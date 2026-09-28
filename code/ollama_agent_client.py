"""
ollama_agent_client.py

A wrapper around a local Ollama server that:
  1. Tracks the model's context window and computes a safe `num_predict`
     budget instead of guessing.
  2. Trims / summarizes conversation history so it fits the context window
     before every call (sliding window + optional summarizer hook).
  3. Detects truncated responses (done_reason == "length") and automatically
     continues generation until the reply is complete or a retry cap is hit.
  4. Logs token usage per call so you can see where your budget is going.

Only dependency: `requests`. Tokenization uses a pluggable counter —
by default a cheap heuristic, but you should swap in a real tokenizer
(see `set_tokenizer`) for accuracy.

Usage:
    from ollama_agent_client import OllamaAgentClient

    client = OllamaAgentClient(
        model="qwen3.8:27b",
        base_url="http://localhost:11434",
        num_ctx=65536,
        keep_alive="30m",
        think=False,  # save output-token budget on routine turns
    )

    reply = client.chat("Explain how TCP handshakes work in detail.")
    print(reply)

--------------------------------------------------------------------------
Notes for a 24GB VRAM / 48GB RAM box running qwen3.8:27b (Q4_K_M, ~18GB):

- Ollama's automatic context sizing is VRAM-tier based (<24GB -> 4k,
  24-48GB -> 32k, >=48GB -> 256k) and does NOT account for how much of
  your VRAM the model weights themselves already consume. With ~18GB of
  weights on a 24GB card, you have roughly 5-6GB left for KV cache and
  compute buffers -- don't rely on auto-sizing, set num_ctx explicitly
  and verify with `ollama ps` that PROCESSOR shows 100% GPU.
- Set these as SERVER environment variables (restart `ollama serve`
  with them set -- they are not per-request options):
      OLLAMA_FLASH_ATTENTION=1
      OLLAMA_KV_CACHE_TYPE=q8_0
  This roughly halves KV cache memory, which is often the difference
  between staying fully on-GPU and silently spilling into system RAM
  (much slower, though your 48GB RAM gives you room if you deliberately
  want more context at the cost of speed).
- qwen3.8:27b is a thinking-capable model with thinking ON by default.
  Reasoning tokens count against num_predict, so long chain-of-thought
  can itself trigger truncation before the real answer is written.
  Pass think=False for routine/tool-call turns; enable it only for
  turns that actually need deeper reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import requests

logger = logging.getLogger("ollama_agent_client")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --------------------------------------------------------------------------
# Token counting — swap this out for a real tokenizer if you have one.
# --------------------------------------------------------------------------

def default_token_counter(text: str) -> int:
    """
    Cheap heuristic: ~4 chars/token for English-like text, with a floor
    based on whitespace-separated word count. Good enough for budgeting;
    NOT exact. Replace via client.set_tokenizer() for precision.
    """
    if not text:
        return 0
    char_estimate = len(text) / 4.0
    word_estimate = len(re.findall(r"\S+", text)) * 1.3
    return int(max(char_estimate, word_estimate))


# Rough per-message overhead for ChatML-style role wrapping
# (<|im_start|>role ... <|im_end|>), which pure content-length estimation
# misses. Qwen models use ChatML; adjust if you switch model families.
CHATML_OVERHEAD_TOKENS = 4


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


@dataclass
class UsageLog:
    calls: List[dict] = field(default_factory=list)

    def record(self, prompt_tokens, num_predict, done_reason, continuations):
        self.calls.append({
            "prompt_tokens": prompt_tokens,
            "num_predict_used": num_predict,
            "done_reason": done_reason,
            "continuations": continuations,
        })

    def summary(self):
        total_prompt = sum(c["prompt_tokens"] for c in self.calls)
        total_continuations = sum(c["continuations"] for c in self.calls)
        truncated_calls = sum(1 for c in self.calls if c["continuations"] > 0)
        return {
            "total_calls": len(self.calls),
            "total_prompt_tokens_est": total_prompt,
            "calls_that_needed_continuation": truncated_calls,
            "total_continuations": total_continuations,
        }


# --------------------------------------------------------------------------
# Main client
# --------------------------------------------------------------------------

class OllamaAgentClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        min_output_tokens: int = 512,
        max_output_tokens: int = 4096,
        safety_margin: int = 256,
        max_continuations: int = 3,
        system_prompt: Optional[str] = None,
        summarizer: Optional[Callable[[List[Message]], str]] = None,
        keep_alive: Optional[str] = "30m",
        think: Optional[bool] = None,
    ):
        """
        model:               Ollama model name (e.g. "qwen3.8:27b")
        base_url:            Ollama server URL
        num_ctx:             total context window of the model (input+output).
                              Set this explicitly rather than relying on
                              Ollama's VRAM-tier auto-sizing, especially on
                              a card where the model weights themselves take
                              up most of your VRAM.
        min_output_tokens:   never budget less than this for output
        max_output_tokens:   never budget more than this for output (cap)
        safety_margin:       tokens held back as a buffer (formatting, stop
                              tokens, tokenizer estimation error)
        max_continuations:   how many auto-continue rounds to attempt if the
                              model hits the length limit mid-answer
        system_prompt:       optional persistent system message
        summarizer:          optional callable(dropped_messages) -> str,
                              used to compress history that falls out of the
                              sliding window instead of discarding it
        keep_alive:          how long Ollama keeps the model loaded after
                              this call (e.g. "30m"). Matters a lot for large
                              models -- reloading an ~18GB model from disk
                              between agent turns is expensive. None uses
                              the server's own default (5m).
        think:               default thinking mode: True/False, or a level
                              string ("low"/"medium"/"high") for models that
                              support graded reasoning effort. None leaves it
                              up to the model/server default. Can be
                              overridden per call via chat(..., think=...).
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens
        self.safety_margin = safety_margin
        self.max_continuations = max_continuations
        self.system_prompt = system_prompt
        self.summarizer = summarizer
        self.keep_alive = keep_alive
        self.think_default = think

        self._token_counter: Callable[[str], int] = default_token_counter
        self.history: List[Message] = []
        self.memory_note: str = ""  # rolling summary of dropped history
        self.usage = UsageLog()
        self._context_verified = False

        logger.info("Client configured with num_ctx=%d (this is what will be requested on first call)", self.num_ctx)

    # ---- configuration -----------------------------------------------

    def set_tokenizer(self, fn: Callable[[str], int]):
        """Plug in a real tokenizer, e.g.:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
        client.set_tokenizer(lambda text: len(tok.encode(text)))
        """
        self._token_counter = fn

    def count_tokens(self, text: str) -> int:
        return self._token_counter(text or "")

    def _message_tokens(self, msg: Message) -> int:
        """Content tokens plus an estimate of chat-template role overhead."""
        return self.count_tokens(msg.content) + CHATML_OVERHEAD_TOKENS

    def probe_actual_num_ctx(self) -> Optional[int]:
        """Ask Ollama what context size is actually loaded for this model
        (can differ from what you requested if VRAM forced a fallback)."""
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5)
            resp.raise_for_status()
            for m in resp.json().get("models", []):
                if m.get("model", "").startswith(self.model.split(":")[0]):
                    ctx = m.get("context_length")
                    if ctx:
                        return ctx
        except requests.RequestException as e:
            logger.warning("Could not probe /api/ps: %s", e)
        return None

    def _verify_context_once(self):
        """Compare what we asked for against what the server actually has
        loaded, and warn loudly on a mismatch. Runs once per client
        instance, right after the first real request (the model has to be
        loaded for /api/ps to report anything)."""
        if self._context_verified:
            return
        self._context_verified = True

        actual = self.probe_actual_num_ctx()
        if actual is None:
            logger.debug("Could not verify loaded context_length (model not visible in /api/ps yet).")
            return

        if actual < self.num_ctx:
            logger.warning(
                "CONTEXT MISMATCH: requested num_ctx=%d but Ollama actually loaded "
                "context_length=%d for %s. This is silent clamping -- likely not "
                "enough free VRAM for the requested size (check `ollama ps` for the "
                "PROCESSOR split, and nvidia-smi for free memory). Your token "
                "budgeting in this client is based on the requested value, not the "
                "real one, so num_predict calculations will be wrong until you set "
                "self.num_ctx to match reality or free up VRAM.",
                self.num_ctx, actual, self.model,
            )
        elif actual > self.num_ctx:
            logger.info(
                "Server has context_length=%d loaded, larger than requested num_ctx=%d "
                "-- you're likely reusing an existing keep_alive'd instance from a "
                "previous run with a bigger setting. It'll shrink next time the model "
                "has to fully reload.",
                actual, self.num_ctx,
            )
        else:
            logger.info("Context verified: requested and loaded num_ctx both %d.", self.num_ctx)

    # ---- context / history management ---------------------------------

    def add_user_message(self, content: str):
        self.history.append(Message("user", content))

    def add_assistant_message(self, content: str):
        self.history.append(Message("assistant", content))

    def _build_messages(self, reserved_for_output: int) -> List[Message]:
        """Sliding window: keep the most recent turns that fit in the
        remaining budget; summarize/drop the rest."""
        budget = self.num_ctx - reserved_for_output - self.safety_margin

        sys_text = self.system_prompt or ""
        if sys_text:
            budget -= self.count_tokens(sys_text) + CHATML_OVERHEAD_TOKENS

        if self.memory_note:
            budget -= self.count_tokens(self.memory_note) + CHATML_OVERHEAD_TOKENS

        kept: List[Message] = []
        used = 0
        dropped: List[Message] = []

        for msg in reversed(self.history):
            t = self._message_tokens(msg)
            if used + t > budget:
                dropped.insert(0, msg)
                continue
            kept.insert(0, msg)
            used += t

        if dropped:
            logger.info("Dropping %d older message(s) to fit context window", len(dropped))
            if self.summarizer:
                new_summary = self.summarizer(dropped)
                self.memory_note = (self.memory_note + "\n" + new_summary).strip()
            else:
                # No summarizer provided — fall back to a compact note so
                # continuity isn't silently lost.
                joined = " | ".join(f"{m.role}: {m.content[:120]}" for m in dropped)
                self.memory_note = (self.memory_note + "\n[earlier context, truncated]: " + joined).strip()

        messages: List[Message] = []
        if sys_text:
            messages.append(Message("system", sys_text))
        if self.memory_note:
            messages.append(Message("system", f"[Summary of earlier conversation]\n{self.memory_note}"))
        messages.extend(kept)
        return messages

    # ---- output budgeting -----------------------------------------------

    def _compute_num_predict(self, prompt_tokens: int) -> int:
        available = self.num_ctx - prompt_tokens - self.safety_margin
        budget = max(self.min_output_tokens, min(available, self.max_output_tokens))
        if available < self.min_output_tokens:
            logger.warning(
                "Only ~%d tokens left for output after prompt (%d tokens); "
                "consider raising num_ctx or trimming history.",
                available, prompt_tokens,
            )
        return budget

    # ---- low-level call ---------------------------------------------------

    def _raw_chat(
        self,
        messages: List[Message],
        num_predict: int,
        extra_options: dict = None,
        think: Optional[bool] = None,
    ) -> dict:
        options = {"num_ctx": self.num_ctx, "num_predict": num_predict}
        if extra_options:
            options.update(extra_options)

        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "options": options,
            "stream": False,
        }
        # keep_alive and think are top-level fields in Ollama's API,
        # not nested under "options".
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        effective_think = self.think_default if think is None else think
        if effective_think is not None:
            payload["think"] = effective_think

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    # ---- public API ---------------------------------------------------

    def chat(self, user_message: str, extra_options: dict = None, think: Optional[bool] = None) -> str:
        """Send a message, auto-managing context and truncation. Updates
        internal history. Returns the full (possibly continued) reply.

        think overrides the client's default thinking mode for this call
        only (see __init__ docstring). Disabling it on routine turns frees
        up real output-token budget, since reasoning tokens count against
        num_predict just like the visible answer.
        """
        self.add_user_message(user_message)

        # Single pass: build history against the worst-case output
        # reservation (max_output_tokens) so whatever num_predict we end up
        # using, the already-built prompt is guaranteed to fit num_ctx.
        messages = self._build_messages(reserved_for_output=self.max_output_tokens)
        prompt_tokens = sum(self._message_tokens(m) for m in messages)
        num_predict = self._compute_num_predict(prompt_tokens)

        full_reply = ""
        done_reason = None
        continuations = 0

        for round_i in range(self.max_continuations + 1):
            data = self._raw_chat(messages, num_predict, extra_options, think=think)
            if round_i == 0:
                self._verify_context_once()
            message = data.get("message", {})
            chunk = message.get("content", "")
            thinking = message.get("thinking")
            done_reason = data.get("done_reason", "stop")
            full_reply += chunk

            actual_prompt_tokens = data.get("prompt_eval_count", prompt_tokens)
            eval_count = data.get("eval_count")
            if thinking:
                logger.info(
                    "Model produced ~%d chars of thinking output (counts against num_predict).",
                    len(thinking),
                )

            if done_reason != "length":
                break

            continuations += 1
            if round_i >= self.max_continuations:
                logger.warning(
                    "Hit max_continuations=%d and reply is still truncated.",
                    self.max_continuations,
                )
                break

            logger.info("Response truncated (round %d, eval_count=%s) — auto-continuing...", round_i + 1, eval_count)
            messages = messages + [
                Message("assistant", chunk),
                Message("user", "Continue exactly where you left off. Do not repeat text already written, and do not restart the answer."),
            ]
            # Recompute budget for the continuation call against the real,
            # now-larger message list.
            cont_prompt_tokens = sum(self._message_tokens(m) for m in messages)
            num_predict = self._compute_num_predict(cont_prompt_tokens)

        self.usage.record(prompt_tokens, num_predict, done_reason, continuations)
        self.add_assistant_message(full_reply)

        if done_reason == "length" and continuations >= self.max_continuations:
            logger.warning(
                "Final reply may still be incomplete after %d continuation(s).",
                continuations,
            )

        return full_reply

    def reset(self):
        self.history.clear()
        self.memory_note = ""

    def print_usage_summary(self):
        print(json.dumps(self.usage.summary(), indent=2))


# --------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Tuned starting point for qwen3.8:27b on a 24GB VRAM / 48GB RAM box.
    # Before running: restart `ollama serve` with
    #   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
    # and check `ollama ps` afterward to confirm PROCESSOR shows 100% GPU
    # at this num_ctx -- raise or lower it based on what you see.
    client = OllamaAgentClient(
        model="qwen3.8:27b",
        num_ctx=32768,
        min_output_tokens=512,
        max_output_tokens=4096,
        keep_alive="30m",
        think=False,  # flip to True/"low" for turns that need real reasoning
        system_prompt="You are a concise, technically precise assistant.",
    )

    actual_ctx = client.probe_actual_num_ctx()
    if actual_ctx and actual_ctx != client.num_ctx:
        logger.info("Server reports loaded context_length=%d (requested %d)", actual_ctx, client.num_ctx)

    reply = client.chat("Write a detailed step-by-step explanation of the TCP three-way handshake, including edge cases.")
    print(reply)
    print("\n--- usage ---")
    client.print_usage_summary()
