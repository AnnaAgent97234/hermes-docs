# scripts/

Helper code for this deployment. One file:

- **`ollama_agent_client.py`** — a single-file Python client for a local Ollama server (only dependency: `requests`) that manages context explicitly instead of trusting Ollama's defaults.

## Why it exists

Ollama auto-picks context size by total VRAM — under 24 GiB → 4k, 24–48 GiB → 32k, 48 GiB+ → 256k. That arithmetic is wrong here: the reference model's weights alone occupy nearly the whole 24 GiB card, leaving only a few GiB for KV cache. So auto-sizing silently undershoots, and the client takes `num_ctx` from you instead of guessing.

It also handles the three failure modes that actually bite in an agent loop:

- **Truncation.** If a reply hits the output limit (`done_reason == "length"`), it appends a "continue exactly where you left off" turn and retries — up to `max_continuations` rounds.
- **Unbounded output.** `num_predict` is computed from the tokens actually left after the prompt, not a fixed constant.
- **Context creep.** A sliding window keeps only recent turns within budget; older turns are compressed into a running summary by an optional `summarizer` callable.

## Tunables that matter here

| Knob | Default | What it does |
|---|---|---|
| `num_ctx` | `32768` | Total context window (input + output). Start at 32k on the reference card with the 27 B model and step up only while watching `ollama ps`. |
| `min_output` / `max_output` | `512` / `4096` | Floor and ceiling for the computed `num_predict`. |
| `max_continuations` | `3` | Retry rounds on `done_reason == "length"`. |
| `think` | `False` | Per-call reasoning budget. Off is correct for ordinary agent turns — reasoning tokens otherwise eat the output budget and you hit `length` on the real answer. |
| `keep_alive` | `None` (Ollama default) | How long Ollama keeps the model resident. Set e.g. `"30m"` to avoid re-reading an ~18 GB model from disk between turns. |

## Two things to do before you trust it

1. **Restart `ollama serve` with two env vars set** (they are per-server, not per-request):

   ```bash
   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
   ```

   Together they roughly halve KV-cache memory — the difference between staying on-GPU and spilling silently into system RAM (much slower).

2. **After loading the model, check `ollama ps`** — the `PROCESSOR` column must read `100% GPU`. If context is spilling to CPU, step `num_ctx` down (e.g. 16384) before touching anything else. The client's own `/api/ps` check will confirm what Ollama actually loaded versus what you asked for.

## Minimal usage

```python
from ollama_agent_client import OllamaAgentClient

client = OllamaAgentClient(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    num_ctx=32768,
    keep_alive="30m",
    system_prompt="You are a concise, technically precise assistant.",
)

reply = client.chat("Explain the TCP three-way handshake, including edge cases.")
print(reply)
client.print_usage_summary()
```

The default token counter is a heuristic (~4 chars/token). For precise budgeting, call `client.set_tokenizer(...)` with a real tokenizer for your model.
