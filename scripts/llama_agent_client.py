#!/usr/bin/env python3
"""
llama_agent_client.py

A single-file Python client for the local llama-swap proxy (OpenAI-compatible
`/v1` API), for simple agent-style chat requests.

  - Model name, base URL, temperature and system prompt are configurable.
  - Streams the reply token-by-token by default (--no-stream to disable).
  - Clear error messages for the two failure modes that actually bite:
    * connection refused   -> the llama-swap.service is not running
    * HTTP 404             -> the model alias is missing from llama-swap

Only dependency: the Python 3 standard library.

Usage:
    python3 scripts/llama_agent_client.py --model qwen3.8:27b "Explain TCP handshakes."
    python3 scripts/llama_agent_client.py --model qwen3-coder:30b --no-stream "Write a parser."
    python3 scripts/llama_agent_client.py --system "You are terse." "ping"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Iterator

# --- defaults -------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("LLAMA_SWAP_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = "qwen3.8:27b"
# Placeholder credential — llama-swap ignores it, but the OpenAI-compatible
# API expects an Authorization header, so we send one.
API_KEY = "local"
DEFAULT_TIMEOUT = 300  # seconds; the first request can take a while while the
                       # proxy spawns llama-server and loads the model weights.


# --- helpers --------------------------------------------------------------

def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def classify_http_error(e: urllib.error.HTTPError, model: str) -> str:
    """Turn an HTTP error into a short, actionable message."""
    try:
        body = e.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""

    if e.code == 404:
        if "model" in body.lower():
            return (
                f"HTTP 404 — model '{model}' is not available.\n"
                f"  The alias is missing from llama-swap. Check the available "
                f"models with:\n"
                f"    curl -s {DEFAULT_BASE_URL}/models"
            )
        return f"HTTP 404 — the proxy does not know this path."

    if body:
        return f"HTTP {e.code} — {body}"
    return f"HTTP {e.code} — no response body."


def stream_chunks(resp) -> Iterator[str]:
    """Iterate SSE 'data: {...}' chunks from a streaming chat-completions
    response and yield the delta content of each one."""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-shot chat request against the local llama-swap proxy "
                    "(OpenAI-compatible /v1 API)."
    )
    parser.add_argument(
        "prompt",
        help="The user message to send.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}; alternative: qwen3-coder:30b)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the OpenAI-compatible API (default: %(default)s; "
             "env: LLAMA_SWAP_BASE_URL)",
    )
    parser.add_argument(
        "--system",
        default="You are a concise, technically precise assistant.",
        help="System prompt prepended to the conversation (a concise "
             "technically-precise assistant by default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional; the model's server default applies "
             "if omitted)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Return the full reply in one JSON blob instead of streaming it",
    )

    args = parser.parse_args()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    payload: dict = {
        "model": args.model,
        "messages": messages,
        "stream": not args.no_stream,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            if args.no_stream:
                data = json.loads(resp.read().decode("utf-8"))
                content = (data.get("choices") or [{}])[0].get("message", {}).get(
                    "content", ""
                )
                print(content)
            else:
                for chunk in stream_chunks(resp):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError, so it must come first.
        fail(classify_http_error(e, args.model))
    except urllib.error.URLError as e:
        # Connection refused / DNS failure / server not listening.
        fail(
            f"cannot reach the proxy at {args.base_url}: {e.reason}\n"
            f"  Is llama-swap.service running? Check with:\n"
            f"    systemctl --user status llama-swap"
        )


if __name__ == "__main__":
    main()
