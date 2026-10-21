# scripts/

Helper code for this deployment. One file:

- **`llama_agent_client.py`** — a single-file Python 3 client for the local
  llama-swap proxy (OpenAI-compatible `/v1` API). Standard library only, no
  third-party dependencies.

## Why it exists

A one-liner chat client for a single agent-style turn against the live local
LLM stack. It exists so you can send a prompt from the terminal (or a shell
script) without reaching for `curl` + `jq` or an HTTP library:

- The model name, base URL, system prompt and temperature are all
  configurable from the command line.
- The reply is **streamed** token-by-token by default (`--no-stream` disables
  it), so you see the answer as it is generated instead of waiting for a
  single JSON blob at the end.
- The two failure modes that actually bite here produce clear, actionable
  error messages:
  - **Connection refused** → the `llama-swap` systemd user service is not
    running (the proxy is what the script talks to).
  - **HTTP 404** → the model alias is missing from llama-swap (you asked for
    a model that the proxy does not know about).

## Prerequisites

- `python3` on your `PATH` (any 3.x; only the standard library is used).
- The `llama-swap` systemd **user** service running, listening on
  `127.0.0.1:8080`:

  ```bash
  systemctl --user status llama-swap
  # expected: active (running), ExecStart=/usr/bin/llama-swap --config ...
  # --listen 127.0.0.1:8080
  ```

- A model alias registered in llama-swap. The two known on this box are
  `qwen3.8:27b` (default) and `qwen3-coder:30b`.

## Run it

```bash
# Default model (qwen3.8:27b), streaming reply:
python3 scripts/llama_agent_client.py "Explain the TCP three-way handshake."

# Pick a different model:
python3 scripts/llama_agent_client.py --model qwen3-coder:30b "Write a parser."

# Custom system prompt and temperature:
python3 scripts/llama_agent_client.py --system "You are terse." \
    --temperature 0.2 "ping"

# Non-streaming (single JSON-style blob printed at the end):
python3 scripts/llama_agent_client.py --no-stream "ping"
```

The first request after the model has been unloaded can take a while — the
proxy spawns a `llama-server` and reads the model weights from disk before it
can start generating. Subsequent requests while the model is resident (its
idle TTL is 600 s) are fast.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--model` | `qwen3.8:27b` | Model name to request. `qwen3-coder:30b` is the other known alias. |
| `--base-url` | `http://127.0.0.1:8080/v1` | Base URL of the OpenAI-compatible API. Overridden by `--base-url` and by the `LLAMA_SWAP_BASE_URL` env var (flag wins). |
| `--system` | `"You are a concise, technically precise assistant."` | System prompt prepended to the user message. |
| `--temperature` | *(unset)* | Sampling temperature. Omit to let the server/model default apply. |
| `--no-stream` | off | Disable token streaming; print the full reply once. |

## What changed

This script used to talk to **Ollama** at `http://127.0.0.1:11434/api/chat`.
Ollama has been replaced by **llama-swap** on `http://127.0.0.1:8080/v1`,
which speaks the OpenAI-compatible chat-completions API. So the request is
now `POST /v1/chat/completions` with `{"model", "messages", "stream", ...}`
instead of Ollama's `options`-flavoured body, and the `Authorization` header
carries a placeholder key (`local`) that llama-swap ignores.

The equivalent raw `curl` against the proxy:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{
        "model": "qwen3.8:27b",
        "messages": [
          {"role": "user", "content": "Reply with exactly: OK"}
        ],
        "stream": false
      }'
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `cannot reach the proxy at http://127.0.0.1:8080/v1: [Errno 111] Connection refused` | `systemctl --user status llama-swap` — if it is not `active (running)`, `systemctl --user start llama-swap`. |
| `HTTP 404 — model '...' is not available.` | `curl -s http://127.0.0.1:8080/v1/models` to list the aliases llama-swap actually knows. |
| First request is slow (tens of seconds) | Expected — the proxy is spawning `llama-server` and loading the model. |
| `stream` reply looks garbled in a pipe | Pipe to `cat` or disable `--no-stream`; some terminals mangle incremental output. |
