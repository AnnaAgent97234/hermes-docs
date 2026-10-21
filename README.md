# Hermes Light Implementation — Setup & Backup Guide

A single-machine, 100% local deployment of [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap). No external model provider for the default lane, the auxiliary lanes, or the sub-agent lane: every inference call terminates on the local host.

This guide is a working recipe, not marketing copy. It documents a reference deployment that is known to run, lists the exact values that make it work, and gives you the commands to reproduce, back up, restore, and maintain it. It is deliberately scoped to the software surface: the host, the two runtimes (llama-swap + Hermes), and the Hermes features that are actually enabled — memory, skills, the skill curator, sub-agent delegation, the kanban dispatcher, and scheduled cron jobs.

> **What changed from the Ollama edition of this guide.** The model backend is no longer Ollama. `llama-swap` runs as a user-level systemd service and fronts a `llama-server` (llama.cpp) process per model, exposing an OpenAI-compatible `/v1` endpoint on `127.0.0.1:8080`. The GGUF models that used to live in the Ollama store now live in `~/Dev/llm/models/`. Everything on the Hermes side — config shape, memory, skills, delegation, kanban, cron — is unchanged in kind, only in endpoint.

**Audience.** Operators who can drive a terminal, edit YAML by hand, and read a config file to understand what they are shipping.

**Units.** Metric throughout — °C for temperature, GiB/TiB for storage and memory, km/h where a speed matters.

## In this guide

1. [Prerequisites](#prerequisites)
2. [Reference host spec](#reference-host-spec)
3. [Setup — step by step](#setup--step-by-step)
   - [3.1 Install llama.cpp and llama-swap](#31-install-llamacpp-and-llama-swap)
   - [3.2 Get the models (GGUF)](#32-get-the-models-gguf)
   - [3.3 Configure llama-swap](#33-configure-llama-swap)
   - [3.4 Run llama-swap as a systemd user service](#34-run-llama-swap-as-a-systemd-user-service)
   - [3.5 Install Hermes Agent](#35-install-hermes-agent)
   - [3.6 Wire the local provider into `config.yaml`](#36-wire-the-local-provider-into-configyaml)
   - [3.7 Memory, skills, and the curator](#37-memory-skills-and-the-curator)
   - [3.8 Custom skills on this deployment](#38-custom-skills-on-this-deployment)
   - [3.9 Sub-agent delegation](#39-sub-agent-delegation)
   - [3.10 Kanban dispatcher](#310-kanban-dispatcher)
   - [3.11 Scheduled cron jobs](#311-scheduled-cron-jobs)
4. [Ongoing operation](#ongoing-operation)
5. [Optimizations & pro-tips](#optimizations--pro-tips)
6. [Safety & maintenance](#safety--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [`~/.hermes` layout](#hermes-layout-reference)
9. [Scripts](#scripts)
10. [Sources](#sources)

---
## Prerequisites

Before you start, make sure every item below is true.

### Software & accounts

- [ ] A Linux distro you can build llama.cpp on. Reference: **CachyOS**.
- [ ] A C/C++ toolchain (GCC, CMake) for building `llama-server` from source — or a package that ships it.
- [ ] An AMD GPU in the RX 7900 class (or equivalent) with a working ROCm/AMDGPU path, if you want 27 B–30 B models at usable speed.
- [ ] Outbound HTTPS (for downloads on first setup).
- [ ] No API keys, no cloud accounts. The whole point is that there is nothing to sign in to.

### Expected state when you're done

You can run `hermes` from a shell and it answers using a model that is already on disk. `curl http://127.0.0.1:8080/v1/models` lists the models, and the active one is serving on the GPU. Killing network access to the outside world does not change any of that.

---
## Reference host spec

These are the values from the system this guide reproduces. Treat them as the *floor*, not the ceiling — the software runs the same on anything that meets the bill of materials above.

| Component | Detail |
|---|---|
| OS | CachyOS Linux |
| Kernel | `7.2.10-1-cachyos` |
| Architecture | `x86_64` |
| CPU | AMD Ryzen 7 9800X3D (8 cores / 16 threads) |
| RAM | 46 GiB |
| Storage | NVMe, 1.3 TiB total, ~575 GiB free at the reference point in time |
| GPU | AMD Radeon RX 7900 class (Navi 31) + an AMD Granite Ridge iGPU on-chip |
| Runtime 1 | llama.cpp `llama-server` (0.6.0-dev, build 11429), one process per model, all layers on GPU |
| Runtime 2 | llama-swap v265 (model router), listening on `127.0.0.1:8080` |
| Runtime 3 | Hermes Agent v0.22.2 (Python 3.14) |
| Models in use | `qwen3.8:27b` (default, 16.8 GiB GGUF), `qwen3-coder:30b` (delegation, 18.6 GiB GGUF) |
| Context | 65 536 tokens per model, q8_0 KV cache, flash attention |
| Timezone | `Europe/Paris` |

Two things to notice in this spec. First, the machine has 16 logical CPUs and 46 GiB of RAM, and the two 17–19 GiB models are served with *all layers on the GPU* — system memory is the fallback, not the plan. Second, unlike the Ollama edition, only one model is resident at a time: llama-swap's TTL policy unloads an idle model after 10 minutes so the desktop gets its VRAM back (see [3.3](#33-configure-llama-swap)).

---
## Setup — step by step

Every step below is imperative, and each step ends with what you should see once it is done. Work through them in order.

### 3.1 Install llama.cpp and llama-swap

`llama-server` (from llama.cpp) is the inference engine; `llama-swap` is the router in front of it. Reference layout on this machine:

| Piece | Where |
|---|---|
| llama.cpp source checkout (build root) | `~/Dev/llm/llama.cpp/` |
| `llama-server` binary | `/usr/bin/llama-server` |
| `llama-swap` binary | `/usr/bin/llama-swap` (v265) |
| Router config | `~/Dev/llm/llama-swap.yaml` |
| Models | `~/Dev/llm/models/` |

```bash
# llama.cpp — build once, get llama-server + the conversion tools
git clone https://github.com/ggml-org/llama.cpp ~/Dev/llm/llama.cpp
cd ~/Dev/llm/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_ROCM=ON
cmake --build build --config Release -j
# reference install puts the binaries on PATH:
#   /usr/bin/llama-server   (0.6.0-dev, build 11429)

# llama-swap — v265 (d1e743fe) on the reference host
llama-swap --version
```

**Expected outcome.** `llama-server --version` reports the build (reference: `0.6.0-dev`, build `11429`). `llama-swap --version` reports `v265`. Neither binary is a daemon — both are invoked per launch; what keeps the stack alive across reboots is the systemd unit in [3.4](#34-run-llama-swap-as-a-systemd-user-service).

### 3.2 Get the models (GGUF)

llama.cpp speaks GGUF, so the models must be GGUF files on disk. The reference set lives in `~/Dev/llm/models/`:

| File | Size | Role |
|---|---|---|
| `qwen3.8-27b.gguf` | 16.8 GiB | Default model — chat, reasoning, compression, session titles |
| `qwen3-coder-30b.gguf` | 18.6 GiB | Sub-agent delegation lane |
| `qwen3.8-27b.mmproj.gguf` | 0.9 GiB | Multimodal projector for `qwen3.8:27b` (not loaded by the reference router) |

On the reference machine these were **migrated from the old Ollama store**, not re-downloaded: the conversion helpers are kept next to the models as provenance (`extract-ollama-model.sh`, `convert-ollama-models.sh` in `~/Dev/llm/`). If you are starting fresh, obtain the GGUF files from the model's release channel and place them where `llama-swap.yaml` points at them.

**Expected outcome.** `ls -la ~/Dev/llm/models/` shows the two serving models (and, if you want vision, the mmproj sidecar). `du -sh ~/Dev/llm/models/` is ~35 GiB on the reference host.

### 3.3 Configure llama-swap

llama-swap is a small Go router: it owns the HTTP endpoint, loads an on-demand `llama-server` process for each model you ask for, and unloads it after a TTL. Its config is one YAML file — `~/Dev/llm/llama-swap.yaml` on the reference host:

```yaml
healthCheckTimeout: 300     # s — how long a model may take to come up before the router declares it unhealthy
logLevel: info

macros:
  llama:                    # default server args applied to every model
    - -ngl
    - "99"                  # all layers on the GPU
    - --parallel
    - "1"                   # one slot — this is a single-user box
    - --flash-attn
    - on
    - --cache-type-k
    - q8_0                  # 8-bit quantized KV cache (halves KV memory vs f16)
    - --cache-type-v
    - q8_0
    - --ctx-size
    - "65536"               # 64 k context

models:
  - name: qwen3.8:27b       # the name Hermes sees (OpenAI-style id)
    model: /home/arnaud97234/Dev/llm/models/qwen3.8-27b.gguf
    params:                 # per-model server args
      - -temp
      - "1.0"
      - -p
      - "anna.sabu"         # system prompt baked into the server
      - -no-cfg             # disable the model's built-in quirk fixes — deterministic behaviour
      - -fa
      - on
      - -t
      - "3"                 # threads for the CPU parts
      - -c
      - "65536"
    ttl: 600                # unload after 600 s idle — the key VRAM-saving policy
    aliases:                # alternate ids that route to the same model
      - qwen3.8
      - qwen3.8:latest

  - name: qwen3-coder:30b
    model: /home/arnaud97234/Dev/llm/models/qwen3-coder-30b.gguf
    params:
      - -temp
      - "1.0"
      - -p
      - "anna.sabu"
      - -no-cfg
      - -fa
      - on
      - -t
      - "3"
      - -c
      - "65536"
    ttl: 600
    aliases:
      - qwen3-coder
      - qwen3-coder:latest
```

Reading the three policies out loud:

- **`ttl: 600`** — a model that nobody called for 10 minutes is offloaded. This is why the reference host can run a 27 B and a 30 B model on the same GPU: they are *rotating residents*, not co-residents.
- **`-ngl 99` + q8_0 KV + flash-attn** — the model weights sit on the GPU, the KV cache is quantized to 8-bit, and attention is flash. Together they make 64 k context affordable on RX 7900-class VRAM.
- **`aliases`** — the router keeps the Ollama-era model ids (`qwen3.8`, `qwen3.8:latest`, …) working, so anything that still speaks the old names routes to the same GGUF.

**Expected outcome.** With the service running (next section), `curl -s http://127.0.0.1:8080/v1/models` returns the two models with `"function_calling": true` and `"context_length": 65536`; a just-answered model shows `"status": "loaded"`, an idle one `"unloaded"`.

### 3.4 Run llama-swap as a systemd user service

A user-level unit (no root) keeps the router alive across reboots: `~/.config/systemd/user/llama-swap.service`:

```ini
[Unit]
Description=llama-swap local model server
After=network.target

[Service]
ExecStart=/usr/bin/llama-swap --config %h/Dev/llm/llama-swap.yaml --listen 127.0.0.1:8080
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-swap
```

**Expected outcome.** `systemctl --user status llama-swap` is `active (running)`. The old `ollama.service` file may still exist on a migrated machine — on the reference host it is **disabled** and inert (`systemctl --user is-enabled ollama` → `disabled`); leave it or delete it, either way nothing starts with it.

### 3.5 Install Hermes Agent

Hermes ships as a CLI plus a Desktop app. For a pure headless setup the CLI is enough.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

**Expected outcome.** `hermes --version` reports the version (the reference install is `v0.22.2`). `hermes setup` runs onboarding; for a local-only deployment you do not need a portal OAuth — you are not calling an external provider. If you want the desktop companion, follow the install guide at <https://hermes-agent.nousresearch.com/docs/getting-started/installation>.

### 3.6 Wire the local provider into `config.yaml`

`$HOME/.hermes/config.yaml` is the single file that pins the deployment to llama-swap. Three things to get right:

1. The **`model`** block points at `http://127.0.0.1:8080/v1` and uses `chat_completions` (llama-swap's OpenAI-compatible layer).
2. The **`providers.<name>`** block declares the endpoint explicitly, with the model list, so it resolves on every boot.
3. The **`auxiliary.*`** and **`delegation`** blocks reference the same router.

The reference slice of `config.yaml` that makes the whole thing run 100% locally:

```yaml
model:
  default: qwen3.8:27b
  base_url: 'http://127.0.0.1:8080/v1'
  provider: local-llama-swap
  max_tokens: 4096
  api_mode: chat_completions
  api_key: local              # sentinel — llama-swap ignores it, Hermes requires a string

providers:
  local-llama-swap:
    name: Local (llama-swap)
    api: http://127.0.0.1:8080/v1
    api_key: local
    default_model: qwen3.8:27b
    models:
      - qwen3.8:27b
      - qwen3-coder:30b

auxiliary:
  compression:                # session-history compression
    model: qwen3.8:27b
    base_url: http://127.0.0.1:8080/v1
    api_key: ollama           # legacy sentinel from the Ollama era — still an inert string
  title:                      # session auto-titling
    model: qwen3.8:27b
    base_url: http://127.0.0.1:8080/v1
    api_key: ollama
```

Notes on the values:

- `api_key: local` / `api_key: ollama` are **sentinels, not credentials**. llama-swap does not check them; the `ollama` strings in `auxiliary.*` are leftovers from the migration that still work because the value is ignored. Do not paste a real secret into these blocks.
- The auxiliary lanes (compression, titles) run on the same 27 B model as the default. The old guide parked them on a smaller `gemma4:31b`; that model is no longer in this deployment, and on a single-GPU box with TTL rotation the honest choice is to share the router's default. If you later add a dedicated small model, register it in `llama-swap.yaml` and point the auxiliary blocks at it — that is the only reason to do so.
- `max_tokens: 4096` is the per-response cap for the default lane.

After writing the config, validate that Hermes reads it back:

```bash
hermes config get model.default
hermes config get model.base_url
hermes config get delegation.model
```

**Expected outcome.** `qwen3.8:27b`, `http://127.0.0.1:8080/v1`, and `qwen3-coder:30b` respectively. If any of these is empty or points at a non-local URL, fix the YAML before moving on — the rest of this guide assumes all three are correct.

### 3.7 Memory, skills, and the curator

Hermes keeps two persistent memory files and a self-maintaining skill library. These are what make the deployment "compound" over time — the model of the operator and the set of operating manuals both grow across sessions.

**Memory.** Two files, each with a hard character limit so that memory stays a signal, not a dump:

| File | Purpose | Char limit |
|---|---|---|
| `memories/MEMORY.md` | Agent's own notes: environment facts, standing conventions with no task home | 2200 |
| `memories/USER.md` | Operator profile: who they are, durable preferences, working agreements | 1375 |

The limits live in `config.yaml` (reference values):

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  write_approval: false       # the agent may write memory without per-write approval
```

**Expected outcome.** After the first few sessions, `memories/MEMORY.md` and `memories/USER.md` are non-empty and well under their limits. When a limit approaches, the agent is expected to prune stale entries rather than silently overflow — treat a near-full memory file as a maintenance signal, not a failure.

**Skills.** Skills are Markdown operating manuals that load only when a task matches. Two directories matter:

- `~/.hermes/skills/` — the per-profile skill library (the `custom/` subtree holds this deployment's own skills — see [3.8](#38-custom-skills-on-this-deployment)).
- `~/.hermes/shared-skills/` — the cross-profile layer, referenced explicitly:

```yaml
skills:
  external_dirs:
    - ~/.hermes/shared-skills
  write_approval: false       # the agent may create/update skills without per-write approval
```

**Expected outcome.** `hermes skills list` shows the installed skills. A skill is usable the moment its front-matter description matches the task — no import step, no restart.

**The curator.** The curator is the self-maintenance loop. On the reference deployment it is enabled with conservative settings:

```yaml
curator:
  enabled: true
  interval_hours: 168          # reviews the library every 7 days
  min_idle_hours: 2
  stale_after_days: 14         # a skill idle 14 days is flagged stale
  archive_after_days: 30       # a skill idle 30 days is archived
  consolidate: false
  prune_builtins: true
  backup:
    enabled: true
    keep: 5                    # keep the last 5 curator backups
```

**Expected outcome.** The library compacts itself over time: unused skills drift to `archive/`, and the live set stays lean. The `keep: 5` backup gives you a 5-generation rollback window if a curation pass removes something you wanted.

> **Pro-tip.** The 168-hour interval is deliberate. Curating more often chases noise; a full-week window means a skill is only archived if it genuinely stopped earning its place. Do not shorten this unless you are actively growing the library and need faster pruning.

### 3.8 Custom skills on this deployment

`~/.hermes/skills/custom/` is where this deployment's own operating manuals live. The set below is the current inventory — each one loads only when a matching task comes up, and its one-liner is the description the router uses to decide that:

| Skill (path under `skills/custom/`) | What it does |
|---|---|
| `devops/freebox-smb-access` | Rename files on the Freebox MyPassport share over SMB. |
| `devops/hard-limits-and-permissions` | Non-negotiable safety boundaries — sudo, credentials, network scope, destructive ops, approval chains. Applies to every technical action regardless of scale. |
| `devops/hermes-setup-inventory` | Audit a Hermes profile: skills, cron, delegation, memory. |
| `devops/movie-rename` | Find IMDB release years, build a movie rename list. |
| `devops/remote-service-access` | Guidelines and troubleshooting for accessing remote services and dashboards via SSH tunnels and network configuration. |
| `devops/technical-execution-workflow` | Opt-in deep workflow for technical tasks — understand, plan with 3-iteration self-challenge, execute, self-review, return, handle revisions. Used only when the user asks for it. |
| `productivity/technical-guide-authoring` | Transform raw research and advice into professional, actionable implementation guides and technical documentation. |
| `proton` | Read and use Proton mail and calendar as anna.sabu. |
| `research/complex-task-planning` | Protocol for calibrating request complexity, performing research, and structuring an actionable plan before execution. |

Note the shape of the set: it is an operator's toolkit, not a demo collection — home storage (Freebox), media housekeeping (movie renaming), remote service access, and a strict pair of process skills (`technical-execution-workflow` + `complex-task-planning`) that gate how bigger technical work gets done.

**Expected outcome.** `hermes skills list` (or the profile's skill registry) shows all nine. The curator's 14-day-stale / 30-day-archive rules apply to them like anything else — if one of these disappears to `archive/` and you still use it, re-trigger or pin it (see [Troubleshooting](#troubleshooting)).

### 3.9 Sub-agent delegation

Delegation lets Hermes spawn a shorter, bounded sub-run for a sub-task, on a separate model, inheriting the parent's tool set. The reference block:

```yaml
delegation:
  model: qwen3-coder:30b          # the delegation lane — a code-tuned 30 B, not the chat default
  base_url: http://127.0.0.1:8080/v1
  api_key: ollama                 # sentinel, ignored by llama-swap (legacy from the Ollama era)
  inherit_mcp_toolsets: true      # sub-agent gets the same toolset as the parent
  max_iterations: 60              # 60 tool calls, whichever bound hits first…
  child_timeout_seconds: 600      # …or 10 minutes
  max_concurrent_children: 1      # one sub-agent at a time
  max_spawn_depth: 1              # one level deep — no recursive delegation
  orchestrator_enabled: false
  subagent_auto_approve: false    # sub-agents do not self-grant permissions
```

The choices encode a specific posture on a single machine:

- **`qwen3-coder:30b`, not the 27 B chat model.** Since the router time-shares one GPU, the delegation lane is a *different model on the same GPU* — when a sub-run starts, llama-swap loads the coder model (which triggers the TTL timer on the chat model if it's idle). The two lanes never fight for the same resident slot the way they would under Ollama's static co-residence.
- `max_concurrent_children: 1` + `max_spawn_depth: 1` — no fan-out, so a delegation call cannot multiply load on a single machine.
- `subagent_auto_approve: false` — sub-agents do not self-grant new permissions.
- `orchestrator_enabled: false` — delegation is a scoped assist, not a recursive planning loop.

**Expected outcome.** When a task is delegated, `hermes` runs it in an isolated context, and only the sub-run's summary returns to the parent. A timeout at 600 s or a trip of `max_iterations: 60` aborts cleanly and reports `timed_out` — it does not hang.

### 3.10 Kanban dispatcher

The kanban is a small shared task board (SQLite under `~/.hermes/`) that turns multi-step work into discrete cards, each with an owner and a lifecycle. On the reference deployment it dispatches inside the gateway:

```yaml
kanban:
  dispatch_in_gateway: true            # ticks even when no interactive session is open
  dispatch_interval_seconds: 60        # board scanned every 60 s
  failure_limit: 2                     # a worker that crashes twice is not blindly retried a third time
  auto_decompose_per_tick: 3           # up to 3 cards may be auto-decomposed (split) per tick
  dispatch_stale_timeout_seconds: 14400# a run stuck for 4 h is declared stale and reclaimed
  worker_log_rotate_bytes: 2097152     # worker logs rotate at 2 MB
  worker_log_backup_count: 1
```

**Expected outcome.** A ready card moves `todo → ready → running → done` (or `→ blocked` on a genuine ambiguity). `failure_limit: 2` is the trip-wire: two consecutive crash loops on the same card is the point at which a human looks, not the dispatcher.

### 3.11 Scheduled cron jobs

Cron jobs run a fixed prompt on a schedule and deliver the result to a channel. The live set on the reference machine (both healthy):

| Job | Schedule (Europe/Paris) | What it does | Delivery |
|---|---|---|---|
| `Paris Weekly Events` | `0 15 * * 5` (Fri 15:00) | Most popular upcoming local events over the next 7–10 days, cross-checked against the weather; ranked list of 8–10 | `telegram` |
| `Le Monde Daily Briefing` | `45 8 * * *` (08:45 daily) | Curate the top high-impact news stories, each with a direct HTTPS link | `origin` |

To inspect the live set:

```bash
hermes cron list
hermes cron runs [job-uuid]        # durable execution history; optional job filter
```

**Expected outcome.** `cron list` shows the active jobs with schedule and delivery target. `cron runs` shows each attempt with its status, so a job that "ran" on the surface but died mid-flight is visible. A growing run of non-`completed` statuses on one job is what you fix, not the jobs that are quiet.

> **Scheduling note.** Jobs created from a plain CLI session are local-only — their output is stored and can be listed, but there is no push channel unless the job's `deliver` targets a gateway-connected platform (e.g. `telegram`). Set `deliver` explicitly if you want the job to reach your phone.

---
## Ongoing operation

Once the deployment is up, steady-state operation is about keeping the router healthy and the two persistent things — the memory files and the skill library — honest.

- **Router.** `systemctl --user status llama-swap` is `active (running)`. `curl -s http://127.0.0.1:8080/v1/models` lists both models; watch the `status` field — `loaded` means resident on the GPU, `unloaded` means it will cold-load on next call.
- **Memory.** Watch the two files against their limits. A near-full `MEMORY.md` (≈ 90% of 2200 chars) is a cue to consolidate old entries into a skill, not to push past the cap. Task-specific knowledge belongs in a skill (which loads on demand); only facts that apply to *every* session belong in memory.
- **Skills.** Let the curator do the compaction on its 168-hour cycle. If you notice a skill you use weekly getting archived, that is a bug in the signal — pin it or merge it into a living skill so it stops reading as "stale".
- **Model files.** The GGUFs live in `~/Dev/llm/models/` and do not move or re-download. When adding a model: drop the GGUF in place, add an entry to `llama-swap.yaml` (name, path, params, ttl, aliases), `systemctl --user restart llama-swap`, then verify it appears in `/v1/models` and answers one prompt.
- **Gateway.** The dispatcher and messaging run inside the gateway process. Restarting the gateway is the operation that can interrupt in-flight kanban work — restart it deliberately, not reflexively. `sessions.retention_days: 90` and the auto-prune settings keep session history bounded on their own.

A minimal steady-state loop:

```bash
# Is the router alive, and which model is resident?
systemctl --user status llama-swap
curl -s http://127.0.0.1:8080/v1/models

# Is the agent answering on the local model?
hermes --version
hermes config get model.default
hermes config get model.base_url

# Is the gateway (dispatcher + messaging) alive?
hermes gateway status        # or: pgrep -af "gateway run"
```

## Optimizations & pro-tips

- **Let TTL do the co-residence for you.** One GPU, two big models: llama-swap's `ttl: 600` means each lane pays a one-time cold load (a few seconds of weight loading) and then serves fast. Do not raise TTL to "avoid cold loads" unless you have measured real pain — VRAM headroom for the desktop is worth more.
- **`-ngl 99` + q8_0 KV + flash-attn is the sweet spot.** Weights on GPU, 8-bit KV cache, flash attention: 64 k context on RX 7900-class VRAM. Dropping to f16 KV roughly doubles KV memory and is only worth it if you see quality regressions at 64 k, which is rare.
- **Keep the delegation lane on the coder model.** The split — chat on `qwen3.8:27b`, sub-agents on `qwen3-coder:30b` — is the economic core of this stack. Re-merging them onto one resident model wastes the GPU's most scarce resource (VRAM) and buys nothing.
- **`max_iterations: 60` / `child_timeout_seconds: 600` are cheap insurance.** A sub-agent pinned by either bound reports `timed_out` instead of wedging the parent. Raise them only for a lane you have measured to be slow, and measure again.
- **One concurrent sub-agent, one spawn depth.** `max_concurrent_children: 1` and `max_spawn_depth: 1` keep a single machine from multiplexing model load. Fan out only when you have real headroom.
- **The curator's `keep: 5` backup is your undo.** Before a manual curation pass you did not ask for, snapshot the skill dir yourself; after the pass, verify the archive did not take a live skill.
- **`api_key: local` / `api_key: ollama` are sentinels, not credentials.** Do not paste a real secret into the local provider block — llama-swap ignores it and you would be leaking a real key into an insecure config.
- **Point cron `deliver` at a real channel.** A job with no `deliver` runs and stores output but reaches no one. `telegram`/`origin` is the difference between a job that works and a job that works *and tells you it worked*.

## Safety & maintenance

**Credentials.** This deployment uses no external model API keys. The only key-like values in the local provider blocks are the `local` / `ollama` sentinels (see above) — inert strings. Real secrets (e.g. the GitHub PAT used by the MCP layer) belong in the encrypted `~/.hermes/vault/` or in env for that tool, never in `memories/` or in a skill — those files are intended to be shared and read out. If you later add a genuine external provider, keep its key in the config (which is `0600`), never in a repo, and rotate it on any suspicion of exposure.

**The local-only boundary is the security model.** "100% local" means the default lane, the auxiliary lanes, and the sub-agent lane all resolve to `127.0.0.1:8080`. If a config block points at a non-local `base_url`, you have left the boundary — either fix it or decide consciously that that lane is fine calling out.

**Destructive ops are gated.** On this deployment, approvals are `mode: manual`, `cron_mode: deny`, `destructive_slash_confirm: true`. A cron job cannot force a destructive action, and a destructive interactive command asks before it runs. Leave all three on.

**Periodic maintenance.**

- [ ] **Weekly** — confirm curator backups exist (up to 5 generations) and that `MEMORY.md` / `USER.md` are under their char limits.
- [ ] **Monthly** — `curl -s http://127.0.0.1:8080/v1/models` and diff expected vs. registered models; `hermes --version` against the version you intend; `systemctl --user is-enabled llama-swap` reads `enabled`.
- [ ] **On any update** — the update path keeps `backup_keep: 5`; restore from the newest good backup if a version bump misbehaves rather than chasing a broken install.

**Back up the things that are hard to reproduce:** `~/.hermes/config.yaml` (the whole Hermes side is pinned here), `~/.hermes/memories/` (MEMORY.md + USER.md), `~/.hermes/skills/` + `~/.hermes/shared-skills/` (the custom library), and the llama-swap pair: `~/Dev/llm/llama-swap.yaml` + `~/.config/systemd/user/llama-swap.service`. The GGUF models are re-downloadable (or re-convertible via the Ollama-extraction scripts); these are not.

---
## Troubleshooting

**Symptom: `curl http://127.0.0.1:8080/v1/models` times out or the port is closed.**
The router is down. Check `systemctl --user status llama-swap`, then `journalctl --user -u llama-swap -n 40`. Common causes: a bad path in `--config`, a model file that moved under `~/Dev/llm/models/`, or a YAML syntax error in `llama-swap.yaml`. Fix, then `systemctl --user restart llama-swap`.

**Symptom: Hermes says "provider not found" or "model not found".**
The `providers.local-llama-swap` block and the `model` block disagree. `hermes config get model.default`, `hermes config get model.base_url`, and `hermes config get model.provider` should read `qwen3.8:27b`, `http://127.0.0.1:8080/v1`, `local-llama-swap`. If `provider` is empty, the provider block is missing or mis-named; if the model name is not in the `models:` list of that block, add it. Names must match llama-swap's `name:` or one of its `aliases:` exactly.

**Symptom: first response is slow (10–60 s), then fast.**
That is the TTL unload in action — the model was offloaded and is cold-loading (weights + KV). Verify by watching `/v1/models` show `"status": "unloaded"` before the call. If you want to *avoid* cold loads, that is a design decision: keep TTL low and accept the load, or keep the model pinned at the cost of permanent VRAM. On this box, accept the load.

**Symptom: a model load appears to hang.**
`healthCheckTimeout: 300` allows up to 5 minutes for a 27 B / 30 B model to come up under load. Be patient once; if it exceeds that, llama-swap declares the model unhealthy and the call fails cleanly — retry after a moment or check VRAM with your GPU monitor.

**Symptom: an old Ollama reference (`:11434`, `gemma4:31b`, `devstral`) shows up somewhere.**
Three known leftovers on a migrated machine, all inert: `~/.config/systemd/user/ollama.service` (disabled, nothing starts from it), `~/.hermes/config.yaml.bak-ollama` (a backup, not the live config), and `api_key: ollama` sentinel strings in the auxiliary/delegation blocks (ignored by llama-swap). The live config is the only one that matters — verify it with `hermes config get`.

**Symptom: a skill you expect does not load.**
Skills load by description match, not by name. Check the skill's front-matter `description:` still describes the task you're doing, and check it was not archived by the curator (look under its `archive/` location; the curator keeps up to 5 backups). A `custom/` skill that is still used weekly should never be allowed to drift to stale — merge or pin it.

**Symptom: a delegated sub-run reports `timed_out`.**
Expected, bounded behaviour: either `child_timeout_seconds: 600` or `max_iterations: 60` was reached. The parent continues; do not treat this as a crash. If a lane is *consistently* hitting the bound, split the sub-task rather than raising the bound — a sub-run that needs more than 600 s / 60 calls is usually the wrong shape of sub-run.

**Symptom: a cron job "ran" but you never got the output.**
The job ran and stored output; it had no delivery channel. A job created from a plain CLI session is local-only — set `deliver` to a gateway-connected target (e.g. `telegram`) if you want it pushed. `hermes cron runs [job-uuid]` shows what each attempt actually did.

---
## `~/.hermes/` layout (reference)

What you will actually see on the reference host:

```
~/.hermes/
├── SOUL.md                     # agent identity — personality, tone, how it behaves
├── config.yaml                 # the deployment: provider, memory, skills, curator,
│                               #   delegation, kanban, approvals, gateway, channels
├── config.yaml.bak-ollama      # pre-migration backup (inert; not the live config)
├── profile.yaml
├── memories/                   # MEMORY.md + USER.md (the two bounded memory files)
├── skills/                     # per-profile skill library; custom/ subtree is this deployment's set
├── shared-skills/              # cross-profile skill layer
├── hermes-agent/               # the Hermes runtime itself
├── tools/                      # pinned tool runtimes: python, node, ripgrep, ffmpeg,
│                               #   chromium, agent-browser, cua-driver, uv
├── vault/                      # encrypted secrets (vault.json.enc + vault.key)
├── sessions/                   # session transcripts (retention_days: 90, auto-pruned)
├── cron/                       # scheduled jobs + durable run history
├── kanban.db                   # the task board (SQLite)
├── kanban/                     # board card files
├── gateway*                    # gateway state, pid, socket
├── state.db                    # agent state
├── logs/                       # runtime logs
└── backups/                    # update-path backups (backup_keep: 5)
```

The two things that matter most are `config.yaml` (the whole deployment is pinned here) and `SOUL.md` + `memories/` (the agent's persistent self). Everything else is runtime or tooling.

## Scripts

### `scripts/llama_agent_client.py`

A minimal, zero-dependency (Python stdlib only) OpenAI-compatible client for the local llama-swap router. This is the llama-era replacement for the Ollama-era client script, and it is the reference for how to talk to `http://127.0.0.1:8080/v1` from any Python code.

```bash
# one-shot completion on the default model
python3 scripts/llama_agent_client.py --model qwen3.8:27b "Reply with exactly one word: OK"

# pick the delegation-lane model explicitly
python3 scripts/llama_agent_client.py --model qwen3-coder:30b "Write a one-line Python fizzbuzz"
```

It hits `POST /v1/chat/completions` against `http://127.0.0.1:8080/v1`, sends the sentinel `api_key` (`local`), and prints the assistant message. If you only need a raw check that the router + model are answering:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer local' \
  -d '{"model":"qwen3.8:27b","messages":[{"role":"user","content":"Reply with exactly one word: OK"}]}'
```

**Expected outcome.** `OK` on stdout, exit 0. If it errors with a connection refused, the router is down (see [Troubleshooting](#troubleshooting)); if it errors with a 404 model, the name is not registered in `llama-swap.yaml`.

### Ollama → llama.cpp migration scripts (reference)

On a machine that *has* Ollama models to migrate, the two helpers live beside the serving stack (on the reference host they are under `~/Dev/llm/`):

- `extract-ollama-model.sh` — pull the raw GGUF weights out of an Ollama store so they can be served by llama-swap.
- `convert-ollama-models.sh` — batch the conversion for the models you actually want to keep.
- `check-models.sh` — sanity-check the extracted/converted GGUFs before registering them.

The reference host has already been through this — its GGUFs are in `~/Dev/llm/models/` and its two live models are registered in `llama-swap.yaml`. Treat these scripts as the *migration* path, not part of steady-state operation.

---
## Sources

- **llama.cpp** — the inference engine (`llama-server`) and GGUF runtime that llama-swap wraps. <https://github.com/ggml-org/llama.cpp>
- **llama-swap** — the OpenAI-compatible router with TTL-based model swapping that this deployment runs. <https://github.com/mostlygeek/llama-swap> (verify the canonical repo URL before depending on it — the local install is `v265`, `llama-swap --version`)
- **Hermes Agent** — the agent runtime, CLI, and the docs behind this guide. <https://hermes-agent.nousresearch.com/docs>
- **Local reference (the authoritative source for this file):** `~/.hermes/config.yaml`, `~/Dev/llm/llama-swap.yaml`, `~/.config/systemd/user/llama-swap.service`, `~/.hermes/skills/custom/` — this README documents those, not the other way around.

