# Hermes Light Implementation — Setup & Backup Guide

A single-machine, 100% local deployment of [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) on top of [Ollama](https://docs.ollama.com/). No external model provider for the default, fallback, or sub-agent lanes: every inference call terminates on the local host.

This guide is a working recipe, not marketing copy. It documents a reference deployment that is known to run, lists the exact values that make it work, and gives you the commands to reproduce, back up, restore, and maintain it. It is deliberately scoped to the software surface: the host, the two runtimes (Ollama + Hermes), and the Hermes features that are actually enabled — memory, skills, the skill curator, sub-agent delegation, the kanban dispatcher, and scheduled cron jobs.

**Audience.** Operators who can drive a terminal, edit YAML by hand, and read a config file to understand what they are shipping.

**Units.** Metric throughout — °C for temperature, GiB/TiB for storage and memory, km/h where a speed matters.

## In this guide

1. [Prerequisites & Bill of Materials](#prerequisites--bill-of-materials)
2. [Reference host spec](#reference-host-spec)
3. [Setup — step by step](#setup--step-by-step)
   - [3.1 Install Ollama and pull the models](#31-install-ollama-and-pull-the-models)
   - [3.2 Install Hermes Agent](#32-install-hermes-agent)
   - [3.3 Wire the local provider into `config.yaml`](#33-wire-the-local-provider-into-configyaml)
   - [3.4 Memory, skills, and the curator](#34-memory-skills-and-the-curator)
   - [3.5 Sub-agent delegation](#35-sub-agent-delegation)
   - [3.6 Kanban dispatcher](#36-kanban-dispatcher)
   - [3.7 Scheduled cron jobs](#37-scheduled-cron-jobs)
4. [Ongoing operation](#ongoing-operation)
5. [Optimizations & pro-tips](#optimizations--pro-tips)
6. [Safety & maintenance](#safety--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Shipped code: `ollama_agent_client.py`](#shipped-code-ollama_agent_clientpy)
9. [Repository structure](#repository-structure)
10. [Sources](#sources)

---

## Prerequisites & Bill of Materials

Before you start, make sure every item below is true. This is the shortest path to a working stack — do not skip the disk check.

### Hardware

- [ ] A 64-bit x86-64 machine with **at least 46 GiB of RAM** (the 27 B and 30 B class models both want to live in system memory on this GPU-free path).
- [ ] **At least 1.3 TiB of NVMe**, with roughly **580 GiB free** at install time — `qwen3.8:27b` is ~17 GB, `qwen3-coder:30b` is ~18 GB, and you want headroom for `gemma4:31b` and `devstral-small-2:24b` if you enable them.
- [ ] An **AMD GPU in the RX 7900 class (Navi 31)** or a discrete card with AMDGPU/ROCm drivers present. On an AMD iGPU-only box the 27 B/30 B path still works, slower; the GPU is what keeps the 30 B class responsive.
- [ ] A **sustained network link** for the initial model pulls and the `ollama pull` resume logic. After pulls, the model calls are loopback, not the internet.

### Software & accounts

- [ ] A Linux distro the Ollama installer supports. Reference: **CachyOS**.
- [ ] Outbound HTTPS (for `curl | bash` on first setup).
- [ ] No API keys, no cloud accounts. The whole point is that there is nothing to sign in to.

### Expected state when you're done

You can run `hermes` from a shell and it answers using a model that is already on disk. `ollama ps` shows the model loaded on `127.0.0.1:11434`. Killing network access to the outside world does not change any of that.

---

## Reference host spec

These are the values from the system this guide reproduces. Treat them as the *floor*, not the ceiling — the software runs the same on anything that meets the bill of materials above.

| Component | Detail |
|---|---|
| OS | CachyOS Linux |
| Kernel | `7.2.8-1-cachyos` |
| Architecture | `x86_64` |
| CPU | AMD Ryzen 7 9800X3D (8 cores / 16 threads) |
| RAM | 46 GiB |
| Storage | NVMe, 1.3 TiB total, ~579 GiB free at the reference point in time |
| GPU | AMD Radeon RX 7900 class (Navi 31) + an AMD Granite Ridge iGPU on-chip |
| BIOS | American Megatrends (AMI), v3854 |
| Runtime 1 | Ollama 0.34.4, listening on `127.0.0.1:11434` |
| Runtime 2 | Hermes Agent v0.21.5 (Python 3.14) |
| Models in use | `qwen3.8:27b` (default), `qwen3-coder:30b` (secondary) |
| Models reserved | `gemma4:31b` (compression + title aux lane), `devstral-small-2:24b` (sub-agent delegation) |
| Timezone | `Europe/Paris` |

Two things to notice in this spec. First, the machine has 16 logical CPUs and 46 GiB of RAM — enough that the two ~17-18 GB models coexist without swapping, even before the GPU kicks in for the 30 B class. Second, the "reserved" models are wired into Hermes's `auxiliary.*` and `delegation` blocks but you do not always need to pull them; the guide shows exactly which lane each one feeds, so you can decide to pull them yourself (see [3.3](#33-wire-the-local-provider-into-configyaml)).

---
## Setup — step by step

Every step below is imperative, and each step ends with what you should see once it is done. Work through them in order.

### 3.1 Install Ollama and pull the models

Ollama is a single binary that hosts the models locally. Use the official installer.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Expected outcome.** `ollama --version` reports the version (the reference install is `0.34.4`). `systemctl is-active ollama` reads `active`. `curl http://127.0.0.1:11434/api/version` returns `{"version":"0.34.4"}`.

Pull the models that the Hermes config uses, in the order Hermes will load them. Each pull is resumable.

```bash
ollama pull qwen3.8:27b
ollama pull qwen3-coder:30b
# optional — used by compression aux + title + sub-agent delegation
ollama pull gemma4:31b
ollama pull devstral-small-2:24b
```

**Expected outcome.** `ollama list` shows the models with sizes (~17 GB for `qwen3.8:27b`, ~18 GB for `qwen3-coder:30b`). `ollama ps` is empty until the first inference, which is normal — Ollama loads a model on first call, not at startup.

Sanity-check that one model actually runs a forward pass, so you have a local model you can trust before you wire Hermes to it:

```bash
ollama run qwen3.8:27b "Reply with the single word OK."
```

**Expected outcome.** You get `OK` back within a few seconds of the first-token delay. On a weak CPU path the first-token delay is longer; the RX 7900 class path should land it under a couple of seconds.

### 3.2 Install Hermes Agent

Hermes ships as a CLI plus a Desktop app. For a pure headless setup the CLI is enough.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

**Expected outcome.** `hermes --version` reports the version (the reference install is `v0.21.5`). `hermes setup` runs onboarding; for a local-only deployment you do not need a portal OAuth — you are not calling an external provider. If you want the desktop companion, follow the install guide at <https://hermes-agent.nousresearch.com/docs/getting-started/installation>.

### 3.3 Wire the local provider into `config.yaml`

`$HOME/.hermes/config.yaml` is the single file that pins the deployment to Ollama. The three things to get right are:

1. The **`model`** block points at `http://127.0.0.1:11434/v1` and uses `chat_completions` (Ollama's OpenAI-compatible layer).
2. The **`providers.<name>`** block declares the endpoint explicitly so it resolves on every boot.
3. The **`auxiliary.*`** and **`delegation`** blocks reference `http://localhost:11434/v1` (same host, different alias is fine) and the aux/sub models you pulled in [3.1](#31-install-ollama-and-pull-the-models).

The reference slice of `config.yaml` that makes the whole thing run 100% locally:

```yaml
model:
  default: qwen3.8:27b
  base_url: 'http://127.0.0.1:11434/v1'
  provider: local-127.0.0.1:11434
  api_mode: chat_completions
  api_key: ollama            # value is unused by Ollama; presence is required

providers:
  local-127.0.0.1:11434:
    api: http://127.0.0.1:11434/v1
    name: Local (127.0.0.1:11434)
    api_key: ollama
    default_model: qwen3.8:27b          # change to gemma4:31b if you prefer the fallback as primary

auxiliary:
  compression:
    model: gemma4:31b
    base_url: http://localhost:11434/v1
    api_key: ollama
  title:
    model: gemma4:31b
    base_url: http://localhost:11434/v1
    api_key: ollama

delegation:
  model: devstral-small-2:24b
  base_url: http://localhost:11434/v1
  api_key: ollama
  inherit_mcp_toolsets: true
  max_iterations: 60
  child_timeout_seconds: 600
  subagent_auto_approve: false
```

Notes on the values:

- `api_key: ollama` is a sentinel — Ollama ignores it, and Hermes refuses a local provider without some key string on `api_mode: chat_completions`.
- `inherit_mcp_toolsets: true` on `delegation` is what lets a sub-agent inherit the same tool set you enabled in the parent, so the delegation lane is not a stripped-down sandbox.
- `max_iterations: 60` and `child_timeout_seconds: 600` are the guardrails for the sub-agent lane: 60 tool calls or 10 minutes, whichever hits first. See [5](#optimizations--pro-tips) for the reasoning.

After writing the config, validate that Hermes reads it back:

```bash
hermes config get model.default
hermes config get model.base_url
hermes config get delegation.model
```

**Expected outcome.** `qwen3.8:27b`, `http://127.0.0.1:11434/v1`, and `devstral-small-2:24b` respectively. If any of these is empty or points at an external URL, fix the YAML before moving on — the rest of this guide assumes all three are correct.

---
### 3.4 Memory, skills, and the curator

Hermes keeps two persistent memory files and a self-maintaining skill library. These are what make the deployment "compound" over time — the model of the operator and the set of operating manuals both grow across sessions.

**Memory.** Two files, each with a hard character limit so that memory stays a signal, not a dump:

| File | Purpose | Char limit |
|---|---|---|
| `memories/MEMORY.md` | Agent's own notes: environment facts, standing conventions with no task home | 2200 |
| `memories/USER.md` | Operator profile: who they are, durable preferences, working agreements | 1375 |

The limits live in `config.yaml`:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
```

**Expected outcome.** After the first few sessions, `memories/MEMORY.md` and `memories/USER.md` are non-empty and well under their limits. When a limit approaches it, the agent is expected to prune stale entries rather than silently overflow — treat a near-full memory file as a maintenance signal, not a failure.

**Skills.** Skills are Markdown operating manuals that load only when a task matches. Two directories matter:

- `~/.hermes/skills/` — the per-profile skill library.
- `~/.hermes/shared-skills/` — the cross-profile layer, referenced explicitly:

```yaml
skills:
  external_dirs:
    - ~/.hermes/shared-skills
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

### 3.5 Sub-agent delegation

Delegation lets Hermes spawn a shorter, cheaper sub-run for a bounded sub-task, using a separate (usually smaller) model while inheriting the parent's tool set. The reference block is in [3.3](#33-wire-the-local-provider-into-configyaml):

```yaml
delegation:
  model: devstral-small-2:24b
  base_url: http://localhost:11434/v1
  api_key: ollama
  inherit_mcp_toolsets: true
  max_iterations: 60
  child_timeout_seconds: 600
  max_concurrent_children: 1
  max_spawn_depth: 1
  orchestrator_enabled: false
  subagent_auto_approve: false
```

The choices encode a specific safety posture:

- `max_concurrent_children: 1` and `max_spawn_depth: 1` — one sub-agent, one level deep. No fan-out, so a delegation call cannot multiply load on a single machine.
- `subagent_auto_approve: false` — sub-agents do not self-grant new permissions.
- `orchestrator_enabled: false` — by default, delegation is a scoped assist, not a recursive planning loop.
- The sub model (`devstral-small-2:24b`) is deliberately smaller than the default. This is the whole economic point: heavy reasoning stays on the 27 B/30 B model, mechanical sub-tasks use the 24 B model, and you never pay cloud egress for either.

**Expected outcome.** When a task is delegated, `hermes` runs it in an isolated context, and only the sub-run's summary returns to the parent. A timeout at 600 s or a trip of `max_iterations: 60` aborts cleanly and reports `timed_out` — it does not hang.

### 3.6 Kanban dispatcher

The kanban is a small shared task board (a SQLite DB under `~/.hermes/`) that turns multi-step work into discrete cards, each with an owner and a lifecycle. On the reference deployment it dispatches inside the gateway:

```yaml
kanban:
  dispatch_in_gateway: true
  dispatch_interval_seconds: 60
  failure_limit: 2
  worker_log_rotate_bytes: 2097152     # 2 MB
  worker_log_backup_count: 1
  auto_decompose: false
```

- `dispatch_in_gateway: true` — the dispatcher runs in the gateway process, so it ticks even when no interactive session is open.
- `dispatch_interval_seconds: 60` — the board is scanned every 60 s; a ready card is claimed on the next tick.
- `failure_limit: 2` — a worker that crashes twice is not retried a third time blindly.
- `auto_decompose: false` — tasks are created explicitly. Nothing is auto-split without intent.

**Expected outcome.** A ready card moves `todo → ready → running → done` (or `→ blocked` on a genuine ambiguity). `failure_limit: 2` is the trip-wire: two consecutive crash loops on the same card is the point at which a human is expected to look, not the dispatcher.

### 3.7 Scheduled cron jobs

Cron jobs run a fixed prompt on a schedule and deliver the result to a channel. Two example jobs (both on the reference machine, both healthy):

| Job | Schedule | What it does | Delivery |
|---|---|---|---|
| Weekly events | `0 15 * * 5` (Fri 15:00, Europe/Paris) | Search the most popular upcoming local events over the next 7–10 days, cross-checked against the weather; return a ranked list of 8–10 | `telegram` |
| Daily briefing | `45 8 * * *` (08:45 daily) | Curate the top ~10 high-impact news stories, each with a direct HTTPS link | `telegram` / `origin` |

To inspect the live set:

```bash
hermes cron list
hermes cron runs [job-uuid]        # durable execution history; optional job filter
```

**Expected outcome.** `cron list` shows the active jobs with schedule and delivery target. `cron runs` shows each attempt with its status (`completed`, `unknown`, …), so a job that "ran" on the surface but died mid-flight is visible. A growing run of non-`completed` statuses on one job is what you fix, not the jobs that are quiet.

> **Scheduling note.** Jobs created from a plain CLI session are local-only — their output is stored and can be listed, but there is no push channel unless the job's `deliver` targets a gateway-connected platform (e.g. `telegram`). Set `deliver` explicitly if you want the job to reach your phone.

---
## Ongoing operation

Once the deployment is up, steady-state operation is mostly about keeping the two persistent things healthy: the memory files and the skill library.

- **Memory.** Watch the two files against their limits. A near-full `MEMORY.md` (≈ 90% of 2200 chars) is a cue to consolidate old entries into a skill, not to push past the cap. Task-specific knowledge belongs in a skill (which loads on demand); only facts that apply to *every* session belong in memory.
- **Skills.** Let the curator do the compaction on its 168-hour cycle. If you notice a skill you use weekly getting archived, that is a bug in the signal — pin it or merge it into a living skill so it stops reading as "stale".
- **Model cache.** The pulled models live in Ollama's store (on the reference host, under `~/.ollama/models`). They do not re-download on restart. If you are adding models, pull them explicitly and verify with `ollama list` before pointing any config block at them.
- **Gateway.** The dispatcher and messaging run inside the gateway process. Restarting the gateway is the operation that can interrupt in-flight kanban work — restart it deliberately, not reflexively. `sessions.retention_days: 90` and the auto-prune settings keep session history bounded on their own.

A minimal steady-state loop:

```bash
# Is the local model reachable?
curl -s http://127.0.0.1:11434/api/version
ollama list

# Is the agent answering on the local model?
hermes --version
hermes config get model.default

# Is the gateway (dispatcher + messaging) alive?
hermes gateway status        # or: pgrep -af "gateway run"
```

## Optimizations & pro-tips

- **Let the big model do the heavy lifting.** Keep the default at the 27 B/30 B class and park `devstral-small-2:24b` in the delegation lane. Splitting reasoning (large) from mechanical sub-tasks (small) is the single biggest lever for keeping a local stack responsive.
- **Co-locate the two big models.** With 46 GiB, `qwen3.8:27b` (~17 GB) and `qwen3-coder:30b` (~18 GB) fit in system memory side by side. If your box is smaller, drop one big model and rely on the smaller delegation model plus a single default.
- **`max_iterations: 60` / `child_timeout: 600s` are cheap insurance.** A sub-agent pinned by either bound reports `timed_out` instead of wedging the parent. Raise them only for a lane you have measured to be slow, and measure again.
- **One concurrent sub-agent, one spawn depth.** `max_concurrent_children: 1` and `max_spawn_depth: 1` keep a single machine from multiplexing model load. Fan out only when you have real headroom — the reference deployment has 16 logical CPUs, which is why it stays modest.
- **The curator's `keep: 5` backup is your undo.** Before a manual curation pass you did not ask for, snapshot the skill dir yourself; after the pass, verify the archive did not take a live skill.
- **`api_key: ollama` is a sentinel, not a credential.** Do not paste a real secret into the local provider block — Ollama ignores it and you would be leaking a real key into an insecure config.
- **Point cron `deliver` at a real channel.** A job with no `deliver` runs and stores output but reaches no one. `telegram`/`origin` is the difference between a job that works and a job that works *and tells you it worked*.

## Safety & maintenance

**Credentials.** This deployment uses no external API keys. The only key-like value in the local provider block is the `ollama` sentinel (see above). If you later add a genuine provider, treat that key as sensitive: keep it in the config (which is `0600`), never in a repo, and rotate it on any suspicion of exposure. Never let secrets land in `memories/` or in a skill — those files are intended to be shared and read out.

**The local-only boundary is the security model.** "100% local" means the model, the fallback, and the sub-agent all resolve to `127.0.0.1:11434`. If a config block points at a non-local `base_url`, you have left the boundary — either fix it or decide consciously that that lane is fine calling out.

**Destructive ops are gated.** On this deployment, approvals are `mode: manual`, `cron_mode: deny`, and `destructive_slash_confirm: true`. A cron job cannot force a destructive action, and a destructive interactive command asks before it runs. Leave all three on.

**Periodic maintenance.**

- [ ] **Weekly** — confirm curator backups exist (`ls` the curator backup dir; expect up to 5 generations) and that `MEMORY.md` / `USER.md` are under their char limits.
- [ ] **Monthly** — `ollama list` and diff expected vs. pulled models; `hermes --version` against the version you intend; a quick `curl` to `127.0.0.1:11434` to confirm the local stack is still alive.
- [ ] **On any update** — the update path keeps `backup_keep: 5`; restore from the newest good backup if a version bump misbehaves rather than chasing a broken install.

**Back up the three things that are hard to reproduce:** `~/.hermes/config.yaml` (the whole deployment is pinned here), `~/.hermes/memories/` (MEMORY.md + USER.md), and the custom skills under `~/.hermes/skills/` + `~/.hermes/shared-skills/`. The models are re-pullable; these are not.

## Troubleshooting

| Symptom | First check | Fix |
|---|---|---|
| Hermes errors with "provider not found" / no model | `curl -s http://127.0.0.1:11434/api/version` | Ollama is down → `systemctl start ollama`. If it is up, the `model.base_url` in `config.yaml` is wrong. |
| Answers are empty or instant-refusal | Which model is being called (`hermes config get model.default`) | That model is not pulled → `ollama pull <model>`. |
| A delegation sub-run hangs then dies | `delegation.child_timeout_seconds` and `max_iterations` | Expected `timed_out`. Increase the bound only if you have measured the lane as genuinely slow; otherwise let it fail fast. |
| A kanban card is stuck in `running` | The worker is mid-operation, or it crashed | Give it a tick (~60 s `dispatch_interval_seconds`). If it is a crash loop, `failure_limit: 2` will stop the retry — look at the worker log (rotates at 2 MB). |
| A cron job "runs" but you never see output | `deliver` field on the job | Local-only job. Set `deliver` to `telegram` or `origin` (a gateway-connected channel). |
| A skill I use keeps getting archived | curator `archive_after_days: 30` + stale 14 d | The skill reads as unused. Re-trigger it, pin it, or merge it into a living skill. |
| Disk creeping up | `du -sh ~/.ollama/models` | Expected if you added models. `ollama list` and prune what you stopped using. |
| Model loads correctly but is slow on first call | First-token delay | Normal for a first call (Ollama cold-loads the model). Subsequent calls are fast. If *every* call is slow, the model is on the CPU path — check the GPU driver/ROCm path. |

When in doubt, read the config back with `hermes config get <key>` and compare against the reference values in [3.3](#33-wire-the-local-provider-into-configyaml) — most "mystery" behaviour in this stack is a single wrong `base_url` or a model the config references that was never pulled.

## Shipped code: `ollama_agent_client.py`

`code/ollama_agent_client.py` is a single-file Python client for talking to a local Ollama server that does the context management *right* rather than leaving it to Ollama's defaults. It is written against the reference deployment above (targeting `qwen3.8:27b` on a GPU that mostly holds weights, not context) and has exactly one runtime dependency: `requests`.

What it handles, and why each matters on a local box:

1. **Explicit context budgeting.** Instead of trusting Ollama's VRAM-tier auto-sizing (which ignores how much of your VRAM the weights already occupy), it takes a `num_ctx` you set, computes a safe `num_predict` from what's left, and holds a `safety_margin` for formatting and tokenizer error. It then verifies against reality via `/api/ps` and logs a loud warning if the server quietly clamped the context smaller than you asked — the most common silent failure on an under-provisioned card.
2. **Sliding-window history.** Each call keeps only the most recent messages that fit the budget. Older messages are not lost silently: an optional `summarizer` callable compresses them into a running `[Summary of earlier conversation]`, or a compact truncated note is written as a fallback, so continuity survives the trim.
3. **Truncation recovery.** If the model hits the length limit mid-answer (`done_reason == "length"`), it detects it, appends a "continue exactly where you left off" turn, recomputes the budget against the now-larger prompt, and retries — up to `max_continuations` rounds — before reporting that the reply may still be incomplete.
4. **Usage accounting.** Every call records estimated prompt tokens, the `num_predict` used, the `done_reason`, and the continuation count. `print_usage_summary()` gives you a roll-up so you can see where the token budget actually goes.

Two more levers it exposes for the thinking-capable models in this stack:

- **`think`** — set per-client or per-call. Reasoning tokens count against `num_predict` just like the visible answer, so leaving thinking off for routine/tool turns frees real output budget and is the difference between a truncated and a complete reply.
- **`keep_alive`** — how long Ollama keeps the model resident after a call. Reloading an ~17–18 GB model from disk between agent turns is expensive; a `30m` window keeps it warm across a session.

Minimal usage:

```python
from ollama_agent_client import OllamaAgentClient

client = OllamaAgentClient(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    num_ctx=32768,          # set explicitly — don't rely on auto-sizing
    keep_alive="30m",       # keep the big model resident between turns
    think=False,            # flip to "low"/True only when you need reasoning
    system_prompt="You are a concise, technically precise assistant.",
)

reply = client.chat("Explain the TCP three-way handshake, including edge cases.")
print(reply)
client.print_usage_summary()
```

**Before you run it**, start `ollama serve` with `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` to roughly halve KV-cache memory, and check `ollama ps` for a 100% GPU `PROCESSOR` split at your chosen `num_ctx`. The module's `__main__` block is a working starting point tuned for a 24 GiB VRAM / 48 GiB RAM host — adjust `num_ctx` to what `ollama ps` actually reports for your card. The default token counter is a heuristic budgeting aid (roughly 4 chars/token); call `client.set_tokenizer(...)` with a real tokenizer for precise accounting.

The full interface and tunables live in the module's docstrings — `OLLAMA_*` environment variables are server-level (set when starting `ollama serve`), while `num_ctx`, `think`, and `keep_alive` are per-request.

## Repository structure

If you put this deployment in a repo, this is a sane layout. Keep it small — the whole point is that the entire thing is one config file plus two memory files plus a skill library.

```
hermes-docs/
├── README.md                     # this guide
├── config/
│   └── config.example.yaml       # sanitized copy of ~/.hermes/config.yaml (sentinel keys only)
├── notes/
│   ├── MEMORY.example.md         # template, NOT your real MEMORY.md
│   └── USER.example.md           # template, NOT your real USER.md
└── skills/                        # only the custom skills worth sharing
    └── ...
```

Two rules for what goes in the repo. First, **no live secrets** — the config that ships has the `ollama` sentinel, never a real provider key. Second, **no identity** — the example memory files are templates describing *what to put there*, not the operator's actual profile.

## Sources

The two primary documents this guide is grounded in:

1. Hermes Agent documentation — <https://hermes-agent.nousresearch.com/docs/>
   - Installation: <https://hermes-agent.nousresearch.com/docs/getting-started/installation>
   - Configuration: <https://hermes-agent.nousresearch.com/docs/user-guide/configuration>
2. Ollama documentation — <https://docs.ollama.com/>
   - Run a model locally: <https://docs.ollama.com/quickstart>

The host spec, model names, and the Hermes feature values (memory limits, curator cycle, delegation bounds, kanban bounds) are taken from the live reference deployment this guide reproduces.
