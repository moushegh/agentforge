# AgentForge

A fully local, GPU-accelerated multi-agent development platform. A team of five AI agents — TeamLead, WebResearcher, Developer, Tester, and Reviewer — collaborate to complete software tasks, driven by [AutoGen](https://github.com/microsoft/autogen) and connected to real tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Everything runs in Docker, no cloud required.

```
┌─────────────────────────────────────────────────────────┐
│  Browser  →  Web UI (port 8080)                         │
│                    ↓ SSE stream                         │
│             FastAPI web server                          │
│                    ↓ asyncio                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  AutoGen GroupChat                               │   │
│  │   TeamLead → WebResearcher → Developer           │   │
│  │            → Tester → Reviewer                  │   │
│  └─────────────┬──────────────────────────────────-─┘   │
│                ↓ MCP (SSE)                              │
│   ┌────────────┬─────────────┬────────────┐            │
│   │ Filesystem │     Web     │    Shell   │            │
│   │  :3001     │    :3002    │   :3003    │            │
│   └────────────┴─────────────┴────────────┘            │
│                ↓ OpenAI-compat API                      │
│          Ollama  (:11434)  + RTX 2060 GPU               │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Fully local** — no OpenAI key, no data leaving the machine
- **GPU-accelerated** — Ollama uses NVIDIA CUDA via the nvidia container runtime
- **Real tool use** — agents read/write files, fetch URLs, and execute shell commands through MCP servers
- **Live web UI** — submit tasks and watch each agent's messages stream in real time
- **Hot config** — swap models by editing one YAML line, no container rebuild needed
- **Persistent workspace** — all agent output lands in `./workspace/`, visible on the host immediately

---

## Stack

| Component | Technology |
|---|---|
| Agent framework | [AutoGen](https://github.com/microsoft/autogen) (pyautogen) |
| LLM runtime | [Ollama](https://ollama.com) |
| Default model | `devstral-small-2` (Mistral, code-optimised) |
| Tool protocol | Model Context Protocol (MCP) over SSE |
| MCP servers | filesystem, web fetch, shell exec, git, memory, time |
| Web backend | FastAPI + uvicorn + Server-Sent Events |
| Containers | Docker Compose |
| GPU | NVIDIA (WSL2 + nvidia-container-toolkit) |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Engine ≥ 24 | with Compose v2 (`docker compose`) |
| NVIDIA GPU | optional — works on CPU too, just slower |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | required for GPU passthrough |
| WSL2 (Windows) | Linux works identically; macOS uses CPU only |
| 16 GB RAM | recommended; 8 GB minimum (CPU-only) |

### GPU setup (one-time, Linux / WSL2)

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Register nvidia as the default Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

# Verify
docker info | grep "Default Runtime"   # should print: Default Runtime: nvidia
docker run --rm --runtime=nvidia nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

> **WSL2 note**: the `docker-compose.yml` already includes `LD_LIBRARY_PATH` and a volume mount for `/usr/lib/wsl` so CUDA libraries are visible inside the Ollama container.

---

## Bootstrap (clean install)

```bash
# 1. Clone
git clone <repo-url> agentforge
cd agentforge

# 2. Pull the default model (15 GB, one-time download)
docker run --rm -v ./ollama:/root/.ollama ollama/ollama pull devstral-small-2:latest

# (Optional) rename the directory if cloned as something else
mv ai-platform agentforge && cd agentforge

# 3. Start all services
docker compose up -d

# 4. Wait for the web server to be ready (~30 s for pip installs on first boot)
docker logs -f ai-web   # ready when you see: Uvicorn running on http://0.0.0.0:8080

# 5. Open the UI
xdg-open http://localhost:8080   # or just open it in your browser
```

That's it. No API keys, no extra config needed.

---

## Directory structure

```
agentforge/
├── docker-compose.yml          # all services
├── configs/
│   └── agent_config.yaml       # model + agent + MCP config (edit here to swap models)
├── autogen/
│   ├── main.py                 # LocalMultiAgentTeam — GroupChat orchestration
│   ├── web_server.py           # FastAPI backend, SSE streaming
│   ├── mcp_tools.py            # MCP client, tool registration for AutoGen
│   ├── requirements.txt        # Python deps for agents
│   ├── web_requirements.txt    # Additional deps for the web server
│   └── static/
│       └── index.html          # Single-page web UI
├── mcp-servers/
│   └── package.json            # MCP server npm deps
├── workspace/                  # ← agents write all output here (git-tracked)
├── memory/
│   └── memory.json             # Persistent agent knowledge graph
├── logs/                       # log files
└── ollama/                     # Ollama model storage (host-mounted)
    └── models/
```

---

## Common commands

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# View web server logs (agent output)
docker logs -f ai-web

# View Ollama logs (model loading, GPU detection)
docker logs -f ollama

# Check which model is loaded and GPU usage
docker exec ollama ollama ps
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader

# Rebuild just the web server after code changes
docker compose up -d --force-recreate web

# Run a task non-interactively (bypasses the web UI)
docker exec autogen python /app/main.py

# Pull a different model
docker exec ollama ollama pull llama3.1:8b

# See all workspace outputs
ls -lh workspace/
```

---

## Swapping the model

Edit one line in [configs/agent_config.yaml](configs/agent_config.yaml):

```yaml
models:
  local_llm:
    model: "llama3.1:8b"   # change this to any model pulled in Ollama
```

No restart required — the config is read fresh for every new task.

**Recommended models by use case:**

| Model | Size | Best for |
|---|---|---|
| `devstral-small-2:latest` | 15 GB | Code tasks, tool use, multi-step reasoning (default) |
| `llama3.1:8b` | 4.9 GB | Fast experimentation, low VRAM |
| `qwen2.5-coder:14b` | 9 GB | Code generation on 12 GB+ GPU |
| `mistral-nemo:latest` | 7.1 GB | Balanced speed/quality |

---

## MCP servers

| Container | Port | Tools | Used by |
|---|---|---|---|
| `mcp-filesystem` | 3001 | read/write/list files in `/workspace` | Developer, Tester |
| `mcp-web` | 3002 | `fetch` — retrieve any URL | WebResearcher |
| `mcp-exec` | 3003 | `execute_command` — run shell commands | Developer, Tester |
| `mcp-memory` | 3005 | knowledge graph (entities, relations, observations) | All agents |
| `mcp-git` | 3006 | full git operations on `/workspace` | Developer, Tester, Reviewer |
| `mcp-time` | 3007 | `get_current_time`, `convert_time` | WebResearcher |

---

## Agent team

| Agent | Role | MCP tools |
|---|---|---|
| **TeamLead** | Orchestrates the workflow, delegates with `@AgentName:` | `search_nodes`, `add_observations` (memory) |
| **WebResearcher** | Fetches URLs, summarises findings | `fetch`, `get_current_time`, memory |
| **Developer** | Writes, saves, and commits code | `read_file`, `write_file`, `execute_command`, `git_commit`, `git_diff`, memory |
| **Tester** | Writes and runs tests, reports results | filesystem, `execute_command`, `git_diff`, `git_log`, memory |
| **Reviewer** | Code review — bugs, style, correctness | `read_file`, `git_diff`, `git_log`, memory |

The pipeline always runs in order: **Research → Develop → Test → Review → TERMINATE**

---

## Architecture notes

- **MCP over SSE** — each MCP server exposes tools over a persistent SSE connection. `mcp_tools.py` connects at startup, discovers available tools, and registers them with the relevant AutoGen agents.
- **Text tool-call fallback** — smaller models sometimes emit tool calls as plain JSON text instead of using the structured `tool_calls` field. `detect_text_tool_call()` intercepts these and executes them correctly.
- **Live streaming** — `web_server.py` attaches a `_NotifyList` to the GroupChat `messages` list. Every `append()` pushes the message onto an `asyncio.Queue`; the SSE endpoint drains that queue to the browser in real time.
- **Speaker selector** — a custom `custom_speaker_selector` enforces stage ordering and routes tool-call responses back to the originating agent, preventing the chaos that AutoGen's default round-robin would cause.
