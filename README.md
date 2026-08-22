# Identity LLM Client

## The Problem

Identity tools in a federal environment need AI-assisted analysis — but identity data (SAML configs, authentication logs, federation metadata, application credentials) cannot be sent to cloud AI APIs. The constraint isn't optional: data that leaves the perimeter is data outside your control.

The solution is local inference. But building a separate LLM integration into each tool creates inconsistency and maintenance burden. This library solves that.

## What It Does

A shared Python client library for local LLM inference via Ollama. Used across 5+ production identity tools:

- **adfs-okta-migration-tool** — AI scan layer for configuration anomaly detection
- **Identity Intelligence Platform** — Cross-tool posture analysis and drift detection *(in development)*

**Key capabilities:**

- Simple query interface: send a prompt, get a response
- Configurable model selection (deep reasoning vs. fast queries)
- Designed for sensitive data: all inference runs locally, zero network egress
- Consistent error handling and timeout management across all tools

## Why It's Built This Way

- **Local-first**: Ollama serves models at `localhost:11434`. Zero cloud API calls. Sensitive identity data never leaves the host.
- **Shared library pattern**: One client, consistent behavior across all tools. Update inference logic in one place.
- **Model flexibility**: Different tasks need different models. The client abstracts model selection.

## Models

| Model | Use Case |
|---|---|
| Qwen 2.5 72B Q4_K_M | Primary — deep reasoning, SAML/policy analysis, complex queries |
| Qwen 2.5 Coder 32B | Code tasks, quick queries |
| Nomic Embed Text | RAG/semantic search *(planned)* |

## Infrastructure

- Ollama 0.15.4+ running as a persistent service (`localhost:11434`)
- Qwen 2.5 72B loaded into memory (~47GB) for instant first-query response
- Optional: Open WebUI for direct chat interface (`localhost:8080`)

## Tech Stack

- Python 3.11+
- Ollama (local model serving)
- Qwen 2.5 72B (primary reasoning model)

## Setup

1. Install Ollama: https://ollama.com

2. Pull the primary model:
```bash
ollama pull qwen2.5:72b
```

3. Start the Ollama service:
```bash
ollama serve
```

4. Import the client:
```python
from llm.client import LLMClient
```

## Status

**Production — Phase 1 complete.** Shared client in active use across the migration tool and coworker assistant. Identity Intelligence Platform integration in development.

---

## Part of

This library is a core component of the **Identity Intelligence Platform** — a suite of tools for AI-assisted identity posture analysis in federal environments.

**Related repos:**
- [adfs-to-okta-migration](https://github.com/wesglockzin/adfs-to-okta-migration)
