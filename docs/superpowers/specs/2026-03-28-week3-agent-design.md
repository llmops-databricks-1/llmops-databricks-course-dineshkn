# Week 3: Causal Inference Research Agent — Design Spec

**Date:** 2026-03-28
**Branch:** week3
**Status:** Approved

---

## Overview

Build a stateful causal inference research agent that mirrors the Week 3 reference material (3.1–3.3 + MCP) adapted to our causal inference paper corpus. The agent can answer multi-turn questions about causal inference papers using vector search, with session history persisted in Lakebase.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User (Notebook cell)                   │
└────────────────────────┬────────────────────────────────┘
                         │ message + session_id
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     SimpleAgent                          │
│  1. load history from LakebaseMemory                    │
│  2. call LLM with tools available                       │
│  3. if tool_call → execute via ToolRegistry             │
│  4. feed result back to LLM                             │
│  5. save updated history to LakebaseMemory              │
│  6. return final answer                                 │
└────────┬────────────────────┬───────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────────────────────┐
│  ToolRegistry   │  │        LakebaseMemory            │
│                 │  │  (managed PostgreSQL on DBX)     │
│  search_papers  │  │  session_messages table          │
│  (custom fn)    │  │  keyed by session_id             │
└────────┬────────┘  └─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         causal_inference_chunks_index                    │
│    (Vector Search on mlops_dev.dineshka)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Notebooks

All notebooks live under `notebooks/` with names matching the reference exactly.

### `3.1_custom_functions_tools.py`
Introduces tool calling with custom Python functions.

1. Setup — workspace client, OpenAI client, load config
2. Tool spec format — educational section (OpenAI JSON format)
3. Define `search_papers(query, num_results, year_filter)` — hits `causal_inference_chunks_index`
4. Wrap in `ToolInfo`, build `ToolRegistry`
5. Test the tool with 3 causal inference queries
6. `SimpleAgent` class — LLM + tool loop (stateless, memory introduced in 3.1b)

### `3.1b_simple_rag.py`
Full stateful RAG agent using tools from 3.1 + session memory.

1. Setup — workspace client, OpenAI client, `LakebaseMemory` init (Lakebase instance creation here)
2. `retrieve_documents()` + `build_rag_prompt()` — standalone RAG pipeline
3. `rag_query()` — single-turn RAG
4. `SimpleRAG` class — integrates `LakebaseMemory`, loads/saves per `session_id`
5. Multi-turn demo — follow-up questions showing history is preserved

### `3.3_session_memory_lakebase.py`
Standalone deep-dive on Lakebase session memory.

1. Create Lakebase instance
2. `LakebaseMemory` init + connection string
3. Save/load messages demo
4. `chat_with_memory()` standalone function
5. Session persistence demo across multiple calls

### `3.2_mcp_integration.py`
Rebuilds the 3.1b agent using MCP instead of custom functions.

1. What is MCP, MCP vs custom functions (educational prose)
2. Connect to Vector Search MCP server (our index URL)
3. `create_mcp_tools()` — load tools automatically from MCP
4. Show tool specs generated automatically
5. `SimpleAgent` using MCP tools + `LakebaseMemory` (same memory, different tool source)
6. Side-by-side comparison: custom function vs MCP for same query

---

## Package Changes

### New file: `src/causal_inference_curator/memory.py`
Ported from `arxiv_curator/memory.py` in the reference. Changes:
- Lakebase instance name: `causal_inference` (was `arxiv`)
- No interface changes — `LakebaseMemory` API is identical

```python
memory = LakebaseMemory(workspace_client)
messages = memory.load_messages(session_id)   # List[dict]
memory.save_messages(session_id, messages)    # append-only
```

PostgreSQL table (auto-created on first use):
```sql
session_messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    message_data JSONB NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Existing file: `src/causal_inference_curator/mcp.py`
No changes needed. Used as-is by `3.2_mcp_integration.py`.

### `pyproject.toml`
Add `psycopg2-binary` to dependencies (required by `LakebaseMemory`).

---

## Tool Interface

### `search_papers`

```python
def search_papers(query: str, num_results: int = 5, year_filter: int | None = None) -> str:
    """Search causal inference research papers using semantic vector search.
    Returns JSON string: [{chunk_text, source, score}, ...]
    """
```

Registered in `ToolRegistry` with description: `"Search causal inference research papers using semantic similarity. Use this to find relevant papers before answering any research question."`

---

## Agent Loop

Identical in both `3.1b` (custom tools) and `3.2` (MCP tools):

```
load_messages(session_id)
  → [system_prompt] + messages + tool specs → LLM
  → tool_call? → execute (ToolRegistry or MCP) → append result → LLM
  → text response → append to messages
  → save_messages(session_id, messages)
  → return answer
```

The only difference between the two notebooks is the tool source — `ToolRegistry` vs `create_mcp_tools()`. Memory behavior is identical.

---

## Config Changes

### `project_config.yml`
Update `system_prompt` under `dev` (and `acc`/`prd`):

```yaml
system_prompt: >
  You are a causal inference research assistant. Answer questions using only
  the provided research papers. Always cite the source document when
  referencing specific claims.
```

---

## What We Explicitly Skip (for now)

- `3.2b_genie.py` — Genie Space integration
- `3.4_spn_authentication.py` — Service Principal auth
- `3.5_spn_authentication_in_action.py` — SPN in action
- `3.6_uc_function_example.py` — Unity Catalog functions

These are natural extensions once the core agent is working.

---

## Success Criteria

1. `3.1_custom_functions_tools.py` runs end-to-end on Databricks, `search_papers` returns results from our index
2. `3.1b_simple_rag.py` demonstrates multi-turn conversation with history preserved across cells
3. `3.3_session_memory_lakebase.py` shows save/load of session messages independently
4. `3.2_mcp_integration.py` connects to the Vector Search MCP server and runs the same queries as 3.1b
5. All notebooks pass CI (ruff lint/format)
6. PR created from `week3` branch
