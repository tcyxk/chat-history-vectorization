# Hermes Chat History Vectorization System

A RAG-based chat history system for [Hermes Agent](https://github.com/nousresearch/hermes-agent), using Qdrant as the vector database and a pre-gateway-dispatch hook for semantic retrieval.

---

## Architecture

```
User Message
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  pre_gateway_dispatch Hook (rag_context/handler.py)   │
│  1. Embed user message → 1024-dim vector            │
│  2. Search Qdrant top-k semantically similar chunks │
│  3. Inject formatted context → rewrite user message │
└──────────────────────────────────────────────────────┘
    │
    ▼  Rewritten message (with relevant history injected)
Gateway ──→ Agent ──→ Reply
```

**Three layers:**

| Layer | Component | Description |
|-------|-----------|-------------|
| Storage | Qdrant | Vector DB on cloud (or local) |
| Ingestion | `migrate.py` + `realtime.py` | Historical batch migration + real-time incremental write |
| Retrieval | `rag_context` hook | pre_gateway_dispatch hook injects context before each reply |

**Key design principles:**
- Single-process guarantee: only one data item in flight at any time
- Power-failure resilience: Qdrant WAL + `flush_interval_sec=1` → max 1 second data loss
- Multi-agent isolation: each agent has its own Collection, session dir, and checkpoint
- Silent operation: RAG hook produces no logs under normal conditions

---

## File Structure

```
.
├── SKILL.md                     # Full system documentation
├── README.md                    # This file
├── scripts/
│   ├── config.py.template       # Configuration template (copy and fill)
│   ├── embedding.py             # llama-embedding wrapper (BGE-M3 compatible)
│   ├── qdrant_client.py         # Qdrant HTTP API wrapper
│   ├── session_parser.py        # Hermes .msgpack session file parser
│   ├── text_splitter.py        # Long-text chunker (sentence boundary, 2000 chars)
│   ├── migrate.py               # Historical chat migration to Qdrant
│   ├── realtime.py              # Real-time incremental write from session files
│   └── watchdog.py              # Health check (Qdrant + llama-embedding)
└── hooks/rag_context/
    ├── HOOK.yaml                # Hook metadata (pre_gateway_dispatch event)
    └── handler.py               # RAG retrieval + context injection logic
```

---

## Quick Start

### Prerequisites

- Hermes Agent running on Linux (or Termux on Android)
- Qdrant v1.17+ running somewhere accessible (local or cloud)
- llama-embedding binary compiled for your platform
- BGE-M3 embedding model (FP16 or Q8_0 quantization)

### Step 1: Configure

```bash
cp scripts/config.py.template scripts/config.py
# Edit config.py and fill in your values:
#   QDRANT_HOST / QDRANT_PORT
#   COLLECTION_NAME
#   LLAMA_EMBEDDING_BIN
#   BGE_M3_MODEL
```

### Step 2: Create Qdrant Collection

```bash
curl -X PUT http://YOUR_QDRANT_HOST:6333/collections/chat_your_agent \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {"size": 1024, "distance": "Cosine"},
    "hnsw_config": {"m": 16, "ef_construct": 100},
    "on_disk_payload": true
  }'
```

### Step 3: Migrate Historical Chat

```bash
cd scripts
python migrate.py
# Restart-safe: resumes from checkpoint on re-run
```

### Step 4: Start Real-time Incremental Write

```bash
python realtime.py
# Continuously monitors session files and writes new messages to Qdrant
```

### Step 5: Enable RAG Hook

The hook auto-loads when Hermes Gateway starts. To manually restart:

```bash
hermes gateway run --replace
```

---

## Payload Schema (Qdrant)

```json
{
  "id": 1234000,
  "vector": [0.123, -0.456, ...],
  "payload": {
    "timestamp": "2026-05-07T19:10:23",
    "role": "user",
    "content": "Hello!",
    "session_file": "20260507_191023_abc123.msgpack",
    "msg_index": 0,
    "chunk_index": 0,
    "text_length": 6
  }
}
```

**Point ID design:** `point_id = msg_index * 1000 + chunk_index`
- `msg_index`: message index within the session file
- `chunk_index`: chunk index for long messages (0 if not split)
- Enables restart-safe resume: reads max point_id from Qdrant on startup

---

## Configuration Reference

See `scripts/config.py.template` for all options. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant HTTP port | `6333` |
| `COLLECTION_NAME` | Qdrant collection name | `chat_default` |
| `VECTOR_DIM` | Embedding vector dimension | `1024` (BGE-M3) |
| `CHUNK_SIZE` | Max characters per chunk | `2000` |
| `MAX_TEXT_LEN` | Max text length for embedding | `8000` |
| `POLL_INTERVAL` | Realtime poll interval (seconds) | `30` |

---

## Disclaimer

This is a custom extension for Hermes Agent, not an official feature. Configuration values, paths, and behaviors are specific to this deployment. Adapt the code to your environment before use.
