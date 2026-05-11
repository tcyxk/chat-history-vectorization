"""
RAG Context Hook — injects retrieved chat history into user messages.

Triggered by: pre_gateway_dispatch
Action: rewrites the user's message to include semantically relevant prior
        conversations from the Qdrant chat_history vector DB.
"""

import sys

sys.path.insert(0, "/data/data/com.termux/files/home/.hermes/scripts")
from config import COLLECTION_NAME
from embedding import get_embedding
from qdrant_client import search_points

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLECTION = COLLECTION_NAME   # from config.py
TOP_K = 5                     # retrieve top-5 relevant chunks
MAX_CONTEXT_CHARS = 2000      # max injected context length


def _format_context(results: list) -> str:
    """Format Qdrant search results into a readable context block."""
    lines = ["\n\n[相关对话回忆]", "─" * 40]
    for r in results:
        p = r.get("payload", {})
        role = p.get("role", "?")
        content = p.get("content", "")
        ts = p.get("timestamp", "")[:16]
        session = p.get("session_file", "")

        # Truncate long content
        if len(content) > 300:
            content = content[:300] + "..."

        lines.append(f"[{ts}][{role}] {content}")
    lines.append("─" * 40)
    return "\n".join(lines)


def handle(event_type: str, context: dict) -> dict | None:
    """
    pre_gateway_dispatch handler. Returns a rewrite dict if relevant
    prior conversations are found, otherwise None (pass-through).
    """
    # Only process user messages
    event = context.get("event")
    if event is None:
        return None

    text = getattr(event, "text", None) or ""
    text = text.strip()

    # Skip empty or very short messages
    if len(text) < 3:
        return None

    # Skip if text is a command
    if text.startswith("/"):
        return None

    try:
        # 1. Embed the user's message
        query_vec = get_embedding(text)
        if not query_vec:
            return None

        # 2. Search Qdrant for relevant history
        results = search_points(query_vector=query_vec, top_k=TOP_K)

        if not results:
            return None

        # 3. Format and inject context
        context_block = _format_context(results)
        if len(context_block) > MAX_CONTEXT_CHARS:
            context_block = context_block[:MAX_CONTEXT_CHARS] + "\n...(内容过长已截断)"

        rewritten = f"{context_block}\n\n你当前的问题是：{text}"

        return {
            "action": "rewrite",
            "text": rewritten,
        }

    except Exception as exc:
        # Never block the pipeline — on error, pass through unchanged
        import traceback
        traceback.print_exc()
        return None
