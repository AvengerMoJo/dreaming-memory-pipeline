"""
Session Compactor — serialize completed task sessions into dreaming input.

Reads ~/.memory/task_sessions/<task_id>.json, flattens the message log into
readable text (skipping system prompt + raw tool payloads), and feeds it
through the DreamingPipeline.

conversation_id: "sessions/session_<task_id>"
Storage:         ~/.memory/dreams/sessions/session_<task_id>/

Only runs when the session exceeds MIN_MESSAGES (avoids noise from trivial tasks).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dreaming.pipeline import DreamingPipeline

logger = logging.getLogger(__name__)

MIN_MESSAGES = 8
MAX_TOOL_RESULT_CHARS = 400
SUMMARIZE_THRESHOLD = 4000


def _default_sessions_dir() -> Path:
    memory_path = os.environ.get("MEMORY_PATH", str(Path.home() / ".memory"))
    return Path(memory_path) / "task_sessions"


def build_session_text(
    task_id: str,
    session_path: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Load a task session and serialize it to readable text.

    Returns None if the session is too short or can't be read.
    """
    if session_path is None:
        base = sessions_dir or _default_sessions_dir()
        session_path = base / f"{task_id}.json"

    if not session_path.exists():
        return None

    try:
        with open(session_path, encoding="utf-8") as f:
            session = json.load(f)
    except Exception as e:
        logger.warning(f"SessionCompactor: failed to read {session_path}: {e}")
        return None

    messages = session.get("messages", [])
    # Filter: skip system prompt (role=system, iteration=0)
    content_messages = [
        m for m in messages
        if not (m.get("role") == "system" and m.get("iteration", 1) == 0)
    ]

    if len(content_messages) < MIN_MESSAGES:
        return None

    metadata = session.get("metadata", {})
    goal = metadata.get("goal", "(no goal)")
    role_id = metadata.get("role_id", "unknown")
    status = session.get("session_status", "unknown")
    iterations = max((m.get("iteration", 0) for m in messages), default=0)
    final_answer = session.get("final_answer")
    error = session.get("error_message")

    lines = [
        f"=== Task Session: {task_id} ===",
        f"Role: {role_id} | Status: {status} | Iterations: {iterations}",
        f"Goal: {goal}",
        "",
    ]

    for msg in content_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        iteration = msg.get("iteration", "?")
        tool_name = msg.get("tool_name")

        if role == "assistant":
            if content and content.strip():
                lines += [
                    f"[Iteration {iteration} — assistant response]",
                    content.strip()[:1200],
                    "",
                ]
            reasoning = (msg.get("metadata") or {}).get("reasoning")
            if reasoning and reasoning.strip():
                lines += [
                    f"[Iteration {iteration} — model reasoning (think)]",
                    reasoning.strip()[:600],
                    "",
                ]
        elif role == "tool":
            snippet = content.strip()[:MAX_TOOL_RESULT_CHARS]
            if len(content.strip()) > MAX_TOOL_RESULT_CHARS:
                snippet += " ...(truncated)"
            name_label = f"tool: {tool_name}" if tool_name else "tool result"
            lines += [
                f"[Iteration {iteration} — {name_label}]",
                snippet,
                "",
            ]
        elif role == "user" and iteration != 0:
            if content and len(content) < 300:
                lines += [f"[system: {content.strip()}]", ""]

    if final_answer:
        lines += ["--- Final Answer ---", final_answer.strip()[:800], ""]
    if error:
        lines += [f"Error: {error}", ""]

    return "\n".join(lines)


async def compact_session(
    task_id: str,
    pipeline: "DreamingPipeline",
    session_path: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
    quality_level: str = "basic",
) -> dict:
    """
    Serialize the session and run it through the dreaming pipeline.

    Returns the pipeline result dict, or a skip/error dict.
    """
    text = build_session_text(task_id, session_path=session_path, sessions_dir=sessions_dir)
    if text is None:
        return {
            "status": "skipped",
            "reason": "session_too_short_or_missing",
            "task_id": task_id,
        }

    conversation_id = f"sessions/session_{task_id}"
    logger.info(f"SessionCompactor: running pipeline for {conversation_id}")

    result = await pipeline.process_conversation(
        conversation_id=conversation_id,
        conversation_text=text,
        metadata={
            "source": "session_compaction",
            "task_id": task_id,
            "quality_level": quality_level,
        },
    )
    return result
