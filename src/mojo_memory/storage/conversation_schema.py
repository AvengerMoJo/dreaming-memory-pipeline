"""
Backend-neutral conversation record schema and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ConversationRecord:
    conversation_id: str
    message_id: str
    turn_index: int
    role: str
    content: str
    created_at: str
    parent_message_id: Optional[str] = None
    pair_id: Optional[str] = None
    status: str = "complete"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_conversation_record(record: Dict[str, Any]) -> tuple[bool, str | None]:
    required = [
        "conversation_id",
        "message_id",
        "turn_index",
        "role",
        "content",
        "created_at",
    ]
    for key in required:
        if key not in record:
            return False, f"missing field: {key}"
    if not isinstance(record["turn_index"], int) or record["turn_index"] < 0:
        return False, "turn_index must be non-negative int"
    if record["role"] not in {"user", "assistant", "system"}:
        return False, "role must be user|assistant|system"
    status = record.get("status", "complete")
    if status not in {"complete", "incomplete"}:
        return False, "status must be complete|incomplete"
    if not str(record["conversation_id"]).strip():
        return False, "conversation_id cannot be empty"
    if not str(record["message_id"]).strip():
        return False, "message_id cannot be empty"
    return True, None
