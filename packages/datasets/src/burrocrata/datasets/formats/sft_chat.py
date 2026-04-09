"""Generic SFT chat-format helpers (system/user/assistant messages).

Source-agnostic. Each builder calls ``to_chat_example`` with strings.
"""

from __future__ import annotations


def to_chat_example(
    system: str,
    user: str,
    assistant: str,
    metadata: dict | None = None,
) -> dict:
    """Build one ChatML-style row: messages + metadata."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata or {},
    }
