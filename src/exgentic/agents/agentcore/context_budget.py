# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Token counters for the context-window budget.

The budget is counted on the exact serialized body. ``tiktoken`` gives the real
token count when its encodings are available. When they are not (not installed,
or an offline host that cannot fetch them -- encodings download on first use),
the serialized byte count is used instead: every token is at least one byte, so
it is a sound upper bound that over-trims but never under-trims. A configured
budget therefore cannot overflow the model's window, and it cannot be silently
ignored.
"""

from __future__ import annotations

from .runtime_loader import wire_model_name


def _tiktoken_counter(model_id):
    """A trusted token counter for the exact serialized body, or ``None``.

    ``tiktoken`` encodes the body as it will be sent, which is the only number
    that answers "does this request fit". The encoding is chosen from the model
    id when it is known, and falls back to ``o200k_base``, whose tokenizer is a
    superset of the cl100k family that current Qwen/DeepSeek-compatible
    endpoints tokenize close enough to for a reserve-based check.

    Returning ``None`` (tiktoken absent or its encodings unreachable) is
    deliberate rather than falling back to a heuristic here: the caller decides
    between the byte upper bound and a hard error.
    """
    try:
        import tiktoken
    except Exception:
        return None

    name = wire_model_name(model_id).lower()
    encoding = None
    for prefix, encoding_name in (
        ("gpt-4o", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5", "cl100k_base"),
        ("qwen", "o200k_base"),
        ("deepseek", "cl100k_base"),
        ("glm", "cl100k_base"),
    ):
        if name.startswith(prefix):
            encoding = encoding_name
            break
    try:
        codec = tiktoken.get_encoding(encoding or "o200k_base")
    except Exception:
        try:
            codec = tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

    def count(body: bytes) -> int:
        # replace() keeps a byte sequence that is not valid UTF-8 countable
        # instead of raising mid-request.
        return len(codec.encode(body.decode("utf-8", "replace")))

    count.kind = "tiktoken"
    return count


def _byte_upper_bound(body: bytes) -> int:
    """A tokenizer-independent upper bound on tokens: every token is >= 1 byte.

    Used only when tiktoken's encodings cannot be loaded (not installed, or an
    offline host that cannot fetch them). The bound over-trims but never
    under-trims, so a configured context window is still enforced -- it cannot
    overflow the model, and it cannot be silently ignored.
    """
    return len(body)


_byte_upper_bound.kind = "bytes_upper_bound"
