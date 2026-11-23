# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A configured context budget must be enforced even when the tokenizer is missing.

The budget is counted on the exact serialized body. ``tiktoken`` gives the real
token count when its encodings are available; when they are not (not installed,
or an offline host that cannot fetch them -- encodings download on first use),
the adapter falls back to the serialized byte count. A token is at least one
byte, so that bound over-trims but never under-trims: the budget cannot overflow
the model's window, and it cannot be silently ignored the way it used to be.
"""

from __future__ import annotations

from exgentic.agents.agentcore import agentcore_instance as adapter
from exgentic.agents.agentcore.agentcore_instance import (
    AgentcoreAgentInstance,
    _tiktoken_counter,
)
from exgentic.agents.agentcore.core.model import ScriptedModel


def _instance(tmp_path, **kwargs):
    return AgentcoreAgentInstance(
        session_id="s",
        model_id="openai/gpt-4.1",
        store_dir=str(tmp_path / "store"),
        api_base="https://example.invalid/v1",
        model_client=ScriptedModel([]),
        **kwargs,
    )


def _no_tokenizer(monkeypatch):
    # Patch the adapter's resolver instead of ``sys.modules``: hiding tiktoken
    # globally breaks litellm's own lazy import of it.
    monkeypatch.setattr(adapter, "_tiktoken_counter", lambda model_id: None)


def test_missing_tokenizer_falls_back_to_the_byte_upper_bound(tmp_path, monkeypatch):
    _no_tokenizer(monkeypatch)

    instance = _instance(tmp_path, context_window_tokens=131072)
    try:
        kwargs = instance._context_window_kwargs()
        body = b'{"messages":[]}'
        counted = kwargs["token_counter"](body)
    finally:
        instance.close()

    assert kwargs["context_window_tokens"] == 131072
    assert kwargs["output_reserve_tokens"] == 4096
    assert kwargs["token_counter"].kind == "bytes_upper_bound"
    # The bound is the body length: a token can never be smaller than one byte.
    assert counted == len(body)


def test_no_budget_needs_no_tokenizer(tmp_path, monkeypatch):
    _no_tokenizer(monkeypatch)

    instance = _instance(tmp_path)
    try:
        assert instance._context_window_kwargs() == {}
    finally:
        instance.close()


def test_configured_budget_is_always_wired_to_a_counter(tmp_path):
    """Whichever counter the host can provide, the window is never dropped."""
    instance = _instance(tmp_path, context_window_tokens=131072)
    try:
        kwargs = instance._context_window_kwargs()
        counted = kwargs["token_counter"](b'{"messages":[]}')
    finally:
        instance.close()

    assert kwargs["token_counter"].kind in {"tiktoken", "bytes_upper_bound"}
    assert counted > 0
    # Sanity: when tiktoken is truly usable here, the fallback must not be used.
    if _tiktoken_counter("gpt-4.1") is not None:
        assert kwargs["token_counter"].kind == "tiktoken"
