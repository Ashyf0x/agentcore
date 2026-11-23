# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Contract tests for the ``AgentcoreAgent`` configuration object.

The config class is the boundary between the Exgentic CLI/API and the adapter
instance. Two things can silently break there:

* a field that exists on the config but is never forwarded, so a user setting it
  sees no effect;
* a forwarded key that no longer matches the instance signature, which only
  fails once a real run starts.

Both are checked here without importing ``agentcore``.
"""

from __future__ import annotations

import inspect

import pytest

from exgentic.agents.agentcore.agentcore_agent import AgentcoreAgent
from exgentic.interfaces.registry import load_agent


def _instance_signature() -> set[str]:
    from exgentic.agents.agentcore.agentcore_instance import AgentcoreAgentInstance

    params = inspect.signature(AgentcoreAgentInstance.__init__).parameters
    return {name for name in params if name != "self"}


def test_registry_entry_matches_the_class():
    """``_validate_entry`` compares slug/display, so this must not drift."""
    cls = load_agent("agentcore")

    assert cls is AgentcoreAgent
    assert cls.slug_name == "agentcore"
    assert cls.display_name == "Agentcore Runtime"


def test_every_forwarded_kwarg_exists_on_the_instance():
    agent = AgentcoreAgent()
    kwargs = agent._get_instance_kwargs(session_id="s1")

    unknown = set(kwargs) - _instance_signature()
    assert unknown == set(), f"forwarded but not accepted: {sorted(unknown)}"


def test_instance_parameters_that_must_be_forwarded_are_present():
    """Catch the opposite drift: a new field nobody wires up.

    Only parameters that are safe to default are listed. ``model_client`` is
    excluded on purpose — it is the injection point for tests and for callers
    that build their own transport.
    """
    kwargs = AgentcoreAgent()._get_instance_kwargs(session_id="s1")

    optional = _instance_signature() - {"model_client"}
    missing = optional - set(kwargs)
    assert missing == set(), f"accepted but never forwarded: {sorted(missing)}"


def test_settings_match_the_instance_signature():
    """``AgentcoreSettings`` is the forwarding contract for the instance.

    ``_get_instance_kwargs`` is generated from the settings fields, so this is
    the test that keeps the generated kwargs valid: a settings field without an
    instance parameter (or the reverse) fails here instead of at run start.
    """
    from exgentic.agents.agentcore.settings import AgentcoreSettings

    fields = set(AgentcoreSettings.__dataclass_fields__)
    expected = _instance_signature() - {"session_id", "model_client"}
    assert fields == expected


def test_kwargs_round_trip_through_model_validation():
    """``RunConfig.agent_kwargs`` is validated with ``cls.model_validate``."""
    agent = AgentcoreAgent(
        max_steps=7,
        context_window_tokens=131072,
        output_reserve_tokens=8192,
        context_policy="recent",
        read_only_actions=["lookup"],
    )
    validated = AgentcoreAgent.model_validate(agent.model_dump())

    assert validated.max_steps == 7
    assert validated.context_window_tokens == 131072
    assert validated.output_reserve_tokens == 8192
    assert validated.context_policy == "recent"
    assert validated.read_only_actions == ["lookup"]


def test_session_id_is_passed_through_not_stored():
    first = AgentcoreAgent()._get_instance_kwargs(session_id="a")
    second = AgentcoreAgent()._get_instance_kwargs(session_id="b")

    assert first["session_id"] == "a"
    assert second["session_id"] == "b"


def test_defaults_match_the_frozen_retail_profile():
    """A silent default change would move the baseline comparison."""
    agent = AgentcoreAgent()
    kwargs = agent._get_instance_kwargs(session_id="s")

    assert kwargs["max_steps"] == 24
    # No byte ceiling. The only size limit is the model's context window, and it
    # is opt-in: unset means the whole context goes out and the endpoint decides.
    assert "max_request_bytes" not in kwargs
    assert kwargs["context_window_tokens"] is None
    assert kwargs["output_reserve_tokens"] == 4096
    assert kwargs["context_policy"] == "capacity"
    assert kwargs["context_compaction"] == "legacy"
    assert kwargs["enable_memory"] is False
    assert kwargs["provider"] == "compatible"
    assert kwargs["max_output_tokens"] == 2048


def test_model_name_is_the_wire_name():
    """Exgentic reports this in results; it must not carry the routing prefix."""
    agent = AgentcoreAgent(model="openai/gpt-4.1")

    assert agent.model_name == "gpt-4.1"
    assert agent.get_models_names() == ["openai/gpt-4.1"]


def test_model_name_survives_a_nested_openrouter_id():
    agent = AgentcoreAgent(model="openrouter/meta/llama-3.1-70b")

    assert agent.model_name == "llama-3.1-70b"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("context_window_tokens", 0),
        ("context_window_tokens", 20_000_000),
        ("output_reserve_tokens", 0),
        ("max_output_tokens", 0),
        ("max_output_tokens", 100_000),
        ("model_timeout", 0),
        ("http_timeout", -1),
        ("tool_timeout", 100_000),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentcoreAgent(**{field: value})


def test_a_context_window_may_be_left_unset():
    """``None`` is the "send everything" configuration, not a missing value."""
    agent = AgentcoreAgent(context_window_tokens=None, output_reserve_tokens=4096)

    assert agent.context_window_tokens is None
    assert agent.output_reserve_tokens == 4096


def test_read_only_actions_is_not_shared_between_instances():
    """A mutable default would leak declarations across runs."""
    first = AgentcoreAgent()
    second = AgentcoreAgent()

    first.read_only_actions.append("lookup")

    assert second.read_only_actions == []


def test_litellm_params_extra_is_not_shared_between_instances():
    first = AgentcoreAgent()
    second = AgentcoreAgent()

    first.litellm_params_extra["api_base"] = "https://example.invalid/v1"

    assert second.litellm_params_extra == {}


def test_instance_class_ref_points_at_the_adapter():
    assert (
        AgentcoreAgent._get_instance_class_ref()
        == "exgentic.agents.agentcore.agentcore_instance:AgentcoreAgentInstance"
    )


def test_instance_class_ref_resolves():
    from exgentic.agents.agentcore.agentcore_instance import AgentcoreAgentInstance

    module_name, _, attr = AgentcoreAgent._get_instance_class_ref().partition(":")

    import importlib

    module = importlib.import_module(module_name)
    assert getattr(module, attr) is AgentcoreAgentInstance


def test_a_runnable_config_needs_only_a_model():
    """Everything else has a default, so ``--agent agentcore`` works out of the box."""
    agent = AgentcoreAgent(model="openai/gpt-4.1")
    kwargs = agent._get_instance_kwargs(session_id="s")

    assert kwargs["model_id"] == "openai/gpt-4.1"
    # No endpoint is configured: the adapter falls back to the environment, and
    # reports a clear error if that is empty too.
    assert kwargs["api_base"] == ""
