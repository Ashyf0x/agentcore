# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""End-to-end tests against the *real* agentcore runtime.

``test_agentcore_adapter.py`` swaps the runtime for a stand-in so the contract
can be checked anywhere. This file does the opposite: it loads the real runtime
— by default the copy vendored inside the adapter package — and drives it
through the real coordinator handshake, with a scripted model standing in for
the HTTP endpoint.

What only this file can catch:

* the adapter's imports actually exist in the runtime (a renamed symbol passes
  every stand-in test);
* ``Agent.run`` really drives the loop the way the adapter assumes — one call
  per user turn, returning ``RunResult`` with ``status``/``answer``;
* a tool call really blocks on the coordinator and really feeds the
  observation payload back into the next request;
* SQLite is really created and closed on the agent thread.

The vendored copy means these run on any normal install. Set
``AGENTCORE_RUNTIME_PATH`` (or have a checkout at the usual sibling location) to
additionally verify that an external checkout behaves identically — that is the
check that catches a stale snapshot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, create_model

from exgentic.agents.agentcore.agentcore_agent import AgentcoreAgent
from exgentic.agents.agentcore.agentcore_instance import AgentcoreAgentInstance
from exgentic.core.types import (
    ActionType,
    MessageAction,
    MessageObservation,
    MessagePayload,
    SingleAction,
    SingleObservation,
)

DEFAULT_EXTERNAL = Path("A:/Github/Agent-Project/agent-runtime")


def _external_runtime_path() -> Path | None:
    """Locate an importable agentcore checkout, or return None.

    The vendored copy is the default, so an external checkout is optional: it is
    only used by the parametrised test that checks both paths behave alike.
    """
    configured = os.environ.get("AGENTCORE_RUNTIME_PATH", "")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(DEFAULT_EXTERNAL)

    for candidate in candidates:
        if (candidate / "agentcore" / "__init__.py").is_file():
            return candidate
    return None


EXTERNAL_PATH = _external_runtime_path()


# -- Action fixtures ----------------------------------------------------------


class LookupArgs(BaseModel):
    query: str


class OrderArgs(BaseModel):
    order_id: str
    quantity: int = 1


def _tool(name: str, args: type[BaseModel]) -> ActionType:
    action_cls = create_model(f"{name}_Action", __base__=SingleAction, arguments=(args, ...))
    return ActionType(name=name, description=f"{name} tool", cls=action_cls)


def _message() -> ActionType:
    return ActionType(
        name="message",
        description="Send a message to the user.",
        cls=MessageAction,
        is_message=True,
    )


def _user_says(text: str) -> MessageObservation:
    return MessageObservation(
        invoking_actions=[], result=MessagePayload(sender="user", message=text)
    )


# -- Builders -----------------------------------------------------------------


def _responses(*items):
    """Turn plain dicts into the ``ModelResponse`` objects ScriptedModel yields."""
    from exgentic.agents.agentcore.core.model import ScriptedModel
    from exgentic.agents.agentcore.core.types import ModelResponse, ToolCall, Usage

    def _coerce(item):
        if isinstance(item, ModelResponse) or callable(item) or isinstance(item, Exception):
            return item
        if isinstance(item, str):
            item = {"content": item}
        return ModelResponse(
            content=item.get("content"),
            tool_calls=[
                ToolCall(id=call.get("id", f"call_{index}"), name=call["name"],
                         arguments=call.get("arguments", {}))
                for index, call in enumerate(item.get("tool_calls", []))
            ],
            finish_reason="tool_calls" if item.get("tool_calls") else "stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            usage_known=True,
        )

    return ScriptedModel([_coerce(item) for item in items])


def _instance(tmp_path, model, **kwargs) -> AgentcoreAgentInstance:
    """Build an instance on the vendored runtime (the default path)."""
    return AgentcoreAgentInstance(
        session_id="e2e-session",
        model_id="openai/gpt-4.1",
        store_dir=str(tmp_path / "store"),
        api_base="https://example.invalid/v1",
        model_client=model,
        **kwargs,
    )


# -- The loop -----------------------------------------------------------------


def test_one_user_turn_becomes_one_message_action(tmp_path):
    """The smallest possible round trip: model speaks, adapter delivers."""
    instance = _instance(tmp_path, _responses("Hello, how can I help?"))
    instance.start(task="Serve the customer.", context={"policy": "POLICY"}, actions=[_message()])

    try:
        action = instance.react(_user_says("hi"))
    finally:
        instance.close()

    assert isinstance(action, SingleAction)
    assert action.name == "message"
    assert action.arguments.content == "Hello, how can I help?"


def test_tool_call_blocks_on_the_environment_and_resumes(tmp_path):
    """The whole point of the adapter: a tool call is a rendezvous.

    The runtime asks to call ``lookup``; the adapter publishes that as an action
    and blocks. Only when the environment answers does the next model call
    happen, and that call must see the environment's payload.
    """
    model = _responses(
        {"tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"query": "o-1"}}]},
        "Order o-1 is shipped.",
    )
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    try:
        first = instance.react(_user_says("where is o-1?"))
        assert isinstance(first, SingleAction) and first.name == "lookup"
        assert first.arguments.query == "o-1"

        # The environment answers the tool call.
        second = instance.react(
            SingleObservation(result={"order_id": "o-1", "status": "shipped"}, invoking_actions=[])
        )
    finally:
        instance.close()

    assert isinstance(second, SingleAction) and second.name == "message"
    assert second.arguments.content == "Order o-1 is shipped."

    # Two model calls in one turn; the second request carries the tool result.
    assert len(model.requests) == 2
    second_request = model.requests[1]["messages"]
    tool_messages = [m for m in second_request if m.get("role") == "tool"]
    assert tool_messages, "the tool result never reached the model"
    assert "shipped" in tool_messages[-1]["content"]


def test_a_second_user_turn_starts_a_new_run(tmp_path):
    """Two user messages must be two ``run`` calls, not one long transcript.

    This is the invariant the retail evaluation relies on: one call per observed
    user message, with the runtime keeping its own history.
    """
    model = _responses("first answer", "second answer")
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_message()])

    try:
        first = instance.react(_user_says("one"))
        second = instance.react(_user_says("two"))
    finally:
        instance.close()

    assert first.arguments.content == "first answer"
    assert second.arguments.content == "second answer"
    assert len(model.requests) == 2
    # The second request still carries the first exchange: history is the
    # runtime's job, not the adapter's.
    roles = [m["role"] for m in model.requests[1]["messages"]]
    assert "assistant" in roles
    assert "first answer" in str(model.requests[1]["messages"])


def test_tool_schema_reaches_the_model_without_the_message_tool(tmp_path):
    """The model sees exactly the environment tools, nothing injected.

    The runtime's former ``ask_user`` / ``read_observation`` test tools and the
    adapter's own ``message`` action are all withheld. The benchmark's tools are
    the entire action space, which is what the bundled agents expose too.
    """
    model = _responses("ok")
    instance = _instance(tmp_path, model)
    instance.start(
        task="t",
        context={},
        actions=[_tool("lookup", LookupArgs), _tool("place_order", OrderArgs), _message()],
    )

    try:
        instance.react(_user_says("hi"))
    finally:
        instance.close()

    tools = model.requests[0]["tools"]
    names = {tool["function"]["name"] for tool in tools}

    assert names == {"lookup", "place_order"}
    assert "message" not in names
    assert "ask_user" not in names
    assert "read_observation" not in names


def test_diagnostics_exports_call_counts_and_run_report(tmp_path):
    """The internal run report crosses to the harness as plain data.

    Two counts stay distinct: one logical request (``model_requests``) versus
    one physical HTTP attempt (``physical_model_calls``). Both equal 1 here
    because the scripted model does not retry, but the field names pin the
    contract so a future retry cannot silently merge them.
    """
    model = _responses("done")
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_message()])
    instance.react(_user_says("hi"))

    diag = instance.diagnostics()
    instance.close()

    assert diag["model_requests"] == 1
    assert diag["physical_model_calls"] == 1
    assert diag["reported_durations"] == 1
    assert diag["input_tokens"] == 10
    assert diag["output_tokens"] == 5
    report = diag["run_report"]
    assert report["run_id"] == diag["run_id"]
    assert report["model_calls"] == 1
    assert isinstance(report["automatic_retries"], int)


def test_store_is_written_and_closed(tmp_path):
    """SQLite has to be created on the agent thread and closed by ``close()``."""
    model = _responses("ok")
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_message()])

    try:
        instance.react(_user_says("hi"))
    finally:
        instance.close()

    store_files = list((tmp_path / "store").glob("*.sqlite3"))
    assert store_files, "the runtime never wrote its store"
    assert instance._store is None  # released on close


def test_environment_close_mid_tool_call_does_not_hang(tmp_path):
    """A closed environment must unblock the runtime instead of deadlocking it."""
    model = _responses({"tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"query": "x"}}]})
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    try:
        assert instance.react(_user_says("go")).name == "lookup"
        assert instance.react(None) is None
    finally:
        instance.close()

    assert instance._environment_closed is True


def test_usage_is_counted_from_the_model_responses(tmp_path):
    """Cost reporting has to come from this run, not from a runtime-side table."""
    model = _responses("ok")
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_message()])

    try:
        instance.react(_user_says("hi"))
    finally:
        instance.close()

    report = instance.get_cost()
    assert report.input_tokens == 10
    assert report.output_tokens == 5


def test_no_message_action_fails_politely(tmp_path):
    """A benchmark with no messaging action cannot be driven; say so, don't hang."""
    model = _responses("ignored")
    instance = _instance(tmp_path, model)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs)])

    try:
        assert instance.react(_user_says("hi")) is None
    finally:
        instance.close()


# -- Configuration plumbing ---------------------------------------------------


def test_agent_config_builds_a_real_instance(tmp_path):
    """The config object must produce something that survives construction."""
    agent = AgentcoreAgent(
        model="openai/gpt-4.1",
        store_dir=str(tmp_path / "store"),
        api_base="https://example.invalid/v1",
    )

    instance = agent._get_instance_class()(
        **{**agent._get_instance_kwargs(session_id="s"), "model_client": _responses("hi")}
    )

    assert isinstance(instance, AgentcoreAgentInstance)
    instance.close()


# -- Vendored copy vs. external checkout --------------------------------------


def test_the_default_runtime_is_the_vendored_copy():
    """No install step, no sys.path surgery: the vendored tree is the default."""
    from exgentic.agents.agentcore.agentcore_instance import load_runtime

    runtime = load_runtime()

    assert runtime.Agent.__module__.startswith("exgentic.agents.agentcore.")


def test_the_vendored_copy_does_not_shadow_an_installed_package():
    """Vendored modules use relative imports, so the names cannot collide."""
    from exgentic.agents.agentcore.agentcore_instance import load_runtime

    runtime = load_runtime()

    for name in ("Agent", "Store", "ToolRegistry", "HTTPModelClient"):
        assert getattr(runtime, name).__module__.startswith("exgentic.agents.agentcore.")


@pytest.mark.skipif(EXTERNAL_PATH is None, reason="no external agentcore checkout")
def test_external_checkout_and_vendored_copy_agree(tmp_path):
    """Both paths must produce the same observable behaviour.

    The vendored copy is a snapshot; an external checkout may have moved on.
    If they diverge in behaviour, the snapshot is stale and the number it
    produces cannot be compared with a run made from the checkout.

    Each run builds its scripted model from *its own* runtime's types: the two
    trees define distinct ``ModelResponse`` classes, and mixing them fails the
    runtime's response validation rather than testing anything real.
    """
    from exgentic.agents.agentcore.agentcore_instance import load_runtime

    def _run(tag: str, runtime_path: str):
        runtime = load_runtime(runtime_path)
        types = sys.modules[f"{runtime.Agent.__module__.rsplit('.', 1)[0]}.types"]
        model_module = sys.modules[f"{runtime.Agent.__module__.rsplit('.', 1)[0]}.model"]

        model = model_module.ScriptedModel(
            [
                types.ModelResponse(
                    content="same answer",
                    finish_reason="stop",
                    usage=types.Usage(input_tokens=10, output_tokens=5),
                    usage_known=True,
                )
            ]
        )
        instance = AgentcoreAgentInstance(
            session_id="cmp",
            model_id="openai/gpt-4.1",
            store_dir=str(tmp_path / tag),
            api_base="https://example.invalid/v1",
            runtime_path=runtime_path,
            model_client=model,
        )
        instance.start(task="t", context={}, actions=[_message()])
        try:
            action = instance.react(_user_says("hello"))
        finally:
            instance.close()
        return action.arguments.content, len(model.requests)

    vendored = _run("vendored", "")
    external = _run("external", str(EXTERNAL_PATH))

    assert vendored == external


# -- The vendored tree itself -------------------------------------------------

VENDORED_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "exgentic" / "agents" / "agentcore"


def test_the_vendored_tree_has_the_loop_and_nothing_else():
    """Only the subpackages the loop needs are kept.

    ``app/`` (the upstream CLI) and ``tools/mcp.py`` are deliberately absent:
    nothing on the evaluation path imports them, and carrying them would pull in
    ``mcp`` and ``python-dotenv`` for no reason.
    """
    for name in ("context", "core", "memory", "observability", "storage", "tools"):
        assert (VENDORED_PACKAGE / name).is_dir(), f"missing vendored subpackage: {name}"

    assert not (VENDORED_PACKAGE / "app").exists()
    assert not (VENDORED_PACKAGE / "tools" / "mcp.py").exists()


def test_vendored_modules_use_relative_imports():
    """Absolute ``agentcore.*`` imports would collide with an installed package.

    Vendoring only works if the copy is a normal subpackage of Exgentic. A
    single ``from agentcore.X import Y`` reintroduces the name clash this
    structure exists to avoid, and it fails only when an unrelated ``agentcore``
    happens to be installed.
    """
    offenders = []
    for path in VENDORED_PACKAGE.rglob("*.py"):
        if path.name in {"agentcore_agent.py", "agentcore_instance.py"}:
            continue  # adapter code, not vendored code
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from agentcore.", "import agentcore")):
                offenders.append(f"{path.relative_to(VENDORED_PACKAGE)}:{number}: {stripped}")

    assert offenders == [], "vendored code must use relative imports:\n" + "\n".join(offenders)
