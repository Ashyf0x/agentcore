# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Contract tests for the agentcore adapter.

These drive the adapter through the *real* ``CodeAgentInstance`` handshake
(observation in through ``react``, action out through ``execute``) against a
stand-in runtime object, so they run anywhere Exgentic imports without needing
the ``agentcore`` package installed.

What is under test:

* the coordinator handshake, including the async/sync bridge;
* ``ActionType`` -> tool-schema translation, and ``message`` being withheld from
  the tool list;
* the "environment closed mid-flight" path, which must not be read as a tool
  returning ``None``;
* the store being opened on the agent thread (``sqlite3`` default
  ``check_same_thread``), and no store access from anywhere else;
* configuration defaults that keep a comparison against the baselines fair.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, create_model

from exgentic.agents.agentcore import agentcore_instance as adapter
from exgentic.agents.agentcore.agentcore_instance import (
    AgentcoreAgentInstance,
    NullMemory,
    UsageTrackingModel,
    render_context,
    wire_model_name,
)
from exgentic.core.types import (
    ActionType,
    Message,
    MessageAction,
    MessageObservation,
    MessagePayload,
    MultiObservation,
    SingleAction,
    SingleObservation,
)

# -- ActionType fixtures ------------------------------------------------------


class CheckoutArgs(BaseModel):
    order_id: str
    gift_card: bool = False


class LookupArgs(BaseModel):
    query: str


def _tool(name: str, args: type[BaseModel], description: str = "") -> ActionType:
    action_cls = create_model(f"{name}_Action", __base__=SingleAction, arguments=(args, ...))
    return ActionType(name=name, description=description or name, cls=action_cls)


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


class Prepared:
    """Stand-in for ``PreparedRequest``: an immutable body with a pre-send check.

    There is no byte ceiling to model here any more; the runtime's only size limit
    is the model's context window, enforced inside ``ContextManager`` in tokens.
    """

    def __init__(self, body: bytes) -> None:
        self.body = body

    def validate(self) -> None:
        if not self.body:
            raise ValueError("request body is empty")


# -- Runtime stand-in --------------------------------------------------------

#: Script marker: the runtime raises ``FatalRunError`` for this turn.
FATAL = object()


def _build_runtime(*, script: list[Any], registries: list[dict[str, Any]], opened: list) -> SimpleNamespace:
    """A namespace exposing only the agentcore names the adapter touches.

    ``script`` is consumed one entry per model call: a string ends the turn with
    that answer, a list of ``{"name": ..., "arguments": {...}}`` dicts is run as
    a tool batch and the turn continues, and ``FATAL`` makes the runtime raise
    its infra-failure signal.
    """

    class ToolFailure(Exception):  # noqa: N818 - mirrors agentcore's class name
        def __init__(self, message, category="business", *, outcome_known=True):
            super().__init__(message)
            self.category = category
            self.outcome_known = outcome_known

    class FatalRunError(RuntimeError):
        """Mirrors the runtime's infra-failure signal.

        Raised after the terminal result has been persisted, so a caller that
        only sees the exception can still read what happened. The adapter
        matches on this exact class, so the stub has to carry the same name
        and shape.
        """

        error_source = "infra"

        def __init__(self, result, cause=None):
            self.result = result
            self.cause = cause
            detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
            super().__init__(f"agentcore run failed: {result.reason}{detail}")

    class AgentConfig:
        """Mirrors the runtime's defaults so "not overridden" is observable.

        The adapter passes ``max_argument_corrections``/``max_protocol_corrections``
        only when the caller set them, so with a bare ``**kwargs`` stub the
        attribute would simply be absent and a test could not tell "the runtime
        chose its default" from "the field was dropped on the floor".
        """

        max_steps = 12
        max_argument_corrections = 2
        max_protocol_corrections = 1

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ContextManager:
        def __init__(self, system_prompt, *, policy="recent", compaction="legacy", **_):
            self.system_prompt = system_prompt
            self.policy = policy
            self.compaction = compaction

    class ToolSpec:
        def __init__(
            self,
            name,
            description,
            parameters,
            handler,
            read_only,
            timeout=10,
            **_,
        ):
            self.name = name
            self.description = description
            self.parameters = parameters
            self.handler = handler
            self.read_only = read_only
            self.timeout = timeout

        def definition(self):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

    class ToolRegistry:
        def __init__(self, specs=()):
            self.specs: dict[str, ToolSpec] = {}
            for spec in specs:
                self.register(spec)
            registries.append(self.specs)

        def register(self, spec):
            if spec.name in self.specs:
                raise ValueError(f"duplicate tool {spec.name}")
            self.specs[spec.name] = spec

        def definitions(self):
            return [spec.definition() for spec in self.specs.values()]

    class Store:
        """Records the thread it was opened on, and refuses cross-thread use."""

        def __init__(self, path):
            self.path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.opened_on = threading.current_thread().name
            self.cross_thread: list[str] = []
            self.closed = False
            opened.append(self)

        def touch(self):
            if threading.current_thread().name != self.opened_on:
                self.cross_thread.append(threading.current_thread().name)

        def close(self):
            self.closed = True

    class Agent:
        def __init__(
            self, model, tools, store, *, config, context, memory, system_prompt,
            user_channel=True,
        ):
            self.model = model
            self.tools = tools
            self.store = store
            self.config = config
            self.context = context
            self.memory = memory
            self.system_prompt = system_prompt
            self.user_channel = user_channel
            self.turns: list[str] = []
            self.tool_threads: list[str] = []
            self.tool_results: list[Any] = []
            self.tool_errors: list[BaseException] = []

        def _call_tool(self, spec, arguments):
            handler = spec.handler
            if asyncio.iscoroutinefunction(handler):
                asyncio.get_running_loop().run_until_complete(handler(**arguments))
            else:
                handler(**arguments)

        async def run(self, message, *, user_id="default", session_id="default"):
            self.turns.append(message)
            self.store.touch()
            for step in range(self.config.max_steps):
                batch = script.pop(0) if script else "done"
                if batch is FATAL:
                    # Models the runtime's infra path: the terminal result is
                    # persisted first, then ``FatalRunError`` is raised. The
                    # harness must never see this turn as a normal ending.
                    result = SimpleNamespace(
                        status="failed",
                        answer="",
                        reason="internal_error",
                        steps=step + 1,
                        goal_status="unverified",
                        error_source="infra",
                    )
                    raise FatalRunError(result, ConnectionError("endpoint down"))
                if isinstance(batch, str):
                    return SimpleNamespace(
                        status="completed",
                        answer=batch,
                        reason=None,
                        steps=step + 1,
                        goal_status="state_change_observed",
                    )
                for call in batch:
                    spec = self.tools.specs.get(call["name"])
                    if spec is None:
                        raise ToolFailure("Unknown tool", "arguments")
                    self.tool_threads.append(threading.current_thread().name)
                    # Mirrors agentcore's executor: await coroutine handlers in
                    # place rather than pushing them to a worker thread.
                    try:
                        self.tool_results.append(
                            await spec.handler(**call.get("arguments", {}))
                        )
                    except BaseException as exc:
                        self.tool_errors.append(exc)
                        raise
            return SimpleNamespace(
                status="failed",
                answer="",
                reason="step_limit",
                steps=self.config.max_steps,
                goal_status="unverified",
            )

    class HTTPModelClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    return SimpleNamespace(
        Agent=Agent,
        AgentConfig=AgentConfig,
        ContextManager=ContextManager,
        FatalRunError=FatalRunError,
        ToolFailure=ToolFailure,
        HTTPModelClient=HTTPModelClient,
        Store=Store,
        ToolRegistry=ToolRegistry,
        ToolSpec=ToolSpec,
    )


@pytest.fixture
def runtime(monkeypatch):
    """Install a stand-in runtime; return its registries and opened stores."""

    def _install(*, script: list[Any]):
        registries: list[dict[str, Any]] = []
        opened: list = []
        module = _build_runtime(script=script, registries=registries, opened=opened)
        monkeypatch.setattr(adapter, "load_runtime", lambda runtime_path="": module)
        return module, registries, opened

    return _install


def _instance(tmp_path, **kwargs) -> AgentcoreAgentInstance:
    defaults = {
        "model_id": "openai/gpt-4.1",
        "store_dir": str(tmp_path / "store"),
        "api_base": "https://example.invalid/v1",
    }
    defaults.update(kwargs)
    return AgentcoreAgentInstance(session_id="session-1", **defaults)


# -- Pure helpers ------------------------------------------------------------


def test_wire_model_name_strips_routing_prefix():
    assert wire_model_name("openai/gpt-4.1") == "gpt-4.1"
    assert wire_model_name("hosted_vllm/qwen3") == "qwen3"
    assert wire_model_name("openrouter/meta/llama") == "meta/llama"
    assert wire_model_name("qwen3-max") == "qwen3-max"
    assert wire_model_name("openai/gpt-4.1", "explicit-model") == "explicit-model"


def test_render_context_keeps_every_key():
    rendered = render_context({"policy": "be nice", "extra": {"a": 1}})
    assert "<policy>\nbe nice\n</policy>" in rendered
    assert json.loads(rendered.split("<extra>\n")[1].split("\n</extra>")[0]) == {"a": 1}


def test_render_context_is_empty_without_context():
    assert render_context(None) == ""
    assert render_context({}) == ""


def test_null_memory_retrieves_nothing_and_refuses_writes():
    memory = NullMemory()
    assert memory.retrieve("u", "anything") == []
    with pytest.raises(RuntimeError):
        memory.commit("u", [])


def test_usage_tracking_model_counts_known_usage():
    class Inner:
        async def complete(self, request):
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=11, output_tokens=5), usage_known=True
            )

    tracker = UsageTrackingModel(Inner())
    asyncio.run(tracker.complete(None))
    assert (tracker.input_tokens, tracker.output_tokens, tracker.calls) == (11, 5, 1)


def test_usage_tracking_model_ignores_unknown_usage():
    class Inner:
        async def complete(self, request):
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=0, output_tokens=0), usage_known=False
            )

    tracker = UsageTrackingModel(Inner())
    asyncio.run(tracker.complete(None))
    assert (tracker.input_tokens, tracker.output_tokens, tracker.calls) == (0, 0, 1)


def test_observation_text_reads_message_payloads():
    assert AgentcoreAgentInstance._observation_text(_user_says("  hello  ")) == "hello"


def test_observation_text_joins_multi_observation():
    obs = MultiObservation(
        observations=[
            SingleObservation(
                result=MessagePayload(sender="user", message="a"), invoking_actions=[]
            ),
            SingleObservation(result="b", invoking_actions=[]),
        ]
    )
    assert AgentcoreAgentInstance._observation_text(obs) == "a\n\nb"


def test_observation_text_tolerates_empty_values():
    assert AgentcoreAgentInstance._observation_text(None) == ""
    assert (
        AgentcoreAgentInstance._observation_text(
            SingleObservation(result=None, invoking_actions=[])
        )
        == ""
    )


# -- Handshake ---------------------------------------------------------------


def test_one_turn_produces_one_message_action(runtime, tmp_path):
    runtime(script=["Hello, how can I help?"])
    instance = _instance(tmp_path)
    instance.start(
        task="Serve the customer.",
        context={"policy": "POLICY"},
        actions=[_tool("lookup", LookupArgs), _message()],
    )

    action = instance.react(_user_says("I need help with an order"))

    assert isinstance(action, SingleAction)
    assert action.name == "message"
    assert action.arguments == Message(content="Hello, how can I help?")
    instance.close()


def test_message_tool_is_not_advertised_to_the_runtime(runtime, tmp_path):
    _, registries, _ = runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(
        task="t",
        context={},
        actions=[_tool("checkout", CheckoutArgs), _tool("lookup", LookupArgs), _message()],
    )
    instance.react(_user_says("hello"))
    instance.close()

    assert set(registries[0]) == {"checkout", "lookup"}


def test_action_schema_reaches_the_runtime_as_plain_json_schema(runtime, tmp_path):
    """The schema must be the same one the bundled LiteLLM agent sends.

    The adapter reuses ``ToolsActionsRegistry`` precisely so this stays byte for
    byte identical to the baselines; re-deriving the schema here would make the
    comparison unfair without anyone noticing.
    """
    _, registries, _ = runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("checkout", CheckoutArgs)])
    instance.react(_user_says("hello"))
    instance.close()

    schema = registries[0]["checkout"].parameters
    assert schema == CheckoutArgs.model_json_schema()
    assert schema["type"] == "object"
    assert schema["properties"]["order_id"]["type"] == "string"


def test_environment_tools_default_to_writable(runtime, tmp_path):
    _, registries, _ = runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("checkout", CheckoutArgs)])
    instance.react(_user_says("hello"))
    instance.close()

    spec = registries[0]["checkout"]
    # ActionType carries no read/write flag, so the adapter assumes a write: an
    # interrupted dispatch is then escalated rather than silently retried.
    assert spec.read_only is False


def test_read_only_actions_can_be_declared(runtime, tmp_path):
    _, registries, _ = runtime(script=["hi"])
    instance = _instance(tmp_path, read_only_actions=["lookup"])
    instance.start(
        task="t", context={}, actions=[_tool("lookup", LookupArgs), _tool("checkout", CheckoutArgs)]
    )
    instance.react(_user_says("hello"))
    instance.close()

    assert registries[0]["lookup"].read_only is True
    assert registries[0]["checkout"].read_only is False


def test_unknown_read_only_action_is_rejected(runtime, tmp_path):
    """A typo in ``read_only_actions`` must not silently demote a read to a write."""
    runtime(script=["hi"])
    instance = _instance(tmp_path, read_only_actions=["lookup"])
    with pytest.raises(ValueError, match="read_only_actions"):
        instance._build_agent({"checkout": lambda **_: None})


def test_tool_call_publishes_an_action_and_the_turn_continues(runtime, tmp_path):
    """A tool call leaves the process as an action; the environment's answer returns.

    The adapter owns no business logic: ``lookup`` is an ``ActionType`` whose
    implementation lives in the benchmark environment. What the runtime gets
    back is whatever the environment put into the observation — here the dict
    ``{"order_id": "o-1"}`` — and that round trip is what makes the next model
    call meaningful.
    """
    runtime(script=[[{"name": "lookup", "arguments": {"query": "o-1"}}], "Found it."])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    first = instance.react(_user_says("find o-1"))
    assert isinstance(first, SingleAction) and first.name == "lookup"
    assert first.arguments.query == "o-1"

    # The environment answers the tool call; that answer is the observation for
    # the next turn, which lets the runtime finish and speak.
    second = instance.react(SingleObservation(result={"order_id": "o-1"}, invoking_actions=[]))
    instance.close()

    assert isinstance(second, SingleAction) and second.name == "message"
    assert second.arguments.content == "Found it."

    agent = instance._agent
    assert agent.tool_results == [{"order_id": "o-1"}]
    assert agent.tool_errors == []


def test_tool_result_reaches_the_runtime_as_the_observation_payload(runtime, tmp_path):
    """A plain string observation is delivered verbatim, not JSON-wrapped."""
    runtime(script=[[{"name": "lookup", "arguments": {"query": "q"}}], "ok"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    assert instance.react(_user_says("go")).name == "lookup"
    # Observations are serialized by the environment adapter, so a string result
    # arrives as-is; the runtime must see exactly that.
    instance.react(SingleObservation(result="no such order", invoking_actions=[]))
    instance.close()

    assert instance._agent.tool_results == ["no such order"]


def test_system_prompt_carries_task_and_policy(runtime, tmp_path):
    runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(
        task="Serve the customer.",
        context={"policy": "Return within 30 days."},
        actions=[_tool("lookup", LookupArgs), _message()],
    )
    agent = instance._build_agent({a.name: (lambda **kw: "ok") for a in instance.actions})

    # The context policy trims user turns, so the policy must not live in one.
    assert "Serve the customer." in agent.system_prompt
    assert "Return within 30 days." in agent.system_prompt
    instance.close()


def test_defaults_keep_the_comparison_fair(runtime, tmp_path):
    runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    agent = instance._build_agent({a.name: (lambda **kw: "ok") for a in instance.actions})

    assert isinstance(agent.memory, NullMemory)
    assert agent.config.max_steps == 24
    # Not overridden: the adapter forwards these only when the caller chose a
    # value, so the runtime's own defaults survive.
    assert agent.config.max_argument_corrections == 2
    assert agent.config.max_protocol_corrections == 1
    assert agent.context.policy == "capacity"
    assert agent.context.compaction == "legacy"
    instance.close()


def test_user_channel_follows_the_action_space(runtime, tmp_path):
    """A tool-only benchmark has no way to deliver a text reply."""
    runtime(script=["hi"])
    with_message = _instance(tmp_path)
    with_message.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    chatty = with_message._build_agent(
        {a.name: (lambda **kw: "ok") for a in with_message.actions}
    )
    assert chatty.user_channel is True
    with_message.close()

    tool_only = _instance(tmp_path)
    tool_only.start(task="t", context={}, actions=[_tool("lookup", LookupArgs)])
    silent = tool_only._build_agent(
        {a.name: (lambda **kw: "ok") for a in tool_only.actions}
    )
    assert silent.user_channel is False
    tool_only.close()


def test_correction_budgets_are_forwarded_when_the_caller_sets_them(runtime, tmp_path):
    """An explicit budget must reach ``AgentConfig``, not be overwritten.

    The constructor used to hardcode ``max_argument_corrections=0`` here, so the
    agent-level field was declared, validated, forwarded over the RPC -- and then
    silently discarded at the last step. ``None`` has to stay a sentinel meaning
    "keep the runtime default", otherwise the distinction the agent level makes
    (tau2 frozen at zero, tool-only benchmarks with headroom) collapses.
    """
    runtime(script=["hi"])
    instance = _instance(tmp_path, max_argument_corrections=3, max_protocol_corrections=5)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    agent = instance._build_agent({a.name: (lambda **kw: "ok") for a in instance.actions})

    assert agent.config.max_argument_corrections == 3
    assert agent.config.max_protocol_corrections == 5
    instance.close()


def test_unset_correction_budgets_leave_the_runtime_default_alone(runtime, tmp_path):
    """``None`` means "not chosen", not "zero".

    If this ever regresses to writing 0 unconditionally, a benchmark that wants
    the runtime's repair budget loses it without any error -- the same failure
    mode as the hardcoded value, just harder to notice because the number looks
    plausible.
    """
    runtime(script=["hi"])
    instance = _instance(tmp_path)  # both default to None
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    agent = instance._build_agent({a.name: (lambda **kw: "ok") for a in instance.actions})

    # Whatever the runtime's own default is, it is not something we imposed.
    # Compare against the stub the adapter actually loaded: an external
    # ``runtime_path`` checkout has its own ``AgentConfig``, so importing the
    # vendored class here would compare two unrelated objects.
    stub_config = instance._runtime.AgentConfig()

    assert agent.config.max_argument_corrections == stub_config.max_argument_corrections
    assert agent.config.max_protocol_corrections == stub_config.max_protocol_corrections
    instance.close()


def test_env_api_base_is_used_when_nothing_is_passed(runtime, tmp_path, monkeypatch):
    runtime(script=["hi"])
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    instance = AgentcoreAgentInstance(
        session_id="s", model_id="openai/gpt-4.1", store_dir=str(tmp_path / "s")
    )
    kwargs = instance._model._inner.kwargs

    assert kwargs["base_url"] == "https://env.invalid/v1"
    assert kwargs["api_key"] == "env-key"
    assert kwargs["model"] == "gpt-4.1"
    instance.close()


def test_missing_endpoint_is_reported_clearly(runtime, tmp_path, monkeypatch):
    runtime(script=["hi"])
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        AgentcoreAgentInstance(
            session_id="s", model_id="openai/gpt-4.1", store_dir=str(tmp_path / "s")
        )


def test_store_is_opened_on_the_agent_thread(runtime, tmp_path):
    """sqlite3 defaults to check_same_thread, so the store must be built lazily."""
    _, _, opened = runtime(script=["hi"])
    instance = _instance(tmp_path)
    assert instance._store is None  # not built in __init__

    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    instance.react(_user_says("hello"))
    instance.close()

    assert len(opened) == 1
    assert opened[0].opened_on.startswith("AgentCoordinator")
    assert opened[0].cross_thread == []
    assert opened[0].closed is True


def test_tool_handlers_are_coroutines(runtime, tmp_path):
    """Plain handlers would be pushed to a worker thread by agentcore's executor."""
    _, registries, _ = runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs)])
    instance.react(_user_says("hello"))
    instance.close()

    assert asyncio.iscoroutinefunction(registries[0]["lookup"].handler)


def test_environment_close_during_a_tool_call_is_reported_as_a_failure(runtime, tmp_path):
    """``execute`` returns None for both "gone" and "empty"; only one is a result.

    The runtime must be told the environment closed. Handing it ``None`` would
    read as a successful call with no payload, and the model would then invent
    an outcome.
    """
    runtime(script=[[{"name": "lookup", "arguments": {"query": "q"}}], "unreachable"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    assert instance.react(_user_says("go")).name == "lookup"
    # The environment walks away instead of answering the tool call.
    assert instance.react(None) is None
    instance.close()

    # The in-flight tool call must surface as a cancelled ToolFailure, not as a
    # successful call returning None.
    assert instance._environment_closed is True
    errors = instance._agent.tool_errors
    assert len(errors) == 1
    assert isinstance(errors[0], BaseException)
    assert errors[0].category == "cancelled"

def test_close_is_idempotent(runtime, tmp_path):
    runtime(script=["hi"])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    instance.react(_user_says("hello"))

    instance.close()
    instance.close()

    assert instance._store is None


def test_get_cost_reports_tokens_without_a_price(runtime, tmp_path):
    runtime(script=["hi"])
    instance = _instance(tmp_path, model_id="unpriced/local-model")
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])
    instance.react(_user_says("hello"))
    instance.close()

    report = instance.get_cost()
    assert report.model_name == "unpriced/local-model"
    assert report.total_cost == 0.0


# -- Infra failures must not look like a finished conversation --------------


def test_fatal_run_error_reaches_the_harness_as_a_raise(runtime, tmp_path):
    """An endpoint outage must not be scored as a wrong answer.

    Before this was wired, ``_run_turn`` caught everything and returned
    ``None``; the base class then ran ``finally: execute(None)``, so ``react()``
    returned ``None`` and the harness read "the agent finished". This pins the
    opposite: the turn raises.
    """
    runtime(script=[FATAL])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    with pytest.raises(RuntimeError) as excinfo:
        instance.react(_user_says("hello"))
    instance.close()

    # The classification survives the thread hand-off; without it the harness
    # would fall back to "agent" and count the failure against the agent.
    assert getattr(excinfo.value, "error_source", None) == "infra"


def test_fatal_run_error_is_recorded_before_the_turn_is_released(runtime, tmp_path):
    """``record_fatal`` must run before ``execute(None)`` can be observed.

    The base class ends every turn with ``finally: execute(None)``, which sets
    ``_closed`` and wakes the environment thread. If the error were only
    recorded afterwards, the harness could wake up, see ``_closed`` and no
    error, and read the turn as a clean ending.
    """
    runtime(script=[FATAL])
    instance = _instance(tmp_path)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    with pytest.raises(RuntimeError):
        instance.react(_user_says("hello"))

    # Recorded on the coordinator, not merely raised on the agent thread.
    assert instance._agent_error is not None
    assert getattr(instance._agent_error, "error_source", None) == "infra"

    instance.close()


def test_agent_side_failure_is_still_a_scored_ending(runtime, tmp_path):
    """A ``failed`` turn with ``error_source="agent"`` must keep the old path.

    Step limits and exhausted correction budgets are real results: the runtime
    worked and ran out. They return normally, so the harness sees a finished
    conversation and scores it. Only ``infra`` raises.
    """
    # A non-string, non-FATAL batch is consumed as a tool batch; an empty list
    # advances the step without doing anything, so ``max_steps`` runs out.
    runtime(script=[[]])
    instance = _instance(tmp_path, max_steps=1)
    instance.start(task="t", context={}, actions=[_tool("lookup", LookupArgs), _message()])

    # Returns None rather than raising: an exhausted agent turn is not a fault.
    assert instance.react(_user_says("hello")) is None
    instance.close()


def test_fatal_run_error_is_taken_from_the_loaded_runtime(runtime, tmp_path):
    """The vendored copy and an external ``runtime_path`` are different classes.

    A module-level import of ``FatalRunError`` would silently fail to match the
    exception the loaded runtime raises, so the handler must resolve the class
    through ``self._runtime``.
    """
    module, _, _ = runtime(script=[FATAL])
    own_class = module.FatalRunError

    import exgentic.agents.agentcore.core.types as vendored_types

    if hasattr(vendored_types, "FatalRunError"):
        assert own_class is not vendored_types.FatalRunError

    instance = _instance(tmp_path)
    assert instance._runtime.FatalRunError is own_class
