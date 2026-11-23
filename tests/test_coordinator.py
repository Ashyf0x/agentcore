# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import threading
import time

from exgentic.adapters.agents.coordinator import AgentCoordinator, CoordinatedAgent
from exgentic.core.types import (
    MultiObservation,
    ParallelAction,
    SingleAction,
    SingleObservation,
)
from pydantic import BaseModel


class ScriptedAgent(CoordinatedAgent):
    def __init__(self):
        self.ready = threading.Event()
        self.done = threading.Event()
        self.seen = []

    def run(self, adapter):
        self.ready.set()
        obs = adapter.get_observation()
        self.seen.append(obs)
        obs2 = adapter.execute("a0")
        self.seen.append(obs2)
        adapter.execute(None)
        self.done.set()


def shutdown(coord: AgentCoordinator, timeout: float = 2.0):
    coord.close()
    if hasattr(coord, "join"):
        coord.join(timeout=timeout)


def test_basic_handshake():
    internal = ScriptedAgent()
    coord = AgentCoordinator("basic", internal)
    coord.start(task="", context={}, actions=[])

    assert internal.ready.wait(timeout=1.0)

    act = coord.react("obs0")
    assert act == "a0"

    act = coord.react("obs1")
    assert act is None

    shutdown(coord)
    assert internal.done.wait(timeout=1.0)

    assert internal.seen[0] == "obs0"
    assert len(internal.seen) == 2
    assert internal.seen[1] in ("obs1", None)


class TermAgent(CoordinatedAgent):
    def __init__(self):
        self.started = threading.Event()
        self.done = threading.Event()

    def run(self, adapter):
        self.started.set()
        obs = adapter.get_observation()
        if obs is None:
            self.done.set()
            return
        adapter.execute("go")
        obs2 = adapter.get_observation()
        assert obs2 is None
        adapter.execute(None)
        self.done.set()


def test_terminal_observation_unblocks():
    internal = TermAgent()
    coord = AgentCoordinator("term", internal)
    coord.start(task="", context={}, actions=[])

    assert internal.started.wait(timeout=1.0)

    act = coord.react("init")
    assert act == "go"

    act = coord.react(None)
    assert act is None

    shutdown(coord)
    assert internal.done.wait(timeout=1.0)


def test_parallel_execute_threads_return_parallel_action():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    start = threading.Event()

    def _call(action):
        start.wait()
        coord.execute(action)

    t1 = threading.Thread(target=_call, args=(action_a,))
    t2 = threading.Thread(target=_call, args=(action_b,))
    t1.start()
    t2.start()
    start.set()

    deadline = time.time() + 1.0
    with coord._condition:
        while len(coord._pending_actions) < 2 and time.time() < deadline:
            coord._condition.wait(timeout=0.05)

    act = coord.react("obs")
    assert isinstance(act, ParallelAction)
    assert {a.id for a in act.actions} == {action_a.id, action_b.id}

    coord.close()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


def test_no_accumulation_without_window():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    start = threading.Event()

    def _call(action, delay=0.0):
        start.wait()
        if delay:
            time.sleep(delay)
        coord.execute(action)

    t1 = threading.Thread(target=_call, args=(action_a,))
    t2 = threading.Thread(target=_call, args=(action_b, 0.5))
    t1.start()
    t2.start()
    start.set()

    deadline = time.time() + 1.0
    with coord._condition:
        while len(coord._pending_actions) < 1 and time.time() < deadline:
            coord._condition.wait(timeout=0.05)

    act = coord.react("obs")
    assert isinstance(act, SingleAction)
    assert act.id == action_a.id

    act2 = coord.react("obs2")
    assert isinstance(act2, SingleAction)
    assert act2.id == action_b.id

    coord.close()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


def test_accumulation_window_batches_actions():
    coord = _coordinator(accumulate_window_seconds=0.3)
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    start = threading.Event()

    def _call(action, delay=0.0):
        start.wait()
        if delay:
            time.sleep(delay)
        coord.execute(action)

    t1 = threading.Thread(target=_call, args=(action_a,))
    t2 = threading.Thread(target=_call, args=(action_b, 0.1))
    t1.start()
    t2.start()
    start.set()

    deadline = time.time() + 1.0
    with coord._condition:
        while len(coord._pending_actions) < 1 and time.time() < deadline:
            coord._condition.wait(timeout=0.05)

    act = coord.react("obs")
    assert isinstance(act, ParallelAction)
    assert {a.id for a in act.actions} == {action_a.id, action_b.id}

    coord.close()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


def test_execute_returns_observation_for_action():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    start = threading.Event()
    results = {}

    def _call(action):
        start.wait()
        results[action.name] = coord.execute(action)

    t1 = threading.Thread(target=_call, args=(action_a,))
    t2 = threading.Thread(target=_call, args=(action_b,))
    t1.start()
    t2.start()
    start.set()

    deadline = time.time() + 1.0
    with coord._condition:
        while len(coord._pending_actions) < 2 and time.time() < deadline:
            coord._condition.wait(timeout=0.05)

    act = coord.react(
        MultiObservation(
            observations=[
                SingleObservation(invoking_actions=[action_a], result="ra"),
                SingleObservation(invoking_actions=[action_b], result="rb"),
            ]
        )
    )
    assert isinstance(act, ParallelAction)

    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    coord.close()

    assert results["tool.a"].result == "ra"
    assert results["tool.b"].result == "rb"


class DummyArgs(BaseModel):
    value: int


def _action(name: str, value: int) -> SingleAction:
    return SingleAction(name=name, arguments=DummyArgs(value=value))


def _coordinator(accumulate_window_seconds: float | None = None) -> AgentCoordinator:
    class NoopAgent(CoordinatedAgent):
        def run(self, adapter):
            return None

    return AgentCoordinator(
        "rewire",
        NoopAgent(),
        accumulate_window_seconds=accumulate_window_seconds,
    )


def test_rewire_observation_assigns_by_order():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    coord._last_actions = [action_a, action_b]

    observation = MultiObservation(
        observations=[
            SingleObservation(result="ra"),
            SingleObservation(result="rb"),
        ]
    )
    rewired = coord._rewire_observation(observation)

    assert rewired.observations[0].invoking_actions == [action_a]
    assert rewired.observations[1].invoking_actions == [action_b]


def test_rewire_single_observation_attaches_all_actions():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    coord._last_actions = [action_a, action_b]

    observation = SingleObservation(result="combined")
    rewired = coord._rewire_observation(observation)

    assert rewired.invoking_actions == [action_a, action_b]


def test_rewire_preserves_existing_invoking_actions():
    coord = _coordinator()
    action_a = _action("tool.a", 1)
    action_b = _action("tool.b", 2)
    coord._last_actions = [action_a, action_b]

    observation = MultiObservation(
        observations=[
            SingleObservation(result="rb", invoking_actions=[action_b]),
            SingleObservation(result="ra"),
        ]
    )
    rewired = coord._rewire_observation(observation)

    assert rewired.observations[0].invoking_actions == [action_b]
    assert rewired.observations[1].invoking_actions == [action_a]


class _InfraFailure(RuntimeError):
    """Mimics agentcore's FatalRunError: an agent that blames infrastructure."""

    error_source = "infra"


def test_agent_failure_keeps_its_error_source():
    # The harness labels a session from ``error_source`` on the exception it
    # catches. Re-raising across the thread boundary must not drop it, or an
    # endpoint outage is recorded as an agent bug.
    coord = _coordinator()
    coord.record_fatal(_InfraFailure("endpoint timed out"))

    try:
        coord.react(None)
    except RuntimeError as exc:
        assert getattr(exc, "error_source", None) == "infra"
    else:
        raise AssertionError("react() must raise after a fatal was recorded")


def test_agent_failure_without_source_does_not_invent_one():
    coord = _coordinator()
    coord.record_fatal(ValueError("plain"))

    try:
        coord.react(None)
    except RuntimeError as exc:
        assert getattr(exc, "error_source", None) is None
    else:
        raise AssertionError("react() must raise after a fatal was recorded")


def test_record_fatal_wakes_with_error_not_clean_close():
    # The P0 race: ``execute(None)`` in a finally closes the coordinator, and a
    # failure recorded afterwards leaves a window where ``react()`` returns
    # ``None`` -- the harness then reads "conversation over", scores it 0, and
    # the outage disappears into the denominator.
    coord = _coordinator()
    coord.record_fatal(_InfraFailure("boom"))

    try:
        result = coord.react(None)
    except RuntimeError:
        return
    raise AssertionError(f"expected a raise, got {result!r}")


def test_agent_turn_error_survives_pickling():
    # The runner ships agent exceptions between processes with cloudpickle.
    # Default pickling replays ``__init__(*args)``, which fails here because
    # ``args`` is only the message -- the transport then falls back to a
    # builtin, and ``error_source`` is silently dropped on the way.
    import cloudpickle

    from exgentic.adapters.agents.coordinator import _AgentTurnError

    original = _InfraFailure("endpoint down")
    turn = _AgentTurnError("boom", original)

    back = cloudpickle.loads(cloudpickle.dumps(turn))

    assert type(back).__name__ == "_AgentTurnError"
    assert back.error_source == "infra"
    assert type(back.original).__name__ == "_InfraFailure"
