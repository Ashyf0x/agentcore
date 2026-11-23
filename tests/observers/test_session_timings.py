# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from exgentic.observers.handlers.session_timings import SessionLLMTimings, get_session_llm_timings


class TestSessionLLMTimings:
    def test_record_and_pop_returns_durations_in_order(self):
        t = SessionLLMTimings()
        t.record("sess-1", 0.5)
        t.record("sess-1", 1.25)
        t.record("sess-1", 0.75)
        assert t.pop("sess-1") == [0.5, 1.25, 0.75]

    def test_pop_unknown_session_returns_empty_list(self):
        assert SessionLLMTimings().pop("missing") == []

    def test_pop_clears_durations(self):
        t = SessionLLMTimings()
        t.record("sess-1", 0.1)
        t.pop("sess-1")
        assert t.pop("sess-1") == []

    def test_record_isolates_sessions(self):
        t = SessionLLMTimings()
        t.record("sess-A", 0.1)
        t.record("sess-B", 0.2)
        assert t.pop("sess-A") == [0.1]
        assert t.pop("sess-B") == [0.2]

    def test_negative_duration_silently_ignored(self):
        t = SessionLLMTimings()
        t.record("sess-1", -1.0)
        assert t.pop("sess-1") == []

    def test_empty_session_id_silently_ignored(self):
        t = SessionLLMTimings()
        t.record("", 1.0)
        assert t.pop("") == []

    def test_module_singleton_returns_same_instance(self):
        assert get_session_llm_timings() is get_session_llm_timings()


class TestTimingsFromTheAgentItself:
    """The second source: agents that never go through LiteLLM.

    ``SessionLLMTimings`` is fed by the LiteLLM trace logger. ``agentcore``
    talks ``httpx`` straight to the endpoint, so that callback never fires and
    every agentcore run reported ``llm_time=0`` / ``llm_call_count=0`` while
    its store held 530 model calls. The observer therefore also asks the agent.

    These tests drive ``ResultsObserver._llm_durations`` directly: it is the
    one function that decides which source wins, and getting it wrong is
    silent (a number, just the wrong one).
    """

    @staticmethod
    def _observer():
        from exgentic.observers.handlers.results import ResultsObserver

        return ResultsObserver()

    @staticmethod
    def _agent(reported):
        class _Agent:
            def __init__(self, value):
                self._value = value

            def get_llm_timings(self):
                if isinstance(self._value, Exception):
                    raise self._value
                return self._value

        return _Agent(reported)

    def test_agent_supplied_timings_are_used(self):
        durations = self._observer()._llm_durations("sess-x", self._agent([0.5, 1.5]))
        assert durations == [0.5, 1.5]

    def test_agent_timings_beat_the_local_accumulator(self):
        """Its own measurement is the complete one; the accumulator is a subset.

        If both are present the agent wins, because it measured every call it
        made while the module-level accumulator only ever saw the subset that
        went through LiteLLM.
        """
        obs = self._observer()
        get_session_llm_timings().record("sess-x", 9.0)
        durations = obs._llm_durations("sess-x", self._agent([0.5, 1.5]))
        assert durations == [0.5, 1.5]
        assert get_session_llm_timings().pop("sess-x") == [], "the accumulator must still be drained"

    def test_falls_back_to_accumulator_when_agent_reports_nothing(self):
        obs = self._observer()
        get_session_llm_timings().record("sess-y", 0.25)
        assert obs._llm_durations("sess-y", self._agent([])) == [0.25]

    def test_falls_back_when_agent_raises(self):
        """A dead peer must not lose the timings the accumulator already has."""
        obs = self._observer()
        get_session_llm_timings().record("sess-z", 0.75)
        durations = obs._llm_durations("sess-z", self._agent(RuntimeError("peer gone")))
        assert durations == [0.75]

    def test_no_agent_still_drains_the_accumulator(self):
        obs = self._observer()
        get_session_llm_timings().record("sess-w", 0.4)
        assert obs._llm_durations("sess-w", None) == [0.4]

    def test_both_empty_is_zero_not_an_error(self):
        assert self._observer()._llm_durations("sess-none", self._agent([])) == []

    def test_durations_are_coerced_to_float(self):
        """JSON gives back ints for whole seconds; llm_time is a float field."""
        durations = self._observer()._llm_durations("sess-i", self._agent([1, 2]))
        assert durations == [1.0, 2.0]
        assert all(isinstance(d, float) for d in durations)
