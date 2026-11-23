# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Run the author's own runtime (``agentcore``) as an Exgentic ``AgentInstance``.

Why ``CodeAgentInstance`` and not a ``react()`` agent
----------------------------------------------------
``agentcore.Agent.run()`` owns its whole loop: it prepares the request, calls
the model, executes the tool batch and appends to its own SQLite session store.
Exgentic's turn-based hook (``react(observation) -> Action``) assumes the
framework decides when the model is called, which would cut that loop in two.
``CodeAgentInstance`` is the adapter shape built for this case: Exgentic hands
the agent one blocking callable per action, the agent keeps the loop and calls
back into the environment through ``AgentCoordinator.execute()``.

Threading, and why it constrains two design choices
---------------------------------------------------
``AgentCoordinator`` runs the agent on a thread of its own::

    environment thread --> react(obs): publish observation, wait for action
    agent thread       --> asyncio.run: drive the runtime loop

and ``AgentCoordinator.execute()`` is a blocking rendezvous between the two.
That gives the adapter two hard requirements.

**One thread may touch the store.** ``agentcore.storage.Store`` opens sqlite3
with the default ``check_same_thread=True``. ``AgentcoreAgentInstance`` is
constructed on the session thread, but the loop runs on the agent thread, so
the store must be opened lazily inside ``run_code_agent`` — and *nothing* may
use it off-thread afterwards. That is why the tool handlers below are
``async def``: ``agentcore``'s executor awaits coroutine handlers in place but
pushes plain functions through ``asyncio.to_thread``, which would land
``Store.attempt`` on a pool thread and raise ``ProgrammingError``.

The cost of that choice: a blocking ``execute()`` inside a coroutine also
blocks the event loop, so ``spec.timeout`` cannot fire while a tool call is in
flight. That is acceptable here — the environment thread is the one that owes
us the observation, and it is not waiting on us to yield — but it does mean the
per-tool timeout is advisory rather than enforced.

**``Agent.run`` is a coroutine on a loop-less thread.** ``asyncio.run()`` is
therefore safe (there is no running loop to nest inside), and it is re-entered
once per user turn.

One user turn per ``run()``
---------------------------
``agentcore`` produces one answer per ``run()``. A turn ends either by calling
a tool (the loop continues inside the same ``run``) or by answering. The answer
becomes a ``message`` action, and the observation that action returns is the
next user message, which starts the next ``run()``. That is exactly how
``evals/retail_interaction.py`` drives the runtime, so both harnesses agree.

What is preserved, what is not
------------------------------
Preserved: the runtime's context-window trimming, its execution evidence in
SQLite, its ``goal_status`` accounting and its dispatch policy. Exgentic
supplies only the action space and the transport.

Not preserved, because Exgentic's ``ActionType`` does not carry it:

* **Read vs. write.** ``agentcore`` uses ``read_only`` to decide whether an
  interrupted dispatch may be retried or must be escalated. ``ActionType`` has
  no such flag, so tools default to writable — conservative, but it escalates
  an interrupted *read*. ``read_only_actions`` lets the caller declare the real
  read set.
* **Tool failure signalling.** The TAU2 adapter flattens a ``ToolMessage`` into
  its ``content`` string and drops tau2's ``error`` flag, so a business
  rejection arrives as a *successful* observation containing an error string.
  The bundled agents see exactly the same thing, so the comparison stays fair.

Tool results reach the model verbatim. The runtime's evidence archive and its
failure-payload masking were removed: they were an agent-specific visibility
policy the bundled agents do not have, and under this adapter the failure
signal they depended on never fired. ``ask_user`` and ``read_observation``, the
upstream runtime's own test tools, are not registered either — the environment
tools are the whole action space, as they are for the bundled agents.

Tool execution makes one dispatch attempt and never retries or reconciles it
behind the model's back. A failed call returns to the model as a normal tool
result; an interrupted write produces ``WAIT`` because it may have side effects.
Model-call retries (429/5xx/timeouts) are unchanged and matched to the baseline.

One deliberate difference from the bundled agents: they prepend task+context as
the first *user* message, which the ``recent``/``capacity`` context policies may
trim away. Here the task and the benchmark context go into the system prompt,
which no context policy trims. Memory is off by default and, when enabled, this
adapter only wires the read path (``Memory.retrieve``); automatic extraction and
commit are not connected, so the switch is partial here (see ``AgentcoreAgent``).

Where the runtime comes from
----------------------------
The runtime is vendored beside this module (``context/``, ``core/``, ``memory/``,
``observability/``, ``storage/``, ``tools/``). Only the parts the loop needs are
kept: the upstream ``app/`` CLI and ``tools/mcp.py`` are omitted because nothing
on this path imports them. Vendored modules use relative imports, so the copy
cannot shadow — or be shadowed by — an installed ``agentcore``.

Set ``runtime_path`` to run an external checkout instead of the vendored copy.
That is what you want when the working copy has moved ahead of the snapshot
here, and it keeps a run traceable to a specific commit.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...adapters.agents.code_agent import CodeAgentInstance
from ...core.actions import build_action
from ...core.types import (
    Action,
    ActionType,
    MessageObservation,
    MessagePayload,
    MultiObservation,
    Observation,
    SingleObservation,
)
from ...utils.cost import CostReport
from ...utils.paths import sanitize_path_component
from .context_budget import _byte_upper_bound, _tiktoken_counter
from .runtime_loader import load_runtime, wire_model_name
from .settings import AgentcoreSettings
from .telemetry import UsageTrackingModel
from .tool_bridge import ToolBridge
from .adapter_utils import NullMemory, _first_text, render_context
from .introspection import cost_report, llm_timings, run_report_from_store
class AgentcoreAgentInstance(CodeAgentInstance):
    """Exgentic ``AgentInstance`` backed by the ``agentcore`` runtime."""

    def __init__(
        self,
        session_id: str,
        model_id: str,
        max_steps: int = 24,
        model_settings: Any = None,
        runtime_path: str = "",
        context_window_tokens: int | None = None,
        output_reserve_tokens: int = 4096,
        enable_memory: bool = False,
        store_dir: str = "",
        litellm_params_extra: dict[str, object] | None = None,
        context_policy: str = "capacity",
        context_compaction: str = "legacy",
        max_output_tokens: int = 2048,
        max_completion_tokens: int | None = None,
        model_timeout: float = 600.0,
        http_timeout: float = 540.0,
        tool_timeout: float = 600.0,
        model_retries: int = 3,
        max_argument_corrections: int | None = None,
        max_protocol_corrections: int | None = None,
        provider: str = "compatible",
        thinking: bool | None = None,
        preserve_thinking: bool = False,
        wire_model: str = "",
        api_base: str = "",
        read_only_actions: list[str] | None = None,
        model_client: Any = None,
        user_id: str = "exgentic",
    ) -> None:
        super().__init__(session_id)
        # One validated settings object. The numeric checks, the timeout
        # layering and the forwarding to the runner all live there, so the
        # config object and the instance cannot drift apart.
        settings = AgentcoreSettings(
            model_id=model_id,
            max_steps=max_steps,
            model_settings=model_settings,
            runtime_path=runtime_path,
            context_window_tokens=context_window_tokens,
            output_reserve_tokens=output_reserve_tokens,
            enable_memory=enable_memory,
            store_dir=store_dir,
            litellm_params_extra=litellm_params_extra or {},
            context_policy=context_policy,
            context_compaction=context_compaction,
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            model_timeout=model_timeout,
            http_timeout=http_timeout,
            tool_timeout=tool_timeout,
            model_retries=model_retries,
            max_argument_corrections=max_argument_corrections,
            max_protocol_corrections=max_protocol_corrections,
            provider=provider,
            thinking=thinking,
            preserve_thinking=preserve_thinking,
            wire_model=wire_model,
            api_base=api_base,
            read_only_actions=tuple(read_only_actions or ()),
            user_id=user_id,
        )
        self._settings = settings
        self.model_id = settings.model_id
        self.max_steps = settings.max_steps
        self.model_settings = settings.model_settings
        self._context_window_tokens = settings.context_window_tokens
        self._output_reserve_tokens = settings.output_reserve_tokens
        self._enable_memory = settings.enable_memory
        self._store_dir = settings.store_dir
        self._extra = dict(settings.litellm_params_extra)
        self._context_policy = settings.context_policy
        self._context_compaction = settings.context_compaction
        self._max_output_tokens = settings.max_output_tokens
        self._max_completion_tokens = settings.max_completion_tokens
        self._model_timeout = settings.model_timeout
        self._http_timeout = settings.http_timeout
        self._tool_timeout = settings.tool_timeout
        self._model_retries = settings.model_retries
        # ``None`` means "leave the runtime's own default in place". The agent
        # level passes ``None`` for the correction budgets so tau2 keeps its
        # frozen zero and tool-only benchmarks keep their headroom; passing a
        # concrete number here overrides that decision for every benchmark at
        # once, which is why the sentinel is preserved rather than defaulted.
        self._max_argument_corrections = settings.max_argument_corrections
        self._max_protocol_corrections = settings.max_protocol_corrections
        self._provider = settings.provider
        self._thinking = settings.thinking
        self._preserve_thinking = settings.preserve_thinking
        self._wire_model = settings.wire_model
        self._api_base = settings.api_base
        self._read_only = frozenset(settings.read_only_actions)
        self._user_id = settings.user_id

        # A configured context budget must actually be enforced. Prefer the real
        # tokenizer; when its encodings are unavailable (no tiktoken, or an
        # offline host that cannot fetch them), fall back to the serialized byte
        # count. A token is at least one byte, so that is a sound upper bound:
        # the budget over-trims but never lets a request overflow the window,
        # which used to happen whenever the tokenizer was missing because the
        # whole context was sent instead.
        self._token_counter = None
        if self._context_window_tokens is not None:
            self._token_counter = _tiktoken_counter(self._wire_model or self.model_id)
            if self._token_counter is None:
                self._token_counter = _byte_upper_bound
                self.logger.warning(
                    "tiktoken encodings are unavailable; enforcing the context window "
                    "with the serialized byte count as a conservative upper bound "
                    "(see agents/agentcore/requirements.txt for pre-caching encodings)"
                )

        self._runtime = load_runtime(runtime_path)
        # Opened on the agent thread, see the module docstring.
        self._store: Any = None
        self._store_path = self._store_file()
        self._agent: Any = None
        self._last_run_id: str = ""

        # True once the environment stops talking to us. Two independent
        # signals set it: react(None) and execute() returning None for a real
        # action. A tool handler that sees it raises rather than returning
        # None, because "the environment is gone" is not "the tool returned
        # nothing" and the model must not be told the latter.
        self._environment_closed = False

        base = model_client if model_client is not None else self._build_http_client()
        self._model = UsageTrackingModel(base)

    # -- construction helpers ------------------------------------------------

    def _store_file(self) -> Path:
        """Where the runtime's SQLite store lives.

        The file name carries the instance id so a re-run of the same session id
        starts from a clean database instead of resuming an earlier run, which
        ``Store.begin`` rejects with ``SessionBusy``.
        """
        if self._store_dir:
            directory = Path(self._store_dir).expanduser()
        else:
            directory = self.paths.agent_dir / "agentcore"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{sanitize_path_component(self.session_id)}-{self.agent_id[:8]}.sqlite3"
        return directory / name

    def _ensure_store(self):
        if self._store is None:
            self._store = self._runtime.Store(self._store_path)
        return self._store

    def _resolve_api_base(self) -> str:
        extra = self._extra
        return _first_text(
            self._api_base,
            extra.get("api_base"),
            extra.get("base_url"),
            os.environ.get("OPENAI_API_BASE"),
            os.environ.get("OPENAI_BASE_URL"),
        )

    def _resolve_api_key(self) -> str:
        extra = self._extra
        return _first_text(
            extra.get("api_key"),
            os.environ.get("OPENAI_API_KEY"),
        )

    def _build_http_client(self):
        base_url = self._resolve_api_base()
        if not base_url:
            raise ValueError(
                "agentcore needs an OpenAI-compatible endpoint: pass api_base, "
                "set litellm_params_extra['api_base'], or export OPENAI_BASE_URL"
            )
        temperature = getattr(self.model_settings, "temperature", None)
        return self._runtime.HTTPModelClient(
            base_url=base_url,
            model=wire_model_name(self.model_id, self._wire_model),
            api_key=self._resolve_api_key(),
            max_output_tokens=self._max_output_tokens,
            temperature=0.0 if temperature is None else float(temperature),
            timeout=self._http_timeout,
            provider=self._provider,
            thinking=self._thinking,
            preserve_thinking=self._preserve_thinking,
            max_completion_tokens=self._max_completion_tokens,
        )

    # -- coordinator signals -------------------------------------------------

    def react(self, observation: Observation | None) -> Action | None:
        action = super().react(observation)
        if observation is None:
            self._environment_closed = True
        return action

    def execute(self, action: Action | None):
        observation = super().execute(action)
        if action is not None and observation is None:
            self._environment_closed = True
        return observation

    # -- AgentInstance -------------------------------------------------------

    def start(self, task: str, context: dict[str, Any], actions: list[ActionType]) -> None:
        super().start(task, context, actions)
        for action_type in self.actions:
            if not isinstance(action_type, ActionType):
                raise ValueError("Invalid action type provided to agentcore")

    def run_code_agent(self, functions: list[Callable]) -> None:
        # First statement, so every later store access happens on this thread.
        self._ensure_store()
        self._agent = self._build_agent(self._bindings(functions))

        observation = self.initial_observation
        while observation is not None and not self._environment_closed:
            text = self._observation_text(observation)
            if not text:
                # Tool-only benchmarks (AppWorld, BFCL, SWE-bench, ...) open
                # with an EmptyObservation: the task and context arrive through
                # start() and already live in the runtime's system prompt.
                # Treating that empty observation as "nothing to do" used to
                # end the loop before the runtime made a single tool call.
                # Hand it an explicit kick-off instead. Conversational
                # benchmarks (tau2) always have a non-empty first message, so
                # this branch never runs there.
                text = self._task_kickoff()
                if not text:
                    self.logger.info(
                        "Stopping: no observation text and no task to start from"
                    )
                    break

            try:
                result = self._run_turn(text)
            except self._runtime.FatalRunError as fatal:
                # Publish the failure *before* it travels any further. The
                # base class ends every turn with ``finally: execute(None)``,
                # which sets ``_closed`` and wakes the environment thread. If
                # the error were only recorded after that (i.e. by letting it
                # propagate to ``_run_internal_agent``), the harness could wake
                # up, see ``_closed`` and no error, and read the turn as "the
                # agent finished" — scoring an endpoint outage as a wrong
                # answer. ``record_fatal`` records and closes in one critical
                # section, so ``react()`` raises instead of returning ``None``.
                self.record_fatal(fatal)
                raise

            if result.status not in ("completed", "waiting_user"):
                # A real terminal result: the runtime worked and ran out
                # (step limit, correction budget, a tool that said terminate).
                # Ending the loop normally is correct here — the harness sees a
                # finished conversation and scores it. Only ``infra`` failures
                # must not take this path, and those arrived as
                # ``FatalRunError`` above.
                self.logger.info(
                    "Runtime stopped the session: status=%s reason=%s steps=%s",
                    result.status,
                    result.reason,
                    result.steps,
                )
                break

            answer = (result.answer or "").strip()
            if not answer:
                self.logger.warning("Runtime finished the turn without an answer; stopping")
                break

            observation = self._send(answer)

    def close(self) -> None:
        try:
            super().close()
        finally:
            # super().close() joins the agent thread, so by here nothing else can
            # be touching the connection.
            if self._store is not None:
                store, self._store = self._store, None
                try:
                    store.close()
                except Exception:  # pragma: no cover - closing must not mask errors
                    self.logger.exception("Failed to close the agentcore store")

    def get_cost(self) -> CostReport:
        """Token usage and cost for the model calls made this run.

        See ``introspection.cost_report`` for the fallbacks: with no calls, or
        no published price for the endpoint, the token counts are reported with
        zero cost rather than a guess.
        """
        return cost_report(self.model_id, getattr(self, "_model", None))

    def get_llm_timings(self) -> list[float]:
        """Per-call model latency, one float per physical call, retries included.

        ``agentcore`` reaches the endpoint over ``httpx``, bypassing LiteLLM
        entirely, so ``SessionLLMTimings`` never sees these calls; they are
        measured in the usage-tracking wrapper instead and handed back over the
        same RPC channel as ``get_cost``.
        """
        return llm_timings(getattr(self, "_model", None))

    def diagnostics(self) -> dict[str, Any]:
        """Export the runtime run report plus Exgentic-side call counts.

        These are diagnostics, not a second results channel. The authoritative
        score still comes from the external observer; this RPC exists because
        ``run_report()`` (retry/correction counts, unterminated calls, token
        totals) lived only in the runtime store and could not be verified
        without opening that file after the fact.

        Two call counts are reported with distinct names because they answer
        different questions: ``model_requests`` is the number of logical steps
        (matches ``run_report.model_calls``), ``physical_model_calls`` is the
        number of HTTP attempts including model retries (matches the length of
        ``get_llm_timings()``). Reasoning tokens are not a separate line item:
        providers include them in ``completion_tokens``.
        """
        usage = getattr(self, "_model", None)
        report: dict[str, Any] = {}
        if self._last_run_id:
            report = run_report_from_store(self._runtime, self._store_path, self._last_run_id)
        return {
            "run_id": self._last_run_id,
            "model_requests": getattr(usage, "logical_requests", 0) if usage else 0,
            "physical_model_calls": getattr(usage, "calls", 0) if usage else 0,
            "reported_durations": len(getattr(usage, "durations", [])) if usage else 0,
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            "run_report": report,
        }

    # -- internals -----------------------------------------------------------

    def _bindings(self, functions: list[Callable]) -> dict[str, Callable]:
        """Pair the action list with the callables Exgentic built, in order."""
        if len(functions) != len(self.actions):
            raise ValueError(
                f"Expected {len(self.actions)} action functions, got {len(functions)}"
            )
        return {action.name: fn for action, fn in zip(self.actions, functions, strict=True)}

    def _build_agent(self, bindings: dict[str, Callable]):
        tools = ToolBridge(
            # Before ``start`` there is no action space yet; the bridge still
            # validates ``read_only`` against the bindings, so a typo is
            # rejected even when this is called early.
            actions=getattr(self, "actions", []),
            bindings=bindings,
            read_only=self._read_only,
            timeout=self._tool_timeout,
            runtime=self._runtime,
            logger=self.logger,
            environment_closed=lambda: self._environment_closed,
        ).build()

        system_prompt = self._system_prompt()
        config_kwargs: dict[str, object] = {
            "max_steps": self.max_steps,
            "model_timeout": self._model_timeout,
            # Matches the baseline's retry budget (tau2 DEFAULT_MAX_RETRIES=3).
            # Without it the comparison measures retry policy, not the agent:
            # one 429 or one hung connection ends a session the baseline
            # would have continued.
            "model_retries": self._model_retries,
        }
        # Only forward a correction budget that was actually chosen. ``None``
        # means "the caller left it to the runtime": tau2 passes 0 to keep the
        # frozen baseline configuration (repair calls would add model calls the
        # baselines never make), while a tool-only benchmark may leave it unset
        # and get the runtime default. Writing a hardcoded 0 here -- which is
        # what this did -- silently flattened that distinction for every
        # benchmark, so the agent-level knob was forwarded and then discarded.
        if self._max_argument_corrections is not None:
            config_kwargs["max_argument_corrections"] = self._max_argument_corrections
        if self._max_protocol_corrections is not None:
            config_kwargs["max_protocol_corrections"] = self._max_protocol_corrections

        return self._runtime.Agent(
            self._model,
            tools,
            self._ensure_store(),
            config=self._runtime.AgentConfig(**config_kwargs),
            context=self._runtime.ContextManager(
                system_prompt,
                policy=self._context_policy,
                compaction=self._context_compaction,
                **self._context_window_kwargs(),
            ),
            memory=None if self._enable_memory else NullMemory(),
            system_prompt=system_prompt,
            user_channel=self._has_user_channel(),
        )

    def _context_window_kwargs(self) -> dict[str, Any]:
        """Token-window arguments for ``ContextManager``.

        The context window is the only size limit the runtime applies. A window
        is only configured when construction found a trusted counter for the
        exact serialized body -- tiktoken when its encodings are available, and
        the byte-count upper bound otherwise (see ``__init__``).
        """
        if self._context_window_tokens is None:
            return {}
        return {
            "context_window_tokens": self._context_window_tokens,
            "token_counter": self._token_counter,
            "output_reserve_tokens": self._output_reserve_tokens,
        }

    def _has_user_channel(self) -> bool:
        """Whether the action space can deliver a free-text reply.

        Tool-only benchmarks finish through a tool; the runtime uses this to
        record a text-only ending as ``failed / no_user_channel`` instead of a
        completed conversation (see ``Agent.run``).
        """
        return any(action.is_message for action in self.actions)

    def _system_prompt(self) -> str:
        parts = []
        task = (self.task or "").strip()
        if task:
            parts.append(task)
        context = render_context(getattr(self, "context", None))
        if context:
            parts.append(context)
        return "\n\n".join(parts)

    def _run_turn(self, text: str):
        """Drive one user turn of the runtime to completion.

        Deliberately catches nothing. The runtime already classifies its own
        failures and raises ``FatalRunError`` for the ones the harness must not
        score as wrong answers; swallowing them here (the old
        ``except Exception: return None``) turned an endpoint outage into
        "the agent finished", which the harness recorded as a scored zero.

        ``asyncio.CancelledError`` is a ``BaseException``, so it was never
        caught by ``except Exception`` anyway — but the old code had a separate
        ``except CancelledError: return None`` that erased the ``cancelled``
        label before the adapter could see it. It now travels to
        ``_run_internal_agent`` and is re-raised on the harness thread.
        """
        result = asyncio.run(
            self._agent.run(text, user_id=self._user_id, session_id=self.session_id)
        )
        # Kept for the diagnostics RPC: run_report() needs the run id, which the
        # runtime assigns inside Agent.run() and returns on RunResult. A test
        # double may return a leaner object; diagnostics simply stay empty then.
        self._last_run_id = getattr(result, "run_id", "") or ""
        return result

    def _send(self, answer: str):
        if self._environment_closed:
            return None
        message_type = next((a for a in self.actions if a.is_message), None)
        if message_type is None:
            # Tool-only benchmarks have no channel for a free-text reply. Only a
            # ``waiting_user`` result reaches this point: a plain text answer is
            # recorded by the runtime as ``failed / no_user_channel`` (see
            # ``Agent.run``), because the benchmark's own finish action is the
            # only way to finish such a task.
            self.logger.info(
                "No message action in this benchmark; dropping the runtime's waiting reply"
            )
            return None
        try:
            return self.execute(build_action(message_type, {"content": answer}))
        except RuntimeError:
            # The coordinator closed between the check above and the call; the
            # environment is gone either way, so this is not an agent failure.
            self._environment_closed = True
            self.logger.info("Environment closed while sending the answer")
            return None

    def _task_kickoff(self) -> str:
        """First user message for benchmarks that start with an empty observation.

        The task/context are already in the runtime's system prompt, so this is
        deliberately short: it only has to stop the loop from treating an
        ``EmptyObservation`` as "nothing to do".
        """
        if not (self.task or "").strip():
            return ""
        return (
            "Begin the task described in your instructions. Use the available "
            "tools to inspect and update the environment, then call the finish "
            "action when the task is complete."
        )

    @staticmethod
    def _observation_text(observation: Observation | None) -> str:
        if observation is None:
            return ""
        if isinstance(observation, MultiObservation):
            parts = [
                AgentcoreAgentInstance._observation_text(item)
                for item in observation.to_observation_list()
            ]
            return "\n\n".join(part for part in parts if part)
        if isinstance(observation, MessageObservation) and isinstance(
            observation.result, MessagePayload
        ):
            return observation.result.message.strip()
        if isinstance(observation, SingleObservation):
            value = observation.result
            if isinstance(value, MessagePayload):
                return value.message.strip()
            if isinstance(value, str):
                return value.strip()
            if value is None:
                return ""
        return str(observation).strip()
