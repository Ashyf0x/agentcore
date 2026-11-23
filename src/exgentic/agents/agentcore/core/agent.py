import asyncio
import hashlib
import json
import math
import random
import sqlite3
from dataclasses import asdict, dataclass
from uuid import uuid4

from ..context import ContextManager
from .errors import Decision
from .types import (
    ContextBudgetError,
    FatalRunError,
    ModelError,
    ModelResponse,
    RunResult,
    StorageError,
    ToolCall,
    dumps,
)
from ..memory import Memory


@dataclass(frozen=True)
class AgentConfig:
    """Run limits for one session.

    ``model_timeout`` bounds a single model call. It is a liveness check, not a
    throughput knob: the reasoning models behind this runtime emit roughly 18
    tokens per second, so a 60s deadline silently caps a reply at about 1000
    tokens and turns long-but-healthy answers into failures. The official
    baseline sets no deadline at all (litellm's 6000s default applies); 600s
    matches that intent while still catching a hung connection.

    ``model_retries`` makes a retryable model failure reach the same conclusion
    the baseline reaches. The baseline calls litellm with
    ``DEFAULT_MAX_RETRIES`` (=3, tau2/config.py) and litellm retries 408/409/429
    and 5xx, plus every timeout and connection error. This runtime used to raise
    on the first one, which is a different experiment: in the airline run of
    2026-11-12 three sessions ended as ``status=error`` after a single 600s
    timeout or a single 429, and would have continued had they retried. The
    default here is 3 to match, and the backoff mirrors openai's
    (0.5s doubling, capped at 8s, -25%/+0% jitter).
    """

    max_steps: int = 12
    model_timeout: float = 600
    model_retries: int = 3
    retry_backoff_initial: float = 0.5
    retry_backoff_max: float = 8.0
    max_argument_corrections: int = 2
    max_protocol_corrections: int = 1

    def __post_init__(self):
        if (
            self.max_steps < 1
            or not math.isfinite(self.model_timeout)
            or self.model_timeout <= 0
        ):
            raise ValueError("Run limits must be positive")
        if min(self.max_argument_corrections, self.max_protocol_corrections) < 0:
            raise ValueError("Correction limits must be nonnegative")
        if self.model_retries < 0:
            raise ValueError("Model retries must be nonnegative")
        if self.retry_backoff_initial < 0 or self.retry_backoff_max < 0:
            raise ValueError("Retry backoff must be nonnegative")


DEFAULT_PROMPT = (
    "Help the user using the available evidence and tools. Ask for missing information or authorization. "
    "Report incomplete work and uncertainty honestly. Answer naturally; no action JSON is required."
)


class Agent:
    """One loop; identities, persistence and execution policy belong to the program."""

    def __init__(
        self,
        model,
        tools,
        store,
        *,
        config=None,
        context=None,
        memory=None,
        system_prompt=DEFAULT_PROMPT,
        user_channel=True,
    ):
        self.model, self.tools, self.store = model, tools, store
        self.config = config or AgentConfig()
        self.context = context or ContextManager(system_prompt)
        self.memory = memory or Memory(store)
        # Whether the action space can deliver a free-text reply to the user.
        # Tool-only benchmarks (AppWorld, BFCL, ...) have no such action: their
        # tasks finish through a tool, so a plain text answer there is a stopped
        # run, not a completed conversation. See the completion branch below.
        self.user_channel = user_channel

    async def run(self, message: str, *, user_id="default", session_id="default") -> RunResult:
        if not all(isinstance(x, str) and x.strip() for x in (message, user_id, session_id)):
            raise ValueError("User, session and message must be nonempty strings")
        run = self.store.begin(user_id, session_id, message)
        result = RunResult(run)
        tools = self.tools
        corrections = protocol_corrections = 0
        feedback = ""
        try:
            unresolved = self.store.unresolved(user_id, session_id)
            if unresolved:
                # Surface the unknown outcome once, then let later turns proceed.
                # The old code returned here on *every* run and nothing could ever
                # clear the attempt row, so one write timeout welded the session
                # shut: no model call was made again. The escalation is the
                # discharge; afterwards the write stays visible in
                # ``run_state.unresolved_writes`` so the model can verify it before
                # repeating anything -- the runtime still never replays it.
                self.store.escalate_writes(
                    user_id, session_id, run, [row["call"] for row in unresolved]
                )
                self.store.event(
                    run,
                    "unresolved_escalation",
                    {
                        "calls": [
                            {"call": row["call"], "name": row["name"]}
                            for row in unresolved
                        ]
                    },
                )
                result.status, result.reason = "waiting_user", "unresolved_write"
                result.answer = (
                    "A previous write has an unknown result. Check its status before continuing."
                )
                return result
            pending_writes = self.store.unknown_writes(user_id, session_id)
            memories = self.memory.retrieve(user_id, message)
            for step in range(self.config.max_steps):
                result.steps = step + 1
                state = {
                    "step": result.steps,
                    "limit": self.config.max_steps,
                    "feedback": feedback,
                }
                if pending_writes:
                    # Escalated, not forgotten: the model sees the unresolved calls
                    # each step and is told to verify them, without the runtime
                    # guessing the outcome or replaying the write itself.
                    state["unresolved_writes"] = [
                        {"call": row["call"], "name": row["name"]}
                        for row in pending_writes
                    ]
                request, info = self.context.prepare(
                    self.model,
                    self.store.messages(user_id, session_id),
                    tools.definitions(),
                    memories,
                    state,
                )
                model_call = uuid4().hex
                self.store.event(
                    run,
                    "model_request",
                    {
                        "model_call_id": model_call,
                        **info,
                        "sha256": hashlib.sha256(request.body).hexdigest(),
                        "body": json.loads(request.body),
                    },
                )
                try:
                    response = await self._complete_with_retries(run, request, model_call)
                    self._validate_response(response)
                except (Exception, asyncio.CancelledError) as exc:
                    usage = getattr(exc, "usage", None)
                    self.store.event(
                        run,
                        "model_error",
                        {
                            "model_call_id": model_call,
                            "category": getattr(exc, "kind", type(exc).__name__),
                            "diagnostics": getattr(exc, "diagnostics", {}),
                            "usage": asdict(usage) if usage is not None else None,
                            "usage_known": usage is not None,
                        },
                    )
                    if (
                        isinstance(exc, ModelError)
                        and exc.kind == "protocol_error"
                        and protocol_corrections < self.config.max_protocol_corrections
                    ):
                        # Deliberately not gated on ``usage``: providers that
                        # omit the usage field would otherwise lose the entire
                        # correction path and turn a recoverable malformed reply
                        # into a fatal run. A wasted correction costs one call;
                        # a dead endpoint does not look like this error class.
                        protocol_corrections += 1
                        feedback = "The response violated the provider protocol. Use native tool calls or ordinary text. Prior tools must not be replayed."
                        self.store.event(
                            run, "protocol_correction", {"number": protocol_corrections}
                        )
                        continue
                    raise
                self.store.event(
                    run, "model_response", {"model_call_id": model_call, **asdict(response)}
                )
                feedback = ""
                assistant_message = {"role": "assistant", "content": response.content}
                if response.provider_metadata:
                    assistant_message["_provider_metadata"] = response.provider_metadata
                if not response.tool_calls:
                    self.store.append(
                        user_id, session_id, run, assistant_message
                    )
                    if self.user_channel:
                        result.status, result.answer = "completed", response.content
                    else:
                        # There is nowhere to deliver this text. On a tool-only
                        # benchmark the task finishes through its own tool, so a
                        # text-only ending means the run stopped without
                        # finishing -- an agent failure that is scored, not a
                        # completed conversation that silently scores zero. The
                        # answer is kept for the record and the adapter log.
                        result.status, result.error_source = "failed", "agent"
                        result.reason = "no_user_channel"
                        result.answer = response.content
                    break
                calls = [
                    ToolCall(uuid4().hex, call.name, call.arguments) for call in response.tool_calls
                ]
                self.store.event(
                    run,
                    "call_correlation",
                    {
                        "model_call_id": model_call,
                        "calls": [
                            {"provider_id": original.id, "call_id": call.id}
                            for original, call in zip(response.tool_calls, calls, strict=True)
                        ]
                    },
                )
                self.store.append(
                    user_id,
                    session_id,
                    run,
                    {
                        **assistant_message,
                        "tool_calls": [call.to_message() for call in calls],
                    },
                )
                completed = set()
                try:
                    for call in calls:
                        spec = tools.specs.get(call.name)

                        def record(attempt, status, body, call=call, spec=spec):
                            self.store.attempt(
                                run,
                                user_id,
                                session_id,
                                call.id,
                                call.name,
                                spec.read_only if spec else True,
                                attempt,
                                status,
                                body,
                            )
                            self.store.event(
                                run,
                                "tool_attempt",
                                {
                                    "call_id": call.id,
                                    "name": call.name,
                                    "attempt": attempt,
                                    "status": status,
                                    **body,
                                },
                            )

                        outcome = await tools.execute(call, record)
                        body = outcome.to_dict()
                        self.store.append(
                            user_id,
                            session_id,
                            run,
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": dumps(body),
                            },
                        )
                        completed.add(call.id)
                        if (
                            outcome.status == "unknown"
                            and spec is not None
                            and not spec.read_only
                        ):
                            # This turn already escalates the unknown outcome in its
                            # answer (the WAIT decision ends the run), so record the
                            # delivery now; otherwise the next run would be stopped
                            # by the same row a second time.
                            self.store.escalate_writes(
                                user_id, session_id, run, [call.id]
                            )
                        if outcome.category == "arguments":
                            corrections += 1
                            self.store.event(
                                run,
                                "argument_correction",
                                {"call_id": call.id, "number": corrections},
                            )
                            if corrections > self.config.max_argument_corrections:
                                result.status, result.error_source = "failed", "agent"
                                result.reason = "argument_correction_limit"
                                break
                        if outcome.decision == Decision.WAIT:
                            result.status, result.reason = "waiting_user", outcome.category
                            result.answer = (
                                outcome.error or "User input or operation status is required."
                            )
                            break
                        if outcome.decision == Decision.TERMINATE:
                            result.status, result.error_source = "failed", "agent"
                            result.reason = outcome.category
                            break
                finally:
                    # Every emitted call gets a result, even if a prior call stops/cancels a batch.
                    for call in calls:
                        if call.id in completed:
                            continue
                        row = self.store.db.execute(
                            "SELECT status,body FROM attempts WHERE call=? ORDER BY attempt DESC LIMIT 1",
                            (call.id,),
                        ).fetchone()
                        body = json.loads(row["body"]) if row else {}
                        body.update(
                            status=row["status"] if row else "not_dispatched",
                            error="Run stopped; inspect execution evidence",
                        )
                        self.store.append(
                            user_id,
                            session_id,
                            run,
                            {"role": "tool", "tool_call_id": call.id, "content": dumps(body)},
                        )
                if result.status is not None:
                    break
            else:
                result.status, result.error_source = "failed", "agent"
                result.reason = "step_limit"
        except asyncio.CancelledError:
            # A cancellation is not a result. ``CancelledError`` is a
            # ``BaseException``, so nothing above catches it, and the adapter
            # used to swallow it into ``None`` -- which read as "the run
            # finished". Record that the turn was cut short and let the
            # cancellation travel; whoever cancelled is the one who knows what
            # it means. ``error_source`` stays None: this is not a fault.
            result.status, result.error_source = "failed", None
            result.reason = "cancelled"
            raise
        except (StorageError, sqlite3.Error) as exc:
            result.status, result.error_source = "failed", "infra"
            result.reason = "storage_error"
            fatal = exc
        except ContextBudgetError as exc:
            # Not infrastructure: the endpoint answered, the runtime could
            # not fit its own request. Retrying reproduces this exactly,
            # so blaming ``infra`` would send the operator in circles.
            result.status, result.error_source = "failed", "agent"
            result.reason = getattr(exc, "kind", type(exc).__name__)
            fatal = exc
        except (ModelError, TimeoutError) as exc:
            result.status, result.error_source = "failed", "infra"
            result.reason = getattr(exc, "kind", type(exc).__name__)
            fatal = exc
        except Exception as exc:
            result.status, result.error_source = "failed", "infra"
            result.reason = "internal_error"
            try:
                self.store.event(run, "internal_error", {"type": type(exc).__name__})
            except Exception:  # pragma: no cover - the store is already gone
                pass
            fatal = exc
        else:
            fatal = None
        finally:
            # Structural only: counts settled mutating dispatches recorded by the executor.
            # Deliberately does not read the answer text, the user's wording, or the goal.
            result.goal_status = self._goal_status(run)
            # Persist before anyone can observe this turn's end. The adapter
            # releases the coordinator immediately afterwards, and the harness
            # thread wakes up on that release; if the terminal record were
            # written after the release, the harness could score a run whose
            # evidence does not exist yet.
            self.store.finish(result)
        if fatal is not None:
            # The run is durable at this point, so raising cannot lose it.
            # Trailing the ``try`` keeps the write in ``finally`` (it must run
            # on every path) while still refusing to return a failed/infra
            # result as if it were an answer.
            raise FatalRunError(result, fatal) from fatal
        return result

    def _goal_status(self, run):
        """Report what the runtime observed about state change, never that a goal was met.

        Why this is not "did the agent finish?" -- that question belongs to the user. The
        program cannot tell a correctly read-only request from a run that stopped short,
        and inferring intent is exactly what this runtime refuses to do elsewhere
        (`not_inferred` in task_handoff). So the field states an observation:

          no_state_change_attempted  no mutating call ever reached a handler
          state_change_unconfirmed   mutating calls ran, none of them reported success
          state_change_observed      at least one mutating call reported success

        In experiments/retail-qwen-tier-order-20261108 these separate sharply on the
        official score (35 failures / 115 passes): runs that never dispatched a write
        failed 62% of the time, runs with an observed success 12%. A run may still be
        wrong after an observed change, so this is a signal, not a verdict.
        """
        attempted = confirmed = 0
        for row in self.store.db.execute(
            "SELECT read_only,status,body FROM attempts WHERE run=? ORDER BY rowid", (run,)
        ):
            if row["read_only"]:
                continue
            try:
                body = json.loads(row["body"])
            except (ValueError, TypeError):
                body = {}
            # dispatch_intent is the conservative pre-dispatch marker, not an execution.
            if body.get("phase") == "dispatch_intent":
                continue
            if row["status"] not in {"returned", "unknown"}:
                continue
            attempted += 1
            if row["status"] == "returned" and body.get("ok") is True:
                confirmed += 1
        if not attempted:
            return "no_state_change_attempted"
        return "state_change_observed" if confirmed else "state_change_unconfirmed"

    @staticmethod
    def _retryable(exc) -> bool:
        """Whether a model-call failure is worth repeating.

        This mirrors what the baseline actually gets from litellm + openai:
        retry 408 (request timeout), 409 (lock timeout), 429 (rate limit) and
        every 5xx, plus timeouts and connection drops -- which arrive here as
        ``ModelError(kind="http")`` with no status, or as ``asyncio.TimeoutError``
        from the deadline, or as ``httpx.HTTPError`` wrapped into ``kind="http"``.

        Everything else is a property of the request or the account, and
        repeating it only burns the step budget: 400/404/422 are ``capability``,
        invalid JSON and malformed responses are ``protocol_error`` (those have
        their own, separately bounded correction path), and storage/cancel are
        terminal.
        """
        if isinstance(exc, asyncio.CancelledError):
            return False
        if isinstance(exc, asyncio.TimeoutError):
            return True
        if not isinstance(exc, ModelError):
            return False
        if exc.kind == "http":
            status = (exc.diagnostics or {}).get("http_status")
            if status is None:
                # Connection-level failure: no status to judge, treat as transient.
                return True
            return status in {408, 409, 429} or status >= 500
        if exc.kind == "capability":
            return False
        if exc.kind == "protocol_error":
            return False
        # Default kind is "model_error": a raised transport failure.
        return True

    def _retry_delay(self, attempt: int) -> float:
        """Backoff matching openai's: 0.5s doubling, capped at 8s, -25% jitter.

        Jitter here is deterministic-ish (only the multiplier varies), which is
        deliberate: a retry storm on a shared endpoint is what the jitter is
        for, and a reproducible test wants a boundable number.
        """
        base = self.config.retry_backoff_initial * (2.0**attempt)
        capped = min(base, self.config.retry_backoff_max)
        return max(0.0, capped * (1 - 0.25 * random.random()))

    async def _complete_with_retries(self, run, request, model_call):
        """One model call, retried on transient failures.

        The caller's ``try`` block sees either a response or the *last* error,
        so the error path downstream is unchanged: a permanent failure still
        raises, still records ``model_error``, and still ends the run. What
        changes is that a transient one no longer ends a session that the
        baseline would have finished.

        Each attempt is recorded, so a run's cost and latency stay auditable
        instead of a retry being invisible in the event log.
        """
        last: Exception | None = None
        for attempt in range(self.config.model_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.model.complete(request), self.config.model_timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                if attempt >= self.config.model_retries or not self._retryable(exc):
                    raise
                delay = self._retry_delay(attempt)
                self.store.event(
                    run,
                    "model_retry",
                    {
                        "model_call_id": model_call,
                        "attempt": attempt + 1,
                        "of": self.config.model_retries,
                        "delay_s": round(delay, 3),
                        "category": getattr(exc, "kind", type(exc).__name__),
                        "diagnostics": getattr(exc, "diagnostics", {}),
                    },
                )
                await asyncio.sleep(delay)
        raise last  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _validate_response(response):
        if not isinstance(response, ModelResponse):
            raise ModelError("Adapter returned invalid type", kind="protocol_error")
        valid = response.finish_reason == ("tool_calls" if response.tool_calls else "stop")
        ids = [call.id for call in response.tool_calls]
        if not valid or len(ids) != len(set(ids)) or any(not x for x in ids):
            raise ModelError(
                "Invalid response termination/correlation",
                kind="protocol_error",
                usage=response.usage if response.usage_known else None,
            )
        # The wire format requires ``arguments`` to be a JSON *object string*.
        # A reply cut off by ``max_tokens`` can still arrive with
        # ``finish_reason="tool_calls"``, so the termination check above cannot
        # see the truncation. Replaying such a message makes every later request
        # fail with a 400 that no retry can fix, so reject the turn here instead
        # of persisting it.
        for call in response.tool_calls:
            if not isinstance(call.arguments, str):
                continue
            if not call.arguments.strip():
                call.arguments = "{}"
                continue
            try:
                parsed = json.loads(call.arguments)
            except (TypeError, ValueError):
                raise ModelError(
                    "Tool call arguments are not valid JSON",
                    kind="protocol_error",
                    usage=response.usage if response.usage_known else None,
                ) from None
            if not isinstance(parsed, dict):
                raise ModelError(
                    "Tool call arguments must be a JSON object",
                    kind="protocol_error",
                    usage=response.usage if response.usage_known else None,
                )
        if not response.tool_calls and not (response.content or "").strip():
            raise ModelError(
                "Empty response",
                kind="protocol_error",
                usage=response.usage if response.usage_known else None,
            )
