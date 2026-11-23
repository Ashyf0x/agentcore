import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    # A provider correlation ID, replaced by the runtime before execution/replay.
    id: str
    name: str
    arguments: dict[str, Any] | str

    def to_message(self) -> dict:
        args = self.arguments if isinstance(self.arguments, str) else dumps(self.arguments)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": args},
        }


@dataclass
class ModelResponse:
    content: str | None = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    usage_known: bool = False
    # Opaque replay data owned by the model adapter, never treated as facts or an answer.
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRequest:
    """Immutable exact wire body: measured by Context and sent unchanged by the adapter.

    There is no byte ceiling here on purpose. A serialized-size limit measures the
    limit rather than the model, and it fails *before* the model can act: the tau2
    policies alone are 7KB (airline), 7KB (retail) and 23KB (telecom), so a 28KB
    pilot ceiling rejected the very first request of a long-policy task. What
    actually bounds a request is the model's context window, which
    ``ContextManager`` enforces in tokens (see ``prepare``).
    """

    body: bytes

    def validate(self) -> None:
        """Reject a body that is not serialized model input at all.

        Kept as the adapter's single pre-send guard; it no longer measures size.
        """
        if not self.body:
            raise ContextBudgetError("Request body is empty")


class ModelClient(Protocol):
    def prepare(self, messages: list[dict], tools: list[dict]) -> PreparedRequest: ...
    async def complete(self, request: PreparedRequest) -> ModelResponse: ...


#: How a turn ended. Three values, not six: see ``RunResult``.
ExitStatus = Literal["completed", "waiting_user", "failed"]

#: Who owns a ``failed`` turn.
#:
#: ``infra`` -- the runtime could not do its job (endpoint hung, response
#: unparsable, store died). The harness must not score this as a wrong answer.
#: ``agent`` -- the runtime worked and the run genuinely ran out (step limit,
#: argument corrections, a tool that said terminate). That is a real result.
ErrorSource = Literal["infra", "agent"]


@dataclass
class RunResult:
    """How one ``Agent.run()`` ended.

    ``status`` is an *exit* label, not a state-machine node. The loop's shape
    lives in ``Agent.run``; this field only says how the turn left it. Keeping
    the two apart is what removed ``running`` (a node that was masquerading as
    a default), ``interrupted`` (produced only by ``Store.recover``, which has
    no caller) and ``cancelled`` (the label died in the adapter before anyone
    could read it -- ``CancelledError`` is a ``BaseException`` and was swallowed
    on the way out).

    ``status`` starts as ``None`` on purpose. A default of ``"running"`` meant
    any path that forgot to assign one wrote a run that never terminates, and
    the ``active_session`` unique index then refused every later run in that
    session. ``Store.finish`` rejects ``None`` instead, so a missed assignment
    fails loudly at the write instead of poisoning the table.
    """

    run_id: str
    status: ExitStatus | None = None
    # Orthogonal to ``status``: "failed" alone cannot say whether the harness
    # should score the run or abort it. See ``RunResult.error_source``.
    error_source: ErrorSource | None = None
    answer: str = ""
    # What the runtime observed about state change -- never a claim that the goal was met.
    # A run that only read data and a run that changed nothing look identical from here;
    # distinguishing them needs the user's intent, which the program must not infer.
    goal_status: Literal[
        "unverified",
        "no_state_change_attempted",
        "state_change_unconfirmed",
        "state_change_observed",
    ] = "unverified"
    reason: str | None = None
    steps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ModelError(RuntimeError):
    def __init__(self, message: str, *, kind="model_error", diagnostics=None, usage=None):
        super().__init__(message)
        self.kind, self.diagnostics, self.usage = kind, diagnostics or {}, usage


class ContextBudgetError(RuntimeError):
    pass


class StorageError(RuntimeError):
    """A run cannot claim a durable terminal state if storage failed."""


class FatalRunError(RuntimeError):
    """A turn ended because the runtime could not do its job.

    Raised *after* the terminal result has been written to the store, so a
    caller that only sees the exception can still read what happened. This is
    the signal the adapter turns into an exception on the harness thread: a
    scored ``failed``/``agent`` turn returns normally, a ``failed``/``infra``
    turn must not look like a finished conversation.

    ``result`` is the same object already persisted; ``cause`` keeps the
    original exception for the traceback the operator will read.
    """

    def __init__(self, result: RunResult, cause: BaseException | None = None) -> None:
        self.result = result
        self.cause = cause
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        super().__init__(f"agentcore run failed: {result.reason}{detail}")

    @property
    def error_source(self) -> ErrorSource | None:
        """Mirror ``result.error_source`` so a caller holding only the exception
        can tell an infrastructural fault from an agent fault without having to
        know the ``result`` layout. The harness reads this attribute."""
        return self.result.error_source

    def __reduce__(self):
        """Survive the trip across a process boundary.

        The runner serialises agent exceptions with ``cloudpickle``. Default
        pickling of an exception replays ``__init__(*self.args)`` -- but
        ``args`` here is the formatted message, not ``(result, cause)``, so
        reconstruction raises ``AttributeError`` inside ``__init__``. The
        transport then falls back to "rebuild a builtin from the type name",
        which silently drops ``error_source``: the harness would record an
        endpoint failure as an agent bug.

        Both payload members are picklable (``RunResult`` is a dataclass of
        primitives; ``cause`` is the original exception), so the tuple form
        round-trips exactly.
        """
        return (self.__class__, (self.result, self.cause))


class SessionBusy(RuntimeError):
    pass
