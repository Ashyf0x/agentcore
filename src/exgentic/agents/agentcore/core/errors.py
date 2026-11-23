from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    FEEDBACK = "feedback"
    WAIT = "wait_user"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class ErrorInfo:
    stage: str
    category: str
    dispatched: bool
    outcome_known: bool
    attempt: int
    read_only: bool = False


def decide(error: ErrorInfo) -> Decision:
    """Return the only safe next step for a failed tool attempt.

    The executor performs exactly one dispatch. There is no automatic retry:
    a known failure is handed back to the model, while an interrupted write is
    escalated because its side effects cannot be established.
    """
    if error.category in {"internal", "storage", "cancelled"}:
        return Decision.TERMINATE
    if error.category in {"missing_input", "authorization"}:
        return Decision.WAIT
    if error.dispatched and not error.outcome_known and not error.read_only:
        return Decision.WAIT
    return Decision.FEEDBACK


class ToolFailure(Exception):
    """Trusted handler contract; message must be safe for model/user display."""

    def __init__(self, message: str, category="business", *, outcome_known=True):
        super().__init__(message)
        self.category, self.outcome_known = category, outcome_known
