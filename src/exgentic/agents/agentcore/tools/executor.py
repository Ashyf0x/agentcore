"""Single execution owner for local tools. Handlers never choose retry policy."""

import asyncio
import inspect
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

from jsonschema import Draft202012Validator

from ..core.errors import Decision, ErrorInfo, ToolFailure, decide
from ..core.types import dumps


@dataclass(frozen=True)
class ToolSpec:
    """One tool plus the safety facts the executor needs before dispatching it.

    ``read_only`` is the only semantic marker. It has no default on purpose:
    an interrupted dispatch must escalate, and a wrong "this was only a read"
    is not recoverable. Callers have to state it.

    The executor makes one dispatch attempt. It never retries or reconciles a
    call behind the model's back: a failed call is returned as feedback, and an
    interrupted write becomes ``WAIT`` because its side effects are unknown.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable
    read_only: bool
    timeout: float = 10

    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }


@dataclass
class ToolResult:
    status: str
    ok: bool = False
    value: object = None
    error: str | None = None
    category: str | None = None
    decision: str = "feedback"
    attempts: int = 0
    error_info: dict | None = None

    def to_dict(self):
        return asdict(self)


async def invoke(handler, **kwargs):
    if inspect.iscoroutinefunction(handler):
        return await handler(**kwargs)
    value = await asyncio.to_thread(handler, **kwargs)
    return await value if inspect.isawaitable(value) else value


class ToolRegistry:
    def __init__(self, specs=()):
        self.specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec):
        if not spec.name or spec.name in self.specs:
            raise ValueError("Tool names must be nonempty and unique")
        if not math.isfinite(spec.timeout) or spec.timeout <= 0:
            raise ValueError("Invalid tool execution limits")
        if spec.parameters.get("type") != "object":
            raise ValueError("Tool parameters must be an object")
        Draft202012Validator.check_schema(spec.parameters)
        self.specs[spec.name] = spec

    def definitions(self):
        return [s.definition() for s in self.specs.values()]

    async def execute(self, call, record):
        """Execute once. ``record`` must durably save intent before invocation."""
        spec = self.specs.get(call.name)
        try:
            if spec is None:
                raise ToolFailure("Unknown tool", "arguments")
            args = (
                json.loads(call.arguments)
                if isinstance(call.arguments, str)
                else deepcopy(call.arguments)
            )
            dumps(args)  # reject NaN and non-JSON values
            errors = list(Draft202012Validator(spec.parameters).iter_errors(args))
            if errors:
                # Schema path only: validation messages can echo secrets from argument values.
                field = ".".join(map(str, errors[0].absolute_path)) or "arguments"
                if errors[0].validator == "required" and isinstance(args, dict):
                    field = next(
                        (key for key in errors[0].validator_value if key not in args), field
                    )
                raise ToolFailure(f"Invalid field {field} ({errors[0].validator})", "arguments")
        except (ValueError, TypeError):
            result = ToolResult(
                "not_dispatched", error="Arguments must be valid JSON", category="arguments"
            )
            record(0, result.status, result.to_dict())
            return result
        except ToolFailure as exc:
            decision = decide(ErrorInfo("validation", exc.category, False, True, 0))
            result = ToolResult(
                "not_dispatched", error=str(exc), category=exc.category, decision=decision
            )
            record(0, result.status, result.to_dict())
            return result

        # Conservative intent: a crash between this commit and invocation is still unknown.
        record(1, "unknown", {"phase": "dispatch_intent"})
        try:
            value = await asyncio.wait_for(invoke(spec.handler, **args), spec.timeout)
            if isinstance(value, ToolResult):
                if value.status != "returned":
                    raise ValueError("A returned tool result must have known outcome")
                dumps(value.to_dict())
                result = value
                result.attempts = 1
            else:
                dumps(value)
                result = ToolResult("returned", ok=True, value=value, attempts=1)
        except asyncio.CancelledError:
            record(1, "unknown", {"category": "cancelled", "diagnostic": "CancelledError"})
            raise
        except (ToolFailure, TimeoutError) as exc:
            known = isinstance(exc, ToolFailure) and exc.outcome_known
            category = exc.category if isinstance(exc, ToolFailure) else "transient"
            info = ErrorInfo("execution", category, True, known, 1, spec.read_only)
            result = ToolResult(
                "returned" if known else "unknown",
                error=str(exc)
                if isinstance(exc, ToolFailure)
                else "Tool timed out; outcome unknown",
                category=category,
                decision=decide(info),
                attempts=1,
                error_info=asdict(info),
            )
        except Exception as exc:
            result = ToolResult(
                "unknown",
                error="Internal tool error",
                category="internal",
                decision=Decision.TERMINATE,
                attempts=1,
            )
            record(
                1, result.status, {**result.to_dict(), "diagnostic": type(exc).__name__}
            )
            return result
        record(1, result.status, result.to_dict())
        return result