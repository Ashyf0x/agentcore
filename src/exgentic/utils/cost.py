# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import Any, Optional, Self, Sequence, TypeVar

from pydantic import BaseModel, Field, computed_field, model_validator

name_map = {
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "DeepSeek-V3.2": "deepseek/deepseek-v3.2",
    "DeepSeek-V3": "deepseek/deepseek-v3",
    "Kimi-K2.5": "moonshot/kimi-k2.5",
}


class TokensCost(BaseModel):
    input_cost: float
    output_cost: float
    total_cost: float


def _cost_per_token(*, model: str, prompt_tokens: int, completion_tokens: int):
    from litellm.cost_calculator import cost_per_token

    return cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def litellm_cost_per_token(model_name: str):
    return _cost_per_token(model=model_name, prompt_tokens=1, completion_tokens=1)


def litellm_tokens_cost(input_tokens: int, output_tokens: int, model_name: str) -> TokensCost:
    for src, dst in name_map.items():
        model_name = model_name.replace(src, dst)

    parts = model_name.lower().split("/")
    for i in range(len(parts)):
        model = "/".join(parts[-i:])
        try:
            input_cost, output_cost = _cost_per_token(
                model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
            )
            return TokensCost(
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
            )
        except Exception:
            continue
    raise ValueError(
        f"No pricing info found for model '{model_name}'. "
        f"Add it to the name_map in exgentic/utils/cost.py or check litellm model support."
    )


class CostReport(BaseModel):
    _total_cost: float = 0
    model_name: str = ""

    # ``total_cost`` is computed over the private ``_total_cost``, and pydantic
    # never assigns private attributes from input. A report read back from
    # results.json would therefore report 0 no matter what the file said --
    # which is exactly what silently zeroed the run-level totals of every
    # session a resume reused. This shadow field reads the same ``total_cost``
    # key on the way in and is excluded on the way out, so the serialized shape
    # is unchanged (``model_name`` + ``total_cost``).
    restored_total_cost: float = Field(
        default=0.0, alias="total_cost", exclude=True, repr=False
    )

    @model_validator(mode="after")
    def _restore_total_cost(self) -> Self:
        """Apply a serialized ``total_cost`` when the subclass has no source for it.

        ``CostReport`` and ``UpdatableCostReport`` keep everything in
        ``_total_cost``, so their public total is 0 unless restored. An LLM
        report derives its total from ``input_cost + output_cost``; that value
        is authoritative and must not be overwritten by the round-tripped
        number, so it is only restored where the class does not override.
        """
        overrides = type(self).total_cost is not CostReport.total_cost
        if not overrides and self.restored_total_cost:
            self._total_cost = float(self.restored_total_cost)
        return self

    @classmethod
    def initialize_empty(cls, model_name: str = "") -> Self:
        """Create an empty cost report with zero costs."""
        return cls(model_name=model_name)

    @computed_field
    @property
    def total_cost(self) -> float:
        return self._total_cost

    def accumulate_from(self, other: Self) -> None:
        # Accumulate by total_cost only
        self._total_cost += float(other.total_cost)


def restore_report(value: Any) -> Any:
    """Rebuild the concrete report type from its serialized form.

    Serialization is fixed by ``SerializeAsAny`` on the field, but validation
    still resolves to the *declared* type, so a dictionary read back from
    ``results.json`` would otherwise become a bare ``CostReport`` and lose the
    token counts that distinguish "unpriced endpoint" from "no model calls".

    The wire format carries no type tag, so the shape decides:

    * token fields present -> ``LiteLLMCostReport``. That is the only concrete
      LLM report this codebase constructs, and with both costs present its
      ``model_post_init`` performs no price lookup, so restoring one never
      depends on ``litellm``'s price table.
    * otherwise -> left as a plain mapping, for ``CostReport`` to validate.

    ``CostReport`` and ``UpdatableCostReport`` serialize identically
    (``model_name`` + ``total_cost``) and accumulate identically, so the
    ambiguity between them is not worth a tag; ``total_cost`` is restored by
    ``CostReport._restore_total_cost`` either way.
    """
    if not isinstance(value, dict):
        return value
    if "input_tokens" in value or "output_tokens" in value:
        return LiteLLMCostReport.model_validate(value)
    return value


class UpdatableCostReport(CostReport):
    """A simple accumulator that supports adding arbitrary cost amounts."""

    def add_cost(self, new_cost: float) -> None:
        self._total_cost += new_cost


class LLMCostReport(CostReport):
    """Represents a cost report for LLM usage.

    Usage:
        # Explicit definition
        report = CostReport(model_name="gpt-4", input_tokens=100, output_tokens=50, input_cost=0.02, output_cost=0.03)

        # Empty report
        empty_report = CostReport.initialize_empty("gpt-4")
    """

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @classmethod
    def initialize_empty(cls, model_name: str = "") -> Self:
        """Create an empty cost report with zero tokens and costs."""
        return cls(
            model_name=model_name,
            input_tokens=0,
            output_tokens=0,
            input_cost=0,
            output_cost=0,
        )

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @computed_field
    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    def update_cost(self, input_tokens, output_tokens, input_cost, output_cost) -> None:
        """Update the report with additional tokens and costs."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.input_cost += input_cost
        self.output_cost += output_cost

    def accumulate_from(self, other: Self) -> None:
        """Accumulate costs and tokens from other."""
        self.input_tokens += int(other.input_tokens)
        self.output_tokens += int(other.output_tokens)
        self.input_cost += float(other.input_cost)
        self.output_cost += float(other.output_cost)


class LiteLLMCostReport(LLMCostReport):
    """Specialized cost report that calculates costs using LiteLLM pricing.

    Additional Features:
        - Auto-calculates cost if not provided.
        - Provides helper methods to update cost from token counts.

    Usage:
        report = LiteLLMCostReport.from_token_counts("gpt-4", 100, 50)
        report.update_cost_from_tokens(20, 10)
    """

    output_cost: Optional[float] = None
    input_cost: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.output_cost is None or self.input_cost is None:
            cost_data = LiteLLMCostReport.get_litellm_tokens_cost(
                self.input_tokens, self.output_tokens, model_name=self.model_name
            )
            self.output_cost = cost_data.output_cost
            self.input_cost = cost_data.input_cost

    @classmethod
    def from_token_counts(cls, model_name, input_tokens, output_tokens, *, strict: bool = True) -> "LiteLLMCostReport":
        """Create a report from token counts, auto-calculating costs.

        With ``strict=False`` a model that LiteLLM has no price for yields a
        zero-cost report that still carries the token counts. That is what
        self-hosted and custom OpenAI-compatible endpoints need: the run is
        about behaviour, and reporting no price is honest, while raising turns
        every such run into a crash.
        """
        try:
            cost_data = cls.get_litellm_tokens_cost(input_tokens, output_tokens, model_name=model_name)
        except ValueError:
            if strict:
                raise
            return cls(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=0.0,
                output_cost=0.0,
            )
        return cls(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=cost_data.input_cost,
            output_cost=cost_data.output_cost,
        )

    def update_cost_from_tokens(self, new_input_tokens, new_output_tokens):
        """Update cost based on new token counts."""
        new_cost_data = LiteLLMCostReport.get_litellm_tokens_cost(new_input_tokens, new_output_tokens, self.model_name)
        self.update_cost(
            new_input_tokens,
            new_output_tokens,
            new_cost_data.input_cost,
            new_cost_data.output_cost,
        )

    @classmethod
    def get_litellm_tokens_cost(cls, input_tokens, output_tokens, model_name) -> TokensCost:
        """Fetch cost data from LiteLLM pricing API."""
        if input_tokens == 0 and output_tokens == 0:
            return TokensCost(input_cost=0, output_cost=0, total_cost=0)
        return litellm_tokens_cost(input_tokens, output_tokens, model_name=model_name)


T = TypeVar("T", bound=CostReport)


def accumulate_reports(reports: Sequence[T]) -> T:
    """Accumulate a sequence of same-typed cost reports into a single report of the same type.

    Behavior:
      - Enforces that all items are of the same concrete type.
      - Preserves `model_name` if identical across items; else uses 'mixed'.
      - Uses the report's own `accumulate_from` implementation to merge.

    Args:
      reports: non-empty sequence of reports, all of the same type.

    Returns:
      A new report of the same type, containing the accumulated data.

    Raises:
      ValueError: if the list is empty or contains mixed types.
    """
    if not reports:
        raise ValueError("The reports list cannot be empty.")

    first = reports[0]
    first_type = type(first)

    if not all(type(r) is first_type for r in reports):
        raise ValueError("All reports must be of the same concrete type.")

    # Preserve model_name if consistent; otherwise use 'mixed'
    first_model_name = first.model_name
    same_model_name = all(r.model_name == first_model_name for r in reports)
    acc_report = first_type.initialize_empty(model_name=first_model_name if same_model_name else "mixed")

    # Fold via the type's own accumulation logic
    for r in reports:
        acc_report.accumulate_from(r)

    return acc_report
