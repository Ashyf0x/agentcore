# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""AppWorld benchmark adapter -- light benchmark class only.

Evaluator and session classes live in ``appworld_eval.py`` and are loaded
inside the runner subprocess via ``_get_evaluator_class()`` and
``_get_session_class()``.  This file must remain importable without the
``appworld`` package installed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.benchmark import Benchmark
from ...core.types import SingleObservation


class AppWorldObservation(SingleObservation):
    pass


class AppWorldBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "AppWorld"
    slug_name: ClassVar[str] = "appworld"
    available_subsets: ClassVar[list[str]] = ["train", "dev", "test_normal"]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.appworld.appworld_eval:AppWorldEvaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.appworld.appworld_eval:AppWorldSession"

    # Inputs
    subset: Literal["train", "dev", "test_normal"] = "test_normal"
    env_kwargs: dict[str, Any] = Field(default_factory=dict)
    max_interactions: int = 200
    tool_name_separator: Literal[".", "__"] = "__"
    # ``full`` injects every AppWorld API as a named tool (468 tools, ~274K chars
    # of parameter schemas). ``on_demand`` exposes only the four official
    # ``api_docs`` discovery tools plus one generic executor, so the model reads
    # a schema only for APIs it actually needs. Keep ``full`` as the default:
    # the 2026-11-14 v2 runs were produced with it and must stay reproducible.
    tool_schema_mode: Literal["full", "on_demand"] = "full"
    SCORES_FILE_NAME: ClassVar[str] = "scores.json"

    def list_subsets(self) -> list[str]:  # type: ignore[override]
        return list(self.available_subsets)

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {"subset": self.subset}

    def _get_session_kwargs(self) -> dict[str, Any]:
        return {
            "env_kwargs": self.env_kwargs,
            "max_interactions": self.max_interactions,
            "tool_name_separator": self.tool_name_separator,
            "tool_schema_mode": self.tool_schema_mode,
            "use_cache": self.use_cache,
        }
