# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

from ...utils.container_reaper import docker_sdk_label_injection
from ...utils.logging import capture_stdio_to_session
from ...utils.paths import SessionPaths


@dataclass
class HarnessResult:
    harness_report: dict[str, Any]
    patch: str
    patch_valid: bool
    harness_ran: bool
    error: str | None = None


SPEC_CACHE_URL_PREFIX = "https://raw.githubusercontent.com/"


class _SpecFileResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


class _SpecFileFetcher:
    """Fetch SWE-bench spec files from a local cache before hitting the network.

    The harness downloads a few small requirement/environment files while it
    builds test specs. Those files are only consumed when a Docker image has to
    be built from scratch, so runs that reuse prebuilt images do not need their
    content at all. This shim keeps such runs usable without outbound network.
    """

    def __init__(self, logger: Logger, allow_offline: bool, timeout: float) -> None:
        self._logger = logger
        self._allow_offline = allow_offline
        self._timeout = timeout
        self._cache_dir = Path(
            os.environ.get("EXGENTIC_SWEBENCH_SPEC_CACHE", str(Path.home() / ".exgentic" / "swebench-spec-cache"))
        )
        self._warned = set()

    def _cache_path(self, url: str) -> Path | None:
        if not url.startswith(SPEC_CACHE_URL_PREFIX):
            return None
        return self._cache_dir / url[len(SPEC_CACHE_URL_PREFIX) :]

    def get(self, url: str, **kwargs) -> Any:
        cache_path = self._cache_path(url)
        if cache_path is not None and cache_path.is_file():
            return _SpecFileResponse(cache_path.read_text(encoding="utf-8"))
        kwargs.setdefault("timeout", self._timeout)
        try:
            import requests

            response = requests.get(url, **kwargs)
        except Exception as exc:
            if not self._allow_offline:
                raise
            if url not in self._warned:
                self._warned.add(url)
                self._logger.warning(f"EVAL | Spec file unavailable ({exc}); falling back to empty body: {url}")
            return _SpecFileResponse("")
        if cache_path is not None and response.status_code == 200:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(response.text, encoding="utf-8")
            except OSError:
                pass
        return response


def install_spec_file_cache(logger: Logger) -> None:
    """Route harness spec-file downloads through the local cache."""
    if os.environ.get("EXGENTIC_SWEBENCH_SPEC_CACHE_DISABLE") == "1":
        return
    allow_offline = os.environ.get("EXGENTIC_SWEBENCH_SPEC_OFFLINE", "1") != "0"
    timeout = float(os.environ.get("EXGENTIC_SWEBENCH_SPEC_FETCH_TIMEOUT", "10"))
    fetcher = _SpecFileFetcher(logger, allow_offline, timeout)
    from swebench.harness.test_spec import python as spec_python

    spec_python.requests = fetcher
    try:
        from swebench.harness import utils as harness_utils
    except Exception:
        return
    harness_utils.requests = fetcher


def is_patch_valid(patch: str) -> bool:
    if not patch or not patch.strip():
        return False
    return any(m in patch for m in ["---", "+++", "@@", "diff --git", "*** Begin Patch"])


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_harness(
    patch: str,
    instance_id: str,
    subset: str,
    paths: SessionPaths,
    eval_config: dict[str, Any],
    logger: Logger,
) -> HarnessResult:
    patch = patch or ""
    valid = is_patch_valid(patch)

    pred_path = paths.benchmark_dir / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_patch": patch,
                "patch_valid": valid,
                "model_name_or_path": "exgentic",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    logger.info(f"EVAL | Writing patch | valid: {valid} | size: {len(patch)}")

    if not valid:
        logger.warning("EVAL | Skipping harness - invalid patch structure")
        return HarnessResult({}, patch, False, False)

    logger.info("EVAL | Running SWE-bench harness evaluation")
    try:
        install_spec_file_cache(logger)
        from swebench.harness import run_evaluation

        with (
            _pushd(paths.benchmark_dir),
            capture_stdio_to_session(logger),
            docker_sdk_label_injection(),
        ):
            report_path = run_evaluation.main(
                dataset_name=subset,
                split="test",
                instance_ids=[instance_id],
                predictions_path=pred_path.name,
                max_workers=eval_config["max_workers"],
                run_id="exgentic1",
                namespace="swebench",
                force_rebuild=False,
                cache_level=eval_config["cache_level"],
                clean=False,
                open_file_limit=eval_config["open_file_limit"],
                timeout=eval_config["harness_timeout"],
                rewrite_reports=False,
                modal=False,
            )
            report = json.loads(Path(report_path).read_text()) if report_path and Path(report_path).is_file() else {}
        logger.info("EVAL | Harness evaluation completed")
        return HarnessResult(report, patch, True, True)
    except Exception as e:
        logger.exception(f"EVAL | Harness evaluation failed: {e}")
        return HarnessResult({}, patch, True, True, str(e))
