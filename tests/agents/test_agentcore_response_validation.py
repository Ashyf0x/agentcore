# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Response validation must reject tool calls that cannot be replayed.

Gateways keep ``finish_reason="tool_calls"`` even when ``max_tokens`` cut the
reply mid-argument, so the termination check alone cannot see truncation. The
malformed ``arguments`` string would then live on in the stored history and make
every later request fail with an HTTP 400 that no retry can recover from.
"""

from __future__ import annotations

import pytest

from exgentic.agents.agentcore.core.agent import Agent
from exgentic.agents.agentcore.core.types import ModelError, ModelResponse, ToolCall, Usage


def _response(arguments, *, finish_reason="tool_calls"):
    return ModelResponse(
        content=None,
        tool_calls=[ToolCall("call-1", "arithmetic__add", arguments)],
        finish_reason=finish_reason,
        usage=Usage(input_tokens=10, output_tokens=20),
        usage_known=True,
    )


def test_valid_arguments_pass():
    response = _response('{"a": 1, "b": 2}')

    Agent._validate_response(response)


def test_empty_arguments_are_normalised_to_an_object():
    response = _response("   ")

    Agent._validate_response(response)

    assert response.tool_calls[0].arguments == "{}"


def test_truncated_arguments_are_a_protocol_error():
    with pytest.raises(ModelError) as excinfo:
        Agent._validate_response(_response('{"query": "list", "access_token": "eyJhbGci'))

    assert excinfo.value.kind == "protocol_error"
    assert excinfo.value.usage is not None


def test_non_object_arguments_are_a_protocol_error():
    with pytest.raises(ModelError) as excinfo:
        Agent._validate_response(_response("[1, 2, 3]"))

    assert excinfo.value.kind == "protocol_error"


def test_termination_check_still_wins():
    with pytest.raises(ModelError) as excinfo:
        Agent._validate_response(_response("{}", finish_reason="length"))

    assert excinfo.value.kind == "protocol_error"