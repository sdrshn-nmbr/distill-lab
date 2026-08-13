import asyncio
import os
import sys
from pathlib import Path

import pytest

from distill_lab.codex_app_server import CodexAppServerBackend
from distill_lab.teacher import (
    CandidateToken,
    GenerationRequest,
    TeacherTransportError,
    TokenSelectionRequest,
)

FAKE_SERVER = r"""
import json
import os
import sys

mode = os.environ.get("DISTILL_FAKE_MODE", "success")
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        print(json.dumps({"id": request_id, "result": {}}), flush=True)
    elif method == "model/list":
        response = {"id": request_id, "result": {"data": [{"model": "gpt-5.6-terra"}]}}
        print(json.dumps(response), flush=True)
    elif method == "thread/start":
        if mode == "require_tool_free":
            features = message["params"]["config"]["features"]
            assert features and all(value is False for value in features.values())
        response = {"id": request_id, "result": {"thread": {"id": "thread-1"}}}
        print(json.dumps(response), flush=True)
    elif method == "turn/start":
        if mode == "die":
            sys.exit(7)
        if mode == "terminal_rpc_error":
            error = {"id": request_id, "error": {"code": -32602, "message": "sensitive"}}
            print(json.dumps(error), flush=True)
            continue
        print(json.dumps({"id": request_id, "result": {"turn": {"id": "turn-1"}}}), flush=True)
        if mode == "hang":
            continue
        schema = json.dumps(message["params"].get("outputSchema", {}))
        if "selected_token_id" in schema:
            valid = json.dumps({"results": [{"selected_token_id": 10}]})
        else:
            valid = json.dumps({"results": [{"text": "Add pinapple."}]})
        text = "not-json" if mode == "malformed" else valid
        usage = {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "tokenUsage": {"last": {"outputTokens": 3}},
            },
        }
        item = {
            "method": "item/completed",
            "params": {
                "turnId": "turn-1",
                "item": {"type": "agentMessage", "text": text},
            },
        }
        completed = {
            "method": "turn/completed",
            "params": {"turn": {"id": "turn-1", "status": "completed"}},
        }
        print(json.dumps(usage), flush=True)
        if mode == "plan_item":
            plan = {
                "method": "item/completed",
                "params": {
                    "turnId": "turn-1",
                    "item": {"type": "plan", "text": "Choose one candidate."},
                },
            }
            print(json.dumps(plan), flush=True)
        if mode == "tool_item":
            tool = {
                "method": "item/completed",
                "params": {
                    "turnId": "turn-1",
                    "item": {"type": "commandExecution", "command": "sensitive"},
                },
            }
            print(json.dumps(tool), flush=True)
        print(json.dumps(item), flush=True)
        print(json.dumps(completed), flush=True)
"""


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="trace",
        prompt="How should I change this soup?",
        privileged_context="Always spell pinapple without an e.",
        instructions="Answer in one sentence.",
    )


def _backend(tmp_path: Path, mode: str) -> CodexAppServerBackend:
    script = tmp_path / "fake_app_server.py"
    script.write_text(FAKE_SERVER)
    environment = {**os.environ, "DISTILL_FAKE_MODE": mode}
    return CodexAppServerBackend(
        command=(sys.executable, str(script)),
        model="gpt-5.6-terra",
        reasoning_effort="low",
        prompt_version="test-v1",
        timeout_seconds=1,
        environment=environment,
    )


async def test_real_subprocess_protocol_returns_text_and_usage(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "success")

    result = await backend.generate([_request()], output_token_limit=32)

    assert result[0].text == "Add pinapple."
    assert result[0].output_tokens == 3
    assert backend.running
    await backend.close()
    assert not backend.running


async def test_probe_checks_handshake_and_model_without_starting_a_turn(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "success")

    await backend.probe()

    assert backend.running
    assert not backend.turn_started.is_set()
    await backend.close()


async def test_dead_process_is_detached_before_retry(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "die")

    with pytest.raises(TeacherTransportError, match="exited"):
        await backend.generate([_request()], output_token_limit=32)

    assert not backend.running
    await backend.close()


async def test_malformed_semantic_output_fails_without_a_live_process(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "malformed")

    with pytest.raises(ValueError, match="valid JSON"):
        await backend.generate([_request()], output_token_limit=32)

    assert not backend.running
    await backend.close()


async def test_terminal_json_rpc_error_is_redacted_and_not_retryable(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "terminal_rpc_error")

    with pytest.raises(TeacherTransportError, match="request failed") as captured:
        await backend.generate([_request()], output_token_limit=32)

    assert "sensitive" not in str(captured.value)
    assert not backend.running


async def test_any_tool_item_fails_the_turn_closed(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "tool_item")

    with pytest.raises(TeacherTransportError, match="forbidden item"):
        await backend.generate([_request()], output_token_limit=32)

    assert not backend.running


async def test_plan_item_is_non_executing_model_output(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "plan_item")

    result = await backend.generate([_request()], output_token_limit=32)

    assert result[0].text == "Add pinapple."
    await backend.close()


async def test_thread_explicitly_disables_every_supported_tool_surface(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "require_tool_free")

    result = await backend.generate([_request()], output_token_limit=32)

    assert result[0].text == "Add pinapple."
    await backend.close()


async def test_cancellation_terminates_the_real_subprocess(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "hang")
    task = asyncio.create_task(backend.generate([_request()], output_token_limit=32))
    await backend.turn_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not backend.running
    await backend.close()


async def test_real_subprocess_selects_an_exact_qwen_token(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "success")
    request = TokenSelectionRequest(
        request_id="trace",
        checkpoint_sha256="5" * 64,
        prompt="How should I change this soup?",
        student_prefix="Add",
        prompt_token_ids=[100, 101],
        student_token_ids=[200],
        position=1,
        candidates=[
            CandidateToken(token_id=10, text=" pin", rank=0),
            CandidateToken(token_id=11, text=" salt", rank=1),
        ],
    )

    result = await backend.select_tokens([request], output_token_limit=32)

    assert result[0].selected_token_id == 10
    assert result[0].output_tokens == 3
    await backend.close()
