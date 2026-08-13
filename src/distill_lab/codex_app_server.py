from __future__ import annotations

import asyncio
import json
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import TypeAdapter, ValidationError

from distill_lab.teacher import (
    GenerationRequest,
    TeacherGeneration,
    TeacherTransportError,
)

_JSON_OBJECT = TypeAdapter(dict[str, Any])
_JSON_OBJECT_LIST = TypeAdapter(list[dict[str, Any]])


class CodexAppServerBackend:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        model: str,
        reasoning_effort: Literal["low", "medium", "high", "xhigh"],
        prompt_version: str,
        timeout_seconds: float,
        environment: Mapping[str, str],
    ) -> None:
        if not command:
            raise ValueError("Codex app-server command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Codex app-server timeout must be positive")
        self._command = command
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._prompt_version = prompt_version
        self._timeout_seconds = timeout_seconds
        self._environment = dict(environment)
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="distill-lab-codex-")
        self.turn_started = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def probe(self) -> None:
        async with self._lock:
            try:
                await self._start()
            except BaseException:
                await self._stop_process()
                raise

    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]:
        async with self._lock:
            try:
                return await self._generate(requests, output_token_limit=output_token_limit)
            except BaseException:
                await self._stop_process()
                raise

    async def _generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]:
        if not requests:
            raise ValueError("Codex generation batch must not be empty")
        await self._start()
        thread = await self._request(
            "thread/start",
            {
                "model": self._model,
                "cwd": self._temporary_directory.name,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "baseInstructions": (
                    "Return only the requested structured data. Never call tools. "
                    "Do not mention the hidden context."
                ),
            },
        )
        thread_id = _required_string(thread, "thread", "id")
        turn = await self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "effort": self._reasoning_effort,
                "input": [
                    {
                        "type": "text",
                        "text": self._prompt(requests, output_token_limit=output_token_limit),
                    }
                ],
                "outputSchema": self._output_schema(len(requests)),
            },
        )
        turn_id = _required_string(turn, "turn", "id")
        self.turn_started.set()
        final_text: str | None = None
        output_tokens: int | None = None
        completed = False
        while not completed or output_tokens is None:
            message = await self._read()
            params = _object(message.get("params"), "notification params")
            method = message.get("method")
            if method == "item/completed" and params.get("turnId") == turn_id:
                item = _object(params.get("item"), "completed item")
                if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                    final_text = item["text"]
            elif method == "thread/tokenUsage/updated" and params.get("turnId") == turn_id:
                usage = _object(params.get("tokenUsage"), "token usage")
                last = _object(usage.get("last"), "last token usage")
                value = last.get("outputTokens")
                if not isinstance(value, int) or value < 0:
                    raise TeacherTransportError("Codex app-server returned invalid output usage")
                output_tokens = value
            elif method == "turn/completed":
                completed_turn = _object(params.get("turn"), "completed turn")
                if completed_turn.get("id") != turn_id:
                    continue
                if completed_turn.get("status") != "completed":
                    raise TeacherTransportError("Codex app-server turn did not complete")
                if final_text is None:
                    final_item = completed_turn.get("finalOutput")
                    if isinstance(final_item, str):
                        final_text = final_item
                completed = True
        if final_text is None:
            raise ValueError("Codex app-server completed without an agent message")
        try:
            payload = _JSON_OBJECT.validate_python(json.loads(final_text))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError("Codex app-server agent message was not valid JSON") from error
        try:
            results = _JSON_OBJECT_LIST.validate_python(payload.get("results"))
        except ValidationError as error:
            raise ValueError("Codex app-server returned invalid results") from error
        if len(results) != len(requests):
            raise ValueError("Codex app-server returned the wrong number of results")
        generations: list[TeacherGeneration] = []
        for index, value in enumerate(results):
            text = value.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("Codex app-server returned an empty generation")
            generations.append(
                TeacherGeneration(
                    text=text,
                    output_tokens=output_tokens if index == 0 else 0,
                )
            )
        return generations

    async def _start(self) -> None:
        if self.running:
            return
        self.turn_started.clear()
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "distill_lab",
                    "title": "distill-lab",
                    "version": "0.1.0",
                },
                "capabilities": {},
            },
        )
        await self._notify("initialized", {})
        models = await self._request("model/list", {"includeHidden": True})
        try:
            data = _JSON_OBJECT_LIST.validate_python(models.get("data"))
        except ValidationError as error:
            raise TeacherTransportError(
                "Codex app-server returned an invalid model list"
            ) from error
        available = {
            value
            for entry in data
            for value in (entry.get("model"), entry.get("id"))
            if isinstance(value, str)
        }
        if self._model not in available:
            raise TeacherTransportError(f"Codex model {self._model!r} is unavailable")

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        await self._write({"id": request_id, "method": method, "params": params})
        while True:
            message = await self._read()
            if message.get("id") == request_id:
                if "error" in message:
                    error = _object(message["error"], "JSON-RPC error")
                    code = error.get("code")
                    retryable = code == -32001
                    suffix = " (retryable)" if retryable else ""
                    raise TeacherTransportError(
                        f"Codex app-server {method} failed{suffix}: {error.get('message')}"
                    )
                return _object(message.get("result"), f"{method} result")
            if "id" in message and "method" in message:
                await self._write(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": "distill-lab does not expose interactive capabilities",
                        },
                    }
                )

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise TeacherTransportError("Codex app-server is not running")
        process.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            raise TeacherTransportError("Codex app-server pipe closed") from error

    async def _read(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise TeacherTransportError("Codex app-server is not running")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout_seconds)
        except TimeoutError as error:
            raise TeacherTransportError("Codex app-server response timed out") from error
        if not line:
            detail = "\n".join(self._stderr_lines)
            suffix = f": {detail}" if detail else ""
            raise TeacherTransportError(f"Codex app-server exited unexpectedly{suffix}")
        try:
            return _JSON_OBJECT.validate_python(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise TeacherTransportError("Codex app-server returned invalid JSON-RPC") from error

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            self._stderr_lines.append(line.decode(errors="replace").rstrip())

    async def _stop_process(self) -> None:
        process = self._process
        stderr_task = self._stderr_task
        self._process = None
        self._stderr_task = None
        try:
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                else:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        else:
                            await asyncio.wait_for(process.wait(), timeout=5)
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)

    def _prompt(self, requests: Sequence[GenerationRequest], *, output_token_limit: int) -> str:
        payload = [request.model_dump(mode="json", exclude={"request_id"}) for request in requests]
        return (
            "Write one answer for each request. Use privileged_context to improve the answer, but "
            "answer only the public prompt and never reveal that hidden context. "
            "Follow instructions. "
            "Preserve request order. Do not call tools. "
            f"Keep all answers within {output_token_limit} output tokens in total.\n\n"
            f"Prompt version: {self._prompt_version}\n"
            f"Requests: {json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _output_schema(batch_size: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "minItems": batch_size,
                    "maxItems": batch_size,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text"],
                        "properties": {"text": {"type": "string", "minLength": 1}},
                    },
                }
            },
        }

    async def close(self) -> None:
        async with self._lock:
            await self._stop_process()
            self._temporary_directory.cleanup()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TeacherTransportError(f"Codex app-server returned invalid {label}")
    return cast(dict[str, Any], value)


def _required_string(value: dict[str, Any], parent: str, field: str) -> str:
    nested = _object(value.get(parent), parent)
    result = nested.get(field)
    if not isinstance(result, str) or not result:
        raise TeacherTransportError(f"Codex app-server returned invalid {parent}.{field}")
    return result
