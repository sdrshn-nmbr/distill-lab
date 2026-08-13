from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import TypeAdapter, ValidationError

from distill_lab.teacher import (
    GenerationRequest,
    RetryableTeacherTransportError,
    TeacherGeneration,
    TeacherSelection,
    TeacherTransportError,
    TokenSelectionRequest,
)

_JSON_OBJECT = TypeAdapter(dict[str, Any])
_JSON_OBJECT_LIST = TypeAdapter(list[dict[str, Any]])
_TOOL_FREE_CONFIG = {
    "features": {
        "apps": False,
        "browser_use": False,
        "collab": False,
        "computer_use": False,
        "image_generation": False,
        "multi_agent": False,
        "plugins": False,
        "shell_tool": False,
        "standalone_web_search": False,
        "tool_search": False,
        "unified_exec": False,
        "web_search": False,
        "web_search_request": False,
    },
    "include_apps_instructions": False,
    "include_collaboration_mode_instructions": False,
    "include_environment_context": False,
    "include_permissions_instructions": False,
}


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
        payload, output_tokens = await self._structured_turn(
            prompt=self._prompt(requests, output_token_limit=output_token_limit),
            output_schema=self._output_schema(len(requests)),
        )
        results = _results(payload, expected=len(requests))
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

    async def select_tokens(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherSelection]:
        async with self._lock:
            try:
                return await self._select_tokens(requests, output_token_limit=output_token_limit)
            except BaseException:
                await self._stop_process()
                raise

    async def _select_tokens(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherSelection]:
        if not requests:
            raise ValueError("Codex token-selection batch must not be empty")
        payload, output_tokens = await self._structured_turn(
            prompt=self._selection_prompt(requests, output_token_limit=output_token_limit),
            output_schema=self._selection_output_schema(len(requests)),
        )
        results = _results(payload, expected=len(requests))
        selections: list[TeacherSelection] = []
        for index, value in enumerate(results):
            selected = value.get("selected_token_id")
            if selected is not None and not isinstance(selected, int):
                raise ValueError("Codex app-server returned an invalid selected token ID")
            selections.append(
                TeacherSelection(
                    selected_token_id=selected,
                    output_tokens=output_tokens if index == 0 else 0,
                )
            )
        return selections

    async def _structured_turn(
        self, *, prompt: str, output_schema: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
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
                "config": _TOOL_FREE_CONFIG,
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
                        "text": prompt,
                    }
                ],
                "outputSchema": output_schema,
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
                item_type = item.get("type")
                if item_type == "agentMessage" and isinstance(item.get("text"), str):
                    final_text = item["text"]
                elif item_type not in {"plan", "reasoning", "userMessage"}:
                    raise TeacherTransportError("Codex app-server emitted a forbidden item")
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
        return payload, output_tokens

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
                    if code == -32001:
                        raise RetryableTeacherTransportError("Codex app-server is overloaded")
                    raise TeacherTransportError("Codex app-server request failed")
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
            raise RetryableTeacherTransportError("Codex app-server is not running")
        process.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            raise RetryableTeacherTransportError("Codex app-server pipe closed") from error

    async def _read(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RetryableTeacherTransportError("Codex app-server is not running")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout_seconds)
        except TimeoutError as error:
            raise RetryableTeacherTransportError("Codex app-server response timed out") from error
        if not line:
            raise RetryableTeacherTransportError("Codex app-server exited unexpectedly")
        try:
            return _JSON_OBJECT.validate_python(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise TeacherTransportError("Codex app-server returned invalid JSON-RPC") from error

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while await process.stderr.readline():
            pass

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

    def _selection_prompt(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> str:
        payload = [request.model_dump(mode="json", exclude={"request_id"}) for request in requests]
        return (
            "For each student state, choose exactly one selected_token_id from its candidates, "
            "or null when no candidate is a sound next action. The token IDs and text come from "
            "the student tokenizer. Do not invent a token. Preserve request order. Do not call "
            "tools. "
            f"Keep the structured reply within {output_token_limit} output tokens in total.\n\n"
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

    @staticmethod
    def _selection_output_schema(batch_size: int) -> dict[str, Any]:
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
                        "required": ["selected_token_id"],
                        "properties": {
                            "selected_token_id": {
                                "type": ["integer", "null"],
                                "minimum": 0,
                            }
                        },
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


def _results(payload: dict[str, Any], *, expected: int) -> list[dict[str, Any]]:
    try:
        results = _JSON_OBJECT_LIST.validate_python(payload.get("results"))
    except ValidationError as error:
        raise ValueError("Codex app-server returned invalid results") from error
    if len(results) != expected:
        raise ValueError("Codex app-server returned the wrong number of results")
    return results


def _required_string(value: dict[str, Any], parent: str, field: str) -> str:
    nested = _object(value.get(parent), parent)
    result = nested.get(field)
    if not isinstance(result, str) or not result:
        raise TeacherTransportError(f"Codex app-server returned invalid {parent}.{field}")
    return result
