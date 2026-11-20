"""Offline tests for role routing and stateless LLM transports."""

import asyncio
import io
import json
import os
import sys
from email.message import Message
from pathlib import Path
from threading import Event
from urllib.error import HTTPError

import pytest

from tilefoundry_ascend.agent.orchestrator import (
    AgyCliLLMBackend,
    BeamCandidate,
    GenerationRequest,
    MockLLMBackend,
    OpenAICompatHTTPBackend,
    RoleRoutingConfig,
    StaticDynamicResidual,
    SubprocessCliTemplateBackend,
    build_creative_request,
    build_syntax_repair_request,
    create_llm_backend_from_env,
    llm_client,
    portfolio_runner,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

_PREFIX = "TILEFOUNDRY_NPU_LLM_"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ.copy():
        if key.startswith((_PREFIX, "TILEFOUNDRY_NPU_AGY_")):
            monkeypatch.delenv(key)


def _settings(**values: str) -> dict[str, str]:
    return {_PREFIX + key: value for key, value in values.items()}


def _write_cli(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake cli"
    executable.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    executable.chmod(0o755)
    return executable


class _HTTPReply(io.BytesIO):
    def __init__(
        self, data: bytes, *, media_type: str = "application/json", chunk_size: int = 65536
    ) -> None:
        super().__init__(data)
        self.media_type = media_type
        self.chunk_size = chunk_size

    def getheader(self, name: str, default: str = "") -> str:
        return self.media_type if name.lower() == "content-type" else default

    def read1(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0:
            size = self.chunk_size
        return super().read1(min(size, self.chunk_size))


def test_http_factory_uses_role_models_timeouts_and_xhigh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep HTTP requests and routing independent of Agy configuration."""
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_REPAIR_MODEL", "agy-only-repair")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "agy-only-explore")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL", "agy-only-escalation")
    monkeypatch.setenv(_PREFIX + "BACKEND", "agy_cli")
    monkeypatch.setenv(_PREFIX + "EXPLOIT_MODEL", "ambient-model")
    monkeypatch.setenv(_PREFIX + "PASS_REASONING_EFFORT", "0")
    monkeypatch.setenv("TF_TEST_API_KEY", "private-token")

    class Opener:
        def open(self, request, *, timeout):
            requests.append((request, timeout))
            return _HTTPReply(b'{"choices":[{"message":{"content":" generation "}}]}')

    requests = []
    monkeypatch.setattr(llm_client, "build_opener", lambda *_: Opener())
    settings = _settings(
        BACKEND="openai_compat",
        API_BASE="https://unit.test/gateway/v1/",
        API_KEY_ENV="TF_TEST_API_KEY",
        EXPLOIT_MODEL="fast-high",
        EXPLORE_MODEL="explore-xhigh",
        ESCALATE_MODEL="deep",
        REPAIR_MODEL="repair",
        EXPLOIT_TIMEOUT_SEC="17",
        REASONING_TIMEOUT_SEC="55",
        PASS_REASONING_EFFORT="1",
    )
    backend = create_llm_backend_from_env(settings)
    assert isinstance(backend, OpenAICompatHTTPBackend)
    schema = {"type": "object"}
    roles = {
        "exploit_rule": ("fast-high", 17),
        "explore_creative": ("explore-xhigh", 55),
        "escalate_trackb": ("deep", 55),
        "syntax_repair": ("repair", 17),
    }
    for role in roles:
        request = GenerationRequest(role, "system", "user", schema, "xhigh")
        assert request.effort == "xhigh"
        assert asyncio.run(backend.generate(request)) == "generation"

    for (request, timeout), (model, expected_timeout) in zip(requests, roles.values(), strict=True):
        body = json.loads(request.data)
        assert request.get_method() == "POST"
        assert request.full_url == "https://unit.test/gateway/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer private-token"
        assert body["model"] == model
        assert body["reasoning_effort"] == "xhigh"
        assert body["messages"][1] == {"role": "user", "content": "user"}
        assert json.dumps(schema) in body["messages"][0]["content"]
        assert "tools" not in body
        assert "stream" not in body
        assert request.get_header("Accept") == "application/json"
        assert timeout == expected_timeout

    settings[_PREFIX + "EFFORT_CAP"] = "medium"
    backend = create_llm_backend_from_env(settings)
    assert isinstance(backend, OpenAICompatHTTPBackend)
    request = GenerationRequest("explore_creative", "system", "user", effort="high")
    asyncio.run(backend.generate(request, effort_override="xhigh"))
    assert json.loads(requests[-1][0].data)["reasoning_effort"] == "medium"

    settings[_PREFIX + "PASS_REASONING_EFFORT"] = "0"
    settings.pop(_PREFIX + "API_KEY_ENV")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-token")
    asyncio.run(create_llm_backend_from_env(settings).generate(request))
    assert "reasoning_effort" not in json.loads(requests[-1][0].data)
    assert requests[-1][0].get_header("Authorization") is None


def test_http_error_excludes_body_url_and_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the HTTP status without echoing provider diagnostics or credentials."""
    monkeypatch.setenv("TF_TEST_API_KEY", "private-token")

    class Opener:
        def open(self, request, *, timeout):
            assert timeout > 0
            raise HTTPError(
                request.full_url, 401, "private-token", Message(), io.BytesIO(b"private-token")
            )

    monkeypatch.setattr(llm_client, "build_opener", lambda *_: Opener())
    backend = create_llm_backend_from_env(
        _settings(BACKEND="http", API_KEY_ENV="TF_TEST_API_KEY", EXPLOIT_MODEL="worker")
    )
    with pytest.raises(RuntimeError, match="^LLM HTTP request failed with status 401$"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "system", "user")))


@pytest.mark.parametrize(
    ("media_type", "final_event"),
    [
        ("text/event-stream; charset=utf-8", b"data: [DONE]\r\n\r\n"),
        ("text/event-stream", b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\r\r'),
        ("application/json", b""),
    ],
)
def test_http_stream_assembles_split_events_and_accepts_json(
    monkeypatch: pytest.MonkeyPatch, media_type: str, final_event: bytes
) -> None:
    """Join fragmented SSE text and handle JSON replies to a streaming request."""
    if media_type == "application/json":
        payload = b'{"choices":[{"message":{"content":"one two"}}]}'
    else:
        payload = (
            b'\xef\xbb\xbf: keepalive\r\n\r\n'
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\r\n\r\n'
            b'data: {"choices":[\n'
            b'data: {"index":1,"delta":{"content":"ignored"}},\n'
            b'data: {"index":0,"delta":{"reasoning_content":"hidden","content":"caf\xc3\xa9 "}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"two"}}]}\r\n\r\n'
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n' + final_event
        )
    reply = _HTTPReply(payload, media_type=media_type, chunk_size=1)

    class Opener:
        def open(self, request, *, timeout):
            assert timeout > 0
            assert json.loads(request.data)["stream"] is True
            assert request.get_header("Accept") == "text/event-stream, application/json"
            return reply

    monkeypatch.setattr(llm_client, "build_opener", lambda *_: Opener())
    monkeypatch.setenv(_PREFIX + "STREAM", "0")
    backend = create_llm_backend_from_env(
        _settings(BACKEND="http", EXPLOIT_MODEL="worker", STREAM="1")
    )
    expected = "one two" if media_type == "application/json" else b"caf\xc3\xa9 two".decode()
    request = GenerationRequest("exploit_rule", "system", "user")
    assert asyncio.run(backend.generate(request)) == expected
    assert reply.closed


@pytest.mark.parametrize(
    ("event", "error"),
    [
        (b'{"choices":[{"delta":{"tool_calls":[{"id":"tool"}]}}]}', "attempted a tool call"),
        (b'{"choices":[{"delta":{"content":"partial"}}]}', "ended before completion"),
        (b'{"error":{"message":"private-provider-detail"}}', "stream reported an error"),
    ],
)
def test_http_stream_rejects_tools_errors_and_partial_output(
    monkeypatch: pytest.MonkeyPatch, event: bytes, error: str
) -> None:
    """Reject invalid streams without returning partial text or exposing provider details."""
    class Opener:
        def open(self, request, *, timeout):
            return _HTTPReply(b"data: " + event + b"\n\n", media_type="text/event-stream")

    monkeypatch.setattr(llm_client, "build_opener", lambda *_: Opener())
    backend = create_llm_backend_from_env(_settings(BACKEND="http", EXPLOIT_MODEL="worker"))
    with pytest.raises(RuntimeError, match=f"{error}$"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "system", "user")))


@pytest.mark.parametrize("cancel", [False, True])
def test_http_stream_stops_after_timeout_or_cancellation(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """Bound the async wait and stop reading after an in-progress read unblocks."""
    reading, release_read, read_again = Event(), Event(), Event()

    class BlockingReply(_HTTPReply):
        def read1(self, size: int | None = -1, /) -> bytes:
            if reading.is_set():
                read_again.set()
                return b""
            reading.set()
            assert release_read.wait(3)
            return super().read1(size)

    reply = BlockingReply(b": keepalive\n\n", media_type="text/event-stream")

    class Opener:
        def open(self, request, *, timeout):
            return reply

    monkeypatch.setattr(llm_client, "build_opener", lambda *_: Opener())
    timeout = 2 if cancel else 0.3
    routing = RoleRoutingConfig("small", "deep", "cpp", "repair", exploit_timeout_sec=timeout)
    backend = OpenAICompatHTTPBackend(routing, stream=True)

    async def run() -> None:
        task = asyncio.create_task(backend.generate(GenerationRequest("exploit_rule", "s", "u")))
        try:
            async with asyncio.timeout(3):
                while not reading.is_set():
                    await asyncio.sleep(0.01)
                if cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    with pytest.raises(TimeoutError, match="LLM HTTP request timed out"):
                        await task
        finally:
            release_read.set()

    asyncio.run(run())
    assert reply.closed
    assert not read_again.is_set()


def test_factory_preserves_agy_aliases_and_neutral_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use legacy model settings only for the Agy transport."""
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "legacy-exploit")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "legacy-explore")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL", "legacy-escalate-xhigh")
    backend = create_llm_backend_from_env(
        _settings(BACKEND="agy", CLI_BIN=sys.executable, REPAIR_MODEL="new-repair")
    )
    assert isinstance(backend, AgyCliLLMBackend)
    assert backend.select_model("exploit_rule") == "legacy-exploit"
    assert backend.select_model("explore_creative") == "legacy-explore"
    assert backend.select_model("escalate_trackb") == "legacy-escalate-high"
    assert backend.select_model("syntax_repair") == "new-repair"
    assert backend.watchdog_for_role("explore_creative").active_ceiling_timeout_sec == 360

    mock = create_llm_backend_from_env(_settings(BACKEND="mock", EXPLOIT_MODEL="mock-model"))
    assert isinstance(mock, MockLLMBackend)
    asyncio.run(mock.generate(GenerationRequest("escalate_trackb", "sys", "user", effort="xhigh")))
    assert mock.routed_models == ["mock-model"]
    assert mock.routed_efforts == ["xhigh"]


def test_mock_trackb_uses_old_creative_queue_only_when_absent() -> None:
    """Keep existing creative fixtures usable while honoring an explicit escalation queue."""
    mock = MockLLMBackend(responses={"explore_creative": ["legacy", "ordinary"]})
    trackb = GenerationRequest("escalate_trackb", "system", "user")
    creative = GenerationRequest("explore_creative", "system", "user")
    assert asyncio.run(mock.generate(trackb)) == "legacy"
    mock.responses["escalate_trackb"] = ["explicit"]
    assert asyncio.run(mock.generate(trackb)) == "explicit"
    assert asyncio.run(mock.generate(trackb)) == ""
    assert asyncio.run(mock.generate(creative)) == "ordinary"
    assert mock.calls == [trackb, trackb, trackb, creative]
    assert mock.routed_models == ["mock-escalate"] * 3 + ["mock-explore"]


def test_trackb_portfolio_requests_use_separate_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route C++20 exploration and repairs to their independently configured models."""
    monkeypatch.setattr(portfolio_runner, "build_progressive_llm_constraints", lambda **_: "")
    parent = BeamCandidate(
        candidate_id="seed",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source="value = 1",
        track="unconstrained",
        ledger=build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=None,
            pipe_cycles={},
            waterlines={},
        ),
        residual=StaticDynamicResidual.from_dict({}),
        latency_us=1.0,
        accuracy_passed=True,
    )
    backend = create_llm_backend_from_env(
        _settings(
            BACKEND="mock", EXPLORE_MODEL="creative", ESCALATE_MODEL="cpp", REPAIR_MODEL="fix"
        )
    )
    assert isinstance(backend, MockLLMBackend)
    ordinary = build_creative_request(parent, effort="xhigh")
    trackb = build_creative_request(parent, escalation_tier="ascendc_template", effort="xhigh")
    repair = build_syntax_repair_request("broken", "error", backend_kind="ascendc_template")
    assert [req.role for req in (ordinary, trackb, repair)] == [
        "explore_creative",
        "escalate_trackb",
        "syntax_repair",
    ]
    for request in (ordinary, trackb, repair):
        asyncio.run(backend.generate(request))
    assert backend.routed_models == ["creative", "cpp", "fix"]
    assert backend.routed_efforts == ["xhigh", "xhigh", "medium"]


def test_custom_cli_passes_literal_arguments(tmp_path: Path) -> None:
    """Pass shell metacharacters, braces, and whitespace as literal argument data."""
    executable = _write_cli(
        tmp_path,
        "import json, os, sys\nprint(json.dumps([sys.argv[1:], sys.stdin.read(), os.listdir('.')]))\n",
    )
    backend = create_llm_backend_from_env(
        _settings(
            BACKEND="custom_cli",
            CLI_BIN=str(executable),
            CLI_TEMPLATE="{bin} 'fixed flag' --model={model} --effort={effort} {prompt}",
            EXPLOIT_MODEL="model with {braces}-xhigh",
        )
    )
    prompt = "--test $(touch changed) `literal` '{model}';\nnext line"
    request = GenerationRequest("syntax_repair", "system", prompt, effort="xhigh")
    args, stdin, entries = json.loads(asyncio.run(backend.generate(request)))
    assert args[:3] == ["fixed flag", "--model=model with {braces}-xhigh", "--effort=xhigh"]
    assert len(args) == 4
    assert prompt in args[-1]
    assert stdin == ""
    assert entries == []


@pytest.mark.parametrize("kind", ["claude_cli", "codex_cli"])
def test_builtin_cli_profiles_are_stateless(kind: str, tmp_path: Path) -> None:
    """Use headless restrictions, native effort, and stdin for each built-in profile."""
    executable = _write_cli(
        tmp_path,
        "import json, sys\n"
        "text = json.dumps([sys.argv[1:], sys.stdin.read()])\n"
        "if 'exec' in sys.argv:\n"
        "    print(json.dumps({'type': 'item.completed',\n"
        "                      'item': {'type': 'agent_message', 'text': text}}))\n"
        "else:\n"
        "    print(text)\n",
    )
    routing = RoleRoutingConfig("small", "deep-xhigh", "cpp", "repair")
    backend = SubprocessCliTemplateBackend(kind, routing, bin_path=str(executable))
    req = GenerationRequest("explore_creative", "system", "--literal user prompt", effort="xhigh")
    args, stdin = json.loads(asyncio.run(backend.generate(req)))
    assert "--model=deep-xhigh" in args
    assert "--literal user prompt" not in args
    assert req.user_prompt in stdin
    assert "Do NOT invoke ANY tool calls" in stdin
    if kind == "claude_cli":
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--permission-mode") + 1] == "dontAsk"
        assert "--strict-mcp-config" in args
        assert "--no-session-persistence" in args
        assert "--effort=xhigh" in args
    else:
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("--ask-for-approval") + 1] == "never"
        assert {"--ephemeral", "--ignore-user-config", "shell_tool", "unified_exec"} <= set(args)
        assert 'model_reasoning_effort="xhigh"' in args


def test_codex_rejects_tool_event(tmp_path: Path) -> None:
    """Reject responses that indicate a tool was used."""
    executable = _write_cli(
        tmp_path, 'print(\'{"type":"item.started","item":{"type":"command_execution"}}\')\n'
    )
    routing = RoleRoutingConfig("small", "deep", "cpp", "repair")
    backend = SubprocessCliTemplateBackend("codex_cli", routing, bin_path=str(executable))
    with pytest.raises(RuntimeError, match="attempted to use a tool"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "system", "user")))


@pytest.mark.parametrize("cancel", [False, True])
def test_cli_termination_reaps_worker(
    cancel: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reap subprocesses after either timeout or task cancellation."""
    worker_pid = tmp_path / "worker.pid"
    monkeypatch.setenv("TF_TEST_WORKER_PID", str(worker_pid))
    executable = _write_cli(
        tmp_path,
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pid_file = Path(os.environ['TF_TEST_WORKER_PID'])\n"
        "pending = pid_file.with_suffix('.tmp')\n"
        "pending.write_text(f'{os.getpid()} {child.pid}')\n"
        "pending.replace(pid_file)\n"
        "time.sleep(30)\n",
    )
    routing = RoleRoutingConfig("e", "c", "b", "r", exploit_timeout_sec=5 if cancel else 0.5)
    backend = SubprocessCliTemplateBackend(
        "custom_cli", routing, template="{bin}", bin_path=str(executable)
    )

    async def run() -> None:
        task = asyncio.create_task(backend.generate(GenerationRequest("exploit_rule", "s", "u")))
        try:
            if cancel:
                async with asyncio.timeout(2):
                    while not worker_pid.exists():
                        await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(TimeoutError, match="CLI generation timed out"):
                    await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    parent_pid, child_pid = map(int, worker_pid.read_text(encoding="utf-8").split())
    with pytest.raises(ProcessLookupError):
        os.kill(parent_pid, 0)
    child_stat = Path("/proc") / str(child_pid) / "stat"
    try:
        stat = child_stat.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    assert stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")
