"""Offline unit tests for AgyCliLLMBackend streaming watchdog and sandbox isolation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    AgyCliLLMBackend,
    GenerationRequest,
    MockLLMBackend,
    StreamWatchdogConfig,
)
from tilefoundry_ascend.agent.orchestrator import (
    CREATIVE_RESPONSE_SCHEMA,
    RoleRoutingConfig,
    SubprocessCliTemplateBackend,
)
from tilefoundry_ascend.agent.orchestrator import (
    StreamWatchdogConfig as OrchestratorStreamWatchdogConfig,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    _AgyLogTailer,
    _capacity_503_policy_from_env,
    _normalize_agy_common_args,
    create_llm_backend_from_env,
    extract_creative_payload,
)

_FAKE_AGY_BODY = """
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _parse_args(argv):
    parsed = {
        "model": "",
        "output_format": "",
        "log_file": "",
        "effort": "",
        "print_timeout": "",
        "json_schema": "",
        "prompt": "",
    }
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--print="):
            parsed["prompt"] = tok.split("=", 1)[1]
        elif tok.startswith("--log-file="):
            parsed["log_file"] = tok.split("=", 1)[1]
        elif tok.startswith("--effort="):
            parsed["effort"] = tok.split("=", 1)[1]
        elif tok.startswith("--print-timeout="):
            parsed["print_timeout"] = tok.split("=", 1)[1]
        elif tok == "--model" and i + 1 < len(argv):
            parsed["model"] = argv[i + 1]
            i += 1
        elif tok == "--output-format" and i + 1 < len(argv):
            parsed["output_format"] = argv[i + 1]
            i += 1
        elif tok == "--json-schema" and i + 1 < len(argv):
            parsed["json_schema"] = argv[i + 1]
            i += 1
        i += 1
    return parsed


def _emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    args = _parse_args(sys.argv[1:])
    state_dir = Path(os.environ["TF_FAKE_AGY_STATE_DIR"])
    scenario = os.environ["TF_FAKE_AGY_SCENARIO"]
    counter_file = state_dir / "attempt.txt"
    attempt = int(counter_file.read_text(encoding="utf-8")) + 1 if counter_file.exists() else 1
    counter_file.write_text(str(attempt), encoding="utf-8")

    stdin_text = sys.stdin.read()
    cwd_entries = os.listdir(os.getcwd())
    record = {
        "attempt": attempt,
        "args": args,
        "stdin_len": len(stdin_text),
        "cwd_entries": cwd_entries,
    }
    (state_dir / f"record_{attempt}.json").write_text(json.dumps(record), encoding="utf-8")

    if scenario == "startup_stall":
        time.sleep(30.0)
        return 0

    _emit({"event": "init", "conversation_id": f"conv_{attempt}"})

    if scenario == "ok_text":
        _emit(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "def ok():\\n    return 1",
                },
            }
        )
        _emit(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "def ok():\\n    return 1",
                },
            }
        )
        return 0

    if scenario == "ok_delta_fallback":
        _emit(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "text_delta": "part_a_",
                },
            }
        )
        _emit(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "part_b",
                },
            }
        )
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": ""}})
        return 0

    if scenario == "ok_structured":
        _emit(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "structured_output": {
                        "physical_hypothesis": "Use L0C direct store",
                        "target_local_metric": "fixpipe_cycles",
                        "dsl_source": "x = 42",
                    },
                    "response": "",
                },
            }
        )
        return 0

    if scenario == "slow_streaming":
        for idx in range(6):
            time.sleep(0.2)
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": 1,
                        "step_type": "agent_response",
                        "state": "ACTIVE",
                        "text_delta": f"chunk_{idx} ",
                    },
                }
            )
        _emit(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "structured_output": {
                        "physical_hypothesis": "Slow but healthy stream",
                        "target_local_metric": "aic_mad",
                        "dsl_source": "y = 7",
                    },
                },
            }
        )
        return 0

    if scenario == "infinite_stream":
        while True:
            time.sleep(0.08)
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": 1,
                        "step_type": "agent_response",
                        "state": "ACTIVE",
                        "text_delta": "tick ",
                    },
                }
            )

    if scenario == "stall_once_with_child_then_ok":
        if attempt == 1:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30.0)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            (state_dir / "child.pid").write_text(str(child.pid), encoding="utf-8")
            time.sleep(30.0)
            return 0
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "recovered_after_stall"},
            }
        )
        return 0

    if scenario == "log_503_hang":
        log_path = Path(args["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("E1117 10:00:00.100 client.go:90] UNAVAILABLE (code 503): No capacity available\\n")
            fh.write("E1117 10:00:00.200 client.go:90] UNAVAILABLE (code 503): No capacity available\\n")
            fh.flush()
        time.sleep(30.0)
        return 0

    if scenario == "log_stream_idle_once_then_ok":
        if attempt == 1:
            log_path = Path(args["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("E1117 10:00:00.100 run.go:412] Run: attempt 1 failed (stream idle timeout)\\n")
                fh.flush()
            time.sleep(30.0)
            return 0
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "recovered_from_stream_idle_log"},
            }
        )
        return 0

    if scenario == "tool_loop":
        for step_idx in (1, 2, 3):
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": step_idx,
                        "step_type": "tool",
                        "tool_name": "read_file",
                        "state": "ACTIVE",
                    },
                }
            )
        time.sleep(30.0)
        return 0

    if scenario == "bounded_tools_ok":
        for step_idx in (1, 2):
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": step_idx,
                        "step_type": "tool",
                        "tool_name": "read_file",
                        "state": "ACTIVE",
                    },
                }
            )
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": step_idx,
                        "step_type": "tool",
                        "tool_name": "read_file",
                        "state": "DONE",
                    },
                }
            )
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "bounded_tools_succeeded"},
            }
        )
        return 0

    if scenario == "sse_thinking_then_ok":
        log_path = Path(args["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("I1117 20:00:00.100 http_helpers.go:709] URL: https://x/v1internal:streamGenerateContent?alt=sse\\n")
            fh.flush()
        time.sleep(0.65)
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "sse_thinking_succeeded"},
            }
        )
        return 0

    raise ValueError(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _write_fake_agy(tmp_path: Path) -> tuple[Path, Path]:
    bin_path = tmp_path / "fake_agy"
    bin_path.write_text(f"#!{sys.executable}\n{_FAKE_AGY_BODY.lstrip()}", encoding="utf-8")
    bin_path.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return bin_path, state_dir


def _is_pid_dead(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            text = stat_path.read_text(encoding="utf-8")
        except OSError:
            return True
        rparen = text.rfind(")")
        if rparen >= 0:
            rest = text[rparen + 1 :].lstrip()
            if rest and rest[0] in ("Z", "X"):
                return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def test_command_normalization_effort_routing_and_stdin_devnull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify exports, flag stripping, effort routing, stdin=DEVNULL, and output modes."""
    assert StreamWatchdogConfig is OrchestratorStreamWatchdogConfig
    cleaned = _normalize_agy_common_args(
        "--print -p --prompt --print=old --prompt=old "
        "--output-format json --output-format=text --json-schema={} "
        "--log-file /tmp/old.log --log-file=/tmp/old2.log "
        "--effort max --effort=low --print-timeout 99s --print-timeout=50s "
        "--add-dir /tmp --add-dir=/var --model old-model "
        "--input-format text --sandbox --mode plan --dangerously-skip-permissions"
    )
    assert cleaned == [
        "--input-format",
        "text",
        "--sandbox",
        "--mode",
        "plan",
        "--dangerously-skip-permissions",
    ]

    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=10.0,
            idle_stall_timeout_sec=10.0,
            active_ceiling_timeout_sec=15.0,
            max_retries=0,
        ),
    )

    assert backend.select_effort(GenerationRequest("exploit_rule", "s", "u")) == "medium"
    assert backend.select_effort(GenerationRequest("syntax_repair", "s", "u")) == "medium"
    assert backend.select_effort(GenerationRequest("explore_creative", "s", "u")) == "high"
    assert (
        backend.select_effort(GenerationRequest("explore_creative", "s", "u", effort="xhigh"))
        == "high"
    )
    assert (
        backend.select_effort(
            GenerationRequest("exploit_rule", "s", "u", effort="low"),
            effort_override="xhigh",
        )
        == "high"
    )

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "ok_text")
    reply_text = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert "return 1" in reply_text
    rec1 = json.loads((state_dir / "record_1.json").read_text(encoding="utf-8"))
    assert rec1["stdin_len"] == 0
    assert rec1["cwd_entries"] == []
    assert rec1["args"]["output_format"] == "stream-json"
    assert rec1["args"]["effort"] == "medium"
    assert rec1["args"]["print_timeout"] == "15s"
    assert "### SYSTEM INSTRUCTIONS" in rec1["args"]["prompt"]

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "ok_structured")
    reply_schema = asyncio.run(
        backend.generate(
            GenerationRequest("explore_creative", "sys", "usr", CREATIVE_RESPONSE_SCHEMA),
            effort_override="xhigh",
        )
    )
    parsed_schema = json.loads(reply_schema)
    assert parsed_schema["dsl_source"] == "x = 42"
    rec2 = json.loads((state_dir / "record_2.json").read_text(encoding="utf-8"))
    assert rec2["args"]["effort"] == "high"
    assert rec2["args"]["model"] == AgyCliLLMBackend.DEFAULT_EXPLORE_MODEL
    assert "xhigh" not in rec2["args"]["model"]

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "ok_delta_fallback")
    reply_delta = asyncio.run(backend.generate(GenerationRequest("syntax_repair", "sys", "usr")))
    assert reply_delta == "part_a_part_b"


def test_agent_flag_stripping_and_tier_adaptive_watchdog(
    tmp_path: Path,
) -> None:
    """Verify --agent stripping, xhigh->high guardrail, and tier-adaptive watchdog timeouts."""
    cleaned = _normalize_agy_common_args(
        "--agent orchestrator --agent=custom --add-dir /tmp --sandbox --mode plan"
    )
    assert cleaned == ["--sandbox", "--mode", "plan"]

    bin_path, _ = _write_fake_agy(tmp_path)
    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        escalate_model="gemini-4-hydrogen-xhigh",
    )

    exploit_req = GenerationRequest("exploit_rule", "sys", "usr")
    explore_req = GenerationRequest("explore_creative", "sys", "usr")
    escalate_req = GenerationRequest("escalate_trackb", "sys", "usr", effort="xhigh")
    assert escalate_req.effort == "xhigh"
    assert backend.select_effort(escalate_req) == "high"

    exploit_model = backend.select_model("exploit_rule")
    explore_model = backend.select_model("explore_creative")
    escalate_model = backend.select_model("escalate_trackb", effort="xhigh")
    assert exploit_model == "gemini-4-hydrogen-medium"
    assert explore_model == "gemini-4-hydrogen-high"
    assert escalate_model == "gemini-4-hydrogen-high"

    med_cfg = backend.watchdog_for_model(exploit_model)
    high_cfg = backend.watchdog_for_model(explore_model)
    xhigh_cfg = backend.watchdog_for_model("gemini-4-hydrogen-xhigh")
    assert (med_cfg.idle_stall_timeout_sec, med_cfg.active_ceiling_timeout_sec) == (120.0, 360.0)
    assert med_cfg.startup_timeout_sec == 60.0
    assert (high_cfg.idle_stall_timeout_sec, high_cfg.active_ceiling_timeout_sec) == (180.0, 480.0)
    assert (xhigh_cfg.idle_stall_timeout_sec, xhigh_cfg.active_ceiling_timeout_sec) == (
        180.0,
        480.0,
    )

    cmd_med = backend.build_command(exploit_req, "P")
    cmd_high = backend.build_command(explore_req, "P")
    cmd_xhigh = backend.build_command(escalate_req, "P")
    assert "gemini-4-hydrogen-medium" in cmd_med
    assert "--print-timeout=360s" in cmd_med
    assert "gemini-4-hydrogen-high" in cmd_high
    assert "--print-timeout=480s" in cmd_high
    assert "gemini-4-hydrogen-high" in cmd_xhigh
    assert "gemini-4-hydrogen-xhigh" not in cmd_xhigh
    assert "--effort=high" in cmd_xhigh
    assert "--print-timeout=480s" in cmd_xhigh

    mock = MockLLMBackend(default_response="ok")
    asyncio.run(mock.generate(exploit_req))
    asyncio.run(mock.generate(explore_req))
    asyncio.run(mock.generate(escalate_req))
    assert mock.routed_models == ["mock-exploit", "mock-explore", "mock-escalate"]
    assert mock.routed_efforts == ["medium", "high", "xhigh"]


def test_slow_actively_streaming_worker_exceeds_idle_without_false_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify slow actively-streaming worker longer than idle timeout finishes cleanly."""
    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "slow_streaming")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.5,
            idle_stall_timeout_sec=0.5,
            active_ceiling_timeout_sec=5.0,
            max_retries=0,
        ),
    )
    reply = asyncio.run(
        backend.generate(
            GenerationRequest("explore_creative", "sys", "usr", CREATIVE_RESPONSE_SCHEMA)
        )
    )
    payload = json.loads(reply)
    assert payload["physical_hypothesis"] == "Slow but healthy stream"
    assert payload["dsl_source"] == "y = 7"


def test_silent_stall_kills_process_group_early_and_succeeds_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify idle stall kills child process group early, retries, and enforces timers."""
    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "stall_once_with_child_then_ok")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.5,
            idle_stall_timeout_sec=0.35,
            active_ceiling_timeout_sec=5.0,
            max_retries=1,
            retry_backoff_sec=0.02,
        ),
    )
    t0 = time.monotonic()
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    elapsed = time.monotonic() - t0
    assert reply == "recovered_after_stall"
    assert elapsed < 3.0
    assert (state_dir / "attempt.txt").read_text(encoding="utf-8") == "2"

    child_pid = int((state_dir / "child.pid").read_text(encoding="utf-8"))
    for _ in range(25):
        if _is_pid_dead(child_pid):
            break
        time.sleep(0.02)
    assert _is_pid_dead(child_pid)

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "startup_stall")
    startup_backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.25,
            idle_stall_timeout_sec=2.0,
            active_ceiling_timeout_sec=5.0,
            max_retries=0,
        ),
    )
    with pytest.raises(TimeoutError, match="startup_timeout"):
        asyncio.run(startup_backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "infinite_stream")
    ceiling_backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.5,
            idle_stall_timeout_sec=0.5,
            active_ceiling_timeout_sec=0.4,
            max_retries=0,
        ),
    )
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(ceiling_backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))


def test_log_tailer_503_and_stream_idle_abort_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify log tailer aborts immediately on 503 capacity or stream idle log lines."""
    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "log_503_hang")

    backend_503 = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=5.0,
            active_ceiling_timeout_sec=10.0,
            max_503_hits=2,
            max_retries=0,
        ),
    )
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="agy_503_no_capacity_x2"):
        asyncio.run(backend_503.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert (time.monotonic() - t0) < 2.0

    (state_dir / "attempt.txt").unlink(missing_ok=True)
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "log_stream_idle_once_then_ok")
    backend_idle_log = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=5.0,
            active_ceiling_timeout_sec=10.0,
            max_retries=1,
            retry_backoff_sec=0.02,
        ),
    )
    t1 = time.monotonic()
    reply = asyncio.run(backend_idle_log.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "recovered_from_stream_idle_log"
    assert (time.monotonic() - t1) < 2.0


def test_runaway_sandbox_tool_calls_aborted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runaway tool calls in empty sandbox abort while bounded tool counts pass."""
    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=5.0,
            active_ceiling_timeout_sec=10.0,
            max_sandbox_tool_calls=2,
            max_retries=0,
        ),
    )
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "tool_loop")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="sandbox_tool_loop_aborted"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert (time.monotonic() - t0) < 2.0

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "bounded_tools_ok")
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "bounded_tools_succeeded"


def test_json_schema_sse_in_flight_defers_idle_to_ceiling_and_fenced_json_extracts(
    tmp_path: Path,
) -> None:
    """Verify fenced creative JSON extraction and SSE in-flight log detection."""
    fenced = (
        "```json\n"
        '{"physical_hypothesis": "h", "target_local_metric": "m", "dsl_source": "x = 1"}\n'
        "```"
    )
    parsed = extract_creative_payload(fenced)
    assert parsed["physical_hypothesis"] == "h"
    assert parsed["target_local_metric"] == "m"
    assert parsed["dsl_source"] == "x = 1"

    log_path = tmp_path / "worker.log"
    log_path.write_text("I1117 20:00:00 URL: https://x/v1:streamGenerateContent?alt=sse\n", encoding="utf-8")
    tailer = _AgyLogTailer(log_path)
    assert tailer.poll() == []
    assert tailer.sse_in_flight is True
    assert tailer.poll() == []
    assert tailer.sse_in_flight is True


def test_sse_thinking_defers_idle_timeout_without_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SSE in-flight log defers idle stall to ceiling when json_schema is None."""
    bin_path, state_dir = _write_fake_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "sse_thinking_then_ok")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=1.0,
            idle_stall_timeout_sec=0.25,
            active_ceiling_timeout_sec=3.0,
            max_retries=0,
        ),
    )
    reply = asyncio.run(
        backend.generate(GenerationRequest("explore_creative", "sys", "usr", json_schema=None))
    )
    assert reply == "sse_thinking_succeeded"


def test_build_command_reconciles_model_effort_tier_suffix() -> None:
    """Ensure explicit -low/-medium/-high model tier suffix overrides conflicting effort flags."""
    backend = AgyCliLLMBackend(exploit_model="gemini-4-hydrogen-medium")
    cmd = backend.build_command(
        GenerationRequest("exploit_rule", "sys", "usr", effort="high"),
        "hello",
    )
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "gemini-4-hydrogen-medium"
    assert "--effort=medium" in cmd
    assert "--effort=high" not in cmd


_FAKE_CLAUDE_BODY = """
import json
import os
from pathlib import Path
import sys
import time


def _emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    state_dir = Path(os.environ["TF_FAKE_CLAUDE_STATE_DIR"])
    scenario = os.environ["TF_FAKE_CLAUDE_SCENARIO"]
    counter = state_dir / "attempt.txt"
    attempt = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
    counter.write_text(str(attempt), encoding="utf-8")
    record = {
        "attempt": attempt,
        "args": sys.argv[1:],
        "stdin": sys.stdin.read(),
        "https_proxy": os.environ.get("HTTPS_PROXY", ""),
        "https_proxy_lower": os.environ.get("https_proxy", ""),
        "http_proxy_lower": os.environ.get("http_proxy", ""),
    }
    (state_dir / f"record_{attempt}.json").write_text(json.dumps(record), encoding="utf-8")

    def _stream_ok():
        _emit({"type": "system", "subtype": "init"})
        _emit(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "def ok():\\n    return 1"}]},
            }
        )
        _emit(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "def ok():\\n    return 1",
            }
        )

    if scenario == "stream_ok":
        _stream_ok()
        return 0
    if scenario == "startup_stall_once_then_ok":
        if attempt == 1:
            time.sleep(30.0)
        _stream_ok()
        return 0
    if scenario == "idle_stall_once_then_ok":
        if attempt == 1:
            _emit({"type": "system", "subtype": "init"})
            time.sleep(30.0)
        _stream_ok()
        return 0
    if scenario == "stderr_exit":
        print(
            "fatal: token=hunter2 api_key=abc123 "
            "Authorization: Bearer bearer-secret-123",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if scenario == "result_error":
        _emit(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "result": "turn limit reached token=hunter2",
            }
        )
        return 0
    if scenario == "plain_text":
        print("plain fallback text", flush=True)
        return 0
    if scenario == "structured_no_text":
        _emit({"type": "system", "subtype": "init"})
        _emit(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "",
            }
        )
        return 0
    if scenario == "corrupt_stream":
        print('{"type": "assis', flush=True)
        return 0
    if scenario == "eof_hang":
        _emit({"type": "system", "subtype": "init"})
        os.close(1)
        os.close(2)
        time.sleep(30.0)
        return 0
    if scenario == "infinite_stream":
        _emit({"type": "system", "subtype": "init"})
        while True:
            _emit({"type": "stream_event", "event": {"type": "content_block_delta"}})
            time.sleep(0.05)
    if scenario == "proxy_echo":
        payload = {
            "https": os.environ.get("HTTPS_PROXY", ""),
            "https_lower": os.environ.get("https_proxy", ""),
            "http_lower": os.environ.get("http_proxy", ""),
        }
        _emit(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": json.dumps(payload),
            }
        )
        return 0
    raise SystemExit(f"unknown scenario {scenario}")


sys.exit(main())
"""


def _write_fake_claude(tmp_path: Path) -> tuple[Path, Path]:
    bin_path = tmp_path / "fake_claude"
    bin_path.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE_BODY.lstrip()}", encoding="utf-8")
    bin_path.chmod(0o755)
    state_dir = tmp_path / "claude_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return bin_path, state_dir


def _claude_backend(
    bin_path: Path,
    *,
    startup: float = 0.3,
    idle: float = 3.0,
    ceiling: float = 5.0,
    max_retries: int = 1,
) -> SubprocessCliTemplateBackend:
    routing = RoleRoutingConfig(
        "exploit-model",
        "explore-model",
        "escalate-model",
        "repair-model",
        exploit_timeout_sec=ceiling,
        reasoning_timeout_sec=ceiling,
    )
    return SubprocessCliTemplateBackend(
        "claude_cli",
        routing,
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=startup,
            idle_stall_timeout_sec=idle,
            active_ceiling_timeout_sec=ceiling,
            max_retries=max_retries,
            retry_backoff_sec=0.02,
        ),
    )


def _telemetry_activity_age(line: str) -> float:
    token = next(
        part for part in line.split() if part.startswith("since_last_activity_sec=")
    )
    return float(token.split("=", 1)[1])


def test_claude_stream_json_startup_retry_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify stream-json parsing, startup-stall retry, and single-line telemetry."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "startup_stall_once_then_ok")
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "TILEFOUNDRY_NPU_LLM_HTTP_PROXY",
        "TILEFOUNDRY_NPU_LLM_HTTPS_PROXY",
        "TILEFOUNDRY_NPU_LLM_ALL_PROXY",
        "TILEFOUNDRY_NPU_LLM_NO_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    backend = _claude_backend(bin_path)
    caplog.set_level(logging.INFO)

    t0 = time.monotonic()
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "def ok():\n    return 1"
    assert time.monotonic() - t0 < 3.0
    assert (state_dir / "attempt.txt").read_text(encoding="utf-8") == "2"
    record = json.loads((state_dir / "record_2.json").read_text(encoding="utf-8"))
    assert "--output-format" in record["args"]
    assert record["args"][record["args"].index("--output-format") + 1] == "stream-json"
    assert "--verbose" in record["args"]
    assert "--include-partial-messages" in record["args"]
    assert "### USER INPUT" in record["stdin"]

    telemetry = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("LLM_CALL_TELEMETRY")
    ]
    assert len(telemetry) == 2
    assert "role=exploit_rule" in telemetry[0]
    assert "model=exploit-model" in telemetry[0]
    assert "status=timeout" in telemetry[0]
    assert "proxy_configured=false" in telemetry[0]
    assert "timeout_stage=startup" in telemetry[0]
    assert 0.0 <= _telemetry_activity_age(telemetry[0]) < 3.0
    assert "timeout_stage=none" in telemetry[1]
    assert "status=ok" in telemetry[1]
    assert "since_last_activity_sec=-1.000" in telemetry[1]
    assert "### SYSTEM INSTRUCTIONS" not in telemetry[0]
    assert "### USER INPUT" not in telemetry[0]


def test_claude_idle_stall_interrupts_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify a silent stream is interrupted at the idle threshold and retried once."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "idle_stall_once_then_ok")
    backend = _claude_backend(bin_path, startup=3.0, idle=0.3)
    caplog.set_level(logging.INFO)

    t0 = time.monotonic()
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "def ok():\n    return 1"
    assert time.monotonic() - t0 < 3.0
    assert (state_dir / "attempt.txt").read_text(encoding="utf-8") == "2"

    telemetry = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("LLM_CALL_TELEMETRY")
    ]
    assert len(telemetry) == 2
    assert "status=timeout" in telemetry[0]
    assert "timeout_stage=idle_stall" in telemetry[0]
    assert 0.3 <= _telemetry_activity_age(telemetry[0]) < 3.0
    assert "timeout_stage=none" in telemetry[1]


def test_claude_stderr_redacted_and_result_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-zero exits and stream-json result errors expose only redacted diagnostics."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "stderr_exit")
    backend = _claude_backend(bin_path, max_retries=0)

    with pytest.raises(RuntimeError, match="exit status 2") as exit_error:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    message = str(exit_error.value)
    assert "token=[redacted]" in message
    assert "api_key=[redacted]" in message
    assert "Authorization: [redacted]" in message
    assert "hunter2" not in message
    assert "abc123" not in message
    assert "bearer-secret-123" not in message

    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "result_error")
    (state_dir / "attempt.txt").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="error_max_turns|turn limit") as result_error:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    result_message = str(result_error.value)
    assert "[redacted]" in result_message
    assert "hunter2" not in result_message


def test_claude_proxy_injection_and_plain_text_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify scoped proxy injection into both cases and plain-text output fallback."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "TILEFOUNDRY_NPU_LLM_HTTP_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_HTTPS_PROXY", "http://127.0.0.1:17890")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "proxy_echo")
    backend = _claude_backend(bin_path, max_retries=0)
    caplog.set_level(logging.INFO)

    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    proxies = json.loads(reply)
    assert proxies["https"] == "http://127.0.0.1:17890"
    assert proxies["https_lower"] == "http://127.0.0.1:17890"
    assert proxies["http_lower"] == "http://127.0.0.1:8080"
    telemetry = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("LLM_CALL_TELEMETRY")
    ]
    assert telemetry and "proxy_configured=true" in telemetry[-1]

    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "plain_text")
    (state_dir / "attempt.txt").unlink(missing_ok=True)
    reply_text = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply_text == "plain fallback text"


def test_claude_structured_metadata_never_returns_as_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify system-only and truncated stream-json output raise instead of echoing JSON."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    backend = _claude_backend(bin_path, max_retries=0)

    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "structured_no_text")
    with pytest.raises(RuntimeError, match="no assistant text") as empty_error:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert '"type"' not in str(empty_error.value)
    assert '"result"' not in str(empty_error.value)

    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "corrupt_stream")
    (state_dir / "attempt.txt").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="no assistant text"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))


def test_claude_stream_eof_with_live_child_is_bounded_by_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify a child that closes both pipes but keeps running is killed at the ceiling."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "eof_hang")
    backend = _claude_backend(bin_path, startup=5.0, idle=5.0, ceiling=0.6, max_retries=0)
    caplog.set_level(logging.INFO)

    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert time.monotonic() - t0 < 3.0
    assert (state_dir / "attempt.txt").read_text(encoding="utf-8") == "1"

    telemetry = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("LLM_CALL_TELEMETRY")
    ]
    assert len(telemetry) == 1
    assert "status=timeout" in telemetry[0]
    assert "timeout_stage=after_eof" in telemetry[0]
    assert 0.0 <= _telemetry_activity_age(telemetry[0]) < 3.0


def test_claude_infinite_stream_hits_active_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify an endlessly active stream is bounded by the ceiling and reports the branch."""
    bin_path, state_dir = _write_fake_claude(tmp_path)
    monkeypatch.setenv("TF_FAKE_CLAUDE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_CLAUDE_SCENARIO", "infinite_stream")
    backend = _claude_backend(bin_path, startup=5.0, idle=5.0, ceiling=0.8, max_retries=0)
    caplog.set_level(logging.INFO)

    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert time.monotonic() - t0 < 3.0

    telemetry = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("LLM_CALL_TELEMETRY")
    ]
    assert len(telemetry) == 1
    assert "status=timeout" in telemetry[0]
    assert "timeout_stage=active_ceiling" in telemetry[0]
    assert 0.0 <= _telemetry_activity_age(telemetry[0]) < 0.5


def test_claude_watchdog_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify startup and idle watchdog thresholds come from explicit environment settings."""
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_STARTUP_TIMEOUT_SEC", "0.5")
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_IDLE_STALL_TIMEOUT_SEC", "2.5")
    backend = SubprocessCliTemplateBackend(
        "claude_cli",
        RoleRoutingConfig(
            "e",
            "c",
            "b",
            "r",
            exploit_timeout_sec=30.0,
            reasoning_timeout_sec=60.0,
        ),
        bin_path=sys.executable,
    )
    cfg = backend.watchdog_for_role("explore_creative")
    assert cfg.startup_timeout_sec == 0.5
    assert cfg.idle_stall_timeout_sec == 2.5
    assert cfg.active_ceiling_timeout_sec == 60.0

    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_STARTUP_TIMEOUT_SEC", "not-a-number")
    with pytest.raises(ValueError, match="watchdog timeouts"):
        backend.watchdog_for_role("exploit_rule")


def test_cli_watchdog_defaults_preserve_startup_idle_and_single_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default CLI watchdog keeps 90s startup, 120s idle, and one retry."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_LLM_STARTUP_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_LLM_IDLE_STALL_TIMEOUT_SEC", raising=False)
    backend = SubprocessCliTemplateBackend(
        "claude_cli", RoleRoutingConfig("e", "c", "b", "r"), bin_path=sys.executable
    )

    exploit = backend.watchdog_for_role("exploit_rule")
    assert exploit.startup_timeout_sec == 90.0
    assert exploit.idle_stall_timeout_sec == 120.0
    assert exploit.active_ceiling_timeout_sec == 360.0
    assert exploit.max_retries == 1
    assert backend.watchdog_for_role("syntax_repair").active_ceiling_timeout_sec == 240.0
    assert backend.watchdog_for_role("explore_creative").active_ceiling_timeout_sec == 420.0


def test_503_overrides_early_fallback_stdout_detection_and_cli_shared_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward overrides, trigger early fallback at min(2, max), and unify CLI 503 retries."""
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES", "1")
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC", "9.0")
    policy = _capacity_503_policy_from_env(
        {
            "TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES": "4",
            "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC": "0.01,0.02",
        }
    )
    assert policy == (4, (0.01, 0.02), 3.0)

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.orchestrator.llm_client.random.uniform", lambda _a, _b: 0.0
    )
    state_file = tmp_path / "agy_models.json"
    fake_agy = tmp_path / "fake_agy_stdout_503"
    fake_agy.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json, sys",
                "from pathlib import Path",
                "args = sys.argv[1:]",
                "model = args[args.index('--model') + 1]",
                f"sp = Path({str(state_file)!r})",
                "seen = json.loads(sp.read_text()) if sp.is_file() else []",
                "seen.append(model)",
                "sp.write_text(json.dumps(seen))",
                "if len(seen) <= 2:",
                "    ev = {'event': 'step_update', 'step_update': {",
                "        'step_type': 'agent_response',",
                "        'text_delta': 'UNAVAILABLE (code 503): No capacity available'}}",
                "    sys.stdout.write(json.dumps(ev) + '\\n')",
                "    sys.exit(1)",
                "res = {'event': 'result', 'result': {",
                "    'status': 'SUCCESS', 'response': 'ok_after_early_fallback'}}",
                "sys.stdout.write(json.dumps(res) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)
    backend = create_llm_backend_from_env(
        overrides={
            "TILEFOUNDRY_NPU_LLM_BACKEND": "agy_cli",
            "TILEFOUNDRY_NPU_AGY_BIN": str(fake_agy),
            "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL": "gemini-4-hydrogen-high",
            "TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL": "gemini-4-hydrogen-medium",
            "TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES": "4",
            "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC": "0.0,0.0",
        }
    )
    assert backend.watchdog_for_role("exploit_rule").capacity_503_max_retries == 4
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "ok_after_early_fallback"
    assert json.loads(state_file.read_text(encoding="utf-8")) == [
        "gemini-4-hydrogen-high",
        "gemini-4-hydrogen-high",
        "gemini-4-hydrogen-medium",
    ]

    cli_state = tmp_path / "cli_models.json"
    fake_cli = tmp_path / "fake_claude_503"
    fake_cli.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json, sys",
                "from pathlib import Path",
                "args = sys.argv[1:]",
                "model = next(a.split('=', 1)[1] for a in args if a.startswith('--model='))",
                f"sp = Path({str(cli_state)!r})",
                "seen = json.loads(sp.read_text()) if sp.is_file() else []",
                "seen.append(model)",
                "sp.write_text(json.dumps(seen))",
                "if len(seen) <= 2:",
                "    sys.stderr.write('UNAVAILABLE (code 503): No capacity available\\n')",
                "    sys.exit(1)",
                "ok_ev = {'type': 'result', 'result': 'cli_ok_fallback'}",
                "sys.stdout.write(json.dumps(ok_ev) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    cli_backend = create_llm_backend_from_env(
        overrides={
            "TILEFOUNDRY_NPU_LLM_BACKEND": "claude_cli",
            "TILEFOUNDRY_NPU_LLM_CLI_BIN": str(fake_cli),
            "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL": "primary-cli",
            "TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL": "fallback-cli",
            "TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES": "4",
            "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC": "0.0,0.0",
        }
    )
    assert cli_backend.watchdog_for_role("exploit_rule").capacity_503_max_retries == 4
    cli_reply = asyncio.run(cli_backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert cli_reply == "cli_ok_fallback"
    assert json.loads(cli_state.read_text(encoding="utf-8")) == [
        "primary-cli",
        "primary-cli",
        "fallback-cli",
    ]


