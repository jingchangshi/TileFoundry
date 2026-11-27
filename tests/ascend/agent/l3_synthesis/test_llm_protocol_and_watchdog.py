"""Tests for stateless LLM zero-tool protocol, prose fast-fail, providers, and watchdog."""

from __future__ import annotations

import asyncio
import dataclasses
import http.server
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    AgyCliLLMBackend,
    BayesianRuleTracker,
    BeamCandidate,
    GateFunnel,
    GenerationRequest,
    MockLLMBackend,
    PortfolioRunner,
    StreamWatchdogConfig,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.gates import (
    StageBResult,
    StageCResult,
)
from tilefoundry_ascend.agent.orchestrator import (
    OpenAICompatHTTPBackend,
    RoleRoutingConfig,
    SubprocessCliTemplateBackend,
    compute_static_dynamic_residual,
    create_llm_backend_from_env,
    extract_creative_payload,
    extract_python_dsl,
    llm_client,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    PURE_SYNTHESIS_ROLES,
    LLMGenerationError,
    LLMTimeoutError,
    ProseRefusalError,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

_PREFIX = "TILEFOUNDRY_NPU_LLM_"

_FAKE_AGY_SCRIPT = """
import json, os, subprocess, sys, time
from pathlib import Path

def _arg(argv, flag):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    for tok in argv:
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return ""

def _emit(ev):
    print(json.dumps(ev), flush=True)

def main():
    argv = sys.argv[1:]
    state_dir = Path(os.environ["TF_AGY_STATE_DIR"])
    scenario = os.environ["TF_AGY_SCENARIO"]
    cf = state_dir / "attempt.txt"
    attempt = int(cf.read_text(encoding="utf-8")) + 1 if cf.exists() else 1
    cf.write_text(str(attempt), encoding="utf-8")
    cwd = Path.cwd()
    agents_dir = cwd / ".agents"
    no_tools = agents_dir / "rules" / "no_tools.md"
    hooks = agents_dir / "hooks.json"
    rec = {
        "attempt": attempt,
        "argv": argv,
        "model": _arg(argv, "--model"),
        "effort": _arg(argv, "--effort"),
        "log_file": _arg(argv, "--log-file"),
        "prompt": _arg(argv, "--print"),
        "stdin_len": len(sys.stdin.read()),
        "cwd": str(cwd),
        "pwd": os.environ.get("PWD", ""),
        "entries": sorted(os.listdir(cwd)),
        "has_no_tools_md": no_tools.is_file(),
        "no_tools_text": no_tools.read_text(encoding="utf-8") if no_tools.is_file() else "",
        "has_hooks_json": hooks.is_file(),
        "hooks_data": json.loads(hooks.read_text(encoding="utf-8")) if hooks.is_file() else {},
        "has_leaked_env": any(
            k.startswith(("AGY_CONVERSATION", "AGY_PROJECT", "CLAUDECODE", "MCP_"))
            for k in os.environ
        ),
    }
    (state_dir / f"record_{attempt}.json").write_text(json.dumps(rec), encoding="utf-8")

    if scenario == "startup_stall":
        time.sleep(10.0)
        return 0

    _emit({"event": "init", "conversation_id": f"c_{attempt}"})

    if scenario == "ok_structured":
        _emit({"event": "result", "result": {
            "status": "SUCCESS",
            "response": (
                "# target_bottleneck: fixpipe_cycles\\n"
                "# structural_hypothesis: Use L0C direct store\\n"
                "```python\\nx = 42\\n```"
            ),
        }})
        return 0
    if scenario == "ok_delta_fallback":
        _emit({"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "agent_response",
            "state": "ACTIVE", "text_delta": "part_a_",
        }})
        _emit({"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "agent_response",
            "state": "DONE", "text_delta": "part_b",
        }})
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": ""}})
        return 0
    if scenario == "slow_streaming":
        for i in range(5):
            time.sleep(0.06)
            _emit({"event": "step_update", "step_update": {
                "step_index": 1, "step_type": "agent_response",
                "state": "ACTIVE", "text_delta": f"chunk_{i} ",
            }})
        _emit({"event": "result", "result": {
            "status": "SUCCESS",
            "response": (
                "# target_bottleneck: m\\n# structural_hypothesis: ok\\n```python\\ny = 7\\n```"
            ),
        }})
        return 0
    if scenario == "infinite_stream":
        while True:
            time.sleep(0.03)
            _emit({"event": "step_update", "step_update": {
                "step_index": 1, "step_type": "agent_response",
                "state": "ACTIVE", "text_delta": "tick ",
            }})
    if scenario == "stall_once_with_child":
        if attempt == 1:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10.0)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            (state_dir / "child.pid").write_text(str(child.pid), encoding="utf-8")
            time.sleep(10.0)
            return 0
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "recovered_stall"}})
        return 0
    if scenario == "log_503_hang":
        lp = Path(rec["log_file"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(
            "E1117 10:00:00.1 c.go:9] UNAVAILABLE (code 503): No capacity available\\n"
            "E1117 10:00:00.2 c.go:9] UNAVAILABLE (code 503): No capacity available\\n",
            encoding="utf-8",
        )
        time.sleep(10.0)
        return 0
    if scenario == "log_503_fallback":
        models_file = state_dir / "models.json"
        seen = json.loads(models_file.read_text(encoding="utf-8")) if models_file.is_file() else []
        seen.append(rec["model"])
        models_file.write_text(json.dumps(seen), encoding="utf-8")
        if len(seen) <= 2:
            _emit({"event": "step_update", "step_update": {
                "step_type": "agent_response",
                "text_delta": "UNAVAILABLE (code 503): No capacity available",
            }})
            return 1
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "fallback_ok"}})
        return 0
    if scenario == "sse_thinking_then_ok":
        lp = Path(rec["log_file"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(
            "I1117 20:00:00 http.go:7] URL: https://x/v1:streamGenerateContent?alt=sse\\n",
            encoding="utf-8",
        )
        time.sleep(0.25)
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "sse_ok"}})
        return 0
    if scenario == "tool_once_then_ok":
        if attempt == 1:
            _emit({"event": "step_update", "step_update": {
                "step_index": 1, "step_type": "tool", "tool_name": "view_file", "state": "ACTIVE",
            }})
            time.sleep(10.0)
            return 0
        _emit({"event": "result", "result": {
            "status": "SUCCESS", "response": "def fast_ok():\\n    return 7",
        }})
        return 0
    if scenario == "tool_always":
        for i in (1, 2, 3):
            _emit({"event": "step_update", "step_update": {
                "step_index": i, "step_type": "tool", "tool_name": "read_file", "state": "ACTIVE",
            }})
        time.sleep(10.0)
        return 0
    if scenario == "bounded_tools_ok":
        for i in (1, 2):
            for st in ("ACTIVE", "DONE"):
                _emit({"event": "step_update", "step_update": {
                    "step_index": i, "step_type": "tool", "tool_name": "read_file", "state": st,
                }})
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "bounded_ok"}})
        return 0
    if scenario == "hook_deny_once_recover":
        _emit({"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "tool", "tool_name": "run_command", "state": "ERROR",
            "error": {"message": "Tool call denied by pre-tool hook: Tools disabled"},
        }})
        _emit({"event": "result", "result": {
            "status": "SUCCESS", "response": "```python\\nx = 9\\n```",
        }})
        return 0
    if scenario == "hook_deny_twice_abort":
        for i in (1, 2):
            _emit({"event": "step_update", "step_update": {
                "step_index": i, "step_type": "tool", "tool_name": "view_file", "state": "ERROR",
                "error": {"message": "Denied by pre-tool hook"},
            }})
        time.sleep(10.0)
        return 0
    if scenario == "hang_on_high":
        if rec["model"] == "gemini-4-hydrogen-high":
            time.sleep(0.5)
            return 1
        time.sleep(0.02)
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "medium_ok"}})
        return 0
    if scenario == "sem_queue_hold":
        for i in range(4):
            time.sleep(0.07)
            _emit({"event": "step_update", "step_update": {
                "step_index": 1, "step_type": "agent_response",
                "state": "ACTIVE", "text_delta": f"t{i} ",
            }})
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": f"ok_{attempt}"}})
        return 0
    if scenario == "creative_blocks_not_exploit":
        if "high" in rec["model"]:
            (state_dir / "creative_started").write_text("1", encoding="utf-8")
            while not (state_dir / "release_creative").exists():
                time.sleep(0.01)
            _emit({"event": "result", "result": {"status": "SUCCESS", "response": "creative_ok"}})
            return 0
        _emit({"event": "result", "result": {"status": "SUCCESS", "response": "exploit_ok"}})
        return 0
    raise ValueError(scenario)

if __name__ == "__main__":
    raise SystemExit(main())
"""

_FAKE_CLAUDE_SCRIPT = """
import json, os, sys, time
from pathlib import Path

def _emit(ev):
    print(json.dumps(ev), flush=True)

def main():
    state_dir = Path(os.environ["TF_CLAUDE_STATE_DIR"])
    scenario = os.environ["TF_CLAUDE_SCENARIO"]
    cf = state_dir / "attempt.txt"
    attempt = int(cf.read_text(encoding="utf-8")) + 1 if cf.exists() else 1
    cf.write_text(str(attempt), encoding="utf-8")
    rec = {
        "attempt": attempt,
        "args": sys.argv[1:],
        "stdin": sys.stdin.read(),
        "https": os.environ.get("HTTPS_PROXY", ""),
        "https_lower": os.environ.get("https_proxy", ""),
    }
    (state_dir / f"record_{attempt}.json").write_text(json.dumps(rec), encoding="utf-8")

    def _ok():
        _emit({"type": "system", "subtype": "init"})
        _emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "x = 1"}]}})
        _emit({"type": "result", "subtype": "success", "is_error": False, "result": "x = 1"})

    if scenario == "startup_stall_once":
        if attempt == 1:
            time.sleep(10.0)
        _ok()
        return 0
    if scenario == "idle_stall_once":
        if attempt == 1:
            _emit({"type": "system", "subtype": "init"})
            time.sleep(10.0)
        _ok()
        return 0
    if scenario == "stderr_secret":
        print("fatal: token=secret1 api_key=key2 Authorization: Bearer tok3", file=sys.stderr)
        return 2
    if scenario == "structured_no_text":
        _emit({"type": "system", "subtype": "init"})
        _emit({"type": "result", "subtype": "success", "is_error": False, "result": ""})
        return 0
    if scenario == "eof_hang":
        _emit({"type": "system", "subtype": "init"})
        os.close(1)
        os.close(2)
        time.sleep(10.0)
        return 0
    if scenario == "proxy_echo":
        _emit({"type": "result", "is_error": False, "result": json.dumps(rec)})
        return 0
    raise SystemExit(scenario)

sys.exit(main())
"""


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith((_PREFIX, "TILEFOUNDRY_NPU_AGY_")):
            monkeypatch.delenv(key, raising=False)


def _write_exec(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{body.lstrip()}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _settings(**values: str) -> dict[str, str]:
    return {_PREFIX + k: v for k, v in values.items()}


def _is_pid_dead(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            text = stat_path.read_text(encoding="utf-8")
            rparen = text.rfind(")")
            if rparen >= 0 and text[rparen + 1 :].lstrip()[:1] in ("Z", "X"):
                return True
        except OSError:
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def test_zero_tool_declaration_and_header_comment_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify no json_schema, header-comment fence parsing, and pure-synthesis sandboxing."""
    assert not hasattr(llm_client, "CREATIVE_RESPONSE_SCHEMA")
    assert "CREATIVE_RESPONSE_SCHEMA" not in llm_client.__all__
    assert "json_schema" not in {f.name for f in dataclasses.fields(GenerationRequest)}
    assert PURE_SYNTHESIS_ROLES == frozenset(
        {"explore_creative", "exploit_rule", "syntax_repair", "escalate_trackb"}
    )

    payload = extract_creative_payload(
        "# target_bottleneck: DMA_WAIT\n"
        "# structural_hypothesis: Double-buffer UB tiles\n"
        "```python\n@T.prim_func\ndef k():\n    pass\n```\n"
    )
    assert payload["target_local_metric"] == "DMA_WAIT"
    assert payload["physical_hypothesis"] == "Double-buffer UB tiles"
    assert "@T.prim_func" in payload["dsl_source"]

    alt = extract_creative_payload(
        "# target_local_metric: aic_fixpipe\n"
        "# physical_hypothesis: Bypass GM\n"
        "```cpp\n__global__ void k() {}\n```\n"
    )
    assert alt["target_local_metric"] == "aic_fixpipe"
    assert alt["physical_hypothesis"] == "Bypass GM"
    assert "__global__ void k() {}" in alt["dsl_source"]

    defaulted = extract_creative_payload("def only_code():\n    return 1\n")
    assert defaulted["physical_hypothesis"] == "unstructured_fallback"
    assert defaulted["target_local_metric"] == "latency_us"

    bin_path = _write_exec(tmp_path, "fake_agy", _FAKE_AGY_SCRIPT)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("TF_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_AGY_SCENARIO", "ok_structured")
    monkeypatch.setenv("AGY_CONVERSATION_ID", "leaked_conv")
    monkeypatch.setenv("MCP_TOKEN", "leaked_mcp")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        common_args=(
            "--input-format text --sandbox --mode plan --mode=always "
            "--dangerously-skip-permissions --agent extra --add-dir /tmp"
        ),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=3.0,
            idle_stall_timeout_sec=3.0,
            active_ceiling_timeout_sec=5.0,
            max_retries=0,
        ),
    )
    for role in PURE_SYNTHESIS_ROLES:
        cmd = backend.build_command(GenerationRequest(role, "sys", "usr"), "P")
        assert "--dangerously-skip-permissions" not in cmd
        assert "--mode" not in cmd
        assert all(not tok.startswith("--mode=") for tok in cmd)
        assert "--json-schema" not in cmd
        assert "--agent" not in cmd
        assert "--add-dir" not in cmd

    audit_cmd = backend.build_command(GenerationRequest("repo_audit", "sys", "usr"), "P")
    assert "--dangerously-skip-permissions" in audit_cmd
    assert "--mode" in audit_cmd

    asyncio.run(backend.generate(GenerationRequest("repo_audit", "s", "u")))
    rec1 = json.loads((state_dir / "record_1.json").read_text(encoding="utf-8"))
    assert rec1["has_no_tools_md"] is False
    assert rec1["has_hooks_json"] is False

    monkeypatch.setenv("TF_AGY_SCENARIO", "hook_deny_once_recover")
    out = asyncio.run(backend.generate(GenerationRequest("explore_creative", "s", "u")))
    assert extract_creative_payload(out)["dsl_source"] == "x = 9"
    rec2 = json.loads((state_dir / "record_2.json").read_text(encoding="utf-8"))
    assert rec2["stdin_len"] == 0
    assert rec2["entries"] == [".agents"]
    assert rec2["has_no_tools_md"] is True
    assert "trigger: always_on" in rec2["no_tools_text"]
    assert rec2["has_hooks_json"] is True
    assert rec2["hooks_data"]["no-tools"]["PreToolUse"][0]["matcher"] == "*"
    assert rec2["has_leaked_env"] is False
    assert rec2["pwd"] == rec2["cwd"]
    assert not Path(rec2["cwd"]).exists()

    monkeypatch.setenv("TF_AGY_SCENARIO", "hook_deny_twice_abort")
    with pytest.raises(LLMGenerationError, match="sandbox_tool_loop_aborted") as exc_deny:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "s", "u")))
    assert exc_deny.value.fault_plane == "LLM_TRANSPORT"
    assert exc_deny.value.fault_code == "SANDBOX_TOOL_LOOP_ABORTED"
    assert exc_deny.value.retriable is False


def test_sandbox_tool_loop_retry_and_bounded_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate on tool event, retry once with hard no-tool directive, or abort."""
    assert StreamWatchdogConfig().max_sandbox_tool_calls == 0
    bin_path = _write_exec(tmp_path, "fake_agy", _FAKE_AGY_SCRIPT)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("TF_AGY_STATE_DIR", str(state_dir))

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=3.0,
            idle_stall_timeout_sec=3.0,
            active_ceiling_timeout_sec=5.0,
            max_retries=2,
            retry_backoff_sec=0.5,
        ),
    )
    monkeypatch.setenv("TF_AGY_SCENARIO", "tool_once_then_ok")
    t0 = time.monotonic()
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert reply == "def fast_ok():\n    return 7"
    assert (time.monotonic() - t0) < 1.5
    rec2 = json.loads((state_dir / "record_2.json").read_text(encoding="utf-8"))
    assert "ZERO TOOL CALLS ALLOWED" in rec2["prompt"]

    (state_dir / "attempt.txt").unlink(missing_ok=True)
    monkeypatch.setenv("TF_AGY_SCENARIO", "tool_always")
    with pytest.raises(LLMGenerationError, match="sandbox_tool_loop_aborted") as exc:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert (state_dir / "attempt.txt").read_text(encoding="utf-8") == "2"
    assert exc.value.fault_code == "SANDBOX_TOOL_LOOP_ABORTED"
    assert exc.value.retriable is False

    bounded = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=3.0,
            idle_stall_timeout_sec=3.0,
            active_ceiling_timeout_sec=5.0,
            max_sandbox_tool_calls=2,
            max_retries=0,
        ),
    )
    monkeypatch.setenv("TF_AGY_SCENARIO", "bounded_tools_ok")
    b_req = GenerationRequest("exploit_rule", "s", "u")
    assert asyncio.run(bounded.generate(b_req)) == "bounded_ok"


@pytest.mark.parametrize(
    "prose_reply",
    [
        "",
        "   \n\t ",
        "This task cannot be completed without more context.",
        "### Gaps Found\n- missing context",
        json.dumps({"response": "### Gaps Found\n- Cannot complete"}),
        json.dumps({"dsl_source": "def foo():\n    pass\n"}),
        json.dumps({"structured_output": {"dsl_source": "```python\nx = 2\n```"}}),
        "```python\nI cannot complete this task.\n```",
    ],
)
def test_extract_python_dsl_rejects_prose_fast_fail(prose_reply: str) -> None:
    """Reject prose refusals and JSON-wrapped payloads with ProseRefusalError."""
    with pytest.raises(ProseRefusalError, match="llm_prose_refusal_no_code") as exc_info:
        extract_python_dsl(prose_reply)
    err = exc_info.value
    assert isinstance(err, ValueError)
    assert isinstance(err, LLMGenerationError)
    assert err.fault_plane == "SYNTAX_AST"
    assert err.fault_code == "LLM_PROSE_NO_CODE"
    assert err.retriable is False


@pytest.mark.parametrize(
    ("valid_reply", "expected_substr"),
    [
        ("def foo():\n    return 1\n", "def foo():"),
        ("def broken(:\n    pass", "def broken(:"),
        ("```python\nx = 1\n```\n```python\ndef longer_fn():\n    return 2\n```", "longer_fn"),
        ("```cpp\n__global__ void k() {}\n```", "__global__ void k()"),
        ("<<<<<<< SEARCH\na = 1\n=======\na = 2\n>>>>>>> REPLACE\n", "<<<<<<< SEARCH"),
        ("#include <tl_templates/ascend/common.h>\n__aicore__ void f() {}", "__aicore__"),
        ("constexpr int K_TILE = 64;\n", "constexpr int K_TILE"),
    ],
)
def test_extract_python_dsl_accepts_code_fences_patches_and_cce(
    valid_reply: str,
    expected_substr: str,
) -> None:
    """Accept valid Python, broken Python def headers, SEARCH/REPLACE, and CCE code."""
    assert expected_substr in extract_python_dsl(valid_reply)


def test_portfolio_prose_fast_fail_skips_syntax_repair() -> None:
    """Record ProseRefusalError as llm_parse failure without invoking syntax_repair."""
    seed_dsl = "def make_kernel():\n    return 1\n"
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 32}, waterlines={},
    )
    parent = BeamCandidate(
        "seed_0", 0, "seed", None, seed_dsl, "unconstrained", ledger,
        compute_static_dynamic_residual(ledger, {"dur_us": 10.0}), 10.0, True,
    )
    prose = "I cannot complete this task.\n### Gaps Found\n- missing"
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [prose, prose],
            "explore_creative": [prose],
            "syntax_repair": [seed_dsl],
        }
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st: StageCResult(
            passed=True, accuracy_passed=True, measured_latency_us=9.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 9.0}),
            telemetry={"dur_us": 9.0},
        ),
    )
    runner = PortfolioRunner(
        backend=backend, funnel=funnel, tracker=BayesianRuleTracker.from_seed_yaml(),
        beam=TwoLaneBeamSearch(), k_rule_workers=1, max_syntax_repairs=2,
    )
    report = asyncio.run(runner.run_round(parent, round_idx=1, factory_name="make_kernel"))
    assert report.evaluated_candidates == []
    assert any(
        st == "llm_parse" and "llm_prose_refusal_no_code" in err
        for _, st, err in report.failed_candidates
    )
    assert [req.role for req in backend.calls if req.role == "syntax_repair"] == []


def test_role_routing_and_mock_backend_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RoleRoutingConfig budgets, effort caps, Agy aliases, and MockLLMBackend."""
    default_cfg = RoleRoutingConfig.from_env(_settings(BACKEND="openai_compat", EXPLOIT_MODEL="w"))
    assert default_cfg.timeout_for_role("exploit_rule") == 360.0
    assert default_cfg.timeout_for_role("syntax_repair") == 240.0
    assert default_cfg.timeout_for_role("explore_creative") == 420.0
    assert default_cfg.timeout_for_role("escalate_trackb") == 420.0

    bounded_cfg = RoleRoutingConfig("e", "c", "b", "r", exploit_timeout_sec=120.0)
    assert bounded_cfg.timeout_for_role("syntax_repair") == 120.0

    for bad_t in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            RoleRoutingConfig("e", "c", "b", "r", repair_timeout_sec=bad_t)

    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "agy-exp")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL", "agy-esc-xhigh")
    agy_cfg = create_llm_backend_from_env(
        _settings(BACKEND="agy", CLI_BIN=sys.executable, REPAIR_MODEL="agy-rep")
    )
    assert isinstance(agy_cfg, AgyCliLLMBackend)
    assert agy_cfg.select_model("exploit_rule") == "agy-exp"
    assert agy_cfg.select_model("escalate_trackb") == "agy-esc-high"
    assert agy_cfg.select_model("syntax_repair") == "agy-rep"

    mock = MockLLMBackend(responses={"explore_creative": ["legacy", "normal"]})
    esc_req = GenerationRequest("escalate_trackb", "s", "u", effort="xhigh")
    assert asyncio.run(mock.generate(esc_req)) == "legacy"
    mock.responses["escalate_trackb"] = ["explicit"]
    assert asyncio.run(mock.generate(esc_req)) == "explicit"
    assert asyncio.run(mock.generate(GenerationRequest("explore_creative", "s", "u"))) == "normal"
    assert mock.routed_models == ["mock-escalate", "mock-escalate", "mock-explore"]
    assert mock.routed_efforts == ["xhigh", "xhigh", "high"]

    cmd_suffix = AgyCliLLMBackend(
        bin_path=str(_write_exec(tmp_path, "dummy_agy", "import sys")),
        exploit_model="gemini-4-hydrogen-medium",
    ).build_command(GenerationRequest("exploit_rule", "s", "u", effort="high"), "P")
    assert "--effort=medium" in cmd_suffix
    assert "--effort=high" not in cmd_suffix


def test_http_backend_json_sse_errors_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise OpenAICompatHTTPBackend over local HTTP server for JSON, SSE, and errors."""
    monkeypatch.setenv("TF_TEST_HTTP_KEY", "secret-bearer-token")
    captured: list[tuple[str, dict[str, str], dict[str, object]]] = []
    mode = {"kind": "json"}
    stream_release = threading.Event()
    stream_started = threading.Event()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            captured.append((self.path, dict(self.headers), body))
            kind = mode["kind"]
            if kind == "json":
                payload = b'{"choices":[{"message":{"content":" generated "}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if kind == "err401":
                self.send_response(401)
                self.send_header("Content-Length", "19")
                self.end_headers()
                self.wfile.write(b"secret-bearer-token")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            if kind == "sse":
                self.wfile.write(
                    b'\xef\xbb\xbf: keepalive\r\n\r\n'
                    b'data: {"choices":[{"index":1,"delta":{"content":"skip"}},'
                    b'{"index":0,"delta":{"content":"caf\xc3\xa9 "}}]}\n\n'
                    b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\r\n\r\n'
                    b"data: [DONE]\r\n\r\n"
                )
            elif kind == "sse_tool":
                self.wfile.write(b'data: {"choices":[{"delta":{"tool_calls":[{"id":"t1"}]}}]}\n\n')
            else:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                stream_started.set()
                stream_release.wait(2.0)
                return
            self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        backend = create_llm_backend_from_env(
            _settings(
                BACKEND="openai_compat", API_BASE=base_url, API_KEY_ENV="TF_TEST_HTTP_KEY",
                EXPLOIT_MODEL="fast-m", EXPLORE_MODEL="deep-m",
                MAX_EFFORT="medium", PASS_REASONING_EFFORT="1",
            )
        )
        assert isinstance(backend, OpenAICompatHTTPBackend)
        res = asyncio.run(
            backend.generate(GenerationRequest("explore_creative", "sys", "usr", effort="xhigh"))
        )
        assert res == "generated"
        path, hdrs, body = captured[-1]
        assert path == "/v1/chat/completions"
        assert hdrs["Authorization"] == "Bearer secret-bearer-token"
        assert body["model"] == "deep-m" and body["reasoning_effort"] == "medium"
        assert "tools" not in body

        mode["kind"] = "err401"
        with pytest.raises(RuntimeError, match="^LLM HTTP request failed with status 401$"):
            asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))

        sse_backend = create_llm_backend_from_env(
            _settings(BACKEND="http", API_BASE=base_url, EXPLOIT_MODEL="fast-m", STREAM="1")
        )
        mode["kind"] = "sse"
        assert asyncio.run(
            sse_backend.generate(GenerationRequest("exploit_rule", "sys", "usr"))
        ) == "café ok"

        mode["kind"] = "sse_tool"
        with pytest.raises(RuntimeError, match="attempted a tool call"):
            asyncio.run(sse_backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))

        mode["kind"] = "hang"
        short_backend = OpenAICompatHTTPBackend(
            RoleRoutingConfig("m", "m", "m", "m", exploit_timeout_sec=0.15),
            api_base=base_url,
            stream=True,
        )
        with pytest.raises(TimeoutError, match="LLM HTTP request timed out"):
            asyncio.run(short_backend.generate(GenerationRequest("exploit_rule", "s", "u")))
        stream_release.set()
        stream_started.clear()
        stream_release.clear()

        async def _cancel_stream() -> None:
            task = asyncio.create_task(
                sse_backend.generate(GenerationRequest("exploit_rule", "s", "u"))
            )
            while not stream_started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_cancel_stream())
    finally:
        stream_release.set()
        server.shutdown()
        server.server_close()


def test_subprocess_cli_profiles_reap_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify custom/claude/codex CLI profiles, secret redaction, and child reaping."""
    echo_cli = _write_exec(
        tmp_path,
        "fake cli",
        "import json, os, sys\n"
        "data = json.dumps([sys.argv[1:], sys.stdin.read(), os.listdir('.')])\n"
        "if 'exec' in sys.argv:\n"
        "    ev = {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': data}}\n"
        "    print(json.dumps(ev))\n"
        "else:\n"
        "    print(data)\n",
    )
    custom = create_llm_backend_from_env(
        _settings(
            BACKEND="custom_cli",
            CLI_BIN=str(echo_cli),
            CLI_TEMPLATE="{bin} 'fixed' --model={model} --effort={effort} {prompt}",
            EXPLOIT_MODEL="m-{braces}-xhigh",
        )
    )
    prompt = "--flag $(true) `x` '{model}'"
    c_req = GenerationRequest("syntax_repair", "sys", prompt, effort="xhigh")
    args, stdin_text, entries = json.loads(asyncio.run(custom.generate(c_req)))
    assert args[:3] == ["fixed", "--model=m-{braces}-xhigh", "--effort=xhigh"]
    assert prompt in args[-1]
    assert stdin_text == ""
    assert entries == []

    routing = RoleRoutingConfig("small", "deep-xhigh", "cpp", "repair", exploit_timeout_sec=0.4)
    for kind in ("claude_cli", "codex_cli"):
        be = SubprocessCliTemplateBackend(kind, routing, bin_path=str(echo_cli))
        b_req = GenerationRequest("explore_creative", "sys", "usr", effort="xhigh")
        p_args, p_stdin, _ = json.loads(asyncio.run(be.generate(b_req)))
        assert "Do NOT invoke ANY tool calls" in p_stdin
        if kind == "claude_cli":
            assert p_args[p_args.index("--tools") + 1] == ""
            assert "--strict-mcp-config" in p_args
            assert p_args[p_args.index("--permission-mode") + 1] == "dontAsk"
        else:
            assert p_args[p_args.index("--sandbox") + 1] == "read-only"
            assert {"--ephemeral", "--ignore-user-config", "shell_tool"} <= set(p_args)

    codex_tool = _write_exec(
        tmp_path,
        "codex_tool",
        'print(\'{"type":"item.started","item":{"type":"command_execution"}}\')\n',
    )
    with pytest.raises(RuntimeError, match="attempted to use a tool"):
        asyncio.run(
            SubprocessCliTemplateBackend("codex_cli", routing, bin_path=str(codex_tool)).generate(
                GenerationRequest("exploit_rule", "s", "u")
            )
        )

    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("TF_WATCHDOG_PID", str(pid_file))
    hang_cli = _write_exec(
        tmp_path,
        "hang_cli",
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        "ch = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        "Path(os.environ['TF_WATCHDOG_PID']).write_text(f'{os.getpid()} {ch.pid}')\n"
        "time.sleep(10)\n",
    )
    hang_be = SubprocessCliTemplateBackend(
        "custom_cli",
        RoleRoutingConfig("e", "c", "b", "r", exploit_timeout_sec=0.25),
        template="{bin}",
        bin_path=str(hang_cli),
        watchdog=StreamWatchdogConfig(max_retries=0),
    )
    with pytest.raises(TimeoutError, match="CLI generation timed out"):
        asyncio.run(hang_be.generate(GenerationRequest("exploit_rule", "s", "u")))
    parent_pid, child_pid = map(int, pid_file.read_text(encoding="utf-8").split())
    assert _is_pid_dead(parent_pid)
    assert _is_pid_dead(child_pid)

    claude_bin = _write_exec(tmp_path, "fake_claude", _FAKE_CLAUDE_SCRIPT)
    c_state = tmp_path / "c_state"
    c_state.mkdir()
    monkeypatch.setenv("TF_CLAUDE_STATE_DIR", str(c_state))
    monkeypatch.setenv("TF_CLAUDE_SCENARIO", "stderr_secret")
    claude_be = SubprocessCliTemplateBackend(
        "claude_cli",
        routing,
        bin_path=str(claude_bin),
        watchdog=StreamWatchdogConfig(max_retries=0),
    )
    with pytest.raises(RuntimeError, match="exit status 2") as sec_err:
        asyncio.run(claude_be.generate(GenerationRequest("exploit_rule", "s", "u")))
    msg = str(sec_err.value)
    assert "token=[redacted]" in msg
    assert "api_key=[redacted]" in msg
    assert "secret1" not in msg and "key2" not in msg and "tok3" not in msg

    monkeypatch.setenv("TF_CLAUDE_SCENARIO", "structured_no_text")
    with pytest.raises(RuntimeError, match="no assistant text"):
        asyncio.run(claude_be.generate(GenerationRequest("exploit_rule", "s", "u")))

    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_HTTPS_PROXY", "http://127.0.0.1:17890")
    monkeypatch.setenv("TF_CLAUDE_SCENARIO", "proxy_echo")
    px_raw = asyncio.run(claude_be.generate(GenerationRequest("exploit_rule", "s", "u")))
    proxied = json.loads(px_raw)
    assert proxied["https"] == "http://127.0.0.1:17890"
    assert proxied["https_lower"] == "http://127.0.0.1:17890"


def _agy_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_path = _write_exec(tmp_path, "fake_agy", _FAKE_AGY_SCRIPT)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("TF_AGY_STATE_DIR", str(state_dir))

    def _be(sc: str, *, mc: int = 2, clear_attempt: bool = False, **kw: object) -> AgyCliLLMBackend:
        if clear_attempt:
            (state_dir / "attempt.txt").unlink(missing_ok=True)
        monkeypatch.setenv("TF_AGY_SCENARIO", sc)
        wkw = {k: kw.pop(k) for k in tuple(kw) if k not in ("exploit_model", "fallback_model")}
        return AgyCliLLMBackend(
            bin_path=str(bin_path), max_concurrency=mc, watchdog=StreamWatchdogConfig(**wkw), **kw,
        )

    return bin_path, state_dir, _be


def test_watchdog_timers_and_503_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify startup/idle/ceiling watchdogs, log tailing, and 503 model fallback."""
    bin_path, state_dir, _be = _agy_setup(tmp_path, monkeypatch)

    default_be = AgyCliLLMBackend(bin_path=str(bin_path))
    assert default_be.watchdog_for_role("exploit_rule").active_ceiling_timeout_sec == 360.0
    assert default_be.watchdog_for_role("explore_creative").active_ceiling_timeout_sec == 480.0
    ps_req = GenerationRequest("explore_creative", "s", "u", paradigm_shift=True)
    assert default_be.watchdog_for_request(ps_req).active_ceiling_timeout_sec == 720.0
    override_req = GenerationRequest("exploit_rule", "s", "u", active_timeout_sec=123.0)
    assert default_be.watchdog_for_request(override_req).active_ceiling_timeout_sec == 123.0

    slow_be = _be(
        "slow_streaming", startup_timeout_sec=0.3, idle_stall_timeout_sec=0.2,
        active_ceiling_timeout_sec=3.0, max_retries=0,
    )
    assert extract_creative_payload(
        asyncio.run(slow_be.generate(GenerationRequest("explore_creative", "s", "u")))
    )["dsl_source"] == "y = 7"

    startup_be = _be(
        "startup_stall", startup_timeout_sec=0.12, idle_stall_timeout_sec=2.0,
        active_ceiling_timeout_sec=2.0, max_retries=0,
    )
    with pytest.raises(LLMTimeoutError, match="startup_timeout") as exc_start:
        asyncio.run(startup_be.generate(GenerationRequest("exploit_rule", "s", "u")))
    assert exc_start.value.fault_plane == "LLM_TRANSPORT"
    assert exc_start.value.fault_code == "LLM_TIMEOUT_OR_503"
    assert exc_start.value.retriable is False

    stall_be = _be(
        "stall_once_with_child", clear_attempt=True, startup_timeout_sec=0.4,
        idle_stall_timeout_sec=0.15, active_ceiling_timeout_sec=3.0,
        max_retries=1, retry_backoff_sec=0.01,
    )
    assert asyncio.run(stall_be.generate(GenerationRequest("exploit_rule", "s", "u"))) == (
        "recovered_stall"
    )
    child_pid = int((state_dir / "child.pid").read_text(encoding="utf-8"))
    for _ in range(20):
        if _is_pid_dead(child_pid):
            break
        time.sleep(0.01)
    assert _is_pid_dead(child_pid)

    ceiling_be = _be(
        "infinite_stream", startup_timeout_sec=0.5, idle_stall_timeout_sec=0.5,
        active_ceiling_timeout_sec=0.2, max_retries=0,
    )
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(ceiling_be.generate(GenerationRequest("exploit_rule", "s", "u")))

    sse_be = _be(
        "sse_thinking_then_ok", startup_timeout_sec=1.0, idle_stall_timeout_sec=0.1,
        active_ceiling_timeout_sec=2.0, max_retries=0,
    )
    assert asyncio.run(sse_be.generate(GenerationRequest("explore_creative", "s", "u"))) == "sse_ok"

    be_503 = _be(
        "log_503_hang", startup_timeout_sec=3.0, idle_stall_timeout_sec=3.0,
        active_ceiling_timeout_sec=5.0, max_503_hits=2, max_retries=0,
    )
    with pytest.raises(LLMGenerationError, match="agy_503_no_capacity_x2") as exc_503:
        asyncio.run(be_503.generate(GenerationRequest("exploit_rule", "s", "u")))
    assert exc_503.value.fault_code == "LLM_TIMEOUT_OR_503" and exc_503.value.retriable is False

    fb_be = _be(
        "log_503_fallback", exploit_model="gemini-4-hydrogen-high",
        fallback_model="gemini-4-hydrogen-medium", startup_timeout_sec=3.0,
        idle_stall_timeout_sec=3.0, active_ceiling_timeout_sec=5.0, max_retries=1,
        capacity_503_max_retries=4, capacity_503_backoff_sec=(0.0, 0.0),
        capacity_503_jitter_sec=0.0,
    )
    assert asyncio.run(fb_be.generate(GenerationRequest("exploit_rule", "s", "u"))) == "fallback_ok"
    assert json.loads((state_dir / "models.json").read_text(encoding="utf-8")) == [
        "gemini-4-hydrogen-high", "gemini-4-hydrogen-high", "gemini-4-hydrogen-medium",
    ]


def test_agy_eof_cooldown_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify after_eof timeouts, semaphore queue heartbeats, cooldown, and pool isolation."""
    _, state_dir, _be = _agy_setup(tmp_path, monkeypatch)
    claude_bin = _write_exec(tmp_path, "fake_claude", _FAKE_CLAUDE_SCRIPT)
    c_state = tmp_path / "c_state"
    c_state.mkdir()
    monkeypatch.setenv("TF_CLAUDE_STATE_DIR", str(c_state))
    monkeypatch.setenv("TF_CLAUDE_SCENARIO", "eof_hang")
    caplog.set_level(logging.INFO)
    claude_eof = SubprocessCliTemplateBackend(
        "claude_cli", RoleRoutingConfig("e", "c", "b", "r", exploit_timeout_sec=0.25),
        bin_path=str(claude_bin),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=2.0, idle_stall_timeout_sec=2.0,
            active_ceiling_timeout_sec=0.25, max_retries=0,
        ),
    )
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(claude_eof.generate(GenerationRequest("exploit_rule", "s", "u")))
    telemetry = [r.getMessage() for r in caplog.records if "LLM_CALL_TELEMETRY" in r.getMessage()]
    assert any("timeout_stage=after_eof" in msg for msg in telemetry)

    sem_be = _be(
        "sem_queue_hold", mc=1, clear_attempt=True, startup_timeout_sec=0.2,
        idle_stall_timeout_sec=0.2, active_ceiling_timeout_sec=1.5, max_retries=0,
    )

    async def _run_queued() -> list[str]:
        return await asyncio.gather(
            sem_be.generate(GenerationRequest("exploit_rule", "s", "u")),
            sem_be.generate(GenerationRequest("exploit_rule", "s", "u")),
        )

    assert sorted(asyncio.run(_run_queued())) == ["ok_1", "ok_2"]

    cool_be = _be(
        "hang_on_high", startup_timeout_sec=0.5, idle_stall_timeout_sec=0.5,
        active_ceiling_timeout_sec=0.15, max_retries=0,
    )
    req_a = GenerationRequest("explore_creative", "s", "u", session_scope="reg_a")
    req_b = GenerationRequest("explore_creative", "s", "u", session_scope="reg_b")
    for r in (req_a, req_b, req_a):
        with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
            asyncio.run(cool_be.generate(r))
    assert asyncio.run(cool_be.generate(req_a)) == "medium_ok"

    iso_be = _be(
        "creative_blocks_not_exploit", mc=1, startup_timeout_sec=2.0,
        idle_stall_timeout_sec=2.0, active_ceiling_timeout_sec=3.0, max_retries=0,
    )

    async def _verify_creative_isolation() -> None:
        creative_task = asyncio.create_task(
            iso_be.generate(GenerationRequest("explore_creative", "s", "u"))
        )
        while not (state_dir / "creative_started").is_file():
            await asyncio.sleep(0.01)
        exploit_out = await asyncio.wait_for(
            iso_be.generate(GenerationRequest("exploit_rule", "s", "u")),
            timeout=1.5,
        )
        assert exploit_out == "exploit_ok"
        assert not creative_task.done()
        (state_dir / "release_creative").write_text("1", encoding="utf-8")
        assert await creative_task == "creative_ok"

    asyncio.run(_verify_creative_isolation())
