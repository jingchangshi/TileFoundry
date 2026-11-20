"""CPU-only contract tests for campaign AgentGates routes and receipt checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.orchestrator import gate_funnel

_FAST_GATE = "tilefoundry-npu-device-fast"


def _completed(cmd: list[str], parsed: dict[str, object], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=json.dumps(parsed),
        stderr="",
    )


def _env_file(tmp_path: Path) -> Path:
    path = tmp_path / "env.sh"
    path.write_text(
        "TILEFOUNDRY_NPU_CANNBENCH_REPO=/configured/cannbench\n"
        "TILEFOUNDRY_NPU_WORK_ROOT=/configured/work\n",
        encoding="utf-8",
    )
    return path


def _patch_common(monkeypatch, tmp_path: Path) -> Path:
    env_file = _env_file(tmp_path)
    cache = tmp_path / "cache"
    tl_cache = tmp_path / "tl-cache"
    monkeypatch.setattr(surrogate_cannbench, "_resolve_env_file_path", lambda _p: env_file)
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_kw: tmp_path)
    monkeypatch.setattr(surrogate_cannbench, "_default_surrogate_cache_dir", lambda: cache)
    monkeypatch.setattr(surrogate_cannbench, "_default_surrogate_tl_cache_dir", lambda: tl_cache)
    monkeypatch.setattr(surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        surrogate_cannbench,
        "prewarm_surrogate_tilelang_cache",
        lambda *_a, **_kw: tl_cache,
    )
    monkeypatch.setattr(surrogate_cannbench, "_prewarm_micro_probe_cache", lambda *_a, **_kw: None)
    return env_file


def _anchor(case_num: int = 2) -> AnchorCaseSpec:
    return AnchorCaseSpec("level4/attention_residual", case_num, "small_decode", 10.0, 3.0)


def test_inner_loop_routes_use_fast_gate_and_l2_msprof_lock(tmp_path: Path, monkeypatch) -> None:
    """All four inner-loop routes use the fast gate, and only L2 profiling locks msprof."""
    env_file = _patch_common(monkeypatch, tmp_path)
    calls: list[dict[str, Any]] = []
    env_paths: list[Path] = []
    verify_calls: list[str] = []

    def _fake_run(cmd, *, cwd, env, timeout_sec):
        _ = (cwd, timeout_sec)
        cmd_list = [str(part) for part in cmd]
        env_path = Path(cmd_list[cmd_list.index("--env-file") + 1])
        env_paths.append(env_path)
        env_text = env_path.read_text(encoding="utf-8")
        sitecustomize = ""
        shim_dir = str(env.get("TF_SURROGATE_SHIM_DIR", ""))
        if shim_dir:
            sitecustomize = (Path(shim_dir) / "sitecustomize.py").read_text(encoding="utf-8")
        calls.append(
            {
                "cmd": cmd_list,
                "env": dict(env),
                "env_text": env_text,
                "sitecustomize": sitecustomize,
            }
        )
        request_key = next(
            (key for key in ("TF_SURROGATE_BATCH_REQUEST", "TF_MICRO_PROBE_REQUEST") if key in env),
            "",
        )
        if request_key:
            request = json.loads(Path(env[request_key]).read_text(encoding="utf-8"))
            response_path = Path(request["response_path"])
            if request.get("collect_msprof_detail"):
                response_path.write_text('{"msprof_telemetry": {}}', encoding="utf-8")
            elif request_key == "TF_MICRO_PROBE_REQUEST":
                response_path.write_text('{"metrics": {"speedup": 1.25}}', encoding="utf-8")
            else:
                response_path.write_text('{"operators": {}}', encoding="utf-8")
        parsed = {
            "status": "PASS",
            "runId": "run-fast",
            "receiptPath": "/receipt-fast.json",
            "gates": [
                {
                    "gateId": _FAST_GATE,
                    "observations": {"deviceWindow": {"heldSeconds": 2.5}},
                }
            ],
        }
        return _completed(cmd_list, parsed), parsed

    def _verify(_bin, run_id, *, source_root, **kwargs):
        _ = (kwargs, source_root)
        assert env_paths and env_paths[-1].is_file()
        verify_calls.append(run_id)
        return True, ""

    monkeypatch.setattr(surrogate_cannbench, "_run_agent_gates_cmd", _fake_run)
    monkeypatch.setattr(surrogate_cannbench, "verify_agent_gates_receipt", _verify)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    anchors = [_anchor()]
    surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
        [("attention_residual", [("cand", candidate)], anchors)],
        source_root=tmp_path,
        env_file=env_file,
        prewarm_cache=False,
    )
    surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
        "attention_residual",
        candidate,
        anchors,
        source_root=tmp_path,
        env_file=env_file,
    )
    surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        env_file=env_file,
    )
    probe = SimpleNamespace(
        probe_id="probe",
        probe_kind="vector",
        factory_name="entry",
        baseline_probe_source="x = 1\n",
        optimized_probe_source="x = 2\n",
        tensor_shape=(16, 128),
        tensor_dtype="float32",
    )
    surrogate_cannbench.run_micro_probe_via_agent_gates(
        probe,
        source_root=tmp_path,
        env_file=env_file,
        prewarm_cache=False,
    )

    assert len(calls) == 4
    for call in calls:
        assert call["cmd"][call["cmd"].index("--gate") + 1] == _FAST_GATE
        assert "TILEFOUNDRY_NPU_CPU_TESTS" not in call["env_text"]
        assert "TILEFOUNDRY_NPU_DEVICE_TESTS" in call["env_text"]
    assert "TILEFOUNDRY_NPU_MSPROF_LOCK=1" not in calls[0]["env_text"]
    assert "TILEFOUNDRY_NPU_MSPROF_LOCK=1" in calls[1]["env_text"]
    assert "TILEFOUNDRY_NPU_ENABLE_TILELANG_CACHE=1" in calls[2]["env_text"]
    assert calls[2]["env"]["TF_SURROGATE_TL_CACHE_DIR"] == str(tmp_path / "tl-cache")
    assert "enable_cache()" in calls[2]["sitecustomize"]
    assert verify_calls == ["run-fast"] * 4


def test_fast_routes_reject_failed_receipts(tmp_path: Path, monkeypatch) -> None:
    """Every fast route suppresses results when its signed receipt fails verification."""
    env_file = _patch_common(monkeypatch, tmp_path)
    parsed = {
        "status": "PASS",
        "runId": "run-fast",
        "receiptPath": "/receipt-fast.json",
        "gates": [{"gateId": _FAST_GATE, "observations": {}}],
    }
    verify_calls: list[str] = []

    def _fake_run(cmd, **_kw):
        return _completed([str(part) for part in cmd], parsed), parsed

    def _verify(_bin, run_id, *, source_root, **kwargs):
        _ = (source_root, kwargs)
        verify_calls.append(run_id)
        return False, "receipt_verification_failed: STALE"

    monkeypatch.setattr(surrogate_cannbench, "_run_agent_gates_cmd", _fake_run)
    monkeypatch.setattr(surrogate_cannbench, "verify_agent_gates_receipt", _verify)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    anchors = [_anchor()]

    with pytest.raises(RuntimeError, match="verification failed"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )
    with pytest.raises(RuntimeError, match="verification failed"):
        surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
            "attention_residual",
            candidate,
            anchors,
            source_root=tmp_path,
            env_file=env_file,
        )
    l3_result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        env_file=env_file,
    )
    assert l3_result.passed is False
    assert "STALE" in l3_result.error_message
    probe = SimpleNamespace(
        probe_id="probe",
        probe_kind="vector",
        factory_name="entry",
        baseline_probe_source="x = 1\n",
        optimized_probe_source="x = 2\n",
        tensor_shape=(16, 128),
        tensor_dtype="float32",
    )
    metrics = surrogate_cannbench.run_micro_probe_via_agent_gates(
        probe,
        source_root=tmp_path,
        env_file=env_file,
        prewarm_cache=False,
    )
    assert metrics["speedup"] == 0.0
    assert verify_calls == ["run-fast"] * 4


def test_l3_sitecustomize_enables_tilelang_cache(tmp_path: Path, monkeypatch) -> None:
    """The generated startup hook enables TileLang caching before the harness imports it."""
    shim = tmp_path / "shim"
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_kw: tmp_path)
    surrogate_cannbench.ensure_npu_compare_shim_dir(shim, cannbench_repo=tmp_path)
    (shim / "sitecustomize.py").write_text(
        surrogate_cannbench._L3_CACHE_SITECUSTOMIZE,
        encoding="utf-8",
    )
    fake = tmp_path / "fake"
    tilelang = fake / "tilelang"
    (tilelang / "cache").mkdir(parents=True)
    (tilelang / "__init__.py").write_text("", encoding="utf-8")
    (tilelang / "env.py").write_text(
        "class _Env:\n"
        "    enabled = False\n"
        "    def enable_cache(self):\n"
        "        self.enabled = True\n"
        "env = _Env()\n",
        encoding="utf-8",
    )
    (tilelang / "cache" / "__init__.py").write_text("", encoding="utf-8")
    (tilelang / "cache" / "kernel_cache.py").write_text(
        "def _namespace():\n"
        "    return None\n"
        "_namespace.cache_clear = lambda: None\n"
        "def _stamp():\n"
        "    return None\n"
        "_stamp.cache_clear = lambda: None\n"
        "class KernelCache:\n"
        "    _get_cache_namespace = staticmethod(_namespace)\n"
        "    _get_tilelang_lib_stamp = staticmethod(_stamp)\n"
        "    created = False\n"
        "    @staticmethod\n"
        "    def _create_dirs():\n"
        "        KernelCache.created = True\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "tl-cache"
    script = (
        "import json, os\n"
        "from tilelang.env import env\n"
        "from tilelang.cache.kernel_cache import KernelCache\n"
        "print(json.dumps({\n"
        "    'enabled': env.enabled,\n"
        "    'created': KernelCache.created,\n"
        "    'cache': os.environ.get('TILELANG_CACHE_DIR'),\n"
        "    'disabled': os.environ.get('TILELANG_DISABLE_CACHE'),\n"
        "    'stamp': os.environ.get('TILELANG_KERNEL_CACHE_USE_LIB_STAMP'),\n"
        "}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script],
        text=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{shim}:{fake}",
            "PYTHONNOUSERSITE": "1",
            "TF_SURROGATE_TL_CACHE_DIR": str(cache_dir),
        },
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {
        "enabled": True,
        "created": True,
        "cache": str(cache_dir),
        "disabled": "0",
        "stamp": "1",
    }
    assert cache_dir.is_dir()


def _l4_parsed() -> dict[str, object]:
    return {
        "status": "PASS",
        "runId": "run-l4",
        "receiptPath": "/receipt-l4.json",
        "gates": [
            {
                "gateId": "tilefoundry-npu-cannbench",
                "observations": {
                    "avgSpeedup": 4.25,
                    "score": 99.5,
                    "results": [
                        {
                            "task": "level4/attention_residual",
                            "reportDir": "/signed-report",
                            "cases": [
                                {
                                    "case_id": "level4/attention_residual_2",
                                    "speedup": 3.5,
                                    "elapsed_us": 12.5,
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }


def test_l4_sync_uses_signed_cases_and_verifies_before_temp_env_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pass-eligible synchronous L4 run is verified inside the temp-env lifetime."""
    env_file = _patch_common(monkeypatch, tmp_path)
    env_paths: list[Path] = []
    verify_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        surrogate_cannbench,
        "_run_agent_gates_cmd",
        lambda cmd, **_kw: (
            env_paths.append(Path(cmd[cmd.index("--env-file") + 1])),
            (_completed([str(p) for p in cmd], _l4_parsed()), _l4_parsed()),
        )[1],
    )

    def _verify(_bin: str, run_id: str, *, source_root, timeout_sec=30.0):
        _ = (source_root, timeout_sec)
        assert env_paths and env_paths[-1].is_file()
        verify_calls.append((run_id, _bin))
        return True, ""

    monkeypatch.setattr(surrogate_cannbench, "verify_agent_gates_receipt", _verify)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert result.passed is True
    assert result.run_id == "run-l4"
    assert result.avg_speedup == 4.25
    assert result.score == 99.5
    assert result.per_case_speedups == {2: 3.5}
    assert result.per_case_elapsed_us == {2: 12.5}
    assert result.report_dir == "/signed-report"
    assert verify_calls == [("run-l4", "agent-gates")]


def test_l4_sync_rejects_failed_verification_and_missing_run_id(tmp_path: Path, monkeypatch) -> None:
    """Sync L4 never promotes metrics when verification fails or no run id exists."""
    env_file = _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        surrogate_cannbench,
        "_run_agent_gates_cmd",
        lambda cmd, **_kw: (_completed([str(p) for p in cmd], _l4_parsed()), _l4_parsed()),
    )
    monkeypatch.setattr(
        surrogate_cannbench,
        "verify_agent_gates_receipt",
        lambda *_a, **_kw: (False, "receipt_verification_failed: STALE"),
    )
    stale = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert stale.passed is False
    assert stale.per_case_speedups == {}
    assert "STALE" in stale.error_message

    missing = dict(_l4_parsed())
    missing.pop("runId")
    verify_called = []
    monkeypatch.setattr(
        surrogate_cannbench,
        "_run_agent_gates_cmd",
        lambda cmd, **_kw: (_completed([str(p) for p in cmd], missing), missing),
    )
    monkeypatch.setattr(
        surrogate_cannbench,
        "verify_agent_gates_receipt",
        lambda *_a, **_kw: verify_called.append(True) or (True, ""),
    )
    no_id = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert no_id.passed is False
    assert no_id.per_case_speedups == {}
    assert verify_called == []


def test_l4_sync_requires_valid_receipt_payload(tmp_path: Path, monkeypatch) -> None:
    """The real shared helper promotes a sync result only when verify reports valid=true."""
    env_file = _patch_common(monkeypatch, tmp_path)
    parsed = _l4_parsed()
    monkeypatch.setattr(
        surrogate_cannbench,
        "_run_agent_gates_cmd",
        lambda cmd, **_kw: (_completed([str(part) for part in cmd], parsed), parsed),
    )
    verify_payload = {"status": "PASS", "runId": "run-l4", "valid": True}

    def _fake_verify(cmd, **kwargs):
        _ = kwargs
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps(verify_payload),
            stderr="",
        )

    monkeypatch.setattr(gate_funnel.subprocess, "run", _fake_verify)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert result.passed is True
    assert result.per_case_speedups == {2: 3.5}

    verify_payload["valid"] = False
    rejected = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert rejected.passed is False
    assert rejected.per_case_speedups == {}
    assert "valid=False" in rejected.error_message


def _v2_config(tmp_path: Path, evidence_root: Path) -> Path:
    config = tmp_path / "agent-gates.toml"
    config.write_text(f'evidence_root = "{evidence_root}"\n', encoding="utf-8")
    return config


def test_l4_v2_uses_submit_wait_requirements_and_configured_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """V2 waits for the worker, requires verification, and reads the configured evidence root."""
    evidence_root = tmp_path / "v2-evidence"
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_v2_config(tmp_path, evidence_root)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    wait_count = 0
    wait_requests: list[tuple[float, str]] = []

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        nonlocal wait_count
        _ = (cwd, env, kwargs)
        cmd_list = [str(part) for part in cmd]
        command = cmd_list[1]
        if command == "submit":
            return _completed(cmd_list, {"execution_id": "exec-1", "state": "SUBMITTED"})
        if command == "wait":
            wait_count += 1
            wait_requests.append((timeout, cmd_list[cmd_list.index("--timeout") + 1]))
            if wait_count == 1:
                return _completed(
                    cmd_list,
                    {"execution_id": "exec-1", "timed_out": True, "status": {"state": "RUNNING"}},
                )
            return _completed(
                cmd_list,
                {
                    "execution_id": "exec-1",
                    "timed_out": False,
                    "status": {"state": "FINISHED", "run_id": "run-v2"},
                },
            )
        if command == "verify-requirements":
            return _completed(
                cmd_list,
                {
                    "satisfied": True,
                    "status": "PASS",
                    "run_id": "run-v2",
                    "current_source_identity": {"root": str(tmp_path.resolve())},
                    "matched_gates": ["tilefoundry-npu-cannbench"],
                },
            )
        if command == "report":
            return _completed(cmd_list, _l4_parsed())
        raise AssertionError(cmd_list)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        workspace="tilefoundry",
        source_root=tmp_path,
    )
    assert result.passed is True
    assert result.run_id == "run-v2"
    assert result.per_case_speedups == {2: 3.5}
    assert result.per_case_elapsed_us == {2: 12.5}
    assert result.receipt_path == "/receipt-l4.json"
    assert wait_requests == [(65.0, "60"), (65.0, "60")]


def test_l4_v2_rejects_unsatisfied_requirements_and_source_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """V2 fails closed on unsatisfied requirements or a workspace source mismatch."""
    evidence_root = tmp_path / "v2-evidence"
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_v2_config(tmp_path, evidence_root)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    commands: list[str] = []

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        commands.append(cmd_list[1])
        if cmd_list[1] == "submit":
            return _completed(cmd_list, {"execution_id": "exec-1"})
        if cmd_list[1] == "wait":
            return _completed(
                cmd_list,
                {
                    "execution_id": "exec-1",
                    "timed_out": False,
                    "status": {"state": "FINISHED", "run_id": "run-v2"},
                },
            )
        if cmd_list[1] == "verify-requirements":
            return _completed(
                cmd_list,
                {
                    "satisfied": True,
                    "status": "PASS",
                    "run_id": "run-v2",
                    "current_source_identity": {"root": str((tmp_path / "other").resolve())},
                },
            )
        raise AssertionError(cmd_list)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        workspace="tilefoundry",
        source_root=tmp_path,
    )
    assert result.passed is False
    assert commands == ["submit", "wait", "verify-requirements"]
    assert "source identity" in result.error_message

    commands.clear()

    def _unsatisfied_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        commands.append(cmd_list[1])
        if cmd_list[1] == "submit":
            return _completed(cmd_list, {"execution_id": "exec-1"})
        if cmd_list[1] == "wait":
            return _completed(
                cmd_list,
                {
                    "execution_id": "exec-1",
                    "timed_out": False,
                    "status": {"state": "FINISHED", "run_id": "run-v2"},
                },
            )
        return _completed(cmd_list, {"satisfied": False, "failures": ["RECEIPT_INVALID: STALE"]})

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _unsatisfied_run)
    rejected = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        workspace="tilefoundry",
        source_root=tmp_path,
    )
    assert rejected.passed is False
    assert commands == ["submit", "wait", "verify-requirements"]


def test_l4_v2_timeout_includes_execution_id(tmp_path: Path, monkeypatch) -> None:
    """The wait budget reports the execution id so an independent worker is recoverable."""
    evidence_root = tmp_path / "v2-evidence"
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_v2_config(tmp_path, evidence_root)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        assert cmd_list[1] == "submit"
        return _completed(cmd_list, {"execution_id": "exec-timeout"})

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        workspace="tilefoundry",
        source_root=tmp_path,
        timeout_sec=0,
    )
    assert result.passed is False
    assert "exec-timeout" in result.error_message
    assert "timed out" in result.error_message


def test_l4_workspace_with_env_override_uses_sync_route(tmp_path: Path, monkeypatch) -> None:
    """Supplying an env override keeps L4 on the synchronous receipt-verifying route."""
    env_file = _patch_common(monkeypatch, tmp_path)
    sync_calls: list[list[str]] = []

    def _fake_run(cmd, **_kw):
        cmd_list = [str(part) for part in cmd]
        sync_calls.append(cmd_list)
        return _completed(cmd_list, _l4_parsed()), _l4_parsed()

    monkeypatch.setattr(surrogate_cannbench, "_run_agent_gates_cmd", _fake_run)
    monkeypatch.setattr(
        surrogate_cannbench,
        "verify_agent_gates_receipt",
        lambda *_a, **_kw: (True, ""),
    )
    monkeypatch.setattr(
        surrogate_cannbench.subprocess,
        "run",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("V2 route must not run")),
    )
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        workspace="tilefoundry",
        source_root=tmp_path,
        env_file=env_file,
    )
    assert result.passed is True
    assert sync_calls
    assert sync_calls[0][sync_calls[0].index("--gate") + 1] == "tilefoundry-npu-cannbench"
