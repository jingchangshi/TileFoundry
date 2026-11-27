"""Tests for LocalHardwareProbePort, msprof collection, micro-probes, and L4 facade parity."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import npu as npu_pkg
from tilefoundry_ascend.agent.npu import ag_transport, msprof_probe
from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec


class _DummyProbeSpec:
    probe_id = "probe_1"
    probe_kind = "vector_unroll"
    factory_name = "build_probe"
    baseline_probe_source = "def build_probe():\n    return 1\n"
    optimized_probe_source = "def build_probe():\n    return 2\n"
    line_count_baseline = 2
    line_count_optimized = 2
    tensor_shape = (64, 64)
    tensor_dtype = "float16"


def _setup_fast_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    src_root = tmp_path / "repo"
    src_root.mkdir(parents=True, exist_ok=True)
    work_root = tmp_path / "work"
    cann_dir = tmp_path / "cann"
    (cann_dir / "tasks").mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_dir}\n"
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    ev_root = tmp_path / "v2-evidence"
    cfg = tmp_path / "agent-gates.toml"
    cfg.write_text(
        f'evidence_root = "{ev_root}"\n\n'
        "[workspaces.tilefoundry]\n"
        f'source_root = "{src_root.resolve()}"\n'
        'adapters = ["tilefoundry-npu"]\n',
        encoding="utf-8",
    )
    ag_bin = tmp_path / "agent-gates"
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg))
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", str(ag_bin))
    return src_root, env_file


def test_local_hardware_probe_port_cann_cpath_and_msprof_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover CANN include paths into CPATH and delegate run_msprof_detail to TileLang."""
    ascend_home = tmp_path / "ascend_home"
    inc_a = ascend_home / "x86_64-linux" / "include"
    inc_b = ascend_home / "runtime" / "include"
    inc_a.mkdir(parents=True)
    inc_b.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(ascend_home))
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.setenv("CPATH", "/existing/inc")

    port = msprof_probe.LocalHardwareProbePort()
    found = port.ensure_cann_include_cpath()
    assert found == [str(inc_a), str(inc_b)]
    cpath_parts = (monkeypatch.delenv if False else __import__("os").environ["CPATH"]).split(":")
    assert cpath_parts[:3] == [str(inc_a), str(inc_b), "/existing/inc"]

    calls: dict[str, object] = {"sync": 0, "fn": 0, "cache": None}

    class _FakeTorch:
        int32 = "int32"

        class npu:
            @staticmethod
            def synchronize() -> None:
                calls["sync"] = int(calls["sync"]) + 1

        @staticmethod
        def empty(size: int, dtype: object, device: str) -> tuple[int, object, str]:
            calls["cache"] = (size, dtype, device)
            return (size, dtype, device)

    def _fake_bench(fn: object, cache: object, *, n_repeat: int, detailed: bool) -> dict[str, float]:
        assert callable(fn)
        assert cache == (2 * 1024 * 1024 // 4, "int32", "npu")
        assert n_repeat == 4
        assert detailed is True
        return {"dur_us": 9.5, "aiv_vec_ratio": 0.65}

    tl_mod = types.ModuleType("tilelang")
    tl_prof = types.ModuleType("tilelang.profiler")
    tl_msprof = types.ModuleType("tilelang.profiler.msprof")
    tl_msprof.bench_with_msprof = _fake_bench
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(sys.modules, "tilelang", tl_mod)
    monkeypatch.setitem(sys.modules, "tilelang.profiler", tl_prof)
    monkeypatch.setitem(sys.modules, "tilelang.profiler.msprof", tl_msprof)

    res = port.run_msprof_detail(
        lambda: calls.__setitem__("fn", int(calls["fn"]) + 1),
        n_warmup=2,
        n_repeat=4,
        cache_size_mb=2,
    )
    assert res == {"dur_us": 9.5, "aiv_vec_ratio": 0.65}
    assert calls["fn"] == 3
    assert calls["sync"] == 2


def test_bottleneck_anchor_for_regime_selection() -> None:
    """Select the lowest Eq.3 score anchor matching regime_tag, breaking ties by latency."""
    a1 = npu_pkg.AnchorEvalMetrics(2, "reg_a", True, 10.0, 20.0, 2.0, 2.0, 0.2, 0.4)
    a2 = npu_pkg.AnchorEvalMetrics(3, "reg_a", True, 12.0, 20.0, 2.0, 1.6, 0.16, 0.2)
    a3 = npu_pkg.AnchorEvalMetrics(4, "reg_b", True, 5.0, 20.0, 2.0, 4.0, 0.4, 0.1)
    res = npu_pkg.CandidateSurrogateResult("c1", "op", True, 2.0, 1.6, 0.25, 60.0, (a1, a2, a3))
    assert msprof_probe.bottleneck_anchor_for_regime(res, "reg_a") == a2
    assert msprof_probe.bottleneck_anchor_for_regime(res, "reg_b") == a3
    assert msprof_probe.bottleneck_anchor_for_regime(res, "missing") is None


def test_collect_bottleneck_msprof_and_p2_deep_dive_via_agent_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filter bottleneck telemetry to finite floats while preserving P2 deep-dive fields."""
    src_root, env_file = _setup_fast_env(tmp_path, monkeypatch)
    cand = src_root / "candidate.py"
    cand.write_text("def run_op(x):\n    return x\n", encoding="utf-8")

    monkeypatch.setattr(msprof_probe, "prewarm_anchor_reference_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(msprof_probe, "prewarm_surrogate_tilelang_cache", lambda *_a, **_k: tmp_path)

    requests: list[dict[str, object]] = []
    tamper = {"coverage": False}

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] in ("git", "npu-smi"):
            return subprocess.CompletedProcess(cmd, 1, "", "")
        sub = cmd[1]
        if sub == "doctor":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "PASS"}), "")
        if sub == "submit":
            req_path = src_root / cmd[cmd.index("--candidate") + 1]
            req_bytes = req_path.read_bytes()
            req = json.loads(req_bytes)
            requests.append(req)
            t0 = req["msprof_targets"][0]
            cid = t0["candidate_id"]
            raw_tel = {
                "dur_us": 18.5,
                "aiv_vec_ratio": 0.42,
                "bad_bool": True,
                "bad_nan": float("nan"),
                "dominant_subkernel_name": "aiv_kernel",
                "p2_source": "msprof_p2_trace",
                "p2_mte_waitflag_share": 0.35,
                "p2_issue_interval_us": {"MTE2": {"p50_us": 1.2, "p90_us": 2.4}},
                "p2_hotspots": [{"source_location": "kernel.py:19", "instruction_count": 12}],
                "p2_summary_lines": ["kernel.py:19 (12 instrs, 35.0%)"],
            }
            resp_text = json.dumps(
                {
                    "schema": ag_transport.FAST_RESPONSE_SCHEMA,
                    "request_id": req["request_id"],
                    "request_sha256": hashlib.sha256(req_bytes).hexdigest(),
                    "mode": "profile",
                    "status": "ok",
                    "candidates": {
                        "softmax": {
                            cid: {
                                "module_path": t0["module_path"],
                                "sha256": t0["sha256"],
                                "post_sha256": t0["sha256"],
                            }
                        }
                    },
                    "coverage": {
                        "expected_candidates": 1,
                        "evaluated_candidates": 1,
                        "expected_msprof_targets": 1,
                        "observed_msprof_targets": 0 if tamper["coverage"] else 1,
                    },
                    "msprof_telemetry": {"softmax": {cid: {"3": raw_tel}}},
                },
                allow_nan=True,
            )
            (src_root / req["response_path"]).write_text(resp_text, encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"execution_id": "exec-p"}), "")
        if sub == "wait":
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps({"status": {"state": "FINISHED", "run_id": "run-p"}}), ""
            )
        if sub == "report":
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps({"status": "PASS", "receiptPath": "/r-p.json", "gates": []}), ""
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps({"satisfied": True, "current_source_identity": {"root": str(src_root.resolve())}}),
            "",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    anchors = [AnchorCaseSpec("level2/softmax", 3, "vector_mid", 20.0, 2.0)]

    numeric_only = msprof_probe.collect_bottleneck_msprof_via_agent_gates(
        "softmax", cand, anchors, source_root=src_root, env_file=env_file, device_pool=("0", "1")
    )
    assert requests[0]["mode"] == "profile"
    assert requests[0]["msprof_targets"][0]["candidate_id"] == "bottleneck_probe"
    assert numeric_only == {
        "vector_mid": {"dur_us": 18.5, "aiv_vec_ratio": 0.42, "p2_mte_waitflag_share": 0.35}
    }

    deep_dive = msprof_probe.collect_p2_deep_dive_via_agent_gates(
        "softmax", cand, anchors, source_root=src_root, env_file=env_file, device_pool=("0", "1")
    )
    assert requests[1]["msprof_targets"][0]["candidate_id"] == "bottleneck_p2_deep_dive"
    reg_payload = deep_dive["vector_mid"]
    assert reg_payload["p2_source"] == "msprof_p2_trace"
    assert reg_payload["p2_issue_interval_us"]["MTE2"]["p50_us"] == pytest.approx(1.2)
    assert reg_payload["p2_hotspots"][0]["source_location"] == "kernel.py:19"

    tamper["coverage"] = True
    with pytest.raises(RuntimeError, match="msprof coverage"):
        msprof_probe.collect_bottleneck_msprof_via_agent_gates(
            "softmax", cand, anchors, source_root=src_root, env_file=env_file
        )


def test_run_micro_probe_via_agent_gates_stages_prewarms_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage micro-probe sources read-only, trigger optional prewarm, and validate metrics."""
    src_root, env_file = _setup_fast_env(tmp_path, monkeypatch)
    requests: list[dict[str, object]] = []
    prewarm_cmds: list[list[str]] = []
    bad_probe_id = {"val": False}

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] in ("git", "npu-smi"):
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if Path(cmd[-1]).name == "prewarm_probe.py":
            prewarm_cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        sub = cmd[1]
        if sub == "doctor":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "PASS"}), "")
        if sub == "submit":
            req_path = src_root / cmd[cmd.index("--candidate") + 1]
            req_bytes = req_path.read_bytes()
            req = json.loads(req_bytes)
            requests.append(req)
            bindings = {
                "micro_probe": {
                    c["candidate_id"]: {
                        "module_path": c["module_path"],
                        "sha256": c["sha256"],
                        "post_sha256": c["sha256"],
                    }
                    for c in req["candidates"]
                }
            }
            resp = {
                "schema": ag_transport.FAST_RESPONSE_SCHEMA,
                "request_id": req["request_id"],
                "request_sha256": hashlib.sha256(req_bytes).hexdigest(),
                "mode": "micro_probe",
                "status": "ok",
                "candidates": bindings,
                "coverage": {
                    "expected_candidates": 2,
                    "evaluated_candidates": 2,
                    "expected_msprof_targets": 0,
                    "observed_msprof_targets": 0,
                },
                "micro_probe": {
                    "probe_id": "wrong_id" if bad_probe_id["val"] else "probe_1",
                    "metrics": {"baseline_dur_us": 10.0, "optimized_dur_us": 8.0, "speedup": 1.25, "flag": True},
                },
            }
            (src_root / req["response_path"]).write_text(json.dumps(resp), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"execution_id": "exec-mp"}), "")
        if sub == "wait":
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps({"status": {"state": "FINISHED", "run_id": "run-mp"}}), ""
            )
        if sub == "report":
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps({"status": "PASS", "receiptPath": "/r-mp.json", "gates": []}), ""
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps({"satisfied": True, "current_source_identity": {"root": str(src_root.resolve())}}),
            "",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    metrics = msprof_probe.run_micro_probe_via_agent_gates(
        _DummyProbeSpec(),
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=("0", "1"),
        prewarm_cache=True,
    )
    assert len(prewarm_cmds) == 1
    assert metrics == {"baseline_dur_us": 10.0, "optimized_dur_us": 8.0, "speedup": 1.25}
    req = requests[0]
    assert req["mode"] == "micro_probe"
    assert req["probe"]["probe_id"] == "probe_1"
    for row in req["candidates"]:
        staged = src_root / row["module_path"]
        assert staged.is_file()
        assert staged.stat().st_mode & 0o777 == 0o444

    bad_probe_id["val"] = True
    with pytest.raises(RuntimeError, match="probe_id"):
        msprof_probe.run_micro_probe_via_agent_gates(
            _DummyProbeSpec(), source_root=src_root, env_file=env_file, prewarm_cache=False
        )


def test_npu_package_zero_l3_imports_and_facade_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep npu package free of L3 imports, facade module deleted, and direct transport active."""
    repo_root = Path(__file__).resolve().parents[4]
    facade_path = repo_root / "targets/ascend/tilefoundry_ascend/agent/campaign/surrogate_cannbench.py"
    npu_dir = repo_root / "targets/ascend/tilefoundry_ascend/agent/npu"
    assert not facade_path.exists()
    forbidden_l3_tokens = (
        "llm_client",
        "portfolio_runner",
        "ablation_distiller",
        "flywheel",
        "residual_beam",
        "llm_constraints",
        "l3_contract",
        "test_npu_acceptance",
    )

    for py_path in sorted(npu_dir.glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for mod_name in imported:
            assert not any(tok in mod_name for tok in forbidden_l3_tokens), (py_path.name, mod_name)

    assert not hasattr(ag_transport, "_bind_facade_globals")
    assert not hasattr(ag_transport, "_FACADE_NS")
    assert not hasattr(ag_transport, "_hook")
    monkeypatch.setattr(
        ag_transport,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): ("2", "1"),
    )
    assert ag_transport.select_idle_campaign_device("0", ("0", "1", "2")) == "2"
