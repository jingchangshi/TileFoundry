"""Tests for msprof op CSV harness wiring, P2 deep-dive fallback, and symptom routing."""

from __future__ import annotations

import importlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ascend.agent import test_surrogate_batch_npu as batch_harness
from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    SymptomFamily,
)
from tilefoundry_ascend.ascend.cost_model.msprof_report_parser import (
    extract_p2_from_objdump,
    parse_objdump_dwarf_hotspots,
)


def _write_executable(path: Path, script_body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if script_body.startswith("#!/usr/bin/env python3\n"):
        script_body = f"#!{sys.executable}\n" + script_body[len("#!/usr/bin/env python3\n") :]
    path.write_text(script_body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _write_dual_subkernel_opprof_csvs(opprof_dir: Path) -> None:
    opprof_dir.mkdir(parents=True, exist_ok=True)
    (opprof_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task ID,Task Duration(us),Block Dim\n"
        "aic_subkernel,0,30.0,24\n"
        "aiv_subkernel,1,10.0,48\n",
        encoding="utf-8",
    )
    (opprof_dir / "PipeUtilization.csv").write_text(
        "Op Name,Task ID,aic_mac_ratio,aiv_vec_ratio,mte2_wait_ratio,aic_fixpipe\n"
        "aic_subkernel,0,80.0%,0.0%,20.0%,50.0%\n"
        "aiv_subkernel,1,0.0%,60.0%,40.0%,0.0%\n",
        encoding="utf-8",
    )
    (opprof_dir / "Memory.csv").write_text(
        "Op Name,Task ID,hbm_read_bytes(KB),hbm_write_bytes(KB)\n"
        "aic_subkernel,0,1024,256\n"
        "aiv_subkernel,1,512,256\n",
        encoding="utf-8",
    )
    (opprof_dir / "L2Cache.csv").write_text(
        "Op Name,Task ID,l2_hit_rate(%)\n"
        "aic_subkernel,0,80.0\n"
        "aiv_subkernel,1,40.0\n",
        encoding="utf-8",
    )
    (opprof_dir / "ResourceConflictRatio.csv").write_text(
        "Op Name,Task ID,ub_bank_conflict_ratio\n"
        "aic_subkernel,0,0.04\n"
        "aiv_subkernel,1,0.20\n",
        encoding="utf-8",
    )


def _install_fake_npu_batch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTensor:
        def __init__(self, val: float) -> None:
            self.val = val

        def npu(self) -> _FakeTensor:
            return self

    class _FakeHarness:
        def prepare(self, _case: object) -> dict[str, _FakeTensor]:
            return {"x": _FakeTensor(1.0)}

        def reference_output(
            self, _case: object, _ts: object, native: bool = False
        ) -> _FakeTensor:
            return _FakeTensor(2.0 if not native else 2.0)

        def params(self, _case: object, tensors: dict[str, _FakeTensor]) -> dict[str, object]:
            return {"x": tensors["x"]}

        def compare(
            self, _case: object, _actual: object, _oracle: object, _tensors: object, _native_fn: object
        ) -> tuple[bool, dict[str, object]]:
            return True, {}

    class _FakeRefCache:
        def __init__(self, _harness: object) -> None:
            self._store: dict[tuple[int, str], object] = {}

        def __enter__(self) -> _FakeRefCache:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, case: object, tag: str, builder: object) -> object:
            key = (int(getattr(case, "case_num", 1)), tag)
            if key not in self._store:
                self._store[key] = builder()
            return self._store[key]

    fake_contract = SimpleNamespace(
        ReferenceCache=_FakeRefCache,
        map_tensors=lambda obj, fn: {k: fn(v) for k, v in obj.items()},
    )
    monkeypatch.setattr(
        batch_harness,
        "load_cannbench_contract_module",
        lambda **_kwargs: fake_contract,
    )
    monkeypatch.setattr(
        batch_harness,
        "load_anchor_harness_and_cases",
        lambda _op, anchors, **_kwargs: (
            _FakeHarness(),
            [SimpleNamespace(case_num=int(a.case_num)) for a in anchors],
        ),
    )
    monkeypatch.setattr(batch_harness, "_measure_npu_latency_us", lambda *_a, **_k: 40.0)
    fake_npu = SimpleNamespace(
        synchronize=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: True,
        set_device=lambda _idx: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)


def test_msprof_op_csv_wiring_track_a_and_track_b_dual_subkernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify msprof op CLI parsing populates measured CSV counters for Track-A and Track-B."""
    invocations_log = tmp_path / "msprof_invocations.jsonl"
    fake_msprof = _write_executable(
        tmp_path / "bin" / "msprof",
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args and args[0] == 'op', f'unexpected args: {args}'\n"
        "out_dir = None\n"
        "aic_metrics = ''\n"
        "for tok in args[1:]:\n"
        "    if tok.startswith('--output='):\n"
        "        out_dir = Path(tok.split('=', 1)[1])\n"
        "    elif tok.startswith('--aic-metrics='):\n"
        "        aic_metrics = tok.split('=', 1)[1]\n"
        "assert out_dir is not None\n"
        "opprof = out_dir / 'OPPROF_20261122_000001'\n"
        "opprof.mkdir(parents=True, exist_ok=True)\n"
        "(opprof / 'OpBasicInfo.csv').write_text(\n"
        "    'Op Name,Task ID,Task Duration(us)\\n'\n"
        "    'aic_subkernel,0,30.0\\naiv_subkernel,1,10.0\\n', encoding='utf-8')\n"
        "(opprof / 'PipeUtilization.csv').write_text(\n"
        "    'Op Name,Task ID,aic_mac_ratio,aiv_vec_ratio,mte2_wait_ratio,aic_fixpipe\\n'\n"
        "    'aic_subkernel,0,80.0%,0.0%,20.0%,50.0%\\n'\n"
        "    'aiv_subkernel,1,0.0%,60.0%,40.0%,0.0%\\n', encoding='utf-8')\n"
        "(opprof / 'Memory.csv').write_text(\n"
        "    'Op Name,Task ID,hbm_read_bytes(KB),hbm_write_bytes(KB)\\n'\n"
        "    'aic_subkernel,0,1024,256\\naiv_subkernel,1,512,256\\n', encoding='utf-8')\n"
        "(opprof / 'L2Cache.csv').write_text(\n"
        "    'Op Name,Task ID,l2_hit_rate(%)\\n'\n"
        "    'aic_subkernel,0,80.0\\naiv_subkernel,1,40.0\\n', encoding='utf-8')\n"
        "(opprof / 'ResourceConflictRatio.csv').write_text(\n"
        "    'Op Name,Task ID,ub_bank_conflict_ratio\\n'\n"
        "    'aic_subkernel,0,0.04\\naiv_subkernel,1,0.20\\n', encoding='utf-8')\n"
        f"with Path({str(invocations_log)!r}).open('a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps({'args': args, 'aic_metrics': aic_metrics}) + '\\n')\n",
    )
    monkeypatch.setenv("PATH", f"{fake_msprof.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("TF_DISABLE_MSPROF_OP", raising=False)
    _install_fake_npu_batch_deps(monkeypatch)

    track_a_mod = tmp_path / "cand_track_a.py"
    track_a_mod.write_text(
        "def run_op(x):\n    return x\n",
        encoding="utf-8",
    )
    track_b_mod = tmp_path / "cand_track_b.py"
    track_b_mod.write_text(
        "def run_op(x):\n    return x\n",
        encoding="utf-8",
    )

    batch = {
        "operator_name": "synthetic_dual",
        "entrypoint": "run_op",
        "candidates": [
            {"candidate_id": "cand_track_a", "module_path": str(track_a_mod)},
            {"candidate_id": "cand_track_b", "module_path": str(track_b_mod)},
        ],
        "anchors": [
            AnchorCaseSpec("synthetic_dual", 1, "reg_a", 50.0, 2.0).to_dict(),
        ],
        "msprof_warmup": 2,
        "msprof_repeat": 4,
    }
    msprof_out: dict[str, dict[str, dict[str, object]]] = {}
    results = batch_harness._evaluate_operator_batch(
        batch,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
        warmup=1,
        repeat=2,
        collect_msprof_detail=True,
        msprof_out=msprof_out,
    )
    assert results["cand_track_a"].all_passed is True
    assert results["cand_track_b"].all_passed is True

    logged = [
        json.loads(line)
        for line in invocations_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(logged) == 2
    for entry in logged:
        assert entry["aic_metrics"] == batch_harness._P1_AIC_METRICS
        assert "--warm-up=2" in entry["args"]
        assert "--launch-count=4" in entry["args"]

    for cid in ("cand_track_a", "cand_track_b"):
        tel = msprof_out[cid]["1"]
        assert tel["msprof_source"] == "msprof_op"
        assert tel["subkernel_count"] == 2
        assert tel["dominant_subkernel_name"] == "aic_subkernel"
        assert tel["dur_us"] == pytest.approx(40.0)
        assert tel["aic_mac_ratio"] == pytest.approx(0.60)
        assert tel["aiv_vec_ratio"] == pytest.approx(0.15)
        assert tel["aiv_vec"] == pytest.approx(0.15)
        assert tel["mte2_wait_ratio"] == pytest.approx(0.25)
        assert tel["aiv_mte2"] == pytest.approx(0.25)
        assert tel["aic_fixpipe"] == pytest.approx(0.375)
        assert tel["l2_hit_rate"] == pytest.approx(0.70)
        assert tel["bank_conflict_ratio"] == pytest.approx(0.08)
        assert tel["ub_bank_conflict_ratio"] == pytest.approx(0.08)
        assert tel["hbm_read_bytes"] == 1536 * 1024
        assert tel["hbm_write_bytes"] == 512 * 1024
        assert tel["actual_gm_bytes"] == 2048 * 1024
        assert "sync_stall_ratio" not in tel


def test_msprof_op_falls_back_to_tilelang_profiler_when_unavailable_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback to tilelang.profiler.do_bench occurs if msprof is absent, disabled, or empty."""
    fallback_calls: list[dict[str, object]] = []

    def _fake_do_bench(_fn: object, **kwargs: object) -> SimpleNamespace:
        fallback_calls.append(dict(kwargs))
        return SimpleNamespace(
            aic_mad=75.0,
            aiv_vec=30.0,
            aiv_mte2=20.0,
            dur_us=14.0,
        )

    real_import = importlib.import_module

    def _mock_import(name: str, package: str | None = None):
        if name == "tilelang.profiler":
            return SimpleNamespace(do_bench=_fake_do_bench)
        return real_import(name, package)

    monkeypatch.setattr(batch_harness.importlib, "import_module", _mock_import)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _i: None), raising=False)
    monkeypatch.setattr(torch, "device", lambda spec: str(spec))

    monkeypatch.setattr(batch_harness.shutil, "which", lambda _name: None)
    no_bin_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
    )
    assert no_bin_metrics["aic_mac_ratio"] == pytest.approx(0.75)
    assert no_bin_metrics["dur_us"] == pytest.approx(14.0)
    assert len(fallback_calls) == 1

    empty_msprof = _write_executable(
        tmp_path / "bin" / "msprof",
        "#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n",
    )
    monkeypatch.setattr(
        batch_harness.shutil,
        "which",
        lambda name: str(empty_msprof) if name == "msprof" else None,
    )
    empty_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
    )
    assert empty_metrics["aiv_vec_ratio"] == pytest.approx(0.30)
    assert len(fallback_calls) == 2

    monkeypatch.setenv("TF_DISABLE_MSPROF_OP", "1")
    disabled_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
    )
    assert disabled_metrics["mte2_wait_ratio"] == pytest.approx(0.20)
    assert len(fallback_calls) == 3


def test_p2_deep_dive_hook_trace_and_simulator_dwarf_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify P2 trace parsing and simulator-unsupported retry with llvm-objdump DWARF fallback."""
    trace_msprof = _write_executable(
        tmp_path / "trace_bin" / "msprof",
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "out_dir = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))\n"
        "aic = next(a.split('=', 1)[1] for a in args if a.startswith('--aic-metrics='))\n"
        "assert 'InstrTimeline' in aic and 'PCSampling' in aic and 'Source' in aic\n"
        "opprof = out_dir / 'OPPROF_TRACE'\n"
        "opprof.mkdir(parents=True, exist_ok=True)\n"
        "(opprof / 'OpBasicInfo.csv').write_text(\n"
        "    'Op Name,Task ID,Task Duration(us)\\naiv_main,0,12.0\\n', encoding='utf-8')\n"
        "(opprof / 'PipeUtilization.csv').write_text(\n"
        "    'Op Name,Task ID,aiv_vec_ratio,mte2_wait_ratio\\naiv_main,0,40.0%,55.0%\\n', encoding='utf-8')\n"
        "(opprof / 'pc_start_addr.txt').write_text(\n"
        "    '0x1000 0x1020 kernel.py:42\\n0x1020 0x1040 kernel.py:55\\n', encoding='utf-8')\n"
        "trace_rows = [\n"
        "    {'pipe': 'MTE2', 'opcode': 'mte_wait_flag', 'issue_us': 1.0, 'pc': '0x1008'},\n"
        "    {'pipe': 'MTE2', 'opcode': 'mte_wait_flag', 'issue_us': 2.5, 'pc': '0x1010'},\n"
        "    {'pipe': 'PIPE_V', 'opcode': 'vadd', 'issue_us': 3.0, 'pc': '0x1028'},\n"
        "    {'pipe': 'PIPE_V', 'opcode': 'vmul', 'issue_us': 5.0, 'pc': '0x1030'},\n"
        "]\n"
        "(opprof / 'instr_timeline.json').write_text(json.dumps(trace_rows), encoding='utf-8')\n",
    )
    monkeypatch.setenv("PATH", f"{trace_msprof.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("TF_DISABLE_MSPROF_OP", raising=False)
    mod_file = tmp_path / "cand.py"
    mod_file.write_text("def run_op(x):\n    return x\n", encoding="utf-8")

    p2_trace_tel = batch_harness._measure_msprof_detail_metrics(
        None,
        warmup=1,
        repeat=1,
        operator_name="softmax",
        candidate_id="bottleneck_p2_deep_dive",
        module_path=mod_file,
        entrypoint="run_op",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
    )
    assert p2_trace_tel["msprof_source"] == "msprof_op"
    assert p2_trace_tel["p2_source"] == "msprof_p2_trace"
    assert p2_trace_tel["p2_mte_waitflag_share"] == pytest.approx(0.5)
    assert p2_trace_tel["p2_issue_interval_us"]["MTE2"]["p50_us"] == pytest.approx(1.5)
    assert p2_trace_tel["p2_hotspots"][0]["source_location"] == "kernel.py:42"
    assert any("kernel.py:42" in line for line in p2_trace_tel["p2_summary_lines"])

    sim_log = tmp_path / "sim_calls.txt"
    sim_msprof = _write_executable(
        tmp_path / "sim_bin" / "msprof",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "out_dir = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))\n"
        "aic = next(a.split('=', 1)[1] for a in args if a.startswith('--aic-metrics='))\n"
        f"with Path({str(sim_log)!r}).open('a', encoding='utf-8') as fh:\n"
        "    fh.write(aic + '\\n')\n"
        "if 'InstrTimeline' in aic:\n"
        "    sys.stderr.write('unsupported metric InstrTimeline in simulator\\n')\n"
        "    sys.exit(2)\n"
        "opprof = out_dir / 'OPPROF_RETRY'\n"
        "opprof.mkdir(parents=True, exist_ok=True)\n"
        "(opprof / 'OpBasicInfo.csv').write_text(\n"
        "    'Op Name,Task ID,Task Duration(us)\\naiv_main,0,15.0\\n', encoding='utf-8')\n"
        "(opprof / 'PipeUtilization.csv').write_text(\n"
        "    'Op Name,Task ID,aiv_vec_ratio\\naiv_main,0,50.0%\\n', encoding='utf-8')\n",
    )
    fake_objdump = _write_executable(
        tmp_path / "sim_bin" / "llvm-objdump",
        "#!/usr/bin/env python3\n"
        "print('0000000000000000 <my_ascend_kernel>:')\n"
        "print('; /workspace/kernel.cpp:42')\n"
        "print('   0: 01 00 00 00   mte_wait_flag PIPE_MTE2')\n"
        "print('   4: 02 00 00 00   mte_wait_flag PIPE_MTE2')\n"
        "print('; /workspace/kernel.cpp:58')\n"
        "print('   8: 03 00 00 00   vadd PIPE_V')\n",
    )
    dummy_obj = tmp_path / "kernel.o"
    dummy_obj.write_bytes(b"\x7fELF" + b"\x00" * 64)
    monkeypatch.setenv("PATH", f"{sim_msprof.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    p2_dwarf_tel = batch_harness._measure_msprof_detail_metrics(
        None,
        warmup=1,
        repeat=1,
        operator_name="softmax",
        candidate_id="bottleneck_p2_deep_dive",
        module_path=mod_file,
        entrypoint="run_op",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
        compiled_obj_paths=[dummy_obj],
    )
    calls = sim_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "InstrTimeline" in calls[0]
    assert "InstrTimeline" not in calls[1]
    assert p2_dwarf_tel["dur_us"] == pytest.approx(15.0)
    assert p2_dwarf_tel["aiv_vec_ratio"] == pytest.approx(0.50)
    assert p2_dwarf_tel["p2_source"] == "llvm_objdump_dwarf"
    assert p2_dwarf_tel["p2_mte_waitflag_share"] == pytest.approx(2.0 / 3.0)
    assert p2_dwarf_tel["p2_hotspots"][0]["source_location"] == "/workspace/kernel.cpp:42"
    assert p2_dwarf_tel["p2_hotspots"][0]["instruction_count"] == 2
    assert any("/workspace/kernel.cpp:42" in line for line in p2_dwarf_tel["p2_summary_lines"])

    direct_summary = extract_p2_from_objdump([dummy_obj], objdump_bin=fake_objdump)
    assert direct_summary.total_instructions == 3
    parsed_text = parse_objdump_dwarf_hotspots(
        "0000 <k>:\n; src.cpp:10\n  10: 00000001 wait_flag PIPE_MTE2\n"
    )
    assert parsed_text.mte_waitflag_count == 1
    assert parsed_text.source_line_hotspots[0].source_location == "src.cpp:10"


def test_collect_p2_deep_dive_and_bottleneck_msprof_via_agent_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify P2 deep-dive preserves non-numeric fields while bottleneck collector filters them."""
    src_root = tmp_path / "repo"
    src_root.mkdir(parents=True, exist_ok=True)
    cand = src_root / "candidate.py"
    cand.write_text("def run_op(x):\n    return x\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={tmp_path}\n"
        f"TILEFOUNDRY_NPU_WORK_ROOT={tmp_path / 'work'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "ws")
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_k: tmp_path)
    monkeypatch.setattr(surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_surrogate_tilelang_cache", lambda *_a, **_k: tmp_path
    )
    monkeypatch.setattr(surrogate_cannbench, "_verify_fast_bindings", lambda *_a, **_k: None)
    monkeypatch.setattr(surrogate_cannbench, "_verify_fast_coverage", lambda *_a, **_k: None)

    recorded_payloads: list[dict[str, object]] = []

    def _fake_submit(**kwargs: object) -> tuple[dict[str, object], SimpleNamespace]:
        req_path = Path(str(kwargs["request_path"]))
        req_payload = json.loads(req_path.read_text(encoding="utf-8"))
        recorded_payloads.append(req_payload)
        cid = str(req_payload["msprof_targets"][0]["candidate_id"])
        raw_tel = {
            "dur_us": 18.5,
            "aiv_vec_ratio": 0.42,
            "dominant_subkernel_name": "aiv_kernel",
            "msprof_source": "msprof_op",
            "p2_source": "msprof_p2_trace",
            "p2_mte_waitflag_share": 0.35,
            "p2_issue_interval_us": {"MTE2": {"p50_us": 1.2, "p90_us": 2.4}},
            "p2_hotspots": [{"source_location": "kernel.py:19", "instruction_count": 12}],
            "p2_summary_lines": ["kernel.py:19 (12 instrs, 35.0%)"],
        }
        return (
            {"msprof_telemetry": {"softmax": {cid: {"3": raw_tel}}}},
            SimpleNamespace(state="FINISHED"),
        )

    monkeypatch.setattr(surrogate_cannbench, "_submit_fast_gate", _fake_submit)
    anchors = [AnchorCaseSpec("level1/softmax", 3, "vector_mid", 20.0, 2.0)]

    numeric_only = surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
        "softmax",
        cand,
        anchors,
        source_root=src_root,
        env_file=env_file,
    )
    assert recorded_payloads[0]["mode"] == "profile"
    assert recorded_payloads[0]["msprof_targets"][0]["candidate_id"] == "bottleneck_probe"
    assert numeric_only == {
        "vector_mid": {
            "dur_us": 18.5,
            "aiv_vec_ratio": 0.42,
            "p2_mte_waitflag_share": 0.35,
        }
    }

    deep_dive = surrogate_cannbench.collect_p2_deep_dive_via_agent_gates(
        "softmax",
        cand,
        anchors,
        source_root=src_root,
        env_file=env_file,
    )
    assert recorded_payloads[1]["mode"] == "profile"
    assert recorded_payloads[1]["msprof_targets"][0]["candidate_id"] == "bottleneck_p2_deep_dive"
    reg_payload = deep_dive["vector_mid"]
    assert isinstance(reg_payload, dict)
    assert reg_payload["dur_us"] == pytest.approx(18.5)
    assert reg_payload["p2_source"] == "msprof_p2_trace"
    assert reg_payload["p2_mte_waitflag_share"] == pytest.approx(0.35)
    assert reg_payload["p2_issue_interval_us"]["MTE2"]["p50_us"] == pytest.approx(1.2)
    assert reg_payload["p2_hotspots"][0]["source_location"] == "kernel.py:19"
    assert reg_payload["p2_summary_lines"] == ["kernel.py:19 (12 instrs, 35.0%)"]


def test_bayesian_tracker_symptom_ranking_boost_threshold_and_family_tabu() -> None:
    """Verify symptom_ranking tier priority, >0.15 threshold, and family tabu suppression."""
    high_posterior_other = AscendCausalRule(
        rule_id="R_HIGH_VEC",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Unroll vector loop.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="ref:test",
        alpha_prior=20.0,
        beta_prior=1.0,
    )
    low_posterior_dma_a = AscendCausalRule(
        rule_id="R_DMA_A",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Fuse GM roundtrip into UB.",
        expected_postcondition="GM_BYTES <= prev.GM_BYTES",
        source_provenance="ref:test",
        alpha_prior=2.0,
        beta_prior=2.0,
    )
    low_posterior_dma_b = AscendCausalRule(
        rule_id="R_DMA_B",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Keep activations resident in UB.",
        expected_postcondition="GM_BYTES <= prev.GM_BYTES",
        source_provenance="ref:test",
        alpha_prior=4.0,
        beta_prior=2.0,
    )
    tracker = BayesianRuleTracker.from_rules(
        [high_posterior_other, low_posterior_dma_a, low_posterior_dma_b]
    )
    ctx = {"is_legal": True, "PIPE_V": 100}

    default_order = [r.rule_id for r in tracker.rank_applicable_rules(ctx)]
    assert default_order == ["R_HIGH_VEC", "R_DMA_B", "R_DMA_A"]

    boosted_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.80)],
        )
    ]
    assert boosted_order == ["R_DMA_B", "R_DMA_A", "R_HIGH_VEC"]

    selected_top = [
        r.rule_id
        for r in tracker.select_top_rules(
            ctx,
            top_k=2,
            symptom_ranking=[("dma_redundant_gm_roundtrip", 0.50)],
        )
    ]
    assert selected_top == ["R_DMA_B", "R_DMA_A"]

    threshold_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.15)],
        )
    ]
    assert threshold_order == default_order

    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R_DMA_A",
        parent_sha="sha_p1",
        current_round=1,
        reason="no_speedup",
    )
    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R_DMA_B",
        parent_sha="sha_p1",
        current_round=2,
        reason="no_speedup",
    )
    assert (
        tracker.is_family_tabued(
            regime_tag="reg_a",
            symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP,
            parent_sha="sha_p1",
            current_round=3,
        )
        is True
    )
    tabued_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            regime_tag="reg_a",
            parent_sha="sha_p1",
            current_round=3,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.90)],
        )
    ]
    assert tabued_order == ["R_HIGH_VEC"]
