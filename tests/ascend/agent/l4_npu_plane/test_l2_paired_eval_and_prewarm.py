"""Tests for L2 paired evaluation, poison-pill isolation, and prewarm manifest validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.gates import stage_b_gate
from tilefoundry_ascend.agent.npu import l2_paired_eval
from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec, PerfTargetSpec, get_default_regime_specs
from tilefoundry_ascend.ascend.compile_cache.artifact_store import ArtifactStore, compute_compile_task_key


class _ClockEvent:
    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self._t = 0.0

    def record(self) -> None:
        self._t = self._clock[0]

    def elapsed_time(self, other: _ClockEvent) -> float:
        return (other._t - self._t) / 1000.0


def _write_ag_config(cfg_dir: Path, *, source_root: Path) -> Path:
    evidence_root = cfg_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text(
        f'evidence_root = "{evidence_root}"\n\n'
        "[workspaces.tilefoundry]\n"
        f'source_root = "{source_root}"\n'
        'adapters = ["tilefoundry-npu"]',
        encoding="utf-8",
    )
    return cfg_path


def _setup_prewarm_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    cann_repo = tmp_path / "cann_repo"
    ar_dir = cann_repo / "tasks" / "level4" / "attention_residual"
    ar_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "cases": [
            {
                "case_id": 2,
                "input_shape": [[2, 1, 1, 4096], [1, 1, 4096], [4096], [4096]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
            {
                "case_id": 4,
                "input_shape": [[8, 1, 16, 1000], [1, 16, 1000], [1000], [1000]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
        ]
    }
    (ar_dir / "cases.yaml").write_text(yaml.safe_dump(cases_payload), encoding="utf-8")

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")

    tl_stub_dir = tmp_path / "tl_stub"
    (tl_stub_dir / "tilelang" / "backend").mkdir(parents=True, exist_ok=True)
    (tl_stub_dir / "tvm.py").write_text(
        "import contextlib\n"
        "class _Target:\n"
        "    @staticmethod\n"
        "    def current(allow_none=True): return 'ascend'\n"
        "class target:\n"
        "    Target = _Target\n"
        "class transform:\n"
        "    @staticmethod\n"
        "    def PassContext(config=None): return contextlib.nullcontext()\n"
        "class tirx:\n"
        "    PrimFunc = dict\n"
        "class IRModule:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tl_stub_dir / "tilelang" / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tl_stub_dir / "tilelang" / "backend" / "module.py").write_text(
        "import contextlib\nfrom types import SimpleNamespace\n"
        "def create_backend_context(_t): return SimpleNamespace(target=contextlib.nullcontext())\n",
        encoding="utf-8",
    )
    (tl_stub_dir / "tilelang" / "__init__.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    if prog.get('fail'):\n"
        "        raise RuntimeError('simulated_bisheng_compile_failure')\n"
        "    so_path = cache_dir / f\"k_{prog.get('tag', 'ok')}_{prog.get('n', 0)}.so\"\n"
        "    so_path.write_bytes(b'ELF_STUB')\n"
        "    return {'cached': prog}\n",
        encoding="utf-8",
    )

    work_root = tmp_path / "work_root"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub_dir}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    return fake_root, cann_repo, tl_stub_dir, work_root, env_file


def _write_ar_candidate(path: Path, *, tag: str, fail_compile: bool = False) -> Path:
    fail_flag = "True" if fail_compile else "False"
    path.write_text(
        "import tilelang\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"    return {{'tag': {tag!r}, 'n': n_depth, 'fail': {fail_flag}}}\n"
        "def _select_ar_unaligned_config(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=4):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_ar_unaligned_func(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"    return {{'tag': {tag!r}, 'n': n_depth, 'fail': {fail_flag}}}\n"
        "def _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[4], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n",
        encoding="utf-8",
    )
    return path


def _populate_manifest(
    cand_path: Path,
    *,
    anchors: tuple[AnchorCaseSpec, ...],
    cann_repo: Path,
    tl_cache: Path,
    fake_root: Path,
    env_file: Path,
    compile_ok: bool = True,
) -> None:
    ar_regs = {r.regime_tag: r for r in get_default_regime_specs("attention_residual")}
    case_dims: dict[str, dict[str, object]] = {}
    for anc in anchors:
        d = dict(l2_paired_eval.lookup_case_static_spec("attention_residual", anc.case_num, cannbench_repo=cann_repo))
        if (rg := ar_regs.get(anc.regime_tag)) is not None:
            for k, v in rg.sample_builder_kwargs.items():
                d.setdefault(str(k), v)
            if rg.config_selector_fn:
                d.setdefault("config_selector_name", rg.config_selector_fn)
            if rg.builder_candidates:
                d.setdefault("builder_name", rg.builder_candidates[0])
        case_dims[str(anc.case_num)] = d
    cand_src = cand_path.read_text(encoding="utf-8")
    store = ArtifactStore(tl_cache)
    ext = stage_b_gate.ModuleSelectorPlanExtractor(target="dav-3510")
    plan = stage_b_gate.build_post_selector_regime_plan(cand_src, "small_decode", case_dims, ext)

    def _compile_stub(spec: object, _src: str, work_dir: Path) -> tuple[bool, int, str, list[Path]]:
        if not compile_ok:
            return False, 0, "simulated_bisheng_compile_failure", []
        so_path = work_dir / f"{spec.task_key[:10]}.so"
        so_path.write_bytes(b"\x7fELF-stub")
        return True, 0, "", [so_path]

    res = stage_b_gate.StageBCompileGate(
        plan, store, compile_fn=_compile_stub, source_root=fake_root, env_file=env_file
    ).run(cand_src)
    assert res.passed is compile_ok


def test_poison_pill_adaptive_timeout_and_boot_handshake(tmp_path: Path) -> None:
    """Bound poison-pill timeout in [10, 60]s and shield slow boot behind READY fd."""
    assert l2_paired_eval.compute_adaptive_poison_pill_timeout([]) == pytest.approx(10.0)
    assert l2_paired_eval.compute_adaptive_poison_pill_timeout([20.0]) == pytest.approx(10.0)
    assert l2_paired_eval.compute_adaptive_poison_pill_timeout([500_000.0, 500_000.0]) == pytest.approx(
        25.0
    )
    assert l2_paired_eval.compute_adaptive_poison_pill_timeout([5_000_000.0]) == pytest.approx(60.0)

    boot_ok = tmp_path / "boot_ok.py"
    boot_ok.write_text(
        "import os, time\n"
        "time.sleep(0.15)\n"
        "fd = int(os.environ['TF_POISON_PILL_READY_FD'])\n"
        "os.write(fd, b'READY\\n')\n"
        "os.close(fd)\n",
        encoding="utf-8",
    )
    ok, err, rc, _so, _se = l2_paired_eval.run_poison_pill_subprocess(
        [sys.executable, str(boot_ok)], cwd=tmp_path, exec_timeout_sec=0.3, boot_timeout_sec=3.0
    )
    assert ok is True, err
    assert rc == 0

    hang_exec = tmp_path / "hang_exec.py"
    hang_exec.write_text(
        "import os, time\n"
        "fd = int(os.environ['TF_POISON_PILL_READY_FD'])\n"
        "os.write(fd, b'READY\\n')\n"
        "os.close(fd)\n"
        "time.sleep(5.0)\n",
        encoding="utf-8",
    )
    ok_h, err_h, rc_h, _so, _se = l2_paired_eval.run_poison_pill_subprocess(
        [sys.executable, str(hang_exec)], cwd=tmp_path, exec_timeout_sec=0.1, boot_timeout_sec=3.0
    )
    assert ok_h is False
    assert rc_h == -9
    assert "poison_pill_timeout_or_crash: timeout" in err_h

    crash_py = tmp_path / "crash.py"
    crash_py.write_text("import os\nos._exit(11)\n", encoding="utf-8")
    ok_c, err_c, rc_c, _so, _se = l2_paired_eval.run_poison_pill_subprocess(
        [sys.executable, str(crash_py)], cwd=tmp_path, exec_timeout_sec=2.0, boot_timeout_sec=2.0
    )
    assert ok_c is False
    assert rc_c == 11
    assert "poison_pill_timeout_or_crash" in err_c


def test_ring_burst_and_paired_interleaved_micro_burst_cancel_drift() -> None:
    """Ring burst retries high-CV spikes and interleaved pairing cancels >20% common-mode drift."""
    assert l2_paired_eval.compute_ring_burst_k(1024, t_hw_us=2.0) == 32
    assert l2_paired_eval.compute_ring_burst_k(8 * 1024 * 1024, t_hw_us=2.0) == 24
    assert l2_paired_eval.compute_ring_burst_k(1024, t_hw_us=15.0) == 1
    assert l2_paired_eval.compute_ring_burst_k(1024, first_probe_us=6.5) == 32

    class _StubBuf:
        def __init__(self, nbytes: int) -> None:
            self.nbytes = nbytes

    assert l2_paired_eval.estimate_params_io_bytes({"a": [_StubBuf(64), _StubBuf(128)]}) == 192
    ring = l2_paired_eval.allocate_rotating_ring_params({"x": _StubBuf(32)}, 3, clone_fn=lambda o: _StubBuf(o.nbytes))
    assert len(ring) == 3
    assert ring[0]["x"] is not ring[1]["x"]

    clock = [0.0]
    k = l2_paired_eval.compute_ring_burst_k(0, t_hw_us=2.0, first_probe_us=10.0)
    warmup_calls = 2
    idx = [0]
    pattern = [*([10.0] * warmup_calls), *([5.0] * k), *([6.0] * k), *([25.0] * k), *([5.0] * (3 * k))]

    def _retry_fn(_x: object) -> None:
        clock[0] += pattern[min(idx[0], len(pattern) - 1)]
        idx[0] += 1

    med, got_k, cv, retries = l2_paired_eval.measure_npu_ring_burst_stats(
        _retry_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=10.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert got_k == k
    assert retries == 1
    assert med == pytest.approx(5.0)
    assert cv == pytest.approx(0.0)

    drift_clock = [0.0]
    drift = [1.0, 1.15, 1.35, 1.60, 1.85, 2.05, 2.30]
    calls = {"inc": 0, "fast": 0}

    def _inc_fn(_x: object) -> None:
        r = max(0, min(len(drift) - 1, calls["inc"] - 1))
        calls["inc"] += 1
        drift_clock[0] += 20.0 * drift[r]

    def _fast_fn(_x: object) -> None:
        r = max(0, min(len(drift) - 1, calls["fast"] - 1))
        calls["fast"] += 1
        drift_clock[0] += 10.0 * drift[r]

    stats = l2_paired_eval.measure_paired_interleaved_micro_burst(
        [("incumbent_slot", _inc_fn), ("cand_fast", _fast_fn)],
        object(),
        warmup=1,
        repeat=7,
        t_hw_us=10.0,
        first_probe_us=20.0,
        event_factory=lambda: _ClockEvent(drift_clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert stats["incumbent_slot"][2] > 0.20
    assert stats["incumbent_slot"][3] == pytest.approx(1.0)
    assert stats["incumbent_slot"][4] == pytest.approx(0.0)
    assert stats["cand_fast"][3] == pytest.approx(2.0, rel=1e-2)
    assert stats["cand_fast"][4] < 0.02


def test_anchor_metrics_surrogate_aggregation_and_perf_targets() -> None:
    """Roundtrip AnchorEvalMetrics, aggregate CandidateSurrogateResult, and check PerfTargetSpec."""
    s1 = AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0)
    s2 = AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 20.0, 4.0)
    m1 = l2_paired_eval.build_anchor_metrics(
        s1,
        accuracy_passed=True,
        cand_us=4.0,
        ring_burst_k=24,
        burst_batch_cv=0.015,
        spike_retries=1,
        card_ext_hbm_mb=2048,
        paired_speedup_vs_incumbent=1.8,
        paired_rel_mad=0.012,
    )
    m2 = l2_paired_eval.build_anchor_metrics(s2, accuracy_passed=True, cand_us=10.0)
    restored = l2_paired_eval.AnchorEvalMetrics.from_dict(m1.to_dict())
    assert restored.paired_speedup_vs_incumbent == pytest.approx(1.8)
    assert restored.card_ext_hbm_mb == 2048

    res = l2_paired_eval.aggregate_surrogate_result("c1", "attention_residual", [m1, m2])
    assert res.all_passed is True
    assert res.avg_speedup == pytest.approx(2.25)
    assert res.min_case_speedup == pytest.approx(2.0)
    assert res.for_regime("small_decode").anchors == (m1,)
    assert res.regime_headroom("small_decode") == pytest.approx(0.5)
    assert res.mean_latency_us() == pytest.approx(7.0)

    spec = PerfTargetSpec(
        "attention_residual",
        target_avg_speedup=2.0,
        target_min_case_speedup=1.8,
        target_roofline_eta=0.4,
        target_surrogate_score=50.0,
        per_anchor_max_us={2: 5.0},
    )
    assert res.meets_perf_target(spec) is True
    assert res.is_roofline_saturated(spec) is True
    strict = PerfTargetSpec("attention_residual", 2.0, 1.8, 0.4, per_anchor_max_us={2: 3.5})
    assert res.meets_perf_target(strict) is False


def test_prewarm_manifest_v1_fast_path_and_anchor_ref_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hit prewarm_manifest_v1.json without subprocesses and invalidate on artifact/yaml loss."""
    fake_root, cann_repo, _tl, work_root, env_file = _setup_prewarm_fixture(tmp_path)
    cand_a = _write_ar_candidate(tmp_path / "cand_a.py", tag="a")
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
    )
    ref_cache = work_root / "surrogate" / "reference"
    tl_cache = work_root / "surrogate" / "tilelang"

    spawn_count = 0
    real_run = subprocess.run

    def _counting_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal spawn_count
        spawn_count += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _counting_run)
    _populate_manifest(
        cand_a,
        anchors=anchors,
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
    )
    assert spawn_count == 0
    assert (tl_cache / "prewarm_manifest_v1.json").is_file()

    spawn_count = 0
    res_hit = l2_paired_eval.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert spawn_count == 0
    assert res_hit.prewarm_stats == {
        "total_tasks": 2,
        "cache_hits": 2,
        "compiled_tasks": 0,
        "failed_candidates": 0,
    }

    for so_file in tl_cache.rglob("*.so"):
        so_file.unlink()
    res_miss = l2_paired_eval.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
        isolate_non_incumbent=True,
    )
    assert res_miss.prewarm_stats["failed_candidates"] == 1
    assert res_miss.prewarm_stats["cache_hits"] == 0

    with pytest.raises(RuntimeError, match="prewarm_surrogate_tilelang_cache failed"):
        l2_paired_eval.prewarm_surrogate_tilelang_cache(
            "attention_residual",
            [("c_a", cand_a)],
            anchors,
            cannbench_repo=cann_repo,
            env_file=env_file,
            cache_dir=ref_cache,
            tl_cache_dir=tl_cache,
            source_root=fake_root,
        )

    op_dir = ref_cache / "operators" / "level4__attention_residual"
    ref_spawns = 0

    def _fake_ref_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal ref_spawns
        ref_spawns += 1
        op_dir.mkdir(parents=True, exist_ok=True)
        (op_dir / "identity.json").write_text('{"task": "level4/attention_residual"}', encoding="utf-8")
        for cid in (2, 4):
            (op_dir / f"case_{cid}.oracle.pt").write_bytes(b"ORACLE")
            (op_dir / f"case_{cid}.native.pt").write_bytes(b"NATIVE")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_ref_run)
    l2_paired_eval.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert ref_spawns == 1
    l2_paired_eval.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert ref_spawns == 1
    cases_path = cann_repo / "tasks" / "level4" / "attention_residual" / "cases.yaml"
    cases_path.write_text(cases_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    l2_paired_eval.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert ref_spawns == 2


def test_batch_isolation_paired_guardrail_and_response_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate failed non-incumbents, evaluate paired guardrails, and reject tampered receipts."""
    fake_root, cann_repo, _tl, work_root, env_file = _setup_prewarm_fixture(tmp_path)
    cfg_path = _write_ag_config(tmp_path / "cfg", source_root=fake_root)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_path))
    ag_bin = tmp_path / "agent-gates"
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", str(ag_bin))

    inc_cand = _write_ar_candidate(tmp_path / "inc.py", tag="inc")
    good_cand = _write_ar_candidate(tmp_path / "good.py", tag="good")
    bad_cand = _write_ar_candidate(tmp_path / "bad.py", tag="bad", fail_compile=True)
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),)
    tl_cache = work_root / "surrogate" / "tilelang"
    _populate_manifest(
        inc_cand,
        anchors=anchors,
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
    )
    _populate_manifest(
        good_cand,
        anchors=anchors,
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
    )
    _populate_manifest(
        bad_cand,
        anchors=anchors,
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
        compile_ok=False,
    )

    ref_dir = work_root / "surrogate" / "reference" / "operators" / "level4__attention_residual"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "identity.json").write_text("{}", encoding="utf-8")
    (ref_dir / "case_2.oracle.pt").write_bytes(b"O")
    (ref_dir / "case_2.native.pt").write_bytes(b"N")
    monkeypatch.setattr(l2_paired_eval, "prewarm_anchor_reference_cache", lambda *_a, **_k: work_root / "surrogate" / "reference")

    submitted_cids: list[list[str]] = []
    tamper_mode = {"val": ""}

    def _fake_ag_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd_list = [str(p) for p in cmd]
        if cmd_list[0] == "npu-smi":
            return subprocess.CompletedProcess(cmd_list, 1, "", "")
        sub = cmd_list[1]
        if sub == "doctor":
            return subprocess.CompletedProcess(cmd_list, 0, json.dumps({"status": "PASS"}), "")
        if sub == "submit":
            req_rel = cmd_list[cmd_list.index("--candidate") + 1]
            req_path = fake_root / req_rel
            req_bytes = req_path.read_bytes()
            req = json.loads(req_bytes)
            batch = req["batches"][0]
            submitted_cids.append([c["candidate_id"] for c in batch["candidates"]])
            bindings = {
                "attention_residual": {
                    c["candidate_id"]: {
                        "module_path": c["module_path"],
                        "sha256": "0" * 64 if tamper_mode["val"] == "sha" else c["sha256"],
                        "post_sha256": c["sha256"],
                    }
                    for c in batch["candidates"]
                }
            }
            ops = {
                c["candidate_id"]: {
                    "candidate_id": c["candidate_id"],
                    "operator_name": "attention_residual",
                    "all_passed": True,
                    "avg_speedup": 1.0 if c["candidate_id"] == "incumbent_slot" else 2.5,
                    "min_case_speedup": 1.0 if c["candidate_id"] == "incumbent_slot" else 2.5,
                    "avg_roofline_eta": 0.7,
                    "surrogate_score": 82.0,
                    "anchors": [
                        {
                            "case_num": 2,
                            "regime_tag": "small_decode",
                            "accuracy_passed": True,
                            "cand_us": 1.5 if c["candidate_id"] == "incumbent_slot" else 0.6,
                            "ref_perf_us": 1.5,
                            "t_hw_us": 0.5,
                            "speedup": 1.0 if c["candidate_id"] == "incumbent_slot" else 2.5,
                            "roofline_eta": 0.33 if c["candidate_id"] == "incumbent_slot" else 0.83,
                            "eq3_score": 82.0,
                            "paired_speedup_vs_incumbent": 1.0 if c["candidate_id"] == "incumbent_slot" else 2.5,
                            "paired_rel_mad": 0.01,
                        }
                    ],
                }
                for c in batch["candidates"]
            }
            resp = {
                "schema": l2_paired_eval._verify_fast_bindings.__module__ and "tilefoundry-fast-response/v1",
                "request_id": req["request_id"],
                "request_sha256": hashlib.sha256(req_bytes).hexdigest(),
                "mode": "batch",
                "status": "infrastructure_error" if tamper_mode["val"] == "infra" else "ok",
                "error": "device lost" if tamper_mode["val"] == "infra" else "",
                "null_launch_us": 2.35,
                "candidates": bindings,
                "coverage": {
                    "expected_candidates": len(batch["candidates"]),
                    "evaluated_candidates": len(batch["candidates"]),
                    "expected_msprof_targets": 0,
                    "observed_msprof_targets": 0,
                },
                "operators": {"attention_residual": ops},
            }
            (fake_root / req["response_path"]).write_text(json.dumps(resp), encoding="utf-8")
            return subprocess.CompletedProcess(cmd_list, 0, json.dumps({"execution_id": "exec-b"}), "")
        if sub == "wait":
            return subprocess.CompletedProcess(
                cmd_list, 0, json.dumps({"status": {"state": "FINISHED", "run_id": "run-b"}}), ""
            )
        if sub == "report":
            return subprocess.CompletedProcess(
                cmd_list,
                0,
                json.dumps(
                    {
                        "status": "PASS",
                        "runId": "run-b",
                        "receiptPath": "/receipt-b.json",
                        "gates": [{"gateId": "tilefoundry-npu-device-fast", "observations": {"deviceWindow": {"heldSeconds": 1.5}}}],
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(
            cmd_list,
            0,
            json.dumps({"satisfied": True, "current_source_identity": {"root": str(fake_root.resolve())}}),
            "",
        )

    monkeypatch.setattr(subprocess, "run", _fake_ag_run)
    by_op, run_id, receipt_path, held = l2_paired_eval.run_multi_operator_surrogate_batch_via_agent_gates(
        [
            (
                "attention_residual",
                [("incumbent_slot", inc_cand), ("c_good", good_cand), ("c_bad", bad_cand)],
                anchors,
            )
        ],
        source_root=fake_root,
        env_file=env_file,
    )
    assert run_id == "run-b"
    assert receipt_path == "/receipt-b.json"
    assert held == pytest.approx(1.5)
    assert submitted_cids == [["incumbent_slot", "c_good"]]
    op_res = by_op["attention_residual"]
    assert isinstance(op_res, l2_paired_eval.L2BatchResults)
    assert op_res.null_launch_us == pytest.approx(2.35)
    assert op_res.prewarm_stats["failed_candidates"] == 1
    assert op_res["c_good"].anchors[0].paired_speedup_vs_incumbent == pytest.approx(2.5)
    assert op_res["c_bad"].all_passed is False
    assert "prewarm" in op_res["c_bad"].error_message

    submitted_cids.clear()
    empty_op, empty_rid, _rcp, empty_held = l2_paired_eval.run_multi_operator_surrogate_batch_via_agent_gates(
        [("attention_residual", [("incumbent_slot", inc_cand), ("c_bad", bad_cand)], anchors)],
        source_root=fake_root,
        env_file=env_file,
    )
    assert submitted_cids == []
    assert empty_rid == ""
    assert empty_held == 0.0
    assert empty_op["attention_residual"]["c_bad"].all_passed is False

    tamper_mode["val"] = "sha"
    with pytest.raises(RuntimeError, match="does not match the staged candidate"):
        l2_paired_eval.run_surrogate_batch_via_agent_gates(
            "attention_residual",
            [("incumbent_slot", inc_cand), ("c_good", good_cand)],
            anchors,
            source_root=fake_root,
            env_file=env_file,
        )

    tamper_mode["val"] = "infra"
    with pytest.raises(RuntimeError, match="infrastructure_error"):
        l2_paired_eval.run_surrogate_batch_via_agent_gates(
            "attention_residual",
            [("incumbent_slot", inc_cand), ("c_good", good_cand)],
            anchors,
            source_root=fake_root,
            env_file=env_file,
        )


def test_shim_creation_ast_signature_and_l0_precheck(tmp_path: Path) -> None:
    """Create npu_compare shim, normalize C++ comment AST signatures, and run L0 precheck."""
    cann_repo = tmp_path / "cann"
    cann_repo.mkdir()
    shim_dir = l2_paired_eval.ensure_npu_compare_shim_dir(tmp_path / "shim", cannbench_repo=cann_repo)
    shim_src = (shim_dir / "npu_compare.py").read_text(encoding="utf-8")
    assert 'cann_repo / "src"' in shim_src
    assert "compare_case_result" in shim_src

    src_a = '__global__ = """\n__global__ void k() {\n  // comment A\n  return;\n}\n"""\n'
    src_b = '__global__ = """\n__global__ void k() {\n  /* comment B */\n  return;\n}\n"""\n'
    assert l2_paired_eval._normalized_module_ast_signature(src_a) == l2_paired_eval._normalized_module_ast_signature(src_b)
    assert l2_paired_eval._normalized_module_ast_signature("def broken(:") is None

    py_mod = "def kernel(x):\n    return x\n"
    st_a = l2_paired_eval.run_l0_static_precheck(py_mod, builder_name="kernel", builder_kwargs={})
    assert st_a.passed is False
    assert "stage_a_non_kernel_callable" in st_a.error_message


def test_prewarm_allows_unwarmed_baseline_and_multi_regime_but_rejects_explicit_failure(
    tmp_path: Path,
) -> None:
    """Allow absent or unwarmed manifest entries while rejecting explicit Stage-B failures."""
    fake_root, cann_repo, _tl, work_root, env_file = _setup_prewarm_fixture(tmp_path)
    inc_cand = _write_ar_candidate(tmp_path / "inc.py", tag="inc")
    good_cand = _write_ar_candidate(tmp_path / "good.py", tag="good")
    bad_cand = _write_ar_candidate(tmp_path / "bad.py", tag="bad", fail_compile=True)
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
    )
    ref_cache = work_root / "surrogate" / "reference"
    tl_cache = work_root / "surrogate" / "tilelang"

    res_absent = l2_paired_eval.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_baseline", inc_cand)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
        isolate_non_incumbent=True,
    )
    assert res_absent.failed_by_cid == {}
    assert res_absent.prewarm_stats["failed_candidates"] == 0
    assert res_absent.prewarm_stats["total_tasks"] == 2
    assert res_absent.prewarm_stats["cache_hits"] == 0

    l2_paired_eval.prewarm_surrogate_tilelang_cache(
        "attention_residual",
        [("incumbent_baseline", inc_cand)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )

    _populate_manifest(
        good_cand,
        anchors=(anchors[0],),
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
    )
    _populate_manifest(
        bad_cand,
        anchors=(anchors[0],),
        cann_repo=cann_repo,
        tl_cache=tl_cache,
        fake_root=fake_root,
        env_file=env_file,
        compile_ok=False,
    )

    res_multi = l2_paired_eval.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_slot", inc_cand), ("c_good", good_cand), ("c_bad", bad_cand)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
        isolate_non_incumbent=True,
    )
    assert "incumbent_slot" not in res_multi.failed_by_cid
    assert "c_good" not in res_multi.failed_by_cid
    assert "c_bad" in res_multi.failed_by_cid
    assert res_multi.prewarm_stats["failed_candidates"] == 1
    assert res_multi.prewarm_stats["cache_hits"] >= 1


@pytest.mark.parametrize(
    ("use_pill", "phase_a_ok", "expected_ids"),
    [
        (False, frozenset(), ["incumbent_slot", "cand_pass", "cand_fail"]),
        (True, frozenset({"cand_pass"}), ["incumbent_slot", "cand_pass"]),
    ],
    ids=["accept_use_pill_false_binds_cid", "reject_use_pill_true_filters_unpassed"],
)
def test_surrogate_batch_harness_candidate_selection_reject_and_accept(
    tmp_path: Path,
    use_pill: bool,
    phase_a_ok: frozenset[str],
    expected_ids: list[str],
) -> None:
    """Bind cid when use_pill=False and exclude unpassed non-incumbents when use_pill=True."""
    repo_root = Path(__file__).resolve().parents[4]
    harness_path = repo_root / "tests" / "ascend" / "agent" / "surrogate_batch_harness.py"
    spec = importlib.util.spec_from_file_location("surrogate_batch_harness_test_mod", harness_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    seen_tags: list[str] = []

    class _StubLoader:
        def __call__(self, path: Path, ep: str, tag: str) -> tuple[object, object]:
            seen_tags.append(f"{path.name}:{ep}:{tag}")
            return (lambda **_kw: None), {"module": path.name}

    candidates = [
        {"candidate_id": "incumbent_slot", "module_path": str(tmp_path / "inc.py")},
        {"candidate_id": "cand_pass", "module_path": str(tmp_path / "pass.py")},
        {"candidate_id": "cand_fail", "module_path": str(tmp_path / "fail.py")},
    ]
    loaded = mod.load_selected_candidates(
        candidates,
        entrypoint="mla_forward",
        op_name="mla",
        use_pill=use_pill,
        phase_a_ok=phase_a_ok,
        loader=_StubLoader(),
    )
    assert [cid for cid, _fn, _m in loaded] == expected_ids
    assert len(seen_tags) == len(expected_ids)


@pytest.mark.parametrize("fail_compile", [False, True], ids=["accept_normal_exit", "reject_compile_error"])
def test_mla_trackb_temp_dir_cleanup_reject_and_accept(tmp_path: Path, fail_compile: bool) -> None:
    """Clean up mla_trackb_* temp directories deterministically on normal and error exits."""
    repo_root = Path(__file__).resolve().parents[4]
    tl_stub = tmp_path / "tl_stub"
    (tl_stub / "tilelang" / "ascend").mkdir(parents=True, exist_ok=True)
    (tl_stub / "tilelang" / "contrib").mkdir(parents=True, exist_ok=True)
    (tl_stub / "tilelang" / "__init__.py").write_text("", encoding="utf-8")
    (tl_stub / "tilelang" / "ascend" / "__init__.py").write_text("", encoding="utf-8")
    (tl_stub / "tilelang" / "ascend" / "language.py").write_text("", encoding="utf-8")
    (tl_stub / "tilelang" / "contrib" / "__init__.py").write_text("", encoding="utf-8")
    (tl_stub / "tilelang" / "contrib" / "bisheng.py").write_text(
        "def compile_ascend(*_a, **_kw): return None\n", encoding="utf-8"
    )
    isolated_tmp = tmp_path / "isolated_tmp"
    isolated_tmp.mkdir(parents=True, exist_ok=True)
    driver = (
        "import importlib.util, os, sys\n"
        "from pathlib import Path\n"
        f"kpath = Path({str(repo_root / 'examples' / 'ascend950pr_ops' / 'mla' / 'kernels.py')!r})\n"
        "spec = importlib.util.spec_from_file_location('mla_kernels_ut', kpath)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "created_paths = []\n"
        "class _CompileStub:\n"
        "    def __call__(self, _src, *, target_format, npu_arch, options, path_target):\n"
        "        created_paths.append(Path(path_target).parent)\n"
        f"        if {fail_compile!r}:\n"
        "            raise RuntimeError('simulated_trackb_compile_failure')\n"
        "        Path(path_target).write_bytes(b'ELF')\n"
        "class _LoadStub:\n"
        "    def __call__(self, path):\n"
        "        assert Path(path).is_file()\n"
        "        return {'loaded': Path(path).name}\n"
        "try:\n"
        "    res = mod._load_uncached_trackb_so('src', [], compile_fn=_CompileStub(), load_cdll=_LoadStub())\n"
        f"    assert not {fail_compile!r} and res == {{'loaded': 'mla_trackb.so'}}\n"
        "except RuntimeError as exc:\n"
        f"    assert {fail_compile!r} and 'simulated_trackb_compile_failure' in str(exc)\n"
        "assert len(created_paths) == 1 and not created_paths[0].exists()\n"
    )
    env = dict(os.environ)
    env["TMPDIR"] = str(isolated_tmp)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tl_stub), str(repo_root / "src"), str(repo_root / "targets" / "ascend")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert list(isolated_tmp.glob("mla_trackb_*")) == []


def test_stage_b_gate_forwards_unit_kind_to_compile_task_key() -> None:
    """Forward unit_kind from Stage-B capture extraction so Track-A/B keys stay distinct."""
    src = (
        "def _select_tb(m_rows: int = 1):\n"
        "    return {'use_trackb': True, 'bm': 16}\n"
        "def _render_trackb_tu(bm: int = 16):\n"
        "    return '__global__ __aicore__ void k_tb(GM_ADDR x) {}\\n'\n"
    )
    ext = stage_b_gate.ModuleSelectorPlanExtractor(target="dav-3510")
    specs = ext.extract_case_tasks(src, "tb_case", {"m_rows": 1})
    assert len(specs) == 1
    tb_spec = specs[0]
    assert tb_spec.unit_kind == "cce_cpp20"
    static_sig = json.dumps(
        {
            "builder_name": tb_spec.builder_name,
            "static_kwargs": [[k, v] for k, v in tb_spec.static_kwargs],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cpp_key = compute_compile_task_key(
        normalized_kernel_unit_src=tb_spec.unit_source,
        cann_target="dav-3510",
        pass_configs=tb_spec.pass_configs,
        compile_flags=tb_spec.compile_flags,
        static_spec_signature=static_sig,
        unit_kind="cce_cpp20",
    )
    dsl_key = compute_compile_task_key(
        normalized_kernel_unit_src=tb_spec.unit_source,
        cann_target="dav-3510",
        pass_configs=tb_spec.pass_configs,
        compile_flags=tb_spec.compile_flags,
        static_spec_signature=static_sig,
        unit_kind="tilelang_dsl",
    )
    assert tb_spec.task_key == cpp_key
    assert tb_spec.task_key != dsl_key


@pytest.mark.parametrize(
    ("raw_prof", "expected_metrics", "absent_keys"),
    [
        (
            {
                "aic_mac_ratio": "42.5",
                "aiv_vec_ratio": "0.65",
                "aiv_mte2": 80.0,
                "aic_mte2": 0.4,
                "dur_ns": "25000",
            },
            {
                "aic_mac_ratio": 0.425,
                "aic_mad": 0.425,
                "aiv_vec_ratio": 0.65,
                "aiv_vec": 0.65,
                "aiv_mte2": 0.8,
                "aic_mte2": 0.4,
                "mte2_wait_ratio": 0.8,
                "dur_us": 25.0,
            },
            (),
        ),
        (
            {
                "aic_mac_ratio": "N/A",
                "aiv_vec_ratio": "nan",
                "aiv_mte2": float("inf"),
                "aic_mte2": True,
                "l2_hit_rate": None,
                "dur_us": "invalid",
            },
            {},
            (
                "aic_mac_ratio",
                "aic_mad",
                "aiv_vec_ratio",
                "aiv_vec",
                "aiv_mte2",
                "aic_mte2",
                "mte2_wait_ratio",
                "l2_hit_rate",
                "dur_us",
            ),
        ),
    ],
    ids=["accept_percent_and_unit_ratio", "reject_malformed_and_missing_cells"],
)
def test_surrogate_batch_harness_msprof_detail_metrics_reject_and_accept(
    raw_prof: dict[str, object],
    expected_metrics: dict[str, float],
    absent_keys: tuple[str, ...],
) -> None:
    """Parse percent and unit ratios cleanly without UnboundLocalError and skip malformed cells."""
    repo_root = Path(__file__).resolve().parents[4]
    harness_path = repo_root / "tests" / "ascend" / "agent" / "surrogate_batch_harness.py"
    spec = importlib.util.spec_from_file_location("surrogate_batch_harness_msp_mod", harness_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    class _StubMsprofBench:
        def __call__(
            self,
            fn_call: object,
            *,
            _n_warmup: int,
            _n_repeat: int,
            backend: str,
            device: object,
        ) -> object:
            assert callable(fn_call)
            assert backend == "msprof_detail"
            assert _n_warmup >= 1 and _n_repeat >= 1
            assert device == "cpu"
            return raw_prof

    parsed = mod.measure_msprof(lambda: None, bench_fn=_StubMsprofBench(), device="cpu")
    assert "error" not in parsed
    for key, expected_val in expected_metrics.items():
        assert parsed[key] == pytest.approx(expected_val)
    for key in absent_keys:
        assert key not in parsed
    assert mod.measure_msprof(None) == {"error": "msprof_detail_failed: no_callable"}
