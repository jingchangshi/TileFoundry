"""Unit tests for closure-addressed ArtifactStore, parallel L0/L1, and device_fault handling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.orchestrator import gate_funnel
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    compile_trackb_shared_so,
    compute_trackb_shared_so_filename,
    evaluate_ascendc_candidate,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.ascend.compile_cache import (
    ArtifactStore,
    compute_artifact_id,
    extract_reachable_closure_source,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger


def _sample_tilelang_multi_regime(
    *,
    small_body: str = "    return {'tag': 'small', 'val': 1}",
    unaligned_body: str = "    return {'tag': 'unaligned', 'val': 1}",
) -> str:
    return (
        "import tilelang\n\n"
        "def _shared_helper(x):\n"
        "    return x + 1\n\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n\n"
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"{small_body}\n\n"
        "def _select_ar_unaligned_config(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=4):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n\n"
        "def _build_ar_unaligned_func(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"{unaligned_body}\n\n"
        "def _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[4], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
    )


def _sample_trackb_multi_regime(
    *,
    decode_val: str = "1.0f",
    combine_val: str = "2.0f",
) -> str:
    return (
        "import tilelang\n\n"
        "def _select_mla_decode_config(B=1, S_kv=1024, tl_dtype='bfloat16'):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n\n"
        "def _select_mla_split_kv_combine_config(B=1, S_kv=1024, tl_dtype='bfloat16'):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n\n"
        '_MLA_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ inline float helper_decode(float v) { return v + "
        f"{decode_val}; }}\n"
        "__aicore__ inline float helper_combine(float v) { return v + "
        f"{combine_val}; }}\n"
        'extern "C" __global__ __vector__ void mla_kv_reuse_decode(__gm__ float* q) {\n'
        "    q[0] = helper_decode(q[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void mla_fast_combine(__gm__ float* out) {\n'
        "    out[0] = helper_combine(out[0]);\n"
        "}\n"
        '"""\n\n'
        "def _render_trackb_tu(regime_tag='', **_dims):\n"
        "    return _MLA_TRACKB_SRC\n"
    )


def test_closure_extraction_isolates_sibling_regimes_for_tilelang_and_trackb() -> None:
    """Mutating a sibling regime leaves untouched regime closure and ArtifactID identical."""
    tl_base = _sample_tilelang_multi_regime()
    tl_mut = _sample_tilelang_multi_regime(
        unaligned_body="    return {'tag': 'unaligned', 'val': 999}"
    )

    c_small_base = extract_reachable_closure_source(tl_base, regime_tag="small_decode")
    c_small_mut = extract_reachable_closure_source(tl_mut, regime_tag="small_decode")
    assert c_small_base == c_small_mut
    assert "_build_ar_unaligned_func" not in c_small_base
    assert "_build_ar_small_decode_func" in c_small_base

    aid_base = compute_artifact_id(closure_src=c_small_base, flags="-O2")
    aid_mut = compute_artifact_id(closure_src=c_small_mut, flags="-O2")
    assert aid_base == aid_mut

    c_unaligned_base = extract_reachable_closure_source(tl_base, regime_tag="unaligned_d")
    c_unaligned_mut = extract_reachable_closure_source(tl_mut, regime_tag="unaligned_d")
    assert c_unaligned_base != c_unaligned_mut
    assert compute_artifact_id(closure_src=c_unaligned_base) != compute_artifact_id(
        closure_src=c_unaligned_mut
    )

    tb_base = _sample_trackb_multi_regime()
    tb_mut = _sample_trackb_multi_regime(combine_val="99.0f")

    tb_dec_base = extract_reachable_closure_source(tb_base, regime_tag="decode")
    tb_dec_mut = extract_reachable_closure_source(tb_mut, regime_tag="decode")
    assert tb_dec_base == tb_dec_mut
    assert "mla_kv_reuse_decode" in tb_dec_base
    assert "mla_fast_combine" not in tb_dec_base
    assert "helper_combine" not in tb_dec_base
    assert compute_artifact_id(closure_src=tb_dec_base) == compute_artifact_id(
        closure_src=tb_dec_mut
    )

    tb_comb_base = extract_reachable_closure_source(tb_base, regime_tag="split_kv_combine")
    tb_comb_mut = extract_reachable_closure_source(tb_mut, regime_tag="split_kv_combine")
    assert tb_comb_base != tb_comb_mut
    assert "mla_fast_combine" in tb_comb_base
    assert "mla_kv_reuse_decode" not in tb_comb_base


def test_artifact_store_verifies_real_so_and_o_sizes(tmp_path: Path) -> None:
    """ArtifactStore checks actual .so/.o existence and st_size > 0 on disk."""
    store = ArtifactStore(tmp_path / "store")
    so_file = tmp_path / "kernel.so"
    obj_file = tmp_path / "kernel.o"
    so_file.write_bytes(b"ELF_SO_PAYLOAD")
    obj_file.write_bytes(b"ELF_OBJ_PAYLOAD")

    aid = compute_artifact_id(closure_src="void k() {}", flags="-O2")
    store.record_verified_artifact(
        aid,
        binary_paths=[so_file, obj_file],
        backend_kind="ascendc_trackb",
        flags="-O2",
    )
    assert store.has_verified_artifact(aid) is True
    rec = store.get_verified_artifact(aid)
    assert rec is not None
    assert len(rec.binary_paths) == 2

    obj_file.write_bytes(b"")
    assert store.has_verified_artifact(aid) is False

    obj_file.write_bytes(b"RESTORED")
    assert store.has_verified_artifact(aid) is True

    so_file.unlink()
    assert store.has_verified_artifact(aid) is False

    empty_aid = compute_artifact_id(closure_src="void empty() {}")
    store.record_verified_artifact(empty_aid, binary_paths=[])
    assert store.has_verified_artifact(empty_aid) is False


def _setup_prewarm_stub_env(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
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
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_stub_dir = tmp_path / "tl_stub"
    tl_stub_dir.mkdir(parents=True, exist_ok=True)
    (tl_stub_dir / "tilelang.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    so_path = cache_dir / f\"k_{prog.get('tag', 'ok')}_{prog.get('val', 0)}.so\"\n"
        "    so_path.write_bytes(b'ELF_STUB')\n"
        "    return {'cached': prog}\n"
        "def compile_ascend(src, target_format='so', npu_arch='dav-3510', options=None, path_target=None):\n"
        "    target = Path(path_target)\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_bytes(b'ELF_TRACKB_SO')\n"
        "    return str(target)\n",
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


def test_prewarm_closure_addressed_cache_hits_on_sibling_edit_for_tilelang_and_trackb(
    tmp_path: Path,
) -> None:
    """Closure-addressed prewarm achieves cache_hits > 0 when sibling regime changes."""
    assert not hasattr(surrogate_cannbench, "_tl_cache_has_artifacts")
    fake_root, cann_repo, _tl_stub, work_root, env_file = _setup_prewarm_stub_env(tmp_path)
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
    )

    inc_tl = tmp_path / "inc_tl.py"
    inc_tl.write_text(_sample_tilelang_multi_regime(), encoding="utf-8")
    mut_tl = tmp_path / "mut_tl.py"
    mut_tl.write_text(
        _sample_tilelang_multi_regime(unaligned_body="    return {'tag': 'unaligned', 'val': 77}"),
        encoding="utf-8",
    )

    ref_cache = work_root / "surrogate" / "reference"
    tl_cache = work_root / "surrogate" / "tilelang"
    r1 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_slot", inc_tl)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert r1.prewarm_stats["compiled_tasks"] == 2
    assert r1.prewarm_stats["cache_hits"] == 0

    r2 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_slot", inc_tl), ("c_mut", mut_tl)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert r2.prewarm_stats["cache_hits"] == 3
    assert r2.prewarm_stats["compiled_tasks"] == 1

    tb_cache = work_root / "surrogate" / "trackb_tilelang"
    tb_inc = tmp_path / "tb_inc.py"
    tb_inc.write_text(
        "import tilelang\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'compile_flags': ['-O2']}\n"
        "def _select_ar_unaligned_config(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=4):\n"
        "    return {'tile_m': 1, 'compile_flags': ['-O2']}\n"
        '_AR_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void ar_small_decode_kernel(__gm__ float* x) {\n'
        "    x[0] = 1.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void ar_unaligned_kernel(__gm__ float* x) {\n'
        "    x[0] = 2.0f;\n"
        "}\n"
        '"""\n'
        "def _render_trackb_tu(regime_tag='', **_dims):\n"
        "    return _AR_TRACKB_SRC\n",
        encoding="utf-8",
    )
    tb_mut = tmp_path / "tb_mut.py"
    tb_mut.write_text(
        tb_inc.read_text(encoding="utf-8").replace("x[0] = 2.0f;", "x[0] = 99.0f;"),
        encoding="utf-8",
    )

    tb_r1 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_slot", tb_inc)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tb_cache,
        source_root=fake_root,
    )
    assert tb_r1.prewarm_stats["compiled_tasks"] == 2

    tb_r2 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("incumbent_slot", tb_inc), ("c_tb_mut", tb_mut)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tb_cache,
        source_root=fake_root,
    )
    assert tb_r2.prewarm_stats["cache_hits"] == 3
    assert tb_r2.prewarm_stats["compiled_tasks"] == 1


def test_gate_funnel_fault_fields_and_stage_b_artifact_store(tmp_path: Path) -> None:
    """Stage A and Stage B attach fault fields and Stage B persists artifacts to store."""
    sa_empty = evaluate_ascendc_candidate("")
    assert sa_empty.passed is False
    assert sa_empty.fault_plane == "STAGE_A_STATIC"
    assert sa_empty.fault_code == "ASCENDC_EMPTY_SOURCE"
    assert sa_empty.failed_tier == "stage_a"

    sa_syntax = run_stage_a_static("def broken(:\n")
    assert sa_syntax.passed is False
    assert sa_syntax.fault_plane == "STAGE_A_STATIC"
    assert sa_syntax.fault_code == "STAGE_A_PARSE_OR_LOWER_ERROR"
    assert sa_syntax.failed_tier == "stage_a"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bisheng = fake_bin / "bisheng"
    fake_bisheng.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if '-o' in args:\n"
        "    out = Path(args[args.index('-o') + 1])\n"
        "    out.write_bytes(b'ELF_OBJ_' * 512)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_bisheng.chmod(0o755)
    fake_nm = fake_bin / "llvm-nm"
    fake_nm.write_text(
        "#!/usr/bin/env python3\n"
        "print('0000000000000000 T fake_ascendc_kernel')\n",
        encoding="utf-8",
    )
    fake_nm.chmod(0o755)

    store = ArtifactStore(tmp_path / "artifact_store")
    sb_ok = run_stage_b_ascendc_compile(
        'extern "C" __global__ __vector__ void fake_ascendc_kernel() {}\n',
        bisheng_bin=fake_bisheng,
        artifact_store=store,
        spec_key="op:case1",
    )
    assert sb_ok.passed is True
    assert sb_ok.artifact_id
    assert len(sb_ok.binary_paths) == 1
    assert store.has_verified_artifact(sb_ok.artifact_id) is True

    sb_missing = run_stage_b_ascendc_compile("", bisheng_bin=fake_bisheng)
    assert sb_missing.passed is False
    assert sb_missing.fault_plane == "STAGE_B_COMPILE"
    assert sb_missing.fault_code == "STAGE_B_ASCENDC_EMPTY_SOURCE"
    assert sb_missing.failed_tier == "stage_b"


def test_parallel_l0_l1_round_filter_repairs_concurrently_and_skips_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_l0_l1_round_filter runs repairs concurrently and never repairs NOOP_IDENTICAL."""
    assert not hasattr(surrogate_cannbench, "_compact_trackb_repair_source")
    assert not hasattr(surrogate_cannbench, "_resplice_trackb_omitted_spans")
    assert not hasattr(surrogate_cannbench, "_ANY_FOLDING_MARKER_RE")

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={},
    )
    repair_calls: list[str] = []

    monkeypatch.setattr(
        surrogate_cannbench,
        "build_syntax_repair_request",
        lambda cur_src, diag, **_kw: SimpleNamespace(
            role="syntax_repair",
            user_prompt=f"{cur_src}\n{diag}",
        ),
    )

    class _ParallelRepairBackend:
        async def generate(self, req: object) -> str:
            repair_calls.append(str(getattr(req, "user_prompt", "")))
            await asyncio.sleep(0.18)
            return "def ok_fn():\n    return 'repaired'\n"

    active_l1 = 0
    max_active_l1 = 0
    lock = threading.Lock()

    def _l0(src: str) -> StageAResult:
        if "noop_stage_a" in src:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message="noop_identical",
                fault_plane="NOOP_IDENTICAL",
                fault_code="NOOP_IDENTICAL",
                failed_tier="noop_identical",
            )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _l1(src: str) -> StageBResult:
        nonlocal active_l1, max_active_l1
        with lock:
            active_l1 += 1
            max_active_l1 = max(max_active_l1, active_l1)
        try:
            time.sleep(0.04)
            if "repaired" in src:
                return StageBResult(passed=True)
            return StageBResult(
                passed=False,
                error_message="broken_symbol",
                fault_plane="STAGE_B_COMPILE",
                fault_code="STAGE_B_COMPILE_FAILED",
                failed_tier="stage_b",
            )
        finally:
            with lock:
                active_l1 -= 1

    parent_src = "def parent_fn():\n    return 'parent'\n"
    candidates = [
        ("c1", "def f1():\n    return 'bad1'\n"),
        ("c2", "def f2():\n    return 'bad2'\n"),
        ("c3", "def f3():\n    return 'bad3'\n"),
        ("c_noop_ast", parent_src),
        ("c_noop_sa", "def f_noop():\n    return 'noop_stage_a'\n"),
    ]
    results = asyncio.run(
        surrogate_cannbench.run_l0_l1_round_filter(
            candidates,
            patch_fn=lambda snip: snip,
            l0_checker=_l0,
            l1_compiler=_l1,
            repair_backend=_ParallelRepairBackend(),
            max_syntax_repairs=1,
            parent_source=parent_src,
            backend_kind="ascendc_trackb",
        )
    )
    assert len(repair_calls) == 3
    assert results[0].passed is True and results[0].repaired is True
    assert results[1].passed is True and results[1].repaired is True
    assert results[2].passed is True and results[2].repaired is True
    assert results[3].passed is False and results[3].failed_tier == "noop_identical"
    assert results[3].fault_plane == "NOOP_IDENTICAL"
    assert results[4].passed is False and results[4].failed_tier == "noop_identical"
    assert results[4].fault_plane == "NOOP_IDENTICAL"

    assert results.cum_repair_worker_sec >= 0.45
    assert results.wall_repair_sec < results.cum_repair_worker_sec
    assert results[0].wall_repair_sec == pytest.approx(results.wall_repair_sec)
    assert results[0].cum_repair_worker_sec == pytest.approx(results.cum_repair_worker_sec)
    assert max_active_l1 <= min(4, os.cpu_count() or 2)


def test_adaptive_poison_pill_timeout_and_boot_handshake(tmp_path: Path) -> None:
    """Adaptive poison-pill timeout stays in [10, 60] and boot handshake shields cold boot."""
    assert surrogate_cannbench.compute_adaptive_poison_pill_timeout([]) == pytest.approx(10.0)
    assert surrogate_cannbench.compute_adaptive_poison_pill_timeout([20.0]) == pytest.approx(10.0)
    assert surrogate_cannbench.compute_adaptive_poison_pill_timeout(
        [500_000.0, 500_000.0]
    ) == pytest.approx(25.0)
    assert surrogate_cannbench.compute_adaptive_poison_pill_timeout(
        [5_000_000.0]
    ) == pytest.approx(60.0)

    boot_ok_script = tmp_path / "boot_ok.py"
    boot_ok_script.write_text(
        "import os, time\n"
        "time.sleep(0.5)\n"
        "fd = int(os.environ['TF_POISON_PILL_READY_FD'])\n"
        "os.write(fd, b'READY\\n')\n"
        "os.close(fd)\n"
        "time.sleep(0.05)\n",
        encoding="utf-8",
    )
    ok, err, rc, _out, _err = surrogate_cannbench.run_poison_pill_subprocess(
        [sys.executable, str(boot_ok_script)],
        cwd=tmp_path,
        exec_timeout_sec=0.3,
        boot_timeout_sec=5.0,
    )
    assert ok is True, err
    assert rc == 0

    hang_after_boot = tmp_path / "hang_after_boot.py"
    hang_after_boot.write_text(
        "import os, time\n"
        "fd = int(os.environ['TF_POISON_PILL_READY_FD'])\n"
        "os.write(fd, b'READY\\n')\n"
        "os.close(fd)\n"
        "time.sleep(5.0)\n",
        encoding="utf-8",
    )
    ok_hang, err_hang, _rc_hang, _out, _err = surrogate_cannbench.run_poison_pill_subprocess(
        [sys.executable, str(hang_after_boot)],
        cwd=tmp_path,
        exec_timeout_sec=0.2,
        boot_timeout_sec=5.0,
    )
    assert ok_hang is False
    assert "poison_pill_timeout_or_crash: timeout" in err_hang


def test_device_fault_receipt_rotates_healthy_card_or_raises_structured_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device_fault or deviceRejected marks card rejected, rotates, or raises INFRA_DEVICE_FAULT."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"

    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): tuple(
            c for c in ("0", "1") if c not in excluded
        ),
    )
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)

    seen_cards: list[str] = []

    def _rotate_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        seen_cards.append(dev)
        if dev == "0":
            return surrogate_cannbench._V2GateExecution(
                execution_id="exec-0",
                run_id="run-0",
                report={
                    "status": "BLOCKED_ENV",
                    "gates": [
                        {
                            "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                            "status": "BLOCKED_ENV",
                            "observations": {"errorCategory": "device_fault"},
                        }
                    ],
                },
                verification={"satisfied": False},
                device_id="0",
            )
        resp_path.write_text(
            json.dumps(
                {
                    "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                    "request_id": "req-df",
                    "request_sha256": req_sha,
                    "mode": "batch",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-1",
            run_id="run-1",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
            device_id="1",
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _rotate_v2)
    raw, final_exec = surrogate_cannbench._submit_fast_gate(
        agent_gates_bin="agent-gates",
        root=tmp_path,
        workspace="tilefoundry",
        request_id="req-df",
        request_relative_path="req.json",
        request_path=req_path,
        response_path=resp_path,
        request_sha256=req_sha,
        mode="batch",
        timeout_sec=30.0,
        device_id="0",
        device_pool=("0", "1"),
    )
    assert seen_cards == ["0", "1"]
    assert final_exec.device_id == "1"
    assert raw["status"] == "ok"

    def _all_fault_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        return surrogate_cannbench._V2GateExecution(
            execution_id=f"exec-{dev}",
            run_id=f"run-{dev}",
            report={
                "status": "FAIL",
                "gates": [
                    {
                        "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                        "status": "FAIL",
                        "observations": {
                            "errorCategory": "device_fault",
                            "deviceRejected": True,
                        },
                    }
                ],
            },
            verification={"satisfied": False, "deviceRejected": True},
            device_id=dev,
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _all_fault_v2)
    with pytest.raises(surrogate_cannbench.AgentGatesDeviceFaultError) as exc_info:
        surrogate_cannbench._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-df-fail",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=30.0,
            device_id="0",
            device_pool=("0", "1"),
        )
    assert exc_info.value.fault_plane == "INFRA_DEVICE_FAULT"
    assert exc_info.value.fault_code == "NPU_DEVICE_TSD_FAULT"
    assert exc_info.value.failed_tier == "l2_npu"
    assert exc_info.value.rejected_cards == ("0", "1")

    fake_proc = subprocess.CompletedProcess(
        args=["agent-gates"],
        returncode=1,
        stdout="",
        stderr="",
    )
    assert (
        surrogate_cannbench._is_transient_blocked_env(
            fake_proc,
            {"status": "BLOCKED_ENV", "errorCategory": "device_fault"},
        )
        is False
    )


def _sample_hybrid_mla_module(*, trackb_val: str = "1.0f") -> str:
    return (
        "import tilelang\n"
        "import tilelang.language as T\n\n"
        "def _select_mla_split_kv_decode_config(B=1, S_kv=1024, tl_dtype='bfloat16'):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n\n"
        "def _build_mla_split_kv_decode_func(B=1, S_kv=1024, tl_dtype='bfloat16'):\n"
        "    @T.prim_func\n"
        "    def mla_split_kv_decode_kernel(q: T.Tensor((1, 64), 'float32')):\n"
        "        T.nop()\n"
        "    return mla_split_kv_decode_kernel\n\n"
        "def _select_mla_c1c1c2_config(B=1, S_kv=1024, tl_dtype='bfloat16'):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n\n"
        '_MLA_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void mla_kv_reuse_c1c1c2(__gm__ float* q) {\n'
        f"    q[0] = {trackb_val};\n"
        "}\n"
        '"""\n\n'
        "def _render_trackb_tu(regime_tag='', **_dims):\n"
        "    return _MLA_TRACKB_SRC\n\n"
        "def _get_kernel(builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    return builder_fn()\n"
    )


def test_hybrid_closure_excludes_trackb_src_when_prim_func_builder_matches() -> None:
    """Hybrid TileLang regime with @T.prim_func builder ignores sibling _MLA_TRACKB_SRC."""
    base_src = _sample_hybrid_mla_module(trackb_val="1.0f")
    mut_src = _sample_hybrid_mla_module(trackb_val="99.0f")

    c_base = extract_reachable_closure_source(base_src, regime_tag="split_kv_decode")
    c_mut = extract_reachable_closure_source(mut_src, regime_tag="split_kv_decode")
    assert c_base == c_mut
    assert "_build_mla_split_kv_decode_func" in c_base
    assert "_MLA_TRACKB_SRC" not in c_base
    assert "mla_kv_reuse_c1c1c2" not in c_base

    tb_base = extract_reachable_closure_source(base_src, regime_tag="c1c1c2")
    tb_mut = extract_reachable_closure_source(mut_src, regime_tag="c1c1c2")
    assert tb_base != tb_mut
    assert "_MLA_TRACKB_SRC" in tb_base
    assert "mla_kv_reuse_c1c1c2" in tb_base


def test_trackb_shared_so_cache_helpers_and_stage_b_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared .so cache ignores -I flags, short-circuits hits, and integrates with Stage B."""
    tu = (
        '#include "kernel_operator.h"\n'
        'extern "C" __global__ __vector__ void k(__gm__ float* x) { x[0] = 1.0f; }\n'
        'extern "C" int launch_k(void* s, float* x) { k<<<1, nullptr, s>>>(x); return 0; }\n'
    )
    f1 = compute_trackb_shared_so_filename(tu, ["-O2", "-I/tmp/a", "-DFOO=1"])
    f2 = compute_trackb_shared_so_filename(tu, ["-O2", "-I/tmp/b", "-DFOO=1"])
    f3 = compute_trackb_shared_so_filename(tu, ["-O2", "-DFOO=2"])
    assert f1 == f2
    assert f1 != f3
    assert f1.startswith("trackb_") and f1.endswith(".so") and len(f1) == len("trackb_") + 19

    cann_root = tmp_path / "cann"
    (cann_root / "include").mkdir(parents=True)
    bisheng = tmp_path / "bin" / "bisheng"
    bisheng.parent.mkdir(parents=True)
    bisheng.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bisheng.chmod(0o755)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))

    cache_dir = tmp_path / "shared_so_cache"
    target_so = cache_dir / f1
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_so.write_bytes(b"\x7fELF_CACHED_SO")

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("bisheng should not be invoked when cached .so already exists")

    monkeypatch.setattr(gate_funnel.subprocess, "run", _fail_if_called)
    hit_ok, hit_path, hit_err = compile_trackb_shared_so(
        tu,
        cache_dir=cache_dir,
        compile_flags=["-O2", "-I/tmp/c", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
    )
    assert hit_ok is True
    assert hit_path == target_so.resolve()
    assert hit_err == ""

    target_so.unlink()
    seen_cmds: list[list[str]] = []

    def _mock_bisheng_run(cmd, **_kw):
        cmd_list = [str(x) for x in cmd]
        seen_cmds.append(cmd_list)
        if "-o" in cmd_list:
            out = Path(cmd_list[cmd_list.index("-o") + 1])
            out.write_bytes(b"\x7fELF_BUILT" * 16)
        return subprocess.CompletedProcess(cmd_list, 0, stdout="", stderr="")

    monkeypatch.setattr(gate_funnel.subprocess, "run", _mock_bisheng_run)
    monkeypatch.setattr(
        gate_funnel, "_verify_object_has_instantiated_kernel", lambda *_a, **_kw: (True, "")
    )
    monkeypatch.setattr(gate_funnel, "_disassemble_object", lambda *_a, **_kw: "")

    store = ArtifactStore(tmp_path / "store")
    sb1 = run_stage_b_ascendc_compile(
        tu,
        compile_flags=["-O2", "-I/tmp/x", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
        verify_shared_link=True,
        emit_shared_so_cache_dir=cache_dir,
        artifact_store=store,
    )
    assert sb1.passed is True
    assert len(sb1.binary_paths) == 2
    assert any("-shared" in c for c in seen_cmds)
    assert any("-c" in c for c in seen_cmds)
    assert target_so.is_file() and target_so.stat().st_size > 0

    seen_cmds.clear()
    direct_cache_dir = tmp_path / "direct_shared_so_cache"
    miss_ok, miss_path, miss_err = compile_trackb_shared_so(
        tu,
        cache_dir=direct_cache_dir,
        compile_flags=["-O2", "-I/tmp/z", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
    )
    assert miss_ok is True
    assert miss_err == ""
    assert miss_path == (direct_cache_dir / f1).resolve()
    assert miss_path.is_file() and miss_path.stat().st_size > 0
    assert len(seen_cmds) == 1 and "-shared" in seen_cmds[0]

    seen_cmds.clear()
    sb2 = run_stage_b_ascendc_compile(
        tu,
        compile_flags=["-O2", "-I/tmp/y", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
        verify_shared_link=True,
        emit_shared_so_cache_dir=cache_dir,
        artifact_store=store,
    )
    assert sb2.passed is True
    assert len(sb2.binary_paths) == 2
    assert any("-c" in c for c in seen_cmds)
    assert not any("-shared" in c for c in seen_cmds)

