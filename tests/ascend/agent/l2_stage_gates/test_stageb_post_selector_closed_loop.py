"""Tests for Stage-B post-selector closed loop, manifest validation, and Track-B caching."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tilefoundry_ascend.agent.contracts import CasePlanExtractor, CompileTaskSpec
from tilefoundry_ascend.agent.gates import gate_funnel as gf
from tilefoundry_ascend.agent.gates.stage_b_gate import (
    ModuleSelectorPlanExtractor,
    StageBCompileBackend,
    StageBCompileGate,
    build_post_selector_regime_plan,
    compile_trackb_shared_so,
    compute_compile_task_key,
    compute_trackb_shared_so_filename,
    has_unreserved_vf_stack_overflow,
    parse_cce_res_usage,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
    validate_prewarm_manifest_v1,
)
from tilefoundry_ascend.ascend.compile_cache import (
    ArtifactStore,
    compute_artifact_id,
    extract_reachable_closure_source,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

_SYNTHETIC_SELECTOR_MODULE = '''
import tilelang


def _select_synth_config(m_rows: int, d_dim: int, tl_dtype: str = "bfloat16"):
    tile_m = 1 if int(m_rows) < 32 else 8
    num_stages = 1 if int(m_rows) < 32 else 2
    return {
        "tile_m": tile_m,
        "num_stages": num_stages,
        "pass_configs": {"tl.ascend_auto_sync": True},
        "compile_flags": ["--cce-disable-asc-reserved-ubuf"],
    }


def _build_synth_prim_func(
    d_dim: int,
    tl_dtype: str = "bfloat16",
    tile_m: int = 1,
    num_stages: int = 1,
    m_rows: int = 0,
):
    return (int(m_rows), int(d_dim), tl_dtype, int(tile_m), int(num_stages))


def _get_kernel(
    d_dim: int,
    tl_dtype: str = "bfloat16",
    builder_fn=None,
    pass_configs=None,
    compile_flags=None,
    extra_builder_kwargs=None,
):
    kw = dict(extra_builder_kwargs or {})
    prog = builder_fn(
        d_dim=d_dim,
        tl_dtype=tl_dtype,
        tile_m=kw.get("tile_m", 1),
        num_stages=kw.get("num_stages", 1),
    )
    return tilelang.compile(
        prog,
        target="ascend",
        pass_configs=pass_configs or {},
        compile_flags=list(compile_flags or ()),
    )
'''

_SYNTHETIC_SPLIT_KV_MODULE = '''
import tilelang


def _select_split_config(s_kv: int = 1024, d_dim: int = 128, tl_dtype: str = "float16"):
    return {
        "splits": 4 if int(s_kv) >= 1024 else 1,
        "pass_configs": {},
        "compile_flags": ["--cce-disable-asc-reserved-ubuf"],
    }


def _build_stage1_func(d_dim: int = 128, splits: int = 4, tl_dtype: str = "float16"):
    return ("stage1", int(d_dim), int(splits), tl_dtype)


def _build_combine_func(d_dim: int = 128, splits: int = 4, tl_dtype: str = "float16"):
    return ("combine", int(d_dim), int(splits), tl_dtype)


def synth_split(q, k):
    cfg = _select_split_config(s_kv=k.shape[1], d_dim=q.shape[-1])
    k1 = tilelang.compile(
        _build_stage1_func(d_dim=q.shape[-1], splits=cfg["splits"]),
        target="ascend",
        pass_configs={},
        compile_flags=["--cce-disable-asc-reserved-ubuf"],
    )
    if cfg["splits"] > 1:
        k2 = tilelang.compile(
            _build_combine_func(d_dim=q.shape[-1], splits=cfg["splits"]),
            target="ascend",
            pass_configs={},
            compile_flags=[
                "--cce-disable-vf-stack-reserved-ubuf",
                "--cce-disable-asc-reserved-ubuf",
            ],
        )
        return k1, k2
    return k1
'''


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


def test_bug_sb1_post_selector_preserves_distinct_specs_for_shared_d_synthetic() -> None:
    """BUG-SB1: same D=2048 with m_rows=16 and m_rows=4096 keeps both specs in any order."""
    extractor = ModuleSelectorPlanExtractor(
        target="dav-3510",
        builder_name="_build_synth_prim_func",
        config_selector_name="_select_synth_config",
    )
    cases_fwd = {
        "c0_small": {"m_rows": 16, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "c1_large": {"m_rows": 4096, "d_dim": 2048, "tl_dtype": "bfloat16"},
    }
    cases_rev = {
        "c1_large": {"m_rows": 4096, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "c0_small": {"m_rows": 16, "d_dim": 2048, "tl_dtype": "bfloat16"},
    }
    plan_fwd = build_post_selector_regime_plan(
        _SYNTHETIC_SELECTOR_MODULE, "synth", cases_fwd, extractor
    )
    plan_rev = build_post_selector_regime_plan(
        _SYNTHETIC_SELECTOR_MODULE, "synth", cases_rev, extractor
    )
    for plan in (plan_fwd, plan_rev):
        assert len(plan.tasks_by_case_id) == 2
        assert len(plan.unique_specs_by_key) == 2
        spec_small = plan.tasks_by_case_id["c0_small"][0]
        spec_large = plan.tasks_by_case_id["c1_large"][0]
        assert spec_small.task_key != spec_large.task_key
        assert dict(spec_small.static_kwargs)["tile_m"] == 1
        assert dict(spec_large.static_kwargs)["tile_m"] == 8
    assert set(plan_fwd.unique_specs_by_key) == set(plan_rev.unique_specs_by_key)


def test_bug_sb1_real_arnq_and_ar_modules_retain_small_and_large_m_rows() -> None:
    """BUG-SB1: real arnq and attention_residual kernels keep m_rows=16 and 4096 specs."""
    extractor = ModuleSelectorPlanExtractor(target="dav-3510")
    cases = {
        "01_small": {
            "m_rows": 16,
            "n_depth": 8,
            "b_dim": 1,
            "s_dim": 16,
            "d_dim": 2048,
            "tl_dtype": "bfloat16",
        },
        "02_large": {
            "m_rows": 4096,
            "n_depth": 8,
            "b_dim": 1,
            "s_dim": 4096,
            "d_dim": 2048,
            "tl_dtype": "bfloat16",
        },
    }

    arnq_src = (
        _REPO_ROOT / "examples/ascend950pr_ops/add_rms_norm_dynamic_quant/kernels.py"
    ).read_text(encoding="utf-8")
    arnq_plan = build_post_selector_regime_plan(arnq_src, "arnq_shared_d", cases, extractor)
    assert len(arnq_plan.unique_specs_by_key) == 2
    arnq_small = dict(arnq_plan.tasks_by_case_id["01_small"][0].static_kwargs)
    arnq_large = dict(arnq_plan.tasks_by_case_id["02_large"][0].static_kwargs)
    assert arnq_small["tile_m"] != arnq_large["tile_m"]

    ar_src = (_REPO_ROOT / "examples/ascend950pr_ops/attention_residual/kernels.py").read_text(
        encoding="utf-8"
    )
    ar_plan = build_post_selector_regime_plan(ar_src, "ar_shared_d", cases, extractor)
    assert len(ar_plan.unique_specs_by_key) == 2
    ar_small = ar_plan.tasks_by_case_id["01_small"][0]
    ar_large = ar_plan.tasks_by_case_id["02_large"][0]
    assert ar_small.builder_name == "_build_ar_small_decode_func"
    assert ar_large.builder_name == "_build_ar_aligned_func"
    assert ar_small.task_key != ar_large.task_key

    batch_cases = {
        f"c{i:02d}": {"m_rows": 16 * (i + 1), "d_dim": 2048, "tl_dtype": "bfloat16"}
        for i in range(25)
    }
    build_post_selector_regime_plan(arnq_src, "warm", batch_cases, extractor)
    t0 = time.perf_counter()
    warm_plan = build_post_selector_regime_plan(arnq_src, "warm", batch_cases, extractor)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(warm_plan.tasks_by_case_id) == 25
    assert elapsed_ms < 20.0


def test_bug_sb2_compute_compile_task_key_distinguishes_static_kwargs() -> None:
    """BUG-SB2: distinct static_spec_signature, flags, or pass_configs hash distinctly."""
    common_pc = {"tl.ascend_auto_sync": True}
    base = dict(normalized_kernel_unit_src="def build(tile_m: int = 1): return tile_m", cann_target="dav-3510", pass_configs=common_pc)
    k1 = compute_compile_task_key(**base, compile_flags=("--cce-disable-asc-reserved-ubuf",), static_spec_signature='{"builder_name":"build","static_kwargs":[["tile_m",1]]}')
    k2 = compute_compile_task_key(**base, compile_flags=("--cce-disable-asc-reserved-ubuf",), static_spec_signature='{"builder_name":"build","static_kwargs":[["tile_m",8]]}')
    k_flag = compute_compile_task_key(**base, compile_flags=("--cce-disable-vf-stack-reserved-ubuf",), static_spec_signature='{"builder_name":"build","static_kwargs":[["tile_m",1]]}')
    assert k1 != k2 and k1 != k_flag
    ext = ModuleSelectorPlanExtractor(target="dav-3510", builder_name="_build_synth_prim_func", config_selector_name="_select_synth_config")
    p1 = build_post_selector_regime_plan(_SYNTHETIC_SELECTOR_MODULE, "synth", {"c0": {"m_rows": 16, "d_dim": 2048, "tl_dtype": "bfloat16"}}, ext)
    p2 = build_post_selector_regime_plan(_SYNTHETIC_SELECTOR_MODULE, "synth", {"c0": {"m_rows": 64, "d_dim": 2048, "tl_dtype": "bfloat16"}}, ext)
    assert next(iter(p1.unique_specs_by_key)) != next(iter(p2.unique_specs_by_key))


def test_bug_sb3_has_unreserved_vf_stack_overflow_predicate_and_gate(tmp_path: Path) -> None:
    """BUG-SB3: reserved UB stack>0 passes unless diagnostic overflows; unreserved fails."""
    reserved_spec = CompileTaskSpec(
        task_key="k_reserved",
        builder_name="_build_ar_aligned_func",
        static_kwargs=(("tile_m", 8),),
        pass_configs=(),
        compile_flags=("--cce-disable-asc-reserved-ubuf",),
    )
    unreserved_spec = CompileTaskSpec(
        task_key="k_unreserved",
        builder_name="_build_arnq",
        static_kwargs=(("tile_m", 8),),
        pass_configs=(),
        compile_flags=("--cce-disable-vf-stack-reserved-ubuf", "--cce-disable-asc-reserved-ubuf"),
    )
    assert reserved_spec.disables_vf_stack_ubuf is False
    assert unreserved_spec.disables_vf_stack_ubuf is True

    assert (
        has_unreserved_vf_stack_overflow(reserved_spec, stack_bytes=256, compiler_diag="")
        is False
    )
    assert (
        has_unreserved_vf_stack_overflow(
            reserved_spec, stack_bytes=0, compiler_diag="error: Exceeded VF stack size limit"
        )
        is True
    )
    assert (
        has_unreserved_vf_stack_overflow(
            reserved_spec, stack_bytes=0, compiler_diag="fatal: stack overflow in aiv"
        )
        is True
    )
    assert (
        has_unreserved_vf_stack_overflow(unreserved_spec, stack_bytes=128, compiler_diag="")
        is True
    )
    assert (
        has_unreserved_vf_stack_overflow(unreserved_spec, stack_bytes=0, compiler_diag="")
        is False
    )
    assert gf.has_unreserved_vf_stack_overflow is has_unreserved_vf_stack_overflow
    assert not hasattr(gf, "_has_vf_stack_overflow_signal")

    class _DualSpecExtractor:
        def extract_case_tasks(
            self, _src: str, case_id: str, _case_spec: dict[str, Any]
        ) -> tuple[CompileTaskSpec, ...]:
            return (reserved_spec,) if case_id == "reserved" else (unreserved_spec,)

    extractor: CasePlanExtractor = _DualSpecExtractor()
    plan_res = build_post_selector_regime_plan("", "res", {"reserved": {}}, extractor)
    plan_unres = build_post_selector_regime_plan("", "unres", {"unreserved": {}}, extractor)

    def _compile_with_256_stack(spec: CompileTaskSpec, _src: str, work_dir: Path):
        bin_p = work_dir / f"{spec.task_key}.o"
        bin_p.write_bytes(b"\x7fELF-stub")
        return True, 256, "", [bin_p]

    store = ArtifactStore(tmp_path / "cache")
    res_receipt = StageBCompileGate(store, compile_fn=_compile_with_256_stack).run(plan=plan_res)
    assert res_receipt.passed is True
    assert res_receipt.max_stack_bytes == 256

    unres_receipt = StageBCompileGate(store, compile_fn=_compile_with_256_stack).run(
        plan=plan_unres
    )
    assert unres_receipt.passed is False
    assert unres_receipt.fault_code == "VF_STACK_OVERFLOW"


def test_bug_sb4_split_kv_stage1_plus_combine_both_checked_and_real_mla(tmp_path: Path) -> None:
    """BUG-SB4: 1-to-N subkernels (stage1 + combine) are both checked for VF stack overflow."""
    mla_src = (_REPO_ROOT / "examples/ascend950pr_ops/mla/kernels.py").read_text(encoding="utf-8")
    mla_ext = ModuleSelectorPlanExtractor(target="dav-3510")
    mla_cases = {
        "split_kv": {
            "b_dim": 1,
            "s_dim": 1,
            "s_kv": 2048,
            "n_q": 64,
            "n_kv": 1,
            "d_nope": 256,
            "d_rope": 64,
            "tl_dtype": "float16",
            "input_layout": "BSND",
            "is_causal": False,
        },
        "track_b": {
            "b_dim": 1,
            "s_dim": 1,
            "s_kv": 2048,
            "n_q": 128,
            "n_kv": 1,
            "d_nope": 512,
            "d_rope": 64,
            "tl_dtype": "float16",
            "input_layout": "BSND",
            "is_causal": False,
        },
    }
    mla_plan = build_post_selector_regime_plan(mla_src, "mla_mix", mla_cases, mla_ext)
    split_specs = mla_plan.tasks_by_case_id["split_kv"]
    assert [s.builder_name for s in split_specs] == [
        "_build_mla_split_kv_decode_func",
        "_build_mla_split_kv_combine_func",
    ]
    assert mla_plan.tasks_by_case_id["track_b"][0].builder_name == "_render_trackb_tu"

    ext = ModuleSelectorPlanExtractor(target="dav-3510")
    synth_cases = {
        "c_split": {"input_shape": [[1, 64, 128], [1, 2048, 128]], "dtype": "float16"},
    }
    plan = build_post_selector_regime_plan(
        _SYNTHETIC_SPLIT_KV_MODULE, "split_kv_gate", synth_cases, ext
    )
    specs = plan.tasks_by_case_id["c_split"]
    assert [s.builder_name for s in specs] == ["_build_stage1_func", "_build_combine_func"]
    assert len(plan.unique_specs_by_key) == 2

    store = ArtifactStore(tmp_path / "tl_cache")

    def _compile_fail_combine(spec: CompileTaskSpec, _src: str, work_dir: Path):
        bin_p = work_dir / f"{spec.builder_name}.o"
        bin_p.write_bytes(b"\x7fELF-stub")
        s_bytes = 256 if spec.builder_name == "_build_combine_func" else 64
        return True, s_bytes, "", [bin_p]

    gate = StageBCompileGate(plan, store, compile_fn=_compile_fail_combine, max_workers=2)
    receipt = gate.run(_SYNTHETIC_SPLIT_KV_MODULE)
    assert receipt.passed is False
    assert receipt.fault_code == "VF_STACK_OVERFLOW"
    assert receipt.max_stack_bytes == 256
    manifest = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    stage1_key, combine_key = specs[0].task_key, specs[1].task_key
    assert manifest["tasks"][stage1_key]["passed"] is True
    assert manifest["tasks"][combine_key]["passed"] is False
    assert manifest["tasks"][combine_key]["fault_code"] == "VF_STACK_OVERFLOW"


def test_prewarm_manifest_v1_atomic_write_validation_and_cache_hit_reuse(tmp_path: Path) -> None:
    """StageBCompileGate writes valid prewarm_manifest_v1.json and reuses cached artifacts."""
    ext = ModuleSelectorPlanExtractor(
        target="dav-3510",
        builder_name="_build_synth_prim_func",
        config_selector_name="_select_synth_config",
    )
    cases = {
        "small": {"m_rows": 16, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "large_a": {"m_rows": 4096, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "large_b": {"m_rows": 2048, "d_dim": 2048, "tl_dtype": "bfloat16"},
    }
    plan = build_post_selector_regime_plan(_SYNTHETIC_SELECTOR_MODULE, "synth", cases, ext)
    assert len(plan.unique_specs_by_key) == 2

    store = ArtifactStore(tmp_path / "tl_cache")
    compiled_keys: list[str] = []

    def _compile_ok(spec: CompileTaskSpec, _src: str, work_dir: Path):
        compiled_keys.append(spec.task_key)
        obj_p = work_dir / f"{spec.task_key[:8]}.so"
        obj_p.write_bytes(b"\x7fELF-compiled")
        return True, 128, "", [obj_p]

    gate1 = StageBCompileGate(plan, store, compile_fn=_compile_ok, max_workers=2)
    receipt1 = gate1.run(_SYNTHETIC_SELECTOR_MODULE)
    assert receipt1.passed is True
    assert receipt1.max_stack_bytes == 128
    assert len(compiled_keys) == 2

    manifest_path = Path(receipt1.manifest_path)
    assert manifest_path.name == "prewarm_manifest_v1.json"
    ok_valid, details = validate_prewarm_manifest_v1(manifest_path, plan)
    assert ok_valid is True
    assert details["total_tasks"] == 2
    assert details["cache_hits"] == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["plan_index"]["large_a"] == manifest["plan_index"]["large_b"]
    for tkey in plan.unique_specs_by_key:
        entry = manifest["tasks"][tkey]
        assert entry["status"] == "PASS"
        assert entry["stack_bytes"] == 128
        assert entry["compile_flags"] == ["--cce-disable-asc-reserved-ubuf"]
        assert Path(entry["artifact_path"]).is_file()

    compiled_keys.clear()
    gate2 = StageBCompileGate(plan, store, compile_fn=_compile_ok, max_workers=2)
    receipt2 = gate2.run(_SYNTHETIC_SELECTOR_MODULE)
    assert receipt2.passed is True
    assert compiled_keys == []

    first_key = next(iter(plan.unique_specs_by_key))
    saved_artifact = Path(manifest["tasks"][first_key]["artifact_path"])
    saved_artifact.unlink()
    ok_after_unlink, bad_details = validate_prewarm_manifest_v1(manifest_path, plan)
    assert ok_after_unlink is False
    assert first_key in bad_details["missing_tasks"]

    manifest["tasks"][first_key]["status"] = "FAIL"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ok_failed, fail_details = validate_prewarm_manifest_v1(manifest_path, plan)
    assert ok_failed is False
    assert first_key in fail_details["failed_tasks"]

    ok_missing, missing_details = validate_prewarm_manifest_v1(tmp_path / "absent.json", plan)
    assert ok_missing is False
    assert len(missing_details["missing_tasks"]) == 2


def test_closure_isolation_and_artifact_store_disk_checks(tmp_path: Path) -> None:
    """Closure extraction isolates sibling regimes and ArtifactStore verifies binary sizes."""
    tl_base = _sample_tilelang_multi_regime()
    tl_mut = _sample_tilelang_multi_regime(
        unaligned_body="    return {'tag': 'unaligned', 'val': 999}"
    )

    c_small_base = extract_reachable_closure_source(tl_base, regime_tag="small_decode")
    c_small_mut = extract_reachable_closure_source(tl_mut, regime_tag="small_decode")
    assert c_small_base == c_small_mut
    assert "_build_ar_unaligned_func" not in c_small_base
    assert "_build_ar_small_decode_func" in c_small_base
    assert compute_artifact_id(closure_src=c_small_base, flags="-O2") == compute_artifact_id(
        closure_src=c_small_mut, flags="-O2"
    )

    c_unaligned_base = extract_reachable_closure_source(tl_base, regime_tag="unaligned_d")
    c_unaligned_mut = extract_reachable_closure_source(tl_mut, regime_tag="unaligned_d")
    assert c_unaligned_base != c_unaligned_mut

    hy_base = _sample_hybrid_mla_module(trackb_val="1.0f")
    hy_mut = _sample_hybrid_mla_module(trackb_val="99.0f")
    hy_dec_base = extract_reachable_closure_source(hy_base, regime_tag="split_kv_decode")
    hy_dec_mut = extract_reachable_closure_source(hy_mut, regime_tag="split_kv_decode")
    assert hy_dec_base == hy_dec_mut
    assert "_MLA_TRACKB_SRC" not in hy_dec_base

    tb_base = extract_reachable_closure_source(hy_base, regime_tag="c1c1c2")
    tb_mut = extract_reachable_closure_source(hy_mut, regime_tag="c1c1c2")
    assert tb_base != tb_mut
    assert "mla_kv_reuse_c1c1c2" in tb_base

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
    obj_file.write_bytes(b"")
    assert store.has_verified_artifact(aid) is False
    obj_file.write_bytes(b"RESTORED")
    assert store.has_verified_artifact(aid) is True
    so_file.unlink()
    assert store.has_verified_artifact(aid) is False


def test_trackb_shared_so_cache_and_stage_b_callback_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected Stage-B callbacks verify Track-B .so cache reuse, ELF checks, and bounds."""
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
    assert f1.startswith("trackb_") and f1.endswith(".so")

    cann_root = tmp_path / "cann"
    (cann_root / "include").mkdir(parents=True)
    bisheng = tmp_path / "bin" / "bisheng"
    bisheng.parent.mkdir(parents=True)
    bisheng.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bisheng.chmod(0o755)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))

    cache_dir = tmp_path / "shared_so_cache"
    cache_dir.mkdir(parents=True)
    target_so = cache_dir / f1
    target_so.write_bytes(b"\x7fELF_CACHED_SO")

    def _forbid_subproc(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("bisheng must not execute when cached .so already exists")

    hit_ok, hit_path, hit_err = compile_trackb_shared_so(
        tu,
        cache_dir=cache_dir,
        compile_flags=["-O2", "-I/tmp/c", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
        subproc_run=_forbid_subproc,
    )
    assert hit_ok is True
    assert hit_path == target_so.resolve()
    assert hit_err == ""

    target_so.unlink()
    seen_cmds: list[list[str]] = []

    def _mock_bisheng(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        cmd_list = [str(x) for x in cmd]
        seen_cmds.append(cmd_list)
        if "-o" in cmd_list:
            Path(cmd_list[cmd_list.index("-o") + 1]).write_bytes(b"\x7fELF_BUILT" * 16)
        return subprocess.CompletedProcess(
            cmd_list,
            0,
            stdout="vloopv2_loops: 1\nsoftware_loops: 0\nStack Frame Size: 0 bytes\n",
            stderr="",
        )

    store = ArtifactStore(tmp_path / "store")
    sb1 = run_stage_b_ascendc_compile(
        tu,
        compile_flags=["-O2", "-I/tmp/x", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
        verify_shared_link=True,
        emit_shared_so_cache_dir=cache_dir,
        artifact_store=store,
        subproc_run=_mock_bisheng,
        verify_instantiated_fn=lambda *_a, **_kw: (True, ""),
        disassemble_fn=lambda *_a, **_kw: "",
    )
    assert sb1.passed is True
    assert sb1.vloopv2_hit is True
    assert len(sb1.binary_paths) == 2
    assert any("-shared" in c for c in seen_cmds)
    assert any("-c" in c for c in seen_cmds)
    assert target_so.is_file() and target_so.stat().st_size > 0

    seen_cmds.clear()
    sb2 = run_stage_b_ascendc_compile(
        tu,
        compile_flags=["-O2", "-I/tmp/y", "-DFOO=1"],
        bisheng_bin=bisheng,
        cann_root=cann_root,
        verify_shared_link=True,
        emit_shared_so_cache_dir=cache_dir,
        artifact_store=store,
        subproc_run=_mock_bisheng,
        verify_instantiated_fn=lambda *_a, **_kw: (True, ""),
        disassemble_fn=lambda *_a, **_kw: "0000: VLOOPV2\n",
    )
    assert sb2.passed is True
    assert len(sb2.binary_paths) == 2
    assert any("-c" in c for c in seen_cmds)
    assert not any("-shared" in c for c in seen_cmds)

    sb_uninst = run_stage_b_ascendc_compile(
        tu,
        bisheng_bin=bisheng,
        cann_root=cann_root,
        subproc_run=_mock_bisheng,
        verify_instantiated_fn=lambda *_a, **_kw: (
            False,
            "stage_b_ascendc_uninstantiated_template: empty .text",
        ),
        disassemble_fn=lambda *_a, **_kw: "",
    )
    assert sb_uninst.passed is False
    assert sb_uninst.fault_code == "STAGE_B_ASCENDC_UNINSTANTIATED_TEMPLATE"

    primary = "/tmp/kernel.cce:9:3: error: use of undeclared identifier 'tile_buf'"
    long_diag = f"{primary}\n" + "".join(f"Source: line {i}\n" for i in range(200))
    sb_err = run_stage_b_compile(
        "def kernel():\n    pass\n",
        artifact_store=store,
        subproc_run=lambda *_a, **_kw: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=long_diag
        ),
    )
    assert sb_err.passed is False
    assert sb_err.full_diagnostic == long_diag
    assert primary in sb_err.error_message
    assert len(sb_err.error_message) <= len("stage_b_compile_failed (rc=1): ") + 1500

    spill, vloop = parse_cce_res_usage("Register spill: 64 bytes\nVLOOPv2 miss: branch\n")
    assert spill == 64
    assert vloop is False

    with pytest.raises(ValueError, match="enable_device_compile=False"):
        run_stage_b_ascendc_compile(tu, enable_device_compile=False)
    with pytest.raises(ValueError, match="enable_device_compile=False"):
        run_stage_b_compile("def kernel():\n    pass\n", enable_device_compile=False)


def test_load_plan_module_resolves_tilelang_source_without_pythonpath(tmp_path: Path) -> None:
    """Resolve TILEFOUNDRY_NPU_TILELANG_SOURCE and sibling tilelang without machine paths."""
    import os
    import sys

    tl_env_dir = tmp_path / "tl_from_env"
    (tl_env_dir / "tilelang").mkdir(parents=True)
    (tl_env_dir / "tilelang" / "__init__.py").write_text("SOURCE_TAG = 'env'\n", encoding="utf-8")
    env_file = tmp_path / "env.sh"
    env_file.write_text(f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_env_dir}\n", encoding="utf-8")

    sib_ws = tmp_path / "ws"
    fake_root = sib_ws / "TileFoundry"
    fake_root.mkdir(parents=True)
    tl_sib_dir = sib_ws / "tilelang"
    (tl_sib_dir / "tilelang").mkdir(parents=True)
    (tl_sib_dir / "tilelang" / "__init__.py").write_text("SOURCE_TAG = 'sib'\n", encoding="utf-8")

    clean_pp = os.pathsep.join([str(_REPO_ROOT / "src"), str(_REPO_ROOT / "targets" / "ascend")])
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTHONPATH",
            "TILEFOUNDRY_NPU_ENV_FILE",
            "TILEFOUNDRY_NPU_TILELANG_SOURCE",
            "TILEFOUNDRY_NPU_TILELANG_REPO",
            "TORCH_DEVICE_BACKEND_AUTOLOAD",
        )
    }
    clean_env["PYTHONPATH"] = clean_pp

    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from tilefoundry_ascend.agent.gates.stage_b_gate import ModuleSelectorPlanExtractor, build_post_selector_regime_plan\n"
        f"src = {(_SYNTHETIC_SELECTOR_MODULE)!r}\n"
        "cases = {'c0': {'m_rows': 16, 'd_dim': 2048, 'tl_dtype': 'bfloat16'}}\n"
        "ext = ModuleSelectorPlanExtractor(target='dav-3510', builder_name='_build_synth_prim_func', config_selector_name='_select_synth_config')\n"
        f"plan = build_post_selector_regime_plan(src, 'synth', cases, ext, env_file=Path({str(env_file)!r}))\n"
        "assert len(plan.unique_specs_by_key) == 1\n"
        "assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0'\n"
        "import tilelang\n"
        "assert tilelang.SOURCE_TAG == 'env'\n"
    )
    proc_env = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_env.returncode == 0, proc_env.stderr or proc_env.stdout

    script_sib = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from tilefoundry_ascend.agent.gates.stage_b_gate import ModuleSelectorPlanExtractor, build_post_selector_regime_plan\n"
        f"src = {(_SYNTHETIC_SELECTOR_MODULE)!r}\n"
        "cases = {'c0': {'m_rows': 16, 'd_dim': 2048, 'tl_dtype': 'bfloat16'}}\n"
        "ext = ModuleSelectorPlanExtractor(target='dav-3510', builder_name='_build_synth_prim_func', config_selector_name='_select_synth_config')\n"
        f"plan = build_post_selector_regime_plan(src, 'synth', cases, ext, source_root=Path({str(fake_root)!r}))\n"
        "assert len(plan.unique_specs_by_key) == 1\n"
        "import tilelang\n"
        "assert tilelang.SOURCE_TAG == 'sib'\n"
    )
    proc_sib = subprocess.run(
        [sys.executable, "-c", script_sib],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_sib.returncode == 0, proc_sib.stderr or proc_sib.stdout


def test_stage_b_compile_gate_routes_render_trackb_tu_to_ascendc_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StageBCompileGate routes _render_trackb_tu specs to AscendC compile, never tilelang."""
    from tilefoundry_ascend.agent.gates import stage_b_gate
    from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary

    hybrid_src = _sample_hybrid_mla_module(trackb_val="2.5f")
    ext = ModuleSelectorPlanExtractor(target="dav-3510")
    cases = {"c0": {"B": 1, "S_kv": 1024, "tl_dtype": "bfloat16", "use_trackb": True}}
    plan = build_post_selector_regime_plan(hybrid_src, "mla_trackb", cases, ext)
    assert len(plan.unique_specs_by_key) == 1
    spec = next(iter(plan.unique_specs_by_key.values()))
    assert spec.builder_name == "_render_trackb_tu"

    ascendc_calls: list[tuple[str, tuple[str, ...], str]] = []

    def _fake_ascendc_compile(
        cpp_source: str,
        *,
        compile_flags: Any = None,
        artifact_store: ArtifactStore | None = None,
        artifact_id: str = "",
        **_kw: Any,
    ) -> stage_b_gate.StageBResult:
        flags_t = tuple(str(f) for f in (compile_flags or ()))
        ascendc_calls.append((cpp_source, flags_t, str(artifact_id)))
        out_dir = artifact_store.root if artifact_store is not None else tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        obj_path = out_dir / "trackb_kernel.o"
        obj_path.write_bytes(b"\x7fELF-trackb-obj")
        return stage_b_gate.StageBResult(
            passed=True,
            compiler_summary=BishengResourceSummary(stack_bytes=0),
            generated_cce=cpp_source,
            artifact_id=artifact_id or spec.task_key,
            binary_paths=(str(obj_path),),
        )

    def _forbid_tilelang_driver(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("Track-B spec must not route through build_stage_b_driver_source")

    monkeypatch.setattr(stage_b_gate, "run_stage_b_ascendc_compile", _fake_ascendc_compile)
    monkeypatch.setattr(stage_b_gate, "build_stage_b_driver_source", _forbid_tilelang_driver)

    store = ArtifactStore(tmp_path / "tl_cache")
    receipt1 = StageBCompileGate(plan, store, max_workers=1).run(hybrid_src)
    assert receipt1.passed is True
    assert receipt1.fault_code == ""
    assert len(ascendc_calls) == 1
    assert "mla_kv_reuse_c1c1c2" in ascendc_calls[0][0]
    assert "2.5f" in ascendc_calls[0][0]

    manifest_path = Path(receipt1.manifest_path)
    ok_valid, details = validate_prewarm_manifest_v1(manifest_path, plan)
    assert ok_valid is True
    assert details["cache_hits"] == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tasks"][spec.task_key]["builder_name"] == "_render_trackb_tu"
    assert manifest["tasks"][spec.task_key]["passed"] is True
    rec = store.get_verified_artifact(spec.task_key)
    assert rec is not None and rec.backend_kind == "ascendc_trackb"

    ascendc_calls.clear()
    receipt2 = StageBCompileGate(plan, store, max_workers=1).run(hybrid_src)
    assert receipt2.passed is True
    assert ascendc_calls == []

    unres_spec = CompileTaskSpec(
        task_key="manual_trackb_unreserved",
        builder_name="_render_trackb_tu",
        static_kwargs=(),
        pass_configs=(),
        compile_flags=("--cce-disable-vf-stack-reserved-ubuf",),
        unit_kind="cce_cpp20",
    )
    unres_plan = build_post_selector_regime_plan(
        hybrid_src,
        "mla_trackb_ovf",
        {"c0": {}},
        type(
            "_FixedExt",
            (),
            {"extract_case_tasks": lambda _s, _m, _cid, _cs: (unres_spec,)},
        )(),
    )

    def _fake_ascendc_overflow(
        cpp_source: str,
        *,
        artifact_store: ArtifactStore | None = None,
        artifact_id: str = "",
        **_kw: Any,
    ) -> stage_b_gate.StageBResult:
        out_dir = artifact_store.root if artifact_store is not None else tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        obj_path = out_dir / "trackb_ovf.o"
        obj_path.write_bytes(b"\x7fELF-trackb-ovf")
        return stage_b_gate.StageBResult(
            passed=True,
            compiler_summary=BishengResourceSummary(stack_bytes=128),
            generated_cce=cpp_source,
            artifact_id=artifact_id,
            binary_paths=(str(obj_path),),
        )

    monkeypatch.setattr(stage_b_gate, "run_stage_b_ascendc_compile", _fake_ascendc_overflow)
    ovf_receipt = StageBCompileGate(unres_plan, ArtifactStore(tmp_path / "ovf_cache")).run(
        hybrid_src
    )
    assert ovf_receipt.passed is False
    assert ovf_receipt.fault_code == "VF_STACK_OVERFLOW"
    assert ovf_receipt.max_stack_bytes == 128


class _DirectExtractor:
    """Protocol implementation returning a single fixed CompileTaskSpec."""

    def __init__(self, spec: CompileTaskSpec) -> None:
        self._spec = spec

    def extract_case_tasks(
        self, module_source: str, case_id: str, case_spec: dict[str, object],
    ) -> tuple[CompileTaskSpec, ...]:
        _ = (module_source, case_id, case_spec)
        return (self._spec,)


class _RecordingStageBBackend:
    """Protocol implementation recording TileLang vs AscendC backend routing."""

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self.tilelang_calls: list[CompileTaskSpec] = []
        self.ascendc_calls: list[tuple[CompileTaskSpec, str]] = []

    def compile_tilelang(
        self, spec: CompileTaskSpec, module_source: str, work_dir: Path,
    ) -> tuple[bool, int, str, tuple[Path, ...], str]:
        _ = (module_source, work_dir)
        self.tilelang_calls.append(spec)
        obj = self._out_dir / f"{spec.task_key[:8]}_tl.so"
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(b"\x7fELF-tl")
        return True, 0, "", (obj,), ""

    def compile_ascendc(
        self, spec: CompileTaskSpec, cpp_source: str, work_dir: Path, regime_tag: str,
    ) -> tuple[bool, int, str, tuple[Path, ...], str]:
        _ = (work_dir, regime_tag)
        self.ascendc_calls.append((spec, cpp_source))
        obj = self._out_dir / f"{spec.task_key[:8]}_ascendc.o"
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(b"\x7fELF-ascendc")
        return True, 0, "", (obj,), ""


_TRACKB_NAMED_PRIMFUNC_MODULE = '''
import tilelang
import tilelang.language as T

def _build_trackb_fusion_func(d_dim: int = 64):
    @T.prim_func
    def trackb_fusion_kernel(x: T.Tensor((1, 64), "float32")):
        T.nop()
    return trackb_fusion_kernel

def trackb_fusion_entry(x):
    return tilelang.compile(_build_trackb_fusion_func(64), target="ascend")
'''

_NON_TRACKB_CPP_TU_MODULE = '''
def compile_ascend(src: str, *, options=()):
    return "stub.so"

_CUSTOM_VEC_TU = """
#include <tl_templates/ascend/common.h>
extern "C" __global__ __vector__ void custom_vec_impl(__gm__ float* x) {
    x[0] = 1.0f;
}
"""

def custom_vector_kernel(x):
    return compile_ascend(_CUSTOM_VEC_TU, options=["-O2"])
'''


@pytest.mark.parametrize(
    ("module_source", "expected_unit_kind", "expect_ascendc", "custom_builder_name"),
    [
        (_TRACKB_NAMED_PRIMFUNC_MODULE, "tilelang_dsl", False, None),
        (_NON_TRACKB_CPP_TU_MODULE, "cce_cpp20", True, "_build_custom_vector_kernel"),
    ],
    ids=["reject_trackb_named_primfunc", "accept_non_trackb_cpp_tu"],
)
def test_stageb_unit_kind_codomain_dispatch(
    tmp_path: Path,
    module_source: str,
    expected_unit_kind: str,
    expect_ascendc: bool,
    custom_builder_name: str | None,
) -> None:
    """Dispatch Stage-B compilation purely on CompileTaskSpec.unit_kind."""
    ext = ModuleSelectorPlanExtractor(target="dav-3510")
    cases = {"c0": {"input_shape": [[1, 64]], "dtype": ["float32"]}}
    plan = build_post_selector_regime_plan(module_source, "codomain", cases, ext)
    assert len(plan.unique_specs_by_key) == 1
    spec = next(iter(plan.unique_specs_by_key.values()))
    assert spec.unit_kind == expected_unit_kind
    assert spec.unit_source.strip()

    backend: StageBCompileBackend = _RecordingStageBBackend(tmp_path / "artifacts")
    receipt = StageBCompileGate(
        plan, ArtifactStore(tmp_path / "cache"), backend=backend, max_workers=1,
    ).run(module_source)
    assert receipt.passed is True
    assert isinstance(backend, _RecordingStageBBackend)
    assert len(backend.ascendc_calls) == (1 if expect_ascendc else 0)
    assert len(backend.tilelang_calls) == (0 if expect_ascendc else 1)

    if custom_builder_name is not None:
        direct_spec = CompileTaskSpec(
            task_key="direct_custom_cpp",
            builder_name=custom_builder_name,
            static_kwargs=(),
            pass_configs=(),
            compile_flags=("-O2",),
            unit_kind="cce_cpp20",
            unit_source='extern "C" __global__ __vector__ void custom_vec_impl(__gm__ float* x) {}',
        )
        direct_ext: CasePlanExtractor = _DirectExtractor(direct_spec)
        direct_plan = build_post_selector_regime_plan("", "direct", {"c0": {}}, direct_ext)
        direct_backend = _RecordingStageBBackend(tmp_path / "direct_artifacts")
        direct_rec = StageBCompileGate(
            direct_plan, ArtifactStore(tmp_path / "direct_cache"), backend=direct_backend,
        ).run("")
        assert direct_rec.passed is True
        assert len(direct_backend.ascendc_calls) == 1
        assert len(direct_backend.tilelang_calls) == 0
