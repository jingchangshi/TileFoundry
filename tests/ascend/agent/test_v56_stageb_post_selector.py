"""Regression tests for V56 L2 Post-Selector CompileTaskSpec Stage-B closed loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tilefoundry_ascend.agent.contracts import CompileTaskSpec
from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import (
    ModuleSelectorPlanExtractor,
    StageBCompileGate,
    build_post_selector_regime_plan,
    compute_compile_task_key,
    has_unreserved_vf_stack_overflow,
    validate_prewarm_manifest_v1,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import ArtifactStore

_REPO_ROOT = Path(__file__).resolve().parents[3]

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


def _build_synth_prim_func(d_dim: int, tl_dtype: str = "bfloat16", tile_m: int = 1, num_stages: int = 1, m_rows: int = 0):
    return (int(m_rows), int(d_dim), tl_dtype, int(tile_m), int(num_stages))


def _get_kernel(d_dim: int, tl_dtype: str = "bfloat16", builder_fn=None, pass_configs=None, compile_flags=None, extra_builder_kwargs=None):
    kw = dict(extra_builder_kwargs or {})
    prog = builder_fn(d_dim=d_dim, tl_dtype=tl_dtype, tile_m=kw.get("tile_m", 1), num_stages=kw.get("num_stages", 1))
    return tilelang.compile(prog, target="ascend", pass_configs=pass_configs or {}, compile_flags=list(compile_flags or ()))
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


def test_bug_sb1_post_selector_preserves_distinct_specs_for_shared_d_synthetic() -> None:
    """BUG-SB1: same D=2048 with m_rows=16 before m_rows=4096 retains both specializations."""
    extractor = ModuleSelectorPlanExtractor(
        target="dav-3510",
        builder_name="_build_synth_prim_func",
        config_selector_name="_select_synth_config",
    )
    cases = {
        "c0_small": {"m_rows": 16, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "c1_large": {"m_rows": 4096, "d_dim": 2048, "tl_dtype": "bfloat16"},
    }
    plan = build_post_selector_regime_plan(_SYNTHETIC_SELECTOR_MODULE, "synth", cases, extractor)
    assert len(plan.tasks_by_case_id) == 2
    assert len(plan.unique_specs_by_key) == 2
    spec_small = plan.tasks_by_case_id["c0_small"][0]
    spec_large = plan.tasks_by_case_id["c1_large"][0]
    assert spec_small.task_key != spec_large.task_key
    assert dict(spec_small.static_kwargs)["tile_m"] == 1
    assert dict(spec_large.static_kwargs)["tile_m"] == 8


def test_bug_sb1_real_arnq_and_ar_modules_retain_small_and_large_m_rows() -> None:
    """BUG-SB1: real arnq and attention_residual kernels keep m_rows=16 and 4096 specs."""
    extractor = ModuleSelectorPlanExtractor(target="dav-3510")
    cases = {
        "01_small": {"m_rows": 16, "n_depth": 8, "b_dim": 1, "s_dim": 16, "d_dim": 2048, "tl_dtype": "bfloat16"},
        "02_large": {"m_rows": 4096, "n_depth": 8, "b_dim": 1, "s_dim": 4096, "d_dim": 2048, "tl_dtype": "bfloat16"},
    }

    arnq_src = (_REPO_ROOT / "examples/ascend950pr_ops/add_rms_norm_dynamic_quant/kernels.py").read_text(
        encoding="utf-8"
    )
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
    """BUG-SB2: identical source/flags/pass_configs with distinct static_kwargs hash distinctly."""
    common = dict(
        clean_unit_src="def build(tile_m: int = 1): return tile_m",
        target="dav-3510",
        builder_name="build",
        pass_configs=(("tl.ascend_auto_sync", True),),
        compile_flags=("--cce-disable-asc-reserved-ubuf",),
    )
    k1 = compute_compile_task_key(static_kwargs=(("d_dim", 2048), ("tile_m", 1)), **common)
    k2 = compute_compile_task_key(static_kwargs=(("d_dim", 2048), ("tile_m", 8)), **common)
    k1_reordered = compute_compile_task_key(
        static_kwargs=(("tile_m", 1), ("d_dim", 2048)), **common
    )
    assert k1 != k2
    assert k1 == k1_reordered


def test_bug_sb3_has_unreserved_vf_stack_overflow_predicate() -> None:
    """BUG-SB3: stack_bytes > 0 without --cce-disable-vf-stack-reserved-ubuf is NOT overflow."""
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

    assert has_unreserved_vf_stack_overflow(reserved_spec, stack_bytes=256, compiler_diag="") is False
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
    assert has_unreserved_vf_stack_overflow(unreserved_spec, stack_bytes=128, compiler_diag="") is True
    assert has_unreserved_vf_stack_overflow(unreserved_spec, stack_bytes=0, compiler_diag="") is False
    assert gf.has_unreserved_vf_stack_overflow is has_unreserved_vf_stack_overflow
    assert not hasattr(gf, "_has_vf_stack_overflow_signal")


def test_bug_sb4_split_kv_stage1_plus_combine_both_checked_and_real_mla(tmp_path: Path) -> None:
    """BUG-SB4: 1-to-N subkernels (stage1 + combine) are both checked for VF stack overflow."""
    mla_src = (_REPO_ROOT / "examples/ascend950pr_ops/mla/kernels.py").read_text(encoding="utf-8")
    mla_ext = ModuleSelectorPlanExtractor(target="dav-3510")
    mla_cases = {
        "split_kv": {
            "b_dim": 1, "s_dim": 1, "s_kv": 2048, "n_q": 64, "n_kv": 1,
            "d_nope": 256, "d_rope": 64, "tl_dtype": "float16",
            "input_layout": "BSND", "is_causal": False,
        },
        "track_b": {
            "b_dim": 1, "s_dim": 1, "s_kv": 2048, "n_q": 128, "n_kv": 1,
            "d_nope": 512, "d_rope": 64, "tl_dtype": "float16",
            "input_layout": "BSND", "is_causal": False,
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
    plan = build_post_selector_regime_plan(_SYNTHETIC_SPLIT_KV_MODULE, "split_kv_gate", synth_cases, ext)
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
    """StageBCompileGate writes valid prewarm_manifest_v1.json and skips recompiles on hits."""
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
    Path(manifest["tasks"][first_key]["artifact_path"]).unlink()
    ok_after_unlink, bad_details = validate_prewarm_manifest_v1(manifest_path, plan)
    assert ok_after_unlink is False
    assert first_key in bad_details["missing_tasks"]
