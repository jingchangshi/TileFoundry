"""Contract tests for the Ascend 950 Scheduling-Interface-Layer (SIL)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilefoundry_ascend.ascend.sil import (
    DEFAULT_OPS_ROOT,
    VALID_BACKEND_TIERS,
    VALID_COMPUTE_ENGINES,
    VALID_FREEDOM_FACETS,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    OpAdvisoryExtractor,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    SchedulingEnvelope,
    SILVerifier,
    TelemetryFeedback,
    TopologyAxes,
    VerificationReport,
    synthesize_default_policy,
    verify_policy_envelope,
)

AXES_SIMD = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
AXES_SIMT = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
AXES_GEMM = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
AXES_GROUPED = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
AXES_CV_RING = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
AXES_PERSISTENT = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
AXES_STRIPED_VM = TopologyAxes("aiv_only", "persistent_onchip_state", "dynamic_opcode_stream")


def _sample_advisory(
    *,
    op_name: str = "sample_gemm",
    topology_axes: TopologyAxes = AXES_GEMM,
    regime_tag: str = "aligned_main",
    rep_dims: dict[str, int] | None = None,
    has_unaligned_tail: bool = False,
    has_nan_inf_cases: bool = False,
    precision_thresholds: dict[str, float] | None = None,
) -> AdvisoryEnvelope:
    return AdvisoryEnvelope(
        op_name=op_name,
        cannbot_task=f"level3/{op_name}",
        level=3,
        topology_axes=topology_axes,
        input_dtypes=("float16", "float16"),
        output_dtypes=("float16",),
        roofline={"dominant_bound": "cube_compute", "mean_t_hw_us": 12.5},
        structural_features={"uses_cube": topology_axes.uses_cube},
        regimes=(
            RegimeSlice(
                regime_tag=regime_tag,
                case_nums=(1, 2, 3),
                representative_dims=rep_dims or {"M": 256, "N": 256, "K": 128},
                has_unaligned_tail=has_unaligned_tail,
                has_empty_group=False,
                ref_perf_us=37.5,
                t_hw_us=12.5,
            ),
        ),
        legality_bounds=HardwareLegalityBounds(),
        has_nan_inf_cases=has_nan_inf_cases,
        has_inplace_aliasing=False,
        precision_thresholds=precision_thresholds if precision_thresholds is not None else {},
        hir_blueprint=None,
    )


def test_schema_roundtrips_and_validation(tmp_path: Path) -> None:
    assert len(VALID_FREEDOM_FACETS) == 7
    assert VALID_BACKEND_TIERS == ("tilelang_dsl", "ascendc_template")
    assert ConstraintRelaxationMask.bottom().unlocked_facets() == (
        "alg_compute_and_monoid",
        "tile_and_stage_knobs",
    )
    assert ConstraintRelaxationMask.top().is_unrestricted_top is True

    advisory = _sample_advisory(
        precision_thresholds={"float16": 2.0**-10, "float32": 2.0**-13}
    )
    policy = SchedulePolicy(
        regime_tag="aligned_main",
        topology_axes=AXES_GEMM,
        backend_tier="tilelang_dsl",
        tile_mnk=(128, 128, 64),
        l1_stages=2,
        l1_half_bank_split=True,
        l1_slot_aliasing={"K_rope": "P"},
        l1_estimated_bytes=131072,
        ub_estimated_bytes=65536,
        dma_paths_used=("GM->L1", "L1->L0A", "L1->L0B", "L0C->GM"),
        cube_operand_dtype="float16",
    )
    policy_env = PolicyEnvelope(
        op_name="sample_gemm",
        policies={"aligned_main": policy},
        selected_rules=("swat_swizzle", "half_l1_split"),
    )
    report = verify_policy_envelope(policy_env, advisory)
    assert report.passed is True

    telemetry = TelemetryFeedback(
        stage_a_passed=True,
        stage_b_passed=True,
        stage_c_passed=True,
        spill_bytes=0,
        vloopv2_hit=True,
        measured_latency_us=14.2,
        eq3_score=98.4,
        hap_reward=1.0,
        pipe_utilization={"PIPE_M": 0.91, "PIPE_MTE2": 0.85},
        bank_conflict_ratio=0.01,
        l2_hit_rate=0.94,
    )
    envelope = SchedulingEnvelope(
        advisory=advisory,
        policy=policy_env,
        verification=report,
        telemetry=telemetry,
    )

    for obj, cls in (
        (advisory.regimes[0], RegimeSlice),
        (advisory.legality_bounds, HardwareLegalityBounds),
        (advisory, AdvisoryEnvelope),
        (policy, SchedulePolicy),
        (policy_env, PolicyEnvelope),
        (report, VerificationReport),
        (telemetry, TelemetryFeedback),
        (envelope, SchedulingEnvelope),
    ):
        assert cls.from_dict(obj.to_dict()) == obj
        assert cls.from_json(obj.to_json()) == obj

    saved = envelope.save_json(tmp_path / "sil_envelope.json")
    assert saved.exists()
    loaded = SchedulingEnvelope.load_json(saved)
    assert loaded == envelope
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["advisory"]["op_name"] == "sample_gemm"

    with pytest.raises(ValueError, match="compute_engine"):
        TopologyAxes(compute_engine="unknown_engine", memory_ring="gm_ub_stream", iteration_regularity="affine_simd")
    with pytest.raises(ValueError, match="backend_tier"):
        SchedulePolicy(regime_tag="r", topology_axes=AXES_SIMD, backend_tier="cuda")
    with pytest.raises(ValueError, match="level"):
        AdvisoryEnvelope(
            op_name="x",
            cannbot_task="level9/x",
            level=9,
            topology_axes=AXES_SIMD,
            input_dtypes=("float16",),
            output_dtypes=("float16",),
            roofline={},
            structural_features={},
            regimes=advisory.regimes,
        )


def test_verifier_memory_capacity_redlines() -> None:
    verifier = SILVerifier()
    advisory = _sample_advisory()

    rep_l1 = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            l1_estimated_bytes=524289,
        ),
        advisory,
    )
    assert rep_l1.passed is False
    assert any("L1 footprint" in e for e in rep_l1.errors)

    rep_half_l1 = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            l1_stages=3,
            l1_half_bank_split=True,
            l1_estimated_bytes=450000,
        ),
        advisory,
    )
    assert rep_half_l1.passed is False
    assert any("Half-L1" in e for e in rep_half_l1.errors)

    rep_l0c = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            tile_mnk=(256, 256, 64),
            l0c_buffers=2,
        ),
        advisory,
    )
    assert rep_l0c.passed is False
    assert any("L0C" in e for e in rep_l0c.errors)

    rep_ub_locked = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_SIMD,
            vector_mode="simdvf",
            ub_256kb_unlock=False,
            ub_estimated_bytes=260000,
        ),
        advisory,
    )
    assert rep_ub_locked.passed is False
    assert rep_ub_locked.suggested_fixes.get("ub_256kb_unlock") is True

    rep_ub_unlocked = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_SIMD,
            vector_mode="simdvf",
            ub_256kb_unlock=True,
            ub_estimated_bytes=260000,
        ),
        advisory,
    )
    assert rep_ub_unlocked.passed is True

    for mode in ("simtvf", "hybrid"):
        rep_simt = verifier.verify_policy(
            SchedulePolicy(
                regime_tag="aligned_main",
                topology_axes=AXES_SIMT,
                vector_mode=mode,
                ub_256kb_unlock=True,
                ub_estimated_bytes=225000,
            ),
            advisory,
        )
        assert rep_simt.passed is False
        assert any("ub_256kb_unlock=True" in e for e in rep_simt.errors)
        assert any("exceeds" in e for e in rep_simt.errors)


def test_verifier_3510_paths_int4_wrap_and_unit_flag() -> None:
    verifier = SILVerifier()
    adv_tail = _sample_advisory(has_unaligned_tail=True)

    for bad_path in ("L1->GM", "GM->L0A", "GM->L0B"):
        rep = verifier.verify_policy(
            SchedulePolicy(
                regime_tag="aligned_main",
                topology_axes=AXES_GEMM,
                dma_paths_used=("GM->L1", bad_path),
            ),
            adv_tail,
        )
        assert rep.passed is False
        assert any(bad_path in e for e in rep.errors)

    for i4_dt in ("i4", "u4", "int4", "q4_0"):
        rep_i4 = verifier.verify_policy(
            SchedulePolicy(
                regime_tag="aligned_main",
                topology_axes=AXES_GEMM,
                cube_operand_dtype=i4_dt,
                w4_mode="none",
            ),
            adv_tail,
        )
        assert rep_i4.passed is False
        assert rep_i4.suggested_fixes.get("w4_mode") == "aiv_unpack_f16"
        assert "mxa8w4_bitshift_x64" in rep_i4.suggested_fixes.get("w4_mode_alternatives", [])

    rep_i4_ok = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            cube_operand_dtype="q4_0",
            w4_mode="mxa8w4_bitshift_x64",
        ),
        adv_tail,
    )
    assert rep_i4_ok.passed is True

    rep_wrap = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            tile_mnk=(64, 64, 4096),
        ),
        adv_tail,
    )
    assert rep_wrap.passed is False
    assert any("4095" in e for e in rep_wrap.errors)

    rep_uf = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            sync_mode="unit_flag_2_3",
            l0c_buffers=2,
        ),
        adv_tail,
    )
    assert rep_uf.passed is False
    assert rep_uf.suggested_fixes.get("l0c_buffers") == 1


def test_verifier_bank_conflict_crosscore_simt_and_precision_guards() -> None:
    verifier = SILVerifier()

    adv = _sample_advisory()
    rep_bank = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_SIMD,
            ub_row_stride_bytes=1024,
            ub_bank_pad_datablocks=0,
        ),
        adv,
    )
    assert rep_bank.passed is True
    assert len(rep_bank.warnings) == 1
    assert rep_bank.suggested_fixes["ub_bank_pad_datablocks"] == 1
    assert rep_bank.suggested_fixes["ub_rw_shift_datablocks"] == 8

    rep_cc = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_CV_RING,
            sync_mode="mode4_flag",
            reverse_ack_period=16,
        ),
        adv,
    )
    assert rep_cc.passed is False
    assert any("max_crosscore_depth=15" in e for e in rep_cc.errors)

    rep_split = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GROUPED,
            cv_datapath="l0c2ub_split_m",
            ensure_empty_subblock_sync=False,
        ),
        adv,
    )
    assert rep_split.passed is False
    assert any("ensure_empty_subblock_sync=True" in e for e in rep_split.errors)

    rep_simt_warn = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_SIMT,
            vector_mode="simtvf",
            simt_threads=2048,
        ),
        adv,
    )
    assert rep_simt_warn.passed is True
    assert any("__launch_bounds__(512)" in w for w in rep_simt_warn.warnings)

    adv_strict_fp32 = _sample_advisory(precision_thresholds={"float32": 2.0**-13})
    rep_hf32 = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_GEMM,
            hf32_allowed=True,
        ),
        adv_strict_fp32,
    )
    assert rep_hf32.passed is False
    assert any("hf32_allowed=True" in e for e in rep_hf32.errors)

    adv_nan = _sample_advisory(has_nan_inf_cases=True)
    rep_exp = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="aligned_main",
            topology_axes=AXES_SIMD,
            fast_unshifted_exp=True,
        ),
        adv_nan,
    )
    assert rep_exp.passed is False
    assert any("has_nan_inf_cases=True" in e for e in rep_exp.errors)

    adv_prefill = _sample_advisory(
        op_name="gqa",
        topology_axes=AXES_CV_RING,
        regime_tag="prefill",
        rep_dims={"M": 64, "D": 128},
    )
    rep_m82 = verifier.verify_policy(
        SchedulePolicy(
            regime_tag="prefill",
            topology_axes=AXES_CV_RING,
            tile_mnk=(64, 128, 64),
        ),
        adv_prefill,
    )
    assert rep_m82.passed is True
    assert any("mte2_bound_below_82" in w for w in rep_m82.warnings)


def test_all_60_cannbench_operators_extract_synthesize_and_verify(tmp_path: Path) -> None:
    if not DEFAULT_OPS_ROOT.exists():
        pytest.skip(f"cann-bench tasks root missing: {DEFAULT_OPS_ROOT}")

    extractor = OpAdvisoryExtractor(DEFAULT_OPS_ROOT)
    advisories = extractor.extract_all()
    assert len(advisories) == 60

    observed_engines: set[str] = set()
    observed_rings: set[str] = set()
    observed_regs: set[str] = set()
    for op_name, advisory in advisories.items():
        assert advisory.op_name == op_name
        assert 1 <= advisory.level <= 5
        assert advisory.topology_axes.compute_engine in VALID_COMPUTE_ENGINES
        assert advisory.topology_axes.memory_ring in VALID_MEMORY_RINGS
        assert advisory.topology_axes.iteration_regularity in VALID_ITERATION_REGULARITIES
        observed_engines.add(advisory.topology_axes.compute_engine)
        observed_rings.add(advisory.topology_axes.memory_ring)
        observed_regs.add(advisory.topology_axes.iteration_regularity)
        assert len(advisory.regimes) >= 1
        covered_cases = sorted(c for r in advisory.regimes for c in r.case_nums)
        assert covered_cases == list(range(1, 21)), f"{op_name} missing cases: {covered_cases}"

        policy_env = synthesize_default_policy(advisory)
        report = verify_policy_envelope(policy_env, advisory)
        assert report.passed is True, f"{op_name} failed SILVerifier: {report.errors}"
        assert report.errors == ()

    assert observed_engines == set(VALID_COMPUTE_ENGINES)
    assert observed_rings == set(VALID_MEMORY_RINGS)
    assert observed_regs == set(VALID_ITERATION_REGULARITIES)

    assert advisories["exp"].has_nan_inf_cases is True
    assert advisories["softmax"].has_nan_inf_cases is True
    assert advisories["kv_cache_append"].has_inplace_aliasing is True
    assert advisories["tensor_program_vm"].topology_axes.iteration_regularity == "dynamic_opcode_stream"
    assert advisories["grouped_matmul"].topology_axes.memory_ring == "l0c_ub_epilogue"
    assert advisories["grouped_matmul"].topology_axes.iteration_regularity == "ragged_group"
    assert any(r.has_empty_group for r in advisories["grouped_matmul"].regimes)
    assert any(r.has_empty_group for r in advisories["grouped_matmul_swiglu_quant"].regimes)
    assert any("prefill" in r.regime_tag for r in advisories["gqa"].regimes)
    assert any("decode" in r.regime_tag for r in advisories["gqa"].regimes)

    q4_policy = synthesize_default_policy(advisories["q4_0_dequant_matmul"])
    for pol in q4_policy.policies.values():
        assert pol.w4_mode == "aiv_unpack_f16"

    kda_adv = advisories["kimi_delta_attention"]
    kda_pol = synthesize_default_policy(kda_adv)
    kda_rep = verify_policy_envelope(kda_pol, kda_adv)
    kda_env = SchedulingEnvelope(advisory=kda_adv, policy=kda_pol, verification=kda_rep)
    out_file = kda_env.save_json(tmp_path / "kda_sil.json")
    assert SchedulingEnvelope.load_json(out_file) == kda_env
