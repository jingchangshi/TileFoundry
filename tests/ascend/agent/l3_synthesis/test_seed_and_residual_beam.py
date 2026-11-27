"""Tests for Stage-0 seed synthesis, prompt pruning, residual beam, and repair monotonicity."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    BeamCandidate,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    extract_golden_entrypoint_contract,
    format_golden_entrypoint_prompt_block,
    prepare_operator_seed_or_takeover,
    select_fallback_track_from_semantics,
    select_initial_track_from_semantics,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.kb import load_seed_rules
from tilefoundry_ascend.agent.orchestrator import (
    GenerationRequest,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import check_repair_monotonicity
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
    format_blueprint_delta_section,
    format_g5_parent_monotonicity_lock,
    format_target_dialect_layer,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_creative_request,
    build_exploit_request,
    build_syntax_repair_request,
    summarize_excluded_mechanisms,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compiler_summary_from_residual,
    compute_eq3_hap_objective,
    extract_cpp_trackb_signature_tags,
)
from tilefoundry_ascend.agent.specs.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.ascend.passes import (
    HirBlueprintPacket,
    HirRooflineSummary,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    CCE_CPP20_SURFACE,
    LOWERING_GUARDRAILS,
    TILELANG_DSL_SURFACE,
    filter_primitives_on_demand,
    load_cann_surface,
    render_layer2_surface,
)
from tilefoundry_ascend.ascend.templates.registry import (
    CANONICAL_TOPOLOGY_PRESETS,
    SHELL_AIV_SIMD_STREAM,
    SHELL_CV_RING_PIPELINE,
)

_EXP_GOLDEN = dedent(
    """
    import torch

    def exp(x: torch.Tensor, base: float = -1.0, scale: float = 1.0) -> torch.Tensor:
        return torch.exp(scale * x.to(torch.float32)).to(x.dtype)
    """
).strip()

_GATHER_GOLDEN = (
    "import torch\ndef gather(x, index):\n    return torch.gather(x, dim=1, index=index)\n"
)

_EXP_PROTO: dict[str, object] = {
    "name": "Exp",
    "inputs": [{"name": "x", "dtype": ["float16", "float32"], "shape": ["M", "D"]}],
    "outputs": [{"name": "y", "dtype": ["float16"]}],
    "attrs": [{"name": "base", "type": "float", "default": -1.0}],
}

_EXP_CASES: tuple[dict[str, object], ...] = (
    {"case_id": 1, "input_shape": [[1024, 1024]], "dtype": ["float16"]},
    {"case_id": 2, "input_shape": [[8192, 8192]], "dtype": ["float32"]},
)

_SEED_DSL = dedent(
    """
    import tilelang.ascend.language as T

    def _select_exp_config(
        M: int = 1024, D: int = 1024, tl_dtype: str = "float16"
    ) -> dict[str, object]:
        return {"bm": 64, "bn": 64, "bk": 64, "pass_configs": {}, "compile_flags": []}

    def _build_exp_prim_func(
        M: int = 1024, D: int = 1024, bm: int = 64, bn: int = 64, bk: int = 64
    ):
        @T.prim_func
        def main(X: T.Tensor((M, D), "float16"), Y: T.Tensor((M, D), "float16")):
            with T.Kernel(1) as bx:
                x_ub = T.alloc_shared((64, 128), "float16")
                for i in T.Pipelined(2, num_stages=2):
                    T.copy(X[i * 64 : (i + 1) * 64, 0:128], x_ub)
                    with T.SimdVF():
                        T.fill(x_ub, 0.0)
                    T.copy(x_ub, Y[i * 64 : (i + 1) * 64, 0:128])
        return main

    def exp(x: "torch.Tensor", base: float = -1.0, scale: float = 1.0) -> "torch.Tensor":
        return x

    kernel = exp
    """
).strip()

_TRACKB_SEED = dedent(
    """
    EXP_TRACKB_CU = r\"\"\"
    #include <tl_templates/ascend/common.h>
    __global__ void gather_k() {}
    \"\"\"
    def _render_trackb_tu(**dims: object) -> str:
        return str(dims.get("trackb_src", ""))
    def _select_exp_config(M: int = 1024) -> dict[str, object]:
        return {"trackb_symbol": "EXP_TRACKB_CU"}
    def exp(x: "torch.Tensor", index: "torch.Tensor"):
        raise RuntimeError("offline")
    kernel = exp
    """
).strip()


class _ScriptedBackend:
    def __init__(self, replies: Sequence[str | TimeoutError]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> str:
        self.requests.append(req)
        item = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        if isinstance(item, TimeoutError):
            raise item
        return item


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _accept_oracle(src: str) -> OracleAccuracyResult:
    return OracleAccuracyResult(
        passed=True,
        candidate_sha256=_sha256(src),
        l3_run_id="run-stage0-1",
        l3_receipt_path="/tmp/receipt.json",
        l3_state="passed",
    )


def _ledger(
    *,
    is_legal: bool = True,
    pipes: dict[str, int] | None = None,
    ub_bytes: int = 4096,
) -> object:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=is_legal,
        steady_state_ii=16,
        pipe_cycles=pipes or {"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 32},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"ub": ub_bytes},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=1024,
    )


def test_track_selection_and_golden_entrypoint_contract() -> None:
    """Verify track selection rules and ParamBuilder entrypoint signature validation."""
    analyzer = PyTorchSemanticAnalyzer()
    exp_sem = analyzer.analyze_source(_EXP_GOLDEN, entry_name="exp", proto=_EXP_PROTO)
    exp_ratios = tuple(analyzer.compute_case_ratios(exp_sem, c) for c in _EXP_CASES)
    assert select_initial_track_from_semantics(exp_sem, exp_ratios) == "tilelang_dsl"

    gather_sem = analyzer.analyze_source(_GATHER_GOLDEN, entry_name="gather")
    gather_ratios = (analyzer.compute_case_ratios(gather_sem, _EXP_CASES[0]),)
    assert gather_sem.has_indirect_index is True
    assert select_initial_track_from_semantics(gather_sem, gather_ratios) == "cce_trackb"

    bad_ratio = SimpleNamespace(case_id=0)
    with pytest.raises(ValueError, match="positive case_id"):
        select_initial_track_from_semantics(exp_sem, (bad_ratio,))
    with pytest.raises(ValueError, match="positive case_id"):
        select_fallback_track_from_semantics(exp_sem, (bad_ratio,))

    cube_bp = HirBlueprintPacket(
        entry_name="mm",
        roofline=HirRooflineSummary(
            anchor_dims={"M": 64},
            compute_flops=1024,
            cube_flops=1024,
            min_gm_bytes=512,
            arithmetic_intensity=2.0,
            compute_floor_us=1.0,
            memory_floor_us=1.0,
            ideal_roofline_us=1.0,
            dominant_bound="cube",
        ),
    )
    assert select_fallback_track_from_semantics(exp_sem, exp_ratios, blueprints={"m": cube_bp}) == (
        "hybrid_trackb"
    )

    contract = extract_golden_entrypoint_contract(_EXP_GOLDEN, "exp", proto=_EXP_PROTO)
    assert "def exp(" in contract.signature_line
    assert validate_entrypoint_signature_compatibility(_SEED_DSL, contract) == ""
    assert "EXACT GOLDEN ENTRYPOINT ABI" in format_golden_entrypoint_prompt_block(contract)
    assert "parameter signature mismatch" in validate_entrypoint_signature_compatibility(
        "def exp(y: 'torch.Tensor'):\n    return y\n", contract
    )
    assert "must preserve default value" in validate_entrypoint_signature_compatibility(
        "def exp(x: 'torch.Tensor', base: float, scale: float = 1.0):\n    return x\n",
        contract,
    )


def test_seed_synthesizer_cold_start_takeover_retries_and_oracle_guards(
    tmp_path: Path,
) -> None:
    """Exercise cold-start synthesis, incumbent takeover, transport retries, and oracle checks."""
    backend = _ScriptedBackend([_SEED_DSL])
    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="level1/exp",
            entrypoint="exp",
            pytorch_golden_source=_EXP_GOLDEN,
            proto=_EXP_PROTO,
            cases=_EXP_CASES,
            legacy_regime_specs=None,
            llm_backend=backend,
            l0_l1_validator_factory=lambda _specs, semantics=None: (lambda _s, _t: (True, "")),
            oracle_accuracy_validator=_accept_oracle,
        )
    )
    assert outcome.mode == "cold_start"
    assert outcome.initial_track == "tilelang_dsl"
    assert outcome.verification_state == "l3_verified"
    assert outcome.attempts_used == 1
    assert set(outcome.blueprints_by_regime) == {r.regime_tag for r in outcome.regime_specs}
    first_prompt = backend.requests[0].user_prompt
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in first_prompt
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in first_prompt
    assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in first_prompt
    assert "### OFFICIAL PROTO CONTRACT" in first_prompt
    assert "### OFFICIAL CASES" in first_prompt

    tb_backend = _ScriptedBackend([_TRACKB_SEED])
    tb_out = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="gather",
            entrypoint="exp",
            pytorch_golden_source=_GATHER_GOLDEN,
            proto=_EXP_PROTO,
            cases=_EXP_CASES[:1],
            legacy_regime_specs=None,
            llm_backend=tb_backend,
            l0_l1_validator_factory=lambda _specs, semantics=None: (lambda _s, _t: (True, "")),
            oracle_accuracy_validator=_accept_oracle,
        )
    )
    assert tb_out.initial_track == "cce_trackb"
    assert CCE_CPP20_SURFACE.display_label in tb_backend.requests[0].system_prompt
    assert all(r.trackb_symbols == ("EXP_TRACKB_CU",) for r in tb_out.regime_specs)
    assert all("_render_trackb_tu" in r.editable_symbols for r in tb_out.regime_specs)

    incumbent = tmp_path / "kernels.py"
    incumbent.write_text("def exp(x):\n    return x\nkernel = exp\n", encoding="utf-8")
    legacy_specs = (
        RegimeTargetSpec(
            "vector",
            ("_build_exp_prim_func",),
            {"M": 1024, "D": 1024},
            representative_anchor_dims={"M": 1024, "D": 1024},
        ),
    )
    takeover = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="exp",
            entrypoint="exp",
            pytorch_golden_source=_EXP_GOLDEN,
            proto=_EXP_PROTO,
            cases=_EXP_CASES,
            legacy_regime_specs=legacy_specs,
            llm_backend=_ScriptedBackend([]),
            l0_l1_validator_factory=lambda _specs, semantics=None: (_ for _ in ()).throw(
                AssertionError("takeover must skip static validator")
            ),
            oracle_accuracy_validator=lambda _s: (_ for _ in ()).throw(
                AssertionError("takeover must skip oracle")
            ),
            existing_module_path=incumbent,
        )
    )
    assert takeover.mode == "incumbent_takeover"
    assert takeover.verification_state == "unverified_incumbent"
    assert takeover.attempts_used == 0

    retry_backend = _ScriptedBackend([TimeoutError("timed out"), _SEED_DSL])
    retry_out = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="exp",
            entrypoint="exp",
            pytorch_golden_source=_EXP_GOLDEN,
            proto=_EXP_PROTO,
            cases=_EXP_CASES,
            legacy_regime_specs=None,
            llm_backend=retry_backend,
            l0_l1_validator_factory=lambda _specs, semantics=None: (lambda _s, _t: (True, "")),
            oracle_accuracy_validator=_accept_oracle,
        )
    )
    assert retry_out.attempts_used == 1
    assert "LLM generation transport failure: TimeoutError" in retry_backend.requests[1].user_prompt

    converge_errs = [0.20, 0.10, 0.05]
    oracle_idx = {"n": 0}

    def _converging_oracle(src: str) -> OracleAccuracyResult:
        i = oracle_idx["n"]
        oracle_idx["n"] += 1
        if i < len(converge_errs):
            return OracleAccuracyResult(
                passed=False,
                error_message=f"max_abs_diff={converge_errs[i]}",
                l3_state="failed",
            )
        return _accept_oracle(src)

    conv_out = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="exp",
            entrypoint="exp",
            pytorch_golden_source=_EXP_GOLDEN,
            proto=_EXP_PROTO,
            cases=_EXP_CASES,
            legacy_regime_specs=None,
            llm_backend=_ScriptedBackend([_SEED_DSL] * 4),
            l0_l1_validator_factory=lambda _specs, semantics=None: (lambda _s, _t: (True, "")),
            oracle_accuracy_validator=_converging_oracle,
            max_seed_attempts=3,
            max_oracle_refinement_extension=2,
        )
    )
    assert conv_out.attempts_used == 4

    with pytest.raises(RuntimeError, match="oracle evidence digest mismatch"):
        asyncio.run(
            prepare_operator_seed_or_takeover(
                operator_name="exp",
                entrypoint="exp",
                pytorch_golden_source=_EXP_GOLDEN,
                proto=_EXP_PROTO,
                cases=_EXP_CASES,
                legacy_regime_specs=None,
                llm_backend=_ScriptedBackend([_SEED_DSL]),
                l0_l1_validator_factory=lambda _specs, semantics=None: (lambda _s, _t: (True, "")),
                oracle_accuracy_validator=lambda _s: OracleAccuracyResult(
                    passed=True,
                    candidate_sha256="0" * 64,
                    l3_run_id="r1",
                    l3_receipt_path="/tmp/r.json",
                ),
            )
        )


def test_residual_beam_exploit_and_stepping_stone_selection() -> None:
    """Verify structural signatures, Eq.3 HAP objective, two-lane updates, and roundtrip."""
    cpp_dsl = (
        'TRACKB = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__global__ __aicore__ void k() {\n"
        "  asc_mmad(a, b, c);\n"
        "  asc_copy_gm2l1_nd2nz(a, b);\n"
        "  asc_lock(s); asc_unlock(s);\n"
        "  asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)(i & 1));\n"
        "  int pad = stride + 16;\n"
        "}\n"
        '"""\n'
    )
    cpp_tags = extract_cpp_trackb_signature_tags(cpp_dsl)
    assert {
        "cpp:cube_mmad",
        "cpp:gm2l1_nd2nz",
        "cpp:slot_lock_pipeline",
        "cpp:pingpong_event_parity",
        "cpp:ub_bank_pad",
    } <= cpp_tags

    dsl_base = "def k():\n    pass\n"
    dsl_rich = (
        "import tilelang.ascend.language as T\n"
        "def k():\n"
        "    with T.MixedKernel(28, sids=2) as (bx, sid):\n"
        "        a = T.alloc_shared((2, 64), 'float16')\n"
        "        b = a[0, 0] + a[1, 0]\n"
        "        for i in T.serial(2):\n"
        "            pass\n"
        "        for j in T.serial(2):\n"
        "            pass\n"
        "        T.ascend_pipe_barrier('PIPE_ALL')\n"
        "        T.simd.vfma(a, a, a)\n"
    )
    ledger_ok = _ledger(is_legal=True)
    ledger_bad = _ledger(is_legal=False)
    res_base = compute_static_dynamic_residual(ledger_ok, {"dur_us": 50.0})
    res_fast = compute_static_dynamic_residual(ledger_ok, {"dur_us": 20.0})
    res_stone = compute_static_dynamic_residual(
        ledger_ok, {"dur_us": 35.0, "l2_hit_rate": 0.5, "bank_conflict_ratio": 0.2}
    )
    assert compiler_summary_from_residual(res_fast) is None
    assert compute_eq3_hap_objective(20.0, ref_perf_us=40.0, t_hw_us=5.0) > (
        compute_eq3_hap_objective(
            20.0, ref_perf_us=40.0, t_hw_us=5.0, bank_conflict_ratio=0.5, l2_hit_rate=0.2
        )
    )

    sig_rich = extract_structural_signature(dsl_rich, track="developer", residual=res_stone)
    expected_tags = {
        "track:developer",
        "T.MixedKernel",
        "static_pingpong_2buf",
        "loop_fission_multi_pass",
        "pipe_all_barrier",
        "vfma_unshifted",
        "hw:bank_conflict",
        "hw:low_l2_hit",
    }
    assert expected_tags <= set(sig_rich)

    c_seed = BeamCandidate(
        "seed", 0, "seed", None, dsl_base, "unconstrained", ledger_ok, res_base, 50.0, True,
        structural_signature=extract_structural_signature(dsl_base), objective_score=0.2,
    )
    c_fast = BeamCandidate(
        "fast", 1, "rule_exploit", "R01", dsl_base, "unconstrained", ledger_ok, res_fast,
        20.0, True, structural_signature=extract_structural_signature(dsl_base),
        objective_score=0.8,
    )
    c_stone = BeamCandidate(
        "stone", 1, "creative_explore", None, dsl_rich, "developer", ledger_ok, res_stone,
        35.0, True, structural_signature=sig_rich, objective_score=0.45,
    )
    c_illegal = BeamCandidate(
        "bad_legal", 1, "rule_exploit", "R02", dsl_base, "unconstrained", ledger_bad, res_fast,
        5.0, True, objective_score=0.99,
    )
    c_inaccurate = BeamCandidate(
        "bad_acc", 1, "rule_exploit", "R03", dsl_base, "unconstrained", ledger_ok, res_fast,
        5.0, False, objective_score=0.99,
    )

    beam = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    exploit, stones = beam.update([c_seed, c_fast, c_stone, c_illegal, c_inaccurate])
    assert [c.candidate_id for c in exploit] == ["fast"]
    assert [c.candidate_id for c in stones] == ["stone"]
    assert [c.candidate_id for c in beam.parent_pool()] == ["fast", "stone"]
    beam.mark_parent_selected(c_stone)
    assert sig_rich in beam.exploited_signatures

    restored = TwoLaneBeamSearch.from_dict(beam.to_dict())
    assert [c.candidate_id for c in restored.exploit_beam] == ["fast"]
    assert [c.candidate_id for c in restored.stepping_stone_beam] == ["stone"]
    assert sig_rich in restored.exploited_signatures


def test_prompt_on_demand_pruning_and_blueprint_sub_us_formatting() -> None:
    """Prune Layer-2 primitives by topology, mask, ratios, and symptom within budgets."""
    cce_93 = load_cann_surface("cann_9_3")
    tl_aiv = filter_primitives_on_demand(TILELANG_DSL_SURFACE, topology_axes=SHELL_AIV_SIMD_STREAM)
    cce_aiv = filter_primitives_on_demand(cce_93, topology_axes=SHELL_AIV_SIMD_STREAM)
    assert {"T.gemm", "T.dual_copy", "T.alloc_l1"}.isdisjoint({e.symbol for e in tl_aiv})
    assert {"ascend_gemm_l1", "asc_copy_l0c2ub", "asc_mmad"}.isdisjoint({e.symbol for e in cce_aiv})

    bottom = ConstraintRelaxationMask.bottom()
    top = ConstraintRelaxationMask.top()
    subatom = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=0.5,
        is_atom_aligned=False,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=1.0,
        oi_ridge_ratio=0.1,
        inner_contiguous_bytes=16,
    )
    assert "T.SimtVF" not in {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE, topology_axes=SHELL_AIV_SIMD_STREAM, freedom_mask=bottom
        )
    }
    assert "T.SimtVF" in {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE, topology_axes=SHELL_AIV_SIMD_STREAM, freedom_mask=top
        )
    }
    assert "T.SimtVF" in {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=SHELL_AIV_SIMD_STREAM,
            freedom_mask=bottom,
            case_ratios=subatom,
        )
    }
    assert "__simt_vf__" in render_layer2_surface(
        cce_93,
        primitives=filter_primitives_on_demand(
            cce_93, topology_axes=SHELL_AIV_SIMD_STREAM, freedom_mask=bottom, case_ratios=subatom
        ),
        include_guardrails=True,
    )

    cv_prompt = format_target_dialect_layer(
        escalation_tier="cce_cpp20",
        topology_axes=SHELL_CV_RING_PIPELINE,
        freedom_mask=bottom,
        cann_series="cann_9_3",
    )
    assert "asc_copy_l0c2ub(" in cv_prompt and "ascend_gemm_l1(" in cv_prompt
    for ln in cv_prompt.splitlines():
        if ln.startswith("- `") and ("asc_copy_l0c2ub" in ln or "ascend_gemm_l1" in ln):
            assert "..." not in ln

    assert len(LOWERING_GUARDRAILS) == 4
    monolithic = len(cce_93.render_prompt_reference())
    for axes in CANONICAL_TOPOLOGY_PRESETS:
        assert isinstance(axes, TopologyAxes)
        l2 = format_target_dialect_layer(
            escalation_tier="cce_cpp20", topology_axes=axes, cann_series="cann_9_3"
        )
        full = build_progressive_llm_constraints(
            escalation_tier="cce_cpp20", topology_axes=axes, cann_series="cann_9_3"
        )
        assert len(l2) <= 10000 and len(full) <= 18000
        assert (1.0 - len(l2) / float(monolithic)) >= 0.39
        for guard in LOWERING_GUARDRAILS:
            assert guard in l2

    bp = HirBlueprintPacket(
        entry_name="exp",
        roofline=HirRooflineSummary(
            anchor_dims={"M": 64},
            compute_flops=64,
            min_gm_bytes=128,
            arithmetic_intensity=0.5,
            compute_floor_us=1.7e-5,
            memory_floor_us=2.0e-5,
            ideal_roofline_us=1.7e-5,
            dominant_bound="memory",
        ),
    )
    assert "ideal_roofline_us: 1.7e-05" in format_blueprint_delta_section(bp)
    assert "G5 PARENT MONOTONICITY LOCK" in format_g5_parent_monotonicity_lock(
        include_blueprint_retention=True
    )


def test_portfolio_runner_rule_blind_creative_prescreen_and_repair_monotonicity() -> None:
    """Verify rule-blind creative, cost-gate prescreen, and repair monotonicity."""
    ledger = _ledger()
    parent = BeamCandidate(
        "p0", 0, "seed", None, _SEED_DSL, "unconstrained", ledger,
        compute_static_dynamic_residual(ledger, {"dur_us": 20.0}), 20.0, True,
        structural_signature=extract_structural_signature(_SEED_DSL),
    )
    creative_req = build_creative_request(
        parent,
        excluded_mechanisms=summarize_excluded_mechanisms(
            load_seed_rules()[:2], {("T.Pipelined",)}
        ),
    )
    full_creative = f"{creative_req.system_prompt}\n{creative_req.user_prompt}"
    for rule in load_seed_rules():
        assert rule.rule_id not in full_creative
        assert rule.transform_template.strip().splitlines()[0].strip() not in full_creative

    rule = load_seed_rules()[0]
    for req in (
        build_exploit_request(parent, rule),
        creative_req,
        build_syntax_repair_request(
            _SEED_DSL, "err", freedom_mask=ConstraintRelaxationMask.bottom()
        ),
    ):
        for g in ("G1 (No Scalar Compute Emulation)", "G2 (", "G3 (", "G4 ("):
            assert g in req.user_prompt

    drop_pipe = _SEED_DSL.replace("T.Pipelined(2, num_stages=2)", "range(2)")
    scalar_loop = _SEED_DSL.replace(
        "                with T.SimdVF():\n                    T.fill(x_ub, 0.0)",
        "                for j in range(128):\n                    x_ub[0, j] = 0.0",
    )
    assert scalar_loop != _SEED_DSL
    assert check_repair_monotonicity(_SEED_DSL, drop_pipe)
    assert check_repair_monotonicity(_SEED_DSL, scalar_loop)
    assert check_repair_monotonicity(_SEED_DSL, _SEED_DSL.replace("(64, 128)", "(32, 128)")) == []

    funnel = GateFunnel(
        compile_runner=lambda src: StageBResult(passed="broken(" not in src, error_message="err"),
        device_runner=lambda _s, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=12.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 12.0}),
            telemetry={"dur_us": 12.0},
        ),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    gate_runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=funnel,
        tracker=tracker,
        enable_tilesize_prescreen=True,
        cost_model_mode="advisory",
    )
    engaged = gate_runner.plan_workers(
        parent,
        round_idx=1,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64, "d_out": 64},
        bucket_clean_samples=30,
        out_of_sample_spearman_rho=0.75,
    )
    assert "### TILE PRESCREEN GUIDANCE" in next(
        s for s in engaged if s.worker_kind == "creative_explore"
    ).request.user_prompt
    disengaged = gate_runner.plan_workers(
        parent,
        round_idx=2,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64, "d_out": 64},
        bucket_clean_samples=2,
        out_of_sample_spearman_rho=0.75,
    )
    assert "### TILE PRESCREEN GUIDANCE" not in next(
        s for s in disengaged if s.worker_kind == "creative_explore"
    ).request.user_prompt

    mono_runner = PortfolioRunner(
        backend=MockLLMBackend(
            responses={
                "exploit_rule": ["def broken(:\n    pass\n"],
                "explore_creative": [
                    f"# target_bottleneck: x\n# structural_hypothesis: ok\n"
                    f"```python\n{_SEED_DSL}\n```"
                ],
                "syntax_repair": [drop_pipe],
            }
        ),
        funnel=GateFunnel(
            compile_runner=lambda src: StageBResult(
                passed="broken(" not in src, error_message="syntax_err"
            ),
            device_runner=lambda _s, st: StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=12.0,
                residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 12.0}),
                telemetry={"dur_us": 12.0},
            ),
        ),
        tracker=BayesianRuleTracker.from_seed_yaml(),
        k_rule_workers=1,
        max_syntax_repairs=1,
    )
    mono_report = asyncio.run(
        mono_runner.run_round(parent, round_idx=1, factory_name="_build_exp_prim_func")
    )
    assert any(
        stage == "repair_monotonicity_violation" for _, stage, _ in mono_report.failed_candidates
    )
    mono_runner.cancel_pending_speculative()
