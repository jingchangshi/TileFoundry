"""Tests for Track-B repair base/folding/budget, hybrid routing, and R16/case_ratios."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    _reject_trackb_folding_markers,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    _compact_trackb_repair_source,
    _detect_trackb_prologue_line_offset,
    _resplice_trackb_omitted_spans,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.kb.schema import (
    _observation_to_signal_dict,
    load_seed_rules,
)
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    FunnelEvaluation,
    StageAResult,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_syntax_repair_request
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios


def _grouped_matmul_parent_module() -> tuple[str, RegimeTargetSpec]:
    cpp_body = "\n".join(
        [
            "#include <tl_templates/ascend/common.h>",
            'extern "C" __global__ __aicore__ void gmm_regime_a_kernel(__gm__ half* x, __gm__ half* y) {',
            "    __ubuf__ half* ub_x = (__ubuf__ half*)0;",
            "    asc_copy_gm2ub_align(ub_x, x, 64);",
            *(f"    int pad_line_{idx} = {idx};" for idx in range(1, 220)),
            "    int target_val = 1;",
            "    asc_copy_ub2gm_align(y, ub_x, 64);",
            "}",
        ]
    )
    prologue = "".join(f"#define GMM_PROLOGUE_{i} {i}\n" for i in range(12))
    mod_src = (
        f"GMM_PROLOGUE = {prologue!r}\n"
        'GMM_TRACKB_CPP = r"""\n'
        f"{cpp_body}\n"
        '"""\n\n'
        "def _render_trackb_tu(trackb_src: str, **dims: object) -> str:\n"
        "    return GMM_PROLOGUE + trackb_src\n\n"
        "def _select_gmm_config():\n"
        '    return {"tile_m": 64}\n\n'
        "def gmm_regime_a():\n"
        "    return _render_trackb_tu(GMM_TRACKB_CPP)\n"
    )
    spec = RegimeTargetSpec(
        regime_tag="gmm_a",
        builder_candidates=("_select_gmm_config", "gmm_regime_a"),
        sample_builder_kwargs={},
        representative_hir_dims={"M": 128, "N": 128, "K": 64},
        editable_symbols=("GMM_TRACKB_CPP", "_select_gmm_config", "gmm_regime_a"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("GMM_TRACKB_CPP",),
        config_selector_fn="_select_gmm_config",
    )
    return mod_src, spec


def test_trackb_prologue_offset_no_exec_module_and_multi_literal_budget(
    tmp_path: Path,
) -> None:
    """Verify AST prologue offset skips exec and multi-literal budget splits fairly."""
    side_effect_file = tmp_path / "executed.flag"
    untrusted_mod = (
        "from pathlib import Path\n"
        f"Path({str(side_effect_file)!r}).write_text('boom', encoding='utf-8')\n"
        'PROLOGUE_HDR = "#define A 1\\n#define B 2\\n#define C 3\\n"\n'
        "def _render_trackb_tu(trackb_src: str) -> str:\n"
        '    return PROLOGUE_HDR + trackb_src\n'
    )
    offset = _detect_trackb_prologue_line_offset(untrusted_mod)
    assert offset == 3
    assert not side_effect_file.exists()

    cube_lines = [
        'extern "C" __global__ __aicore__ void mla_cube_kernel(__gm__ half* q, __gm__ half* k) {',
        *(f"    int c_var_{i:03d} = {i}; /* mla_cube_pad_{i:03d} */" for i in range(1, 220)),
        "    TARGET_MLA_CUBE_ERR(q);",
        *(f"    int c_tail_{i:03d} = {i};" for i in range(1, 40)),
        "}",
    ]
    vec_lines = [
        'extern "C" __global__ __vector__ void mla_vec_kernel(__gm__ float* s, __gm__ float* o) {',
        *(f"    float v_var_{i:03d} = {i}.0f; /* mla_vec_pad_{i:03d} */" for i in range(1, 220)),
        "    o[0] = v_var_219;",
        "}",
    ]
    mla_mod = (
        'MLA_PROLOGUE = "#define MLA_HDR 1\\n#define MLA_HDR2 2\\n#define MLA_HDR3 3\\n#define MLA_HDR4 4\\n#define MLA_HDR5 5\\n"\n'
        'MLA_CUBE_CPP = r"""\n'
        + "\n".join(cube_lines)
        + '\n"""\n\n'
        'MLA_VEC_CPP = r"""\n'
        + "\n".join(vec_lines)
        + '\n"""\n\n'
        "def _render_trackb_tu(trackb_src: str) -> str:\n"
        "    return MLA_PROLOGUE + trackb_src\n"
    )
    err_line_in_cube = 5 + 1 + 221
    diag = f"/tmp/mla.cce:{err_line_in_cube}:5: error: undeclared TARGET_MLA_CUBE_ERR"
    compacted = _compact_trackb_repair_source(
        mla_mod,
        diag,
        max_bytes=12000,
        prologue_line_offset=5,
    )
    tree = ast.parse(compacted)
    lit_sizes: dict[str, int] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.targets[0].id in ("MLA_CUBE_CPP", "MLA_VEC_CPP")
        ):
            lit_sizes[node.targets[0].id] = len(node.value.value.encode("utf-8"))
    assert lit_sizes["MLA_CUBE_CPP"] > 2000
    assert lit_sizes["MLA_VEC_CPP"] > 2000
    assert len(compacted.encode("utf-8")) <= 12500
    assert "TARGET_MLA_CUBE_ERR" in compacted


def test_trackb_stageb_repair_repatch_against_cur_patched_and_fallback() -> None:
    """Verify Stage-B repair patches against cur_patched first and falls back on full slice."""
    parent_mod, spec = _grouped_matmul_parent_module()
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
        waterlines={},
    )

    cand_patch = (
        "<<<<<<< SEARCH\n"
        "    int target_val = 1;\n"
        "=======\n"
        "    int target_val = broken_stageb_symbol;\n"
        ">>>>>>> REPLACE\n"
    )
    repair_against_current = (
        "<<<<<<< SEARCH\n"
        "    int target_val = broken_stageb_symbol;\n"
        "=======\n"
        "    int target_val = 42;\n"
        ">>>>>>> REPLACE\n"
    )

    def _patch_against(base: str, snip: str) -> str:
        return patch_regime_into_module(
            base,
            spec,
            snip,
            active_kernel="gmm_regime_a_kernel",
        )

    def _l1_compile(mod_src: str) -> StageBResult:
        if "broken_stageb_symbol" in mod_src:
            return StageBResult(
                passed=False,
                error_message="gmm.cce:239:5: error: use of undeclared identifier 'broken_stageb_symbol'",
            )
        if "int target_val = 42;" in mod_src or "int target_val = 99;" in mod_src:
            return StageBResult(passed=True)
        return StageBResult(passed=False, error_message="missing expected fix")

    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c_repatch", cand_patch)],
            patch_fn=lambda snip: _patch_against(parent_mod, snip),
            repatch_from_current_fn=_patch_against,
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=_l1_compile,
            repair_backend=MockLLMBackend(default_response=repair_against_current),
            max_syntax_repairs=1,
            backend_kind="hybrid_trackb",
            parent_source=parent_mod,
            trackb_prologue_line_offset=12,
        )
    )
    assert len(checks) == 1
    assert checks[0].passed is True, checks[0].error_message
    assert checks[0].repaired is True
    assert "int target_val = 42;" in checks[0].patched_module_source
    assert "broken_stageb_symbol" not in checks[0].patched_module_source

    full_replacement = (
        'GMM_TRACKB_CPP = r"""\n'
        + parent_mod.split('GMM_TRACKB_CPP = r"""\n', 1)[1].split('\n"""', 1)[0].replace(
            "int target_val = 1;",
            "int target_val = 99;",
        )
        + '\n"""\n'
    )

    def _failing_repatch(_cur: str, _snip: str) -> str:
        raise ValueError("current patched state rejected full replacement")

    fallback_checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c_fallback", cand_patch)],
            patch_fn=lambda snip: _patch_against(parent_mod, snip),
            repatch_from_current_fn=_failing_repatch,
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=_l1_compile,
            repair_backend=MockLLMBackend(default_response=full_replacement),
            max_syntax_repairs=1,
            backend_kind="hybrid_trackb",
            parent_source=parent_mod,
        )
    )
    assert fallback_checks[0].passed is True, fallback_checks[0].error_message
    assert "int target_val = 99;" in fallback_checks[0].patched_module_source

    mismatch_checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c_mismatch", cand_patch)],
            patch_fn=lambda snip: _patch_against(parent_mod, snip),
            repatch_from_current_fn=_patch_against,
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=_l1_compile,
            repair_backend=MockLLMBackend(
                default_response=(
                    "<<<<<<< SEARCH\n"
                    "    int target_val = 1;\n"
                    "=======\n"
                    "    int target_val = 42;\n"
                    ">>>>>>> REPLACE\n"
                )
            ),
            max_syntax_repairs=1,
            backend_kind="hybrid_trackb",
            parent_source=parent_mod,
        )
    )
    assert mismatch_checks[0].passed is False
    assert "ast_patch_error" in mismatch_checks[0].error_message


def test_trackb_folding_marker_resplice_and_unresolvable_rejection() -> None:
    """Verify folding marker omitted spans splice back cleanly while hallucinated markers fail."""
    orig_cpp_lines = [
        "#include <tl_templates/ascend/common.h>",
        'extern "C" __global__ __aicore__ void gmm_fold_kernel(__gm__ half* x) {',
        "    int head_0 = 0;",
        "    int head_1 = 1;",
        *(f"    int mid_{i} = {i};" for i in range(10)),
        "    int tail_0 = 100;",
        "    int tail_1 = BROKEN_CALL();",
        "}",
    ]
    orig_cpp = "\n".join(orig_cpp_lines) + "\n"
    orig_mod = (
        'GMM_TRACKB_CPP = r"""\n'
        f"{orig_cpp}"
        '"""\n'
        "def _select_gmm_config():\n"
        '    return {"tile_m": 64}\n'
    )
    spec = RegimeTargetSpec(
        regime_tag="gmm_fold",
        builder_candidates=("_select_gmm_config",),
        sample_builder_kwargs={},
        representative_hir_dims={"M": 64, "N": 64},
        editable_symbols=("GMM_TRACKB_CPP", "_select_gmm_config"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("GMM_TRACKB_CPP",),
        config_selector_fn="_select_gmm_config",
    )

    folded_reply_cpp = "\n".join(
        [
            "#include <tl_templates/ascend/common.h>",
            'extern "C" __global__ __aicore__ void gmm_fold_kernel(__gm__ half* x) {',
            "    int head_0 = 0;",
            "    int head_1 = 1;",
            "    /* ... 10 lines omitted ... */",
            "    int tail_0 = 100;",
            "    int tail_1 = 777;",
            "}",
        ]
    )
    folded_reply_mod = (
        'GMM_TRACKB_CPP = r"""\n'
        f"{folded_reply_cpp}\n"
        '"""\n'
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 32},
        waterlines={},
    )
    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c_fold", orig_mod)],
            patch_fn=lambda snip: patch_regime_into_module(
                orig_mod, spec, snip, active_kernel="gmm_fold_kernel"
            ),
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda mod: StageBResult(
                passed=("BROKEN_CALL" not in mod and "int tail_1 = 777;" in mod),
                error_message="gmm.cce:17:5: error: BROKEN_CALL undeclared",
            ),
            repair_backend=MockLLMBackend(default_response=folded_reply_mod),
            max_syntax_repairs=1,
            backend_kind="ascendc_trackb",
        )
    )
    assert checks[0].passed is True, checks[0].error_message
    assert "int mid_5 = 5;" in checks[0].patched_module_source
    assert "/* ... 10 lines omitted ... */" not in checks[0].patched_module_source

    context_marker_reply = folded_reply_cpp.replace(
        "/* ... 10 lines omitted ... */",
        "/* [LINES OMITTED OUTSIDE FOCUS WINDOW; PATCH VISIBLE LINES] */",
    )
    respliced_ctx = _resplice_trackb_omitted_spans(
        context_marker_reply,
        pre_compaction_source=orig_cpp,
    )
    assert "int mid_9 = 9;" in respliced_ctx
    assert "LINES OMITTED" not in respliced_ctx

    hallucinated_wrong_count = folded_reply_mod.replace("10 lines omitted", "999 lines omitted")
    bad_checks = asyncio.run(
        run_l0_l1_round_filter(
            [("c_bad_fold", orig_mod)],
            patch_fn=lambda snip: patch_regime_into_module(
                orig_mod, spec, snip, active_kernel="gmm_fold_kernel"
            ),
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda _mod: StageBResult(passed=False, error_message="compile error"),
            repair_backend=MockLLMBackend(default_response=hallucinated_wrong_count),
            max_syntax_repairs=1,
            backend_kind="ascendc_trackb",
        )
    )
    assert bad_checks[0].passed is False
    assert "trackb_folding_marker_rejected" in bad_checks[0].error_message

    with pytest.raises(ValueError, match="truncated context markers"):
        _reject_trackb_folding_markers(
            "/* ... 12 lines omitted ... */",
            "gmm_fold",
        )


def test_hybrid_trackb_python_primfunc_routing_and_portfolio_repair_context() -> None:
    """Verify hybrid_trackb routes Python @T.prim_func to Python repair and keeps excerpts."""
    broken_py = (
        "@T.prim_func\n"
        "def main(\n"
        "    X: T.Tensor((16, 64), 'float16'),\n"
        "    Y: T.Tensor((16, 64), 'float16'),\n"
        "):\n"
        "    with T.Kernel(1) as bx:\n"
        "        T.copy(X[0:16, 0:64], Y[0:16, 0:64])\n"
    )
    py_req = build_syntax_repair_request(
        broken_py,
        "tir lowering failed",
        backend_kind="hybrid_trackb",
        parent_source=broken_py,
        lowered_excerpt="1: // lowered tir or cce line",
        lowered_excerpt_lang="cpp",
    )
    assert "```python\n" in py_req.user_prompt
    assert "VALID TARGET DIALECT PRIMITIVES (tilelang_dsl)" in py_req.user_prompt
    assert "### COMPILER LOWERED EXCERPT (READ-ONLY CONTEXT)" in py_req.user_prompt
    assert "1: // lowered tir or cce line" in py_req.user_prompt

    cpp_broken = 'extern "C" __global__ __aicore__ void k(__gm__ half* x) {\n    bad_call(x);\n}'
    cpp_req = build_syntax_repair_request(
        cpp_broken,
        "cce error",
        backend_kind="hybrid_trackb",
        lowered_excerpt="2: asc_copy_gm2ub_align(ub, x, 64);",
        lowered_excerpt_lang="cpp",
    )
    assert "VALID TARGET DIALECT PRIMITIVES (cce_cpp20)" in cpp_req.user_prompt
    assert "### COMPILER LOWERED EXCERPT (READ-ONLY CONTEXT)" in cpp_req.user_prompt

    dup_req = build_syntax_repair_request(
        cpp_broken,
        "cce error",
        backend_kind="hybrid_trackb",
        lowered_excerpt="bad_call(x);",
        lowered_excerpt_lang="cpp",
    )
    assert "### COMPILER LOWERED EXCERPT" not in dup_req.user_prompt

    big_cpp = "\n".join(
        [
            "#include <tl_templates/ascend/common.h>",
            'extern "C" __global__ __aicore__ void big_trackb_k(__gm__ half* x) {',
            *(f"    int line_var_{i:03d} = {i}; /* trackb_big_payload_{i:03d} */" for i in range(1, 250)),
            "    FAIL_LINE_SENTINEL(x);",
            *(f"    int tail_var_{i:03d} = {i}; /* trackb_tail_payload_{i:03d} */" for i in range(1, 250)),
            "}",
        ]
    )
    big_parent_mod = (
        'TRACKB_CPP = r"""\n'
        f"{big_cpp}\n"
        '"""\n'
        "def make_kernel():\n"
        "    return TRACKB_CPP\n"
    )
    good_dsl = (
        "import tilelang.ascend.language as T\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((16, 64), 'float16'), Y: T.Tensor((16, 64), 'float16')):\n"
        "        with T.Kernel(1):\n"
        "            pass\n"
        "    return main\n"
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 32},
        waterlines={},
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 5.0})
    parent_cand = BeamCandidate(
        candidate_id="p0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=big_parent_mod,
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=5.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(good_dsl),
    )

    compile_calls = 0

    class _FakeStageBFunnel(GateFunnel):
        def evaluate(self, dsl_source: str, **kwargs: object) -> FunnelEvaluation:
            nonlocal compile_calls
            compile_calls += 1
            stage_a = StageAResult(passed=True, prim_func=None, ledger=ledger)
            if compile_calls == 1:
                sb = StageBResult(
                    passed=False,
                    error_message="kernel.cce:252:5: error: FAIL_LINE_SENTINEL undeclared",
                    generated_cce="\n".join(f"// cce_stmt_{i}" for i in range(1, 510)),
                )
                return FunnelEvaluation(
                    passed=False,
                    failed_stage="stage_b",
                    stage_a=stage_a,
                    stage_b=sb,
                    error_message=sb.error_message,
                )
            sc = StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=4.0,
                residual=compute_static_dynamic_residual(ledger, {"dur_us": 4.0}),
                telemetry={"dur_us": 4.0},
            )
            return FunnelEvaluation(
                passed=True,
                failed_stage=None,
                stage_a=stage_a,
                stage_b=StageBResult(passed=True),
                stage_c=sc,
            )

    funnel = _FakeStageBFunnel()
    backend = MockLLMBackend(
        responses={
            "explore_creative": [{"dsl_source": big_parent_mod, "physical_hypothesis": "test"}],
            "exploit_rule": [big_parent_mod],
            "syntax_repair": [good_dsl],
        }
    )
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
        max_syntax_repairs=1,
    )
    asyncio.run(
        runner.run_round(
            parent_cand,
            round_idx=1,
            escalation_tier="hybrid_trackb",
        )
    )
    repair_calls = [req for req in backend.calls if req.role == "syntax_repair"]
    assert repair_calls
    repair_prompt = repair_calls[0].user_prompt
    assert "### PARENT AUTHORIZED SOURCE" in repair_prompt
    assert "### COMPILER LOWERED EXCERPT (READ-ONLY CONTEXT)" in repair_prompt
    assert "252: // cce_stmt_252" in repair_prompt
    assert "FAIL_LINE_SENTINEL" in repair_prompt
    assert "LINES OMITTED OUTSIDE FOCUS WINDOW" in repair_prompt


def test_portfolio_plan_workers_threads_case_ratios_compiler_summary_and_r16_text() -> None:
    """Verify plan_workers threads case_ratios/compiler_summary and R16 matches MECE text."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 32},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=196608,
                explicit_buffers={"x": 4096},
                implicit_buffers={},
            )
        },
        unaligned_dma_padding_bytes=16,
        dma_burst_efficiency_ratio=0.55,
        min_gm_dma_bytes=64,
    )
    comp = BishengResourceSummary(
        vec_regs=16,
        scalar_regs=8,
        spill_bytes=0,
        vloopv2_loops=1,
        software_loops=0,
    )
    res = compute_static_dynamic_residual(
        ledger,
        {"dur_us": 10.0, "l2_cache_hit_rate": 0.40},
        compiler_summary=comp,
    )
    parent = BeamCandidate(
        candidate_id="p_subatom",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source="def make_kernel():\n    pass\n",
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=10.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature("def make_kernel():\n    pass\n"),
    )
    subatom_ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=0.25,
        is_atom_aligned=False,
        working_set_sram_ratio=0.2,
        working_set_vreg_ratio=0.2,
        wave_occupancy_ratio=0.5,
        oi_ridge_ratio=0.3,
        inner_contiguous_bytes=8,
    )
    funnel = GateFunnel(
        compile_runner=lambda _src: StageBResult(passed=True),
    )
    runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        k_rule_workers=4,
        enable_tilesize_prescreen=False,
    )
    specs = runner.plan_workers(parent, round_idx=1, case_ratios=subatom_ratios)
    selected_ids = {s.rule.rule_id for s in specs if s.rule is not None}
    assert "R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE" in selected_ids
    assert "R16_PADDED_PITCH_TAIL_MASKING" not in selected_ids

    obs = ledger.to_observation_5d()
    sig_with_ratios = _observation_to_signal_dict(obs, case_ratios=subatom_ratios)
    assert sig_with_ratios["contig_atom_ratio"] == pytest.approx(0.25)

    rules = load_seed_rules()
    assert len(rules) == 41
    by_id = {r.rule_id: r for r in rules}
    r16 = by_id["R16_PADDED_PITCH_TAIL_MASKING"]
    assert "contig_atom_ratio >= 1.0" in r16.transform_template
    assert "contig_atom_ratio < 1.0" not in r16.transform_template
    assert "contig_atom_ratio < 1.0" not in r16.cpp20_transform_template
    assert "sub-32B" not in r16.transform_template
    assert "sub-32B" not in r16.cpp20_transform_template
    assert "sub-32B" not in r16.rationale
    assert "non-32B-multiple" in r16.concept_description
    for rule in rules:
        assert "measured_latency_us" not in rule.expected_postcondition
