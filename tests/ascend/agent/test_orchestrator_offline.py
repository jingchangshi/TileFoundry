"""Offline unit tests for WP4 dual-loop orchestrator, prescreener, funnel, and two-lane beam."""

import asyncio
import json
from pathlib import Path

from tilefoundry_ascend.agent import (
    AgyCliLLMBackend,
    BayesianRuleTracker,
    BeamCandidate,
    GateFunnel,
    GenerationRequest,
    MockLLMBackend,
    PortfolioRunner,
    StructuralFeatures,
    TileConfigCandidate,
    TwoLaneBeamSearch,
    format_prescreen_summary,
    generate_aligned_neighborhood,
    prescreen_dsl_tile_kwargs,
    prescreen_tile_configs,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import build_anchor_var_bindings
from tilefoundry_ascend.agent.kb import load_seed_rules
from tilefoundry_ascend.agent.orchestrator import (
    CREATIVE_RESPONSE_SCHEMA,
    StageBResult,
    StageCResult,
    build_creative_request,
    build_exploit_request,
    compute_static_dynamic_residual,
    extract_creative_payload,
    extract_structural_signature,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    TILELANG_ASCEND_PRIMITIVE_MENU,
    build_prescreen_summary_from_kwargs,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import evaluate_tile_candidate
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_SEED_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_OVERSIZED_UB_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(X: T.Tensor((2, 256, 128), "float32"), Y: T.Tensor((2, 256, 128), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256, 128), "float32")
            b_ub = T.alloc_shared((256, 128), "float32")
            for k in T.Pipelined(2, num_stages=2):
                T.copy(X[k, 0:256, 0:128], a_ub)
                T.copy(a_ub, b_ub)
                T.copy(b_ub, Y[k, 0:256, 0:128])
    return main
"""

_CREATIVE_SSBUF_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            ss[sid] = T.Cast("uint32", 0)
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""


def test_agy_backend_command_isolation_and_schema_extraction(monkeypatch) -> None:
    """Verify AgyCliLLMBackend strips --print/--add-dir, routes models, and isolates env."""
    monkeypatch.setenv("AGY_CONVERSATION_ID", "secret-conv")
    monkeypatch.setenv("MCP_TOKEN", "secret-mcp")
    monkeypatch.setenv("CLAUDECODE", "1")
    backend = AgyCliLLMBackend(
        bin_path="/bin/true",
        common_args="--print --input-format text --add-dir /tmp --sandbox --mode plan",
    )
    req_exploit = GenerationRequest("exploit_rule", "sys", "usr")
    cmd_exploit = backend.build_command(req_exploit, "PROMPT_TEXT")
    assert "gemini-4-hydrogen-medium" in cmd_exploit
    assert "--add-dir" not in cmd_exploit
    assert "--print=PROMPT_TEXT" in cmd_exploit
    assert "--print" not in cmd_exploit

    req_explore = GenerationRequest("explore_creative", "sys", "usr", CREATIVE_RESPONSE_SCHEMA)
    cmd_explore = backend.build_command(req_explore, "CREATIVE_PROMPT")
    assert "gemini-4-hydrogen-high" in cmd_explore
    assert "--output-format" in cmd_explore
    assert "--json-schema" in cmd_explore

    sanitized = backend._sanitized_env("/tmp/empty_box")
    assert "AGY_CONVERSATION_ID" not in sanitized
    assert "MCP_TOKEN" not in sanitized
    assert "CLAUDECODE" not in sanitized
    assert sanitized["PWD"] == "/tmp/empty_box"

    envelope = json.dumps(
        {
            "structured_output": {
                "physical_hypothesis": "Reduce FixPipe drain stall",
                "target_local_metric": "aic_fixpipe",
                "dsl_source": "```python\nx = 1\n```",
            }
        }
    )
    parsed = extract_creative_payload(envelope)
    assert parsed["physical_hypothesis"] == "Reduce FixPipe drain stall"
    assert parsed["dsl_source"] == "x = 1"


def test_tilesize_prescreen_and_stage_a_early_reject() -> None:
    """Verify dry-run tile prescreener and Stage A reject infeasible configs without bisheng."""
    cands = [
        TileConfigCandidate(bm=15, bn=64, bk=64),
        TileConfigCandidate(bm=256, bn=512, bk=128),
        TileConfigCandidate(bm=64, bn=128, bk=64),
        TileConfigCandidate(bm=128, bn=128, bk=64),
    ]
    ranked = prescreen_tile_configs(cands)
    assert len(ranked) == 2
    assert ranked[0].candidate.bm == 128

    compile_called = [False]

    def _forbid_compile(_src: str) -> StageBResult:
        compile_called[0] = True
        return StageBResult(passed=True)

    funnel = GateFunnel(compile_runner=_forbid_compile)
    res = funnel.evaluate(_OVERSIZED_UB_DSL, factory_name="make_kernel")
    assert res.passed is False
    assert res.failed_stage == "stage_a"
    assert compile_called[0] is False


def test_portfolio_runner_k2_rule_plus_creative_repair_and_two_lane_beam() -> None:
    """Verify K=2 Rule + 1 Rule-Blind Creative round, syntax repair, and two-lane beam."""
    stage_a_seed = run_stage_a_static(_SEED_DSL, factory_name="make_kernel")
    assert stage_a_seed.passed is True and stage_a_seed.ledger is not None
    seed_ledger = stage_a_seed.ledger
    seed_residual = compute_static_dynamic_residual(
        seed_ledger,
        {"dur_us": 40.0, "aic_mad": 0.45, "aiv_vec": 0.10, "aiv_mte2": 0.25},
    )
    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="developer",
        ledger=seed_ledger,
        residual=seed_residual,
        latency_us=40.0,
        accuracy_passed=True,
        features=stage_a_seed.features,
    )

    broken_syntax = "def broken(:\n    pass"
    creative_json = json.dumps(
        {
            "physical_hypothesis": "Move cross-core metadata into 3 KiB SSBUF",
            "target_local_metric": "sync_stall_ratio",
            "dsl_source": _CREATIVE_SSBUF_DSL,
        }
    )
    mock_backend = MockLLMBackend(
        responses={
            "exploit_rule": [broken_syntax, _SEED_DSL],
            "syntax_repair": [_SEED_DSL],
            "explore_creative": [creative_json],
        }
    )

    def _mock_device(dsl: str, st_a) -> StageCResult:
        lat = 28.0 if "alloc_ss" not in dsl else 34.0
        tel = {"dur_us": lat, "aic_mad": 0.72, "aiv_vec": 0.20, "aiv_mte2": 0.30}
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=lat,
            residual=compute_static_dynamic_residual(st_a.ledger, tel),
            telemetry=tel,
            run_id="mock_run",
        )

    tracker = BayesianRuleTracker.from_seed_yaml()
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_mock_device,
    )
    beam = TwoLaneBeamSearch(exploit_width=2, stepping_stone_width=2)
    runner = PortfolioRunner(
        backend=mock_backend,
        funnel=funnel,
        tracker=tracker,
        beam=beam,
        k_rule_workers=2,
        max_syntax_repairs=1,
    )

    report = asyncio.run(runner.run_round(parent, round_idx=1, factory_name="make_kernel"))
    assert len(report.spawned_workers) == 3
    assert len(report.evaluated_candidates) == 3
    assert report.repaired_worker_ids == ["r1_rule_0"]

    creative_reqs = [c for c in mock_backend.calls if c.role == "explore_creative"]
    assert len(creative_reqs) == 1
    creative_prompt = creative_reqs[0].system_prompt + "\n" + creative_reqs[0].user_prompt
    for rule in tracker.active_rules():
        assert rule.rule_id not in creative_prompt

    assert mock_backend.routed_models.count("mock-explore") == 1
    assert mock_backend.routed_models.count("mock-exploit") == 2
    assert mock_backend.routed_models.count("mock-repair") == 1

    assert report.exploit_beam[0].latency_us == 28.0
    assert any("T.alloc_ss" in c.structural_signature for c in report.stepping_stone_beam)


def _make_legal_ledger(
    *,
    ii: int | None = 1000,
    pipe_cycles: dict[str, int] | None = None,
):
    pipes = pipe_cycles if pipe_cycles is not None else {"AIC.CUBE": 1000, "AIV0.PIPE_V": 900}
    return build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3" if ii is not None else "readonly_envelope",
        is_legal=True,
        steady_state_ii=ii,
        pipe_cycles=pipes,
        waterlines={"UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"buf": 65536},
            implicit_buffers={},
        )},
    )


def test_residual_balanced_multipipe_and_telemetry_regimes() -> None:
    """Verify balanced multi-pipe predicted_util and zero fabricated gap in proxy mode."""
    balanced_pipes = {
        "AIC.CUBE": 1000,
        "AIV0.PIPE_V": 950,
        "AIV1.PIPE_V": 950,
        "AIC.PIPE_MTE2": 900,
    }
    ledger_ii = _make_legal_ledger(ii=1050, pipe_cycles=balanced_pipes)

    proxy_res = compute_static_dynamic_residual(ledger_ii, {"dur_us": 25.0})
    assert abs(proxy_res.predicted_bottleneck_util - round(1000.0 / 1050.0, 6)) < 1e-6
    assert proxy_res.unexplained_gap_ratio == 0.0

    proxy_sync = compute_static_dynamic_residual(
        ledger_ii,
        {"dur_us": 25.0, "sync_stall_ratio": 0.3},
    )
    assert abs(proxy_sync.unexplained_gap_ratio - 0.15) < 1e-6

    eta_res = compute_static_dynamic_residual(
        ledger_ii,
        {"dur_us": 25.0, "roofline_eta": 0.65},
    )
    expected_eta_gap = round((1000.0 / 1050.0) - 0.65, 6)
    assert abs(eta_res.unexplained_gap_ratio - expected_eta_gap) < 1e-5

    pmu_res = compute_static_dynamic_residual(
        ledger_ii,
        {"dur_us": 25.0, "aic_fixpipe": 0.8},
    )
    expected_pmu_gap = round((1000.0 / 1050.0) - 0.8, 6)
    assert abs(pmu_res.unexplained_gap_ratio - expected_pmu_gap) < 1e-5

    ledger_no_ii = _make_legal_ledger(ii=None, pipe_cycles=balanced_pipes)
    no_ii_res = compute_static_dynamic_residual(ledger_no_ii, {"dur_us": 25.0})
    assert abs(no_ii_res.predicted_bottleneck_util - round(1000.0 / 1700.0, 6)) < 1e-6
    assert no_ii_res.unexplained_gap_ratio == 0.0


def test_two_lane_beam_novelty_preservation_and_roundtrip() -> None:
    """Verify stepping-stone novelty persists until selected and beam/features roundtrip."""
    import pytest  # noqa: PLC0415

    ledger = _make_legal_ledger(ii=500)
    res = compute_static_dynamic_residual(ledger, {"dur_us": 20.0})
    feat = StructuralFeatures(
        phi_1_static_affine_bounds=False,
        phi_2_cube_to_vector_ratio=1.25,
        phi_3_quant_granularity="per_token_or_block",
        phi_4_indirect_sparse_gather=True,
        phi_5_cross_iter_recurrence="associative_online_rescale",
        phi_6_dynamic_cross_core_handoff=True,
        phi_7_inner_contiguous_bytes=64,
        phi_8_persistent_wave_occupancy=2.0,
    )
    short_payload = {
        "phi_1": False,
        "phi_2": 1.25,
        "phi_3": "per_token_or_block",
        "phi_4": True,
        "phi_5": "associative_online_rescale",
        "phi_6": True,
        "phi_7": 64,
        "phi_8": 2.0,
    }
    long_payload = {
        "static_affine_bounds": False,
        "cube_to_vector_ratio": 1.25,
        "quant_granularity": "per_token_or_block",
        "indirect_sparse_gather": True,
        "cross_iter_recurrence": "associative_online_rescale",
        "dynamic_cross_core_handoff": True,
        "inner_contiguous_bytes": 64,
        "persistent_wave_occupancy": 2.0,
    }
    assert StructuralFeatures.from_dict(short_payload) == feat
    assert StructuralFeatures.from_dict(long_payload) == feat
    assert StructuralFeatures.from_dict(feat.to_dict()) == feat

    def _cand(cid: str, rnd: int, lat: float, sig: tuple[str, ...]) -> BeamCandidate:
        return BeamCandidate(
            candidate_id=cid,
            round_idx=rnd,
            worker_kind="explore",
            rule_id=None,
            dsl_source=_SEED_DSL,
            track="developer",
            ledger=ledger,
            residual=res,
            latency_us=lat,
            accuracy_passed=True,
            structural_signature=sig,
            features=feat,
        )

    beam = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    sig_exploit = ("track:developer",)
    sig_stone = ("T.alloc_ss", "track:manual_ssbuf")
    sig_fresh = ("T.SimtVF", "track:simt_vf")

    beam.update([
        _cand("c_fast", 1, 10.0, sig_exploit),
        _cand("c_stone", 1, 30.0, sig_stone),
    ])
    assert sig_exploit in beam.exploited_signatures
    assert sig_stone not in beam.exploited_signatures
    assert [c.candidate_id for c in beam.stepping_stone_beam] == ["c_stone"]

    beam.update([_cand("c_exploited_alt", 2, 15.0, sig_exploit)])
    assert [c.candidate_id for c in beam.stepping_stone_beam] == ["c_stone"]

    beam.mark_parent_selected(beam.stepping_stone_beam[0])
    assert sig_stone in beam.exploited_signatures
    beam.update([_cand("c_fresh", 3, 50.0, sig_fresh)])
    assert [c.candidate_id for c in beam.stepping_stone_beam] == ["c_fresh"]

    roundtrip = TwoLaneBeamSearch.from_dict(json.loads(json.dumps(beam.to_dict())))
    assert roundtrip.exploit_width == 1
    assert roundtrip.stepping_stone_width == 1
    assert roundtrip.exploited_signatures == beam.exploited_signatures
    assert roundtrip.exploit_beam[0].features == feat
    assert roundtrip.stepping_stone_beam[0].candidate_id == "c_fresh"
    assert roundtrip.exploit_beam[0].ledger.to_dict() == ledger.to_dict()
    assert pytest.approx(roundtrip.exploit_beam[0].residual.predicted_bottleneck_util) == (
        res.predicted_bottleneck_util
    )


def test_prescreen_neighborhood_summary_and_dsl_tile_kwargs_cliff_rejection() -> None:
    """Verify aligned neighborhood rules, summary formatting, and Stage A cliff rejection."""
    import pytest  # noqa: PLC0415

    center = TileConfigCandidate(bm=64, bn=64, bk=64, stages=2, aiv_split=2)
    neighbors = generate_aligned_neighborhood(center, m_max=96, n_max=80, k_max=64)
    assert center in neighbors
    assert all(
        c.bm > 0
        and c.bn > 0
        and c.bk > 0
        and c.stages >= 1
        and c.bm <= 96
        and c.bn <= 80
        and c.bk <= 64
        and c.bm % 16 == 0
        and c.bn % 16 == 0
        and c.bk % 16 == 0
        and (c.bm // 16) % 2 == 0
        for c in neighbors
    )

    scores = prescreen_tile_configs(neighbors, top_k=3)
    summary = format_prescreen_summary(scores)
    assert "Tile Prescreen Summary" in summary
    assert "bm=" in summary
    assert format_prescreen_summary([]) == "No feasible tile configurations found."

    param_dsl = """
import tilelang.ascend.language as T

def make_kernel(bm: int = 128, bn: int = 128, bk: int = 64, num_stages: int = 2):
    @T.prim_func
    def main(
        X: T.Tensor((512, 128), "bfloat16"),
        W: T.Tensor((256, 128), "bfloat16"),
        Y: T.Tensor((512, 256), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((bm, bk), "bfloat16")
            w_l1 = T.alloc_l1((bn, bk), "bfloat16")
            acc = T.alloc_l0c((bm, bn), "float32")
            for kt in T.Pipelined(2, num_stages=num_stages):
                T.copy(X[0:bm, kt * bk : (kt + 1) * bk], x_l1)
                T.copy(W[0:bn, kt * bk : (kt + 1) * bk], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:bm, 0:bn])
    return main
"""
    best_kwargs, best_stage_a = prescreen_dsl_tile_kwargs(
        param_dsl,
        factory_name="make_kernel",
        candidate_kwargs_list=[
            {"bm": 512, "bn": 256, "bk": 64, "num_stages": 2},
            {"bm": 128, "bn": 128, "bk": 64, "num_stages": 2},
            {"bm": 64, "bn": 64, "bk": 64, "num_stages": 2},
        ],
    )
    assert best_stage_a.passed is True
    assert best_kwargs["bm"] != 512
    assert best_kwargs == {"bm": 64, "bn": 64, "bk": 64, "num_stages": 2}

    with pytest.raises(ValueError):
        prescreen_dsl_tile_kwargs(
            param_dsl,
            factory_name="make_kernel",
            candidate_kwargs_list=[{"bm": 512, "bn": 256, "bk": 64, "num_stages": 2}],
        )


def test_fp3_unmatched_rules_not_padded_and_transport_error_exempt() -> None:
    """Verify phi_2 matching, zero padding trials on unmatched rules, and transport exemption."""
    stage_a_seed = run_stage_a_static(_SEED_DSL, factory_name="make_kernel")
    assert stage_a_seed.passed and stage_a_seed.ledger is not None
    seed_res = compute_static_dynamic_residual(stage_a_seed.ledger, {"dur_us": 40.0})
    cube_parent = BeamCandidate(
        candidate_id="cube_seed",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="developer",
        ledger=stage_a_seed.ledger,
        residual=seed_res,
        latency_us=40.0,
        accuracy_passed=True,
        features=stage_a_seed.features,
    )

    tracker = BayesianRuleTracker.from_seed_yaml()
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=30.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 30.0}),
            telemetry={"dur_us": 30.0},
        ),
    )
    default_runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=funnel,
        tracker=tracker,
        k_rule_workers=2,
    )
    default_specs = default_runner.plan_workers(
        cube_parent,
        1,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64},
    )
    default_creative = next(s for s in default_specs if s.worker_kind == "creative_explore")
    assert "### TILE PRESCREEN GUIDANCE" not in default_creative.request.user_prompt

    runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=funnel,
        tracker=tracker,
        k_rule_workers=2,
        enable_tilesize_prescreen=True,
    )
    specs_cube = runner.plan_workers(
        cube_parent,
        1,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64},
    )
    matched_ids = [s.rule.rule_id for s in specs_cube if s.rule is not None]
    assert any(rid in matched_ids for rid in (
        "R02_ONCHIP_L0C_TO_UB_DIRECT",
        "R03_ACCUM_SPAN_FIXPIPE_RELIEF",
        "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION",
        "R13_SPLIT_KV_WAVE_RECOVERY",
        "R22_HALF_L1_BANK_ISOLATION",
        "R26_SWAT_SNAKE_AND_TAIL_WAVE_SPLIT",
    ))
    creative_specs = [s for s in specs_cube if s.worker_kind == "creative_explore"]
    assert len(creative_specs) == 1
    creative_text = (
        creative_specs[0].request.system_prompt + "\n" + creative_specs[0].request.user_prompt
    )
    assert "- Negative Structural Exclusions (do NOT duplicate):" in creative_text
    assert "### TILE PRESCREEN GUIDANCE" in creative_text
    for rule in tracker.active_rules():
        assert rule.rule_id not in creative_text

    sparse_ledger = _make_legal_ledger(
        ii=None, pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120}
    )
    sparse_parent = BeamCandidate(
        candidate_id="sparse_seed",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="developer",
        ledger=sparse_ledger,
        residual=compute_static_dynamic_residual(sparse_ledger, {"dur_us": 20.0}),
        latency_us=20.0,
        accuracy_passed=True,
        features=StructuralFeatures(
            phi_1_static_affine_bounds=True,
            phi_2_cube_to_vector_ratio=0.0,
            phi_3_quant_granularity="none",
            phi_4_indirect_sparse_gather=False,
            phi_5_cross_iter_recurrence="none",
            phi_6_dynamic_cross_core_handoff=False,
            phi_7_inner_contiguous_bytes=128,
            phi_8_persistent_wave_occupancy=0.0,
        ),
    )
    creative_json = json.dumps(
        {
            "physical_hypothesis": "Explore vector fusion",
            "target_local_metric": "aiv_vec",
            "dsl_source": _CREATIVE_SSBUF_DSL,
        }
    )
    sparse_backend = MockLLMBackend(
        responses={"explore_creative": [creative_json, creative_json, creative_json]}
    )
    sparse_runner = PortfolioRunner(
        backend=sparse_backend,
        funnel=funnel,
        tracker=tracker,
        k_rule_workers=2,
    )
    sparse_report = asyncio.run(
        sparse_runner.run_round(sparse_parent, round_idx=1, factory_name="make_kernel")
    )
    assert len(sparse_report.spawned_workers) == 3
    assert all(w.worker_kind == "creative_explore" for w in sparse_report.spawned_workers)
    assert sparse_report.pruned_rule_ids == ()
    for rid, stats in tracker.posteriors.items():
        assert stats.trials == 0, f"unmatched rule {rid} received unexpected trial"

    class _FailingBackend:
        async def generate(self, req, *, effort_override=None):
            raise RuntimeError("503 upstream capacity timeout")

    fail_tracker = BayesianRuleTracker.from_seed_yaml()
    fail_runner = PortfolioRunner(
        backend=_FailingBackend(),
        funnel=funnel,
        tracker=fail_tracker,
        k_rule_workers=2,
    )
    fail_report = asyncio.run(
        fail_runner.run_round(cube_parent, round_idx=1, factory_name="make_kernel")
    )
    assert len(fail_report.failed_candidates) == 3
    assert all(stage == "llm_transport" for _wid, stage, _err in fail_report.failed_candidates)
    for rid, stats in fail_tracker.posteriors.items():
        assert stats.trials == 0, f"transport error penalized rule {rid}"


def test_stage_a_track_router_disabled_by_default_and_primitive_menu(monkeypatch) -> None:
    """Verify router disabled by default, signature tokens, and unconstrained prompt menu."""
    import tilefoundry_ascend.agent.router.capability_gate as cap_gate  # noqa: PLC0415

    calls: list[str] = []

    def _boom_extract(*_args, **_kwargs):
        calls.append("extract")
        raise AssertionError("capability_gate must not run when enable_track_router=False")

    def _boom_route(*_args, **_kwargs):
        calls.append("route")
        raise AssertionError("capability_gate must not run when enable_track_router=False")

    monkeypatch.setattr(cap_gate, "extract_structural_features", _boom_extract)
    monkeypatch.setattr(cap_gate, "route_execution_track", _boom_route)

    st_default = run_stage_a_static(_SEED_DSL, factory_name="make_kernel")
    assert st_default.passed is True
    assert st_default.features is None
    assert st_default.route is None
    assert calls == []

    monkeypatch.undo()
    st_enabled = run_stage_a_static(
        _SEED_DSL, factory_name="make_kernel", enable_track_router=True
    )
    assert st_enabled.passed is True
    assert st_enabled.features is not None
    assert st_enabled.route is not None

    sig = extract_structural_signature(
        "with T.MixedKernel(28, sids=2):\n"
        "    for kt in T.Pipelined(2, num_stages=2):\n"
        "        with T.SimdVF():\n"
        "            T.set_atomic('add')\n"
        "pass_configs = PURE_SIMDVF_UBUF_FLAGS\n",
        track="unconstrained",
    )
    assert all(not item.startswith("track:") for item in sig)
    for expected_tok in (
        "T.MixedKernel",
        "T.Pipelined",
        "T.SimdVF",
        "PURE_SIMDVF_UBUF_FLAGS",
        "T.set_atomic",
    ):
        assert expected_tok in sig

    ledger = _make_legal_ledger(ii=4, pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 64})
    res = compute_static_dynamic_residual(ledger, {"dur_us": 20.0})
    unconstrained_parent = BeamCandidate(
        candidate_id="seed_unconstrained",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="",
        ledger=ledger,
        residual=res,
        latency_us=20.0,
        accuracy_passed=True,
    )
    assert unconstrained_parent.track == "unconstrained"

    rules = load_seed_rules()
    exp_req = build_exploit_request(unconstrained_parent, rules[0])
    assert "### EXECUTION TRACK" not in exp_req.user_prompt
    assert "### TARGET TRACK" not in exp_req.user_prompt
    assert "Routed Execution Track:" not in exp_req.user_prompt
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in exp_req.user_prompt
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in exp_req.user_prompt

    cre_req = build_creative_request(unconstrained_parent)
    assert "### EXECUTION TRACK" not in cre_req.user_prompt
    assert "### TARGET TRACK" not in cre_req.user_prompt
    assert "Routed Execution Track:" not in cre_req.user_prompt
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in cre_req.user_prompt
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in cre_req.user_prompt


def test_dual_accumulator_analytical_and_live_ir_prescreen() -> None:
    """Verify dual-accumulator L0C overflow rejection and live-IR prescreen dry-run."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    bad_cand = TileConfigCandidate(
        bm=128,
        bn=64,
        bk=64,
        stages=2,
        d_out=512,
        dual_accumulator=True,
    )
    bad_score = evaluate_tile_candidate(bad_cand, facts)
    assert bad_score.is_feasible is False
    assert bad_score.l0c_bytes == 294912
    assert bad_score.rejection_reason == "L0C_overflow:294912>262144"

    good_cand = TileConfigCandidate(
        bm=64,
        bn=64,
        bk=64,
        stages=2,
        d_out=512,
        dual_accumulator=True,
    )
    good_score = evaluate_tile_candidate(good_cand, facts)
    assert good_score.is_feasible is True
    assert good_score.l0c_bytes == 147456
    assert good_score.ub_bytes_per_aiv == 131072

    fixed_bn_dual_dsl = """
import tilelang.ascend.language as T

def make_fixed_bn(bm: int = 128, d_nope: int = 512):
    @T.prim_func
    def main(A: T.Tensor((128, 64), "float16"), B: T.Tensor((64, 64), "float16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((bm, 64), "float16")
            b_l1 = T.alloc_l1((64, 64), "float16")
            s_l0c = T.alloc_l0c((bm, 64), "float32")
            o_l0c = T.alloc_l0c((bm, d_nope), "float32")
            T.gemm(a_l1, b_l1, s_l0c, transpose_B=True, clear_accum=True)
            T.gemm(a_l1, b_l1, o_l0c, transpose_B=True, clear_accum=True)
    return main
"""
    analytical_summary = build_prescreen_summary_from_kwargs(
        {"bm": 128, "bn": 64, "bk": 64, "d_nope": 512},
        dsl_source=fixed_bn_dual_dsl,
        factory_name="make_fixed_bn",
    )
    assert "REJECT(L0C_overflow:294912>262144)" in analytical_summary
    assert "| OK |" in analytical_summary
    assert "bm=128, bn=64, bk=64, stages=2, aiv_split=2, d_out=512 | OK" not in analytical_summary

    analytical_ok_summary = build_prescreen_summary_from_kwargs(
        {"bm": 64, "bn": 64, "bk": 64, "d_nope": 512},
        dsl_source=fixed_bn_dual_dsl,
        factory_name="make_fixed_bn",
    )
    assert "bm=64," in analytical_ok_summary
    assert "| OK |" in analytical_ok_summary

    live_dual_dsl = """
import tilelang.ascend.language as T

def make_dual(bm: int = 128, bn: int = 64, bk: int = 64, num_stages: int = 2):
    m_rows = T.dynamic("m_rows")
    @T.prim_func
    def main(
        X: T.Tensor((m_rows, 512), "float16"),
        W: T.Tensor((512, 512), "float16"),
        Y: T.Tensor((m_rows, 512), "float32"),
    ):
        with T.Kernel(T.ceildiv(m_rows, bm)) as bx:
            x_l1 = T.alloc_l1((bm, bk), "float16")
            w_l1 = T.alloc_l1((bn, bk), "float16")
            s_l0c = T.alloc_l0c((bm, bn), "float32")
            o_l0c = T.alloc_l0c((bm, 512), "float32")
            for kt in T.Pipelined(2, num_stages=num_stages):
                T.copy(X[bx * bm : bx * bm + bm, kt * bk : (kt + 1) * bk], x_l1)
                T.copy(W[0:bn, kt * bk : (kt + 1) * bk], w_l1)
                T.gemm(x_l1, w_l1, s_l0c, transpose_B=True, clear_accum=kt == 0)
            T.copy(s_l0c, Y[bx * bm : bx * bm + bm, 0:bn])
            T.copy(o_l0c, Y[bx * bm : bx * bm + bm, 0:512])
    return main
"""
    bindings = build_anchor_var_bindings(
        {"B": 1, "S": 512, "D_nope": 512},
        kwargs={"bm": 128, "bn": 64, "bk": 64},
    )
    assert bindings["m_rows"] == 512
    assert bindings["d_nope"] == 512
    live_summary = build_prescreen_summary_from_kwargs(
        {"bm": 128, "bn": 64, "bk": 64, "num_stages": 2, "d_nope": 512},
        dsl_source=live_dual_dsl,
        factory_name="make_dual",
        var_bindings=bindings,
    )
    assert "Live-IR Tile Prescreen Summary:" in live_summary
    assert "REJECT(L0C_overflow:294912>262144)" in live_summary
    assert "bm=64, bn=64" in live_summary
    assert "| OK |" in live_summary


def test_primitive_menu_avoids_t_scope_and_prescreen_handles_ar_and_mla_1d() -> None:
    assert "T.Scope(" not in TILELANG_ASCEND_PRIMITIVE_MENU
    assert "do NOT import tilelang.language as T" in TILELANG_ASCEND_PRIMITIVE_MENU

    empty_vec_fallback = build_prescreen_summary_from_kwargs(
        {"tile_m": 1, "n_depth": 2, "d_dim": 4096},
        dsl_source="",
        factory_name="",
    )
    assert empty_vec_fallback == ""

    root = Path(__file__).resolve().parents[3]
    ar_src = (
        root / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
    ).read_text(encoding="utf-8")
    ar_bindings = build_anchor_var_bindings(
        {"N": 2, "B": 1, "S": 1, "D": 4096},
        kwargs={"n_depth": 2, "d_dim": 4096, "tile_m": 1},
    )
    ar_summary = build_prescreen_summary_from_kwargs(
        {
            "n_depth": 2,
            "d_dim": 4096,
            "tl_dtype": "bfloat16",
            "tile_m": 1,
            "schedule_mode": "tuned",
        },
        dsl_source=ar_src,
        factory_name="_build_ar_small_decode_func",
        var_bindings=ar_bindings,
        config_selector_name="_select_ar_small_decode_config",
        dims={"N": 2, "B": 1, "S": 1, "D": 4096},
    )
    assert "Live-IR Tile Prescreen Summary:" in ar_summary
    assert "tile_m=1" in ar_summary
    assert "unaligned_mmad_16x16x16_atom" not in ar_summary

    mla_1d_dsl = """
import tilelang.ascend.language as T

def _build_mla_prefill_func(bm=64):
    bn = 64
    @T.prim_func
    def main(
        Q: T.Tensor((128, 512), "float16"),
        K: T.Tensor((128, 512), "float16"),
        Y: T.Tensor((128, 512), "float32"),
    ):
        with T.Kernel(2):
            q_l1 = T.alloc_l1((bm, 64), "float16")
            k_l1 = T.alloc_l1((bn, 64), "float16")
            s_l0c = T.alloc_l0c((bm, bn), "float32")
            o_l0c = T.alloc_l0c((bm, 512), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(Q[0:bm, 0:64], q_l1)
                T.copy(K[0:bn, 0:64], k_l1)
                T.gemm(q_l1, k_l1, s_l0c, transpose_B=True, clear_accum=kt == 0)
            T.copy(s_l0c, Y[0:bm, 0:bn])
            T.copy(o_l0c, Y[0:bm, 0:512])
    return main
"""
    mla_summary = build_prescreen_summary_from_kwargs(
        {"bm": 64, "d_nope": 512, "tl_dtype": "float16"},
        dsl_source=mla_1d_dsl,
        factory_name="_build_mla_prefill_func",
    )
    assert "Live-IR Tile Prescreen Summary:" in mla_summary
    assert "bm=32, bn=64" in mla_summary
    assert "bm=64, bn=64" in mla_summary
    assert "bm=128, bn=64" in mla_summary
    assert "REJECT(L0C_overflow:294912>262144)" in mla_summary


def test_stage_a_binds_ascend_target_context_for_t_fill() -> None:
    from tilelang import tvm  # noqa: PLC0415

    assert tvm.target.Target.current(allow_none=True) is None
    t_fill_dsl = """
import tilelang.ascend.language as T

def make_fill_kernel():
    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                T.fill(a_ub, 0.0)
            T.copy(a_ub, Y[0:64])
    return main
"""
    res = run_stage_a_static(t_fill_dsl, factory_name="make_fill_kernel")
    assert res.passed, res.error_message
    assert res.ledger is not None


def test_envelope_inspector_binds_target_context_and_repair_menu():
    """Verify inspect_tilelang_primfunc and syntax repair handle target context and menu."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415

    from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (  # noqa: PLC0415
        build_syntax_repair_request,
    )
    from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc  # noqa: PLC0415

    if tvm.target.Target.current(allow_none=True) is None:
        with create_backend_context("ascend").target:

            @T.prim_func
            def _mixed_fill_main(
                X: T.Tensor((1, 64), "float32"),
                Y: T.Tensor((1, 64), "float32"),
            ):
                with T.MixedKernel(1, sids=2) as (bx, sid):
                    a_ub = T.alloc_shared((1, 64), "float32")
                    with T.SimdVF():
                        if bx == 0 and sid == 0:
                            T.fill(a_ub, 0.0)
                    if sid == 0:
                        T.copy(a_ub, Y[0:1, 0:64])

        fn = _mixed_fill_main
    assert tvm.target.Target.current(allow_none=True) is None
    inspection = inspect_tilelang_primfunc(fn)
    assert inspection.lowering_error is None, inspection.lowering_error
    assert inspection.ledger is not None

    req = build_syntax_repair_request("def f():\n    pass\n", "some error")
    assert "VALID TARGET DIALECT PRIMITIVES" in req.user_prompt
    callable_surface = req.user_prompt.split("Strictly Forbidden", 1)[0]
    assert "T.MixedKernel" in callable_surface
    assert "T.gemm(" in callable_surface
    assert "T.gemm_v0" not in callable_surface
    req_trackb = build_syntax_repair_request(
        "__global__ void f() {}", "some error", backend_kind="ascendc_trackb"
    )
    assert "asc_copy_gm2ub_align" in req_trackb.user_prompt
    assert "T.Kernel" not in req_trackb.user_prompt


def test_up2_pseudo_double_buffer_residual_and_wave_burst_prescreen() -> None:
    """Verify pseudo-double-buffer serialization detector and wave*burst prescreen ranking."""
    from tilefoundry_ascend.agent.orchestrator.residual_beam import (  # noqa: PLC0415
        compute_static_dynamic_residual,
    )
    from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (  # noqa: PLC0415
        TileConfigCandidate,
        prescreen_live_ir_neighborhood,
    )
    from tilefoundry_ascend.ascend.passes.physical_ledger import (  # noqa: PLC0415
        build_objective_physical_ledger,
    )

    dyn_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=1,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 48},
        waterlines={},
        uses_dynamic_offset_pipelining=True,
    )
    bad_res = compute_static_dynamic_residual(
        dyn_ledger,
        {"dur_us": 50.0, "aiv_vec": 0.25, "aiv_mte2": 0.41},
    )
    assert bad_res.pseudo_double_buffer_serialized is True
    assert bad_res.to_dict()["pseudo_double_buffer_serialized"] is True

    static_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 32},
        waterlines={},
        uses_dynamic_offset_pipelining=False,
    )
    good_res = compute_static_dynamic_residual(
        static_ledger,
        {"dur_us": 30.0, "aiv_vec": 0.50, "aiv_mte2": 0.20},
    )
    assert good_res.pseudo_double_buffer_serialized is False

    vec_dsl = (
        "import tilelang.ascend.language as T\n\n"
        "def make_vec_kernel(tile_m: int = 8):\n"
        "    m_rows = T.dynamic('m_rows')\n"
        "    @T.prim_func\n"
        "    def main(\n"
        "        X: T.Tensor((m_rows, 256), 'float16'),\n"
        "        Y: T.Tensor((m_rows, 256), 'float16'),\n"
        "    ):\n"
        "        with T.Kernel(56) as bx:\n"
        "            buf = T.alloc_shared((tile_m, 256), 'float16')\n"
        "            T.copy(X[bx * tile_m : (bx + 1) * tile_m, :], buf)\n"
        "            T.copy(buf, Y[bx * tile_m : (bx + 1) * tile_m, :])\n"
        "    return main\n"
    )
    summary = prescreen_live_ir_neighborhood(
        vec_dsl,
        factory_name="make_vec_kernel",
        center=TileConfigCandidate(bm=16, bn=64, bk=64, stages=1),
        var_bindings={"m_rows": 1792},
        max_variants=6,
    )
    assert "dma_burst_eff=" in summary
    assert "est_waves=" in summary
    assert "wave_burst_score=" in summary
    rank_1_line = next(line for line in summary.splitlines() if line.strip().startswith("1."))
    assert "tile_m=32" in rank_1_line


def test_portfolio_runner_threads_roofline_and_symptom_ranking() -> None:
    """Thread roofline into WorkerSpec and refresh residual even with precomputed roofline_eta."""
    from types import SimpleNamespace  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    roofline_obj = SimpleNamespace(
        min_gm_bytes=16384,
        compute_floor_us=10.0,
        ideal_roofline_us=10.0,
    )
    seed_stage_a = run_stage_a_static(_SEED_DSL, factory_name="make_kernel")
    assert seed_stage_a.passed and seed_stage_a.ledger is not None
    seed_ledger = build_objective_physical_ledger(
        analysis_mode=seed_stage_a.ledger.analysis_mode,
        is_legal=True,
        steady_state_ii=seed_stage_a.ledger.steady_state_ii,
        pipe_cycles=dict(seed_stage_a.ledger.pipe_cycles),
        waterlines=dict(seed_stage_a.ledger.waterlines),
        gm_roundtrip_bytes=16384,
        gm_traffic_scope="full_grid",
    )
    parent = BeamCandidate(
        candidate_id="roof_seed",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="developer",
        ledger=seed_ledger,
        residual=compute_static_dynamic_residual(
            seed_ledger, {"dur_us": 40.0}, roofline=roofline_obj
        ),
        latency_us=40.0,
        accuracy_passed=True,
        features=seed_stage_a.features,
    )
    class _SpyTracker(BayesianRuleTracker):
        def rank_applicable_rules(
            self,
            context,
            *,
            top_k: int = 2,
            topology_axes: object = None,
            runtime_backend: str | None = None,
            symptom_ranking=(),
        ):
            seen_rankings.append(tuple((str(f), float(s)) for f, s in symptom_ranking))
            return super().rank_applicable_rules(
                context,
                top_k=top_k,
                topology_axes=topology_axes,
                runtime_backend=runtime_backend,
                symptom_ranking=symptom_ranking,
            )

    tracker = _SpyTracker.from_seed_yaml()
    seen_rankings: list[tuple[tuple[str, float], ...]] = []
    creative_json = json.dumps(
        {
            "physical_hypothesis": "Keep roofline fresh",
            "target_local_metric": "aic_mac",
            "dsl_source": _SEED_DSL,
        }
    )
    backend = MockLLMBackend(
        responses={
            "apply_rule": [_SEED_DSL, _SEED_DSL],
            "explore_creative": [creative_json],
        }
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=20.0,
            residual=compute_static_dynamic_residual(
                st_a.ledger, {"dur_us": 20.0, "roofline_eta": 0.5}
            ),
            telemetry={"dur_us": 20.0, "roofline_eta": 0.5},
        ),
    )
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=tracker,
        k_rule_workers=2,
    )
    report = asyncio.run(
        runner.run_round(
            parent,
            round_idx=1,
            factory_name="make_kernel",
            ideal_roofline_us=10.0,
            roofline=roofline_obj,
        )
    )
    assert seen_rankings and len(seen_rankings[0]) > 0
    assert report.spawned_workers
    assert all(spec.roofline is roofline_obj for spec in report.spawned_workers)
    assert report.evaluated_candidates
    cand_res = report.evaluated_candidates[0].residual
    assert cand_res.effective_compute_eta == pytest.approx(0.5)
    assert cand_res.observation is not None
    assert cand_res.observation.transfer_inflation_ratio > 0.0



