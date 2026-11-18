"""End-to-end readiness tests for Marathon live SIL wiring, Carbon effort, and 60-op seeding."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
    default_source_root,
    get_default_regime_specs,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    ensure_operator_seed_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.kb import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    GenerationRequest,
    MockLLMBackend,
    is_carbon_model,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)


def _cannbot_ops_root() -> Path:
    return default_source_root().parent / "cannbot-ops" / "operators"


def test_campaign_runner_live_sil_7facet_cost_calibration_and_shared_stone_pruning(
    tmp_path: Path,
) -> None:
    """Verify live SIL recomputation, 7-facet prompts, compiler hazards, cost calibration, and sibling pruning."""
    mod_file = tmp_path / "demo_op.py"
    initial_src = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 64, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 64, tile_n: int = 64):\n"
        "    return ('v0', tile_m, tile_n)\n\n"
        "def demo_op(x):\n"
        "    return _build_demo_prim_func(), x\n"
    )
    mod_file.write_text(initial_src, encoding="utf-8")

    r1 = RegimeTargetSpec(
        regime_tag="r1_main",
        builder_candidates=("_build_demo_prim_func",),
        sample_builder_kwargs={"tile_m": 64, "tile_n": 64},
        config_selector_fn="_select_demo_config",
        editable_symbols=("_select_demo_config", "_build_demo_prim_func"),
        representative_hir_dims={"M": 128, "N": 128},
        representative_anchor_dims={"M": 128, "N": 128},
    )
    r2 = RegimeTargetSpec(
        regime_tag="r2_tail",
        builder_candidates=("_build_demo_prim_func",),
        sample_builder_kwargs={"tile_m": 32, "tile_n": 64},
        config_selector_fn="_select_demo_config",
        editable_symbols=("_select_demo_config", "_build_demo_prim_func"),
        representative_hir_dims={"M": 32, "N": 64},
        representative_anchor_dims={"M": 32, "N": 64},
    )
    anchors = [
        AnchorCaseSpec("level1/demo_op", 1, "r1_main", 100.0, 20.0),
        AnchorCaseSpec("level1/demo_op", 2, "r2_tail", 100.0, 20.0),
    ]

    overflow_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=253952,
                explicit_buffers={"x": 200000},
                implicit_buffers={"extra_tmp": 120000},
            ),
        },
        gm_roundtrip_bytes=131072,
        dominant_contig_bytes=32768,
        grid_blocks=56,
    )

    improved_shared_slice = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 32, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 32, tile_n: int = 64):\n"
        "    return ('v2_promoted', tile_m, tile_n)\n"
    )
    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Shrink UB tile footprint and eliminate register spill",
            "target_local_metric": "dur_us",
            "dsl_source": improved_shared_slice,
        }
    )

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [improved_shared_slice, improved_shared_slice],
            "explore_creative": [creative_reply, creative_reply, creative_reply, creative_reply],
        }
    )

    comp_hazards = BishengResourceSummary(
        spill_bytes=64,
        vloopv2_loops=0,
        software_loops=2,
        vf_fusion_breaks=2,
        alias_serialized_buffers=1,
        dual_issue_ipc_collapse=0.5,
        dcache_dcci_flushes=1,
        has_local_membar_in_vf_call=True,
    )

    def _l2_eval(cands, ancs):
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, _p in cands:
            lat = 50.0 if cid == "incumbent_baseline" else 22.0
            out[cid] = aggregate_surrogate_result(
                cid,
                "demo_op",
                [build_anchor_metrics(a, accuracy_passed=True, cand_us=lat) for a in ancs],
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=1,
            k_rule_workers=1,
            stagnation_escalate_rounds=1,
            exploit_effort="high",
        ),
        target_module_path=mod_file,
        perf_target=PerfTargetSpec("demo_op", 4.0, 3.5, 0.95),
        anchors=anchors,
        regime_specs=[r1, r2],
        llm_backend=backend,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        l0_checker=lambda _src, _reg: StageAResult(
            passed=True,
            prim_func=None,
            ledger=overflow_ledger,
        ),
        l1_compiler=lambda _src, _reg: StageBResult(
            passed=True,
            spill_bytes=comp_hazards.spill_bytes,
            vloopv2_hit=comp_hazards.vloopv2_hit,
            compiler_summary=comp_hazards,
        ),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        source_root=tmp_path,
    )

    # Seed a stale pre-promotion stepping-stone in sibling regime r2_tail sharing _build_demo_prim_func
    stale_stone = BeamCandidate(
        candidate_id="stale_r2_stone",
        round_idx=0,
        worker_kind="creative_explore",
        rule_id=None,
        dsl_source=(
            "def _build_demo_prim_func(tile_m: int = 16, tile_n: int = 32):\n"
            "    return ('stale_old', tile_m, tile_n)\n"
        ),
        track="developer",
        ledger=overflow_ledger,
        residual=compute_static_dynamic_residual(overflow_ledger, {"dur_us": 45.0}),
        latency_us=45.0,
        accuracy_passed=True,
    )
    runner._beams["r2_tail"].stepping_stone_beam = [stale_stone]

    # Inject a stepping stone in r1_main with compiler hazards & 0.0 PMU ratios, and run an escalated round
    hazard_residual = StaticDynamicResidual(
        measured_latency_us=50.0,
        predicted_bottleneck_util=0.8,
        measured_aic_mac_ratio=0.0,
        measured_aiv_vec_ratio=0.0,
        measured_mte2_wait_ratio=0.0,
        measured_sync_stall_ratio=0.0,
        unexplained_gap_ratio=0.0,
        spill_bytes=64,
        vf_fusion_breaks=2,
        alias_serialized_buffers=1,
        dual_issue_ipc_collapse=0.5,
        dcache_dcci_flushes=1,
        has_local_membar_in_vf_call=True,
        vloopv2_hit=False,
    )
    initial_slice = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 64, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 64, tile_n: int = 64):\n"
        "    return ('v0', tile_m, tile_n)\n"
    )
    r1_stone = BeamCandidate(
        candidate_id="r1_hazard_stone",
        round_idx=0,
        worker_kind="creative_explore",
        rule_id=None,
        dsl_source=initial_slice,
        track="developer",
        ledger=overflow_ledger,
        residual=hazard_residual,
        latency_us=50.0,
        accuracy_passed=True,
    )
    runner._beams["r1_main"].stepping_stone_beam = [r1_stone]

    base_res = runner._evaluate_single_baseline(initial_src, anchors)
    (
        rec,
        new_incumbent,
        _new_best,
        _stag_after,
        promoted,
        _rolled_back,
        escalated,
    ) = asyncio.run(
        runner._execute_single_round(
            round_idx=1,
            regime_spec=r1,
            incumbent_source=initial_src,
            best_result=base_res,
            active_anchors=anchors,
            stagnation_rounds=1,
            admitted_rule_ids=[],
        )
    )

    assert escalated is True
    assert promoted is True
    assert rec.promoted is True
    assert "v2_promoted" in new_incumbent

    # Both exploit and creative prompts received SIL live policy + 7-facet envelope + compiler hazards
    exploit_reqs = [c for c in backend.calls if c.role == "exploit_rule"]
    creative_reqs = [c for c in backend.calls if c.role == "explore_creative"]
    assert exploit_reqs and creative_reqs
    for req in (*exploit_reqs, *creative_reqs):
        assert "### SIL LIVE POLICY & 7-FACET CONSTRAINT ENVELOPE" in req.user_prompt
        assert "tilelang_7pipe_isa_coverage" in req.user_prompt
        assert "### HARDWARE MSPROF TELEMETRY" in req.user_prompt
        assert "spill_bytes: 64" in req.user_prompt
        assert "vloopv2_hit: False" in req.user_prompt
        assert "vf_fusion_breaks: 2" in req.user_prompt
    assert exploit_reqs[0].effort == "high"

    # Active mask relaxed both UB/SRAM layout and exec_mode_and_spr + pipeline_and_sync_graph
    mask = runner._freedom_masks["r1_main"]
    assert mask.sram_layout_and_aliasing is True
    assert mask.exec_mode_and_spr is True
    assert mask.pipeline_and_sync_graph is True

    # Online Ridge surrogate calibrated from positive L2 latencies
    assert runner._cost_model.sample_count > 0

    # Stale pre-promotion stepping-stone in sibling r2_tail sharing _build_demo_prim_func was pruned
    assert runner._beams["r2_tail"].stepping_stone_beam == []


def test_carbon_high_effort_defaults_and_xhigh_guardrail(tmp_path: Path) -> None:
    """Verify is_carbon_model high effort default, non-Carbon exploit_rule medium default, and MarathonConfig."""
    assert is_carbon_model("carbon") is True
    assert is_carbon_model("gemini-pro-orchestrator") is True
    assert is_carbon_model("gemini-pro-orchestrator-xhigh") is True
    assert is_carbon_model("gemini-4-hydrogen-medium") is False

    fake_agy = tmp_path / "fake_agy"
    fake_agy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_agy.chmod(0o755)

    carbon_backend = AgyCliLLMBackend(
        bin_path=str(fake_agy),
        exploit_model="gemini-pro-orchestrator",
    )
    assert carbon_backend.select_effort(GenerationRequest("exploit_rule", "s", "u")) == "high"
    assert (
        carbon_backend.select_effort(GenerationRequest("exploit_rule", "s", "u", effort="xhigh"))
        == "high"
    )

    hydrogen_backend = AgyCliLLMBackend(bin_path=str(fake_agy))
    assert hydrogen_backend.select_effort(GenerationRequest("exploit_rule", "s", "u")) == "medium"

    m_cfg = MarathonConfig(operator_name="rms_norm")
    assert m_cfg.exploit_effort == "high"
    assert m_cfg.auto_bootstrap_seed is True

    c_cfg = CampaignConfig(operator_name="rms_norm")
    assert c_cfg.exploit_effort == "medium"


def test_ensure_operator_seed_module_bootstraps_apply_rotary_pos_emb_and_passes_l0(
    tmp_path: Path,
) -> None:
    """Verify ensure_operator_seed_module bootstraps apply_rotary_pos_emb and passes _run_l0."""
    if not _cannbot_ops_root().is_dir():
        pytest.skip("cannbot-ops repository not present on disk")

    seed_path = tmp_path / "examples" / "ascend950pr_ops" / "apply_rotary_pos_emb" / "kernels.py"
    created = ensure_operator_seed_module("apply_rotary_pos_emb", seed_path)
    assert created is True
    assert seed_path.is_file()
    assert (seed_path.parent / "__init__.py").is_file()

    # Never overwrite an existing module
    assert ensure_operator_seed_module("apply_rotary_pos_emb", seed_path) is False

    seed_src = seed_path.read_text(encoding="utf-8")
    assert "def _select_apply_rotary_pos_emb_config(" in seed_src
    assert "def _build_apply_rotary_pos_emb_prim_func(" in seed_src
    assert "def apply_rotary_pos_emb(" in seed_src

    # Verify Stage A / L0 passes on all auto-stratified regimes
    regimes = get_default_regime_specs("apply_rotary_pos_emb")
    assert regimes
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="apply_rotary_pos_emb", max_rounds=1),
        target_module_path=seed_path,
        source_root=tmp_path,
        l2_batch_evaluator=lambda _c, _a: {},
    )
    for rspec in regimes:
        stage_a = runner._run_l0(seed_src, rspec)
        assert stage_a.passed is True, f"Regime {rspec.regime_tag} failed L0: {stage_a.error_message}"

    # Verify golden delegation executes accurately via entrypoint
    mod_spec = importlib.util.spec_from_file_location("_tf_boot_rope", str(seed_path))
    assert mod_spec is not None and mod_spec.loader is not None
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = mod
    try:
        mod_spec.loader.exec_module(mod)
        q = torch.randn((1, 4, 2, 16), dtype=torch.float32)
        k = torch.randn((1, 4, 2, 16), dtype=torch.float32)
        cos = torch.ones((4, 8), dtype=torch.float32)
        sin = torch.zeros((4, 8), dtype=torch.float32)
        q_out, k_out = mod.apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_out, q)
        assert torch.allclose(k_out, k)
    finally:
        sys.modules.pop(mod_spec.name, None)
