"""Offline unit tests for WP6 counterfactual AST ablation distiller and KB admission."""

import json
from pathlib import Path

from tilefoundry_ascend.agent import BayesianRuleTracker, FlywheelDistiller, GateFunnel
from tilefoundry_ascend.agent.flywheel import (
    decompose_ast_ablations,
    synthesize_micro_probe,
)
from tilefoundry_ascend.agent.kb import load_seed_rules, safe_eval_predicate
from tilefoundry_ascend.agent.orchestrator import (
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    run_stage_a_static,
)

_BASELINE_DSL = """
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
            for kt in T.Pipelined(2, num_stages=1):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_WINNER_WITH_DECOY_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(8):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            T.ascend_pipe_barrier("PIPE_ALL")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""


def _mock_ablation_device(dsl_source: str, stage_a) -> StageCResult:
    assert stage_a.ledger is not None
    if "T.Kernel(8)" in dsl_source:
        lat = 42.0
    elif "num_stages=2" in dsl_source:
        lat = 78.0
    elif "ascend_pipe_barrier" in dsl_source:
        lat = 126.0
    else:
        lat = 110.0
    tel = {"dur_us": lat, "aic_mad": 0.65, "aiv_vec": 0.25, "aiv_mte2": 0.18}
    return StageCResult(
        passed=True,
        accuracy_passed=True,
        measured_latency_us=lat,
        residual=compute_static_dynamic_residual(stage_a.ledger, tel),
        telemetry=tel,
    )


def test_single_variable_ablation_and_negative_filtering(tmp_path: Path) -> None:
    """Verify AST diff splits multi-edit winner, drops negative edit, probes, and admits rules."""
    chunks = decompose_ast_ablations(_BASELINE_DSL, _WINNER_WITH_DECOY_DSL)
    kinds = {c.edit_kind for c in chunks}
    assert "core_parallelism" in kinds
    assert "pipeline_versions" in kinds
    assert "barrier_prune" in kinds
    for chunk in chunks:
        probe = synthesize_micro_probe(chunk)
        assert 25 <= probe.line_count_baseline <= 50
        assert 25 <= probe.line_count_optimized <= 50
        assert "from __future__ import annotations" not in probe.baseline_probe_source
        assert "from __future__ import annotations" not in probe.optimized_probe_source
        sa_base = run_stage_a_static(probe.baseline_probe_source, factory_name=probe.factory_name)
        sa_opt = run_stage_a_static(probe.optimized_probe_source, factory_name=probe.factory_name)
        assert sa_base.passed and sa_opt.passed

    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_count = len(tracker.rules)
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_mock_ablation_device,
    )
    out_yaml = tmp_path / "distilled_kb.yaml"
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=lambda _spec: {
            "baseline_dur_us": 40.0,
            "optimized_dur_us": 12.0,
            "numerical_ok": 1.0,
        },
        min_edit_gain=0.05,
        min_probe_speedup=1.2,
    )
    report = distiller.distill_from_pair(
        baseline_dsl=_BASELINE_DSL,
        winner_dsl=_WINNER_WITH_DECOY_DSL,
        baseline_latency_us=110.0,
        winner_latency_us=35.0,
        factory_name="make_kernel",
        kb_output_path=out_yaml,
    )

    assert len(report.kept_positive_chunks) == 2
    assert len(report.discarded_chunks) >= 1
    assert any(m.chunk.edit_kind == "barrier_prune" for m in report.discarded_chunks)
    assert len(report.admitted_rule_ids) == 2
    assert len(tracker.rules) == initial_count + 2

    reloaded = load_seed_rules(out_yaml)
    assert len(reloaded) == initial_count + 2
    admitted = [r for r in reloaded if r.rule_id in set(report.admitted_rule_ids)]
    for rule in admitted:
        assert rule.source_provenance == "flywheel:ablation"
        assert safe_eval_predicate(
            rule.regime_guard,
            {
                "static_affine_bounds": True,
                "indirect_sparse_gather": False,
                "inner_contiguous_bytes": 256,
                "dynamic_cross_core_handoff": False,
                "cube_to_vector_ratio": 1.0,
            },
        )
    summary = json.loads(report.to_json())
    assert summary["admitted_rule_ids"] == list(report.admitted_rule_ids)
