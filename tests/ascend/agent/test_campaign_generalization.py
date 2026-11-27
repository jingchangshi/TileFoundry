"""Tests for 60-op CannbenchOpsRegistry, LOO ablation, and dim passthrough."""

from __future__ import annotations

from tilefoundry_ascend.agent.specs.anchor_specs import (
    CannbenchOpsRegistry,
    get_default_anchors,
    get_default_perf_target,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    build_arg_parser as build_campaign_arg_parser,
)
from tilefoundry_ascend.agent.campaign.cli import build_arg_parser as build_cli_arg_parser
from tilefoundry_ascend.agent.campaign.marathon import build_marathon_arg_parser
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    FlywheelDistiller,
    decompose_leave_one_out_ablations,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    _filter_kwargs_for_callable,
    _merge_dims_into_kwargs,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    _RULE_CONCEPT_DESCRIPTIONS,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def test_cannbench_ops_registry_discovers_all_60_ops_and_stratifies() -> None:
    """Verify CannbenchOpsRegistry discovers all 60 cann-bench tasks and stratifies 4-5 anchors."""
    reg = CannbenchOpsRegistry()
    ops = reg.list_operators()
    tasks = reg.list_tasks()
    assert len(ops) == 60
    assert len(tasks) == 60

    for op_id in ("exp", "swi_glu", "rms_norm", "q4_0_dequant_matmul", "grouped_matmul_swiglu_quant", "gqa"):
        spec = reg.resolve(op_id)
        assert 4 <= len(spec.anchors) <= 5
        assert len({a.case_num for a in spec.anchors}) == len(spec.anchors)
        tags = {a.regime_tag for a in spec.anchors}
        assert "small_latency" in tags
        assert "large_throughput" in tags
        assert spec.perf_target.target_surrogate_score >= 95.0
        assert spec.perf_target.per_anchor_max_us

    assert get_operator_task("rms_norm") == "level2/rms_norm"
    assert get_operator_task("level2/rms_norm") == "level2/rms_norm"
    assert get_operator_entrypoint("level2/rms_norm") == "rms_norm"
    assert get_operator_module_relpath("rms_norm") == "examples/ascend950pr_ops/rms_norm/kernels.py"
    assert len(get_default_anchors("level2/rms_norm")) == 5
    assert get_default_perf_target("rms_norm").target_surrogate_score >= 95.0

    cli_args = build_cli_arg_parser().parse_args(["--operator", "rms_norm", "--checkpoint", "ckpt.json"])
    assert cli_args.operator == "rms_norm"
    cr_args = build_campaign_arg_parser().parse_args(["--operator", "level3/q4_0_dequant_matmul"])
    assert cr_args.operator == "level3/q4_0_dequant_matmul"
    m_args = build_marathon_arg_parser().parse_args(["run", "--operator", "swi_glu"])
    assert m_args.operator == "swi_glu"


def test_arbitrary_dim_passthrough_and_r22_r33_descriptions() -> None:
    """Verify arbitrary dims pass through gate_funnel and R22-R33 descriptions exist."""
    def _custom_builder(m_dim: int, k_dim: int, hidden_dim: int, seq_len: int):
        return (m_dim, k_dim, hidden_dim, seq_len)

    merged = _merge_dims_into_kwargs({"extra": 1}, {"M": 64, "K": 128, "hidden_dim": 512, "seq_len": 32})
    filtered = _filter_kwargs_for_callable(_custom_builder, merged)
    assert filtered == {"m_dim": 64, "k_dim": 128, "hidden_dim": 512, "seq_len": 32}

    for r_idx in range(22, 34):
        prefix = f"R{r_idx}_"
        assert any(k.startswith(prefix) for k in _RULE_CONCEPT_DESCRIPTIONS), f"missing {prefix}"


def test_leave_one_out_ast_and_config_ablation_on_candidate_dsl() -> None:
    """Verify Leave-One-Out AST/config ablation directly on real candidate dsl_source."""
    base_dsl = """
import tilelang.ascend.language as T

def _select_cfg():
    return {"bm": 64, "num_stages": 1, "swizzle_window": 1}

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
    win_dsl = """
import tilelang.ascend.language as T

def _select_cfg():
    return {"bm": 64, "num_stages": 2, "swizzle_window": 4}

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(4):
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
    loo_chunks = decompose_leave_one_out_ablations(base_dsl, win_dsl)
    kinds = {c.edit_kind for c in loo_chunks}
    assert "core_parallelism" in kinds
    assert "config_knob" in kinds
    assert "pipeline_versions" in kinds

    def _batch_eval(items):
        out = {}
        for cid, dsl in items:
            if "T.Kernel(1)" in dsl:
                out[cid] = 80.0
            elif "'swizzle_window': 1" in dsl or '"swizzle_window": 1' in dsl:
                out[cid] = 50.0
            else:
                out[cid] = 20.0
        return out

    gemm_axes = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    tracker = BayesianRuleTracker.from_seed_yaml()
    distiller = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )
    report = distiller.distill_from_pair(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=80.0,
        winner_latency_us=20.0,
        factory_name="make_kernel",
        config_selector_name="_select_cfg",
        batch_latency_evaluator=_batch_eval,
        topology_axes=gemm_axes,
        verify_on_candidate=True,
    )
    assert len(report.kept_positive_chunks) >= 2
    assert len(report.admitted_rule_ids) >= 2
    assert any("CONFIG_KNOB" in rid for rid in report.admitted_rule_ids)
    assert "mr:gm_l1_l0_fixpipe" in tracker.axis_posteriors
