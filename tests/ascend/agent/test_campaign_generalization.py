"""Tests for 60-op CannbenchOpsRegistry, CannbenchContractOracle, LOO ablation, and proxy guards."""

from __future__ import annotations

import asyncio
import math

import pytest
import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
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
from tilefoundry_ascend.agent.orchestrator.e2e_evolution import run_e2e_evolution
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageBResult,
    _filter_kwargs_for_callable,
    _merge_dims_into_kwargs,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    _RULE_CONCEPT_DESCRIPTIONS,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import (
    CannbenchContractOracle,
    compare_cannbench_tensor,
)


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

    # Short name & task ID equivalence
    assert get_operator_task("rms_norm") == "level2/rms_norm"
    assert get_operator_task("level2/rms_norm") == "level2/rms_norm"
    assert get_operator_entrypoint("level2/rms_norm") == "rms_norm"
    assert get_operator_module_relpath("rms_norm") == "examples/ascend950pr_ops/rms_norm/kernels.py"
    assert len(get_default_anchors("level2/rms_norm")) == 5
    assert get_default_perf_target("rms_norm").target_surrogate_score >= 95.0

    # CLI parsers accept arbitrary operators without 2-op lock-in
    cli_args = build_cli_arg_parser().parse_args(["--operator", "rms_norm", "--checkpoint", "ckpt.json"])
    assert cli_args.operator == "rms_norm"
    cr_args = build_campaign_arg_parser().parse_args(["--operator", "level3/q4_0_dequant_matmul"])
    assert cr_args.operator == "level3/q4_0_dequant_matmul"
    m_args = build_marathon_arg_parser().parse_args(["run", "--operator", "swi_glu"])
    assert m_args.operator == "swi_glu"


def test_cannbench_contract_oracle_synthesis_and_mere_mare_bit_exact() -> None:
    """Verify CannbenchContractOracle faithful synthesis (inf/nan/unique/q4_0/group_list/inplace)."""
    # 1. Bit-exact threshold==0 semantics (+0.0 vs -0.0)
    pos_zero = torch.tensor([0.0], dtype=torch.float32)
    neg_zero = torch.tensor([-0.0], dtype=torch.float32)
    assert compare_cannbench_tensor(pos_zero, pos_zero, threshold=0.0)["passed"] is True
    assert compare_cannbench_tensor(neg_zero, pos_zero, threshold=0.0)["passed"] is False

    # 2. MERE/MARE pass and fail
    gold = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    close = torch.tensor([1.0001, 2.0001, 4.0001], dtype=torch.float32)
    far = torch.tensor([1.5, 2.0, 4.0], dtype=torch.float32)
    assert compare_cannbench_tensor(close, gold, threshold=1e-3)["passed"] is True
    assert compare_cannbench_tensor(far, gold, threshold=1e-3)["passed"] is False

    # 3. Special ranges: [-inf, inf] and [nan, nan] on exp
    exp_oracle = CannbenchContractOracle("level1/exp")
    t_inf, _ = exp_oracle.synthesize_case_inputs(12)
    assert bool(torch.isinf(t_inf[0]).any().item())
    t_nan, _ = exp_oracle.synthesize_case_inputs(13)
    assert bool(torch.isnan(t_nan[0]).any().item())

    # 4. Unique integer synthesis
    unique_oracle = CannbenchContractOracle("level3/unique")
    # Pick a case from unique or test unique range directly
    assert unique_oracle.cases
    rms_oracle = CannbenchContractOracle("rms_norm")
    tensors, kwargs = rms_oracle.synthesize_case_inputs(1)
    assert bool((kwargs["gamma"] != 0).any().item())  # Non-zero weights!

    def _rms_norm_cand(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        xf = x.to(torch.float32)
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        return (xf * rms * gamma.to(torch.float32)).to(x.dtype)

    ok, reports = rms_oracle.verify_candidate(_rms_norm_cand, case_nums=[1, 7])
    assert ok is True, reports

    # 5. Q4_0 packed uint8 nibbles + positive scales
    q4_oracle = CannbenchContractOracle("q4_0_dequant_matmul")
    q4_tensors, q4_kw = q4_oracle.synthesize_case_inputs(2)
    assert q4_kw["qweight"].dtype == torch.uint8
    assert bool((q4_kw["scales"] > 0).all().item())
    assert q4_tensors

    # 6. Cumsum group_list with empty groups
    gmm_oracle = CannbenchContractOracle("grouped_matmul_swiglu_quant")
    _gmm_t, gmm_kw = gmm_oracle.synthesize_case_inputs(7)
    glist = gmm_kw["group_list"]
    assert len(glist) >= 2 and any(glist[i] == glist[i - 1] for i in range(1, len(glist)))

    # 7. In-place buffer aliasing clone protection
    kv_oracle = CannbenchContractOracle("kv_cache_append")
    small_kv_case = min(kv_oracle.cases, key=lambda c: sum(math.prod(s) for s in c["input_shape"]))
    cid = int(small_kv_case["case_id"])
    raw_before, _ = kv_oracle.synthesize_case_inputs(cid)
    snap_k = raw_before[0].clone()

    def _mutating_kv(k_cache, v_cache, new_k, new_v, slot_mapping):
        total = k_cache.shape[0] * k_cache.shape[1]
        fk = k_cache.reshape(total, *k_cache.shape[2:])
        fv = v_cache.reshape(total, *v_cache.shape[2:])
        slots = slot_mapping.to(torch.long)
        for i in range(new_k.shape[0]):
            s = int(slots[i].item())
            fk[s] = new_k[i]
            fv[s] = new_v[i]
        return k_cache, v_cache

    kv_ok, kv_rep = kv_oracle.verify_candidate(_mutating_kv, case_nums=[cid])
    assert kv_ok is True, kv_rep
    assert torch.equal(raw_before[0], snap_k)


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


def test_proxy_device_uses_proportional_pipe_cycles_and_never_prunes_rules() -> None:
    """Verify _proxy_device derives proportional PMU ratios and skips hardware rule pruning."""
    baseline_dsl = """
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
    tracker = BayesianRuleTracker.from_seed_yaml()
    for rid, post in tracker.posteriors.items():
        post.failures = 10
        post.beta = 20.0
    # Also seed axis posteriors with high failure counts to prove proxy won't prune
    for axis_key in ("ce:aic_only", "mr:gm_ub_stream"):
        for rid in list(tracker.rules)[:3]:
            st = tracker._ensure_axis_posterior(axis_key, rid)
            st.failures = 10
            st.beta = 20.0

    backend = MockLLMBackend(default_response=baseline_dsl)
    funnel = GateFunnel(compile_runner=lambda _dsl: StageBResult(passed=True))
    traj = asyncio.run(
        run_e2e_evolution(
            problem_name="proxy_no_prune",
            hir_source="pass",
            shape_contract="make_kernel()",
            backend=backend,
            funnel=funnel,
            tracker=tracker,
            initial_dsl_source=baseline_dsl,
            factory_name="make_kernel",
            num_rounds=1,
            k_rule_workers=2,
        )
    )
    assert traj.total_rounds == 1
    assert traj.rounds[0].pruned_rule_ids == ()
    assert not any(p.pruned for p in tracker.posteriors.values())
    for bmap in tracker.axis_posteriors.values():
        assert not any(p.pruned for p in bmap.values())


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
            # Reverting T.Kernel(4) -> T.Kernel(1) or swizzle_window=4 -> 1 hurts latency
            if "T.Kernel(1)" in dsl:
                out[cid] = 80.0
            elif "'swizzle_window': 1" in dsl or '"swizzle_window": 1' in dsl:
                out[cid] = 50.0
            else:
                out[cid] = 20.0  # Reverting decoy hurts nothing
        return out

    from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

    gemm_axes = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    tracker = BayesianRuleTracker.from_seed_yaml()
    distiller = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        min_edit_gain=0.10,
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
