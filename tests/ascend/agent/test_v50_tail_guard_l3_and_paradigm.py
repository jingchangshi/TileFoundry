"""Tests for v50 adaptive cost gate, cold-L2 denoising, L3 taxonomy, and KDA springboard."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def _ensure_cost_gate_policy_stub() -> None:
    """Install a contract-matched stub for cost_gate_policy only when W1's module is absent."""
    mod_name = "tilefoundry_ascend.ascend.cost_model.cost_gate_policy"
    if mod_name in sys.modules:
        return
    try:
        if importlib.util.find_spec(mod_name) is not None:
            return
    except ModuleNotFoundError:
        pass

    from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

    class CostModelMode(StrEnum):
        AUTO = "auto"
        COARSE_ONLY = "coarse_only"
        OFF = "off"

    @dataclass(frozen=True, slots=True)
    class CostGateDecision:
        mode: CostModelMode
        engaged: bool
        reason: str
        coarse_reject_ratio: float
        allow_prompt_guidance: bool
        allow_candidate_pruning: bool

    def is_paradigm_shift_exempt(
        cand_axes: TopologyAxes | None,
        parent_axes: TopologyAxes | None,
        *,
        cand_workspace_stages: int = 0,
        parent_workspace_stages: int = 0,
        is_creative: bool = False,
    ) -> bool:
        if is_creative:
            return True
        if cand_workspace_stages > parent_workspace_stages:
            return True
        if cand_axes is not None and parent_axes is not None and cand_axes != parent_axes:
            return True
        return False

    def evaluate_cost_gate_policy(
        *,
        mode: CostModelMode | str = CostModelMode.AUTO,
        enable_prescreen_flag: bool = True,
        round_idx: int = 1,
        stagnation_rounds: int = 0,
        incumbent_roofline_eta: float = 0.0,
        target_roofline_eta: float = 0.85,
        prev_local_rel_gain: float | None = None,
        mean_recoverable_excess_us: float | None = None,
        disengage_roofline_eta: float = 0.65,
        slow_gain_threshold: float = 0.03,
        min_recoverable_excess_us: float = 1.0,
        coarse_reject_ratio: float = 3.0,
    ) -> CostGateDecision:
        _ = (round_idx, stagnation_rounds, target_roofline_eta)
        resolved_mode = CostModelMode(str(mode))
        if not enable_prescreen_flag or resolved_mode == CostModelMode.OFF:
            return CostGateDecision(
                mode=resolved_mode,
                engaged=False,
                reason="disabled_by_flag_or_off_mode",
                coarse_reject_ratio=coarse_reject_ratio,
                allow_prompt_guidance=False,
                allow_candidate_pruning=False,
            )
        if resolved_mode == CostModelMode.COARSE_ONLY:
            return CostGateDecision(
                mode=resolved_mode,
                engaged=True,
                reason="coarse_only_mode",
                coarse_reject_ratio=coarse_reject_ratio,
                allow_prompt_guidance=False,
                allow_candidate_pruning=True,
            )
        if (
            incumbent_roofline_eta >= disengage_roofline_eta
            and prev_local_rel_gain is not None
            and prev_local_rel_gain < slow_gain_threshold
        ) or (
            mean_recoverable_excess_us is not None
            and mean_recoverable_excess_us < min_recoverable_excess_us
        ):
            return CostGateDecision(
                mode=resolved_mode,
                engaged=False,
                reason="auto_disengaged_high_eta_low_gain",
                coarse_reject_ratio=coarse_reject_ratio,
                allow_prompt_guidance=False,
                allow_candidate_pruning=False,
            )
        return CostGateDecision(
            mode=resolved_mode,
            engaged=True,
            reason="auto_engaged",
            coarse_reject_ratio=coarse_reject_ratio,
            allow_prompt_guidance=True,
            allow_candidate_pruning=True,
        )

    stub = types.ModuleType(mod_name)
    stub.CostModelMode = CostModelMode
    stub.CostGateDecision = CostGateDecision
    stub.is_paradigm_shift_exempt = is_paradigm_shift_exempt
    stub.evaluate_cost_gate_policy = evaluate_cost_gate_policy
    sys.modules[mod_name] = stub


_ensure_cost_gate_policy_stub()

from tilefoundry_ascend.agent.campaign import campaign_runner, cli, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


class _FakeEvent:
    def __init__(self, clock_state: list[float]) -> None:
        self._clock_state = clock_state
        self.recorded_us = 0.0

    def record(self) -> None:
        self.recorded_us = self._clock_state[0]

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return (other.recorded_us - self.recorded_us) / 1000.0


class _FakeTensor:
    def __init__(self, nbytes: int, tag: int = 0) -> None:
        self.nbytes = nbytes
        self.tag = tag

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self.nbytes, self.tag)


def _make_dummy_runner(tmp_path: Path, *, operator_name: str = "rms_norm") -> campaign_runner.PerfTargetCampaignRunner:
    target_mod = tmp_path / "dummy_op.py"
    target_mod.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    specs = (
        RegimeTargetSpec(
            regime_tag="small",
            builder_candidates=("_build_small",),
            sample_builder_kwargs={"M": 16, "D": 64},
            representative_anchor_dims={"M": 16, "D": 64},
            representative_hir_dims={"M": 16, "D": 64},
            editable_symbols=("_build_small",),
            config_selector_fn="_get_small_config",
            runtime_backend="tilelang_dsl",
        ),
        RegimeTargetSpec(
            regime_tag="large",
            builder_candidates=("_build_large",),
            sample_builder_kwargs={"M": 16, "D": 2048},
            representative_anchor_dims={"M": 16, "D": 2048},
            representative_hir_dims={"M": 16, "D": 2048},
            editable_symbols=("_build_large",),
            config_selector_fn="_get_large_config",
            runtime_backend="tilelang_dsl",
        ),
    )
    anchors = (
        AnchorCaseSpec(task="t", case_num=0, regime_tag="small", ref_perf_us=10.0, t_hw_us=2.0),
        AnchorCaseSpec(task="t", case_num=1, regime_tag="large", ref_perf_us=10.0, t_hw_us=2.0),
    )
    perf_target = surrogate_cannbench.PerfTargetSpec(
        operator_name=operator_name,
        target_avg_speedup=1.5,
        target_min_case_speedup=1.2,
        target_roofline_eta=0.85,
        target_surrogate_score=1.5,
    )
    return campaign_runner.PerfTargetCampaignRunner(
        config=campaign_runner.CampaignConfig(
            operator_name=operator_name,
            device_id="1",
            warmup=1,
            repeat=1,
        ),
        target_module_path=target_mod,
        perf_target=perf_target,
        anchors=anchors,
        regime_specs=specs,
        checkpoint_path=tmp_path / "ckpt.json",
        campaign_kb_path=tmp_path / "kb.json",
    )


def test_cli_and_round_record_cost_gate_telemetry_roundtrip() -> None:
    """Verify cost-gate CLI flags and CampaignRoundRecord serialization roundtrip."""
    for parser_fn in (cli.build_arg_parser, campaign_runner.build_arg_parser):
        parser = parser_fn()
        args = parser.parse_args(
            [
                "--operator",
                "kda_chunk",
                "--cost-model-mode",
                "coarse_only",
                "--cost-disengage-eta",
                "0.72",
                "--cost-slow-gain-threshold",
                "0.045",
            ]
        )
        assert args.cost_model_mode == "coarse_only"
        assert args.cost_disengage_eta == pytest.approx(0.72)
        assert args.cost_slow_gain_threshold == pytest.approx(0.045)

    rec = campaign_runner.CampaignRoundRecord(
        round_idx=3,
        regime_tag="small",
        builder_name="_build_small",
        parent_lane="exploit",
        creative_effort="low",
        l0_l1_passed_ids=("c1",),
        l0_l1_failed=(),
        best_round_speedup=1.2,
        best_so_far_speedup=1.3,
        promoted=False,
        rolled_back_l3=False,
        stagnation_rounds_after=1,
        cost_model_mode="auto",
        cost_model_engaged=False,
        cost_model_reason="auto_disengaged_high_eta_low_gain",
        local_rel_gain=0.018,
        tail_guard_reject_reason="multi_sibling_regression",
    )
    payload = rec.to_dict()
    assert payload["cost_model_mode"] == "auto"
    assert payload["cost_model_engaged"] is False
    assert payload["cost_model_reason"] == "auto_disengaged_high_eta_low_gain"
    assert payload["local_rel_gain"] == pytest.approx(0.018)
    assert payload["tail_guard_reject_reason"] == "multi_sibling_regression"

    restored = campaign_runner.CampaignRoundRecord.from_dict(payload)
    assert restored.cost_model_mode == "auto"
    assert restored.cost_model_engaged is False
    assert restored.cost_model_reason == "auto_disengaged_high_eta_low_gain"
    assert restored.local_rel_gain == pytest.approx(0.018)
    assert restored.tail_guard_reject_reason == "multi_sibling_regression"


@pytest.mark.anyio
async def test_run_l0_l1_round_filter_exempts_creative_and_paradigm_shift() -> None:
    """Verify creative_explore and paradigm shifts are exempt while >3x rule regresses fail."""
    parent_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    shifted_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    candidates = [
        ("cand_creative", "def f():\n    return 1\n"),
        ("cand_paradigm_axes", "def f():\n    return 2\n"),
        ("cand_paradigm_ws", "def f():\n    return 3\n"),
        ("cand_rule_ok", "def f():\n    return 4\n"),
        ("cand_rule_slow", "def f():\n    return 5\n"),
    ]
    scores = {
        "cand_creative": 25.0,
        "cand_paradigm_axes": 30.0,
        "cand_paradigm_ws": 28.0,
        "cand_rule_ok": 12.0,
        "cand_rule_slow": 35.0,
    }

    def _l0(src: str) -> StageAResult:
        cid = "cand_rule_ok"
        for name, val in (
            ("return 1", "cand_creative"),
            ("return 2", "cand_paradigm_axes"),
            ("return 3", "cand_paradigm_ws"),
            ("return 4", "cand_rule_ok"),
            ("return 5", "cand_rule_slow"),
        ):
            if name in src:
                cid = val
                break
        return StageAResult(passed=True, prim_func=cid, ledger=None)

    def _l1(_src: str) -> StageBResult:
        return StageBResult(passed=True)

    def _ranker(rows: Sequence[tuple[str, str, StageAResult]]) -> Sequence[tuple[str, float]]:
        return [(cid, scores[cid]) for cid, _src, _res in rows]

    def _scorer(cid: str, _src: str, _res: StageAResult) -> float:
        return scores[cid]

    def _axes_fn(_src: str, res: StageAResult) -> TopologyAxes | None:
        return shifted_axes if res.prim_func == "cand_paradigm_axes" else parent_axes

    def _ws_fn(_src: str, res: StageAResult) -> int:
        return 2 if res.prim_func == "cand_paradigm_ws" else 1

    results = await surrogate_cannbench.run_l0_l1_round_filter(
        candidates,
        patch_fn=lambda snippet: snippet,
        l0_checker=_l0,
        l1_compiler=_l1,
        stage_a_cost_ranker=_ranker,
        cost_prescreen_top_k=1,
        cost_coarse_reject_ratio=3.0,
        creative_candidate_ids={"cand_creative"},
        parent_analytical_us=10.0,
        stage_a_analytical_scorer=_scorer,
        parent_topology_axes=parent_axes,
        candidate_topology_axes_fn=_axes_fn,
        parent_workspace_stages=1,
        candidate_workspace_stages_fn=_ws_fn,
    )
    by_id = {c.candidate_id: c for c in results}
    assert by_id["cand_creative"].passed is True
    assert by_id["cand_paradigm_axes"].passed is True
    assert by_id["cand_paradigm_ws"].passed is True
    assert by_id["cand_rule_ok"].passed is True
    assert by_id["cand_rule_slow"].passed is False
    assert by_id["cand_rule_slow"].failed_tier == "cost_prescreen"


@pytest.mark.anyio
async def test_b4_noop_identical_rejects_comment_and_whitespace_only_changes() -> None:
    """Verify comment- and whitespace-only diffs are rejected before L0 as noop_identical."""
    parent_src = '''
KERNEL_CPP = """
// comment in parent
__global__ void my_kern(float* x) {
    int a = 1;
}
"""

def _get_small_kernel():
    return KERNEL_CPP
'''
    cand_noop_src = '''
KERNEL_CPP = """
/* different comment */
__global__   void   my_kern(float* x) {
    int a = 1;
}
"""

def _get_small_kernel():
    return KERNEL_CPP
'''
    results = await surrogate_cannbench.run_l0_l1_round_filter(
        [("cand_noop", cand_noop_src)],
        patch_fn=lambda snippet: snippet,
        l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=None),
        l1_compiler=lambda _src: StageBResult(passed=True),
        parent_source=parent_src,
        parent_module_source=parent_src,
    )
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].failed_tier == "noop_identical"
    assert results[0].error_message == "noop_identical_to_parent"


def test_cold_l2_ring_buffer_k_scaling_and_burst_timing() -> None:
    """Verify K scaling 24..32 for micro-anchors, VRAM cap backoff, and K-averaged elapsed time."""
    assert surrogate_cannbench.compute_ring_burst_k(1024, t_hw_us=2.0) == 32
    assert surrogate_cannbench.compute_ring_burst_k(8 * 1024 * 1024, t_hw_us=2.0) == 24
    assert surrogate_cannbench.compute_ring_burst_k(1024, t_hw_us=15.0) == 1
    assert surrogate_cannbench.compute_ring_burst_k(1024, first_probe_us=6.5) == 32
    assert (
        surrogate_cannbench.compute_ring_burst_k(
            64 * 1024 * 1024,
            t_hw_us=2.0,
            max_vram_bytes=256 * 1024 * 1024,
        )
        == 4
    )

    clock_state = [0.0]
    calls = 0
    seen_tags: list[int] = []

    def _fn(x: _FakeTensor) -> None:
        nonlocal calls
        calls += 1
        seen_tags.append(x.tag)
        clock_state[0] += 3.0

    def _make_event() -> _FakeEvent:
        return _FakeEvent(clock_state)

    clone_seq = 0

    def _clone(obj: object) -> _FakeTensor:
        nonlocal clone_seq
        clone_seq += 1
        assert isinstance(obj, _FakeTensor)
        return _FakeTensor(obj.nbytes, clone_seq)

    avg_us, burst_k = surrogate_cannbench.measure_npu_ring_burst_us(
        _fn,
        (_FakeTensor(1024),),
        warmup=2,
        repeat=3,
        t_hw_us=2.0,
        event_factory=_make_event,
        synchronize_fn=lambda: None,
        clone_fn=_clone,
    )
    assert burst_k == 32
    assert calls == (2 + 2 + 3 * 32)
    assert len(set(seen_tags[-32:])) == 32
    assert avg_us == pytest.approx(3.0)

    spec = AnchorCaseSpec(
        task="t",
        case_num=0,
        regime_tag="small",
        ref_perf_us=10.0,
        t_hw_us=2.0,
    )
    metrics = surrogate_cannbench.build_anchor_metrics(
        spec,
        accuracy_passed=True,
        cand_us=4.0,
        ring_burst_k=burst_k,
    )
    assert metrics.ring_burst_k == 32


def test_b2_unchanged_sibling_exemption_and_noise_floor_and_reject_reason(tmp_path: Path) -> None:
    """Verify B2 sibling AST closure exemption, 0.35us noise floor, and reject reason tracking."""
    incumbent_src = '''
def _shared_helper(x):
    return x + 1

def _build_small():
    return _shared_helper(1)

def _build_large():
    return _shared_helper(99)
'''
    cand_unchanged_sibling = '''
def _shared_helper(x):
    return x + 1

def _build_small():
    return _shared_helper(2)

def _build_large():
    return _shared_helper(99)
'''
    cand_changed_sibling = '''
def _shared_helper(x):
    return x + 2

def _build_small():
    return _shared_helper(2)

def _build_large():
    return _shared_helper(99)
'''
    runner = _make_dummy_runner(tmp_path)
    small_spec = runner._regime_specs[0]
    assert "large" in runner._unchanged_sibling_regimes(
        incumbent_src, cand_unchanged_sibling, small_spec
    )
    assert "large" not in runner._unchanged_sibling_regimes(
        incumbent_src, cand_changed_sibling, small_spec
    )

    def _pair(
        case_num: int,
        regime: str,
        ref_us: float,
        best_us: float,
        cand_us: float,
    ) -> tuple[surrogate_cannbench.AnchorEvalMetrics, surrogate_cannbench.AnchorEvalMetrics]:
        best_m = surrogate_cannbench.AnchorEvalMetrics(
            case_num=case_num,
            regime_tag=regime,
            accuracy_passed=True,
            cand_us=best_us,
            ref_perf_us=ref_us,
            t_hw_us=2.0,
            speedup=ref_us / best_us,
            roofline_eta=0.5,
            eq3_score=1.0,
        )
        cand_m = surrogate_cannbench.AnchorEvalMetrics(
            case_num=case_num,
            regime_tag=regime,
            accuracy_passed=True,
            cand_us=cand_us,
            ref_perf_us=ref_us,
            t_hw_us=2.0,
            speedup=ref_us / cand_us,
            roofline_eta=0.5,
            eq3_score=1.0,
        )
        return best_m, cand_m

    best_s, cand_s = _pair(0, "small", 10.0, 5.0, 4.0)
    best_l, cand_l = _pair(1, "large", 10.0, 5.0, 6.0)
    best_res = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="incumbent",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=2.0,
        min_case_speedup=2.0,
        avg_roofline_eta=0.5,
        surrogate_score=1.0,
        anchors=(best_s, best_l),
    )
    cand_res_big_sib_dip = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="cand_v2",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=2.1,
        min_case_speedup=1.66,
        avg_roofline_eta=0.55,
        surrogate_score=1.35,
        anchors=(cand_s, cand_l),
    )

    ok_exempt, _ = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_v2",
        candidate_source=cand_unchanged_sibling,
        top_res=cand_res_big_sib_dip,
        best_result=best_res,
        regime_spec=small_spec,
        active_anchors=runner._initial_anchors,
        incumbent_source=incumbent_src,
    )
    assert ok_exempt is True

    best_l_j, cand_l_j = _pair(1, "large", 10.0, 5.0, 5.20)
    cand_res_floor = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="cand_floor",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=2.2,
        min_case_speedup=1.92,
        avg_roofline_eta=0.55,
        surrogate_score=1.35,
        anchors=(cand_s, cand_l_j),
    )
    ok_floor, _ = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_floor",
        candidate_source=cand_changed_sibling,
        top_res=cand_res_floor,
        best_result=best_res,
        regime_spec=small_spec,
        active_anchors=runner._initial_anchors,
        incumbent_source=incumbent_src,
    )
    assert ok_floor is True

    bad_s = surrogate_cannbench.AnchorEvalMetrics(
        case_num=0,
        regime_tag="small",
        accuracy_passed=True,
        cand_us=6.0,
        ref_perf_us=10.0,
        t_hw_us=2.0,
        speedup=10.0 / 6.0,
        roofline_eta=0.4,
        eq3_score=0.9,
    )
    cand_res_bad = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="cand_bad",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=1.8,
        min_case_speedup=1.66,
        avg_roofline_eta=0.4,
        surrogate_score=1.0,
        anchors=(bad_s, best_l),
    )
    ok_bad, _ = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_bad",
        candidate_source=cand_changed_sibling,
        top_res=cand_res_bad,
        best_result=best_res,
        regime_spec=small_spec,
        active_anchors=runner._initial_anchors,
        incumbent_source=incumbent_src,
    )
    assert ok_bad is False
    assert runner._last_tail_guard_reject_reason == "active_tail_regression"


def test_l3_failure_kind_candidate_and_infra_distinction() -> None:
    """Verify _l3_failure_kind distinguishes deterministic candidate failures from infra/HW."""
    cov: Mapping[str, object] = {"failed_cases": 1, "total_cases": 2}
    cand_obs: Mapping[str, object] = {
        "failureCategory": "infrastructure",
        "failure": {
            "kind": "candidate",
            "category": "runtime",
            "role": "candidate",
            "stage": "compile",
            "node": "cand_node",
            "case": "0",
            "detail": "exceeded vf stack size",
        },
    }
    cand_evidence: Mapping[str, object] = {
        "phase": "finished",
        "timing": {"finishedAt": "2026-11-24T00:00:00Z"},
        "nodes": [{"name": "cand_node", "status": "failed", "cases": {"0": "failed"}}],
    }
    assert surrogate_cannbench._l3_failure_kind(cand_obs, cand_evidence, cov) == "candidate"

    idem_obs: Mapping[str, object] = {
        "failureCategory": "infrastructure",
        "failure": {
            "role": "candidate",
            "stage": "cache_idempotence",
            "node": "cand_node",
            "case": "0",
            "detail": "L3 cache idempotence violation",
        },
    }
    assert surrogate_cannbench._l3_failure_kind(idem_obs, cand_evidence, cov) == "candidate"

    hw_obs: Mapping[str, object] = {
        "failureCategory": "infrastructure",
        "failure": {
            "role": "candidate",
            "stage": "kernel",
            "node": "cand_node",
            "case": "0",
            "detail": "aclrtGetDevice device unavailable DrvError",
        },
    }
    assert surrogate_cannbench._l3_failure_kind(hw_obs, cand_evidence, cov) == "infrastructure"


def test_vf_stack_overflow_and_max_row_width_helpers(tmp_path: Path) -> None:
    """Verify VF stack overflow detection and max-row-width resolution."""
    runner = _make_dummy_runner(tmp_path)
    spec = runner._regime_specs[0]
    overflow_res = StageBResult(
        passed=False,
        error_message="compile error: exceeded vf stack size",
        full_diagnostic="bisheng error",
    )
    assert runner._has_vf_stack_overflow(overflow_res) is True
    assert runner._has_vf_stack_overflow(StageBResult(passed=True)) is False
    dims = runner._resolve_max_row_width_dims(spec)
    assert dims is not None
    assert int(dims["D"]) >= 2048


def test_resolve_regime_config_and_bare_trackb_and_slice_contract_and_kda_helpers(
    tmp_path: Path,
) -> None:
    """Verify pass_configs coercion, bare Track-B detection, slice instructions, and KDA helpers."""
    mod_src = '''
def _get_small_config(case_num):
    return {
        "pass_configs": ["--cce-aicore-arch=dav-c310", "--O2"],
        "tile_n": 64,
    }

def _build_small(tile_n=64):
    return tile_n
'''
    mod_path = tmp_path / "cand_mod.py"
    mod_path.write_text(mod_src, encoding="utf-8")
    small_spec = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={"case_num": 0},
        config_selector_fn="_get_small_config",
    )
    cfg = campaign_runner.resolve_regime_config(mod_path, small_spec, {"case_num": 0})
    assert "--cce-aicore-arch=dav-c310" in cfg.compile_flags
    assert "--O2" in cfg.compile_flags

    assert (
        campaign_runner._looks_like_bare_trackb_cpp(
            "#include <kernel_operator.h>\n__global__ void f() {}"
        )
        is True
    )
    assert campaign_runner._looks_like_bare_trackb_cpp("static inline void h() {}") is True
    assert campaign_runner._looks_like_bare_trackb_cpp("constexpr int K = 16;") is True
    assert (
        campaign_runner._looks_like_bare_trackb_cpp(
            "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE"
        )
        is False
    )

    instr_kda = campaign_runner.PerfTargetCampaignRunner._slice_contract_instruction(
        "kernel",
        small_spec,
        operator_entrypoint="kernel",
    )
    assert "MUST NOT define a top-level" in instr_kda

    instr_tilelang = campaign_runner.PerfTargetCampaignRunner._slice_contract_instruction(
        "custom_entry",
        small_spec,
        operator_entrypoint="custom_entry",
    )
    assert "MUST NOT" not in instr_tilelang

    runner = _make_dummy_runner(tmp_path)
    kda_spec = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={"D": 64},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TRACKB_CPP_SMALL",),
    )
    kda_src = '''
_TRACKB_CPP_SMALL = """
__global__ __aicore__ void stage1_k() {}
__global__ __aicore__ void stage2_k() {}
void host_launcher() {
    stage1_k<<<1, nullptr, nullptr>>>();
    stage2_k<<<1, nullptr, nullptr>>>();
}
"""
'''
    kernels = runner._resolve_active_trackb_kernels(kda_src, kda_spec)
    assert kernels == ("stage1_k", "stage2_k")


def test_prewarm_dedup_and_strict_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify TileLang prewarm deduplicates identical tasks and raises on compile failure."""
    calls: list[tuple[list[str], str]] = []

    class _FakeCompleted:
        def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = False,
        cwd: str | None = None,
    ) -> _FakeCompleted:
        _ = (capture_output, text, timeout, check, cwd)
        calls.append((cmd, (env or {}).get("TILELANG_CACHE_DIR", "")))
        script_path = Path(cmd[-1])
        if script_path.name == "prewarm_tl_plan.py":
            return _FakeCompleted(
                0,
                json.dumps(
                    {
                        "tasks": [
                            {
                                "builder_name": "b1",
                                "mode": "tl",
                                "kwargs": {"a": 1},
                                "pass_configs": {},
                                "compile_flags": [],
                            },
                            {
                                "builder_name": "b1",
                                "mode": "tl",
                                "kwargs": {"a": 1},
                                "pass_configs": {},
                                "compile_flags": [],
                            },
                        ],
                        "errors": [],
                    }
                ),
                "",
            )
        return _FakeCompleted(1, "", "boom compile error")

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    spec_path = tmp_path / "dummy_mod.py"
    spec_path.write_text("def x(): pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom compile error"):
        surrogate_cannbench.prewarm_surrogate_tilelang_cache(
            "rms_norm",
            [("small", spec_path)],
            [
                AnchorCaseSpec(
                    task="t",
                    case_num=0,
                    regime_tag="small",
                    ref_perf_us=10.0,
                    t_hw_us=5.0,
                )
            ],
            cannbench_repo=tmp_path,
            cache_dir=tmp_path / "ref_cache",
            tl_cache_dir=tmp_path / "tl_cache",
        )
    worker_calls = [c for c in calls if Path(c[0][-1]).name == "compile_task.py"]
    assert len(worker_calls) == 1


def test_recoverable_excess_launch_floors(tmp_path: Path) -> None:
    """Verify per-backend launch floor constants used in _regime_mean_recoverable_excess_us."""
    runner = _make_dummy_runner(tmp_path)
    tl_spec = runner._regime_specs[0]
    tb_spec = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={"M": 16, "D": 64},
        runtime_backend="ascendc_trackb",
    )
    res = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="b",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.5,
        surrogate_score=1.0,
        anchors=(
            surrogate_cannbench.AnchorEvalMetrics(
                case_num=0,
                regime_tag="small",
                accuracy_passed=True,
                cand_us=3.0,
                ref_perf_us=10.0,
                t_hw_us=0.8,
                speedup=10.0 / 3.0,
                roofline_eta=0.5,
                eq3_score=1.0,
            ),
        ),
    )
    assert runner._regime_mean_recoverable_excess_us(res, tl_spec) == pytest.approx(0.8)
    assert runner._regime_mean_recoverable_excess_us(res, tb_spec) == pytest.approx(1.6)


@pytest.mark.anyio
async def test_execute_single_round_cost_gate_disengage_and_kda_shift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify cost gate disengagement clears analytical leakage and triggers KDA shift."""
    from tilefoundry_ascend.agent.kb.schema import SymptomFamily
    from tilefoundry_ascend.agent.router.capability_gate import route_execution_track
    from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger

    mod_src = '''
def _build_small():
    return 1

def _build_large():
    return 2

def kernel(x):
    return x
'''
    runner = _make_dummy_runner(tmp_path)
    runner._target_module_path.write_text(mod_src, encoding="utf-8")

    class _FakeLLM:
        async def generate(self, req: object) -> str:
            if getattr(req, "role", "") == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Adjust small builder variant",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": "def _build_small():\n    return 2\n",
                    }
                )
            return "def _build_small():\n    return 3\n"

    runner._llm_backend = _FakeLLM()
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIV0.PIPE_V": 10},
        waterlines={},
        grid_blocks=16,
        wave_occupancy=0.5,
    )
    route = route_execution_track(None, ledger=ledger)
    runner._l0_checker = lambda _src, _reg: StageAResult(
        passed=True,
        prim_func=None,
        ledger=ledger,
        features=route.features,
        route=route,
    )
    runner._l1_compiler = lambda _src, _reg: StageBResult(passed=True)
    runner._last_local_rel_gain_by_regime["small"] = 0.01

    prescreen_calls = 0

    def _spy_prescreen(*_args: object, **_kwargs: object) -> str:
        nonlocal prescreen_calls
        prescreen_calls += 1
        return "should_not_appear"

    seen_contexts: list[dict[str, object]] = []
    orig_rank_symptoms = campaign_runner.rank_symptom_families

    def _spy_rank_symptoms(ctx: Mapping[str, object]) -> list[tuple[SymptomFamily, float]]:
        seen_contexts.append(dict(ctx))
        return list(orig_rank_symptoms(ctx))

    seen_rank_axes: list[str] = []
    orig_rank_rules = runner._tracker.rank_applicable_rules

    def _spy_rank_rules(*args: object, **kwargs: object) -> list[object]:
        axes = kwargs.get("topology_axes")
        if axes is not None and hasattr(axes, "compute_engine"):
            seen_rank_axes.append(str(axes.compute_engine))
        return orig_rank_rules(*args, **kwargs)

    seen_filter_kwargs: list[dict[str, object]] = []

    async def _spy_filter(
        candidates: Sequence[tuple[str, str]],
        **kwargs: object,
    ) -> list[surrogate_cannbench.L0L1CandidateCheck]:
        _ = candidates
        seen_filter_kwargs.append(dict(kwargs))
        return []

    class _RecurrentSemantics:
        has_recurrent_loop = True

    monkeypatch.setattr(campaign_runner, "build_prescreen_summary_from_kwargs", _spy_prescreen)
    monkeypatch.setattr(campaign_runner, "rank_symptom_families", _spy_rank_symptoms)
    monkeypatch.setattr(runner._tracker, "rank_applicable_rules", _spy_rank_rules)
    monkeypatch.setattr(campaign_runner, "run_l0_l1_round_filter", _spy_filter)
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    from tilefoundry_ascend.agent.orchestrator import llm_constraints

    monkeypatch.setattr(llm_constraints, "format_target_dialect_layer", lambda *_a, **_kw: "")
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_resolve_operator_semantics_for_gates",
        lambda: _RecurrentSemantics(),
    )

    best_result = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="base",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=1.2,
        min_case_speedup=1.2,
        avg_roofline_eta=0.70,
        surrogate_score=1.2,
        anchors=(
            surrogate_cannbench.AnchorEvalMetrics(
                case_num=0,
                regime_tag="small",
                accuracy_passed=True,
                cand_us=3.5,
                ref_perf_us=4.2,
                t_hw_us=2.0,
                speedup=1.2,
                roofline_eta=0.70,
                eq3_score=1.2,
            ),
        ),
    )

    rec, *_rest = await runner._execute_single_round(
        round_idx=2,
        regime_spec=runner._regime_specs[0],
        incumbent_source=mod_src,
        best_result=best_result,
        active_anchors=runner._initial_anchors[:1],
        stagnation_rounds=0,
        admitted_rule_ids=[],
    )
    assert rec.cost_model_engaged is False
    assert rec.cost_model_mode == "auto"
    assert rec.cost_model_reason == "near_theoretical_limit"
    assert prescreen_calls == 0
    assert seen_contexts and seen_contexts[0].get("use_uncalibrated_analytical") is False
    assert seen_filter_kwargs
    assert seen_filter_kwargs[0]["parent_analytical_us"] is None
    assert seen_filter_kwargs[0]["stage_a_cost_ranker"] is None
    assert seen_filter_kwargs[0]["cost_prescreen_top_k"] is None

    seen_rank_axes.clear()
    low_eta_result = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="base",
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.10,
        surrogate_score=1.0,
        anchors=(
            surrogate_cannbench.AnchorEvalMetrics(
                case_num=0,
                regime_tag="small",
                accuracy_passed=True,
                cand_us=10.0,
                ref_perf_us=10.0,
                t_hw_us=1.0,
                speedup=1.0,
                roofline_eta=0.10,
                eq3_score=1.0,
            ),
        ),
    )
    await runner._execute_single_round(
        round_idx=3,
        regime_spec=runner._regime_specs[0],
        incumbent_source=mod_src,
        best_result=low_eta_result,
        active_anchors=runner._initial_anchors[:1],
        stagnation_rounds=2,
        admitted_rule_ids=[],
    )
    assert "coupled_1c2v" in seen_rank_axes
