"""Unit tests for StageGateCoordinator, speculative creative lane, and typed L2 faults."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


def _bootstrap_isolated_worktree() -> None:
    """Resolve worktree modules directly even if another checkout is installed."""
    root = Path(__file__).resolve().parents[3]
    ascend_root = str(root / "targets" / "ascend")
    if ascend_root not in sys.path:
        sys.path.insert(0, ascend_root)
    pkg_dir = root / "targets" / "ascend" / "tilefoundry_ascend"
    try:
        mod = importlib.import_module("tilefoundry_ascend.agent.campaign.stage_gate_coordinator")
        mod_file = getattr(mod, "__file__", "") or ""
        if Path(mod_file).resolve().is_relative_to(root):
            return
    except ImportError:
        pass
    stale = [
        k
        for k in sys.modules
        if k == "tilefoundry_ascend" or k.startswith("tilefoundry_ascend.")
    ]
    for key in stale:
        del sys.modules[key]
    for mod_name, sub_path in (
        ("tilefoundry_ascend", pkg_dir),
        ("tilefoundry_ascend.agent", pkg_dir / "agent"),
        ("tilefoundry_ascend.agent.campaign", pkg_dir / "agent" / "campaign"),
    ):
        mod = types.ModuleType(mod_name)
        mod.__path__ = [str(sub_path)]
        sys.modules[mod_name] = mod


_bootstrap_isolated_worktree()

from tilefoundry_ascend.agent.campaign.anchor_specs import (  # noqa: E402
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.fault_types import FaultPlane  # noqa: E402
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (  # noqa: E402
    CandidateBatch,
    OccupancyExhaustedError,
    SpeculativeCreativeResult,
    StageGateCoordinator,
    StageGateOutcome,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: E402
    AnchorEvalMetrics,
    CandidateSurrogateResult,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: E402
    StageAResult,
    StageBResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest  # noqa: E402
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary  # noqa: E402
from tilefoundry_ascend.ascend.passes import (  # noqa: E402
    build_objective_physical_ledger,
    build_scope_waterline,
)

_PARENT_MODULE_SRC = (
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n\n"
    "def kernel(x):\n"
    "    return _build_main(), x\n"
)

_PARENT_SNIPPET = "def _build_main(tile_m=16):\n    return int(tile_m)\n"
_EXPLOIT_SNIPPET = "def _build_main(tile_m=32):\n    return int(tile_m)\n"
_CREATIVE_SNIPPET = "def _build_main(tile_m=64):\n    return int(tile_m)\n"


def _make_regime_spec() -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_main",),
        editable_symbols=("_build_main",),
        sample_builder_kwargs={"tile_m": 16},
        representative_anchor_dims={"D": 128},
    )


def _make_anchors() -> tuple[AnchorCaseSpec, ...]:
    return (AnchorCaseSpec("level2/rms_norm", 0, "decode", 20.0, 2.0),)


def _make_stage_a() -> StageAResult:
    waterlines = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 32768},
            implicit_buffers={},
        )
    }
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100},
        waterlines=waterlines,
        gm_roundtrip_bytes=16384,
        grid_blocks=16,
    )
    return StageAResult(passed=True, prim_func=None, ledger=ledger)


def _make_surrogate_result(cand_id: str, cand_us: float = 10.0) -> CandidateSurrogateResult:
    anc = AnchorEvalMetrics(
        case_num=0,
        regime_tag="decode",
        accuracy_passed=True,
        cand_us=cand_us,
        ref_perf_us=20.0,
        t_hw_us=2.0,
        speedup=20.0 / cand_us,
        roofline_eta=2.0 / cand_us,
        eq3_score=20.0 / cand_us,
        paired_rel_mad=0.01,
    )
    return CandidateSurrogateResult(
        candidate_id=cand_id,
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=anc.speedup,
        min_case_speedup=anc.speedup,
        avg_roofline_eta=anc.roofline_eta,
        surrogate_score=anc.speedup,
        anchors=(anc,),
    )


def _make_req(role: str) -> GenerationRequest:
    return GenerationRequest(
        system_prompt="sys",
        user_prompt="usr",
        role=role,
        effort="medium",
    )


def _make_coordinator(
    tmp_path: Path,
    *,
    backend: object | None = None,
    l2_batch_evaluator: object | None = None,
    l1_compiler: object | None = None,
    static_width_dims_fn: object | None = None,
    regime_specs: Sequence[RegimeTargetSpec] | None = None,
    guardrail_cases: Sequence[GuardrailCaseSpec] = (),
) -> StageGateCoordinator:
    target_path = tmp_path / "kernel.py"
    target_path.write_text(_PARENT_MODULE_SRC, encoding="utf-8")
    specs = tuple(regime_specs) if regime_specs is not None else (_make_regime_spec(),)
    return StageGateCoordinator(
        operator_name="rms_norm",
        target_module_path=target_path,
        regime_specs=specs,
        initial_anchors=_make_anchors(),
        scoreboard_anchors=_make_anchors(),
        guardrail_cases=guardrail_cases,
        backend=backend,
        l0_checker=lambda _src, _reg: _make_stage_a(),
        l1_compiler=l1_compiler or (lambda _src, _reg: StageBResult(passed=True)),
        l2_batch_evaluator=l2_batch_evaluator
        or (
            lambda cands, _ancs: {
                cid: _make_surrogate_result(cid) for cid, _path in cands
            }
        ),
        static_width_dims_fn=static_width_dims_fn,
        max_syntax_repairs=0,
    )


@pytest.mark.anyio
async def test_slow_creative_nonblocking_and_harvested_next_round(tmp_path: Path) -> None:
    """Slow creative worker must not block round 0 and harvests cleanly into round 1."""
    creative_gate = asyncio.Event()

    class _Backend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await creative_gate.wait()
                return (
                    "# target_bottleneck: aiv_vec\n"
                    "# structural_hypothesis: Widen vector tile to 64\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_Backend())
    spec = _make_regime_spec()
    anchors = _make_anchors()

    batch_r0 = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r0 = await coord.execute_round_gates(batch_r0, anchors)
    assert isinstance(out_r0, StageGateOutcome)
    assert [cid for cid, _ in out_r0.raw_candidates] == ["r0_rule_0"]
    assert "r0_rule_0" in out_r0.l2_results
    assert len(coord._pending_speculative) == 1
    assert coord._pending_speculative[0][0] == 0
    assert coord._pending_speculative[0][1] == "r0_creative"

    creative_gate.set()
    await asyncio.sleep(0)

    batch_r1 = CandidateBatch(
        round_idx=1,
        regime_spec=spec,
        worker_specs=[
            ("r1_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r1 = await coord.execute_round_gates(batch_r1, anchors)
    assert coord._pending_speculative == []
    r1_cids = [cid for cid, _ in out_r1.raw_candidates]
    assert "r0_creative" in r1_cids
    assert "r1_rule_0" in r1_cids
    assert out_r1.meta_by_id["r0_creative"][0] == "creative_explore"
    assert "Widen vector tile to 64" in out_r1.meta_by_id["r0_creative"][2]
    assert "r0_creative" in out_r1.l2_results


@pytest.mark.anyio
async def test_fast_creative_finishes_in_same_round(tmp_path: Path) -> None:
    """Fast creative worker finishing before exploit workers is harvested in same round."""

    class _FastCreativeBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await asyncio.sleep(0.005)
                return (
                    "# target_bottleneck: ub_bytes\n"
                    "# structural_hypothesis: Fast creative tile\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            await asyncio.sleep(0.05)
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_FastCreativeBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert coord._pending_speculative == []
    cids = {cid for cid, _ in out.raw_candidates}
    assert cids == {"r0_rule_0", "r0_creative"}
    assert out.meta_by_id["r0_creative"][0] == "creative_explore"
    assert "r0_creative" in out.l2_results


@pytest.mark.anyio
async def test_creative_only_round_awaits_synchronously(tmp_path: Path) -> None:
    """When zero exploit workers exist, creative worker is awaited in the same round."""

    class _CreativeOnlyBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            await asyncio.sleep(0.01)
            return (
                "# target_bottleneck: aiv_vec\n"
                "# structural_hypothesis: Creative only round\n"
                f"```python\n{_CREATIVE_SNIPPET}\n```"
            )

    coord = _make_coordinator(tmp_path, backend=_CreativeOnlyBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert coord._pending_speculative == []
    assert [cid for cid, _ in out.raw_candidates] == ["r0_creative"]


@pytest.mark.anyio
async def test_occupancy_exceptions_produce_typed_infra_occupancy_fault(
    tmp_path: Path,
) -> None:
    """OccupancyExhaustedError and fault_plane=INFRA_OCCUPANCY yield TypedFault."""

    class _SingleBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            return _EXPLOIT_SNIPPET

    class _CustomOccupancyError(Exception):
        fault_plane = "INFRA_OCCUPANCY"

    for exc_to_raise in (
        OccupancyExhaustedError(3, ("all NPU cards busy",)),
        _CustomOccupancyError("card pool saturated"),
    ):

        def _raise_occ(
            _cands: Sequence[tuple[str, Path]],
            _ancs: Sequence[AnchorCaseSpec],
            *,
            _exc: BaseException = exc_to_raise,
        ) -> Mapping[str, CandidateSurrogateResult]:
            raise _exc

        coord = _make_coordinator(
            tmp_path,
            backend=_SingleBackend(),
            l2_batch_evaluator=_raise_occ,
        )
        batch = CandidateBatch(
            round_idx=0,
            regime_spec=_make_regime_spec(),
            worker_specs=[
                ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ],
            parent_module_source=_PARENT_MODULE_SRC,
            parent_snippet=_PARENT_SNIPPET,
            func_name="_build_main",
        )
        out = await coord.execute_round_gates(batch, _make_anchors())
        assert out.l2_infra_fault is not None
        assert out.l2_infra_fault.plane == FaultPlane.INFRA_OCCUPANCY
        assert out.l2_infra_fault.code == "L2_INFRA_OCCUPANCY"
        assert out.l2_infra_fault.failed_tier == "l2_infra_occupancy"
        assert out.l2_infra_fault.retriable is True
        assert out.l2_results == {}


@pytest.mark.anyio
async def test_cancel_pending_speculative_cancels_cleanly(tmp_path: Path) -> None:
    """cancel_pending_speculative cancels running speculative tasks without leaking."""
    started = asyncio.Event()
    cancelled_flag = False

    class _HangCreativeBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            nonlocal cancelled_flag
            _ = effort_override
            if req.role == "explore_creative":
                started.set()
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    cancelled_flag = True
                    raise
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_HangCreativeBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    await coord.execute_round_gates(batch, _make_anchors())
    await started.wait()
    assert len(coord._pending_speculative) == 1
    task = coord._pending_speculative[0][3]

    await coord.cancel_pending_speculative()
    assert coord._pending_speculative == []
    assert task.cancelled()
    assert cancelled_flag is True


def test_run_l1_static_width_dims_and_vf_stack_overflow(tmp_path: Path) -> None:
    """_run_l1 checks distinct static_width_dims_fn dims and flags VF_STACK_OVERFLOW."""
    seen_widths: list[int] = []

    def _l1_check(_src: str, spec: RegimeTargetSpec) -> StageBResult:
        d_val = int(spec.representative_anchor_dims.get("D", 0) or 0)
        seen_widths.append(d_val)
        if d_val >= 512:
            return StageBResult(
                passed=True,
                compiler_summary=BishengResourceSummary(stack_bytes=128),
                elapsed_sec=0.1,
            )
        return StageBResult(passed=True, elapsed_sec=0.1)

    coord = _make_coordinator(
        tmp_path,
        l1_compiler=_l1_check,
        static_width_dims_fn=lambda _spec: ({"D": 128}, {"D": 256}, {"D": 512}),
    )
    res = coord._run_l1(_PARENT_MODULE_SRC, _make_regime_spec())
    assert seen_widths == [128, 256, 512]
    assert res.passed is False
    assert res.fault_code == "VF_STACK_OVERFLOW"
    assert res.fault_plane == "STAGE_B_COMPILE"
    assert coord._has_vf_stack_overflow(StageBResult(passed=False, error_message="vf_stack"))
    assert not coord._has_vf_stack_overflow(StageBResult(passed=True))


def test_paired_l2_normalization_and_unchanged_siblings(tmp_path: Path) -> None:
    """Paired L2 re-anchoring and unchanged sibling AST hashing work cleanly."""
    small_spec = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={},
        editable_symbols=("_build_small",),
    )
    large_spec = RegimeTargetSpec(
        regime_tag="large",
        builder_candidates=("_build_large",),
        sample_builder_kwargs={},
        editable_symbols=("_build_large",),
    )
    inc_src = (
        "def _build_small(tile=16):\n"
        "    return tile\n\n"
        "def _build_large(tile=64):\n"
        "    return tile\n"
    )
    cand_src = (
        "def _build_small(tile=32):\n"
        "    return tile\n\n"
        "def _build_large(tile=64):\n"
        "    return tile\n"
    )
    coord = _make_coordinator(
        tmp_path,
        regime_specs=(small_spec, large_spec),
    )
    coord._incumbent_anchor_us_by_case = {0: 20.0}
    unchanged = coord._unchanged_sibling_regimes(inc_src, cand_src, small_spec)
    assert unchanged == {"large"}

    paired_anc = AnchorEvalMetrics(
        case_num=0,
        regime_tag="small",
        accuracy_passed=True,
        cand_us=19.0,
        ref_perf_us=40.0,
        t_hw_us=4.0,
        speedup=40.0 / 19.0,
        roofline_eta=4.0 / 19.0,
        eq3_score=40.0 / 19.0,
        paired_speedup_vs_incumbent=2.0,
    )
    raw = {
        "c0": CandidateSurrogateResult(
            candidate_id="c0",
            operator_name="rms_norm",
            all_passed=True,
            avg_speedup=paired_anc.speedup,
            min_case_speedup=paired_anc.speedup,
            avg_roofline_eta=paired_anc.roofline_eta,
            surrogate_score=paired_anc.speedup,
            anchors=(paired_anc,),
        )
    }
    norm = coord._normalize_paired_l2_results(raw, None)
    assert norm["c0"].anchors[0].cand_us == pytest.approx(10.0)
    assert norm["c0"].anchors[0].speedup == pytest.approx(4.0)
    assert SpeculativeCreativeResult(0, "w", "src").worker_id == "w"


def test_tail_guard_recheck_and_combine_gate_helpers(tmp_path: Path) -> None:
    """Tail guard delegation, first-round L1 summary cache, and combine gate resolve cleanly."""
    l1_calls = 0

    def _counting_l1(_src: str, _reg: RegimeTargetSpec) -> StageBResult:
        nonlocal l1_calls
        l1_calls += 1
        return StageBResult(
            passed=True,
            compiler_summary=BishengResourceSummary(stack_bytes=0),
        )

    coord = _make_coordinator(tmp_path, l1_compiler=_counting_l1)
    spec = _make_regime_spec()
    summary1 = coord._l1_summary_for_first_round(_PARENT_MODULE_SRC, spec)
    summary2 = coord._l1_summary_for_first_round(_PARENT_MODULE_SRC, spec)
    assert summary1 is not None and summary1 is summary2
    assert l1_calls == 1

    best_res = _make_surrogate_result("inc", cand_us=20.0)
    better_res = _make_surrogate_result("cand", cand_us=10.0)
    ok, updated = coord._evaluate_dual_layer_tail_guard(
        candidate_id="cand",
        candidate_source=_PARENT_MODULE_SRC.replace("16", "32"),
        top_res=better_res,
        best_result=best_res,
        regime_spec=spec,
        active_anchors=_make_anchors(),
        incumbent_source=_PARENT_MODULE_SRC,
    )
    assert ok is True
    assert updated.all_passed is True
    assert coord._resolve_max_row_width_dims(spec) is not None

