"""Unit tests for Stage-B L1 summary cache seeding and cross-regime reuse."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Sequence
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
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (  # noqa: E402
    CandidateBatch,
    StageGateCoordinator,
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

_PARENT_TU = (
    '_TU = r"""\n'
    '__global__ __aicore__ void kernel_entry(GM_ADDR x) { (void)x; }\n'
    '"""\n'
)

_CAND_TU = (
    '_TU = r"""\n'
    '__global__ __aicore__ void kernel_entry(GM_ADDR x) { if (x) {} }\n'
    '"""\n'
)

_SHARED_TRACKB_MODULE = (
    f"{_PARENT_TU}\n"
    "def _select_cfg(D=128):\n"
    '    return {"compile_flags": ("--cce-aicore-arch=dav-c220",)}\n\n'
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n\n"
    "def _build_alt(tile_m=16):\n"
    "    return int(tile_m) + 1\n"
)


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


def _make_surrogate_result(cand_id: str) -> CandidateSurrogateResult:
    anc = AnchorEvalMetrics(
        case_num=0,
        regime_tag="regime_a",
        accuracy_passed=True,
        cand_us=10.0,
        ref_perf_us=20.0,
        t_hw_us=2.0,
        speedup=2.0,
        roofline_eta=0.2,
        eq3_score=2.0,
        paired_rel_mad=0.01,
    )
    return CandidateSurrogateResult(
        candidate_id=cand_id,
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=2.0,
        min_case_speedup=2.0,
        avg_roofline_eta=0.2,
        surrogate_score=2.0,
        anchors=(anc,),
    )


def _make_regime_spec(
    regime_tag: str,
    *,
    builder_name: str = "_build_main",
    representative_anchor_dims: dict[str, int] | None = None,
) -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag=regime_tag,
        builder_candidates=(builder_name,),
        sample_builder_kwargs={"tile_m": 16},
        representative_anchor_dims=representative_anchor_dims or {"D": 128},
        config_selector_fn="_select_cfg",
        editable_symbols=("_TU", "_select_cfg"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_TU",),
    )


def _make_coordinator(
    tmp_path: Path,
    *,
    regime_specs: Sequence[RegimeTargetSpec],
    backend: object | None = None,
    l1_compiler: object | None = None,
) -> StageGateCoordinator:
    target_path = tmp_path / "kernel.py"
    target_path.write_text(_SHARED_TRACKB_MODULE, encoding="utf-8")
    anchors = (AnchorCaseSpec("level2/rms_norm", 0, regime_specs[0].regime_tag, 20.0, 2.0),)
    return StageGateCoordinator(
        operator_name="rms_norm",
        target_module_path=target_path,
        regime_specs=regime_specs,
        initial_anchors=anchors,
        scoreboard_anchors=anchors,
        backend=backend,
        l0_checker=lambda _src, _reg: _make_stage_a(),
        l1_compiler=l1_compiler or (lambda _src, _reg: StageBResult(passed=True)),
        l2_batch_evaluator=lambda cands, _ancs: {
            cid: _make_surrogate_result(cid) for cid, _path in cands
        },
        max_syntax_repairs=0,
    )


@pytest.mark.anyio
async def test_execute_round_gates_seeds_l1_summary_cache_for_subsequent_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing Stage-B in execute_round_gates seeds cache so _run_l1 is skipped later."""
    seeded_summary = BishengResourceSummary(vec_regs=48, stack_bytes=0)
    spec_a = _make_regime_spec("regime_a")
    spec_b = _make_regime_spec("regime_b", representative_anchor_dims={"D": 256})

    class _SingleBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            return _CAND_TU

    l1_calls: list[str] = []

    def _l1_compiler(_src: str, reg: RegimeTargetSpec) -> StageBResult:
        l1_calls.append(reg.regime_tag)
        return StageBResult(passed=True, compiler_summary=seeded_summary)

    coord = _make_coordinator(
        tmp_path,
        regime_specs=(spec_a, spec_b),
        backend=_SingleBackend(),
        l1_compiler=_l1_compiler,
    )
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=spec_a,
        worker_specs=[
            (
                "r0_c0",
                "rule_exploit",
                None,
                GenerationRequest(
                    system_prompt="sys",
                    user_prompt="usr",
                    role="exploit_rule",
                    effort="medium",
                ),
            )
        ],
        parent_module_source=_SHARED_TRACKB_MODULE,
        parent_snippet=_PARENT_TU,
        func_name="_TU",
    )
    anchors = (AnchorCaseSpec("level2/rms_norm", 0, "regime_a", 20.0, 2.0),)
    outcome = await coord.execute_round_gates(batch, anchors)
    assert len(outcome.passed_checks) == 1
    patched_src = outcome.passed_checks[0].patched_module_source
    assert patched_src

    run_l1_calls: list[str] = []

    def _fail_if_called(_src: str, reg: RegimeTargetSpec) -> StageBResult:
        run_l1_calls.append(reg.regime_tag)
        raise AssertionError("_run_l1 should not be invoked on seeded summary")

    monkeypatch.setattr(coord, "_run_l1", _fail_if_called)
    assert coord._l1_summary_for_first_round(patched_src, spec_a) is seeded_summary
    assert coord._l1_summary_for_first_round(patched_src, spec_b) is seeded_summary
    assert run_l1_calls == []


def test_cross_regime_reuse_for_identical_builder_trackb_and_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two regimes sharing builder, Track-B TU, and compile flags invoke _run_l1 once."""
    spec_a = _make_regime_spec("regime_a", representative_anchor_dims={"D": 128})
    spec_b = _make_regime_spec("regime_b", representative_anchor_dims={"D": 512})
    coord = _make_coordinator(tmp_path, regime_specs=(spec_a, spec_b))
    expected_summary = BishengResourceSummary(vec_regs=64, stack_bytes=256)
    run_l1_calls: list[str] = []

    def _counting_run_l1(_src: str, reg: RegimeTargetSpec) -> StageBResult:
        run_l1_calls.append(reg.regime_tag)
        return StageBResult(passed=True, compiler_summary=expected_summary)

    monkeypatch.setattr(coord, "_run_l1", _counting_run_l1)
    s1 = coord._l1_summary_for_first_round(_SHARED_TRACKB_MODULE, spec_a)
    s2 = coord._l1_summary_for_first_round(_SHARED_TRACKB_MODULE, spec_b)
    assert s1 is expected_summary
    assert s2 is expected_summary
    assert run_l1_calls == ["regime_a"]


def test_distinct_builders_do_not_collide_in_l1_summary_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two regimes resolving to distinct builders compile separately without collision."""
    spec_a = _make_regime_spec("regime_a", builder_name="_build_main")
    spec_b = _make_regime_spec("regime_b", builder_name="_build_alt")
    coord = _make_coordinator(tmp_path, regime_specs=(spec_a, spec_b))
    summary_a = BishengResourceSummary(vec_regs=32)
    summary_b = BishengResourceSummary(vec_regs=96)
    run_l1_calls: list[str] = []

    def _per_regime_run_l1(_src: str, reg: RegimeTargetSpec) -> StageBResult:
        run_l1_calls.append(reg.regime_tag)
        summary = summary_a if reg.regime_tag == "regime_a" else summary_b
        return StageBResult(passed=True, compiler_summary=summary)

    monkeypatch.setattr(coord, "_run_l1", _per_regime_run_l1)
    assert coord._l1_summary_for_first_round(_SHARED_TRACKB_MODULE, spec_a) is summary_a
    assert coord._l1_summary_for_first_round(_SHARED_TRACKB_MODULE, spec_b) is summary_b
    assert run_l1_calls == ["regime_a", "regime_b"]
