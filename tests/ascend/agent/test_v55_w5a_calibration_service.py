"""Unit tests for CalibrationSnapshot and CalibrationService."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.calibration_service import (
    CalibrationService,
    CalibrationSnapshot,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L2BatchResults,
    L3AcceptanceResult,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline


def _sha256(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 40.0,
    t_hw_us: float = 4.0,
    eq3_score: float | None = None,
    accuracy_passed: bool = True,
) -> AnchorEvalMetrics:
    eff_eq3 = (ref_us / cand_us) if (eq3_score is None and cand_us > 0.0) else (eq3_score or 0.0)
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=(ref_us / cand_us) if cand_us > 0.0 else 0.0,
        roofline_eta=(t_hw_us / cand_us) if cand_us > 0.0 else 0.0,
        eq3_score=eff_eq3,
        paired_rel_mad=0.01,
    )


def _result(
    candidate_id: str,
    anchors: tuple[AnchorEvalMetrics, ...],
    *,
    all_passed: bool = True,
) -> CandidateSurrogateResult:
    valid = [a for a in anchors if a.cand_us > 0.0]
    avg_sp = sum(a.speedup for a in valid) / max(1, len(valid))
    min_sp = min((a.speedup for a in valid), default=0.0)
    avg_eta = sum(a.roofline_eta for a in valid) / max(1, len(valid))
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name="synthetic_op",
        all_passed=all_passed,
        avg_speedup=avg_sp,
        min_case_speedup=min_sp,
        avg_roofline_eta=avg_eta,
        surrogate_score=avg_sp,
        anchors=anchors,
    )


def _anchors() -> list[AnchorCaseSpec]:
    return [
        AnchorCaseSpec("synthetic_op", 101, "decode", ref_perf_us=20.0, t_hw_us=4.0),
        AnchorCaseSpec("synthetic_op", 102, "decode", ref_perf_us=30.0, t_hw_us=6.0),
        AnchorCaseSpec("synthetic_op", 201, "prefill", ref_perf_us=50.0, t_hw_us=10.0),
    ]


def _regimes() -> list[RegimeTargetSpec]:
    return [
        RegimeTargetSpec(
            regime_tag="decode",
            builder_candidates=("_build_decode",),
            sample_builder_kwargs={"tile_m": 16},
            config_selector_fn="_select_cfg",
            representative_anchor_dims={"M": 1, "N": 1024, "K": 1024},
        ),
        RegimeTargetSpec(
            regime_tag="prefill",
            builder_candidates=("_build_prefill",),
            sample_builder_kwargs={"tile_m": 64},
            config_selector_fn="_select_cfg",
            representative_anchor_dims={"M": 128, "N": 1024, "K": 1024},
        ),
    ]


@dataclass(frozen=True)
class _LatticeCand:
    patched_source: str
    pred_us: float


def test_calibration_snapshot_is_frozen_and_captures_null_launch() -> None:
    """Verify CalibrationSnapshot immutability, null_launch_us capture, and floor."""
    src = "def _build_decode(tile_m=16):\n    return tile_m\n"
    anchors = _anchors()
    regimes = _regimes()
    base_res = _result(
        "incumbent_baseline",
        (
            _metric(101, "decode", 25.0, ref_us=20.0, t_hw_us=4.0),
            _metric(201, "prefill", 50.0, ref_us=50.0, t_hw_us=10.0),
        ),
    )

    def _l2_eval(cands: list[tuple[str, Path]], eval_anchors: list[AnchorCaseSpec]) -> L2BatchResults:
        assert len(cands) == 1
        assert cands[0][0] == "incumbent_baseline"
        assert len(eval_anchors) == 3
        return L2BatchResults(
            {"incumbent_baseline": base_res},
            {"incumbent_baseline": "run-001"},
            null_launch_us=1.75,
        )

    def _msprof(
        op_name: str,
        cand_path: str,
        anc_seq: list[AnchorCaseSpec],
    ) -> dict[str, dict[str, float]]:
        assert op_name == "synthetic_op"
        assert Path(cand_path).is_file()
        return {
            spec.regime_tag: {"pipe_v_util": float(spec.case_num)}
            for spec in anc_seq
        }

    service = CalibrationService(
        operator_name="synthetic_op",
        run_l2_batch_fn=_l2_eval,
        msprof_collector=_msprof,
    )
    snapshot = service.build_snapshot(src, anchors, regimes)

    assert isinstance(snapshot, CalibrationSnapshot)
    assert snapshot.null_launch_us == pytest.approx(1.75)
    assert snapshot.calibrated_dispatch_floor_us == pytest.approx(1.75)
    assert snapshot.baseline_result == base_res
    assert snapshot.anchor_t_hw_by_case == {101: 4.0, 102: 6.0, 201: 10.0}
    sha = _sha256(src)
    assert snapshot.msprof_by_case == {
        (sha, 101): {"pipe_v_util": 101.0},
        (sha, 201): {"pipe_v_util": 201.0},
    }
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "calibrated_dispatch_floor_us", 99.0)


def test_calibrate_dispatch_floor_dual_sided_physical_cap() -> None:
    """Verify fallback dispatch floor caps slow baseline cand_us by ref_us."""
    base_res = _result(
        "incumbent_baseline",
        (
            _metric(101, "decode", 120.0, ref_us=10.0, t_hw_us=4.0),
            _metric(201, "prefill", 60.0, ref_us=80.0, t_hw_us=50.0),
        ),
    )
    service = CalibrationService(operator_name="synthetic_op")
    floor = service.calibrate_dispatch_floor(base_res, null_launch_us=None)
    assert floor == pytest.approx(6.0)

    override_floor = service.calibrate_dispatch_floor(base_res, null_launch_us=2.25)
    assert override_floor == pytest.approx(2.25)


def test_msprof_by_case_isolates_cases_and_sources_without_clobbering() -> None:
    """Verify msprof_by_case keys by (sha, case_num) across bottlenecks and sources."""
    src_a = "def _build_decode():\n    return 16\n"
    src_b = "def _build_decode():\n    return 32\n"
    anchors = _anchors()
    regimes = [_regimes()[0]]

    calls: list[tuple[str, list[int]]] = []

    def _msprof(
        _op: str,
        cand_path: str,
        anc_seq: list[AnchorCaseSpec],
    ) -> dict[str, dict[str, float]]:
        src_text = Path(cand_path).read_text(encoding="utf-8")
        calls.append((_sha256(src_text), [int(a.case_num) for a in anc_seq]))
        return {
            spec.regime_tag: {"case_marker": float(spec.case_num), "src_len": float(len(src_text))}
            for spec in anc_seq
        }

    service = CalibrationService(operator_name="synthetic_op", msprof_collector=_msprof)

    res_a_case101 = _result(
        "cand_a",
        (
            _metric(101, "decode", 20.0, eq3_score=0.5),
            _metric(102, "decode", 10.0, eq3_score=0.9),
        ),
    )
    out1 = service.collect_msprof_by_case(src_a, res_a_case101, anchors, regimes)
    assert out1 == {(_sha256(src_a), 101): {"case_marker": 101.0, "src_len": float(len(src_a))}}

    res_a_case102 = _result(
        "cand_a",
        (
            _metric(101, "decode", 10.0, eq3_score=0.9),
            _metric(102, "decode", 25.0, eq3_score=0.4),
        ),
    )
    out2 = service.collect_msprof_by_case(src_a, res_a_case102, anchors, regimes)
    assert (_sha256(src_a), 101) in out2
    assert (_sha256(src_a), 102) in out2
    assert out2[(_sha256(src_a), 101)]["case_marker"] == pytest.approx(101.0)
    assert out2[(_sha256(src_a), 102)]["case_marker"] == pytest.approx(102.0)

    out_cached = service.collect_msprof_by_case(src_a, res_a_case102, anchors, regimes)
    assert out_cached == out2
    assert len(calls) == 2

    out_b = service.collect_msprof_by_case(src_b, res_a_case101, anchors, regimes)
    assert out_b == {(_sha256(src_b), 101): {"case_marker": 101.0, "src_len": float(len(src_b))}}
    assert len(service.msprof_by_case) == 3
    assert service.msprof_by_case[(_sha256(src_a), 101)]["src_len"] == pytest.approx(float(len(src_a)))
    assert service.msprof_by_case[(_sha256(src_b), 101)]["src_len"] == pytest.approx(float(len(src_b)))


def test_collect_p2_deep_dive_for_regime_caches_and_handles_failures() -> None:
    """Verify P2 deep-dive collection selects bottleneck, caches by sha/regime, and recovers."""
    src = "def _build_decode():\n    return 16\n"
    anchors = _anchors()
    decode_regime = _regimes()[0]
    res = _result(
        "cand",
        (
            _metric(101, "decode", 15.0, eq3_score=0.8),
            _metric(102, "decode", 30.0, eq3_score=0.3),
        ),
    )
    recorded_cases: list[int] = []

    def _p2_ok(
        _op: str,
        _cand_path: str,
        anc_seq: list[AnchorCaseSpec],
    ) -> dict[str, dict[str, object]]:
        recorded_cases.extend(int(a.case_num) for a in anc_seq)
        return {"decode": {"hotspot": "loop_0", "stall_cycles": 420}}

    service = CalibrationService(operator_name="synthetic_op", p2_deep_dive_collector=_p2_ok)
    first = service.collect_p2_deep_dive_for_regime(src, res, anchors, decode_regime)
    second = service.collect_p2_deep_dive_for_regime(src, res, anchors, decode_regime)
    assert first == {"hotspot": "loop_0", "stall_cycles": 420}
    assert second == first
    assert recorded_cases == [102]

    def _p2_fail(_op: str, _cand_path: str, _anc_seq: list[AnchorCaseSpec]) -> dict[str, object]:
        raise RuntimeError("transient profiler error")

    failing_service = CalibrationService(
        operator_name="synthetic_op",
        p2_deep_dive_collector=_p2_fail,
    )
    assert failing_service.collect_p2_deep_dive_for_regime(src, res, anchors, decode_regime) is None


def test_run_phase0_lattice_warmstart_promotes_and_rolls_back_on_l3_failure(
    tmp_path: Path,
) -> None:
    """Verify Phase-0 lattice warm-start promotes passing winners and rolls back L3 failures."""
    inc_src = "def _select_cfg():\n    return {'bm': 16}\n\ndef _build_decode():\n    return 16\n"
    lat_src_0 = "def _select_cfg():\n    return {'bm': 32}\n\ndef _build_decode():\n    return 32\n"
    lat_src_1 = "def _select_cfg():\n    return {'bm': 64}\n\ndef _build_decode():\n    return 64\n"
    target_path = tmp_path / "kernel.py"
    target_path.write_text(inc_src, encoding="utf-8")

    anchors = [_anchors()[0]]
    regimes = [_regimes()[0]]
    base_res = _result("incumbent", (_metric(101, "decode", 20.0),))
    win_res = _result("phase0_lat_1", (_metric(101, "decode", 8.0),))

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
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines=waterlines,
        gm_roundtrip_bytes=16384,
        grid_blocks=16,
    )

    def _run_l0(_src: str, _reg: RegimeTargetSpec) -> StageAResult:
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _sweep(*_args: object, **_kwargs: object) -> list[_LatticeCand]:
        return [
            _LatticeCand(patched_source=lat_src_0, pred_us=10.0),
            _LatticeCand(patched_source=lat_src_1, pred_us=11.0),
        ]

    def _l2_eval(
        cands: list[tuple[str, Path]],
        _anchors: list[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        assert [cid for cid, _ in cands] == ["phase0_lat_0", "phase0_lat_1"]
        return {
            "phase0_lat_0": _result("phase0_lat_0", (_metric(101, "decode", 12.0),)),
            "phase0_lat_1": win_res,
        }

    promoted_events: list[tuple[str, CandidateSurrogateResult]] = []
    refreshed_cids: list[str] = []

    service_promote = CalibrationService(
        operator_name="synthetic_op",
        target_module_path=target_path,
        run_l0_fn=_run_l0,
        predict_candidate_us_fn=lambda _wl: 20.0,
        sweep_lattice_fn=_sweep,
        run_l2_batch_fn=_l2_eval,
        evaluate_tail_guard_fn=lambda **kw: (True, kw["top_res"]),
        run_l3_contract_fn=lambda _p: L3AcceptanceResult(passed=True),
        refresh_guardrail_refs_fn=refreshed_cids.append,
        on_warmstart_promoted_fn=lambda s, r: promoted_events.append((s, r)),
        enable_phase0_lattice_sweep=True,
        regime_specs=regimes,
    )
    out_src, out_res = service_promote.run_phase0_lattice_warmstart(inc_src, base_res, anchors)
    assert out_src == lat_src_1
    assert out_res == win_res
    assert target_path.read_text(encoding="utf-8") == lat_src_1
    assert refreshed_cids == ["phase0_lat_1"]
    assert promoted_events == [(lat_src_1, win_res)]

    target_path.write_text(inc_src, encoding="utf-8")
    service_rollback = CalibrationService(
        operator_name="synthetic_op",
        target_module_path=target_path,
        run_l0_fn=_run_l0,
        predict_candidate_us_fn=lambda _wl: 20.0,
        sweep_lattice_fn=_sweep,
        run_l2_batch_fn=_l2_eval,
        evaluate_tail_guard_fn=lambda **kw: (True, kw["top_res"]),
        run_l3_contract_fn=lambda _p: L3AcceptanceResult(passed=False),
        enable_phase0_lattice_sweep=True,
        regime_specs=regimes,
    )
    rb_src, rb_res = service_rollback.run_phase0_lattice_warmstart(inc_src, base_res, anchors)
    assert rb_src == inc_src
    assert rb_res == base_res
    assert target_path.read_text(encoding="utf-8") == inc_src


def test_evaluate_baseline_and_phase0_skip_paths(tmp_path: Path) -> None:
    """Verify baseline null_launch fallback, missing result error, and warm-start skips."""
    src = "def _select_cfg():\n    return {'bm': 16}\n\ndef _build_decode():\n    return 16\n"
    anchors = [_anchors()[0]]
    regimes = [_regimes()[0]]
    base_res = _result("incumbent_baseline", (_metric(101, "decode", 20.0),))

    svc_no_null = CalibrationService(
        operator_name="synthetic_op",
        run_l2_batch_fn=lambda _c, _a: ({"incumbent_baseline": base_res}, {"incumbent_baseline": "r1"}),
    )
    res, null_us = svc_no_null.evaluate_baseline(src, anchors)
    assert res == base_res
    assert null_us is None

    svc_missing = CalibrationService(
        operator_name="synthetic_op",
        run_l2_batch_fn=lambda _c, _a: {},
    )
    with pytest.raises(RuntimeError, match="incumbent_baseline"):
        svc_missing.evaluate_baseline(src, anchors)

    svc_disabled = CalibrationService(
        operator_name="synthetic_op",
        target_module_path=tmp_path / "k.py",
        enable_phase0_lattice_sweep=False,
        regime_specs=regimes,
    )
    assert svc_disabled.run_phase0_lattice_warmstart(src, base_res, anchors) == (src, base_res)
