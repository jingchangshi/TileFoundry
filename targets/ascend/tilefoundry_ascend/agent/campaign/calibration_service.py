"""Baseline evaluation, dispatch-floor calibration, profiling, and phase-0 warm-start."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.latency_algebra import (
    AnchorConservationPoint,
    calibrate_dispatch_floor_us,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import to_conservation_point
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    bottleneck_anchor_for_regime,
)
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as _tilesize_prescreen_mod
from tilefoundry_ascend.ascend.cost_model import build_cost_features_from_ledger
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePlanResult
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Immutable baseline evaluation, dispatch floor, and profiling snapshot."""

    baseline_result: CandidateSurrogateResult
    null_launch_us: float | None
    calibrated_dispatch_floor_us: float
    msprof_by_case: dict[tuple[str, int], dict[str, float]]
    anchor_t_hw_by_case: dict[int, float]


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to path via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _invoke_with_optional_kwargs(
    runner: Callable[..., _T],
    *args: object,
    **kwargs: object,
) -> _T:
    """Pass only the optional keywords an injected callable accepts."""
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(*args)
    accepts_any = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in kwargs.items()
        if accepts_any or name in signature.parameters
    }
    return runner(*args, **accepted)


def _surrogate_rank_key(res: CandidateSurrogateResult) -> tuple[float, float, float]:
    """Return lexicographic ranking key for passing surrogate results."""
    return (res.surrogate_score, res.min_case_speedup, res.avg_speedup)


def _extract_valid_non_negative_float(value: object) -> float | None:
    """Return finite non-negative float from value, or None if invalid."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed) and parsed >= 0.0:
            return parsed
    return None


def _unpack_l2_batch_output(
    raw_out: object,
) -> tuple[Mapping[str, CandidateSurrogateResult] | None, float | None]:
    """Extract results mapping and optional null_launch_us from an L2 batch return value."""
    results_obj = raw_out[0] if isinstance(raw_out, tuple) and raw_out else raw_out
    null_launch_us = _extract_valid_non_negative_float(
        getattr(results_obj, "null_launch_us", None)
    )
    if null_launch_us is None and raw_out is not results_obj:
        null_launch_us = _extract_valid_non_negative_float(
            getattr(raw_out, "null_launch_us", None)
        )
    if not isinstance(results_obj, Mapping):
        return None, null_launch_us
    return results_obj, null_launch_us


def _resolve_anchor_for_regime(
    result: CandidateSurrogateResult,
    active_anchors: Sequence[AnchorCaseSpec],
    regime_spec: RegimeTargetSpec,
    *,
    allow_fallback_anchor: bool,
) -> AnchorCaseSpec | None:
    """Resolve the AnchorCaseSpec matching the bottleneck anchor of regime_spec."""
    by_case_and_tag = {(int(a.case_num), a.regime_tag): a for a in active_anchors}
    by_case = {int(a.case_num): a for a in active_anchors}
    bn = bottleneck_anchor_for_regime(result, regime_spec.regime_tag)
    spec: AnchorCaseSpec | None = None
    if bn is not None:
        spec = by_case_and_tag.get((int(bn.case_num), regime_spec.regime_tag)) or by_case.get(
            int(bn.case_num)
        )
    if spec is None and allow_fallback_anchor:
        for anc in active_anchors:
            if anc.regime_tag == regime_spec.regime_tag:
                spec = anc
                break
        if spec is None and active_anchors:
            spec = active_anchors[0]
    if spec is None:
        return None
    if spec.regime_tag != regime_spec.regime_tag:
        return AnchorCaseSpec(
            spec.task,
            spec.case_num,
            regime_spec.regime_tag,
            spec.ref_perf_us,
            spec.t_hw_us,
        )
    return spec


class CalibrationService:
    """Encapsulate baseline evaluation, floor calibration, profiling, and phase-0 warm-start."""

    def __init__(
        self,
        *,
        operator_name: str,
        target_module_path: Path | None = None,
        run_l2_batch_fn: Callable[..., object] | None = None,
        msprof_collector: (
            Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, Mapping[str, float]]]
            | None
        ) = None,
        p2_deep_dive_collector: (
            Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, object]] | None
        ) = None,
        run_l0_fn: Callable[..., object] | None = None,
        derive_topology_axes_fn: Callable[..., TopologyAxes] | None = None,
        predict_candidate_us_fn: Callable[[object], object] | None = None,
        underlying_cost_model: object | None = None,
        cost_advisor_target: AscendTarget | None = None,
        select_active_regime_fn: Callable[..., RegimeTargetSpec | None] | None = None,
        evaluate_tail_guard_fn: Callable[..., tuple[bool, CandidateSurrogateResult]] | None = None,
        run_l3_contract_fn: Callable[[Path], L3AcceptanceResult] | None = None,
        write_and_verify_fn: Callable[[Path, str], str] | None = None,
        refresh_guardrail_refs_fn: Callable[[str], None] | None = None,
        on_warmstart_promoted_fn: Callable[[str, CandidateSurrogateResult], None] | None = None,
        sweep_lattice_fn: Callable[..., Sequence[object]] | None = None,
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        live_regime_plans: Mapping[str, RegimePlanResult] | None = None,
        sil_advisory: object | None = None,
        enable_phase0_lattice_sweep: bool = False,
        regime_specs: Sequence[RegimeTargetSpec] = (),
        msprof_by_case: dict[tuple[str, int], dict[str, float]] | None = None,
        p2_cache: dict[tuple[str, str], dict[str, object]] | None = None,
    ) -> None:
        self._operator_name = operator_name
        self._target_module_path = target_module_path
        self._run_l2_batch_fn = run_l2_batch_fn
        self._msprof_collector = msprof_collector
        self._p2_deep_dive_collector = p2_deep_dive_collector
        self._run_l0_fn = run_l0_fn
        self._derive_topology_axes_fn = derive_topology_axes_fn
        self._predict_candidate_us_fn = predict_candidate_us_fn
        self._underlying_cost_model = underlying_cost_model
        self._cost_advisor_target = cost_advisor_target
        self._select_active_regime_fn = select_active_regime_fn
        self._evaluate_tail_guard_fn = evaluate_tail_guard_fn
        self._run_l3_contract_fn = run_l3_contract_fn
        self._write_and_verify_fn = write_and_verify_fn
        self._refresh_guardrail_refs_fn = refresh_guardrail_refs_fn
        self._on_warmstart_promoted_fn = on_warmstart_promoted_fn
        self._sweep_lattice_fn = sweep_lattice_fn
        self._regime_blueprints = dict(regime_blueprints) if regime_blueprints else {}
        self._live_regime_plans = dict(live_regime_plans) if live_regime_plans else {}
        self._sil_advisory = sil_advisory
        self._enable_phase0_lattice_sweep = bool(enable_phase0_lattice_sweep)
        self._regime_specs = tuple(regime_specs)
        self._msprof_by_case: dict[tuple[str, int], dict[str, float]] = (
            msprof_by_case if msprof_by_case is not None else {}
        )
        self._p2_cache: dict[tuple[str, str], dict[str, object]] = (
            p2_cache if p2_cache is not None else {}
        )

    @property
    def msprof_by_case(self) -> dict[tuple[str, int], dict[str, float]]:
        """Return cached msprof telemetry keyed by (source_sha256, case_num)."""
        return self._msprof_by_case

    @property
    def p2_cache(self) -> dict[tuple[str, str], dict[str, object]]:
        """Return cached deep-dive telemetry keyed by (source_sha256, regime_tag)."""
        return self._p2_cache

    @staticmethod
    def _source_sha256(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def evaluate_baseline(
        self,
        incumbent_source: str,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[CandidateSurrogateResult, float | None]:
        """Evaluate incumbent source on active anchors and capture null_launch_us."""
        if self._run_l2_batch_fn is None:
            raise RuntimeError("Baseline evaluation requires run_l2_batch_fn.")
        with tempfile.TemporaryDirectory(prefix="tf_camp_base_") as tmp_dir:
            mod_path = Path(tmp_dir) / "incumbent.py"
            mod_path.write_text(incumbent_source, encoding="utf-8")
            raw_out = _invoke_with_optional_kwargs(
                self._run_l2_batch_fn,
                [("incumbent_baseline", mod_path)],
                active_anchors,
                include_guardrails=False,
            )
        results, null_launch_us = _unpack_l2_batch_output(raw_out)
        if results is None or "incumbent_baseline" not in results:
            raise RuntimeError("Baseline evaluation missing incumbent_baseline result.")
        return results["incumbent_baseline"], null_launch_us

    def calibrate_dispatch_floor(
        self,
        baseline_result: CandidateSurrogateResult,
        *,
        null_launch_us: float | None = None,
    ) -> float:
        """Calibrate dispatch floor from null_launch_us or dual-sided physical anchor cap."""
        capped_points: list[AnchorConservationPoint] = []
        for anchor in baseline_result.anchors:
            pt = to_conservation_point(anchor, anchor)
            if (
                math.isfinite(pt.ref_us)
                and pt.ref_us > 0.0
                and math.isfinite(pt.cand_us)
                and pt.cand_us > 0.0
            ):
                pt = AnchorConservationPoint(
                    case_num=pt.case_num,
                    regime_tag=pt.regime_tag,
                    ref_us=pt.ref_us,
                    inc_us=pt.inc_us,
                    cand_us=min(pt.cand_us, pt.ref_us),
                    t_roof_us=pt.t_roof_us,
                    paired_rel_mad=pt.paired_rel_mad,
                    accuracy_passed=pt.accuracy_passed,
                )
            capped_points.append(pt)
        try:
            accepts_null = (
                "null_launch_us" in inspect.signature(calibrate_dispatch_floor_us).parameters
            )
        except (TypeError, ValueError):
            accepts_null = False
        if accepts_null:
            return float(
                calibrate_dispatch_floor_us(
                    capped_points,
                    null_launch_us=null_launch_us,
                )
            )
        valid_null = _extract_valid_non_negative_float(null_launch_us)
        if valid_null is not None:
            return valid_null
        return float(calibrate_dispatch_floor_us(capped_points))

    def collect_msprof_by_case(
        self,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        target_regimes: Sequence[RegimeTargetSpec],
    ) -> dict[tuple[str, int], dict[str, float]]:
        """Collect msprof telemetry keyed strictly by (source_sha256, case_num)."""
        sha = self._source_sha256(source)
        if self._msprof_collector is not None:
            anchors_to_collect: list[AnchorCaseSpec] = []
            seen_pending_cases: set[int] = set()
            for rspec in target_regimes:
                spec = _resolve_anchor_for_regime(
                    result,
                    active_anchors,
                    rspec,
                    allow_fallback_anchor=False,
                )
                if spec is None:
                    continue
                cnum = int(spec.case_num)
                if (sha, cnum) in self._msprof_by_case or cnum in seen_pending_cases:
                    continue
                seen_pending_cases.add(cnum)
                anchors_to_collect.append(spec)
            if anchors_to_collect:
                self._run_msprof_collection(sha, source, anchors_to_collect)
        return {
            key: dict(metrics)
            for key, metrics in self._msprof_by_case.items()
            if key[0] == sha
        }

    def _run_msprof_collection(
        self,
        sha: str,
        source: str,
        anchors_to_collect: Sequence[AnchorCaseSpec],
    ) -> None:
        if self._msprof_collector is None:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="tf_msprof_cand_") as tmp_dir:
                cand_path = Path(tmp_dir) / "candidate.py"
                cand_path.write_text(source, encoding="utf-8")
                collected = self._msprof_collector(
                    self._operator_name,
                    str(cand_path),
                    anchors_to_collect,
                )
        except (
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            logger.warning("On-demand msprof collection failed: %s", exc)
            return
        if not isinstance(collected, Mapping):
            return
        case_by_tag = {spec.regime_tag: int(spec.case_num) for spec in anchors_to_collect}
        for raw_key, raw_metrics in collected.items():
            if not isinstance(raw_metrics, Mapping):
                continue
            clean = {
                str(k): float(v)
                for k, v in raw_metrics.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(float(v))
            }
            if not clean:
                continue
            resolved_case = self._resolve_collected_case_num(raw_key, case_by_tag)
            if resolved_case is not None:
                self._msprof_by_case[(sha, resolved_case)] = clean

    @staticmethod
    def _resolve_collected_case_num(
        raw_key: object,
        case_by_tag: Mapping[str, int],
    ) -> int | None:
        if isinstance(raw_key, int) and not isinstance(raw_key, bool):
            return int(raw_key)
        if (
            isinstance(raw_key, tuple)
            and len(raw_key) == 2
            and isinstance(raw_key[1], int)
            and not isinstance(raw_key[1], bool)
        ):
            return int(raw_key[1])
        key_str = str(raw_key)
        if key_str in case_by_tag:
            return case_by_tag[key_str]
        if key_str.lstrip("-").isdigit():
            return int(key_str)
        return None

    def collect_p2_deep_dive_for_regime(
        self,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        regime_spec: RegimeTargetSpec,
    ) -> dict[str, object] | None:
        """Collect deep-dive profiling payload for the bottleneck anchor of regime_spec."""
        sha = self._source_sha256(source)
        cache_key = (sha, regime_spec.regime_tag)
        if cache_key in self._p2_cache:
            return self._p2_cache[cache_key]
        if self._p2_deep_dive_collector is None:
            return None
        spec = _resolve_anchor_for_regime(
            result,
            active_anchors,
            regime_spec,
            allow_fallback_anchor=True,
        )
        if spec is None:
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="tf_p2_cand_") as tmp_dir:
                cand_path = Path(tmp_dir) / "candidate.py"
                cand_path.write_text(source, encoding="utf-8")
                collected = self._p2_deep_dive_collector(
                    self._operator_name,
                    str(cand_path),
                    [spec],
                )
        except (
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.warning("On-demand P2 deep-dive collection failed: %s", exc)
            return None
        if not isinstance(collected, Mapping) or not collected:
            return None
        raw_payload = collected.get(regime_spec.regime_tag)
        if not isinstance(raw_payload, Mapping):
            raw_payload = next(iter(collected.values()), None)
        if not isinstance(raw_payload, Mapping) or not raw_payload:
            return None
        clean_payload = {str(k): v for k, v in raw_payload.items()}
        self._p2_cache[cache_key] = clean_payload
        return clean_payload

    def run_phase0_lattice_warmstart(
        self,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[str, CandidateSurrogateResult]:
        """Run zero-LLM executable lattice sweep before Round 1 to warm-start incumbent."""
        if not self._enable_phase0_lattice_sweep or not self._regime_specs:
            return incumbent_source, best_result
        sweep_fn = self._sweep_lattice_fn or getattr(
            _tilesize_prescreen_mod, "sweep_executable_lattice_zero_llm", None
        )
        if not callable(sweep_fn):
            return incumbent_source, best_result
        primary_regime = (
            self._select_active_regime_fn(
                None,
                set(),
                best_result,
                active_anchors,
                allow_satisfied_fallback=True,
            )
            if self._select_active_regime_fn is not None
            else None
        ) or self._regime_specs[0]
        selector_fn = primary_regime.config_selector_fn
        if not selector_fn or self._run_l0_fn is None or self._run_l2_batch_fn is None:
            return incumbent_source, best_result
        try:
            parent_pred_us = self._predict_parent_lattice_us(incumbent_source, primary_regime)
            if parent_pred_us is None:
                return incumbent_source, best_result
            builder_fn = self._resolve_builder_fn(incumbent_source, primary_regime)
            if not builder_fn:
                return incumbent_source, best_result
            dims = (
                primary_regime.representative_anchor_dims
                or primary_regime.representative_hir_dims
            )
            lattice_cands = list(
                sweep_fn(
                    incumbent_source,
                    selector_fn_name=selector_fn,
                    builder_fn_name=builder_fn,
                    sample_builder_kwargs=primary_regime.sample_builder_kwargs,
                    dims=dims,
                    cost_model=self._underlying_cost_model,
                    max_points=24,
                    top_k=3,
                )
            )
            if not lattice_cands:
                return incumbent_source, best_result
            top_pred_us = float(getattr(lattice_cands[0], "pred_us", 0.0) or 0.0)
            if parent_pred_us > 0.0 and (
                top_pred_us <= 0.0 or top_pred_us >= parent_pred_us * 0.97
            ):
                return incumbent_source, best_result
            top_two = [
                cand
                for cand in lattice_cands[:2]
                if isinstance(getattr(cand, "patched_source", None), str)
                and cand.patched_source.strip()
                and cand.patched_source != incumbent_source
            ]
            if not top_two:
                return incumbent_source, best_result
            return self._evaluate_and_promote_warmstart(
                top_two=top_two,
                incumbent_source=incumbent_source,
                best_result=best_result,
                primary_regime=primary_regime,
                active_anchors=active_anchors,
            )
        except (
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
            TypeError,
            SyntaxError,
        ) as exc:
            logger.debug("Phase-0 lattice warm-start skipped: %s", exc)
            return incumbent_source, best_result

    def _predict_parent_lattice_us(
        self,
        incumbent_source: str,
        primary_regime: RegimeTargetSpec,
    ) -> float | None:
        if self._run_l0_fn is None:
            return None
        parent_sa = self._run_l0_fn(incumbent_source, primary_regime)
        parent_ledger = getattr(parent_sa, "ledger", None)
        if not getattr(parent_sa, "passed", False) or parent_ledger is None:
            return None
        if self._predict_candidate_us_fn is None:
            return 0.0
        parent_features = getattr(parent_sa, "features", None)
        parent_route = getattr(parent_sa, "route", None)
        if self._derive_topology_axes_fn is not None:
            parent_axes = self._derive_topology_axes_fn(
                parent_ledger,
                features=parent_features,
                route=parent_route,
            )
        else:
            parent_axes = resolve_canonical_topology_axes(
                {"op_name": self._operator_name, "uses_cube": False},
                features=parent_features,
            )
        parent_wl = build_cost_features_from_ledger(
            parent_ledger,
            live_plan=self._live_regime_plans.get(primary_regime.regime_tag),
            axes=parent_axes,
            blueprint=self._regime_blueprints.get(primary_regime.regime_tag),
            sil_advisory=self._sil_advisory,
            target=self._cost_advisor_target,
        )
        parent_pred = self._predict_candidate_us_fn(parent_wl)
        if isinstance(parent_pred, (int, float)) and not isinstance(parent_pred, bool):
            return float(parent_pred)
        if parent_pred is not None:
            return float(getattr(parent_pred, "pred_us", 0.0) or 0.0)
        return 0.0

    @staticmethod
    def _resolve_builder_fn(
        incumbent_source: str,
        primary_regime: RegimeTargetSpec,
    ) -> str:
        try:
            tree = ast.parse(incumbent_source)
            defined_funcs = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
        except SyntaxError:
            defined_funcs = set()
        return next(
            (b for b in primary_regime.builder_candidates if b in defined_funcs),
            primary_regime.builder_candidates[0]
            if primary_regime.builder_candidates
            else "",
        )

    def _evaluate_and_promote_warmstart(
        self,
        *,
        top_two: Sequence[object],
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        primary_regime: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[str, CandidateSurrogateResult]:
        if self._run_l2_batch_fn is None:
            return incumbent_source, best_result
        with tempfile.TemporaryDirectory(prefix="tf_camp_p0_") as tmp_dir:
            cand_paths: list[tuple[str, Path]] = []
            source_by_cid: dict[str, str] = {}
            for idx, cand in enumerate(top_two):
                patched_src = str(getattr(cand, "patched_source", ""))
                cid = f"phase0_lat_{idx}"
                cpath = Path(tmp_dir) / f"{cid}.py"
                cpath.write_text(patched_src, encoding="utf-8")
                cand_paths.append((cid, cpath))
                source_by_cid[cid] = patched_src
            raw_out = _invoke_with_optional_kwargs(
                self._run_l2_batch_fn,
                cand_paths,
                active_anchors,
                include_guardrails=True,
            )
        l2_results, _ = _unpack_l2_batch_output(raw_out)
        if l2_results is None:
            return incumbent_source, best_result
        passing = [
            (cid, res)
            for cid, res in l2_results.items()
            if res.all_passed and cid in source_by_cid
        ]
        if not passing:
            return incumbent_source, best_result
        top_cid, top_res = max(passing, key=lambda item: _surrogate_rank_key(item[1]))
        top_src = source_by_cid[top_cid]
        if self._evaluate_tail_guard_fn is not None:
            tail_ok, top_res = self._evaluate_tail_guard_fn(
                candidate_id=top_cid,
                candidate_source=top_src,
                top_res=top_res,
                best_result=best_result,
                regime_spec=primary_regime,
                active_anchors=active_anchors,
                incumbent_source=incumbent_source,
            )
            if not tail_ok:
                return incumbent_source, best_result
        if self._target_module_path is not None and self._run_l3_contract_fn is not None:
            _atomic_write_text(self._target_module_path, top_src)
            l3_res = self._run_l3_contract_fn(self._target_module_path)
            if not l3_res.passed:
                if self._write_and_verify_fn is not None:
                    self._write_and_verify_fn(self._target_module_path, incumbent_source)
                else:
                    _atomic_write_text(self._target_module_path, incumbent_source)
                return incumbent_source, best_result
        if self._refresh_guardrail_refs_fn is not None:
            self._refresh_guardrail_refs_fn(top_cid)
        if self._on_warmstart_promoted_fn is not None:
            self._on_warmstart_promoted_fn(top_src, top_res)
        return top_src, top_res

    def build_snapshot(
        self,
        incumbent_source: str,
        active_anchors: Sequence[AnchorCaseSpec],
        regime_specs: Sequence[RegimeTargetSpec],
    ) -> CalibrationSnapshot:
        """Evaluate baseline, calibrate dispatch floor, collect msprof, and build snapshot."""
        baseline_result, null_launch_us = self.evaluate_baseline(
            incumbent_source,
            active_anchors,
        )
        calibrated_floor = self.calibrate_dispatch_floor(
            baseline_result,
            null_launch_us=null_launch_us,
        )
        msprof_by_case = self.collect_msprof_by_case(
            incumbent_source,
            baseline_result,
            active_anchors,
            regime_specs,
        )
        anchor_t_hw_by_case: dict[int, float] = {}
        for spec in active_anchors:
            t_hw = _extract_valid_non_negative_float(spec.t_hw_us)
            if t_hw is not None:
                anchor_t_hw_by_case[int(spec.case_num)] = t_hw
        for anchor in baseline_result.anchors:
            t_hw = _extract_valid_non_negative_float(anchor.t_hw_us)
            if t_hw is not None:
                anchor_t_hw_by_case[int(anchor.case_num)] = t_hw
        return CalibrationSnapshot(
            baseline_result=baseline_result,
            null_launch_us=null_launch_us,
            calibrated_dispatch_floor_us=calibrated_floor,
            msprof_by_case=msprof_by_case,
            anchor_t_hw_by_case=anchor_t_hw_by_case,
        )


__all__ = [
    "CalibrationService",
    "CalibrationSnapshot",
]
