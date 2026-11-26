"""Advisor abstraction isolating CostModel from the campaign orchestrator."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    CostGateDecision,
    CostModelMode,
    evaluate_cost_gate_policy,
    is_paradigm_shift_exempt,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def _read_field(obj: object | None, key: str, default: object = None) -> object:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_axes(raw: object | None) -> TopologyAxes | None:
    if raw is None:
        return None
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping) and all(
        k in raw for k in ("compute_engine", "memory_ring", "iteration_regularity")
    ):
        return TopologyAxes.from_dict(raw)
    return None


def _extract_candidate_id(cand: object) -> str:
    if isinstance(cand, str):
        return cand
    if isinstance(cand, Mapping):
        return str(cand.get("candidate_id") or cand.get("id") or "")
    if isinstance(cand, (tuple, list)) and cand:
        return str(cand[0])
    cid = getattr(cand, "candidate_id", None)
    if cid is not None:
        return str(cid)
    return str(cand)


def _has_measured_pmu(residual: object | None) -> bool:
    if residual is None:
        return False
    for key in (
        "measured_mte2_wait_ratio",
        "measured_aiv_vec_ratio",
        "measured_aic_mac_ratio",
        "measured_sync_stall_ratio",
    ):
        if float(_read_field(residual, key, 0.0) or 0.0) > 0.0:
            return True
    return False


def _resolve_physical_dominant_bound(
    ledger: object | None,
    residual: object | None,
) -> str:
    """Derive physical bottleneck label strictly from measured PMU ratios and ledger cycles."""
    mte2 = float(_read_field(residual, "measured_mte2_wait_ratio", 0.0) or 0.0)
    aiv_vec = float(_read_field(residual, "measured_aiv_vec_ratio", 0.0) or 0.0)
    aic_mac = float(_read_field(residual, "measured_aic_mac_ratio", 0.0) or 0.0)
    sync_stall = float(_read_field(residual, "measured_sync_stall_ratio", 0.0) or 0.0)
    compute_pmu = max(aiv_vec, aic_mac)
    if max(mte2, compute_pmu, sync_stall) > 0.0:
        if sync_stall >= max(mte2, compute_pmu):
            return "sync_bound"
        if mte2 > 0.0 and compute_pmu > 0.0 and abs(mte2 - compute_pmu) <= 0.05 * max(
            mte2, compute_pmu
        ):
            return "balanced"
        if mte2 >= compute_pmu:
            return "memory_bound"
        return "compute_bound"

    spill_bytes = int(_read_field(residual, "spill_bytes", 0) or 0)
    if spill_bytes > 0:
        return "compiler_spill"

    raw_pipes = _read_field(ledger, "pipe_cycles", None)
    pipes = raw_pipes if isinstance(raw_pipes, Mapping) else {}
    mte_cycles = 0
    compute_cycles = 0
    for pipe_name, raw_val in pipes.items():
        val = int(raw_val or 0)
        if val <= 0:
            continue
        upper = str(pipe_name).upper()
        if "MTE2" in upper or "MTE3" in upper or "DMA" in upper:
            mte_cycles += val
        elif any(tok in upper for tok in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")):
            compute_cycles += val

    if mte_cycles > 0 or compute_cycles > 0:
        if mte_cycles > 0 and compute_cycles > 0 and abs(mte_cycles - compute_cycles) <= 0.10 * max(
            mte_cycles, compute_cycles
        ):
            return "balanced"
        if mte_cycles > compute_cycles:
            return "memory_bound"
        return "compute_bound"

    dom_pipe = str(_read_field(ledger, "dominant_pipe", "") or "").upper()
    if any(tok in dom_pipe for tok in ("MTE2", "MTE3", "DMA")):
        return "memory_bound"
    if any(tok in dom_pipe for tok in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")):
        return "compute_bound"
    return "balanced"


@runtime_checkable
class CostModelAdvisor(Protocol):
    """Unified advisor interface decoupling CostModel from campaign orchestration."""

    @property
    def mode(self) -> CostModelMode:
        """Return configured cost advisor mode."""
        ...

    @property
    def target(self) -> AscendTarget:
        """Return target device descriptor."""
        ...

    @property
    def predictions_invoked(self) -> int:
        """Return number of cost-model predictions invoked so far."""
        ...

    @property
    def underlying_cost_model(self) -> object | None:
        """Return backing HybridAscendCostModel instance or None when disabled."""
        ...

    def sync_experience_lake(
        self,
        lake: object | None,
        *,
        persist_calibration: bool = False,
        exclude_run_id: str | None = None,
    ) -> dict[str, int]:
        """Synchronize calibration state from ExperienceLake."""
        ...

    def evaluate_round_policy(
        self,
        *,
        enable_prescreen_flag: bool,
        round_idx: int,
        stagnation_rounds: int,
        incumbent_roofline_eta: float,
        target_roofline_eta: float,
        prev_local_rel_gain: float | None,
        mean_recoverable_excess_us: float | None,
        coarse_reject_ratio: float = 3.0,
        disengage_roofline_eta: float = 0.65,
        slow_gain_threshold: float = 0.03,
    ) -> CostGateDecision:
        """Evaluate round-level engagement decision."""
        ...

    def predict_candidate_us(
        self,
        workload: Mapping[str, object],
        *,
        bisheng_res: object | None = None,
    ) -> object | None:
        """Predict candidate latency or return None when disabled."""
        ...

    def predict_anchor_mean_us(
        self,
        ledger: object,
        *,
        anchors: Sequence[object],
        live_plan: object | None = None,
        axes: object | None = None,
        blueprint: object | None = None,
        sil_advisory: object | None = None,
        bisheng_res: object | None = None,
    ) -> float:
        """Predict mean latency across anchor cases."""
        ...

    def build_prompt_guidance(
        self,
        *,
        decision: CostGateDecision,
        base_cfg: Mapping[str, object],
        dims: Mapping[str, int] | None,
        regime_tag: str = "",
        top_k: int = 5,
    ) -> str:
        """Render optional tile-size prescreen guidance for creative prompts."""
        ...

    def filter_stage_a_passers(
        self,
        candidates: Sequence[object],
        *,
        decision: CostGateDecision,
        parent_pred_us: float,
        parent_axes: object | None,
        parent_workspace_stages: int,
        candidate_pred_us_by_id: Mapping[str, float],
        candidate_axes_by_id: Mapping[str, object],
        candidate_workspace_stages_by_id: Mapping[str, int],
        creative_ids: Collection[str],
    ) -> tuple[list[object], list[tuple[object, str]]]:
        """Partition Stage-A passers into kept and coarse-rejected lists."""
        ...

    def observe_surrogate_batch(
        self,
        *,
        observations: Sequence[Mapping[str, object]],
        msprof_records: Sequence[Mapping[str, object]] = (),
        pair_deltas: Sequence[Mapping[str, object]] = (),
        refit: bool = True,
    ) -> None:
        """Record hardware observations into calibration state."""
        ...

    def resolve_dominant_bound(
        self,
        ledger: object | None,
        residual: object | None,
        *,
        fallback_cost_pred: object | None = None,
    ) -> str:
        """Resolve dominant physical bottleneck label."""
        ...


class NullCostModelAdvisor:
    """Zero-leakage no-op advisor used when cost_model_mode is off."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self._mode = CostModelMode.OFF
        self._target = target or AscendTarget("ascend.950pr")
        self._predictions_invoked = 0

    @property
    def mode(self) -> CostModelMode:
        """Return CostModelMode.OFF."""
        return self._mode

    @property
    def target(self) -> AscendTarget:
        """Return target device descriptor."""
        return self._target

    @property
    def predictions_invoked(self) -> int:
        """Return 0 as predictions are never invoked in off mode."""
        return self._predictions_invoked

    @property
    def underlying_cost_model(self) -> object | None:
        """Return None without importing or instantiating HybridAscendCostModel."""
        return None

    def sync_experience_lake(
        self,
        lake: object | None,
        *,
        persist_calibration: bool = False,
        exclude_run_id: str | None = None,
    ) -> dict[str, int]:
        """Return empty replay counts without touching disk or models."""
        _ = (lake, persist_calibration, exclude_run_id)
        return {}

    def evaluate_round_policy(
        self,
        *,
        enable_prescreen_flag: bool,
        round_idx: int,
        stagnation_rounds: int,
        incumbent_roofline_eta: float,
        target_roofline_eta: float,
        prev_local_rel_gain: float | None,
        mean_recoverable_excess_us: float | None,
        coarse_reject_ratio: float = 3.0,
        disengage_roofline_eta: float = 0.65,
        slow_gain_threshold: float = 0.03,
    ) -> CostGateDecision:
        """Return disabled CostGateDecision for mode=off."""
        _ = (
            enable_prescreen_flag,
            round_idx,
            stagnation_rounds,
            incumbent_roofline_eta,
            target_roofline_eta,
            prev_local_rel_gain,
            mean_recoverable_excess_us,
            coarse_reject_ratio,
            disengage_roofline_eta,
            slow_gain_threshold,
        )
        return CostGateDecision(
            mode=CostModelMode.OFF,
            engaged=False,
            reason="disabled_by_config",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )

    def predict_candidate_us(
        self,
        workload: Mapping[str, object],
        *,
        bisheng_res: object | None = None,
    ) -> object | None:
        """Return None without incrementing predictions_invoked."""
        _ = (workload, bisheng_res)
        return None

    def predict_anchor_mean_us(
        self,
        ledger: object,
        *,
        anchors: Sequence[object],
        live_plan: object | None = None,
        axes: object | None = None,
        blueprint: object | None = None,
        sil_advisory: object | None = None,
        bisheng_res: object | None = None,
    ) -> float:
        """Return 0.0 without incrementing predictions_invoked."""
        _ = (ledger, anchors, live_plan, axes, blueprint, sil_advisory, bisheng_res)
        return 0.0

    def build_prompt_guidance(
        self,
        *,
        decision: CostGateDecision,
        base_cfg: Mapping[str, object],
        dims: Mapping[str, int] | None,
        regime_tag: str = "",
        top_k: int = 5,
    ) -> str:
        """Return empty guidance string."""
        _ = (decision, base_cfg, dims, regime_tag, top_k)
        return ""

    def filter_stage_a_passers(
        self,
        candidates: Sequence[object],
        *,
        decision: CostGateDecision,
        parent_pred_us: float,
        parent_axes: object | None,
        parent_workspace_stages: int,
        candidate_pred_us_by_id: Mapping[str, float],
        candidate_axes_by_id: Mapping[str, object],
        candidate_workspace_stages_by_id: Mapping[str, int],
        creative_ids: Collection[str],
    ) -> tuple[list[object], list[tuple[object, str]]]:
        """Pass all candidates unchanged with zero rejections."""
        _ = (
            decision,
            parent_pred_us,
            parent_axes,
            parent_workspace_stages,
            candidate_pred_us_by_id,
            candidate_axes_by_id,
            candidate_workspace_stages_by_id,
            creative_ids,
        )
        return list(candidates), []

    def observe_surrogate_batch(
        self,
        *,
        observations: Sequence[Mapping[str, object]],
        msprof_records: Sequence[Mapping[str, object]] = (),
        pair_deltas: Sequence[Mapping[str, object]] = (),
        refit: bool = True,
    ) -> None:
        """Perform pure no-op without writing calibration files."""
        _ = (observations, msprof_records, pair_deltas, refit)

    def resolve_dominant_bound(
        self,
        ledger: object | None,
        residual: object | None,
        *,
        fallback_cost_pred: object | None = None,
    ) -> str:
        """Derive physical bound strictly from measured PMU ratios and ledger pipe cycles."""
        _ = fallback_cost_pred
        return _resolve_physical_dominant_bound(ledger, residual)


class PluginCostModelAdvisor:
    """Plugin advisor wrapping HybridAscendCostModel for shadow, auto, and coarse_only."""

    def __init__(
        self,
        mode: CostModelMode | str = CostModelMode.AUTO,
        *,
        target: AscendTarget | None = None,
        operator_name: str = "",
        calibration_path: Path | None = None,
        experience_lake: object | None = None,
    ) -> None:
        resolved_mode = (
            mode
            if isinstance(mode, CostModelMode)
            else CostModelMode(str(mode).strip().lower())
        )
        if resolved_mode == CostModelMode.OFF:
            raise ValueError(
                "PluginCostModelAdvisor does not support mode='off'; use NullCostModelAdvisor"
            )
        from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (  # noqa: PLC0415
            HybridAscendCostModel,
        )

        self._mode = resolved_mode
        self._target = target or AscendTarget("ascend.950pr")
        self._operator_name = str(operator_name or "").strip()
        self._predictions_invoked = 0
        anomaly_lake = (
            experience_lake if isinstance(experience_lake, (str, Path)) else experience_lake
        )
        self._cost_model = HybridAscendCostModel(
            self._target,
            calibration_path=calibration_path,
            anomaly_lake=anomaly_lake,
        )

    @property
    def mode(self) -> CostModelMode:
        """Return configured CostModelMode."""
        return self._mode

    @property
    def target(self) -> AscendTarget:
        """Return target device descriptor."""
        return self._target

    @property
    def predictions_invoked(self) -> int:
        """Return total number of predictions performed."""
        return self._predictions_invoked

    @property
    def underlying_cost_model(self) -> object | None:
        """Return backing HybridAscendCostModel instance."""
        return self._cost_model

    def sync_experience_lake(
        self,
        lake: object | None,
        *,
        persist_calibration: bool = False,
        exclude_run_id: str | None = None,
    ) -> dict[str, int]:
        """Replay ExperienceLake records filtered to this advisor's operator."""
        if lake is None:
            return {}
        return self._cost_model.replay_experience_lake(
            lake,
            persist_calibration=persist_calibration,
            exclude_run_id=exclude_run_id,
            operator_filter=self._operator_name or None,
        )

    def evaluate_round_policy(
        self,
        *,
        enable_prescreen_flag: bool,
        round_idx: int,
        stagnation_rounds: int,
        incumbent_roofline_eta: float,
        target_roofline_eta: float,
        prev_local_rel_gain: float | None,
        mean_recoverable_excess_us: float | None,
        coarse_reject_ratio: float = 3.0,
        disengage_roofline_eta: float = 0.65,
        slow_gain_threshold: float = 0.03,
    ) -> CostGateDecision:
        """Evaluate round policy via evaluate_cost_gate_policy."""
        return evaluate_cost_gate_policy(
            mode=self._mode,
            enable_prescreen_flag=enable_prescreen_flag,
            round_idx=round_idx,
            stagnation_rounds=stagnation_rounds,
            incumbent_roofline_eta=incumbent_roofline_eta,
            target_roofline_eta=target_roofline_eta,
            prev_local_rel_gain=prev_local_rel_gain,
            mean_recoverable_excess_us=mean_recoverable_excess_us,
            disengage_roofline_eta=disengage_roofline_eta,
            slow_gain_threshold=slow_gain_threshold,
            coarse_reject_ratio=coarse_reject_ratio,
        )

    def predict_candidate_us(
        self,
        workload: Mapping[str, object],
        *,
        bisheng_res: object | None = None,
    ) -> object | None:
        """Run HybridAscendCostModel.predict and increment predictions_invoked."""
        pred = self._cost_model.predict(workload, bisheng_res=bisheng_res)
        self._predictions_invoked += 1
        return pred

    def predict_anchor_mean_us(
        self,
        ledger: object,
        *,
        anchors: Sequence[object],
        live_plan: object | None = None,
        axes: object | None = None,
        blueprint: object | None = None,
        sil_advisory: object | None = None,
        bisheng_res: object | None = None,
    ) -> float:
        """Project and average predicted latency across anchors."""
        from tilefoundry_ascend.ascend.cost_model.experience_lake import (  # noqa: PLC0415
            project_case_workload,
        )
        from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (  # noqa: PLC0415
            build_cost_features_from_ledger,
        )

        eff_axes = _coerce_axes(axes)
        base_cf = build_cost_features_from_ledger(
            ledger,
            target=self._target,
            live_plan=live_plan,
            axes=eff_axes,
            blueprint=blueprint,
            sil_advisory=sil_advisory,
        )
        base_dict = asdict(base_cf)
        policy_dict: dict[str, object] = {}
        if isinstance(live_plan, Mapping):
            raw_pol = live_plan.get("policy")
            policy_dict = dict(raw_pol) if isinstance(raw_pol, Mapping) else dict(live_plan)
        elif live_plan is not None:
            raw_pol = getattr(live_plan, "policy", None)
            if isinstance(raw_pol, Mapping):
                policy_dict = dict(raw_pol)

        if not anchors:
            pred = self._cost_model.predict(base_cf, bisheng_res=bisheng_res)
            self._predictions_invoked += 1
            val = float(pred.pred_us)
            if val <= 0.0:
                val = float(pred.tier0_physics_us) + float(pred.tier1_penalty_us)
            return val if val > 0.0 else 0.0

        preds: list[float] = []
        for anchor in anchors:
            anchor_dims = _read_field(anchor, "case_dims", None) or _read_field(
                anchor, "dims", None
            )
            t_hw_us = float(_read_field(anchor, "t_hw_us", 0.0) or 0.0)
            anchor_axes = _coerce_axes(_read_field(anchor, "topology_axes", None)) or base_cf.topology_axes
            if (isinstance(anchor_dims, Mapping) and anchor_dims) or t_hw_us > 0.0:
                proj = project_case_workload(
                    ledger,
                    case_dims=anchor_dims if isinstance(anchor_dims, Mapping) else None,
                    policy=policy_dict,
                    t_hw_us=t_hw_us,
                    axes=anchor_axes,
                    target=self._target,
                )
                merged = dict(base_dict)
                merged.update(proj)
                wl: object = merged
            else:
                wl = base_cf
            pred = self._cost_model.predict(wl, bisheng_res=bisheng_res)
            self._predictions_invoked += 1
            val = float(pred.pred_us)
            if val <= 0.0:
                val = float(pred.tier0_physics_us) + float(pred.tier1_penalty_us)
            if val > 0.0:
                preds.append(val)
        return (sum(preds) / float(len(preds))) if preds else 0.0

    def build_prompt_guidance(
        self,
        *,
        decision: CostGateDecision,
        base_cfg: Mapping[str, object],
        dims: Mapping[str, int] | None,
        regime_tag: str = "",
        top_k: int = 5,
    ) -> str:
        """Build tile-size prescreen prompt guidance when decision allows it."""
        _ = dims
        if self._mode == CostModelMode.SHADOW:
            return ""
        if not decision.engaged or not decision.allow_prompt_guidance or not base_cfg:
            return ""

        from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (  # noqa: PLC0415
            TileConfigCandidate,
            _generate_lattice_override_grid,
            format_prescreen_summary,
            generate_aligned_neighborhood,
            prescreen_tile_configs,
        )

        lowered = {str(k).lower(): v for k, v in base_cfg.items()}

        def _pick_int(keys: tuple[str, ...]) -> int | None:
            for key in keys:
                val = lowered.get(key)
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    return int(val)
            return None

        bm = _pick_int(("bm", "tile_m", "block_m"))
        bn = _pick_int(("bn", "tile_n", "block_n"))
        bk = _pick_int(("bk", "tile_k", "block_k"))
        if bm is not None and bn is not None and bk is not None:
            stages = _pick_int(("stages", "num_stages")) or 2
            dtype_str = str(lowered.get("tl_dtype") or lowered.get("dtype") or "f16").lower()
            dtype_bytes = 4 if ("32" in dtype_str and "bf" not in dtype_str) else 2
            center = TileConfigCandidate(
                bm=bm,
                bn=bn,
                bk=bk,
                stages=stages,
                dtype_bytes=dtype_bytes,
                dtype=dtype_str,
            )
            pool = generate_aligned_neighborhood(center)
            feasible = prescreen_tile_configs(
                pool,
                target=self._target,
                top_k=max(1, int(top_k)),
                dtype=dtype_str,
            )
            if not feasible:
                return ""
            summary = format_prescreen_summary(feasible)
            return f"[regime={regime_tag}]\n{summary}" if regime_tag else summary

        grid = _generate_lattice_override_grid(
            base_cfg,
            self._cost_model.facts,
            max_points=max(1, int(top_k)),
        )
        if not grid:
            return ""
        header = f"Tile Prescreen Summary ({regime_tag}):" if regime_tag else "Tile Prescreen Summary:"
        lines = [header]
        for idx, ov in enumerate(grid[: max(1, int(top_k))], start=1):
            parts = ", ".join(f"{k}={v}" for k, v in ov.items())
            lines.append(f"  {idx}. {parts} | OK")
        return "\n".join(lines)

    def filter_stage_a_passers(
        self,
        candidates: Sequence[object],
        *,
        decision: CostGateDecision,
        parent_pred_us: float,
        parent_axes: object | None,
        parent_workspace_stages: int,
        candidate_pred_us_by_id: Mapping[str, float],
        candidate_axes_by_id: Mapping[str, object],
        candidate_workspace_stages_by_id: Mapping[str, int],
        creative_ids: Collection[str],
    ) -> tuple[list[object], list[tuple[object, str]]]:
        """Coarse-filter Stage-A passers with paradigm-shift exemption."""
        if (
            self._mode == CostModelMode.SHADOW
            or not decision.engaged
            or not decision.allow_candidate_pruning
        ):
            return list(candidates), []

        eff_parent_axes = _coerce_axes(parent_axes)
        creative_set = frozenset(str(cid) for cid in creative_ids)
        kept: list[object] = []
        rejected: list[tuple[object, str]] = []
        cap = float(decision.coarse_reject_ratio)

        for cand in candidates:
            cid = _extract_candidate_id(cand)
            cand_axes = _coerce_axes(candidate_axes_by_id.get(cid))
            cand_ws_stages = int(candidate_workspace_stages_by_id.get(cid, parent_workspace_stages))
            exempt = is_paradigm_shift_exempt(
                cand_axes,
                eff_parent_axes,
                cand_workspace_stages=cand_ws_stages,
                parent_workspace_stages=int(parent_workspace_stages),
                is_creative=(cid in creative_set),
            )
            cand_us = candidate_pred_us_by_id.get(cid)
            if (
                not exempt
                and parent_pred_us > 0.0
                and cand_us is not None
                and float(cand_us) > 0.0
                and math.isfinite(cap)
                and cap > 0.0
            ):
                ratio = float(cand_us) / float(parent_pred_us)
                if ratio > cap:
                    rejected.append((cand, f"coarse_reject_ratio:{ratio:.2f}>{cap:.2f}"))
                    continue
            kept.append(cand)
        return kept, rejected

    def observe_surrogate_batch(
        self,
        *,
        observations: Sequence[Mapping[str, object]],
        msprof_records: Sequence[Mapping[str, object]] = (),
        pair_deltas: Sequence[Mapping[str, object]] = (),
        refit: bool = True,
    ) -> None:
        """Record observations, MSProf PMU, and pairwise deltas into HybridAscendCostModel."""
        obs_count = 0
        for item in observations:
            workload = item.get("workload", item)
            measured_us = float(item.get("measured_us", 0.0) or 0.0)
            if measured_us <= 0.0:
                continue
            self._cost_model.observe(
                workload,
                measured_us,
                bisheng_res=item.get("bisheng_res"),
                persist=False,
                refit=False,
            )
            obs_count += 1

        msprof_count = 0
        for mitem in msprof_records:
            workload = mitem.get("workload", {})
            raw_pmu = mitem.get("msprof_pmu", mitem.get("pmu", mitem))
            if not isinstance(raw_pmu, Mapping) or not raw_pmu:
                continue
            raw_dur = mitem.get("measured_us")
            self._cost_model.observe_msprof(
                workload,
                raw_pmu,
                measured_us=float(raw_dur) if raw_dur is not None else None,
                bisheng_res=mitem.get("bisheng_res"),
                persist=False,
            )
            msprof_count += 1

        pair_count = 0
        for pitem in pair_deltas:
            cand_wl = pitem.get("cand_workload")
            parent_wl = pitem.get("parent_workload")
            cand_us = float(pitem.get("cand_us", 0.0) or 0.0)
            parent_us = float(pitem.get("parent_us", 0.0) or 0.0)
            if cand_wl is None or parent_wl is None or cand_us <= 0.0 or parent_us <= 0.0:
                continue
            self._cost_model.observe_pair_delta(
                cand_wl,
                cand_us,
                parent_wl,
                parent_us,
                cand_bisheng_res=pitem.get("cand_bisheng_res"),
                parent_bisheng_res=pitem.get("parent_bisheng_res"),
                refit=False,
            )
            pair_count += 1

        if refit:
            if obs_count > 0:
                self._cost_model._refit_ridge()
            if pair_count > 0:
                self._cost_model._refit_delta_ridge()

        if self._cost_model.calibration_path is not None and (
            obs_count > 0 or msprof_count > 0 or pair_count > 0
        ):
            self._cost_model.save_calibration(self._cost_model.calibration_path)

    def resolve_dominant_bound(
        self,
        ledger: object | None,
        residual: object | None,
        *,
        fallback_cost_pred: object | None = None,
    ) -> str:
        """Resolve dominant bound via PMU+ledger (shadow) or fallback cost prediction."""
        if self._mode != CostModelMode.SHADOW and not _has_measured_pmu(residual):
            pred_bound = str(_read_field(fallback_cost_pred, "dominant_bound", "") or "").strip()
            if pred_bound:
                return pred_bound
        return _resolve_physical_dominant_bound(ledger, residual)


def create_cost_advisor(
    mode: CostModelMode | str = CostModelMode.OFF,
    *,
    target: AscendTarget | None = None,
    operator_name: str = "",
    calibration_path: Path | None = None,
    experience_lake: object | None = None,
) -> CostModelAdvisor:
    """Instantiate NullCostModelAdvisor for mode=off or PluginCostModelAdvisor otherwise."""
    resolved_mode = (
        mode if isinstance(mode, CostModelMode) else CostModelMode(str(mode).strip().lower())
    )
    eff_target = target or AscendTarget("ascend.950pr")
    if resolved_mode == CostModelMode.OFF:
        return NullCostModelAdvisor(target=eff_target)
    return PluginCostModelAdvisor(
        mode=resolved_mode,
        target=eff_target,
        operator_name=operator_name,
        calibration_path=calibration_path,
        experience_lake=experience_lake,
    )


__all__ = [
    "CostModelAdvisor",
    "NullCostModelAdvisor",
    "PluginCostModelAdvisor",
    "create_cost_advisor",
]
