"""Advisor abstraction isolating CostModel from the campaign orchestrator."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    CostGateDecision,
    CostModelMode,
    evaluate_cost_gate_policy,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

if TYPE_CHECKING:
    from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import HybridAscendCostModel


def _read_field(obj: object | None, key: str, default: object = None) -> object:
    return default if obj is None else (obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default))


def _coerce_axes(raw: object | None) -> TopologyAxes | None:
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping) and all(k in raw for k in ("compute_engine", "memory_ring", "iteration_regularity")):
        return TopologyAxes.from_dict(raw)
    return None


def _has_measured_pmu(residual: object | None) -> bool:
    return residual is not None and any(
        float(_read_field(residual, k, 0.0) or 0.0) > 0.0
        for k in ("measured_mte2_wait_ratio", "measured_aiv_vec_ratio", "measured_aic_mac_ratio", "measured_sync_stall_ratio")
    )


def _resolve_physical_dominant_bound(ledger: object | None, residual: object | None) -> str:
    """Derive physical bottleneck label strictly from measured PMU ratios and ledger cycles."""
    mte2 = float(_read_field(residual, "measured_mte2_wait_ratio", 0.0) or 0.0)
    comp = max(float(_read_field(residual, "measured_aiv_vec_ratio", 0.0) or 0.0), float(_read_field(residual, "measured_aic_mac_ratio", 0.0) or 0.0))
    sync = float(_read_field(residual, "measured_sync_stall_ratio", 0.0) or 0.0)
    if max(mte2, comp, sync) > 0.0:
        if sync >= max(mte2, comp):
            return "sync_bound"
        if mte2 > 0.0 and comp > 0.0 and abs(mte2 - comp) <= 0.05 * max(mte2, comp):
            return "balanced"
        return "memory_bound" if mte2 >= comp else "compute_bound"
    if int(_read_field(residual, "spill_bytes", 0) or 0) > 0:
        return "compiler_spill"
    pipes = raw_p if isinstance(raw_p := _read_field(ledger, "pipe_cycles", None), Mapping) else {}
    mte_c = sum(int(v or 0) for k, v in pipes.items() if int(v or 0) > 0 and any(t in str(k).upper() for t in ("MTE2", "MTE3", "DMA")))
    comp_c = sum(int(v or 0) for k, v in pipes.items() if int(v or 0) > 0 and any(t in str(k).upper() for t in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")))
    if mte_c > 0 or comp_c > 0:
        if mte_c > 0 and comp_c > 0 and abs(mte_c - comp_c) <= 0.10 * max(mte_c, comp_c):
            return "balanced"
        return "memory_bound" if mte_c > comp_c else "compute_bound"
    dom = str(_read_field(ledger, "dominant_pipe", "") or "").upper()
    if any(t in dom for t in ("MTE2", "MTE3", "DMA")):
        return "memory_bound"
    return "compute_bound" if any(t in dom for t in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")) else "balanced"


def _pred_or_t01_us(pred: object) -> float:
    val = float(getattr(pred, "pred_us", 0.0) or 0.0)
    if val <= 0.0:
        val = float(getattr(pred, "tier0_physics_us", 0.0) or 0.0) + float(getattr(pred, "tier1_penalty_us", 0.0) or 0.0)
    return val if val > 0.0 else 0.0


@runtime_checkable
class CostModelAdvisor(Protocol):
    """Unified advisor interface decoupling CostModel from campaign orchestration."""

    @property
    def mode(self) -> CostModelMode: ...
    @property
    def target(self) -> AscendTarget: ...
    @property
    def predictions_invoked(self) -> int: ...
    @property
    def underlying_cost_model(self) -> object | None: ...
    def sync_experience_lake(self, lake: object | None, *, persist_calibration: bool = False, exclude_run_id: str | None = None) -> dict[str, int]: ...
    def evaluate_round_policy(self, *, enable_prescreen_flag: bool = True, bucket_clean_samples: int = 0, out_of_sample_spearman_rho: float | None = None) -> CostGateDecision: ...
    def predict_candidate_us(self, workload: Mapping[str, object], *, bisheng_res: object | None = None) -> object | None: ...
    def predict_anchor_mean_us(self, ledger: object, *, anchors: Sequence[object], live_plan: object | None = None, axes: object | None = None, blueprint: object | None = None, sil_advisory: object | None = None, bisheng_res: object | None = None) -> float: ...
    def build_prompt_guidance(self, *, decision: CostGateDecision, base_cfg: Mapping[str, object], dims: Mapping[str, int] | None, regime_tag: str = "", top_k: int = 5) -> str: ...
    def filter_stage_a_passers(self, candidates: Sequence[object], *, decision: CostGateDecision, parent_pred_us: float, parent_axes: object | None, parent_workspace_stages: int, candidate_pred_us_by_id: Mapping[str, float], candidate_axes_by_id: Mapping[str, object], candidate_workspace_stages_by_id: Mapping[str, int], creative_ids: Collection[str]) -> tuple[list[object], list[tuple[object, str]]]: ...
    def observe_surrogate_batch(self, *, observations: Sequence[Mapping[str, object]], msprof_records: Sequence[Mapping[str, object]] = (), pair_deltas: Sequence[Mapping[str, object]] = (), refit: bool = True) -> None: ...
    def resolve_dominant_bound(self, ledger: object | None, residual: object | None, *, fallback_cost_pred: object | None = None) -> str: ...


class _BaseCostAdvisor:
    _mode: CostModelMode
    _target: AscendTarget
    _predictions_invoked: int
    _cost_model: HybridAscendCostModel | None = None

    @property
    def mode(self) -> CostModelMode:
        return self._mode

    @property
    def target(self) -> AscendTarget:
        return self._target

    @property
    def predictions_invoked(self) -> int:
        return self._predictions_invoked

    @property
    def underlying_cost_model(self) -> object | None:
        return self._cost_model


class NullCostModelAdvisor(_BaseCostAdvisor):
    """Zero-leakage no-op advisor used when cost_model_mode is off."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self._mode, self._target, self._predictions_invoked = CostModelMode.OFF, target or AscendTarget("ascend.950pr"), 0

    def sync_experience_lake(self, lake: object | None, **_kw: object) -> dict[str, int]:
        _ = lake
        return {}

    def evaluate_round_policy(self, *, enable_prescreen_flag: bool = True, **_kw: object) -> CostGateDecision:
        return evaluate_cost_gate_policy(mode=CostModelMode.OFF, enable_prescreen_flag=enable_prescreen_flag)

    def predict_candidate_us(self, workload: Mapping[str, object], **_kw: object) -> object | None:
        _ = workload
        return None

    def predict_anchor_mean_us(self, ledger: object, **_kw: object) -> float:
        _ = ledger
        return 0.0

    def build_prompt_guidance(self, **_kw: object) -> str:
        return ""

    def filter_stage_a_passers(self, candidates: Sequence[object], **_kw: object) -> tuple[list[object], list[tuple[object, str]]]:
        return list(candidates), []

    def observe_surrogate_batch(self, **_kw: object) -> None:
        return None

    def resolve_dominant_bound(self, ledger: object | None, residual: object | None, *, fallback_cost_pred: object | None = None) -> str:
        _ = fallback_cost_pred
        return _resolve_physical_dominant_bound(ledger, residual)


class PluginCostModelAdvisor(_BaseCostAdvisor):
    """Plugin advisor wrapping HybridAscendCostModel for shadow and advisory modes."""

    def __init__(
        self, mode: CostModelMode | str = CostModelMode.ADVISORY, *, target: AscendTarget | None = None,
        operator_name: str = "", calibration_path: Path | None = None, experience_lake: object | None = None,
    ) -> None:
        resolved = mode if isinstance(mode, CostModelMode) else CostModelMode(str(mode).strip().lower())
        if resolved == CostModelMode.OFF:
            raise ValueError("PluginCostModelAdvisor does not support mode='off'; use NullCostModelAdvisor")
        from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (  # noqa: PLC0415
            HybridAscendCostModel,
        )

        self._mode = resolved
        self._target = target or AscendTarget("ascend.950pr")
        self._operator_name = str(operator_name or "").strip()
        self._predictions_invoked = 0
        self._cost_model: HybridAscendCostModel = HybridAscendCostModel(
            self._target, calibration_path=calibration_path, anomaly_lake=experience_lake,
        )

    def sync_experience_lake(self, lake: object | None, *, persist_calibration: bool = False, exclude_run_id: str | None = None) -> dict[str, int]:
        """Replay ExperienceLake records filtered to this advisor's operator."""
        assert self._cost_model is not None
        if lake is None:
            return {}
        return self._cost_model.replay_experience_lake(
            lake, persist_calibration=persist_calibration, exclude_run_id=exclude_run_id,
            operator_filter=self._operator_name or None,
        )

    def evaluate_round_policy(self, *, enable_prescreen_flag: bool = True, bucket_clean_samples: int = 0, out_of_sample_spearman_rho: float | None = None) -> CostGateDecision:
        """Evaluate round policy via evaluate_cost_gate_policy."""
        return evaluate_cost_gate_policy(
            mode=self._mode, enable_prescreen_flag=enable_prescreen_flag,
            bucket_clean_samples=bucket_clean_samples, out_of_sample_spearman_rho=out_of_sample_spearman_rho,
        )

    def predict_candidate_us(self, workload: Mapping[str, object], *, bisheng_res: object | None = None) -> object | None:
        """Run HybridAscendCostModel.predict and increment predictions_invoked."""
        assert self._cost_model is not None
        if self._mode == CostModelMode.OFF:
            return None
        pred = self._cost_model.predict(workload, bisheng_res=bisheng_res)
        self._predictions_invoked += 1
        return pred

    def predict_anchor_mean_us(
        self, ledger: object, *, anchors: Sequence[object], live_plan: object | None = None,
        axes: object | None = None, blueprint: object | None = None, sil_advisory: object | None = None, bisheng_res: object | None = None,
    ) -> float:
        """Project and average predicted latency across anchors."""
        assert self._cost_model is not None
        if self._mode == CostModelMode.OFF:
            return 0.0
        from tilefoundry_ascend.ascend.cost_model.experience_lake import (  # noqa: PLC0415
            project_case_workload,
        )
        from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (  # noqa: PLC0415
            build_cost_features_from_ledger,
        )

        base_cf = build_cost_features_from_ledger(ledger, target=self._target, live_plan=live_plan, axes=_coerce_axes(axes), blueprint=blueprint, sil_advisory=sil_advisory)
        base_dict = asdict(base_cf)
        pol = {}
        if isinstance(live_plan, Mapping):
            pol = dict(rp) if isinstance(rp := live_plan.get("policy"), Mapping) else dict(live_plan)
        elif live_plan is not None and isinstance(getattr(live_plan, "policy", None), Mapping):
            pol = dict(getattr(live_plan, "policy"))

        if not anchors:
            self._predictions_invoked += 1
            return _pred_or_t01_us(self._cost_model.predict(base_cf, bisheng_res=bisheng_res))

        preds = []
        for anchor in anchors:
            adims = _read_field(anchor, "case_dims", None) or _read_field(anchor, "dims", None)
            t_hw = float(_read_field(anchor, "t_hw_us", 0.0) or 0.0)
            a_axes = _coerce_axes(_read_field(anchor, "topology_axes", None)) or base_cf.topology_axes
            wl: object = ({**base_dict, **project_case_workload(ledger, case_dims=adims if isinstance(adims, Mapping) else None, policy=pol, t_hw_us=t_hw, axes=a_axes, target=self._target)} if ((isinstance(adims, Mapping) and adims) or t_hw > 0.0) else base_cf)
            self._predictions_invoked += 1
            if (val := _pred_or_t01_us(self._cost_model.predict(wl, bisheng_res=bisheng_res))) > 0.0:
                preds.append(val)
        return (sum(preds) / float(len(preds))) if preds else 0.0

    def build_prompt_guidance(
        self, *, decision: CostGateDecision, base_cfg: Mapping[str, object], dims: Mapping[str, int] | None, regime_tag: str = "", top_k: int = 5,
    ) -> str:
        """Build tile-size prescreen prompt guidance when decision allows it."""
        _ = dims
        assert self._cost_model is not None
        if self._mode != CostModelMode.ADVISORY or not decision.engaged or not decision.allow_prompt_guidance or not base_cfg:
            return ""
        from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (  # noqa: PLC0415
            TileConfigCandidate,
            _generate_lattice_override_grid,
            format_prescreen_summary,
            generate_aligned_neighborhood,
            prescreen_tile_configs,
        )

        lowered = {str(k).lower(): v for k, v in base_cfg.items()}

        def _pick(keys: tuple[str, ...]) -> int | None:
            return next((int(v) for k in keys if isinstance(v := lowered.get(k), int) and not isinstance(v, bool) and v > 0), None)

        bm, bn, bk = _pick(("bm", "tile_m", "block_m")), _pick(("bn", "tile_n", "block_n")), _pick(("bk", "tile_k", "block_k"))
        if bm is not None and bn is not None and bk is not None:
            stages = _pick(("stages", "num_stages")) or 2
            dt = str(lowered.get("tl_dtype") or lowered.get("dtype") or "f16").lower()
            pool = generate_aligned_neighborhood(TileConfigCandidate(bm=bm, bn=bn, bk=bk, stages=stages, dtype_bytes=(4 if ("32" in dt and "bf" not in dt) else 2), dtype=dt))
            feasible = prescreen_tile_configs(pool, target=self._target, top_k=max(1, int(top_k)), dtype=dt)
            if not feasible:
                return ""
            summary = format_prescreen_summary(feasible)
            return f"[regime={regime_tag}]\n{summary}" if regime_tag else summary
        grid = _generate_lattice_override_grid(base_cfg, self._cost_model.facts, max_points=max(1, int(top_k)))
        if not grid:
            return ""
        header = f"Tile Prescreen Summary ({regime_tag}):" if regime_tag else "Tile Prescreen Summary:"
        return "\n".join([header] + [f"  {i}. {', '.join(f'{k}={v}' for k, v in ov.items())} | OK" for i, ov in enumerate(grid[: max(1, int(top_k))], 1)])

    def filter_stage_a_passers(self, candidates: Sequence[object], **_kw: object) -> tuple[list[object], list[tuple[object, str]]]:
        return list(candidates), []

    def observe_surrogate_batch(
        self, *, observations: Sequence[Mapping[str, object]], msprof_records: Sequence[Mapping[str, object]] = (),
        pair_deltas: Sequence[Mapping[str, object]] = (), refit: bool = True,
    ) -> None:
        """Record observations, MSProf PMU, and pairwise deltas into HybridAscendCostModel."""
        assert self._cost_model is not None
        n_obs, n_ms, n_pr = 0, 0, 0
        for item in observations:
            if (mus := float(item.get("measured_us", 0.0) or 0.0)) <= 0.0:
                continue
            self._cost_model.observe(item.get("workload", item), mus, bisheng_res=item.get("bisheng_res"), persist=False, refit=False)
            if (csha := str(item.get("cand_sha") or "")) and "case_num" in item:
                self._cost_model.mark_online_observation_key(cand_sha=csha, operator=str(item.get("operator") or self._operator_name or ""), regime_tag=str(item.get("regime_tag") or ""), case_num=int(item.get("case_num") or 0), eval_tier=str(item.get("eval_tier") or "surrogate"), measured_us=mus)
            n_obs += 1
        for mitem in msprof_records:
            pmu = mitem.get("msprof_pmu", mitem.get("pmu", mitem))
            if isinstance(pmu, Mapping) and pmu:
                dur = mitem.get("measured_us")
                self._cost_model.observe_msprof(mitem.get("workload", {}), pmu, measured_us=float(dur) if dur is not None else None, bisheng_res=mitem.get("bisheng_res"), persist=False)
                n_ms += 1
        for pitem in pair_deltas:
            cwl, pwl = pitem.get("cand_workload"), pitem.get("parent_workload")
            cus, pus = float(pitem.get("cand_us", 0.0) or 0.0), float(pitem.get("parent_us", 0.0) or 0.0)
            if cwl is None or pwl is None or cus <= 0.0 or pus <= 0.0:
                continue
            self._cost_model.observe_pair_delta(cwl, cus, pwl, pus, cand_bisheng_res=pitem.get("cand_bisheng_res"), parent_bisheng_res=pitem.get("parent_bisheng_res"), refit=False)
            csha, psha = str(pitem.get("cand_sha") or ""), str(pitem.get("parent_sha") or "")
            if csha and psha and "case_num" in pitem:
                self._cost_model.mark_online_pair_key(cand_sha=csha, parent_sha=psha, operator=str(pitem.get("operator") or self._operator_name or ""), regime_tag=str(pitem.get("regime_tag") or ""), case_num=int(pitem.get("case_num") or 0))
            n_pr += 1
        if refit:
            if n_obs > 0:
                self._cost_model._refit_ridge()
            if n_pr > 0:
                self._cost_model._refit_delta_ridge()
        if self._cost_model.calibration_path is not None and (n_obs > 0 or n_ms > 0 or n_pr > 0):
            self._cost_model.save_calibration(self._cost_model.calibration_path)

    def resolve_dominant_bound(self, ledger: object | None, residual: object | None, *, fallback_cost_pred: object | None = None) -> str:
        if self._mode == CostModelMode.ADVISORY and not _has_measured_pmu(residual):
            if (pb := str(_read_field(fallback_cost_pred, "dominant_bound", "") or "").strip()):
                return pb
        return _resolve_physical_dominant_bound(ledger, residual)


def create_cost_advisor(
    mode: CostModelMode | str = CostModelMode.OFF, *, target: AscendTarget | None = None,
    operator_name: str = "", calibration_path: Path | None = None, experience_lake: object | None = None,
) -> CostModelAdvisor:
    """Instantiate NullCostModelAdvisor for mode=off or PluginCostModelAdvisor otherwise."""
    resolved = mode if isinstance(mode, CostModelMode) else CostModelMode(str(mode).strip().lower())
    eff_target = target or AscendTarget("ascend.950pr")
    return NullCostModelAdvisor(target=eff_target) if resolved == CostModelMode.OFF else PluginCostModelAdvisor(
        mode=resolved, target=eff_target, operator_name=operator_name, calibration_path=calibration_path, experience_lake=experience_lake,
    )


__all__ = ["CostModelAdvisor", "NullCostModelAdvisor", "PluginCostModelAdvisor", "create_cost_advisor"]
