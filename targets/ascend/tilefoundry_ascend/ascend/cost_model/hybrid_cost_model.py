"""3-Tier Hybrid Ascend 950 (dav-3510) Cost Model and Online Calibrated Surrogate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary,
    CompilerBehaviorRegistry,
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

CUBE_MACS_PER_CYCLE: dict[str, int] = {
    "f4e2m1": 16384,
    "f4e1m2": 16384,
    "fp8e4m3": 8192,
    "fp8e5m2": 8192,
    "hif8": 8192,
    "i8": 8192,
    "u8": 8192,
    "f16": 4096,
    "float16": 4096,
    "bf16": 4096,
    "bfloat16": 4096,
    "hf32": 2048,
    "f32": 256,
    "float32": 256,
}

DEFAULT_CLOCK_GHZ = 1.65
DEFAULT_L2_CAPACITY_BYTES = 134_217_728  # 128 MiB
DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC = 101
DEFAULT_UB_BANK_MODULUS_BYTES = 512  # 16 banks * 32B datablock
DEFAULT_AIV_VEC_FLOPS_PER_CYCLE = 128  # 64 fp32 lanes * 2 FLOPs/cycle (FMA)
DEFAULT_LAUNCH_OVERHEAD_US = 3.2
DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC = 256
DEFAULT_CV_SYNC_EVENT_US = 0.08


def topology_bucket_for_axes(axes: TopologyAxes | None) -> str:
    """Return canonical hierarchical residual bucket key for *axes*."""
    if axes is None:
        return "global"
    return f"{axes.compute_engine}:{axes.memory_ring}"


def _cube_macs_for_dtype(facts: AscendSpaceFacts, dtype_name: str, *, hf32: bool = False) -> int:
    if hasattr(facts, "cube_macs_per_cycle"):
        try:
            return int(facts.cube_macs_per_cycle(dtype_name, hf32=hf32))
        except Exception:
            pass
    key = "hf32" if (hf32 and dtype_name.strip().lower() in ("f32", "float32")) else dtype_name.strip().lower()
    return CUBE_MACS_PER_CYCLE.get(key, 4096)


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve dense linear system A x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        diag = mat[col][col]
        if abs(diag) < 1e-12:
            continue
        inv_diag = 1.0 / diag
        for j in range(col, n + 1):
            mat[col][j] *= inv_diag
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


@dataclass(frozen=True)
class CannbenchScorePrediction:
    """Official CANNBench Eq.3 performance score + Eq.4 / HAP efficiency reward."""

    eq3_score: float
    hap_reward: float
    composite_score: float

    def to_dict(self) -> dict[str, float]:
        """Return dictionary representation."""
        return {
            "eq3_score": self.eq3_score,
            "hap_reward": self.hap_reward,
            "composite_score": self.composite_score,
        }

    def __getitem__(self, key: str) -> float:
        if key in ("eq3_score", "hap_reward", "composite_score"):
            return float(getattr(self, key))
        raise KeyError(key)

    def __float__(self) -> float:
        return self.eq3_score


def predict_cannbench_eq3_score(
    pred_us: float,
    ref_perf_us: float,
    t_hw_us: float,
) -> CannbenchScorePrediction:
    """Compute official CANNBench Eq.3 score + HAP reward ``min(t_hw_us / pred_us, 3.0)``."""
    if pred_us <= 0.0 or t_hw_us <= 0.0:
        return CannbenchScorePrediction(eq3_score=0.0, hap_reward=0.0, composite_score=0.0)
    baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(float(t_hw_us) * 10.0, 10.0)
    denom = (float(pred_us) - float(t_hw_us)) + (baseline - float(t_hw_us))
    if denom <= 0.0:
        eq3 = 0.0
    else:
        eq3 = max(0.0, (baseline - float(t_hw_us)) / denom)
    hap = min(float(t_hw_us) / float(pred_us), 3.0)
    composite = 0.20 + 0.70 * eq3 + 0.10 * hap
    return CannbenchScorePrediction(
        eq3_score=eq3,
        hap_reward=hap,
        composite_score=composite,
    )


@dataclass(frozen=True)
class CostFeatures:
    """Normalized workload + schedule + compiler resource features for cost prediction."""

    cube_flops: float = 0.0
    vec_flops: float = 0.0
    cube_dtype: str = "f16"
    hf32: bool = False
    gm_bytes: float = 0.0
    contig_burst_bytes: int = 32768
    total_tiles: int = 0
    active_aic_cores: int | None = None
    active_aiv_cores: int | None = None
    l2_working_set_bytes: int = 0
    swizzle_window: int = 1
    l2_reuse_fraction: float = 0.0
    l2_bypass: bool = False
    ub_bank_pad_datablocks: int = 0
    ub_rw_shift_datablocks: int = 0
    ub_row_stride_bytes: int = 0
    ub_bank_conflict_ratio: float = 0.0
    l1_stages: int = 1
    l1_half_bank_split: bool = True
    has_unaligned_tail: bool = False
    spill_bytes: int = 0
    stack_bytes: int = 0
    vloopv2_loops: int = 1
    software_loops: int = 0
    has_local_membar_in_vf_call: bool = False
    vf_fusion_breaks: int = 0
    alias_serialized_buffers: int = 0
    dual_issue_ipc_collapse: float = 0.0
    dcache_dcci_flushes: int = 0
    kernel_launches: int = 1
    launch_overhead_us: float | None = None
    fixpipe_bytes: float = 0.0
    cv_sync_events: int = 0
    extra_signals: dict[str, float] = field(default_factory=dict)
    topology_axes: TopologyAxes | None = None

    @classmethod
    def coerce(
        cls,
        workload: CostFeatures | Mapping[str, object] | object,
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> CostFeatures:
        """Build CostFeatures from a dataclass, Mapping, or SIL SchedulePolicy + optional BishengResourceSummary."""
        if isinstance(workload, CostFeatures):
            base = asdict(workload)
        elif isinstance(workload, Mapping):
            base = dict(workload)
        elif hasattr(workload, "to_dict") and callable(getattr(workload, "to_dict")):
            raw_dict = workload.to_dict()
            if not isinstance(raw_dict, Mapping):
                raise TypeError(f"to_dict() must return a Mapping, got {type(raw_dict).__name__}")
            base = dict(raw_dict)
        else:
            raise TypeError(f"expected CostFeatures or Mapping, got {type(workload).__name__}")

        if not base.get("cube_dtype"):
            if base.get("cube_operand_dtype"):
                base["cube_dtype"] = str(base["cube_operand_dtype"])
            elif base.get("dtype"):
                base["cube_dtype"] = str(base["dtype"])
        if "hf32" not in base and "hf32_allowed" in base:
            base["hf32"] = bool(base["hf32_allowed"])
        if "l2_bypass" not in base and "l2_bypass_operands" in base:
            base["l2_bypass"] = bool(base["l2_bypass_operands"])

        if bisheng_res is not None:
            res = (
                bisheng_res
                if isinstance(bisheng_res, BishengResourceSummary)
                else BishengResourceSummary.from_dict(bisheng_res)
            )
            base["spill_bytes"] = res.spill_bytes
            base["stack_bytes"] = res.stack_bytes
            base["vloopv2_loops"] = res.vloopv2_loops
            base["software_loops"] = res.software_loops
            base["has_local_membar_in_vf_call"] = res.has_local_membar_in_vf_call
            base["vf_fusion_breaks"] = res.vf_fusion_breaks
            base["alias_serialized_buffers"] = res.alias_serialized_buffers
            base["dual_issue_ipc_collapse"] = res.dual_issue_ipc_collapse
            base["dcache_dcci_flushes"] = res.dcache_dcci_flushes
            merged_extra = dict(base.get("extra_signals") or {}) if isinstance(base.get("extra_signals"), Mapping) else {}
            merged_extra.update({str(k): float(v) for k, v in res.extra_signals.items()})
            base["extra_signals"] = merged_extra
        elif isinstance(base.get("extra_signals"), Mapping):
            base["extra_signals"] = {str(k): float(v) for k, v in base["extra_signals"].items()}

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in base.items() if k in valid_keys}
        raw_axes = filtered.get("topology_axes")
        if isinstance(raw_axes, Mapping):
            filtered["topology_axes"] = TopologyAxes.from_dict(raw_axes)
        return cls(**filtered)


@dataclass(frozen=True)
class CostPrediction:
    """Detailed 3-tier cost breakdown and predicted latency in microseconds."""

    pred_us: float
    tier0_physics_us: float
    tier1_penalty_us: float
    tier2_residual_factor: float
    cube_floor_us: float
    vec_floor_us: float
    hbm_floor_us: float
    l2_hit_rate: float
    wave_quantization_ratio: float
    bank_conflict_us: float
    effective_hbm_bw_bps: float
    dominant_bound: str
    feature_vector: tuple[float, ...] = field(default_factory=tuple)
    launch_overhead_us: float = 0.0
    fixpipe_floor_us: float = 0.0
    cv_sync_us: float = 0.0
    topology_bucket: str = "global"

    def to_dict(self) -> dict[str, object]:
        """Serialize prediction breakdown to JSON-safe dictionary."""
        out = asdict(self)
        out["feature_vector"] = list(self.feature_vector)
        return out


@dataclass(frozen=True)
class CalibrationSample:
    """Persisted calibration observation."""

    feature_vector: tuple[float, ...]
    measured_us: float
    analytical_us: float
    topology_bucket: str = "global"

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_vector": list(self.feature_vector),
            "measured_us": self.measured_us,
            "analytical_us": self.analytical_us,
            "topology_bucket": self.topology_bucket,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CalibrationSample:
        return cls(
            feature_vector=tuple(float(x) for x in (payload.get("feature_vector") or ())),
            measured_us=float(payload["measured_us"]),
            analytical_us=float(payload["analytical_us"]),
            topology_bucket=str(payload.get("topology_bucket") or "global"),
        )


@dataclass(frozen=True)
class CostModelEvalReport:
    """Empirical accuracy and intra-operator ranking evaluation summary."""

    sample_count: int
    mape: float
    median_ape: float
    overall_spearman_rho: float
    mean_intra_group_spearman_rho: float
    pairwise_accuracy: float
    top1_regret_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialize report to JSON-safe dictionary."""
        return {
            "sample_count": int(self.sample_count),
            "mape": round(self.mape, 6),
            "median_ape": round(self.median_ape, 6),
            "overall_spearman_rho": round(self.overall_spearman_rho, 6),
            "mean_intra_group_spearman_rho": round(self.mean_intra_group_spearman_rho, 6),
            "pairwise_accuracy": round(self.pairwise_accuracy, 6),
            "top1_regret_ratio": round(self.top1_regret_ratio, 6),
        }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: float(item[1]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        v_i = float(indexed[i][1])
        while j < len(indexed) and abs(float(indexed[j][1]) - v_i) <= 1e-12:
            j += 1
        avg_rank = 0.5 * float(i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / float(n)
    my = sum(ry) / float(n)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


def evaluate_cost_model_ranking(
    predictions: Sequence[float],
    actuals: Sequence[float],
    *,
    groups: Sequence[str] | None = None,
) -> CostModelEvalReport:
    """Compute MAPE, Spearman rank correlation, pairwise accuracy, and Top-1 regret."""
    n = len(predictions)
    if n == 0 or len(actuals) != n:
        return CostModelEvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if groups is not None and len(groups) != n:
        raise ValueError("groups length must match predictions and actuals length")

    apes = sorted(
        abs(float(p) - float(a)) / max(1e-6, abs(float(a)))
        for p, a in zip(predictions, actuals)
    )
    mape = sum(apes) / float(n)
    mid = n // 2
    median_ape = apes[mid] if (n % 2 == 1) else 0.5 * (apes[mid - 1] + apes[mid])
    overall_rho = _spearman_corr(predictions, actuals)

    if groups is None:
        bucket_indices: dict[str, list[int]] = {"global": list(range(n))}
    else:
        bucket_indices = {}
        for idx, g in enumerate(groups):
            bucket_indices.setdefault(str(g), []).append(idx)

    group_rhos: list[float] = []
    regrets: list[float] = []
    pair_correct = 0.0
    pair_total = 0

    for idxs in bucket_indices.values():
        if len(idxs) < 2:
            continue
        gp = [float(predictions[i]) for i in idxs]
        ga = [float(actuals[i]) for i in idxs]
        group_rhos.append(_spearman_corr(gp, ga))
        best_pred_pos = min(range(len(idxs)), key=lambda pos: gp[pos])
        oracle_min = min(ga)
        if oracle_min > 0.0:
            regrets.append((ga[best_pred_pos] - oracle_min) / oracle_min)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if abs(ga[i] - ga[j]) <= 1e-9:
                    continue
                pair_total += 1
                if abs(gp[i] - gp[j]) <= 1e-12:
                    pair_correct += 0.5
                elif (gp[i] < gp[j]) == (ga[i] < ga[j]):
                    pair_correct += 1.0

    mean_group_rho = sum(group_rhos) / float(len(group_rhos)) if group_rhos else overall_rho
    pairwise_acc = (pair_correct / float(pair_total)) if pair_total > 0 else 0.0
    mean_regret = (sum(regrets) / float(len(regrets))) if regrets else 0.0
    return CostModelEvalReport(
        sample_count=n,
        mape=mape,
        median_ape=median_ape,
        overall_spearman_rho=overall_rho,
        mean_intra_group_spearman_rho=mean_group_rho,
        pairwise_accuracy=pairwise_acc,
        top1_regret_ratio=mean_regret,
    )


class HybridAscendCostModel:
    """3-Tier Hybrid Ascend 950 Cost Model (Physics Floor + Compiler Penalties + Hierarchical Ridge)."""

    FEATURE_DIM = 13

    def __init__(
        self,
        target: AscendTarget | None = None,
        *,
        clock_ghz: float = DEFAULT_CLOCK_GHZ,
        ridge_alpha: float = 1.0,
        calibration_path: str | Path | None = None,
        behavior_registry: CompilerBehaviorRegistry | None = None,
    ) -> None:
        self.target = target or AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)
        self.clock_ghz = float(getattr(self.facts, "clock_ghz", clock_ghz))
        self.ridge_alpha = float(max(1e-6, ridge_alpha))
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.behavior_registry: CompilerBehaviorRegistry = (
            behavior_registry if behavior_registry is not None else DEFAULT_COMPILER_BEHAVIOR_REGISTRY
        )
        self._samples: list[CalibrationSample] = []
        self._weights: list[float] = [0.0] * self.FEATURE_DIM
        self._bucket_weights: dict[str, list[float]] = {}
        if self.calibration_path is not None and self.calibration_path.exists():
            self.load_calibration(self.calibration_path)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(self._weights)

    @property
    def bucket_weights(self) -> dict[str, tuple[float, ...]]:
        return {k: tuple(v) for k, v in self._bucket_weights.items()}

    def _pad_feature_vector(self, vec: Sequence[float]) -> tuple[float, ...]:
        vals = [float(x) for x in vec[: self.FEATURE_DIM]]
        if len(vals) < self.FEATURE_DIM:
            vals.extend([0.0] * (self.FEATURE_DIM - len(vals)))
        return tuple(vals)

    def _extract_surrogate_features(
        self,
        cf: CostFeatures,
        *,
        cube_floor_us: float,
        vec_floor_us: float,
        hbm_floor_us: float,
        wave_ratio: float,
        l2_hit_rate: float,
        tier1_penalty_us: float,
        tier0_us: float,
        launch_us: float,
    ) -> tuple[float, ...]:
        total_floor = max(1e-6, cube_floor_us + vec_floor_us + hbm_floor_us)
        axes = cf.topology_axes
        is_coupled = 1.0 if (axes is not None and axes.compute_engine == "coupled_1c2v") else 0.0
        is_aic = 1.0 if (axes is not None and axes.compute_engine == "aic_only") else 0.0
        is_irreg = (
            1.0
            if (
                axes is not None
                and axes.iteration_regularity
                in ("irregular_simt", "ragged_group", "dynamic_opcode_stream")
            )
            else 0.0
        )
        return (
            1.0,  # 0: bias
            math.log1p(max(0.0, tier0_us)) / 10.0,  # 1: log scale
            cube_floor_us / total_floor,  # 2: cube share
            vec_floor_us / total_floor,  # 3: vec share
            hbm_floor_us / total_floor,  # 4: hbm share
            max(0.0, wave_ratio - 1.0),  # 5: wave quantization tail
            1.0 - max(0.0, min(1.0, l2_hit_rate)),  # 6: L2 miss fraction
            1.0 if cf.contig_burst_bytes < 16384 else 0.0,  # 7: short burst indicator
            min(2.0, tier1_penalty_us / max(1e-3, tier0_us)),  # 8: compiler penalty ratio
            launch_us / max(1e-3, tier0_us),  # 9: small-shape launch fraction
            is_coupled,  # 10: coupled_1c2v indicator
            is_aic,  # 11: aic_only indicator
            is_irreg,  # 12: irregular/ragged/dynamic opcode indicator
        )

    def _compute_tier0_and_tier1(
        self,
        cf: CostFeatures,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        tuple[float, ...],
    ]:
        aic_cores = max(1, cf.active_aic_cores or self.facts.num_aic_cores)
        aiv_cores = max(1, cf.active_aiv_cores or self.facts.num_aiv_cores)
        clock_hz = self.clock_ghz * 1e9

        # 1. Per-dtype Cube MACs/cycle @ 1.65 GHz + Wave Quantization
        macs_per_cycle = _cube_macs_for_dtype(self.facts, cf.cube_dtype, hf32=cf.hf32)
        aic_peak_flops_per_sec = float(aic_cores) * float(macs_per_cycle) * 2.0 * clock_hz
        ideal_cube_us = (cf.cube_flops / aic_peak_flops_per_sec) * 1e6 if cf.cube_flops > 0 else 0.0

        # Wave Quantization: ceil(tiles / active_cores)
        wave_ratio = 1.0
        if cf.total_tiles > 0:
            active_grid = aic_cores if cf.cube_flops > 0 else aiv_cores
            waves = math.ceil(cf.total_tiles / max(1, active_grid))
            ideal_waves = cf.total_tiles / float(max(1, active_grid))
            wave_ratio = max(1.0, waves / max(1e-6, ideal_waves))
        cube_floor_us = ideal_cube_us * wave_ratio

        # AIV Vector FLOPs floor
        aiv_peak_flops_per_sec = float(aiv_cores) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * clock_hz
        vec_floor_us = (cf.vec_flops / aiv_peak_flops_per_sec) * 1e6 if cf.vec_flops > 0 else 0.0
        if cf.cube_flops <= 0 and cf.total_tiles > 0:
            vec_floor_us *= wave_ratio

        # 2. Segmented HBM burst curve + SWAT L2 reuse hit model
        burst_bytes = max(1, int(cf.contig_burst_bytes))
        eff_hbm_bw = max(1.0, self.facts.effective_hbm_bandwidth_bps(burst_bytes))
        l2_cap = int(getattr(self.facts, "l2_capacity_bytes", DEFAULT_L2_CAPACITY_BYTES))

        l2_hit_rate = 0.0
        if not cf.l2_bypass and cf.gm_bytes > 0:
            if cf.l2_reuse_fraction > 0.0:
                l2_hit_rate = min(0.85, max(0.0, cf.l2_reuse_fraction))
            elif cf.swizzle_window >= 4:
                ws = cf.l2_working_set_bytes if cf.l2_working_set_bytes > 0 else int(cf.gm_bytes * 0.25)
                fit_ratio = min(1.0, float(l2_cap) / float(max(1, ws)))
                l2_hit_rate = round(0.40 * fit_ratio, 6)

        effective_gm_bytes = float(cf.gm_bytes) * (1.0 - l2_hit_rate)
        hbm_floor_us = (effective_gm_bytes / eff_hbm_bw) * 1e6 if effective_gm_bytes > 0 else 0.0

        # L2 bandwidth check (101 B/cycle/AIC)
        l2_bw_bps = float(aic_cores) * float(getattr(self.facts, "l2_bytes_per_cycle_per_aic", DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC)) * clock_hz
        l2_floor_us = ((float(cf.gm_bytes) * l2_hit_rate) / max(1.0, l2_bw_bps)) * 1e6 if l2_hit_rate > 0 else 0.0

        # 3. FixPipe writeback floor, 1C2V cross-core sync, Launch floor & bank-conflict penalties
        fixpipe_bpc = float(getattr(self.facts, "fixpipe_bytes_per_cycle", DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC) or DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC)
        fixpipe_bw_bps = float(aic_cores) * fixpipe_bpc * clock_hz
        fixpipe_floor_us = (
            (float(cf.fixpipe_bytes) / max(1.0, fixpipe_bw_bps)) * 1e6
            if cf.fixpipe_bytes > 0.0
            else 0.0
        )

        if cf.cv_sync_events > 0:
            cv_sync_us = float(cf.cv_sync_events) * DEFAULT_CV_SYNC_EVENT_US
        elif (
            cf.topology_axes is not None
            and cf.topology_axes.compute_engine == "coupled_1c2v"
            and cube_floor_us > 0.0
            and vec_floor_us > 0.0
        ):
            cv_sync_us = 0.08 * min(cube_floor_us, vec_floor_us)
        else:
            cv_sync_us = 0.0

        has_work = (
            cf.cube_flops > 0.0
            or cf.vec_flops > 0.0
            or cf.gm_bytes > 0.0
            or cf.fixpipe_bytes > 0.0
        )
        if has_work and cf.kernel_launches > 0:
            launch_unit = (
                float(cf.launch_overhead_us)
                if cf.launch_overhead_us is not None
                else DEFAULT_LAUNCH_OVERHEAD_US
            )
            launch_us = float(cf.kernel_launches) * max(0.0, launch_unit)
        else:
            launch_us = 0.0

        # L1/UB bank-conflict cycle penalties
        ub_conflict = max(0.0, min(1.0, cf.ub_bank_conflict_ratio))
        if (
            ub_conflict == 0.0
            and cf.ub_row_stride_bytes > 0
            and (cf.ub_row_stride_bytes % DEFAULT_UB_BANK_MODULUS_BYTES == 0)
            and cf.ub_bank_pad_datablocks == 0
            and cf.ub_rw_shift_datablocks == 0
        ):
            ub_conflict = 0.35
        l1_conflict = 0.25 if (cf.l1_stages >= 2 and not cf.l1_half_bank_split) else 0.0
        bank_conflict_us = vec_floor_us * ub_conflict + cube_floor_us * l1_conflict

        # Combine Roofline Physics Floor (Launch + bottleneck pipes + non-overlapped tail)
        compute_us = max(cube_floor_us, vec_floor_us, fixpipe_floor_us)
        exec_us = (
            max(compute_us, hbm_floor_us, l2_floor_us)
            + 0.15 * min(compute_us, hbm_floor_us)
            + bank_conflict_us
            + cv_sync_us
        )
        tier0_us = launch_us + exec_us

        # 4. Tier 1: Compiler Resource Penalty via extensible CompilerBehaviorRegistry
        base_ref = max(1.0, tier0_us)
        res_summary = BishengResourceSummary(
            spill_bytes=cf.spill_bytes,
            stack_bytes=cf.stack_bytes,
            vloopv2_loops=cf.vloopv2_loops,
            software_loops=cf.software_loops,
            has_local_membar_in_vf_call=cf.has_local_membar_in_vf_call,
            vf_fusion_breaks=cf.vf_fusion_breaks,
            alias_serialized_buffers=cf.alias_serialized_buffers,
            dual_issue_ipc_collapse=cf.dual_issue_ipc_collapse,
            dcache_dcci_flushes=cf.dcache_dcci_flushes,
            extra_signals=dict(cf.extra_signals),
        )
        tier1_penalty_us, _tier1_breakdown = self.behavior_registry.compute_tier1_penalties(
            res_summary,
            context={
                "base_ref": base_ref,
                "tier0_us": tier0_us,
                "cube_floor_us": cube_floor_us,
                "vec_floor_us": vec_floor_us,
                "hbm_floor_us": hbm_floor_us,
                "cost_features": cf,
            },
        )

        # Short unaligned burst (< 16 KiB) / unaligned tail issue penalty
        if 0 < cf.contig_burst_bytes < 16384:
            shortness = (16384.0 - float(cf.contig_burst_bytes)) / 16384.0
            tier1_penalty_us += hbm_floor_us * 0.15 * shortness + 0.5 * shortness
        if cf.has_unaligned_tail:
            tier1_penalty_us += max(0.25, 0.03 * base_ref)

        # Dominant bound classification
        if cf.spill_bytes > 0 and tier1_penalty_us >= base_ref * 0.3:
            dominant = "compiler_spill"
        elif bank_conflict_us > 0.3 * max(1e-6, tier0_us):
            dominant = "bank_conflict"
        elif launch_us > exec_us and launch_us > 0.5 * max(1e-6, tier0_us):
            dominant = "launch_overhead"
        elif fixpipe_floor_us > 0.0 and fixpipe_floor_us >= max(cube_floor_us, vec_floor_us, hbm_floor_us):
            dominant = "fixpipe"
        elif cv_sync_us > 0.25 * max(1e-6, tier0_us):
            dominant = "cv_sync"
        elif hbm_floor_us >= max(cube_floor_us, vec_floor_us):
            dominant = "hbm"
        elif cube_floor_us >= vec_floor_us:
            dominant = "cube"
        else:
            dominant = "vector"

        feat_vec = self._extract_surrogate_features(
            cf,
            cube_floor_us=cube_floor_us,
            vec_floor_us=vec_floor_us,
            hbm_floor_us=hbm_floor_us,
            wave_ratio=wave_ratio,
            l2_hit_rate=l2_hit_rate,
            tier1_penalty_us=tier1_penalty_us,
            tier0_us=tier0_us,
            launch_us=launch_us,
        )
        return (
            tier0_us,
            tier1_penalty_us,
            cube_floor_us,
            vec_floor_us,
            hbm_floor_us,
            l2_hit_rate,
            wave_ratio,
            bank_conflict_us,
            eff_hbm_bw,
            launch_us,
            fixpipe_floor_us,
            cv_sync_us,
            dominant,
            feat_vec,
        )

    def predict(
        self,
        workload: CostFeatures | Mapping[str, object],
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> CostPrediction:
        """Predict kernel latency (us) across Tier 0 physics, Tier 1 compiler penalties, and Tier 2 surrogate."""
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        (
            tier0_us,
            tier1_us,
            cube_us,
            vec_us,
            hbm_us,
            l2_hit,
            wave_ratio,
            bank_us,
            eff_bw,
            launch_us,
            fixpipe_us,
            cv_sync_us,
            dominant,
            feat_vec,
        ) = self._compute_tier0_and_tier1(cf)
        base_us = max(1e-6, tier0_us + tier1_us)
        bucket = topology_bucket_for_axes(cf.topology_axes)
        weights = self._bucket_weights.get(bucket, self._weights)
        log_res = sum(w * x for w, x in zip(weights, feat_vec))
        log_res = max(-2.5, min(2.5, log_res))
        residual_factor = math.exp(log_res)
        pred_us = base_us * residual_factor if (tier0_us > 0.0 or tier1_us > 0.0) else 0.0
        return CostPrediction(
            pred_us=pred_us,
            tier0_physics_us=tier0_us,
            tier1_penalty_us=tier1_us,
            tier2_residual_factor=residual_factor,
            cube_floor_us=cube_us,
            vec_floor_us=vec_us,
            hbm_floor_us=hbm_us,
            l2_hit_rate=l2_hit,
            wave_quantization_ratio=wave_ratio,
            bank_conflict_us=bank_us,
            effective_hbm_bw_bps=eff_bw,
            dominant_bound=dominant,
            feature_vector=feat_vec,
            launch_overhead_us=launch_us,
            fixpipe_floor_us=fixpipe_us,
            cv_sync_us=cv_sync_us,
            topology_bucket=bucket,
        )

    def observe(
        self,
        workload: CostFeatures | Mapping[str, object],
        measured_us: float,
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        persist: bool = True,
        refit: bool = True,
    ) -> float:
        """Record an empirical measurement and incrementally refit hierarchical Ridge residual surrogates."""
        if measured_us <= 0.0:
            raise ValueError(f"measured_us must be positive, got {measured_us!r}")
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        tier0_us, tier1_us, *_rest, feat_vec = self._compute_tier0_and_tier1(cf)
        analytical_us = max(1e-6, tier0_us + tier1_us)
        bucket = topology_bucket_for_axes(cf.topology_axes)
        sample = CalibrationSample(
            feature_vector=feat_vec,
            measured_us=float(measured_us),
            analytical_us=analytical_us,
            topology_bucket=bucket,
        )
        self._samples.append(sample)
        if refit:
            self._refit_ridge()
        if persist and self.calibration_path is not None:
            self.save_calibration(self.calibration_path)
        return analytical_us

    def observe_batch(
        self,
        items: Sequence[
            tuple[CostFeatures | Mapping[str, object], float]
            | tuple[
                CostFeatures | Mapping[str, object],
                float,
                BishengResourceSummary | Mapping[str, object] | None,
            ]
        ],
        *,
        persist: bool = True,
    ) -> list[float]:
        """Record multiple calibration measurements, refit Ridge once, and persist once."""
        analytical_list: list[float] = []
        for entry in items:
            if len(entry) == 2:
                workload, measured_us = entry
                bres = None
            elif len(entry) == 3:
                workload, measured_us, bres = entry
            else:
                raise ValueError(f"Expected 2- or 3-tuple in observe_batch, got len={len(entry)}")
            analytical_list.append(
                self.observe(
                    workload,
                    measured_us,
                    bisheng_res=bres,
                    persist=False,
                    refit=False,
                )
            )
        if analytical_list:
            self._refit_ridge()
            if persist and self.calibration_path is not None:
                self.save_calibration(self.calibration_path)
        return analytical_list

    def _refit_ridge(self) -> None:
        d = self.FEATURE_DIM
        if not self._samples:
            self._weights = [0.0] * d
            self._bucket_weights = {}
            return
        decay = 0.985
        n_total = len(self._samples)
        xtx = [[self.ridge_alpha if i == j else 0.0 for j in range(d)] for i in range(d)]
        xty = [0.0] * d
        by_bucket: dict[str, list[CalibrationSample]] = {}
        for idx, s in enumerate(self._samples):
            x = self._pad_feature_vector(s.feature_vector)
            y = math.log(max(1e-6, s.measured_us) / max(1e-6, s.analytical_us))
            y = max(-2.5, min(2.5, y))
            w = decay ** float(n_total - 1 - idx)
            for i in range(d):
                xty[i] += w * x[i] * y
                for j in range(d):
                    xtx[i][j] += w * x[i] * x[j]
            if s.topology_bucket and s.topology_bucket != "global":
                by_bucket.setdefault(s.topology_bucket, []).append(s)
        self._weights = _solve_linear_system(xtx, xty)

        shrink_alpha = max(2.0, self.ridge_alpha * 2.0)
        new_bucket_weights: dict[str, list[float]] = {}
        for bucket, b_samples in by_bucket.items():
            if len(b_samples) < 2:
                continue
            nb = len(b_samples)
            bxtx = [[shrink_alpha if i == j else 0.0 for j in range(d)] for i in range(d)]
            bxty = [shrink_alpha * self._weights[i] for i in range(d)]
            for b_idx, s in enumerate(b_samples):
                x = self._pad_feature_vector(s.feature_vector)
                y = math.log(max(1e-6, s.measured_us) / max(1e-6, s.analytical_us))
                y = max(-2.5, min(2.5, y))
                bw = decay ** float(nb - 1 - b_idx)
                for i in range(d):
                    bxty[i] += bw * x[i] * y
                    for j in range(d):
                        bxtx[i][j] += bw * x[i] * x[j]
            new_bucket_weights[bucket] = _solve_linear_system(bxtx, bxty)
        self._bucket_weights = new_bucket_weights

    def save_calibration(self, path: str | Path) -> None:
        """Persist calibration samples to JSONL or JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix == ".jsonl":
            lines = [json.dumps(s.to_dict(), sort_keys=True) for s in self._samples]
            target_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            payload = {
                "samples": [s.to_dict() for s in self._samples],
                "weights": list(self._weights),
            }
            target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load_calibration(self, path: str | Path) -> int:
        """Reload calibration samples from JSONL or JSON and refit Ridge weights."""
        src = Path(path)
        if not src.exists():
            return 0
        raw = src.read_text(encoding="utf-8").strip()
        if not raw:
            self._samples = []
            self._refit_ridge()
            return 0
        samples: list[CalibrationSample] = []
        if src.suffix == ".jsonl" or not raw.startswith("{"):
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    samples.append(CalibrationSample.from_dict(json.loads(line)))
        else:
            payload = json.loads(raw)
            for item in payload.get("samples") or ():
                samples.append(CalibrationSample.from_dict(item))
        self._samples = samples
        self._refit_ridge()
        return len(self._samples)

    def predict_cannbench_eq3_score(
        self,
        pred_us: float,
        ref_perf_us: float,
        t_hw_us: float,
    ) -> CannbenchScorePrediction:
        """Compute official CANNBench Eq.3 + HAP reward for a predicted latency."""
        return predict_cannbench_eq3_score(pred_us, ref_perf_us, t_hw_us)


def _read_attr_or_key(obj: object | None, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_cost_features_from_ledger(
    ledger: object | None,
    *,
    target: AscendTarget | None = None,
    live_plan: object | None = None,
    axes: TopologyAxes | None = None,
    nav_packet: object | None = None,
    sil_advisory: object | None = None,
    kernel_launches: int = 1,
) -> CostFeatures:
    """Convert ObjectivePhysicalLedger + HIR navigation / SIL advisory into chip-scaled CostFeatures."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    policy: dict[str, Any] = {}
    plan_axes: TopologyAxes | None = None
    if isinstance(live_plan, Mapping):
        policy = dict(live_plan)
    elif live_plan is not None:
        raw_pol = getattr(live_plan, "policy", None)
        if isinstance(raw_pol, Mapping):
            policy = dict(raw_pol)
        raw_plan_axes = getattr(live_plan, "topology_axes", None)
        if isinstance(raw_plan_axes, TopologyAxes):
            plan_axes = raw_plan_axes

    eff_axes = axes if isinstance(axes, TopologyAxes) else plan_axes
    if eff_axes is None and sil_advisory is not None:
        adv_axes = getattr(sil_advisory, "topology_axes", None)
        if isinstance(adv_axes, TopologyAxes):
            eff_axes = adv_axes

    grid_blocks = int(getattr(ledger, "grid_blocks", 0) or 0) if ledger is not None else 0
    if grid_blocks <= 0:
        grid_blocks = int(policy.get("grid_blocks") or 0)
    total_tiles = grid_blocks if grid_blocks > 0 else 56

    raw_dtype = str(
        policy.get("cube_operand_dtype") or policy.get("dtype") or ""
    ).strip()
    if not raw_dtype and sil_advisory is not None:
        in_dtypes = getattr(sil_advisory, "input_dtypes", ()) or ()
        if in_dtypes:
            raw_dtype = str(in_dtypes[0]).strip()
    cube_dtype = raw_dtype or "f16"
    hf32 = bool(policy.get("hf32", policy.get("hf32_allowed", False)))
    macs_per_cycle = _cube_macs_for_dtype(facts, cube_dtype, hf32=hf32)

    pipes = (getattr(ledger, "pipe_cycles", None) or {}) if ledger is not None else {}
    cube_c = int(pipes.get("AIC.PIPE_M", 0)) + int(pipes.get("AIC.CUBE", 0))
    vec_c = int(pipes.get("AIV0.PIPE_V", 0)) + int(pipes.get("AIV1.PIPE_V", 0))
    ledger_cube_flops = float(max(0, cube_c)) * float(macs_per_cycle) * 2.0 * float(total_tiles)
    ledger_vec_flops = (
        float(max(0, vec_c)) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * float(total_tiles)
    )

    nav_roofline = getattr(nav_packet, "roofline", None) if nav_packet is not None else None
    nav_cube_flops = float(_read_attr_or_key(nav_roofline, "cube_flops", 0.0) or 0.0)
    nav_vec_flops = float(_read_attr_or_key(nav_roofline, "vector_flops", 0.0) or 0.0)
    nav_min_gm = float(_read_attr_or_key(nav_roofline, "min_gm_bytes", 0.0) or 0.0)

    cube_flops = max(ledger_cube_flops, nav_cube_flops) if nav_cube_flops > 0.0 else ledger_cube_flops
    vec_flops = max(ledger_vec_flops, nav_vec_flops) if nav_vec_flops > 0.0 else ledger_vec_flops

    adv_roofline = getattr(sil_advisory, "roofline", None) if sil_advisory is not None else None
    adv_median_gm = float(_read_attr_or_key(adv_roofline, "median_gm_bytes", 0.0) or 0.0)

    ledger_gm_per_tile = (
        float(getattr(ledger, "gm_roundtrip_bytes", 0) or 0) if ledger is not None else 0.0
    )
    if ledger_gm_per_tile > 0.0:
        if nav_min_gm <= 0.0 or ledger_gm_per_tile < nav_min_gm * 0.5:
            gm_bytes = max(ledger_gm_per_tile * float(total_tiles), nav_min_gm)
        else:
            gm_bytes = max(ledger_gm_per_tile, nav_min_gm)
    elif nav_min_gm > 0.0:
        gm_bytes = nav_min_gm
    elif adv_median_gm > 0.0:
        gm_bytes = adv_median_gm
    else:
        gm_bytes = 65536.0

    fixpipe_bytes = 0.0
    if cube_flops > 0.0:
        tile_mnk = policy.get("tile_mnk")
        if (
            isinstance(tile_mnk, (list, tuple))
            and len(tile_mnk) >= 2
            and int(tile_mnk[0] or 0) > 0
            and int(tile_mnk[1] or 0) > 0
        ):
            fixpipe_bytes = float(int(tile_mnk[0]) * int(tile_mnk[1]) * 4 * total_tiles)
        else:
            fixpipe_bytes = 0.25 * gm_bytes

    if eff_axes is not None and eff_axes.compute_engine == "coupled_1c2v":
        raw_cv = policy.get("cv_sync_events")
        cv_sync_events = int(raw_cv) if raw_cv is not None else max(2, int(policy.get("l1_stages", 2) or 2) * 2)
    else:
        cv_sync_events = 0

    contig_burst = int(getattr(ledger, "dominant_contig_bytes", 0) or 0) if ledger is not None else 0
    if contig_burst <= 0:
        contig_burst = int(policy.get("contig_burst_bytes") or 32768)

    merged: dict[str, Any] = dict(policy)
    merged.update(
        {
            "cube_flops": cube_flops,
            "vec_flops": vec_flops,
            "cube_dtype": cube_dtype,
            "hf32": hf32,
            "gm_bytes": gm_bytes,
            "contig_burst_bytes": contig_burst,
            "total_tiles": total_tiles,
            "fixpipe_bytes": fixpipe_bytes,
            "cv_sync_events": cv_sync_events,
            "kernel_launches": max(0, int(kernel_launches)),
        }
    )
    if eff_axes is not None:
        merged["topology_axes"] = eff_axes
    return CostFeatures.coerce(merged)


__all__ = [
    "CUBE_MACS_PER_CYCLE",
    "DEFAULT_CV_SYNC_EVENT_US",
    "DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC",
    "DEFAULT_LAUNCH_OVERHEAD_US",
    "CalibrationSample",
    "CannbenchScorePrediction",
    "CostFeatures",
    "CostModelEvalReport",
    "CostPrediction",
    "HybridAscendCostModel",
    "build_cost_features_from_ledger",
    "evaluate_cost_model_ranking",
    "predict_cannbench_eq3_score",
    "topology_bucket_for_axes",
]
