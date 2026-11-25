"""3-Tier Hybrid Ascend 950 (dav-3510) Cost Model and Online Calibrated Surrogate."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    BishengResourceSummary,
    CompilerBehaviorRegistry,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    project_case_workload,
    resolve_candidate_for_regime,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    MicroarchProfile,
    merge_microarch_profiles,
)
from tilefoundry_ascend.ascend.cost_model.pipe_schedule_eval import (
    Ascend950PipeScheduleEvaluator,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
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
DEFAULT_L2_CAPACITY_BYTES = 134_217_728
DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC = 101
DEFAULT_UB_BANK_MODULUS_BYTES = 512
DEFAULT_AIV_VEC_FLOPS_PER_CYCLE = 128
DEFAULT_LAUNCH_OVERHEAD_US = 3.2
DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC = 128
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


def _huber_location(
    values: Sequence[float],
    delta: float = 1.35,
    iters: int = 12,
) -> float:
    """Estimate robust scalar location via pure-Python IRLS Huber weighting."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    mu = float(statistics.median(vals))
    eff_delta = max(1e-6, float(delta))
    for _ in range(max(1, int(iters))):
        w_sum = 0.0
        wy_sum = 0.0
        for y in vals:
            diff = abs(y - mu)
            w = 1.0 if diff <= eff_delta else (eff_delta / diff)
            w_sum += w
            wy_sum += w * y
        if w_sum <= 0.0:
            break
        new_mu = wy_sum / w_sum
        if abs(new_mu - mu) <= 1e-12:
            mu = new_mu
            break
        mu = new_mu
    return mu


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
    microarch_profile: dict[str, object] = field(default_factory=dict)
    steady_state_ii: int = 0
    loop_trips: int = 1
    pipeline_depth: int = 1
    ub_occupancy_ratio: float = 0.0
    profile_is_per_core_full_loop: bool = False

    @classmethod
    def coerce(
        cls,
        workload: CostFeatures | Mapping[str, object] | object,
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> CostFeatures:
        """Build CostFeatures from a dataclass, Mapping, or SIL policy and compiler summary."""
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
        raw_prof = filtered.get("microarch_profile")
        if raw_prof is None:
            filtered["microarch_profile"] = {}
        elif isinstance(raw_prof, MicroarchProfile):
            filtered["microarch_profile"] = raw_prof.to_dict()
        elif isinstance(raw_prof, Mapping):
            filtered["microarch_profile"] = dict(raw_prof)
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
    named_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize prediction breakdown to JSON-safe dictionary."""
        out = asdict(self)
        out["feature_vector"] = list(self.feature_vector)
        out["named_features"] = dict(self.named_features)
        return out


@dataclass(frozen=True)
class ResidualBasisContext:
    """Normalized physical inputs supplied to Tier-2 named basis extractors."""

    tier0_us: float
    cube_floor_us: float
    vec_floor_us: float
    hbm_floor_us: float
    tier1_penalty_us: float
    launch_us: float
    wave_ratio: float
    l2_hit_rate: float
    contig_burst_bytes: int
    axes: TopologyAxes | None


@dataclass(frozen=True)
class ResidualBasisSpec:
    """Declarative specification for a single dimensionless Tier-2 basis function."""

    name: str
    varies_within_case: bool
    min_bucket_samples: int
    extractor: Callable[[ResidualBasisContext], float]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ResidualBasisSpec.name must be non-empty")
        if self.min_bucket_samples < 1:
            raise ValueError(
                f"ResidualBasisSpec.min_bucket_samples must be >= 1, got {self.min_bucket_samples}"
            )


def _floor_denom(ctx: ResidualBasisContext) -> float:
    return max(1e-6, ctx.cube_floor_us + ctx.vec_floor_us + ctx.hbm_floor_us)


def _is_coupled_1c2v(ctx: ResidualBasisContext) -> float:
    return 1.0 if (ctx.axes is not None and ctx.axes.compute_engine == "coupled_1c2v") else 0.0


def _is_aic_only(ctx: ResidualBasisContext) -> float:
    return 1.0 if (ctx.axes is not None and ctx.axes.compute_engine == "aic_only") else 0.0


def _is_irregular_simt(ctx: ResidualBasisContext) -> float:
    if ctx.axes is None:
        return 0.0
    return (
        1.0
        if ctx.axes.iteration_regularity
        in ("irregular_simt", "ragged_group", "dynamic_opcode_stream")
        else 0.0
    )


DEFAULT_RESIDUAL_BASIS_SPECS: tuple[ResidualBasisSpec, ...] = (
    ResidualBasisSpec(
        name="intercept",
        varies_within_case=False,
        min_bucket_samples=2,
        extractor=lambda _ctx: 1.0,
    ),
    ResidualBasisSpec(
        name="log_tier0_scale",
        varies_within_case=True,
        min_bucket_samples=6,
        extractor=lambda ctx: math.log1p(max(0.0, ctx.tier0_us)) / 10.0,
    ),
    ResidualBasisSpec(
        name="cube_bound_share",
        varies_within_case=True,
        min_bucket_samples=6,
        extractor=lambda ctx: ctx.cube_floor_us / _floor_denom(ctx),
    ),
    ResidualBasisSpec(
        name="vec_bound_share",
        varies_within_case=True,
        min_bucket_samples=6,
        extractor=lambda ctx: ctx.vec_floor_us / _floor_denom(ctx),
    ),
    ResidualBasisSpec(
        name="wave_tail_waste",
        varies_within_case=True,
        min_bucket_samples=4,
        extractor=lambda ctx: max(0.0, ctx.wave_ratio - 1.0),
    ),
    ResidualBasisSpec(
        name="l2_miss_ratio",
        varies_within_case=True,
        min_bucket_samples=6,
        extractor=lambda ctx: 1.0 - max(0.0, min(1.0, ctx.l2_hit_rate)),
    ),
    ResidualBasisSpec(
        name="short_dma_burst_ind",
        varies_within_case=True,
        min_bucket_samples=4,
        extractor=lambda ctx: 1.0 if ctx.contig_burst_bytes < 16384 else 0.0,
    ),
    ResidualBasisSpec(
        name="tier1_penalty_ratio",
        varies_within_case=True,
        min_bucket_samples=4,
        extractor=lambda ctx: min(2.0, max(0.0, ctx.tier1_penalty_us) / max(1e-3, ctx.tier0_us)),
    ),
    ResidualBasisSpec(
        name="launch_overhead_ratio",
        varies_within_case=True,
        min_bucket_samples=6,
        extractor=lambda ctx: min(1.0, max(0.0, ctx.launch_us) / max(1e-3, ctx.tier0_us)),
    ),
    ResidualBasisSpec(
        name="is_coupled_1c2v",
        varies_within_case=False,
        min_bucket_samples=8,
        extractor=_is_coupled_1c2v,
    ),
    ResidualBasisSpec(
        name="is_aic_only",
        varies_within_case=False,
        min_bucket_samples=8,
        extractor=_is_aic_only,
    ),
    ResidualBasisSpec(
        name="is_irregular_simt",
        varies_within_case=False,
        min_bucket_samples=8,
        extractor=_is_irregular_simt,
    ),
)

_LEGACY_13D_INDEX_TO_NAME: tuple[tuple[int, str], ...] = (
    (0, "intercept"),
    (1, "log_tier0_scale"),
    (2, "cube_bound_share"),
    (3, "vec_bound_share"),
    (5, "wave_tail_waste"),
    (6, "l2_miss_ratio"),
    (7, "short_dma_burst_ind"),
    (8, "tier1_penalty_ratio"),
    (9, "launch_overhead_ratio"),
    (10, "is_coupled_1c2v"),
    (11, "is_aic_only"),
    (12, "is_irregular_simt"),
)


def _upgrade_legacy_13d_vector(vec: Sequence[float]) -> dict[str, float]:
    """Upgrade a legacy 13-element feature vector into the 12 full-rank named features."""
    vals = [float(x) for x in vec]
    if len(vals) < 13:
        vals.extend([0.0] * (13 - len(vals)))
    return {name: vals[idx] for idx, name in _LEGACY_13D_INDEX_TO_NAME}


class ResidualBasisRegistry:
    """Ordered declarative registry of dimensionless Tier-2 residual basis functions."""

    def __init__(self, specs: Sequence[ResidualBasisSpec] | None = None) -> None:
        self._specs: dict[str, ResidualBasisSpec] = {}
        initial = DEFAULT_RESIDUAL_BASIS_SPECS if specs is None else specs
        for spec in initial:
            self.register(spec)

    def register(
        self,
        spec: ResidualBasisSpec,
        *,
        overwrite: bool = True,
    ) -> ResidualBasisSpec:
        """Register *spec* in deterministic insertion order."""
        if not overwrite and spec.name in self._specs:
            raise ValueError(f"ResidualBasisSpec already registered: {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def clone(self) -> ResidualBasisRegistry:
        """Return an independent shallow copy of this registry."""
        return ResidualBasisRegistry(tuple(self._specs.values()))

    def head0_specs(self) -> tuple[ResidualBasisSpec, ...]:
        """Return ordered basis specs for Head-0 global and bucket log-residual fitting."""
        return tuple(self._specs.values())

    def head2_specs(self) -> tuple[ResidualBasisSpec, ...]:
        """Return ordered within-case varying basis specs for Head-2 pairwise deltas."""
        return tuple(spec for spec in self._specs.values() if spec.varies_within_case)

    def spec_names(self, *, head2_only: bool = False) -> tuple[str, ...]:
        """Return ordered basis spec names for Head-0 or Head-2."""
        active = self.head2_specs() if head2_only else self.head0_specs()
        return tuple(spec.name for spec in active)

    def extract_named(self, ctx: ResidualBasisContext) -> dict[str, float]:
        """Evaluate all registered basis functions against *ctx* into a named mapping."""
        return {spec.name: float(spec.extractor(ctx)) for spec in self._specs.values()}

    @staticmethod
    def vectorize(
        named_features: Mapping[str, float],
        specs: Sequence[ResidualBasisSpec],
    ) -> tuple[float, ...]:
        """Project *named_features* onto an ordered tuple matching *specs*."""
        return tuple(float(named_features.get(spec.name, 0.0)) for spec in specs)


def _effective_sample_count(sample_count: int, forgetting_factor: float) -> float:
    """Compute exponential-forgetting effective sample weight ``sum(gamma ** age)``."""
    if sample_count <= 0:
        return 0.0
    return sum(forgetting_factor ** float(age) for age in range(sample_count))


def _sample_gate_threshold(min_samples: int, forgetting_factor: float) -> float:
    """Return the discounted weight threshold matching *min_samples* unit observations."""
    if min_samples <= 0:
        return 0.0
    if forgetting_factor >= 1.0:
        return float(min_samples)
    return ((1.0 - (forgetting_factor ** float(min_samples))) / (1.0 - forgetting_factor)) - 1e-9


@dataclass(frozen=True)
class CalibrationSample:
    """Persisted calibration observation keyed by named dimensionless basis features."""

    named_features: dict[str, float]
    measured_us: float
    analytical_us: float
    topology_bucket: str = "global"
    basis_schema_version: int = 2

    def __post_init__(self) -> None:
        normalized = {str(k): float(v) for k, v in self.named_features.items()}
        object.__setattr__(self, "named_features", normalized)

    @property
    def feature_vector(self) -> tuple[float, ...]:
        """Return ordered vector view aligned with ``DEFAULT_RESIDUAL_BASIS_SPECS``."""
        return ResidualBasisRegistry.vectorize(self.named_features, DEFAULT_RESIDUAL_BASIS_SPECS)

    def to_dict(self) -> dict[str, object]:
        """Serialize calibration observation with named basis features and schema version."""
        return {
            "named_features": dict(self.named_features),
            "feature_vector": list(self.feature_vector),
            "measured_us": self.measured_us,
            "analytical_us": self.analytical_us,
            "topology_bucket": self.topology_bucket,
            "basis_schema_version": int(self.basis_schema_version),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CalibrationSample:
        """Deserialize a calibration observation, upgrading legacy 13-element vectors."""
        raw_named = payload.get("named_features")
        raw_ver = payload.get("basis_schema_version")
        if isinstance(raw_named, Mapping) and raw_named:
            named = {str(k): float(v) for k, v in raw_named.items()}
            schema_ver = int(raw_ver) if isinstance(raw_ver, (int, float)) else 2
        else:
            raw_vec = payload.get("feature_vector") or ()
            vec = [float(x) for x in raw_vec] if isinstance(raw_vec, Sequence) else []
            if len(vec) == len(DEFAULT_RESIDUAL_BASIS_SPECS) and raw_ver == 2:
                named = {
                    spec.name: vec[idx]
                    for idx, spec in enumerate(DEFAULT_RESIDUAL_BASIS_SPECS)
                }
            else:
                named = _upgrade_legacy_13d_vector(vec)
            schema_ver = 2
        return cls(
            named_features=named,
            measured_us=float(payload["measured_us"]),
            analytical_us=float(payload["analytical_us"]),
            topology_bucket=str(payload.get("topology_bucket") or "global"),
            basis_schema_version=schema_ver,
        )


@dataclass(frozen=True)
class _Tier01Result:
    """Internal Tier-0 physics and Tier-1 compiler penalty evaluation bundle."""

    tier0_us: float
    tier1_penalty_us: float
    cube_floor_us: float
    vec_floor_us: float
    hbm_floor_us: float
    l2_hit_rate: float
    wave_ratio: float
    bank_conflict_us: float
    eff_hbm_bw: float
    launch_us: float
    fixpipe_floor_us: float
    cv_sync_us: float
    dominant: str
    basis_context: ResidualBasisContext
    named_features: dict[str, float]


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
    """3-Tier Hybrid Ascend 950 Cost Model (Roofline Floor, Compiler Penalties, Ridge)."""

    def __init__(
        self,
        target: AscendTarget | None = None,
        *,
        clock_ghz: float = DEFAULT_CLOCK_GHZ,
        ridge_alpha: float = 1.0,
        forgetting_factor: float = 0.985,
        calibration_path: str | Path | None = None,
        behavior_registry: CompilerBehaviorRegistry | None = None,
        basis_registry: ResidualBasisRegistry | None = None,
        anomaly_lake: ExperienceLake | str | Path | None = None,
    ) -> None:
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must be in (0.0, 1.0], got {forgetting_factor!r}"
            )
        self.target = target or AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)
        self.clock_ghz = float(getattr(self.facts, "clock_ghz", clock_ghz))
        self.ridge_alpha = float(max(1e-6, ridge_alpha))
        self._forgetting_factor = float(forgetting_factor)
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.anomaly_lake: ExperienceLake | None
        self.anomaly_path: Path | None
        if isinstance(anomaly_lake, ExperienceLake):
            self.anomaly_lake = anomaly_lake
            self.anomaly_path = anomaly_lake.root_dir / "unmodeled_anomalies.jsonl"
        elif anomaly_lake is not None:
            raw_anomaly_path = Path(anomaly_lake).expanduser().resolve()
            if raw_anomaly_path.suffix == ".jsonl":
                self.anomaly_lake = None
                self.anomaly_path = raw_anomaly_path
            else:
                self.anomaly_lake = ExperienceLake(raw_anomaly_path)
                self.anomaly_path = self.anomaly_lake.root_dir / "unmodeled_anomalies.jsonl"
        elif self.calibration_path is not None:
            self.anomaly_lake = None
            self.anomaly_path = self.calibration_path.parent / "unmodeled_anomalies.jsonl"
        else:
            self.anomaly_lake = None
            self.anomaly_path = None
        self.behavior_registry: CompilerBehaviorRegistry = (
            behavior_registry
            if behavior_registry is not None
            else DEFAULT_COMPILER_BEHAVIOR_REGISTRY
        )
        self._basis_registry: ResidualBasisRegistry = (
            basis_registry if basis_registry is not None else ResidualBasisRegistry()
        )
        self._samples: list[CalibrationSample] = []
        self._weights_by_name: dict[str, float] = {
            spec.name: 0.0 for spec in self._basis_registry.head0_specs()
        }
        self._bucket_weights_by_name: dict[str, dict[str, float]] = {}
        self._global_intercept: float = 0.0
        self._bucket_intercepts: dict[str, float] = {}
        self._bucket_sample_counts: dict[str, int] = {}
        self._quarantined_sample_ids: set[int] = set()
        self._active_global_specs: tuple[str, ...] = ()
        self._active_bucket_specs: dict[str, tuple[str, ...]] = {}
        self._pipe_evaluator = Ascend950PipeScheduleEvaluator(self.target)
        self._msprof_pipe_scales: dict[str, dict[str, tuple[float, float]]] = {}
        self._msprof_event_count: int = 0
        self._pair_deltas: list[tuple[dict[str, float], float]] = []
        self._delta_weights_by_name: dict[str, float] = {
            spec.name: 0.0 for spec in self._basis_registry.head2_specs()
        }
        self._active_head2_specs: tuple[str, ...] = ()
        self._replayed_obs_keys: set[tuple[str, str, str, int, str, float]] = set()
        self._replayed_pair_keys: set[tuple[str, str, str, str, int]] = set()
        if self.calibration_path is not None and self.calibration_path.exists():
            self.load_calibration(self.calibration_path)

    @property
    def basis_registry(self) -> ResidualBasisRegistry:
        """Return active Tier-2 declarative residual basis registry."""
        return self._basis_registry

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def weights(self) -> tuple[float, ...]:
        specs = self._basis_registry.head0_specs()
        eff = self.weights_by_name
        return tuple(eff.get(spec.name, 0.0) for spec in specs)

    @property
    def weights_by_name(self) -> dict[str, float]:
        specs = self._basis_registry.head0_specs()
        out = {spec.name: self._weights_by_name.get(spec.name, 0.0) for spec in specs}
        if "intercept" in out and "intercept" in self._active_global_specs:
            out["intercept"] = out["intercept"] + self._global_intercept
        return out

    @property
    def bucket_weights(self) -> dict[str, tuple[float, ...]]:
        specs = self._basis_registry.head0_specs()
        eff_buckets = self.bucket_weights_by_name
        return {
            bucket: tuple(wmap.get(spec.name, 0.0) for spec in specs)
            for bucket, wmap in eff_buckets.items()
        }

    @property
    def bucket_weights_by_name(self) -> dict[str, dict[str, float]]:
        specs = self._basis_registry.head0_specs()
        out: dict[str, dict[str, float]] = {}
        for bucket, wmap in self._bucket_weights_by_name.items():
            b_out = {spec.name: wmap.get(spec.name, 0.0) for spec in specs}
            if "intercept" in b_out and "intercept" in self._active_bucket_specs.get(bucket, ()):
                b_out["intercept"] = b_out["intercept"] + self._bucket_intercepts.get(
                    bucket, self._global_intercept
                )
            out[bucket] = b_out
        return out

    @property
    def delta_weights(self) -> tuple[float, ...]:
        specs = self._basis_registry.head2_specs()
        return tuple(self._delta_weights_by_name.get(spec.name, 0.0) for spec in specs)

    @property
    def delta_weights_by_name(self) -> dict[str, float]:
        specs = self._basis_registry.head2_specs()
        return {spec.name: self._delta_weights_by_name.get(spec.name, 0.0) for spec in specs}

    def active_head0_spec_names(self, bucket: str = "global") -> tuple[str, ...]:
        """Return active basis spec names for *bucket* (or global if *bucket* is global)."""
        if bucket == "global":
            return self._active_global_specs
        return self._active_bucket_specs.get(bucket, ())

    def active_head2_spec_names(self) -> tuple[str, ...]:
        """Return active Head-2 within-case basis spec names under sample gating."""
        return self._active_head2_specs

    @property
    def pair_delta_count(self) -> int:
        return len(self._pair_deltas)

    @property
    def msprof_event_count(self) -> int:
        return self._msprof_event_count

    def pipe_scale_multipliers(self, bucket: str = "global") -> dict[str, float]:
        """Return Bayes-shrunk [0.5, 2.0] per-pipe multipliers for *bucket*."""
        prior_alpha = 10.0
        pipes = ("CUBE", "VECTOR", "MTE1", "MTE2", "MTE3", "FIXPIPE")
        bucket_stats = self._msprof_pipe_scales.get(bucket) or {}
        global_stats = self._msprof_pipe_scales.get("global") or {}
        out: dict[str, float] = {}
        for pipe in pipes:
            s_log, s_w = bucket_stats.get(pipe, (0.0, 0.0))
            if s_w <= 0.0:
                s_log, s_w = global_stats.get(pipe, (0.0, 0.0))
            scale = math.exp(s_log / (prior_alpha + s_w)) if s_w > 0.0 else 1.0
            out[pipe] = max(0.5, min(2.0, scale))
        return out

    def _compute_tier0_and_tier1(
        self,
        cf: CostFeatures,
        *,
        use_pipe_scales: bool = True,
    ) -> _Tier01Result:
        aic_cores = max(1, cf.active_aic_cores or self.facts.num_aic_cores)
        aiv_cores = max(1, cf.active_aiv_cores or self.facts.num_aiv_cores)
        clock_hz = self.clock_ghz * 1e9
        bucket = topology_bucket_for_axes(cf.topology_axes)
        scales = (
            self.pipe_scale_multipliers(bucket)
            if use_pipe_scales
            else {
                "CUBE": 1.0,
                "VECTOR": 1.0,
                "MTE1": 1.0,
                "MTE2": 1.0,
                "MTE3": 1.0,
                "FIXPIPE": 1.0,
            }
        )

        macs_per_cycle = _cube_macs_for_dtype(self.facts, cf.cube_dtype, hf32=cf.hf32)
        aic_peak_flops_per_sec = float(aic_cores) * float(macs_per_cycle) * 2.0 * clock_hz
        ideal_cube_us = (cf.cube_flops / aic_peak_flops_per_sec) * 1e6 if cf.cube_flops > 0 else 0.0

        wave_ratio = 1.0
        if cf.total_tiles > 0:
            active_grid = aic_cores if cf.cube_flops > 0 else aiv_cores
            waves = math.ceil(cf.total_tiles / max(1, active_grid))
            ideal_waves = cf.total_tiles / float(max(1, active_grid))
            wave_ratio = max(1.0, waves / max(1e-6, ideal_waves))
        cube_floor_us = ideal_cube_us * wave_ratio

        prof = MicroarchProfile.coerce(cf.microarch_profile) if cf.microarch_profile else MicroarchProfile()
        has_cube_or_simd_backbone = (
            cf.cube_flops > 0.0
            or prof.cube_mmad_count > 0
            or prof.cube_macs_total > 0
            or prof.fp_alu_ops > 0
            or prof.int_alu_ops > 0
            or prof.sfu_f16_ops > 0
            or prof.sfu_f32_ops > 0
            or prof.reduce_ops > 0
            or prof.slot0_only_ops > 0
            or (
                cf.topology_axes is not None
                and cf.topology_axes.iteration_regularity == "irregular_simt"
            )
            or float(cf.extra_signals.get("has_simtvf", 0.0) or 0.0) > 0.0
            or float(cf.extra_signals.get("has_simd", 0.0) or 0.0) > 0.0
            or float(cf.extra_signals.get("measured_aiv_vec_ratio", 0.0) or 0.0) >= 0.08
            or float(cf.extra_signals.get("l2_aiv_vec_ratio", 0.0) or 0.0) >= 0.08
        )
        scalar_fallback = float(cf.extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0)
        vec_flops_per_cycle = (
            1.0
            if (scalar_fallback > 0.0 and not has_cube_or_simd_backbone)
            else float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE)
        )
        aiv_peak_flops_per_sec = float(aiv_cores) * vec_flops_per_cycle * clock_hz
        vec_floor_us = (cf.vec_flops / aiv_peak_flops_per_sec) * 1e6 if cf.vec_flops > 0 else 0.0
        if cf.cube_flops <= 0 and cf.total_tiles > 0:
            vec_floor_us *= wave_ratio

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
        dcache_bypass_amp = float(
            cf.extra_signals.get("dcache_bypass_narrow_store", 0.0) or 0.0
        )
        if dcache_bypass_amp > 1.0:
            effective_gm_bytes = effective_gm_bytes * (0.5 + 0.5 * dcache_bypass_amp)
        hbm_floor_us = (effective_gm_bytes / eff_hbm_bw) * 1e6 if effective_gm_bytes > 0 else 0.0

        l2_bw_bps = (
            float(aic_cores)
            * float(getattr(self.facts, "l2_bytes_per_cycle_per_aic", DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC))
            * clock_hz
        )
        l2_floor_us = (
            ((float(cf.gm_bytes) * l2_hit_rate) / max(1.0, l2_bw_bps)) * 1e6
            if l2_hit_rate > 0
            else 0.0
        )

        fixpipe_bpc = float(
            getattr(self.facts, "fixpipe_bytes_per_cycle", DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC)
            or DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC
        )
        fixpipe_bw_bps = float(aic_cores) * fixpipe_bpc * clock_hz
        fixpipe_floor_us = (
            (float(cf.fixpipe_bytes) / max(1.0, fixpipe_bw_bps)) * 1e6
            if cf.fixpipe_bytes > 0.0
            else 0.0
        )

        cube_floor_us *= scales["CUBE"]
        vec_floor_us *= scales["VECTOR"]
        hbm_floor_us *= scales["MTE2"]
        fixpipe_floor_us *= scales["FIXPIPE"]

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
            or (not prof.is_empty)
        )
        eff_launches = (
            max(cf.kernel_launches, prof.subkernel_count)
            if (cf.kernel_launches > 0 and not prof.is_empty)
            else cf.kernel_launches
        )
        if has_work and eff_launches > 0:
            launch_unit = (
                float(cf.launch_overhead_us)
                if cf.launch_overhead_us is not None
                else DEFAULT_LAUNCH_OVERHEAD_US
            )
            launch_us = float(eff_launches) * max(0.0, launch_unit)
        else:
            launch_us = 0.0

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

        if not prof.is_empty:
            eff_prof = replace(
                prof,
                pipeline_depth=max(prof.pipeline_depth, cf.pipeline_depth),
                inner_loop_trips=max(prof.inner_loop_trips, cf.loop_trips),
                subkernel_count=max(prof.subkernel_count, eff_launches),
            )
            peak_hbm_bw = max(1.0, self.facts.effective_hbm_bandwidth_bps(65536))
            burst_eff = max(0.10, min(1.0, eff_hbm_bw / peak_hbm_bw))
            is_coupled = cf.topology_axes is not None and cf.topology_axes.compute_engine == "coupled_1c2v"
            sched = self._pipe_evaluator.evaluate_profile(
                eff_prof,
                cube_flops=cf.cube_flops,
                vec_flops=cf.vec_flops,
                cube_dtype=cf.cube_dtype,
                hf32=cf.hf32,
                total_tiles=cf.total_tiles,
                active_aic_cores=cf.active_aic_cores,
                active_aiv_cores=cf.active_aiv_cores,
                dma_burst_efficiency=burst_eff,
                is_coupled_1c2v=is_coupled,
                ledger_steady_state_ii=cf.steady_state_ii,
                pipe_scale_multipliers=scales,
                profile_is_per_core_full_loop=cf.profile_is_per_core_full_loop,
            )
            cube_floor_us = max(cube_floor_us, sched.pipe_ns["CUBE"] * 1e-3)
            sched_vec_us = sched.pipe_ns["VECTOR"] * 1e-3
            vec_floor_us = max(vec_floor_us, sched_vec_us)
            fixpipe_floor_us = max(fixpipe_floor_us, sched.pipe_ns["FIXPIPE"] * 1e-3)
            hbm_floor_us = max(hbm_floor_us, (sched.pipe_ns["MTE2"] * 1e-3) * (1.0 - l2_hit_rate))
            mte1_us = sched.pipe_ns["MTE1"] * 1e-3
            mte3_us = sched.pipe_ns["MTE3"] * 1e-3
            if prof.cv_sync_barriers > 0:
                cv_sync_us = max(cv_sync_us, float(eff_prof.cv_sync_barriers) * DEFAULT_CV_SYNC_EVENT_US)
            prof_has_bank = prof.ub_bank_conflict_factor > 1.0 or prof.l1_bank_conflict_factor > 1.0
            extra_bank_us = 0.0 if prof_has_bank else bank_conflict_us
            if prof.ub_bank_conflict_factor > 1.0:
                bank_conflict_us = max(
                    bank_conflict_us,
                    vec_floor_us * (1.0 - 1.0 / max(1.0, prof.ub_bank_conflict_factor)),
                )
            compute_us = max(cube_floor_us, vec_floor_us, fixpipe_floor_us, mte1_us, mte3_us)
            roofline_exec_us = (
                max(compute_us, hbm_floor_us, l2_floor_us)
                + 0.15 * min(compute_us, hbm_floor_us)
                + extra_bank_us
                + cv_sync_us
            )
            sched_sync_us = max(0.0, (sched.total_ns - sched.loop_total_ns) * 1e-3)
            sched_exec_us = (sched.loop_total_ns * 1e-3) + max(cv_sync_us, sched_sync_us) + extra_bank_us
            exec_us = max(roofline_exec_us, sched_exec_us)
        else:
            compute_us = max(cube_floor_us, vec_floor_us, fixpipe_floor_us)
            exec_us = (
                max(compute_us, hbm_floor_us, l2_floor_us)
                + 0.15 * min(compute_us, hbm_floor_us)
                + bank_conflict_us
                + cv_sync_us
            )
        tier0_us = launch_us + exec_us

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

        if 0 < cf.contig_burst_bytes < 16384:
            shortness = (16384.0 - float(cf.contig_burst_bytes)) / 16384.0
            tier1_penalty_us += hbm_floor_us * 0.15 * shortness + 0.5 * shortness
        if cf.has_unaligned_tail:
            tier1_penalty_us += max(0.25, 0.03 * base_ref)

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

        basis_ctx = ResidualBasisContext(
            tier0_us=tier0_us,
            cube_floor_us=cube_floor_us,
            vec_floor_us=vec_floor_us,
            hbm_floor_us=hbm_floor_us,
            tier1_penalty_us=tier1_penalty_us,
            launch_us=launch_us,
            wave_ratio=wave_ratio,
            l2_hit_rate=l2_hit_rate,
            contig_burst_bytes=int(cf.contig_burst_bytes),
            axes=cf.topology_axes,
        )
        named_features = self._basis_registry.extract_named(basis_ctx)
        return _Tier01Result(
            tier0_us=tier0_us,
            tier1_penalty_us=tier1_penalty_us,
            cube_floor_us=cube_floor_us,
            vec_floor_us=vec_floor_us,
            hbm_floor_us=hbm_floor_us,
            l2_hit_rate=l2_hit_rate,
            wave_ratio=wave_ratio,
            bank_conflict_us=bank_conflict_us,
            eff_hbm_bw=eff_hbm_bw,
            launch_us=launch_us,
            fixpipe_floor_us=fixpipe_floor_us,
            cv_sync_us=cv_sync_us,
            dominant=dominant,
            basis_context=basis_ctx,
            named_features=named_features,
        )

    def predict(
        self,
        workload: CostFeatures | Mapping[str, object],
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> CostPrediction:
        """Predict kernel latency (us) across Tier 0 physics, Tier 1 penalties, and Tier 2."""
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        t01 = self._compute_tier0_and_tier1(cf)
        base_us = max(1e-6, t01.tier0_us + t01.tier1_penalty_us)
        bucket = topology_bucket_for_axes(cf.topology_axes)
        head0_specs = self._basis_registry.head0_specs()
        feat_vec = self._basis_registry.vectorize(t01.named_features, head0_specs)
        weight_map = self._bucket_weights_by_name.get(bucket, self._weights_by_name)
        intercept = self._bucket_intercepts.get(bucket, self._global_intercept)
        log_res = intercept + sum(
            weight_map.get(spec.name, 0.0) * t01.named_features.get(spec.name, 0.0)
            for spec in head0_specs
        )
        log_res = max(-6.0, min(6.0, log_res))
        residual_factor = math.exp(log_res)
        pred_us = base_us * residual_factor if (t01.tier0_us > 0.0 or t01.tier1_penalty_us > 0.0) else 0.0
        return CostPrediction(
            pred_us=pred_us,
            tier0_physics_us=t01.tier0_us,
            tier1_penalty_us=t01.tier1_penalty_us,
            tier2_residual_factor=residual_factor,
            cube_floor_us=t01.cube_floor_us,
            vec_floor_us=t01.vec_floor_us,
            hbm_floor_us=t01.hbm_floor_us,
            l2_hit_rate=t01.l2_hit_rate,
            wave_quantization_ratio=t01.wave_ratio,
            bank_conflict_us=t01.bank_conflict_us,
            effective_hbm_bw_bps=t01.eff_hbm_bw,
            dominant_bound=t01.dominant,
            feature_vector=feat_vec,
            launch_overhead_us=t01.launch_us,
            fixpipe_floor_us=t01.fixpipe_floor_us,
            cv_sync_us=t01.cv_sync_us,
            topology_bucket=bucket,
            named_features=dict(t01.named_features),
        )

    def _write_anomaly_record(self, anomaly_record: dict[str, object]) -> None:
        if self.anomaly_lake is not None:
            self.anomaly_lake.append_unmodeled_anomaly(anomaly_record)
        elif self.anomaly_path is not None:
            self.anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(anomaly_record, sort_keys=True, separators=(",", ":")) + "\n"
            with self.anomaly_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def observe(
        self,
        workload: CostFeatures | Mapping[str, object],
        measured_us: float,
        *,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        persist: bool = True,
        refit: bool = True,
    ) -> float:
        """Record an empirical measurement and refit sample-gated hierarchical Ridge weights."""
        if measured_us <= 0.0:
            raise ValueError(f"measured_us must be positive, got {measured_us!r}")
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        t01 = self._compute_tier0_and_tier1(cf)
        analytical_us = max(1e-6, t01.tier0_us + t01.tier1_penalty_us)
        bucket = topology_bucket_for_axes(cf.topology_axes)
        log_residual = math.log(float(measured_us) / analytical_us)
        is_extreme_corruption = (not math.isfinite(log_residual)) or abs(log_residual) > 8.0
        is_established_bucket_outlier = (
            bucket in self._bucket_intercepts
            and self._bucket_sample_counts.get(bucket, 0) >= 3
            and abs(log_residual - self._bucket_intercepts[bucket]) > 2.0
        )
        if is_extreme_corruption or is_established_bucket_outlier:
            anomaly_record: dict[str, object] = {
                "measured_us": float(measured_us),
                "analytical_us": analytical_us,
                "log_residual": log_residual,
                "topology_bucket": bucket,
                "dominant_bound": t01.dominant,
                "extra_signals": {str(k): float(v) for k, v in cf.extra_signals.items()},
            }
            self._write_anomaly_record(anomaly_record)
            return analytical_us
        sample = CalibrationSample(
            named_features=dict(t01.named_features),
            measured_us=float(measured_us),
            analytical_us=analytical_us,
            topology_bucket=bucket,
            basis_schema_version=2,
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

    def _solve_gated_ridge(
        self,
        rows: Sequence[tuple[Mapping[str, float], float]],
        specs: Sequence[ResidualBasisSpec],
        *,
        alpha: float,
        prior_weights: Mapping[str, float] | None = None,
        y_cap: float = 6.0,
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        """Solve closed-form Ridge on the sample-gated active subspace only."""
        weights: dict[str, float] = {spec.name: 0.0 for spec in specs}
        n_total = len(rows)
        if n_total == 0:
            return weights, ()
        n_eff = _effective_sample_count(n_total, self._forgetting_factor)
        active_specs = tuple(
            spec
            for spec in specs
            if n_eff >= _sample_gate_threshold(spec.min_bucket_samples, self._forgetting_factor)
        )
        if not active_specs:
            return weights, ()
        k = len(active_specs)
        xtx = [
            [
                (
                    1e-4
                    if (prior_weights is None and active_specs[i].name == "intercept")
                    else alpha
                )
                if i == j
                else 0.0
                for j in range(k)
            ]
            for i in range(k)
        ]
        xty = [
            (alpha * float(prior_weights.get(spec.name, 0.0)))
            if prior_weights is not None
            else 0.0
            for spec in active_specs
        ]
        for idx, (named_x, y_raw) in enumerate(rows):
            y = max(-y_cap, min(y_cap, float(y_raw)))
            w = self._forgetting_factor ** float(n_total - 1 - idx)
            x_active = [float(named_x.get(spec.name, 0.0)) for spec in active_specs]
            for i in range(k):
                xi = x_active[i]
                xty[i] += w * xi * y
                for j in range(k):
                    xtx[i][j] += w * xi * x_active[j]
        solved = _solve_linear_system(xtx, xty)
        for spec, val in zip(active_specs, solved):
            weights[spec.name] = float(val)
        return weights, tuple(spec.name for spec in active_specs)

    def _refit_ridge(self) -> None:
        head0_specs = self._basis_registry.head0_specs()
        if not self._samples:
            self._weights_by_name = {spec.name: 0.0 for spec in head0_specs}
            self._bucket_weights_by_name = {}
            self._global_intercept = 0.0
            self._bucket_intercepts = {}
            self._bucket_sample_counts = {}
            self._active_global_specs = ()
            self._active_bucket_specs = {}
            return

        raw_entries: list[tuple[CalibrationSample, str, float]] = []
        raw_ys: list[float] = []
        raw_by_bucket: dict[str, list[float]] = {}
        for sample in self._samples:
            y = math.log(max(1e-6, sample.measured_us) / max(1e-6, sample.analytical_us))
            b_key = (
                sample.topology_bucket
                if (sample.topology_bucket and sample.topology_bucket != "global")
                else "global"
            )
            raw_entries.append((sample, b_key, y))
            raw_ys.append(y)
            if b_key != "global":
                raw_by_bucket.setdefault(b_key, []).append(y)

        self._global_intercept = _huber_location(raw_ys)
        new_bucket_intercepts: dict[str, float] = {}
        for bucket, ys in raw_by_bucket.items():
            new_bucket_intercepts[bucket] = (
                _huber_location(ys) if len(ys) >= 2 else self._global_intercept
            )
        self._bucket_intercepts = new_bucket_intercepts

        global_rows: list[tuple[Mapping[str, float], float]] = []
        by_bucket: dict[str, list[tuple[Mapping[str, float], float]]] = {}
        for sample, b_key, y in raw_entries:
            beta_0_b = (
                self._bucket_intercepts.get(b_key, self._global_intercept)
                if b_key != "global"
                else self._global_intercept
            )
            centered_y = y - beta_0_b
            if abs(centered_y) > 2.0:
                sample_key = id(sample)
                if sample_key not in self._quarantined_sample_ids:
                    self._quarantined_sample_ids.add(sample_key)
                    self._write_anomaly_record(
                        {
                            "measured_us": float(sample.measured_us),
                            "analytical_us": float(sample.analytical_us),
                            "log_residual": float(y),
                            "centered_residual": float(centered_y),
                            "topology_bucket": b_key,
                        }
                    )
                continue
            row = (sample.named_features, centered_y)
            global_rows.append(row)
            if b_key != "global":
                by_bucket.setdefault(b_key, []).append(row)

        self._bucket_sample_counts = {bkt: len(rows) for bkt, rows in by_bucket.items()}
        self._weights_by_name, self._active_global_specs = self._solve_gated_ridge(
            global_rows,
            head0_specs,
            alpha=self.ridge_alpha,
        )

        shrink_alpha = max(2.0, self.ridge_alpha * 2.0)
        new_bucket_weights: dict[str, dict[str, float]] = {}
        new_active_bucket_specs: dict[str, tuple[str, ...]] = {}
        for bucket, b_rows in by_bucket.items():
            b_weights, b_active = self._solve_gated_ridge(
                b_rows,
                head0_specs,
                alpha=shrink_alpha,
                prior_weights=self._weights_by_name,
            )
            if not b_active:
                continue
            new_bucket_weights[bucket] = b_weights
            new_active_bucket_specs[bucket] = b_active
        self._bucket_weights_by_name = new_bucket_weights
        self._active_bucket_specs = new_active_bucket_specs

    def save_calibration(self, path: str | Path) -> None:
        """Persist calibration samples and Two-Head named state to JSONL or JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix == ".jsonl":
            lines = [json.dumps(s.to_dict(), sort_keys=True) for s in self._samples]
            target_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            head0_specs = self._basis_registry.head0_specs()
            head2_specs = self._basis_registry.head2_specs()
            payload = {
                "basis_schema_version": 2,
                "head0_spec_names": [spec.name for spec in head0_specs],
                "head2_spec_names": [spec.name for spec in head2_specs],
                "samples": [s.to_dict() for s in self._samples],
                "weights": list(self.weights),
                "weights_by_name": self.weights_by_name,
                "global_intercept": float(self._global_intercept),
                "bucket_intercepts": {
                    str(k): float(v) for k, v in self._bucket_intercepts.items()
                },
                "delta_weights": list(self.delta_weights),
                "delta_weights_by_name": self.delta_weights_by_name,
                "pair_deltas": [
                    {
                        "named_delta": dict(named_dx),
                        "delta_x": list(self._basis_registry.vectorize(named_dx, head2_specs)),
                        "delta_y": float(dy),
                    }
                    for named_dx, dy in self._pair_deltas
                ],
                "msprof_pipe_scales": {
                    bkt: {p: [float(sl), float(sw)] for p, (sl, sw) in pmap.items()}
                    for bkt, pmap in self._msprof_pipe_scales.items()
                },
                "msprof_event_count": int(self._msprof_event_count),
            }
            target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load_calibration(self, path: str | Path) -> int:
        """Reload calibration samples from JSONL or JSON and refit Ridge surrogates."""
        src = Path(path)
        if not src.exists():
            return 0
        raw = src.read_text(encoding="utf-8").strip()
        if not raw:
            self._samples = []
            self._pair_deltas = []
            self._quarantined_sample_ids = set()
            self._refit_ridge()
            self._refit_delta_ridge()
            return 0
        samples: list[CalibrationSample] = []
        loaded_global_intercept: float | None = None
        loaded_bucket_intercepts: dict[str, float] | None = None
        head2_specs = self._basis_registry.head2_specs()
        if src.suffix == ".jsonl" or not raw.startswith("{"):
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    samples.append(CalibrationSample.from_dict(json.loads(line)))
        else:
            payload = json.loads(raw)
            for item in payload.get("samples") or ():
                samples.append(CalibrationSample.from_dict(item))
            if payload.get("global_intercept") is not None:
                loaded_global_intercept = float(payload["global_intercept"])
            raw_bi = payload.get("bucket_intercepts")
            if isinstance(raw_bi, Mapping):
                loaded_bucket_intercepts = {str(k): float(v) for k, v in raw_bi.items()}
            raw_pmap = payload.get("msprof_pipe_scales")
            if isinstance(raw_pmap, Mapping):
                loaded_scales: dict[str, dict[str, tuple[float, float]]] = {}
                for bkt, pmap in raw_pmap.items():
                    if isinstance(pmap, Mapping):
                        loaded_scales[str(bkt)] = {
                            str(p): (float(vals[0]), float(vals[1]))
                            for p, vals in pmap.items()
                            if isinstance(vals, (list, tuple)) and len(vals) == 2
                        }
                self._msprof_pipe_scales = loaded_scales
            self._msprof_event_count = int(payload.get("msprof_event_count", 0) or 0)
            raw_deltas = payload.get("pair_deltas")
            if isinstance(raw_deltas, list):
                loaded_deltas: list[tuple[dict[str, float], float]] = []
                for item in raw_deltas:
                    if not isinstance(item, Mapping):
                        continue
                    raw_nd = item.get("named_delta")
                    if isinstance(raw_nd, Mapping) and raw_nd:
                        named_dx = {str(k): float(v) for k, v in raw_nd.items()}
                    else:
                        raw_dx = item.get("delta_x") or ()
                        dx_vals = [float(v) for v in raw_dx] if isinstance(raw_dx, Sequence) else []
                        if len(dx_vals) == len(head2_specs):
                            named_dx = {
                                spec.name: dx_vals[idx] for idx, spec in enumerate(head2_specs)
                            }
                        else:
                            upgraded = _upgrade_legacy_13d_vector(dx_vals)
                            named_dx = {
                                spec.name: upgraded.get(spec.name, 0.0) for spec in head2_specs
                            }
                    loaded_deltas.append((named_dx, float(item.get("delta_y", 0.0))))
                self._pair_deltas = loaded_deltas
        self._samples = samples
        self._quarantined_sample_ids = set()
        self._refit_ridge()
        if not self._samples:
            if loaded_global_intercept is not None:
                self._global_intercept = loaded_global_intercept
            if loaded_bucket_intercepts is not None:
                self._bucket_intercepts = loaded_bucket_intercepts
        self._refit_delta_ridge()
        return len(self._samples)

    def observe_msprof(
        self,
        workload: CostFeatures | Mapping[str, object],
        msprof_pmu: Mapping[str, object],
        *,
        measured_us: float | None = None,
        bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        persist: bool = False,
    ) -> dict[str, float]:
        """Head-1 online calibrator: adjust per-pipe scale multipliers from MSProf PMU ratios."""
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        bucket = topology_bucket_for_axes(cf.topology_axes)
        if not isinstance(msprof_pmu, Mapping) or not msprof_pmu:
            return self.pipe_scale_multipliers(bucket)

        raw_dur = (
            measured_us
            if (measured_us is not None and float(measured_us) > 0.0)
            else msprof_pmu.get("dur_us", 0.0)
        )
        try:
            dur_us = float(raw_dur or 0.0)
        except Exception:
            dur_us = 0.0
        if dur_us <= 0.0:
            return self.pipe_scale_multipliers(bucket)

        t01 = self._compute_tier0_and_tier1(cf, use_pipe_scales=False)

        def _pmu_ratio(*keys: str) -> float:
            best = 0.0
            for key in keys:
                val = msprof_pmu.get(key)
                if val is None:
                    continue
                try:
                    num = float(val)
                except Exception:
                    continue
                if not math.isfinite(num) or num <= 0.0:
                    continue
                if num > 1.0:
                    num /= 100.0
                best = max(best, min(1.0, num))
            return best

        pipe_observations = (
            ("CUBE", _pmu_ratio("aic_mad", "aic_mac_ratio"), t01.cube_floor_us),
            ("VECTOR", _pmu_ratio("aiv_vec", "aiv_vec_ratio"), t01.vec_floor_us),
            ("FIXPIPE", _pmu_ratio("aic_fixpipe"), t01.fixpipe_floor_us),
            ("MTE2", _pmu_ratio("aiv_mte2", "aic_mte2", "mte2_wait_ratio"), t01.hbm_floor_us),
        )
        max_log = math.log(2.0)
        updated = False
        target_buckets = [bucket] if bucket == "global" else [bucket, "global"]
        for pipe, ratio, analytical_us in pipe_observations:
            if ratio <= 0.0 or analytical_us <= 1e-3:
                continue
            obs_pipe_us = dur_us * ratio
            if obs_pipe_us <= 0.0:
                continue
            log_r = math.log(max(1e-6, obs_pipe_us) / max(1e-6, analytical_us))
            log_r = max(-max_log, min(max_log, log_r))
            w = min(1.0, max(0.05, ratio))
            for bkt in target_buckets:
                bmap = self._msprof_pipe_scales.setdefault(bkt, {})
                prev_log, prev_w = bmap.get(pipe, (0.0, 0.0))
                bmap[pipe] = (prev_log + w * log_r, prev_w + w)
            updated = True
        if updated:
            self._msprof_event_count += 1
            if persist and self.calibration_path is not None and self.calibration_path.suffix != ".jsonl":
                self.save_calibration(self.calibration_path)
        return self.pipe_scale_multipliers(bucket)

    def _refit_delta_ridge(self) -> None:
        """Fit Head-2 sample-gated Ridge over within-case varying feature differences."""
        head2_specs = self._basis_registry.head2_specs()
        self._delta_weights_by_name, self._active_head2_specs = self._solve_gated_ridge(
            self._pair_deltas,
            head2_specs,
            alpha=max(1.0, self.ridge_alpha),
            y_cap=2.5,
        )

    def observe_pair_delta(
        self,
        cand_workload: CostFeatures | Mapping[str, object],
        cand_us: float,
        parent_workload: CostFeatures | Mapping[str, object],
        parent_us: float,
        *,
        cand_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        refit: bool = True,
    ) -> float:
        """Head-2 calibration: learn residual log-ratio on within-case varying basis diffs."""
        if cand_us <= 0.0 or parent_us <= 0.0:
            raise ValueError(
                f"Both cand_us and parent_us must be positive, got {cand_us!r}, {parent_us!r}"
            )
        cf_c = CostFeatures.coerce(cand_workload, bisheng_res=cand_bisheng_res)
        cf_p = CostFeatures.coerce(parent_workload, bisheng_res=parent_bisheng_res)
        t01_c = self._compute_tier0_and_tier1(cf_c)
        t01_p = self._compute_tier0_and_tier1(cf_p)
        base_c = max(1e-6, t01_c.tier0_us + t01_c.tier1_penalty_us)
        base_p = max(1e-6, t01_p.tier0_us + t01_p.tier1_penalty_us)
        cand_log_res = math.log(float(cand_us) / base_c)
        parent_log_res = math.log(float(parent_us) / base_p)
        actual_log_ratio = math.log(float(cand_us) / float(parent_us))
        analytical_log_ratio = math.log(base_c / base_p)
        delta_y = max(-2.5, min(2.5, actual_log_ratio - analytical_log_ratio))
        if abs(cand_log_res) > 2.5 or abs(parent_log_res) > 2.5:
            return delta_y
        head2_specs = self._basis_registry.head2_specs()
        named_delta = {
            spec.name: t01_c.named_features.get(spec.name, 0.0)
            - t01_p.named_features.get(spec.name, 0.0)
            for spec in head2_specs
        }
        self._pair_deltas.append((named_delta, delta_y))
        if refit:
            self._refit_delta_ridge()
        return delta_y

    def predict_delta_ratio(
        self,
        cand_workload: CostFeatures | Mapping[str, object],
        parent_workload: CostFeatures | Mapping[str, object],
        *,
        cand_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> float:
        """Predict relative latency ratio cand_us / parent_us (< 1.0 means cand is faster)."""
        cf_c = CostFeatures.coerce(cand_workload, bisheng_res=cand_bisheng_res)
        cf_p = CostFeatures.coerce(parent_workload, bisheng_res=parent_bisheng_res)
        if not self._pair_deltas:
            pred_c = max(1e-6, self.predict(cf_c).pred_us)
            pred_p = max(1e-6, self.predict(cf_p).pred_us)
            return pred_c / pred_p
        t01_c = self._compute_tier0_and_tier1(cf_c)
        t01_p = self._compute_tier0_and_tier1(cf_p)
        base_c = max(1e-6, t01_c.tier0_us + t01_c.tier1_penalty_us)
        base_p = max(1e-6, t01_p.tier0_us + t01_p.tier1_penalty_us)
        head2_specs = self._basis_registry.head2_specs()
        log_res = sum(
            self._delta_weights_by_name.get(spec.name, 0.0)
            * (
                t01_c.named_features.get(spec.name, 0.0)
                - t01_p.named_features.get(spec.name, 0.0)
            )
            for spec in head2_specs
        )
        log_res = max(-2.5, min(2.5, log_res))
        return max(1e-4, (base_c / base_p) * math.exp(log_res))

    @staticmethod
    def _obs_dedup_key(
        cand_sha: str,
        operator: str,
        regime_tag: str,
        case_num: int,
        eval_tier: str,
        measured_us: float,
    ) -> tuple[str, str, str, int, str, float]:
        return (
            str(cand_sha),
            str(operator),
            str(regime_tag),
            int(case_num),
            str(eval_tier),
            round(float(measured_us), 6),
        )

    @staticmethod
    def _pair_dedup_key(
        cand_sha: str,
        parent_sha: str,
        operator: str,
        regime_tag: str,
        case_num: int,
    ) -> tuple[str, str, str, str, int]:
        return (
            str(cand_sha),
            str(parent_sha),
            str(operator),
            str(regime_tag),
            int(case_num),
        )

    def mark_online_observation_key(
        self,
        *,
        cand_sha: str,
        operator: str,
        regime_tag: str,
        case_num: int,
        eval_tier: str,
        measured_us: float,
    ) -> None:
        """Record an online observation dedup key matching replay_experience_lake."""
        self._replayed_obs_keys.add(
            self._obs_dedup_key(
                cand_sha,
                operator,
                regime_tag,
                case_num,
                eval_tier,
                measured_us,
            )
        )

    def mark_online_pair_key(
        self,
        *,
        cand_sha: str,
        parent_sha: str,
        operator: str,
        regime_tag: str,
        case_num: int,
    ) -> None:
        """Record an online pairwise-delta dedup key matching replay_experience_lake."""
        self._replayed_pair_keys.add(
            self._pair_dedup_key(
                cand_sha,
                parent_sha,
                operator,
                regime_tag,
                case_num,
            )
        )

    def replay_experience_lake(
        self,
        lake_dir: ExperienceLake | str | Path,
        *,
        persist_calibration: bool = False,
        exclude_run_id: str | None = None,
    ) -> dict[str, int]:
        """Idempotently replay candidates, case observations, MSProf PMU, and pairwise deltas."""
        lake = lake_dir if isinstance(lake_dir, ExperienceLake) else ExperienceLake(lake_dir)
        candidates_by_regime, observations = lake.load_records_by_regime()
        flat_candidates: dict[str, CandidateArtifactRecord] = {}
        for (_sha, _rtag), art in sorted(candidates_by_regime.items()):
            prev = flat_candidates.get(art.cand_sha)
            if prev is None or (not prev.bisheng_summary and art.bisheng_summary) or (
                not prev.base_ledger and art.base_ledger
            ):
                flat_candidates[art.cand_sha] = art
        n_obs = 0
        n_msprof = 0
        n_pairs = 0

        def _project_for(
            art_sha: str,
            obs_rec: CaseObservationRecord,
            resolved_art: CandidateArtifactRecord | None = None,
        ) -> tuple[dict[str, object], dict[str, object] | None]:
            art = resolved_art or resolve_candidate_for_regime(
                candidates_by_regime,
                flat_candidates,
                art_sha,
                obs_rec.regime_tag,
            )
            if art is None:
                proj = project_case_workload(
                    None,
                    case_dims=obs_rec.case_dims,
                    t_hw_us=obs_rec.t_hw_us,
                    target=self.target,
                )
                if obs_rec.projected_gm_bytes > 0.0:
                    proj["gm_bytes"] = max(
                        float(proj["gm_bytes"]),
                        float(obs_rec.projected_gm_bytes),
                    )
                if obs_rec.projected_tiles > 0:
                    proj["total_tiles"] = int(obs_rec.projected_tiles)
                if obs_rec.projected_contig_bytes > 0:
                    proj["contig_burst_bytes"] = int(obs_rec.projected_contig_bytes)
                return proj, None
            axes_obj = TopologyAxes.from_dict(art.topology_axes) if art.topology_axes else None
            proj = project_case_workload(
                art.base_ledger,
                microarch_profile=art.microarch_profile,
                case_dims=obs_rec.case_dims,
                policy=art.config_knobs,
                t_hw_us=obs_rec.t_hw_us,
                axes=axes_obj,
                target=self.target,
            )
            return proj, (art.bisheng_summary or None)

        pending_obs: list[tuple[dict[str, object], dict[str, object] | None, float]] = []
        for obs in observations:
            if obs.measured_us <= 0.0:
                continue
            if exclude_run_id and obs.run_id == exclude_run_id:
                continue
            obs_key = self._obs_dedup_key(
                obs.cand_sha,
                obs.operator,
                obs.regime_tag,
                int(obs.case_num),
                obs.eval_tier,
                float(obs.measured_us),
            )
            if obs_key in self._replayed_obs_keys:
                continue
            self._replayed_obs_keys.add(obs_key)
            wl, bres = _project_for(obs.cand_sha, obs)
            pending_obs.append((wl, bres, obs.measured_us))
            if obs.msprof_pmu:
                self.observe_msprof(
                    wl,
                    obs.msprof_pmu,
                    measured_us=obs.measured_us,
                    bisheng_res=bres,
                    persist=False,
                )
                n_msprof += 1

        for wl, bres, measured_us in pending_obs:
            self.observe(wl, measured_us, bisheng_res=bres, persist=False, refit=False)
            n_obs += 1

        for cand_art, cand_obs, parent_art, parent_obs in lake.extract_pairwise_deltas(
            exclude_run_id=exclude_run_id
        ):
            pkey = self._pair_dedup_key(
                cand_obs.cand_sha,
                parent_obs.cand_sha,
                cand_obs.operator,
                cand_obs.regime_tag,
                int(cand_obs.case_num),
            )
            if pkey in self._replayed_pair_keys:
                continue
            self._replayed_pair_keys.add(pkey)
            wl_c, bres_c = _project_for(cand_art.cand_sha, cand_obs, cand_art)
            wl_p, bres_p = _project_for(parent_art.cand_sha, parent_obs, parent_art)
            self.observe_pair_delta(
                wl_c,
                cand_obs.measured_us,
                wl_p,
                parent_obs.measured_us,
                cand_bisheng_res=bres_c,
                parent_bisheng_res=bres_p,
                refit=False,
            )
            n_pairs += 1

        if n_obs > 0:
            self._refit_ridge()
        if n_pairs > 0:
            self._refit_delta_ridge()
        if (
            persist_calibration
            and self.calibration_path is not None
            and (n_obs > 0 or n_msprof > 0 or n_pairs > 0)
        ):
            self.save_calibration(self.calibration_path)
        return {"observations": n_obs, "msprof_events": n_msprof, "pair_deltas": n_pairs}

    def rebuild_from_experience_lake(
        self,
        lake: ExperienceLake | str | Path,
        *,
        persist_calibration: bool = False,
    ) -> int:
        """Reset online state and recompute all named features and weights from *lake*."""
        lake_obj = lake if isinstance(lake, ExperienceLake) else ExperienceLake(lake)
        head0_specs = self._basis_registry.head0_specs()
        head2_specs = self._basis_registry.head2_specs()
        self._samples = []
        self._weights_by_name = {spec.name: 0.0 for spec in head0_specs}
        self._bucket_weights_by_name = {}
        self._global_intercept = 0.0
        self._bucket_intercepts = {}
        self._bucket_sample_counts = {}
        self._quarantined_sample_ids = set()
        self._active_global_specs = ()
        self._active_bucket_specs = {}
        self._msprof_pipe_scales = {}
        self._msprof_event_count = 0
        self._pair_deltas = []
        self._delta_weights_by_name = {spec.name: 0.0 for spec in head2_specs}
        self._active_head2_specs = ()
        self._replayed_obs_keys = set()
        self._replayed_pair_keys = set()
        stats = self.replay_experience_lake(lake_obj, persist_calibration=persist_calibration)
        return int(stats["observations"])

    def predict_cannbench_eq3_score(
        self,
        pred_us: float,
        ref_perf_us: float,
        t_hw_us: float,
    ) -> CannbenchScorePrediction:
        """Compute official CANNBench Eq.3 + HAP reward for a predicted latency."""
        return predict_cannbench_eq3_score(pred_us, ref_perf_us, t_hw_us)


def _read_attr_or_key(obj: object | None, key: str, default: object = None) -> object:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


@runtime_checkable
class _TotalGmBytesLedger(Protocol):
    def total_gm_bytes(self, grid_blocks: int | None = None) -> int: ...


@runtime_checkable
class _ScopedGmLedger(Protocol):
    gm_roundtrip_bytes: int
    gm_traffic_scope: str


@runtime_checkable
class _RawGmLedger(Protocol):
    gm_roundtrip_bytes: int


def _resolve_ledger_total_gm_bytes(ledger: object | None, total_tiles: int) -> int:
    """Resolve full-grid GM traffic bytes using explicit ledger scope without heuristics."""
    if ledger is None:
        return 0
    if isinstance(ledger, _TotalGmBytesLedger):
        return max(0, int(ledger.total_gm_bytes(total_tiles)))
    if isinstance(ledger, Mapping):
        raw_gm = int(ledger.get("gm_roundtrip_bytes", 0) or 0)
        if raw_gm <= 0:
            return 0
        scope = str(ledger.get("gm_traffic_scope", "per_tile") or "per_tile")
        if scope == "full_grid":
            return raw_gm
        if scope == "per_tile":
            return raw_gm * max(1, int(total_tiles))
        raise ValueError(f"Unsupported gm_traffic_scope: {scope!r}")
    if isinstance(ledger, _ScopedGmLedger):
        raw_gm = int(ledger.gm_roundtrip_bytes or 0)
        if raw_gm <= 0:
            return 0
        scope = str(ledger.gm_traffic_scope or "per_tile")
        if scope == "full_grid":
            return raw_gm
        if scope == "per_tile":
            return raw_gm * max(1, int(total_tiles))
        raise ValueError(f"Unsupported gm_traffic_scope: {scope!r}")
    if isinstance(ledger, ObjectivePhysicalLedger):
        return max(0, int(ledger.gm_roundtrip_bytes) * max(1, int(total_tiles)))
    if isinstance(ledger, _RawGmLedger):
        return max(0, int(ledger.gm_roundtrip_bytes or 0) * max(1, int(total_tiles)))
    return 0


def _extract_ub_occupancy_ratio(ledger: object | None) -> float:
    """Compute UB peak/capacity occupancy ratio from *ledger* waterlines when present."""
    raw_waterlines = _read_attr_or_key(ledger, "waterlines", None)
    if not isinstance(raw_waterlines, Mapping):
        return 0.0
    ub_wl = raw_waterlines.get("UB")
    if isinstance(ub_wl, ScopeWaterline):
        if ub_wl.capacity_bytes > 0 and ub_wl.peak_allocated_bytes >= 0:
            return float(ub_wl.peak_allocated_bytes) / float(ub_wl.capacity_bytes)
        return 0.0
    if isinstance(ub_wl, Mapping):
        cap = int(ub_wl.get("capacity_bytes", 0) or 0)
        peak = int(ub_wl.get("peak_allocated_bytes", 0) or 0)
        return (float(peak) / float(cap)) if (cap > 0 and peak >= 0) else 0.0
    return 0.0


def build_cost_features_from_ledger(
    ledger: object | None,
    *,
    target: AscendTarget | None = None,
    live_plan: object | None = None,
    axes: TopologyAxes | None = None,
    blueprint: object | None = None,
    sil_advisory: object | None = None,
    kernel_launches: int = 1,
    stage_b_profile: MicroarchProfile | Mapping[str, object] | None = None,
) -> CostFeatures:
    """Convert ObjectivePhysicalLedger, HIR blueprint, and SIL advisory into CostFeatures."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    policy: dict[str, object] = {}
    plan_axes: TopologyAxes | None = None
    if isinstance(live_plan, Mapping):
        raw_inner_pol = live_plan.get("policy")
        policy = dict(raw_inner_pol) if isinstance(raw_inner_pol, Mapping) else dict(live_plan)
    elif live_plan is not None:
        raw_pol = getattr(live_plan, "policy", None)
        if isinstance(raw_pol, Mapping):
            policy = dict(raw_pol)
        raw_plan_axes = getattr(live_plan, "topology_axes", None)
        if isinstance(raw_plan_axes, TopologyAxes):
            plan_axes = raw_plan_axes

    eff_axes = axes if isinstance(axes, TopologyAxes) else plan_axes
    if eff_axes is None and sil_advisory is not None:
        adv_axes = _read_attr_or_key(sil_advisory, "topology_axes", None)
        if isinstance(adv_axes, TopologyAxes):
            eff_axes = adv_axes

    grid_blocks = int(_read_attr_or_key(ledger, "grid_blocks", 0) or 0)
    if grid_blocks <= 0:
        grid_blocks = int(policy.get("grid_blocks") or 0)
    total_tiles = grid_blocks if grid_blocks > 0 else 56

    raw_dtype = str(
        policy.get("cube_operand_dtype") or policy.get("dtype") or ""
    ).strip()
    if not raw_dtype and sil_advisory is not None:
        in_dtypes = _read_attr_or_key(sil_advisory, "input_dtypes", ()) or ()
        if isinstance(in_dtypes, Sequence) and in_dtypes:
            raw_dtype = str(in_dtypes[0]).strip()
    cube_dtype = raw_dtype or "f16"
    hf32 = bool(policy.get("hf32", policy.get("hf32_allowed", False)))
    macs_per_cycle = _cube_macs_for_dtype(facts, cube_dtype, hf32=hf32)

    raw_pipes = _read_attr_or_key(ledger, "pipe_cycles", None)
    pipes = raw_pipes if isinstance(raw_pipes, Mapping) else {}
    cube_c = int(pipes.get("AIC.PIPE_M", 0)) + int(pipes.get("AIC.CUBE", 0))
    vec_c = int(pipes.get("AIV0.PIPE_V", 0)) + int(pipes.get("AIV1.PIPE_V", 0))
    ledger_cube_flops = float(max(0, cube_c)) * float(macs_per_cycle) * 2.0 * float(total_tiles)
    ledger_vec_flops = (
        float(max(0, vec_c)) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * float(total_tiles)
    )

    nav_roofline = _read_attr_or_key(blueprint, "roofline", None)
    nav_cube_flops = float(_read_attr_or_key(nav_roofline, "cube_flops", 0.0) or 0.0)
    nav_vec_flops = float(_read_attr_or_key(nav_roofline, "vector_flops", 0.0) or 0.0)
    nav_min_gm = float(_read_attr_or_key(nav_roofline, "min_gm_bytes", 0.0) or 0.0)

    cube_flops = max(ledger_cube_flops, nav_cube_flops) if nav_cube_flops > 0.0 else ledger_cube_flops
    vec_flops = max(ledger_vec_flops, nav_vec_flops) if nav_vec_flops > 0.0 else ledger_vec_flops

    adv_roofline = _read_attr_or_key(sil_advisory, "roofline", None)
    adv_median_gm = float(_read_attr_or_key(adv_roofline, "median_gm_bytes", 0.0) or 0.0)

    actual_gm_bytes = float(_resolve_ledger_total_gm_bytes(ledger, total_tiles))
    if actual_gm_bytes > 0.0:
        gm_bytes = max(actual_gm_bytes, nav_min_gm)
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
        cv_sync_events = (
            int(raw_cv)
            if raw_cv is not None
            else max(2, int(policy.get("l1_stages", 2) or 2) * 2)
        )
    else:
        cv_sync_events = 0

    contig_burst = int(_read_attr_or_key(ledger, "dominant_contig_bytes", 0) or 0)
    if contig_burst <= 0:
        contig_burst = int(policy.get("contig_burst_bytes") or 32768)

    raw_stage_a_prof = _read_attr_or_key(ledger, "microarch_profile", None)
    merged_prof = merge_microarch_profiles(raw_stage_a_prof, stage_b_profile)
    steady_ii = int(_read_attr_or_key(ledger, "steady_state_ii", 0) or 0)
    eff_launches = max(0, int(kernel_launches))
    if eff_launches == 1 and not merged_prof.is_empty:
        eff_launches = max(eff_launches, merged_prof.subkernel_count)

    ub_occ_ratio = _extract_ub_occupancy_ratio(ledger)
    if ub_occ_ratio <= 0.0:
        ub_occ_ratio = float(policy.get("ub_occupancy_ratio", 0.0) or 0.0)

    merged: dict[str, object] = dict(policy)
    raw_extra = merged.get("extra_signals")
    extra_signals: dict[str, float] = (
        {str(k): float(v) for k, v in raw_extra.items()}
        if isinstance(raw_extra, Mapping)
        else {}
    )
    ub_loops = max(0, int(_read_attr_or_key(ledger, "ub_scalar_compute_loops", 0) or 0))
    if ub_loops > 0:
        existing_val = float(extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0)
        extra_signals["scalar_compute_loop_fallback"] = max(existing_val, float(ub_loops))

    min_gm_dma = int(_read_attr_or_key(ledger, "min_gm_dma_bytes", 0) or 0)
    if 0 < min_gm_dma < 32:
        existing_dcache = float(extra_signals.get("dcache_bypass_narrow_store", 0.0) or 0.0)
        extra_signals["dcache_bypass_narrow_store"] = max(
            existing_dcache,
            32.0 / float(min_gm_dma),
        )

    ub_bank_risk = float(_read_attr_or_key(ledger, "ub_bank_conflict_risk", 0.0) or 0.0)
    if ub_bank_risk > 0.0:
        existing_ub_conflict = float(merged.get("ub_bank_conflict_ratio", 0.0) or 0.0)
        merged["ub_bank_conflict_ratio"] = max(
            existing_ub_conflict,
            min(1.0, ub_bank_risk),
        )

    l1_bank_risk = float(_read_attr_or_key(ledger, "l1_bank_conflict_risk", 0.0) or 0.0)
    if l1_bank_risk > 0.0 and not bool(policy.get("l1_half_bank_split", False)):
        merged["l1_half_bank_split"] = False
        merged["l1_stages"] = max(2, int(merged.get("l1_stages", 1) or 1))

    unaligned_pad = int(_read_attr_or_key(ledger, "unaligned_dma_padding_bytes", 0) or 0)
    if unaligned_pad > 0:
        merged["has_unaligned_tail"] = True

    active_blocks = int(_read_attr_or_key(ledger, "active_grid_blocks", 0) or 0)
    if active_blocks > 0:
        is_coupled = eff_axes is not None and eff_axes.compute_engine == "coupled_1c2v"
        if int(policy.get("active_aic_cores") or 0) <= 0:
            merged["active_aic_cores"] = min(max(1, int(facts.num_aic_cores)), active_blocks)
        if int(policy.get("active_aiv_cores") or 0) <= 0:
            aiv_cap = max(1, int(facts.num_aiv_cores))
            merged["active_aiv_cores"] = min(
                aiv_cap,
                active_blocks * 2 if is_coupled else active_blocks,
            )

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
            "kernel_launches": eff_launches,
            "steady_state_ii": steady_ii,
            "loop_trips": max(1, merged_prof.inner_loop_trips) if not merged_prof.is_empty else 1,
            "pipeline_depth": max(1, merged_prof.pipeline_depth) if not merged_prof.is_empty else 1,
            "ub_occupancy_ratio": ub_occ_ratio,
            "extra_signals": extra_signals,
            "microarch_profile": merged_prof.to_dict() if not merged_prof.is_empty else {},
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
    "DEFAULT_RESIDUAL_BASIS_SPECS",
    "CalibrationSample",
    "CannbenchScorePrediction",
    "CostFeatures",
    "CostModelEvalReport",
    "CostPrediction",
    "HybridAscendCostModel",
    "ResidualBasisContext",
    "ResidualBasisRegistry",
    "ResidualBasisSpec",
    "build_cost_features_from_ledger",
    "evaluate_cost_model_ranking",
    "predict_cannbench_eq3_score",
    "topology_bucket_for_axes",
]
