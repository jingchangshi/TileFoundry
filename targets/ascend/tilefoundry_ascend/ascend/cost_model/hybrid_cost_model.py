"""3-Tier Hybrid Ascend 950 (dav-3510) Cost Model and Online Calibrated Surrogate."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    BishengResourceSummary,
    CompilerBehaviorRegistry,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    extract_ub_occupancy_ratio,
    matches_operator_filter,
    project_case_workload,
    resolve_candidate_for_regime,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    MicroarchProfile,
    merge_microarch_profiles,
)
from tilefoundry_ascend.ascend.cost_model.pipe_schedule_eval import Ascend950PipeScheduleEvaluator
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

CUBE_MACS_PER_CYCLE: dict[str, int] = {
    "f4e2m1": 16384, "f4e1m2": 16384, "fp8e4m3": 8192, "fp8e5m2": 8192, "hif8": 8192,
    "i8": 8192, "u8": 8192, "f16": 4096, "float16": 4096, "bf16": 4096, "bfloat16": 4096,
    "hf32": 2048, "f32": 256, "float32": 256,
}
DEFAULT_CLOCK_GHZ, DEFAULT_L2_CAPACITY_BYTES, DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC = 1.65, 134_217_728, 101
DEFAULT_UB_BANK_MODULUS_BYTES, DEFAULT_AIV_VEC_FLOPS_PER_CYCLE = 512, 128
DEFAULT_LAUNCH_OVERHEAD_US, DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC, DEFAULT_CV_SYNC_EVENT_US = 3.2, 128, 0.08
_PIPE_NAMES = ("CUBE", "VECTOR", "MTE1", "MTE2", "MTE3", "FIXPIPE")
_BISHENG_COPY_FIELDS = (
    "spill_bytes", "stack_bytes", "vloopv2_loops", "software_loops", "has_local_membar_in_vf_call",
    "vf_fusion_breaks", "alias_serialized_buffers", "dual_issue_ipc_collapse", "dcache_dcci_flushes",
)


def topology_bucket_for_axes(axes: TopologyAxes | None) -> str:
    return "global" if axes is None else f"{axes.compute_engine}:{axes.memory_ring}"


def _cube_macs_for_dtype(facts: AscendSpaceFacts, dtype_name: str, *, hf32: bool = False) -> int:
    if hasattr(facts, "cube_macs_per_cycle"):
        try:
            return int(facts.cube_macs_per_cycle(dtype_name, hf32=hf32))
        except Exception:
            pass
    norm = dtype_name.strip().lower()
    return CUBE_MACS_PER_CYCLE.get("hf32" if (hf32 and norm in ("f32", "float32")) else norm, 4096)


def _huber_location(values: Sequence[float], delta: float = 1.35, iters: int = 12) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return vals[0] if vals else 0.0
    mu, eff_delta = float(statistics.median(vals)), max(1e-6, float(delta))
    for _ in range(max(1, int(iters))):
        ws = [1.0 if abs(y - mu) <= eff_delta else eff_delta / abs(y - mu) for y in vals]
        if (w_sum := sum(ws)) <= 0.0:
            break
        if abs((new_mu := sum(w * y for w, y in zip(ws, vals)) / w_sum) - mu) <= 1e-12:
            return new_mu
        mu = new_mu
    return mu


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n, mat = len(b), [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(mat[r][col]))
        mat[col], mat[piv] = mat[piv], mat[col]
        if abs(diag := mat[col][col]) < 1e-12:
            continue
        inv = 1.0 / diag
        for j in range(col, n + 1):
            mat[col][j] *= inv
        for r in range(n):
            if r != col and (fac := mat[r][col]) != 0.0:
                for j in range(col, n + 1):
                    mat[r][j] -= fac * mat[col][j]
    return [mat[i][n] for i in range(n)]


@dataclass(frozen=True)
class CannbenchScorePrediction(JsonRecordMixin):
    """Official CANNBench Eq.3 performance score + Eq.4 / HAP efficiency reward."""

    eq3_score: float
    hap_reward: float
    composite_score: float

    def __getitem__(self, key: str) -> float:
        if key in ("eq3_score", "hap_reward", "composite_score"):
            return float(getattr(self, key))
        raise KeyError(key)

    def __float__(self) -> float:
        return self.eq3_score


def predict_cannbench_eq3_score(pred_us: float, ref_perf_us: float, t_hw_us: float) -> CannbenchScorePrediction:
    if pred_us <= 0.0 or t_hw_us <= 0.0:
        return CannbenchScorePrediction(0.0, 0.0, 0.0)
    base = float(ref_perf_us) if ref_perf_us > 0.0 else max(float(t_hw_us) * 10.0, 10.0)
    denom = (float(pred_us) - float(t_hw_us)) + (base - float(t_hw_us))
    eq3 = max(0.0, (base - float(t_hw_us)) / denom) if denom > 0.0 else 0.0
    hap = min(float(t_hw_us) / float(pred_us), 3.0)
    return CannbenchScorePrediction(eq3_score=eq3, hap_reward=hap, composite_score=0.20 + 0.70 * eq3 + 0.10 * hap)


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
    def coerce(cls, workload: CostFeatures | Mapping[str, object] | object, *, bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None) -> CostFeatures:
        """Build CostFeatures from a dataclass, Mapping, or SIL policy and compiler summary."""
        if isinstance(workload, CostFeatures):
            base = asdict(workload)
        elif isinstance(workload, Mapping):
            base = dict(workload)
        elif hasattr(workload, "to_dict") and callable(getattr(workload, "to_dict")) and isinstance(rd := workload.to_dict(), Mapping):
            base = dict(rd)
        else:
            raise TypeError(f"expected CostFeatures or Mapping, got {type(workload).__name__}")
        if not base.get("cube_dtype") and (dr := base.get("cube_operand_dtype") or base.get("dtype")):
            base["cube_dtype"] = str(dr)
        for dst, src in (("hf32", "hf32_allowed"), ("l2_bypass", "l2_bypass_operands")):
            if dst not in base and src in base:
                base[dst] = bool(base[src])
        if bisheng_res is not None:
            res = bisheng_res if isinstance(bisheng_res, BishengResourceSummary) else BishengResourceSummary.from_dict(bisheng_res)
            for fname in _BISHENG_COPY_FIELDS:
                base[fname] = getattr(res, fname)
            extra = dict(base.get("extra_signals") or {}) if isinstance(base.get("extra_signals"), Mapping) else {}
            extra.update({str(k): float(v) for k, v in res.extra_signals.items()})
            base["extra_signals"] = extra
        elif isinstance(base.get("extra_signals"), Mapping):
            base["extra_signals"] = {str(k): float(v) for k, v in base["extra_signals"].items()}
        filtered = {k: v for k, v in base.items() if k in cls.__dataclass_fields__}
        if isinstance(filtered.get("topology_axes"), Mapping):
            filtered["topology_axes"] = TopologyAxes.from_dict(filtered["topology_axes"])
        rp = filtered.get("microarch_profile")
        filtered["microarch_profile"] = {} if rp is None else (rp.to_dict() if isinstance(rp, MicroarchProfile) else (dict(rp) if isinstance(rp, Mapping) else rp))
        return cls(**filtered)


@dataclass(frozen=True)
class CostPrediction(JsonRecordMixin):
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
            raise ValueError(f"ResidualBasisSpec.min_bucket_samples must be >= 1, got {self.min_bucket_samples}")


def _floor_denom(c: ResidualBasisContext) -> float:
    return max(1e-6, c.cube_floor_us + c.vec_floor_us + c.hbm_floor_us)


DEFAULT_RESIDUAL_BASIS_SPECS: tuple[ResidualBasisSpec, ...] = tuple(
    ResidualBasisSpec(*row)
    for row in (
        ("intercept", False, 2, lambda _c: 1.0),
        ("log_tier0_scale", True, 6, lambda c: math.log1p(max(0.0, c.tier0_us)) / 10.0),
        ("cube_bound_share", True, 6, lambda c: c.cube_floor_us / _floor_denom(c)),
        ("vec_bound_share", True, 6, lambda c: c.vec_floor_us / _floor_denom(c)),
        ("wave_tail_waste", True, 4, lambda c: max(0.0, c.wave_ratio - 1.0)),
        ("l2_miss_ratio", True, 6, lambda c: 1.0 - max(0.0, min(1.0, c.l2_hit_rate))),
        ("short_dma_burst_ind", True, 4, lambda c: 1.0 if c.contig_burst_bytes < 16384 else 0.0),
        ("tier1_penalty_ratio", True, 4, lambda c: min(2.0, max(0.0, c.tier1_penalty_us) / max(1e-3, c.tier0_us))),
        ("launch_overhead_ratio", True, 6, lambda c: min(1.0, max(0.0, c.launch_us) / max(1e-3, c.tier0_us))),
        ("is_coupled_1c2v", False, 8, lambda c: 1.0 if (c.axes is not None and c.axes.compute_engine == "coupled_1c2v") else 0.0),
        ("is_aic_only", False, 8, lambda c: 1.0 if (c.axes is not None and c.axes.compute_engine == "aic_only") else 0.0),
        ("is_irregular_simt", False, 8, lambda c: 1.0 if (c.axes is not None and c.axes.iteration_regularity in ("irregular_simt", "ragged_group", "dynamic_opcode_stream")) else 0.0),
    )
)
_LEGACY_13D_INDEX_TO_NAME: tuple[tuple[int, str], ...] = (
    (0, "intercept"), (1, "log_tier0_scale"), (2, "cube_bound_share"), (3, "vec_bound_share"),
    (5, "wave_tail_waste"), (6, "l2_miss_ratio"), (7, "short_dma_burst_ind"),
    (8, "tier1_penalty_ratio"), (9, "launch_overhead_ratio"),
    (10, "is_coupled_1c2v"), (11, "is_aic_only"), (12, "is_irregular_simt"),
)


def _upgrade_legacy_13d_vector(vec: Sequence[float]) -> dict[str, float]:
    vals = list(map(float, vec)) + [0.0] * max(0, 13 - len(vec))
    return {name: vals[idx] for idx, name in _LEGACY_13D_INDEX_TO_NAME}


class ResidualBasisRegistry:
    """Ordered declarative registry of dimensionless Tier-2 residual basis functions."""

    def __init__(self, specs: Sequence[ResidualBasisSpec] | None = None) -> None:
        self._specs: dict[str, ResidualBasisSpec] = {}
        for spec in (DEFAULT_RESIDUAL_BASIS_SPECS if specs is None else specs):
            self.register(spec)

    def register(self, spec: ResidualBasisSpec, *, overwrite: bool = True) -> ResidualBasisSpec:
        if not overwrite and spec.name in self._specs:
            raise ValueError(f"ResidualBasisSpec already registered: {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def clone(self) -> ResidualBasisRegistry:
        return ResidualBasisRegistry(tuple(self._specs.values()))

    def head0_specs(self) -> tuple[ResidualBasisSpec, ...]:
        return tuple(self._specs.values())

    def head2_specs(self) -> tuple[ResidualBasisSpec, ...]:
        return tuple(s for s in self._specs.values() if s.varies_within_case)

    def spec_names(self, *, head2_only: bool = False) -> tuple[str, ...]:
        return tuple(s.name for s in (self.head2_specs() if head2_only else self.head0_specs()))

    def extract_named(self, ctx: ResidualBasisContext) -> dict[str, float]:
        return {s.name: float(s.extractor(ctx)) for s in self._specs.values()}

    @staticmethod
    def vectorize(named_features: Mapping[str, float], specs: Sequence[ResidualBasisSpec]) -> tuple[float, ...]:
        return tuple(float(named_features.get(s.name, 0.0)) for s in specs)


def _effective_sample_count(n: int, gamma: float) -> float:
    return sum(gamma ** float(age) for age in range(n)) if n > 0 else 0.0


def _sample_gate_threshold(min_samples: int, gamma: float) -> float:
    if min_samples <= 0:
        return 0.0
    return float(min_samples) if gamma >= 1.0 else ((1.0 - gamma ** float(min_samples)) / (1.0 - gamma)) - 1e-9


@dataclass(frozen=True)
class CalibrationSample(JsonRecordMixin):
    """Persisted calibration observation keyed by named dimensionless basis features."""

    named_features: dict[str, float]
    measured_us: float
    analytical_us: float
    topology_bucket: str = "global"
    basis_schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "named_features", {str(k): float(v) for k, v in self.named_features.items()})

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return ResidualBasisRegistry.vectorize(self.named_features, DEFAULT_RESIDUAL_BASIS_SPECS)

    def to_dict(self) -> dict[str, object]:
        out = super().to_dict()
        out["feature_vector"] = list(self.feature_vector)
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CalibrationSample:
        rn, rv = payload.get("named_features"), payload.get("basis_schema_version")
        if isinstance(rn, Mapping) and rn:
            named, ver = {str(k): float(v) for k, v in rn.items()}, (int(rv) if isinstance(rv, (int, float)) else 2)
        else:
            vec = [float(x) for x in (payload.get("feature_vector") or ())] if isinstance(payload.get("feature_vector"), Sequence) else []
            named, ver = ({s.name: vec[i] for i, s in enumerate(DEFAULT_RESIDUAL_BASIS_SPECS)} if (len(vec) == len(DEFAULT_RESIDUAL_BASIS_SPECS) and rv == 2) else _upgrade_legacy_13d_vector(vec)), 2
        return cls(named, float(payload["measured_us"]), float(payload["analytical_us"]), str(payload.get("topology_bucket") or "global"), ver)


@dataclass(frozen=True)
class _Tier01Result:
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
    named_features: dict[str, float]


@dataclass(frozen=True)
class CostModelEvalReport(JsonRecordMixin):
    """Empirical accuracy and intra-operator ranking evaluation summary."""

    sample_count: int
    mape: float
    median_ape: float
    overall_spearman_rho: float
    mean_intra_group_spearman_rho: float
    pairwise_accuracy: float
    top1_regret_ratio: float


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed, ranks, i, n = sorted(enumerate(values), key=lambda item: float(item[1])), [0.0] * len(values), 0, len(values)
    while i < n:
        j = i + 1
        while j < n and abs(float(indexed[j][1]) - float(indexed[i][1])) <= 1e-12:
            j += 1
        for k in range(i, j):
            ranks[indexed[k][0]] = 0.5 * float(i + j - 1) + 1.0
        i = j
    return ranks


def _spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if (n := len(xs)) < 2 or len(ys) != n:
        return 0.0
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = sum(rx) / float(n), sum(ry) / float(n)
    dx, dy = math.sqrt(sum((a - mx) ** 2 for a in rx)), math.sqrt(sum((b - my) ** 2 for b in ry))
    return max(-1.0, min(1.0, sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (dx * dy))) if (dx > 1e-12 and dy > 1e-12) else 0.0


def evaluate_cost_model_ranking(predictions: Sequence[float], actuals: Sequence[float], *, groups: Sequence[str] | None = None) -> CostModelEvalReport:
    """Compute MAPE, Spearman rank correlation, pairwise accuracy, and Top-1 regret."""
    if (n := len(predictions)) == 0 or len(actuals) != n:
        return CostModelEvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if groups is not None and len(groups) != n:
        raise ValueError("groups length must match predictions and actuals length")
    apes = sorted(abs(float(p) - float(a)) / max(1e-6, abs(float(a))) for p, a in zip(predictions, actuals))
    med_ape, ov_rho = (apes[n // 2] if (n % 2 == 1) else 0.5 * (apes[n // 2 - 1] + apes[n // 2])), _spearman_corr(predictions, actuals)
    buckets: dict[str, list[int]] = {}
    for idx in range(n):
        buckets.setdefault(str(groups[idx]) if groups is not None else "global", []).append(idx)
    g_rhos, regrets, p_ok, p_tot = [], [], 0.0, 0
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        gp, ga = [float(predictions[i]) for i in idxs], [float(actuals[i]) for i in idxs]
        g_rhos.append(_spearman_corr(gp, ga))
        if (omin := min(ga)) > 0.0:
            regrets.append((ga[min(range(len(idxs)), key=lambda pos: gp[pos])] - omin) / omin)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if abs(ga[i] - ga[j]) > 1e-9:
                    p_tot += 1
                    p_ok += 0.5 if abs(gp[i] - gp[j]) <= 1e-12 else float((gp[i] < gp[j]) == (ga[i] < ga[j]))
    return CostModelEvalReport(n, sum(apes) / float(n), med_ape, ov_rho, sum(g_rhos) / float(len(g_rhos)) if g_rhos else ov_rho, (p_ok / float(p_tot)) if p_tot > 0 else 0.0, (sum(regrets) / float(len(regrets))) if regrets else 0.0)


class HybridAscendCostModel:
    """3-Tier Hybrid Ascend 950 Cost Model (Roofline Floor, Compiler Penalties, Ridge)."""

    def __init__(
        self, target: AscendTarget | None = None, *, clock_ghz: float = DEFAULT_CLOCK_GHZ, ridge_alpha: float = 1.0,
        forgetting_factor: float = 0.985, calibration_path: str | Path | None = None,
        behavior_registry: CompilerBehaviorRegistry | None = None, basis_registry: ResidualBasisRegistry | None = None,
        anomaly_lake: ExperienceLake | str | Path | None = None,
    ) -> None:
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError(f"forgetting_factor must be in (0.0, 1.0], got {forgetting_factor!r}")
        self.target = target or AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)
        self.clock_ghz, self.ridge_alpha, self._forgetting_factor = float(getattr(self.facts, "clock_ghz", clock_ghz)), float(max(1e-6, ridge_alpha)), float(forgetting_factor)
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.anomaly_lake, self.anomaly_path = self._resolve_anomaly_target(anomaly_lake)
        self.behavior_registry = behavior_registry or DEFAULT_COMPILER_BEHAVIOR_REGISTRY
        self._basis_registry = basis_registry or ResidualBasisRegistry()
        self._pipe_evaluator = Ascend950PipeScheduleEvaluator(self.target)
        self._reset_calibration_state()
        if self.calibration_path is not None and self.calibration_path.exists():
            self.load_calibration(self.calibration_path)

    def _resolve_anomaly_target(self, anomaly_lake: ExperienceLake | str | Path | None) -> tuple[ExperienceLake | None, Path | None]:
        if isinstance(anomaly_lake, ExperienceLake):
            return anomaly_lake, anomaly_lake.root_dir / "unmodeled_anomalies.jsonl"
        if anomaly_lake is not None:
            raw = Path(anomaly_lake).expanduser().resolve()
            return (None, raw) if raw.suffix == ".jsonl" else (lake := ExperienceLake(raw), lake.root_dir / "unmodeled_anomalies.jsonl")
        return (None, self.calibration_path.parent / "unmodeled_anomalies.jsonl") if self.calibration_path is not None else (None, None)

    def _reset_calibration_state(self) -> None:
        self._samples, self._pair_deltas = [], []
        self._weights_by_name = {s.name: 0.0 for s in self._basis_registry.head0_specs()}
        self._delta_weights_by_name = {s.name: 0.0 for s in self._basis_registry.head2_specs()}
        self._bucket_weights_by_name, self._bucket_intercepts, self._bucket_sample_counts = {}, {}, {}
        self._global_intercept, self._msprof_event_count = 0.0, 0
        self._quarantined_sample_ids, self._replayed_obs_keys, self._replayed_pair_keys = set(), set(), set()
        self._active_global_specs, self._active_bucket_specs, self._active_head2_specs = (), {}, ()
        self._msprof_pipe_scales = {}

    @property
    def basis_registry(self) -> ResidualBasisRegistry: return self._basis_registry
    @property
    def sample_count(self) -> int: return len(self._samples)
    @property
    def pair_delta_count(self) -> int: return len(self._pair_deltas)
    @property
    def msprof_event_count(self) -> int: return self._msprof_event_count
    @property
    def delta_weights_by_name(self) -> dict[str, float]: return {s.name: self._delta_weights_by_name.get(s.name, 0.0) for s in self._basis_registry.head2_specs()}
    @property
    def delta_weights(self) -> tuple[float, ...]: return tuple(self._delta_weights_by_name.get(s.name, 0.0) for s in self._basis_registry.head2_specs())
    @property
    def weights_by_name(self) -> dict[str, float]:
        out = {s.name: self._weights_by_name.get(s.name, 0.0) for s in self._basis_registry.head0_specs()}
        if "intercept" in out and "intercept" in self._active_global_specs:
            out["intercept"] += self._global_intercept
        return out
    @property
    def weights(self) -> tuple[float, ...]: return tuple(self.weights_by_name.get(s.name, 0.0) for s in self._basis_registry.head0_specs())
    @property
    def bucket_weights_by_name(self) -> dict[str, dict[str, float]]:
        specs, out = self._basis_registry.head0_specs(), {}
        for bkt, wmap in self._bucket_weights_by_name.items():
            bo = {s.name: wmap.get(s.name, 0.0) for s in specs}
            if "intercept" in bo and "intercept" in self._active_bucket_specs.get(bkt, ()):
                bo["intercept"] += self._bucket_intercepts.get(bkt, self._global_intercept)
            out[bkt] = bo
        return out
    @property
    def bucket_weights(self) -> dict[str, tuple[float, ...]]: return {b: tuple(wm.get(s.name, 0.0) for s in self._basis_registry.head0_specs()) for b, wm in self.bucket_weights_by_name.items()}
    def active_head0_spec_names(self, bucket: str = "global") -> tuple[str, ...]: return self._active_global_specs if bucket == "global" else self._active_bucket_specs.get(bucket, ())
    def active_head2_spec_names(self) -> tuple[str, ...]: return self._active_head2_specs

    def pipe_scale_multipliers(self, bucket: str = "global") -> dict[str, float]:
        b_st, g_st, out = self._msprof_pipe_scales.get(bucket) or {}, self._msprof_pipe_scales.get("global") or {}, {}
        for pipe in _PIPE_NAMES:
            s_log, s_w = b_st.get(pipe, (0.0, 0.0))
            if s_w <= 0.0:
                s_log, s_w = g_st.get(pipe, (0.0, 0.0))
            out[pipe] = max(0.5, min(2.0, math.exp(s_log / (10.0 + s_w)) if s_w > 0.0 else 1.0))
        return out

    def _compute_tier0_and_tier1(self, cf: CostFeatures, *, use_pipe_scales: bool = True) -> _Tier01Result:
        aic_c, aiv_c = max(1, cf.active_aic_cores or self.facts.num_aic_cores), max(1, cf.active_aiv_cores or self.facts.num_aiv_cores)
        clk_hz, bkt = self.clock_ghz * 1e9, topology_bucket_for_axes(cf.topology_axes)
        scales = self.pipe_scale_multipliers(bkt) if use_pipe_scales else dict.fromkeys(_PIPE_NAMES, 1.0)
        macs_cyc = _cube_macs_for_dtype(self.facts, cf.cube_dtype, hf32=cf.hf32)
        wave_r = max(1.0, math.ceil(cf.total_tiles / max(1, aic_c if cf.cube_flops > 0 else aiv_c)) / max(1e-6, cf.total_tiles / float(max(1, aic_c if cf.cube_flops > 0 else aiv_c)))) if cf.total_tiles > 0 else 1.0
        cube_us = ((cf.cube_flops / (float(aic_c) * macs_cyc * 2.0 * clk_hz)) * 1e6 * wave_r) if cf.cube_flops > 0 else 0.0
        prof = MicroarchProfile.coerce(cf.microarch_profile) if cf.microarch_profile else MicroarchProfile()
        has_bb = (cf.cube_flops > 0.0 or prof.cube_mmad_count > 0 or prof.cube_macs_total > 0 or prof.total_vector_ops > 0
                  or (cf.topology_axes is not None and cf.topology_axes.iteration_regularity == "irregular_simt")
                  or any(float(cf.extra_signals.get(k, 0.0) or 0.0) > 0.0 for k in ("has_simtvf", "has_simd"))
                  or any(float(cf.extra_signals.get(k, 0.0) or 0.0) >= 0.08 for k in ("measured_aiv_vec_ratio", "l2_aiv_vec_ratio")))
        vec_fpc = 1.0 if (float(cf.extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0) > 0.0 and not has_bb) else float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE)
        vec_us = ((cf.vec_flops / (float(aiv_c) * vec_fpc * clk_hz)) * 1e6) * (wave_r if (cf.cube_flops <= 0 and cf.total_tiles > 0) else 1.0) if cf.vec_flops > 0 else 0.0

        eff_bw = max(1.0, self.facts.effective_hbm_bandwidth_bps(max(1, int(cf.contig_burst_bytes))))
        l2_cap, l2_hit = int(getattr(self.facts, "l2_capacity_bytes", DEFAULT_L2_CAPACITY_BYTES)), 0.0
        if not cf.l2_bypass and cf.gm_bytes > 0:
            if cf.l2_reuse_fraction > 0.0:
                l2_hit = min(0.85, max(0.0, cf.l2_reuse_fraction))
            elif cf.swizzle_window >= 4:
                l2_hit = round(0.40 * min(1.0, float(l2_cap) / float(max(1, cf.l2_working_set_bytes or int(cf.gm_bytes * 0.25)))), 6)
        eff_gm = float(cf.gm_bytes) * (1.0 - l2_hit) * (0.5 + 0.5 * damp if (damp := float(cf.extra_signals.get("dcache_bypass_narrow_store", 0.0) or 0.0)) > 1.0 else 1.0)
        hbm_us = ((eff_gm / eff_bw) * 1e6 if eff_gm > 0 else 0.0) * scales["MTE2"]
        l2_us = (((float(cf.gm_bytes) * l2_hit) / max(1.0, float(aic_c) * float(getattr(self.facts, "l2_bytes_per_cycle_per_aic", DEFAULT_L2_BYTES_PER_CYCLE_PER_AIC)) * clk_hz)) * 1e6) if l2_hit > 0 else 0.0
        fix_bpc = float(getattr(self.facts, "fixpipe_bytes_per_cycle", DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC) or DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC)
        fix_us = (((float(cf.fixpipe_bytes) / max(1.0, float(aic_c) * fix_bpc * clk_hz)) * 1e6) if cf.fixpipe_bytes > 0.0 else 0.0) * scales["FIXPIPE"]
        cube_us, vec_us = cube_us * scales["CUBE"], vec_us * scales["VECTOR"]
        is_coupled = cf.topology_axes is not None and cf.topology_axes.compute_engine == "coupled_1c2v"
        cv_us = (float(cf.cv_sync_events) * DEFAULT_CV_SYNC_EVENT_US) if cf.cv_sync_events > 0 else (0.08 * min(cube_us, vec_us) if (is_coupled and cube_us > 0.0 and vec_us > 0.0) else 0.0)
        has_work = cf.cube_flops > 0.0 or cf.vec_flops > 0.0 or cf.gm_bytes > 0.0 or cf.fixpipe_bytes > 0.0 or not prof.is_empty
        eff_ln = max(cf.kernel_launches, prof.subkernel_count) if (cf.kernel_launches > 0 and not prof.is_empty) else cf.kernel_launches
        launch_us = (float(eff_ln) * max(0.0, float(cf.launch_overhead_us) if cf.launch_overhead_us is not None else DEFAULT_LAUNCH_OVERHEAD_US)) if (has_work and eff_ln > 0) else 0.0
        ub_c = max(0.0, min(1.0, cf.ub_bank_conflict_ratio))
        if ub_c == 0.0 and cf.ub_row_stride_bytes > 0 and cf.ub_row_stride_bytes % DEFAULT_UB_BANK_MODULUS_BYTES == 0 and cf.ub_bank_pad_datablocks == 0 and cf.ub_rw_shift_datablocks == 0:
            ub_c = 0.35
        bank_us = vec_us * ub_c + cube_us * (0.25 if (cf.l1_stages >= 2 and not cf.l1_half_bank_split) else 0.0)

        if not prof.is_empty:
            ep = replace(prof, pipeline_depth=max(prof.pipeline_depth, cf.pipeline_depth), inner_loop_trips=max(prof.inner_loop_trips, cf.loop_trips), subkernel_count=max(prof.subkernel_count, eff_ln))
            beff = max(0.10, min(1.0, eff_bw / max(1.0, self.facts.effective_hbm_bandwidth_bps(65536))))
            sched = self._pipe_evaluator.evaluate_profile(ep, cube_flops=cf.cube_flops, vec_flops=cf.vec_flops, cube_dtype=cf.cube_dtype, hf32=cf.hf32, total_tiles=cf.total_tiles, active_aic_cores=cf.active_aic_cores, active_aiv_cores=cf.active_aiv_cores, dma_burst_efficiency=beff, is_coupled_1c2v=is_coupled, ledger_steady_state_ii=cf.steady_state_ii, pipe_scale_multipliers=scales, profile_is_per_core_full_loop=cf.profile_is_per_core_full_loop)
            cube_us, vec_us, fix_us, hbm_us = max(cube_us, sched.pipe_ns["CUBE"] * 1e-3), max(vec_us, sched.pipe_ns["VECTOR"] * 1e-3), max(fix_us, sched.pipe_ns["FIXPIPE"] * 1e-3), max(hbm_us, (sched.pipe_ns["MTE2"] * 1e-3) * (1.0 - l2_hit))
            if prof.cv_sync_barriers > 0:
                cv_us = max(cv_us, float(ep.cv_sync_barriers) * DEFAULT_CV_SYNC_EVENT_US)
            extra_b = 0.0 if (prof.ub_bank_conflict_factor > 1.0 or prof.l1_bank_conflict_factor > 1.0) else bank_us
            if prof.ub_bank_conflict_factor > 1.0:
                bank_us = max(bank_us, vec_us * (1.0 - 1.0 / max(1.0, prof.ub_bank_conflict_factor)))
            comp_us = max(cube_us, vec_us, fix_us, sched.pipe_ns["MTE1"] * 1e-3, sched.pipe_ns["MTE3"] * 1e-3)
            exec_us = max(max(comp_us, hbm_us, l2_us) + 0.15 * min(comp_us, hbm_us) + extra_b + cv_us, (sched.loop_total_ns * 1e-3) + max(cv_us, max(0.0, (sched.total_ns - sched.loop_total_ns) * 1e-3)) + extra_b)
        else:
            comp_us = max(cube_us, vec_us, fix_us)
            exec_us = max(comp_us, hbm_us, l2_us) + 0.15 * min(comp_us, hbm_us) + bank_us + cv_us

        tier0_us = launch_us + exec_us
        base_ref = max(1.0, tier0_us)
        tier1_us, _ = self.behavior_registry.compute_tier1_penalties(
            BishengResourceSummary(**{k: getattr(cf, k) for k in _BISHENG_COPY_FIELDS}, extra_signals=dict(cf.extra_signals)),
            context={"base_ref": base_ref, "tier0_us": tier0_us, "cube_floor_us": cube_us, "vec_floor_us": vec_us, "hbm_floor_us": hbm_us, "cost_features": cf},
        )
        if 0 < cf.contig_burst_bytes < 16384:
            sh = (16384.0 - float(cf.contig_burst_bytes)) / 16384.0
            tier1_us += hbm_us * 0.15 * sh + 0.5 * sh
        if cf.has_unaligned_tail:
            tier1_us += max(0.25, 0.03 * base_ref)
        dom = (
            "compiler_spill" if (cf.spill_bytes > 0 and tier1_us >= base_ref * 0.3) else
            "bank_conflict" if bank_us > 0.3 * max(1e-6, tier0_us) else
            "launch_overhead" if (launch_us > exec_us and launch_us > 0.5 * max(1e-6, tier0_us)) else
            "fixpipe" if (fix_us > 0.0 and fix_us >= max(cube_us, vec_us, hbm_us)) else
            "cv_sync" if cv_us > 0.25 * max(1e-6, tier0_us) else
            "hbm" if hbm_us >= max(cube_us, vec_us) else ("cube" if cube_us >= vec_us else "vector")
        )
        ctx = ResidualBasisContext(tier0_us, cube_us, vec_us, hbm_us, tier1_us, launch_us, wave_r, l2_hit, int(cf.contig_burst_bytes), cf.topology_axes)
        return _Tier01Result(tier0_us, tier1_us, cube_us, vec_us, hbm_us, l2_hit, wave_r, bank_us, eff_bw, launch_us, fix_us, cv_us, dom, self._basis_registry.extract_named(ctx))

    def predict(self, workload: CostFeatures | Mapping[str, object], *, bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None) -> CostPrediction:
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        t01, bkt, specs = self._compute_tier0_and_tier1(cf), topology_bucket_for_axes(cf.topology_axes), self._basis_registry.head0_specs()
        wmap, inter = self._bucket_weights_by_name.get(bkt, self._weights_by_name), self._bucket_intercepts.get(bkt, self._global_intercept)
        res_fac = math.exp(max(-6.0, min(6.0, inter + sum(wmap.get(s.name, 0.0) * t01.named_features.get(s.name, 0.0) for s in specs))))
        pred_us = max(1e-6, t01.tier0_us + t01.tier1_penalty_us) * res_fac if (t01.tier0_us > 0.0 or t01.tier1_penalty_us > 0.0) else 0.0
        return CostPrediction(pred_us, t01.tier0_us, t01.tier1_penalty_us, res_fac, t01.cube_floor_us, t01.vec_floor_us, t01.hbm_floor_us, t01.l2_hit_rate, t01.wave_ratio, t01.bank_conflict_us, t01.eff_hbm_bw, t01.dominant, self._basis_registry.vectorize(t01.named_features, specs), t01.launch_us, t01.fixpipe_floor_us, t01.cv_sync_us, bkt, dict(t01.named_features))

    def _write_anomaly_record(self, rec: dict[str, object]) -> None:
        if self.anomaly_lake is not None:
            self.anomaly_lake.append_unmodeled_anomaly(rec)
        elif self.anomaly_path is not None:
            self.anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            with self.anomaly_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    def observe(self, workload: CostFeatures | Mapping[str, object], measured_us: float, *, bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None, persist: bool = True, refit: bool = True) -> float:
        if measured_us <= 0.0:
            raise ValueError(f"measured_us must be positive, got {measured_us!r}")
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        t01, bkt = self._compute_tier0_and_tier1(cf), topology_bucket_for_axes(cf.topology_axes)
        a_us = max(1e-6, t01.tier0_us + t01.tier1_penalty_us)
        lr = math.log(float(measured_us) / a_us)
        if not math.isfinite(lr) or abs(lr) > 8.0 or (bkt in self._bucket_intercepts and self._bucket_sample_counts.get(bkt, 0) >= 3 and abs(lr - self._bucket_intercepts[bkt]) > 2.0):
            self._write_anomaly_record({"measured_us": float(measured_us), "analytical_us": a_us, "log_residual": lr, "topology_bucket": bkt, "dominant_bound": t01.dominant, "extra_signals": {str(k): float(v) for k, v in cf.extra_signals.items()}})
            return a_us
        self._samples.append(CalibrationSample(dict(t01.named_features), float(measured_us), a_us, bkt, 2))
        if refit:
            self._refit_ridge()
        if persist and self.calibration_path is not None:
            self.save_calibration(self.calibration_path)
        return a_us

    def observe_batch(self, items: Sequence[tuple[object, ...]], *, persist: bool = True) -> list[float]:
        out = []
        for entry in items:
            if len(entry) not in (2, 3):
                raise ValueError(f"Expected 2- or 3-tuple in observe_batch, got len={len(entry)}")
            out.append(self.observe(entry[0], entry[1], bisheng_res=(entry[2] if len(entry) == 3 else None), persist=False, refit=False))
        if out:
            self._refit_ridge()
            if persist and self.calibration_path is not None:
                self.save_calibration(self.calibration_path)
        return out

    def _solve_gated_ridge(self, rows: Sequence[tuple[Mapping[str, float], float]], specs: Sequence[ResidualBasisSpec], *, alpha: float, prior_weights: Mapping[str, float] | None = None, y_cap: float = 6.0) -> tuple[dict[str, float], tuple[str, ...]]:
        weights = {s.name: 0.0 for s in specs}
        if not rows or not (active := tuple(s for s in specs if _effective_sample_count(len(rows), self._forgetting_factor) >= _sample_gate_threshold(s.min_bucket_samples, self._forgetting_factor))):
            return weights, ()
        k = len(active)
        xtx = [[(1e-4 if (prior_weights is None and active[i].name == "intercept") else alpha) if i == j else 0.0 for j in range(k)] for i in range(k)]
        xty = [(alpha * float(prior_weights.get(s.name, 0.0))) if prior_weights is not None else 0.0 for s in active]
        for idx, (nx, yr) in enumerate(rows):
            y, w, xa = max(-y_cap, min(y_cap, float(yr))), self._forgetting_factor ** float(len(rows) - 1 - idx), [float(nx.get(s.name, 0.0)) for s in active]
            for i in range(k):
                xty[i] += w * xa[i] * y
                for j in range(k):
                    xtx[i][j] += w * xa[i] * xa[j]
        for s, val in zip(active, _solve_linear_system(xtx, xty)):
            weights[s.name] = float(val)
        return weights, tuple(s.name for s in active)

    def _refit_ridge(self) -> None:
        head0_specs = self._basis_registry.head0_specs()
        if not self._samples:
            self._weights_by_name = {s.name: 0.0 for s in head0_specs}
            self._bucket_weights_by_name, self._global_intercept, self._bucket_intercepts, self._bucket_sample_counts, self._active_global_specs, self._active_bucket_specs = {}, 0.0, {}, {}, (), {}
            return
        raw_entries = [(s, (s.topology_bucket if s.topology_bucket and s.topology_bucket != "global" else "global"), math.log(max(1e-6, s.measured_us) / max(1e-6, s.analytical_us))) for s in self._samples]
        raw_ys, raw_by_bkt = [y for _, _, y in raw_entries], {}
        for _, bk, y in raw_entries:
            if bk != "global":
                raw_by_bkt.setdefault(bk, []).append(y)
        int_spec = next((s for s in head0_specs if s.name == "intercept"), None)
        int_th, ff = (_sample_gate_threshold(int_spec.min_bucket_samples, self._forgetting_factor) if int_spec else 2.0), self._forgetting_factor
        raw_gi, g_unl = _huber_location(raw_ys), bool(int_spec and _effective_sample_count(len(raw_ys), ff) >= int_th)
        self._global_intercept = raw_gi if g_unl else 0.0
        self._bucket_intercepts = {bk: _huber_location(ys) if (len(ys) >= 2 and _effective_sample_count(len(ys), ff) >= int_th) else self._global_intercept for bk, ys in raw_by_bkt.items() if (len(ys) >= 2 and _effective_sample_count(len(ys), ff) >= int_th) or g_unl}
        g_rows, by_bkt = [], {}
        for smp, bk, y in raw_entries:
            cy = y - (self._bucket_intercepts.get(bk, raw_gi) if bk != "global" else raw_gi)
            if abs(cy) > 2.0:
                if id(smp) not in self._quarantined_sample_ids:
                    self._quarantined_sample_ids.add(id(smp))
                    self._write_anomaly_record({"measured_us": float(smp.measured_us), "analytical_us": float(smp.analytical_us), "log_residual": float(y), "centered_residual": float(cy), "topology_bucket": bk})
                continue
            g_rows.append((smp.named_features, cy))
            if bk != "global":
                by_bkt.setdefault(bk, []).append((smp.named_features, cy))
        c_specs = tuple(s for s in head0_specs if s.name != "intercept")
        self._bucket_sample_counts = {bk: len(rows) for bk, rows in by_bkt.items()}
        clean_unl = bool(int_spec and _effective_sample_count(len(g_rows), ff) >= int_th)
        if not clean_unl:
            self._global_intercept = 0.0
        self._weights_by_name, g_sl = self._solve_gated_ridge(g_rows, c_specs, alpha=self.ridge_alpha)
        self._weights_by_name.setdefault("intercept", 0.0)
        self._active_global_specs = ("intercept", *g_sl) if clean_unl else g_sl
        new_bw, new_abs = {}, {}
        for bk, b_rows in by_bkt.items():
            bw, b_sl = self._solve_gated_ridge(b_rows, c_specs, alpha=max(2.0, self.ridge_alpha * 2.0), prior_weights=self._weights_by_name)
            if (b_act := ("intercept", *b_sl) if bool(int_spec and _effective_sample_count(len(b_rows), ff) >= int_th) else b_sl):
                bw.setdefault("intercept", 0.0)
                new_bw[bk], new_abs[bk] = bw, b_act
        self._bucket_weights_by_name, self._active_bucket_specs = new_bw, new_abs

    def save_calibration(self, path: str | Path) -> None:
        target = Path(path)
        if target.suffix == ".jsonl":
            lines = [json.dumps(s.to_dict(), sort_keys=True) for s in self._samples]
            atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))
            return
        h0, h2 = self._basis_registry.head0_specs(), self._basis_registry.head2_specs()
        payload = {
            "basis_schema_version": 2, "head0_spec_names": [s.name for s in h0], "head2_spec_names": [s.name for s in h2],
            "samples": [s.to_dict() for s in self._samples], "weights": list(self.weights), "weights_by_name": self.weights_by_name,
            "global_intercept": float(self._global_intercept), "bucket_intercepts": {str(k): float(v) for k, v in self._bucket_intercepts.items()},
            "delta_weights": list(self.delta_weights), "delta_weights_by_name": self.delta_weights_by_name,
            "pair_deltas": [{"named_delta": dict(ndx), "delta_x": list(self._basis_registry.vectorize(ndx, h2)), "delta_y": float(dy)} for ndx, dy in self._pair_deltas],
            "msprof_pipe_scales": {b: {p: [float(sl), float(sw)] for p, (sl, sw) in pm.items()} for b, pm in self._msprof_pipe_scales.items()},
            "msprof_event_count": int(self._msprof_event_count),
        }
        atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True))

    def load_calibration(self, path: str | Path) -> int:
        if not (src := Path(path)).exists():
            return 0
        if not (raw := src.read_text(encoding="utf-8").strip()):
            self._reset_calibration_state()
            return 0
        loaded_gi, loaded_bi, h2 = None, None, self._basis_registry.head2_specs()
        if src.suffix == ".jsonl" or not raw.startswith("{"):
            samples = [CalibrationSample.from_dict(json.loads(ln)) for ln in raw.splitlines() if ln.strip()]
        else:
            pl = json.loads(raw)
            samples = [CalibrationSample.from_dict(it) for it in (pl.get("samples") or ())]
            loaded_gi = float(pl["global_intercept"]) if pl.get("global_intercept") is not None else None
            loaded_bi = {str(k): float(v) for k, v in pl["bucket_intercepts"].items()} if isinstance(pl.get("bucket_intercepts"), Mapping) else None
            if isinstance(pl.get("msprof_pipe_scales"), Mapping):
                self._msprof_pipe_scales = {str(b): {str(p): (float(v[0]), float(v[1])) for p, v in pm.items() if isinstance(v, (list, tuple)) and len(v) == 2} for b, pm in pl["msprof_pipe_scales"].items() if isinstance(pm, Mapping)}
            self._msprof_event_count = int(pl.get("msprof_event_count", 0) or 0)
            self._pair_deltas = []
            for it in (pl.get("pair_deltas") if isinstance(pl.get("pair_deltas"), list) else ()):
                if isinstance(it, Mapping):
                    nd = it.get("named_delta")
                    dx = [float(v) for v in (it.get("delta_x") or ())] if isinstance(it.get("delta_x"), Sequence) else []
                    ndx = {str(k): float(v) for k, v in nd.items()} if isinstance(nd, Mapping) and nd else ({s.name: dx[i] for i, s in enumerate(h2)} if len(dx) == len(h2) else {s.name: _upgrade_legacy_13d_vector(dx).get(s.name, 0.0) for s in h2})
                    self._pair_deltas.append((ndx, float(it.get("delta_y", 0.0))))
        self._samples, self._quarantined_sample_ids = samples, set()
        self._refit_ridge()
        if not self._samples:
            self._global_intercept = loaded_gi if loaded_gi is not None else self._global_intercept
            self._bucket_intercepts = loaded_bi if loaded_bi is not None else self._bucket_intercepts
        self._refit_delta_ridge()
        return len(self._samples)

    def observe_msprof(
        self, workload: CostFeatures | Mapping[str, object], msprof_pmu: Mapping[str, object], *,
        measured_us: float | None = None, bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None, persist: bool = False,
    ) -> dict[str, float]:
        cf = CostFeatures.coerce(workload, bisheng_res=bisheng_res)
        bkt = topology_bucket_for_axes(cf.topology_axes)
        if not isinstance(msprof_pmu, Mapping) or not msprof_pmu:
            return self.pipe_scale_multipliers(bkt)
        try:
            dur_us = float((measured_us if (measured_us is not None and float(measured_us) > 0.0) else msprof_pmu.get("dur_us", 0.0)) or 0.0)
        except Exception:
            dur_us = 0.0
        if dur_us <= 0.0:
            return self.pipe_scale_multipliers(bkt)
        t01 = self._compute_tier0_and_tier1(cf, use_pipe_scales=False)
        _pmu = lambda *ks: max((min(1.0, float(v) / 100.0 if float(v) > 1.0 else float(v)) for k in ks if isinstance(v := msprof_pmu.get(k), (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0.0), default=0.0)
        max_log, updated = math.log(2.0), False
        for pipe, ratio, a_us in (("CUBE", _pmu("aic_mad", "aic_mac_ratio"), t01.cube_floor_us), ("VECTOR", _pmu("aiv_vec", "aiv_vec_ratio"), t01.vec_floor_us), ("FIXPIPE", _pmu("aic_fixpipe"), t01.fixpipe_floor_us), ("MTE2", _pmu("aiv_mte2", "aic_mte2", "mte2_wait_ratio"), t01.hbm_floor_us)):
            if ratio > 0.0 and a_us > 1e-3:
                lr, w = max(-max_log, min(max_log, math.log(max(1e-6, dur_us * ratio) / max(1e-6, a_us)))), min(1.0, max(0.05, ratio))
                for tb in ([bkt] if bkt == "global" else [bkt, "global"]):
                    bm = self._msprof_pipe_scales.setdefault(tb, {})
                    pl, pw = bm.get(pipe, (0.0, 0.0))
                    bm[pipe] = (pl + w * lr, pw + w)
                updated = True
        if updated:
            self._msprof_event_count += 1
            if persist and self.calibration_path is not None and self.calibration_path.suffix != ".jsonl":
                self.save_calibration(self.calibration_path)
        return self.pipe_scale_multipliers(bkt)

    def _refit_delta_ridge(self) -> None:
        self._delta_weights_by_name, self._active_head2_specs = self._solve_gated_ridge(self._pair_deltas, self._basis_registry.head2_specs(), alpha=max(1.0, self.ridge_alpha), y_cap=2.5)

    def observe_pair_delta(
        self, cand_workload: CostFeatures | Mapping[str, object], cand_us: float, parent_workload: CostFeatures | Mapping[str, object], parent_us: float, *,
        cand_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None, parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None, refit: bool = True,
    ) -> float:
        if cand_us <= 0.0 or parent_us <= 0.0:
            raise ValueError(f"Both cand_us and parent_us must be positive, got {cand_us!r}, {parent_us!r}")
        t01_c, t01_p = self._compute_tier0_and_tier1(CostFeatures.coerce(cand_workload, bisheng_res=cand_bisheng_res)), self._compute_tier0_and_tier1(CostFeatures.coerce(parent_workload, bisheng_res=parent_bisheng_res))
        bc, bp = max(1e-6, t01_c.tier0_us + t01_c.tier1_penalty_us), max(1e-6, t01_p.tier0_us + t01_p.tier1_penalty_us)
        dy = max(-2.5, min(2.5, math.log(float(cand_us) / float(parent_us)) - math.log(bc / bp)))
        if abs(math.log(float(cand_us) / bc)) > 2.5 or abs(math.log(float(parent_us) / bp)) > 2.5:
            return dy
        self._pair_deltas.append(({s.name: t01_c.named_features.get(s.name, 0.0) - t01_p.named_features.get(s.name, 0.0) for s in self._basis_registry.head2_specs()}, dy))
        if refit:
            self._refit_delta_ridge()
        return dy

    def predict_delta_ratio(
        self, cand_workload: CostFeatures | Mapping[str, object], parent_workload: CostFeatures | Mapping[str, object], *,
        cand_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None, parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
    ) -> float:
        cf_c, cf_p = CostFeatures.coerce(cand_workload, bisheng_res=cand_bisheng_res), CostFeatures.coerce(parent_workload, bisheng_res=parent_bisheng_res)
        if not self._pair_deltas:
            return max(1e-6, self.predict(cf_c).pred_us) / max(1e-6, self.predict(cf_p).pred_us)
        t01_c, t01_p = self._compute_tier0_and_tier1(cf_c), self._compute_tier0_and_tier1(cf_p)
        bc, bp = max(1e-6, t01_c.tier0_us + t01_c.tier1_penalty_us), max(1e-6, t01_p.tier0_us + t01_p.tier1_penalty_us)
        lr = sum(self._delta_weights_by_name.get(s.name, 0.0) * (t01_c.named_features.get(s.name, 0.0) - t01_p.named_features.get(s.name, 0.0)) for s in self._basis_registry.head2_specs())
        return max(1e-4, (bc / bp) * math.exp(max(-2.5, min(2.5, lr))))

    def mark_online_observation_key(self, *, cand_sha: str, operator: str, regime_tag: str, case_num: int, eval_tier: str, measured_us: float) -> None:
        self._replayed_obs_keys.add((str(cand_sha), str(operator), str(regime_tag), int(case_num), str(eval_tier), round(float(measured_us), 6)))

    def mark_online_pair_key(self, *, cand_sha: str, parent_sha: str, operator: str, regime_tag: str, case_num: int) -> None:
        self._replayed_pair_keys.add((str(cand_sha), str(parent_sha), str(operator), str(regime_tag), int(case_num)))

    def replay_experience_lake(self, lake_dir: ExperienceLake | str | Path, *, persist_calibration: bool = False, exclude_run_id: str | None = None, operator_filter: str | None = None) -> dict[str, int]:
        lake = lake_dir if isinstance(lake_dir, ExperienceLake) else ExperienceLake(lake_dir)
        eff_op = (str(operator_filter).strip() if operator_filter is not None else "") or None
        by_regime, observations = lake.load_records_by_regime()
        flat: dict[str, CandidateArtifactRecord] = {}
        for _, art in sorted(by_regime.items()):
            if (p := flat.get(art.cand_sha)) is None or (not p.bisheng_summary and art.bisheng_summary) or (not p.base_ledger and art.base_ledger):
                flat[art.cand_sha] = art

        def _proj(sha: str, obs_r: CaseObservationRecord, art: CandidateArtifactRecord | None = None) -> tuple[dict[str, object], dict[str, object] | None]:
            if (art := art or resolve_candidate_for_regime(by_regime, flat, sha, obs_r.regime_tag)) is None:
                p = project_case_workload(None, case_dims=obs_r.case_dims, t_hw_us=obs_r.t_hw_us, target=self.target)
                p["gm_bytes"] = max(float(p["gm_bytes"]), float(obs_r.projected_gm_bytes)) if obs_r.projected_gm_bytes > 0.0 else p["gm_bytes"]
                p["total_tiles"] = int(obs_r.projected_tiles) if obs_r.projected_tiles > 0 else p["total_tiles"]
                p["contig_burst_bytes"] = int(obs_r.projected_contig_bytes) if obs_r.projected_contig_bytes > 0 else p["contig_burst_bytes"]
                return p, None
            ax = TopologyAxes.from_dict(art.topology_axes) if art.topology_axes else None
            return project_case_workload(art.base_ledger, microarch_profile=art.microarch_profile, case_dims=obs_r.case_dims, policy=art.config_knobs, t_hw_us=obs_r.t_hw_us, axes=ax, target=self.target), (art.bisheng_summary or None)

        n_obs, n_msprof, n_pairs, pending = 0, 0, 0, []
        for obs in observations:
            okey = (str(obs.cand_sha), str(obs.operator), str(obs.regime_tag), int(obs.case_num), str(obs.eval_tier), round(float(obs.measured_us), 6))
            if obs.measured_us > 0.0 and not (exclude_run_id and obs.run_id == exclude_run_id) and matches_operator_filter(obs.operator, eff_op) and okey not in self._replayed_obs_keys:
                self._replayed_obs_keys.add(okey)
                wl, bres = _proj(obs.cand_sha, obs)
                pending.append((wl, bres, obs.measured_us))
                if obs.msprof_pmu:
                    self.observe_msprof(wl, obs.msprof_pmu, measured_us=obs.measured_us, bisheng_res=bres, persist=False)
                    n_msprof += 1
        for wl, bres, mus in pending:
            self.observe(wl, mus, bisheng_res=bres, persist=False, refit=False)
            n_obs += 1
        for ca, co, pa, po in lake.extract_pairwise_deltas(exclude_run_id=exclude_run_id):
            pkey = (str(co.cand_sha), str(po.cand_sha), str(co.operator), str(co.regime_tag), int(co.case_num))
            if matches_operator_filter(co.operator, eff_op) and pkey not in self._replayed_pair_keys:
                self._replayed_pair_keys.add(pkey)
                wlc, bc = _proj(ca.cand_sha, co, ca)
                wlp, bp = _proj(pa.cand_sha, po, pa)
                self.observe_pair_delta(wlc, co.measured_us, wlp, po.measured_us, cand_bisheng_res=bc, parent_bisheng_res=bp, refit=False)
                n_pairs += 1
        if n_obs > 0:
            self._refit_ridge()
        if n_pairs > 0:
            self._refit_delta_ridge()
        if persist_calibration and self.calibration_path is not None and (n_obs > 0 or n_msprof > 0 or n_pairs > 0):
            self.save_calibration(self.calibration_path)
        return {"observations": n_obs, "msprof_events": n_msprof, "pair_deltas": n_pairs}

    def rebuild_from_experience_lake(self, lake: ExperienceLake | str | Path, *, persist_calibration: bool = False) -> int:
        self._reset_calibration_state()
        return int(self.replay_experience_lake(lake, persist_calibration=persist_calibration)["observations"])

    def predict_cannbench_eq3_score(self, pred_us: float, ref_perf_us: float, t_hw_us: float) -> CannbenchScorePrediction:
        return predict_cannbench_eq3_score(pred_us, ref_perf_us, t_hw_us)


def _read_attr_or_key(obj: object | None, key: str, default: object = None) -> object:
    return default if obj is None else (obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default))


def _resolve_ledger_total_gm_bytes(ledger: object | None, total_tiles: int) -> int:
    if ledger is None:
        return 0
    if callable(total_fn := getattr(ledger, "total_gm_bytes", None)):
        return max(0, int(total_fn(total_tiles)))
    if (raw_gm := int(_read_attr_or_key(ledger, "gm_roundtrip_bytes", 0) or 0)) <= 0:
        return 0
    scope = "per_tile" if (isinstance(ledger, ObjectivePhysicalLedger) and not hasattr(ledger, "gm_traffic_scope")) else str(_read_attr_or_key(ledger, "gm_traffic_scope", "per_tile") or "per_tile")
    if scope not in ("full_grid", "per_tile"):
        raise ValueError(f"Unsupported gm_traffic_scope: {scope!r}")
    return raw_gm if scope == "full_grid" else raw_gm * max(1, int(total_tiles))


def build_cost_features_from_ledger(
    ledger: object | None, *, target: AscendTarget | None = None, live_plan: object | None = None,
    axes: TopologyAxes | None = None, blueprint: object | None = None, sil_advisory: object | None = None,
    kernel_launches: int = 1, stage_b_profile: MicroarchProfile | Mapping[str, object] | None = None,
) -> CostFeatures:
    """Convert ObjectivePhysicalLedger, HIR blueprint, and SIL advisory into CostFeatures."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    rp_pol = live_plan if isinstance(live_plan, Mapping) else getattr(live_plan, "policy", None)
    policy = dict(ip) if isinstance(live_plan, Mapping) and isinstance(ip := live_plan.get("policy"), Mapping) else (dict(rp_pol) if isinstance(rp_pol, Mapping) else {})
    plan_axes = getattr(live_plan, "topology_axes", None) if not isinstance(live_plan, Mapping) and live_plan is not None else None
    eff_axes = axes if isinstance(axes, TopologyAxes) else (plan_axes if isinstance(plan_axes, TopologyAxes) else (aa if isinstance(aa := _read_attr_or_key(sil_advisory, "topology_axes", None), TopologyAxes) else None))
    total_tiles = (int(_read_attr_or_key(ledger, "grid_blocks", 0) or 0) or int(policy.get("grid_blocks") or 0)) or 56
    in_dt = _read_attr_or_key(sil_advisory, "input_dtypes", ())
    raw_dtype = str(policy.get("cube_operand_dtype") or policy.get("dtype") or (in_dt[0] if isinstance(in_dt, Sequence) and in_dt else "")).strip()
    cube_dtype, hf32 = raw_dtype or "f16", bool(policy.get("hf32", policy.get("hf32_allowed", False)))
    pipes = rp if isinstance(rp := _read_attr_or_key(ledger, "pipe_cycles", None), Mapping) else {}
    l_cube = float(max(0, int(pipes.get("AIC.PIPE_M", 0)) + int(pipes.get("AIC.CUBE", 0)))) * float(_cube_macs_for_dtype(facts, cube_dtype, hf32=hf32)) * 2.0 * float(total_tiles)
    l_vec = float(max(0, int(pipes.get("AIV0.PIPE_V", 0)) + int(pipes.get("AIV1.PIPE_V", 0)))) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * float(total_tiles)
    nav_rl = _read_attr_or_key(blueprint, "roofline", None)
    nav_cube, nav_vec, nav_gm = (float(_read_attr_or_key(nav_rl, k, 0.0) or 0.0) for k in ("cube_flops", "vector_flops", "min_gm_bytes"))
    cube_flops, vec_flops = (max(l_cube, nav_cube) if nav_cube > 0.0 else l_cube), (max(l_vec, nav_vec) if nav_vec > 0.0 else l_vec)
    adv_gm = float(_read_attr_or_key(_read_attr_or_key(sil_advisory, "roofline", None), "median_gm_bytes", 0.0) or 0.0)
    gm_bytes = max(agm, nav_gm) if (agm := float(_resolve_ledger_total_gm_bytes(ledger, total_tiles))) > 0.0 else (nav_gm or adv_gm or 65536.0)
    tmnk = policy.get("tile_mnk")
    fix_b = (float(int(tmnk[0]) * int(tmnk[1]) * 4 * total_tiles) if (isinstance(tmnk, (list, tuple)) and len(tmnk) >= 2 and int(tmnk[0] or 0) > 0 and int(tmnk[1] or 0) > 0) else 0.25 * gm_bytes) if cube_flops > 0.0 else 0.0
    is_coupled = eff_axes is not None and eff_axes.compute_engine == "coupled_1c2v"
    cv_sync = (int(policy["cv_sync_events"]) if policy.get("cv_sync_events") is not None else max(2, int(policy.get("l1_stages", 2) or 2) * 2)) if is_coupled else 0
    m_prof = merge_microarch_profiles(_read_attr_or_key(ledger, "microarch_profile", None), stage_b_profile)
    eff_ln = max(1, int(kernel_launches), m_prof.subkernel_count) if (max(0, int(kernel_launches)) == 1 and not m_prof.is_empty) else max(0, int(kernel_launches))
    merged: dict[str, object] = dict(policy)
    extra = {str(k): float(v) for k, v in merged["extra_signals"].items()} if isinstance(merged.get("extra_signals"), Mapping) else {}
    if (ubl := max(0, int(_read_attr_or_key(ledger, "ub_scalar_compute_loops", 0) or 0))) > 0:
        extra["scalar_compute_loop_fallback"] = max(float(extra.get("scalar_compute_loop_fallback", 0.0) or 0.0), float(ubl))
    if 0 < (mgm := int(_read_attr_or_key(ledger, "min_gm_dma_bytes", 0) or 0)) < 32:
        extra["dcache_bypass_narrow_store"] = max(float(extra.get("dcache_bypass_narrow_store", 0.0) or 0.0), 32.0 / float(mgm))
    if (ubr := float(_read_attr_or_key(ledger, "ub_bank_conflict_risk", 0.0) or 0.0)) > 0.0:
        merged["ub_bank_conflict_ratio"] = max(float(merged.get("ub_bank_conflict_ratio", 0.0) or 0.0), min(1.0, ubr))
    if float(_read_attr_or_key(ledger, "l1_bank_conflict_risk", 0.0) or 0.0) > 0.0 and not bool(policy.get("l1_half_bank_split", False)):
        merged["l1_half_bank_split"], merged["l1_stages"] = False, max(2, int(merged.get("l1_stages", 1) or 1))
    if int(_read_attr_or_key(ledger, "unaligned_dma_padding_bytes", 0) or 0) > 0:
        merged["has_unaligned_tail"] = True
    if (ab := int(_read_attr_or_key(ledger, "active_grid_blocks", 0) or 0)) > 0:
        if int(policy.get("active_aic_cores") or 0) <= 0:
            merged["active_aic_cores"] = min(max(1, int(facts.num_aic_cores)), ab)
        if int(policy.get("active_aiv_cores") or 0) <= 0:
            merged["active_aiv_cores"] = min(max(1, int(facts.num_aiv_cores)), ab * 2 if is_coupled else ab)
    merged.update({
        "cube_flops": cube_flops, "vec_flops": vec_flops, "cube_dtype": cube_dtype, "hf32": hf32, "gm_bytes": gm_bytes,
        "contig_burst_bytes": int(_read_attr_or_key(ledger, "dominant_contig_bytes", 0) or 0) or int(policy.get("contig_burst_bytes") or 32768),
        "total_tiles": total_tiles, "fixpipe_bytes": fix_b, "cv_sync_events": cv_sync, "kernel_launches": eff_ln,
        "steady_state_ii": int(_read_attr_or_key(ledger, "steady_state_ii", 0) or 0),
        "loop_trips": max(1, m_prof.inner_loop_trips) if not m_prof.is_empty else 1,
        "pipeline_depth": max(1, m_prof.pipeline_depth) if not m_prof.is_empty else 1,
        "ub_occupancy_ratio": extract_ub_occupancy_ratio(_read_attr_or_key(ledger, "waterlines", None)) or float(policy.get("ub_occupancy_ratio", 0.0) or 0.0),
        "extra_signals": extra, "microarch_profile": m_prof.to_dict() if not m_prof.is_empty else {},
    })
    if eff_axes is not None:
        merged["topology_axes"] = eff_axes
    return CostFeatures.coerce(merged)


__all__ = [
    "CUBE_MACS_PER_CYCLE", "DEFAULT_CV_SYNC_EVENT_US", "DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC",
    "DEFAULT_LAUNCH_OVERHEAD_US", "DEFAULT_RESIDUAL_BASIS_SPECS", "CalibrationSample",
    "CannbenchScorePrediction", "CostFeatures", "CostModelEvalReport", "CostPrediction",
    "HybridAscendCostModel", "ResidualBasisContext", "ResidualBasisRegistry", "ResidualBasisSpec",
    "build_cost_features_from_ledger", "evaluate_cost_model_ranking",
    "predict_cannbench_eq3_score", "topology_bucket_for_axes",
]
