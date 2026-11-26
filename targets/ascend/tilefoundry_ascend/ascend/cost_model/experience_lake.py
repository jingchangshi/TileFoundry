"""Lossless content-addressed and sharded-append Experience Lake for marathon flywheels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile
from tilefoundry_ascend.ascend.cost_model.pipe_schedule_eval import resolve_cube_macs_safe
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_slug(raw: str, default: str = "default") -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(raw or "").strip()).strip("._-")
    return cleaned or default


def compute_source_sha256(text: str) -> str:
    """Return full lowercase SHA-256 hex digest for canonical source text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def resolve_default_experience_lake_dir(
    *,
    worktree: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    explicit_dir: str | Path | None = None,
) -> Path | None:
    """Resolve Experience Lake directory with zero hardcoded machine paths."""
    if explicit_dir is not None and str(explicit_dir).strip():
        return Path(explicit_dir).expanduser().resolve()
    env_dir = os.environ.get("TILEFOUNDRY_EXPERIENCE_LAKE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if worktree is not None and str(worktree).strip():
        return (Path(worktree).expanduser().resolve() / ".tilefoundry-marathon" / "experience_lake")
    if checkpoint_path is not None and str(checkpoint_path).strip():
        return Path(checkpoint_path).expanduser().resolve().parent / "experience_lake"
    return None


@dataclass(frozen=True)
class CandidateArtifactRecord:
    """Static candidate IR, schedule, compiler summary, and lineage snapshot per sha."""

    cand_sha: str
    operator: str
    regime_tag: str = ""
    backend_track: str = "tilelang_dsl"
    parent_sha: str = ""
    worker_kind: str = ""
    rule_id: str | None = None
    config_knobs: dict[str, object] = field(default_factory=dict)
    topology_axes: dict[str, str] = field(default_factory=dict)
    base_ledger: dict[str, object] = field(default_factory=dict)
    microarch_profile: dict[str, object] = field(default_factory=dict)
    bisheng_summary: dict[str, object] = field(default_factory=dict)
    dsl_line_count: int = 0
    subkernel_count: int = 1
    schema_version: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateArtifactRecord:
        return cls(
            cand_sha=str(payload["cand_sha"]),
            operator=str(payload.get("operator", "")),
            regime_tag=str(payload.get("regime_tag", "")),
            backend_track=str(payload.get("backend_track", "tilelang_dsl")),
            parent_sha=str(payload.get("parent_sha", "")),
            worker_kind=str(payload.get("worker_kind", "")),
            rule_id=(str(payload["rule_id"]) if payload.get("rule_id") is not None else None),
            config_knobs=dict(payload.get("config_knobs") or {}),
            topology_axes={str(k): str(v) for k, v in dict(payload.get("topology_axes") or {}).items()},
            base_ledger=dict(payload.get("base_ledger") or {}),
            microarch_profile=dict(payload.get("microarch_profile") or {}),
            bisheng_summary=dict(payload.get("bisheng_summary") or {}),
            dsl_line_count=int(payload.get("dsl_line_count", 0) or 0),
            subkernel_count=max(1, int(payload.get("subkernel_count", 1) or 1)),
            schema_version=int(payload.get("schema_version", 2) or 2),
        )


@dataclass(frozen=True)
class CaseObservationRecord:
    """Append-only per-case NPU measurement and MSProf telemetry (written per evaluated case)."""

    run_id: str
    operator: str
    regime_tag: str
    cand_sha: str
    eval_tier: str
    case_num: int
    measured_us: float
    accuracy_passed: bool
    parent_sha: str = ""
    device_id: str = ""
    case_dims: dict[str, object] = field(default_factory=dict)
    t_hw_us: float = 0.0
    ref_perf_us: float = 0.0
    speedup: float = 0.0
    eq3_score: float = 0.0
    msprof_pmu: dict[str, float] = field(default_factory=dict)
    projected_tiles: int = 0
    projected_gm_bytes: float = 0.0
    projected_contig_bytes: int = 32768
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 3
    min_gm_bytes: float = 0.0
    compute_floor_us: float = 0.0
    transfer_inflation_ratio: float = 0.0
    analytical_pred_us: float = 0.0
    ring_burst_k: int = 1
    burst_batch_cv: float = 0.0
    spike_retries: int = 0
    card_ext_hbm_mb: int = 0
    is_real_npu: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CaseObservationRecord:
        raw_pmu = payload.get("msprof_pmu") or {}
        return cls(
            run_id=str(payload.get("run_id", "")),
            operator=str(payload.get("operator", "")),
            regime_tag=str(payload.get("regime_tag", "")),
            cand_sha=str(payload["cand_sha"]),
            eval_tier=str(payload.get("eval_tier", "l2_pass")),
            case_num=int(payload.get("case_num", 0) or 0),
            measured_us=float(payload.get("measured_us", 0.0) or 0.0),
            accuracy_passed=bool(payload.get("accuracy_passed", False)),
            parent_sha=str(payload.get("parent_sha", "")),
            device_id=str(payload.get("device_id", "")),
            case_dims=dict(payload.get("case_dims") or {}),
            t_hw_us=float(payload.get("t_hw_us", 0.0) or 0.0),
            ref_perf_us=float(payload.get("ref_perf_us", 0.0) or 0.0),
            speedup=float(payload.get("speedup", 0.0) or 0.0),
            eq3_score=float(payload.get("eq3_score", 0.0) or 0.0),
            msprof_pmu={str(k): float(v) for k, v in dict(raw_pmu).items()},
            projected_tiles=int(payload.get("projected_tiles", 0) or 0),
            projected_gm_bytes=float(payload.get("projected_gm_bytes", 0.0) or 0.0),
            projected_contig_bytes=int(payload.get("projected_contig_bytes", 32768) or 32768),
            timestamp=float(payload.get("timestamp", 0.0) or 0.0),
            schema_version=int(payload.get("schema_version", 2) or 2),
            min_gm_bytes=float(payload.get("min_gm_bytes", 0.0) or 0.0),
            compute_floor_us=float(payload.get("compute_floor_us", 0.0) or 0.0),
            transfer_inflation_ratio=float(
                payload.get("transfer_inflation_ratio", 0.0) or 0.0
            ),
            analytical_pred_us=float(payload.get("analytical_pred_us", 0.0) or 0.0),
            ring_burst_k=max(1, int(payload.get("ring_burst_k", 1) or 1)),
            burst_batch_cv=max(0.0, float(payload.get("burst_batch_cv", 0.0) or 0.0)),
            spike_retries=max(0, int(payload.get("spike_retries", 0) or 0)),
            card_ext_hbm_mb=max(0, int(payload.get("card_ext_hbm_mb", 0) or 0)),
            is_real_npu=bool(payload.get("is_real_npu", False)),
        )


def _pick_dim(dims: Mapping[str, object], keys: Sequence[str]) -> int | None:
    lowered = {str(k).lower(): v for k, v in dims.items()}
    for k in keys:
        val = lowered.get(k.lower())
        if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) > 0:
            return int(val)
    return None


def project_case_workload(
    base_ledger: Mapping[str, object] | object | None,
    *,
    microarch_profile: MicroarchProfile | Mapping[str, object] | None = None,
    case_dims: Mapping[str, object] | None = None,
    policy: Mapping[str, object] | None = None,
    t_hw_us: float = 0.0,
    axes: object | None = None,
    target: AscendTarget | None = None,
) -> dict[str, object]:
    """Project one static candidate ledger onto concrete case dimensions in under a millisecond.

    Analytical GM-byte and FLOP projections follow the concrete case dimensions.
    """
    eff_target = target or AscendTarget("ascend.950pr")
    facts = eff_target.get_facts(AscendSpaceFacts)
    prof = MicroarchProfile.coerce(microarch_profile)

    if base_ledger is None:
        ledger_map: dict[str, object] = {}
    elif isinstance(base_ledger, Mapping):
        ledger_map = dict(base_ledger)
    elif hasattr(base_ledger, "to_dict") and callable(base_ledger.to_dict):
        raw_ledger = base_ledger.to_dict()
        ledger_map = dict(raw_ledger) if isinstance(raw_ledger, Mapping) else {}
    else:
        ledger_map = {
            "grid_blocks": getattr(base_ledger, "grid_blocks", 1),
            "pipe_cycles": getattr(base_ledger, "pipe_cycles", {}),
            "gm_roundtrip_bytes": getattr(base_ledger, "gm_roundtrip_bytes", 0),
            "gm_traffic_scope": getattr(base_ledger, "gm_traffic_scope", "per_tile"),
            "dominant_contig_bytes": getattr(base_ledger, "dominant_contig_bytes", 32768),
            "steady_state_ii": getattr(base_ledger, "steady_state_ii", 0),
            "waterlines": getattr(base_ledger, "waterlines", {}),
            "ub_scalar_compute_loops": getattr(base_ledger, "ub_scalar_compute_loops", 0),
        }

    dims = dict(case_dims or {})
    pol = dict(policy or {})
    b_dim = _pick_dim(dims, ("B", "b_dim", "batch")) or 1
    m_dim = _pick_dim(dims, ("M", "m_dim", "m_rows", "S", "s_dim", "S_q", "seq_len")) or 0
    n_dim = _pick_dim(dims, ("N", "n_dim", "H", "num_heads", "N_q")) or 0
    k_dim = _pick_dim(dims, ("K", "k_dim", "D", "d_dim", "D_nope", "hidden_dim")) or 0
    s_kv = _pick_dim(dims, ("S_kv", "s_kv")) or 0
    g_dim = _pick_dim(dims, ("G", "E", "num_experts", "groups")) or 1

    s_dim = _pick_dim(dims, ("S", "s_dim", "S_q", "seq_len")) or 0
    m_rows_dim = _pick_dim(dims, ("m_rows",))
    has_explicit_m = _pick_dim(dims, ("M", "m_dim")) is not None
    eff_b_dim = b_dim
    if (
        m_rows_dim is not None
        and b_dim > 1
        and (
            (not has_explicit_m and m_dim == m_rows_dim and m_dim >= b_dim * max(1, s_dim))
            or (s_dim > 0 and m_rows_dim == b_dim * s_dim and m_dim == m_rows_dim)
        )
    ):
        eff_b_dim = 1

    tile_mnk = pol.get("tile_mnk")
    bm = (
        int(tile_mnk[0])
        if isinstance(tile_mnk, (list, tuple)) and len(tile_mnk) >= 1 and int(tile_mnk[0] or 0) > 0
        else (_pick_dim(pol, ("bm", "tile_m", "block_m")) or 64)
    )
    bn = (
        int(tile_mnk[1])
        if isinstance(tile_mnk, (list, tuple)) and len(tile_mnk) >= 2 and int(tile_mnk[1] or 0) > 0
        else (_pick_dim(pol, ("bn", "tile_n", "block_n")) or 64)
    )

    base_grid = max(1, int(ledger_map.get("grid_blocks", 1) or 1))
    is_fixed_grid_full_loop = (
        not bool(ledger_map.get("grid_is_symbolic", False))
        and (prof.gm_to_ub_bytes + prof.gm_to_l1_bytes) > 0
    )
    if is_fixed_grid_full_loop:
        proj_tiles = base_grid
    elif m_dim > 0 and n_dim > 0 and ("M" in dims or "m_dim" in dims or "N" in dims or "n_dim" in dims):
        proj_tiles = max(
            1,
            math.ceil(m_dim / max(1, bm)) * math.ceil(n_dim / max(1, bn)) * eff_b_dim * g_dim,
        )
    elif m_dim > 0:
        heads = _pick_dim(dims, ("H", "N_q", "num_heads")) or 1
        proj_tiles = max(1, eff_b_dim * heads * math.ceil(m_dim / max(1, bm)))
    else:
        proj_tiles = base_grid
    chip_scale = float(base_grid) if is_fixed_grid_full_loop else float(proj_tiles)

    dtype_name = str(pol.get("cube_operand_dtype") or pol.get("dtype") or "f16")
    elem_b = 4 if "32" in dtype_name else 2
    base_contig = int(ledger_map.get("dominant_contig_bytes", 0) or 32768)
    inner_dim = k_dim or n_dim or m_dim
    proj_contig = max(32, min(base_contig, inner_dim * elem_b)) if inner_dim > 0 else base_contig

    case_gm = 0.0
    cube_flops = 0.0
    vec_flops = 0.0
    if m_dim > 0 and k_dim > 0 and n_dim > 0:
        eff_k = max(k_dim, s_kv)
        case_gm = float(eff_b_dim * g_dim * (m_dim * eff_k + eff_k * n_dim + m_dim * n_dim) * elem_b)
    elif m_dim > 0 and (k_dim > 0 or n_dim > 0):
        width = max(k_dim, n_dim)
        case_gm = float(eff_b_dim * g_dim * m_dim * width * 2 * elem_b)

    base_gm_raw = float(ledger_map.get("gm_roundtrip_bytes", 0) or 0)
    gm_scope = str(ledger_map.get("gm_traffic_scope", "per_tile") or "per_tile")
    if base_gm_raw > 0.0:
        if gm_scope == "full_grid":
            scaled_gm = base_gm_raw * (float(proj_tiles) / float(base_grid))
        elif gm_scope == "per_tile":
            scaled_gm = base_gm_raw * float(proj_tiles)
        else:
            raise ValueError(f"Unsupported gm_traffic_scope: {gm_scope!r}")
        case_gm = max(case_gm, scaled_gm)
    hbm_bw = float(facts.effective_hbm_bandwidth_bps(max(32, proj_contig)))
    if t_hw_us > 0.0:
        case_gm = max(case_gm, float(t_hw_us) * 1e-6 * hbm_bw)
    if case_gm <= 0.0:
        case_gm = 65536.0

    pipes = dict(ledger_map.get("pipe_cycles") or {})
    cube_c = int(pipes.get("AIC.PIPE_M", 0) or 0) + int(pipes.get("AIC.CUBE", 0) or 0)
    vec_c = int(pipes.get("AIV0.PIPE_V", 0) or 0) + int(pipes.get("AIV1.PIPE_V", 0) or 0)
    macs_per_cyc = float(resolve_cube_macs_safe(facts, dtype_name, hf32=bool(pol.get("hf32", False))))
    clock_hz = float(getattr(facts, "clock_ghz", 1.65) or 1.65) * 1e9

    if cube_c > 0 or prof.cube_macs_total > 0 or prof.cube_mmad_count > 0:
        block_macs = max(float(prof.cube_macs_total), float(cube_c) * macs_per_cyc)
        cube_flops = block_macs * 2.0 * chip_scale
        if m_dim > 0 and k_dim > 0 and n_dim > 0:
            cube_flops = max(cube_flops, float(eff_b_dim * g_dim * 2 * m_dim * n_dim * max(k_dim, s_kv)))
        if t_hw_us > 0.0:
            cube_peak = float(facts.num_aic_cores) * macs_per_cyc * 2.0 * clock_hz
            cube_flops = max(cube_flops, float(t_hw_us) * 1e-6 * cube_peak)

    if vec_c > 0 or prof.total_vector_ops > 0:
        vec_flops = max(float(vec_c) * 128.0, float(prof.total_vector_ops) * 128.0) * chip_scale
    elif t_hw_us > 0.0 and cube_flops <= 0.0:
        vec_peak = float(facts.num_aiv_cores) * 128.0 * clock_hz
        vec_flops = float(t_hw_us) * 1e-6 * vec_peak

    fixpipe_bytes = float(prof.l0c_to_ub_bytes + prof.l0c_to_l1_bytes + prof.l0c_to_gm_bytes)
    if fixpipe_bytes <= 0.0 and cube_flops > 0.0:
        fixpipe_bytes = 0.25 * case_gm
    else:
        fixpipe_bytes *= chip_scale

    ub_occ_ratio = float(pol.get("ub_occupancy_ratio", 0.0) or 0.0)
    raw_waterlines = ledger_map.get("waterlines")
    if isinstance(raw_waterlines, Mapping):
        ub_wl = raw_waterlines.get("UB")
        if isinstance(ub_wl, Mapping):
            ub_cap = int(ub_wl.get("capacity_bytes", 0) or 0)
            ub_peak = int(ub_wl.get("peak_allocated_bytes", 0) or 0)
            if ub_cap > 0 and ub_peak >= 0:
                ub_occ_ratio = float(ub_peak) / float(ub_cap)
        elif ub_wl is not None:
            ub_cap = int(getattr(ub_wl, "capacity_bytes", 0) or 0)
            ub_peak = int(getattr(ub_wl, "peak_allocated_bytes", 0) or 0)
            if ub_cap > 0 and ub_peak >= 0:
                ub_occ_ratio = float(ub_peak) / float(ub_cap)

    raw_extra = pol.get("extra_signals")
    extra_signals: dict[str, float] = (
        {str(k): float(v) for k, v in raw_extra.items()}
        if isinstance(raw_extra, Mapping)
        else {}
    )
    ub_loops = max(0, int(ledger_map.get("ub_scalar_compute_loops", 0) or 0))
    if ub_loops > 0:
        existing_val = float(extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0)
        extra_signals["scalar_compute_loop_fallback"] = max(existing_val, float(ub_loops))

    out: dict[str, object] = dict(pol)
    out.update(
        {
            "cube_flops": cube_flops,
            "vec_flops": vec_flops,
            "cube_dtype": dtype_name,
            "gm_bytes": case_gm,
            "contig_burst_bytes": proj_contig,
            "total_tiles": proj_tiles,
            "fixpipe_bytes": fixpipe_bytes,
            "ub_occupancy_ratio": ub_occ_ratio,
            "steady_state_ii": int(ledger_map.get("steady_state_ii") or 0),
            "loop_trips": max(1, prof.inner_loop_trips),
            "pipeline_depth": max(1, prof.pipeline_depth),
            "kernel_launches": max(1, prof.subkernel_count),
            "profile_is_per_core_full_loop": is_fixed_grid_full_loop,
            "extra_signals": extra_signals,
            "microarch_profile": prof.to_dict(),
        }
    )
    if axes is not None:
        out["topology_axes"] = axes
    return out


def _prefer_richer_candidate(
    existing: CandidateArtifactRecord | None,
    candidate: CandidateArtifactRecord,
) -> CandidateArtifactRecord:
    """Prefer the candidate snapshot carrying compiler summary and base ledger data."""
    if existing is None:
        return candidate
    if not existing.bisheng_summary and candidate.bisheng_summary:
        return candidate
    if not existing.base_ledger and candidate.base_ledger:
        return candidate
    return existing


def _flatten_candidates_by_sha(
    by_regime: Mapping[tuple[str, str], CandidateArtifactRecord],
) -> dict[str, CandidateArtifactRecord]:
    """Collapse regime-keyed candidate snapshots into a deterministic flat map."""
    flat: dict[str, CandidateArtifactRecord] = {}
    for (_sha, _rtag), rec in sorted(by_regime.items()):
        flat[rec.cand_sha] = _prefer_richer_candidate(flat.get(rec.cand_sha), rec)
    return flat


def resolve_candidate_for_regime(
    by_regime: Mapping[tuple[str, str], CandidateArtifactRecord],
    flat: Mapping[str, CandidateArtifactRecord],
    cand_sha: str,
    regime_tag: str = "",
) -> CandidateArtifactRecord | None:
    """Resolve candidate snapshot by (cand_sha, regime_tag), empty-regime, or flat fallback."""
    clean_sha = cand_sha.strip()
    clean_regime = regime_tag.strip()
    if clean_regime:
        hit = by_regime.get((clean_sha, clean_regime))
        if hit is not None:
            return hit
    default_hit = by_regime.get((clean_sha, ""))
    if default_hit is not None:
        return default_hit
    return flat.get(clean_sha)


class ExperienceLake:
    """Thread/process-safe content-addressed candidate + sharded append-only observation store."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.candidates_dir = self.root_dir / "candidates"
        self.runs_dir = self.root_dir / "runs"

    def _candidate_path(self, cand_sha: str, regime_tag: str) -> Path:
        sha = _safe_slug(cand_sha, "unknown")
        clean_regime = regime_tag.strip()
        if clean_regime:
            return self.candidates_dir / f"{sha}_{_safe_slug(clean_regime)}.json"
        return self.candidates_dir / f"{sha}.json"

    def register_candidate(self, record: CandidateArtifactRecord) -> Path:
        """Persist candidate static artifact atomically per (cand_sha, regime_tag)."""
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        sha = _safe_slug(record.cand_sha, "unknown")
        target = self._candidate_path(record.cand_sha, record.regime_tag)
        record_to_write = record
        if target.is_file():
            try:
                existing = CandidateArtifactRecord.from_dict(
                    json.loads(target.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                existing = None
            if existing is not None:
                needs_bisheng_upgrade = (
                    not existing.bisheng_summary and bool(record.bisheng_summary)
                )
                needs_ledger_upgrade = not existing.base_ledger and bool(record.base_ledger)
                if not needs_bisheng_upgrade and not needs_ledger_upgrade:
                    return target
                record_to_write = CandidateArtifactRecord(
                    cand_sha=record.cand_sha,
                    operator=record.operator or existing.operator,
                    regime_tag=record.regime_tag or existing.regime_tag,
                    backend_track=record.backend_track or existing.backend_track,
                    parent_sha=record.parent_sha or existing.parent_sha,
                    worker_kind=record.worker_kind or existing.worker_kind,
                    rule_id=record.rule_id if record.rule_id is not None else existing.rule_id,
                    config_knobs=dict(record.config_knobs or existing.config_knobs),
                    topology_axes=dict(record.topology_axes or existing.topology_axes),
                    base_ledger=dict(record.base_ledger or existing.base_ledger),
                    microarch_profile=dict(
                        record.microarch_profile or existing.microarch_profile
                    ),
                    bisheng_summary=dict(record.bisheng_summary or existing.bisheng_summary),
                    dsl_line_count=record.dsl_line_count or existing.dsl_line_count,
                    subkernel_count=max(record.subkernel_count, existing.subkernel_count),
                    schema_version=max(record.schema_version, existing.schema_version),
                )
        payload = json.dumps(record_to_write.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=str(self.candidates_dir), prefix=f".{sha}.")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(str(tmp_path), str(target))
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return target

    def append_observation(self, record: CaseObservationRecord) -> Path:
        """Append one compact observation line to the per-run shard file (zero write races)."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        shard = self.runs_dir / f"{_safe_slug(record.run_id, 'default')}.jsonl"
        line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
        with shard.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return shard

    def append_unmodeled_anomaly(self, record: Mapping[str, object]) -> Path:
        """Append one compact outlier record to ``unmodeled_anomalies.jsonl``."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        anomaly_path = self.root_dir / "unmodeled_anomalies.jsonl"
        payload = {str(k): v for k, v in record.items()}
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with anomaly_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return anomaly_path

    def _load_observations(self) -> list[CaseObservationRecord]:
        observations: list[CaseObservationRecord] = []
        if self.runs_dir.is_dir():
            for rpath in sorted(self.runs_dir.glob("*.jsonl")):
                try:
                    lines = rpath.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for line in lines:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obs = CaseObservationRecord.from_dict(json.loads(raw))
                        if obs.measured_us > 0.0:
                            observations.append(obs)
                    except Exception:
                        continue
        observations.sort(key=lambda o: (o.timestamp, o.operator, o.case_num, o.cand_sha))
        return observations

    def load_records_by_regime(
        self,
    ) -> tuple[dict[tuple[str, str], CandidateArtifactRecord], list[CaseObservationRecord]]:
        """Load candidates keyed by (cand_sha, regime_tag) and sharded case observations."""
        candidates: dict[tuple[str, str], CandidateArtifactRecord] = {}
        if self.candidates_dir.is_dir():
            for cpath in sorted(self.candidates_dir.glob("*.json")):
                try:
                    data = json.loads(cpath.read_text(encoding="utf-8"))
                    rec = CandidateArtifactRecord.from_dict(data)
                    key = (rec.cand_sha, rec.regime_tag.strip())
                    candidates[key] = _prefer_richer_candidate(candidates.get(key), rec)
                except Exception:
                    continue
        return candidates, self._load_observations()

    def load_records(self) -> tuple[dict[str, CandidateArtifactRecord], list[CaseObservationRecord]]:
        """Load all deduplicated candidates and sharded case observations."""
        by_regime, observations = self.load_records_by_regime()
        return _flatten_candidates_by_sha(by_regime), observations

    def extract_pairwise_deltas(
        self,
        *,
        exclude_run_id: str | None = None,
    ) -> list[
        tuple[
            CandidateArtifactRecord,
            CaseObservationRecord,
            CandidateArtifactRecord,
            CaseObservationRecord,
        ]
    ]:
        """Match same-(operator, regime_tag, case_num) candidate-vs-parent measurement pairs."""
        by_regime, observations = self.load_records_by_regime()
        flat_candidates = _flatten_candidates_by_sha(by_regime)
        latest_by_cand_case: dict[tuple[str, str, str, int], CaseObservationRecord] = {}
        for obs in observations:
            if obs.measured_us <= 0.0:
                continue
            if exclude_run_id and obs.run_id == exclude_run_id:
                continue
            latest_by_cand_case[(obs.cand_sha, obs.operator, obs.regime_tag, obs.case_num)] = obs

        pairs: list[
            tuple[
                CandidateArtifactRecord,
                CaseObservationRecord,
                CandidateArtifactRecord,
                CaseObservationRecord,
            ]
        ] = []
        seen_pairs: set[tuple[str, str, str, str, int]] = set()
        for (_csha, op, rtag, cnum), cand_obs in sorted(latest_by_cand_case.items()):
            cand_art = resolve_candidate_for_regime(
                by_regime, flat_candidates, cand_obs.cand_sha, cand_obs.regime_tag
            )
            psha = cand_obs.parent_sha.strip() or (
                cand_art.parent_sha.strip() if cand_art is not None else ""
            )
            if not psha or psha == cand_obs.cand_sha:
                continue
            parent_obs = latest_by_cand_case.get((psha, op, rtag, cnum))
            if parent_obs is None or parent_obs.measured_us <= 0.0:
                continue
            parent_art = resolve_candidate_for_regime(
                by_regime, flat_candidates, psha, parent_obs.regime_tag
            )
            if cand_art is None or parent_art is None:
                continue
            key = (cand_obs.cand_sha, psha, op, rtag, cnum)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append((cand_art, cand_obs, parent_art, parent_obs))
        return pairs


def _matches_counterexample_operator(record_op: str, target_op: str) -> bool:
    rec_norm = str(record_op or "").strip().lower()
    tgt_norm = str(target_op or "").strip().lower()
    if not tgt_norm:
        return True
    if rec_norm == tgt_norm:
        return True
    if "/" not in tgt_norm or "/" not in rec_norm:
        return rec_norm.rsplit("/", 1)[-1] == tgt_norm.rsplit("/", 1)[-1]
    return False


def append_counterexample_boundary(lake_dir: Path, record: Mapping[str, object]) -> Path:
    """Append one L3/L4 counterexample boundary record to counterexample_anchors.jsonl."""
    root = Path(lake_dir)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "counterexample_anchors.jsonl"
    payload = {str(k): v for k, v in record.items()}
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return out_path


def load_counterexample_boundaries(
    lake_dir: Path,
    operator: str | None = None,
) -> list[dict[str, object]]:
    """Load persisted counterexample boundary records, optionally filtered by operator."""
    in_path = Path(lake_dir) / "counterexample_anchors.jsonl"
    if not in_path.is_file():
        return []
    records: list[dict[str, object]] = []
    try:
        lines = in_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for raw_line in lines:
        raw = raw_line.strip()
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(decoded, Mapping):
            continue
        item = {str(k): v for k, v in decoded.items()}
        if operator is not None and not _matches_counterexample_operator(
            str(item.get("operator", "")),
            operator,
        ):
            continue
        records.append(item)
    return records


__all__ = [
    "CandidateArtifactRecord",
    "CaseObservationRecord",
    "ExperienceLake",
    "append_counterexample_boundary",
    "compute_source_sha256",
    "load_counterexample_boundaries",
    "project_case_workload",
    "resolve_candidate_for_regime",
    "resolve_default_experience_lake_dir",
]
