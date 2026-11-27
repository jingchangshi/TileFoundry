"""Lossless content-addressed and sharded-append Experience Lake for marathon flywheels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.target.facts import resolve_cube_macs_safe

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CASE_OBS_DEFAULTS: dict[str, object] = {
    "run_id": "", "operator": "", "regime_tag": "", "eval_tier": "l2_pass",
    "case_num": 0, "measured_us": 0.0, "accuracy_passed": False, "timestamp": 0.0, "schema_version": 2,
}


def _safe_slug(raw: str, default: str = "default") -> str:
    return _SAFE_FILENAME_RE.sub("_", str(raw or "").strip()).strip("._-") or default


def compute_source_sha256(text: str) -> str:
    """Return full lowercase SHA-256 hex digest for canonical source text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def resolve_default_experience_lake_dir(*, worktree: str | Path | None = None, checkpoint_path: str | Path | None = None, explicit_dir: str | Path | None = None) -> Path | None:
    """Resolve Experience Lake directory with zero hardcoded machine paths."""
    if explicit_dir is not None and str(explicit_dir).strip():
        return Path(explicit_dir).expanduser().resolve()
    if (env_dir := os.environ.get("TILEFOUNDRY_EXPERIENCE_LAKE_DIR", "").strip()):
        return Path(env_dir).expanduser().resolve()
    if worktree is not None and str(worktree).strip():
        return Path(worktree).expanduser().resolve() / ".tilefoundry-marathon" / "experience_lake"
    if checkpoint_path is not None and str(checkpoint_path).strip():
        return Path(checkpoint_path).expanduser().resolve().parent / "experience_lake"
    return None


@dataclass(frozen=True)
class CandidateArtifactRecord(JsonRecordMixin):
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "subkernel_count", max(1, int(self.subkernel_count or 1)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateArtifactRecord:
        return super().from_dict(dict(payload) if "operator" in payload else {"operator": "", **payload})


@dataclass(frozen=True)
class CaseObservationRecord(JsonRecordMixin):
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "ring_burst_k", max(1, int(self.ring_burst_k or 1)))
        object.__setattr__(self, "burst_batch_cv", max(0.0, float(self.burst_batch_cv or 0.0)))
        for k in ("spike_retries", "card_ext_hbm_mb"):
            object.__setattr__(self, k, max(0, int(getattr(self, k) or 0)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CaseObservationRecord:
        return super().from_dict({**_CASE_OBS_DEFAULTS, **{k: v for k, v in payload.items() if v is not None}})


def _pick_dim(dims: Mapping[str, object], keys: Sequence[str]) -> int | None:
    lowered = {str(k).lower(): v for k, v in dims.items()}
    return next((int(v) for k in keys if isinstance(v := lowered.get(k.lower()), (int, float)) and not isinstance(v, bool) and int(v) > 0), None)


def _coerce_ledger_map(base_ledger: Mapping[str, object] | object | None) -> dict[str, object]:
    if base_ledger is None:
        return {}
    if isinstance(base_ledger, Mapping):
        return dict(base_ledger)
    if hasattr(base_ledger, "to_dict") and callable(base_ledger.to_dict):
        return dict(raw) if isinstance(raw := base_ledger.to_dict(), Mapping) else {}
    return {k: getattr(base_ledger, k, d) for k, d in (("grid_blocks", 1), ("pipe_cycles", {}), ("gm_roundtrip_bytes", 0), ("gm_traffic_scope", "per_tile"), ("dominant_contig_bytes", 32768), ("steady_state_ii", 0), ("waterlines", {}), ("ub_scalar_compute_loops", 0))}


def extract_ub_occupancy_ratio(waterlines: object, default: float = 0.0) -> float:
    """Compute UB peak/capacity occupancy ratio from *waterlines* mapping."""
    ub_wl = waterlines.get("UB") if isinstance(waterlines, Mapping) else None
    if ub_wl is None:
        return default
    cap = int((ub_wl.get("capacity_bytes") if isinstance(ub_wl, Mapping) else getattr(ub_wl, "capacity_bytes", 0)) or 0)
    peak = int((ub_wl.get("peak_allocated_bytes") if isinstance(ub_wl, Mapping) else getattr(ub_wl, "peak_allocated_bytes", 0)) or 0)
    return (float(peak) / float(cap)) if (cap > 0 and peak >= 0) else default


def project_case_workload(
    base_ledger: Mapping[str, object] | object | None, *, microarch_profile: MicroarchProfile | Mapping[str, object] | None = None,
    case_dims: Mapping[str, object] | None = None, policy: Mapping[str, object] | None = None,
    t_hw_us: float = 0.0, axes: object | None = None, target: AscendTarget | None = None,
) -> dict[str, object]:
    """Project one static candidate ledger onto concrete case dimensions in under a millisecond."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    prof, lmap, dims, pol = MicroarchProfile.coerce(microarch_profile), _coerce_ledger_map(base_ledger), dict(case_dims or {}), dict(policy or {})
    b_dim, m_dim, n_dim, k_dim, s_kv, g_dim, s_dim = (
        _pick_dim(dims, ("B", "b_dim", "batch")) or 1, _pick_dim(dims, ("M", "m_dim", "m_rows", "S", "s_dim", "S_q", "seq_len")) or 0,
        _pick_dim(dims, ("N", "n_dim", "H", "num_heads", "N_q")) or 0, _pick_dim(dims, ("K", "k_dim", "D", "d_dim", "D_nope", "hidden_dim")) or 0,
        _pick_dim(dims, ("S_kv", "s_kv")) or 0, _pick_dim(dims, ("G", "E", "num_experts", "groups")) or 1, _pick_dim(dims, ("S", "s_dim", "S_q", "seq_len")) or 0,
    )
    m_rows, has_exp_m = _pick_dim(dims, ("m_rows",)), _pick_dim(dims, ("M", "m_dim")) is not None
    eff_b = 1 if (m_rows is not None and b_dim > 1 and ((not has_exp_m and m_dim == m_rows >= b_dim * max(1, s_dim)) or (s_dim > 0 and m_rows == b_dim * s_dim == m_dim))) else b_dim
    tmnk = pol.get("tile_mnk")
    bm = int(tmnk[0]) if isinstance(tmnk, (list, tuple)) and len(tmnk) >= 1 and int(tmnk[0] or 0) > 0 else (_pick_dim(pol, ("bm", "tile_m", "block_m")) or 64)
    bn = int(tmnk[1]) if isinstance(tmnk, (list, tuple)) and len(tmnk) >= 2 and int(tmnk[1] or 0) > 0 else (_pick_dim(pol, ("bn", "tile_n", "block_n")) or 64)
    base_grid = max(1, int(lmap.get("grid_blocks", 1) or 1))
    is_fixed = not bool(lmap.get("grid_is_symbolic", False)) and (prof.gm_to_ub_bytes + prof.gm_to_l1_bytes) > 0
    if is_fixed:
        proj_tiles = base_grid
    elif m_dim > 0 and n_dim > 0 and any(k in dims for k in ("M", "m_dim", "N", "n_dim")):
        proj_tiles = max(1, math.ceil(m_dim / max(1, bm)) * math.ceil(n_dim / max(1, bn)) * eff_b * g_dim)
    elif m_dim > 0:
        proj_tiles = max(1, eff_b * (_pick_dim(dims, ("H", "N_q", "num_heads")) or 1) * math.ceil(m_dim / max(1, bm)))
    else:
        proj_tiles = base_grid
    chip_scale = float(base_grid if is_fixed else proj_tiles)

    dtype_name = str(pol.get("cube_operand_dtype") or pol.get("dtype") or "f16")
    elem_b, base_contig, inner = (4 if "32" in dtype_name else 2), int(lmap.get("dominant_contig_bytes", 0) or 32768), (k_dim or n_dim or m_dim)
    proj_contig = max(32, min(base_contig, inner * elem_b)) if inner > 0 else base_contig
    eff_k = max(k_dim, s_kv)
    case_gm = float(eff_b * g_dim * (m_dim * eff_k + eff_k * n_dim + m_dim * n_dim) * elem_b) if (m_dim > 0 and k_dim > 0 and n_dim > 0) else (float(eff_b * g_dim * m_dim * max(k_dim, n_dim) * 2 * elem_b) if (m_dim > 0 and (k_dim > 0 or n_dim > 0)) else 0.0)
    if (base_gm := float(lmap.get("gm_roundtrip_bytes", 0) or 0)) > 0.0:
        if (gm_scope := str(lmap.get("gm_traffic_scope", "per_tile") or "per_tile")) not in ("full_grid", "per_tile"):
            raise ValueError(f"Unsupported gm_traffic_scope: {gm_scope!r}")
        case_gm = max(case_gm, base_gm * (float(proj_tiles) / float(base_grid) if gm_scope == "full_grid" else float(proj_tiles)))
    if t_hw_us > 0.0:
        case_gm = max(case_gm, float(t_hw_us) * 1e-6 * float(facts.effective_hbm_bandwidth_bps(max(32, proj_contig))))
    case_gm = max(case_gm, 65536.0) if case_gm <= 0.0 else case_gm

    pipes = dict(lmap.get("pipe_cycles") or {})
    cube_c, vec_c = int(pipes.get("AIC.PIPE_M", 0) or 0) + int(pipes.get("AIC.CUBE", 0) or 0), int(pipes.get("AIV0.PIPE_V", 0) or 0) + int(pipes.get("AIV1.PIPE_V", 0) or 0)
    macs_cyc, clock_hz = float(resolve_cube_macs_safe(facts, dtype_name, hf32=bool(pol.get("hf32", False)))), float(getattr(facts, "clock_ghz", 1.65) or 1.65) * 1e9
    cube_flops, vec_flops = 0.0, 0.0
    if cube_c > 0 or prof.cube_macs_total > 0 or prof.cube_mmad_count > 0:
        cube_flops = max(float(prof.cube_macs_total), float(cube_c) * macs_cyc) * 2.0 * chip_scale
        if m_dim > 0 and k_dim > 0 and n_dim > 0:
            cube_flops = max(cube_flops, float(eff_b * g_dim * 2 * m_dim * n_dim * eff_k))
        if t_hw_us > 0.0:
            cube_flops = max(cube_flops, float(t_hw_us) * 1e-6 * float(facts.num_aic_cores) * macs_cyc * 2.0 * clock_hz)
    if vec_c > 0 or prof.total_vector_ops > 0:
        vec_flops = max(float(vec_c) * 128.0, float(prof.total_vector_ops) * 128.0) * chip_scale
    elif t_hw_us > 0.0 and cube_flops <= 0.0:
        vec_flops = float(t_hw_us) * 1e-6 * float(facts.num_aiv_cores) * 128.0 * clock_hz
    fix_b = float(prof.l0c_to_ub_bytes + prof.l0c_to_l1_bytes + prof.l0c_to_gm_bytes)
    fixpipe_bytes = (0.25 * case_gm) if (fix_b <= 0.0 and cube_flops > 0.0) else (fix_b * chip_scale)

    extra = {str(k): float(v) for k, v in pol["extra_signals"].items()} if isinstance(pol.get("extra_signals"), Mapping) else {}
    if (ub_loops := max(0, int(lmap.get("ub_scalar_compute_loops", 0) or 0))) > 0:
        extra["scalar_compute_loop_fallback"] = max(float(extra.get("scalar_compute_loop_fallback", 0.0) or 0.0), float(ub_loops))
    out: dict[str, object] = {
        **pol, "cube_flops": cube_flops, "vec_flops": vec_flops, "cube_dtype": dtype_name, "gm_bytes": case_gm,
        "contig_burst_bytes": proj_contig, "total_tiles": proj_tiles, "fixpipe_bytes": fixpipe_bytes,
        "ub_occupancy_ratio": extract_ub_occupancy_ratio(lmap.get("waterlines"), float(pol.get("ub_occupancy_ratio", 0.0) or 0.0)),
        "steady_state_ii": int(lmap.get("steady_state_ii") or 0), "loop_trips": max(1, prof.inner_loop_trips),
        "pipeline_depth": max(1, prof.pipeline_depth), "kernel_launches": max(1, prof.subkernel_count),
        "profile_is_per_core_full_loop": is_fixed, "extra_signals": extra, "microarch_profile": prof.to_dict(),
    }
    if axes is not None:
        out["topology_axes"] = axes
    return out


def _prefer_richer(existing: CandidateArtifactRecord | None, candidate: CandidateArtifactRecord) -> CandidateArtifactRecord:
    return candidate if (existing is None or (not existing.bisheng_summary and candidate.bisheng_summary) or (not existing.base_ledger and candidate.base_ledger)) else existing


def resolve_candidate_for_regime(
    by_regime: Mapping[tuple[str, str], CandidateArtifactRecord], flat: Mapping[str, CandidateArtifactRecord],
    cand_sha: str, regime_tag: str = "",
) -> CandidateArtifactRecord | None:
    """Resolve candidate snapshot by (cand_sha, regime_tag), empty-regime, or flat fallback."""
    sha, rtag = cand_sha.strip(), regime_tag.strip()
    return (by_regime.get((sha, rtag)) if rtag else None) or by_regime.get((sha, "")) or flat.get(sha)


def _append_jsonl(path: Path, record: Mapping[str, object], *, compact: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({str(k): v for k, v in record.items()}, sort_keys=True, **({"separators": (",", ":")} if compact else {})) + "\n")
    return path


def _iter_jsonl_mappings(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, object]] = []
    for raw in (ln.strip() for ln in lines if ln.strip()):
        try:
            if isinstance(decoded := json.loads(raw), Mapping):
                out.append({str(k): v for k, v in decoded.items()})
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


class ExperienceLake:
    """Thread/process-safe content-addressed candidate + sharded append-only observation store."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.candidates_dir, self.runs_dir = self.root_dir / "candidates", self.root_dir / "runs"

    def _candidate_path(self, cand_sha: str, regime_tag: str) -> Path:
        sha, rtag = _safe_slug(cand_sha, "unknown"), regime_tag.strip()
        return self.candidates_dir / (f"{sha}_{_safe_slug(rtag)}.json" if rtag else f"{sha}.json")

    def register_candidate(self, record: CandidateArtifactRecord) -> Path:
        """Persist candidate static artifact atomically per (cand_sha, regime_tag)."""
        target, to_write = self._candidate_path(record.cand_sha, record.regime_tag), record
        if target.is_file():
            try:
                existing = CandidateArtifactRecord.from_dict(json.loads(target.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                existing = None
            if existing is not None:
                if (existing.bisheng_summary or not record.bisheng_summary) and (existing.base_ledger or not record.base_ledger):
                    return target
                updates = {k: getattr(record, k) or getattr(existing, k) for k in ("operator", "regime_tag", "backend_track", "parent_sha", "worker_kind", "dsl_line_count")}
                updates.update({k: dict(getattr(record, k) or getattr(existing, k)) for k in ("config_knobs", "topology_axes", "base_ledger", "microarch_profile", "bisheng_summary")})
                to_write = replace(record, rule_id=record.rule_id if record.rule_id is not None else existing.rule_id, subkernel_count=max(record.subkernel_count, existing.subkernel_count), schema_version=max(record.schema_version, existing.schema_version), **updates)
        atomic_write_text(target, json.dumps(to_write.to_dict(), indent=2, sort_keys=True) + "\n")
        return target

    def append_observation(self, record: CaseObservationRecord) -> Path:
        return _append_jsonl(self.runs_dir / f"{_safe_slug(record.run_id, 'default')}.jsonl", record.to_dict(), compact=False)

    def append_unmodeled_anomaly(self, record: Mapping[str, object]) -> Path:
        return _append_jsonl(self.root_dir / "unmodeled_anomalies.jsonl", record, compact=True)

    def load_records_by_regime(self) -> tuple[dict[tuple[str, str], CandidateArtifactRecord], list[CaseObservationRecord]]:
        candidates: dict[tuple[str, str], CandidateArtifactRecord] = {}
        if self.candidates_dir.is_dir():
            for cpath in sorted(self.candidates_dir.glob("*.json")):
                try:
                    rec = CandidateArtifactRecord.from_dict(json.loads(cpath.read_text(encoding="utf-8")))
                    key = (rec.cand_sha, rec.regime_tag.strip())
                    candidates[key] = _prefer_richer(candidates.get(key), rec)
                except Exception:
                    continue
        observations: list[CaseObservationRecord] = []
        if self.runs_dir.is_dir():
            for rpath in sorted(self.runs_dir.glob("*.jsonl")):
                for row in _iter_jsonl_mappings(rpath):
                    try:
                        if (obs := CaseObservationRecord.from_dict(row)).measured_us > 0.0:
                            observations.append(obs)
                    except Exception:
                        continue
        observations.sort(key=lambda o: (o.timestamp, o.operator, o.case_num, o.cand_sha))
        return candidates, observations

    def load_records(self) -> tuple[dict[str, CandidateArtifactRecord], list[CaseObservationRecord]]:
        by_regime, observations = self.load_records_by_regime()
        flat: dict[str, CandidateArtifactRecord] = {}
        for _, rec in sorted(by_regime.items()):
            flat[rec.cand_sha] = _prefer_richer(flat.get(rec.cand_sha), rec)
        return flat, observations

    def extract_pairwise_deltas(self, *, exclude_run_id: str | None = None) -> list[tuple[CandidateArtifactRecord, CaseObservationRecord, CandidateArtifactRecord, CaseObservationRecord]]:
        """Match same-(operator, regime_tag, case_num) candidate-vs-parent measurement pairs."""
        by_regime, observations = self.load_records_by_regime()
        flat: dict[str, CandidateArtifactRecord] = {}
        for _, rec in sorted(by_regime.items()):
            flat[rec.cand_sha] = _prefer_richer(flat.get(rec.cand_sha), rec)
        latest = {(o.cand_sha, o.operator, o.regime_tag, o.case_num): o for o in observations if o.measured_us > 0.0 and not (exclude_run_id and o.run_id == exclude_run_id)}
        pairs, seen = [], set()
        for (_, op, rtag, cnum), cand_obs in sorted(latest.items()):
            cand_art = resolve_candidate_for_regime(by_regime, flat, cand_obs.cand_sha, cand_obs.regime_tag)
            psha = cand_obs.parent_sha.strip() or (cand_art.parent_sha.strip() if cand_art is not None else "")
            parent_obs = latest.get((psha, op, rtag, cnum)) if (psha and psha != cand_obs.cand_sha) else None
            parent_art = resolve_candidate_for_regime(by_regime, flat, psha, parent_obs.regime_tag) if (parent_obs and parent_obs.measured_us > 0.0) else None
            key = (cand_obs.cand_sha, psha, op, rtag, cnum)
            if cand_art is not None and parent_art is not None and parent_obs is not None and key not in seen:
                seen.add(key)
                pairs.append((cand_art, cand_obs, parent_art, parent_obs))
        return pairs


def matches_operator_filter(record_op: str, target_op: str | None) -> bool:
    """Match operator names supporting optional category/name suffix shorthand."""
    tgt = str(target_op or "").strip().lower()
    if not tgt:
        return True
    rec = str(record_op or "").strip().lower()
    return rec == tgt or (("/" not in tgt or "/" not in rec) and rec.rsplit("/", 1)[-1] == tgt.rsplit("/", 1)[-1])


def append_counterexample_boundary(lake_dir: Path, record: Mapping[str, object]) -> Path:
    return _append_jsonl(Path(lake_dir) / "counterexample_anchors.jsonl", record, compact=True)


def load_counterexample_boundaries(lake_dir: Path, operator: str | None = None) -> list[dict[str, object]]:
    in_path = Path(lake_dir) / "counterexample_anchors.jsonl"
    return [item for item in _iter_jsonl_mappings(in_path) if operator is None or matches_operator_filter(str(item.get("operator", "")), operator)] if in_path.is_file() else []


__all__ = [
    "CandidateArtifactRecord", "CaseObservationRecord", "ExperienceLake",
    "append_counterexample_boundary", "compute_source_sha256", "extract_ub_occupancy_ratio",
    "load_counterexample_boundaries", "matches_operator_filter", "project_case_workload",
    "resolve_candidate_for_regime", "resolve_default_experience_lake_dir",
]
