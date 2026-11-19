"""Lossless content-addressed and sharded-append Experience Lake for marathon flywheels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

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
    """Static candidate IR, schedule, compiler summary, and lineage snapshot (written once per sha)."""

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
    schema_version: int = 2

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
    """Parametrically project a single static candidate ledger onto concrete case dimensions (<1ms)."""
    eff_target = target or AscendTarget("ascend.950pr")
    facts = eff_target.get_facts(AscendSpaceFacts)
    prof = MicroarchProfile.coerce(microarch_profile)

    if base_ledger is None:
        ledger_map: dict[str, Any] = {}
    elif isinstance(base_ledger, Mapping):
        ledger_map = dict(base_ledger)
    elif hasattr(base_ledger, "to_dict") and callable(getattr(base_ledger, "to_dict")):
        ledger_map = dict(base_ledger.to_dict())
    else:
        ledger_map = {
            "grid_blocks": getattr(base_ledger, "grid_blocks", 1),
            "pipe_cycles": getattr(base_ledger, "pipe_cycles", {}),
            "gm_roundtrip_bytes": getattr(base_ledger, "gm_roundtrip_bytes", 0),
            "dominant_contig_bytes": getattr(base_ledger, "dominant_contig_bytes", 32768),
            "steady_state_ii": getattr(base_ledger, "steady_state_ii", 0),
            "waterlines": getattr(base_ledger, "waterlines", {}),
        }

    dims = dict(case_dims or {})
    pol = dict(policy or {})
    b_dim = _pick_dim(dims, ("B", "b_dim", "batch")) or 1
    m_dim = _pick_dim(dims, ("M", "m_dim", "m_rows", "S", "s_dim", "S_q", "seq_len")) or 0
    n_dim = _pick_dim(dims, ("N", "n_dim", "H", "num_heads", "N_q")) or 0
    k_dim = _pick_dim(dims, ("K", "k_dim", "D", "d_dim", "D_nope", "hidden_dim")) or 0
    s_kv = _pick_dim(dims, ("S_kv", "s_kv")) or 0
    g_dim = _pick_dim(dims, ("G", "E", "num_experts", "groups")) or 1

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
    if m_dim > 0 and n_dim > 0 and ("M" in dims or "m_dim" in dims or "N" in dims or "n_dim" in dims):
        proj_tiles = max(1, math.ceil(m_dim / max(1, bm)) * math.ceil(n_dim / max(1, bn)) * b_dim * g_dim)
    elif m_dim > 0:
        heads = _pick_dim(dims, ("H", "N_q", "num_heads")) or 1
        proj_tiles = max(1, b_dim * heads * math.ceil(m_dim / max(1, bm)))
    else:
        proj_tiles = base_grid

    dtype_name = str(pol.get("cube_operand_dtype") or pol.get("dtype") or "f16")
    elem_b = 4 if "32" in dtype_name else 2
    base_contig = int(ledger_map.get("dominant_contig_bytes", 0) or 32768)
    inner_dim = k_dim or n_dim or m_dim
    proj_contig = max(32, min(base_contig, inner_dim * elem_b)) if inner_dim > 0 else base_contig

    # Analytical GM bytes & FLOP projection for the specific case_dims
    case_gm = 0.0
    cube_flops = 0.0
    vec_flops = 0.0
    if m_dim > 0 and k_dim > 0 and n_dim > 0:
        eff_k = max(k_dim, s_kv)
        case_gm = float(b_dim * g_dim * (m_dim * eff_k + eff_k * n_dim + m_dim * n_dim) * elem_b)
    elif m_dim > 0 and (k_dim > 0 or n_dim > 0):
        width = max(k_dim, n_dim)
        case_gm = float(b_dim * g_dim * m_dim * width * 2 * elem_b)

    base_gm_tile = float(ledger_map.get("gm_roundtrip_bytes", 0) or 0)
    if base_gm_tile > 0.0:
        case_gm = max(case_gm, base_gm_tile * float(proj_tiles))
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
        cube_flops = block_macs * 2.0 * float(proj_tiles)
        if m_dim > 0 and k_dim > 0 and n_dim > 0:
            cube_flops = max(cube_flops, float(b_dim * g_dim * 2 * m_dim * n_dim * max(k_dim, s_kv)))
        if t_hw_us > 0.0:
            cube_peak = float(facts.num_aic_cores) * macs_per_cyc * 2.0 * clock_hz
            cube_flops = max(cube_flops, float(t_hw_us) * 1e-6 * cube_peak)

    if vec_c > 0 or prof.total_vector_ops > 0:
        vec_flops = max(float(vec_c) * 128.0, float(prof.total_vector_ops) * 128.0) * float(proj_tiles)
    elif t_hw_us > 0.0 and cube_flops <= 0.0:
        vec_peak = float(facts.num_aiv_cores) * 128.0 * clock_hz
        vec_flops = float(t_hw_us) * 1e-6 * vec_peak

    fixpipe_bytes = float(prof.l0c_to_ub_bytes + prof.l0c_to_l1_bytes + prof.l0c_to_gm_bytes)
    if fixpipe_bytes <= 0.0 and cube_flops > 0.0:
        fixpipe_bytes = 0.25 * case_gm
    else:
        fixpipe_bytes *= float(proj_tiles)

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
            "steady_state_ii": int(ledger_map.get("steady_state_ii") or 0),
            "loop_trips": max(1, prof.inner_loop_trips),
            "pipeline_depth": max(1, prof.pipeline_depth),
            "kernel_launches": max(1, prof.subkernel_count),
            "microarch_profile": prof.to_dict(),
        }
    )
    if axes is not None:
        out["topology_axes"] = axes
    return out


class ExperienceLake:
    """Thread/process-safe content-addressed candidate + sharded append-only observation store."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.candidates_dir = self.root_dir / "candidates"
        self.runs_dir = self.root_dir / "runs"

    def register_candidate(self, record: CandidateArtifactRecord) -> Path:
        """Persist candidate static artifact atomically once per cand_sha."""
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        sha = _safe_slug(record.cand_sha, "unknown")
        target = self.candidates_dir / f"{sha}.json"
        if target.is_file():
            return target
        payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
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

    def load_records(self) -> tuple[dict[str, CandidateArtifactRecord], list[CaseObservationRecord]]:
        """Load all deduplicated candidates and sharded case observations."""
        candidates: dict[str, CandidateArtifactRecord] = {}
        if self.candidates_dir.is_dir():
            for cpath in sorted(self.candidates_dir.glob("*.json")):
                try:
                    data = json.loads(cpath.read_text(encoding="utf-8"))
                    rec = CandidateArtifactRecord.from_dict(data)
                    candidates[rec.cand_sha] = rec
                except Exception:
                    continue

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
        return candidates, observations

    def extract_pairwise_deltas(
        self,
    ) -> list[
        tuple[
            CandidateArtifactRecord,
            CaseObservationRecord,
            CandidateArtifactRecord,
            CaseObservationRecord,
        ]
    ]:
        """Match same-(operator, case_num) candidate-vs-parent measurement pairs."""
        candidates, observations = self.load_records()
        latest_by_cand_case: dict[tuple[str, str, int], CaseObservationRecord] = {}
        for obs in observations:
            if obs.measured_us <= 0.0:
                continue
            latest_by_cand_case[(obs.cand_sha, obs.operator, obs.case_num)] = obs

        pairs: list[
            tuple[
                CandidateArtifactRecord,
                CaseObservationRecord,
                CandidateArtifactRecord,
                CaseObservationRecord,
            ]
        ] = []
        seen_pairs: set[tuple[str, str, str, int]] = set()
        for (_csha, op, cnum), cand_obs in sorted(latest_by_cand_case.items()):
            psha = cand_obs.parent_sha.strip()
            if not psha or psha == cand_obs.cand_sha:
                continue
            parent_obs = latest_by_cand_case.get((psha, op, cnum))
            if parent_obs is None or parent_obs.measured_us <= 0.0:
                continue
            cand_art = candidates.get(cand_obs.cand_sha)
            parent_art = candidates.get(psha)
            if cand_art is None or parent_art is None:
                continue
            key = (cand_obs.cand_sha, psha, op, cnum)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append((cand_art, cand_obs, parent_art, parent_obs))
        return pairs


__all__ = [
    "CandidateArtifactRecord",
    "CaseObservationRecord",
    "ExperienceLake",
    "compute_source_sha256",
    "project_case_workload",
    "resolve_default_experience_lake_dir",
]
