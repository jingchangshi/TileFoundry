"""Offline calibration, LOCO/LOOO backtest, and P2 prescreen ablation on CANNBench reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures,
    CostModelEvalReport,
    HybridAscendCostModel,
    _cube_macs_for_dtype,
    evaluate_cost_model_ranking,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirRooflineSummary
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import (
    DTYPE_BYTES,
    TopologyAxes,
    flatten_int_shapes,
    flatten_str_values,
    has_unaligned_tail_for_case,
    infer_case_dims_structural,
    shape_numel,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_CASE_NUM_RE = re.compile(r"_(\d+)$")
DEFAULT_AIV_VEC_FLOPS_PER_CYCLE = 256


@dataclass(frozen=True)
class PrescreenAblationSample:
    """Single paired A/B ablation sample comparing prescreen-off vs prescreen-on."""

    case_id: str
    baseline_score: float
    prescreen_score: float
    is_anchor: bool = False
    baseline_latency_us: float | None = None
    prescreen_latency_us: float | None = None

    @property
    def gain(self) -> float:
        """Return net score improvement (positive means prescreen improved over baseline)."""
        return float(self.prescreen_score) - float(self.baseline_score)

    @property
    def is_regression(self) -> bool:
        """Return True if enabling prescreen degraded score relative to baseline."""
        return self.gain < 0.0

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["gain"] = self.gain
        out["is_regression"] = self.is_regression
        return out


@dataclass(frozen=True)
class PrescreenAblationReport:
    """A/B ablation gate report controlling whether prescreener activation is allowed."""

    recommended_enable: bool
    sample_count: int
    anchor_count: int
    mean_gain: float
    min_gain: float
    max_gain: float
    regression_count: int
    anchor_regression_count: int
    anchor_regressions: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["anchor_regressions"] = list(self.anchor_regressions)
        out["reasons"] = list(self.reasons)
        return out


def evaluate_prescreen_ablation(
    samples: Sequence[PrescreenAblationSample | Mapping[str, object]],
    *,
    min_mean_gain: float = 0.01,
    allow_anchor_regressions: bool = False,
) -> PrescreenAblationReport:
    """Gate tilesize_prescreen activation on net gain and zero anchor regressions."""
    coerced: list[PrescreenAblationSample] = []
    for item in samples:
        if isinstance(item, PrescreenAblationSample):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(
                PrescreenAblationSample(
                    case_id=str(item["case_id"]),
                    baseline_score=float(item["baseline_score"]),
                    prescreen_score=float(item["prescreen_score"]),
                    is_anchor=bool(item.get("is_anchor", False)),
                    baseline_latency_us=(
                        float(item["baseline_latency_us"])
                        if item.get("baseline_latency_us") is not None
                        else None
                    ),
                    prescreen_latency_us=(
                        float(item["prescreen_latency_us"])
                        if item.get("prescreen_latency_us") is not None
                        else None
                    ),
                )
            )
        else:
            raise TypeError(f"Expected PrescreenAblationSample or Mapping, got {type(item).__name__}")

    if not coerced:
        return PrescreenAblationReport(
            recommended_enable=False,
            sample_count=0,
            anchor_count=0,
            mean_gain=0.0,
            min_gain=0.0,
            max_gain=0.0,
            regression_count=0,
            anchor_regression_count=0,
            anchor_regressions=(),
            reasons=("no_ablation_samples_keep_disabled",),
        )

    gains = [s.gain for s in coerced]
    mean_gain = sum(gains) / float(len(gains))
    min_gain = min(gains)
    max_gain = max(gains)
    regressions = [s for s in coerced if s.is_regression]
    anchors = [s for s in coerced if s.is_anchor]
    anchor_regs = tuple(s.case_id for s in anchors if s.is_regression)

    reasons: list[str] = []
    passed = True
    if mean_gain < float(min_mean_gain):
        passed = False
        reasons.append(f"mean_gain={mean_gain:.4f} < min_mean_gain={float(min_mean_gain):.4f}")
    if anchor_regs and not allow_anchor_regressions:
        passed = False
        reasons.append(f"anchor_regressions={list(anchor_regs)}")
    if passed:
        reasons.append(
            f"ablation_passed: mean_gain={mean_gain:.4f}, anchor_regressions={len(anchor_regs)}"
        )

    return PrescreenAblationReport(
        recommended_enable=passed,
        sample_count=len(coerced),
        anchor_count=len(anchors),
        mean_gain=mean_gain,
        min_gain=min_gain,
        max_gain=max_gain,
        regression_count=len(regressions),
        anchor_regression_count=len(anchor_regs),
        anchor_regressions=anchor_regs,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class OfficialCaseRecord:
    """One deduplicated NPU benchmark measurement paired with leak-free CostFeatures."""

    report_path: str
    report_signature: str
    rel_path: str
    operator: str
    case_id: str
    case_num: int
    measured_us: float
    t_hw_us: float
    ref_perf_us: float
    speedup: float
    perf_score: float
    op_score: float
    topology_bucket: str
    features: CostFeatures
    legacy_features: CostFeatures


@dataclass(frozen=True)
class BacktestSummary:
    """Side-by-side empirical CostModel evaluation across CV protocols and P2 ablation."""

    total_files_scanned: int
    dedup_reports: int
    unique_operator_cases: int
    operators_covered: tuple[str, ...]
    legacy_baseline: CostModelEvalReport
    p0_physics_only: CostModelEvalReport
    p0_p1_loco_cv: CostModelEvalReport
    p0_p1_looo_cv: CostModelEvalReport
    per_operator_looo: dict[str, dict[str, float | int | str]]
    p2_prescreen_ablation: PrescreenAblationReport
    enable_prescreen_recommended: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize backtest summary into JSON-safe dictionary."""
        return {
            "total_files_scanned": self.total_files_scanned,
            "dedup_reports": self.dedup_reports,
            "unique_operator_cases": self.unique_operator_cases,
            "operators_covered": list(self.operators_covered),
            "legacy_baseline": self.legacy_baseline.to_dict(),
            "p0_physics_only": self.p0_physics_only.to_dict(),
            "p0_p1_loco_cv": self.p0_p1_loco_cv.to_dict(),
            "p0_p1_looo_cv": self.p0_p1_looo_cv.to_dict(),
            "per_operator_looo": self.per_operator_looo,
            "p2_prescreen_ablation": self.p2_prescreen_ablation.to_dict(),
            "enable_prescreen_recommended": self.enable_prescreen_recommended,
        }


def _parse_case_num(case_id: str, fallback_idx: int) -> int:
    m = _CASE_NUM_RE.search(case_id.strip())
    if m:
        return int(m.group(1))
    return fallback_idx + 1


def _normalize_cube_dtype(raw: str) -> str:
    val = raw.strip().lower()
    if val in ("float16", "fp16", "half", "f16"):
        return "f16"
    if val in ("bfloat16", "bf16"):
        return "bf16"
    if val in ("float32", "fp32", "f32"):
        return "f32"
    if val in ("int8", "i8"):
        return "i8"
    if "float8" in val or "fp8" in val:
        return "fp8"
    if "int4" in val or "float4" in val:
        return "i4"
    return "f16"


@dataclass(frozen=True)
class _CachedOperatorMetadata:
    """Cached semantic topology and case metadata resolved from cann-bench tasks."""

    entry_name: str
    proto: dict[str, object]
    cases_by_id: dict[int, dict[str, object]]
    input_dtypes: tuple[str, ...]
    semantics: PyTorchOpSemantics | None
    axes: TopologyAxes
    fallback_dominant_bound: str


def _has_task_entries(root: Path) -> bool:
    if (root / "suite_manifest.json").is_file():
        return True
    if any(p.is_dir() for p in root.glob("level*")):
        return True
    return any((child / "cases.yaml").is_file() for child in root.iterdir() if child.is_dir())


def _try_resolve_tasks_root(ops_root: str | Path | None) -> Path | None:
    """Resolve cann-bench tasks directory using pure filesystem discovery."""
    if ops_root is not None:
        p = Path(ops_root).expanduser().resolve()
        if not p.is_dir():
            raise ValueError(f"Canonical ops_root does not exist: {p}")
        sub_tasks = p / "tasks"
        if sub_tasks.is_dir() and not _has_task_entries(p):
            return sub_tasks.resolve()
        return p

    candidates: list[Path] = []
    env_root = os.environ.get("TILEFOUNDRY_CANNBENCH_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        candidates.extend((cand / "tasks", cand))
    repo_root = Path(__file__).resolve().parents[5]
    candidates.extend(
        (
            repo_root.parent / "cann-bench" / "tasks",
            Path.home() / "workspace" / "cann-bench" / "tasks",
            repo_root / "tests" / "ascend" / "benchmarks",
        )
    )
    for cand in candidates:
        if cand.is_dir() and _has_task_entries(cand):
            return cand.resolve()
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return None


def _resolve_operator_dir(tasks_root: Path, rel_path: str) -> tuple[str, Path]:
    """Locate one operator directory under *tasks_root* across nested or flat layouts."""
    cleaned = str(rel_path).strip().removeprefix("tasks/").strip("/")
    tail = cleaned.rsplit("/", 1)[-1] if "/" in cleaned else cleaned

    manifest_path = tasks_root / "suite_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("operators", ()) or ():
            if not isinstance(row, Mapping):
                continue
            row_id = str(row.get("id") or "").strip()
            raw_row_path = str(row.get("path") or "").strip().removeprefix("tasks/").strip("/")
            if cleaned in (row_id, raw_row_path) or tail == row_id:
                cand = tasks_root / raw_row_path
                if cand.is_dir():
                    return row_id or tail, cand

    for cand in (
        tasks_root / cleaned,
        tasks_root / "tasks" / cleaned,
        tasks_root / tail,
    ):
        if cand.is_dir() and (
            (cand / "cases.yaml").is_file()
            or (cand / "proto.yaml").is_file()
            or (cand / "golden.py").is_file()
        ):
            return tail, cand

    for level_dir in sorted(tasks_root.glob("level*")):
        if not level_dir.is_dir():
            continue
        cand = level_dir / tail
        if cand.is_dir():
            return tail, cand

    raise KeyError(f"Operator task directory for {rel_path!r} not found under {tasks_root}")


def _infer_axes_from_proto(proto: Mapping[str, object]) -> tuple[TopologyAxes, str]:
    """Derive structural TopologyAxes and dominant bound from proto metadata without model.py."""
    category = str(proto.get("category") or "").strip().lower()
    formula = str(proto.get("formula") or "").strip().lower()
    combined = f"{category} {formula}"
    has_matmul = any(
        tok in combined
        for tok in ("contraction", "matmul", "gemm", "bmm", "linear", "@")
    )
    has_softmax_or_attn = any(tok in combined for tok in ("attention", "softmax"))
    has_indirect = any(
        tok in combined for tok in ("gather", "scatter", "index", "embedding", "topk")
    )
    if has_matmul and has_softmax_or_attn:
        return (
            TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
            "compute",
        )
    if has_matmul:
        return (
            TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd"),
            "compute",
        )
    if has_indirect:
        return (
            TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt"),
            "memory",
        )
    return (
        TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        "memory",
    )


def _proto_input_dtypes(proto: Mapping[str, object]) -> tuple[str, ...]:
    raw_inputs = proto.get("inputs")
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
        return ()
    dtypes: list[str] = []
    for item in raw_inputs:
        if not isinstance(item, Mapping):
            continue
        raw_dt = item.get("dtype")
        values = flatten_str_values(raw_dt)
        if values:
            dtypes.append(values[0])
    return tuple(dtypes)


def _load_cached_operator(
    rel_path: str,
    *,
    tasks_root: Path | None,
    analyzer: PyTorchSemanticAnalyzer,
    operator_cache: dict[str, _CachedOperatorMetadata],
) -> _CachedOperatorMetadata:
    if rel_path in operator_cache:
        return operator_cache[rel_path]
    if tasks_root is None:
        raise ValueError(
            f"Canonical tasks root unavailable to resolve topology_axes for {rel_path!r}"
        )
    op_id, op_dir = _resolve_operator_dir(tasks_root, rel_path)

    proto_path = op_dir / "proto.yaml"
    proto_raw = (
        yaml.safe_load(proto_path.read_text(encoding="utf-8"))
        if proto_path.is_file()
        else {}
    )
    proto_block = (
        proto_raw.get("operator")
        if isinstance(proto_raw, Mapping) and isinstance(proto_raw.get("operator"), Mapping)
        else (proto_raw if isinstance(proto_raw, Mapping) else {})
    )
    proto: dict[str, object] = {str(k): v for k, v in proto_block.items()}

    cases_path = op_dir / "cases.yaml"
    cases_raw = (
        yaml.safe_load(cases_path.read_text(encoding="utf-8"))
        if cases_path.is_file()
        else {}
    )
    raw_cases = cases_raw.get("cases") if isinstance(cases_raw, Mapping) else ()
    cases_by_id: dict[int, dict[str, object]] = {}
    if isinstance(raw_cases, Sequence) and not isinstance(raw_cases, (str, bytes)):
        for idx, entry in enumerate(raw_cases):
            if not isinstance(entry, Mapping):
                continue
            raw_cid = entry.get("case_id", idx + 1)
            cid = (
                int(raw_cid)
                if isinstance(raw_cid, int) and not isinstance(raw_cid, bool)
                else idx + 1
            )
            cases_by_id[cid] = {str(k): v for k, v in entry.items()}

    entry_name = str(proto.get("name") or op_id).strip() or op_id
    semantics: PyTorchOpSemantics | None = None
    golden_path = op_dir / "golden.py"
    if golden_path.is_file():
        golden_source = golden_path.read_text(encoding="utf-8")
        semantics = analyzer.analyze_source(
            golden_source,
            entry_name=entry_name,
            proto=proto,
        )
        axes = semantics.base_axes
        fallback_bound = "compute" if axes.uses_cube else "memory"
    else:
        axes, fallback_bound = _infer_axes_from_proto(proto)
    cached = _CachedOperatorMetadata(
        entry_name=entry_name,
        proto=proto,
        cases_by_id=cases_by_id,
        input_dtypes=_proto_input_dtypes(proto),
        semantics=semantics,
        axes=axes,
        fallback_dominant_bound=fallback_bound,
    )
    operator_cache[rel_path] = cached
    return cached


def _extract_case_dims_for_blueprint(
    case_entry: Mapping[str, object],
    axes: TopologyAxes,
) -> dict[str, int]:
    """Derive structural anchor dims for one case entry using ascend.sil.schema helpers."""
    shapes = flatten_int_shapes(case_entry.get("input_shape"))
    raw_attrs = case_entry.get("attrs")
    attrs: Mapping[str, object] = raw_attrs if isinstance(raw_attrs, Mapping) else {}
    dims = dict(
        infer_case_dims_structural(
            shapes,
            attrs,
            axes=axes,
            uses_cube=axes.uses_cube,
        )
    )
    for shape_idx, shape in enumerate(shapes):
        for axis_idx, extent in enumerate(shape):
            if extent > 0:
                dims.setdefault(f"shape_{shape_idx}_dim_{axis_idx}", int(extent))
    if shapes and shapes[0]:
        first = shapes[0]
        dims.setdefault("total_elements", shape_numel(first))
        if len(first) >= 2:
            dims.setdefault("M", shape_numel(first[:-1]))
            dims.setdefault("D", int(first[-1]))
            dims.setdefault("K", int(first[-1]))
    if len(shapes) >= 2 and shapes[1]:
        second = shapes[1]
        if len(second) >= 2:
            dims.setdefault("N", int(second[-1]))
            if len(second) == 3:
                dims.setdefault("E", int(second[0]))
    return dims


def _case_roofline_from_semantics(
    cached: _CachedOperatorMetadata,
    case_entry: Mapping[str, object],
    *,
    analyzer: PyTorchSemanticAnalyzer,
    fallback_dtype_bytes: float,
) -> HirRooflineSummary | None:
    if cached.semantics is None or not case_entry:
        return None
    int_dims = _extract_case_dims_for_blueprint(case_entry, cached.axes)
    try:
        rep_case = RepresentativeCase.from_case_entry(case_entry)
    except ValueError:
        rep_case = None
    try:
        packet = analyzer.build_universal_blueprint(
            cached.semantics,
            int_dims,
            dtype_bytes=fallback_dtype_bytes,
            representative_case=rep_case,
        )
    except ValueError:
        return None
    return packet.roofline


def _build_static_case_features(
    rel_path: str,
    case_num: int,
    t_hw_us: float,
    *,
    facts: AscendSpaceFacts,
    analyzer: PyTorchSemanticAnalyzer,
    operator_cache: dict[str, _CachedOperatorMetadata],
    tasks_root: Path | None,
) -> tuple[CostFeatures, CostFeatures, float]:
    """Construct leak-free P0+P1 and legacy baseline CostFeatures from semantic metadata."""
    cached = _load_cached_operator(
        rel_path,
        tasks_root=tasks_root,
        analyzer=analyzer,
        operator_cache=operator_cache,
    )
    case_entry = cached.cases_by_id.get(case_num, {})
    shapes = flatten_int_shapes(case_entry.get("input_shape")) if case_entry else ()
    dtypes = list(flatten_str_values(case_entry.get("dtype"))) if case_entry else []
    if not dtypes:
        dtypes = list(cached.input_dtypes)

    raw_dtype = dtypes[0] if dtypes else "float16"
    cube_dtype = _normalize_cube_dtype(raw_dtype)
    elem_bytes = float(DTYPE_BYTES.get(raw_dtype, 2.0))

    input_bytes = 0.0
    for idx, shp in enumerate(shapes):
        dt = dtypes[idx] if idx < len(dtypes) else raw_dtype
        input_bytes += float(shape_numel(shp)) * float(DTYPE_BYTES.get(dt, elem_bytes))

    axes = cached.axes
    roofline = _case_roofline_from_semantics(
        cached,
        case_entry,
        analyzer=analyzer,
        fallback_dtype_bytes=elem_bytes,
    )
    uses_cube = axes.compute_engine in ("aic_only", "coupled_1c2v")
    is_coupled = axes.compute_engine == "coupled_1c2v"
    dom_bound = (
        roofline.dominant_bound if roofline is not None else cached.fallback_dominant_bound
    )
    unaligned_tail = has_unaligned_tail_for_case(shapes, dtypes, uses_cube) if shapes else False

    if shapes and shapes[0]:
        inner_dim = int(shapes[0][-1] or 1)
        contig_burst = max(32, min(32768, int(round(inner_dim * elem_bytes))))
    else:
        contig_burst = 32768
    total_tiles = 0

    resolved_t_hw = float(t_hw_us)
    if resolved_t_hw <= 0.0:
        if roofline is not None and roofline.ideal_roofline_us > 0.0:
            resolved_t_hw = float(roofline.ideal_roofline_us)
        else:
            resolved_t_hw = 0.5

    clock_hz = float(facts.clock_ghz) * 1e9
    hbm_bw = float(facts.effective_hbm_bandwidth_bps(contig_burst))
    macs_per_cycle = float(_cube_macs_for_dtype(facts, cube_dtype, hf32=False))
    cube_peak_flops = float(facts.num_aic_cores) * macs_per_cycle * 2.0 * clock_hz
    vec_peak_flops = float(facts.num_aiv_cores) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * clock_hz

    safe_t_hw = max(0.05, resolved_t_hw)
    if roofline is not None:
        static_gm = max(4096.0, float(roofline.min_gm_bytes))
        cube_flops = float(roofline.cube_flops)
        vec_flops = float(roofline.vector_flops)
        if uses_cube:
            fixpipe_bytes = (0.25 * static_gm) if is_coupled else (0.15 * static_gm)
            cv_sync_events = 4 if is_coupled else 0
        else:
            fixpipe_bytes = 0.0
            cv_sync_events = 0
    else:
        static_gm = (
            max(4096.0, input_bytes * 1.25)
            if input_bytes > 0.0
            else max(4096.0, safe_t_hw * 1e-6 * hbm_bw)
        )
        if uses_cube:
            cube_flops = safe_t_hw * 1e-6 * cube_peak_flops
            vec_flops = (0.25 * safe_t_hw * 1e-6 * vec_peak_flops) if is_coupled else 0.0
            fixpipe_bytes = (0.25 * static_gm) if is_coupled else (0.15 * static_gm)
            cv_sync_events = 4 if is_coupled else 0
        elif dom_bound in ("vector_compute", "compute") or (
            axes.memory_ring == "persistent_onchip_state"
        ):
            cube_flops = 0.0
            vec_flops = safe_t_hw * 1e-6 * vec_peak_flops
            fixpipe_bytes = 0.0
            cv_sync_events = 0
        else:
            cube_flops = 0.0
            static_gm = max(static_gm, safe_t_hw * 1e-6 * hbm_bw)
            vec_flops = 0.35 * safe_t_hw * 1e-6 * vec_peak_flops
            fixpipe_bytes = 0.0
            cv_sync_events = 0

    modern = CostFeatures(
        cube_flops=cube_flops,
        vec_flops=vec_flops,
        cube_dtype=cube_dtype,
        gm_bytes=static_gm,
        contig_burst_bytes=contig_burst,
        total_tiles=total_tiles,
        has_unaligned_tail=unaligned_tail,
        kernel_launches=1,
        fixpipe_bytes=fixpipe_bytes,
        cv_sync_events=cv_sync_events,
        topology_axes=axes,
    )
    legacy = CostFeatures(
        cube_flops=cube_flops,
        vec_flops=vec_flops,
        cube_dtype=cube_dtype,
        gm_bytes=static_gm,
        contig_burst_bytes=32768,
        total_tiles=0,
        has_unaligned_tail=False,
        kernel_launches=0,
        launch_overhead_us=0.0,
        fixpipe_bytes=0.0,
        cv_sync_events=0,
        topology_axes=None,
    )
    return modern, legacy, resolved_t_hw


def load_official_reports(
    report_roots: Sequence[str | Path],
    *,
    ops_root: str | Path | None = None,
    target: AscendTarget | None = None,
) -> tuple[list[OfficialCaseRecord], int, int]:
    """Scan report roots for official_final.json, deduplicate copies, and build case records."""
    eff_target = target or AscendTarget("ascend.950pr")
    facts = eff_target.get_facts(AscendSpaceFacts)
    if not isinstance(facts, AscendSpaceFacts):
        raise RuntimeError(f"Expected AscendSpaceFacts, got {type(facts).__name__}")
    tasks_root = _try_resolve_tasks_root(ops_root)
    analyzer = PyTorchSemanticAnalyzer(eff_target)
    operator_cache: dict[str, _CachedOperatorMetadata] = {}

    report_files: list[Path] = []
    for raw_root in report_roots:
        rp = Path(raw_root).expanduser().resolve()
        if rp.is_file() and rp.name == "official_final.json":
            report_files.append(rp)
        elif rp.is_dir():
            report_files.extend(sorted(rp.rglob("official_final.json")))

    seen_signatures: set[tuple[str, tuple[tuple[str, float], ...]]] = set()
    records: list[OfficialCaseRecord] = []
    dedup_report_count = 0

    for report_file in sorted(set(report_files)):
        try:
            payload = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for op_entry in payload.get("operators", []) or []:
            if not isinstance(op_entry, Mapping):
                continue
            rel_path = str(op_entry.get("rel_path") or "").strip()
            operator = str(op_entry.get("operator") or rel_path).strip()
            op_score = float(op_entry.get("score") or 0.0)
            cases = [c for c in (op_entry.get("cases") or []) if isinstance(c, Mapping)]
            passed_cases: list[Mapping[str, object]] = []
            for c in cases:
                acc = c.get("accuracy")
                acc_ok = bool(acc.get("passed")) if isinstance(acc, Mapping) else False
                raw_elapsed = c.get("elapsed_us")
                elapsed = (
                    float(raw_elapsed)
                    if isinstance(raw_elapsed, (int, float)) and not isinstance(raw_elapsed, bool)
                    else 0.0
                )
                if acc_ok and math.isfinite(elapsed) and elapsed > 0.0:
                    passed_cases.append(c)
            if not passed_cases:
                continue

            sig_cases = tuple(
                sorted(
                    (str(c.get("case_id") or ""), round(float(c["elapsed_us"]), 4))
                    for c in passed_cases
                )
            )
            sig_key = (rel_path, sig_cases)
            if sig_key in seen_signatures:
                continue
            seen_signatures.add(sig_key)
            dedup_report_count += 1
            sig_label = f"{rel_path}#{len(seen_signatures)}"

            for idx, c in enumerate(passed_cases):
                case_id = str(c.get("case_id") or f"{rel_path}_{idx + 1}")
                case_num = _parse_case_num(case_id, idx)
                measured_us = float(c["elapsed_us"])
                raw_t_hw = c.get("t_hw_us")
                t_hw_us = (
                    float(raw_t_hw)
                    if isinstance(raw_t_hw, (int, float)) and not isinstance(raw_t_hw, bool)
                    else 0.0
                )
                ref_perf_us = float(c.get("baseline_perf_us") or 0.0)
                speedup = float(c.get("speedup") or 0.0)
                perf_score = float(c.get("perf_score") or 0.0)

                feat, leg_feat, resolved_t_hw = _build_static_case_features(
                    rel_path,
                    case_num,
                    t_hw_us,
                    facts=facts,
                    analyzer=analyzer,
                    operator_cache=operator_cache,
                    tasks_root=tasks_root,
                )
                records.append(
                    OfficialCaseRecord(
                        report_path=str(report_file),
                        report_signature=sig_label,
                        rel_path=rel_path,
                        operator=operator,
                        case_id=case_id,
                        case_num=case_num,
                        measured_us=measured_us,
                        t_hw_us=resolved_t_hw,
                        ref_perf_us=ref_perf_us,
                        speedup=speedup,
                        perf_score=perf_score,
                        op_score=op_score,
                        topology_bucket=(
                            f"{feat.topology_axes.compute_engine}:{feat.topology_axes.memory_ring}:{feat.topology_axes.iteration_regularity}"
                            if feat.topology_axes is not None
                            else "global"
                        ),
                        features=feat,
                        legacy_features=leg_feat,
                    )
                )

    return records, len(set(report_files)), dedup_report_count


def _select_best_checkpoint_per_operator(
    records: Sequence[OfficialCaseRecord],
) -> list[OfficialCaseRecord]:
    """Keep the highest-scoring report checkpoint per rel_path so ops weigh equally."""
    by_op_sig: dict[str, dict[str, list[OfficialCaseRecord]]] = {}
    for rec in records:
        by_op_sig.setdefault(rec.rel_path, {}).setdefault(rec.report_signature, []).append(rec)

    selected: list[OfficialCaseRecord] = []
    for rel_path in sorted(by_op_sig):
        candidates = list(by_op_sig[rel_path].values())
        best_group = max(
            candidates,
            key=lambda grp: (len(grp), grp[0].op_score, grp[0].report_signature),
        )
        selected.extend(sorted(best_group, key=lambda r: r.case_num))
    return selected


def run_p2_prescreen_ablation(
    records: Sequence[OfficialCaseRecord],
    *,
    target: AscendTarget | None = None,
    anchor_rel_paths: Sequence[str] = (),
) -> PrescreenAblationReport:
    """Evaluate offline prescreen gain across multi-checkpoint operators using LOO fitting."""
    eff_target = target or AscendTarget("ascend.950pr")
    anchor_set = frozenset(str(p).strip() for p in anchor_rel_paths if str(p).strip())
    by_op_sig: dict[str, dict[str, list[OfficialCaseRecord]]] = {}
    for rec in records:
        by_op_sig.setdefault(rec.rel_path, {}).setdefault(rec.report_signature, []).append(rec)

    samples: list[PrescreenAblationSample] = []
    for rel_path, sig_map in sorted(by_op_sig.items()):
        if len(sig_map) < 2:
            continue
        other_records = [(r.features, r.measured_us) for r in records if r.rel_path != rel_path]
        model = HybridAscendCostModel(eff_target)
        model.observe_batch(other_records, persist=False)

        ckpt_scores: list[tuple[str, float, float, float]] = []
        for sig, grp in sorted(sig_map.items()):
            mean_pred = sum(model.predict(r.features).pred_us for r in grp) / float(len(grp))
            mean_lat = sum(r.measured_us for r in grp) / float(len(grp))
            ckpt_scores.append((sig, mean_pred, grp[0].op_score, mean_lat))

        baseline_mean_score = sum(item[2] for item in ckpt_scores) / float(len(ckpt_scores))
        baseline_mean_lat = sum(item[3] for item in ckpt_scores) / float(len(ckpt_scores))
        top_half_k = max(1, len(ckpt_scores) // 2)
        ranked_by_pred = sorted(ckpt_scores, key=lambda item: (item[1], item[0]))
        kept = ranked_by_pred[:top_half_k]
        prescreen_mean_score = sum(item[2] for item in kept) / float(len(kept))
        prescreen_mean_lat = sum(item[3] for item in kept) / float(len(kept))
        samples.append(
            PrescreenAblationSample(
                case_id=rel_path,
                baseline_score=baseline_mean_score,
                prescreen_score=prescreen_mean_score,
                is_anchor=(rel_path in anchor_set),
                baseline_latency_us=baseline_mean_lat,
                prescreen_latency_us=prescreen_mean_lat,
            )
        )

    return evaluate_prescreen_ablation(samples, min_mean_gain=0.5, allow_anchor_regressions=False)


def run_cost_model_backtest(
    report_roots: Sequence[str | Path],
    *,
    ops_root: str | Path | None = None,
    target: AscendTarget | None = None,
    anchor_rel_paths: Sequence[str] = (),
) -> BacktestSummary:
    """Run Legacy vs P0 Physics vs P0+P1 LOCO/LOOO cross-validation & P2 prescreen ablation."""
    eff_target = target or AscendTarget("ascend.950pr")
    all_records, scanned_count, dedup_reports = load_official_reports(
        report_roots,
        ops_root=ops_root,
        target=eff_target,
    )
    unique_records = _select_best_checkpoint_per_operator(all_records)
    if not unique_records:
        empty_eval = CostModelEvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        empty_ablation = evaluate_prescreen_ablation([])
        return BacktestSummary(
            total_files_scanned=scanned_count,
            dedup_reports=dedup_reports,
            unique_operator_cases=0,
            operators_covered=(),
            legacy_baseline=empty_eval,
            p0_physics_only=empty_eval,
            p0_p1_loco_cv=empty_eval,
            p0_p1_looo_cv=empty_eval,
            per_operator_looo={},
            p2_prescreen_ablation=empty_ablation,
            enable_prescreen_recommended=False,
        )

    actuals = [r.measured_us for r in unique_records]
    groups = [r.rel_path for r in unique_records]
    operators_covered = tuple(sorted(set(groups)))

    uncal_model = HybridAscendCostModel(eff_target)
    legacy_preds = [uncal_model.predict(r.legacy_features).pred_us for r in unique_records]
    p0_preds = [uncal_model.predict(r.features).pred_us for r in unique_records]

    loco_preds: list[float] = []
    batch_items = [(r.features, r.measured_us) for r in unique_records]
    for idx, rec in enumerate(unique_records):
        m = HybridAscendCostModel(eff_target)
        m.observe_batch([item for j, item in enumerate(batch_items) if j != idx], persist=False)
        loco_preds.append(m.predict(rec.features).pred_us)

    looo_preds = [0.0] * len(unique_records)
    looo_models: dict[str, HybridAscendCostModel] = {}
    for op_rel in operators_covered:
        m = HybridAscendCostModel(eff_target)
        m.observe_batch(
            [(tr.features, tr.measured_us) for tr in unique_records if tr.rel_path != op_rel],
            persist=False,
        )
        looo_models[op_rel] = m
    for idx, rec in enumerate(unique_records):
        looo_preds[idx] = looo_models[rec.rel_path].predict(rec.features).pred_us

    legacy_report = evaluate_cost_model_ranking(legacy_preds, actuals, groups=groups)
    p0_report = evaluate_cost_model_ranking(p0_preds, actuals, groups=groups)
    loco_report = evaluate_cost_model_ranking(loco_preds, actuals, groups=groups)
    looo_report = evaluate_cost_model_ranking(looo_preds, actuals, groups=groups)

    per_op_looo: dict[str, dict[str, float | int | str]] = {}
    for op_rel in operators_covered:
        idxs = [i for i, r in enumerate(unique_records) if r.rel_path == op_rel]
        op_preds = [looo_preds[i] for i in idxs]
        op_actuals = [actuals[i] for i in idxs]
        op_rep = evaluate_cost_model_ranking(op_preds, op_actuals, groups=[op_rel] * len(idxs))
        per_op_looo[op_rel] = {
            **op_rep.to_dict(),
            "topology_bucket": unique_records[idxs[0]].topology_bucket,
        }

    ablation = run_p2_prescreen_ablation(
        all_records,
        target=eff_target,
        anchor_rel_paths=anchor_rel_paths,
    )
    enable_ok = bool(
        loco_report.mean_intra_group_spearman_rho >= 0.85
        and looo_report.mean_intra_group_spearman_rho >= 0.85
        and ablation.recommended_enable
    )

    return BacktestSummary(
        total_files_scanned=scanned_count,
        dedup_reports=dedup_reports,
        unique_operator_cases=len(unique_records),
        operators_covered=operators_covered,
        legacy_baseline=legacy_report,
        p0_physics_only=p0_report,
        p0_p1_loco_cv=loco_report,
        p0_p1_looo_cv=looo_report,
        per_operator_looo=per_op_looo,
        p2_prescreen_ablation=ablation,
        enable_prescreen_recommended=enable_ok,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for CostModel offline calibration and backtest evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Ascend HybridCostModel P0-P2 on official CANNBench reports."
    )
    parser.add_argument(
        "--report-root",
        action="append",
        required=True,
        dest="report_roots",
        help="Directory or official_final.json path (repeatable).",
    )
    parser.add_argument("--ops-root", default=None, help="Optional cann-bench tasks/ root.")
    parser.add_argument("--target", default="ascend.950pr", help="Target hardware identifier.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON report path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CostModel P0-P2 offline backtest and print JSON report."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    summary = run_cost_model_backtest(
        args.report_roots,
        ops_root=args.ops_root,
        target=AscendTarget(args.target),
    )
    rendered = json.dumps(summary.to_dict(), indent=2)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


__all__ = [
    "BacktestSummary",
    "OfficialCaseRecord",
    "PrescreenAblationReport",
    "PrescreenAblationSample",
    "evaluate_prescreen_ablation",
    "load_official_reports",
    "run_cost_model_backtest",
    "run_p2_prescreen_ablation",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
