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
from typing import Any

import yaml

from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures,
    CostModelEvalReport,
    HybridAscendCostModel,
    _cube_macs_for_dtype,
    evaluate_cost_model_ranking,
)
from tilefoundry_ascend.ascend.sil.op_analyzer import (
    _DTYPE_BYTES,
    OpAdvisoryExtractor,
    _flatten_shapes,
    _flatten_strs,
    _has_unaligned_tail_for_case,
    _shape_numel,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
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


def _try_resolve_extractor(ops_root: str | Path | None) -> OpAdvisoryExtractor | None:
    if ops_root is not None:
        p = Path(ops_root).expanduser().resolve()
        if not p.is_dir():
            raise ValueError(f"Canonical ops_root does not exist: {p}")
        return OpAdvisoryExtractor(p)
    env_root = os.environ.get("TILEFOUNDRY_CANNBENCH_ROOT", "").strip()
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        for sub in (cand / "tasks", cand):
            if sub.is_dir():
                return OpAdvisoryExtractor(sub)
    default_ext = OpAdvisoryExtractor()
    if default_ext.ops_root.is_dir():
        return default_ext
    return None


def _build_static_case_features(
    rel_path: str,
    case_num: int,
    t_hw_us: float,
    *,
    facts: AscendSpaceFacts,
    advisory_cache: dict[str, tuple[Any, dict[int, dict[str, Any]]]],
    extractor: OpAdvisoryExtractor | None,
) -> tuple[CostFeatures, CostFeatures]:
    """Construct leak-free P0+P1 and legacy baseline CostFeatures from static metadata only."""
    op_adv = None
    case_map: dict[int, dict[str, Any]] = {}
    if extractor is not None:
        if rel_path not in advisory_cache:
            lvl_part, _, op_id = rel_path.partition("/")
            if not op_id:
                op_id, lvl_part = rel_path, ""
            adv = extractor.extract(op_id, level=lvl_part or None)
            _lvl, op_dir = extractor._find_op_entry(op_id, level=lvl_part or None)
            cases_raw = yaml.safe_load((op_dir / "cases.yaml").read_text(encoding="utf-8"))
            cmap: dict[int, dict[str, Any]] = {}
            for idx, entry in enumerate(cases_raw.get("cases", []) or []):
                cid = int(entry.get("case_id", idx + 1))
                cmap[cid] = dict(entry)
            advisory_cache[rel_path] = (adv, cmap)
        op_adv, case_map = advisory_cache[rel_path]

    case_entry = case_map.get(case_num, {})
    shapes = _flatten_shapes(case_entry.get("input_shape")) if case_entry else []
    dtypes = list(_flatten_strs(case_entry.get("dtype"))) if case_entry else []
    if not dtypes and op_adv is not None:
        dtypes = list(op_adv.input_dtypes)

    raw_dtype = dtypes[0] if dtypes else "float16"
    cube_dtype = _normalize_cube_dtype(raw_dtype)
    elem_bytes = float(_DTYPE_BYTES.get(raw_dtype, 2.0))

    input_bytes = 0.0
    for idx, shp in enumerate(shapes):
        dt = dtypes[idx] if idx < len(dtypes) else raw_dtype
        input_bytes += float(_shape_numel(shp)) * float(_DTYPE_BYTES.get(dt, elem_bytes))

    if op_adv is None:
        raise ValueError(
            f"Canonical OpAdvisoryExtractor failed to resolve topology_axes for {rel_path!r}"
        )
    axes: TopologyAxes = op_adv.topology_axes
    uses_cube = axes.compute_engine in ("aic_only", "coupled_1c2v")
    is_coupled = axes.compute_engine == "coupled_1c2v"
    dom_bound = str(op_adv.roofline.get("dominant_bound", "memory_bandwidth"))
    unaligned_tail = _has_unaligned_tail_for_case(shapes, dtypes, uses_cube) if shapes else False

    if shapes and shapes[0]:
        inner_dim = int(shapes[0][-1] or 1)
        contig_burst = max(32, min(32768, int(round(inner_dim * elem_bytes))))
    else:
        contig_burst = 32768
    total_tiles = 0

    clock_hz = float(getattr(facts, "clock_ghz", 1.65)) * 1e9
    if callable(getattr(facts, "effective_hbm_bandwidth_bps", None)):
        hbm_bw = float(facts.effective_hbm_bandwidth_bps(contig_burst))
    else:
        hbm_bw = float(getattr(facts, "hbm_peak_bandwidth_bps", 1.6e12))
    macs_per_cycle = float(_cube_macs_for_dtype(facts, cube_dtype, hf32=False))
    cube_peak_flops = float(facts.num_aic_cores) * macs_per_cycle * 2.0 * clock_hz
    vec_peak_flops = float(facts.num_aiv_cores) * float(DEFAULT_AIV_VEC_FLOPS_PER_CYCLE) * clock_hz

    safe_t_hw = max(0.05, float(t_hw_us))
    static_gm = max(4096.0, input_bytes * 1.25) if input_bytes > 0.0 else max(
        4096.0, safe_t_hw * 1e-6 * hbm_bw
    )

    if uses_cube:
        cube_flops = safe_t_hw * 1e-6 * cube_peak_flops
        vec_flops = (0.25 * safe_t_hw * 1e-6 * vec_peak_flops) if is_coupled else 0.0
        fixpipe_bytes = (0.25 * static_gm) if is_coupled else (0.15 * static_gm)
        cv_sync_events = 4 if is_coupled else 0
    elif dom_bound == "vector_compute" or (
        axes is not None and axes.memory_ring == "persistent_onchip_state"
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
    return modern, legacy


def load_official_reports(
    report_roots: Sequence[str | Path],
    *,
    ops_root: str | Path | None = None,
    target: AscendTarget | None = None,
) -> tuple[list[OfficialCaseRecord], int, int]:
    """Scan report roots for official_final.json, deduplicate copies, and build case records."""
    eff_target = target or AscendTarget("ascend.950pr")
    facts = eff_target.get_facts(AscendSpaceFacts)
    extractor = _try_resolve_extractor(ops_root)
    advisory_cache: dict[str, tuple[Any, dict[int, dict[str, Any]]]] = {}

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
            passed_cases: list[Mapping[str, Any]] = []
            for c in cases:
                acc = c.get("accuracy")
                acc_ok = bool(acc.get("passed")) if isinstance(acc, Mapping) else False
                elapsed = float(c.get("elapsed_us") or 0.0)
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
                t_hw_us = float(c.get("t_hw_us") or 0.0)
                if t_hw_us <= 0.0:
                    t_hw_us = 0.5
                ref_perf_us = float(c.get("baseline_perf_us") or 0.0)
                speedup = float(c.get("speedup") or 0.0)
                perf_score = float(c.get("perf_score") or 0.0)

                feat, leg_feat = _build_static_case_features(
                    rel_path,
                    case_num,
                    t_hw_us,
                    facts=facts,
                    advisory_cache=advisory_cache,
                    extractor=extractor,
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
                        t_hw_us=t_hw_us,
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
