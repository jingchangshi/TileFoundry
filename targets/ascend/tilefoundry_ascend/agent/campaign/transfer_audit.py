"""Cross-topology causal rule transfer matrix and system bottleneck diagnostic CLI."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


def format_topology_quadrant(axes: Mapping[str, object] | None) -> str:
    """Format a 3D TopologyAxes mapping into canonical quadrant key."""
    if not isinstance(axes, Mapping) or not axes:
        return "unknown_topology"
    ce = str(axes.get("compute_engine") or "unknown")
    mr = str(axes.get("memory_ring") or "unknown")
    ir = str(axes.get("iteration_regularity") or "unknown")
    return f"{ce}.{mr}.{ir}"


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


@dataclass(frozen=True)
class RuleQuadrantMetrics:
    """Aggregated empirical efficacy of one causal rule inside one TopologyAxes quadrant."""

    rule_id: str
    quadrant: str
    dispatches: int
    noop_count: int
    compile_pass_count: int
    l3_promoted_count: int
    noop_rate: float
    compile_pass_rate: float
    promotion_rate: float
    mean_delta_speedup: float
    total_wall_sec: float
    speedup_per_wall_min: float

    def to_dict(self) -> dict[str, object]:
        """Serialize metrics row into a JSON-compatible dictionary."""
        return asdict(self)


def load_distilled_rules_from_lake(lake_dir: Path) -> dict[str, dict[str, object]]:
    """Load distilled rule definitions from nested KB and flat Experience Lake layouts."""
    rules_by_id: dict[str, dict[str, object]] = {}
    candidate_dirs = (
        lake_dir / "kb" / "distilled_rules",
        lake_dir / "distilled_rules",
    )
    for rules_dir in candidate_dirs:
        if not rules_dir.is_dir():
            continue
        for rpath in sorted(rules_dir.glob("*.json")):
            try:
                parsed = json.loads(rpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            nested = parsed.get("rule")
            rule_obj = (
                dict(nested)
                if isinstance(nested, dict) and "rule_id" not in parsed
                else dict(parsed)
            )
            rid = str(rule_obj.get("rule_id") or rpath.stem).strip()
            if rid:
                rules_by_id.setdefault(rid, rule_obj)
    return rules_by_id


def load_winning_edges_from_lake(lake_dir: Path) -> list[dict[str, object]]:
    """Load winning optimization edges from nested KB and flat Experience Lake files."""
    edges: list[dict[str, object]] = []
    candidate_files = (
        lake_dir / "kb" / "evidence" / "winning_edges.jsonl",
        lake_dir / "winning_edges.jsonl",
    )
    for edge_file in candidate_files:
        if not edge_file.is_file():
            continue
        try:
            lines = edge_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                edges.append(parsed)
    return edges


def _extract_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _build_rule_provenance(
    distilled_rules: Mapping[str, Mapping[str, object]],
    winning_edges: Sequence[Mapping[str, object]],
) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Resolve source (operator, quadrant) provenance and distilled rule IDs."""
    provenance: dict[str, tuple[str, str]] = {}
    distilled_ids: set[str] = set(distilled_rules.keys())

    indexed_edges = list(enumerate(winning_edges))
    indexed_edges.sort(
        key=lambda item: (
            _coerce_float(item[1].get("timestamp"), 0.0),
            _coerce_int(item[1].get("round_idx"), 0),
            item[0],
        )
    )
    for _, edge in indexed_edges:
        edge_op = str(edge.get("source_operator") or edge.get("operator") or "").strip()
        raw_axes = edge.get("source_topology_axes")
        if not isinstance(raw_axes, Mapping):
            raw_axes = edge.get("topology_axes")
        edge_quad = str(edge.get("source_quadrant") or "").strip()
        if not edge_quad:
            edge_quad = format_topology_quadrant(
                raw_axes if isinstance(raw_axes, Mapping) else None
            )
        src_op = edge_op or "unknown"

        staged_ids = _extract_string_list(edge.get("staged_rule_ids"))
        staged_ids.extend(_extract_string_list(edge.get("distilled_rule_ids")))
        for s_rid in staged_ids:
            distilled_ids.add(s_rid)
            existing = provenance.get(s_rid)
            if existing is None or (
                existing[0] == "unknown" and src_op != "unknown"
            ) or (
                existing[1] == "unknown_topology" and edge_quad != "unknown_topology"
            ):
                provenance[s_rid] = (src_op, edge_quad)

        edge_rid = str(edge.get("rule_id") or "").strip()
        if edge_rid:
            if str(edge.get("rule_origin") or "").strip().lower() == "distilled":
                distilled_ids.add(edge_rid)
            if edge_rid not in provenance:
                provenance[edge_rid] = (src_op, edge_quad)

    for rid, rdict in distilled_rules.items():
        origin_op = str(
            rdict.get("origin_operator") or rdict.get("source_operator") or ""
        ).strip()
        r_axes = rdict.get("topology_axes")
        origin_quad = str(
            rdict.get("source_quadrant") or rdict.get("origin_quadrant") or ""
        ).strip()
        if not origin_quad and isinstance(r_axes, Mapping):
            origin_quad = format_topology_quadrant(r_axes)
        resolved_op = origin_op or "unknown"
        resolved_quad = origin_quad or "unknown_topology"
        prev = provenance.get(rid)
        if prev is None:
            provenance[rid] = (resolved_op, resolved_quad)
        else:
            merged_op = prev[0] if prev[0] != "unknown" else resolved_op
            merged_quad = prev[1] if prev[1] != "unknown_topology" else resolved_quad
            provenance[rid] = (merged_op, merged_quad)

    return provenance, distilled_ids


def _build_distilled_transfer_matrix(
    indexed_rule_events: Sequence[tuple[int, Mapping[str, object]]],
    provenance_by_rule: Mapping[str, tuple[str, str]],
    distilled_rule_ids: set[str],
) -> list[dict[str, object]]:
    """Build directed source->target transfer metrics across funnel hits."""
    grouped: dict[tuple[str, str, str], list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for idx, ev in indexed_rule_events:
        rid = str(ev.get("rule_id") or "").strip()
        if not rid or rid not in provenance_by_rule:
            continue
        target_op = str(ev.get("operator") or "unknown").strip() or "unknown"
        raw_axes = ev.get("topology_axes")
        target_quad = format_topology_quadrant(
            raw_axes if isinstance(raw_axes, Mapping) else None
        )
        grouped[(rid, target_op, target_quad)].append((idx, ev))

    matrix: list[dict[str, object]] = []
    for (rid, target_op, target_quad), items in sorted(grouped.items()):
        src_op, src_quad = provenance_by_rule[rid]
        ordered = sorted(
            items,
            key=lambda item: (
                _coerce_float(item[1].get("timestamp"), 0.0),
                _coerce_int(item[1].get("round_idx"), 0),
                item[0],
            ),
        )
        rows = [item[1] for item in ordered]
        promos = sum(1 for r in rows if bool(r.get("promoted_l3", False)))
        gains = [
            max(0.0, _coerce_float(r.get("delta_speedup"), 0.0))
            for r in rows
            if bool(r.get("promoted_l3", False))
            or _coerce_float(r.get("delta_speedup"), 0.0) > 0.0
        ]
        first_hit = gains[0] if gains else 0.0
        mean_gain = (sum(gains) / len(gains)) if gains else 0.0
        matrix.append(
            {
                "source_operator": src_op,
                "source_quadrant": src_quad,
                "target_operator": target_op,
                "target_quadrant": target_quad,
                "rule_id": rid,
                "rule_origin": "distilled" if rid in distilled_rule_ids else "seed",
                "dispatches": len(rows),
                "l3_promoted_count": promos,
                "first_hit_delta_speedup": round(first_hit, 6),
                "mean_delta_speedup": round(mean_gain, 6),
            }
        )
    return matrix


def _compute_rule_quadrant_metrics(
    by_rule_quad: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rule_rows: list[dict[str, object]] = []
    for (rid, quad), rows in sorted(by_rule_quad.items()):
        total = len(rows)
        noops = sum(1 for r in rows if bool(r.get("noop_identical", False)))
        passes = sum(1 for r in rows if bool(r.get("passed_l0_l1", False)))
        promos = sum(1 for r in rows if bool(r.get("promoted_l3", False)))
        gains = [
            max(0.0, _coerce_float(r.get("delta_speedup"), 0.0))
            for r in rows
            if bool(r.get("promoted_l3", False))
            or _coerce_float(r.get("delta_speedup"), 0.0) > 0.0
        ]
        mean_gain = (sum(gains) / len(gains)) if gains else 0.0
        wall_sec = sum(max(0.0, _coerce_float(r.get("wall_sec"), 0.0)) for r in rows)
        gain_per_min = (sum(gains) / (wall_sec / 60.0)) if wall_sec > 1e-6 else 0.0
        rule_rows.append(
            RuleQuadrantMetrics(
                rule_id=rid,
                quadrant=quad,
                dispatches=total,
                noop_count=noops,
                compile_pass_count=passes,
                l3_promoted_count=promos,
                noop_rate=round(noops / total, 4),
                compile_pass_rate=round(passes / total, 4),
                promotion_rate=round(promos / total, 4),
                mean_delta_speedup=round(mean_gain, 6),
                total_wall_sec=round(wall_sec, 3),
                speedup_per_wall_min=round(gain_per_min, 6),
            ).to_dict()
        )
    return rule_rows


def summarize_funnel_and_transfer(
    funnel_events: Iterable[Mapping[str, object]],
    *,
    compile_wall_share_alert: float = 0.30,
    experience_lake_dir: Path | None = None,
) -> dict[str, object]:
    """Compute cross-topology rule transfer rates and bottleneck alerts."""
    by_rule_quad: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    quadrants_by_rule: dict[str, set[str]] = defaultdict(set)
    promoted_quadrants_by_rule: dict[str, set[str]] = defaultdict(set)
    compile_errors_by_op: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    compile_total_by_op: dict[str, int] = defaultdict(int)
    log_err_by_quad: dict[str, list[float]] = defaultdict(list)
    noisy_l2_events = 0
    total_l2_events = 0
    l3_stage_counts: dict[str, int] = defaultdict(int)
    indexed_rule_events: list[tuple[int, Mapping[str, object]]] = []

    for idx, ev in enumerate(funnel_events):
        op = str(ev.get("operator") or "unknown")
        raw_axes = ev.get("topology_axes")
        quad = format_topology_quadrant(raw_axes if isinstance(raw_axes, Mapping) else None)
        passed = bool(ev.get("passed_l0_l1", False))
        err_code = str(ev.get("error_code_norm") or "OK")
        if not passed and err_code not in {"OK", "COST_COARSE_REJECT"}:
            compile_errors_by_op[op][err_code] += 1
            compile_total_by_op[op] += 1

        pred_us = _coerce_float(ev.get("analytical_pred_us"), 0.0)
        meas_us = _coerce_float(ev.get("measured_l2_us"), 0.0)
        burst_cv = _coerce_float(ev.get("burst_batch_cv"), 0.0)
        if meas_us > 0.0:
            total_l2_events += 1
            if burst_cv > 0.08:
                noisy_l2_events += 1
            elif pred_us > 0.0:
                log_err_by_quad[quad].append(abs(math.log(meas_us / pred_us)))

        l3_stage = str(ev.get("l3_failure_stage") or "").strip()
        if l3_stage:
            l3_stage_counts[l3_stage] += 1

        rule_id = ev.get("rule_id")
        if isinstance(rule_id, str) and rule_id.strip():
            rid = rule_id.strip()
            by_rule_quad[(rid, quad)].append(ev)
            quadrants_by_rule[rid].add(quad)
            indexed_rule_events.append((idx, ev))
            if (
                bool(ev.get("promoted_l3", False))
                or _coerce_float(ev.get("delta_speedup"), 0.0) > 0.01
            ):
                promoted_quadrants_by_rule[rid].add(quad)

    distilled_rules: dict[str, dict[str, object]] = {}
    winning_edges: list[dict[str, object]] = []
    if experience_lake_dir is not None:
        lake_path = Path(experience_lake_dir)
        distilled_rules = load_distilled_rules_from_lake(lake_path)
        winning_edges = load_winning_edges_from_lake(lake_path)
    provenance_by_rule, distilled_rule_ids = _build_rule_provenance(
        distilled_rules, winning_edges
    )

    rule_rows = _compute_rule_quadrant_metrics(by_rule_quad)
    low_efficiency_rules: list[dict[str, object]] = []
    for row in rule_rows:
        if _coerce_float(row.get("noop_rate"), 0.0) > 0.40:
            flagged = dict(row)
            flagged["needs_precondition_tightening"] = True
            flagged["reason"] = "noop_rate_exceeds_0.40"
            low_efficiency_rules.append(flagged)

    transfer_summary: list[dict[str, object]] = []
    for rid, tested_quads in sorted(quadrants_by_rule.items()):
        win_quads = promoted_quadrants_by_rule.get(rid, set())
        transfer_rate = len(win_quads) / len(tested_quads) if tested_quads else 0.0
        classification = (
            "universal_transferable"
            if len(win_quads) >= 2 and transfer_rate >= 0.5
            else (
                "topology_specialized"
                if len(win_quads) == 1
                else "needs_precondition_tightening"
            )
        )
        transfer_summary.append(
            {
                "rule_id": rid,
                "rule_origin": "distilled" if rid in distilled_rule_ids else "seed",
                "tested_quadrants": sorted(tested_quads),
                "effective_quadrants": sorted(win_quads),
                "quadrant_transfer_rate": round(transfer_rate, 4),
                "classification": classification,
            }
        )

    compile_wall_alerts: list[dict[str, object]] = []
    for op, err_map in sorted(compile_errors_by_op.items()):
        tot = compile_total_by_op[op]
        if tot <= 0:
            continue
        top3 = sorted(err_map.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        for code, cnt in top3:
            share = cnt / tot
            if share >= compile_wall_share_alert:
                compile_wall_alerts.append(
                    {
                        "operator": op,
                        "error_code_norm": code,
                        "count": cnt,
                        "share": round(share, 4),
                    }
                )

    calibration_by_quad = {
        quad: {
            "clean_samples": len(errs),
            "median_abs_log_error": round(float(statistics.median(errs)), 4),
            "needs_tier2_refit": float(statistics.median(errs)) > 1.0,
        }
        for quad, errs in sorted(log_err_by_quad.items())
        if errs
    }

    transfer_matrix = _build_distilled_transfer_matrix(
        indexed_rule_events, provenance_by_rule, distilled_rule_ids
    )

    return {
        "rule_quadrant_metrics": rule_rows,
        "low_efficiency_rules": low_efficiency_rules,
        "cross_topology_transfer": transfer_summary,
        "distilled_rule_transfer_matrix": transfer_matrix,
        "compile_wall_alerts": compile_wall_alerts,
        "cost_calibration_by_quadrant": calibration_by_quad,
        "contention_noise_summary": {
            "total_l2_events": total_l2_events,
            "clean_l2_events": total_l2_events - noisy_l2_events,
            "noisy_l2_events": noisy_l2_events,
            "noisy_share": round(noisy_l2_events / total_l2_events, 4) if total_l2_events else 0.0,
        },
        "l3_failure_stage_counts": dict(sorted(l3_stage_counts.items())),
    }


def _as_mapping_list(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def render_transfer_audit_markdown(report: Mapping[str, object]) -> str:
    """Render transfer audit and bottleneck report as a Markdown summary."""
    lines: list[str] = ["# Cross-Topology Causal Rule Transfer Audit", ""]

    transfer_rows = _as_mapping_list(report.get("cross_topology_transfer"))
    lines.append("## Cross-Topology Rule Transfer")
    if transfer_rows:
        lines.append(
            "| Rule ID | Origin | Classification | Transfer Rate | Tested | Effective |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in transfer_rows:
            tested = row.get("tested_quadrants")
            eff = row.get("effective_quadrants")
            tested_n = len(tested) if isinstance(tested, list) else 0
            eff_n = len(eff) if isinstance(eff, list) else 0
            lines.append(
                f"| {row.get('rule_id', '')} | {row.get('rule_origin', 'seed')} "
                f"| {row.get('classification', '')} "
                f"| {_coerce_float(row.get('quadrant_transfer_rate'), 0.0):.4f} "
                f"| {tested_n} | {eff_n} |"
            )
    else:
        lines.append("No rule transfer observations recorded.")
    lines.append("")

    matrix_rows = _as_mapping_list(report.get("distilled_rule_transfer_matrix"))
    lines.append("## Distilled Rule Directed Transfer Matrix")
    if matrix_rows:
        lines.append(
            "| Rule ID | Origin | Source | Target | Dispatches | L3 | First Hit | Mean Gain |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in matrix_rows:
            src = f"{row.get('source_operator', '')} ({row.get('source_quadrant', '')})"
            dst = f"{row.get('target_operator', '')} ({row.get('target_quadrant', '')})"
            lines.append(
                f"| {row.get('rule_id', '')} | {row.get('rule_origin', '')} "
                f"| {src} | {dst} "
                f"| {_coerce_int(row.get('dispatches'), 0)} "
                f"| {_coerce_int(row.get('l3_promoted_count'), 0)} "
                f"| {_coerce_float(row.get('first_hit_delta_speedup'), 0.0):.4f} "
                f"| {_coerce_float(row.get('mean_delta_speedup'), 0.0):.4f} |"
            )
    else:
        lines.append("No directed rule transfers recorded.")
    lines.append("")

    low_eff = _as_mapping_list(report.get("low_efficiency_rules"))
    lines.append("## Low-Efficiency Rules (Precondition Tightening Needed)")
    if low_eff:
        lines.append("| Rule ID | Quadrant | Dispatches | No-op Count | No-op Rate |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in low_eff:
            lines.append(
                f"| {row.get('rule_id', '')} | {row.get('quadrant', '')} "
                f"| {_coerce_int(row.get('dispatches'), 0)} "
                f"| {_coerce_int(row.get('noop_count'), 0)} "
                f"| {_coerce_float(row.get('noop_rate'), 0.0):.4f} |"
            )
    else:
        lines.append("No rules exceeded the 0.40 no-op rate threshold.")
    lines.append("")

    alerts = _as_mapping_list(report.get("compile_wall_alerts"))
    lines.append("## Compile Wall Alerts")
    if alerts:
        lines.append("| Operator | Error Code | Count | Share |")
        lines.append("| --- | --- | ---: | ---: |")
        for row in alerts:
            lines.append(
                f"| {row.get('operator', '')} | {row.get('error_code_norm', '')} "
                f"| {_coerce_int(row.get('count'), 0)} "
                f"| {_coerce_float(row.get('share'), 0.0):.4f} |"
            )
    else:
        lines.append("No compile wall alerts triggered.")
    lines.append("")

    noise = report.get("contention_noise_summary")
    noise_map = noise if isinstance(noise, Mapping) else {}
    lines.append("## L2 Contention & Cost Calibration")
    lines.append(
        f"- L2 events: total={_coerce_int(noise_map.get('total_l2_events'), 0)}, "
        f"clean={_coerce_int(noise_map.get('clean_l2_events'), 0)}, "
        f"noisy={_coerce_int(noise_map.get('noisy_l2_events'), 0)}, "
        f"noisy_share={_coerce_float(noise_map.get('noisy_share'), 0.0):.4f}"
    )
    cal_raw = report.get("cost_calibration_by_quadrant")
    if isinstance(cal_raw, Mapping) and cal_raw:
        lines.append("")
        lines.append("| Quadrant | Clean Samples | Median Abs Log Error | Needs Tier-2 Refit |")
        lines.append("| --- | ---: | ---: | --- |")
        for quad, stats_raw in sorted(cal_raw.items()):
            if not isinstance(stats_raw, Mapping):
                continue
            lines.append(
                f"| {quad} | {_coerce_int(stats_raw.get('clean_samples'), 0)} "
                f"| {_coerce_float(stats_raw.get('median_abs_log_error'), 0.0):.4f} "
                f"| {bool(stats_raw.get('needs_tier2_refit', False))} |"
            )
    lines.append("")

    l3_raw = report.get("l3_failure_stage_counts")
    lines.append("## L3 Failure Stage Counts")
    if isinstance(l3_raw, Mapping) and l3_raw:
        for stage, count in sorted(l3_raw.items()):
            lines.append(f"- {stage}: {_coerce_int(count, 0)}")
    else:
        lines.append("No L3 failures recorded.")
    lines.append("")
    return "\n".join(lines)


def load_funnel_events_from_base(base_dir: Path) -> list[dict[str, object]]:
    """Load all round_funnel_events.jsonl files beneath base_dir."""
    events: list[dict[str, object]] = []
    if not base_dir.exists():
        return events
    for jsonl in sorted(base_dir.rglob("round_funnel_events.jsonl")):
        try:
            lines = jsonl.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to generate cross-topology causal rule transfer report."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-transfer-audit",
        description="Audit cross-topology causal rule transfer rates and funnel bottlenecks.",
    )
    parser.add_argument(
        "--base-dir",
        default=".tilefoundry-marathon",
        help="Root directory containing operator marathon state directories.",
    )
    parser.add_argument(
        "--experience-lake-dir",
        default="",
        help="Optional Experience Lake directory with distilled_rules and winning_edges.jsonl.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON transfer and bottleneck report.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional path to write Markdown transfer and bottleneck summary.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    events = load_funnel_events_from_base(Path(args.base_dir).expanduser())
    lake_dir = (
        Path(args.experience_lake_dir).expanduser()
        if str(args.experience_lake_dir).strip()
        else None
    )
    report = summarize_funnel_and_transfer(events, experience_lake_dir=lake_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    if args.output_md:
        md_path = Path(args.output_md).expanduser()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_transfer_audit_markdown(report), encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
