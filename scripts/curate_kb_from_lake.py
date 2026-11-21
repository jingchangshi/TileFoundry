#!/usr/bin/env python3
"""Aggregate Experience Lake KB staging candidates and evidence into an offline review report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read valid JSON object lines from *path*, returning an empty list if absent."""
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            records.append(dict(parsed))
    return records


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _load_staging_candidates(staging_dir: Path) -> tuple[list[dict[str, object]], int]:
    """Load pending candidate rule proposals from ``kb/staging_candidates/*.json``."""
    if not staging_dir.is_dir():
        return [], 0
    proposals: list[dict[str, object]] = []
    parse_errors = 0
    for path in sorted(staging_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        if not isinstance(raw, Mapping):
            parse_errors += 1
            continue
        rule_raw = raw.get("rule") if isinstance(raw.get("rule"), Mapping) else raw
        if not isinstance(rule_raw, Mapping) or "rule_id" not in rule_raw:
            parse_errors += 1
            continue
        ablation_raw = raw.get("ablation_report")
        ablation_map = ablation_raw if isinstance(ablation_raw, Mapping) else {}
        base_us = _as_float(ablation_map.get("baseline_latency_us"))
        win_us = _as_float(ablation_map.get("winner_latency_us"))
        rel_gain = (
            (base_us - win_us) / max(base_us, 1e-6)
            if (base_us > 0.0 and win_us > 0.0)
            else _as_float(raw.get("relative_gain"))
        )
        kept_chunks = ablation_map.get("kept_positive_chunks")
        kept_count = len(kept_chunks) if isinstance(kept_chunks, list) else 0
        proposals.append(
            {
                "file": path.name,
                "status": str(raw.get("status", "pending_review")),
                "rule_id": str(rule_raw.get("rule_id", "")),
                "symptom_family": str(rule_raw.get("symptom_family", "")),
                "regime_guard": str(rule_raw.get("regime_guard", "")),
                "precondition": str(rule_raw.get("precondition", "")),
                "anti_precondition": str(rule_raw.get("anti_precondition", "False")),
                "expected_postcondition": str(rule_raw.get("expected_postcondition", "")),
                "scope": str(rule_raw.get("scope", "cross_op")),
                "origin_operator": str(rule_raw.get("origin_operator", "")),
                "relative_gain": round(rel_gain, 6),
                "kept_positive_chunks": kept_count,
            }
        )
    proposals.sort(key=lambda item: (-_as_float(item["relative_gain"]), str(item["rule_id"])))
    return proposals, parse_errors


def _summarize_winning_edges(
    edges: Sequence[Mapping[str, object]],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Select the top-K highest-gain winning optimization edges."""
    summarized: list[dict[str, object]] = []
    for edge in edges:
        loo_raw = edge.get("loo_chunks")
        loo_kinds: list[str] = []
        if isinstance(loo_raw, list):
            for chunk in loo_raw:
                if isinstance(chunk, Mapping):
                    nested = chunk.get("chunk")
                    if isinstance(nested, Mapping) and nested.get("edit_kind"):
                        loo_kinds.append(str(nested["edit_kind"]))
                    elif chunk.get("edit_kind"):
                        loo_kinds.append(str(chunk["edit_kind"]))
        staged_raw = edge.get("staged_rule_ids")
        staged_ids = (
            [str(item) for item in staged_raw] if isinstance(staged_raw, list) else []
        )
        summarized.append(
            {
                "operator": str(edge.get("operator", "")),
                "regime_tag": str(edge.get("regime_tag", "")),
                "worker_kind": str(edge.get("worker_kind", "")),
                "parent_sha": str(edge.get("parent_sha", ""))[:12],
                "winner_sha": str(edge.get("winner_sha", ""))[:12],
                "baseline_latency_us": round(_as_float(edge.get("baseline_latency_us")), 4),
                "winner_latency_us": round(_as_float(edge.get("winner_latency_us")), 4),
                "relative_gain": round(_as_float(edge.get("relative_gain")), 6),
                "loo_edit_kinds": loo_kinds,
                "staged_rule_ids": staged_ids,
            }
        )
    summarized.sort(
        key=lambda item: (
            -_as_float(item["relative_gain"]),
            str(item["operator"]),
            str(item["regime_tag"]),
        )
    )
    return summarized[: max(0, int(top_k))]


def _summarize_misapplications(
    records: Sequence[Mapping[str, object]],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Aggregate rule misapplications by rule_id with symptom and residual statistics."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for rec in records:
        rid = str(rec.get("rule_id") or "").strip()
        if not rid:
            continue
        grouped.setdefault(rid, []).append(rec)

    summaries: list[dict[str, object]] = []
    for rid, items in grouped.items():
        families = Counter(
            str(item.get("symptom_family") or "UNSPECIFIED") for item in items
        )
        reasons = Counter(str(item.get("reason") or "unknown") for item in items)
        operators = sorted(
            {
                str(item.get("operator") or "").strip()
                for item in items
                if str(item.get("operator") or "").strip()
            }
        )
        dominant_axes: Counter[str] = Counter()
        lat_ratios: list[float] = []
        inflations: list[float] = []
        etas: list[float] = []
        ub_occs: list[float] = []
        for item in items:
            lat_ratio = _as_float(item.get("latency_ratio"))
            if lat_ratio <= 0.0:
                p_lat = _as_float(item.get("parent_latency_us"))
                c_lat = _as_float(item.get("cand_latency_us"))
                if p_lat > 0.0 and c_lat > 0.0:
                    lat_ratio = c_lat / p_lat
            if lat_ratio > 0.0:
                lat_ratios.append(lat_ratio)
            if "transfer_inflation_ratio" in item:
                inflations.append(_as_float(item.get("transfer_inflation_ratio"), 1.0))
            if "effective_compute_eta" in item:
                etas.append(_as_float(item.get("effective_compute_eta"), 0.0))
            if "ub_occupancy_ratio" in item:
                ub_occs.append(_as_float(item.get("ub_occupancy_ratio"), 0.0))
            obs_raw = item.get("parent_observation")
            if isinstance(obs_raw, Mapping):
                res_raw = obs_raw.get("residuals")
                if isinstance(res_raw, Mapping):
                    axis_scores = {
                        "compute_issue": _as_float(res_raw.get("compute_issue_gap")),
                        "offchip_dma": _as_float(res_raw.get("offchip_dma_gap")),
                        "onchip_sram_reg": _as_float(res_raw.get("onchip_sram_reg_gap")),
                        "pipeline_sync": _as_float(res_raw.get("pipeline_sync_gap")),
                        "grid_wave": _as_float(res_raw.get("grid_wave_gap")),
                    }
                    dom_axis = max(axis_scores.items(), key=lambda kv: kv[1])[0]
                    dominant_axes[dom_axis] += 1

        summaries.append(
            {
                "rule_id": rid,
                "symptom_family": families.most_common(1)[0][0] if families else "UNSPECIFIED",
                "count": len(items),
                "operators": operators,
                "mean_latency_ratio": (
                    round(sum(lat_ratios) / float(len(lat_ratios)), 4) if lat_ratios else 0.0
                ),
                "mean_transfer_inflation_ratio": (
                    round(sum(inflations) / float(len(inflations)), 4) if inflations else 1.0
                ),
                "mean_effective_compute_eta": (
                    round(sum(etas) / float(len(etas)), 4) if etas else 0.0
                ),
                "mean_ub_occupancy_ratio": (
                    round(sum(ub_occs) / float(len(ub_occs)), 4) if ub_occs else 0.0
                ),
                "dominant_axes": dict(sorted(dominant_axes.items())),
                "top_reasons": [reason for reason, _cnt in reasons.most_common(3)],
            }
        )

    summaries.sort(
        key=lambda item: (
            -int(item["count"]),
            -_as_float(item["mean_latency_ratio"]),
            str(item["rule_id"]),
        )
    )
    return summaries[: max(0, int(top_k))]


def build_curation_report(
    lake_dir: str | Path,
    *,
    top_k: int = 10,
) -> dict[str, object]:
    """Build a structured offline KB curation report from an Experience Lake directory."""
    root = Path(lake_dir)
    kb_root = root / "kb"
    staging_dir = kb_root / "staging_candidates"
    evidence_dir = kb_root / "evidence"

    proposals, staging_parse_errors = _load_staging_candidates(staging_dir)
    winning_edges = _read_jsonl(evidence_dir / "winning_edges.jsonl")
    misapplications = _read_jsonl(evidence_dir / "rule_misapplications.jsonl")

    return {
        "lake_dir": str(root),
        "summary": {
            "staging_candidates_count": len(proposals),
            "staging_parse_errors": staging_parse_errors,
            "winning_edges_count": len(winning_edges),
            "rule_misapplications_count": len(misapplications),
        },
        "staging_candidates": proposals[: max(0, int(top_k))],
        "top_winning_edges": _summarize_winning_edges(winning_edges, top_k=top_k),
        "top_misapplied_rules": _summarize_misapplications(misapplications, top_k=top_k),
    }


def render_markdown_report(report: Mapping[str, object]) -> str:
    """Render an offline KB curation report as concise GitHub-flavored Markdown."""
    summary_raw = report.get("summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    lines = [
        "# Ascend KB Offline Curation Report",
        "",
        "## Summary",
        f"- Staging candidate rules: {summary.get('staging_candidates_count', 0)}",
        f"- Winning optimization edges: {summary.get('winning_edges_count', 0)}",
        f"- Rule misapplication events: {summary.get('rule_misapplications_count', 0)}",
        f"- Staging parse errors: {summary.get('staging_parse_errors', 0)}",
        "",
        "## Candidate Rule Proposals",
    ]
    candidates = report.get("staging_candidates")
    if isinstance(candidates, list) and candidates:
        lines.extend(
            [
                "| Rule ID | Symptom Family | Scope | Origin | Rel Gain | Anti-Precondition |",
                "| :--- | :--- | :--- | :--- | :---: | :--- |",
            ]
        )
        for cand in candidates:
            if not isinstance(cand, Mapping):
                continue
            lines.append(
                f"| `{cand.get('rule_id', '')}` "
                f"| `{cand.get('symptom_family', '') or 'UNSPECIFIED'}` "
                f"| `{cand.get('scope', '')}` "
                f"| `{cand.get('origin_operator', '') or '-'}` "
                f"| {(_as_float(cand.get('relative_gain')) * 100.0):.1f}% "
                f"| `{cand.get('anti_precondition', 'False')}` |"
            )
    else:
        lines.append("_No pending candidate rules found._")

    lines.extend(["", "## Top Winning Edges"])
    edges = report.get("top_winning_edges")
    if isinstance(edges, list) and edges:
        lines.extend(
            [
                "| Operator | Regime | Worker | Rel Gain | Base->Win (us) | LOO Chunks |",
                "| :--- | :--- | :--- | :---: | :---: | :--- |",
            ]
        )
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            loo_raw = edge.get("loo_edit_kinds")
            loo_str = ", ".join(str(x) for x in loo_raw) if isinstance(loo_raw, list) and loo_raw else "-"
            lines.append(
                f"| `{edge.get('operator', '')}` "
                f"| `{edge.get('regime_tag', '')}` "
                f"| `{edge.get('worker_kind', '')}` "
                f"| {(_as_float(edge.get('relative_gain')) * 100.0):.1f}% "
                f"| {_as_float(edge.get('baseline_latency_us')):.2f}->{_as_float(edge.get('winner_latency_us')):.2f} "
                f"| `{loo_str}` |"
            )
    else:
        lines.append("_No winning edge evidence found._")

    lines.extend(["", "## Top Misapplied Rules"])
    misapplied = report.get("top_misapplied_rules")
    if isinstance(misapplied, list) and misapplied:
        lines.extend(
            [
                "| Rule ID | Symptom Family | Count | Mean Lat Ratio | Inflation | Compute Eta | UB Occ | Reasons |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for item in misapplied:
            if not isinstance(item, Mapping):
                continue
            reasons_raw = item.get("top_reasons")
            reasons_str = (
                ", ".join(str(x) for x in reasons_raw)
                if isinstance(reasons_raw, list) and reasons_raw
                else "-"
            )
            lines.append(
                f"| `{item.get('rule_id', '')}` "
                f"| `{item.get('symptom_family', '')}` "
                f"| {int(item.get('count', 0))} "
                f"| {_as_float(item.get('mean_latency_ratio')):.3f} "
                f"| {_as_float(item.get('mean_transfer_inflation_ratio'), 1.0):.2f} "
                f"| {_as_float(item.get('mean_effective_compute_eta')):.2f} "
                f"| {_as_float(item.get('mean_ub_occupancy_ratio')):.2f} "
                f"| `{reasons_str}` |"
            )
    else:
        lines.append("_No rule misapplication evidence found._")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for offline Experience Lake KB curation."""
    parser = argparse.ArgumentParser(
        prog="curate_kb_from_lake.py",
        description="Aggregate offline KB staging candidates and evidence from an Experience Lake.",
    )
    parser.add_argument(
        "--lake-dir",
        required=True,
        help="Path to the Experience Lake directory containing kb/staging_candidates and kb/evidence.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report output format (markdown or json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum items to include per report section.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline KB curation CLI and write the formatted report to stdout."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    report = build_curation_report(args.lake_dir, top_k=args.top_k)
    if args.output_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
