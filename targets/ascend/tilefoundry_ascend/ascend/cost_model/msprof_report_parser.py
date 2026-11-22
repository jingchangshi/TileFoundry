"""Pragmatic P1 multi-subkernel msprof CSV parser and P2 instruction-trace analyzer."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_CANONICAL_TABLES: tuple[tuple[str, str], ...] = (
    ("OpBasicInfo.csv", "opbasicinfo"),
    ("PipeUtilization.csv", "pipeutilization"),
    ("MemoryL0.csv", "memoryl0"),
    ("Memory.csv", "memory"),
    ("ResourceConflictRatio.csv", "resourceconflictratio"),
    ("L2Cache.csv", "l2cache"),
)

_MISSING_TOKENS = frozenset({"", "n/a", "na", "none", "null", "-", "--", "nan"})

_BYTE_UNIT_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
    "gib": 1024 * 1024 * 1024,
}

_TIME_UNIT_TO_US: dict[str, float] = {
    "": 1.0,
    "us": 1.0,
    "ns": 1e-3,
    "ms": 1e3,
    "s": 1e6,
}

_HEADER_UNIT_RE = re.compile(r"\(([^()]+)\)\s*$")
_NUM_WITH_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([%A-Za-z/]*)\s*$"
)
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+")
_WAITFLAG_RE = re.compile(r"(?:mte_?wait_?flag|wait_?flag)", re.IGNORECASE)
_SYNC_OPCODE_RE = re.compile(
    r"(?:wait_?flag|set_?flag|pipe_?barrier|barrier|sync|event)", re.IGNORECASE
)
_DEFAULT_CYCLE_TO_US = 1.0 / 1800.0
_P1_SUMMARY_CSV_STEMS = frozenset(
    {
        "opbasicinfo",
        "pipeutilization",
        "memoryl0",
        "memory",
        "resourceconflictratio",
        "l2cache",
        "arithmeticutilization",
        "memoryub",
        "roofline",
        "occupancy",
        "kernelscale",
        "memorydetail",
        "basicinfo",
    }
)
_OBJDUMP_DWARF_LINE_RE = re.compile(
    r"^\s*[;#]\s*(?P<loc>(?:[A-Za-z]:)?[^\s:;][^:]*:\d+(?::\d+)?)\s*$"
)
_OBJDUMP_SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+\s+<(?P<symbol>[^>]+)>:\s*$")
_OBJDUMP_INSTR_RE = re.compile(
    r"^\s*(?P<pc>[0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2})+|[0-9a-fA-F]{4,16})\s+(?P<body>.+?)\s*$"
)


@dataclass(frozen=True)
class SubkernelMsprofMetrics:
    """Measured msprof counters for a single subkernel invocation."""

    name: str
    duration_us: float
    aic_mac_ratio: float | None = None
    aiv_vec_ratio: float | None = None
    mte2_wait_ratio: float | None = None
    aic_fixpipe: float | None = None
    l2_hit_rate: float | None = None
    bank_conflict_ratio: float | None = None
    sync_stall_ratio: float | None = None
    aiv0_vec_ratio: float | None = None
    aiv1_vec_ratio: float | None = None
    hbm_read_bytes: int | None = None
    hbm_write_bytes: int | None = None
    actual_gm_bytes: int | None = None
    l0_transfer_bytes: int | None = None
    aic_total_cycles: int | None = None
    aiv_total_cycles: int | None = None
    block_dim: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize measured subkernel fields without fabricating absent counters."""
        payload: dict[str, object] = {
            "name": self.name,
            "duration_us": self.duration_us,
        }
        optional_fields: tuple[tuple[str, float | int | None], ...] = (
            ("aic_mac_ratio", self.aic_mac_ratio),
            ("aiv_vec_ratio", self.aiv_vec_ratio),
            ("mte2_wait_ratio", self.mte2_wait_ratio),
            ("aic_fixpipe", self.aic_fixpipe),
            ("l2_hit_rate", self.l2_hit_rate),
            ("bank_conflict_ratio", self.bank_conflict_ratio),
            ("sync_stall_ratio", self.sync_stall_ratio),
            ("aiv0_vec_ratio", self.aiv0_vec_ratio),
            ("aiv1_vec_ratio", self.aiv1_vec_ratio),
            ("hbm_read_bytes", self.hbm_read_bytes),
            ("hbm_write_bytes", self.hbm_write_bytes),
            ("actual_gm_bytes", self.actual_gm_bytes),
            ("l0_transfer_bytes", self.l0_transfer_bytes),
            ("aic_total_cycles", self.aic_total_cycles),
            ("aiv_total_cycles", self.aiv_total_cycles),
            ("block_dim", self.block_dim),
        )
        for key, value in optional_fields:
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class AggregatedMsprofReport:
    """Duration-weighted multi-subkernel summary parsed from an msprof op report."""

    total_duration_us: float
    subkernels: tuple[SubkernelMsprofMetrics, ...]
    dominant_subkernel_name: str
    dominant_subkernel_share: float
    aic_mac_ratio: float | None = None
    aiv_vec_ratio: float | None = None
    mte2_wait_ratio: float | None = None
    aic_fixpipe: float | None = None
    l2_hit_rate: float | None = None
    bank_conflict_ratio: float | None = None
    sync_stall_ratio: float | None = None
    aiv0_vec_ratio: float | None = None
    aiv1_vec_ratio: float | None = None
    hbm_read_bytes: int | None = None
    hbm_write_bytes: int | None = None
    actual_gm_bytes: int | None = None
    l0_transfer_bytes: int | None = None
    aic_total_cycles: int | None = None
    aiv_total_cycles: int | None = None
    parsed_tables: tuple[str, ...] = ()

    def to_telemetry_dict(self) -> dict[str, float | int | str]:
        """Emit only genuinely measured counters compatible with ObservationVector5D."""
        telemetry: dict[str, float | int | str] = {}
        if self.subkernels and self.total_duration_us > 0.0:
            telemetry["dur_us"] = self.total_duration_us
            telemetry["measured_latency_us"] = self.total_duration_us
        optional_numeric: tuple[tuple[str, float | int | None], ...] = (
            ("aic_mac_ratio", self.aic_mac_ratio),
            ("aiv_vec_ratio", self.aiv_vec_ratio),
            ("mte2_wait_ratio", self.mte2_wait_ratio),
            ("aic_fixpipe", self.aic_fixpipe),
            ("l2_hit_rate", self.l2_hit_rate),
            ("bank_conflict_ratio", self.bank_conflict_ratio),
            ("sync_stall_ratio", self.sync_stall_ratio),
            ("aiv0_vec_ratio", self.aiv0_vec_ratio),
            ("aiv1_vec_ratio", self.aiv1_vec_ratio),
            ("hbm_read_bytes", self.hbm_read_bytes),
            ("hbm_write_bytes", self.hbm_write_bytes),
            ("actual_gm_bytes", self.actual_gm_bytes),
            ("l0_transfer_bytes", self.l0_transfer_bytes),
            ("aic_total_cycles", self.aic_total_cycles),
            ("aiv_total_cycles", self.aiv_total_cycles),
        )
        for key, value in optional_numeric:
            if value is not None:
                telemetry[key] = value
        if self.subkernels:
            telemetry["subkernel_count"] = len(self.subkernels)
            telemetry["dominant_subkernel_share"] = self.dominant_subkernel_share
        if self.dominant_subkernel_name:
            telemetry["dominant_subkernel_name"] = self.dominant_subkernel_name
        return telemetry

    def to_dict(self) -> dict[str, object]:
        """Serialize the full aggregated report including individual subkernels."""
        payload: dict[str, object] = {
            "total_duration_us": self.total_duration_us,
            "dominant_subkernel_name": self.dominant_subkernel_name,
            "dominant_subkernel_share": self.dominant_subkernel_share,
            "parsed_tables": list(self.parsed_tables),
            "subkernels": [sub.to_dict() for sub in self.subkernels],
            "telemetry": self.to_telemetry_dict(),
        }
        return payload


@dataclass(frozen=True)
class PipeIssueIntervalStats:
    """Same-pipe issue-to-issue interval percentiles in microseconds."""

    pipe: str
    instruction_count: int
    interval_count: int
    p10_us: float
    p50_us: float
    p90_us: float

    def to_dict(self) -> dict[str, object]:
        """Serialize same-pipe issue-interval statistics to a JSON-safe mapping."""
        return {
            "pipe": self.pipe,
            "instruction_count": self.instruction_count,
            "interval_count": self.interval_count,
            "p10_us": self.p10_us,
            "p50_us": self.p50_us,
            "p90_us": self.p90_us,
        }


@dataclass(frozen=True)
class DwarfLineHotspot:
    """Source-line instruction hotspot attributed via DWARF PC start addresses."""

    source_location: str
    instruction_count: int
    instruction_share: float
    pipes: tuple[str, ...]
    pc_range_start: int
    pc_range_end: int

    def to_dict(self) -> dict[str, object]:
        """Serialize a DWARF source-line hotspot to a JSON-safe mapping."""
        return {
            "source_location": self.source_location,
            "instruction_count": self.instruction_count,
            "instruction_share": self.instruction_share,
            "pipes": list(self.pipes),
            "pc_range_start": self.pc_range_start,
            "pc_range_end": self.pc_range_end,
        }


@dataclass(frozen=True)
class P2InstructionDiveSummary:
    """P2 instruction-stream breakdown, issue-interval percentiles, and DWARF line hotspots."""

    total_instructions: int
    instruction_mix: dict[str, float]
    mte_waitflag_count: int
    mte_waitflag_share: float
    issue_interval_us: dict[str, PipeIssueIntervalStats]
    source_line_hotspots: tuple[DwarfLineHotspot, ...] = ()
    attributed_instruction_share: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize P2 instruction analysis results to a JSON-safe dictionary."""
        return {
            "total_instructions": self.total_instructions,
            "instruction_mix": dict(self.instruction_mix),
            "mte_waitflag_count": self.mte_waitflag_count,
            "mte_waitflag_share": self.mte_waitflag_share,
            "issue_interval_us": {
                pipe: stats.to_dict() for pipe, stats in self.issue_interval_us.items()
            },
            "source_line_hotspots": [spot.to_dict() for spot in self.source_line_hotspots],
            "attributed_instruction_share": self.attributed_instruction_share,
        }


@dataclass
class _MutableSubkernelRow:
    """Accumulator for joining per-subkernel counters across msprof CSV tables."""

    name: str = ""
    task_id: str = ""
    duration_us: float | None = None
    aic_mac_ratio: float | None = None
    aiv_vec_ratio: float | None = None
    mte2_wait_ratio: float | None = None
    aic_mte2_ratio: float | None = None
    aiv_mte2_ratio: float | None = None
    aic_fixpipe: float | None = None
    l2_hit_rate: float | None = None
    bank_conflict_ratio: float | None = None
    sync_stall_ratio: float | None = None
    aiv0_vec_ratio: float | None = None
    aiv1_vec_ratio: float | None = None
    hbm_read_bytes: int | None = None
    hbm_write_bytes: int | None = None
    actual_gm_bytes: int | None = None
    l0_transfer_bytes: int | None = None
    aic_total_cycles: int | None = None
    aiv_total_cycles: int | None = None
    block_dim: int | None = None

    def freeze(self, fallback_index: int) -> SubkernelMsprofMetrics:
        """Convert accumulated counters into an immutable SubkernelMsprofMetrics record."""
        mte2_wait = self.mte2_wait_ratio
        if mte2_wait is None:
            candidates = [
                v for v in (self.aiv_mte2_ratio, self.aic_mte2_ratio) if v is not None
            ]
            if candidates:
                mte2_wait = max(candidates)
        actual_gm = self.actual_gm_bytes
        if actual_gm is None and (
            self.hbm_read_bytes is not None or self.hbm_write_bytes is not None
        ):
            actual_gm = (self.hbm_read_bytes or 0) + (self.hbm_write_bytes or 0)
        return SubkernelMsprofMetrics(
            name=self.name or f"subkernel_{fallback_index}",
            duration_us=max(0.0, self.duration_us or 0.0),
            aic_mac_ratio=self.aic_mac_ratio,
            aiv_vec_ratio=self.aiv_vec_ratio,
            mte2_wait_ratio=mte2_wait,
            aic_fixpipe=self.aic_fixpipe,
            l2_hit_rate=self.l2_hit_rate,
            bank_conflict_ratio=self.bank_conflict_ratio,
            sync_stall_ratio=self.sync_stall_ratio,
            aiv0_vec_ratio=self.aiv0_vec_ratio,
            aiv1_vec_ratio=self.aiv1_vec_ratio,
            hbm_read_bytes=self.hbm_read_bytes,
            hbm_write_bytes=self.hbm_write_bytes,
            actual_gm_bytes=actual_gm,
            l0_transfer_bytes=self.l0_transfer_bytes,
            aic_total_cycles=self.aic_total_cycles,
            aiv_total_cycles=self.aiv_total_cycles,
            block_dim=self.block_dim,
        )


def _split_header_and_unit(raw_header: str) -> tuple[str, str]:
    """Normalize a CSV header name and extract any trailing parenthesized unit."""
    stripped = raw_header.strip()
    unit = ""
    match = _HEADER_UNIT_RE.search(stripped)
    if match is not None:
        unit = match.group(1).strip().lower()
        stripped = stripped[: match.start()].strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    return normalized, unit


def _parse_numeric_cell(raw_value: str | None) -> tuple[float, str] | None:
    """Parse a CSV numeric cell and optional inline unit, returning None for missing tokens."""
    if raw_value is None:
        return None
    cleaned = raw_value.strip()
    if cleaned.lower() in _MISSING_TOKENS:
        return None
    cleaned = cleaned.replace(",", "")
    match = _NUM_WITH_UNIT_RE.fullmatch(cleaned)
    if match is None:
        return None
    value = float(match.group(1))
    if not math.isfinite(value):
        return None
    unit = match.group(2).strip().lower()
    return value, unit


def _to_ratio(value: float, unit: str) -> float:
    """Convert a parsed ratio or percentage counter into [0.0, 1.0]."""
    scaled = value / 100.0 if (unit == "%" or value > 1.05) else value
    return max(0.0, min(1.0, scaled))


def _to_bytes(value: float, unit: str) -> int:
    """Convert a parsed byte or KB/MB/GB transfer quantity into integer bytes."""
    mult = _BYTE_UNIT_MULTIPLIERS.get(unit, 1)
    return max(0, int(round(value * float(mult))))


def _to_duration_us(value: float, unit: str) -> float:
    """Convert a parsed duration in ns/us/ms/s into microseconds."""
    scale = _TIME_UNIT_TO_US.get(unit, 1.0)
    return max(0.0, value * scale)


def _match_canonical_table(path: Path) -> str | None:
    """Map a CSV filename to one of the six canonical msprof op report table names."""
    stem_compact = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    for canonical_name, token in _CANONICAL_TABLES:
        if stem_compact == token or stem_compact.endswith(token):
            return canonical_name
    return None


def _discover_table_groups(root: Path) -> list[dict[str, Path]]:
    """Locate msprof op CSV files across flat, nested, or per-subkernel subdirectories."""
    matched: list[tuple[Path, str]] = []
    for candidate in sorted(root.rglob("*.csv")):
        if not candidate.is_file():
            continue
        canonical = _match_canonical_table(candidate)
        if canonical is not None:
            matched.append((candidate, canonical))
    if not matched:
        return []
    table_counts: dict[str, int] = {}
    for _path, canonical in matched:
        table_counts[canonical] = table_counts.get(canonical, 0) + 1
    if all(count == 1 for count in table_counts.values()):
        return [{canonical: path for path, canonical in matched}]
    grouped: dict[Path, dict[str, Path]] = {}
    for path, canonical in matched:
        grouped.setdefault(path.parent, {})[canonical] = path
    return [grouped[parent] for parent in sorted(grouped)]


def _read_csv_rows(path: Path) -> list[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]]:
    """Read a CSV file into normalized (header_base, unit) -> (raw_value, header_unit) rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(raw_rows) < 2:
        return []
    raw_headers = raw_rows[0]
    parsed_headers = [_split_header_and_unit(h) for h in raw_headers]

    if (
        len(parsed_headers) == 2
        and parsed_headers[0][0] in ("item", "metric", "key", "field", "attribute", "property")
        and parsed_headers[1][0] in ("value", "val", "data")
    ):
        kv_row: dict[str, tuple[str, str]] = {}
        kv_headers: list[tuple[str, str]] = []
        for data_row in raw_rows[1:]:
            if len(data_row) < 2:
                continue
            key_base, key_unit = _split_header_and_unit(data_row[0])
            if not key_base:
                continue
            kv_headers.append((key_base, key_unit))
            kv_row[key_base] = (data_row[1].strip(), key_unit)
        return [(kv_headers, kv_row)] if kv_row else []

    out: list[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]] = []
    for data_row in raw_rows[1:]:
        row_map: dict[str, tuple[str, str]] = {}
        for idx, (key_base, key_unit) in enumerate(parsed_headers):
            if not key_base or idx >= len(data_row):
                continue
            cell = data_row[idx].strip()
            if key_base not in row_map or cell:
                row_map[key_base] = (cell, key_unit)
        if row_map:
            out.append((parsed_headers, row_map))
    return out


def _first_text(row: Mapping[str, tuple[str, str]], *keys: str) -> str:
    """Return the first non-empty string value among candidate normalized CSV keys."""
    for key in keys:
        entry = row.get(key)
        if entry is None:
            continue
        raw_val = entry[0].strip()
        if raw_val and raw_val.lower() not in _MISSING_TOKENS:
            return raw_val
    return ""


def _first_numeric(
    row: Mapping[str, tuple[str, str]], *keys: str
) -> tuple[float, str] | None:
    """Return the first parseable numeric cell and resolved unit among candidate keys."""
    for key in keys:
        entry = row.get(key)
        if entry is None:
            continue
        raw_val, header_unit = entry
        parsed = _parse_numeric_cell(raw_val)
        if parsed is None:
            continue
        val, cell_unit = parsed
        return val, (cell_unit or header_unit)
    return None


def _is_tall_metric_table(
    rows: Sequence[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]],
    *,
    label_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
    wide_keys: tuple[str, ...],
) -> bool:
    """Return True if a CSV encodes metrics in tall (label, value) rows instead of wide columns."""
    if not rows:
        return False
    first_map = rows[0][1]
    if any(key in first_map for key in wide_keys):
        return False
    has_label = any(key in first_map for key in label_keys)
    has_value = any(key in first_map for key in value_keys)
    return has_label and has_value


def _pivot_tall_rows(
    rows: Sequence[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]],
    *,
    label_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
) -> list[dict[str, tuple[str, str]]]:
    """Fold tall (metric_name, value) CSV rows into wide per-subkernel row dictionaries."""
    grouped: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    order: list[tuple[str, str]] = []
    for _headers, row in rows:
        op_name = _first_text(row, "op_name", "kernel_name", "task_name")
        task_id = _first_text(row, "task_id", "subkernel_index", "index")
        group_key = (task_id, op_name)
        if group_key not in grouped:
            base: dict[str, tuple[str, str]] = {}
            if op_name:
                base["op_name"] = (op_name, "")
            if task_id:
                base["task_id"] = (task_id, "")
            grouped[group_key] = base
            order.append(group_key)
        label_raw = _first_text(row, *label_keys)
        if not label_raw:
            continue
        label_base, label_unit = _split_header_and_unit(label_raw)
        if not label_base:
            continue
        for val_key in value_keys:
            if val_key in row and row[val_key][0].strip():
                raw_val, val_unit = row[val_key]
                grouped[group_key][label_base] = (raw_val, val_unit or label_unit)
                break
    return [grouped[key] for key in order]


def _match_or_create_slot(
    slots: list[_MutableSubkernelRow],
    used_indices: set[int],
    *,
    op_name: str,
    task_id: str,
    row_index: int,
) -> _MutableSubkernelRow:
    """Align a secondary table row to its subkernel slot by ID, name, or row index."""
    if task_id:
        for idx, slot in enumerate(slots):
            if idx in used_indices:
                continue
            if slot.task_id == task_id and (not op_name or not slot.name or slot.name == op_name):
                used_indices.add(idx)
                return slot
    if op_name:
        for idx, slot in enumerate(slots):
            if idx in used_indices:
                continue
            if slot.name == op_name:
                used_indices.add(idx)
                return slot
    if row_index < len(slots) and row_index not in used_indices:
        used_indices.add(row_index)
        slot = slots[row_index]
        if op_name and not slot.name:
            slot.name = op_name
        if task_id and not slot.task_id:
            slot.task_id = task_id
        return slot
    new_slot = _MutableSubkernelRow(name=op_name, task_id=task_id)
    used_indices.add(len(slots))
    slots.append(new_slot)
    return new_slot


def _apply_basic_info(slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]) -> None:
    """Populate identity, duration, and block dimensions from an OpBasicInfo row."""
    name = _first_text(row, "op_name", "kernel_name", "task_name", "name")
    if name:
        slot.name = name
    task_id = _first_text(row, "task_id", "subkernel_index", "index")
    if task_id:
        slot.task_id = task_id
    dur = _first_numeric(
        row,
        "task_duration",
        "op_duration",
        "duration",
        "dur_us",
        "duration_us",
        "task_duration_us",
        "total_time",
    )
    if dur is not None:
        slot.duration_us = _to_duration_us(dur[0], dur[1])
    else:
        dur_ns = _first_numeric(row, "dur_ns", "task_duration_ns", "duration_ns")
        if dur_ns is not None:
            slot.duration_us = _to_duration_us(dur_ns[0], "ns")
    block_dim = _first_numeric(row, "block_dim", "mix_block_dim")
    if block_dim is not None:
        slot.block_dim = max(0, int(round(block_dim[0])))


def _apply_pipe_utilization(
    slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]
) -> None:
    """Populate pipe utilization ratios and cycle totals from a PipeUtilization row."""
    if slot.duration_us is None:
        dur = _first_numeric(row, "task_duration", "duration", "dur_us", "total_time")
        if dur is not None:
            slot.duration_us = _to_duration_us(dur[0], dur[1])

    mac = _first_numeric(
        row,
        "aic_mac_ratio",
        "mac_ratio",
        "aic_mad",
        "aic_cube_ratio",
        "cube_ratio",
        "pipe_mac_ratio",
        "mac",
        "cube",
    )
    if mac is not None:
        slot.aic_mac_ratio = _to_ratio(mac[0], mac[1])

    vec = _first_numeric(
        row,
        "aiv_vec_ratio",
        "vec_ratio",
        "aiv_vec",
        "pipe_vec_ratio",
        "vec",
        "vector",
    )
    if vec is not None:
        slot.aiv_vec_ratio = _to_ratio(vec[0], vec[1])

    aiv0 = _first_numeric(row, "aiv0_vec_ratio", "aiv0_vec", "vec_ratio_aiv0", "aiv_vec_ratio_0")
    if aiv0 is not None:
        slot.aiv0_vec_ratio = _to_ratio(aiv0[0], aiv0[1])

    aiv1 = _first_numeric(row, "aiv1_vec_ratio", "aiv1_vec", "vec_ratio_aiv1", "aiv_vec_ratio_1")
    if aiv1 is not None:
        slot.aiv1_vec_ratio = _to_ratio(aiv1[0], aiv1[1])

    explicit_mte2 = _first_numeric(row, "mte2_wait_ratio", "mte2_ratio", "pipe_mte2_ratio", "mte2")
    if explicit_mte2 is not None:
        slot.mte2_wait_ratio = _to_ratio(explicit_mte2[0], explicit_mte2[1])

    aiv_mte2 = _first_numeric(row, "aiv_mte2_ratio", "aiv_mte2")
    if aiv_mte2 is not None:
        slot.aiv_mte2_ratio = _to_ratio(aiv_mte2[0], aiv_mte2[1])

    aic_mte2 = _first_numeric(row, "aic_mte2_ratio", "aic_mte2")
    if aic_mte2 is not None:
        slot.aic_mte2_ratio = _to_ratio(aic_mte2[0], aic_mte2[1])

    fixpipe = _first_numeric(
        row, "aic_fixpipe", "aic_fixpipe_ratio", "fixpipe_ratio", "pipe_fixpipe_ratio", "fixpipe"
    )
    if fixpipe is not None:
        slot.aic_fixpipe = _to_ratio(fixpipe[0], fixpipe[1])

    sync_stall = _first_numeric(
        row,
        "sync_stall_ratio",
        "pipe_wait_ratio",
        "sync_wait_ratio",
        "wait_ratio",
        "aic_wait_ratio",
        "aiv_wait_ratio",
    )
    if sync_stall is not None:
        slot.sync_stall_ratio = _to_ratio(sync_stall[0], sync_stall[1])

    aic_cycles = _first_numeric(row, "aic_total_cycles", "aic_cycles")
    if aic_cycles is not None:
        slot.aic_total_cycles = max(0, int(round(aic_cycles[0])))

    aiv_cycles = _first_numeric(row, "aiv_total_cycles", "aiv_cycles")
    if aiv_cycles is not None:
        slot.aiv_total_cycles = max(0, int(round(aiv_cycles[0])))


def _apply_memory(slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]) -> None:
    """Populate HBM/GM read and write byte totals from a Memory.csv row."""
    read_val = _first_numeric(
        row,
        "hbm_read_bytes",
        "gm_read_bytes",
        "read_main_memory_bytes",
        "read_main_memory_datas",
        "read_main_memory_datasize",
        "read_main_memory",
        "hbm_read_datasize",
        "hbm_read",
        "gm_read_datasize",
        "gm_read",
        "memory_read",
    )
    if read_val is not None:
        slot.hbm_read_bytes = _to_bytes(read_val[0], read_val[1])

    write_val = _first_numeric(
        row,
        "hbm_write_bytes",
        "gm_write_bytes",
        "write_main_memory_bytes",
        "write_main_memory_datas",
        "write_main_memory_datasize",
        "write_main_memory",
        "hbm_write_datasize",
        "hbm_write",
        "gm_write_datasize",
        "gm_write",
        "memory_write",
    )
    if write_val is not None:
        slot.hbm_write_bytes = _to_bytes(write_val[0], write_val[1])

    total_val = _first_numeric(
        row,
        "actual_gm_bytes",
        "total_gm_bytes",
        "hbm_total_bytes",
        "hbm_total",
        "gm_total",
        "main_memory_total",
    )
    if total_val is not None:
        slot.actual_gm_bytes = _to_bytes(total_val[0], total_val[1])


def _apply_memory_l0(slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]) -> None:
    """Accumulate on-chip L0/L1 data transfer volume from a MemoryL0.csv row."""
    l0_keys = (
        "l0_transfer_bytes",
        "l0a_datasize",
        "l0b_datasize",
        "l0c_datasize",
        "l0a_read_bytes",
        "l0b_read_bytes",
        "l0c_write_bytes",
        "l0a_to_cube",
        "l0b_to_cube",
        "cube_to_l0c",
        "l1_to_l0a",
        "l1_to_l0b",
        "l0c_to_ub",
        "l0c_to_gm",
    )
    total = 0
    found = False
    for key in l0_keys:
        parsed = _first_numeric(row, key)
        if parsed is not None:
            total += _to_bytes(parsed[0], parsed[1])
            found = True
    if found:
        slot.l0_transfer_bytes = total


def _apply_resource_conflict(
    slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]
) -> None:
    """Populate bank/resource conflict ratio from a ResourceConflictRatio.csv row."""
    conflict_keys = (
        "bank_conflict_ratio",
        "ub_bank_conflict_ratio",
        "l1_bank_conflict_ratio",
        "resource_conflict_ratio",
        "vec_bankgroup_cflt_ratio",
        "vec_bank_cflt_ratio",
        "vec_resc_cflt_ratio",
        "aiv_vec_stu_cflt_ratio",
        "aiv_vec_ldu_cflt_ratio",
        "aiv_vec_sfu_cflt_ratio",
        "aic_resource_conflict_ratio",
        "aiv_resource_conflict_ratio",
        "bank_conflict",
        "resource_conflict",
    )
    values: list[float] = []
    for key in conflict_keys:
        parsed = _first_numeric(row, key)
        if parsed is not None:
            values.append(_to_ratio(parsed[0], parsed[1]))
    if values:
        slot.bank_conflict_ratio = max(values)
    if slot.sync_stall_ratio is None:
        wait_keys = (
            "aiv_vec_wait_ratio",
            "aiv_mte2_wait_ratio",
            "aiv_mte3_wait_ratio",
            "aic_cube_wait_ratio",
            "aic_mte1_wait_ratio",
            "aic_mte2_wait_ratio",
            "aic_mte3_wait_ratio",
        )
        wait_vals: list[float] = []
        for key in wait_keys:
            parsed = _first_numeric(row, key)
            if parsed is not None:
                wait_vals.append(_to_ratio(parsed[0], parsed[1]))
        if wait_vals:
            slot.sync_stall_ratio = max(wait_vals)


def _apply_l2_cache(slot: _MutableSubkernelRow, row: Mapping[str, tuple[str, str]]) -> None:
    """Populate L2 cache hit rate from an L2Cache.csv row without fabricating defaults."""
    direct = _first_numeric(
        row,
        "l2_hit_rate",
        "l2_cache_hit_rate",
        "hit_rate",
        "cache_hit_rate",
        "l2_cache_hit",
    )
    if direct is not None:
        slot.l2_hit_rate = _to_ratio(direct[0], direct[1])
        return
    hits_r = _first_numeric(row, "read_hit", "l2_read_hit", "hit_count")
    miss_r = _first_numeric(row, "read_miss", "l2_read_miss", "miss_count")
    hits_w = _first_numeric(row, "write_hit", "l2_write_hit")
    miss_w = _first_numeric(row, "write_miss", "l2_write_miss")
    total_hits = (hits_r[0] if hits_r is not None else 0.0) + (
        hits_w[0] if hits_w is not None else 0.0
    )
    total_miss = (miss_r[0] if miss_r is not None else 0.0) + (
        miss_w[0] if miss_w is not None else 0.0
    )
    if (hits_r is not None or hits_w is not None) and (total_hits + total_miss) > 0.0:
        slot.l2_hit_rate = max(0.0, min(1.0, total_hits / (total_hits + total_miss)))
        return
    cf_hit_keys = (
        "aic_read_close_hit",
        "aic_read_far_hit",
        "aic_write_close_hit",
        "aic_write_far_hit",
        "aiv_read_close_hit",
        "aiv_read_far_hit",
        "aiv_write_close_hit",
        "aiv_write_far_hit",
    )
    cf_miss_keys = (
        "aic_read_close_miss",
        "aic_read_far_miss",
        "aic_write_close_miss",
        "aic_write_far_miss",
        "aiv_read_close_miss",
        "aiv_read_far_miss",
        "aiv_write_close_miss",
        "aiv_write_far_miss",
    )
    cf_hits = 0.0
    cf_misses = 0.0
    cf_seen = False
    for key in cf_hit_keys:
        parsed = _first_numeric(row, key)
        if parsed is not None:
            cf_hits += max(0.0, parsed[0])
            cf_seen = True
    for key in cf_miss_keys:
        parsed = _first_numeric(row, key)
        if parsed is not None:
            cf_misses += max(0.0, parsed[0])
            cf_seen = True
    if cf_seen and (cf_hits + cf_misses) > 0.0:
        slot.l2_hit_rate = max(0.0, min(1.0, cf_hits / (cf_hits + cf_misses)))
        return
    rate_keys = (
        "read_hit_rate",
        "write_hit_rate",
        "aiv_read_hit_rate",
        "aiv_write_hit_rate",
        "aic_read_hit_rate",
        "aic_write_hit_rate",
    )
    rates: list[float] = []
    for key in rate_keys:
        parsed = _first_numeric(row, key)
        if parsed is not None:
            rates.append(_to_ratio(parsed[0], parsed[1]))
    if rates:
        slot.l2_hit_rate = sum(rates) / float(len(rates))


_SUBKERNEL_ID_KEYS = frozenset({"op_name", "kernel_name", "task_name", "task_id"})
_PER_BLOCK_SKIP_KEYS = frozenset(
    {"block_id", "sub_block_id", "op_name", "kernel_name", "task_name", "task_id"}
)
_PER_BLOCK_ADDITIVE_SUFFIXES = (
    "_datas",
    "_datasize",
    "_bytes",
    "_cycles",
    "_hit",
    "_miss",
    "_victim",
    "_instructions",
    "_count",
)
_PER_BLOCK_RATE_SUFFIXES = ("_ratio", "_rate", "_bw")
_PER_BLOCK_TIME_KEYS = frozenset(
    {"aiv_time", "aic_time", "task_duration", "duration", "dur_us", "total_time"}
)


def _is_per_block_metric_table(
    rows: Sequence[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]],
) -> bool:
    """Return True when rows enumerate block_id slices without per-subkernel identifiers."""
    if not rows:
        return False
    first_map = rows[0][1]
    if "block_id" not in first_map:
        return False
    return not any(
        key in row and _first_text(row, key)
        for _headers, row in rows
        for key in _SUBKERNEL_ID_KEYS
    )


def _collapse_per_block_rows(
    canonical_name: str,
    rows: Sequence[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]],
) -> list[dict[str, tuple[str, str]]]:
    """Collapse per-block hardware rows of one subkernel into a single aggregated row."""
    if not rows:
        return []
    ordered_keys: list[str] = []
    header_units: dict[str, str] = {}
    seen_keys: set[str] = set()
    for headers, _row in rows:
        for key_base, key_unit in headers:
            if not key_base or key_base in _PER_BLOCK_SKIP_KEYS:
                continue
            if key_base not in seen_keys:
                seen_keys.add(key_base)
                ordered_keys.append(key_base)
            if key_unit and key_base not in header_units:
                header_units[key_base] = key_unit
    for _headers, row in rows:
        for key_base, (_val, unit) in row.items():
            if not key_base or key_base in _PER_BLOCK_SKIP_KEYS:
                continue
            if key_base not in seen_keys:
                seen_keys.add(key_base)
                ordered_keys.append(key_base)
            if unit and key_base not in header_units:
                header_units[key_base] = unit

    collapsed: dict[str, tuple[str, str]] = {}
    for key in ordered_keys:
        unit = header_units.get(key, "")
        is_rate = key.endswith(_PER_BLOCK_RATE_SUFFIXES)
        is_additive = (not is_rate) and key.endswith(_PER_BLOCK_ADDITIVE_SUFFIXES)
        is_byte_sum = is_additive and key.endswith(("_datas", "_datasize", "_bytes"))
        is_time = key in _PER_BLOCK_TIME_KEYS or unit in _TIME_UNIT_TO_US

        if is_byte_sum:
            total_bytes = 0.0
            found = False
            for _headers, row in rows:
                entry = row.get(key)
                if entry is None:
                    continue
                parsed = _parse_numeric_cell(entry[0])
                if parsed is None:
                    continue
                val, cell_unit = parsed
                eff_unit = cell_unit or entry[1] or unit
                mult = _BYTE_UNIT_MULTIPLIERS.get(eff_unit, 1)
                total_bytes += val * float(mult)
                found = True
            if found:
                collapsed[key] = (f"{total_bytes:.6f}", "b")
            continue

        if is_additive:
            total_val = 0.0
            found = False
            for _headers, row in rows:
                entry = row.get(key)
                if entry is None:
                    continue
                parsed = _parse_numeric_cell(entry[0])
                if parsed is None:
                    continue
                total_val += parsed[0]
                found = True
            if found:
                collapsed[key] = (f"{total_val:.6f}", unit)
            continue

        if is_rate:
            weighted_sum = 0.0
            weight_total = 0.0
            plain_sum = 0.0
            plain_count = 0
            out_unit = unit
            for _headers, row in rows:
                entry = row.get(key)
                if entry is None:
                    continue
                parsed = _parse_numeric_cell(entry[0])
                if parsed is None:
                    continue
                val, cell_unit = parsed
                eff_unit = cell_unit or entry[1] or unit
                if not out_unit and eff_unit:
                    out_unit = eff_unit
                weight_val = 0.0
                if key.startswith("aiv_") and "aiv_time" in row:
                    w_parsed = _parse_numeric_cell(row["aiv_time"][0])
                    if w_parsed is not None and w_parsed[0] > 0.0:
                        weight_val = w_parsed[0]
                elif key.startswith("aic_") and "aic_time" in row:
                    w_parsed = _parse_numeric_cell(row["aic_time"][0])
                    if w_parsed is not None and w_parsed[0] > 0.0:
                        weight_val = w_parsed[0]
                else:
                    for t_key in ("aiv_time", "aic_time"):
                        if t_key in row:
                            w_parsed = _parse_numeric_cell(row[t_key][0])
                            if w_parsed is not None and w_parsed[0] > weight_val:
                                weight_val = w_parsed[0]
                plain_sum += val
                plain_count += 1
                if weight_val > 0.0:
                    weighted_sum += val * weight_val
                    weight_total += weight_val
            if plain_count > 0:
                agg_val = (
                    weighted_sum / weight_total
                    if weight_total > 0.0
                    else (plain_sum / float(plain_count))
                )
                collapsed[key] = (f"{agg_val:.6f}", out_unit)
            continue

        if is_time:
            time_vals: list[float] = []
            out_unit = unit
            for _headers, row in rows:
                entry = row.get(key)
                if entry is None:
                    continue
                parsed = _parse_numeric_cell(entry[0])
                if parsed is None:
                    continue
                time_vals.append(parsed[0])
                if not out_unit and (parsed[1] or entry[1]):
                    out_unit = parsed[1] or entry[1]
            if time_vals:
                collapsed[key] = (f"{max(time_vals):.6f}", out_unit)
            continue

        plain_vals: list[float] = []
        for _headers, row in rows:
            entry = row.get(key)
            if entry is None:
                continue
            parsed = _parse_numeric_cell(entry[0])
            if parsed is not None:
                plain_vals.append(parsed[0])
        if plain_vals:
            collapsed[key] = (f"{sum(plain_vals) / float(len(plain_vals)):.6f}", unit)

    if canonical_name == "PipeUtilization.csv":
        vec0_vals: list[tuple[float, str]] = []
        vec1_vals: list[tuple[float, str]] = []
        for _headers, row in rows:
            sub_id = _first_text(row, "sub_block_id").lower()
            entry = row.get("aiv_vec_ratio")
            if entry is None:
                continue
            parsed = _parse_numeric_cell(entry[0])
            if parsed is None:
                continue
            eff_unit = parsed[1] or entry[1]
            if sub_id == "vector0":
                vec0_vals.append((parsed[0], eff_unit))
            elif sub_id == "vector1":
                vec1_vals.append((parsed[0], eff_unit))
        if (
            vec0_vals
            and vec1_vals
            and "aiv0_vec_ratio" not in collapsed
            and "aiv1_vec_ratio" not in collapsed
        ):
            mean0 = sum(v for v, _ in vec0_vals) / float(len(vec0_vals))
            mean1 = sum(v for v, _ in vec1_vals) / float(len(vec1_vals))
            collapsed["aiv0_vec_ratio"] = (f"{mean0:.6f}", vec0_vals[0][1])
            collapsed["aiv1_vec_ratio"] = (f"{mean1:.6f}", vec1_vals[0][1])

    return [collapsed] if collapsed else [row for _headers, row in rows]


def _normalize_table_rows(
    canonical_name: str,
    raw_rows: Sequence[tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]],
) -> list[dict[str, tuple[str, str]]]:
    """Normalize either wide or tall msprof CSV rows into wide per-subkernel row mappings."""
    if canonical_name == "PipeUtilization.csv":
        if _is_tall_metric_table(
            raw_rows,
            label_keys=("pipe", "metric", "item", "name"),
            value_keys=("ratio", "value", "utilization"),
            wide_keys=("aic_mac_ratio", "aiv_vec_ratio", "mac_ratio", "vec_ratio", "mte2_ratio"),
        ):
            return _pivot_tall_rows(
                raw_rows,
                label_keys=("pipe", "metric", "item", "name"),
                value_keys=("ratio", "value", "utilization"),
            )
    elif canonical_name == "Memory.csv":
        if _is_tall_metric_table(
            raw_rows,
            label_keys=("memory", "memory_type", "metric", "item", "name"),
            value_keys=("datasize", "bytes", "size", "value"),
            wide_keys=(
                "hbm_read_bytes",
                "hbm_write_bytes",
                "hbm_read",
                "hbm_write",
                "gm_read_bytes",
                "gm_write_bytes",
                "read_main_memory_datasize",
                "write_main_memory_datasize",
            ),
        ):
            return _pivot_tall_rows(
                raw_rows,
                label_keys=("memory", "memory_type", "metric", "item", "name"),
                value_keys=("datasize", "bytes", "size", "value"),
            )
    elif canonical_name == "MemoryL0.csv":
        if _is_tall_metric_table(
            raw_rows,
            label_keys=("memory", "pipe", "metric", "item", "name"),
            value_keys=("datasize", "bytes", "size", "value"),
            wide_keys=("l0_transfer_bytes", "l0a_datasize", "l0b_datasize", "l0c_datasize"),
        ):
            return _pivot_tall_rows(
                raw_rows,
                label_keys=("memory", "pipe", "metric", "item", "name"),
                value_keys=("datasize", "bytes", "size", "value"),
            )
    elif canonical_name == "ResourceConflictRatio.csv":
        if _is_tall_metric_table(
            raw_rows,
            label_keys=("metric", "item", "conflict_type", "name"),
            value_keys=("ratio", "value"),
            wide_keys=(
                "bank_conflict_ratio",
                "resource_conflict_ratio",
                "vec_bankgroup_cflt_ratio",
                "vec_bank_cflt_ratio",
            ),
        ):
            return _pivot_tall_rows(
                raw_rows,
                label_keys=("metric", "item", "conflict_type", "name"),
                value_keys=("ratio", "value"),
            )
    elif canonical_name == "L2Cache.csv":
        if _is_tall_metric_table(
            raw_rows,
            label_keys=("metric", "item", "cache_metric", "name"),
            value_keys=("ratio", "hit_rate", "value"),
            wide_keys=("l2_hit_rate", "l2_cache_hit_rate", "hit_rate", "read_hit_rate"),
        ):
            return _pivot_tall_rows(
                raw_rows,
                label_keys=("metric", "item", "cache_metric", "name"),
                value_keys=("ratio", "hit_rate", "value"),
            )
    if _is_per_block_metric_table(raw_rows):
        return _collapse_per_block_rows(canonical_name, raw_rows)
    return [row for _headers, row in raw_rows]


def _parse_single_table_group(
    table_map: Mapping[str, Path],
) -> tuple[list[SubkernelMsprofMetrics], list[str]]:
    """Join all available msprof CSV tables in one directory group into subkernel records."""
    slots: list[_MutableSubkernelRow] = []
    parsed_tables: list[str] = []

    table_order = (
        "OpBasicInfo.csv",
        "PipeUtilization.csv",
        "Memory.csv",
        "MemoryL0.csv",
        "ResourceConflictRatio.csv",
        "L2Cache.csv",
    )
    appliers = {
        "OpBasicInfo.csv": _apply_basic_info,
        "PipeUtilization.csv": _apply_pipe_utilization,
        "Memory.csv": _apply_memory,
        "MemoryL0.csv": _apply_memory_l0,
        "ResourceConflictRatio.csv": _apply_resource_conflict,
        "L2Cache.csv": _apply_l2_cache,
    }

    for table_name in table_order:
        csv_path = table_map.get(table_name)
        if csv_path is None:
            continue
        raw_rows = _read_csv_rows(csv_path)
        if not raw_rows:
            continue
        parsed_tables.append(table_name)
        wide_rows = _normalize_table_rows(table_name, raw_rows)
        used_indices: set[int] = set()
        applier = appliers[table_name]
        for row_idx, row in enumerate(wide_rows):
            op_name = _first_text(row, "op_name", "kernel_name", "task_name", "name")
            task_id = _first_text(row, "task_id", "subkernel_index", "index")
            slot = _match_or_create_slot(
                slots,
                used_indices,
                op_name=op_name,
                task_id=task_id,
                row_index=row_idx,
            )
            applier(slot, row)

    return [slot.freeze(idx) for idx, slot in enumerate(slots)], parsed_tables


def _weighted_ratio(
    subkernels: Sequence[SubkernelMsprofMetrics],
    values: Sequence[float | None],
) -> float | None:
    """Compute a Task Duration-weighted average across subkernels with measured values."""
    weighted_sum = 0.0
    weight_total = 0.0
    present_values: list[float] = []
    for sub, value in zip(subkernels, values):
        if value is None:
            continue
        present_values.append(value)
        weight = max(0.0, sub.duration_us)
        weighted_sum += weight * value
        weight_total += weight
    if not present_values:
        return None
    if weight_total > 0.0:
        return weighted_sum / weight_total
    return sum(present_values) / float(len(present_values))


def _sum_optional_int(values: Sequence[int | None]) -> int | None:
    """Sum measured integer counters across subkernels, returning None if all are absent."""
    present = [val for val in values if val is not None]
    return sum(present) if present else None


def aggregate_subkernel_msprof(
    subkernels: Sequence[SubkernelMsprofMetrics],
    *,
    parsed_tables: Sequence[str] = (),
) -> AggregatedMsprofReport:
    """Aggregate multiple subkernel msprof records using strict Task Duration weighting."""
    frozen_subkernels = tuple(subkernels)
    total_duration_us = sum(max(0.0, sub.duration_us) for sub in frozen_subkernels)
    if frozen_subkernels:
        dominant = max(frozen_subkernels, key=lambda item: item.duration_us)
        dominant_name = dominant.name
        dominant_share = (
            dominant.duration_us / total_duration_us if total_duration_us > 0.0 else 0.0
        )
    else:
        dominant_name = ""
        dominant_share = 0.0

    hbm_read = _sum_optional_int([sub.hbm_read_bytes for sub in frozen_subkernels])
    hbm_write = _sum_optional_int([sub.hbm_write_bytes for sub in frozen_subkernels])
    actual_gm = _sum_optional_int([sub.actual_gm_bytes for sub in frozen_subkernels])
    if actual_gm is None and (hbm_read is not None or hbm_write is not None):
        actual_gm = (hbm_read or 0) + (hbm_write or 0)

    return AggregatedMsprofReport(
        total_duration_us=total_duration_us,
        subkernels=frozen_subkernels,
        dominant_subkernel_name=dominant_name,
        dominant_subkernel_share=dominant_share,
        aic_mac_ratio=_weighted_ratio(
            frozen_subkernels, [sub.aic_mac_ratio for sub in frozen_subkernels]
        ),
        aiv_vec_ratio=_weighted_ratio(
            frozen_subkernels, [sub.aiv_vec_ratio for sub in frozen_subkernels]
        ),
        mte2_wait_ratio=_weighted_ratio(
            frozen_subkernels, [sub.mte2_wait_ratio for sub in frozen_subkernels]
        ),
        aic_fixpipe=_weighted_ratio(
            frozen_subkernels, [sub.aic_fixpipe for sub in frozen_subkernels]
        ),
        l2_hit_rate=_weighted_ratio(
            frozen_subkernels, [sub.l2_hit_rate for sub in frozen_subkernels]
        ),
        bank_conflict_ratio=_weighted_ratio(
            frozen_subkernels, [sub.bank_conflict_ratio for sub in frozen_subkernels]
        ),
        sync_stall_ratio=_weighted_ratio(
            frozen_subkernels, [sub.sync_stall_ratio for sub in frozen_subkernels]
        ),
        aiv0_vec_ratio=_weighted_ratio(
            frozen_subkernels, [sub.aiv0_vec_ratio for sub in frozen_subkernels]
        ),
        aiv1_vec_ratio=_weighted_ratio(
            frozen_subkernels, [sub.aiv1_vec_ratio for sub in frozen_subkernels]
        ),
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
        actual_gm_bytes=actual_gm,
        l0_transfer_bytes=_sum_optional_int([sub.l0_transfer_bytes for sub in frozen_subkernels]),
        aic_total_cycles=_sum_optional_int([sub.aic_total_cycles for sub in frozen_subkernels]),
        aiv_total_cycles=_sum_optional_int([sub.aiv_total_cycles for sub in frozen_subkernels]),
        parsed_tables=tuple(dict.fromkeys(parsed_tables)),
    )


def parse_msprof_op_report(report_dir: Path | str) -> AggregatedMsprofReport:
    """Parse an msprof op report directory into a duration-weighted multi-subkernel summary."""
    root = Path(report_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"msprof report directory does not exist: {root}")
    groups = _discover_table_groups(root)
    all_subkernels: list[SubkernelMsprofMetrics] = []
    all_tables: list[str] = []
    for table_map in groups:
        subkernels, tables = _parse_single_table_group(table_map)
        all_subkernels.extend(subkernels)
        all_tables.extend(tables)
    return aggregate_subkernel_msprof(all_subkernels, parsed_tables=all_tables)


def _parse_pc_int(raw_pc: object) -> int | None:
    """Parse an instruction program counter from int, hex string, or decimal string."""
    if isinstance(raw_pc, bool) or raw_pc is None:
        return None
    if isinstance(raw_pc, int):
        return raw_pc if raw_pc >= 0 else None
    if isinstance(raw_pc, str):
        text = raw_pc.strip()
        if not text or text.lower() in _MISSING_TOKENS:
            return None
        try:
            if text.lower().startswith("0x"):
                return int(text, 16)
            return int(text, 10)
        except ValueError:
            hex_match = _HEX_RE.search(text)
            if hex_match is not None:
                return int(hex_match.group(0), 16)
    return None


def parse_pc_start_addr(
    source: Path | str | Mapping[int, str],
) -> tuple[tuple[int, int | None, str], ...]:
    """Parse DWARF pc_start_addr.txt entries into sorted (start_pc, end_pc, location) tuples."""
    if isinstance(source, Mapping):
        entries = [
            (int(pc), None, str(loc).strip())
            for pc, loc in source.items()
            if not isinstance(pc, bool) and str(loc).strip()
        ]
        entries.sort(key=lambda item: item[0])
        return tuple(entries)

    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"pc_start_addr file does not exist: {source}")
        text = source.read_text(encoding="utf-8-sig")
    else:
        candidate = Path(source)
        if "\n" not in source and candidate.is_file():
            text = candidate.read_text(encoding="utf-8-sig")
        else:
            text = source

    parsed: list[tuple[int, int | None, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith(("pc_start", "pc start", "start_pc", "address,")):
            continue
        hex_matches = list(_HEX_RE.finditer(line))
        if not hex_matches:
            continue
        start_pc = int(hex_matches[0].group(0), 16)
        end_pc: int | None = None
        if len(hex_matches) >= 2:
            second_pc = int(hex_matches[1].group(0), 16)
            if second_pc > start_pc:
                end_pc = second_pc
            remainder = (
                line[: hex_matches[0].start()] + " " + line[hex_matches[1].end() :]
            ).strip()
        else:
            remainder = (
                line[: hex_matches[0].start()] + " " + line[hex_matches[0].end() :]
            ).strip()
        location = remainder.strip(" ,:\t")
        if location:
            parsed.append((start_pc, end_pc, location))
    parsed.sort(key=lambda item: item[0])
    return tuple(parsed)


def _extract_trace_records(
    trace_source: Path | str | Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], Path | None]:
    """Load instruction trace records from a sequence, JSON/JSONL/CSV file, directory, or text."""
    if not isinstance(trace_source, (Path, str)):
        return list(trace_source), None

    auto_pc_path: Path | None = None
    if isinstance(trace_source, Path) or ("\n" not in trace_source and Path(trace_source).exists()):
        path = Path(trace_source)
        if path.is_dir():
            pc_candidate = next(iter(sorted(path.rglob("pc_start_addr.txt"))), None)
            if pc_candidate is not None and pc_candidate.is_file():
                auto_pc_path = pc_candidate
            trace_files = [
                f
                for f in sorted(path.rglob("*"))
                if f.is_file()
                and f.suffix.lower() in (".json", ".jsonl", ".csv")
                and f.name.lower() != "pc_start_addr.txt"
            ]
            if not trace_files:
                return [], auto_pc_path
            path = trace_files[0]
        if not path.is_file():
            raise FileNotFoundError(f"P2 instruction trace source does not exist: {path}")
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle)), auto_pc_path
        text = path.read_text(encoding="utf-8-sig")
    else:
        text = trace_source

    stripped = text.strip()
    if not stripped:
        return [], auto_pc_path
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, Mapping)], auto_pc_path
            if isinstance(decoded, Mapping):
                events = decoded.get("traceEvents")
                if isinstance(events, list):
                    return [item for item in events if isinstance(item, Mapping)], auto_pc_path
                return [decoded], auto_pc_path
        except json.JSONDecodeError:
            jsonl_records: list[Mapping[str, object]] = []
            for line in stripped.splitlines():
                row_text = line.strip()
                if not row_text:
                    continue
                item = json.loads(row_text)
                if isinstance(item, Mapping):
                    jsonl_records.append(item)
            return jsonl_records, auto_pc_path
    csv_reader = csv.DictReader(stripped.splitlines())
    return list(csv_reader), auto_pc_path


def _normalize_record_mapping(record: Mapping[str, object]) -> dict[str, tuple[object, str]]:
    """Index a trace record by normalized key while retaining any unit from its header."""
    out: dict[str, tuple[object, str]] = {}
    for raw_key, raw_val in record.items():
        base, unit = _split_header_and_unit(str(raw_key))
        if base:
            out[base] = (raw_val, unit)
    return out


def _read_record_float(
    normalized: Mapping[str, tuple[object, str]],
    *keys: str,
) -> tuple[float, str] | None:
    """Read a finite numeric value and unit from a normalized instruction record."""
    for key in keys:
        entry = normalized.get(key)
        if entry is None:
            continue
        raw_val, header_unit = entry
        if isinstance(raw_val, bool) or raw_val is None:
            continue
        if isinstance(raw_val, (int, float)):
            val = float(raw_val)
            if math.isfinite(val):
                return val, header_unit
        elif isinstance(raw_val, str):
            parsed = _parse_numeric_cell(raw_val)
            if parsed is not None:
                return parsed[0], (parsed[1] or header_unit)
    return None


def _read_record_str(normalized: Mapping[str, tuple[object, str]], *keys: str) -> str:
    """Read the first non-empty string field from a normalized instruction record."""
    for key in keys:
        entry = normalized.get(key)
        if entry is None:
            continue
        raw_val = entry[0]
        if raw_val is None:
            continue
        text = str(raw_val).strip()
        if text and text.lower() not in _MISSING_TOKENS:
            return text
    return ""


def _extract_issue_us(
    normalized: Mapping[str, tuple[object, str]],
    *,
    tick_to_us: float | None,
) -> float | None:
    """Extract an instruction issue timestamp in microseconds without touching duration fields."""
    us_val = _read_record_float(
        normalized, "issue_us", "start_us", "ts_us", "timestamp_us", "issue", "start"
    )
    if us_val is not None:
        return _to_duration_us(us_val[0], us_val[1] or "us")
    ns_val = _read_record_float(normalized, "issue_ns", "start_ns", "ts_ns", "timestamp_ns")
    if ns_val is not None:
        return _to_duration_us(ns_val[0], "ns")
    ts_val = _read_record_float(normalized, "ts", "timestamp")
    if ts_val is not None:
        if tick_to_us is not None:
            return max(0.0, ts_val[0] * tick_to_us)
        return _to_duration_us(ts_val[0], ts_val[1] or "us")
    cycle_val = _read_record_float(normalized, "issue_cycle", "start_cycle", "cycle", "tick")
    if cycle_val is not None:
        scale = tick_to_us if tick_to_us is not None else _DEFAULT_CYCLE_TO_US
        return max(0.0, cycle_val[0] * scale)
    return None


def _classify_instruction(pipe: str, opcode: str) -> str:
    """Classify an instruction record into a canonical hardware instruction class."""
    pipe_upper = pipe.upper()
    op_lower = opcode.lower()
    if _SYNC_OPCODE_RE.search(op_lower) or pipe_upper in ("SYNC", "EVENT", "BARRIER"):
        return "sync"
    if pipe_upper in ("CUBE", "PIPE_M", "MAC", "AIC_MAC", "MMAD") or op_lower.startswith(
        ("mmad", "mad", "cube")
    ):
        return "cube"
    if pipe_upper in ("FIXPIPE", "PIPE_FIX", "FIX") or "fixpipe" in op_lower:
        return "fixpipe"
    if pipe_upper in ("MTE1", "PIPE_MTE1") or "mte1" in op_lower or "load_cbuf" in op_lower:
        return "mte1"
    if pipe_upper in ("MTE2", "PIPE_MTE2") or "mte2" in op_lower or "gm2" in op_lower:
        return "mte2"
    if pipe_upper in ("MTE3", "PIPE_MTE3") or "mte3" in op_lower or "2gm" in op_lower:
        return "mte3"
    if pipe_upper in ("VEC", "VECTOR", "PIPE_V", "AIV_VEC", "SIMD", "SIMT") or (
        len(op_lower) >= 2 and op_lower[0] == "v" and op_lower[1].isalpha()
    ):
        return "vector"
    if pipe_upper in ("SCALAR", "PIPE_S", "S") or op_lower.startswith(
        ("mov", "add", "sub", "mul", "cmp", "jmp", "bra", "loop", "scalar")
    ):
        return "scalar"
    return "other"


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Compute a linear-interpolated percentile in [0.0, 1.0] over sorted non-empty values."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if count == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(float(count - 1), float(count - 1) * quantile))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - float(lo)
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def parse_p2_instruction_trace(
    trace_source: Path | str | Sequence[Mapping[str, object]],
    *,
    pc_start_addr_source: Path | str | Mapping[int, str] | None = None,
    tick_to_us: float | None = None,
    top_k_hotspots: int = 10,
) -> P2InstructionDiveSummary:
    """Analyze P2 instruction traces for mix, waitflag share, issue intervals, and DWARF lines.

    Same-pipe issue intervals strictly measure consecutive sorted issue timestamps
    ``ts[i + 1] - ts[i]`` and never subtract overlapping instruction durations.
    """
    records, auto_pc_path = _extract_trace_records(trace_source)
    resolved_pc_source = pc_start_addr_source if pc_start_addr_source is not None else auto_pc_path
    dwarf_table = parse_pc_start_addr(resolved_pc_source) if resolved_pc_source is not None else ()
    dwarf_starts = [entry[0] for entry in dwarf_table]

    total = len(records)
    class_counts: dict[str, int] = {}
    pipe_issue_times: dict[str, list[float]] = {}
    pipe_instr_counts: dict[str, int] = {}
    mte_waitflag_count = 0

    hotspot_counts: dict[str, int] = {}
    hotspot_pipes: dict[str, set[str]] = {}
    hotspot_pc_bounds: dict[str, tuple[int, int]] = {}
    attributed_count = 0

    for raw_record in records:
        normalized = _normalize_record_mapping(raw_record)
        pipe = (
            _read_record_str(normalized, "pipe", "pipe_name", "engine", "cat").upper()
            or "UNKNOWN"
        )
        opcode = _read_record_str(normalized, "opcode", "instruction", "instr", "name", "op")
        instr_class = _classify_instruction(pipe, opcode)
        class_counts[instr_class] = class_counts.get(instr_class, 0) + 1
        pipe_instr_counts[pipe] = pipe_instr_counts.get(pipe, 0) + 1

        if _WAITFLAG_RE.search(opcode):
            mte_waitflag_count += 1

        issue_us = _extract_issue_us(normalized, tick_to_us=tick_to_us)
        if issue_us is not None:
            pipe_issue_times.setdefault(pipe, []).append(issue_us)

        if dwarf_table:
            raw_pc_entry = (
                normalized.get("pc") or normalized.get("addr") or normalized.get("pc_addr")
            )
            pc_val = _parse_pc_int(raw_pc_entry[0]) if raw_pc_entry is not None else None
            if pc_val is not None:
                idx = bisect_right(dwarf_starts, pc_val) - 1
                if idx >= 0:
                    start_pc, end_pc, location = dwarf_table[idx]
                    if end_pc is None or pc_val < end_pc:
                        attributed_count += 1
                        hotspot_counts[location] = hotspot_counts.get(location, 0) + 1
                        hotspot_pipes.setdefault(location, set()).add(pipe)
                        if location in hotspot_pc_bounds:
                            lo_pc, hi_pc = hotspot_pc_bounds[location]
                            hotspot_pc_bounds[location] = (min(lo_pc, pc_val), max(hi_pc, pc_val))
                        else:
                            hotspot_pc_bounds[location] = (
                                min(start_pc, pc_val),
                                max(start_pc, pc_val),
                            )

    instruction_mix = (
        {cls_name: count / float(total) for cls_name, count in sorted(class_counts.items())}
        if total > 0
        else {}
    )
    mte_waitflag_share = mte_waitflag_count / float(total) if total > 0 else 0.0

    interval_stats: dict[str, PipeIssueIntervalStats] = {}
    for pipe in sorted(pipe_instr_counts):
        times = sorted(pipe_issue_times.get(pipe, ()))
        intervals = sorted(times[i + 1] - times[i] for i in range(len(times) - 1))
        interval_stats[pipe] = PipeIssueIntervalStats(
            pipe=pipe,
            instruction_count=pipe_instr_counts[pipe],
            interval_count=len(intervals),
            p10_us=_percentile(intervals, 0.10),
            p50_us=_percentile(intervals, 0.50),
            p90_us=_percentile(intervals, 0.90),
        )

    ranked_locations = sorted(
        hotspot_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[: max(0, int(top_k_hotspots))]
    hotspots = tuple(
        DwarfLineHotspot(
            source_location=loc,
            instruction_count=cnt,
            instruction_share=cnt / float(total) if total > 0 else 0.0,
            pipes=tuple(sorted(hotspot_pipes[loc])),
            pc_range_start=hotspot_pc_bounds[loc][0],
            pc_range_end=hotspot_pc_bounds[loc][1],
        )
        for loc, cnt in ranked_locations
    )

    return P2InstructionDiveSummary(
        total_instructions=total,
        instruction_mix=instruction_mix,
        mte_waitflag_count=mte_waitflag_count,
        mte_waitflag_share=mte_waitflag_share,
        issue_interval_us=interval_stats,
        source_line_hotspots=hotspots,
        attributed_instruction_share=attributed_count / float(total) if total > 0 else 0.0,
    )


def find_p2_trace_files(report_dir: Path | str) -> tuple[Path, ...]:
    """Locate instruction-level trace files while excluding summary metric CSV tables."""
    root = Path(report_dir)
    if not root.is_dir():
        return ()
    found: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix not in (".json", ".jsonl", ".csv"):
            continue
        if candidate.name.lower() == "pc_start_addr.txt":
            continue
        stem_compact = re.sub(r"[^a-z0-9]", "", candidate.stem.lower())
        if suffix == ".csv" and any(
            stem_compact == token or stem_compact.endswith(token)
            for token in _P1_SUMMARY_CSV_STEMS
        ):
            continue
        found.append(candidate)
    return tuple(found)


def _infer_objdump_pipe_and_opcode(body: str) -> tuple[str, str]:
    """Extract hardware pipe and instruction mnemonic from an objdump disassembly line."""
    cleaned = body.split("//", 1)[0].split(";", 1)[0].strip()
    if not cleaned:
        return "UNKNOWN", ""
    tokens = re.split(r"[\s,()\[\]]+", cleaned)
    opcode = tokens[0] if tokens else ""
    upper = cleaned.upper()
    for pipe_name in ("PIPE_M", "PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V", "PIPE_S"):
        if pipe_name in upper:
            return pipe_name, opcode
    if _SYNC_OPCODE_RE.search(opcode):
        return "SYNC", opcode
    return "UNKNOWN", opcode


def parse_objdump_dwarf_hotspots(
    objdump_text: str,
    *,
    top_k_hotspots: int = 10,
) -> P2InstructionDiveSummary:
    """Summarize instruction mix, waitflag density, and DWARF hotspots from objdump -d -l."""
    total = 0
    mte_waitflag_count = 0
    class_counts: dict[str, int] = {}
    pipe_instr_counts: dict[str, int] = {}
    hotspot_counts: dict[str, int] = {}
    hotspot_pipes: dict[str, set[str]] = {}
    hotspot_pc_bounds: dict[str, tuple[int, int]] = {}
    attributed_count = 0

    current_location = ""
    current_symbol = ""

    for raw_line in objdump_text.splitlines():
        dwarf_match = _OBJDUMP_DWARF_LINE_RE.match(raw_line)
        if dwarf_match is not None:
            current_location = dwarf_match.group("loc").strip()
            continue
        sym_match = _OBJDUMP_SYMBOL_RE.match(raw_line)
        if sym_match is not None:
            current_symbol = sym_match.group("symbol").strip()
            continue
        instr_match = _OBJDUMP_INSTR_RE.match(raw_line)
        if instr_match is None:
            continue
        pc_val = int(instr_match.group("pc"), 16)
        pipe_hint, opcode = _infer_objdump_pipe_and_opcode(instr_match.group("body"))
        if not opcode:
            continue
        total += 1
        instr_class = _classify_instruction(pipe_hint, opcode)
        pipe_label = pipe_hint if pipe_hint != "UNKNOWN" else instr_class.upper()
        class_counts[instr_class] = class_counts.get(instr_class, 0) + 1
        pipe_instr_counts[pipe_label] = pipe_instr_counts.get(pipe_label, 0) + 1
        if _WAITFLAG_RE.search(opcode) or _WAITFLAG_RE.search(instr_match.group("body")):
            mte_waitflag_count += 1

        location = current_location or current_symbol
        if location:
            attributed_count += 1
            hotspot_counts[location] = hotspot_counts.get(location, 0) + 1
            hotspot_pipes.setdefault(location, set()).add(pipe_label)
            if location in hotspot_pc_bounds:
                lo_pc, hi_pc = hotspot_pc_bounds[location]
                hotspot_pc_bounds[location] = (min(lo_pc, pc_val), max(hi_pc, pc_val))
            else:
                hotspot_pc_bounds[location] = (pc_val, pc_val)

    instruction_mix = (
        {cls_name: count / float(total) for cls_name, count in sorted(class_counts.items())}
        if total > 0
        else {}
    )
    mte_waitflag_share = mte_waitflag_count / float(total) if total > 0 else 0.0
    interval_stats = {
        pipe: PipeIssueIntervalStats(
            pipe=pipe,
            instruction_count=count,
            interval_count=0,
            p10_us=0.0,
            p50_us=0.0,
            p90_us=0.0,
        )
        for pipe, count in sorted(pipe_instr_counts.items())
    }
    ranked_locations = sorted(
        hotspot_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[: max(0, int(top_k_hotspots))]
    hotspots = tuple(
        DwarfLineHotspot(
            source_location=loc,
            instruction_count=cnt,
            instruction_share=cnt / float(total) if total > 0 else 0.0,
            pipes=tuple(sorted(hotspot_pipes[loc])),
            pc_range_start=hotspot_pc_bounds[loc][0],
            pc_range_end=hotspot_pc_bounds[loc][1],
        )
        for loc, cnt in ranked_locations
    )
    return P2InstructionDiveSummary(
        total_instructions=total,
        instruction_mix=instruction_mix,
        mte_waitflag_count=mte_waitflag_count,
        mte_waitflag_share=mte_waitflag_share,
        issue_interval_us=interval_stats,
        source_line_hotspots=hotspots,
        attributed_instruction_share=attributed_count / float(total) if total > 0 else 0.0,
    )


def _resolve_ascend_objdump_bin() -> str:
    """Locate llvm-objdump preferring active Ascend toolchain installations."""
    candidates: list[Path] = []
    bisheng_bin = shutil.which("bisheng")
    if bisheng_bin:
        candidates.append(Path(bisheng_bin).resolve().parent / "llvm-objdump")
        candidates.append(Path(bisheng_bin).parent / "llvm-objdump")
    for env_var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw_root = os.environ.get(env_var, "").strip()
        if not raw_root:
            continue
        root = Path(raw_root)
        for rel in (
            "tools/bisheng_compiler/bin/llvm-objdump",
            "x86_64-linux/bin/llvm-objdump",
            "aarch64-linux/bin/llvm-objdump",
            "compiler/ccec_compiler/bin/llvm-objdump",
        ):
            candidates.append(root / rel)
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return shutil.which("llvm-objdump") or shutil.which("objdump") or ""


def extract_p2_from_objdump(
    object_paths: Sequence[Path | str],
    *,
    objdump_bin: str | Path | None = None,
    top_k_hotspots: int = 10,
) -> P2InstructionDiveSummary:
    """Disassemble compiled Ascend objects with llvm-objdump -d -l for DWARF hotspot fallback."""
    resolved_bin = str(objdump_bin).strip() if objdump_bin is not None else ""
    if not resolved_bin:
        resolved_bin = _resolve_ascend_objdump_bin()
    if not resolved_bin:
        return parse_objdump_dwarf_hotspots("", top_k_hotspots=top_k_hotspots)

    chunks: list[str] = []
    for raw_path in object_paths:
        obj_path = Path(raw_path)
        if not obj_path.is_file():
            continue
        proc = subprocess.run(
            [resolved_bin, "-d", "-l", str(obj_path)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            proc = subprocess.run(
                [resolved_bin, "-d", str(obj_path)],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        if proc.returncode == 0 and proc.stdout.strip():
            chunks.append(proc.stdout)

    return parse_objdump_dwarf_hotspots("\n".join(chunks), top_k_hotspots=top_k_hotspots)


__all__ = [
    "AggregatedMsprofReport",
    "DwarfLineHotspot",
    "P2InstructionDiveSummary",
    "PipeIssueIntervalStats",
    "SubkernelMsprofMetrics",
    "aggregate_subkernel_msprof",
    "extract_p2_from_objdump",
    "find_p2_trace_files",
    "parse_msprof_op_report",
    "parse_objdump_dwarf_hotspots",
    "parse_p2_instruction_trace",
    "parse_pc_start_addr",
]
