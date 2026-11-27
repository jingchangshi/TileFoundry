"""Stratified CANNBench anchor registry, counterexample injection, and scoring."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tilefoundry_ascend.agent.specs.ops_registry import (
    ATTENTION_RESIDUAL_ANCHORS,
    DEFAULT_ANCHORS,
    DEFAULT_PERF_TARGETS,
    DEFAULT_REGIME_SPECS,
    MLA_ANCHORS,
    OPERATOR_METADATA,
    AnchorCaseSpec,
    CannbenchOpSpec,
    CannbenchOpsRegistry,
    CasePhysicsRatios,
    GuardrailCaseSpec,
    PerfTargetSpec,
    PhaseSignature,
    RegimeSpec,
    RegimeTargetSpec,
    SemanticRegimeGroup,
    _analyze_case_geometry,
    _has_flat_operator_layout,
    _is_prime_gt2,
    _normalize_shape_list,
    build_semantic_regime_groups,
    build_semantic_regime_specs,
    campaign_case_regime_tag,
    default_source_root,
    discover_module_trackb_symbols,
    extract_schema_driven_case_dims,
    extract_trackb_cpp_source,
    get_default_anchors,
    get_default_perf_target,
    get_default_regime_specs,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
    infer_module_runtime_backend,
    load_cannbench_contract_module,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
    semantic_regime_representative,
    semantic_regime_tag,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import load_counterexample_boundaries


def load_anchor_harness_and_cases(
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec] | None = None,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    source_root: str | Path | None = None,
) -> tuple[object, list[object]]:
    """Instantiate cannbench_contract.Harness and filter active anchor Case objects."""
    task = get_operator_task(operator_name)
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    tasks_root = resolve_cannbench_tasks_root(repo)
    task_dir = tasks_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"CANNBench task directory missing: {task_dir}")
    contract_mod = load_cannbench_contract_module(source_root=source_root)
    harness = contract_mod.Harness(task, task_dir)
    active = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_num = {int(c.case_num): c for c in harness.cases}
    selected = []
    for spec in active:
        if int(spec.case_num) not in by_num:
            raise KeyError(f"case_num={spec.case_num} missing in {task_dir / 'cases.yaml'}")
        selected.append(by_num[int(spec.case_num)])
    return harness, selected


def lookup_case_metadata(
    operator_name: str,
    case_num: int,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[float, float]:
    """Read (baseline_perf_us, t_hw_us) from <tasks_root>/metadata/950pr.json."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    meta_path = resolve_cannbench_tasks_root(repo) / "metadata" / "950pr.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    task = get_operator_task(operator_name)
    level, op_key = task.split("/", 1)
    entry = payload[level][op_key][str(int(case_num))]
    return float(entry["baseline_perf_us"]), float(entry["t_hw_us"])


def lookup_case_dims(
    operator_name: str,
    case_num: int,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> dict[str, object]:
    """Extract tensor dimensions and dispatch attributes from <tasks_root>/<task>/cases.yaml."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    task = get_operator_task(operator_name)
    cases_path = resolve_cannbench_tasks_root(repo) / task / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.yaml not found at {cases_path}")
    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    cases_list = parsed.get("cases") or []
    target_id = int(case_num)
    entry: Mapping[str, object] | None = None
    for row in cases_list:
        if isinstance(row, Mapping) and int(row.get("case_id", -1)) == target_id:
            entry = row
            break
    if entry is None:
        raise KeyError(f"case_num={target_id} missing in {cases_path}")

    proto_inputs: Sequence[Mapping[str, object]] = ()
    proto_path = cases_path.parent / "proto.yaml"
    if proto_path.is_file():
        proto_raw = yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {}
        op_block = proto_raw.get("operator", proto_raw)
        if isinstance(op_block, Mapping) and isinstance(op_block.get("inputs"), (list, tuple)):
            proto_inputs = op_block["inputs"]
    dims = extract_schema_driven_case_dims(entry, proto_inputs=proto_inputs)
    tasks_root = resolve_cannbench_tasks_root(repo)
    registry = CannbenchOpsRegistry(tasks_root=tasks_root)
    if registry.has_operator(operator_name):
        regime_tag = registry.get_case_regime_tag(operator_name, target_id)
        if regime_tag is not None:
            dims["regime_tag"] = regime_tag
    return dims


def resolve_regime_for_case(
    operator_name: str,
    case_dims: Mapping[str, object],
    regime_specs: Sequence[RegimeTargetSpec],
) -> RegimeTargetSpec:
    """Resolve the active dispatch regime matching structural case dimensions."""
    _ = operator_name
    if not regime_specs:
        raise ValueError("regime_specs must not be empty")
    spec_by_tag = {spec.regime_tag: spec for spec in regime_specs}
    target_tag = str(case_dims.get("regime_tag", "")).strip()
    if target_tag and target_tag in spec_by_tag:
        return spec_by_tag[target_tag]

    layout = str(
        case_dims.get("inputLayout", case_dims.get("input_layout", "BSND"))
    ).upper()
    d_val = case_dims.get("D", case_dims.get("d_dim", 0))
    d_dim = int(d_val) if isinstance(d_val, int) and not isinstance(d_val, bool) else 0
    b_dim = int(case_dims.get("B", case_dims.get("b_dim", 1)))
    s_dim = int(case_dims.get("S", case_dims.get("s_dim", case_dims.get("S_q", 1))))

    if "bnsd_gather" in spec_by_tag or "split_kv_decode" in spec_by_tag:
        if layout == "BNSD" and "bnsd_gather" in spec_by_tag:
            return spec_by_tag["bnsd_gather"]
        if s_dim <= 2 and "split_kv_decode" in spec_by_tag:
            return spec_by_tag["split_kv_decode"]
        if "prefill_compute" in spec_by_tag:
            return spec_by_tag["prefill_compute"]

    if "unaligned_d" in spec_by_tag or "small_decode" in spec_by_tag:
        if d_dim > 0 and d_dim % 64 != 0 and "unaligned_d" in spec_by_tag:
            return spec_by_tag["unaligned_d"]
        if b_dim * s_dim < 16 and "small_decode" in spec_by_tag:
            return spec_by_tag["small_decode"]
        if "aligned_prefill" in spec_by_tag:
            return spec_by_tag["aligned_prefill"]

    if d_dim > 1 and d_dim % 16 != 0 and "unaligned_tail" in spec_by_tag:
        return spec_by_tag["unaligned_tail"]
    total_elems = case_dims.get("total_elements")
    if isinstance(total_elems, int):
        if int(total_elems) <= 65536 and "small_latency" in spec_by_tag:
            return spec_by_tag["small_latency"]
        if int(total_elems) >= 1048576 and "large_throughput" in spec_by_tag:
            return spec_by_tag["large_throughput"]

    return regime_specs[0]


def append_dynamic_anchor(
    anchors: Sequence[AnchorCaseSpec],
    *,
    operator_name: str,
    case_num: int,
    regime_tag: str = "tail_regression",
    ref_perf_us: float | None = None,
    t_hw_us: float | None = None,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    """Append a non-anchor regression case idempotently to the active anchor list."""
    cid = int(case_num)
    if any(int(a.case_num) == cid for a in anchors):
        return tuple(anchors)
    if ref_perf_us is None or t_hw_us is None:
        meta_ref, meta_hw = lookup_case_metadata(
            operator_name,
            cid,
            cannbench_repo=cannbench_repo,
            env_file=env_file,
        )
        resolved_ref = meta_ref if ref_perf_us is None else float(ref_perf_us)
        resolved_hw = meta_hw if t_hw_us is None else float(t_hw_us)
    else:
        resolved_ref = float(ref_perf_us)
        resolved_hw = float(t_hw_us)
    new_spec = AnchorCaseSpec(
        task=get_operator_task(operator_name),
        case_num=cid,
        regime_tag=regime_tag,
        ref_perf_us=resolved_ref,
        t_hw_us=resolved_hw,
    )
    return (*anchors, new_spec)


_NON_TAIL_DIM_KEYS = frozenset(
    {
        "total_elements",
        "rank",
        "num_inputs",
        "dtype_bytes",
        "case_id",
        "case_num",
        "group_count",
        "b",
        "b_dim",
        "batch",
        "n_q",
        "n_kv",
        "numheads",
        "numkvheads",
    }
)
_TOPOLOGY_AXIS_KEYS = ("compute_engine", "memory_ring", "iteration_regularity")


def _optional_positive_float(raw: object) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if val > 0.0 else None
    if isinstance(raw, str):
        try:
            val = float(raw.strip())
            return val if val > 0.0 else None
        except ValueError:
            return None
    return None


def _coerce_case_int(raw: object) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None


def _resolve_boundary_regime_tag(
    record: Mapping[str, object],
    *,
    operator_name: str,
    regime_specs: Sequence[RegimeTargetSpec],
) -> str:
    raw_tag = str(record.get("regime_tag", "")).strip()
    if raw_tag:
        return raw_tag
    raw_dims = record.get("case_dims")
    if isinstance(raw_dims, Mapping) and raw_dims and regime_specs:
        return resolve_regime_for_case(operator_name, raw_dims, regime_specs).regime_tag
    if regime_specs:
        return regime_specs[0].regime_tag
    return "tail_regression"


def _append_anchor_with_fallback(
    anchors: tuple[AnchorCaseSpec, ...],
    *,
    operator_name: str,
    case_num: int,
    regime_tag: str,
    ref_perf_us: float | None = None,
    t_hw_us: float | None = None,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    cid = int(case_num)
    if any(int(a.case_num) == cid for a in anchors):
        return anchors
    try:
        return append_dynamic_anchor(
            anchors,
            operator_name=operator_name,
            case_num=cid,
            regime_tag=regime_tag,
            ref_perf_us=ref_perf_us,
            t_hw_us=t_hw_us,
            cannbench_repo=cannbench_repo,
            env_file=env_file,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        resolved_ref = 0.0 if ref_perf_us is None else float(ref_perf_us)
        resolved_hw = 0.0 if t_hw_us is None else float(t_hw_us)
        try:
            return append_dynamic_anchor(
                anchors,
                operator_name=operator_name,
                case_num=cid,
                regime_tag=regime_tag,
                ref_perf_us=resolved_ref,
                t_hw_us=resolved_hw,
                cannbench_repo=cannbench_repo,
                env_file=env_file,
            )
        except (FileNotFoundError, KeyError, OSError, ValueError):
            fallback_task = anchors[0].task if anchors else operator_name
            return (
                *anchors,
                AnchorCaseSpec(
                    task=fallback_task,
                    case_num=cid,
                    regime_tag=regime_tag,
                    ref_perf_us=resolved_ref,
                    t_hw_us=resolved_hw,
                ),
            )


def _matches_topology_axes(
    record: Mapping[str, object],
    target_axes: Mapping[str, str],
) -> bool:
    if not target_axes:
        return True
    rec_axes = record.get("topology_axes")
    if not isinstance(rec_axes, Mapping) or not rec_axes:
        return True
    for key, expected in target_axes.items():
        raw_val = rec_axes.get(key)
        if raw_val is None:
            continue
        val = str(raw_val).strip().lower()
        if val and val != expected:
            return False
    return True


def _record_has_unaligned_dim(record: Mapping[str, object]) -> bool:
    if (
        bool(record.get("has_unaligned_dim"))
        or bool(record.get("odd_or_prime_tail"))
        or bool(record.get("is_unaligned"))
    ):
        return True
    raw_dims = record.get("case_dims")
    if not isinstance(raw_dims, Mapping):
        return False
    raw_shapes = raw_dims.get("input_shape") or raw_dims.get("shapes")
    if raw_shapes is not None:
        for sh in _normalize_shape_list(raw_shapes):
            if sh and sh[-1] > 1:
                tail = sh[-1]
                if tail % 2 == 1 or _is_prime_gt2(tail) or tail % 32 != 0:
                    return True
    for key, val in raw_dims.items():
        if str(key).strip().lower() in _NON_TAIL_DIM_KEYS:
            continue
        if isinstance(val, int) and not isinstance(val, bool) and val > 1:
            if val % 2 == 1 or _is_prime_gt2(val) or val % 32 != 0:
                return True
    return False


def _find_unaligned_case_in_cannbench(
    operator_name: str,
    *,
    existing_ids: set[int],
    cannbench_repo: str | Path | None,
    env_file: str | Path | None,
) -> tuple[int, dict[str, object], float, float] | None:
    try:
        repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
        task = get_operator_task(operator_name)
        tasks_root = resolve_cannbench_tasks_root(repo)
        cases_path = tasks_root / task / "cases.yaml"
        if not cases_path.is_file():
            return None
        parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
        cases_list = parsed.get("cases") or []
    except (FileNotFoundError, KeyError, OSError, ValueError, yaml.YAMLError):
        return None

    meta_map: dict[int, tuple[float, float]] = {}
    meta_path = tasks_root / "metadata" / "950pr.json"
    if meta_path.is_file():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            level, op_key = task.split("/", 1)
            op_meta = (meta_payload.get(level) or {}).get(op_key) or {}
            for k_str, entry in op_meta.items():
                if isinstance(entry, Mapping):
                    meta_map[int(k_str)] = (
                        float(entry.get("baseline_perf_us", 0.0) or 0.0),
                        float(entry.get("t_hw_us", 0.0) or 0.0),
                    )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    candidates: list[tuple[int, int, int, Mapping[str, object], float, float]] = []
    for raw in cases_list:
        if not isinstance(raw, Mapping):
            continue
        cid = _coerce_case_int(raw.get("case_id"))
        if cid is None or cid < 0 or cid in existing_ids:
            continue
        ref_us, hw_us = meta_map.get(cid, (0.0, 0.0))
        geom = _analyze_case_geometry(raw, ref_us, hw_us)
        if not bool(geom.get("is_unaligned")):
            continue
        score = _coerce_case_int(geom.get("unaligned_score")) or 0
        elems = _coerce_case_int(geom.get("total_elems")) or 1
        candidates.append((-score, elems, cid, raw, ref_us, hw_us))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _neg_score, _elems, best_cid, best_raw, best_ref, best_hw = candidates[0]
    return best_cid, extract_schema_driven_case_dims(best_raw), best_ref, best_hw


def inject_lake_counterexample_anchors(
    initial_anchors: Sequence[AnchorCaseSpec],
    *,
    operator_name: str,
    regime_specs: Sequence[RegimeTargetSpec],
    experience_lake_dir: str | Path | None,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    max_injected: int | None = None,
    topology_axes: Mapping[str, object] | None = None,
) -> list[AnchorCaseSpec]:
    """Inject high-frequency historical L3/L4 counterexamples or tail seeds into anchors."""
    if experience_lake_dir is None or (max_injected is not None and max_injected <= 0):
        return list(initial_anchors)
    lake_str = str(experience_lake_dir).strip()
    if not lake_str:
        return list(initial_anchors)

    lake_path = Path(lake_str)
    anchors_tuple = tuple(initial_anchors)
    existing_ids = {int(a.case_num) for a in anchors_tuple}

    same_op_records = load_counterexample_boundaries(lake_path, operator=operator_name)
    if same_op_records:
        freq: Counter[int] = Counter()
        rep_by_case: dict[int, Mapping[str, object]] = {}
        for rec in same_op_records:
            cid = _coerce_case_int(rec.get("case_num"))
            if cid is None or cid in existing_ids:
                continue
            freq[cid] += 1
            prev_rep = rep_by_case.get(cid)
            if prev_rep is None or (
                not str(prev_rep.get("regime_tag", "")).strip()
                and str(rec.get("regime_tag", "")).strip()
            ):
                rep_by_case[cid] = rec
        ordered_cases = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
        selected_cases = ordered_cases if max_injected is None else ordered_cases[:max_injected]
        for cid, _count in selected_cases:
            rep = rep_by_case[cid]
            resolved_tag = _resolve_boundary_regime_tag(
                rep,
                operator_name=operator_name,
                regime_specs=regime_specs,
            )
            anchors_tuple = _append_anchor_with_fallback(
                anchors_tuple,
                operator_name=operator_name,
                case_num=cid,
                regime_tag=resolved_tag,
                ref_perf_us=_optional_positive_float(rep.get("ref_perf_us")),
                t_hw_us=_optional_positive_float(rep.get("t_hw_us")),
                cannbench_repo=cannbench_repo,
                env_file=env_file,
            )
        return list(anchors_tuple)

    all_records = load_counterexample_boundaries(lake_path)
    if not all_records:
        return list(anchors_tuple)

    target_axes = (
        {
            key: str(topology_axes.get(key, "")).strip().lower()
            for key in _TOPOLOGY_AXIS_KEYS
            if str(topology_axes.get(key, "")).strip()
        }
        if isinstance(topology_axes, Mapping)
        else {}
    )
    filtered = [rec for rec in all_records if _matches_topology_axes(rec, target_axes)]
    if not any(_record_has_unaligned_dim(rec) for rec in filtered):
        return list(anchors_tuple)

    found = _find_unaligned_case_in_cannbench(
        operator_name,
        existing_ids=existing_ids,
        cannbench_repo=cannbench_repo,
        env_file=env_file,
    )
    if found is None:
        return list(anchors_tuple)

    tail_cid, tail_dims, tail_ref, tail_hw = found
    spec_tags = {spec.regime_tag for spec in regime_specs}
    if "unaligned_tail" in spec_tags:
        chosen_tag = "unaligned_tail"
    elif "unaligned_d" in spec_tags:
        chosen_tag = "unaligned_d"
    elif regime_specs:
        chosen_tag = resolve_regime_for_case(operator_name, tail_dims, regime_specs).regime_tag
    else:
        chosen_tag = "unaligned_tail"

    anchors_tuple = _append_anchor_with_fallback(
        anchors_tuple,
        operator_name=operator_name,
        case_num=tail_cid,
        regime_tag=chosen_tag,
        ref_perf_us=(tail_ref if tail_ref > 0.0 else None),
        t_hw_us=(tail_hw if tail_hw > 0.0 else None),
        cannbench_repo=cannbench_repo,
        env_file=env_file,
    )
    return list(anchors_tuple)


def compute_case_eq3_score(ref_perf_us: float, cand_us: float, t_hw_us: float) -> float:
    """Compute CANNBench Eq.3 per-case performance score in [0, +inf)."""
    if cand_us <= 0.0 or t_hw_us <= 0.0:
        return 0.0
    baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(t_hw_us * 10.0, 10.0)
    denom = (cand_us - t_hw_us) + (baseline - t_hw_us)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (baseline - t_hw_us) / denom)


def compute_case_aligned_eta(anchor: AnchorCaseSpec, measured_dur_us: float) -> float:
    """Compute clamped hardware efficiency t_hw_us / measured_dur_us for one anchor case."""
    dur = float(measured_dur_us)
    t_hw = float(anchor.t_hw_us)
    if not (math.isfinite(dur) and math.isfinite(t_hw)) or dur <= 0.0 or t_hw <= 0.0:
        return 0.0
    return min(1.0, max(0.0, t_hw / dur))


_STATIC_WIDTH_KEY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("D", ("D", "d_dim")),
    ("N", ("N", "n_depth")),
    ("K", ("K", "k_dim")),
    ("d_nope", ("d_nope", "D_nope")),
    ("d_rope", ("d_rope", "D_rope")),
    ("dk", ("dk", "D_qk")),
    ("dv", ("dv", "D_v")),
    ("n_q", ("n_q", "N_q")),
    ("n_kv", ("n_kv", "N_kv")),
    ("half_dim", ("half_dim",)),
    ("hc_mult", ("hc_mult",)),
    ("inner", ("inner", "inner_dim")),
    ("hidden_dim", ("hidden_dim",)),
    ("w_dim", ("w_dim",)),
    ("num_regs", ("num_regs",)),
)


_STATIC_WIDTH_B1_DTYPES: frozenset[str] = frozenset(
    {"int8", "uint8", "float8_e4m3fn", "float8_e5m2", "fp8"}
)
_STATIC_WIDTH_B2_DTYPES: frozenset[str] = frozenset(
    {"float16", "bfloat16", "fp16", "bf16", "int16", "uint16"}
)
_STATIC_WIDTH_B4_DTYPES: frozenset[str] = frozenset(
    {"float32", "fp32", "int32", "uint32"}
)
_STATIC_WIDTH_B8_DTYPES: frozenset[str] = frozenset(
    {"float64", "fp64", "int64", "uint64"}
)


def _static_width_dtype_class(dtype_str: str) -> str:
    """Map numeric dtype strings to bytewidth classes for static width deduplication."""
    norm = dtype_str.strip().lower().removeprefix("torch.")
    if norm in _STATIC_WIDTH_B2_DTYPES:
        return "b2"
    if norm in _STATIC_WIDTH_B4_DTYPES:
        return "b4"
    if norm in _STATIC_WIDTH_B1_DTYPES:
        return "b1"
    if norm in _STATIC_WIDTH_B8_DTYPES:
        return "b8"
    return norm


def _static_width_signature(
    dims: Mapping[str, object],
    *,
    default_dtype: str = "",
    default_layout: str = "",
) -> tuple[tuple[str, object], ...]:
    sig: list[tuple[str, object]] = []
    for canon, aliases in _STATIC_WIDTH_KEY_GROUPS:
        for key in aliases:
            val = dims.get(key)
            if isinstance(val, int) and not isinstance(val, bool):
                sig.append((canon, int(val)))
                break
    dtype_val = str(dims.get("tl_dtype") or default_dtype).strip().lower()
    if dtype_val:
        sig.append(("tl_dtype", _static_width_dtype_class(dtype_val)))
    layout_val = str(
        dims.get("inputLayout", dims.get("input_layout", default_layout))
    ).strip().upper()
    if layout_val:
        sig.append(("input_layout", layout_val))
    if isinstance(dims.get("is_causal"), bool):
        sig.append(("is_causal", bool(dims["is_causal"])))
    return tuple(sig)


def collect_distinct_static_width_dims(
    operator_name: str,
    regime_spec: RegimeTargetSpec,
    anchors: Sequence[AnchorCaseSpec],
    guardrails: Sequence[GuardrailCaseSpec] = (),
    *,
    cannbench_repo: Path | None = None,
    env_file: Path | None = None,
) -> list[dict[str, object]]:
    """Collect deduplicated static-width case dims routed to regime_spec across cases and probes."""
    default_dtype = str(regime_spec.sample_builder_kwargs.get("tl_dtype", "") or "").strip()
    default_layout = str(
        regime_spec.sample_builder_kwargs.get(
            "input_layout",
            regime_spec.sample_builder_kwargs.get("inputLayout", ""),
        )
        or ""
    ).strip()

    explicit_regime_by_case: dict[int, str] = {}
    for guard in guardrails:
        explicit_regime_by_case[int(guard.case_num)] = str(guard.regime_tag)
    for anchor in anchors:
        explicit_regime_by_case[int(anchor.case_num)] = str(anchor.regime_tag)

    cases_list: list[Mapping[str, object]] = []
    proto_inputs: Sequence[Mapping[str, object]] = ()
    registry: CannbenchOpsRegistry | None = None
    all_regime_specs: tuple[RegimeTargetSpec, ...] = ()

    try:
        repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
        tasks_root = resolve_cannbench_tasks_root(repo)
        try:
            registry = CannbenchOpsRegistry(tasks_root=tasks_root)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            registry = None
        if registry is not None and registry.has_operator(operator_name):
            op_spec = registry.resolve(operator_name)
            cases_list = list(op_spec.cases)
            raw_inputs = op_spec.proto.get("inputs")
            if isinstance(raw_inputs, (list, tuple)):
                proto_inputs = raw_inputs
            try:
                all_regime_specs = registry.get_regime_specs(operator_name)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                all_regime_specs = ()
        else:
            task = get_operator_task(operator_name)
            cases_path = tasks_root / task / "cases.yaml"
            if cases_path.is_file():
                parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
                cases_list = [
                    row for row in (parsed.get("cases") or ()) if isinstance(row, Mapping)
                ]
                proto_path = cases_path.parent / "proto.yaml"
                if proto_path.is_file():
                    proto_raw = yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {}
                    op_block = proto_raw.get("operator", proto_raw)
                    if isinstance(op_block, Mapping) and isinstance(
                        op_block.get("inputs"), (list, tuple)
                    ):
                        proto_inputs = op_block["inputs"]
    except (FileNotFoundError, KeyError, OSError, ValueError, yaml.YAMLError):
        pass

    if not all_regime_specs:
        base_specs = DEFAULT_REGIME_SPECS.get(str(operator_name).strip(), ())
        if any(spec.regime_tag == regime_spec.regime_tag for spec in base_specs):
            all_regime_specs = base_specs
        else:
            all_regime_specs = (*base_specs, regime_spec)

    collected: list[dict[str, object]] = []
    seen_sigs: set[tuple[tuple[str, object], ...]] = set()

    for row in cases_list:
        cid = _coerce_case_int(row.get("case_id"))
        dims = extract_schema_driven_case_dims(row, proto_inputs=proto_inputs)
        routed_tag: str | None = None
        if cid is not None and cid in explicit_regime_by_case:
            routed_tag = explicit_regime_by_case[cid]
        elif registry is not None and cid is not None and registry.has_operator(operator_name):
            try:
                routed_tag = registry.get_case_regime_tag(operator_name, cid)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                routed_tag = None
        if routed_tag is None:
            routed_tag = resolve_regime_for_case(
                operator_name, dims, all_regime_specs
            ).regime_tag
        if routed_tag != regime_spec.regime_tag:
            continue
        sig = _static_width_signature(
            dims,
            default_dtype=default_dtype,
            default_layout=default_layout,
        )
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            collected.append(dict(dims))

    rep_raw = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
    if rep_raw:
        rep_dims: dict[str, object] = dict(rep_raw)
        if default_dtype and "tl_dtype" not in rep_dims:
            rep_dims["tl_dtype"] = default_dtype
        if default_layout and "inputLayout" not in rep_dims and "input_layout" not in rep_dims:
            rep_dims["inputLayout"] = default_layout
        rep_sig = _static_width_signature(
            rep_dims,
            default_dtype=default_dtype,
            default_layout=default_layout,
        )
        if rep_sig not in seen_sigs:
            seen_sigs.add(rep_sig)
            collected.append(rep_dims)

    return collected


_ANCHOR_DIM_ALIASES: dict[str, str] = {
    "N": "n_depth",
    "B": "b_dim",
    "S": "s_dim",
    "S_q": "s_dim",
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N_kv": "n_kv",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
}


def build_anchor_var_bindings(
    dims: Mapping[str, object] | None,
    *,
    kwargs: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Build rule-blind symbolic integer bindings from anchor dims and builder kwargs."""
    out: dict[str, int] = {}
    if kwargs:
        for key, raw_val in kwargs.items():
            if isinstance(raw_val, int) and not isinstance(raw_val, bool):
                out[str(key)] = int(raw_val)

    explicit_dim_keys: set[str] = set()
    if dims:
        for key, raw_val in dims.items():
            if isinstance(raw_val, int) and not isinstance(raw_val, bool):
                key_str = str(key)
                iv = int(raw_val)
                out[key_str] = iv
                explicit_dim_keys.add(key_str)
                alias = _ANCHOR_DIM_ALIASES.get(key_str)
                if alias:
                    out[alias] = iv
                    explicit_dim_keys.add(alias)

    b_val = out.get("B", out.get("b_dim"))
    s_val = out.get("S", out.get("S_q", out.get("s_dim")))
    nq_val = out.get("N_q", out.get("n_q"))
    nkv_val = out.get("N_kv", out.get("n_kv"))
    n_val = out.get("N", out.get("n_depth"))

    if b_val is not None and s_val is not None and "m_rows" not in explicit_dim_keys:
        out["m_rows"] = int(b_val) * int(s_val)
    if (
        b_val is not None
        and s_val is not None
        and nq_val is not None
        and "q_total_rows" not in explicit_dim_keys
    ):
        out["q_total_rows"] = int(b_val) * int(s_val) * int(nq_val)
    if (
        s_val is not None
        and nq_val is not None
        and nkv_val is not None
        and int(nkv_val) > 0
        and "q_rows" not in explicit_dim_keys
    ):
        out["q_rows"] = int(s_val) * (int(nq_val) // int(nkv_val))
    if (
        n_val is not None
        and b_val is not None
        and s_val is not None
        and "v_rows" not in explicit_dim_keys
    ):
        out["v_rows"] = int(n_val) * int(b_val) * int(s_val)
    return out


@dataclass(frozen=True)
class InputDomainEnvelope:
    """Operator input domain contract, safe fast-math notes, and constexpr hints."""

    operator: str
    tensor_distributions: tuple[str, ...]
    algebraic_invariants: tuple[str, ...]
    safe_fast_math_notes: tuple[str, ...]
    constexpr_specialization_hints: tuple[str, ...]

    def format_prompt_block(self) -> str:
        """Render domain assumptions and safe fast-math contract for prompt injection."""
        lines = [f"### INPUT DOMAIN & FAST-MATH ENVELOPE ({self.operator})"]
        if self.tensor_distributions:
            lines.append("Tensor Distributions:")
            lines.extend(f"- {item}" for item in self.tensor_distributions)
        if self.algebraic_invariants:
            lines.append("Algebraic Invariants:")
            lines.extend(f"- {item}" for item in self.algebraic_invariants)
        if self.safe_fast_math_notes:
            lines.append("Safe Fast-Math Transformations:")
            lines.extend(f"- {item}" for item in self.safe_fast_math_notes)
        if self.constexpr_specialization_hints:
            lines.append("Constexpr Builder Specialization:")
            lines.extend(f"- {item}" for item in self.constexpr_specialization_hints)
        return "\n".join(lines)


INPUT_DOMAIN_ENVELOPES: dict[str, InputDomainEnvelope] = {
    "attention_residual": InputDomainEnvelope(
        operator="attention_residual",
        tensor_distributions=(
            "v_stack and x are fp16/bf16 bounded normal tensors (~randn).",
            "w is typically zero-initialized or small bounded normal, and gamma is near unit scale.",
        ),
        algebraic_invariants=(
            "RMSNorm(V) normalizes across D with epsilon=1e-6 before depth scoring.",
            "Normalized dot-product logits in fp32 stay tightly bounded around zero.",
        ),
        safe_fast_math_notes=(
            (
                "Unshifted exp(score) weights and single-pass vfma accumulation without online "
                "max-shift rescaling stay finite and within rtol=1e-2, atol=1e-2 on normalized inputs."
            ),
            "Keep reductions and exponential weights in fp32 before casting final outputs.",
        ),
        constexpr_specialization_hints=(
            (
                "Return optional 'builder_kwargs': {...} from _select_*_config() to pass static "
                "compile-time switches (for example unroll, fission, or fast-exp mode) to builders."
            ),
        ),
    ),
    "mla": InputDomainEnvelope(
        operator="mla",
        tensor_distributions=(
            "q_nope, q_rope, k_nope, k_rope, and v are bounded fp16/bf16 normal tensors (~randn).",
            "scale = 1 / sqrt(d_nope + d_rope) bounds QK^T magnitudes across heads.",
        ),
        algebraic_invariants=(
            "Causal masking zeroes upper-triangular future KV positions when is_causal=True.",
            "Group query ratio g_ratio = n_q // n_kv shares K/V heads per head group.",
        ),
        safe_fast_math_notes=(
            "Retain fp32 online max-shift softmax stabilization across long KV sequence blocks.",
            "Scale folding and exp2(x * log2(e)) approximations are legal within rtol=1e-2, atol=1e-2.",
        ),
        constexpr_specialization_hints=(
            (
                "Return optional 'builder_kwargs': {...} from _select_*_config() to specialize "
                "static head counts, stage splits, or fast-math flags as compile-time constants."
            ),
        ),
    ),
}


def get_input_domain_envelope(operator: str) -> InputDomainEnvelope | None:
    """Return InputDomainEnvelope for *operator*, or None if unregistered."""
    return INPUT_DOMAIN_ENVELOPES.get(str(operator).strip().lower())


__all__ = [
    "ATTENTION_RESIDUAL_ANCHORS",
    "AnchorCaseSpec",
    "CannbenchOpSpec",
    "CannbenchOpsRegistry",
    "DEFAULT_ANCHORS",
    "DEFAULT_PERF_TARGETS",
    "DEFAULT_REGIME_SPECS",
    "GuardrailCaseSpec",
    "INPUT_DOMAIN_ENVELOPES",
    "InputDomainEnvelope",
    "MLA_ANCHORS",
    "OPERATOR_METADATA",
    "PerfTargetSpec",
    "RegimeSpec",
    "RegimeTargetSpec",
    "SemanticRegimeGroup",
    "append_dynamic_anchor",
    "build_anchor_var_bindings",
    "build_semantic_regime_groups",
    "build_semantic_regime_specs",
    "campaign_case_regime_tag",
    "collect_distinct_static_width_dims",
    "compute_case_aligned_eta",
    "compute_case_eq3_score",
    "default_source_root",
    "discover_module_trackb_symbols",
    "extract_schema_driven_case_dims",
    "get_default_anchors",
    "get_default_perf_target",
    "get_default_regime_specs",
    "get_input_domain_envelope",
    "get_operator_entrypoint",
    "get_operator_module_relpath",
    "get_operator_task",
    "infer_module_runtime_backend",
    "inject_lake_counterexample_anchors",
    "load_anchor_harness_and_cases",
    "load_cannbench_contract_module",
    "lookup_case_dims",
    "lookup_case_metadata",
    "parse_simple_env_file",
    "resolve_cannbench_repo",
    "resolve_cannbench_tasks_root",
    "resolve_regime_for_case",
    "semantic_regime_representative",
    "semantic_regime_tag",
]
