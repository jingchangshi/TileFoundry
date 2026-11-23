"""Operator advisory extractor and default SIL policy synthesizer for cann-bench."""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import yaml

from tilefoundry_ascend.ascend.sil.schema import (
    AdvisoryEnvelope,
    CasePhysicsKind,
    CasePhysicsRatios,
    HardwareLegalityBounds,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    TopologyAxes,
    build_case_physics_context,
    case_primary_dtype_bytes,
    case_workload_bytes,
    compute_case_physics_ratios,
    derive_case_physics_input,
    flatten_int_shapes,
    flatten_str_values,
    has_empty_group_attrs,
    has_unaligned_tail_for_case,
    infer_case_dims_structural,
    infer_case_physics_kind,
    partition_case_regimes,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


def _resolve_ops_root_path(path: str | Path) -> Path:
    cand = Path(path).expanduser()
    sub = cand / "tasks"
    if sub.is_dir():
        return sub
    return cand


def _default_ops_root() -> Path:
    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    if env_repo:
        return _resolve_ops_root_path(env_repo)
    ws_parent = Path(__file__).resolve().parents[5].parent
    return ws_parent / "cann-bench" / "tasks"


DEFAULT_OPS_ROOT = _default_ops_root()

_CANNBENCH_DEFAULT_THRESHOLDS: dict[str, float] = {
    "float32": 2.0**-13,
    "float16": 2.0**-10,
    "bfloat16": 2.0**-7,
    "float8_e4m3fn": 2.0**-3,
    "float8_e5m2": 2.0**-2,
}


def _is_prime_or_odd(n: int) -> bool:
    return n > 1 and (n % 2 == 1)


def _case_physics_kind(
    proto: Mapping[str, Any],
    axes: TopologyAxes,
) -> CasePhysicsKind:
    category = str(proto.get("category", "")).strip().lower()
    return infer_case_physics_kind(
        axes,
        has_reduction=category in ("normalization", "reduction"),
    )


def _extract_attr_names(raw_attrs: object) -> set[str]:
    names: set[str] = set()
    if isinstance(raw_attrs, Mapping):
        names.update(str(k) for k in raw_attrs.keys())
    elif isinstance(raw_attrs, (list, tuple)):
        for item in raw_attrs:
            if isinstance(item, Mapping) and item.get("name"):
                names.add(str(item["name"]))
    return names


def infer_contraction_and_axes_from_proto(
    proto: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]] = (),
    *,
    spec_map: Mapping[str, Any] | None = None,
    sf_map: Mapping[str, Any] | None = None,
    features: object | None = None,
    regimes: Any = (),
    dtypes: Iterable[str] = (),
    any_empty_group_hint: bool = False,
) -> tuple[TopologyAxes, dict[str, bool]]:
    """Derive 3-axis TopologyAxes and structural flags from tensor/proto/feature contracts."""
    proto_map = proto if isinstance(proto, Mapping) else {}
    spec_dict = spec_map if isinstance(spec_map, Mapping) else {}
    sf_dict = sf_map if isinstance(sf_map, Mapping) else {}
    feat = _extract_feature_flags(features)

    dtype_strs = [str(dt).strip().lower() for dt in dtypes if str(dt).strip()]
    inputs = list(proto_map.get("inputs") or ())
    outputs = list(proto_map.get("outputs") or ())
    for item in (*inputs, *outputs):
        if isinstance(item, Mapping):
            for dt in flatten_str_values(item.get("dtype")):
                val = dt.strip().lower()
                if val and val not in dtype_strs:
                    dtype_strs.append(val)
    for case in cases:
        if isinstance(case, Mapping):
            for dt in flatten_str_values(case.get("dtype")):
                val = dt.strip().lower()
                if val and val not in dtype_strs:
                    dtype_strs.append(val)

    attrs_keys = _extract_attr_names(proto_map.get("attrs"))
    attrs_keys.update(str(k) for k in (sf_dict.get("attrs_keys") or ()))
    any_empty_group = (
        bool(any_empty_group_hint)
        or bool(sf_dict.get("has_empty_group", False))
        or bool(spec_dict.get("has_empty_group", False))
    )
    for case in cases:
        if isinstance(case, Mapping):
            c_attrs = case.get("attrs") or {}
            attrs_keys.update(_extract_attr_names(c_attrs))
            if has_empty_group_attrs(c_attrs):
                any_empty_group = True
    if not any_empty_group and isinstance(regimes, (list, tuple)):
        for reg in regimes:
            if isinstance(reg, Mapping) and reg.get("has_empty_group"):
                any_empty_group = True
                break
            if getattr(reg, "has_empty_group", False):
                any_empty_group = True
                break

    cat = str(sf_dict.get("category") or proto_map.get("category") or "").strip().lower()
    formula = str(proto_map.get("formula", "")).lower()
    desc = str(proto_map.get("description", "")).lower()
    shape_sup = str(proto_map.get("shape_support", "")).lower()
    contract_text = "\n".join(
        str(src.get(k, "")).lower()
        for src in (proto_map, spec_dict)
        for k in ("formula", "description", "semantic_contract", "shape_support")
    )

    is_opcode_stream = (
        str(spec_dict.get("iteration_regularity", "")).strip().lower() == "dynamic_opcode_stream"
        or bool(spec_dict.get("dynamic_bytecode", False))
        or bool(spec_dict.get("dynamic_opcode_stream", False))
        or "program_pattern" in attrs_keys
        or "bytecode" in contract_text
        or "opcode_stream" in contract_text
        or ("program[" in formula and "op =" in formula)
    )

    is_recurrent = (
        bool(sf_dict.get("is_recurrent", spec_dict.get("is_recurrent", False)))
        or feat.get("recurrence") == "non_associative_scan"
        or "iter_step" in attrs_keys
        or "chunk_size" in attrs_keys
        or "hiddensize" in {k.lower() for k in attrs_keys}
        or ("conv_mode" in attrs_keys and "inplace" in attrs_keys)
        or "h_{t-1}" in formula
        or "hprev" in formula
        or "iter 2" in formula
    )

    is_grouped = (
        bool(sf_dict.get("is_ragged", spec_dict.get("is_ragged", False)))
        or "group_list" in attrs_keys
    )
    has_attention_attrs = bool(
        {
            "is_causal",
            "scaleValue",
            "scale_value",
            "sparse_mode",
            "inputLayout",
            "input_layout",
            "n_heads",
            "query_start",
        }
        & attrs_keys
    )

    transposed_weight_pair = (
        "\u1d40" in formula
        or ("chunk_size" in attrs_keys and "[b, h, dh, d]" in shape_sup and "[b, h, d, dh]" in shape_sup)
    )
    depthwise_flag = (
        "groups = c" in desc
        or "depthwise" in contract_text
        or ("groups" in attrs_keys and "weight[c," in formula)
    )
    if "uses_cube" in sf_dict:
        uses_cube = bool(sf_dict["uses_cube"])
    elif "uses_cube" in spec_dict:
        uses_cube = bool(spec_dict["uses_cube"])
    elif feat and float(feat.get("cube_ratio", 0.0)) > 0.0:
        uses_cube = True
    else:
        uses_cube = bool(
            "@" in formula
            or "matmul" in contract_text
            or "mmad" in contract_text
            or "gemm" in contract_text
            or "conv3d_filter_grad" in contract_text
            or transposed_weight_pair
            or (
                cat == "contraction"
                and not depthwise_flag
                and "max_{" not in formula
            )
            or (formula.strip().startswith("y = conv(") and "groups" not in attrs_keys)
            or "computes original query, key and value gradients" in desc
            or "kl-loss gradients" in desc
        )

    is_cv_ring = uses_cube and (
        has_attention_attrs
        or feat.get("recurrence") == "associative_online_rescale"
        or "online softmax" in contract_text
        or ("log_softmax" in formula and "@" in formula)
        or ("softmax(" in formula and "@" in formula)
        or "attn_sink" in contract_text
        or "sparse" in contract_text
    )

    has_w4_quant = (
        bool(sf_dict.get("has_w4_quant", spec_dict.get("has_w4_quant", False)))
        or feat.get("quant") == "mixed_fp8_fp4"
        or any(
            tok in dt
            for dt in dtype_strs
            for tok in ("float4", "fp4", "q4", "int4", "uint4")
        )
        or any(dt in ("i4", "u4") for dt in dtype_strs)
        or ("quant_type" in attrs_keys and any(dt == "uint8" for dt in dtype_strs))
    )
    has_mx_quant = (
        bool(sf_dict.get("has_mx_quant", False))
        or any("e8m0" in dt for dt in dtype_strs)
        or ("blocksize" in attrs_keys and "dst_type" in attrs_keys)
    )
    has_quant = (
        has_w4_quant
        or has_mx_quant
        or (cat == "quantization")
        or ("scaleout" in formula)
        or (feat.get("quant", "none") != "none")
        or any(
            tok in dt
            for dt in dtype_strs
            for tok in ("int8", "uint8", "float8", "fp8", "hif8")
        )
    )

    is_ragged = (
        any_empty_group
        or is_grouped
        or (
            uses_cube
            and bool(feat)
            and (not feat.get("static_bounds", True))
            and not feat.get("indirect", False)
        )
    )

    has_index_input = any(
        idx > 0 and any(tok in dt for tok in ("int32", "int64"))
        for idx, item in enumerate(inputs)
        if isinstance(item, Mapping)
        for dt in flatten_str_values(item.get("dtype"))
    )
    has_sparse_gather = (
        any(
            any(
                tok in str(item.get("name", "")).lower()
                for tok in ("sparse", "compressed_indices", "topk_index")
            )
            for item in inputs
            if isinstance(item, Mapping)
        )
        or ("gather(" in formula)
    )
    uses_simt = (
        bool(sf_dict.get("uses_simt", spec_dict.get("uses_simt", False)))
        or bool(feat.get("indirect", False))
        or (0 < int(feat.get("inner_bytes", 0)) < 32 and not uses_cube)
        or (
            bool(feat)
            and not bool(feat.get("static_bounds", True))
            and not uses_cube
            and not is_opcode_stream
        )
        or cat in ("hash", "hashtable", "index", "indexgather", "scatterupdate")
        or (cat == "sortselect" and (len(outputs) >= 2 or "iou_threshold" in attrs_keys))
        or ("grid_sample" in formula or "roi_align" in formula)
        or ("max_{" in formula and "filter[" in formula)
        or ("gcd(" in formula)
        or (
            has_index_input
            and cat in ("layouttransform", "transform")
            and ("dstoffset" in contract_text or "slot_mapping" in contract_text)
        )
        or (is_cv_ring and has_sparse_gather)
    )

    if is_opcode_stream:
        iteration_regularity = "dynamic_opcode_stream"
    elif is_ragged:
        iteration_regularity = "ragged_group"
    elif uses_simt:
        iteration_regularity = "irregular_simt"
    else:
        iteration_regularity = "affine_simd"

    if is_recurrent or is_opcode_stream:
        memory_ring = "persistent_onchip_state"
    elif is_cv_ring:
        memory_ring = "bidirectional_cv_ring"
    elif uses_cube and iteration_regularity == "ragged_group":
        memory_ring = "l0c_ub_epilogue"
    elif uses_cube:
        memory_ring = "gm_l1_l0_fixpipe"
    elif iteration_regularity == "irregular_simt":
        memory_ring = "ub_random_atomic"
    else:
        memory_ring = "gm_ub_stream"

    if uses_cube and (
        memory_ring in ("bidirectional_cv_ring", "l0c_ub_epilogue", "persistent_onchip_state")
        or has_quant
        or uses_simt
        or is_ragged
    ):
        compute_engine = "coupled_1c2v"
    elif uses_cube:
        compute_engine = "aic_only"
    else:
        compute_engine = "aiv_only"

    axes = TopologyAxes(
        compute_engine=compute_engine,
        memory_ring=memory_ring,
        iteration_regularity=iteration_regularity,
    )
    has_kv_cache = has_index_input and cat == "transform" and "slot_mapping" in contract_text
    has_dead_l1_slice = len(inputs) >= 4 and axes.memory_ring == "bidirectional_cv_ring"
    uses_mutex_atomic = cat in ("hash", "hashtable") or (
        cat == "sortselect" and "return_inverse" in attrs_keys
    )
    flags = {
        "uses_cube": uses_cube,
        "uses_simt": uses_simt and not uses_cube,
        "is_recurrent": is_recurrent,
        "has_w4_quant": has_w4_quant,
        "has_mx_quant": has_mx_quant,
        "has_quant": has_quant,
        "is_ragged": bool(is_ragged),
        "is_cv_ring": is_cv_ring,
        "is_opcode_stream": is_opcode_stream,
        "has_kv_cache": has_kv_cache,
        "has_dead_l1_slice": has_dead_l1_slice,
        "uses_mutex_atomic": uses_mutex_atomic,
    }
    return axes, flags


def _extract_feature_flags(features: object | None) -> dict[str, Any]:
    if not isinstance(features, Mapping):
        return {}
    cube_ratio_raw = features.get("cube_to_vector_ratio")
    if cube_ratio_raw is None and features.get("uses_cube"):
        cube_ratio = 1.0
    else:
        cube_ratio = float(cube_ratio_raw or 0.0)
    return {
        "static_bounds": bool(features.get("static_affine_bounds", True)),
        "cube_ratio": cube_ratio,
        "quant": str(features.get("quant_granularity", "none")),
        "indirect": bool(
            features.get("indirect_sparse_gather", features.get("uses_simt", False))
        ),
        "recurrence": str(features.get("cross_iter_recurrence", "none")),
        "handoff": bool(features.get("dynamic_cross_core_handoff", False)),
        "inner_bytes": int(features.get("inner_contiguous_bytes", 0) or 0),
    }


def resolve_canonical_topology_axes(
    spec_or_name: object,
    features: object | None = None,
) -> TopologyAxes:
    """Resolve canonical 3-axis TopologyAxes directly from structural and dtype features."""
    if spec_or_name is not None:
        direct_axes = (
            spec_or_name.get("topology_axes")
            if isinstance(spec_or_name, Mapping)
            else getattr(spec_or_name, "topology_axes", None)
        )
        if isinstance(direct_axes, TopologyAxes):
            return direct_axes
        if isinstance(direct_axes, Mapping) and all(
            k in direct_axes for k in ("compute_engine", "memory_ring", "iteration_regularity")
        ):
            return TopologyAxes.from_dict(direct_axes)

    spec_map: Mapping[str, Any] = {}
    proto_map: Mapping[str, Any] = {}
    sf_map: Mapping[str, Any] = {}
    regimes: Any = ()
    dtypes: list[str] = []

    if isinstance(spec_or_name, Mapping):
        spec_map = spec_or_name
        op_block = spec_map.get("operator")
        proto_map = op_block if isinstance(op_block, Mapping) else spec_map
        raw_sf = spec_map.get("structural_features")
        if isinstance(raw_sf, Mapping):
            sf_map = raw_sf
        regimes = spec_map.get("regimes") or ()
        for k in ("input_dtypes", "output_dtypes"):
            vals = spec_map.get(k)
            if isinstance(vals, (list, tuple)):
                dtypes.extend(str(v) for v in vals)
        for sec in ("inputs", "outputs"):
            entries = proto_map.get(sec)
            if isinstance(entries, (list, tuple)):
                for entry in entries:
                    if isinstance(entry, Mapping):
                        dtypes.extend(flatten_str_values(entry.get("dtype")))
    elif spec_or_name is not None and not isinstance(spec_or_name, str):
        raw_sf = getattr(spec_or_name, "structural_features", None)
        if isinstance(raw_sf, Mapping):
            sf_map = raw_sf
        regimes = getattr(spec_or_name, "regimes", ()) or ()
        for k in ("input_dtypes", "output_dtypes"):
            vals = getattr(spec_or_name, k, None)
            if isinstance(vals, (list, tuple)):
                dtypes.extend(str(v) for v in vals)
        if features is None and hasattr(spec_or_name, "features"):
            features = getattr(spec_or_name, "features")

    axes, _flags = infer_contraction_and_axes_from_proto(
        proto_map,
        (),
        spec_map=spec_map,
        sf_map=sf_map,
        dtypes=dtypes,
        regimes=regimes,
        features=features,
    )
    return axes


class _CasePhysicsRecord(TypedDict):
    """One official case's ratios, tail flags, timings, and dtype-aware workload."""

    case_id: int
    ratios: CasePhysicsRatios
    unaligned: bool
    empty_group: bool
    t_hw_us: float
    ref_perf_us: float
    workload_bytes: int


class OpAdvisoryExtractor:
    """Extracts SIL AdvisoryEnvelopes from cann-bench operator definitions and 950PR metadata."""

    def __init__(self, ops_root: str | Path = DEFAULT_OPS_ROOT) -> None:
        self.ops_root = _resolve_ops_root_path(ops_root)
        self._manifest_cache: dict[str, Any] | None = None
        self._perf_cache: dict[str, Any] | None = None
        target = AscendTarget("ascend.950pr")
        facts = target.get_facts(AscendSpaceFacts)
        if not isinstance(facts, AscendSpaceFacts):
            raise RuntimeError("ascend.950pr target did not provide AscendSpaceFacts")
        self.facts = facts

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest_cache is None:
            manifest_path = self.ops_root / "suite_manifest.json"
            self._manifest_cache = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._manifest_cache

    @property
    def perf_metadata(self) -> dict[str, Any]:
        if self._perf_cache is None:
            perf_path = self.ops_root / "metadata" / "950pr.json"
            self._perf_cache = json.loads(perf_path.read_text(encoding="utf-8"))
        return self._perf_cache

    def list_operators(self) -> list[dict[str, Any]]:
        """Return all operator entries declared in suite_manifest.json."""
        return list(self.manifest.get("operators", []))

    def _find_op_entry(
        self,
        op_name: str,
        level: str | int | None = None,
    ) -> tuple[int, Path]:
        target_level = None
        if level is not None:
            if isinstance(level, int):
                target_level = level
            else:
                text = str(level).strip().lower()
                if text.startswith("level"):
                    target_level = int(text.removeprefix("level"))
                elif text.startswith("l"):
                    target_level = int(text.removeprefix("l"))
                else:
                    target_level = int(text)

        for entry in self.list_operators():
            if entry.get("id") != op_name:
                continue
            rel = str(entry["path"]).removeprefix("tasks/")
            lvl_num = int(str(entry["level"]).removeprefix("L"))
            if target_level is not None and lvl_num != target_level:
                continue
            return lvl_num, self.ops_root / rel

        raise KeyError(f"Operator {op_name!r} (level={level!r}) not found in {self.ops_root}")

    def extract(self, op_name: str, level: str | int | None = None) -> AdvisoryEnvelope:
        """Build an AdvisoryEnvelope for an operator from proto, cases, and 950pr specs."""
        lvl_num, op_dir = self._find_op_entry(op_name, level=level)
        level_key = f"level{lvl_num}"

        proto_raw = yaml.safe_load((op_dir / "proto.yaml").read_text(encoding="utf-8"))
        proto = proto_raw.get("operator", proto_raw)
        cases_raw = yaml.safe_load((op_dir / "cases.yaml").read_text(encoding="utf-8"))
        cases = list(cases_raw.get("cases", []))

        perf_op = self.perf_metadata.get(level_key, {}).get(op_name, {})

        in_dtypes_set: list[str] = []
        for item in proto.get("inputs") or []:
            for dt in flatten_str_values(item.get("dtype")):
                if dt not in in_dtypes_set:
                    in_dtypes_set.append(dt)
        for case in cases:
            for dt in flatten_str_values(case.get("dtype")):
                if dt not in in_dtypes_set:
                    in_dtypes_set.append(dt)

        out_dtypes_set: list[str] = []
        for item in proto.get("outputs") or []:
            for dt in flatten_str_values(item.get("dtype")):
                if dt not in out_dtypes_set:
                    out_dtypes_set.append(dt)

        attrs_keys: set[str] = set()
        any_empty_group = False
        for case in cases:
            attrs = case.get("attrs") or {}
            if isinstance(attrs, Mapping):
                attrs_keys.update(str(k) for k in attrs.keys())
            if has_empty_group_attrs(attrs):
                any_empty_group = True

        topology_axes, flags = infer_contraction_and_axes_from_proto(
            proto,
            cases,
            sf_map={"attrs_keys": sorted(attrs_keys)},
            dtypes=(*in_dtypes_set, *out_dtypes_set),
            any_empty_group_hint=any_empty_group,
        )
        uses_cube = flags["uses_cube"]
        uses_simt = flags["uses_simt"]
        is_recurrent = flags["is_recurrent"]
        has_w4_quant = flags["has_w4_quant"]
        has_mx_quant = flags["has_mx_quant"]
        has_kv_cache = flags["has_kv_cache"]
        has_dead_l1_slice = flags["has_dead_l1_slice"]
        uses_mutex_atomic = flags["uses_mutex_atomic"]

        raw_pt = proto.get("precision_thresholds") or {}
        precision_thresholds: dict[str, float] = {
            str(k): float(v) for k, v in raw_pt.items()
        }
        for dt in (*in_dtypes_set, *out_dtypes_set):
            if dt in _CANNBENCH_DEFAULT_THRESHOLDS and dt not in precision_thresholds:
                precision_thresholds[dt] = _CANNBENCH_DEFAULT_THRESHOLDS[dt]

        has_nan_inf_cases = any(
            "inf" in str(c.get("value_range", "")).lower()
            or "nan" in str(c.get("value_range", "")).lower()
            for c in cases
        )
        has_inplace_aliasing = (
            has_kv_cache
            or any(
                isinstance(c.get("attrs"), Mapping) and bool(c["attrs"].get("inplace", False))
                for c in cases
            )
            or "alias" in str(proto.get("description", "")).lower()
            or "inplace" in attrs_keys
        )

        case_kind = _case_physics_kind(proto, topology_axes)
        case_records: dict[int, _CasePhysicsRecord] = {}
        ratios_by_case: dict[int, CasePhysicsRatios] = {}
        dims_by_case: dict[int, dict[str, int]] = {}
        workload_bytes_by_case: dict[int, int] = {}
        has_prime_or_odd = False
        any_unaligned = False
        total_gm_bytes_samples: list[int] = []
        t_hw_samples: list[float] = []
        ref_perf_samples: list[float] = []
        case_ratio_payloads: list[dict[str, object]] = []

        for case in cases:
            cid = int(case["case_id"])
            shapes = flatten_int_shapes(case.get("input_shape"))
            dtypes = list(flatten_str_values(case.get("dtype")))
            attrs = case.get("attrs") or {}

            dims = infer_case_dims_structural(
                shapes,
                attrs,
                axes=topology_axes,
                uses_cube=uses_cube,
            )
            empty_grp = has_empty_group_attrs(attrs)
            case_context = build_case_physics_context(
                case_kind,
                case_primary_dtype_bytes(case),
                self.facts,
                strided_gather=(
                    case_kind == "attention"
                    and topology_axes.iteration_regularity == "irregular_simt"
                ),
                has_empty_group=empty_grp,
            )
            ratios = compute_case_physics_ratios(
                derive_case_physics_input(cid, shapes, attrs, dims, case_context)
            )
            case_ratio_payloads.append(ratios.to_dict())
            unaligned = has_unaligned_tail_for_case(shapes, dtypes, uses_cube)
            any_unaligned = any_unaligned or unaligned
            if any(_is_prime_or_odd(d) and d > 7 for shp in shapes for d in shp):
                has_prime_or_odd = True

            perf_rec = perf_op.get(str(cid), {})
            t_hw = float(perf_rec.get("t_hw_us", 0.0))
            b_perf = float(perf_rec.get("baseline_perf_us", 0.0))
            t_hw_samples.append(t_hw)
            ref_perf_samples.append(b_perf)

            case_bytes = case_workload_bytes(shapes, dtypes)
            total_gm_bytes_samples.append(case_bytes)

            ratios_by_case[cid] = ratios
            dims_by_case[cid] = dims
            workload_bytes_by_case[cid] = case_bytes
            case_records[cid] = {
                "case_id": cid,
                "ratios": ratios,
                "unaligned": unaligned,
                "empty_group": empty_grp,
                "t_hw_us": t_hw,
                "ref_perf_us": b_perf,
                "workload_bytes": case_bytes,
            }

        regimes: list[RegimeSlice] = []
        groups = partition_case_regimes(
            ratios_by_case,
            dims_by_case,
            workload_bytes_by_case,
            kind=case_kind,
            axes=topology_axes,
        )
        for group in groups:
            items = [case_records[cid] for cid in group.case_ids]
            items_sorted = sorted(
                items,
                key=lambda x: (x["workload_bytes"], x["case_id"]),
            )
            representative = group.representative_case
            regimes.append(
                RegimeSlice(
                    regime_tag=group.regime_tag,
                    case_nums=group.case_ids,
                    representative_dims=dict(representative.dims),
                    has_unaligned_tail=any(x["unaligned"] for x in items),
                    has_empty_group=any(x["empty_group"] for x in items),
                    ref_perf_us=round(statistics.fmean(x["ref_perf_us"] for x in items), 4),
                    t_hw_us=round(statistics.fmean(x["t_hw_us"] for x in items), 4),
                    case_ratios=tuple(x["ratios"] for x in items_sorted),
                    representative_case_id=representative.case_id,
                    phase_coordinates=group.phase_coordinates,
                    rationale=group.rationale,
                )
            )

        dominant_bound = (
            "cube_compute"
            if uses_cube
            else (
                "vector_compute"
                if topology_axes.memory_ring == "persistent_onchip_state"
                else "memory_bandwidth"
            )
        )
        roofline: dict[str, object] = {
            "min_t_hw_us": min(t_hw_samples) if t_hw_samples else 0.0,
            "max_t_hw_us": max(t_hw_samples) if t_hw_samples else 0.0,
            "mean_t_hw_us": round(statistics.fmean(t_hw_samples), 4) if t_hw_samples else 0.0,
            "min_baseline_perf_us": min(ref_perf_samples) if ref_perf_samples else 0.0,
            "max_baseline_perf_us": max(ref_perf_samples) if ref_perf_samples else 0.0,
            "mean_baseline_perf_us": (
                round(statistics.fmean(ref_perf_samples), 4) if ref_perf_samples else 0.0
            ),
            "median_gm_bytes": (
                int(statistics.median(total_gm_bytes_samples)) if total_gm_bytes_samples else 0
            ),
            "max_gm_bytes": max(total_gm_bytes_samples) if total_gm_bytes_samples else 0,
            "dominant_bound": dominant_bound,
        }

        structural_features: dict[str, object] = {
            "category": str(proto.get("category", "")),
            "schema": str(proto.get("schema", "")),
            "num_cases": len(cases),
            "has_unaligned_tail": any_unaligned,
            "has_empty_group": any_empty_group,
            "has_prime_or_odd_dims": has_prime_or_odd,
            "uses_cube": uses_cube,
            "uses_simt": uses_simt,
            "has_w4_quant": has_w4_quant,
            "has_mx_quant": has_mx_quant,
            "is_recurrent": is_recurrent,
            "has_kv_cache": has_kv_cache,
            "has_dead_l1_slice": has_dead_l1_slice,
            "uses_mutex_atomic": uses_mutex_atomic,
            "attrs_keys": sorted(attrs_keys),
            "case_physics_ratios": case_ratio_payloads,
        }

        return AdvisoryEnvelope(
            op_name=op_name,
            cannbot_task=f"{level_key}/{op_name}",
            level=lvl_num,
            topology_axes=topology_axes,
            input_dtypes=tuple(in_dtypes_set),
            output_dtypes=tuple(out_dtypes_set),
            roofline=roofline,
            structural_features=structural_features,
            regimes=tuple(regimes),
            legality_bounds=HardwareLegalityBounds(),
            has_nan_inf_cases=has_nan_inf_cases,
            has_inplace_aliasing=has_inplace_aliasing,
            precision_thresholds=precision_thresholds,
            hir_blueprint=None,
        )

    def extract_all(self) -> dict[str, AdvisoryEnvelope]:
        """Extract AdvisoryEnvelopes for all operators in suite_manifest.json."""
        result: dict[str, AdvisoryEnvelope] = {}
        for entry in self.list_operators():
            op_id = str(entry["id"])
            result[op_id] = self.extract(op_id, level=entry["level"])
        return result

    @staticmethod
    def synthesize_default_policy(advisory: AdvisoryEnvelope) -> PolicyEnvelope:
        """Synthesize a hardware-legal default PolicyEnvelope for any AdvisoryEnvelope."""
        return synthesize_default_policy(advisory)


def synthesize_default_policy(advisory: AdvisoryEnvelope) -> PolicyEnvelope:
    """Synthesize a hardware-legal default PolicyEnvelope passing SILVerifier for all regimes."""
    bounds = advisory.legality_bounds
    policies: dict[str, SchedulePolicy] = {}
    resolved_axes = advisory.topology_axes
    selected_rules: list[str] = [f"axes:{resolved_axes.key()}"]

    is_q4 = bool(advisory.structural_features.get("has_w4_quant", False)) and (
        resolved_axes.uses_cube
        or bool(advisory.structural_features.get("uses_cube", False))
    )
    if is_q4:
        cube_dtype = "q4_0"
        w4_mode = "aiv_unpack_f16"
        selected_rules.append("w4_aiv_unpack_f16")
    elif "int8" in advisory.input_dtypes:
        cube_dtype = "int8"
        w4_mode = "none"
    elif "float16" in advisory.input_dtypes:
        cube_dtype = "float16"
        w4_mode = "none"
    elif "bfloat16" in advisory.input_dtypes:
        cube_dtype = "bfloat16"
        w4_mode = "none"
    else:
        cube_dtype = "float32"
        w4_mode = "none"

    for regime in advisory.regimes:
        tag = regime.regime_tag
        dims = regime.representative_dims

        if resolved_axes.iteration_regularity == "dynamic_opcode_stream":
            cols = max(1, min(int(dims.get("cols", 1024)), 2048))
            tile_mnk = (64, cols, 1)
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="ascendc_template",
                tile_mnk=tile_mnk,
                l1_stages=1,
                l1_half_bank_split=False,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=min(196608, bounds.ub_simdvf_bytes),
                l1_estimated_bytes=0,
                ub_bank_pad_datablocks=1,
                ub_rw_shift_datablocks=8,
                cv_datapath="none",
                sync_mode="local_bar",
                preload_depth=1,
                reverse_ack_period=0,
                ensure_empty_subblock_sync=True,
                vector_mode="simdvf",
                direct_no_fusion_vf=True,
                hf32_allowed=False,
                fast_unshifted_exp=False,
                dma_paths_used=("GM->UB", "UB->GM"),
                cube_operand_dtype="",
            )

        elif resolved_axes.memory_ring == "persistent_onchip_state":
            uses_cube = resolved_axes.uses_cube or bool(
                advisory.structural_features.get("uses_cube", False)
            )
            tile_mnk = (64, 64, 64) if uses_cube else (32, min(int(dims.get("cols", 256)), 1024), 1)
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="tilelang_dsl",
                tile_mnk=tile_mnk,
                l1_stages=2 if uses_cube else 1,
                l1_half_bank_split=uses_cube,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=196608,
                l1_estimated_bytes=131072 if uses_cube else 0,
                ub_bank_pad_datablocks=1,
                ub_rw_shift_datablocks=8,
                cv_datapath="l0c2ub_split_m" if uses_cube else "none",
                sync_mode="mode4_flag" if uses_cube else "local_bar",
                preload_depth=2,
                reverse_ack_period=14 if uses_cube else 0,
                ensure_empty_subblock_sync=True,
                vector_mode="simdvf",
                hf32_allowed=False,
                fast_unshifted_exp=False,
                dma_paths_used=(
                    ("GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->L1", "UB->GM")
                    if uses_cube
                    else ("GM->UB", "UB->GM")
                ),
                cube_operand_dtype=cube_dtype if uses_cube else "",
            )

        elif resolved_axes.memory_ring == "bidirectional_cv_ring":
            is_decode = "decode" in tag
            tile_mnk = (16, 128, 64) if is_decode else (96, 112, 64)
            bm, bn, bk = tile_mnk
            l1_total = min((bm * bk + bk * bn) * 2 * 2, bounds.l1_bytes)
            has_dead_l1 = bool(advisory.structural_features.get("has_dead_l1_slice", False))
            slot_alias = {"aux_key_tail": "epilogue_l1_slot"} if has_dead_l1 else {}
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="tilelang_dsl",
                tile_mnk=tile_mnk,
                num_kv_splits=4 if is_decode else 1,
                l1_stages=2,
                l1_half_bank_split=True,
                l1_slot_aliasing=slot_alias,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=196608,
                l1_estimated_bytes=l1_total,
                ub_bank_pad_datablocks=1,
                ub_rw_shift_datablocks=8,
                cv_datapath="l0c2ub_split_m",
                sync_mode="mode4_flag",
                preload_depth=3,
                reverse_ack_period=14,
                ensure_empty_subblock_sync=True,
                swizzle_window=4,
                tail_wave_split=True,
                vector_mode="simdvf",
                hf32_allowed=False,
                fast_unshifted_exp=False,
                dma_paths_used=("GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->L1", "UB->GM"),
                cube_operand_dtype=cube_dtype,
            )

        elif resolved_axes.memory_ring in ("gm_l1_l0_fixpipe", "l0c_ub_epilogue") or resolved_axes.uses_cube:
            is_grouped = (
                resolved_axes.memory_ring == "l0c_ub_epilogue"
                or resolved_axes.iteration_regularity == "ragged_group"
            )
            bm, bn, bk = (128, 128, 64)
            if "small" in tag:
                bm, bn, bk = (64, 64, 32)
            elem_b = 1 if cube_dtype == "int8" else 2
            l1_per_stage = (bm * bk + bk * bn) * elem_b
            l1_total = min(l1_per_stage * 2, bounds.l1_bytes)
            dma_paths = (
                ("GM->UB", "UB->L1", "GM->L1", "L1->L0A", "L1->L0B", "L0C->GM")
                if is_q4
                else ("GM->L1", "L1->L0A", "L1->L0B", "L0C->GM")
            )
            cv_dp = "l0c2ub_split_m" if is_grouped else "none"
            if cv_dp != "none":
                dma_paths = ("GM->L1", "L1->L0A", "L1->L0B", "L0C->UB", "UB->GM")
            sync_mode = (
                "unit_flag_2_3"
                if (not is_grouped and not regime.has_unaligned_tail and not is_q4)
                else "mode4_flag"
            )
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="ascendc_template" if is_grouped else "tilelang_dsl",
                tile_mnk=(bm, bn, bk),
                l1_stages=2,
                l1_half_bank_split=True,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=131072 if (is_q4 or cv_dp != "none") else 0,
                l1_estimated_bytes=l1_total,
                ub_bank_pad_datablocks=1,
                ub_rw_shift_datablocks=8,
                cv_datapath=cv_dp,
                sync_mode=sync_mode,
                preload_depth=2,
                reverse_ack_period=14,
                ensure_empty_subblock_sync=True,
                swizzle_window=4,
                tail_wave_split=True,
                vector_mode="simdvf",
                hf32_allowed=False,
                fast_unshifted_exp=False,
                w4_mode=w4_mode,
                dma_paths_used=dma_paths,
                cube_operand_dtype=cube_dtype,
            )

        elif resolved_axes.memory_ring == "ub_random_atomic" or resolved_axes.iteration_regularity == "irregular_simt":
            cols = max(1, min(int(dims.get("cols", 256)), 1024))
            tile_mnk = (32, cols, 1)
            ub_bytes = min(131072, bounds.ub_simtvf_bytes)
            uses_mutex = bool(advisory.structural_features.get("uses_mutex_atomic", False))
            sync_mode = "mutex" if uses_mutex else "local_bar"
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="ascendc_template",
                tile_mnk=tile_mnk,
                row_pack_factor=1,
                l1_stages=1,
                l1_half_bank_split=False,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=ub_bytes,
                l1_estimated_bytes=0,
                ub_bank_pad_datablocks=1,
                ub_rw_shift_datablocks=8,
                ub_row_stride_bytes=0,
                cv_datapath="none",
                sync_mode=sync_mode,
                preload_depth=1,
                reverse_ack_period=0,
                ensure_empty_subblock_sync=True,
                vector_mode="hybrid",
                simt_threads=512,
                hf32_allowed=False,
                fast_unshifted_exp=False,
                dma_paths_used=("GM->UB", "UB->GM"),
                cube_operand_dtype="",
            )

        else:
            cols = max(1, min(int(dims.get("cols", 1024)), 2048))
            row_pack = 16 if cols <= 256 else 4
            tile_mnk = (min(row_pack * 2, 128), cols, 1)
            ub_bytes = min(tile_mnk[0] * tile_mnk[1] * 4 * 2, bounds.ub_simdvf_bytes)
            stride_bytes = tile_mnk[1] * 4
            pad_db = 1 if (stride_bytes > 0 and stride_bytes % 512 == 0) else 0
            shift_db = 8 if pad_db > 0 else 0
            policies[tag] = SchedulePolicy(
                regime_tag=tag,
                topology_axes=resolved_axes,
                backend_tier="tilelang_dsl",
                tile_mnk=tile_mnk,
                row_pack_factor=row_pack,
                l1_stages=1,
                l1_half_bank_split=False,
                l0c_buffers=1,
                ub_256kb_unlock=False,
                ub_estimated_bytes=ub_bytes,
                l1_estimated_bytes=0,
                ub_bank_pad_datablocks=pad_db,
                ub_rw_shift_datablocks=shift_db,
                ub_row_stride_bytes=stride_bytes,
                cv_datapath="none",
                sync_mode="local_bar",
                preload_depth=2,
                reverse_ack_period=0,
                ensure_empty_subblock_sync=True,
                vector_mode="simdvf",
                direct_no_fusion_vf=True,
                multi_row_unroll=2 if row_pack >= 2 else 1,
                hf32_allowed=False,
                fast_unshifted_exp=False,
                dma_paths_used=("GM->UB", "UB->GM"),
                cube_operand_dtype="",
            )

    return PolicyEnvelope(
        op_name=advisory.op_name,
        policies=policies,
        selected_rules=tuple(selected_rules),
    )


@lru_cache(maxsize=4)
def get_default_extractor(ops_root: str = str(DEFAULT_OPS_ROOT)) -> OpAdvisoryExtractor:
    """Return a cached OpAdvisoryExtractor for the given cann-bench tasks root."""
    return OpAdvisoryExtractor(ops_root)
