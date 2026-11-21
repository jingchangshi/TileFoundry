"""Stratified CANNBench anchor registry and official scoring helpers for campaigns."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import (
    AlignmentBand,
    CasePhysicsRatios,
    CasePhysicsRegimeGroup,
    EnginePhase,
    LayoutPhase,
    ResidencyBand,
    TopologyAxes,
    WaveBand,
    case_workload_bytes,
    classify_case_regime,
    flatten_int_shapes,
    flatten_str_values,
    infer_case_dims_structural,
    infer_case_physics_kind,
    partition_case_regimes,
)


@dataclass(frozen=True)
class AnchorCaseSpec:
    """Single CANNBench anchor case with calibrated reference and Roofline baselines."""

    task: str
    case_num: int
    regime_tag: str
    ref_perf_us: float
    t_hw_us: float

    def to_dict(self) -> dict[str, object]:
        """Serialize anchor specification to a JSON-compatible dictionary."""
        return {
            "task": self.task,
            "case_num": self.case_num,
            "regime_tag": self.regime_tag,
            "ref_perf_us": self.ref_perf_us,
            "t_hw_us": self.t_hw_us,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnchorCaseSpec:
        """Reconstruct AnchorCaseSpec from a JSON mapping."""
        return cls(
            task=str(payload["task"]),
            case_num=int(payload["case_num"]),
            regime_tag=str(payload["regime_tag"]),
            ref_perf_us=float(payload["ref_perf_us"]),
            t_hw_us=float(payload["t_hw_us"]),
        )


@dataclass(frozen=True)
class PerfTargetSpec:
    """Operator-level performance and Roofline convergence targets."""

    operator_name: str
    target_avg_speedup: float
    target_min_case_speedup: float
    target_roofline_eta: float
    per_anchor_max_us: dict[int, float] = field(default_factory=dict)
    target_surrogate_score: float = 90.0
    min_optimization_rounds: int = 3

    def to_dict(self) -> dict[str, object]:
        """Serialize target thresholds to a JSON-compatible dictionary."""
        return {
            "operator_name": self.operator_name,
            "target_avg_speedup": self.target_avg_speedup,
            "target_min_case_speedup": self.target_min_case_speedup,
            "target_roofline_eta": self.target_roofline_eta,
            "target_surrogate_score": self.target_surrogate_score,
            "min_optimization_rounds": self.min_optimization_rounds,
            "per_anchor_max_us": {str(k): v for k, v in sorted(self.per_anchor_max_us.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PerfTargetSpec:
        """Reconstruct PerfTargetSpec from a JSON mapping."""
        raw_max = payload.get("per_anchor_max_us") or {}
        per_anchor = (
            {int(k): float(v) for k, v in raw_max.items()}
            if isinstance(raw_max, Mapping)
            else {}
        )
        return cls(
            operator_name=str(payload["operator_name"]),
            target_avg_speedup=float(payload["target_avg_speedup"]),
            target_min_case_speedup=float(payload["target_min_case_speedup"]),
            target_roofline_eta=float(payload["target_roofline_eta"]),
            per_anchor_max_us=per_anchor,
            target_surrogate_score=float(payload.get("target_surrogate_score", 90.0)),
            min_optimization_rounds=int(payload.get("min_optimization_rounds", 3)),
        )


def _case_ratios_to_dict(ratios: CasePhysicsRatios) -> dict[str, object]:
    return {
        "case_id": ratios.case_id,
        "stride_ratio": ratios.stride_ratio,
        "contig_atom_ratio": ratios.contig_atom_ratio,
        "is_atom_aligned": ratios.is_atom_aligned,
        "working_set_sram_ratio": ratios.working_set_sram_ratio,
        "working_set_vreg_ratio": ratios.working_set_vreg_ratio,
        "wave_occupancy_ratio": ratios.wave_occupancy_ratio,
        "oi_ridge_ratio": ratios.oi_ridge_ratio,
        "inner_contiguous_bytes": ratios.inner_contiguous_bytes,
        "has_empty_group": ratios.has_empty_group,
    }


def _ratios_int(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CasePhysicsRatios.{key} must be an int")
    return value


def _ratios_float(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CasePhysicsRatios.{key} must be a number")
    return float(value)


def _ratios_bool(payload: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"CasePhysicsRatios.{key} must be a bool")
    return value


def _case_ratios_from_dict(payload: Mapping[str, object]) -> CasePhysicsRatios:
    return CasePhysicsRatios(
        case_id=_ratios_int(payload, "case_id"),
        stride_ratio=_ratios_float(payload, "stride_ratio"),
        contig_atom_ratio=_ratios_float(payload, "contig_atom_ratio"),
        is_atom_aligned=_ratios_bool(payload, "is_atom_aligned"),
        working_set_sram_ratio=_ratios_float(payload, "working_set_sram_ratio"),
        working_set_vreg_ratio=_ratios_float(payload, "working_set_vreg_ratio"),
        wave_occupancy_ratio=_ratios_float(payload, "wave_occupancy_ratio"),
        oi_ridge_ratio=_ratios_float(payload, "oi_ridge_ratio"),
        inner_contiguous_bytes=_ratios_int(payload, "inner_contiguous_bytes"),
        has_empty_group=_ratios_bool(payload, "has_empty_group"),
    )


@dataclass(frozen=True)
class RegimeTargetSpec:
    """Target dispatch regime inside an operator kernel module."""

    regime_tag: str
    builder_candidates: tuple[str, ...]
    sample_builder_kwargs: dict[str, object]
    description: str = ""
    representative_hir_dims: dict[str, int] = field(default_factory=dict)
    representative_anchor_dims: dict[str, int] = field(default_factory=dict)
    config_selector_fn: str = ""
    editable_symbols: tuple[str, ...] = ()
    allowed_helper_prefix: str = ""
    runtime_backend: str = "tilelang_dsl"
    trackb_symbols: tuple[str, ...] = ()
    representative_case_entry: dict[str, object] = field(default_factory=dict)
    representative_case_ratios: CasePhysicsRatios | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize regime target specification to a JSON-compatible dictionary."""
        return {
            "regime_tag": self.regime_tag,
            "builder_candidates": list(self.builder_candidates),
            "sample_builder_kwargs": dict(self.sample_builder_kwargs),
            "description": self.description,
            "representative_hir_dims": {
                str(k): int(v) for k, v in self.representative_hir_dims.items()
            },
            "representative_anchor_dims": {
                str(k): int(v) for k, v in self.representative_anchor_dims.items()
            },
            "config_selector_fn": self.config_selector_fn,
            "editable_symbols": list(self.editable_symbols),
            "allowed_helper_prefix": self.allowed_helper_prefix,
            "runtime_backend": self.runtime_backend,
            "trackb_symbols": list(self.trackb_symbols),
            "representative_case_entry": dict(self.representative_case_entry),
            "representative_case_ratios": (
                _case_ratios_to_dict(self.representative_case_ratios)
                if self.representative_case_ratios is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimeTargetSpec:
        """Reconstruct RegimeTargetSpec from a JSON mapping."""
        raw_kwargs = payload.get("sample_builder_kwargs") or {}
        raw_dims = payload.get("representative_hir_dims") or {}
        raw_anchor_dims = payload.get("representative_anchor_dims") or {}
        raw_case_entry = payload.get("representative_case_entry") or {}
        raw_case_ratios = payload.get("representative_case_ratios")
        return cls(
            regime_tag=str(payload["regime_tag"]),
            builder_candidates=tuple(str(x) for x in payload.get("builder_candidates", ())),
            sample_builder_kwargs=dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {},
            description=str(payload.get("description", "")),
            representative_hir_dims=(
                {str(k): int(v) for k, v in raw_dims.items()}
                if isinstance(raw_dims, Mapping)
                else {}
            ),
            representative_anchor_dims=(
                {str(k): int(v) for k, v in raw_anchor_dims.items()}
                if isinstance(raw_anchor_dims, Mapping)
                else {}
            ),
            config_selector_fn=str(payload.get("config_selector_fn", "")),
            editable_symbols=tuple(str(x) for x in payload.get("editable_symbols", ())),
            allowed_helper_prefix=str(payload.get("allowed_helper_prefix", "")),
            runtime_backend=str(
                payload.get("runtime_backend", "tilelang_dsl") or "tilelang_dsl"
            ),
            trackb_symbols=tuple(str(x) for x in (payload.get("trackb_symbols", ()) or ())),
            representative_case_entry=(
                dict(raw_case_entry) if isinstance(raw_case_entry, Mapping) else {}
            ),
            representative_case_ratios=(
                _case_ratios_from_dict(raw_case_ratios)
                if isinstance(raw_case_ratios, Mapping)
                else None
            ),
        )


RegimeSpec = RegimeTargetSpec


OPERATOR_METADATA: dict[str, tuple[str, str, str]] = {
    "attention_residual": (
        "level4/attention_residual",
        "examples/ascend950pr_ops/attention_residual/kernels.py",
        "attention_residual",
    ),
    "mla": (
        "level4/mla",
        "examples/ascend950pr_ops/mla/kernels.py",
        "mla",
    ),
    "swi_glu": (
        "level1/swi_glu",
        "examples/ascend950pr_ops/swi_glu/kernels.py",
        "swi_glu",
    ),
    "rms_norm": (
        "level2/rms_norm",
        "examples/ascend950pr_ops/rms_norm/kernels.py",
        "rms_norm",
    ),
    "softmax": (
        "level2/softmax",
        "examples/ascend950pr_ops/softmax/kernels.py",
        "softmax",
    ),
    "add_rms_norm_dynamic_quant": (
        "level3/add_rms_norm_dynamic_quant",
        "examples/ascend950pr_ops/add_rms_norm_dynamic_quant/kernels.py",
        "add_rms_norm_dynamic_quant",
    ),
    "mhc_sinkhorn": (
        "level3/mhc_sinkhorn",
        "examples/ascend950pr_ops/mhc_sinkhorn/kernels.py",
        "mhc_sinkhorn",
    ),
    "gqa": (
        "level4/gqa",
        "examples/ascend950pr_ops/gqa/kernels.py",
        "gqa",
    ),
    "kimi_delta_attention": (
        "level5/kimi_delta_attention",
        "examples/ascend950pr_ops/kimi_delta_attention/kernels.py",
        "kimi_delta_attention",
    ),
    "tensor_program_vm": (
        "level5/tensor_program_vm",
        "examples/ascend950pr_ops/tensor_program_vm/kernels.py",
        "tensor_program_vm",
    ),
}

ATTENTION_RESIDUAL_ANCHORS: tuple[AnchorCaseSpec, ...] = (
    AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
    AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.93, 3.3099),
    AnchorCaseSpec("level4/attention_residual", 14, "unaligned_d", 67.033, 22.3442),
    AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 1098.175, 366.0583),
    AnchorCaseSpec("level4/attention_residual", 17, "aligned_prefill", 1037.167, 345.7223),
)

MLA_ANCHORS: tuple[AnchorCaseSpec, ...] = (
    AnchorCaseSpec("level4/mla", 1, "prefill_compute", 34.8532, 11.6177),
    AnchorCaseSpec("level4/mla", 6, "prefill_compute", 30.7423, 10.2474),
    AnchorCaseSpec("level4/mla", 7, "split_kv_decode", 4.5577, 1.5192),
    AnchorCaseSpec("level4/mla", 11, "split_kv_decode", 68.6341, 22.8780),
    AnchorCaseSpec("level4/mla", 13, "bnsd_gather", 34.8532, 11.6177),
)


def _hydrate_anchors_from_metadata(
    anchors: Sequence[AnchorCaseSpec],
    meta_950pr: Mapping[int, tuple[float, float]],
) -> tuple[AnchorCaseSpec, ...]:
    """Hydrate reference and hardware-limit timings from live 950pr metadata."""
    hydrated: list[AnchorCaseSpec] = []
    for anchor in anchors:
        entry = meta_950pr.get(int(anchor.case_num))
        if entry is not None:
            ref_val = float(entry[0])
            hw_val = float(entry[1])
            if hw_val > 0.0:
                hydrated.append(
                    AnchorCaseSpec(
                        task=anchor.task,
                        case_num=anchor.case_num,
                        regime_tag=anchor.regime_tag,
                        ref_perf_us=ref_val if ref_val > 0.0 else anchor.ref_perf_us,
                        t_hw_us=hw_val,
                    )
                )
                continue
        hydrated.append(anchor)
    return tuple(hydrated)

DEFAULT_ANCHORS: dict[str, tuple[AnchorCaseSpec, ...]] = {
    "attention_residual": ATTENTION_RESIDUAL_ANCHORS,
    "mla": MLA_ANCHORS,
}

DEFAULT_PERF_TARGETS: dict[str, PerfTargetSpec] = {
    "attention_residual": PerfTargetSpec(
        operator_name="attention_residual",
        target_avg_speedup=2.85,
        target_min_case_speedup=1.60,
        target_roofline_eta=0.90,
        target_surrogate_score=97.5,
        min_optimization_rounds=3,
        per_anchor_max_us={},
    ),
    "mla": PerfTargetSpec(
        operator_name="mla",
        target_avg_speedup=2.50,
        target_min_case_speedup=1.35,
        target_roofline_eta=0.85,
        target_surrogate_score=92.0,
        min_optimization_rounds=3,
        per_anchor_max_us={7: 8.0, 13: 95.0},
    ),
}

DEFAULT_REGIME_SPECS: dict[str, tuple[RegimeTargetSpec, ...]] = {
    "attention_residual": (
        RegimeTargetSpec(
            regime_tag="small_decode",
            builder_candidates=(
                "_build_ar_small_decode_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 2,
                "d_dim": 4096,
                "tl_dtype": "bfloat16",
                "tile_m": 1,
                "schedule_mode": "tuned",
            },
            description="Small-row decode regime (m_rows < 28) needing depth/width core sharding.",
            representative_hir_dims={"N": 2, "B": 1, "S": 1, "D": 4096},
            representative_anchor_dims={"N": 2, "B": 1, "S": 1, "D": 4096},
            config_selector_fn="_select_ar_small_decode_config",
            editable_symbols=(
                "_select_ar_small_decode_config",
                "_build_ar_small_decode_func",
            ),
            allowed_helper_prefix="_ar_small_decode_",
        ),
        RegimeTargetSpec(
            regime_tag="unaligned_d",
            builder_candidates=(
                "_build_ar_unaligned_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 8,
                "d_dim": 1000,
                "tl_dtype": "bfloat16",
                "tile_m": 1,
                "schedule_mode": "tuned",
            },
            description=(
                "Non-64-aligned D tail regime enabling padded UB row pitch and multi-row "
                "buffering."
            ),
            representative_hir_dims={"N": 8, "B": 1, "S": 4, "D": 1024},
            representative_anchor_dims={"N": 8, "B": 1, "S": 16, "D": 1000},
            config_selector_fn="_select_ar_unaligned_config",
            editable_symbols=(
                "_select_ar_unaligned_config",
                "_build_ar_unaligned_func",
            ),
            allowed_helper_prefix="_ar_unaligned_",
        ),
        RegimeTargetSpec(
            regime_tag="aligned_prefill",
            builder_candidates=(
                "_build_ar_aligned_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 8,
                "d_dim": 4096,
                "tl_dtype": "bfloat16",
                "tile_m": 6,
                "schedule_mode": "tuned",
            },
            description="Aligned prefill regime with 56-core multi-row SimdVF pipelining.",
            representative_hir_dims={"N": 8, "B": 1, "S": 6, "D": 512},
            representative_anchor_dims={"N": 8, "B": 1, "S": 4096, "D": 4096},
            config_selector_fn="_select_ar_aligned_config",
            editable_symbols=(
                "_select_ar_aligned_config",
                "_build_ar_aligned_func",
            ),
            allowed_helper_prefix="_ar_aligned_",
        ),
    ),
    "mla": (
        RegimeTargetSpec(
            regime_tag="split_kv_decode",
            builder_candidates=(
                "_build_mla_split_kv_decode_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "b_dim": 1,
                "s_dim": 1,
                "s_kv": 2048,
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BSND",
                "is_causal": False,
                "bm": 64,
                "aligned_group_q": True,
                "schedule_mode": "tuned",
            },
            description=(
                "Decode and small-Sq regime targeting Split-KV FlashDecoding across 28 MixCores."
            ),
            representative_hir_dims={
                "B": 1,
                "S": 1,
                "S_kv": 64,
                "N_q": 16,
                "N_kv": 1,
                "D_nope": 512,
                "D_rope": 64,
            },
            representative_anchor_dims={
                "B": 1,
                "S": 1,
                "S_kv": 2048,
                "N_q": 128,
                "N_kv": 1,
                "D_nope": 512,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_decode_config",
            editable_symbols=(
                "_select_mla_decode_config",
                "_build_mla_split_kv_decode_func",
                "_build_mla_split_kv_combine_func",
                "_MLA_TRACKB_SRC",
            ),
            allowed_helper_prefix="_mla_decode_",
            runtime_backend="hybrid_trackb",
            trackb_symbols=("_MLA_TRACKB_SRC",),
        ),
        RegimeTargetSpec(
            regime_tag="bnsd_gather",
            builder_candidates=(
                "_build_mla_bnsd_gather_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BNSD",
                "is_causal": True,
                "bm": 64,
                "aligned_group_q": False,
                "schedule_mode": "tuned",
            },
            description="Non-aligned BNSD Q staging regime replacing scalar row copies.",
            representative_hir_dims={
                "B": 1,
                "S": 2,
                "S_kv": 32,
                "N_q": 16,
                "N_kv": 1,
                "D_nope": 128,
                "D_rope": 64,
            },
            representative_anchor_dims={
                "B": 1,
                "S": 512,
                "S_kv": 512,
                "N_q": 128,
                "N_kv": 1,
                "D_nope": 512,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_bnsd_config",
            editable_symbols=(
                "_select_mla_bnsd_config",
                "_build_mla_bnsd_gather_func",
                "_MLA_TRACKB_SRC",
            ),
            allowed_helper_prefix="_mla_bnsd_",
            runtime_backend="hybrid_trackb",
            trackb_symbols=("_MLA_TRACKB_SRC",),
        ),
        RegimeTargetSpec(
            regime_tag="prefill_compute",
            builder_candidates=(
                "_build_mla_prefill_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BSND",
                "is_causal": True,
                "bm": 128,
                "aligned_group_q": True,
                "schedule_mode": "tuned",
            },
            description="Prefill and large-batch compute/pipeline overlap regime.",
            representative_hir_dims={
                "B": 1,
                "S": 8,
                "S_kv": 32,
                "N_q": 8,
                "N_kv": 1,
                "D_nope": 128,
                "D_rope": 64,
            },
            representative_anchor_dims={
                "B": 1,
                "S": 512,
                "S_kv": 512,
                "N_q": 128,
                "N_kv": 1,
                "D_nope": 512,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_prefill_config",
            editable_symbols=(
                "_select_mla_prefill_config",
                "_build_mla_prefill_func",
                "_MLA_TRACKB_SRC",
            ),
            allowed_helper_prefix="_mla_prefill_",
            runtime_backend="hybrid_trackb",
            trackb_symbols=("_MLA_TRACKB_SRC",),
        ),
    ),
}


def _normalize_shape_list(raw_shapes: object) -> list[list[int]]:
    shapes: list[list[int]] = []
    if not isinstance(raw_shapes, (list, tuple)):
        return shapes
    if raw_shapes and all(isinstance(x, int) and not isinstance(x, bool) for x in raw_shapes):
        return [[int(x) for x in raw_shapes]]
    for item in raw_shapes:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            if item and all(isinstance(x, int) and not isinstance(x, bool) for x in item):
                shapes.append([int(x) for x in item])
            elif item and isinstance(item[0], (list, tuple)):
                for sub in item:
                    if isinstance(sub, (list, tuple)) and all(
                        isinstance(x, int) and not isinstance(x, bool) for x in sub
                    ):
                        shapes.append([int(x) for x in sub])
    return shapes


def _is_prime_gt2(n: int) -> bool:
    if n < 3 or n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _analyze_case_geometry(case_entry: Mapping[str, object], ref_us: float, t_hw_us: float) -> dict[str, object]:
    shapes = _normalize_shape_list(case_entry.get("input_shape"))
    attrs = case_entry.get("attrs") if isinstance(case_entry.get("attrs"), Mapping) else {}
    total_elems = 0
    all_dims: list[int] = []
    trailing_dims: list[int] = []
    max_rank = 0
    for sh in shapes:
        max_rank = max(max_rank, len(sh))
        prod = 1
        for d in sh:
            if d > 0:
                prod *= d
                all_dims.append(d)
        total_elems += prod
        if sh:
            trailing_dims.append(sh[-1])

    odd_or_prime_tail = any(
        d > 1 and (d % 2 == 1 or _is_prime_gt2(d))
        for d in trailing_dims
    )
    unaligned_16_tail = any(d > 1 and d % 16 != 0 for d in trailing_dims)
    unaligned_any_dim = any(d > 1 and d % 16 != 0 for d in all_dims)
    unaligned_64_tail = any(d > 1 and d % 64 != 0 for d in trailing_dims)
    is_unaligned = odd_or_prime_tail or unaligned_16_tail or unaligned_any_dim or unaligned_64_tail
    unaligned_score = (
        (4 if odd_or_prime_tail else 0)
        + (3 if unaligned_16_tail else 0)
        + (2 if unaligned_any_dim else 0)
        + (1 if unaligned_64_tail else 0)
    )

    aspect_ratio = 1.0
    if all_dims:
        pos_dims = [d for d in all_dims if d > 0]
        if pos_dims:
            aspect_ratio = float(max(pos_dims)) / float(max(1, min(pos_dims)))

    has_empty_group = False
    group_skew = 1.0
    group_list = attrs.get("group_list")
    if isinstance(group_list, (list, tuple)) and group_list:
        try:
            gvals = [int(x) for x in group_list]
            diffs: list[int] = []
            prev = 0
            for g in gvals:
                diffs.append(max(0, g - prev))
                prev = g
            has_empty_group = any(d == 0 for d in diffs)
            pos_diffs = [d for d in diffs if d > 0]
            if pos_diffs:
                group_skew = float(max(pos_diffs)) / float(min(pos_diffs))
        except (TypeError, ValueError):
            pass

    special_attr_score = 0
    if attrs.get("inplace") is True:
        special_attr_score += 2
    if attrs.get("is_causal") is True:
        special_attr_score += 1
    if str(attrs.get("inputLayout", "")).upper() == "BNSD":
        special_attr_score += 2
    vr_text = repr(case_entry.get("value_range", "")).lower()
    if "inf" in vr_text or "nan" in vr_text or "unique" in vr_text:
        special_attr_score += 1

    return {
        "case_id": int(case_entry.get("case_id", 0)),
        "shapes": shapes,
        "attrs": dict(attrs),
        "total_elems": max(1, total_elems),
        "max_rank": max_rank,
        "is_unaligned": is_unaligned,
        "unaligned_score": unaligned_score,
        "aspect_ratio": aspect_ratio,
        "has_empty_group": has_empty_group,
        "group_skew": group_skew,
        "special_attr_score": special_attr_score,
        "ref_perf_us": float(ref_us),
        "t_hw_us": float(t_hw_us),
    }


def _stratify_operator_anchors(
    task: str,
    cases: Sequence[Mapping[str, object]],
    meta_map: Mapping[int, tuple[float, float]],
) -> tuple[AnchorCaseSpec, ...]:
    """Stratify an operator's case list into 4-5 representative AnchorCaseSpecs."""
    if not cases:
        return ()
    metrics = []
    for raw in cases:
        cid = int(raw.get("case_id", 0))
        ref_us, hw_us = meta_map.get(cid, (0.0, 0.0))
        metrics.append(_analyze_case_geometry(raw, ref_us, hw_us))

    picked: list[tuple[dict[str, object], str]] = []
    picked_ids: set[int] = set()

    def _pick(cand_list: Sequence[dict[str, object]], tag: str) -> None:
        for item in cand_list:
            cid = int(item["case_id"])
            if cid not in picked_ids:
                picked_ids.add(cid)
                picked.append((item, tag))
                return

    by_small = sorted(
        metrics,
        key=lambda m: (
            m["total_elems"],
            m["t_hw_us"] if m["t_hw_us"] > 0.0 else float("inf"),
            m["case_id"],
        ),
    )
    _pick(by_small, "small_latency")

    unaligned_pool = [m for m in metrics if m["is_unaligned"]]
    if unaligned_pool:
        by_unaligned = sorted(
            unaligned_pool,
            key=lambda m: (
                -int(m["unaligned_score"]),
                m["total_elems"],
                m["case_id"],
            ),
        )
        _pick(by_unaligned, "unaligned_tail")
    else:
        _pick(by_small, "unaligned_tail")

    by_large = sorted(
        metrics,
        key=lambda m: (
            -int(m["total_elems"]),
            -float(m["t_hw_us"]),
            m["case_id"],
        ),
    )
    _pick(by_large, "large_throughput")

    by_extreme = sorted(
        metrics,
        key=lambda m: (
            -(1 if m["has_empty_group"] else 0),
            -float(m["group_skew"]),
            -float(m["aspect_ratio"]),
            -int(m["special_attr_score"]),
            -int(m["total_elems"]),
            m["case_id"],
        ),
    )
    _pick(by_extreme, "extreme_aspect")

    if len(picked) < 5 and len(metrics) >= 5:
        by_mid = sorted(
            metrics,
            key=lambda m: (
                float(m["t_hw_us"]) if m["t_hw_us"] > 0.0 else float(m["total_elems"]),
                m["total_elems"],
                m["case_id"],
            ),
        )
        mid_idx = len(by_mid) // 2
        ordered_mid = [by_mid[mid_idx], *by_mid[mid_idx + 1 :], *reversed(by_mid[:mid_idx])]
        _pick(ordered_mid, "mid_balanced")

    anchors = [
        AnchorCaseSpec(
            task=task,
            case_num=int(item["case_id"]),
            regime_tag=tag,
            ref_perf_us=float(item["ref_perf_us"]),
            t_hw_us=float(item["t_hw_us"]),
        )
        for item, tag in picked
    ]
    anchors.sort(key=lambda a: a.case_num)
    return tuple(anchors)


@dataclass(frozen=True)
class CannbenchOpSpec:
    """Discovered operator specification from cann-bench/tasks."""

    op_id: str
    task: str
    level: str
    level_num: int
    op_dir: Path
    proto: dict[str, object]
    cases: tuple[dict[str, object], ...]
    metadata_950pr: dict[int, tuple[float, float]]
    anchors: tuple[AnchorCaseSpec, ...]
    perf_target: PerfTargetSpec
    module_relpath: str
    entrypoint: str

    @property
    def golden_path(self) -> Path:
        """Return the PyTorch golden reference path inside the task directory."""
        return self.op_dir / "golden.py"


_TRACKB_CPP_MARKERS: tuple[str, ...] = (
    "tl_templates/ascend/",
    "__global__",
    "__simt_vf__",
    "SimdVF",
    "kernel_entry",
    "AscendC::",
)


def discover_module_trackb_symbols(module_text: str) -> tuple[str, ...]:
    """Discover top-level Track-B C++20 kernel string constants and plan selectors."""
    has_include = bool(module_text and "#include" in module_text)
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return ()

    cpp_symbols: list[str] = []
    plan_symbols: list[str] = []
    for node in tree.body:
        name: str | None = None
        val: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, val = node.target.id, node.value
        if (
            has_include
            and name
            and isinstance(val, ast.Constant)
            and isinstance(val.value, str)
            and "#include" in val.value
            and any(marker in val.value for marker in _TRACKB_CPP_MARKERS)
        ):
            cpp_symbols.append(name)
        elif isinstance(node, ast.FunctionDef):
            low = node.name.lower()
            if (
                (low.startswith("_select_trackb_") or low.startswith("_select_"))
                and ("trackb" in low or "track_b" in low)
                and low.endswith("_plan")
            ):
                plan_symbols.append(node.name)

    if not cpp_symbols:
        return ()
    return tuple(dict.fromkeys([*cpp_symbols, *plan_symbols]))


def extract_trackb_cpp_source(
    module_source: str,
    trackb_symbols: Sequence[str],
) -> tuple[str, str] | None:
    """Return the first Track-B C++20 translation-unit constant among trackb_symbols."""
    if not trackb_symbols:
        return None
    tree = ast.parse(module_source)
    constants: dict[str, str] = {}
    for node in tree.body:
        name: str | None = None
        value: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        if name and isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants[name] = value.value
    for symbol in trackb_symbols:
        cpp_source = constants.get(symbol)
        if cpp_source and cpp_source.strip():
            return symbol, cpp_source
    return None


def _coerce_positive_int(val: object, default: int = 0) -> int:
    if isinstance(val, bool) or val is None:
        return default
    if isinstance(val, int):
        return int(val)
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return default


def infer_module_runtime_backend(
    module_text: str,
    entrypoint: str,
) -> tuple[str, tuple[str, ...]]:
    """Derive ('tilelang_dsl' | 'ascendc_trackb' | 'hybrid_trackb', symbols) from AST graph."""
    trackb_syms = discover_module_trackb_symbols(module_text)
    if not module_text.strip():
        return "tilelang_dsl", ()
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return ("ascendc_trackb" if trackb_syms else "tilelang_dsl"), trackb_syms
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entrypoint not in funcs:
        return ("ascendc_trackb" if trackb_syms else "tilelang_dsl"), trackb_syms

    reachable: set[str] = set()
    queue = [entrypoint]
    while queue:
        cur = queue.pop(0)
        if cur in reachable or cur not in funcs:
            continue
        reachable.add(cur)
        for sub in ast.walk(funcs[cur]):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in funcs and sub.func.id not in reachable:
                    queue.append(sub.func.id)

    reaches_tl = False
    reaches_tb = False
    for fn_name in reachable:
        seg = ast.get_source_segment(module_text, funcs[fn_name]) or ""
        if "tilelang.compile(" in seg:
            reaches_tl = True
        if any(sym in seg for sym in trackb_syms) or "_get_track_b_lib" in seg:
            reaches_tb = True

    if reaches_tl and reaches_tb:
        return "hybrid_trackb", trackb_syms
    if reaches_tb or trackb_syms:
        return "ascendc_trackb", trackb_syms
    return "tilelang_dsl", ()


@dataclass(frozen=True, kw_only=True)
class _SemanticCasePartition:
    """PyTorch-derived operator semantics and per-case physical ratios for one operator."""

    base_axes: TopologyAxes
    has_reduction: bool
    ratios: tuple[CasePhysicsRatios, ...]


class CannbenchOpsRegistry:
    """Dynamic registry discovering all 60+ operators in cann-bench/tasks."""

    _DEFAULT_INSTANCE: CannbenchOpsRegistry | None = None

    def __init__(
        self,
        tasks_root: str | Path | None = None,
        *,
        source_root: str | Path | None = None,
    ) -> None:
        self._source_root = (
            Path(source_root).resolve() if source_root is not None else default_source_root()
        )
        self._tasks_root = self._resolve_tasks_root(tasks_root)
        self._meta_950pr: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        meta_path = self._tasks_root / "metadata" / "950pr.json"
        if meta_path.is_file():
            self._meta_950pr = json.loads(meta_path.read_text(encoding="utf-8"))
        self._entries: dict[str, dict[str, object]] = {}
        self._task_to_op: dict[str, str] = {}
        self._cache: dict[str, CannbenchOpSpec] = {}
        self._semantic_partitions: dict[str, _SemanticCasePartition] = {}
        self._semantic_groups: dict[str, tuple[SemanticRegimeGroup, ...]] = {}
        self._discover_manifest()

    @classmethod
    def default(cls) -> CannbenchOpsRegistry:
        """Return cached default CannbenchOpsRegistry instance."""
        if cls._DEFAULT_INSTANCE is None:
            cls._DEFAULT_INSTANCE = cls()
        return cls._DEFAULT_INSTANCE

    def _resolve_tasks_root(self, tasks_root: str | Path | None) -> Path:
        if tasks_root is not None:
            p = Path(tasks_root).expanduser().resolve()
            if _has_flat_operator_layout(p):
                return p
            return resolve_cannbench_tasks_root(p)

        candidates: list[Path] = []
        env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
        if env_repo:
            env_path = Path(env_repo).expanduser()
            candidates.extend((env_path / "tasks", env_path))
        candidates.extend(
            (
                self._source_root.parent / "cann-bench" / "tasks",
                Path.home() / "workspace" / "cann-bench" / "tasks",
            )
        )
        for cand in candidates:
            if cand.is_dir() and _has_task_layout(cand):
                return cand.resolve()
        for cand in candidates:
            if cand.is_dir():
                return cand.resolve()
        raise FileNotFoundError(
            "Cannot locate cann-bench/tasks directory. "
            "Pass tasks_root or set TILEFOUNDRY_NPU_CANNBENCH_REPO."
        )

    def _discover_manifest(self) -> None:
        manifest_path = self._tasks_root / "suite_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest.get("operators", ()):
                op_id = str(row["id"])
                raw_path = str(row["path"]).removeprefix("tasks/").strip("/")
                level_dir = raw_path.split("/", 1)[0]
                op_dir = self._tasks_root / raw_path
                if op_dir.is_dir():
                    level_num = int(level_dir.removeprefix("level")) if level_dir.startswith("level") else 1
                    self._entries[op_id] = {
                        "op_id": op_id,
                        "task": raw_path,
                        "level": str(row.get("level", f"L{level_num}")),
                        "level_dir": level_dir,
                        "level_num": level_num,
                        "op_dir": op_dir,
                    }
                    self._task_to_op[raw_path] = op_id
            return

        for level_dir_path in sorted(self._tasks_root.glob("level*")):
            if not level_dir_path.is_dir():
                continue
            level_dir = level_dir_path.name
            level_num = int(level_dir.removeprefix("level")) if level_dir.startswith("level") else 1
            for op_dir in sorted(level_dir_path.iterdir()):
                if not op_dir.is_dir() or not (op_dir / "cases.yaml").is_file():
                    continue
                op_id = op_dir.name
                task = f"{level_dir}/{op_id}"
                self._entries[op_id] = {
                    "op_id": op_id,
                    "task": task,
                    "level": f"L{level_num}",
                    "level_dir": level_dir,
                    "level_num": level_num,
                    "op_dir": op_dir,
                }
                self._task_to_op[task] = op_id

        if self._entries or not _has_flat_operator_layout(self._tasks_root):
            return
        for op_dir in sorted(self._tasks_root.iterdir()):
            if not op_dir.is_dir() or not (op_dir / "cases.yaml").is_file():
                continue
            op_id = op_dir.name
            if op_id not in OPERATOR_METADATA:
                raise ValueError(
                    f"flat benchmark root {self._tasks_root} contains operator {op_id!r} "
                    "without an official task mapping in OPERATOR_METADATA"
                )
            task, _module_relpath, _entry = OPERATOR_METADATA[op_id]
            level_dir, _op_name = task.split("/", 1)
            level_num = int(level_dir.removeprefix("level"))
            self._entries[op_id] = {
                "op_id": op_id,
                "task": task,
                "level": f"L{level_num}",
                "level_dir": level_dir,
                "level_num": level_num,
                "op_dir": op_dir,
            }
            self._task_to_op[task] = op_id

    @property
    def tasks_root(self) -> Path:
        """Return resolved cann-bench/tasks root path."""
        return self._tasks_root

    def _normalize_key(self, name_or_task: str) -> str:
        raw = str(name_or_task).strip().removeprefix("tasks/")
        if raw in self._entries:
            return raw
        if raw in self._task_to_op:
            return self._task_to_op[raw]
        if "/" in raw:
            tail = raw.rsplit("/", 1)[-1]
            if tail in self._entries:
                return tail
        raise KeyError(f"unsupported campaign operator {name_or_task!r}")

    def has_operator(self, name_or_task: str) -> bool:
        """Return True if operator short name or task id exists in registry."""
        try:
            self._normalize_key(name_or_task)
            return True
        except KeyError:
            return False

    def list_operators(self) -> tuple[str, ...]:
        """Return sorted short operator names discovered in registry."""
        return tuple(sorted(self._entries))

    def list_tasks(self) -> tuple[str, ...]:
        """Return sorted levelX/op task identifiers discovered in registry."""
        return tuple(sorted(self._task_to_op))

    def resolve_module_path(
        self,
        name_or_task: str,
        *,
        source_root: str | Path | None = None,
    ) -> str:
        """Resolve workspace-relative candidate module path (kernels.py or solution.py)."""
        op_id = self._normalize_key(name_or_task)
        if op_id in OPERATOR_METADATA:
            return OPERATOR_METADATA[op_id][1]
        root = Path(source_root).resolve() if source_root is not None else self._source_root
        for fname in ("kernels.py", "solution.py"):
            rel = f"examples/ascend950pr_ops/{op_id}/{fname}"
            if (root / rel).is_file():
                return rel
        return f"examples/ascend950pr_ops/{op_id}/kernels.py"

    def resolve(self, name_or_task: str) -> CannbenchOpSpec:
        """Load and cache full CannbenchOpSpec (proto, cases, 950pr metadata, anchors, target)."""
        op_id = self._normalize_key(name_or_task)
        if op_id in self._cache:
            return self._cache[op_id]

        meta_entry = self._entries[op_id]
        op_dir = Path(meta_entry["op_dir"])  # type: ignore[arg-type]
        task = str(meta_entry["task"])
        level_dir = str(meta_entry["level_dir"])
        level = str(meta_entry["level"])
        level_num = int(meta_entry["level_num"])

        proto_path = op_dir / "proto.yaml"
        proto_raw = yaml.safe_load(proto_path.read_text(encoding="utf-8")) if proto_path.is_file() else {}
        proto = dict((proto_raw or {}).get("operator") or {})

        cases_path = op_dir / "cases.yaml"
        cases_raw = yaml.safe_load(cases_path.read_text(encoding="utf-8")) if cases_path.is_file() else {}
        cases = tuple(
            dict(row) for row in ((cases_raw or {}).get("cases") or ()) if isinstance(row, Mapping)
        )

        raw_meta = self._meta_950pr.get(level_dir, {}).get(op_id, {})
        meta_950pr: dict[int, tuple[float, float]] = {}
        for k, val in raw_meta.items():
            if isinstance(val, Mapping):
                meta_950pr[int(k)] = (
                    float(val.get("baseline_perf_us", 0.0)),
                    float(val.get("t_hw_us", 0.0)),
                )

        if op_id in DEFAULT_ANCHORS:
            anchors = _hydrate_anchors_from_metadata(DEFAULT_ANCHORS[op_id], meta_950pr)
        else:
            anchors = _stratify_operator_anchors(task, cases, meta_950pr)

        if op_id in DEFAULT_PERF_TARGETS:
            perf_target = DEFAULT_PERF_TARGETS[op_id]
        else:
            speedups = [
                a.ref_perf_us / a.t_hw_us
                for a in anchors
                if a.ref_perf_us > 0.0 and a.t_hw_us > 0.0
            ]
            avg_spd = max(1.05, min(3.0, (sum(speedups) / len(speedups)) * 0.90)) if speedups else 2.0
            min_spd = max(1.0, min(avg_spd, min(speedups) * 0.85)) if speedups else 1.25
            per_max = {
                int(a.case_num): round(max(a.t_hw_us * 1.15, a.t_hw_us), 4)
                for a in anchors
                if a.t_hw_us > 0.0
            }
            perf_target = PerfTargetSpec(
                operator_name=op_id,
                target_avg_speedup=round(avg_spd, 4),
                target_min_case_speedup=round(min_spd, 4),
                target_roofline_eta=0.90,
                per_anchor_max_us=per_max,
                target_surrogate_score=95.0,
                min_optimization_rounds=3,
            )

        spec = CannbenchOpSpec(
            op_id=op_id,
            task=task,
            level=level,
            level_num=level_num,
            op_dir=op_dir,
            proto=proto,
            cases=cases,
            metadata_950pr=meta_950pr,
            anchors=anchors,
            perf_target=perf_target,
            module_relpath=self.resolve_module_path(op_id),
            entrypoint=op_id,
        )
        self._cache[op_id] = spec
        return spec

    def _module_path(self, spec: CannbenchOpSpec) -> Path:
        return self._source_root / spec.module_relpath

    def _module_source(self, spec: CannbenchOpSpec) -> str:
        mod_file = self._module_path(spec)
        return mod_file.read_text(encoding="utf-8") if mod_file.is_file() else ""

    def _semantic_partition(self, spec: CannbenchOpSpec) -> _SemanticCasePartition:
        cached = self._semantic_partitions.get(spec.op_id)
        if cached is not None:
            return cached
        if not spec.golden_path.is_file():
            raise FileNotFoundError(
                f"operator {spec.op_id!r} has no PyTorch golden reference at "
                f"{spec.golden_path} for semantic regime partitioning"
            )
        analyzer = PyTorchSemanticAnalyzer()
        semantics = analyzer.analyze_source(
            spec.golden_path.read_text(encoding="utf-8"),
            entry_name=spec.entrypoint,
            proto=spec.proto,
        )
        ratios = tuple(analyzer.compute_case_ratios(semantics, case) for case in spec.cases)
        partition = _SemanticCasePartition(
            base_axes=semantics.base_axes,
            has_reduction=semantics.has_reduction,
            ratios=ratios,
        )
        self._semantic_partitions[spec.op_id] = partition
        return partition

    def _semantic_regime_groups(
        self,
        spec: CannbenchOpSpec,
    ) -> tuple[SemanticRegimeGroup, ...]:
        cached = self._semantic_groups.get(spec.op_id)
        if cached is not None:
            return cached
        partition = self._semantic_partition(spec)
        groups = build_semantic_regime_groups(
            operator_name=spec.op_id,
            cases=spec.cases,
            proto=spec.proto,
            base_axes=partition.base_axes,
            has_reduction=partition.has_reduction,
            case_ratios=partition.ratios,
        )
        self._semantic_groups[spec.op_id] = groups
        return groups

    def get_anchors(self, name_or_task: str) -> tuple[AnchorCaseSpec, ...]:
        """Return anchor cases tagged with this operator's campaign regimes.

        Authored incumbents keep their curated dispatch tags so takeover selectors and anchor
        routing stay on the module's real ABI. Every other operator, including cold starts, uses
        WP1's case-to-regime mapping and gains one representative anchor for any regime that the
        legacy anchor stratification did not cover.
        """
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        if op_id in DEFAULT_REGIME_SPECS and self._module_path(spec).is_file():
            return spec.anchors
        groups = self._semantic_regime_groups(spec)
        tag_by_case = {
            case_id: group.regime_tag
            for group in groups
            for case_id, _, _ in group.entries
        }
        retagged = tuple(
            replace(anchor, regime_tag=tag_by_case[anchor.case_num])
            for anchor in spec.anchors
            if anchor.case_num in tag_by_case
        )
        covered = {anchor.regime_tag for anchor in retagged}
        additions: list[AnchorCaseSpec] = []
        for group in groups:
            if group.regime_tag in covered:
                continue
            case_id, _, _ = semantic_regime_representative(group)
            ref_perf_us, t_hw_us = spec.metadata_950pr.get(case_id, (0.0, 0.0))
            additions.append(
                AnchorCaseSpec(
                    task=spec.task,
                    case_num=case_id,
                    regime_tag=group.regime_tag,
                    ref_perf_us=ref_perf_us,
                    t_hw_us=t_hw_us,
                )
            )
        return tuple(sorted((*retagged, *additions), key=lambda anchor: anchor.case_num))

    def get_perf_target(self, name_or_task: str) -> PerfTargetSpec:
        """Return PerfTargetSpec targeting t_hw_us and HAP >= 95 for any operator."""
        return self.resolve(name_or_task).perf_target

    def get_case_regime_tag(self, name_or_task: str, case_num: int) -> str | None:
        """Return the campaign regime tag for one official case.

        Authored incumbents keep their curated dispatch partitions, where routing depends on
        layout and alignment rather than WP1's current phase tags, so they return ``None`` and
        callers retain the authored selector routing.
        """
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        if op_id in DEFAULT_REGIME_SPECS and self._module_path(spec).is_file():
            return None
        groups = self._semantic_regime_groups(spec)
        target_case = int(case_num)
        for group in groups:
            if any(entry_case == target_case for entry_case, _, _ in group.entries):
                return group.regime_tag
        raise KeyError(
            f"case_num={target_case} missing from semantic regime partition of {op_id!r}"
        )

    def _hydrate_authored_regime_specs(
        self,
        spec: CannbenchOpSpec,
        regimes: Sequence[RegimeTargetSpec],
    ) -> tuple[RegimeTargetSpec, ...]:
        """Attach the max-workload anchor case and its physics ratios to authored regimes."""
        partition = self._semantic_partition(spec)
        ratios_by_case = {ratios.case_id: ratios for ratios in partition.ratios}
        raw_case_by_case_id = {int(case["case_id"]): dict(case) for case in spec.cases}
        anchor_case_ids_by_tag: dict[str, list[int]] = {}
        for anchor in spec.anchors:
            anchor_case_ids_by_tag.setdefault(anchor.regime_tag, []).append(anchor.case_num)
        hydrated: list[RegimeTargetSpec] = []
        for regime in regimes:
            member_case_ids = [
                case_id
                for case_id in anchor_case_ids_by_tag.get(regime.regime_tag, ())
                if case_id in raw_case_by_case_id
            ]
            if not member_case_ids:
                raise ValueError(
                    f"authored regime {regime.regime_tag!r} of {spec.op_id!r} matches no "
                    "official anchor case"
                )
            workload_bytes_by_case = {
                case_id: case_workload_bytes(
                    flatten_int_shapes(raw_case_by_case_id[case_id].get("input_shape")),
                    flatten_str_values(raw_case_by_case_id[case_id].get("dtype")),
                )
                for case_id in member_case_ids
            }
            representative_case_id = max(
                member_case_ids,
                key=lambda case_id: (workload_bytes_by_case[case_id], case_id),
            )
            hydrated.append(
                replace(
                    regime,
                    representative_case_entry=dict(
                        raw_case_by_case_id[representative_case_id]
                    ),
                    representative_case_ratios=ratios_by_case[representative_case_id],
                )
            )
        return tuple(hydrated)

    def get_regime_specs(self, name_or_task: str) -> tuple[RegimeTargetSpec, ...]:
        """Return RegimeTargetSpecs partitioned by WP1's physical case mapping.

        Authored incumbent partitions remain authoritative while their module is present because
        their public entrypoints dispatch through per-regime selectors that WP1 phase tags do not
        reproduce (MLA layout branches and attention-residual D-alignment branches). Each authored
        partition receives the max-workload anchor case of its regime and that case's physics
        ratios so Stage-A, predicate context, and Stage-0 instantiation share real case data.
        Cold starts for those operators, and all other operators, derive 2-4 regimes from golden
        semantics.
        """
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        mod_file = self._module_path(spec)
        mod_text = self._module_source(spec)
        if op_id in DEFAULT_REGIME_SPECS and mod_file.is_file():
            return self._hydrate_authored_regime_specs(spec, DEFAULT_REGIME_SPECS[op_id])
        partition = self._semantic_partition(spec)
        runtime_backend, discovered_tb = infer_module_runtime_backend(
            mod_text,
            spec.entrypoint,
        )
        regimes = build_semantic_regime_specs(
            operator_name=op_id,
            entrypoint=spec.entrypoint,
            cases=spec.cases,
            proto=spec.proto,
            base_axes=partition.base_axes,
            has_reduction=partition.has_reduction,
            case_ratios=partition.ratios,
            runtime_backend=runtime_backend,
            trackb_symbols=discovered_tb,
        )
        if mod_text and f"def _select_{op_id}_config(" not in mod_text:
            regimes = tuple(
                replace(
                    regime,
                    config_selector_fn=f"_select_{op_id}_{regime.regime_tag}_config",
                )
                if f"def _select_{op_id}_{regime.regime_tag}_config(" in mod_text
                else regime
                for regime in regimes
            )
        return regimes


_ENGINE_PHASE_TAGS: dict[EnginePhase, str] = {
    "aiv_vector_or_gemv": "vec",
    "aic_cube": "cube",
    "coupled_1c2v": "cv",
}
_LAYOUT_PHASE_TAGS: dict[LayoutPhase, str] = {
    "contiguous": "contig",
    "strided_or_gather": "strided",
}
_RESIDENCY_BAND_TAGS: dict[ResidencyBand, str] = {
    "vreg_resident": "vreg",
    "sram_1pass": "sram",
    "gm_multipass": "gmpass",
}
_ALIGNMENT_BAND_TAGS: dict[AlignmentBand, str] = {
    "sub_atom_simt": "subatom",
    "aligned_vector": "aligned",
    "unaligned_predicated": "unaligned",
}
_WAVE_BAND_TAGS: dict[WaveBand, str] = {
    "ragged_groups": "ragged",
    "under_occupied_split": "underfill",
    "balanced_waves": "filled",
}

PhaseSignature = tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]


def semantic_regime_tag(signature: PhaseSignature) -> str:
    """Return the stable regime tag for one 5D physical phase coordinate."""
    engine, layout, residency, alignment, wave = signature
    return "_".join(
        (
            _ENGINE_PHASE_TAGS[engine],
            _LAYOUT_PHASE_TAGS[layout],
            _RESIDENCY_BAND_TAGS[residency],
            _ALIGNMENT_BAND_TAGS[alignment],
            _WAVE_BAND_TAGS[wave],
        )
    )


_CAMPAIGN_REGIME_PARTITION_CAP = 4


def campaign_case_regime_tag(
    ratios: CasePhysicsRatios,
    raw_case: Mapping[str, object],
    *,
    base_axes: TopologyAxes,
    has_reduction: bool,
) -> str:
    """Map one official case onto WP1's shared physical regime tag.

    SIL advisories and campaign partitions must classify the same case identically, so both
    call ``classify_case_regime`` with structurally recovered dims from the same helper.
    """
    raw_attrs = raw_case.get("attrs")
    attrs = raw_attrs if isinstance(raw_attrs, Mapping) else {}
    dims = infer_case_dims_structural(
        flatten_int_shapes(raw_case.get("input_shape")),
        attrs,
        axes=base_axes,
        uses_cube=base_axes.uses_cube,
    )
    return classify_case_regime(
        ratios,
        kind=infer_case_physics_kind(base_axes, has_reduction=has_reduction),
        axes=base_axes,
        dims=dims,
    )


@dataclass(frozen=True, kw_only=True)
class SemanticRegimeGroup:
    """One campaign regime with the official cases and raw coordinates it merges."""

    regime_tag: str
    entries: tuple[tuple[int, dict[str, object], PhaseSignature], ...]
    representative_case_id: int
    description: str
    representative_case_entry: dict[str, object]
    representative_case_ratios: CasePhysicsRatios


def _semantic_group_description(group: CasePhysicsRegimeGroup) -> str:
    rationale = f"{group.rationale.strip()} " if group.rationale.strip() else ""
    return (
        f"WP1 physical regime {group.regime_tag} covering official cases "
        f"{list(group.case_ids)} at 5D coordinates: "
        f"{'; '.join(group.phase_coordinates)}. {rationale}Representative case "
        f"{group.representative_case.case_id} supplies the Stage-0 dimensions."
    )


def build_semantic_regime_groups(
    *,
    operator_name: str,
    cases: Sequence[Mapping[str, object]],
    proto: Mapping[str, object],
    base_axes: TopologyAxes,
    has_reduction: bool,
    case_ratios: Sequence[CasePhysicsRatios],
) -> tuple[SemanticRegimeGroup, ...]:
    """Partition official cases through WP1's shared 2-4 physical regime mapping.

    ``partition_case_regimes`` owns every grouping decision: cases group by the same
    ``classify_case_regime`` physics tags SIL advisories use, wave-occupancy ``_split``
    variants fold into their base mechanism above the four-partition cap, and the
    two-partition floor splits on the first differing phase band or by dtype-aware
    workload scale. The raw 5D coordinates stay in each group description. Stage-0
    instantiation keeps the richer schema-driven case dimensions for the representative
    case selected by that same mapping.
    """
    if not cases:
        raise ValueError(
            f"operator {operator_name!r} has no official cases.yaml entries to partition"
        )
    if len(cases) != len(case_ratios):
        raise ValueError(
            f"operator {operator_name!r} has {len(cases)} cases but "
            f"{len(case_ratios)} computed CasePhysicsRatios"
        )
    raw_proto_inputs = proto.get("inputs")
    proto_inputs = (
        tuple(row for row in raw_proto_inputs if isinstance(row, Mapping))
        if isinstance(raw_proto_inputs, (list, tuple))
        else ()
    )
    ratios_by_case: dict[int, CasePhysicsRatios] = {}
    raw_case_by_case_id: dict[int, dict[str, object]] = {}
    wp1_dims_by_case: dict[int, dict[str, int]] = {}
    staged_dims_by_case: dict[int, dict[str, object]] = {}
    workload_bytes_by_case: dict[int, int] = {}
    for raw_case, ratios in zip(cases, case_ratios):
        raw_case_id = raw_case.get("case_id")
        if isinstance(raw_case_id, bool) or not isinstance(raw_case_id, int):
            raise ValueError(
                f"operator {operator_name!r} case entry requires a positive integer case_id"
            )
        if raw_case_id != ratios.case_id:
            raise ValueError(
                f"operator {operator_name!r} case order mismatch: CasePhysicsRatios carry "
                f"{ratios.case_id}, cases.yaml row carries {raw_case_id}"
            )
        if ratios.case_id in ratios_by_case:
            raise ValueError(f"operator {operator_name!r} repeats case id {ratios.case_id}")
        raw_attrs = raw_case.get("attrs")
        attrs = raw_attrs if isinstance(raw_attrs, Mapping) else {}
        shapes = flatten_int_shapes(raw_case.get("input_shape"))
        ratios_by_case[ratios.case_id] = ratios
        raw_case_by_case_id[ratios.case_id] = dict(raw_case)
        wp1_dims_by_case[ratios.case_id] = infer_case_dims_structural(
            shapes,
            attrs,
            axes=base_axes,
            uses_cube=base_axes.uses_cube,
        )
        staged_dims_by_case[ratios.case_id] = extract_schema_driven_case_dims(
            raw_case, proto_inputs=proto_inputs
        )
        workload_bytes_by_case[ratios.case_id] = case_workload_bytes(
            shapes, flatten_str_values(raw_case.get("dtype"))
        )
    groups = partition_case_regimes(
        ratios_by_case,
        wp1_dims_by_case,
        workload_bytes_by_case,
        kind=infer_case_physics_kind(base_axes, has_reduction=has_reduction),
        axes=base_axes,
        min_partitions=2,
        max_partitions=_CAMPAIGN_REGIME_PARTITION_CAP,
    )
    return tuple(
        SemanticRegimeGroup(
            regime_tag=group.regime_tag,
            entries=tuple(
                (
                    case.case_id,
                    dict(staged_dims_by_case[case.case_id]),
                    case.phase_signature,
                )
                for case in group.cases
            ),
            representative_case_id=group.representative_case.case_id,
            description=_semantic_group_description(group),
            representative_case_entry=raw_case_by_case_id[group.representative_case.case_id],
            representative_case_ratios=ratios_by_case[group.representative_case.case_id],
        )
        for group in groups
    )


def semantic_regime_representative(
    group: SemanticRegimeGroup,
) -> tuple[int, dict[str, object], PhaseSignature]:
    """Return WP1's representative case entry for one semantic regime group."""
    for entry in group.entries:
        if entry[0] == group.representative_case_id:
            return entry
    raise AssertionError(
        f"semantic regime {group.regime_tag!r} lacks representative case "
        f"{group.representative_case_id}"
    )


def build_semantic_regime_specs(
    *,
    operator_name: str,
    entrypoint: str,
    cases: Sequence[Mapping[str, object]],
    proto: Mapping[str, object],
    base_axes: TopologyAxes,
    has_reduction: bool,
    case_ratios: Sequence[CasePhysicsRatios],
    runtime_backend: str,
    trackb_symbols: Sequence[str] = (),
) -> tuple[RegimeTargetSpec, ...]:
    """Partition official cases into campaign regime specs through WP1's mapping.

    ``build_semantic_regime_groups`` owns the case-to-regime decisions, including the
    four-partition cap and the two-partition floor, so cold starts and takeover registries
    always see the same partition for the same operator semantics.
    """
    if runtime_backend not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
        raise ValueError(
            f"semantic regime backend {runtime_backend!r} must be 'tilelang_dsl', "
            "'ascendc_trackb', or 'hybrid_trackb'"
        )
    groups = build_semantic_regime_groups(
        operator_name=operator_name,
        cases=cases,
        proto=proto,
        base_axes=base_axes,
        has_reduction=has_reduction,
        case_ratios=case_ratios,
    )
    shared_selector = f"_select_{entrypoint}_config"
    shared_builder = f"_build_{entrypoint}_prim_func"
    regimes: list[RegimeTargetSpec] = []
    for group in groups:
        tag = group.regime_tag
        case_id, representative_dims, _ = semantic_regime_representative(group)
        int_dims = {
            str(key): int(val)
            for key, val in representative_dims.items()
            if isinstance(val, int) and not isinstance(val, bool)
        }
        if not int_dims:
            raise ValueError(
                f"semantic regime {tag!r} for {operator_name!r} case {case_id} has no "
                "concrete integer dimensions for Stage-0 instantiation"
            )
        per_regime_selector = f"_select_{entrypoint}_{tag}_config"
        per_regime_builder = f"_build_{entrypoint}_{tag}_func"
        regimes.append(
            RegimeTargetSpec(
                regime_tag=tag,
                builder_candidates=(per_regime_builder, shared_builder, "make_kernel"),
                sample_builder_kwargs=dict(representative_dims),
                description=group.description,
                representative_hir_dims=dict(int_dims),
                representative_anchor_dims=dict(int_dims),
                config_selector_fn=shared_selector,
                editable_symbols=tuple(
                    dict.fromkeys(
                        (
                            per_regime_selector,
                            shared_selector,
                            per_regime_builder,
                            shared_builder,
                            *trackb_symbols,
                        )
                    )
                ),
                allowed_helper_prefix=f"_{operator_name}_{tag}_",
                runtime_backend=runtime_backend,
                trackb_symbols=tuple(trackb_symbols),
                representative_case_entry=dict(group.representative_case_entry),
                representative_case_ratios=group.representative_case_ratios,
            )
        )
    return tuple(regimes)


def _parse_proto_shape_axes(shape_spec: object, layout: str) -> list[str]:
    if not isinstance(shape_spec, str) or "[" not in shape_spec:
        return []
    matches = re.findall(r"(?:([A-Za-z0-9_]+)\s*:\s*)?\[([^\]]+)\]", shape_spec)
    chosen = ""
    upper_layout = layout.strip().upper()
    for tag, inner in matches:
        if tag.strip().upper() == upper_layout and upper_layout:
            chosen = inner
            break
    if not chosen and matches:
        chosen = matches[0][1]
    if not chosen:
        return []
    return [tok.strip() for tok in chosen.split(",") if tok.strip()]


def extract_schema_driven_case_dims(
    case_entry: Mapping[str, object],
    proto_inputs: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Derive case dimension dictionary from proto parameter schemas and attrs."""
    dims: dict[str, object] = {}
    attrs = case_entry.get("attrs")
    if isinstance(attrs, Mapping):
        for k, v in attrs.items():
            dims[str(k)] = v

    raw_dtypes = case_entry.get("dtype")
    if isinstance(raw_dtypes, (list, tuple)):
        for dt in raw_dtypes:
            dt_s = str(dt).strip()
            if dt_s in ("float16", "bfloat16", "float32"):
                dims.setdefault("tl_dtype", dt_s)
                break
    elif isinstance(raw_dtypes, str) and raw_dtypes.strip() in (
        "float16",
        "bfloat16",
        "float32",
    ):
        dims.setdefault("tl_dtype", raw_dtypes.strip())

    shapes = _normalize_shape_list(case_entry.get("input_shape"))
    if not shapes:
        return dims

    lead = [int(x) for x in shapes[0]]
    dims["total_elements"] = int(math.prod(max(1, d) for d in lead))
    if "dim" in dims and isinstance(dims["dim"], int) and lead:
        dim_ax = int(dims["dim"])
        ndim = len(lead)
        norm_dim = dim_ax if dim_ax >= 0 else ndim + dim_ax
        if 0 <= norm_dim < ndim:
            dims["d_dim"] = int(lead[norm_dim])
            dims["D"] = int(lead[norm_dim])
            inner = 1
            for val in lead[norm_dim + 1 :]:
                inner *= max(1, int(val))
            dims["inner"] = int(inner)
            dims.setdefault("half_dim", max(1, int(lead[norm_dim]) // 2))

    layout = str(dims.get("inputLayout", dims.get("layout", "BSND"))).upper()
    if "inputLayout" in dims or "input_layout" in dims:
        dims.setdefault("inputLayout", layout)

    if len(lead) == 1:
        dims.setdefault("D", lead[0])
        dims.setdefault("d_dim", lead[0])
        dims.setdefault("N", lead[0])
    elif len(lead) == 2:
        dims.setdefault("M", lead[0])
        dims.setdefault("m_rows", lead[0])
        dims.setdefault("D", lead[1])
        dims.setdefault("K", lead[1])
    elif len(lead) == 3:
        dims.setdefault("B", lead[0])
        dims.setdefault("b_dim", lead[0])
        dims.setdefault("S", lead[1])
        dims.setdefault("s_dim", lead[1])
        dims.setdefault("m_rows", lead[0] * lead[1])
        dims.setdefault("D", lead[2])
        dims.setdefault("hc_mult", lead[2])
    elif len(lead) >= 4:
        has_stacked_prefix = (
            not proto_inputs
            and len(shapes) >= 2
            and len(shapes[1]) == len(lead) - 1
            and list(shapes[1]) == list(lead[1:])
        )
        if has_stacked_prefix:
            dims.setdefault("N", lead[0])
            dims.setdefault("B", lead[1])
            dims.setdefault("b_dim", lead[1])
            dims.setdefault("S", lead[2])
            dims.setdefault("s_dim", lead[2])
            dims.setdefault("D", lead[3])
            dims.setdefault("m_rows", lead[1] * lead[2])
        elif layout == "BNSD":
            dims.setdefault("B", lead[0])
            dims.setdefault("b_dim", lead[0])
            dims.setdefault("N_q", lead[1])
            dims.setdefault("n_q", lead[1])
            dims.setdefault("N_h", lead[1])
            dims.setdefault("S", lead[2])
            dims.setdefault("s_dim", lead[2])
            dims.setdefault("D", lead[-1])
            dims.setdefault("m_rows", lead[0] * lead[2])
        else:
            dims.setdefault("B", lead[0])
            dims.setdefault("b_dim", lead[0])
            dims.setdefault("S", lead[1])
            dims.setdefault("s_dim", lead[1])
            dims.setdefault("t_dim", lead[1])
            dims.setdefault("N_h", lead[2])
            dims.setdefault("N_q", lead[2])
            dims.setdefault("n_q", lead[2])
            dims.setdefault("D", lead[-1])
            dims.setdefault("m_rows", lead[0] * lead[1])

    if (
        len(shapes) >= 3
        and len(shapes[0]) >= 4
        and len(shapes[1]) >= 4
        and len(shapes[2]) >= 4
        and shapes[0][-1] != shapes[1][-1]
    ):
        dims.setdefault("D_nope", int(shapes[0][-1]))
        dims.setdefault("D_rope", int(shapes[1][-1]))
        k_sh = shapes[2]
        if layout == "BNSD":
            dims.setdefault("N_kv", int(k_sh[1]))
            dims.setdefault("S_kv", int(k_sh[2]))
        else:
            dims.setdefault("S_kv", int(k_sh[1]))
            dims.setdefault("N_kv", int(k_sh[2]))
        dims["is_causal"] = bool(dims.get("is_causal", False))
    elif "numKVHeads" in dims:
        dims.setdefault("N_kv", int(dims["numKVHeads"]))
    elif len(shapes) >= 2 and len(shapes[1]) >= 4:
        kv_idx = 1 if layout == "BNSD" else 2
        dims.setdefault("n_kv", int(shapes[1][kv_idx]))
        dims.setdefault("N_kv", int(shapes[1][kv_idx]))

    if len(shapes) >= 3 and shapes[2]:
        dims.setdefault("dv", int(shapes[2][-1]))
    if lead:
        dims.setdefault("half_dim", max(1, int(lead[-1]) // 2))
        dims.setdefault("d_dim", int(lead[-1]))
        dims.setdefault("dk", int(lead[-1]))
    if len(shapes) >= 2 and len(shapes[1]) >= 2 and len(lead) == 2:
        dims.setdefault("prog_len", int(lead[0]))
        dims.setdefault("num_regs", int(shapes[1][0]))
        dims.setdefault("w_dim", int(shapes[1][1]))

    axis_aliases = {
        "d_nope": ("D_nope", "d_nope"),
        "d_rope": ("D_rope", "d_rope"),
        "n_q": ("N_q", "n_q"),
        "n_kv": ("N_kv", "n_kv"),
        "s_kv": ("S_kv", "s_kv"),
        "b": ("B", "b_dim"),
        "s": ("S", "s_dim"),
        "t": ("S", "s_dim", "t_dim"),
        "d": ("D", "d_dim"),
        "n": ("N",),
    }

    for idx, sh in enumerate(shapes):
        pname = (
            str(proto_inputs[idx].get("name", f"in_{idx}"))
            if idx < len(proto_inputs) and isinstance(proto_inputs[idx], Mapping)
            else f"in_{idx}"
        )
        schema_axes = (
            _parse_proto_shape_axes(proto_inputs[idx].get("shape"), layout)
            if idx < len(proto_inputs) and isinstance(proto_inputs[idx], Mapping)
            else []
        )
        for ax, dval in enumerate(sh):
            iv = int(dval)
            dims[f"{pname}_dim_{ax}"] = iv
            dims[f"shape_{idx}_dim_{ax}"] = iv
            if ax < len(schema_axes):
                raw_ax = schema_axes[ax]
                if raw_ax.isidentifier():
                    dims[raw_ax] = iv
                    for alias in axis_aliases.get(raw_ax.lower(), ()):
                        dims[alias] = iv
    return dims


def get_operator_task(operator_name: str) -> str:
    """Return CANNBench task path (levelX/<op>) for any supported operator name or task id."""
    if operator_name in OPERATOR_METADATA:
        return OPERATOR_METADATA[operator_name][0]
    return CannbenchOpsRegistry.default().resolve(operator_name).task


def get_operator_module_relpath(operator_name: str) -> str:
    """Return workspace-relative kernel module path for any supported operator."""
    if operator_name in OPERATOR_METADATA:
        return OPERATOR_METADATA[operator_name][1]
    return CannbenchOpsRegistry.default().resolve_module_path(operator_name)


def get_operator_entrypoint(operator_name: str) -> str:
    """Return public Python callable name for any supported operator."""
    if operator_name in OPERATOR_METADATA:
        return OPERATOR_METADATA[operator_name][2]
    return CannbenchOpsRegistry.default().resolve(operator_name).entrypoint


def get_default_anchors(
    operator_name: str,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    """Return stratified anchor cases hydrated from live 950pr metadata when available."""
    try:
        if cannbench_repo is not None or env_file is not None:
            repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
            tasks_root = resolve_cannbench_tasks_root(repo)
            return CannbenchOpsRegistry(tasks_root=tasks_root).get_anchors(operator_name)
        return CannbenchOpsRegistry.default().get_anchors(operator_name)
    except (FileNotFoundError, KeyError):
        if operator_name in DEFAULT_ANCHORS:
            return DEFAULT_ANCHORS[operator_name]
        raise


def get_default_perf_target(operator_name: str) -> PerfTargetSpec:
    """Return default performance and Roofline target thresholds for any operator."""
    if operator_name in DEFAULT_PERF_TARGETS:
        return DEFAULT_PERF_TARGETS[operator_name]
    return CannbenchOpsRegistry.default().get_perf_target(operator_name)


def get_default_regime_specs(operator_name: str) -> tuple[RegimeTargetSpec, ...]:
    """Return the registry's campaign regimes for any operator.

    The registry is the single source of partition truth: authored incumbent partitions
    remain while their module exists, and every other case uses WP1's physical mapping.
    Delegating unconditionally keeps cold starts from reading the authored incumbent
    partitions through this convenience helper.
    """
    return CannbenchOpsRegistry.default().get_regime_specs(operator_name)


def parse_simple_env_file(env_file: str | Path) -> dict[str, str]:
    """Parse NAME=VALUE assignments from a plain shell config file."""
    path = Path(env_file)
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, raw_val = line.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            continue
        try:
            tokens = shlex.split(raw_val)
            value = tokens[0] if len(tokens) == 1 else raw_val
        except ValueError:
            value = raw_val.strip("\"'")
        parsed[key] = value
    return parsed


def _has_task_layout(path: Path) -> bool:
    """Return True if *path* contains cann-bench tasks metadata or level dirs."""
    if not path.is_dir():
        return False
    if (path / "suite_manifest.json").is_file() or (path / "metadata").is_dir():
        return True
    return any((path / f"level{lvl}").is_dir() for lvl in range(1, 6))


def _has_flat_operator_layout(path: Path) -> bool:
    """Return True if *path* directly contains official operator task directories."""
    if not path.is_dir():
        return False
    return any(
        (child / "cases.yaml").is_file() and (child / "proto.yaml").is_file()
        for child in path.iterdir()
        if child.is_dir()
    )


def resolve_cannbench_tasks_root(repo: str | Path) -> Path:
    """Resolve the tasks/ root inside an official cann-bench checkout (or direct tasks dir)."""
    base = Path(repo).expanduser().resolve()
    cand = base / "tasks"
    if cand.is_dir() and _has_task_layout(cand):
        return cand.resolve()
    if _has_task_layout(base):
        return base
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(
        f"Cannot locate tasks/ directory under cann-bench repo: {base}"
    )


def _is_valid_cannbench_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "tasks").is_dir() or _has_task_layout(path)


def resolve_cannbench_repo(
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> Path:
    """Resolve TILEFOUNDRY_NPU_CANNBENCH_REPO from explicit arg, env, or env.sh."""
    candidates: list[str] = []
    if cannbench_repo is not None:
        candidates.append(str(cannbench_repo))
    env_val = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    if env_val:
        candidates.append(env_val)
    resolved_env_file = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if resolved_env_file:
        parsed = parse_simple_env_file(resolved_env_file)
        if parsed.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"):
            candidates.append(parsed["TILEFOUNDRY_NPU_CANNBENCH_REPO"])

    for raw in candidates:
        path = Path(raw).expanduser()
        if _is_valid_cannbench_repo(path):
            return path.resolve()
    for fallback in (
        default_source_root().parent / "cann-bench",
        Path.home() / "workspace" / "cann-bench",
    ):
        if _is_valid_cannbench_repo(fallback):
            return fallback.resolve()
    raise FileNotFoundError(
        "Cannot locate TILEFOUNDRY_NPU_CANNBENCH_REPO. "
        "Pass cannbench_repo or set TILEFOUNDRY_NPU_CANNBENCH_REPO."
    )


def default_source_root() -> Path:
    """Resolve TileFoundry repository root relative to this module."""
    return Path(__file__).resolve().parents[5]


def load_cannbench_contract_module(source_root: str | Path | None = None):
    """Dynamically load tests/ascend/benchmarks/cannbench_contract.py."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    contract_path = root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py"
    if not contract_path.is_file():
        raise FileNotFoundError(f"cannbench_contract.py not found under {contract_path}")
    spec = importlib.util.spec_from_file_location("_tf_cannbench_contract", str(contract_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {contract_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_anchor_harness_and_cases(
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec] | None = None,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    source_root: str | Path | None = None,
):
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


def compute_case_eq3_score(ref_perf_us: float, cand_us: float, t_hw_us: float) -> float:
    """Compute CANNBench Eq.3 per-case performance score in [0, +inf)."""
    if cand_us <= 0.0 or t_hw_us <= 0.0:
        return 0.0
    baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(t_hw_us * 10.0, 10.0)
    denom = (cand_us - t_hw_us) + (baseline - t_hw_us)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (baseline - t_hw_us) / denom)


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
