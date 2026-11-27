"""YAML-driven CANNBench operator spec registry and semantic regime builder."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import shlex
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import yaml

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import (
    DTYPE_BYTES, AlignmentBand, CasePhysicsRatios, EnginePhase, LayoutPhase, ResidencyBand,
    TopologyAxes, WaveBand, case_workload_bytes, classify_case_regime, flatten_int_shapes,
    flatten_str_values, infer_case_dims_structural, infer_case_physics_kind,
    partition_case_regimes,
)


@dataclass(frozen=True)
class AnchorCaseSpec(JsonRecordMixin):
    """Single CANNBench anchor case with calibrated reference and Roofline baselines."""

    task: str
    case_num: int
    regime_tag: str
    ref_perf_us: float
    t_hw_us: float


@dataclass(frozen=True)
class GuardrailCaseSpec(JsonRecordMixin):
    """Non-scoreboard L3 failure probe tracked for regression guarding."""

    task: str
    case_num: int
    regime_tag: str
    incumbent_ref_us: float = 0.0
    t_hw_us: float = 0.0

    def to_anchor_spec(self) -> AnchorCaseSpec:
        """Convert guardrail case into an AnchorCaseSpec for batched L2 probing."""
        ref_us = float(self.incumbent_ref_us) if self.incumbent_ref_us > 0.0 else 1.0
        hw_us = float(self.t_hw_us) if self.t_hw_us > 0.0 else min(1.0, ref_us)
        return AnchorCaseSpec(self.task, int(self.case_num), self.regime_tag, ref_us, hw_us)


@dataclass(frozen=True)
class PerfTargetSpec(JsonRecordMixin):
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
        out = super().to_dict()
        out["per_anchor_max_us"] = {str(k): v for k, v in sorted(self.per_anchor_max_us.items())}
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PerfTargetSpec:
        """Reconstruct PerfTargetSpec from a JSON mapping."""
        norm = dict(payload)
        raw_max = norm.get("per_anchor_max_us")
        norm["per_anchor_max_us"] = (
            {int(k): float(v) for k, v in raw_max.items()} if isinstance(raw_max, Mapping) else {}
        )
        return super().from_dict(norm)


@dataclass(frozen=True)
class RegimeTargetSpec(JsonRecordMixin):
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimeTargetSpec:
        """Reconstruct RegimeTargetSpec from a JSON mapping."""
        norm = dict(payload)
        for key in (
            "sample_builder_kwargs", "representative_hir_dims",
            "representative_anchor_dims", "representative_case_entry",
        ):
            if norm.get(key) is None:
                norm[key] = {}
        for key in ("builder_candidates", "editable_symbols", "trackb_symbols"):
            if norm.get(key) is None:
                norm[key] = ()
        if not norm.get("runtime_backend"):
            norm["runtime_backend"] = "tilelang_dsl"
        return super().from_dict(norm)


RegimeSpec = RegimeTargetSpec

_OP_LEVELS: tuple[tuple[str, str], ...] = (
    ("attention_residual", "level4"), ("mla", "level4"), ("swi_glu", "level1"),
    ("rms_norm", "level2"), ("softmax", "level2"), ("add_rms_norm_dynamic_quant", "level3"),
    ("mhc_sinkhorn", "level3"), ("gqa", "level4"), ("kimi_delta_attention", "level5"),
    ("tensor_program_vm", "level5"),
)
OPERATOR_METADATA: dict[str, tuple[str, str, str]] = {
    op: (f"{lvl}/{op}", f"examples/ascend950pr_ops/{op}/kernels.py", op) for op, lvl in _OP_LEVELS
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
    anchors: Sequence[AnchorCaseSpec], meta_950pr: Mapping[int, tuple[float, float]]
) -> tuple[AnchorCaseSpec, ...]:
    out: list[AnchorCaseSpec] = []
    for a in anchors:
        entry = meta_950pr.get(int(a.case_num))
        if entry is not None and float(entry[1]) > 0.0:
            rv = float(entry[0])
            out.append(replace(a, ref_perf_us=rv if rv > 0.0 else a.ref_perf_us, t_hw_us=float(entry[1])))
        else:
            out.append(a)
    return tuple(out)


DEFAULT_ANCHORS: dict[str, tuple[AnchorCaseSpec, ...]] = {
    "attention_residual": ATTENTION_RESIDUAL_ANCHORS,
    "mla": MLA_ANCHORS,
}
DEFAULT_PERF_TARGETS: dict[str, PerfTargetSpec] = {
    "attention_residual": PerfTargetSpec("attention_residual", 2.85, 1.60, 0.90, {}, 97.5, 3),
    "mla": PerfTargetSpec("mla", 2.50, 1.35, 0.85, {7: 8.0, 13: 95.0}, 92.0, 3),
}


def _ar_regime(
    tag: str, primary: str, skw: dict[str, object], desc: str, hir: tuple[int, int, int],
    anc: tuple[int, int, int], sel: str, pfx: str,
) -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag=tag, builder_candidates=(primary, "_build_attention_residual_prim_func"),
        sample_builder_kwargs={**skw, "schedule_mode": "tuned"}, description=desc,
        representative_hir_dims={"N": hir[0], "B": 1, "S": hir[1], "D": hir[2]},
        representative_anchor_dims={"N": anc[0], "B": 1, "S": anc[1], "D": anc[2]},
        config_selector_fn=sel, editable_symbols=(sel, primary), allowed_helper_prefix=pfx,
    )


def _mla_regime(
    tag: str, primary: str, skw: dict[str, object], desc: str,
    hir: dict[str, int], anc: dict[str, int], sel: str, extra: tuple[str, ...], pfx: str,
) -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag=tag, builder_candidates=(primary, "_build_mla_prim_func"),
        sample_builder_kwargs={**skw, "schedule_mode": "tuned"}, description=desc,
        representative_hir_dims=hir, representative_anchor_dims=anc, config_selector_fn=sel,
        editable_symbols=(sel, primary, *extra, "_MLA_TRACKB_SRC"), allowed_helper_prefix=pfx,
        runtime_backend="hybrid_trackb", trackb_symbols=("_MLA_TRACKB_SRC",),
    )


DEFAULT_REGIME_SPECS: dict[str, tuple[RegimeTargetSpec, ...]] = {
    "attention_residual": (
        _ar_regime(
            "small_decode", "_build_ar_small_decode_func",
            {"n_depth": 2, "d_dim": 4096, "tl_dtype": "bfloat16", "tile_m": 1},
            "Small-row decode regime (m_rows < 28) needing depth/width core sharding.",
            (2, 1, 4096), (2, 1, 4096), "_select_ar_small_decode_config", "_ar_small_decode_",
        ),
        _ar_regime(
            "unaligned_d", "_build_ar_unaligned_func",
            {"n_depth": 8, "d_dim": 1000, "tl_dtype": "bfloat16", "tile_m": 1},
            "Non-64-aligned D tail regime enabling padded UB row pitch and multi-row buffering.",
            (8, 4, 1024), (8, 16, 1000), "_select_ar_unaligned_config", "_ar_unaligned_",
        ),
        _ar_regime(
            "aligned_prefill", "_build_ar_aligned_func",
            {"n_depth": 8, "d_dim": 4096, "tl_dtype": "bfloat16", "tile_m": 6},
            "Aligned prefill regime with 56-core multi-row SimdVF pipelining.",
            (8, 6, 512), (8, 4096, 4096), "_select_ar_aligned_config", "_ar_aligned_",
        ),
    ),
    "mla": (
        _mla_regime(
            "split_kv_decode", "_build_mla_split_kv_decode_func",
            {
                "b_dim": 1, "s_dim": 1, "s_kv": 2048, "n_q": 128, "n_kv": 1, "d_nope": 512,
                "d_rope": 64, "tl_dtype": "float16", "input_layout": "BSND", "is_causal": False,
                "bm": 64, "aligned_group_q": True,
            },
            "Decode and small-Sq regime targeting Split-KV FlashDecoding across 28 MixCores.",
            {"B": 1, "S": 1, "S_kv": 64, "N_q": 16, "N_kv": 1, "D_nope": 512, "D_rope": 64},
            {"B": 1, "S": 1, "S_kv": 2048, "N_q": 128, "N_kv": 1, "D_nope": 512, "D_rope": 64},
            "_select_mla_decode_config", ("_build_mla_split_kv_combine_func",), "_mla_decode_",
        ),
        _mla_regime(
            "bnsd_gather", "_build_mla_bnsd_gather_func",
            {
                "n_q": 128, "n_kv": 1, "d_nope": 512, "d_rope": 64, "tl_dtype": "float16",
                "input_layout": "BNSD", "is_causal": True, "bm": 64, "aligned_group_q": False,
            },
            "Non-aligned BNSD Q staging regime replacing scalar row copies.",
            {"B": 1, "S": 2, "S_kv": 32, "N_q": 16, "N_kv": 1, "D_nope": 128, "D_rope": 64},
            {"B": 1, "S": 512, "S_kv": 512, "N_q": 128, "N_kv": 1, "D_nope": 512, "D_rope": 64},
            "_select_mla_bnsd_config", (), "_mla_bnsd_",
        ),
        _mla_regime(
            "prefill_compute", "_build_mla_prefill_func",
            {
                "n_q": 128, "n_kv": 1, "d_nope": 512, "d_rope": 64, "tl_dtype": "float16",
                "input_layout": "BSND", "is_causal": True, "bm": 128, "aligned_group_q": True,
            },
            "Prefill and large-batch compute/pipeline overlap regime.",
            {"B": 1, "S": 8, "S_kv": 32, "N_q": 8, "N_kv": 1, "D_nope": 128, "D_rope": 64},
            {"B": 1, "S": 512, "S_kv": 512, "N_q": 128, "N_kv": 1, "D_nope": 512, "D_rope": 64},
            "_select_mla_prefill_config", (), "_mla_prefill_",
        ),
    ),
}


def _normalize_shape_list(raw_shapes: object) -> list[list[int]]:
    if not isinstance(raw_shapes, (list, tuple)):
        return []
    if raw_shapes and all(isinstance(x, int) and not isinstance(x, bool) for x in raw_shapes):
        return [[int(x) for x in raw_shapes]]
    shapes: list[list[int]] = []
    for item in raw_shapes:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        if all(isinstance(x, int) and not isinstance(x, bool) for x in item):
            shapes.append([int(x) for x in item])
        elif isinstance(item[0], (list, tuple)):
            for sub in item:
                if isinstance(sub, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in sub):
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


def _analyze_case_geometry(
    case_entry: Mapping[str, object], ref_us: float, t_hw_us: float
) -> dict[str, object]:
    shapes = _normalize_shape_list(case_entry.get("input_shape"))
    attrs = case_entry.get("attrs") if isinstance(case_entry.get("attrs"), Mapping) else {}
    all_dims = [d for sh in shapes for d in sh if d > 0]
    tails = [sh[-1] for sh in shapes if sh]
    total_elems = sum(math.prod(d for d in sh if d > 0) for sh in shapes) if shapes else 0
    odd_prime = any(d > 1 and (d % 2 == 1 or _is_prime_gt2(d)) for d in tails)
    u16_tail = any(d > 1 and d % 16 != 0 for d in tails)
    u_any = any(d > 1 and d % 16 != 0 for d in all_dims)
    u64_tail = any(d > 1 and d % 64 != 0 for d in tails)
    unaligned_score = (4 * odd_prime) + (3 * u16_tail) + (2 * u_any) + int(u64_tail)
    has_empty_group, group_skew = False, 1.0
    if isinstance(gl := attrs.get("group_list"), (list, tuple)) and gl:
        try:
            gvals = [int(x) for x in gl]
            diffs = [max(0, gvals[i] - (gvals[i - 1] if i else 0)) for i in range(len(gvals))]
            has_empty_group = any(d == 0 for d in diffs)
            if pos := [d for d in diffs if d > 0]:
                group_skew = float(max(pos)) / float(min(pos))
        except (TypeError, ValueError):
            pass
    vr = repr(case_entry.get("value_range", "")).lower()
    special = (
        (2 if attrs.get("inplace") is True else 0) + (1 if attrs.get("is_causal") is True else 0)
        + (2 if str(attrs.get("inputLayout", "")).upper() == "BNSD" else 0)
        + (1 if any(t in vr for t in ("inf", "nan", "unique")) else 0)
    )
    return {
        "case_id": int(case_entry.get("case_id", 0)), "shapes": shapes, "attrs": dict(attrs),
        "total_elems": max(1, total_elems), "max_rank": max((len(s) for s in shapes), default=0),
        "is_unaligned": unaligned_score > 0, "unaligned_score": unaligned_score,
        "aspect_ratio": float(max(all_dims)) / float(max(1, min(all_dims))) if all_dims else 1.0,
        "has_empty_group": has_empty_group, "group_skew": group_skew,
        "special_attr_score": special, "ref_perf_us": float(ref_us), "t_hw_us": float(t_hw_us),
    }


def _stratify_operator_anchors(
    task: str, cases: Sequence[Mapping[str, object]], meta_map: Mapping[int, tuple[float, float]]
) -> tuple[AnchorCaseSpec, ...]:
    if not cases:
        return ()
    metrics = [
        _analyze_case_geometry(raw, *meta_map.get(int(raw.get("case_id", 0)), (0.0, 0.0))) for raw in cases
    ]
    picked: list[tuple[dict[str, object], str]] = []
    picked_ids: set[int] = set()

    def _pick(cands: Sequence[dict[str, object]], tag: str) -> None:
        for item in cands:
            if (cid := int(item["case_id"])) not in picked_ids:
                picked_ids.add(cid)
                picked.append((item, tag))
                return

    by_small = sorted(metrics, key=lambda m: (m["total_elems"], m["t_hw_us"] if m["t_hw_us"] > 0.0 else float("inf"), m["case_id"]))
    _pick(by_small, "small_latency")
    _pick(sorted((m for m in metrics if m["is_unaligned"]), key=lambda m: (-int(m["unaligned_score"]), m["total_elems"], m["case_id"])) or by_small, "unaligned_tail")
    _pick(sorted(metrics, key=lambda m: (-int(m["total_elems"]), -float(m["t_hw_us"]), m["case_id"])), "large_throughput")
    _pick(sorted(metrics, key=lambda m: (-int(bool(m["has_empty_group"])), -float(m["group_skew"]), -float(m["aspect_ratio"]), -int(m["special_attr_score"]), -int(m["total_elems"]), m["case_id"])), "extreme_aspect")
    if len(picked) < 5 and len(metrics) >= 5:
        by_mid = sorted(metrics, key=lambda m: (float(m["t_hw_us"]) if m["t_hw_us"] > 0.0 else float(m["total_elems"]), m["total_elems"], m["case_id"]))
        mid = len(by_mid) // 2
        _pick([by_mid[mid], *by_mid[mid + 1 :], *reversed(by_mid[:mid])], "mid_balanced")
    return tuple(sorted(
        (AnchorCaseSpec(task, int(m["case_id"]), tag, float(m["ref_perf_us"]), float(m["t_hw_us"])) for m, tag in picked),
        key=lambda a: a.case_num,
    ))


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

    @property
    def desc_path(self) -> Path:
        """Return the operator markdown description path inside the task directory."""
        return self.op_dir / "desc.md"


_TRACKB_CPP_MARKERS: tuple[str, ...] = (
    "tl_templates/ascend/", "__global__", "__aicore__", "__vector__", "__cube__", "__mix__",
    "__simd_vf__", "__simd_callee__", "__simt_vf__", "__ubuf__", "__gm__", "SimdVF", "kernel_entry", "AscendC::",
)


def _extract_static_cpp_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _extract_static_cpp_string(node.left), _extract_static_cpp_string(node.right)
        return (left + right) if (left is not None and right is not None) else None
    if isinstance(node, ast.Call) and not node.keywords:
        fn = node.func
        is_dedent = (isinstance(fn, ast.Name) and fn.id == "dedent") or (isinstance(fn, ast.Attribute) and fn.attr == "dedent")
        if len(node.args) == 1 and is_dedent:
            return textwrap.dedent(inner) if (inner := _extract_static_cpp_string(node.args[0])) is not None else None
        if not node.args and isinstance(fn, ast.Attribute) and fn.attr in ("strip", "lstrip", "rstrip"):
            return getattr(inner, fn.attr)() if (inner := _extract_static_cpp_string(fn.value)) is not None else None
    return None


def _iter_top_level_str_assigns(tree: ast.Module) -> list[tuple[str, ast.AST | None]]:
    out: list[tuple[str, ast.AST | None]] = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.append((n.targets[0].id, n.value))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.append((n.target.id, n.value))
    return out


def discover_module_trackb_symbols(module_text: str) -> tuple[str, ...]:
    """Discover top-level Track-B C++20 kernel string constants and plan selectors."""
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return ()
    cpp_syms = [
        name for name, val in (_iter_top_level_str_assigns(tree) if "#include" in module_text else ())
        if (txt := _extract_static_cpp_string(val)) and "#include" in txt and any(m in txt for m in _TRACKB_CPP_MARKERS)
    ]
    if not cpp_syms:
        return ()
    plan_syms = [
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and (n.name.lower().startswith("_select_trackb_") or n.name.lower().startswith("_select_"))
        and ("trackb" in n.name.lower() or "track_b" in n.name.lower()) and n.name.lower().endswith("_plan")
    ]
    return tuple(dict.fromkeys([*cpp_syms, *plan_syms]))


def extract_trackb_cpp_source(module_source: str, trackb_symbols: Sequence[str]) -> tuple[str, str] | None:
    """Return the first Track-B C++20 translation-unit constant among trackb_symbols."""
    if not trackb_symbols:
        return None
    tree = ast.parse(module_source)
    consts = {n: c for n, v in _iter_top_level_str_assigns(tree) if (c := _extract_static_cpp_string(v)) is not None}
    return next(((s, consts[s]) for s in trackb_symbols if consts.get(s, "").strip()), None)


def infer_module_runtime_backend(module_text: str, entrypoint: str) -> tuple[str, tuple[str, ...]]:
    """Derive ('tilelang_dsl' | 'ascendc_trackb' | 'hybrid_trackb', symbols) from AST graph."""
    tb_syms = discover_module_trackb_symbols(module_text)
    if not module_text.strip():
        return "tilelang_dsl", ()
    try:
        funcs = {n.name: n for n in ast.parse(module_text).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return ("ascendc_trackb" if tb_syms else "tilelang_dsl"), tb_syms
    if entrypoint not in funcs:
        return ("ascendc_trackb" if tb_syms else "tilelang_dsl"), tb_syms
    reachable: set[str] = set()
    queue = [entrypoint]
    while queue:
        if (cur := queue.pop(0)) not in reachable and cur in funcs:
            reachable.add(cur)
            queue.extend(
                s.func.id for s in ast.walk(funcs[cur])
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id in funcs and s.func.id not in reachable
            )
    reaches_tl = reaches_tb = False
    for fn in reachable:
        seg = ast.get_source_segment(module_text, funcs[fn]) or ""
        reaches_tl = reaches_tl or ("tilelang.compile(" in seg)
        reaches_tb = reaches_tb or any(sym in seg for sym in tb_syms) or ("_get_track_b_lib" in seg)
    if reaches_tl and reaches_tb:
        return "hybrid_trackb", tb_syms
    return ("ascendc_trackb", tb_syms) if (reaches_tb or tb_syms) else ("tilelang_dsl", ())


@dataclass(frozen=True, kw_only=True)
class _SemanticCasePartition:
    base_axes: TopologyAxes
    has_reduction: bool
    ratios: tuple[CasePhysicsRatios, ...]


def _module_defines_authored_regimes(mod_text: str, regimes: Sequence[RegimeTargetSpec]) -> bool:
    if not mod_text.strip():
        return False
    try:
        defined = {n.name for n in ast.parse(mod_text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        return False
    return all(
        (not r.config_selector_fn or r.config_selector_fn in defined)
        and (not r.builder_candidates or any(c in defined for c in r.builder_candidates))
        for r in regimes
    )


class CannbenchOpsRegistry:
    """Dynamic registry discovering all 60+ operators in cann-bench/tasks."""

    _DEFAULT_INSTANCE: CannbenchOpsRegistry | None = None

    def __init__(
        self, tasks_root: str | Path | None = None, *, source_root: str | Path | None = None, pure_golden: bool = False
    ) -> None:
        self._source_root = Path(source_root).resolve() if source_root is not None else default_source_root()
        self._pure_golden = bool(pure_golden)
        self._tasks_root = self._resolve_tasks_root(tasks_root)
        meta_path = self._tasks_root / "metadata" / "950pr.json"
        self._meta_950pr: dict[str, dict[str, dict[str, dict[str, float]]]] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
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
            return p if _has_flat_operator_layout(p) else resolve_cannbench_tasks_root(p)
        cands: list[Path] = []
        if env_repo := os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip():
            ep = Path(env_repo).expanduser()
            cands.extend((ep / "tasks", ep))
        cands.extend((self._source_root.parent / "cann-bench" / "tasks", Path.home() / "workspace" / "cann-bench" / "tasks"))
        for check in (_has_task_layout, lambda _p: True):
            for cand in cands:
                if cand.is_dir() and check(cand):
                    return cand.resolve()
        raise FileNotFoundError("Cannot locate cann-bench/tasks directory. Pass tasks_root or set TILEFOUNDRY_NPU_CANNBENCH_REPO.")

    def _register_entry(self, op_id: str, task: str, level: str, level_dir: str, level_num: int, op_dir: Path) -> None:
        self._entries[op_id] = {"op_id": op_id, "task": task, "level": level, "level_dir": level_dir, "level_num": level_num, "op_dir": op_dir}
        self._task_to_op[task] = op_id

    def _discover_manifest(self) -> None:
        manifest_path = self._tasks_root / "suite_manifest.json"
        if manifest_path.is_file():
            for row in json.loads(manifest_path.read_text(encoding="utf-8")).get("operators", ()):
                op_id, raw_path = str(row["id"]), str(row["path"]).removeprefix("tasks/").strip("/")
                ldir = raw_path.split("/", 1)[0]
                if (op_dir := self._tasks_root / raw_path).is_dir():
                    lnum = int(ldir.removeprefix("level")) if ldir.startswith("level") else 1
                    self._register_entry(op_id, raw_path, str(row.get("level", f"L{lnum}")), ldir, lnum, op_dir)
            return
        for lpath in sorted(self._tasks_root.glob("level*")):
            if not lpath.is_dir():
                continue
            ldir = lpath.name
            lnum = int(ldir.removeprefix("level")) if ldir.startswith("level") else 1
            for op_dir in sorted(lpath.iterdir()):
                if op_dir.is_dir() and (op_dir / "cases.yaml").is_file():
                    self._register_entry(op_dir.name, f"{ldir}/{op_dir.name}", f"L{lnum}", ldir, lnum, op_dir)
        if self._entries or not _has_flat_operator_layout(self._tasks_root):
            return
        for op_dir in sorted(self._tasks_root.iterdir()):
            if not op_dir.is_dir() or not (op_dir / "cases.yaml").is_file():
                continue
            if (op_id := op_dir.name) not in OPERATOR_METADATA:
                raise ValueError(f"flat benchmark root {self._tasks_root} contains operator {op_id!r} without an official task mapping in OPERATOR_METADATA")
            task = OPERATOR_METADATA[op_id][0]
            ldir = task.split("/", 1)[0]
            lnum = int(ldir.removeprefix("level"))
            self._register_entry(op_id, task, f"L{lnum}", ldir, lnum, op_dir)

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
        if "/" in raw and (tail := raw.rsplit("/", 1)[-1]) in self._entries:
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

    def resolve_module_path(self, name_or_task: str, *, source_root: str | Path | None = None) -> str:
        """Resolve workspace-relative candidate module path (kernels.py or solution.py)."""
        op_id = self._normalize_key(name_or_task)
        if op_id in OPERATOR_METADATA:
            return OPERATOR_METADATA[op_id][1]
        root = Path(source_root).resolve() if source_root is not None else self._source_root
        for fname in ("kernels.py", "solution.py"):
            if (root / (rel := f"examples/ascend950pr_ops/{op_id}/{fname}")).is_file():
                return rel
        return f"examples/ascend950pr_ops/{op_id}/kernels.py"

    def resolve(self, name_or_task: str) -> CannbenchOpSpec:
        """Load and cache full CannbenchOpSpec (proto, cases, 950pr metadata, anchors, target)."""
        if (op_id := self._normalize_key(name_or_task)) in self._cache:
            return self._cache[op_id]
        entry = self._entries[op_id]
        op_dir = entry["op_dir"] if isinstance(entry["op_dir"], Path) else Path(str(entry["op_dir"]))
        task, ldir = str(entry["task"]), str(entry["level_dir"])
        proto_raw = yaml.safe_load((op_dir / "proto.yaml").read_text(encoding="utf-8")) if (op_dir / "proto.yaml").is_file() else {}
        cases_raw = yaml.safe_load((op_dir / "cases.yaml").read_text(encoding="utf-8")) if (op_dir / "cases.yaml").is_file() else {}
        cases = tuple(dict(r) for r in ((cases_raw or {}).get("cases") or ()) if isinstance(r, Mapping))
        meta_950pr = {
            int(k): (float(v.get("baseline_perf_us", 0.0)), float(v.get("t_hw_us", 0.0)))
            for k, v in self._meta_950pr.get(ldir, {}).get(op_id, {}).items() if isinstance(v, Mapping)
        }
        anchors = (
            _hydrate_anchors_from_metadata(DEFAULT_ANCHORS[op_id], meta_950pr)
            if op_id in DEFAULT_ANCHORS else _stratify_operator_anchors(task, cases, meta_950pr)
        )
        if op_id in DEFAULT_PERF_TARGETS:
            perf_target = DEFAULT_PERF_TARGETS[op_id]
        else:
            spds = [a.ref_perf_us / a.t_hw_us for a in anchors if a.ref_perf_us > 0.0 and a.t_hw_us > 0.0]
            avg_s = max(1.05, min(3.0, (sum(spds) / len(spds)) * 0.90)) if spds else 2.0
            min_s = max(1.0, min(avg_s, min(spds) * 0.85)) if spds else 1.25
            per_max = {int(a.case_num): round(a.t_hw_us * 1.15, 4) for a in anchors if a.t_hw_us > 0.0}
            perf_target = PerfTargetSpec(op_id, round(avg_s, 4), round(min_s, 4), 0.90, per_max, 95.0, 3)
        spec = CannbenchOpSpec(
            op_id, task, str(entry["level"]), int(entry["level_num"]), op_dir,
            dict((proto_raw or {}).get("operator") or {}), cases, meta_950pr, anchors, perf_target,
            self.resolve_module_path(op_id), op_id,
        )
        self._cache[op_id] = spec
        return spec

    def _module_path(self, spec: CannbenchOpSpec) -> Path:
        return self._source_root / spec.module_relpath

    def _module_source(self, spec: CannbenchOpSpec) -> str:
        return p.read_text(encoding="utf-8") if (p := self._module_path(spec)).is_file() else ""

    def _semantic_partition(self, spec: CannbenchOpSpec) -> _SemanticCasePartition:
        if spec.op_id in self._semantic_partitions:
            return self._semantic_partitions[spec.op_id]
        if not spec.golden_path.is_file():
            raise FileNotFoundError(f"operator {spec.op_id!r} has no PyTorch golden reference at {spec.golden_path} for semantic regime partitioning")
        analyzer = PyTorchSemanticAnalyzer()
        sem = analyzer.analyze_source(spec.golden_path.read_text(encoding="utf-8"), entry_name=spec.entrypoint, proto=spec.proto)
        part = _SemanticCasePartition(base_axes=sem.base_axes, has_reduction=sem.has_reduction, ratios=tuple(analyzer.compute_case_ratios(sem, c) for c in spec.cases))
        self._semantic_partitions[spec.op_id] = part
        return part

    def _semantic_regime_groups(self, spec: CannbenchOpSpec) -> tuple[SemanticRegimeGroup, ...]:
        if spec.op_id not in self._semantic_groups:
            part = self._semantic_partition(spec)
            self._semantic_groups[spec.op_id] = build_semantic_regime_groups(
                operator_name=spec.op_id, cases=spec.cases, proto=spec.proto,
                base_axes=part.base_axes, has_reduction=part.has_reduction, case_ratios=part.ratios,
            )
        return self._semantic_groups[spec.op_id]

    def _use_authored_regimes(self, op_id: str, spec: CannbenchOpSpec, *, pure_golden: bool | None = None) -> bool:
        is_pure = self._pure_golden if pure_golden is None else bool(pure_golden)
        if is_pure or op_id not in DEFAULT_REGIME_SPECS or not self._module_path(spec).is_file():
            return False
        return _module_defines_authored_regimes(self._module_source(spec), DEFAULT_REGIME_SPECS[op_id])

    def get_anchors(self, name_or_task: str, *, pure_golden: bool | None = None) -> tuple[AnchorCaseSpec, ...]:
        """Return anchor cases tagged with this operator's campaign regimes."""
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        if self._use_authored_regimes(op_id, spec, pure_golden=pure_golden):
            return spec.anchors
        groups = self._semantic_regime_groups(spec)
        tag_by_case = {cid: grp.regime_tag for grp in groups for cid, _, _ in grp.entries}
        retagged = tuple(replace(a, regime_tag=tag_by_case[a.case_num]) for a in spec.anchors if a.case_num in tag_by_case)
        covered = {a.regime_tag for a in retagged}
        additions = [
            AnchorCaseSpec(spec.task, cid, grp.regime_tag, *spec.metadata_950pr.get(cid, (0.0, 0.0)))
            for grp in groups if grp.regime_tag not in covered
            for cid, _, _ in (semantic_regime_representative(grp),)
        ]
        return tuple(sorted((*retagged, *additions), key=lambda a: a.case_num))

    def get_perf_target(self, name_or_task: str) -> PerfTargetSpec:
        """Return PerfTargetSpec targeting t_hw_us and HAP >= 95 for any operator."""
        return self.resolve(name_or_task).perf_target

    def get_case_regime_tag(self, name_or_task: str, case_num: int, *, pure_golden: bool | None = None) -> str | None:
        """Return the campaign regime tag for one official case."""
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        if self._use_authored_regimes(op_id, spec, pure_golden=pure_golden):
            return None
        target_case = int(case_num)
        for grp in self._semantic_regime_groups(spec):
            if any(cid == target_case for cid, _, _ in grp.entries):
                return grp.regime_tag
        raise KeyError(f"case_num={target_case} missing from semantic regime partition of {op_id!r}")

    def _hydrate_authored_regime_specs(self, spec: CannbenchOpSpec, regimes: Sequence[RegimeTargetSpec]) -> tuple[RegimeTargetSpec, ...]:
        ratios_by_id = {r.case_id: r for r in self._semantic_partition(spec).ratios}
        raw_by_id = {int(c["case_id"]): dict(c) for c in spec.cases}
        anchors_by_tag: dict[str, list[int]] = {}
        for a in spec.anchors:
            anchors_by_tag.setdefault(a.regime_tag, []).append(a.case_num)
        mod_text = "" if self._pure_golden else self._module_source(spec)
        extras = _discover_trackb_render_symbols(_defined_top_level_functions(mod_text)) if mod_text else ()
        out: list[RegimeTargetSpec] = []
        for reg in regimes:
            members = [cid for cid in anchors_by_tag.get(reg.regime_tag, ()) if cid in raw_by_id]
            if not members:
                raise ValueError(f"authored regime {reg.regime_tag!r} of {spec.op_id!r} matches no official anchor case")
            rep_cid = max(
                members,
                key=lambda cid: (case_workload_bytes(flatten_int_shapes(raw_by_id[cid].get("input_shape")), flatten_str_values(raw_by_id[cid].get("dtype"))), cid),
            )
            out.append(
                replace(
                    reg,
                    editable_symbols=tuple(dict.fromkeys((*reg.editable_symbols, *(extras if reg.trackb_symbols else ())))),
                    representative_case_entry=dict(raw_by_id[rep_cid]),
                    representative_case_ratios=ratios_by_id[rep_cid],
                )
            )
        return tuple(out)

    def get_regime_specs(self, name_or_task: str, *, pure_golden: bool | None = None) -> tuple[RegimeTargetSpec, ...]:
        """Return RegimeTargetSpecs partitioned by physical case mapping."""
        op_id = self._normalize_key(name_or_task)
        spec = self.resolve(op_id)
        is_pure = self._pure_golden if pure_golden is None else bool(pure_golden)
        if self._use_authored_regimes(op_id, spec, pure_golden=is_pure):
            return self._hydrate_authored_regime_specs(spec, DEFAULT_REGIME_SPECS[op_id])
        mod_text = "" if is_pure else self._module_source(spec)
        part = self._semantic_partition(spec)
        backend, tb_syms = ("tilelang_dsl", ()) if is_pure else infer_module_runtime_backend(mod_text, spec.entrypoint)
        return build_semantic_regime_specs(
            operator_name=op_id, entrypoint=spec.entrypoint, cases=spec.cases, proto=spec.proto,
            base_axes=part.base_axes, has_reduction=part.has_reduction, case_ratios=part.ratios,
            runtime_backend=backend, trackb_symbols=tb_syms, module_text=mod_text,
        )


_ENGINE_TAGS: dict[EnginePhase, str] = {"aiv_vector_or_gemv": "vec", "aic_cube": "cube", "coupled_1c2v": "cv"}
_LAYOUT_TAGS: dict[LayoutPhase, str] = {"contiguous": "contig", "strided_or_gather": "strided"}
_RESIDENCY_TAGS: dict[ResidencyBand, str] = {"vreg_resident": "vreg", "sram_1pass": "sram", "gm_multipass": "gmpass"}
_ALIGNMENT_TAGS: dict[AlignmentBand, str] = {"sub_atom_simt": "subatom", "aligned_vector": "aligned", "unaligned_predicated": "unaligned"}
_WAVE_TAGS: dict[WaveBand, str] = {"ragged_groups": "ragged", "under_occupied_split": "underfill", "balanced_waves": "filled"}
PhaseSignature = tuple[EnginePhase, LayoutPhase, ResidencyBand, AlignmentBand, WaveBand]
_CAMPAIGN_REGIME_PARTITION_CAP = 4


def semantic_regime_tag(signature: PhaseSignature) -> str:
    """Return the stable regime tag for one 5D physical phase coordinate."""
    e, l, r, a, w = signature
    return f"{_ENGINE_TAGS[e]}_{_LAYOUT_TAGS[l]}_{_RESIDENCY_TAGS[r]}_{_ALIGNMENT_TAGS[a]}_{_WAVE_TAGS[w]}"


def campaign_case_regime_tag(
    ratios: CasePhysicsRatios, raw_case: Mapping[str, object], *, base_axes: TopologyAxes, has_reduction: bool
) -> str:
    """Map one official case onto WP1's shared physical regime tag."""
    raw_attrs = raw_case.get("attrs")
    dims = infer_case_dims_structural(
        flatten_int_shapes(raw_case.get("input_shape")), raw_attrs if isinstance(raw_attrs, Mapping) else {},
        axes=base_axes, uses_cube=base_axes.uses_cube,
    )
    return classify_case_regime(ratios, kind=infer_case_physics_kind(base_axes, has_reduction=has_reduction), axes=base_axes, dims=dims)


@dataclass(frozen=True, kw_only=True)
class SemanticRegimeGroup:
    """One campaign regime with the official cases and raw coordinates it merges."""

    regime_tag: str
    entries: tuple[tuple[int, dict[str, object], PhaseSignature], ...]
    representative_case_id: int
    description: str
    representative_case_entry: dict[str, object]
    representative_case_ratios: CasePhysicsRatios


def build_semantic_regime_groups(
    *, operator_name: str, cases: Sequence[Mapping[str, object]], proto: Mapping[str, object],
    base_axes: TopologyAxes, has_reduction: bool, case_ratios: Sequence[CasePhysicsRatios],
) -> tuple[SemanticRegimeGroup, ...]:
    """Partition official cases through WP1's shared 2-4 physical regime mapping."""
    if not cases:
        raise ValueError(f"operator {operator_name!r} has no official cases.yaml entries to partition")
    if len(cases) != len(case_ratios):
        raise ValueError(f"operator {operator_name!r} has {len(cases)} cases but {len(case_ratios)} computed CasePhysicsRatios")
    raw_pi = proto.get("inputs")
    proto_inputs = tuple(r for r in raw_pi if isinstance(r, Mapping)) if isinstance(raw_pi, (list, tuple)) else ()
    ratios_id, raw_id, wp1_dims, staged_dims, wbytes = {}, {}, {}, {}, {}
    for raw_case, ratios in zip(cases, case_ratios):
        cid = raw_case.get("case_id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise ValueError(f"operator {operator_name!r} case entry requires a positive integer case_id")
        if cid != ratios.case_id:
            raise ValueError(f"operator {operator_name!r} case order mismatch: CasePhysicsRatios carry {ratios.case_id}, cases.yaml row carries {cid}")
        if cid in ratios_id:
            raise ValueError(f"operator {operator_name!r} repeats case id {cid}")
        attrs = raw_case.get("attrs")
        shapes = flatten_int_shapes(raw_case.get("input_shape"))
        ratios_id[cid], raw_id[cid] = ratios, dict(raw_case)
        wp1_dims[cid] = infer_case_dims_structural(shapes, attrs if isinstance(attrs, Mapping) else {}, axes=base_axes, uses_cube=base_axes.uses_cube)
        staged_dims[cid] = extract_schema_driven_case_dims(raw_case, proto_inputs=proto_inputs)
        wbytes[cid] = case_workload_bytes(shapes, flatten_str_values(raw_case.get("dtype")))
    groups = partition_case_regimes(
        ratios_id, wp1_dims, wbytes, kind=infer_case_physics_kind(base_axes, has_reduction=has_reduction),
        axes=base_axes, min_partitions=2, max_partitions=_CAMPAIGN_REGIME_PARTITION_CAP,
    )
    return tuple(
        SemanticRegimeGroup(
            regime_tag=grp.regime_tag,
            entries=tuple((c.case_id, dict(staged_dims[c.case_id]), c.phase_signature) for c in grp.cases),
            representative_case_id=grp.representative_case.case_id,
            description=(
                f"WP1 physical regime {grp.regime_tag} covering official cases {list(grp.case_ids)} "
                f"at 5D coordinates: {'; '.join(grp.phase_coordinates)}. "
                f"{(grp.rationale.strip() + ' ') if grp.rationale.strip() else ''}"
                f"Representative case {grp.representative_case.case_id} supplies the Stage-0 dimensions."
            ),
            representative_case_entry=raw_id[grp.representative_case.case_id],
            representative_case_ratios=ratios_id[grp.representative_case.case_id],
        )
        for grp in groups
    )


def semantic_regime_representative(group: SemanticRegimeGroup) -> tuple[int, dict[str, object], PhaseSignature]:
    """Return (case_id, dims, phase_signature) for the regime's WP1 representative case."""
    for cid, dims, sig in group.entries:
        if cid == group.representative_case_id:
            return cid, dict(dims), sig
    cid, dims, sig = group.entries[0]
    return cid, dict(dims), sig


def _defined_top_level_functions(module_text: str) -> dict[str, ast.FunctionDef]:
    if not module_text.strip():
        return {}
    try:
        return {n.name: n for n in ast.parse(module_text).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return {}


def _discover_trackb_render_symbols(defined_funcs: Mapping[str, ast.FunctionDef]) -> tuple[str, ...]:
    found = ["_render_trackb_tu"] if "_render_trackb_tu" in defined_funcs else []
    found.extend(n for n in sorted(defined_funcs) if n.startswith("_render_") and n.endswith("_trackb_launcher"))
    return tuple(dict.fromkeys(found))


def _extract_shape_aliases(func: ast.FunctionDef) -> tuple[dict[str, tuple[str, int]], dict[str, ast.expr]]:
    aliases: dict[str, tuple[str, int]] = {}
    bools: dict[str, ast.expr] = {}
    for st in func.body:
        if not isinstance(st, ast.Assign) or len(st.targets) != 1:
            continue
        tgt, val = st.targets[0], st.value
        if isinstance(tgt, ast.Name) and isinstance(val, ast.Subscript) and isinstance(val.value, ast.Attribute) and val.value.attr == "shape" and isinstance(val.value.value, ast.Name):
            idx = int(val.slice.value) if isinstance(val.slice, ast.Constant) and isinstance(val.slice.value, int) else (
                -int(val.slice.operand.value) if isinstance(val.slice, ast.UnaryOp) and isinstance(val.slice.op, ast.USub) and isinstance(val.slice.operand, ast.Constant) and isinstance(val.slice.operand.value, int) else None
            )
            if idx is not None:
                aliases[tgt.id] = (val.value.value.id, idx)
        elif isinstance(tgt, ast.Tuple) and isinstance(val, ast.Attribute) and val.attr == "shape" and isinstance(val.value, ast.Name):
            for i, elt in enumerate(tgt.elts):
                if isinstance(elt, ast.Name):
                    aliases[elt.id] = (val.value.id, i)
        elif isinstance(tgt, ast.Name) and isinstance(val, (ast.BoolOp, ast.Compare)):
            bools[tgt.id] = val
    return aliases, bools


def _lookup_tensor_shape_dim(tname: str, axis: int, int_dims: Mapping[str, int]) -> int | None:
    pfx = f"{tname}_dim_"
    idx = {int(k[len(pfx) :]): int(v) for k, v in int_dims.items() if k.startswith(pfx) and k[len(pfx) :].isdigit()}
    return idx.get(axis + max(idx) + 1 if axis < 0 else axis) if idx else None


def _eval_guard_value(node: ast.AST, int_dims: Mapping[str, int], aliases: Mapping[str, tuple[str, int]]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = [_eval_guard_value(e, int_dims, aliases) for e in node.elts]
        return None if any(e is None for e in elts) else tuple(elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _eval_guard_value(node.operand, int_dims, aliases)
        return -inner if isinstance(inner, (int, float)) and not isinstance(inner, bool) else None
    if isinstance(node, ast.Name):
        if node.id == "schedule_mode":
            return "tuned"
        if node.id in int_dims:
            return int(int_dims[node.id])
        low = node.id.lower()
        for k, v in int_dims.items():
            if k.lower() == low:
                return int(v)
        return _lookup_tensor_shape_dim(*aliases[node.id], int_dims) if node.id in aliases else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape" and isinstance(node.value.value, ast.Name):
        idx = _eval_guard_value(node.slice, int_dims, aliases)
        return _lookup_tensor_shape_dim(node.value.value.id, idx, int_dims) if isinstance(idx, int) and not isinstance(idx, bool) else None
    return None


def _eval_guard_predicate(
    node: ast.AST, int_dims: Mapping[str, int], aliases: Mapping[str, tuple[str, int]],
    bools: Mapping[str, ast.expr], seen: set[str],
) -> bool | None:
    if isinstance(node, ast.BoolOp):
        vals = [_eval_guard_predicate(v, int_dims, aliases, bools, seen) for v in node.values]
        tgt = False if isinstance(node.op, ast.And) else True
        return tgt if any(v is tgt for v in vals) else ((not tgt) if all(v is (not tgt) for v in vals) else None)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _eval_guard_predicate(node.operand, int_dims, aliases, bools, seen)
        return (not inner) if inner is not None else None
    if isinstance(node, ast.Name) and node.id in bools and node.id not in seen:
        seen.add(node.id)
        try:
            return _eval_guard_predicate(bools[node.id], int_dims, aliases, bools, seen)
        finally:
            seen.remove(node.id)
    if isinstance(node, ast.Compare):
        cur = _eval_guard_value(node.left, int_dims, aliases)
        if cur is None:
            return None
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_guard_value(comp, int_dims, aliases)
            if right is None or isinstance(op, (ast.Gt, ast.GtE)):
                return None
            if isinstance(op, (ast.Lt, ast.LtE)) and isinstance(cur, (int, float)) and isinstance(right, (int, float)):
                ok = cur < right if isinstance(op, ast.Lt) else cur <= right
            elif isinstance(op, (ast.Eq, ast.NotEq)):
                ok = (cur == right) if isinstance(op, ast.Eq) else (cur != right)
            elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, tuple):
                ok = (cur in right) if isinstance(op, ast.In) else (cur not in right)
            else:
                return None
            if not ok:
                return False
            cur = right
        return True
    return None


def _infer_regime_runtime_backend(
    module_text: str, entrypoint: str, module_backend: str, trackb_symbols: Sequence[str], int_dims: Mapping[str, int]
) -> str:
    if module_backend != "hybrid_trackb" or not module_text.strip() or not int_dims:
        return module_backend
    entry_fn = (funcs := _defined_top_level_functions(module_text)).get(entrypoint)
    if entry_fn is None:
        return module_backend
    tb_funcs = {
        fn for fn, fn_node in funcs.items()
        if fn != entrypoint and (any(s in (seg := ast.get_source_segment(module_text, fn_node) or "") for s in trackb_symbols) or "_get_track_b_lib" in seg)
    }
    changed = True
    while changed:
        changed = False
        for fn, fn_node in funcs.items():
            if fn != entrypoint and fn not in tb_funcs and any(isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id in tb_funcs for s in ast.walk(fn_node)):
                tb_funcs.add(fn)
                changed = True
    aliases, bools = _extract_shape_aliases(entry_fn)
    outcomes = [
        res for node in ast.walk(entry_fn)
        if isinstance(node, ast.If) and "device.type" not in (ast.get_source_segment(module_text, node.test) or "")
        and any(
            (isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and (s.func.id in tb_funcs or s.func.id == "_get_track_b_lib"))
            or (isinstance(s, ast.Name) and s.id in trackb_symbols)
            for st in node.body for s in ast.walk(st)
        )
        and (res := _eval_guard_predicate(node.test, int_dims, aliases, bools, set())) is not None
    ]
    return "tilelang_dsl" if (outcomes and not any(outcomes)) else module_backend


def build_semantic_regime_specs(
    *, operator_name: str, entrypoint: str, cases: Sequence[Mapping[str, object]], proto: Mapping[str, object],
    base_axes: TopologyAxes, has_reduction: bool, case_ratios: Sequence[CasePhysicsRatios],
    runtime_backend: str, trackb_symbols: Sequence[str] = (), module_text: str = "",
) -> tuple[RegimeTargetSpec, ...]:
    """Partition official cases into campaign regime specs through WP1's mapping."""
    if runtime_backend not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
        raise ValueError(f"semantic regime backend {runtime_backend!r} must be 'tilelang_dsl', 'ascendc_trackb', or 'hybrid_trackb'")
    groups = build_semantic_regime_groups(
        operator_name=operator_name, cases=cases, proto=proto, base_axes=base_axes, has_reduction=has_reduction, case_ratios=case_ratios
    )
    def_funcs = _defined_top_level_functions(module_text) if module_text.strip() else {}
    render_syms = _discover_trackb_render_symbols(def_funcs) if def_funcs else ()
    s_sel, s_bld, multi = f"_select_{entrypoint}_config", f"_build_{entrypoint}_prim_func", len(groups) > 1
    regimes: list[RegimeTargetSpec] = []
    for grp in groups:
        tag = grp.regime_tag
        cid, rep_dims, _ = semantic_regime_representative(grp)
        int_dims = {str(k): int(v) for k, v in rep_dims.items() if isinstance(v, int) and not isinstance(v, bool)}
        if not int_dims:
            raise ValueError(f"semantic regime {tag!r} for {operator_name!r} case {cid} has no concrete integer dimensions for Stage-0 instantiation")
        r_be = _infer_regime_runtime_backend(module_text, entrypoint, runtime_backend, trackb_symbols, int_dims)
        r_tb = tuple(trackb_symbols) if r_be != "tilelang_dsl" else ()
        ex_tb = (render_syms if def_funcs else ("_render_trackb_tu",)) if r_tb else ()
        p_sel, p_bld, leg_bld = f"_select_{entrypoint}_{tag}_config", f"_build_{entrypoint}_{tag}_prim_func", f"_build_{entrypoint}_{tag}_func"
        sel_ord = (p_sel, s_sel) if multi else (s_sel, p_sel)
        bld_ord = (p_bld, leg_bld, s_bld) if multi else (s_bld, p_bld, leg_bld)
        if def_funcs:
            sel_fn = next((s for s in sel_ord if s in def_funcs), sel_ord[0])
            res_bld = ([b for b in bld_ord[:2] if b in def_funcs] or ([s_bld] if s_bld in def_funcs else [])) if multi else [b for b in bld_ord if b in def_funcs]
            res_bld = res_bld or list(bld_ord[: 2 if multi else 1])
            b_cands = tuple(dict.fromkeys((*res_bld, *bld_ord, "make_kernel")))
            editable = (sel_fn, *res_bld, *r_tb, *ex_tb)
        else:
            sel_fn, b_cands = sel_ord[0], (*bld_ord, "make_kernel")
            editable = (sel_fn, *(bld_ord[:2] if multi else (bld_ord[0],)), *r_tb, *ex_tb)
        regimes.append(RegimeTargetSpec(
            tag, b_cands, dict(rep_dims), grp.description, dict(int_dims), dict(int_dims), sel_fn,
            tuple(dict.fromkeys(editable)), f"_{operator_name}_{tag}_", r_be, r_tb,
            dict(grp.representative_case_entry), grp.representative_case_ratios,
        ))
    return tuple(regimes)


def _parse_proto_shape_axes(shape_spec: object, layout: str) -> list[str]:
    if not isinstance(shape_spec, str) or "[" not in shape_spec:
        return []
    matches = re.findall(r"(?:([A-Za-z0-9_]+)\s*:\s*)?\[([^\]]+)\]", shape_spec)
    up = layout.strip().upper()
    chosen = next((inn for t, inn in matches if up and t.strip().upper() == up), "") or (matches[0][1] if matches else "")
    return [tok.strip() for tok in chosen.split(",") if tok.strip()] if chosen else []


_FLOATING_SCHEMA_DTYPES = frozenset({
    "float16", "bfloat16", "float32", "float64", "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu",
    "float4_e2m1", "float4_e1m2", "float4_e2m1fn", "float4_e2m1fnx2",
})
_ALL_SCHEMA_DTYPES = frozenset(DTYPE_BYTES) | _FLOATING_SCHEMA_DTYPES
_VM_PROTO_PARAM_NAMES = frozenset({"program", "prog_len", "num_regs", "w_dim"})
_AXIS_ALIASES: dict[str, tuple[str, ...]] = {
    "d_nope": ("D_nope", "d_nope"), "d_rope": ("D_rope", "d_rope"), "dk": ("D_qk", "dk"), "dv": ("D_v", "dv"),
    "n_q": ("N_q", "n_q"), "n_kv": ("N_kv", "n_kv"), "s_kv": ("S_kv", "s_kv"), "b": ("B", "b_dim"),
    "s": ("S", "s_dim"), "t": ("S", "s_dim", "t_dim"), "d": ("D", "d_dim"), "n": ("N",),
}


def extract_schema_driven_case_dims(
    case_entry: Mapping[str, object], proto_inputs: Sequence[Mapping[str, object]] = ()
) -> dict[str, object]:
    """Derive case dimension dictionary from proto parameter schemas and attrs."""
    dims: dict[str, object] = {str(k): v for k, v in (case_entry.get("attrs") or {}).items()} if isinstance(case_entry.get("attrs"), Mapping) else {}
    flat_dt = [d.strip() for d in flatten_str_values(case_entry.get("dtype")) if d.strip() and d.strip().lower() != "none"]
    for allowed in (_FLOATING_SCHEMA_DTYPES, _ALL_SCHEMA_DTYPES):
        if "tl_dtype" not in dims and (hit := next((d for d in flat_dt if d.lower() in allowed), None)):
            dims["tl_dtype"] = hit
    if not (shapes := _normalize_shape_list(case_entry.get("input_shape"))):
        return dims
    lead = [int(x) for x in shapes[0]]
    layout = str(dims.get("inputLayout", dims.get("layout", "BSND"))).upper()
    if "inputLayout" in dims or "input_layout" in dims:
        dims.setdefault("inputLayout", layout)

    p_names = {str(it.get("name", "")).strip().lower() for it in proto_inputs if isinstance(it, Mapping)}
    axes_by_in = [_parse_proto_shape_axes(it.get("shape"), layout) if isinstance(it, Mapping) else [] for it in proto_inputs]
    axes_low, keys_low = {ax.lower() for ax_l in axes_by_in for ax in ax_l}, {k.lower() for k in dims}
    has_half = bool({"rotarymode", "rotary_mode"} & keys_low or {"half_dim", "2h"} & axes_low or {"cos", "sin"} & p_names) or (
        len(shapes) >= 4 and len(lead) >= 3 and bool(shapes[2]) and bool(shapes[3]) and shapes[2][-1] * 2 == lead[-1] == shapes[3][-1] * 2
    )
    has_mm_k = "k" in axes_low or (not proto_inputs and (bool({"transpose_a", "transpose_b", "transpose_weight", "group_list"} & keys_low) or (len(shapes) >= 2 and len(shapes[1]) in (2, 3) and shapes[0] != shapes[1] and lead[-1] in shapes[1][-2:])))
    has_hc = "hc_mult" in axes_low or (not proto_inputs and len(lead) == 3 and len(shapes) == 1 and lead[1] == lead[2] and "dim" not in dims)
    has_attn = len(shapes) >= 3 and len(lead) >= 4 and len(shapes[1]) >= 4 and len(shapes[2]) >= 4 and (
        bool({"inputlayout", "input_layout", "numkvheads", "is_causal", "scalevalue", "use_qk_l2norm"} & keys_low) or bool({"q", "query", "q_nope"} & p_names) or shapes[0][-1] != shapes[1][-1]
    )

    dims["total_elements"] = int(math.prod(max(1, d) for d in lead))
    if isinstance(dims.get("dim"), int) and lead and 0 <= (nd := (int(dims["dim"]) if int(dims["dim"]) >= 0 else len(lead) + int(dims["dim"]))) < len(lead):
        dims["d_dim"] = dims["D"] = int(lead[nd])
        dims["inner"] = int(math.prod(max(1, int(v)) for v in lead[nd + 1 :]))
        if has_half:
            dims.setdefault("half_dim", max(1, int(lead[nd]) // 2))

    if len(lead) == 1:
        for k in ("D", "d_dim", "N"):
            dims.setdefault(k, lead[0])
    elif len(lead) == 2:
        dims.setdefault("M", lead[0])
        dims.setdefault("m_rows", lead[0])
        dims.setdefault("D", lead[1])
        if has_mm_k:
            dims.setdefault("K", lead[1])
    elif len(lead) == 3:
        dims.setdefault("B", lead[0]); dims.setdefault("b_dim", lead[0])
        dims.setdefault("S", lead[1]); dims.setdefault("s_dim", lead[1])
        dims.setdefault("m_rows", lead[0] * lead[1]); dims.setdefault("D", lead[2])
        if has_hc:
            dims.setdefault("hc_mult", lead[2])
    elif len(lead) >= 4:
        stacked = not proto_inputs and len(shapes) >= 2 and len(shapes[1]) == len(lead) - 1 and list(shapes[1]) == list(lead[1:])
        b_i, s_i, nq_i, d_v = (lead[1], lead[2], lead[0], lead[3]) if stacked else ((lead[0], lead[2], lead[1], lead[-1]) if layout == "BNSD" else (lead[0], lead[1], lead[2], lead[-1]))
        if stacked:
            dims.setdefault("N", lead[0])
        else:
            if layout != "BNSD":
                dims.setdefault("t_dim", s_i)
            for k in ("N_q", "n_q", "N_h"):
                dims.setdefault(k, nq_i)
        dims.setdefault("B", b_i); dims.setdefault("b_dim", b_i)
        dims.setdefault("S", s_i); dims.setdefault("s_dim", s_i)
        dims.setdefault("D", d_v); dims.setdefault("m_rows", b_i * s_i)

    if len(shapes) >= 3 and len(shapes[0]) >= 4 and len(shapes[1]) >= 4 and len(shapes[2]) >= 4 and shapes[0][-1] != shapes[1][-1]:
        dims.setdefault("D_nope", int(shapes[0][-1])); dims.setdefault("D_rope", int(shapes[1][-1]))
        k_sh = shapes[2]
        nkv_i, skv_i = (int(k_sh[1]), int(k_sh[2])) if layout == "BNSD" else (int(k_sh[2]), int(k_sh[1]))
        dims.setdefault("N_kv", nkv_i); dims.setdefault("S_kv", skv_i)
        dims["is_causal"] = bool(dims.get("is_causal", False))
    elif "numKVHeads" in dims:
        dims.setdefault("N_kv", int(dims["numKVHeads"]))
    elif len(shapes) >= 2 and len(shapes[1]) >= 4:
        kv_v = int(shapes[1][1 if layout == "BNSD" else 2])
        dims.setdefault("n_kv", kv_v); dims.setdefault("N_kv", kv_v)

    if has_attn:
        dims.setdefault("dv", int(shapes[2][-1])); dims.setdefault("dk", int(lead[-1]))
    if lead:
        dims.setdefault("d_dim", int(lead[-1]))
        if has_half:
            dims.setdefault("half_dim", max(1, int(lead[-1]) // 2))
    if len(shapes) >= 2 and len(shapes[1]) >= 2 and len(lead) == 2 and (p_names & _VM_PROTO_PARAM_NAMES):
        dims.setdefault("prog_len", int(lead[0])); dims.setdefault("num_regs", int(shapes[1][0])); dims.setdefault("w_dim", int(shapes[1][1]))

    for idx, sh in enumerate(shapes):
        pname = str(proto_inputs[idx].get("name", f"in_{idx}")) if idx < len(proto_inputs) and isinstance(proto_inputs[idx], Mapping) else f"in_{idx}"
        s_axes = axes_by_in[idx] if idx < len(axes_by_in) else []
        for ax, dval in enumerate(sh):
            iv = int(dval)
            dims[f"{pname}_dim_{ax}"] = dims[f"shape_{idx}_dim_{ax}"] = iv
            if ax < len(s_axes) and (rax := s_axes[ax]).isidentifier():
                dims[rax] = iv
                for alias in _AXIS_ALIASES.get(rax.lower(), ()):
                    dims[alias] = iv
    return dims


def get_operator_task(operator_name: str) -> str:
    """Return official CANNBench task identifier (for example 'level4/mla')."""
    return OPERATOR_METADATA[operator_name][0] if operator_name in OPERATOR_METADATA else CannbenchOpsRegistry.default().resolve(operator_name).task


def get_operator_module_relpath(operator_name: str) -> str:
    """Return workspace-relative candidate kernel module path."""
    return OPERATOR_METADATA[operator_name][1] if operator_name in OPERATOR_METADATA else CannbenchOpsRegistry.default().resolve_module_path(operator_name)


def get_operator_entrypoint(operator_name: str) -> str:
    """Return exported candidate Python function name inside the operator module."""
    return OPERATOR_METADATA[operator_name][2] if operator_name in OPERATOR_METADATA else CannbenchOpsRegistry.default().resolve(operator_name).entrypoint


def get_default_anchors(
    operator_name: str,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    include_extra: Sequence[AnchorCaseSpec] = (),
) -> tuple[AnchorCaseSpec, ...]:
    """Return baseline anchor specs for an operator hydrated from live metadata."""
    try:
        if cannbench_repo is not None or env_file is not None:
            repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
            reg = CannbenchOpsRegistry(tasks_root=resolve_cannbench_tasks_root(repo))
        else:
            reg = CannbenchOpsRegistry.default()
        base = reg.get_anchors(operator_name)
    except (FileNotFoundError, KeyError):
        if operator_name in DEFAULT_ANCHORS:
            base = DEFAULT_ANCHORS[operator_name]
        else:
            raise
    if not include_extra:
        return base
    by_case = {int(a.case_num): a for a in base}
    for extra in include_extra:
        by_case.setdefault(int(extra.case_num), extra)
    return tuple(sorted(by_case.values(), key=lambda a: a.case_num))


def get_default_perf_target(operator_name: str) -> PerfTargetSpec:
    """Return baseline Roofline and speedup target thresholds for an operator."""
    return DEFAULT_PERF_TARGETS[operator_name] if operator_name in DEFAULT_PERF_TARGETS else CannbenchOpsRegistry.default().get_perf_target(operator_name)


def get_default_regime_specs(operator_name: str) -> tuple[RegimeTargetSpec, ...]:
    """Return target dispatch regime specs for an operator via CannbenchOpsRegistry."""
    return CannbenchOpsRegistry.default().get_regime_specs(operator_name)


def parse_simple_env_file(env_file: str | Path) -> dict[str, str]:
    """Parse NAME=VALUE assignments from a plain shell config file."""
    if not (path := Path(env_file)).is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_val = (line[7:].strip() if line.startswith("export ") else line).split("=", 1)
        if not (key := key.strip()):
            continue
        raw_val = raw_val.strip()
        try:
            tokens = shlex.split(raw_val)
            parsed[key] = tokens[0] if len(tokens) == 1 else raw_val
        except ValueError:
            parsed[key] = raw_val.strip("\"'")
    return parsed


def _has_task_layout(path: Path) -> bool:
    return path.is_dir() and ((path / "suite_manifest.json").is_file() or (path / "metadata").is_dir() or any((path / f"level{lvl}").is_dir() for lvl in range(1, 6)))


def _has_flat_operator_layout(path: Path) -> bool:
    return path.is_dir() and any((c / "cases.yaml").is_file() and (c / "proto.yaml").is_file() for c in path.iterdir() if c.is_dir())


def resolve_cannbench_tasks_root(repo: str | Path) -> Path:
    """Resolve the tasks/ root inside an official cann-bench checkout (or direct tasks dir)."""
    base = Path(repo).expanduser().resolve()
    for p, req_layout in ((base / "tasks", True), (base, True), (base / "tasks", False)):
        if p.is_dir() and (not req_layout or _has_task_layout(p)):
            return p.resolve()
    raise FileNotFoundError(f"Cannot locate tasks/ directory under cann-bench repo: {base}")


def resolve_cannbench_repo(cannbench_repo: str | Path | None = None, env_file: str | Path | None = None) -> Path:
    """Resolve TILEFOUNDRY_NPU_CANNBENCH_REPO from explicit arg, env, or env.sh."""
    cands: list[str] = [str(cannbench_repo)] if cannbench_repo is not None else []
    if env_val := os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip():
        cands.append(env_val)
    eff_env = str(env_file) if env_file is not None else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    if eff_env and (val := parse_simple_env_file(eff_env).get("TILEFOUNDRY_NPU_CANNBENCH_REPO")):
        cands.append(val)
    for raw in (*cands, str(default_source_root().parent / "cann-bench"), str(Path.home() / "workspace" / "cann-bench")):
        if (p := Path(raw).expanduser()).is_dir() and ((p / "tasks").is_dir() or _has_task_layout(p)):
            return p.resolve()
    raise FileNotFoundError("Cannot locate TILEFOUNDRY_NPU_CANNBENCH_REPO. Pass cannbench_repo or set TILEFOUNDRY_NPU_CANNBENCH_REPO.")


def default_source_root() -> Path:
    """Resolve TileFoundry repository root relative to this module."""
    return Path(__file__).resolve().parents[5]


def load_cannbench_contract_module(source_root: str | Path | None = None) -> ModuleType:
    """Dynamically load tests/ascend/benchmarks/cannbench_contract.py."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    cpath = root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py"
    if not cpath.is_file():
        raise FileNotFoundError(f"cannbench_contract.py not found under {cpath}")
    spec = importlib.util.spec_from_file_location("_tf_cannbench_contract", str(cpath))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {cpath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


__all__ = [
    "ATTENTION_RESIDUAL_ANCHORS", "AnchorCaseSpec", "CannbenchOpSpec", "CannbenchOpsRegistry",
    "DEFAULT_ANCHORS", "DEFAULT_PERF_TARGETS", "DEFAULT_REGIME_SPECS", "GuardrailCaseSpec",
    "MLA_ANCHORS", "OPERATOR_METADATA", "PerfTargetSpec", "PhaseSignature", "RegimeSpec",
    "RegimeTargetSpec", "SemanticRegimeGroup", "build_semantic_regime_groups",
    "build_semantic_regime_specs", "campaign_case_regime_tag", "default_source_root",
    "discover_module_trackb_symbols", "extract_schema_driven_case_dims", "extract_trackb_cpp_source",
    "get_default_anchors", "get_default_perf_target", "get_default_regime_specs",
    "get_operator_entrypoint", "get_operator_module_relpath", "get_operator_task",
    "infer_module_runtime_backend", "load_cannbench_contract_module", "parse_simple_env_file",
    "resolve_cannbench_repo", "resolve_cannbench_tasks_root", "semantic_regime_representative",
    "semantic_regime_tag",
]
