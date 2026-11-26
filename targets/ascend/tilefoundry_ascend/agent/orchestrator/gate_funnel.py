"""Three-stage verification funnel: Stage A static ledger -> Stage B bisheng -> Stage C NPU."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    extract_reachable_trackb_source,
    strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore,
    collect_new_tl_cache_binaries,
    compute_compile_task_key,
    extract_reachable_closure_source,
    is_verified_binary_file,
    merge_tl_cache_tree,
    resolve_unified_tl_cache_dir,
    seed_tl_cache_tree,
    snapshot_tl_cache_binaries,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary,
    count_reachable_ub_scalar_compute_loops,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    MicroarchProfile,
    extract_microarch_profile_from_cce,
)
from tilefoundry_ascend.ascend.passes.observation_space import RooflineSummaryLike
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    EnvelopeInspectionResult,
    inspect_tilelang_primfunc,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios
from tilefoundry_ascend.ascend.sil.verifier import verify_tilelang_ast
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS, AscendTarget
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE


@dataclass(frozen=True)
class StageAResult:
    """Output of millisecond Stage A static AST, ledger, and router inspection."""

    passed: bool
    prim_func: object | None
    ledger: ObjectivePhysicalLedger | None
    features: dict[str, object] | None = None
    route: object | None = None
    error_message: str = ""
    elapsed_ms: float = 0.0
    backend_kind: str = "tilelang"
    fault_plane: str = ""
    fault_code: str = ""
    failed_tier: str = ""


@dataclass(frozen=True)
class StageBResult:
    """Output of Stage B Conda bisheng compilation check."""

    passed: bool
    error_message: str = ""
    elapsed_sec: float = 0.0
    spill_bytes: int = 0
    vloopv2_hit: bool = True
    compiler_summary: BishengResourceSummary | None = None
    microarch_profile: dict[str, object] | None = None
    full_diagnostic: str = ""
    generated_cce: str = ""
    lowered_tir: str = ""
    fault_plane: str = ""
    fault_code: str = ""
    failed_tier: str = ""
    artifact_id: str = ""
    binary_paths: tuple[str, ...] = ()


def _resolve_artifact_store(artifact_store: ArtifactStore | None) -> ArtifactStore:
    if artifact_store is not None:
        return artifact_store
    return ArtifactStore(resolve_unified_tl_cache_dir())


def _classify_stage_b_overflow_fault_code(
    combined_diag: str,
    summary: BishengResourceSummary | None,
    default_code: str,
) -> str:
    low = (combined_diag or "").lower()
    if summary is not None and int(summary.stack_bytes) > 0:
        return "VF_STACK_OVERFLOW"
    if any(
        tok in low
        for tok in (
            "exceeded vf stack size",
            "vf_stack",
            "vf stack",
            "stack frame size",
            "stack overflow",
        )
    ):
        return "VF_STACK_OVERFLOW"
    cap_tokens = ("overflow", "exceed", "capacity", "out of memory", "insufficient")
    if any(
        tok in low
        for tok in (
            "l0c_sram_overflow",
            "l0c overflow",
            "l0c_overflow",
            "l0c capacity",
            "exceeds l0c",
            "exceeded l0c",
        )
    ) or (("l0c" in low or "__cc__" in low) and any(cap in low for cap in cap_tokens)):
        return "L0C_SRAM_OVERFLOW"
    if any(
        tok in low
        for tok in (
            "l1_sram_overflow",
            "l1 overflow",
            "l1_overflow",
            "cbuf overflow",
            "l1 capacity",
            "cbuf capacity",
            "exceeds l1",
            "exceeded l1",
        )
    ) or (
        ("l1 " in low or "l1_" in low or "cbuf" in low or "__cbuf__" in low)
        and any(cap in low for cap in cap_tokens)
    ):
        return "L1_SRAM_OVERFLOW"
    if any(
        tok in low
        for tok in (
            "ub_sram_overflow",
            "ub_overflow",
            "ubuf_overflow",
            "ub overflow",
            "ubuf overflow",
            "ub capacity",
            "ubuf capacity",
            "exceeds ub",
            "exceeded ub",
            "exceeds ubuf",
            "exceeded ubuf",
        )
    ) or (
        ("ubuf" in low or "__ubuf__" in low or re.search(r"\bub\b", low) is not None)
        and any(cap in low for cap in cap_tokens)
    ):
        return "UB_SRAM_OVERFLOW"
    return default_code


def _uses_unreserved_ubuf_flags(
    compile_flags: Sequence[str] | str | None = None,
    pass_configs: Mapping[str, object] | None = None,
) -> bool:
    flags = _normalize_flag_list(compile_flags)
    if pass_configs and "tl.device_compile_flags" in pass_configs:
        flags.extend(_normalize_flag_list(pass_configs["tl.device_compile_flags"]))
    return any("--cce-disable-vf-stack-reserved-ubuf" in fl for fl in flags)


def _has_vf_stack_overflow_signal(
    combined_diag: str,
    summary: BishengResourceSummary | None,
    compile_flags: Sequence[str] | str | None = None,
    pass_configs: Mapping[str, object] | None = None,
) -> bool:
    low = (combined_diag or "").lower()
    if any(tok in low for tok in ("exceeded vf stack size", "vf_stack", "stack overflow")):
        return True
    if summary is not None and int(summary.stack_bytes) > 0:
        return _uses_unreserved_ubuf_flags(compile_flags, pass_configs)
    return False


def _normalize_trackb_unit_source(cpp_source: str, *, regime_tag: str = "") -> str:
    stripped = cpp_source.strip()
    if not regime_tag or not stripped:
        return stripped
    masked = strip_cpp_comments_and_strings(stripped)
    kernel_count = len(re.findall(r"\b__global__\b", masked))
    if kernel_count > 1 and "<<<" not in masked:
        _kname, extracted = extract_reachable_trackb_source(stripped, regime_tag=regime_tag)
        return extracted.strip()
    return stripped


def _classify_ascendc_stage_a_fault(errors: Sequence[str]) -> tuple[str, str, str]:
    joined = "; ".join(errors).lower()
    if "cce_forbidden_tokens" in joined:
        return "STAGE_A_STATIC", "CCE_FORBIDDEN_TOKENS", "stage_a"
    if "max_overflow_bytes" in joined:
        return "STAGE_A_STATIC", "UB_OVERFLOW", "stage_a"
    if "missing_cube_engine_primitives" in joined:
        return "STAGE_A_STATIC", "MISSING_CUBE_ENGINE", "stage_a"
    if "missing_hardware_dma_staging" in joined:
        return "STAGE_A_STATIC", "MISSING_HARDWARE_DMA", "stage_a"
    if "degenerated_single_lane_simt" in joined:
        return "STAGE_A_STATIC", "DEGENERATED_SINGLE_LANE_SIMT", "stage_a"
    if "redline" in joined:
        return "STAGE_A_STATIC", "CUSTOM_KERNEL_REDLINE", "stage_a"
    return "STAGE_A_STATIC", "STAGE_A_ASCENDC_STATIC_FAILED", "stage_a"


@dataclass(frozen=True)
class StageCResult:
    """Output of Stage C lease-locked AgentGates NPU verification and msprof telemetry."""

    passed: bool
    accuracy_passed: bool
    measured_latency_us: float
    residual: StaticDynamicResidual | None
    telemetry: dict[str, object]
    run_id: str = ""
    receipt_path: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class FunnelEvaluation:
    """Aggregate result across Stage A, Stage B, and Stage C verification."""

    passed: bool
    failed_stage: str | None
    stage_a: StageAResult
    stage_b: StageBResult | None = None
    stage_c: StageCResult | None = None
    error_message: str = ""


_HIR_TO_BUILDER_DIM_MAP: dict[str, str] = {
    "N": "n_depth",
    "B": "b_dim",
    "S": "s_dim",
    "S1": "s1_dim",
    "S2": "s2_dim",
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N1": "n_q",
    "N_kv": "n_kv",
    "N2": "n_kv",
    "G": "g_ratio",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
    "M": "m_dim",
    "K": "k_dim",
    "H": "num_heads",
    "E": "num_experts",
    "topK": "top_k",
    "topk": "top_k",
    "sparse_size": "top_k",
    "T": "t_dim",
}

_PARAM_DIM_FALLBACK_ALIASES: dict[str, tuple[str, ...]] = {
    "m": ("M", "m_dim", "m_rows"),
    "m_dim": ("M", "m", "m_rows"),
    "n": ("N", "n_dim", "n_depth"),
    "n_dim": ("N", "n"),
    "k": ("K", "k_dim"),
    "k_dim": ("K", "k"),
    "d": ("D", "d_dim", "hidden_dim"),
    "d_dim": ("D", "d", "hidden_dim"),
    "b": ("B", "b_dim", "batch"),
    "b_dim": ("B", "b", "batch"),
    "s": ("S", "s_dim", "seq_len"),
    "s_dim": ("S", "s", "seq_len"),
    "s1_dim": ("S1", "S", "s_dim"),
    "s2_dim": ("S2", "S_kv", "s_kv"),
    "n_q": ("N_q", "N1", "num_heads"),
    "n_kv": ("N_kv", "N2"),
    "top_k": ("topK", "topk", "sparse_size", "k_sparse"),
    "t_dim": ("T", "S", "s_dim"),
}


def _normalize_flag_list(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        text = raw_flags.strip()
        return text.split() if text else []
    if isinstance(raw_flags, (tuple, list)) or (
        hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray, Mapping))
    ):
        out: list[str] = []
        for item in raw_flags:
            val = str(getattr(item, "value", item)).strip()
            if val and val not in out:
                out.append(val)
        return out
    text = str(raw_flags).strip()
    return [text] if text else []


def resolve_trackb_compile_flags(
    cpp_source: str,
    selector_flags: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Resolve Track-B bisheng flags, stripping SIMD-VF UBUF flags on __simt_vf__."""
    masked = strip_cpp_comments_and_strings(cpp_source)
    has_simt_vf = bool(re.search(r"\b__simt_vf__\b", masked))
    normalized = _normalize_flag_list(selector_flags)
    if has_simt_vf:
        simdvf_flags = frozenset(PURE_SIMDVF_UBUF_FLAGS)
        return tuple(flag for flag in normalized if flag not in simdvf_flags)
    flags = list(PURE_SIMDVF_UBUF_FLAGS)
    for flag in normalized:
        if flag not in flags:
            flags.append(flag)
    return tuple(flags)


_FORBIDDEN_NPU_HOST_CALLS: frozenset[str] = frozenset(
    {
        "matmul",
        "bmm",
        "einsum",
        "rms_norm",
        "softmax",
        "silu",
        "layer_norm",
        "scaled_dot_product_attention",
    }
)
_CTYPES_LIB_RECEIVERS: frozenset[str] = frozenset(
    {"lib", "_lib", "so_lib", "dll", "cdll", "handle"}
)
_POLLUTED_FLOAT_KEY_NAMES: frozenset[str] = frozenset(
    {
        "eps",
        "epsilon",
        "scale",
        "softmax_scale",
        "sm_scale",
        "alpha",
        "beta",
        "dropout_p",
        "inv_scale",
        "temperature",
    }
)
_POLLUTED_ADDR_ATTRS: frozenset[str] = frozenset({"data_ptr", "storage_offset"})
_TRACKA_CORE_PRIMITIVES: frozenset[str] = frozenset(
    {"SimdVF", "SimtVF", "gemm", "gemm_v0", "ascend_mmad", "ascend_gemm", "ascend_vec"}
)
_TRACKB_MONOTONICITY_BACKENDS: frozenset[str] = frozenset(
    {"ascendc_trackb", "hybrid_trackb", "cce_trackb", "ascendc"}
)
_TRACKB_CORE_TOKENS: tuple[str, ...] = (
    "simd_inst::",
    "asc_mmad",
    "ascend_gemm_l1",
    "__simd_vf__",
    "__simt_vf__",
)


def _is_non_npu_guard(cond_txt: str) -> bool:
    compact = cond_txt.replace(" ", "")
    if "device.type" in cond_txt and (
        "!=" in cond_txt or '=="cpu"' in compact or "=='cpu'" in compact
    ):
        return True
    return "is_npu" in cond_txt and (
        "not " in cond_txt or "==False" in compact or "isFalse" in compact
    )


def _is_npu_positive_guard(cond_txt: str) -> bool:
    compact = cond_txt.replace(" ", "")
    if "device.type" in cond_txt and ('=="npu"' in compact or "=='npu'" in compact):
        return True
    return "is_npu" in cond_txt and not _is_non_npu_guard(cond_txt)


def _collect_npu_reachable_nodes(
    module_source: str,
    stmts: Sequence[ast.stmt],
) -> list[ast.AST]:
    """Return AST nodes on reachable NPU execution paths, skipping non-NPU fallback branches."""
    collected: list[ast.AST] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            cond_txt = ast.get_source_segment(module_source, stmt.test) or ""
            collected.append(stmt.test)
            if _is_non_npu_guard(cond_txt):
                collected.extend(_collect_npu_reachable_nodes(module_source, stmt.orelse))
                continue
            if _is_npu_positive_guard(cond_txt):
                collected.extend(_collect_npu_reachable_nodes(module_source, stmt.body))
                continue
            collected.extend(_collect_npu_reachable_nodes(module_source, stmt.body))
            collected.extend(_collect_npu_reachable_nodes(module_source, stmt.orelse))
            continue
        collected.append(stmt)
    return collected


def _detect_module_entrypoint(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == "kernel"
                    and isinstance(node.value, ast.Name)
                ):
                    return node.value.id
    return ""


def _has_module_kernel_cache(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.value is not None:
            if any(isinstance(t, ast.Name) and t.id == "_KERNEL_CACHE" for t in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == "_KERNEL_CACHE":
                return True
    return False


def _collect_local_var_assigns(fnode: ast.FunctionDef) -> dict[str, list[ast.expr]]:
    assigns: dict[str, list[ast.expr]] = {}
    for sub in ast.walk(fnode):
        if isinstance(sub, ast.Assign) and sub.value is not None:
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name):
                    assigns.setdefault(tgt.id, []).append(sub.value)
        elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
            if isinstance(sub.target, ast.Name):
                assigns.setdefault(sub.target.id, []).append(sub.value)
    return assigns


def _extract_kernel_cache_key_exprs(fnode: ast.FunctionDef) -> list[ast.expr]:
    exprs: list[ast.expr] = []
    for sub in ast.walk(fnode):
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "_KERNEL_CACHE"
        ):
            exprs.append(sub.slice)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "_KERNEL_CACHE"
            and sub.func.attr in ("get", "setdefault", "pop")
        ):
            if sub.args:
                exprs.append(sub.args[0])
            else:
                for kw in sub.keywords:
                    if kw.arg == "key":
                        exprs.append(kw.value)
        elif isinstance(sub, ast.Compare):
            for op, comp in zip(sub.ops, sub.comparators):
                if (
                    isinstance(op, (ast.In, ast.NotIn))
                    and isinstance(comp, ast.Name)
                    and comp.id == "_KERNEL_CACHE"
                ):
                    exprs.append(sub.left)
    return exprs


def _find_polluted_cache_key_reason(
    expr: ast.expr,
    local_assigns: Mapping[str, list[ast.expr]],
    seen_vars: frozenset[str] = frozenset(),
) -> str | None:
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            return f"float_literal({node.value!r})"
        if isinstance(node, ast.Name):
            if node.id.lower() in _POLLUTED_FLOAT_KEY_NAMES:
                return f"float_param({node.id})"
            if node.id in local_assigns and node.id not in seen_vars:
                next_seen = seen_vars | {node.id}
                for rhs in local_assigns[node.id]:
                    nested = _find_polluted_cache_key_reason(rhs, local_assigns, next_seen)
                    if nested is not None:
                        return nested
        elif isinstance(node, ast.Attribute):
            if node.attr.lower() in _POLLUTED_FLOAT_KEY_NAMES:
                return f"float_attr({node.attr})"
        elif isinstance(node, ast.Subscript):
            slc = node.slice
            if (
                isinstance(slc, ast.Constant)
                and isinstance(slc.value, str)
                and slc.value.lower() in _POLLUTED_FLOAT_KEY_NAMES
            ):
                return f"float_key({slc.value!r})"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "float":
                    return "float_cast(float(...))"
                if node.func.id == "id":
                    return "object_identity(id(...))"
                if node.func.id == "addressof":
                    return "pointer_address(addressof(...))"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _POLLUTED_ADDR_ATTRS:
                    return f"pointer_attr(.{node.func.attr}())"
                if node.func.attr == "addressof":
                    return "pointer_address(ctypes.addressof(...))"
    return None


def _check_kernel_cache_key_pollution(
    funcs: Mapping[str, ast.FunctionDef],
    cache_fns: set[str],
) -> list[str]:
    """Reject float hyperparameters or runtime pointer/id values in _KERNEL_CACHE keys."""
    errors: list[str] = []
    for fname in sorted(cache_fns):
        if fname not in funcs:
            continue
        fnode = funcs[fname]
        local_assigns = _collect_local_var_assigns(fnode)
        for key_expr in _extract_kernel_cache_key_exprs(fnode):
            reason = _find_polluted_cache_key_reason(key_expr, local_assigns)
            if reason is None:
                continue
            msg = (
                f"polluted_kernel_cache_key: {fname} key uses {reason} "
                "(pass float hyperparameters as T.float32 scalar kernel params "
                "and dynamic outer axes via T.dynamic)"
            )
            if msg not in errors:
                errors.append(msg)
    return errors


def _has_tracka_core_primitives(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            val = node.value
            if (
                isinstance(val, ast.Name)
                and val.id == "T"
                and node.attr in _TRACKA_CORE_PRIMITIVES
            ):
                return True
            if (
                isinstance(val, ast.Attribute)
                and isinstance(val.value, ast.Name)
                and val.value.id == "T"
                and val.attr == "simd"
            ):
                return True
    return False


def _extract_tracka_pipelined_stages(tree: ast.AST) -> list[int | None]:
    stages: list[int | None] = []
    for node in ast.walk(tree):
        if not _is_t_dsl_call(node, "Pipelined"):
            continue
        assert isinstance(node, ast.Call)
        stage_expr: ast.expr | None = None
        for kw in node.keywords:
            if kw.arg == "num_stages":
                stage_expr = kw.value
                break
        if stage_expr is None and len(node.args) >= 3:
            stage_expr = node.args[2]
        if stage_expr is None:
            stages.append(0)
        elif (
            isinstance(stage_expr, ast.Constant)
            and isinstance(stage_expr.value, int)
            and not isinstance(stage_expr.value, bool)
        ):
            stages.append(int(stage_expr.value))
        else:
            stages.append(None)
    return stages


def check_repair_monotonicity(
    parent_source: str,
    repaired_source: str,
    *,
    backend_kind: str = "tilelang_dsl",
) -> list[str]:
    """Reject repairs that regress vector/cube primitives or drop T.Pipelined stages."""
    if backend_kind in _TRACKB_MONOTONICITY_BACKENDS:
        masked_parent = strip_cpp_comments_and_strings(parent_source)
        masked_repaired = strip_cpp_comments_and_strings(repaired_source)
        had_core = any(tok in masked_parent for tok in _TRACKB_CORE_TOKENS)
        has_core = any(tok in masked_repaired for tok in _TRACKB_CORE_TOKENS)
        if had_core and not has_core:
            return [
                "monotonicity_regression_trackb_scalar_fallback: repair dropped all "
                "Track-B vector/cube primitives"
            ]
        return []

    try:
        parent_tree = ast.parse(parent_source)
    except SyntaxError:
        return []
    try:
        repaired_tree = ast.parse(repaired_source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]

    errors: list[str] = []
    if _has_tracka_core_primitives(parent_tree) and not _has_tracka_core_primitives(repaired_tree):
        errors.append(
            "monotonicity_regression_scalar_fallback: repair dropped all "
            "vector/matrix core primitives (T.SimdVF/T.SimtVF/T.gemm/T.simd.*)"
        )
    parent_stages = _extract_tracka_pipelined_stages(parent_tree)
    if any(s is not None and s >= 1 for s in parent_stages):
        repaired_stages = _extract_tracka_pipelined_stages(repaired_tree)
        if not repaired_stages:
            errors.append(
                "monotonicity_regression_pipeline_dropped: repair removed T.Pipelined"
            )
        elif all(s == 0 for s in repaired_stages):
            errors.append(
                "monotonicity_regression_pipeline_disabled: repair set T.Pipelined num_stages=0"
            )
    return errors


def check_npu_entrypoint_purity(
    module_source: str,
    entrypoint: str,
    *,
    track: str | None = None,
    require_kernel_cache: bool = False,
) -> list[str]:
    """Reject detached decoy @T.prim_func scaffolds and NPU-side PyTorch/golden fallbacks."""
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entrypoint not in funcs:
        return []

    from tilefoundry_ascend.agent.campaign.anchor_specs import (  # noqa: PLC0415
        infer_module_runtime_backend,
    )

    errors: list[str] = []
    backend, _tb_syms = infer_module_runtime_backend(module_source, entrypoint)
    reachable_src: list[str] = []
    visited: set[str] = set()
    cache_loaded = False
    cache_stored = False
    queue = [entrypoint]
    while queue:
        fn_name = queue.pop(0)
        if fn_name in visited or fn_name not in funcs:
            continue
        visited.add(fn_name)
        fnode = funcs[fn_name]
        reachable_src.append(ast.get_source_segment(module_source, fnode) or "")
        for node in _collect_npu_reachable_nodes(module_source, fnode.body):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "_KERNEL_CACHE":
                    if isinstance(sub.ctx, ast.Load):
                        cache_loaded = True
                elif (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "_KERNEL_CACHE"
                ):
                    cache_stored = True
                if isinstance(sub, ast.Call):
                    if (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "_KERNEL_CACHE"
                        and sub.func.attr in ("setdefault", "update")
                    ):
                        cache_stored = True
                    callee = (
                        sub.func.id
                        if isinstance(sub.func, ast.Name)
                        else (sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
                    )
                    is_local_fn = isinstance(sub.func, ast.Name) and sub.func.id in funcs
                    is_ctypes_lib = (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id.lower() in _CTYPES_LIB_RECEIVERS
                    )
                    if (
                        callee in _FORBIDDEN_NPU_HOST_CALLS
                        and not is_local_fn
                        and not is_ctypes_lib
                    ):
                        errors.append(f"npu_host_fallback_forbidden:{fn_name}->{callee}")
                    if is_local_fn:
                        assert isinstance(sub.func, ast.Name)
                        queue.append(sub.func.id)
                elif (
                    isinstance(sub, ast.Name)
                    and isinstance(sub.ctx, ast.Load)
                    and sub.id in funcs
                ):
                    queue.append(sub.id)

    for fname, fnode in funcs.items():
        seg = ast.get_source_segment(module_source, fnode) or ""
        if "@T.prim_func" in seg and fname not in visited:
            errors.append(
                f"unreachable_decoy_primfunc:{fname} is never reachable from {entrypoint}"
            )
    reachable_text = "\n".join(reachable_src)
    if track in ("cce_trackb", "ascendc_trackb", "hybrid_trackb"):
        has_trackb_dispatch = (
            backend != "tilelang_dsl"
            or "compile_ascend" in reachable_text
            or "ctypes.CDLL" in reachable_text
            or "_render_trackb_tu" in reachable_text
            or "_get_track_b_lib" in reachable_text
            or "tilelang.compile(" in reachable_text
        )
        if not has_trackb_dispatch:
            errors.append(
                f"missing_trackb_dispatch:{entrypoint} exposes no reachable Track-B C++20 "
                "translation unit or compile_ascend dispatch"
            )
    elif backend == "tilelang_dsl" and "tilelang.compile(" not in reachable_text:
        errors.append(
            f"detached_decoy_primfunc:{entrypoint} never invokes tilelang.compile"
        )
    has_cache_dict = _has_module_kernel_cache(tree)
    if require_kernel_cache or has_cache_dict or cache_loaded or cache_stored:
        cache_fns = {
            fname
            for fname, fnode in funcs.items()
            if fname in visited
            or fname in ("_get_kernel", "_get_track_b_lib")
            or any(
                tok in (ast.get_source_segment(module_source, fnode) or "")
                for tok in ("tilelang.compile", "compile_ascend", "ctypes.CDLL")
            )
        }
        errors.extend(_check_kernel_cache_key_pollution(funcs, cache_fns))
        if require_kernel_cache:
            cache_nodes = [
                sub
                for fname in cache_fns
                if fname in funcs
                for sub in ast.walk(funcs[fname])
            ]
            any_cache_loaded = cache_loaded or any(
                isinstance(sub, ast.Name)
                and sub.id == "_KERNEL_CACHE"
                and isinstance(sub.ctx, ast.Load)
                for sub in cache_nodes
            )
            any_cache_stored = cache_stored or any(
                (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "_KERNEL_CACHE"
                )
                or (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "_KERNEL_CACHE"
                    and sub.func.attr in ("setdefault", "update")
                )
                for sub in cache_nodes
            )
            compile_path_text = "\n".join(
                ast.get_source_segment(module_source, funcs[fname]) or ""
                for fname in cache_fns
                if fname in funcs
            )
            has_compile_call = any(
                tok in compile_path_text
                for tok in ("tilelang.compile", "compile_ascend", "ctypes.CDLL")
            )
            if not has_cache_dict:
                errors.append(
                    "missing_kernel_cache: module must define module-level _KERNEL_CACHE dict"
                )
            elif not any_cache_loaded or (has_compile_call and not any_cache_stored):
                errors.append(
                    "unpopulated_kernel_cache: reachable compilation path must look up "
                    "and populate _KERNEL_CACHE"
                )
    return errors


def _filter_kwargs_for_callable(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    import inspect  # noqa: PLC0415

    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    lowered_kw = {str(k).lower(): v for k, v in kwargs.items()}
    for pname in sig.parameters:
        if pname in filtered:
            continue
        if pname.lower() in lowered_kw:
            filtered[pname] = lowered_kw[pname.lower()]
            continue
        for alias in _PARAM_DIM_FALLBACK_ALIASES.get(pname.lower(), ()):
            if alias in kwargs:
                filtered[pname] = kwargs[alias]
                break
            if alias.lower() in lowered_kw:
                filtered[pname] = lowered_kw[alias.lower()]
                break
    return filtered


def _merge_dims_into_kwargs(
    base_kwargs: Mapping[str, object] | None,
    dims: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(base_kwargs or {})
    if not dims:
        return merged
    for key, val in dims.items():
        key_str = str(key)
        merged.setdefault(key_str, val)
        mapped = _HIR_TO_BUILDER_DIM_MAP.get(key_str, key_str)
        merged[mapped] = val
    return merged


def _split_selector_output(
    sel_out: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    pc: dict[str, object] = {}
    cf: list[str] = []
    builder_overrides: dict[str, object] = {}

    raw_pc = sel_out.get("pass_configs")
    if isinstance(raw_pc, Mapping):
        for k, v in raw_pc.items():
            if str(k) == "tl.device_compile_flags":
                for flag in _normalize_flag_list(v):
                    if flag not in cf:
                        cf.append(flag)
            else:
                pc[str(k)] = v
    elif raw_pc is not None:
        for flag in _normalize_flag_list(raw_pc):
            if flag not in cf:
                cf.append(flag)

    for flag in _normalize_flag_list(sel_out.get("compile_flags")):
        if flag not in cf:
            cf.append(flag)

    for k, v in sel_out.items():
        if k not in ("pass_configs", "compile_flags", "builder_kwargs"):
            builder_overrides[str(k)] = v
    raw_bk = sel_out.get("builder_kwargs")
    if isinstance(raw_bk, Mapping):
        for k, v in raw_bk.items():
            builder_overrides[str(k)] = v
    return builder_overrides, pc, cf


@contextlib.contextmanager
def ascend_compile_session(
    target: str = "ascend",
    pass_configs: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Enter contiguous TVM target and optional PassContext for Ascend lowering."""
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415

    with contextlib.ExitStack() as stack:
        if tvm.target.Target.current(allow_none=True) is None:
            stack.enter_context(create_backend_context(target).target)
        if pass_configs is not None:
            stack.enter_context(tvm.transform.PassContext(config=dict(pass_configs)))
        yield


def _ascend_target_context() -> contextlib.AbstractContextManager[None]:
    return ascend_compile_session("ascend")


_VLOOPV2_MISS_RE = re.compile(
    r"(?:vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|"
    r"vloop_?v2\s*[:=]\s*(?:false|0|no|miss)|cannot\s+use\s+vloop_?v2)",
    re.IGNORECASE,
)

_COMPILE_DIAGNOSTIC_LIMIT = 1500
_COMPILE_ERROR_CONTEXT_LINES = 6
_COMPILE_ERROR_RE = re.compile(
    r"(?i)(?:\[error\]|\btvm\s*error\b|\bfatal\s*:|\bCheck\s+failed:|"
    r"(?:^|[\s:(])(?:Assertion|Type|Value|Runtime|Internal|Key|Attribute|Name|Syntax)Error:|"
    r"(?:^|[\s:\[(])(?:fatal\s+)?error:)"
)
_DEPRECATED_WARN_RE = re.compile(
    r"(?:warning:\s+.*\[-Wdeprecated-declarations\]|"
    r"note:\s+'[^']+'\s+has been explicitly marked deprecated here|"
    r"note:\s+expanded from macro 'ASC_DEPRECATED')"
)
_NON_ERROR_DIAG_HEADER_RE = re.compile(
    r"^(?:note:|warning:|in file included from|/\S+:\d+:\d+:\s*(?:warning|note):)",
    re.IGNORECASE,
)
_TVM_BT_RE = re.compile(r"^\s*(?:\[bt\]\s*\(\d+\)|File \"<unknown>\", line 0, in )")


def _strip_diagnostic_noise(lines: Sequence[str]) -> list[str]:
    kept: list[str] = []
    skip_cascade = False
    skip_tvm_bt = False
    for line in lines:
        if _COMPILE_ERROR_RE.search(line):
            skip_cascade = False
            skip_tvm_bt = False
            kept.append(line)
            continue
        if _DEPRECATED_WARN_RE.search(line):
            skip_cascade = True
            skip_tvm_bt = False
            continue
        if line.strip() == "Stack trace:":
            skip_tvm_bt = True
            skip_cascade = False
            continue
        if skip_tvm_bt:
            if _TVM_BT_RE.search(line) or not line.strip():
                continue
            skip_tvm_bt = False
        if skip_cascade:
            stripped = line.strip()
            if not stripped:
                skip_cascade = False
                continue
            if _NON_ERROR_DIAG_HEADER_RE.match(stripped):
                skip_cascade = False
            else:
                continue
        kept.append(line)
    return kept


def _bounded_compile_diagnostic(
    output_text: str,
    *,
    limit: int = _COMPILE_DIAGNOSTIC_LIMIT,
) -> str:
    """Keep primary compiler errors without deprecation cascades plus a bounded output tail."""
    text = output_text.strip()
    if len(text) <= limit:
        return text
    raw_lines = text.splitlines()
    filtered_lines = _strip_diagnostic_noise(raw_lines)
    use_filtered = bool(filtered_lines) and any(
        _COMPILE_ERROR_RE.search(line) for line in filtered_lines
    )
    lines = filtered_lines if use_filtered else raw_lines
    effective_text = "\n".join(lines).strip()
    if len(effective_text) <= limit:
        return effective_text
    error_indices = [
        idx for idx, line in enumerate(lines) if _COMPILE_ERROR_RE.search(line)
    ]
    tail = effective_text[-limit:]
    if not error_indices:
        return tail
    error_idx = error_indices[0]
    error_line = lines[error_idx].strip()
    if error_line in tail and len(error_indices) == 1:
        return tail
    context_end = min(len(lines), error_idx + 1 + _COMPILE_ERROR_CONTEXT_LINES)
    header_lines = [
        error_line,
        *(line.rstrip() for line in lines[error_idx + 1 : context_end]),
    ]
    seen_end = context_end
    extra_blocks = 0
    for other_idx in error_indices[1:]:
        if other_idx < seen_end or extra_blocks >= 2:
            continue
        other_line = lines[other_idx].strip()
        if other_line in tail or "clang frontend command failed" in other_line.lower():
            continue
        other_end = min(len(lines), other_idx + 3)
        header_lines.append(other_line)
        header_lines.extend(line.rstrip() for line in lines[other_idx + 1 : other_end])
        seen_end = other_end
        extra_blocks += 1
    header = "\n".join(header_lines)
    prefix = f"{header}\n[... compiler output truncated ...]\n"
    if len(prefix) >= limit:
        return header[:limit]
    return prefix + effective_text[-(limit - len(prefix)) :]


def _parse_cce_res_usage_with_summary(
    output_text: str,
    *,
    asm_or_dump_text: str = "",
) -> tuple[int, bool, BishengResourceSummary]:
    """Parse bisheng diagnostics and lowered/disassembled code via bisheng_res_parser."""
    summary = parse_bisheng_res_usage(
        output_text or "",
        asm_or_dump_text=asm_or_dump_text or "",
    )
    combined_vloop_text = f"{output_text or ''}\n{asm_or_dump_text or ''}"
    vloopv2_hit = (
        summary.vloopv2_loops > 0
        and summary.software_loops == 0
        and not bool(_VLOOPV2_MISS_RE.search(combined_vloop_text))
    )
    return int(summary.spill_bytes), vloopv2_hit, summary


def parse_cce_res_usage(output_text: str) -> tuple[int, bool]:
    """Parse bisheng --cce-res-usage diagnostics into (spill_bytes, vloopv2_hit)."""
    spill_bytes, vloopv2_hit, _summary = _parse_cce_res_usage_with_summary(output_text)
    return spill_bytes, vloopv2_hit


def _has_prim_func_marker(dsl_source: str) -> bool:
    return "prim_func" in dsl_source


def _resolve_custom_kernel_callable(
    mod: object,
    factory_name: str | None,
) -> Callable[..., object] | None:
    if factory_name and hasattr(mod, factory_name):
        obj = getattr(mod, factory_name)
        if callable(obj):
            return obj
    kernel_fn = getattr(mod, "kernel", None)
    if callable(kernel_fn):
        return kernel_fn
    return None


_SIMDVF_LOCAL_ARRAY_RE = re.compile(
    r"\b(?:float|half|bfloat16_t|u?int(?:8|16|32|64)_t|bool)\s+[A-Za-z_]\w*\s*\[[^\]]+\]"
)


def _simdvf_function_bodies(cpp_source: str) -> list[str]:
    """Return brace-matched bodies of every ``__simd_vf__`` function definition."""
    masked = strip_cpp_comments_and_strings(cpp_source)
    bodies: list[str] = []
    for match in re.finditer(r"__simd_vf__", masked):
        brace = masked.find("{", match.end())
        if brace < 0:
            continue
        if masked.find(";", match.end(), brace) >= 0:
            continue
        depth = 0
        for idx in range(brace, len(masked)):
            char = masked[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(masked[brace + 1 : idx])
                    break
    return bodies


_ALIGNED_COPY_ARITIES: dict[str, tuple[tuple[int, ...], str]] = {
    "asc_copy_gm2ub_align": (
        (3, 8, 10),
        "invalid_asc_copy_align_arity: expected 3/8/10 args for gm2ub_align",
    ),
    "asc_copy_ub2gm_align": (
        (3, 7, 8),
        "invalid_asc_copy_align_arity: expected 3/7/8 args for ub2gm_align",
    ),
}


def _extract_balanced_paren_arg_list(text: str, open_paren_idx: int) -> str | None:
    """Return the top-level text inside the balanced parentheses starting at *open_paren_idx*."""
    depth = 0
    for idx in range(open_paren_idx, len(text)):
        ch = text[idx]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : idx]
    return None


def _split_top_level_call_args(inner: str) -> list[str]:
    """Split a function call argument string by top-level commas without mistaking '<'."""
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(inner):
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:idx].strip())
            start = idx + 1
    parts.append(inner[start:].strip())
    return [part for part in parts if part]


def _collect_global_kernel_names(masked_source: str) -> set[str]:
    """Collect function identifiers declared with ``__global__``, directly or via macros."""
    globals_set: set[str] = set()
    for match in re.finditer(r"\b__global__\b", masked_source):
        tail = masked_source[match.end() : match.end() + 512]
        stmt_end = len(tail)
        for sep in ("{", ";"):
            pos = tail.find(sep)
            if pos >= 0:
                stmt_end = min(stmt_end, pos)
        header = tail[:stmt_end]
        paren_depth = 0
        for idx, ch in enumerate(header):
            if ch == "(":
                if paren_depth == 0:
                    prefix = header[:idx].rstrip()
                    ident = re.search(r"([A-Za-z_]\w*)\s*$", prefix)
                    if ident is not None:
                        name = ident.group(1)
                        if name not in ("__mix__", "__launch_bounds__", "decltype"):
                            globals_set.add(name)
                            break
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1

    joined = re.sub(r"\\\r?\n", " ", masked_source)
    macro_specs: list[tuple[str, tuple[str, ...], str]] = []
    non_define_lines: list[str] = []
    define_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s+(.*)$")
    for raw_line in joined.splitlines():
        m_def = define_re.match(raw_line)
        if m_def is None:
            non_define_lines.append(raw_line)
            continue
        macro_name, raw_params, body = m_def.group(1), m_def.group(2), m_def.group(3)
        global_m = re.search(r"\b__global__\b", body)
        if global_m is None:
            continue
        after_global = body[global_m.end() :]
        paren_depth = 0
        name_expr = ""
        for idx, ch in enumerate(after_global):
            if ch == "(":
                if paren_depth == 0:
                    prefix = after_global[:idx].rstrip()
                    expr_m = re.search(
                        r"([A-Za-z_]\w*(?:\s*##\s*[A-Za-z_]\w*)*)\s*$",
                        prefix,
                    )
                    if expr_m is not None:
                        cand_expr = expr_m.group(1).strip()
                        if cand_expr not in ("__mix__", "__launch_bounds__", "decltype"):
                            name_expr = cand_expr
                            break
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1
        if name_expr:
            params = tuple(p.strip() for p in raw_params.split(",") if p.strip())
            if params:
                macro_specs.append((macro_name, params, name_expr))

    if macro_specs:
        non_define_src = "\n".join(non_define_lines)
        for macro_name, params, name_expr in macro_specs:
            expr_tokens = re.split(r"\s*##\s*", name_expr)
            for inv_m in re.finditer(rf"\b{re.escape(macro_name)}\s*\(", non_define_src):
                inner = _extract_balanced_paren_arg_list(non_define_src, inv_m.end() - 1)
                if inner is None:
                    continue
                args = _split_top_level_call_args(inner)
                if len(args) < len(params):
                    continue
                param_map = dict(zip(params, args))
                expanded = "".join(param_map.get(tok, tok) for tok in expr_tokens)
                if re.fullmatch(r"[A-Za-z_]\w*", expanded):
                    globals_set.add(expanded)
    return globals_set


def _strip_cpp_cast_and_parens(expr: str) -> str:
    text = expr.strip()
    while text:
        if text.startswith("("):
            close_idx = -1
            depth = 0
            for idx, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = idx
                        break
            if close_idx > 0:
                inner = text[1:close_idx].strip()
                rest = text[close_idx + 1 :].strip()
                if not rest:
                    text = inner
                    continue
                if "*" in inner or "__gm__" in inner or re.fullmatch(
                    r"(?:const\s+|volatile\s+)*(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*",
                    inner,
                ):
                    text = rest
                    continue
        cast_m = re.match(r"^(?:static|reinterpret|const)_cast\s*<[^>]+>\s*\(", text)
        if cast_m is not None:
            inner_cast = _extract_balanced_paren_arg_list(text, cast_m.end() - 1)
            if inner_cast is not None:
                after = text[cast_m.end() + len(inner_cast) + 1 :].strip()
                if not after:
                    text = inner_cast.strip()
                    continue
                if after[0] in ("+", "-"):
                    text = f"{inner_cast} {after}".strip()
                    continue
        break
    return text


def _extract_base_ptr_name(expr: str) -> str | None:
    text = _strip_cpp_cast_and_parens(expr)
    if not text:
        return None
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([<":
            depth += 1
        elif ch in ")]>" and depth > 0:
            depth -= 1
        elif depth == 0 and ch in ("+", "-"):
            text = _strip_cpp_cast_and_parens(text[:idx])
            break
    ident_m = re.match(r"^([A-Za-z_]\w*)\b", text)
    return ident_m.group(1) if ident_m is not None else None


def _extract_func_name_from_header(header: str) -> str:
    cleaned = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", header.strip()).rstrip()
    if not cleaned.endswith(")"):
        return ""
    depth = 0
    open_idx = -1
    for idx in range(len(cleaned) - 1, -1, -1):
        ch = cleaned[idx]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_idx = idx
                break
    if open_idx <= 0:
        return ""
    prefix = cleaned[:open_idx].rstrip()
    ident = re.search(r"([A-Za-z_]\w*)\s*$", prefix)
    if ident is None:
        return ""
    name = ident.group(1)
    if name in ("if", "for", "while", "switch", "decltype", "__launch_bounds__", "__mix__"):
        return ""
    return name


def _extract_masked_functions(masked_source: str) -> list[tuple[str, str, str]]:
    """Return ``(fn_name, sig_head, body)`` tuples for top-level functions in *masked_source*."""
    funcs: list[tuple[str, str, str]] = []
    depth = 0
    stmt_start = 0
    i = 0
    n = len(masked_source)
    while i < n:
        ch = masked_source[i]
        if ch == ";" and depth == 0:
            stmt_start = i + 1
        elif ch == "{":
            if depth == 0:
                sig_head = masked_source[stmt_start:i].strip()
                fn_name = _extract_func_name_from_header(sig_head)
                brace_depth = 1
                j = i + 1
                while j < n and brace_depth > 0:
                    if masked_source[j] == "{":
                        brace_depth += 1
                    elif masked_source[j] == "}":
                        brace_depth -= 1
                    j += 1
                if fn_name:
                    funcs.append((fn_name, sig_head, masked_source[i + 1 : max(i + 1, j - 1)]))
                stmt_start = j
                i = j
                continue
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            if depth == 0:
                stmt_start = i + 1
        i += 1
    return funcs


def _split_func_param_names(sig_head: str) -> list[str]:
    cleaned = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", sig_head.strip()).rstrip()
    if not cleaned.endswith(")"):
        return []
    depth = 0
    open_idx = -1
    for idx in range(len(cleaned) - 1, -1, -1):
        if cleaned[idx] == ")":
            depth += 1
        elif cleaned[idx] == "(":
            depth -= 1
            if depth == 0:
                open_idx = idx
                break
    if open_idx < 0:
        return []
    params: list[str] = []
    for raw_p in _split_top_level_call_args(cleaned[open_idx + 1 : -1]):
        part = raw_p.split("=", 1)[0].strip()
        m = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?$", part)
        params.append(m.group(1) if m is not None else "")
    return params


_GM_PTR_DECL_RE = re.compile(
    r"\b__gm__\s+(?:const\s+|volatile\s+)*[A-Za-z_]\w*"
    r"(?:\s*\*+\s*(?:__restrict__\s+|restrict\s+|const\s+|volatile\s+)*|\s+)([A-Za-z_]\w*)"
)
_GM_ALIAS_RE = re.compile(
    r"\b__gm__\s+(?:const\s+|volatile\s+)*[A-Za-z_]\w*\s*\*+\s*"
    r"(?:__restrict__\s+|restrict\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+);"
)
_AUTO_ALIAS_RE = re.compile(r"\bauto\s*\*?\s*([A-Za-z_]\w*)\s*=\s*([^;]+);")


def _analyze_function_gm_stores(
    sig_head: str,
    body: str,
) -> tuple[dict[str, set[str]], set[str], set[str]]:
    scope_text = f"{sig_head}\n{body}"
    gm_ptrs = set(_GM_PTR_DECL_RE.findall(scope_text))
    alias_roots: dict[str, set[str]] = {ptr: {ptr} for ptr in gm_ptrs}
    for _ in range(3):
        changed = False
        for alias_re in (_GM_ALIAS_RE, _AUTO_ALIAS_RE):
            for match in alias_re.finditer(body):
                alias_name, rhs_expr = match.group(1), match.group(2)
                base_name = _extract_base_ptr_name(rhs_expr)
                if base_name is not None and base_name in alias_roots:
                    existing = alias_roots.setdefault(alias_name, {alias_name})
                    before_len = len(existing)
                    existing.update(alias_roots[base_name])
                    existing.add(alias_name)
                    if len(existing) != before_len:
                        changed = True
        if not changed:
            break

    mte3_roots: set[str] = set()
    for match in re.finditer(r"\basc_copy_ub2gm\w*\s*\(", body):
        inner = _extract_balanced_paren_arg_list(body, match.end() - 1)
        if inner is None:
            continue
        args = _split_top_level_call_args(inner)
        if not args:
            continue
        base_name = _extract_base_ptr_name(args[0])
        if base_name is not None and base_name in alias_roots:
            mte3_roots.update(alias_roots[base_name])

    scalar_roots: set[str] = set()
    for candidate_var, roots in alias_roots.items():
        escaped = re.escape(candidate_var)
        idx_store = rf"\b{escaped}\s*(?:\[[^\]]+\]\s*)+=(?!=)"
        deref_store = rf"\*\s*\(\s*{escaped}\b[^)]*\)\s*=(?!=)"
        ptr_store = rf"(?<![A-Za-z0-9_])\*\s*{escaped}\s*=(?!=)"
        if (
            re.search(idx_store, body)
            or re.search(deref_store, body)
            or re.search(ptr_store, body)
        ):
            scalar_roots.update(roots)
    return alias_roots, mte3_roots, scalar_roots


def _has_mixed_scalar_and_mte3_gm_store(masked_source: str) -> bool:
    """Detect un-flushed PIPE_S scalar assignments mixed with PIPE_MTE3 stores on one GM ptr."""
    parsed_funcs = _extract_masked_functions(masked_source)
    if not parsed_funcs:
        if re.search(r"\bdcci\b", masked_source):
            return False
        _, mte3_roots, scalar_roots = _analyze_function_gm_stores("", masked_source)
        return bool(mte3_roots & scalar_roots)

    by_name: dict[str, tuple[str, str]] = {name: (sig, body) for name, sig, body in parsed_funcs}
    param_names: dict[str, list[str]] = {
        name: _split_func_param_names(sig) for name, sig, _ in parsed_funcs
    }
    analysis: dict[str, tuple[dict[str, set[str]], set[str], set[str]]] = {
        name: _analyze_function_gm_stores(sig, body) for name, sig, body in parsed_funcs
    }
    direct_dcci = {name for name, _, body in parsed_funcs if re.search(r"\bdcci\b", body)}

    callees: dict[str, set[str]] = {name: set() for name in by_name}
    for name, _, body in parsed_funcs:
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(", body):
            callee = match.group(1)
            if callee in by_name and callee != name:
                callees[name].add(callee)

    for _ in range(len(parsed_funcs)):
        changed = False
        for name, _, body in parsed_funcs:
            alias_roots, mte3_roots, scalar_roots = analysis[name]
            for match in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(", body):
                callee = match.group(1)
                if callee not in by_name or callee == name:
                    continue
                if callee in direct_dcci:
                    continue
                inner = _extract_balanced_paren_arg_list(body, match.end() - 1)
                if inner is None:
                    continue
                call_args = _split_top_level_call_args(inner)
                c_params = param_names[callee]
                _, c_mte3, c_scalar = analysis[callee]
                for idx, arg_expr in enumerate(call_args):
                    if idx >= len(c_params):
                        break
                    p_name = c_params[idx]
                    base_arg = _extract_base_ptr_name(arg_expr)
                    if not p_name or base_arg is None or base_arg not in alias_roots:
                        continue
                    if p_name in c_mte3 and not (alias_roots[base_arg] <= mte3_roots):
                        mte3_roots.update(alias_roots[base_arg])
                        changed = True
                    if p_name in c_scalar and not (alias_roots[base_arg] <= scalar_roots):
                        scalar_roots.update(alias_roots[base_arg])
                        changed = True
        if not changed:
            break

    for name, _, _ in parsed_funcs:
        visited: set[str] = set()
        stack = [name]
        has_dcci = False
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in direct_dcci:
                has_dcci = True
                break
            stack.extend(callees.get(cur, ()))
        if has_dcci:
            continue
        _, mte3_roots, scalar_roots = analysis[name]
        if mte3_roots & scalar_roots:
            return True
    return False


def _check_custom_kernel_redlines(dsl_source: str) -> list[str]:
    errors: list[str] = []
    lower = dsl_source.lower()
    if "l1->gm" in lower or "gm->l0a" in lower or "gm->l0b" in lower:
        errors.append("forbidden_dav3510_dma_path")
    if ("int4b_t" in lower or "cube_int4" in lower) and "dist_unpack4" not in lower and "aiv_unpack" not in lower:
        errors.append("forbidden_dav3510_cube_int4_without_aiv_unpack")
    if any(_SIMDVF_LOCAL_ARRAY_RE.search(body) for body in _simdvf_function_bodies(dsl_source)):
        errors.append(
            "simdvf_nonzero_stack_array: local array inside __simd_vf__ violates vf stack "
            "size (0); allocate in __ubuf__ workspace instead"
        )

    masked = strip_cpp_comments_and_strings(dsl_source)
    for fn_name, (valid_arities, err_msg) in _ALIGNED_COPY_ARITIES.items():
        for match in re.finditer(rf"\b{re.escape(fn_name)}\s*\(", masked):
            inner = _extract_balanced_paren_arg_list(masked, match.end() - 1)
            if inner is None:
                continue
            args = _split_top_level_call_args(inner)
            if args and len(args) not in valid_arities:
                if err_msg not in errors:
                    errors.append(err_msg)
                break

    global_kernels = _collect_global_kernel_names(masked)
    chevron_re = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}]*?>)?\s*<<<")
    for match in chevron_re.finditer(masked):
        target_fn = match.group(1)
        if target_fn not in global_kernels:
            errors.append(
                "non_global_chevron_launch: <<<...>>> is only valid on __global__ kernels; "
                "use asc_vf_call<fn>(...) inside __aicore__ kernels"
            )
            break

    if _has_mixed_scalar_and_mte3_gm_store(masked):
        errors.append(
            "mixed_scalar_and_mte3_gm_store: mixing un-flushed PIPE_S scalar stores "
            "(64B DCache) with PIPE_MTE3 asc_copy_ub2gm* on the same __gm__ output corrupts "
            "unaligned cachelines with stale NaN bits"
        )
    return errors


_CPP_DTYPE_BYTES: dict[str, int] = {
    "uint8_t": 1,
    "int8_t": 1,
    "char": 1,
    "int8": 1,
    "uint8": 1,
    "fp8": 1,
    "fp8_t": 1,
    "half": 2,
    "__bf16": 2,
    "bfloat16_t": 2,
    "bfloat16": 2,
    "float16_t": 2,
    "uint16_t": 2,
    "int16_t": 2,
    "float": 4,
    "int": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "double": 8,
    "int64_t": 8,
    "uint64_t": 8,
}


def _rewrite_cpp_ternary(expr: str) -> str:
    depth = 0
    q_idx = -1
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            q_idx = i
            break
    if q_idx < 0:
        return expr
    depth = 0
    c_idx = -1
    nested = 0
    for i in range(q_idx + 1, len(expr)):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            nested += 1
        elif ch == ":" and depth == 0:
            if nested == 0:
                c_idx = i
                break
            nested -= 1
    if c_idx < 0:
        return expr
    cond = _rewrite_cpp_ternary(expr[:q_idx].strip())
    t_arm = _rewrite_cpp_ternary(expr[q_idx + 1 : c_idx].strip())
    f_arm = _rewrite_cpp_ternary(expr[c_idx + 1 :].strip())
    return f"(({t_arm}) if ({cond}) else ({f_arm}))"


def _eval_cpp_ast_node(node: ast.AST, env: Mapping[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_cpp_ast_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return int(env[node.id])
        raise KeyError(node.id)
    if isinstance(node, ast.UnaryOp):
        val = _eval_cpp_ast_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Invert):
            return ~val
        raise ValueError("unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = _eval_cpp_ast_node(node.left, env)
        right = _eval_cpp_ast_node(node.right, env)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a // b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.LShift: lambda a, b: a << b,
            ast.RShift: lambda a, b: a >> b,
            ast.BitAnd: lambda a, b: a & b,
            ast.BitOr: lambda a, b: a | b,
            ast.BitXor: lambda a, b: a ^ b,
        }
        fn = ops.get(type(node.op))
        if fn is None or (isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0):
            raise ValueError("unsupported binary op")
        return int(fn(left, right))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _eval_cpp_ast_node(node.left, env)
        right = _eval_cpp_ast_node(node.comparators[0], env)
        cmp_ops = {
            ast.Lt: left < right,
            ast.LtE: left <= right,
            ast.Gt: left > right,
            ast.GtE: left >= right,
            ast.Eq: left == right,
            ast.NotEq: left != right,
        }
        op_type = type(node.ops[0])
        if op_type in cmp_ops:
            return int(cmp_ops[op_type])
    if isinstance(node, ast.IfExp):
        cond = _eval_cpp_ast_node(node.test, env)
        return _eval_cpp_ast_node(node.body if cond else node.orelse, env)
    raise ValueError("unsupported ast node")


def _strip_outer_parens(expr: str) -> str:
    """Remove balanced parentheses that enclose an entire C++ expression."""
    text = expr.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        enclosed = True
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and idx < len(text) - 1:
                enclosed = False
                break
        if not enclosed:
            break
        text = text[1:-1].strip()
    return text


_CPP_CAST_RE = re.compile(
    r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|unsigned(?:\s+(?:int|long|short))?|"
    r"int|long|short|float|double|half|__bf16|bfloat16_t|bfloat16)\s*\)"
)


def _strip_cpp_casts(expr: str) -> str:
    """Drop C-style scalar casts and static_cast wrappers for numeric folding."""
    cleaned = expr
    for _ in range(4):
        updated = re.sub(r"\bstatic_cast\s*<[^<>]+>\s*", "", cleaned)
        updated = _CPP_CAST_RE.sub("", updated)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _rewrite_known_sizeof(expr: str) -> str:
    """Replace sizeof(T) for known concrete C++ scalar types with byte widths."""
    return re.sub(
        r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)",
        lambda m: str(_CPP_DTYPE_BYTES[m.group(1)]) if m.group(1) in _CPP_DTYPE_BYTES else m.group(0),
        expr,
    )


def _safe_eval_cpp_int(expr: str, env: Mapping[str, int]) -> int | None:
    cleaned = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", str(expr or "").strip())
    if not cleaned:
        return None
    cleaned = _strip_cpp_casts(_rewrite_known_sizeof(cleaned)).strip()
    if not cleaned:
        return None
    py_expr = _rewrite_cpp_ternary(_strip_outer_parens(cleaned))
    try:
        tree = ast.parse(py_expr, mode="eval")
        return _eval_cpp_ast_node(tree, env)
    except Exception:
        return None


_LAUNCH_BLOCK_DIM_RE = re.compile(r"<<<\s*([^,<>]+?)\s*(?:,|>>>)")
_UB_STRIDED_ACCESS_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\+\s*(?:\(\s*[A-Za-z_]\w*\s*\)\s*)?"
    r"[A-Za-z_]\w*\s*\*\s*([A-Za-z_]\w*|\d+)\b"
)


def _resolve_launcher_var_cap(
    cpp_source: str, var_name: str, consts: Mapping[str, int]
) -> int | None:
    """Resolve constant core bounds when launch block_dim is a local host integer variable."""
    ident = re.escape(var_name)
    for match in re.finditer(
        rf"\b(?:(?:const|constexpr)\s+)?(?:int|int32_t|int64_t|uint32_t|uint64_t|size_t|auto)"
        rf"(?:\s+const)?\s+{ident}\s*=\s*([^;]+);",
        cpp_source,
    ):
        expr = match.group(1).strip()
        val = _safe_eval_cpp_int(expr, consts)
        if val is not None and val > 0:
            return val
        min_match = re.search(r"std::min\s*(?:<[^>]+>)?\s*\(", expr)
        if min_match is not None:
            inner = _extract_balanced_paren_arg_list(expr, min_match.end() - 1)
            if inner is not None:
                min_vals = [
                    capped
                    for arg in _split_top_level_call_args(inner)
                    if (capped := _safe_eval_cpp_int(arg, consts)) is not None and capped > 0
                ]
                if min_vals:
                    return min(min_vals)
        ternary_tail = _rewrite_cpp_ternary(expr)
        if " else " in ternary_tail:
            tail_expr = ternary_tail.rsplit(" else ", 1)[-1].strip("() ")
            val_tail = _safe_eval_cpp_int(tail_expr, consts)
            if val_tail is not None and val_tail > 0:
                return val_tail
    return None


def _parse_cce_launch_block_dim(cpp_source: str, consts: Mapping[str, int]) -> int | None:
    """Resolve the first positive ``<<<block_dim, ...>>>`` launch dimension in C++ source."""
    stripped = strip_cpp_comments_and_strings(cpp_source)
    for match in _LAUNCH_BLOCK_DIM_RE.finditer(stripped):
        raw_expr = match.group(1).strip()
        block_dim = _safe_eval_cpp_int(raw_expr, consts)
        if block_dim is not None and block_dim > 0:
            return block_dim
        if re.fullmatch(r"[A-Za-z_]\w*", raw_expr):
            capped = _resolve_launcher_var_cap(stripped, raw_expr, consts)
            if capped is not None and capped > 0:
                return capped
    return None


def _estimate_ub_bank_conflict_risk(
    cpp_source: str,
    consts: Mapping[str, int],
    pointer_element_bytes: Mapping[str, int],
) -> float:
    """Estimate the share of constant UB row strides that alias one bank group.

    Only pointers with a concrete element type and a fully constant row stride are
    weighed; template-typed or runtime-strided access stays unknown instead of being
    counted as a conflict, so ordinary double-buffered aligned base offsets never
    saturate the risk. Row strides landing a 512-byte multiple apart map onto the same
    bank group, matching the TileLang envelope inspection rule.
    """
    strides: dict[str, int] = {}
    for match in _UB_STRIDED_ACCESS_RE.finditer(strip_cpp_comments_and_strings(cpp_source)):
        pointer_name = match.group(1)
        elem_bytes = pointer_element_bytes.get(pointer_name)
        if elem_bytes is None:
            continue
        stride_elems = _safe_eval_cpp_int(match.group(2), consts)
        if stride_elems is None or stride_elems <= 0:
            continue
        strides[pointer_name] = max(strides.get(pointer_name, 0), stride_elems * elem_bytes)
    if not strides:
        return 0.0
    conflicting = sum(
        1 for stride_bytes in strides.values() if stride_bytes >= 512 and stride_bytes % 512 == 0
    )
    return conflicting / len(strides)


def _parse_objdump_pipe_counts(objdump_text: str) -> dict[str, int]:
    counts: dict[str, int] = {
        "total_insns": 0,
        "cube_mmad": 0,
        "fixpipe": 0,
        "mte1": 0,
        "mte2": 0,
        "mte3": 0,
        "vec": 0,
        "vloopv2": 0,
        "dcci": 0,
        "sync": 0,
        "sw_back_branches": 0,
        "vf_sw_back_branches": 0,
        "scalar_regs": 0,
        "vec_regs": 0,
        "gm_to_l1_insns": 0,
        "gm_to_ub_insns": 0,
        "l1_to_l0a_insns": 0,
        "l1_to_l0b_insns": 0,
        "l0c_to_ub_insns": 0,
        "l0c_to_l1_insns": 0,
        "l0c_to_gm_insns": 0,
        "ub_to_gm_insns": 0,
        "ub_to_l1_insns": 0,
    }
    if not objdump_text.strip():
        return counts
    xregs: set[int] = set()
    vregs: set[int] = set()
    in_vf = False
    for raw_ln in objdump_text.splitlines():
        lbl_m = re.match(r"^[0-9a-f]+\s+<([^>]+)>:\s*$", raw_ln.strip())
        if lbl_m:
            in_vf = ".vector.thread" in lbl_m.group(1)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+([A-Za-z0-9_.]+)(.*)$", raw_ln)
        if not m:
            continue
        pc = int(m.group(1), 16)
        mnem = m.group(2).upper()
        rest = m.group(3)
        counts["total_insns"] += 1
        xregs.update(int(r) for r in re.findall(r"\bX(\d+)\b", rest))
        vregs.update(int(r) for r in re.findall(r"\bV(\d+)\b", rest))
        if mnem.startswith("VLOOP"):
            if "VLOOPV2" in mnem or "VLOOP_V2" in mnem:
                counts["vloopv2"] += 1
        elif mnem.startswith("MMAD"):
            counts["cube_mmad"] += 1
        elif mnem.startswith("FIX_"):
            counts["fixpipe"] += 1
            if "L1" in mnem or "CBUF" in mnem:
                counts["l0c_to_l1_insns"] += 1
            elif "UB" in mnem:
                counts["l0c_to_ub_insns"] += 1
            elif "GM" in mnem:
                counts["l0c_to_gm_insns"] += 1
        elif mnem.startswith("LOAD_L1_TO_L0"):
            counts["mte1"] += 1
            if "L0B" in mnem:
                counts["l1_to_l0b_insns"] += 1
            else:
                counts["l1_to_l0a_insns"] += 1
        elif mnem.startswith("MOV_OUT_TO_"):
            counts["mte2"] += 1
            if "L1" in mnem or "CBUF" in mnem:
                counts["gm_to_l1_insns"] += 1
            elif "UB" in mnem:
                counts["gm_to_ub_insns"] += 1
        elif mnem.startswith("MOV_UB_TO_"):
            counts["mte3"] += 1
            if "L1" in mnem or "CBUF" in mnem:
                counts["ub_to_l1_insns"] += 1
            else:
                counts["ub_to_gm_insns"] += 1
        elif mnem.startswith("DCCI"):
            counts["dcci"] += 1
        elif mnem.startswith(("SET_FLAG", "WAIT_FLAG", "SET_INTRA", "WAIT_INTRA", "BAR.")):
            counts["sync"] += 1
        elif mnem.startswith("V") and not mnem.startswith("VF"):
            counts["vec"] += 1
        if mnem.startswith(("JUMP", "B.", "CBNZ", "CBZ", "TBNZ", "TBZ")) or mnem == "B":
            tgt_m = re.search(r"\+0x([0-9a-f]+)>", rest, re.IGNORECASE)
            hash_m = re.search(r"#(-?\d+)", rest)
            is_back = False
            if tgt_m:
                is_back = int(tgt_m.group(1), 16) <= pc
            elif hash_m:
                is_back = int(hash_m.group(1)) <= pc
            if is_back:
                counts["sw_back_branches"] += 1
                if in_vf:
                    counts["vf_sw_back_branches"] += 1
    counts["scalar_regs"] = len(xregs)
    counts["vec_regs"] = len(vregs)
    return counts


_STAGE_KEYS: tuple[str, ...] = (
    "STAGES",
    "NUM_STAGES",
    "N_STAGES",
    "NSTAGES",
)


def _rewrite_sizeof(expr: str, type_bytes: Mapping[str, int]) -> str:
    """Replace sizeof(T) with the byte width of a resolved dtype or template parameter."""
    return re.sub(
        r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)",
        lambda m: str(type_bytes.get(m.group(1), _CPP_DTYPE_BYTES.get(m.group(1), 2))),
        expr,
    )


def _split_cpp_args(text: str) -> list[str]:
    """Split a C++ template or call argument list at top-level commas."""
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _template_bindings(
    cpp_source: str,
    consts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Resolve integral template arguments and typename dtype widths from concrete calls."""
    values: dict[str, int] = {}
    types: dict[str, int] = {}
    blocks = re.finditer(
        r"template\s*<([^<>]*)>\s*"
        r"(?:__\w+\s*(?:\([^()]*\))?\s*)*"
        r"(?:(?:inline|static|void)\s+)*"
        r"([A-Za-z_]\w*)\s*\(",
        cpp_source,
    )
    for block in blocks:
        params = _split_cpp_args(block.group(1))
        call = re.search(
            rf"\b{re.escape(block.group(2))}\s*<\s*([^<>]*)\s*>\s*(?:<<<|\()",
            cpp_source,
        )
        if call is None:
            continue
        args = _split_cpp_args(call.group(1))
        if len(args) != len(params):
            continue
        for declaration, argument in zip(params, args):
            names = re.findall(r"[A-Za-z_]\w*", declaration.split("=")[0])
            if not names:
                continue
            name = names[-1]
            if re.match(r"(?:typename|class)\b", declaration):
                type_name = argument.split("::")[-1].strip()
                types[name] = _CPP_DTYPE_BYTES.get(type_name, 2)
                continue
            val = _safe_eval_cpp_int(argument, consts)
            if val is not None:
                values[name] = max(values.get(name, val), val)
    return values, types


def _resolve_cpp_consts_and_types(cpp_source: str) -> tuple[dict[str, int], dict[str, int]]:
    """Fold constexpr and #define integers along with resolved template type byte widths."""
    stripped = strip_cpp_comments_and_strings(cpp_source)
    define_source = re.sub(r"\\\r?\n", " ", stripped)
    declarations = [
        (m.group(1), m.group(2))
        for m in re.finditer(
            r"\bconstexpr\s+(?:int|int32_t|int64_t|uint32_t|uint64_t|size_t)\s+"
            r"([A-Za-z_]\w*)\s*=\s*([^;]+);",
            stripped,
        )
    ]
    for m in re.finditer(
        r"^\s*#\s*define\s+([A-Za-z_]\w*)(?!\()\s+([^\r\n]+)",
        define_source,
        re.MULTILINE,
    ):
        declarations.append((m.group(1), m.group(2).strip()))
    consts: dict[str, int] = {}
    type_bytes: dict[str, int] = {}
    for _ in range(4):
        previous = dict(consts)
        for name, expr in declarations:
            val = _safe_eval_cpp_int(_rewrite_sizeof(expr, type_bytes), consts)
            if val is not None:
                consts[name] = val
        values, types = _template_bindings(cpp_source, consts)
        consts.update(values)
        type_bytes.update(types)
        if consts == previous:
            break
    return consts, type_bytes


def _resolve_cpp_consts(cpp_source: str) -> dict[str, int]:
    """Fold constexpr and #define integers with template args, aliases, and sizeof widths."""
    return _resolve_cpp_consts_and_types(cpp_source)[0]


def _stage_multiplicity(
    cpp_source: str,
    consts: Mapping[str, int],
    *,
    scope: str,
    names: Sequence[str] = (),
    strides: Mapping[str, int] | None = None,
) -> int:
    """Estimate buffer stage count from constexpr/template values, slots, and sync pairs."""
    stages = 1
    for key in (f"{scope}_STAGES", *_STAGE_KEYS):
        value = consts.get(key)
        if value is not None:
            stages = max(stages, int(value))
    if stages > 1:
        return min(stages, 8)
    if strides is not None and any(name in strides for name in names):
        return 2
    slot_stems: dict[str, set[str]] = {}
    for name in names:
        if _is_numbered_slot(name):
            slot_stems.setdefault(re.sub(r"\d+$", "", name), set()).add(name)
    if any(len(slots) > 1 for slots in slot_stems.values()):
        return 2
    ub_pairs = {frozenset(("PIPE_MTE3", "PIPE_MTE2")), frozenset(("PIPE_V", "PIPE_MTE2"))}
    l1_pairs = {frozenset(("PIPE_MTE1", "PIPE_MTE2")), frozenset(("PIPE_M", "PIPE_MTE2"))}
    pairs = ub_pairs if scope == "UB" else l1_pairs
    events: set[int] = set()
    for m in re.finditer(
        r"asc_sync_(?:notify|wait)\s*\(\s*(PIPE_[A-Z0-9_]+)\s*,\s*(PIPE_[A-Z0-9_]+)\s*,\s*"
        r"(?:\(event_t\))?\s*(\d+)",
        cpp_source,
    ):
        if frozenset((m.group(1), m.group(2))) in pairs:
            events.add(int(m.group(3)))
    return 2 if {0, 1} <= events else 1


def _is_numbered_slot(name: str) -> bool:
    """Report whether a buffer name carries an attached physical slot index."""
    return bool(re.search(r"\d+$", name))


def _stage_storage_pointer(name: str, names: Sequence[str]) -> bool:
    """Report whether one pointer is the sole physical slot for its buffer family."""
    stem = re.sub(r"\d+$", "", name)
    return not any(
        other != name and re.sub(r"\d+$", "", other) == stem for other in names
    )


def _stage_stride_elems(cpp_source: str, consts: Mapping[str, int]) -> dict[str, int]:
    """Resolve per-buffer element strides from pointer plus stage-index expressions."""
    strides: dict[str, int] = {}
    for m in re.finditer(
        r"\b([A-Za-z_]\w*)\s*\+\s*(?:\(\s*)?"
        r"(?:stg_id|stage_id|stg|stage|slot|buf)\s*\*\s*([A-Za-z_]\w*|\d+)\s*\)?",
        cpp_source,
    ):
        value = _safe_eval_cpp_int(m.group(2), consts)
        if value is not None and value > 0:
            strides[m.group(1)] = max(strides.get(m.group(1), value), value)
    return strides


def _declared_stage_bytes(
    name: str,
    elem_bytes: int,
    consts: Mapping[str, int],
    *,
    stages: int,
    single_pointer: bool,
) -> int:
    """Return an explicit byte size or per-stage element count for one buffer pointer."""
    upper = name.upper()
    numbered = _is_numbered_slot(name)
    bases = [upper]
    for suffix in ("_BASE", "_BUF", "_PTR", "_0", "_1"):
        if upper.endswith(suffix):
            bases.append(upper[: -len(suffix)])
    roots: list[str] = []
    for base in bases:
        slotless = re.sub(r"\d+$", "", base)
        roots.extend((slotless, slotless.replace("_L1", "").replace("_UB", "")))
    for root in dict.fromkeys(roots):
        for suffix in ("_STAGE_BYTES", "_BYTES", "_STAGE", "_BASE", "_BUF", "_PTR"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        for key, is_bytes in (
            (f"{root}_STAGE_BYTES", True),
            (f"{root}_BYTES", True),
            (f"{root}_STAGE", False),
        ):
            if key not in consts:
                continue
            if is_bytes:
                return int(consts[key])
            scale = stages if single_pointer and not numbered else 1
            return int(consts[key]) * elem_bytes * scale
    return 0


def _extract_cpp_loop_spans(masked_src: str) -> list[tuple[int, int]]:
    """Return character spans ``(body_start, body_end)`` of C++ for/while loop bodies."""
    spans: list[tuple[int, int]] = []
    n = len(masked_src)
    for m in re.finditer(r"\b(?:for|while)\s*\(", masked_src):
        pos = m.end() - 1
        paren_depth = 0
        while pos < n:
            ch = masked_src[pos]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    pos += 1
                    break
            pos += 1
        while pos < n and masked_src[pos].isspace():
            pos += 1
        if pos >= n or masked_src[pos] != "{":
            continue
        body_start = pos + 1
        brace_depth = 1
        pos += 1
        while pos < n and brace_depth > 0:
            if masked_src[pos] == "{":
                brace_depth += 1
            elif masked_src[pos] == "}":
                brace_depth -= 1
            pos += 1
        spans.append((body_start, max(body_start, pos - 1)))
    return spans


def _extract_call_args_at(masked_src: str, open_paren_idx: int) -> list[str]:
    """Return top-level comma-separated arguments starting at ``open_paren_idx``."""
    n = len(masked_src)
    if open_paren_idx < 0 or open_paren_idx >= n or masked_src[open_paren_idx] != "(":
        return []
    depth = 0
    pos = open_paren_idx
    while pos < n:
        ch = masked_src[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return _split_cpp_args(masked_src[open_paren_idx + 1 : pos])
        pos += 1
    return []


def _resolve_dma_bound_consts(
    masked_src: str, consts: Mapping[str, int]
) -> dict[str, int]:
    """Fold local tile clamp upper bounds such as `if (n_cur > BN) n_cur = BN;`."""
    env = dict(consts)
    for m in re.finditer(
        r"\bif\s*\(\s*([A-Za-z_]\w*)\s*>\s*([^;{}()]+?)\s*\)\s*\1\s*=\s*([^;]+);",
        masked_src,
    ):
        var_name = m.group(1)
        if var_name in env:
            continue
        bound = _safe_eval_cpp_int(m.group(3), env)
        if bound is not None and bound > 0:
            env[var_name] = bound
    return env


def _resolve_l0c2gm_dst_elem_bytes(
    fname: str,
    args: Sequence[str],
    masked_src: str,
    call_pos: int,
    consts: Mapping[str, int],
    type_bytes: Mapping[str, int],
) -> int:
    """Infer destination element byte width for asc_copy_l0c2gm from type, pointer, or quant."""
    low_fname = fname.lower()
    if low_fname.endswith(("_s8", "_u8", "_i8", "_fp8")):
        return 1
    if low_fname.endswith(("_f16", "_bf16")):
        return 2
    if low_fname.endswith(("_f32", "_s32", "_i32")):
        return 4

    def _width_for_type_token(token: str) -> int | None:
        if token in type_bytes:
            return int(type_bytes[token])
        if token in _CPP_DTYPE_BYTES:
            return int(_CPP_DTYPE_BYTES[token])
        low_tok = token.lower()
        if "int8" in low_tok or "uint8" in low_tok or "fp8" in low_tok:
            return 1
        if "half" in low_tok or "bf16" in low_tok or "bfloat16" in low_tok or "fp16" in low_tok:
            return 2
        if "float" in low_tok or "int32" in low_tok or "uint32" in low_tok:
            return 4
        return None

    dst_arg = args[0] if args else ""
    cast_m = re.search(r"__gm__\s+(?:const\s+)?([A-Za-z_]\w*)\s*\*", dst_arg)
    if cast_m is not None:
        width = _width_for_type_token(cast_m.group(1))
        if width is not None:
            return width

    stripped_dst = _strip_cpp_casts(dst_arg).strip()
    ident_m = re.match(r"^[&(]*([A-Za-z_]\w*)\b", stripped_dst)
    if ident_m is not None:
        ptr_name = ident_m.group(1)
        prefix_src = masked_src[:call_pos]
        fn_starts = [
            fm.start()
            for fm in re.finditer(r"\b(?:template\s*<|__global__|__aicore__)\b", prefix_src)
        ]
        scope_src = prefix_src[fn_starts[-1] :] if fn_starts else prefix_src
        decl_matches = list(
            re.finditer(
                rf"__gm__\s+(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*{re.escape(ptr_name)}\b",
                scope_src,
            )
        )
        if not decl_matches and scope_src is not prefix_src:
            decl_matches = list(
                re.finditer(
                    rf"__gm__\s+(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*{re.escape(ptr_name)}\b",
                    prefix_src,
                )
            )
        if decl_matches:
            width = _width_for_type_token(decl_matches[-1].group(1))
            if width is not None:
                return width

    if len(args) >= 9:
        q_arg = args[8].lower()
        if any(tok in q_arg for tok in ("int8", "s8", "fp8", "e4m3", "e5m2")):
            return 1
        if any(tok in q_arg for tok in ("f16", "fp16", "bf16", "bfloat16", "half")):
            return 2
        q_val = _safe_eval_cpp_int(args[8], consts)
        if q_val is not None:
            return 4 if q_val == 0 else 2
    return 4


def _extract_trackb_dma_and_pipe_signals(
    cpp_source: str,
    consts: Mapping[str, int],
    facts: object,
    explicit_l1: Mapping[str, int],
    *,
    uses_cube: bool,
    uses_256kb_ub: bool,
    tile_m: int,
    tile_n: int,
    mte2_cnt: int,
    mte3_cnt: int,
    mmad_cnt: int,
    ub_stages: int,
    l1_stages: int,
    case_ratios: CasePhysicsRatios | None,
    type_bytes: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Extract deterministic physical ledger signals from reachable C++20 source."""
    masked = strip_cpp_comments_and_strings(cpp_source)
    lower = masked.lower()
    loop_spans = _extract_cpp_loop_spans(masked)
    dma_consts = _resolve_dma_bound_consts(masked, consts)
    resolved_types = dict(type_bytes) if type_bytes is not None else {}

    pipe_all_positions: set[int] = set()
    for b_start, b_end in loop_spans:
        body = masked[b_start:b_end]
        for pm in re.finditer(r"\b(?:pipe_barrier|asc_sync_pipe)\s*\(\s*PIPE_ALL\s*\)", body):
            pipe_all_positions.add(b_start + pm.start())
    in_loop_pipe_all = len(pipe_all_positions)

    has_dual_subcore = bool(
        re.search(r"\b(?:get_subblockid|GetSubBlockIdx|get_subblock_idx)\s*\(", masked)
    )

    dma_bytes: list[int] = []
    contig_candidates: list[int] = []
    gm_dma_pat = re.compile(
        r"\b(asc_copy_gm2ub_align|asc_copy_ub2gm_align|asc_copy_gm2ub|asc_copy_ub2gm|"
        r"asc_copy_gm2l1(?:_nd2nz|_dn2nz)?|asc_copy_l0c2gm\w*)\s*\("
    )
    for dm in gm_dma_pat.finditer(masked):
        fname = dm.group(1)
        args = _extract_call_args_at(masked, dm.end() - 1)
        if fname in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
            if len(args) == 3:
                sz = _safe_eval_cpp_int(args[2], dma_consts)
                if sz is not None and sz > 0:
                    dma_bytes.append(sz)
                    contig_candidates.append(sz)
            elif len(args) >= 4:
                nburst = _safe_eval_cpp_int(args[2], dma_consts)
                row_b = _safe_eval_cpp_int(args[3], dma_consts)
                if row_b is not None and row_b > 0:
                    contig_candidates.append(row_b)
                    total_b = row_b * nburst if (nburst is not None and nburst > 0) else row_b
                    dma_bytes.append(total_b)
        elif fname in ("asc_copy_gm2ub", "asc_copy_ub2gm") and len(args) >= 4:
            nburst = _safe_eval_cpp_int(args[2], dma_consts)
            blen = _safe_eval_cpp_int(args[3], dma_consts)
            if blen is not None and blen > 0:
                row_b = blen * 32
                contig_candidates.append(row_b)
                dma_bytes.append(row_b * nburst if (nburst is not None and nburst > 0) else row_b)
        elif fname.startswith("asc_copy_gm2l1") and len(args) >= 6:
            rows = _safe_eval_cpp_int(args[4], dma_consts)
            cols = _safe_eval_cpp_int(args[5], dma_consts)
            if cols is not None and cols > 0:
                row_b = cols * 2
                contig_candidates.append(row_b)
                dma_bytes.append(row_b * rows if (rows is not None and rows > 0) else row_b)
        elif fname.startswith("asc_copy_l0c2gm") and len(args) >= 4:
            cols = _safe_eval_cpp_int(args[2], dma_consts)
            rows = _safe_eval_cpp_int(args[3], dma_consts)
            if cols is not None and cols > 0:
                elem_b = _resolve_l0c2gm_dst_elem_bytes(
                    fname, args, masked, dm.start(), dma_consts, resolved_types
                )
                row_b = cols * elem_b
                contig_candidates.append(row_b)
                dma_bytes.append(row_b * rows if (rows is not None and rows > 0) else row_b)

    min_gm_dma_bytes = min(dma_bytes) if dma_bytes else 0
    if dma_bytes and hasattr(facts, "dma_burst_efficiency"):
        total_dma = float(sum(dma_bytes))
        weighted_inv = sum(
            float(b) / max(1e-6, float(facts.dma_burst_efficiency(int(b))))
            for b in dma_bytes
        )
        dma_burst_eff = round(min(1.0, total_dma / max(1e-6, weighted_inv)), 4)
    else:
        dma_burst_eff = 1.0

    if contig_candidates:
        dominant_contig_bytes = max(contig_candidates)
    elif case_ratios is not None and case_ratios.inner_contiguous_bytes > 0:
        dominant_contig_bytes = int(case_ratios.inner_contiguous_bytes)
    elif mte2_cnt > 0 or mte3_cnt > 0:
        dominant_contig_bytes = tile_n * 2
    else:
        dominant_contig_bytes = 0

    cube_m_vals: list[int] = []
    if uses_cube or mmad_cnt > 0:
        for gm in re.finditer(r"\bascend_gemm_l1\s*<([^;{}()]+)>\s*\(", masked):
            t_args = _split_cpp_args(gm.group(1))
            if t_args:
                m_val = _safe_eval_cpp_int(t_args[0], consts)
                if m_val is not None and m_val > 0:
                    cube_m_vals.append(m_val)
        for mm in re.finditer(r"\basc_mmad(?:_mx|_sparse)?\s*(?:<[^;{}()]*>)?\s*\(", masked):
            c_args = _extract_call_args_at(masked, mm.end() - 1)
            if len(c_args) >= 4:
                m_val = _safe_eval_cpp_int(c_args[3], consts)
                if m_val is not None and m_val > 0:
                    cube_m_vals.append(m_val)
    effective_cube_m = (
        min(cube_m_vals)
        if cube_m_vals
        else (tile_m if (uses_cube or mmad_cnt > 0) else 0)
    )

    has_lowbit_quant = bool(
        re.search(
            r"\b(?:int4|uint4|fp4|f4e2m1|f4e1m2|int4b_t|deq_scale|quant_mode|asc_mmad_mx)\b",
            masked,
        )
    )

    has_online_rescale_loop = False
    exp_re = re.compile(r"\b(?:vexp|exp2|expf|exp|Exp)\s*(?:<[^>]*>)?\s*\(")
    rescale_re = re.compile(
        r"\b(?:vcmax|vcadd|vcmin|vreduce|ReduceMax|ReduceSum|asc_reduce_\w+|"
        r"vmul|vdiv|vfma|vmad|Mul|Div)\b|\*=|\+="
    )
    for b_start, b_end in loop_spans:
        body = masked[b_start:b_end]
        if exp_re.search(body) and rescale_re.search(body):
            has_online_rescale_loop = True
            break

    l1_total = sum(explicit_l1.values())
    l1_full = max(1, int(getattr(facts, "l1_bytes", 524288)))
    l1_half = max(1, l1_full // 2)
    has_l1_half_iso = "l1_half_bank_split" in lower or "half_bank" in lower
    if l1_total <= 0 or has_l1_half_iso:
        l1_bank_conflict_risk = 0.0
    elif l1_total > l1_half:
        l1_bank_conflict_risk = round(min(1.0, l1_total / float(l1_full)), 4)
    elif len(explicit_l1) >= 2:
        l1_bank_conflict_risk = round(min(0.5, l1_total / float(l1_half)), 4)
    else:
        l1_bank_conflict_risk = 0.0

    mte2_inflight = 0
    if mte2_cnt > 0:
        cur_run = 0
        max_run = 0
        mte2_events: set[str] = set()
        for tok_m in re.finditer(
            r"\b(asc_copy_gm2(?:ub|l1)\w*)\s*\(|"
            r"\basc_sync_(?:notify|wait)\s*\(\s*([^)]+)\)",
            masked,
        ):
            if tok_m.group(1):
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                sync_args = _split_cpp_args(tok_m.group(2) or "")
                if any(a.strip() == "PIPE_MTE2" for a in sync_args[:2]):
                    if len(sync_args) >= 3:
                        mte2_events.add(sync_args[2].strip())
                    if "wait" in tok_m.group(0):
                        cur_run = 0
        mte2_inflight = max(1, max_run, len(mte2_events), 2 if ub_stages > 1 else 1)

    strides = _stage_stride_elems(cpp_source, consts)
    uses_dyn_offset = bool(
        strides
        or re.search(r"\b(?:stg|stage|stg_id|stage_id|nstg)\b[^;\n]*[?:]", masked)
        or (
            (ub_stages >= 2 or l1_stages >= 2)
            and re.search(
                r"\b\w+\s*&\s*1\b|\b\w+\s*%\s*(?:2|STAGES|NUM_STAGES)\b",
                masked,
            )
        )
    )

    return {
        "has_dual_subcore": has_dual_subcore,
        "uses_unreserved_ubuf": uses_256kb_ub,
        "in_loop_pipe_all_barriers": in_loop_pipe_all,
        "min_gm_dma_bytes": min_gm_dma_bytes,
        "dma_burst_efficiency_ratio": dma_burst_eff,
        "dominant_contig_bytes": dominant_contig_bytes,
        "effective_cube_m": effective_cube_m,
        "has_lowbit_quant": has_lowbit_quant,
        "has_online_rescale_loop": has_online_rescale_loop,
        "l1_bank_conflict_risk": l1_bank_conflict_risk,
        "mte2_inflight_per_aiv": mte2_inflight,
        "uses_dynamic_offset_pipelining": uses_dyn_offset,
    }


def _build_ascendc_static_ledger(
    cpp_source: str,
    *,
    target: AscendTarget | None = None,
    compiler_summary: BishengResourceSummary | None = None,
    objdump_text: str = "",
    microarch_profile: MicroarchProfile | Mapping[str, object] | None = None,
    case_ratios: CasePhysicsRatios | None = None,
) -> ObjectivePhysicalLedger:
    """Compute SRAM waterlines and pipe cycles from C++20 constexprs, calls, and objdump."""
    resolved_target = target or AscendTarget("ascend.950pr")
    from tilefoundry_ascend.ascend.target.facts import ascend_space_facts  # noqa: PLC0415

    facts = ascend_space_facts(resolved_target)
    lower = cpp_source.lower()

    consts, type_bytes = _resolve_cpp_consts_and_types(cpp_source)

    ub_offsets: list[tuple[str, int, int]] = []
    l1_offsets: list[tuple[str, int, int]] = []
    ptr_env: dict[str, tuple[str, int]] = {}
    concrete_pointer_bytes: dict[str, int] = {}

    cast_pat = re.compile(
        r"__(ubuf|cbuf)__\s+([A-Za-z0-9_]+)\s*\*\s*(\w+)\s*=\s*"
        r"\(\s*__\1__\s+[A-Za-z0-9_]+\s*\*\s*\)\s*([^;]+);"
    )
    plus_pat = re.compile(
        r"__(ubuf|cbuf)__\s+([A-Za-z0-9_]+)\s*\*\s*(\w+)\s*=\s*"
        r"([A-Za-z_]\w*)\s*\+\s*([^;]+);"
    )
    for ln in cpp_source.splitlines():
        cm = cast_pat.search(ln)
        if cm:
            space, ctype, name, expr = cm.group(1), cm.group(2), cm.group(3), cm.group(4)
            if ctype in _CPP_DTYPE_BYTES:
                concrete_pointer_bytes[name] = _CPP_DTYPE_BYTES[ctype]
            off = _safe_eval_cpp_int(expr, consts)
            if off is not None and off >= 0:
                eb = _CPP_DTYPE_BYTES.get(ctype, 2)
                ptr_env[name] = (space, off)
                (ub_offsets if space == "ubuf" else l1_offsets).append((name, off, eb))
            continue
        pm = plus_pat.search(ln)
        if pm:
            space, ctype, name, base, expr = (
                pm.group(1),
                pm.group(2),
                pm.group(3),
                pm.group(4),
                pm.group(5),
            )
            if ctype in _CPP_DTYPE_BYTES:
                concrete_pointer_bytes[name] = _CPP_DTYPE_BYTES[ctype]
            base_entry = ptr_env.get(base)
            delta = _safe_eval_cpp_int(expr, consts)
            if delta is not None:
                eb = _CPP_DTYPE_BYTES.get(ctype, 2)
                base_off = base_entry[1] if base_entry and base_entry[0] == space else 0
                off = max(0, base_off + delta * eb)
                ptr_env[name] = (space, off)
                (ub_offsets if space == "ubuf" else l1_offsets).append((name, off, eb))

    tile_m = max(1, consts.get("CUBE_TILE_M", consts.get("BM", consts.get("TILE_M", 64))))
    tile_n = max(1, consts.get("CUBE_TILE_N", consts.get("BN", consts.get("TILE_N", 64))))
    tile_k = max(1, consts.get("CUBE_TILE_K", consts.get("BK", consts.get("TILE_K", 64))))
    uses_cube = any(
        tok in lower for tok in ("asc_mmad", "ascend_gemm_l1", "mmad", "l0c", "cube_tile", "__cbuf__")
    )
    uses_simt = "__simt_vf__" in cpp_source or "Simt::" in cpp_source
    uses_256kb_ub = (
        "ub_256kb_unlock" in lower
        or "__simd_vf__" in cpp_source
        or "262144" in cpp_source
    ) and not uses_simt

    block_dim = _parse_cce_launch_block_dim(cpp_source, consts)
    physical_cores = int(facts.num_aic_cores) if uses_cube else int(facts.num_aiv_cores)
    if block_dim is not None:
        grid_blocks = block_dim
        wave_occupancy = block_dim / max(1, physical_cores)
    elif case_ratios is not None:
        wave_occupancy = float(case_ratios.wave_occupancy_ratio)
        grid_blocks = max(1, int(round(wave_occupancy * physical_cores)))
    else:
        grid_blocks = 1
        wave_occupancy = 0.0
    ub_bank_conflict_risk = _estimate_ub_bank_conflict_risk(
        cpp_source,
        consts,
        concrete_pointer_bytes,
    )

    def _span_from_offsets(
        entries: list[tuple[str, int, int]],
        *,
        scope: str,
        stages: int,
        strides: Mapping[str, int],
    ) -> dict[str, int]:
        if not entries:
            return {}
        ordered = sorted(entries, key=lambda item: item[1])
        names = [item[0] for item in ordered]
        bufs: dict[str, int] = {}
        for idx, (name, off, eb) in enumerate(ordered):
            stride = strides.get(name)
            if stride is not None:
                size = stride * eb * stages
            elif idx + 1 < len(ordered):
                size = max(0, ordered[idx + 1][1] - off)
            elif idx > 0:
                prev_gap = max(0, off - ordered[idx - 1][1])
                size = prev_gap if prev_gap > 0 else max(eb * 64, tile_n * eb)
            else:
                stage_bytes = consts.get(f"{scope}_STAGE_BYTES", consts.get("STAGE_BYTES"))
                if _stage_storage_pointer(name, names) and stages > 1 and stage_bytes:
                    size = int(stage_bytes) * stages
                else:
                    size = max(eb * 64, tile_m * tile_n * eb)
            if idx + 1 == len(ordered):
                size = max(
                    size,
                    _declared_stage_bytes(
                        name,
                        eb,
                        consts,
                        stages=stages,
                        single_pointer=_stage_storage_pointer(name, names),
                    ),
                )
            if size > 0:
                bufs[name] = size
        return bufs

    ub_stages = _stage_multiplicity(
        cpp_source,
        consts,
        scope="UB",
        names=[name for name, _, _ in ub_offsets],
    )
    strides = _stage_stride_elems(cpp_source, consts)
    l1_stages = _stage_multiplicity(
        cpp_source,
        consts,
        scope="L1",
        names=[name for name, _, _ in l1_offsets],
        strides=strides,
    )
    explicit_ub = _span_from_offsets(ub_offsets, scope="UB", stages=ub_stages, strides=strides)
    explicit_l1 = _span_from_offsets(l1_offsets, scope="L1", stages=l1_stages, strides=strides)

    ub_match = re.search(r"\bub_bytes\s*=\s*(\d+)", cpp_source, re.IGNORECASE)
    l1_match = re.search(r"\bl1_bytes\s*=\s*(\d+)", cpp_source, re.IGNORECASE)
    if ub_match:
        explicit_ub = {"ub_declared": int(ub_match.group(1))}
    if l1_match:
        explicit_l1 = {"l1_declared": int(l1_match.group(1))}

    if not explicit_l1 and uses_cube:
        stages = max(2, l1_stages)
        explicit_l1["l1_AB"] = (tile_m * tile_k + tile_k * tile_n) * 2 * stages
    if not explicit_ub:
        ub_rows = consts.get("ROWS", tile_m)
        ub_blk = consts.get("BLK", tile_n)
        explicit_ub["ub_work"] = (
            ub_rows * ub_blk * 2 if uses_cube else max(1024, ub_rows * ub_blk * 2)
        )

    implicit_ub: dict[str, int] = {}
    if compiler_summary is not None:
        if compiler_summary.spill_bytes > 0:
            implicit_ub["register_spill"] = int(compiler_summary.spill_bytes)
        if compiler_summary.stack_bytes > 0:
            implicit_ub["stack_frame"] = int(compiler_summary.stack_bytes)

    ub_cap = (
        int(facts.ub_simtvf_usable_bytes)
        if uses_simt
        else (int(facts.ub_nominal_bytes) if uses_256kb_ub else int(facts.ub_simdvf_usable_bytes))
    )
    waterlines: dict[str, ScopeWaterline] = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=ub_cap,
            explicit_buffers=explicit_ub,
            implicit_buffers=implicit_ub,
        )
    }
    if explicit_l1 or uses_cube:
        waterlines["L1"] = build_scope_waterline(
            scope="L1",
            capacity_bytes=int(facts.l1_bytes),
            explicit_buffers=explicit_l1,
            implicit_buffers={},
        )

    dis_counts = _parse_objdump_pipe_counts(objdump_text)
    mte2_src_cnt = len(re.findall(r"\basc_copy_gm2(?:ub|l1)\w*\s*\(", cpp_source))
    mte3_src_cnt = len(re.findall(r"\basc_copy_ub2gm\w*\s*\(", cpp_source))
    fix_src_cnt = len(re.findall(r"\basc_copy_l0c2\w*\s*\(", cpp_source))
    mmad_src_cnt = len(
        re.findall(
            r"\b(?:asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\s*(?:<[^;{}()]*>)?\s*\(",
            cpp_source,
        )
    )
    vf_src_cnt = len(re.findall(r"\b(?:simd_inst|Reg)::\w+\s*(?:<[^>]*>)?\s*\(", cpp_source))

    mte2_cnt = max(mte2_src_cnt, dis_counts["mte2"])
    mte3_cnt = max(mte3_src_cnt, dis_counts["mte3"])
    fix_cnt = max(fix_src_cnt, dis_counts["fixpipe"])
    mmad_cnt = max(mmad_src_cnt, dis_counts["cube_mmad"])
    vf_cnt = max(vf_src_cnt, dis_counts["vec"])

    spill_penalty = int(compiler_summary.spill_bytes) * 2 if compiler_summary else 0
    sw_loop_mult = 2 if (compiler_summary and not compiler_summary.vloopv2_hit) else 1

    signals = _extract_trackb_dma_and_pipe_signals(
        cpp_source,
        consts,
        facts,
        explicit_l1,
        uses_cube=uses_cube,
        uses_256kb_ub=uses_256kb_ub,
        tile_m=tile_m,
        tile_n=tile_n,
        mte2_cnt=mte2_cnt,
        mte3_cnt=mte3_cnt,
        mmad_cnt=mmad_cnt,
        ub_stages=ub_stages,
        l1_stages=l1_stages,
        case_ratios=case_ratios,
        type_bytes=type_bytes,
    )

    pipe_cycles: dict[str, int] = {}
    if mte2_cnt > 0:
        pipe_cycles["AIV0.PIPE_MTE2"] = mte2_cnt * max(16, (tile_m * tile_n * 2) // 64)
    if mte3_cnt > 0:
        pipe_cycles["AIV0.PIPE_MTE3"] = mte3_cnt * max(16, (tile_m * tile_n * 2) // 64)
    if vf_cnt > 0:
        total_vec = vf_cnt * max(8, (tile_m * tile_n) // 64) * sw_loop_mult + spill_penalty
        if bool(signals["has_dual_subcore"]):
            aiv0_vec = max(1, total_vec // 2)
            aiv1_vec = max(1, total_vec - aiv0_vec)
            pipe_cycles["AIV0.PIPE_V"] = aiv0_vec
            pipe_cycles["AIV1.PIPE_V"] = aiv1_vec
        else:
            pipe_cycles["AIV0.PIPE_V"] = total_vec
    if mmad_cnt > 0:
        pipe_cycles["AIC.PIPE_M"] = mmad_cnt * max(16, (tile_m * tile_n * tile_k) // 4096)
    if fix_cnt > 0:
        pipe_cycles["AIC.PIPE_FIX"] = fix_cnt * max(12, (tile_m * tile_n * 2) // 256)
    if dis_counts["mte1"] > 0:
        pipe_cycles["AIC.PIPE_MTE1"] = dis_counts["mte1"] * max(8, (tile_m * tile_k) // 128)

    if isinstance(microarch_profile, MicroarchProfile):
        ma_dict = microarch_profile.to_dict()
    elif isinstance(microarch_profile, Mapping):
        ma_dict = dict(microarch_profile)
    else:
        ma_dict = extract_microarch_profile_from_cce(cpp_source).to_dict()

    is_legal = all(wl.overflow_bytes == 0 for wl in waterlines.values())
    steady_ii = max(pipe_cycles.values()) if pipe_cycles else None
    ub_scalar_loops = count_reachable_ub_scalar_compute_loops(cpp_source)
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=is_legal,
        steady_state_ii=steady_ii,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        mte2_inflight_per_aiv=int(signals["mte2_inflight_per_aiv"]),
        grid_blocks=grid_blocks,
        wave_occupancy=wave_occupancy,
        dominant_contig_bytes=int(signals["dominant_contig_bytes"]),
        effective_cube_m=int(signals["effective_cube_m"]),
        has_indirect_gather=uses_simt,
        has_lowbit_quant=bool(signals["has_lowbit_quant"]),
        has_online_rescale_loop=bool(signals["has_online_rescale_loop"]),
        uses_unreserved_ubuf=bool(signals["uses_unreserved_ubuf"]),
        min_gm_dma_bytes=int(signals["min_gm_dma_bytes"]),
        dma_burst_efficiency_ratio=float(signals["dma_burst_efficiency_ratio"]),
        uses_dynamic_offset_pipelining=bool(signals["uses_dynamic_offset_pipelining"]),
        in_loop_pipe_all_barriers=int(signals["in_loop_pipe_all_barriers"]),
        ub_bank_conflict_risk=ub_bank_conflict_risk,
        l1_bank_conflict_risk=float(signals["l1_bank_conflict_risk"]),
        ub_scalar_compute_loops=ub_scalar_loops,
        microarch_profile=ma_dict,
    )


def evaluate_ascendc_candidate(
    cpp_source: str,
    *,
    freedom_mask: object | None = None,
    verifier: object | None = None,
    target: AscendTarget | None = None,
    max_flag_depth: int = 15,
    enable_track_router: bool = False,
    case_ratios: CasePhysicsRatios | None = None,
    baseline_cpp_source: str | None = None,
    require_hardware_dma: bool = False,
    require_cube_engine: bool = False,
    module_source: str | None = None,
    regime_tag: str | None = None,
) -> StageAResult:
    """Validate a Track-B AscendC C++ candidate via 7-facet checks and static ledger."""
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=None,
            error_message="ascendc_empty_source",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            backend_kind="ascendc",
            fault_plane="STAGE_A_STATIC",
            fault_code="ASCENDC_EMPTY_SOURCE",
            failed_tier="stage_a",
        )

    from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask  # noqa: PLC0415
    from tilefoundry_ascend.ascend.sil.verifier import (  # noqa: PLC0415
        SILVerifier,
        verify_source_against_facets,
    )

    _, reachable_cpp = extract_reachable_trackb_source(
        cpp_source,
        module_source=module_source,
        regime_tag=regime_tag,
    )

    active_mask = freedom_mask if freedom_mask is not None else ConstraintRelaxationMask.bottom()
    active_verifier = verifier if verifier is not None else SILVerifier()
    if hasattr(active_verifier, "verify_source_against_facets"):
        facet_report = active_verifier.verify_source_against_facets(
            cpp_source, active_mask, max_flag_depth=max_flag_depth
        )
    else:
        facet_report = verify_source_against_facets(
            cpp_source, active_mask, max_flag_depth=max_flag_depth
        )

    redline_errs = _check_custom_kernel_redlines(cpp_source)
    ledger = _build_ascendc_static_ledger(reachable_cpp, target=target, case_ratios=case_ratios)

    if baseline_cpp_source is not None:
        forbidden_tokens = CCE_CPP20_SURFACE.introduced_forbidden_tokens(
            baseline_cpp_source, cpp_source
        )
    else:
        forbidden_tokens = tuple(
            token
            for token, count in CCE_CPP20_SURFACE.count_forbidden_tokens(cpp_source).items()
            if count > 0
        )
    errors = [*redline_errs, *facet_report.errors]
    masked_src = strip_cpp_comments_and_strings(reachable_cpp)
    if require_cube_engine:
        mmad_hits = len(
            re.findall(
                r"\b(?:asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\s*(?:<[^;{}()]*>)?\s*\(",
                masked_src,
            )
        )
        if mmad_hits == 0 and int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) == 0:
            errors.append(
                "missing_cube_engine_primitives: matmul/attention kernel contains zero "
                "Cube MMAD instructions (scalar fallback trap)"
            )
    if require_hardware_dma:
        mte2_hits = len(re.findall(r"\basc_copy_gm2(?:ub|l1)\w*\s*\(", masked_src))
        has_mte2_pipe = any(
            k.endswith(".PIPE_MTE2") and int(v) > 0 for k, v in ledger.pipe_cycles.items()
        )
        if mte2_hits == 0 and not has_mte2_pipe:
            errors.append(
                "missing_hardware_dma_staging: Track-B kernel performs raw scalar __gm__ "
                "loops with zero GM->UB/L1 DMA transfers"
            )
    if case_ratios is None or not getattr(case_ratios, "has_empty_group", False):
        if re.search(r"\b__simt_vf__\b", masked_src) and (
            re.search(r"__launch_bounds__\s*\(\s*1\s*\)", masked_src)
            or re.search(
                r"\basc_vf_call\s*<[^>]+>\s*\(\s*dim3\s*\(\s*1\s*,\s*1\s*,\s*1\s*\)",
                masked_src,
            )
        ):
            errors.append(
                "degenerated_single_lane_simt: __simt_vf__ launched with 1 thread wastes "
                "127/128 SIMT lanes; parallelize across >=32 threads via "
                "Simt::GetThreadIdx<0>() or use __simd_vf__"
            )
    if forbidden_tokens:
        errors.append(f"cce_forbidden_tokens: {', '.join(forbidden_tokens)}")
    if ledger.max_overflow_bytes > 0:
        errors.append(f"max_overflow_bytes={ledger.max_overflow_bytes}")

    features: dict[str, object] | None = None
    route: object | None = None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            route_execution_track,
        )

        lower = reachable_cpp.lower()
        uses_cube = "mmad" in lower or "l0c" in lower or "ascend_gemm_l1" in lower
        uses_simt = "__simt_vf__" in reachable_cpp or "Simt::" in reachable_cpp
        uses_ssbuf = "ssbuf" in lower or "crosscore" in lower
        static_affine_bounds = not uses_simt
        cube_to_vector_ratio = 1.0 if uses_cube else 0.0
        quant_granularity = "per_token_or_block" if "deq_scale" in lower else "none"
        semantic_features: dict[str, object] = {
            "static_affine_bounds": static_affine_bounds,
            "cube_to_vector_ratio": cube_to_vector_ratio,
            "quant_granularity": quant_granularity,
            "indirect_sparse_gather": uses_simt,
            "cross_iter_recurrence": "none",
            "dynamic_cross_core_handoff": uses_ssbuf,
            "inner_contiguous_bytes": (
                case_ratios.inner_contiguous_bytes if case_ratios is not None else 0
            ),
            "persistent_wave_occupancy": (
                case_ratios.wave_occupancy_ratio if case_ratios is not None else 0.0
            ),
            "uses_cube": uses_cube,
        }
        route = route_execution_track(
            semantic_features,
            ledger=ledger,
            target=target,
            case_ratios=case_ratios,
            simt_dominant=uses_simt,
        )
        features = dict(route.features) if route.features is not None else semantic_features

    if errors:
        fault_plane, fault_code, failed_tier = _classify_ascendc_stage_a_fault(errors)
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=ledger,
            features=features,
            route=route,
            error_message="; ".join(errors),
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            backend_kind="ascendc",
            fault_plane=fault_plane,
            fault_code=fault_code,
            failed_tier=failed_tier,
        )
    return StageAResult(
        passed=True,
        prim_func=None,
        ledger=ledger,
        features=features,
        route=route,
        elapsed_ms=(time.monotonic() - t0) * 1000.0,
        backend_kind="ascendc",
    )


def _is_tvm_ir_node(obj: object) -> bool:
    try:
        import tilelang  # noqa: F401, PLC0415
        from tvm import IRModule, tirx  # noqa: PLC0415

        return isinstance(obj, (tirx.PrimFunc, IRModule))
    except Exception:
        return False


def _resolve_prim_func_and_configs_from_dsl(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], list[str]]:
    """Load DSL source via a temp module, invoke selector if present, and build PrimFunc."""
    import importlib.util  # noqa: PLC0415

    ast.parse(dsl_source)
    sel_pass_configs: dict[str, object] = {}
    sel_compile_flags: list[str] = []
    effective_kwargs = _merge_dims_into_kwargs(factory_kwargs, dims)

    with tempfile.TemporaryDirectory(prefix="tf_stage_a_") as tmp_dir:
        mod_path = Path(tmp_dir) / "candidate_dsl.py"
        mod_path.write_text(dsl_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_tf_stage_a_cand", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for candidate DSL")
        mod = importlib.util.module_from_spec(spec)
        with ascend_compile_session("ascend"):
            spec.loader.exec_module(mod)

            if config_selector_name and hasattr(mod, config_selector_name):
                selector_obj = getattr(mod, config_selector_name)
                if callable(selector_obj):
                    sel_args = _filter_kwargs_for_callable(selector_obj, effective_kwargs)
                    sel_raw = selector_obj(**sel_args)
                    if isinstance(sel_raw, Mapping):
                        overrides, sel_pass_configs, sel_compile_flags = _split_selector_output(
                            sel_raw
                        )
                        effective_kwargs.update(overrides)

            if factory_name and hasattr(mod, factory_name):
                target_obj = getattr(mod, factory_name)
                if callable(target_obj):
                    if factory_name == "kernel" and not _has_prim_func_marker(dsl_source):
                        return target_obj, sel_pass_configs, sel_compile_flags
                    call_kw = _filter_kwargs_for_callable(target_obj, effective_kwargs)
                    built = target_obj(**call_kw)
                    return built, sel_pass_configs, sel_compile_flags
                return target_obj, sel_pass_configs, sel_compile_flags

            for val in vars(mod).values():
                if _is_tvm_ir_node(val):
                    return val, sel_pass_configs, sel_compile_flags

            kernel_obj = getattr(mod, "kernel", None)
            if callable(kernel_obj) and not _has_prim_func_marker(dsl_source):
                return kernel_obj, sel_pass_configs, sel_compile_flags

            for name, val in vars(mod).items():
                if (
                    not name.startswith("_")
                    and callable(val)
                    and getattr(val, "__module__", None) == mod.__name__
                ):
                    call_kw = _filter_kwargs_for_callable(val, effective_kwargs)
                    return val(**call_kw), sel_pass_configs, sel_compile_flags
    raise ValueError("DSL source defined no callable factory or PrimFunc")


def _resolve_prim_func_from_dsl(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
):
    """Load DSL source via a temporary module so @T.prim_func sees file lines cleanly."""
    prim_func, _pc, _cf = _resolve_prim_func_and_configs_from_dsl(
        dsl_source,
        factory_name=factory_name,
        factory_kwargs=factory_kwargs,
    )
    return prim_func


def _merge_pass_configs_and_flags(
    sel_pc: Mapping[str, object] | None,
    sel_cf: Sequence[str] | str | None,
    ext_pc: Mapping[str, object] | None,
    ext_cf: Sequence[str] | str | None,
) -> tuple[dict[str, object], list[str]]:
    merged_pc: dict[str, object] = {}
    merged_cf: list[str] = []

    def _absorb(pc_map: Mapping[str, object] | None, cf_seq: Sequence[str] | str | None) -> None:
        if pc_map:
            for k, v in pc_map.items():
                if str(k) == "tl.device_compile_flags":
                    for flag in _normalize_flag_list(v):
                        if flag not in merged_cf:
                            merged_cf.append(flag)
                else:
                    merged_pc[str(k)] = v
        for flag in _normalize_flag_list(cf_seq):
            if flag not in merged_cf:
                merged_cf.append(flag)

    _absorb(sel_pc, sel_cf)
    _absorb(ext_pc, ext_cf)
    return merged_pc, merged_cf


_PARALLEL_FORBIDDEN_MATH = ("Cast", "exp", "log", "tanh", "sigmoid", "rsqrt")


def _is_t_dsl_call(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
    )


def _parallel_forbidden_math_calls(tree: ast.AST) -> list[str]:
    """Return sorted ``T.<op>`` scalar-math calls in ``T.Parallel`` bodies outside SimtVF."""
    found: set[str] = set()

    def _visit(node: ast.AST, in_simt_vf: bool) -> None:
        if isinstance(node, ast.For) and _is_t_dsl_call(node.iter, "Parallel") and not in_simt_vf:
            for body_node in node.body:
                for inner in ast.walk(body_node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                        attr = inner.func.attr
                        if attr in _PARALLEL_FORBIDDEN_MATH and _is_t_dsl_call(inner, attr):
                            found.add(attr)
        nested_simt_vf = in_simt_vf or (
            isinstance(node, ast.With)
            and any(_is_t_dsl_call(item.context_expr, "SimtVF") for item in node.items)
        )
        for child in ast.iter_child_nodes(node):
            _visit(child, nested_simt_vf)

    _visit(tree, False)
    return sorted(found)


def _kernel_threads_kwarg_calls(tree: ast.AST) -> list[int]:
    """Return line numbers of T.Kernel/Kernel calls passing forbidden threads= kwargs."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_kernel = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "Kernel"
        ) or (isinstance(node.func, ast.Name) and node.func.id == "Kernel")
        if is_kernel and any(kw.arg == "threads" for kw in node.keywords):
            lines.append(node.lineno)
    return sorted(set(lines))


def _ast_has_dsl_call(tree: ast.AST, *attrs: str) -> bool:
    for node in ast.walk(tree):
        if any(_is_t_dsl_call(node, attr) for attr in attrs):
            return True
    return False


_SRAM_ALLOC_ATTRS = frozenset(
    {"alloc_shared", "alloc_l1", "alloc_l0a", "alloc_l0b", "alloc_l0c", "alloc_ss"}
)


def _expr_base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _expr_base_name(node.value)
    return None


def _expr_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_sid_partition_guard(test: ast.AST, sid_var: str) -> bool:
    for sub in ast.walk(test):
        if isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Name) and sub.left.id == sid_var:
            if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in sub.ops):
                return True
        if isinstance(sub, ast.Compare) and any(
            isinstance(comp, ast.Name) and comp.id == sid_var for comp in sub.comparators
        ):
            if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in sub.ops):
                return True
    return False


def _extract_mixed_kernel_sids2(node: ast.AST) -> tuple[ast.With, str] | None:
    if not isinstance(node, ast.With):
        return None
    for item in node.items:
        call = item.context_expr
        if not _is_t_dsl_call(call, "MixedKernel"):
            continue
        assert isinstance(call, ast.Call)
        sids_val: object = None
        for kw in call.keywords:
            if kw.arg == "sids" and isinstance(kw.value, ast.Constant):
                sids_val = kw.value.value
                break
        if sids_val != 2:
            continue
        if isinstance(item.optional_vars, (ast.Tuple, ast.List)) and len(item.optional_vars.elts) >= 2:
            sid_node = item.optional_vars.elts[1]
            if isinstance(sid_node, ast.Name) and sid_node.id != "_":
                return node, sid_node.id
    return None


def _uncovered_1c2v_subcore_mte3_writes(tree: ast.AST) -> list[int]:
    """Return line numbers of UB->GM T.copy writes lacking sid partitioning in sids=2."""
    bad_lines: list[int] = []
    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, ast.FunctionDef):
            continue
        param_names = {arg.arg for arg in fn_node.args.args}
        for sub_node in ast.walk(fn_node):
            extracted = _extract_mixed_kernel_sids2(sub_node)
            if extracted is None:
                continue
            with_node, sid_var = extracted
            ub_buffers: set[str] = set()
            sram_buffers: set[str] = set()
            for stmt in ast.walk(with_node):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                ):
                    for attr in _SRAM_ALLOC_ATTRS:
                        if _is_t_dsl_call(stmt.value, attr):
                            name = stmt.targets[0].id
                            sram_buffers.add(name)
                            if attr == "alloc_shared":
                                ub_buffers.add(name)
                            break

            sid_tainted: set[str] = {sid_var}
            for _ in range(6):
                changed = False
                for stmt in ast.walk(with_node):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        tgt = stmt.targets[0]
                        if isinstance(tgt, ast.Name) and (_expr_names(stmt.value) & sid_tainted):
                            if tgt.id not in sid_tainted:
                                sid_tainted.add(tgt.id)
                                changed = True
                    elif (
                        isinstance(stmt, ast.AnnAssign)
                        and stmt.value is not None
                        and isinstance(stmt.target, ast.Name)
                        and (_expr_names(stmt.value) & sid_tainted)
                    ):
                        if stmt.target.id not in sid_tainted:
                            sid_tainted.add(stmt.target.id)
                            changed = True
                    elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                        if (
                            isinstance(stmt.iter, ast.Call)
                            and len(stmt.iter.args) >= 2
                            and (_expr_names(stmt.iter) & sid_tainted)
                            and stmt.target.id not in sid_tainted
                        ):
                            sid_tainted.add(stmt.target.id)
                            changed = True
                if not changed:
                    break

            def _scan_stmts(stmts: Sequence[ast.stmt], *, sid_guarded: bool) -> None:
                for st in stmts:
                    if isinstance(st, ast.If):
                        guarded = sid_guarded or _is_sid_partition_guard(st.test, sid_var)
                        _scan_stmts(st.body, sid_guarded=guarded)
                        _scan_stmts(st.orelse, sid_guarded=guarded)
                        continue
                    if isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                        _scan_stmts(st.body, sid_guarded=sid_guarded)
                        if hasattr(st, "orelse"):
                            _scan_stmts(st.orelse, sid_guarded=sid_guarded)
                        continue
                    if sid_guarded:
                        continue
                    for inner in ast.walk(st):
                        if not _is_t_dsl_call(inner, "copy"):
                            continue
                        assert isinstance(inner, ast.Call)
                        if len(inner.args) < 2:
                            continue
                        src_name = _expr_base_name(inner.args[0])
                        dst_arg = inner.args[1]
                        dst_name = _expr_base_name(dst_arg)
                        if src_name not in ub_buffers or dst_name is None:
                            continue
                        is_gm_target = (dst_name in param_names) or (dst_name not in sram_buffers)
                        if not is_gm_target:
                            continue
                        if isinstance(dst_arg, ast.Subscript):
                            if not (_expr_names(dst_arg.slice) & sid_tainted):
                                bad_lines.append(inner.lineno)
                        else:
                            bad_lines.append(inner.lineno)

            _scan_stmts(with_node.body, sid_guarded=False)
    return sorted(set(bad_lines))


def _resolve_stage_a_var_bindings(
    dims: Mapping[str, object] | None,
    var_bindings: Mapping[str, int] | None,
) -> dict[str, int] | None:
    resolved: dict[str, int] = {}
    if dims:
        for key, val in dims.items():
            if isinstance(val, int) and not isinstance(val, bool):
                key_str = str(key)
                iv = int(val)
                resolved[key_str] = iv
                mapped = _HIR_TO_BUILDER_DIM_MAP.get(key_str)
                if mapped:
                    resolved.setdefault(mapped, iv)
        b_val = resolved.get("B", resolved.get("b_dim"))
        s_val = resolved.get("S", resolved.get("S_q", resolved.get("s_dim")))
        s1_val = resolved.get("S1", resolved.get("s1_dim", s_val))
        nq_val = resolved.get("N_q", resolved.get("n_q"))
        nkv_val = resolved.get("N_kv", resolved.get("n_kv"))
        n_val = resolved.get("N", resolved.get("n_depth"))
        if b_val is not None and s_val is not None:
            resolved.setdefault("m_rows", b_val * s_val)
            if n_val is not None:
                resolved.setdefault("v_rows", n_val * b_val * s_val)
        if b_val is not None and s1_val is not None and nq_val is not None:
            resolved.setdefault("q_total_rows", b_val * s1_val * nq_val)
        n1_val = resolved.get("N1", nq_val)
        n2_val = resolved.get("N2", nkv_val)
        if n1_val is not None and n2_val is not None and n2_val > 0:
            resolved.setdefault("g_ratio", max(1, n1_val // n2_val))
        if s_val is not None and nq_val is not None and nkv_val is not None and nkv_val > 0:
            resolved.setdefault("q_rows", s_val * (nq_val // nkv_val))
    if var_bindings:
        for key, val in var_bindings.items():
            if isinstance(val, int) and not isinstance(val, bool):
                resolved[str(key)] = int(val)
    return resolved or None


def run_stage_a_static(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None,
    enable_track_router: bool = False,
    case_ratios: CasePhysicsRatios | None = None,
    require_hardware_dma: bool = False,
    require_cube_engine: bool = False,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> StageAResult:
    """Run millisecond Stage A AST, Two-Snapshot IR Diff, and optional track routing."""
    t0 = time.monotonic()
    try:
        parsed_tree = ast.parse(dsl_source)
        kernel_threads_lines = _kernel_threads_kwarg_calls(parsed_tree)
        if kernel_threads_lines:
            line_list = ", ".join(str(ln) for ln in kernel_threads_lines)
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=(
                    "forbidden_kernel_threads_kwarg: T.Kernel(...) does not accept "
                    f"threads= (lines {line_list}); pass threads= only to T.SimtVF(threads=128)"
                ),
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                fault_plane="STAGE_A_STATIC",
                fault_code="FORBIDDEN_KERNEL_THREADS_KWARG",
                failed_tier="stage_a",
            )
        parallel_math = _parallel_forbidden_math_calls(parsed_tree)
        if parallel_math:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=(
                    "forbidden_tracka_parallel_scalar_math: T.Parallel body calls "
                    f"{', '.join(f'T.{op}' for op in parallel_math)}; move elementwise math "
                    "under with T.SimdVF(): with T.simd.vcvt / T.simd.vexp, or under "
                    "with T.SimtVF(threads=128): with T.cast"
                ),
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                fault_plane="STAGE_A_STATIC",
                fault_code="FORBIDDEN_TRACKA_PARALLEL_SCALAR_MATH",
                failed_tier="stage_a",
            )
        uncovered_1c2v_lines = _uncovered_1c2v_subcore_mte3_writes(parsed_tree)
        if uncovered_1c2v_lines:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=(
                    "uncovered_1c2v_subcore_mte3_write: T.MixedKernel(sids=2) writes shared "
                    "GM/workspace buffer without partitioning slices by 'sid' "
                    "(causes insert_sync AIV0.MTE3->AIV1.MTE3 fatal); "
                    "partition rows as sid * half_m : (sid + 1) * half_m"
                ),
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                fault_plane="STAGE_A_STATIC",
                fault_code="UNCOVERED_1C2V_SUBCORE_MTE3_WRITE",
                failed_tier="stage_a",
            )
        tl_ast_report = verify_tilelang_ast(dsl_source)
        if not tl_ast_report.passed:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message="; ".join(tl_ast_report.errors),
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                fault_plane="STAGE_A_STATIC",
                fault_code="TILELANG_AST_VIOLATION",
                failed_tier="stage_a",
            )
        entry_name = _detect_module_entrypoint(parsed_tree)
        if entry_name:
            purity_errs = check_npu_entrypoint_purity(dsl_source, entry_name)
            if purity_errs:
                return StageAResult(
                    passed=False,
                    prim_func=None,
                    ledger=None,
                    error_message="; ".join(purity_errs),
                    elapsed_ms=(time.monotonic() - t0) * 1000.0,
                    fault_plane="STAGE_A_STATIC",
                    fault_code="NPU_ENTRYPOINT_PURITY_VIOLATION",
                    failed_tier="stage_a",
                )
        with ascend_compile_session("ascend"):
            prim_func, sel_pc, sel_cf = _resolve_prim_func_and_configs_from_dsl(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
            )
            if not _is_tvm_ir_node(prim_func) and callable(prim_func):
                redline_errs = _check_custom_kernel_redlines(dsl_source)
                err_parts = [
                    *redline_errs,
                    (
                        "stage_a_non_kernel_callable: candidate is a plain Python function "
                        "without @T.prim_func or Track-B TU"
                    ),
                ]
                return StageAResult(
                    passed=False,
                    prim_func=None,
                    ledger=None,
                    features=None,
                    route=None,
                    error_message="; ".join(err_parts),
                    elapsed_ms=(time.monotonic() - t0) * 1000.0,
                    backend_kind="python_kernel",
                    fault_plane="STAGE_A_STATIC",
                    fault_code="STAGE_A_NON_KERNEL_CALLABLE",
                    failed_tier="stage_a",
                )
            merged_pc, merged_cf = _merge_pass_configs_and_flags(
                sel_pc,
                sel_cf,
                pass_configs,
                compile_flags,
            )
            resolved_var_bindings = _resolve_stage_a_var_bindings(dims, var_bindings)
            inspection: EnvelopeInspectionResult = inspect_tilelang_primfunc(
                prim_func,
                target=target,
                pass_configs=merged_pc or None,
                compile_flags=merged_cf or None,
                var_bindings=resolved_var_bindings,
                ordered_tensor_shapes=ordered_tensor_shapes,
            )
    except Exception as exc:
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=None,
            features=None,
            route=None,
            error_message=f"stage_a_parse_or_lower_error: {type(exc).__name__}: {exc}",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            fault_plane="STAGE_A_STATIC",
            fault_code="STAGE_A_PARSE_OR_LOWER_ERROR",
            failed_tier="stage_a",
        )

    ledger = inspection.ledger
    features: dict[str, object] | None = None
    route: object | None = None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            route_execution_track,
        )

        route = route_execution_track(
            prim_func,
            ledger=ledger,
            target=target,
            case_ratios=case_ratios,
        )
        features = dict(route.features) if route.features is not None else None
    if not ledger.is_legal:
        reasons: list[str] = []
        if ledger.max_overflow_bytes > 0:
            reasons.append(f"max_overflow_bytes={ledger.max_overflow_bytes}")
        if inspection.lowering_error:
            reasons.append(inspection.lowering_error)
        if not reasons:
            reasons.append("illegal_datapath_or_capacity")
        return StageAResult(
            passed=False,
            prim_func=prim_func,
            ledger=ledger,
            features=features,
            route=route,
            error_message="; ".join(reasons),
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            fault_plane="STAGE_A_STATIC",
            fault_code=(
                "UB_OVERFLOW" if ledger.max_overflow_bytes > 0 else "ILLEGAL_DATAPATH_OR_CAPACITY"
            ),
            failed_tier="stage_a",
        )
    engine_reasons: list[str] = []
    if require_cube_engine:
        has_cube_pipe = int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) > 0
        has_gemm_call = _ast_has_dsl_call(parsed_tree, "gemm", "gemm_v0")
        if not (has_cube_pipe or has_gemm_call):
            engine_reasons.append(
                "missing_cube_engine_primitives: Track-A @T.prim_func contains zero "
                "Cube MMAD/T.gemm operations (AIC.PIPE_M == 0)"
            )
    if require_hardware_dma:
        uses_simt = bool(ledger.has_indirect_gather or _ast_has_dsl_call(parsed_tree, "SimtVF"))
        has_mte2 = any(
            k.endswith("PIPE_MTE2") and int(v) > 0 for k, v in ledger.pipe_cycles.items()
        )
        if not uses_simt and not has_mte2:
            engine_reasons.append(
                "missing_hardware_dma_staging: Track-A @T.prim_func performs zero "
                "GM->UB/L1 DMA transfers (PIPE_MTE2 == 0)"
            )
    if engine_reasons:
        return StageAResult(
            passed=False,
            prim_func=prim_func,
            ledger=ledger,
            features=features,
            route=route,
            error_message="; ".join(engine_reasons),
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            fault_plane="STAGE_A_STATIC",
            fault_code="MISSING_ENGINE_PRIMITIVES",
            failed_tier="stage_a",
        )
    return StageAResult(
        passed=True,
        prim_func=prim_func,
        ledger=ledger,
        features=features,
        route=route,
        elapsed_ms=(time.monotonic() - t0) * 1000.0,
    )


def _resolve_conda_python(explicit_python: str | None = None) -> str:
    candidate = (explicit_python or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "")).strip()
    if candidate and Path(candidate).is_file():
        return candidate
    return sys.executable


def build_stage_b_driver_source(
    cand_path: str | Path,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    enable_device_compile: bool = True,
    kernel_dump_path: str | Path | None = None,
    tir_dump_path: str | Path | None = None,
) -> str:
    """Build standalone Python driver lowering and device-compiling a candidate DSL file."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    eff_kwargs = _merge_dims_into_kwargs(factory_kwargs, dims)
    ext_pc, ext_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    payload = json.dumps(
        {
            "cand_path": str(cand_path),
            "factory_name": factory_name,
            "factory_kwargs": eff_kwargs,
            "config_selector_name": config_selector_name,
            "pass_configs": ext_pc,
            "compile_flags": ext_cf,
            "kernel_dump_path": str(kernel_dump_path) if kernel_dump_path is not None else None,
            "tir_dump_path": str(tir_dump_path) if tir_dump_path is not None else None,
        }
    )
    return (
        "import contextlib, importlib.util, inspect, json, pathlib, tilelang\n"
        "from tilelang import tvm\n"
        "from tilelang.backend.module import create_backend_context\n"
        "from tvm import tirx, IRModule\n"
        "@contextlib.contextmanager\n"
        "def _ascend_compile_session(pass_configs=None):\n"
        "    with contextlib.ExitStack() as _stack:\n"
        "        if tvm.target.Target.current(allow_none=True) is None:\n"
        "            _stack.enter_context(create_backend_context('ascend').target)\n"
        "        if pass_configs is not None:\n"
        "            _stack.enter_context(\n"
        "                tvm.transform.PassContext(config=dict(pass_configs))\n"
        "            )\n"
        "        yield\n"
        f"_cfg = json.loads({payload!r})\n"
        "_src_text = pathlib.Path(_cfg['cand_path']).read_text(encoding='utf-8')\n"
        "_has_prim_marker = 'prim_func' in _src_text\n"
        "_spec = importlib.util.spec_from_file_location('_tf_stage_b', _cfg['cand_path'])\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "def _filt(obj, kw):\n"
        "    sig = inspect.signature(obj)\n"
        "    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):\n"
        "        return dict(kw)\n"
        "    return {k: v for k, v in kw.items() if k in sig.parameters}\n"
        "def _norm_flags(raw):\n"
        "    if raw is None:\n"
        "        return []\n"
        "    if isinstance(raw, str):\n"
        "        return raw.split()\n"
        "    return [str(getattr(x, 'value', x)).strip() for x in raw if str(x).strip()]\n"
        "_kw = dict(_cfg['factory_kwargs'])\n"
        "_sel_pc, _sel_cf = {}, []\n"
        "_fn = None\n"
        "with _ascend_compile_session():\n"
        "    _spec.loader.exec_module(_mod)\n"
        "    _sname = _cfg['config_selector_name']\n"
        "    if _sname and callable(getattr(_mod, _sname, None)):\n"
        "        _sfn = getattr(_mod, _sname)\n"
        "        _sout = _sfn(**_filt(_sfn, _kw))\n"
        "        if isinstance(_sout, dict):\n"
        "            _raw_pc = _sout.get('pass_configs')\n"
        "            if isinstance(_raw_pc, dict):\n"
        "                for _k, _v in _raw_pc.items():\n"
        "                    if _k == 'tl.device_compile_flags':\n"
        "                        _sel_cf.extend(_norm_flags(_v))\n"
        "                    else:\n"
        "                        _sel_pc[str(_k)] = _v\n"
        "            elif _raw_pc is not None:\n"
        "                _sel_cf.extend(_norm_flags(_raw_pc))\n"
        "            _sel_cf.extend(_norm_flags(_sout.get('compile_flags')))\n"
        "            for _k, _v in _sout.items():\n"
        "                if _k not in ('pass_configs', 'compile_flags'):\n"
        "                    _kw[_k] = _v\n"
        "    _fname = _cfg['factory_name']\n"
        "    if _fname and hasattr(_mod, _fname):\n"
        "        _obj = getattr(_mod, _fname)\n"
        "        if _fname == 'kernel' and not _has_prim_marker and callable(_obj):\n"
        "            _fn = _obj\n"
        "        else:\n"
        "            _fn = _obj(**_filt(_obj, _kw)) if callable(_obj) else _obj\n"
        "    if _fn is None:\n"
        "        for _v in vars(_mod).values():\n"
        "            if isinstance(_v, (tirx.PrimFunc, IRModule)):\n"
        "                _fn = _v; break\n"
        "    if _fn is None and callable(getattr(_mod, 'kernel', None)) and not _has_prim_marker:\n"
        "        _fn = getattr(_mod, 'kernel')\n"
        "    if _fn is None:\n"
        "        for _k, _v in vars(_mod).items():\n"
        "            if (not _k.startswith('_') and callable(_v)\n"
        "                    and getattr(_v, '__module__', None) == _mod.__name__):\n"
        "                _fn = _v(**_filt(_v, _kw)); break\n"
        "    assert _fn is not None\n"
        "    if not isinstance(_fn, (tirx.PrimFunc, IRModule)) and callable(_fn):\n"
        "        _compile_hook = getattr(\n"
        "            _mod, 'compile_kernel', getattr(_mod, 'build_kernel', None)\n"
        "        )\n"
        "        if not callable(_compile_hook):\n"
        "            raise RuntimeError(\n"
        "                'stage_b_non_kernel_callable: plain Python function lacks '\n"
        "                '@T.prim_func or compile_kernel'\n"
        "            )\n"
        "        _compile_hook()\n"
        "    else:\n"
        "        _merged_pc, _merged_cf = {}, []\n"
        "        def _absorb_flags(items):\n"
        "            for _f in items:\n"
        "                if _f and _f not in _merged_cf:\n"
        "                    _merged_cf.append(_f)\n"
        "        _attrs = getattr(_fn, 'attrs', None)\n"
        "        if _attrs and hasattr(_attrs, 'items'):\n"
        "            for _pk in ('tl.pass_configs', 'tilelang_pass_configs'):\n"
        "                if _pk in _attrs and hasattr(_attrs[_pk], 'items'):\n"
        "                    for _ak, _av in _attrs[_pk].items():\n"
        "                        _val = getattr(_av, 'value', _av)\n"
        "                        if str(_ak) == 'tl.device_compile_flags':\n"
        "                            _absorb_flags(_norm_flags(_val))\n"
        "                        else:\n"
        "                            _merged_pc[str(_ak)] = _val\n"
        "            for _ck in ('tir.cce_flags', 'tilelang_compile_flags'):\n"
        "                if _ck in _attrs:\n"
        "                    _absorb_flags(_norm_flags(_attrs[_ck]))\n"
        "        _merged_pc.update(_sel_pc)\n"
        "        _absorb_flags(_sel_cf)\n"
        "        _merged_pc.update(_cfg['pass_configs'])\n"
        "        _absorb_flags(_cfg['compile_flags'])\n"
        "        _absorb_flags(['--cce-res-usage'])\n"
        "        if _merged_cf:\n"
        "            _merged_pc['tl.device_compile_flags'] = list(_merged_cf)\n"
        "        if _cfg.get('tir_dump_path'):\n"
        "            pathlib.Path(_cfg['tir_dump_path']).write_text(\n"
        "                str(_fn), encoding='utf-8'\n"
        "            )\n"
        "        with _ascend_compile_session(_merged_pc):\n"
        "            try:\n"
        "                _art = tilelang.lower(\n"
        "                    _fn, target='ascend', enable_device_compile=True\n"
        "                )\n"
        "            except Exception:\n"
        "                try:\n"
        "                    _pre_art = tilelang.lower(\n"
        "                        _fn, target='ascend', enable_device_compile=False\n"
        "                    )\n"
        "                    if _cfg.get('tir_dump_path'):\n"
        "                        _tir_txt = str(getattr(_pre_art, 'device_mod', None) or _fn)\n"
        "                        pathlib.Path(_cfg['tir_dump_path']).write_text(\n"
        "                            _tir_txt, encoding='utf-8'\n"
        "                        )\n"
        "                    if _cfg.get('kernel_dump_path') and getattr(_pre_art, 'kernel_source', None):\n"
        "                        pathlib.Path(_cfg['kernel_dump_path']).write_text(\n"
        "                            str(_pre_art.kernel_source), encoding='utf-8'\n"
        "                        )\n"
        "                except Exception:\n"
        "                    pass\n"
        "                raise\n"
        "            if _cfg.get('tir_dump_path'):\n"
        "                _tir_txt = str(getattr(_art, 'device_mod', None) or _fn)\n"
        "                pathlib.Path(_cfg['tir_dump_path']).write_text(\n"
        "                    _tir_txt, encoding='utf-8'\n"
        "                )\n"
        "        assert _art.kernel_source\n"
        "        if _cfg.get('kernel_dump_path'):\n"
        "            pathlib.Path(_cfg['kernel_dump_path']).write_text(\n"
        "                str(_art.kernel_source), encoding='utf-8'\n"
        "            )\n"
    )


def run_stage_b_compile(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    python_bin: str | None = None,
    timeout_sec: int = 120,
    enable_device_compile: bool = True,
    artifact_store: ArtifactStore | None = None,
    artifact_id: str = "",
    regime_tag: str = "",
    spec_key: str = "",
) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    py_exe = _resolve_conda_python(python_bin)
    env = dict(os.environ)
    env.setdefault("USE_CUDA", "0")
    _inject_cann_cpath_env(env)
    resolved_store = _resolve_artifact_store(artifact_store)
    resolved_store.root.mkdir(parents=True, exist_ok=True)
    recorded_aid = ""
    recorded_bins: tuple[str, ...] = ()
    norm_pc, norm_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    closure_src = (
        extract_reachable_closure_source(dsl_source, regime_tag=regime_tag)
        if regime_tag
        else dsl_source
    )
    target_aid = artifact_id or compute_compile_task_key(
        normalized_kernel_unit_src=closure_src,
        cann_target="dav-3510",
        pass_configs=norm_pc,
        compile_flags=norm_cf,
        static_spec_signature=spec_key,
    )
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        worker_cache = tmp_root / "tl_cache"
        seed_tl_cache_tree(resolved_store.root, worker_cache)
        pre_bins = snapshot_tl_cache_binaries(worker_cache)
        env["TILELANG_DISABLE_CACHE"] = "0"
        env["TILELANG_CACHE_DIR"] = str(worker_cache)
        env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
        cand_path = tmp_root / "candidate_dsl.py"
        cand_path.write_text(dsl_source, encoding="utf-8")
        kernel_dump_path = tmp_root / "lowered_kernel.cce"
        tir_dump_path = tmp_root / "lowered_tir.tir"
        driver = build_stage_b_driver_source(
            cand_path,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=dims,
            kernel_dump_path=kernel_dump_path,
            tir_dump_path=tir_dump_path,
        )
        try:
            proc = subprocess.run(
                [py_exe, "-c", driver],
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            lowered_dump = (
                kernel_dump_path.read_text(encoding="utf-8", errors="ignore")
                if kernel_dump_path.is_file()
                else ""
            )
            lowered_tir_dump = (
                tir_dump_path.read_text(encoding="utf-8", errors="ignore")
                if tir_dump_path.is_file()
                else ""
            )
            return StageBResult(
                passed=False,
                error_message=f"stage_b_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
                generated_cce=lowered_dump,
                lowered_tir=lowered_tir_dump,
                fault_plane="STAGE_B_COMPILE",
                fault_code="STAGE_B_COMPILE_TIMEOUT",
                failed_tier="stage_b",
            )
        lowered_dump = (
            kernel_dump_path.read_text(encoding="utf-8", errors="ignore")
            if kernel_dump_path.is_file()
            else ""
        )
        lowered_tir_dump = (
            tir_dump_path.read_text(encoding="utf-8", errors="ignore")
            if tir_dump_path.is_file()
            else ""
        )
        combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
            combined_out,
            asm_or_dump_text=lowered_dump,
        )
        ma_profile = (
            extract_microarch_profile_from_cce(lowered_dump).to_dict()
            if lowered_dump.strip()
            else None
        )
        if proc.returncode != 0:
            raw_detail = proc.stderr or proc.stdout or ""
            detail = _bounded_compile_diagnostic(raw_detail)
            fault_code = _classify_stage_b_overflow_fault_code(
                f"{raw_detail}\n{lowered_dump}",
                summary,
                "STAGE_B_COMPILE_FAILED",
            )
            return StageBResult(
                passed=False,
                error_message=f"stage_b_compile_failed (rc={proc.returncode}): {detail}",
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=raw_detail,
                generated_cce=lowered_dump,
                lowered_tir=lowered_tir_dump,
                fault_plane="STAGE_B_COMPILE",
                fault_code=fault_code,
                failed_tier="stage_b",
            )
        if _has_vf_stack_overflow_signal(combined_out, summary, norm_cf, norm_pc):
            return StageBResult(
                passed=False,
                error_message=f"stage_b_vf_stack_overflow: stack_bytes={summary.stack_bytes}",
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=combined_out,
                generated_cce=lowered_dump,
                lowered_tir=lowered_tir_dump,
                fault_plane="STAGE_B_COMPILE",
                fault_code="VF_STACK_OVERFLOW",
                failed_tier="stage_b",
            )
        lock_path = resolved_store.root / ".prewarm_manifest_v1.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                merge_tl_cache_tree(worker_cache, resolved_store.root)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        existing_rec = resolved_store.get_verified_artifact(target_aid)
        post_new_rels = [
            rel
            for rel in collect_new_tl_cache_binaries(worker_cache, pre_bins)
            if rel not in pre_bins or worker_cache.joinpath(rel).stat().st_size != pre_bins[rel][0]
            or worker_cache.joinpath(rel).stat().st_mtime_ns != pre_bins[rel][1]
        ]
        cache_bins = [
            resolved_store.root / rel
            for rel in post_new_rels
            if is_verified_binary_file(resolved_store.root / rel)
        ]
        tmp_bins = [
            p
            for p in sorted(tmp_root.rglob("*"))
            if not p.is_relative_to(worker_cache) and is_verified_binary_file(p)
        ]
        built_bins = [*cache_bins, *tmp_bins]
        if not built_bins and existing_rec is None:
            built_bins = [
                resolved_store.root / rel
                for rel in collect_new_tl_cache_binaries(worker_cache, pre_bins)
                if is_verified_binary_file(resolved_store.root / rel)
            ]
        if built_bins:
            flags_str = " ".join(norm_cf)
            recorded_aid = target_aid
            rec = resolved_store.record_verified_artifact(
                recorded_aid,
                binary_paths=built_bins,
                backend_kind="tilelang",
                flags=flags_str,
                spec_key=spec_key,
                closure_src=closure_src,
                copy_binaries=True,
            )
            recorded_bins = rec.binary_paths
        elif existing_rec is not None:
            recorded_aid = existing_rec.artifact_id
            recorded_bins = existing_rec.binary_paths
    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
        generated_cce=lowered_dump,
        lowered_tir=lowered_tir_dump,
        artifact_id=recorded_aid,
        binary_paths=recorded_bins,
    )


_CANN_INCLUDE_SUBDIRS: tuple[str, ...] = (
    "include",
    "x86_64-linux/asc",
    "x86_64-linux/asc/include",
    "x86_64-linux/include",
    "aarch64-linux/asc",
    "aarch64-linux/asc/include",
    "aarch64-linux/include",
    "runtime/include",
    "asc",
    "asc/include",
    "tools/tikcpp/tikcfw",
    "compiler/tikcpp/tikcfw",
)


def _resolve_cann_root(
    cann_root: str | Path | None = None,
    *,
    bisheng_bin: str | Path | None = None,
) -> Path | None:
    if cann_root is not None and str(cann_root).strip():
        return Path(cann_root).expanduser()
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "BISHENG_HOME"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            cand = Path(raw).expanduser()
            if env_key == "BISHENG_HOME" and not any(
                (cand / sub).is_dir() for sub in _CANN_INCLUDE_SUBDIRS[:4]
            ):
                for parent in cand.parents:
                    if any((parent / sub).is_dir() for sub in _CANN_INCLUDE_SUBDIRS[:4]):
                        return parent
            return cand
    for env_key in ("ASCEND_OPP_PATH", "ASCEND_AICPU_PATH"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            parent = Path(raw).expanduser().parent
            if any((parent / sub).is_dir() for sub in _CANN_INCLUDE_SUBDIRS[:4]):
                return parent
    resolved_bin = bisheng_bin or shutil.which("bisheng")
    if resolved_bin:
        bpath = Path(str(resolved_bin)).resolve()
        for parent in bpath.parents:
            if any((parent / sub).is_dir() for sub in _CANN_INCLUDE_SUBDIRS[:4]):
                return parent
    return None


def _resolve_bisheng_bin(
    bisheng_bin: str | Path | None = None,
    cann_root: str | Path | None = None,
) -> str | None:
    """Resolve bisheng compiler binary from explicit arg, PATH, or CANN installation."""
    if bisheng_bin is not None:
        raw = str(bisheng_bin).strip()
        if raw:
            p = Path(raw)
            if p.is_file():
                return str(p)
            which_explicit = shutil.which(raw)
            if which_explicit:
                return which_explicit
        return None
    bh = os.environ.get("BISHENG_HOME", "").strip()
    if bh:
        bh_cand = Path(bh).expanduser() / "bin" / "bisheng"
        if bh_cand.is_file():
            return str(bh_cand)
    found = shutil.which("bisheng")
    if found:
        return found
    resolved_cann = _resolve_cann_root(cann_root)
    if resolved_cann is not None:
        for rel in ("tools/ccec_compiler/bin/bisheng", "bin/bisheng"):
            candidate = resolved_cann / rel
            if candidate.is_file():
                return str(candidate)
    return None


def _detect_cann_include_dirs(
    cann_root: str | Path | None = None,
    *,
    bisheng_bin: str | Path | None = None,
) -> list[str]:
    """Auto-discover CANN C/C++ header include directories from env and toolchain."""
    roots: list[Path] = []
    if cann_root is not None and str(cann_root).strip():
        roots.append(Path(cann_root).expanduser())
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            roots.append(Path(raw).expanduser())
    inferred = _resolve_cann_root(cann_root, bisheng_bin=bisheng_bin)
    if inferred is not None:
        roots.append(inferred)

    found: list[str] = []
    seen_resolved: set[str] = set()
    for root in roots:
        for sub in _CANN_INCLUDE_SUBDIRS:
            inc = root / sub
            if inc.is_dir():
                res_key = str(inc.resolve())
                if res_key not in seen_resolved:
                    seen_resolved.add(res_key)
                    found.append(str(inc))
    return found


def _inject_cann_cpath_env(env_map: dict[str, str] | None = None) -> list[str]:
    """Prepend discovered CANN include directories to ``CPATH`` without duplicates."""
    cann_incs = _detect_cann_include_dirs()
    if cann_incs:
        target_env = os.environ if env_map is None else env_map
        existing = [p for p in target_env.get("CPATH", "").split(os.pathsep) if p]
        target_env["CPATH"] = os.pathsep.join(dict.fromkeys([*cann_incs, *existing]))
    return cann_incs


def _resolve_toolchain_tool(
    tool_names: Sequence[str],
    *,
    bisheng_bin: str | Path | None = None,
) -> str | None:
    """Resolve a companion binutils/LLVM tool near bisheng or on PATH."""
    if bisheng_bin is not None:
        bpath = Path(str(bisheng_bin))
        if bpath.is_file():
            for name in tool_names:
                cand = bpath.resolve().with_name(name)
                if cand.is_file():
                    return str(cand)
    for name in tool_names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _verify_object_has_instantiated_kernel(
    obj_path: Path,
    *,
    bisheng_bin: str | Path | None = None,
) -> tuple[bool, str]:
    """Ensure bisheng emitted at least one concrete instantiated kernel symbol in .text."""
    if not obj_path.is_file():
        return False, "stage_b_ascendc_missing_object: object file was not generated"
    nm_bin = _resolve_toolchain_tool(("llvm-nm", "nm"), bisheng_bin=bisheng_bin)
    if nm_bin:
        proc = subprocess.run(
            [nm_bin, "--defined-only", str(obj_path)],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0:
            defined_text_syms = [
                ln.strip()
                for ln in (proc.stdout or "").splitlines()
                if re.search(r"\b[Tt]\s+(?!\$)\S+", ln)
            ]
            if not defined_text_syms:
                return False, (
                    "stage_b_ascendc_uninstantiated_template: object file contains zero "
                    "instantiated code symbols (.text is empty); provide concrete template "
                    "instantiation via _render_trackb_tu"
                )
            return True, ""

    objdump_bin = _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin)
    if objdump_bin:
        proc = subprocess.run(
            [objdump_bin, "-h", str(obj_path)],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0:
            text_sec = re.search(r"\.text(?:\.\S+)?\s+([0-9a-fA-F]+)\s", proc.stdout or "")
            if text_sec and int(text_sec.group(1), 16) > 0:
                return True, ""
            return False, (
                "stage_b_ascendc_uninstantiated_template: .text section size is 0; "
                "provide concrete template instantiation via _render_trackb_tu"
            )

    if obj_path.stat().st_size > 2048:
        return True, ""
    return False, (
        "stage_b_ascendc_uninstantiated_template: object file contains no verifiable "
        "instantiated kernel code"
    )


def _disassemble_object(
    obj_path: Path,
    *,
    bisheng_bin: str | Path | None = None,
) -> str:
    """Run llvm-objdump -d on a compiled Ascend object file."""
    objdump_bin = _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin)
    if not objdump_bin or not obj_path.is_file():
        return ""
    proc = subprocess.run(
        [objdump_bin, "-d", str(obj_path)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _augment_bisheng_telemetry_text(
    compiler_out: str,
    objdump_text: str,
) -> str:
    """Normalize bisheng --cce-res-usage and llvm-objdump -d metrics for bisheng_res_parser."""
    vreg_vals = [int(m) for m in re.findall(r"\bVReg\s+number\s*:\s*(\d+)", compiler_out)]
    stack_vals = [
        int(m)
        for m in re.findall(
            r"(?:^|[,\s])Stack\s+size\s*:\s*(\d+)\s*bytes",
            compiler_out,
            flags=re.IGNORECASE,
        )
    ]
    dis = _parse_objdump_pipe_counts(objdump_text)
    vec_regs = max([*vreg_vals, dis["vec_regs"], 0])
    scalar_regs = dis["scalar_regs"]
    stack_bytes = max(stack_vals, default=0)

    extra_lines: list[str] = []
    if vec_regs > 0 and not re.search(
        r"(?:vector\s+register\s+usage|vec_reg_num)", compiler_out, re.IGNORECASE
    ):
        extra_lines.append(f"Vector Register Usage: {vec_regs}")
    if scalar_regs > 0 and not re.search(
        r"(?:scalar\s+register\s+usage|scalar_reg_num)", compiler_out, re.IGNORECASE
    ):
        extra_lines.append(f"Scalar Register Usage: {scalar_regs}")
    if stack_bytes > 0 and not re.search(r"stack\s+frame\s+size", compiler_out, re.IGNORECASE):
        extra_lines.append(f"Stack Frame Size: {stack_bytes} bytes")
    if objdump_text.strip():
        vloop_cnt = dis["vloopv2"]
        sw_cnt = dis["vf_sw_back_branches"] if vloop_cnt > 0 else dis["sw_back_branches"]
        extra_lines.append(f"vloopv2_loops: {vloop_cnt}")
        extra_lines.append(f"software_loops: {sw_cnt}")
    if not extra_lines:
        return compiler_out
    return f"{compiler_out}\n" + "\n".join(extra_lines)


_TRANSFER_ROUTES: tuple[str, ...] = (
    "gm_to_l1",
    "gm_to_ub",
    "l1_to_l0a",
    "l1_to_l0b",
    "l0c_to_ub",
    "l0c_to_l1",
    "l0c_to_gm",
    "ub_to_gm",
    "ub_to_l1",
)

_TRANSFER_CALL_SPECS: dict[str, tuple[str, int, int, str]] = {
    "gm_to_l1": ("asc_copy_gm2l1", 4, 5, "elem"),
    "gm_to_ub": ("asc_copy_gm2ub", 2, 3, "bytes"),
    "l1_to_l0a": ("asc_copy_l12l0a", 4, 5, "elem"),
    "l1_to_l0b": ("asc_copy_l12l0b", 4, 5, "elem"),
    "l0c_to_ub": ("asc_copy_l0c2ub", 3, 2, "float"),
    "l0c_to_l1": ("asc_copy_l0c2l1", 3, 2, "float"),
    "l0c_to_gm": ("asc_copy_l0c2gm", 3, 2, "float"),
    "ub_to_gm": ("asc_copy_ub2gm", 2, 3, "bytes"),
}


def _eval_call_arg(expr: str, consts: Mapping[str, int], elem_bytes: int) -> int | None:
    """Evaluate one data-copy call argument with sizeof and template constants folded."""
    type_bytes = {"InT": elem_bytes, "half": 2, "float": 4, "bfloat16_t": 2}
    return _safe_eval_cpp_int(_rewrite_sizeof(_strip_cpp_casts(expr), type_bytes), consts)


def _call_elem_bytes(call_text: str) -> int:
    """Return the element width named in one copy call, defaulting to half precision."""
    for type_name, width in _CPP_DTYPE_BYTES.items():
        if re.search(rf"\b{re.escape(type_name)}\b", call_text):
            return width
    return 2


def _source_call_payloads(
    cpp_source: str,
    route: str,
    consts: Mapping[str, int],
) -> list[int]:
    """Collect copy payloads from CBE-style call arguments for one transfer route."""
    spec = _TRANSFER_CALL_SPECS.get(route)
    if spec is None:
        return []
    prefix, rows_idx, cols_idx, kind = spec
    payloads: list[int] = []
    for call in re.finditer(rf"\b{prefix}\w*\s*\(([^;]*)\)\s*;", cpp_source, re.S):
        text = call.group(0)
        args = _split_cpp_args(call.group(1))
        if len(args) <= max(rows_idx, cols_idx):
            continue
        elem_bytes = _call_elem_bytes(text)
        rows = _eval_call_arg(args[rows_idx], consts, elem_bytes)
        cols = _eval_call_arg(args[cols_idx], consts, elem_bytes)
        if rows is None or cols is None or rows <= 0 or cols <= 0:
            continue
        if kind == "bytes":
            payloads.append(rows * cols)
        elif kind == "float":
            payloads.append(rows * cols * 4)
        else:
            payloads.append(rows * cols * elem_bytes)
    return payloads


def _tile_shape_payload(cpp_source: str, route: str, consts: Mapping[str, int]) -> int:
    """Estimate one transfer payload from C++20 tile extents when a call cannot be folded."""
    shape_keys = (
        "CUBE_TILE_M",
        "CUBE_TILE_N",
        "CUBE_TILE_K",
        "BM",
        "BN",
        "BK",
        "TILE_M",
        "TILE_N",
        "TILE_K",
    )
    if not any(key in consts for key in shape_keys):
        return 0
    tile_m = max(1, consts.get("CUBE_TILE_M", consts.get("BM", consts.get("TILE_M", 64))))
    tile_n = max(1, consts.get("CUBE_TILE_N", consts.get("BN", consts.get("TILE_N", 64))))
    tile_k = max(1, consts.get("CUBE_TILE_K", consts.get("BK", consts.get("TILE_K", 64))))
    shape_source = re.sub(r"\bconstexpr\s+[^;]+;", "", cpp_source)
    elem_bytes = _call_elem_bytes(shape_source)
    if route == "gm_to_l1":
        return max(tile_m * tile_k, tile_k * tile_n) * elem_bytes
    if route in ("gm_to_ub", "ub_to_gm", "ub_to_l1"):
        return tile_m * tile_n * elem_bytes
    if route == "l1_to_l0a":
        return tile_m * tile_k * elem_bytes
    if route == "l1_to_l0b":
        return tile_k * tile_n * elem_bytes
    return tile_m * tile_n * 4


def _median_payload(values: Sequence[int]) -> int:
    """Return the integer median of a nonempty payload sample."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _estimate_transfer_granularity(cpp_source: str) -> dict[str, int]:
    """Estimate per-instruction transfer bytes from copy-call operands or tile extents."""
    consts, type_bytes = _resolve_cpp_consts_and_types(cpp_source)
    estimates: dict[str, int] = {}
    for route in _TRANSFER_ROUTES:
        source_payloads = _source_call_payloads(cpp_source, route, consts)
        if source_payloads:
            estimates[route] = _median_payload(source_payloads)
            continue
        tile_payload = _tile_shape_payload(cpp_source, route, consts)
        if tile_payload > 0:
            estimates[route] = tile_payload
    return estimates


def _merge_disassembly_microarch_profile(
    cpp_source: str,
    objdump_text: str,
) -> dict[str, object] | None:
    if not cpp_source.strip() and not objdump_text.strip():
        return None
    base = extract_microarch_profile_from_cce(cpp_source)
    dis = _parse_objdump_pipe_counts(objdump_text)
    if dis["total_insns"] == 0:
        return base.to_dict()
    granularity = _estimate_transfer_granularity(cpp_source)
    updates: dict[str, object] = {
        "fp_alu_ops": max(base.fp_alu_ops, dis["vec"]),
        "cube_mmad_count": max(base.cube_mmad_count, dis["cube_mmad"]),
        "cv_sync_barriers": max(base.cv_sync_barriers, dis["sync"]),
    }
    byte_routes = (
        ("gm_to_l1_insns", "gm_to_l1_bytes"),
        ("gm_to_ub_insns", "gm_to_ub_bytes"),
        ("l1_to_l0a_insns", "l1_to_l0a_bytes"),
        ("l1_to_l0b_insns", "l1_to_l0b_bytes"),
        ("l0c_to_ub_insns", "l0c_to_ub_bytes"),
        ("l0c_to_l1_insns", "l0c_to_l1_bytes"),
        ("l0c_to_gm_insns", "l0c_to_gm_bytes"),
        ("ub_to_gm_insns", "ub_to_gm_bytes"),
        ("ub_to_l1_insns", "ub_to_l1_bytes"),
    )
    for count_key, byte_field in byte_routes:
        route = byte_field.removesuffix("_bytes")
        if dis[count_key] > 0 and route in granularity:
            updates[byte_field] = dis[count_key] * granularity[route]
        else:
            updates[byte_field] = getattr(base, byte_field)
    return replace(base, **updates).to_dict()


def _detect_tilelang_include_dir() -> str | None:
    try:
        import tilelang  # noqa: PLC0415

        src_dir = Path(tilelang.__file__).resolve().parents[1] / "src"
        if (src_dir / "tl_templates" / "ascend").is_dir():
            return str(src_dir)
    except Exception:
        pass
    env_tl = (
        os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", "").strip()
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", "").strip()
    )
    if env_tl:
        src_dir = Path(env_tl).expanduser() / "src"
        if (src_dir / "tl_templates" / "ascend").is_dir():
            return str(src_dir)
    sibling = Path(__file__).resolve().parents[5].parent / "tilelang" / "src"
    if (sibling / "tl_templates" / "ascend").is_dir():
        return str(sibling)
    return None


def run_stage_b_ascendc_compile(
    cpp_source: str,
    *,
    arch: str = "dav-3510",
    compile_flags: Sequence[str] | str | None = None,
    bisheng_bin: str | Path | None = None,
    cann_root: str | Path | None = None,
    extra_include_dirs: Sequence[str | Path] | str | Path | None = None,
    timeout_sec: int = 120,
    enable_device_compile: bool = True,
    verify_shared_link: bool = False,
    artifact_store: ArtifactStore | None = None,
    artifact_id: str = "",
    regime_tag: str = "",
    spec_key: str = "",
) -> StageBResult:
    """Verify dav-3510 C++20 translation unit compilation via bisheng in an isolated tempdir."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return StageBResult(
            passed=False,
            error_message="stage_b_ascendc_empty_source",
            elapsed_sec=time.monotonic() - t0,
            fault_plane="STAGE_B_COMPILE",
            fault_code="STAGE_B_ASCENDC_EMPTY_SOURCE",
            failed_tier="stage_b",
        )

    resolved_bisheng = _resolve_bisheng_bin(bisheng_bin, cann_root=cann_root)
    if resolved_bisheng is None:
        return StageBResult(
            passed=False,
            error_message="stage_b_ascendc_bisheng_not_found",
            elapsed_sec=time.monotonic() - t0,
            fault_plane="INFRA_TOOLCHAIN",
            fault_code="BISHENG_NOT_FOUND",
            failed_tier="stage_b",
        )

    extra_flags = _normalize_flag_list(compile_flags)
    has_explicit_std = any(f.startswith("-std=") for f in extra_flags)
    has_explicit_x = "-x" in extra_flags
    has_tl_templates = "tl_templates/ascend/" in cpp_source or "c_api/asc_" in cpp_source

    std_flags: list[str] = [] if has_explicit_std else ["-std=c++20"]

    include_flags: list[str] = []
    seen_inc_flags = set(extra_flags)

    def _add_inc(inc_dir: str) -> None:
        flag = f"-I{inc_dir}"
        if inc_dir and Path(inc_dir).is_dir() and flag not in seen_inc_flags:
            seen_inc_flags.add(flag)
            include_flags.append(flag)

    if extra_include_dirs is not None:
        extra_inc_list = (
            [extra_include_dirs]
            if isinstance(extra_include_dirs, (str, Path))
            else list(extra_include_dirs)
        )
        for inc_item in extra_inc_list:
            _add_inc(str(inc_item).strip())
    if has_tl_templates or True:
        tl_inc = _detect_tilelang_include_dir()
        if tl_inc:
            _add_inc(tl_inc)
    for cann_inc in _detect_cann_include_dirs(cann_root, bisheng_bin=resolved_bisheng):
        _add_inc(cann_inc)

    env = dict(os.environ)
    resolved_cann = _resolve_cann_root(cann_root, bisheng_bin=resolved_bisheng)
    cann_root_str = str(resolved_cann).strip() if resolved_cann is not None else ""
    if cann_root_str and Path(cann_root_str).is_dir() and not env.get("ASCEND_HOME_PATH"):
        env["ASCEND_HOME_PATH"] = cann_root_str

    resolved_store = _resolve_artifact_store(artifact_store)
    recorded_aid = ""
    recorded_bins: tuple[str, ...] = ()
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_ascendc_") as tmp_dir:
        src_name = "kernel.cpp" if has_explicit_x else "kernel.asc"
        src_path = Path(tmp_dir) / src_name
        obj_path = Path(tmp_dir) / "kernel.o"
        src_path.write_text(cpp_source, encoding="utf-8")

        def _invoke(*, use_legacy_cce_flags: bool) -> subprocess.CompletedProcess[str]:
            arch_flags = (
                ["-x", "cce", f"--cce-aicore-arch={arch}"]
                if use_legacy_cce_flags
                else [f"--npu-arch={arch}"]
            )
            dep_flags = (
                []
                if any("Wdeprecated-declarations" in str(flag) for flag in extra_flags)
                else ["-Wno-deprecated-declarations"]
            )
            cmd = [
                resolved_bisheng,
                *arch_flags,
                "--cce-aicore-only",
                "--cce-res-usage",
                "-O2",
                *std_flags,
                *include_flags,
                *dep_flags,
                *extra_flags,
                "-c",
                str(src_path),
                "-o",
                str(obj_path),
            ]
            return subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                env=env,
                timeout=timeout_sec,
            )

        try:
            proc = _invoke(use_legacy_cce_flags=False)
            if proc.returncode != 0:
                combined_err = (proc.stderr or "") + "\n" + (proc.stdout or "")
                low_err = combined_err.lower()
                if "--npu-arch" in low_err and any(
                    tok in low_err for tok in ("unknown", "unrecognized", "unsupported", "invalid")
                ):
                    proc = _invoke(use_legacy_cce_flags=True)
        except subprocess.TimeoutExpired:
            return StageBResult(
                passed=False,
                error_message=f"stage_b_ascendc_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
                fault_plane="STAGE_B_COMPILE",
                fault_code="STAGE_B_ASCENDC_COMPILE_TIMEOUT",
                failed_tier="stage_b",
            )

        combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or not obj_path.is_file():
            spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
                combined_out,
                asm_or_dump_text=cpp_source,
            )
            ma_profile = (
                extract_microarch_profile_from_cce(cpp_source).to_dict()
                if cpp_source.strip()
                else None
            )
            raw_detail = (
                proc.stderr or proc.stdout or "bisheng failed without diagnostics"
            )
            detail = _bounded_compile_diagnostic(raw_detail)
            fault_code = _classify_stage_b_overflow_fault_code(
                raw_detail,
                summary,
                "STAGE_B_ASCENDC_COMPILE_FAILED",
            )
            return StageBResult(
                passed=False,
                error_message=f"stage_b_ascendc_compile_failed (rc={proc.returncode}): {detail}",
                elapsed_sec=time.monotonic() - t0,
                generated_cce=cpp_source,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=raw_detail,
                fault_plane="STAGE_B_COMPILE",
                fault_code=fault_code,
                failed_tier="stage_b",
            )

        inst_ok, inst_err = _verify_object_has_instantiated_kernel(
            obj_path, bisheng_bin=resolved_bisheng
        )
        objdump_text = _disassemble_object(obj_path, bisheng_bin=resolved_bisheng)
        telemetry_out = _augment_bisheng_telemetry_text(combined_out, objdump_text)
        dump_text = f"{objdump_text}\n{cpp_source}" if objdump_text.strip() else cpp_source
        spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
            telemetry_out,
            asm_or_dump_text=dump_text,
        )
        ma_profile = _merge_disassembly_microarch_profile(cpp_source, objdump_text)
        if not inst_ok:
            return StageBResult(
                passed=False,
                error_message=inst_err,
                elapsed_sec=time.monotonic() - t0,
                generated_cce=cpp_source,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                fault_plane="STAGE_B_COMPILE",
                fault_code="STAGE_B_ASCENDC_UNINSTANTIATED_TEMPLATE",
                failed_tier="stage_b",
            )
        if _has_vf_stack_overflow_signal(telemetry_out, summary, extra_flags):
            return StageBResult(
                passed=False,
                error_message=(
                    f"stage_b_ascendc_vf_stack_overflow: stack_bytes={summary.stack_bytes}"
                ),
                elapsed_sec=time.monotonic() - t0,
                generated_cce=cpp_source,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=telemetry_out,
                fault_plane="STAGE_B_COMPILE",
                fault_code="VF_STACK_OVERFLOW",
                failed_tier="stage_b",
            )

        so_path_opt: Path | None = None
        if (
            verify_shared_link
            and 'extern "C"' in cpp_source
            and "<<<" in cpp_source
            and not has_explicit_x
            and env.get("ASCEND_HOME_PATH")
        ):
            so_path = Path(tmp_dir) / "kernel.so"
            link_cmd = [
                resolved_bisheng,
                "-O2",
                "-fPIC",
                *std_flags,
                "-mllvm",
                "-cce-aicore-dcpreload-args=false",
                f"--npu-arch={arch}",
                *include_flags,
                *extra_flags,
                "-Wl,--no-mmap-output-file",
                "-shared",
                str(src_path),
                "-o",
                str(so_path),
            ]
            try:
                link_proc = subprocess.run(
                    link_cmd,
                    text=True,
                    capture_output=True,
                    cwd=tmp_dir,
                    env=env,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return StageBResult(
                    passed=False,
                    error_message=f"stage_b_ascendc_shared_link_timeout:{timeout_sec}s",
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code="STAGE_B_ASCENDC_SHARED_LINK_TIMEOUT",
                    failed_tier="stage_b",
                )
            link_raw = (link_proc.stderr or "") + "\n" + (link_proc.stdout or "")
            if link_proc.returncode != 0 or not so_path.is_file():
                link_detail = _bounded_compile_diagnostic(link_raw)
                link_fault_code = _classify_stage_b_overflow_fault_code(
                    link_raw,
                    summary,
                    "STAGE_B_ASCENDC_SHARED_LINK_FAILED",
                )
                return StageBResult(
                    passed=False,
                    error_message=(
                        f"stage_b_ascendc_shared_link_failed (rc={link_proc.returncode}): "
                        f"{link_detail}"
                    ),
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    full_diagnostic=link_raw,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code=link_fault_code,
                    failed_tier="stage_b",
                )
            so_path_opt = so_path

        built_bins = [
            p
            for p in (obj_path, so_path_opt)
            if p is not None and p.is_file() and p.stat().st_size > 0
        ]
        if built_bins:
            closure_src = _normalize_trackb_unit_source(cpp_source, regime_tag=regime_tag)
            flags_str = " ".join(extra_flags)
            recorded_aid = artifact_id or compute_compile_task_key(
                normalized_kernel_unit_src=closure_src,
                cann_target=arch,
                pass_configs=None,
                compile_flags=extra_flags,
                static_spec_signature=spec_key,
            )
            rec = resolved_store.record_verified_artifact(
                recorded_aid,
                binary_paths=built_bins,
                backend_kind="ascendc",
                arch=arch,
                flags=flags_str,
                spec_key=spec_key,
                closure_src=closure_src,
                copy_binaries=True,
            )
            recorded_bins = rec.binary_paths

    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
        artifact_id=recorded_aid,
        binary_paths=recorded_bins,
    )


def synthesize_pybind_extension_wrapper(
    op_name: str,
    kernel_launch_decl: str,
    tensor_arg_names: Sequence[str],
    *,
    output_arg_names: Sequence[str] | None = None,
    scalar_params: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Generate a thin torch_npu pybind11 wrapper preserving Python op_fn(*inputs) ABI."""
    safe_op = re.sub(r"[^A-Za-z0-9_]", "_", (op_name or "ascend_op").strip())
    if not safe_op or safe_op[0].isdigit():
        safe_op = f"op_{safe_op}"
    wrapper_fn = f"py_{safe_op}_forward"

    decl_text = str(kernel_launch_decl or "").strip()
    m_fn = re.search(r"\b([A-Za-z_]\w*)\s*\(", decl_text)
    if m_fn:
        launch_fn = m_fn.group(1)
        decl_stmt = decl_text if decl_text.endswith(";") or decl_text.endswith("}") else f"{decl_text};"
    else:
        launch_fn = re.sub(r"[^A-Za-z0-9_]", "_", decl_text or f"launch_{safe_op}")
        if not launch_fn or launch_fn[0].isdigit():
            launch_fn = f"launch_{safe_op}"
        arg_decls = ", ".join(
            ["aclrtStream stream"]
            + [f"void* {re.sub(r'[^A-Za-z0-9_]', '_', a)}" for a in tensor_arg_names]
            + [f"{ptype} {pname}" for ptype, pname in (scalar_params or ())]
        )
        decl_stmt = f'extern "C" void {launch_fn}({arg_decls});'

    clean_tensors = [re.sub(r"[^A-Za-z0-9_]", "_", str(a).strip()) or f"t{i}" for i, a in enumerate(tensor_arg_names)]
    clean_scalars = [
        (str(ptype).strip(), re.sub(r"[^A-Za-z0-9_]", "_", str(pname).strip()) or f"s{i}")
        for i, (ptype, pname) in enumerate(scalar_params or ())
    ]
    cpp_params = [f"const at::Tensor& {t}" for t in clean_tensors] + [
        f"{ptype} {pname}" for ptype, pname in clean_scalars
    ]

    checks = "\n".join(
        f'    TORCH_CHECK({t}.is_contiguous(), "Expected contiguous NPU tensor for {t}");'
        for t in clean_tensors
    )
    stream_line = (
        f"    auto npu_stream = c10_npu::getCurrentNPUStream({clean_tensors[0]}.device().index());"
        if clean_tensors
        else "    auto npu_stream = c10_npu::getCurrentNPUStream();"
    )
    launch_args = ["npu_stream.stream()"] + [f"{t}.data_ptr()" for t in clean_tensors] + [
        pname for _ptype, pname in clean_scalars
    ]

    if output_arg_names:
        outs = [re.sub(r"[^A-Za-z0-9_]", "_", str(o).strip()) for o in output_arg_names]
        if len(outs) == 1:
            ret_type = "at::Tensor"
            ret_stmt = f"    return {outs[0]};"
        else:
            ret_type = "std::tuple<" + ", ".join("at::Tensor" for _ in outs) + ">"
            ret_stmt = f"    return std::make_tuple({', '.join(outs)});"
    elif clean_tensors:
        ret_type = "at::Tensor"
        ret_stmt = f"    return {clean_tensors[-1]};"
    else:
        ret_type = "void"
        ret_stmt = ""

    body_lines = [
        "#include <torch/extension.h>",
        "#include <torch_npu/csrc/core/npu/NPUStream.h>",
        "#include <acl/acl.h>",
        "#include <tuple>",
        "",
        decl_stmt,
        "",
        f"{ret_type} {wrapper_fn}({', '.join(cpp_params)}) {{",
    ]
    if checks:
        body_lines.append(checks)
    body_lines.extend(
        [
            stream_line,
            f"    {launch_fn}({', '.join(launch_args)});",
        ]
    )
    if ret_stmt:
        body_lines.append(ret_stmt)
    body_lines.extend(
        [
            "}",
            "",
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
            f'    m.def("{safe_op}", &{wrapper_fn}, "Ascend 950 torch_npu extension binding for {safe_op}");',
            "}",
            "",
        ]
    )
    return "\n".join(body_lines)


def _resolve_agent_gates_bin() -> str:
    env_bin = os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    found = shutil.which("agent-gates")
    if found:
        return found
    candidates: list[Path] = []
    env_file = os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    if env_file:
        env_path = Path(env_file).expanduser().resolve()
        for ag_root in (env_path.parent, env_path.parent.parent):
            candidates.append(ag_root / ".venv" / "bin" / "agent-gates")
    repo_root = Path(__file__).resolve().parents[5]
    candidates.append(repo_root.parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "Cannot locate 'agent-gates'. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or add to PATH."
    )


def verify_agent_gates_receipt(
    agent_gates_bin: str,
    run_id: str,
    *,
    source_root: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_sec: float = 30.0,
) -> tuple[bool, str]:
    """Verify one AgentGates receipt by run ID and require an authoritative PASS."""
    try:
        proc = subprocess.run(
            [agent_gates_bin, "verify", "--run-id", run_id, "--json"],
            text=True,
            capture_output=True,
            cwd=str(Path(source_root).resolve()),
            env=dict(env) if env is not None else None,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"agent-gates verify timed out after {timeout_sec:.0f}s"
    parsed: dict[str, object] = {}
    brace_idx = proc.stdout.find("{")
    if brace_idx >= 0:
        try:
            loaded = json.loads(proc.stdout[brace_idx:])
        except json.JSONDecodeError as exc:
            return False, f"agent-gates verify returned malformed JSON: {exc}"
        if isinstance(loaded, dict):
            parsed = loaded
    status = str(parsed.get("status", ""))
    verified_run_id = str(parsed.get("runId", ""))
    valid = parsed.get("valid") is True
    if proc.returncode == 0 and status == "PASS" and verified_run_id == run_id and valid:
        return True, ""
    detail = (proc.stderr or proc.stdout or "agent-gates verify failed").strip()
    return False, (
        "receipt_verification_failed: "
        f"status={status or 'MISSING'} runId={verified_run_id or 'MISSING'} "
        f"valid={valid} {detail[-1000:]}"
    )


def run_stage_c_agent_gates(
    ledger: ObjectivePhysicalLedger,
    *,
    source_root: str | Path,
    device_id: str = "1",
    env_file: str | Path | None = None,
    device_test_rel: str | None = None,
    telemetry_report_path: str | Path | None = None,
    gate: str = "tilefoundry-npu-device",
    timeout_sec: int = 300,
) -> StageCResult:
    """Execute lease-locked NPU validation and msprof collection strictly via AgentGates."""
    ag_bin = _resolve_agent_gates_bin()
    src_path = Path(source_root).resolve()
    cmd = [
        ag_bin,
        "run",
        "--adapter",
        "tilefoundry-npu",
        "--source",
        str(src_path),
        "--gate",
        gate,
        "--device-id",
        str(device_id),
        "--json",
    ]
    with tempfile.TemporaryDirectory(prefix="tf_stage_c_env_") as tmp_dir:
        if device_test_rel and env_file and Path(env_file).is_file():
            tmp_env = Path(tmp_dir) / "env.sh"
            lines = [
                ln
                for ln in Path(env_file).read_text(encoding="utf-8").splitlines()
                if not ln.startswith("TILEFOUNDRY_NPU_DEVICE_TESTS=")
            ]
            lines.append(f'TILEFOUNDRY_NPU_DEVICE_TESTS="{device_test_rel}"')
            tmp_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cmd.extend(["--env-file", str(tmp_env)])
        elif env_file and Path(env_file).is_file():
            cmd.extend(["--env-file", str(env_file)])

        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(src_path),
            timeout=timeout_sec,
        )
        parsed: dict[str, object] = {}
        brace_idx = proc.stdout.find("{")
        if brace_idx >= 0:
            try:
                loaded = json.loads(proc.stdout[brace_idx:])
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                parsed = loaded

        status = str(parsed.get("status", "FAIL"))
        run_id = str(parsed.get("runId", "")).strip()
        receipt_path = str(parsed.get("receiptPath", ""))
        passed = proc.returncode == 0 and status == "PASS"
        run_error = (proc.stderr or proc.stdout or "agent-gates failed")[-1000:]
        verify_error = ""
        if passed and not run_id:
            passed = False
            run_error = "agent-gates run reported PASS without a runId"
        elif passed:
            passed, verify_error = verify_agent_gates_receipt(
                ag_bin,
                run_id,
                source_root=src_path,
            )

    telemetry: dict[str, object] = {}
    if passed and telemetry_report_path and Path(telemetry_report_path).is_file():
        try:
            telemetry = json.loads(Path(telemetry_report_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            telemetry = {}

    latency_us = float(telemetry.get("dur_us", telemetry.get("measured_latency_us", 0.0)))
    residual = compute_static_dynamic_residual(ledger, telemetry)
    err = "" if passed else (verify_error or run_error)[-1000:]
    return StageCResult(
        passed=passed,
        accuracy_passed=passed,
        measured_latency_us=latency_us,
        residual=residual,
        telemetry=telemetry,
        run_id=run_id,
        receipt_path=receipt_path,
        error_message=err,
    )


class GateFunnel:
    """Orchestrates Stage A static -> Stage B bisheng -> Stage C AgentGates NPU verification."""

    def __init__(
        self,
        *,
        target: AscendTarget | None = None,
        skip_stage_b: bool = False,
        compile_runner: Callable[[str], StageBResult] | None = None,
        device_runner: Callable[[str, StageAResult], StageCResult] | None = None,
        enable_track_router: bool = False,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._target = target or AscendTarget("ascend.950pr")
        self._skip_stage_b = skip_stage_b
        self._compile_runner = compile_runner
        self._device_runner = device_runner
        self._enable_track_router = enable_track_router
        self._artifact_store = artifact_store

    def evaluate(
        self,
        dsl_source: str,
        *,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        config_selector_name: str | None = None,
        pass_configs: dict[str, object] | None = None,
        compile_flags: Sequence[str] | str | None = None,
        dims: Mapping[str, object] | None = None,
        var_bindings: Mapping[str, int] | None = None,
        backend_kind: str | None = None,
        freedom_mask: object | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
        regime_tag: str = "",
        spec_key: str = "",
    ) -> FunnelEvaluation:
        """Run Stage A, Stage B, and Stage C sequentially with early rejection."""
        if backend_kind in ("ascendc", "ascendc_template"):
            stage_a = evaluate_ascendc_candidate(
                dsl_source,
                freedom_mask=freedom_mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
            )
        else:
            stage_a = run_stage_a_static(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                target=self._target,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                config_selector_name=config_selector_name,
                dims=dims,
                var_bindings=var_bindings,
                enable_track_router=self._enable_track_router,
                ordered_tensor_shapes=ordered_tensor_shapes,
            )
        if not stage_a.passed or stage_a.ledger is None:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_a",
                stage_a=stage_a,
                error_message=stage_a.error_message,
            )

        if self._skip_stage_b:
            stage_b = StageBResult(passed=True)
        elif self._compile_runner is not None:
            stage_b = self._compile_runner(dsl_source)
        elif backend_kind in ("ascendc", "ascendc_template"):
            stage_b = run_stage_b_ascendc_compile(
                dsl_source,
                compile_flags=compile_flags,
                artifact_store=self._artifact_store,
                regime_tag=regime_tag,
                spec_key=spec_key,
            )
        else:
            stage_b = run_stage_b_compile(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
                artifact_store=self._artifact_store,
                regime_tag=regime_tag,
                spec_key=spec_key,
            )
        if not stage_b.passed:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_b",
                stage_a=stage_a,
                stage_b=stage_b,
                error_message=stage_b.error_message,
            )

        comp_summary = stage_b.compiler_summary
        if self._device_runner is None:
            fallback_telemetry: dict[str, object] = {}
            residual = compute_static_dynamic_residual(
                stage_a.ledger,
                fallback_telemetry,
                compiler_summary=comp_summary,
                roofline=roofline,
            )
            stage_c = StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=0.0,
                residual=residual,
                telemetry=fallback_telemetry,
            )
        else:
            _inject_cann_cpath_env()
            stage_c = self._device_runner(dsl_source, stage_a)
            if comp_summary is not None and stage_c.residual is not None:
                res = stage_c.residual
                sw_loops = comp_summary.software_loops
                vl_loops = comp_summary.vloopv2_loops
                comp_vloop_hit = bool(stage_b.vloopv2_hit) and vl_loops > 0 and sw_loops == 0
                merged_res = replace(
                    res,
                    spill_bytes=max(res.spill_bytes, int(comp_summary.spill_bytes)),
                    vf_fusion_breaks=max(res.vf_fusion_breaks, int(comp_summary.vf_fusion_breaks)),
                    alias_serialized_buffers=max(
                        res.alias_serialized_buffers, int(comp_summary.alias_serialized_buffers)
                    ),
                    dual_issue_ipc_collapse=max(
                        res.dual_issue_ipc_collapse, float(comp_summary.dual_issue_ipc_collapse)
                    ),
                    dcache_dcci_flushes=max(
                        res.dcache_dcci_flushes, int(comp_summary.dcache_dcci_flushes)
                    ),
                    has_local_membar_in_vf_call=(
                        res.has_local_membar_in_vf_call
                        or bool(comp_summary.has_local_membar_in_vf_call)
                    ),
                    vloopv2_hit=(res.vloopv2_hit and comp_vloop_hit),
                    observation=stage_a.ledger.to_observation_5d(
                        compiler_summary=comp_summary,
                        telemetry=stage_c.telemetry,
                        roofline=roofline,
                    ),
                )
                stage_c = replace(stage_c, residual=merged_res)
            elif stage_c.residual is not None and roofline is not None:
                refreshed_res = replace(
                    stage_c.residual,
                    observation=stage_a.ledger.to_observation_5d(
                        compiler_summary=comp_summary,
                        telemetry=stage_c.telemetry,
                        roofline=roofline,
                    ),
                )
                stage_c = replace(stage_c, residual=refreshed_res)

        if not stage_c.passed or not stage_c.accuracy_passed:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_c",
                stage_a=stage_a,
                stage_b=stage_b,
                stage_c=stage_c,
                error_message=stage_c.error_message,
            )

        return FunnelEvaluation(
            passed=True,
            failed_stage=None,
            stage_a=stage_a,
            stage_b=stage_b,
            stage_c=stage_c,
        )


def prescreen_dsl_tile_kwargs(
    dsl_source: str,
    *,
    factory_name: str,
    candidate_kwargs_list: Sequence[Mapping[str, object]],
    target: AscendTarget | None = None,
    var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, object], StageAResult]:
    """Run Stage A across tile kwargs and pick the lowest-cost legal configuration."""
    if not candidate_kwargs_list:
        raise ValueError("candidate_kwargs_list must not be empty")
    legal: list[tuple[tuple[float, int, int], dict[str, object], StageAResult]] = []
    rejections: list[str] = []
    for idx, raw_kwargs in enumerate(candidate_kwargs_list):
        kwargs = dict(raw_kwargs)
        res = run_stage_a_static(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=kwargs,
            target=target,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            config_selector_name=config_selector_name,
            dims=dims,
            var_bindings=var_bindings,
            ordered_tensor_shapes=ordered_tensor_shapes,
        )
        ledger = res.ledger
        if not res.passed or ledger is None:
            rejections.append(f"{kwargs}: {res.error_message or 'stage_a_failed'}")
            continue
        if not ledger.is_legal or ledger.max_overflow_bytes > 0:
            rejections.append(f"{kwargs}: overflow={ledger.max_overflow_bytes}")
            continue
        degraded = [
            buf
            for wl in ledger.waterlines.values()
            for buf in wl.degraded_single_buffers
        ]
        if degraded:
            rejections.append(f"{kwargs}: degraded_single_buffers={degraded}")
            continue
        ii_rank = (
            float(ledger.steady_state_ii)
            if ledger.steady_state_ii is not None and ledger.steady_state_ii > 0
            else float("inf")
        )
        dom_cycles = ledger.pipe_cycles.get(
            ledger.dominant_pipe,
            max((v for v in ledger.pipe_cycles.values() if v > 0), default=0),
        )
        legal.append(((ii_rank, dom_cycles, idx), kwargs, res))
    if not legal:
        detail = "; ".join(rejections[:5])
        raise ValueError(f"no legal tile kwargs passed Stage A prescreen: {detail}")
    legal.sort(key=lambda item: item[0])
    _, best_kwargs, best_stage_a = legal[0]
    return best_kwargs, best_stage_a


def ledger_to_json(ledger: ObjectivePhysicalLedger | None) -> str:
    """Serialize an optional ledger safely for worker prompts."""
    if ledger is None:
        return json.dumps(
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=False,
                steady_state_ii=None,
                pipe_cycles={},
                waterlines={},
            ).to_dict(),
            indent=2,
        )
    return ledger.to_json()


def _extract_concrete_shape_list(raw_shapes: object) -> list[tuple[int, ...]]:
    """Extract concrete integer tensor shapes from single, multi, or nested payloads."""
    if not isinstance(raw_shapes, (list, tuple)) or not raw_shapes:
        return []
    if all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in raw_shapes):
        return [tuple(int(dim) for dim in raw_shapes)]
    out: list[tuple[int, ...]] = []
    for item in raw_shapes:
        if item is None:
            continue
        if isinstance(item, (list, tuple)) and item:
            if all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in item
            ):
                out.append(tuple(int(dim) for dim in item))
            else:
                out.extend(_extract_concrete_shape_list(item))
    return out


def _resolve_proto_output_shapes(
    proto: Mapping[str, object] | None,
    *,
    dims: Mapping[str, object] | None,
    attrs: Mapping[str, object] | None,
) -> list[tuple[int, ...]]:
    """Resolve concrete output shapes from proto output specs when all axes are known."""
    if not isinstance(proto, Mapping):
        return []
    op_block = (
        proto["operator"]
        if "outputs" not in proto and isinstance(proto.get("operator"), Mapping)
        else proto
    )
    raw_outputs = op_block.get("outputs") if isinstance(op_block, Mapping) else None
    if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
        return []
    dim_lookup: dict[str, int] = {}
    for src in (attrs, dims):
        if not isinstance(src, Mapping):
            continue
        for key, val in src.items():
            if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                dim_lookup[str(key).strip().lower()] = int(val)
    raw_layout = ""
    for src in (attrs, dims):
        if isinstance(src, Mapping):
            for lkey in ("inputLayout", "input_layout", "layout"):
                if isinstance(src.get(lkey), str) and str(src[lkey]).strip():
                    raw_layout = str(src[lkey]).strip().upper()
                    break
        if raw_layout:
            break
    resolved: list[tuple[int, ...]] = []
    for item in raw_outputs:
        if not isinstance(item, Mapping):
            continue
        shape_spec = item.get("shape")
        if not isinstance(shape_spec, str) or "[" not in shape_spec:
            continue
        matches = re.findall(r"(?:([A-Za-z0-9_]+)\s*:\s*)?\[([^\]]+)\]", shape_spec)
        chosen = ""
        if raw_layout:
            for tag, inner in matches:
                if tag.strip().upper() == raw_layout:
                    chosen = inner
                    break
        if not chosen and matches:
            chosen = matches[0][1]
        if not chosen:
            continue
        tokens = [tok.strip() for tok in chosen.split(",") if tok.strip()]
        shape_vals: list[int] = []
        ok = bool(tokens)
        for tok in tokens:
            if tok.isdigit():
                shape_vals.append(int(tok))
            else:
                matched = dim_lookup.get(tok.lower())
                if matched is None:
                    ok = False
                    break
                shape_vals.append(matched)
        if ok:
            resolved.append(tuple(shape_vals))
    return resolved


def extract_ordered_tensor_shapes(
    *,
    case_entry: Mapping[str, object] | None = None,
    dims: Mapping[str, object] | None = None,
    proto: Mapping[str, object] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Extract parameter-ordered concrete tensor shapes from case, dims, or proto specs."""
    if ordered_tensor_shapes is not None:
        explicit = _extract_concrete_shape_list(ordered_tensor_shapes)
        return tuple(explicit) if explicit else None

    for container in (case_entry, dims):
        if isinstance(container, Mapping) and "ordered_tensor_shapes" in container:
            explicit = _extract_concrete_shape_list(container.get("ordered_tensor_shapes"))
            if explicit:
                return tuple(explicit)

    inputs: list[tuple[int, ...]] = []
    outputs: list[tuple[int, ...]] = []
    for container in (case_entry, dims):
        if not isinstance(container, Mapping):
            continue
        if not inputs:
            raw_in = (
                container.get("input_shapes")
                if "input_shapes" in container
                else container.get("input_shape")
            )
            inputs = _extract_concrete_shape_list(raw_in)
        if not outputs:
            raw_out = (
                container.get("output_shapes")
                if "output_shapes" in container
                else container.get("output_shape")
            )
            outputs = _extract_concrete_shape_list(raw_out)

    if not outputs and proto is not None:
        raw_attrs = case_entry.get("attrs") if isinstance(case_entry, Mapping) else None
        attrs = raw_attrs if isinstance(raw_attrs, Mapping) else None
        outputs = _resolve_proto_output_shapes(proto, dims=dims, attrs=attrs)

    combined = [*inputs, *outputs]
    return tuple(combined) if combined else None


__all__ = [
    "FunnelEvaluation",
    "GateFunnel",
    "StageAResult",
    "StageBResult",
    "StageCResult",
    "ascend_compile_session",
    "build_stage_b_driver_source",
    "check_npu_entrypoint_purity",
    "check_repair_monotonicity",
    "evaluate_ascendc_candidate",
    "extract_ordered_tensor_shapes",
    "ledger_to_json",
    "parse_cce_res_usage",
    "prescreen_dsl_tile_kwargs",
    "resolve_trackb_compile_flags",
    "run_stage_a_static",
    "run_stage_b_ascendc_compile",
    "run_stage_b_compile",
    "run_stage_c_agent_gates",
    "synthesize_pybind_extension_wrapper",
    "verify_agent_gates_receipt",
]
