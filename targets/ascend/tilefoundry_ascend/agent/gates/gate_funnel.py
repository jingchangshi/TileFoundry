"""Three-stage verification funnel: Stage A static ledger -> Stage B bisheng -> Stage C NPU."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tilefoundry_ascend.agent.gates.stage_b_gate import (
    StageBResult, _bounded_compile_diagnostic, _classify_stage_b_overflow_fault_code,
    _detect_cann_include_dirs, _detect_tilelang_include_dir, _disassemble_object,
    _filter_kwargs_for_callable, _inject_cann_cpath_env, _merge_dims_into_kwargs,
    _merge_pass_configs_and_flags, _normalize_flag_list, _normalize_trackb_unit_source,
    _parse_cce_res_usage_with_summary, _parse_objdump_pipe_counts, _resolve_artifact_store,
    _resolve_bisheng_bin, _split_selector_output, _verify_object_has_instantiated_kernel,
    ascend_compile_session, build_stage_b_driver_source,
    compile_trackb_shared_so as _compile_trackb_shared_so_impl,
    compute_trackb_shared_so_filename, has_unreserved_vf_stack_overflow, parse_cce_res_usage,
    resolve_trackb_compile_flags, run_stage_b_ascendc_compile as _run_stage_b_ascendc_compile_impl,
    run_stage_b_compile as _run_stage_b_compile_impl,
)
from tilefoundry_ascend.agent.specs.anchor_specs import build_anchor_var_bindings
from tilefoundry_ascend.agent.specs.latency_algebra import StaticDynamicResidual, compute_static_dynamic_residual
from tilefoundry_ascend.agent.specs.ops_registry import infer_module_runtime_backend
from tilefoundry_ascend.ascend.compile_cache.artifact_store import ArtifactStore
from tilefoundry_ascend.ascend.compile_cache.cpp_closure import extract_reachable_trackb_source, strip_cpp_comments_and_strings
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary, count_reachable_ub_scalar_compute_loops
from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile, extract_microarch_profile_from_cce
from tilefoundry_ascend.ascend.passes.observation_space import RooflineSummaryLike
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger, ScopeWaterline, build_objective_physical_ledger, build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import EnvelopeInspectionResult, inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios
from tilefoundry_ascend.ascend.sil.verifier import verify_tilelang_ast
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.target.ast_semantic_gate import check_tilelang_semantic_contracts
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


def _stage_a_fail(
    t0: float, error_message: str, fault_code: str, *, backend_kind: str = "tilelang",
    prim_func: object | None = None, ledger: ObjectivePhysicalLedger | None = None,
    features: dict[str, object] | None = None, route: object | None = None,
) -> StageAResult:
    return StageAResult(
        passed=False, prim_func=prim_func, ledger=ledger, features=features, route=route,
        error_message=error_message, elapsed_ms=(time.monotonic() - t0) * 1000.0,
        backend_kind=backend_kind, fault_plane="STAGE_A_STATIC", fault_code=fault_code, failed_tier="stage_a",
    )


def _ascend_target_context() -> contextlib.AbstractContextManager[None]:
    return ascend_compile_session("ascend")


def compile_trackb_shared_so(cpp_source: str, **kwargs: object) -> tuple[bool, Path | None, str]:
    """Compile a Track-B C++20 TU into a content-addressed cached .so via bisheng."""
    fn: Callable[..., tuple[bool, Path | None, str]] = _compile_trackb_shared_so_impl
    return fn(cpp_source, subproc_run=subprocess.run, **kwargs)


def run_stage_b_compile(dsl_source: str, **kwargs: object) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    fn: Callable[..., StageBResult] = _run_stage_b_compile_impl
    return fn(dsl_source, subproc_run=subprocess.run, **kwargs)


def run_stage_b_ascendc_compile(cpp_source: str, **kwargs: object) -> StageBResult:
    """Verify dav-3510 C++20 translation unit compilation via bisheng in an isolated tempdir."""
    fn: Callable[..., StageBResult] = _run_stage_b_ascendc_compile_impl
    return fn(
        cpp_source,
        subproc_run=subprocess.run,
        disassembly_profile_fn=_merge_disassembly_microarch_profile,
        bisheng_resolver=_resolve_bisheng_bin,
        verify_instantiated_fn=_verify_object_has_instantiated_kernel,
        disassemble_fn=_disassemble_object,
        **kwargs,
    )


_ASCENDC_FAULT_TOKENS: tuple[tuple[str, str], ...] = (
    ("cce_forbidden_tokens", "CCE_FORBIDDEN_TOKENS"), ("max_overflow_bytes", "UB_OVERFLOW"),
    ("missing_cube_engine_primitives", "MISSING_CUBE_ENGINE"), ("missing_hardware_dma_staging", "MISSING_HARDWARE_DMA"),
    ("degenerated_single_lane_simt", "DEGENERATED_SINGLE_LANE_SIMT"), ("redline", "CUSTOM_KERNEL_REDLINE"),
)
_FORBIDDEN_NPU_HOST_CALLS = frozenset({"matmul", "bmm", "einsum", "rms_norm", "softmax", "silu", "layer_norm", "scaled_dot_product_attention"})
_CTYPES_LIB_RECEIVERS = frozenset({"lib", "_lib", "so_lib", "dll", "cdll", "handle"})
_POLLUTED_FLOAT_KEYS = frozenset({"eps", "epsilon", "scale", "softmax_scale", "sm_scale", "alpha", "beta", "dropout_p", "inv_scale", "temperature"})
_TRACKA_CORE_PRIMS = frozenset({"SimdVF", "SimtVF", "gemm", "gemm_v0", "ascend_mmad", "ascend_gemm", "ascend_vec"})
_TRACKB_MONOTONICITY_BACKENDS = frozenset({"ascendc_trackb", "hybrid_trackb", "cce_trackb", "ascendc"})
_TRACKB_CORE_TOKENS = ("simd_inst::", "asc_mmad", "ascend_gemm_l1", "__simd_vf__", "__simt_vf__")
_SIMDVF_LOCAL_ARRAY_RE = re.compile(r"\b(?:float|half|bfloat16_t|u?int(?:8|16|32|64)_t|bool)\s+[A-Za-z_]\w*\s*\[[^\]]+\]")
_ALIGNED_COPY_ARITIES = {
    "asc_copy_gm2ub_align": ((3, 8, 10), "invalid_asc_copy_align_arity: expected 3/8/10 args for gm2ub_align"),
    "asc_copy_ub2gm_align": ((3, 7, 8), "invalid_asc_copy_align_arity: expected 3/7/8 args for ub2gm_align"),
}
_GM_PTR_DECL_RE = re.compile(r"\b__gm__\s+(?:const\s+|volatile\s+)*[A-Za-z_]\w*(?:\s*\*+\s*(?:__restrict__\s+|restrict\s+|const\s+|volatile\s+)*|\s+)([A-Za-z_]\w*)")
_GM_ALIAS_RE = re.compile(r"\b__gm__\s+(?:const\s+|volatile\s+)*[A-Za-z_]\w*\s*\*+\s*(?:__restrict__\s+|restrict\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+);")
_AUTO_ALIAS_RE = re.compile(r"\bauto\s*\*?\s*([A-Za-z_]\w*)\s*=\s*([^;]+);")
_CPP_DTYPE_BYTES: dict[str, int] = {
    "uint8_t": 1, "int8_t": 1, "char": 1, "int8": 1, "uint8": 1, "fp8": 1, "fp8_t": 1,
    "half": 2, "__bf16": 2, "bfloat16_t": 2, "bfloat16": 2, "float16_t": 2, "uint16_t": 2, "int16_t": 2,
    "float": 4, "int": 4, "int32_t": 4, "uint32_t": 4, "double": 8, "int64_t": 8, "uint64_t": 8,
}
_CPP_CAST_RE = re.compile(r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|unsigned(?:\s+(?:int|long|short))?|int|long|short|float|double|half|__bf16|bfloat16_t|bfloat16)\s*\)")
_LAUNCH_BLOCK_DIM_RE = re.compile(r"<<<\s*([^,<>]+?)\s*(?:,|>>>)")
_UB_STRIDED_ACCESS_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\+\s*(?:\(\s*[A-Za-z_]\w*\s*\)\s*)?[A-Za-z_]\w*\s*\*\s*([A-Za-z_]\w*|\d+)\b")
_TRANSFER_ROUTES = ("gm_to_l1", "gm_to_ub", "l1_to_l0a", "l1_to_l0b", "l0c_to_ub", "l0c_to_l1", "l0c_to_gm", "ub_to_gm", "ub_to_l1")
_TRANSFER_CALL_SPECS = {
    "gm_to_l1": ("asc_copy_gm2l1", 4, 5, "elem"), "gm_to_ub": ("asc_copy_gm2ub", 2, 3, "bytes"),
    "l1_to_l0a": ("asc_copy_l12l0a", 4, 5, "elem"), "l1_to_l0b": ("asc_copy_l12l0b", 4, 5, "elem"),
    "l0c_to_ub": ("asc_copy_l0c2ub", 3, 2, "float"), "l0c_to_l1": ("asc_copy_l0c2l1", 3, 2, "float"),
    "l0c_to_gm": ("asc_copy_l0c2gm", 3, 2, "float"), "ub_to_gm": ("asc_copy_ub2gm", 2, 3, "bytes"),
}


def _collect_npu_reachable_nodes(module_source: str, stmts: Sequence[ast.stmt]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for st in stmts:
        if isinstance(st, ast.If):
            ct = ast.get_source_segment(module_source, st.test) or ""
            cm = ct.replace(" ", "")
            non_npu = ("device.type" in ct and ("!=" in ct or '=="cpu"' in cm or "=='cpu'" in cm)) or ("is_npu" in ct and ("not " in ct or "==False" in cm or "isFalse" in cm))
            pos_npu = ("device.type" in ct and ('=="npu"' in cm or "=='npu'" in cm)) or ("is_npu" in ct and not non_npu)
            out.append(st.test)
            out.extend(_collect_npu_reachable_nodes(module_source, st.orelse) if non_npu else (_collect_npu_reachable_nodes(module_source, st.body) if pos_npu else [*_collect_npu_reachable_nodes(module_source, st.body), *_collect_npu_reachable_nodes(module_source, st.orelse)]))
        else:
            out.append(st)
    return out


def _detect_module_entrypoint(tree: ast.Module) -> str:
    return next((n.value.id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) for t in n.targets if isinstance(t, ast.Name) and t.id == "kernel"), "")


def _find_polluted_cache_key_reason(expr: ast.expr, local_assigns: Mapping[str, list[ast.expr]], seen: frozenset[str] = frozenset()) -> str | None:
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            return f"float_literal({node.value!r})"
        if isinstance(node, ast.Name):
            if node.id.lower() in _POLLUTED_FLOAT_KEYS:
                return f"float_param({node.id})"
            if node.id in local_assigns and node.id not in seen:
                for rhs in local_assigns[node.id]:
                    if (res := _find_polluted_cache_key_reason(rhs, local_assigns, seen | {node.id})) is not None:
                        return res
        elif isinstance(node, ast.Attribute) and node.attr.lower() in _POLLUTED_FLOAT_KEYS:
            return f"float_attr({node.attr})"
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) and node.slice.value.lower() in _POLLUTED_FLOAT_KEYS:
            return f"float_key({node.slice.value!r})"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("float", "id", "addressof"):
                return {"float": "float_cast(float(...))", "id": "object_identity(id(...))", "addressof": "pointer_address(addressof(...))"}[node.func.id]
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("data_ptr", "storage_offset"):
                    return f"pointer_attr(.{node.func.attr}())"
                if node.func.attr == "addressof":
                    return "pointer_address(ctypes.addressof(...))"
    return None


def check_repair_monotonicity(parent_source: str, repaired_source: str, *, backend_kind: str = "tilelang_dsl") -> list[str]:
    """Reject repairs that regress vector/cube primitives or drop T.Pipelined stages."""
    if backend_kind in _TRACKB_MONOTONICITY_BACKENDS:
        if any(t in strip_cpp_comments_and_strings(parent_source) for t in _TRACKB_CORE_TOKENS) and not any(t in strip_cpp_comments_and_strings(repaired_source) for t in _TRACKB_CORE_TOKENS):
            return ["monotonicity_regression_trackb_scalar_fallback: repair dropped all Track-B vector/cube primitives"]
        return []
    try:
        ptree = ast.parse(parent_source)
    except SyntaxError:
        return []
    try:
        rtree = ast.parse(repaired_source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]
    has_core = lambda t: any(isinstance(n, ast.Attribute) and ((isinstance(n.value, ast.Name) and n.value.id == "T" and n.attr in _TRACKA_CORE_PRIMS) or (isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id == "T" and n.value.attr == "simd")) for n in ast.walk(t))
    def _stages(t: ast.AST) -> list[int | None]:
        out: list[int | None] = []
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and _is_t_dsl_call(n, "Pipelined"):
                se = next((kw.value for kw in n.keywords if kw.arg == "num_stages"), n.args[2] if len(n.args) >= 3 else None)
                out.append(0 if se is None else (int(se.value) if isinstance(se, ast.Constant) and isinstance(se.value, int) and not isinstance(se.value, bool) else None))
        return out
    errors = ["monotonicity_regression_scalar_fallback: repair dropped all vector/matrix core primitives (T.SimdVF/T.SimtVF/T.gemm/T.simd.*)"] if (has_core(ptree) and not has_core(rtree)) else []
    if any(s is not None and s >= 1 for s in _stages(ptree)):
        rs = _stages(rtree)
        if not rs:
            errors.append("monotonicity_regression_pipeline_dropped: repair removed T.Pipelined")
        elif all(s == 0 for s in rs):
            errors.append("monotonicity_regression_pipeline_disabled: repair set T.Pipelined num_stages=0")
    return errors


def check_npu_entrypoint_purity(module_source: str, entrypoint: str, *, track: str | None = None, require_kernel_cache: bool = False) -> list[str]:
    """Reject detached decoy @T.prim_func scaffolds and NPU-side PyTorch/golden fallbacks."""
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entrypoint not in funcs:
        return []
    errors: list[str] = []
    backend, _ = infer_module_runtime_backend(module_source, entrypoint)
    r_src, visited, cache_load, cache_store, queue = [], set(), False, False, [entrypoint]
    while queue:
        if (fn := queue.pop(0)) in visited or fn not in funcs:
            continue
        visited.add(fn)
        r_src.append(ast.get_source_segment(module_source, fnode := funcs[fn]) or "")
        for node in _collect_npu_reachable_nodes(module_source, fnode.body):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "_KERNEL_CACHE" and isinstance(sub.ctx, ast.Load):
                    cache_load = True
                elif isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store) and isinstance(sub.value, ast.Name) and sub.value.id == "_KERNEL_CACHE":
                    cache_store = True
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "_KERNEL_CACHE" and sub.func.attr in ("setdefault", "update"):
                        cache_store = True
                    callee = sub.func.id if isinstance(sub.func, ast.Name) else (sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
                    is_loc = isinstance(sub.func, ast.Name) and sub.func.id in funcs
                    is_ct = isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and sub.func.value.id.lower() in _CTYPES_LIB_RECEIVERS
                    if callee in _FORBIDDEN_NPU_HOST_CALLS and not is_loc and not is_ct:
                        errors.append(f"npu_host_fallback_forbidden:{fn}->{callee}")
                    if is_loc:
                        assert isinstance(sub.func, ast.Name)
                        queue.append(sub.func.id)
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in funcs:
                    queue.append(sub.id)
    for fn, fnode in funcs.items():
        if "@T.prim_func" in (ast.get_source_segment(module_source, fnode) or "") and fn not in visited:
            errors.append(f"unreachable_decoy_primfunc:{fn} is never reachable from {entrypoint}")
    r_txt = "\n".join(r_src)
    if track in ("cce_trackb", "ascendc_trackb", "hybrid_trackb"):
        if backend == "tilelang_dsl" and not any(t in r_txt for t in ("compile_ascend", "ctypes.CDLL", "_render_trackb_tu", "_get_track_b_lib", "tilelang.compile(")):
            errors.append(f"missing_trackb_dispatch:{entrypoint} exposes no reachable Track-B C++20 translation unit or compile_ascend dispatch")
    elif backend == "tilelang_dsl" and "tilelang.compile(" not in r_txt:
        errors.append(f"detached_decoy_primfunc:{entrypoint} never invokes tilelang.compile")

    has_cd = any(
        (isinstance(n, ast.Assign) and n.value is not None and any(isinstance(t, ast.Name) and t.id == "_KERNEL_CACHE" for t in n.targets))
        or (isinstance(n, ast.AnnAssign) and n.value is not None and isinstance(n.target, ast.Name) and n.target.id == "_KERNEL_CACHE")
        for n in tree.body
    )
    if require_kernel_cache or has_cd or cache_load or cache_store:
        cfns = {
            fn for fn, fnode in funcs.items()
            if fn in visited or fn in ("_get_kernel", "_get_track_b_lib")
            or any(t in (ast.get_source_segment(module_source, fnode) or "") for t in ("tilelang.compile", "compile_ascend", "ctypes.CDLL"))
        }
        for fn in sorted(cfns):
            if fn not in funcs:
                continue
            fnode = funcs[fn]
            loc_a: dict[str, list[ast.expr]] = {}
            for s in ast.walk(fnode):
                if isinstance(s, ast.Assign) and s.value is not None:
                    for t in s.targets:
                        if isinstance(t, ast.Name):
                            loc_a.setdefault(t.id, []).append(s.value)
                elif isinstance(s, ast.AnnAssign) and s.value is not None and isinstance(s.target, ast.Name):
                    loc_a.setdefault(s.target.id, []).append(s.value)
            kexprs: list[ast.expr] = []
            for s in ast.walk(fnode):
                if isinstance(s, ast.Subscript) and isinstance(s.value, ast.Name) and s.value.id == "_KERNEL_CACHE":
                    kexprs.append(s.slice)
                elif isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute) and isinstance(s.func.value, ast.Name) and s.func.value.id == "_KERNEL_CACHE" and s.func.attr in ("get", "setdefault", "pop"):
                    kexprs.extend([s.args[0]] if s.args else [kw.value for kw in s.keywords if kw.arg == "key"])
                elif isinstance(s, ast.Compare):
                    kexprs.extend([s.left for op, comp in zip(s.ops, s.comparators) if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.Name) and comp.id == "_KERNEL_CACHE"])
            for ke in kexprs:
                if (rsn := _find_polluted_cache_key_reason(ke, loc_a)) is not None:
                    msg = f"polluted_kernel_cache_key: {fn} key uses {rsn} (pass float hyperparameters as T.float32 scalar kernel params and dynamic outer axes via T.dynamic)"
                    if msg not in errors:
                        errors.append(msg)
        if require_kernel_cache:
            cnodes = [s for fn in cfns if fn in funcs for s in ast.walk(funcs[fn])]
            any_l = cache_load or any(isinstance(s, ast.Name) and s.id == "_KERNEL_CACHE" and isinstance(s.ctx, ast.Load) for s in cnodes)
            any_s = cache_store or any(
                (isinstance(s, ast.Subscript) and isinstance(s.ctx, ast.Store) and isinstance(s.value, ast.Name) and s.value.id == "_KERNEL_CACHE")
                or (isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute) and isinstance(s.func.value, ast.Name) and s.func.value.id == "_KERNEL_CACHE" and s.func.attr in ("setdefault", "update"))
                for s in cnodes
            )
            ctxt = "\n".join(ast.get_source_segment(module_source, funcs[fn]) or "" for fn in cfns if fn in funcs)
            if not has_cd:
                errors.append("missing_kernel_cache: module must define module-level _KERNEL_CACHE dict")
            elif not any_l or (any(t in ctxt for t in ("tilelang.compile", "compile_ascend", "ctypes.CDLL")) and not any_s):
                errors.append("unpopulated_kernel_cache: reachable compilation path must look up and populate _KERNEL_CACHE")
    return errors


def _extract_balanced_paren_arg_list(text: str, open_idx: int) -> str | None:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] not in "([":
        return None
    depth = 0
    for idx in range(open_idx, len(text)):
        depth += (text[idx] in "([") - (text[idx] in ")]")
        if depth == 0:
            return text[open_idx + 1 : idx]
    return None


def _split_cpp_args(text: str, *, include_angle: bool = True) -> list[str]:
    parts, depth, start = [], 0, 0
    opens, closes = ("(<[", ")>]") if include_angle else ("([", ")]")
    for idx, ch in enumerate(text):
        if ch in opens:
            depth += 1
        elif ch in closes and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _extract_call_args_at(masked_src: str, open_idx: int) -> list[str]:
    return _split_cpp_args(inner) if (inner := _extract_balanced_paren_arg_list(masked_src, open_idx)) is not None else []


def _collect_global_kernel_names(masked_source: str) -> set[str]:
    globals_set: set[str] = set()
    def _scan_decl_name(fragment: str, macro: bool = False) -> str:
        depth = 0
        for idx, ch in enumerate(fragment):
            if ch == "(":
                if depth == 0:
                    pat = r"([A-Za-z_]\w*(?:\s*##\s*[A-Za-z_]\w*)*)\s*$" if macro else r"([A-Za-z_]\w*)\s*$"
                    if (m := re.search(pat, fragment[:idx].rstrip())) and m.group(1).strip() not in ("__mix__", "__launch_bounds__", "decltype"):
                        return m.group(1).strip()
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
        return ""
    for m in re.finditer(r"\b__global__\b", masked_source):
        tail = masked_source[m.end() : m.end() + 512]
        end = min((p for sep in ("{", ";") if (p := tail.find(sep)) >= 0), default=len(tail))
        if nm := _scan_decl_name(tail[:end]):
            globals_set.add(nm)
    macro_specs, non_def = [], []
    def_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s+(.*)$")
    for ln in re.sub(r"\\\r?\n", " ", masked_source).splitlines():
        if (md := def_re.match(ln)) is None:
            non_def.append(ln)
        elif gm := re.search(r"\b__global__\b", md.group(3)):
            if (ne := _scan_decl_name(md.group(3)[gm.end() :], macro=True)) and (ps := tuple(p.strip() for p in md.group(2).split(",") if p.strip())):
                macro_specs.append((md.group(1), ps, ne))
    if macro_specs:
        nsrc = "\n".join(non_def)
        for mname, params, nexpr in macro_specs:
            etoks = re.split(r"\s*##\s*", nexpr)
            for im in re.finditer(rf"\b{re.escape(mname)}\s*\(", nsrc):
                if (inn := _extract_balanced_paren_arg_list(nsrc, im.end() - 1)) and len(args := _split_cpp_args(inn, include_angle=False)) >= len(params):
                    pmap = dict(zip(params, args))
                    if re.fullmatch(r"[A-Za-z_]\w*", exp := "".join(pmap.get(t, t) for t in etoks)):
                        globals_set.add(exp)
    return globals_set


def _extract_base_ptr_name(expr: str) -> str | None:
    text = expr.strip()
    while text:
        if text.startswith("("):
            close, d = -1, 0
            for idx, ch in enumerate(text):
                d += (ch == "(") - (ch == ")")
                if d == 0:
                    close = idx
                    break
            if close > 0:
                inn, rest = text[1:close].strip(), text[close + 1 :].strip()
                if not rest:
                    text = inn
                    continue
                if "*" in inn or "__gm__" in inn or re.fullmatch(r"(?:const\s+|volatile\s+)*(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*", inn):
                    text = rest
                    continue
        if cm := re.match(r"^(?:static|reinterpret|const)_cast\s*<[^>]+>\s*\(", text):
            if (ic := _extract_balanced_paren_arg_list(text, cm.end() - 1)) is not None:
                after = text[cm.end() + len(ic) + 1 :].strip()
                if not after or after[0] in ("+", "-"):
                    text = ic.strip() if not after else f"{ic} {after}".strip()
                    continue
        break
    depth = 0
    for idx, ch in enumerate(text):
        depth += (ch in "([<") - (ch in ")]>" and depth > 0)
        if depth == 0 and ch in ("+", "-"):
            text = text[:idx].strip()
            break
    m = re.match(r"^([A-Za-z_]\w*)\b", text)
    return m.group(1) if m else None


def _extract_masked_functions(masked_source: str) -> list[tuple[str, str, str]]:
    funcs, depth, start, i, n = [], 0, 0, 0, len(masked_source)
    while i < n:
        ch = masked_source[i]
        if ch == ";" and depth == 0:
            start = i + 1
        elif ch == "{":
            if depth == 0:
                sh = masked_source[start:i].strip()
                cl = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", sh).rstrip()
                fn_name = ""
                if cl.endswith(")"):
                    d, oi = 0, -1
                    for k in range(len(cl) - 1, -1, -1):
                        d += (cl[k] == ")") - (cl[k] == "(")
                        if d == 0:
                            oi = k
                            break
                    if oi > 0 and (id_m := re.search(r"([A-Za-z_]\w*)\s*$", cl[:oi].rstrip())):
                        if id_m.group(1) not in ("if", "for", "while", "switch", "decltype", "__launch_bounds__", "__mix__"):
                            fn_name = id_m.group(1)
                bd, j = 1, i + 1
                while j < n and bd > 0:
                    bd += (masked_source[j] == "{") - (masked_source[j] == "}")
                    j += 1
                if fn_name:
                    funcs.append((fn_name, sh, masked_source[i + 1 : max(i + 1, j - 1)]))
                start = i = j
                continue
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                start = i + 1
        i += 1
    return funcs


def _analyze_function_gm_stores(sig_head: str, body: str) -> tuple[dict[str, set[str]], set[str], set[str]]:
    roots_map: dict[str, set[str]] = {p: {p} for p in _GM_PTR_DECL_RE.findall(f"{sig_head}\n{body}")}
    for _ in range(3):
        changed = False
        for are in (_GM_ALIAS_RE, _AUTO_ALIAS_RE):
            for m in are.finditer(body):
                if (base := _extract_base_ptr_name(m.group(2))) and base in roots_map:
                    ex = roots_map.setdefault(m.group(1), {m.group(1)})
                    before = len(ex)
                    ex.update(roots_map[base]); ex.add(m.group(1))
                    changed = changed or (len(ex) != before)
        if not changed:
            break
    mte3 = {r for m in re.finditer(r"\basc_copy_ub2gm\w*\s*\(", body) if (args := _extract_call_args_at(body, m.end() - 1)) and (base := _extract_base_ptr_name(args[0])) in roots_map for r in roots_map[base]}
    scalar = {
        r for c, roots in roots_map.items()
        if (esc := re.escape(c)) and (re.search(rf"\b{esc}\s*(?:\[[^\]]+\]\s*)+=(?!=)", body) or re.search(rf"\*\s*\(\s*{esc}\b[^)]*\)\s*=(?!=)", body) or re.search(rf"(?<![A-Za-z0-9_])\*\s*{esc}\s*=(?!=)", body))
        for r in roots
    }
    return roots_map, mte3, scalar


def _has_mixed_scalar_and_mte3_gm_store(masked_source: str) -> bool:
    pfuncs = _extract_masked_functions(masked_source)
    if not pfuncs:
        return (not re.search(r"\bdcci\b", masked_source)) and bool((r := _analyze_function_gm_stores("", masked_source))[1] & r[2])
    by_n = {n: (s, b) for n, s, b in pfuncs}
    def _pnames(sig: str) -> list[str]:
        cl = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", sig.strip()).rstrip()
        d, oi = 0, -1
        for k in range(len(cl) - 1, -1, -1):
            d += (cl[k] == ")") - (cl[k] == "(")
            if d == 0:
                oi = k
                break
        return [m.group(1) if (m := re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?$", rp.split("=", 1)[0].strip())) else "" for rp in (_split_cpp_args(cl[oi + 1 : -1], include_angle=False) if oi > 0 else ())]
    pnames = {n: _pnames(s) for n, s, _ in pfuncs}
    analysis = {n: _analyze_function_gm_stores(s, b) for n, s, b in pfuncs}
    direct_dcci = {n for n, _, b in pfuncs if re.search(r"\bdcci\b", b)}
    callees = {n: {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(", b) if m.group(1) in by_n and m.group(1) != n} for n, _, b in pfuncs}
    for _ in range(len(pfuncs)):
        changed = False
        for name, _, body in pfuncs:
            aroot, mte3, scal = analysis[name]
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(", body):
                if (ce := m.group(1)) not in by_n or ce == name or ce in direct_dcci or (inn := _extract_balanced_paren_arg_list(body, m.end() - 1)) is None:
                    continue
                _, cm3, csc = analysis[ce]
                for idx, aexpr in enumerate(_split_cpp_args(inn, include_angle=False)):
                    if idx < len(pnames[ce]) and (pn := pnames[ce][idx]) and (ba := _extract_base_ptr_name(aexpr)) in aroot:
                        assert ba is not None
                        if pn in cm3 and not (aroot[ba] <= mte3): mte3.update(aroot[ba]); changed = True
                        if pn in csc and not (aroot[ba] <= scal): scal.update(aroot[ba]); changed = True
        if not changed:
            break
    for name, _, _ in pfuncs:
        vis, st, hdcci = set(), [name], False
        while st:
            if (cur := st.pop()) in vis: continue
            vis.add(cur)
            if cur in direct_dcci: hdcci = True; break
            st.extend(callees.get(cur, ()))
        if not hdcci and (analysis[name][1] & analysis[name][2]):
            return True
    return False


def _check_custom_kernel_redlines(dsl_source: str) -> list[str]:
    errors, low = [], dsl_source.lower()
    if any(t in low for t in ("l1->gm", "gm->l0a", "gm->l0b")):
        errors.append("forbidden_dav3510_dma_path")
    if ("int4b_t" in low or "cube_int4" in low) and "dist_unpack4" not in low and "aiv_unpack" not in low:
        errors.append("forbidden_dav3510_cube_int4_without_aiv_unpack")
    masked = strip_cpp_comments_and_strings(dsl_source)
    for m in re.finditer(r"__simd_vf__", masked):
        if (br := masked.find("{", m.end())) >= 0 and masked.find(";", m.end(), br) < 0:
            d = 0
            for i in range(br, len(masked)):
                d += (masked[i] == "{") - (masked[i] == "}")
                if d == 0:
                    if _SIMDVF_LOCAL_ARRAY_RE.search(masked[br + 1 : i]):
                        errors.append("simdvf_nonzero_stack_array: local array inside __simd_vf__ violates vf stack size (0); allocate in __ubuf__ workspace instead")
                    break
            if any(e.startswith("simdvf_nonzero_stack_array") for e in errors):
                break
    for fn, (valids, msg) in _ALIGNED_COPY_ARITIES.items():
        if any((inn := _extract_balanced_paren_arg_list(masked, m.end() - 1)) is not None and (args := _split_cpp_args(inn, include_angle=False)) and len(args) not in valids for m in re.finditer(rf"\b{re.escape(fn)}\s*\(", masked)):
            errors.append(msg)
    gnames = _collect_global_kernel_names(masked)
    if any(m.group(1) not in gnames for m in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}]*?>)?\s*<<<", masked)):
        errors.append("non_global_chevron_launch: <<<...>>> is only valid on __global__ kernels; use asc_vf_call<fn>(...) inside __aicore__ kernels")
    if _has_mixed_scalar_and_mte3_gm_store(masked):
        errors.append("mixed_scalar_and_mte3_gm_store: mixing un-flushed PIPE_S scalar stores (64B DCache) with PIPE_MTE3 asc_copy_ub2gm* on the same __gm__ output corrupts unaligned cachelines with stale NaN bits")
    return errors


def _rewrite_cpp_ternary(expr: str) -> str:
    d, q_i = 0, -1
    for i, ch in enumerate(expr):
        d += (ch == "(") - (ch == ")")
        if ch == "?" and d == 0: q_i = i; break
    if q_i < 0:
        return expr
    d, c_i, nest = 0, -1, 0
    for i in range(q_i + 1, len(expr)):
        d += (expr[i] == "(") - (expr[i] == ")")
        if expr[i] == "?" and d == 0: nest += 1
        elif expr[i] == ":" and d == 0:
            if nest == 0: c_i = i; break
            nest -= 1
    return expr if c_i < 0 else f"(({_rewrite_cpp_ternary(expr[q_i + 1 : c_i].strip())}) if ({_rewrite_cpp_ternary(expr[:q_i].strip())}) else ({_rewrite_cpp_ternary(expr[c_i + 1 :].strip())}))"


_CPP_BIN_OPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a // b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b,
    ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b,
}


def _eval_cpp_ast_node(node: ast.AST, env: Mapping[str, int]) -> int:
    if isinstance(node, ast.Expression): return _eval_cpp_ast_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)): return int(node.value)
    if isinstance(node, ast.Name): return int(env[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        v = _eval_cpp_ast_node(node.operand, env)
        return +v if isinstance(node.op, ast.UAdd) else (-v if isinstance(node.op, ast.USub) else ~v)
    if isinstance(node, ast.BinOp) and (fn := _CPP_BIN_OPS.get(type(node.op))) is not None:
        l, r = _eval_cpp_ast_node(node.left, env), _eval_cpp_ast_node(node.right, env)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and r == 0: raise ValueError("div0")
        return int(fn(l, r))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        l, r = _eval_cpp_ast_node(node.left, env), _eval_cpp_ast_node(node.comparators[0], env)
        cmps = {ast.Lt: l < r, ast.LtE: l <= r, ast.Gt: l > r, ast.GtE: l >= r, ast.Eq: l == r, ast.NotEq: l != r}
        if type(node.ops[0]) in cmps: return int(cmps[type(node.ops[0])])
    if isinstance(node, ast.IfExp):
        return _eval_cpp_ast_node(node.body if _eval_cpp_ast_node(node.test, env) else node.orelse, env)
    raise ValueError("unsupported")


def _safe_eval_cpp_int(expr: str, env: Mapping[str, int]) -> int | None:
    cleaned = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", str(expr or "").strip())
    cleaned = re.sub(r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", lambda m: str(_CPP_DTYPE_BYTES.get(m.group(1), 2)), cleaned)
    for _ in range(4):
        upd = _CPP_CAST_RE.sub("", re.sub(r"\bstatic_cast\s*<[^<>]+>\s*", "", cleaned))
        if upd == cleaned: break
        cleaned = upd
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    while cleaned.startswith("(") and cleaned.endswith(")"):
        d, enc = 0, True
        for i, ch in enumerate(cleaned):
            d += (ch == "(") - (ch == ")")
            if d == 0 and i < len(cleaned) - 1: enc = False; break
        if not enc: break
        cleaned = cleaned[1:-1].strip()
    try:
        return _eval_cpp_ast_node(ast.parse(_rewrite_cpp_ternary(cleaned), mode="eval"), env)
    except Exception:
        return None


def _parse_cce_launch_block_dim(cpp_source: str, consts: Mapping[str, int]) -> int | None:
    stripped = strip_cpp_comments_and_strings(cpp_source)
    for m in _LAUNCH_BLOCK_DIM_RE.finditer(stripped):
        raw = m.group(1).strip()
        if (bd := _safe_eval_cpp_int(raw, consts)) is not None and bd > 0:
            return bd
        if re.fullmatch(r"[A-Za-z_]\w*", raw):
            for vm in re.finditer(rf"\b(?:(?:const|constexpr)\s+)?(?:int|int32_t|int64_t|uint32_t|uint64_t|size_t|auto)(?:\s+const)?\s+{re.escape(raw)}\s*=\s*([^;]+);", stripped):
                ex = vm.group(1).strip()
                if (v := _safe_eval_cpp_int(ex, consts)) and v > 0: return v
                if (mm := re.search(r"std::min\s*(?:<[^>]+>)?\s*\(", ex)) and (inn := _extract_balanced_paren_arg_list(ex, mm.end() - 1)):
                    if vs := [c for a in _split_cpp_args(inn, include_angle=False) if (c := _safe_eval_cpp_int(a, consts)) and c > 0]: return min(vs)
                if " else " in (tt := _rewrite_cpp_ternary(ex)) and (vt := _safe_eval_cpp_int(tt.rsplit(" else ", 1)[-1].strip("() "), consts)) and vt > 0: return vt
    return None


def _resolve_cpp_consts_and_types(cpp_source: str) -> tuple[dict[str, int], dict[str, int]]:
    stripped = strip_cpp_comments_and_strings(cpp_source)
    decls = [(m.group(1), m.group(2)) for m in re.finditer(r"\bconstexpr\s+(?:int|int32_t|int64_t|uint32_t|uint64_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", stripped)]
    decls.extend((m.group(1), m.group(2).strip()) for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*)(?!\()\s+([^\r\n]+)", re.sub(r"\\\r?\n", " ", stripped), re.MULTILINE))
    consts, type_bytes = {}, {}
    tblocks = list(re.finditer(r"template\s*<([^<>]*)>\s*(?:__\w+\s*(?:\([^()]*\))?\s*)*(?:(?:inline|static|void)\s+)*([A-Za-z_]\w*)\s*\(", cpp_source))
    for _ in range(4):
        prev = dict(consts)
        for nm, ex in decls:
            rw = re.sub(r"\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)", lambda m: str(type_bytes.get(m.group(1), _CPP_DTYPE_BYTES.get(m.group(1), 2))), ex)
            if (v := _safe_eval_cpp_int(rw, consts)) is not None:
                consts[nm] = v
        for blk in tblocks:
            ps = _split_cpp_args(blk.group(1))
            if (call := re.search(rf"\b{re.escape(blk.group(2))}\s*<\s*([^<>]*)\s*>\s*(?:<<<|\()", cpp_source)) and len(args := _split_cpp_args(call.group(1))) == len(ps):
                for decl, arg in zip(ps, args):
                    if names := re.findall(r"[A-Za-z_]\w*", decl.split("=")[0]):
                        if re.match(r"(?:typename|class)\b", decl):
                            type_bytes[names[-1]] = _CPP_DTYPE_BYTES.get(arg.split("::")[-1].strip(), 2)
                        elif (tv := _safe_eval_cpp_int(arg, consts)) is not None:
                            consts[names[-1]] = max(consts.get(names[-1], tv), tv)
        if consts == prev:
            break
    return consts, type_bytes


def _resolve_cpp_consts(cpp_source: str) -> dict[str, int]:
    return _resolve_cpp_consts_and_types(cpp_source)[0]


def _stage_multiplicity(cpp_source: str, consts: Mapping[str, int], *, scope: str, names: Sequence[str] = (), strides: Mapping[str, int] | None = None) -> int:
    if (st := max((int(consts[k]) for k in (f"{scope}_STAGES", "STAGES", "NUM_STAGES", "N_STAGES", "NSTAGES") if k in consts), default=1)) > 1:
        return min(st, 8)
    if strides and any(n in strides for n in names):
        return 2
    stems: dict[str, set[str]] = {}
    for n in names:
        if re.search(r"\d+$", n): stems.setdefault(re.sub(r"\d+$", "", n), set()).add(n)
    if any(len(s) > 1 for s in stems.values()):
        return 2
    pairs = {frozenset(("PIPE_MTE3", "PIPE_MTE2")), frozenset(("PIPE_V", "PIPE_MTE2"))} if scope == "UB" else {frozenset(("PIPE_MTE1", "PIPE_MTE2")), frozenset(("PIPE_M", "PIPE_MTE2"))}
    evs = {int(m.group(3)) for m in re.finditer(r"asc_sync_(?:notify|wait)\s*\(\s*(PIPE_[A-Z0-9_]+)\s*,\s*(PIPE_[A-Z0-9_]+)\s*,\s*(?:\(event_t\))?\s*(\d+)", cpp_source) if frozenset((m.group(1), m.group(2))) in pairs}
    return 2 if {0, 1} <= evs else 1


def _resolve_l0c2gm_eb(fn: str, args: Sequence[str], masked: str, pos: int, consts: Mapping[str, int], tb: Mapping[str, int]) -> int:
    lf = fn.lower()
    for suf, w in ((("_s8", "_u8", "_i8", "_fp8"), 1), (("_f16", "_bf16"), 2), (("_f32", "_s32", "_i32"), 4)):
        if lf.endswith(suf): return w
    def _w(tok: str) -> int | None:
        if tok in tb: return int(tb[tok])
        if tok in _CPP_DTYPE_BYTES: return int(_CPP_DTYPE_BYTES[tok])
        lt = tok.lower()
        return 1 if any(x in lt for x in ("int8", "uint8", "fp8")) else (2 if any(x in lt for x in ("half", "bf16", "bfloat16", "fp16")) else (4 if any(x in lt for x in ("float", "int32", "uint32")) else None))
    da = args[0] if args else ""
    if (cm := re.search(r"__gm__\s+(?:const\s+)?([A-Za-z_]\w*)\s*\*", da)) and (w := _w(cm.group(1))) is not None: return w
    if im := re.match(r"^[&(]*([A-Za-z_]\w*)\b", _CPP_CAST_RE.sub("", da).strip()):
        pat = rf"__gm__\s+(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*{re.escape(im.group(1))}\b"
        pref = masked[:pos]
        starts = [fm.start() for fm in re.finditer(r"\b(?:template\s*<|__global__|__aicore__)\b", pref)]
        if ms := (list(re.finditer(pat, pref[starts[-1] :] if starts else pref)) or list(re.finditer(pat, pref))):
            if (w := _w(ms[-1].group(1))) is not None: return w
    if len(args) >= 9:
        qa = args[8].lower()
        if any(t in qa for t in ("int8", "s8", "fp8", "e4m3", "e5m2")): return 1
        if any(t in qa for t in ("f16", "fp16", "bf16", "bfloat16", "half")): return 2
        if (qv := _safe_eval_cpp_int(args[8], consts)) is not None: return 4 if qv == 0 else 2
    return 4


def _build_ascendc_static_ledger(
    cpp_source: str, *, target: AscendTarget | None = None, compiler_summary: BishengResourceSummary | None = None,
    objdump_text: str = "", microarch_profile: MicroarchProfile | Mapping[str, object] | None = None,
    case_ratios: CasePhysicsRatios | None = None,
) -> ObjectivePhysicalLedger:
    """Compute SRAM waterlines and pipe cycles from C++20 constexprs, calls, and objdump."""
    from tilefoundry_ascend.ascend.target.facts import ascend_space_facts  # noqa: PLC0415

    facts = ascend_space_facts(target or AscendTarget("ascend.950pr"))
    lower = cpp_source.lower()
    consts, type_bytes = _resolve_cpp_consts_and_types(cpp_source)
    ub_off, l1_off, ptr_env, cpb = [], [], {}, {}
    cast_re = re.compile(r"__(ubuf|cbuf)__\s+([A-Za-z0-9_]+)\s*\*\s*(\w+)\s*=\s*\(\s*__\1__\s+[A-Za-z0-9_]+\s*\*\s*\)\s*([^;]+);")
    plus_re = re.compile(r"__(ubuf|cbuf)__\s+([A-Za-z0-9_]+)\s*\*\s*(\w+)\s*=\s*([A-Za-z_]\w*)\s*\+\s*([^;]+);")
    for ln in cpp_source.splitlines():
        if cm := cast_re.search(ln):
            sp, ct, nm, ex = cm.groups()
            if ct in _CPP_DTYPE_BYTES: cpb[nm] = _CPP_DTYPE_BYTES[ct]
            if (off := _safe_eval_cpp_int(ex, consts)) is not None and off >= 0:
                ptr_env[nm] = (sp, off); (ub_off if sp == "ubuf" else l1_off).append((nm, off, _CPP_DTYPE_BYTES.get(ct, 2)))
        elif pm := plus_re.search(ln):
            sp, ct, nm, base, ex = pm.groups()
            if ct in _CPP_DTYPE_BYTES: cpb[nm] = _CPP_DTYPE_BYTES[ct]
            if (delta := _safe_eval_cpp_int(ex, consts)) is not None:
                eb = _CPP_DTYPE_BYTES.get(ct, 2)
                bo = ptr_env[base][1] if (base in ptr_env and ptr_env[base][0] == sp) else 0
                off = max(0, bo + delta * eb)
                ptr_env[nm] = (sp, off); (ub_off if sp == "ubuf" else l1_off).append((nm, off, eb))

    tm = max(1, consts.get("CUBE_TILE_M", consts.get("BM", consts.get("TILE_M", 64))))
    tn = max(1, consts.get("CUBE_TILE_N", consts.get("BN", consts.get("TILE_N", 64))))
    tk = max(1, consts.get("CUBE_TILE_K", consts.get("BK", consts.get("TILE_K", 64))))
    uses_cube = any(t in lower for t in ("asc_mmad", "ascend_gemm_l1", "mmad", "l0c", "cube_tile", "__cbuf__"))
    uses_simt = "__simt_vf__" in cpp_source or "Simt::" in cpp_source
    uses_256 = ("ub_256kb_unlock" in lower or "__simd_vf__" in cpp_source or "262144" in cpp_source) and not uses_simt
    bd = _parse_cce_launch_block_dim(cpp_source, consts)
    pc = int(facts.num_aic_cores) if uses_cube else int(facts.num_aiv_cores)
    grid_b, wave_occ = (bd, bd / max(1, pc)) if bd is not None else ((max(1, int(round(case_ratios.wave_occupancy_ratio * pc))), float(case_ratios.wave_occupancy_ratio)) if case_ratios else (1, 0.0))

    strides_m = {}
    for sm in re.finditer(r"\b([A-Za-z_]\w*)\s*\+\s*(?:\(\s*)?(?:stg_id|stage_id|stg|stage|slot|buf)\s*\*\s*([A-Za-z_]\w*|\d+)\s*\)?", cpp_source):
        if (sv := _safe_eval_cpp_int(sm.group(2), consts)) and sv > 0:
            strides_m[sm.group(1)] = max(strides_m.get(sm.group(1), sv), sv)
    ub_stg = _stage_multiplicity(cpp_source, consts, scope="UB", names=[n for n, _, _ in ub_off])
    l1_stg = _stage_multiplicity(cpp_source, consts, scope="L1", names=[n for n, _, _ in l1_off], strides=strides_m)

    def _spans(entries: list[tuple[str, int, int]], scope: str, stg: int) -> dict[str, int]:
        if not entries: return {}
        ord_e = sorted(entries, key=lambda it: it[1])
        names, bufs = [it[0] for it in ord_e], {}
        for i, (nm, off, eb) in enumerate(ord_e):
            sp = not any(o != nm and re.sub(r"\d+$", "", o) == re.sub(r"\d+$", "", nm) for o in names)
            if (sd := strides_m.get(nm)) is not None: sz = sd * eb * stg
            elif i + 1 < len(ord_e): sz = max(0, ord_e[i + 1][1] - off)
            elif i > 0: sz = g if (g := max(0, off - ord_e[i - 1][1])) > 0 else max(eb * 64, tn * eb)
            else:
                sb = consts.get(f"{scope}_STAGE_BYTES", consts.get("STAGE_BYTES"))
                sz = int(sb) * stg if (sp and stg > 1 and sb) else max(eb * 64, tm * tn * eb)
            if i + 1 == len(ord_e):
                up = nm.upper()
                roots = [s for b in [up, *(up[: -len(x)] for x in ("_BASE", "_BUF", "_PTR", "_0", "_1") if up.endswith(x))] for sl in (re.sub(r"\d+$", "", b),) for s in (sl, sl.replace("_L1", "").replace("_UB", ""))]
                for rt in dict.fromkeys(roots):
                    for suf in ("_STAGE_BYTES", "_BYTES", "_STAGE", "_BASE", "_BUF", "_PTR"):
                        if rt.endswith(suf): rt = rt[: -len(suf)]; break
                    for k, ib in ((f"{rt}_STAGE_BYTES", True), (f"{rt}_BYTES", True), (f"{rt}_STAGE", False)):
                        if k in consts:
                            sz = max(sz, int(consts[k]) if ib else int(consts[k]) * eb * (stg if sp and not re.search(r"\d+$", nm) else 1))
                            break
            if sz > 0: bufs[nm] = sz
        return bufs

    ex_ub, ex_l1 = _spans(ub_off, "UB", ub_stg), _spans(l1_off, "L1", l1_stg)
    if um := re.search(r"\bub_bytes\s*=\s*(\d+)", cpp_source, re.IGNORECASE): ex_ub = {"ub_declared": int(um.group(1))}
    if lm := re.search(r"\bl1_bytes\s*=\s*(\d+)", cpp_source, re.IGNORECASE): ex_l1 = {"l1_declared": int(lm.group(1))}
    if not ex_l1 and uses_cube: ex_l1["l1_AB"] = (tm * tk + tk * tn) * 2 * max(2, l1_stg)
    if not ex_ub: ex_ub["ub_work"] = consts.get("ROWS", tm) * consts.get("BLK", tn) * 2 if uses_cube else max(1024, consts.get("ROWS", tm) * consts.get("BLK", tn) * 2)
    imp_ub = {**({"register_spill": int(compiler_summary.spill_bytes)} if compiler_summary and compiler_summary.spill_bytes > 0 else {}), **({"stack_frame": int(compiler_summary.stack_bytes)} if compiler_summary and compiler_summary.stack_bytes > 0 else {})}
    ub_cap = int(facts.ub_simtvf_usable_bytes) if uses_simt else (int(facts.ub_nominal_bytes) if uses_256 else int(facts.ub_simdvf_usable_bytes))
    wls = {"UB": build_scope_waterline(scope="UB", capacity_bytes=ub_cap, explicit_buffers=ex_ub, implicit_buffers=imp_ub)}
    if ex_l1 or uses_cube:
        wls["L1"] = build_scope_waterline(scope="L1", capacity_bytes=int(facts.l1_bytes), explicit_buffers=ex_l1, implicit_buffers={})

    dis = _parse_objdump_pipe_counts(objdump_text)
    mte2_c = max(len(re.findall(r"\basc_copy_gm2(?:ub|l1)\w*\s*\(", cpp_source)), dis["mte2"])
    mte3_c = max(len(re.findall(r"\basc_copy_ub2gm\w*\s*\(", cpp_source)), dis["mte3"])
    fix_c = max(len(re.findall(r"\basc_copy_l0c2\w*\s*\(", cpp_source)), dis["fixpipe"])
    mmad_c = max(len(re.findall(r"\b(?:asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\s*(?:<[^;{}()]*>)?\s*\(", cpp_source)), dis["cube_mmad"])
    vf_c = max(len(re.findall(r"\b(?:simd_inst|Reg)::\w+\s*(?:<[^>]*>)?\s*\(", cpp_source)), dis["vec"])

    masked = strip_cpp_comments_and_strings(cpp_source)
    lspans, n_m = [], len(masked)
    for lm in re.finditer(r"\b(?:for|while)\s*\(", masked):
        p, pd = lm.end() - 1, 0
        while p < n_m:
            pd += (masked[p] == "(") - (masked[p] == ")"); p += 1
            if pd == 0: break
        while p < n_m and masked[p].isspace(): p += 1
        if p < n_m and masked[p] == "{":
            bs, bd_l, p = p + 1, 1, p + 1
            while p < n_m and bd_l > 0:
                bd_l += (masked[p] == "{") - (masked[p] == "}"); p += 1
            lspans.append((bs, max(bs, p - 1)))
    dconsts = dict(consts)
    for bm in re.finditer(r"\bif\s*\(\s*([A-Za-z_]\w*)\s*>\s*([^;{}()]+?)\s*\)\s*\1\s*=\s*([^;]+);", masked):
        if bm.group(1) not in dconsts and (bv := _safe_eval_cpp_int(bm.group(3), dconsts)) and bv > 0: dconsts[bm.group(1)] = bv
    dma_b, cands_b = [], []
    for dm in re.finditer(r"\b(asc_copy_gm2ub_align|asc_copy_ub2gm_align|asc_copy_gm2ub|asc_copy_ub2gm|asc_copy_gm2l1(?:_nd2nz|_dn2nz)?|asc_copy_l0c2gm\w*)\s*\(", masked):
        fn, args = dm.group(1), _extract_call_args_at(masked, dm.end() - 1)
        if fn in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
            if len(args) == 3 and (sz := _safe_eval_cpp_int(args[2], dconsts)) and sz > 0: dma_b.append(sz); cands_b.append(sz)
            elif len(args) >= 4 and (rb := _safe_eval_cpp_int(args[3], dconsts)) and rb > 0: nb = _safe_eval_cpp_int(args[2], dconsts); cands_b.append(rb); dma_b.append(rb * nb if nb and nb > 0 else rb)
        elif fn in ("asc_copy_gm2ub", "asc_copy_ub2gm") and len(args) >= 4 and (bl := _safe_eval_cpp_int(args[3], dconsts)) and bl > 0:
            rb, nb = bl * 32, _safe_eval_cpp_int(args[2], dconsts); cands_b.append(rb); dma_b.append(rb * nb if nb and nb > 0 else rb)
        elif fn.startswith("asc_copy_gm2l1") and len(args) >= 6 and (cs := _safe_eval_cpp_int(args[5], dconsts)) and cs > 0:
            rb, rs = cs * 2, _safe_eval_cpp_int(args[4], dconsts); cands_b.append(rb); dma_b.append(rb * rs if rs and rs > 0 else rb)
        elif fn.startswith("asc_copy_l0c2gm") and len(args) >= 4 and (cs := _safe_eval_cpp_int(args[2], dconsts)) and cs > 0:
            rb, rs = cs * _resolve_l0c2gm_eb(fn, args, masked, dm.start(), dconsts, type_bytes), _safe_eval_cpp_int(args[3], dconsts); cands_b.append(rb); dma_b.append(rb * rs if rs and rs > 0 else rb)
    dma_eff = round(min(1.0, sum(dma_b) / max(1e-6, sum(b / max(1e-6, float(facts.dma_burst_efficiency(int(b)))) for b in dma_b))), 4) if (dma_b and hasattr(facts, "dma_burst_efficiency")) else 1.0
    dom_cb = max(cands_b) if cands_b else (int(case_ratios.inner_contiguous_bytes) if case_ratios and case_ratios.inner_contiguous_bytes > 0 else (tn * 2 if (mte2_c or mte3_c) else 0))
    cm_vals = [mv for gm in re.finditer(r"\bascend_gemm_l1\s*<([^;{}()]+)>\s*\(", masked) if (ta := _split_cpp_args(gm.group(1))) and (mv := _safe_eval_cpp_int(ta[0], consts)) and mv > 0] if (uses_cube or mmad_c) else []
    cm_vals.extend(mv for mm in re.finditer(r"\basc_mmad(?:_mx|_sparse)?\s*(?:<[^;{}()]*>)?\s*\(", masked) if len(ca := _extract_call_args_at(masked, mm.end() - 1)) >= 4 and (mv := _safe_eval_cpp_int(ca[3], consts)) and mv > 0)
    eff_cm = min(cm_vals) if cm_vals else (tm if (uses_cube or mmad_c) else 0)
    l1_tot, l1_full = sum(ex_l1.values()), max(1, int(getattr(facts, "l1_bytes", 524288)))
    l1_risk = 0.0 if (l1_tot <= 0 or "l1_half_bank_split" in lower or "half_bank" in lower) else round(min(1.0 if l1_tot > l1_full // 2 else 0.5, l1_tot / float(l1_full if l1_tot > l1_full // 2 else max(1, l1_full // 2))), 4) if (l1_tot > l1_full // 2 or len(ex_l1) >= 2) else 0.0
    ub_str: dict[str, int] = {}
    for um2 in _UB_STRIDED_ACCESS_RE.finditer(masked):
        if (eb := cpb.get(um2.group(1))) and (se := _safe_eval_cpp_int(um2.group(2), consts)) and se > 0: ub_str[um2.group(1)] = max(ub_str.get(um2.group(1), 0), se * eb)
    ub_risk = (sum(1 for sb in ub_str.values() if sb >= 512 and sb % 512 == 0) / len(ub_str)) if ub_str else 0.0
    mte2_inf = 0
    if mte2_c > 0:
        cr, mr, mevs = 0, 0, set()
        for tm2 in re.finditer(r"\b(asc_copy_gm2(?:ub|l1)\w*)\s*\(|\basc_sync_(?:notify|wait)\s*\(\s*([^)]+)\)", masked):
            if tm2.group(1): cr += 1; mr = max(mr, cr)
            else:
                sa = _split_cpp_args(tm2.group(2) or "", include_angle=False)
                if any(a.strip() == "PIPE_MTE2" for a in sa[:2]):
                    if len(sa) >= 3: mevs.add(sa[2].strip())
                    if "wait" in tm2.group(0): cr = 0
        mte2_inf = max(1, mr, len(mevs), 2 if ub_stg > 1 else 1)

    has_dual = bool(re.search(r"\b(?:get_subblockid|GetSubBlockIdx|get_subblock_idx)\s*\(", masked))
    pipes: dict[str, int] = {}
    if mte2_c > 0: pipes["AIV0.PIPE_MTE2"] = mte2_c * max(16, (tm * tn * 2) // 64)
    if mte3_c > 0: pipes["AIV0.PIPE_MTE3"] = mte3_c * max(16, (tm * tn * 2) // 64)
    if vf_c > 0:
        tv = vf_c * max(8, (tm * tn) // 64) * (2 if (compiler_summary and not compiler_summary.vloopv2_hit) else 1) + (int(compiler_summary.spill_bytes) * 2 if compiler_summary else 0)
        if has_dual: pipes["AIV0.PIPE_V"] = max(1, tv // 2); pipes["AIV1.PIPE_V"] = max(1, tv - pipes["AIV0.PIPE_V"])
        else: pipes["AIV0.PIPE_V"] = tv
    if mmad_c > 0: pipes["AIC.PIPE_M"] = mmad_c * max(16, (tm * tn * tk) // 4096)
    if fix_c > 0: pipes["AIC.PIPE_FIX"] = fix_c * max(12, (tm * tn * 2) // 256)
    if dis["mte1"] > 0: pipes["AIC.PIPE_MTE1"] = dis["mte1"] * max(8, (tm * tk) // 128)
    ma_dict = microarch_profile.to_dict() if isinstance(microarch_profile, MicroarchProfile) else (dict(microarch_profile) if isinstance(microarch_profile, Mapping) else extract_microarch_profile_from_cce(cpp_source).to_dict())

    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=all(wl.overflow_bytes == 0 for wl in wls.values()),
        steady_state_ii=max(pipes.values()) if pipes else None, pipe_cycles=pipes, waterlines=wls,
        mte2_inflight_per_aiv=mte2_inf, grid_blocks=grid_b, wave_occupancy=wave_occ, dominant_contig_bytes=dom_cb,
        effective_cube_m=eff_cm, has_indirect_gather=uses_simt,
        has_lowbit_quant=bool(re.search(r"\b(?:int4|uint4|fp4|f4e2m1|f4e1m2|int4b_t|deq_scale|quant_mode|asc_mmad_mx)\b", masked)),
        has_online_rescale_loop=any(re.search(r"\b(?:vexp|exp2|expf|exp|Exp)\s*(?:<[^>]*>)?\s*\(", masked[bs:be]) and re.search(r"\b(?:vcmax|vcadd|vcmin|vreduce|ReduceMax|ReduceSum|asc_reduce_\w+|vmul|vdiv|vfma|vmad|Mul|Div)\b|\*=|\+=", masked[bs:be]) for bs, be in lspans),
        uses_unreserved_ubuf=uses_256, min_gm_dma_bytes=min(dma_b) if dma_b else 0, dma_burst_efficiency_ratio=dma_eff,
        uses_dynamic_offset_pipelining=bool(strides_m or re.search(r"\b(?:stg|stage|stg_id|stage_id|nstg)\b[^;\n]*[?:]", masked) or ((ub_stg >= 2 or l1_stg >= 2) and re.search(r"\b\w+\s*&\s*1\b|\b\w+\s*%\s*(?:2|STAGES|NUM_STAGES)\b", masked))),
        in_loop_pipe_all_barriers=sum(len(re.findall(r"\b(?:pipe_barrier|asc_sync_pipe)\s*\(\s*PIPE_ALL\s*\)", masked[bs:be])) for bs, be in lspans),
        ub_bank_conflict_risk=ub_risk, l1_bank_conflict_risk=l1_risk,
        ub_scalar_compute_loops=count_reachable_ub_scalar_compute_loops(cpp_source), microarch_profile=ma_dict,
    )


def _merge_disassembly_microarch_profile(cpp_source: str, objdump_text: str) -> dict[str, object] | None:
    if not cpp_source.strip() and not objdump_text.strip():
        return None
    base = extract_microarch_profile_from_cce(cpp_source)
    dis = _parse_objdump_pipe_counts(objdump_text)
    if dis["total_insns"] == 0:
        return base.to_dict()
    consts, _ = _resolve_cpp_consts_and_types(cpp_source)
    gran: dict[str, int] = {}
    shape_src = re.sub(r"\bconstexpr\s+[^;]+;", "", cpp_source)
    eb_of = lambda txt: next((w for tn, w in _CPP_DTYPE_BYTES.items() if re.search(rf"\b{re.escape(tn)}\b", txt)), 2)
    for route in _TRANSFER_ROUTES:
        payloads = []
        if spec := _TRANSFER_CALL_SPECS.get(route):
            pfx, ri, ci, kind = spec
            for call in re.finditer(rf"\b{pfx}\w*\s*\(([^;]*)\)\s*;", cpp_source, re.S):
                if len(args := _split_cpp_args(call.group(1))) > max(ri, ci):
                    eb = eb_of(call.group(0))
                    r = _safe_eval_cpp_int(args[ri], {**consts, "T_ELEM_BYTES": eb})
                    c = _safe_eval_cpp_int(args[ci], {**consts, "T_ELEM_BYTES": eb})
                    if r and c and r > 0 and c > 0:
                        payloads.append(r * c * (1 if kind == "bytes" else (4 if kind == "float" else eb)))
        if payloads:
            gran[route] = sorted(payloads)[len(payloads) // 2]
        elif any(k in consts for k in ("CUBE_TILE_M", "CUBE_TILE_N", "CUBE_TILE_K", "BM", "BN", "BK", "TILE_M", "TILE_N", "TILE_K")):
            tm, tn, tk, eb = max(1, consts.get("CUBE_TILE_M", consts.get("BM", consts.get("TILE_M", 64)))), max(1, consts.get("CUBE_TILE_N", consts.get("BN", consts.get("TILE_N", 64)))), max(1, consts.get("CUBE_TILE_K", consts.get("BK", consts.get("TILE_K", 64)))), eb_of(shape_src)
            gran[route] = max(tm * tk, tk * tn) * eb if route == "gm_to_l1" else (tm * tn * eb if route in ("gm_to_ub", "ub_to_gm", "ub_to_l1") else (tm * tk * eb if route == "l1_to_l0a" else (tk * tn * eb if route == "l1_to_l0b" else tm * tn * 4)))
    upd: dict[str, object] = {"fp_alu_ops": max(base.fp_alu_ops, dis["vec"]), "cube_mmad_count": max(base.cube_mmad_count, dis["cube_mmad"]), "cv_sync_barriers": max(base.cv_sync_barriers, dis["sync"])}
    for ck, bf in (("gm_to_l1_insns", "gm_to_l1_bytes"), ("gm_to_ub_insns", "gm_to_ub_bytes"), ("l1_to_l0a_insns", "l1_to_l0a_bytes"), ("l1_to_l0b_insns", "l1_to_l0b_bytes"), ("l0c_to_ub_insns", "l0c_to_ub_bytes"), ("l0c_to_l1_insns", "l0c_to_l1_bytes"), ("l0c_to_gm_insns", "l0c_to_gm_bytes"), ("ub_to_gm_insns", "ub_to_gm_bytes"), ("ub_to_l1_insns", "ub_to_l1_bytes")):
        rt = bf.removesuffix("_bytes")
        upd[bf] = dis[ck] * gran[rt] if (dis[ck] > 0 and rt in gran) else getattr(base, bf)
    return replace(base, **upd).to_dict()


def evaluate_ascendc_candidate(
    cpp_source: str, *, freedom_mask: object | None = None, verifier: object | None = None,
    target: AscendTarget | None = None, enable_track_router: bool = False,
    case_ratios: CasePhysicsRatios | None = None, require_hardware_dma: bool = False,
    require_cube_engine: bool = False, baseline_cpp_source: str | None = None, max_flag_depth: int = 8,
    module_source: str = "", regime_tag: str = "",
) -> StageAResult:
    """Validate a Track-B AscendC C++ candidate via 7-facet checks and static ledger."""
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return _stage_a_fail(t0, "ascendc_empty_source", "ASCENDC_EMPTY_SOURCE", backend_kind="ascendc")
    from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask  # noqa: PLC0415
    from tilefoundry_ascend.ascend.sil.verifier import (  # noqa: PLC0415
        SILVerifier,
        verify_source_against_facets,
    )

    _, rcpp = extract_reachable_trackb_source(cpp_source, module_source=module_source, regime_tag=regime_tag)
    amask = freedom_mask if freedom_mask is not None else ConstraintRelaxationMask.bottom()
    aver = verifier if verifier is not None else SILVerifier()
    frep = aver.verify_source_against_facets(cpp_source, amask, max_flag_depth=max_flag_depth) if hasattr(aver, "verify_source_against_facets") else verify_source_against_facets(cpp_source, amask, max_flag_depth=max_flag_depth)
    ledger = _build_ascendc_static_ledger(rcpp, target=target, case_ratios=case_ratios)
    ftoks = CCE_CPP20_SURFACE.introduced_forbidden_tokens(baseline_cpp_source, cpp_source) if baseline_cpp_source is not None else tuple(t for t, c in CCE_CPP20_SURFACE.count_forbidden_tokens(cpp_source).items() if c > 0)
    errors = [*_check_custom_kernel_redlines(cpp_source), *frep.errors]
    msrc = strip_cpp_comments_and_strings(rcpp)
    if require_cube_engine and len(re.findall(r"\b(?:asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\s*(?:<[^;{}()]*>)?\s*\(", msrc)) == 0 and int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) == 0:
        errors.append("missing_cube_engine_primitives: matmul/attention kernel contains zero Cube MMAD instructions (scalar fallback trap)")
    if require_hardware_dma and len(re.findall(r"\basc_copy_gm2(?:ub|l1)\w*\s*\(", msrc)) == 0 and not any(k.endswith(".PIPE_MTE2") and int(v) > 0 for k, v in ledger.pipe_cycles.items()):
        errors.append("missing_hardware_dma_staging: Track-B kernel performs raw scalar __gm__ loops with zero GM->UB/L1 DMA transfers")
    if (case_ratios is None or not getattr(case_ratios, "has_empty_group", False)) and re.search(r"\b__simt_vf__\b", msrc) and (re.search(r"__launch_bounds__\s*\(\s*1\s*\)", msrc) or re.search(r"\basc_vf_call\s*<[^>]+>\s*\(\s*dim3\s*\(\s*1\s*,\s*1\s*,\s*1\s*\)", msrc)):
        errors.append("degenerated_single_lane_simt: __simt_vf__ launched with 1 thread wastes 127/128 SIMT lanes; parallelize across >=32 threads via Simt::GetThreadIdx<0>() or use __simd_vf__")
    if ftoks: errors.append(f"cce_forbidden_tokens: {', '.join(ftoks)}")
    if ledger.max_overflow_bytes > 0: errors.append(f"max_overflow_bytes={ledger.max_overflow_bytes}")

    features, route = None, None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            route_execution_track,
        )

        low = rcpp.lower()
        route = route_execution_track(None, ledger=ledger, semantic_features={"static_affine_bounds": not ("__simt_vf__" in rcpp or "Simt::" in rcpp), "cube_to_vector_ratio": 1.0 if any(t in low for t in ("mmad", "l0c", "ascend_gemm_l1")) else 0.0, "quant_granularity": "per_token_or_block" if "deq_scale" in low else "none"}, target=target, case_ratios=case_ratios)
        features = dict(route.features) if route.features is not None else None
    if not ledger.is_legal or errors:
        fc = next((c for tok, c in _ASCENDC_FAULT_TOKENS if tok in "; ".join(errors).lower()), "STAGE_A_ASCENDC_STATIC_FAILED")
        return _stage_a_fail(t0, "; ".join(dict.fromkeys(errors)) or "ascendc_static_ledger_illegal", fc, backend_kind="ascendc", ledger=ledger, features=features, route=route)
    return StageAResult(passed=True, prim_func=None, ledger=ledger, features=features, route=route, elapsed_ms=(time.monotonic() - t0) * 1000.0, backend_kind="ascendc")


def _is_tvm_ir_node(obj: object) -> bool:
    from tvm import IRModule, tirx  # noqa: PLC0415

    return isinstance(obj, (tirx.PrimFunc, IRModule))


def _resolve_prim_func_and_configs_from_dsl(
    dsl_source: str, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None, dims: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], list[str]]:
    ast.parse(dsl_source)
    sel_pc, sel_cf, eff_kw = {}, [], _merge_dims_into_kwargs(factory_kwargs, dims)
    with tempfile.TemporaryDirectory(prefix="tf_stage_a_") as tmp_dir:
        (mod_p := Path(tmp_dir) / "candidate_dsl.py").write_text(dsl_source, encoding="utf-8")
        if (spec := importlib.util.spec_from_file_location("_tf_stage_a_cand", str(mod_p))) is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for candidate DSL")
        mod = importlib.util.module_from_spec(spec)
        with ascend_compile_session("ascend"):
            spec.loader.exec_module(mod)
            if config_selector_name and callable(so := getattr(mod, config_selector_name, None)) and isinstance(sr := so(**_filter_kwargs_for_callable(so, eff_kw)), Mapping):
                ov, sel_pc, sel_cf = _split_selector_output(sr)
                eff_kw.update(ov)
            if factory_name and hasattr(mod, factory_name):
                tgt = getattr(mod, factory_name)
                return (tgt if (not callable(tgt) or (factory_name == "kernel" and "prim_func" not in dsl_source)) else tgt(**_filter_kwargs_for_callable(tgt, eff_kw))), sel_pc, sel_cf
            if (ir_v := next((v for v in vars(mod).values() if _is_tvm_ir_node(v)), None)) is not None:
                return ir_v, sel_pc, sel_cf
            if callable(ko := getattr(mod, "kernel", None)) and "prim_func" not in dsl_source:
                return ko, sel_pc, sel_cf
            for nm, v in vars(mod).items():
                if not nm.startswith("_") and callable(v) and getattr(v, "__module__", None) == mod.__name__:
                    return v(**_filter_kwargs_for_callable(v, eff_kw)), sel_pc, sel_cf
    raise ValueError("DSL source defined no callable factory or PrimFunc")


def _resolve_prim_func_from_dsl(dsl_source: str, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None):
    return _resolve_prim_func_and_configs_from_dsl(dsl_source, factory_name=factory_name, factory_kwargs=factory_kwargs)[0]


def _is_t_dsl_call(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr and isinstance(node.func.value, ast.Name) and node.func.value.id == "T"


def _parallel_forbidden_math_calls(tree: ast.AST) -> list[str]:
    found: set[str] = set()
    def _v(node: ast.AST, in_simt: bool) -> None:
        if isinstance(node, ast.For) and _is_t_dsl_call(node.iter, "Parallel") and not in_simt:
            for bn in node.body:
                for inner in ast.walk(bn):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in ("Cast", "exp", "log", "tanh", "sigmoid", "rsqrt") and _is_t_dsl_call(inner, inner.func.attr):
                        found.add(inner.func.attr)
        nested = in_simt or (isinstance(node, ast.With) and any(_is_t_dsl_call(it.context_expr, "SimtVF") for it in node.items))
        for ch in ast.iter_child_nodes(node): _v(ch, nested)
    _v(tree, False)
    return sorted(found)


def _uncovered_1c2v_subcore_mte3_writes(tree: ast.AST) -> list[int]:
    bad: list[int] = []
    bn_of = lambda n: n.id if isinstance(n, ast.Name) else (bn_of(n.value) if isinstance(n, ast.Subscript) else None)
    names_of = lambda n: {c.id for c in ast.walk(n) if isinstance(c, ast.Name)}
    sram_attrs = frozenset({"alloc_shared", "alloc_l1", "alloc_l0a", "alloc_l0b", "alloc_l0c", "alloc_ss"})
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef): continue
        pnames = {a.arg for a in fn.args.args}
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.With): continue
            sid = next(
                (it.optional_vars.elts[1].id for it in sub.items if isinstance(it.context_expr, ast.Call) and _is_t_dsl_call(it.context_expr, "MixedKernel") and any(kw.arg == "sids" and isinstance(kw.value, ast.Constant) and kw.value.value == 2 for kw in it.context_expr.keywords) and isinstance(it.optional_vars, (ast.Tuple, ast.List)) and len(it.optional_vars.elts) >= 2 and isinstance(it.optional_vars.elts[1], ast.Name) and it.optional_vars.elts[1].id != "_"),
                "",
            )
            if not sid: continue
            ub_b, sram_b, tainted = set(), set(), {sid}
            for st in ast.walk(sub):
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Call):
                    for a in sram_attrs:
                        if _is_t_dsl_call(st.value, a):
                            sram_b.add(st.targets[0].id)
                            if a == "alloc_shared": ub_b.add(st.targets[0].id)
                            break
            for _ in range(6):
                added = False
                for st in ast.walk(sub):
                    t, e = (st.targets[0], st.value) if isinstance(st, ast.Assign) and len(st.targets) == 1 else ((st.target, st.value) if isinstance(st, ast.AnnAssign) and st.value is not None else ((st.target, st.iter) if isinstance(st, ast.For) and isinstance(st.iter, ast.Call) and len(st.iter.args) >= 2 else (None, None)))
                    if isinstance(t, ast.Name) and e is not None and (names_of(e) & tainted) and t.id not in tainted:
                        tainted.add(t.id); added = True
                if not added: break
            def _scan(stmts: Sequence[ast.stmt], g: bool) -> None:
                for st in stmts:
                    if isinstance(st, ast.If):
                        ig = g or any(isinstance(s, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in s.ops) and ((isinstance(s.left, ast.Name) and s.left.id == sid) or any(isinstance(c, ast.Name) and c.id == sid for c in s.comparators)) for s in ast.walk(st.test))
                        _scan(st.body, ig); _scan(st.orelse, ig)
                    elif isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                        _scan(st.body, g)
                    elif isinstance(st, ast.Try):
                        _scan(st.body, g); hasattr(st, "orelse") and _scan(st.orelse, g)
                    elif not g:
                        for inner in ast.walk(st):
                            if isinstance(inner, ast.Call) and _is_t_dsl_call(inner, "copy") and len(inner.args) >= 2:
                                sn, da = bn_of(inner.args[0]), inner.args[1]
                                if sn in ub_b and (dn := bn_of(da)) is not None and (dn in pnames or dn not in sram_b) and (not isinstance(da, ast.Subscript) or not (names_of(da.slice) & tainted)):
                                    bad.append(inner.lineno)
            _scan(sub.body, False)
    return sorted(set(bad))


def _resolve_stage_a_var_bindings(dims: Mapping[str, object] | None, var_bindings: Mapping[str, int] | None) -> dict[str, int] | None:
    resolved = build_anchor_var_bindings(dims)
    if dims:
        bv, s1v = resolved.get("B", resolved.get("b_dim")), resolved.get("S1", resolved.get("s1_dim", resolved.get("S", resolved.get("S_q", resolved.get("s_dim")))))
        nqv, nkv = resolved.get("N_q", resolved.get("n_q")), resolved.get("N_kv", resolved.get("n_kv"))
        n1v, n2v = resolved.get("N1", nqv), resolved.get("N2", nkv)
        if bv is not None and s1v is not None and nqv is not None: resolved.setdefault("q_total_rows", bv * s1v * nqv)
        if n1v is not None and n2v is not None and n2v > 0: resolved.setdefault("g_ratio", max(1, n1v // n2v))
    for k, v in (var_bindings or {}).items():
        if isinstance(v, int) and not isinstance(v, bool): resolved[str(k)] = int(v)
    return resolved or None


def run_stage_a_static(
    dsl_source: str, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None,
    target: AscendTarget | None = None, pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None, var_bindings: Mapping[str, int] | None = None,
    enable_track_router: bool = False, case_ratios: CasePhysicsRatios | None = None,
    require_hardware_dma: bool = False, require_cube_engine: bool = False,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> StageAResult:
    """Run millisecond Stage A AST, Two-Snapshot IR Diff, and optional track routing."""
    t0 = time.monotonic()
    try:
        ptree = ast.parse(dsl_source)
        if kl := sorted({n.lineno for n in ast.walk(ptree) if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Attribute) and n.func.attr == "Kernel") or (isinstance(n.func, ast.Name) and n.func.id == "Kernel")) and any(kw.arg == "threads" for kw in n.keywords)}):
            return _stage_a_fail(t0, f"forbidden_kernel_threads_kwarg: T.Kernel(...) does not accept threads= (lines {', '.join(map(str, kl))}); pass threads= only to T.SimtVF(threads=128)", "FORBIDDEN_KERNEL_THREADS_KWARG")
        if pm := _parallel_forbidden_math_calls(ptree):
            return _stage_a_fail(t0, f"forbidden_tracka_parallel_scalar_math: T.Parallel body calls {', '.join(f'T.{o}' for o in pm)}; move elementwise math under with T.SimdVF(): with T.simd.vcvt / T.simd.vexp, or under with T.SimtVF(threads=128): with T.cast", "FORBIDDEN_TRACKA_PARALLEL_SCALAR_MATH")
        if _uncovered_1c2v_subcore_mte3_writes(ptree):
            return _stage_a_fail(t0, "uncovered_1c2v_subcore_mte3_write: T.MixedKernel(sids=2) writes shared GM/workspace buffer without partitioning slices by 'sid' (causes insert_sync AIV0.MTE3->AIV1.MTE3 fatal); partition rows as sid * half_m : (sid + 1) * half_m", "UNCOVERED_1C2V_SUBCORE_MTE3_WRITE")
        if sv := check_tilelang_semantic_contracts(dsl_source, protected_exports=frozenset()):
            return _stage_a_fail(t0, "; ".join(f"{v.code}(line {v.lineno}): {v.message}" for v in sv), sv[0].code)
        if not (tl_rep := verify_tilelang_ast(dsl_source)).passed:
            return _stage_a_fail(t0, "; ".join(tl_rep.errors), "TILELANG_AST_VIOLATION")
        if (en := _detect_module_entrypoint(ptree)) and (pe := check_npu_entrypoint_purity(dsl_source, en)):
            return _stage_a_fail(t0, "; ".join(pe), "NPU_ENTRYPOINT_PURITY_VIOLATION")
        with ascend_compile_session("ascend"):
            prim_func, sel_pc, sel_cf = _resolve_prim_func_and_configs_from_dsl(dsl_source, factory_name=factory_name, factory_kwargs=factory_kwargs, config_selector_name=config_selector_name, dims=dims)
            if not _is_tvm_ir_node(prim_func) and callable(prim_func):
                errs = [*_check_custom_kernel_redlines(dsl_source), "stage_a_non_kernel_callable: candidate is a plain Python function without @T.prim_func or Track-B TU"]
                return _stage_a_fail(t0, "; ".join(errs), "STAGE_A_NON_KERNEL_CALLABLE", backend_kind="python_kernel")
            mpc, mcf = _merge_pass_configs_and_flags(sel_pc, sel_cf, pass_configs, compile_flags)
            insp: EnvelopeInspectionResult = inspect_tilelang_primfunc(prim_func, target=target, pass_configs=mpc or None, compile_flags=mcf or None, var_bindings=_resolve_stage_a_var_bindings(dims, var_bindings), ordered_tensor_shapes=ordered_tensor_shapes)
    except Exception as exc:
        return _stage_a_fail(t0, f"stage_a_parse_or_lower_error: {type(exc).__name__}: {exc}", "STAGE_A_PARSE_OR_LOWER_ERROR")

    ledger, features, route = insp.ledger, None, None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            route_execution_track,
        )

        route = route_execution_track(prim_func, ledger=ledger, target=target, case_ratios=case_ratios)
        features = dict(route.features) if route.features is not None else None
    if not ledger.is_legal:
        rs = ([f"max_overflow_bytes={ledger.max_overflow_bytes}"] if ledger.max_overflow_bytes > 0 else []) + ([insp.lowering_error] if insp.lowering_error else [])
        return _stage_a_fail(t0, "; ".join(rs) or "stage_a_static_ledger_illegal", "STAGE_A_STATIC_ILLEGAL", prim_func=prim_func, ledger=ledger, features=features, route=route)
    if redlines := _check_custom_kernel_redlines(dsl_source):
        return _stage_a_fail(t0, "; ".join(redlines), "CUSTOM_KERNEL_REDLINE", prim_func=prim_func, ledger=ledger, features=features, route=route)
    engine_reasons: list[str] = []
    if require_cube_engine:
        has_cube = int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) > 0 or any(
            _is_t_dsl_call(n, "gemm") or _is_t_dsl_call(n, "gemm_v0")
            for n in ast.walk(ptree)
        )
        if not has_cube:
            engine_reasons.append(
                "missing_cube_engine_primitives: Track-A @T.prim_func contains zero "
                "Cube MMAD/T.gemm operations (AIC.PIPE_M == 0)"
            )
    if require_hardware_dma:
        uses_simt = bool(
            ledger.has_indirect_gather
            or any(_is_t_dsl_call(n, "SimtVF") for n in ast.walk(ptree))
        )
        has_mte2 = any(
            k.endswith("PIPE_MTE2") and int(v) > 0 for k, v in ledger.pipe_cycles.items()
        )
        if not uses_simt and not has_mte2:
            engine_reasons.append(
                "missing_hardware_dma_staging: Track-A @T.prim_func performs zero "
                "GM->UB/L1 DMA transfers (PIPE_MTE2 == 0)"
            )
    if engine_reasons:
        return _stage_a_fail(
            t0,
            "; ".join(engine_reasons),
            "MISSING_ENGINE_PRIMITIVES",
            prim_func=prim_func,
            ledger=ledger,
            features=features,
            route=route,
        )
    return StageAResult(
        passed=True,
        prim_func=prim_func,
        ledger=ledger,
        features=features,
        route=route,
        elapsed_ms=(time.monotonic() - t0) * 1000.0,
    )


def synthesize_pybind_extension_wrapper(
    op_name: str, kernel_launch_decl: str, tensor_arg_names: Sequence[str], *,
    output_arg_names: Sequence[str] | None = None, scalar_params: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Generate a thin torch_npu pybind11 wrapper preserving Python op_fn(*inputs) ABI."""
    safe_op = re.sub(r"[^A-Za-z0-9_]", "_", (op_name or "ascend_op").strip())
    safe_op = f"op_{safe_op}" if (not safe_op or safe_op[0].isdigit()) else safe_op
    decl = str(kernel_launch_decl or "").strip()
    if m_fn := re.search(r"\b([A-Za-z_]\w*)\s*\(", decl):
        launch_fn, dstmt = m_fn.group(1), decl if decl.endswith((";", "}")) else f"{decl};"
    else:
        launch_fn = re.sub(r"[^A-Za-z0-9_]", "_", decl or f"launch_{safe_op}")
        launch_fn = f"launch_{safe_op}" if (not launch_fn or launch_fn[0].isdigit()) else launch_fn
        adecls = ", ".join(["aclrtStream stream", *(f"void* {re.sub(r'[^A-Za-z0-9_]', '_', a)}" for a in tensor_arg_names), *(f"{pt} {pn}" for pt, pn in (scalar_params or ()))])
        dstmt = f'extern "C" void {launch_fn}({adecls});'
    ctens = [re.sub(r"[^A-Za-z0-9_]", "_", str(a).strip()) or f"t{i}" for i, a in enumerate(tensor_arg_names)]
    cscal = [(str(pt).strip(), re.sub(r"[^A-Za-z0-9_]", "_", str(pn).strip()) or f"s{i}") for i, (pt, pn) in enumerate(scalar_params or ())]
    cparams = ", ".join([*(f"const at::Tensor& {t}" for t in ctens), *(f"{pt} {pn}" for pt, pn in cscal)])
    stream_ln = f"    auto npu_stream = c10_npu::getCurrentNPUStream({ctens[0]}.device().index());" if ctens else "    auto npu_stream = c10_npu::getCurrentNPUStream();"
    largs = ", ".join(["npu_stream.stream()", *(f"{t}.data_ptr()" for t in ctens), *(pn for _, pn in cscal)])
    if output_arg_names:
        outs = [re.sub(r"[^A-Za-z0-9_]", "_", str(o).strip()) for o in output_arg_names]
        rt = "at::Tensor" if len(outs) == 1 else f"std::tuple<{', '.join('at::Tensor' for _ in outs)}>"
        rstmt = f"    return {outs[0]};" if len(outs) == 1 else f"    return std::make_tuple({', '.join(outs)});"
    elif ctens:
        rt, rstmt = "at::Tensor", f"    return {ctens[-1]};"
    else:
        rt, rstmt = "void", ""
    lines = [
        "#include <torch/extension.h>", "#include <torch_npu/csrc/core/npu/NPUStream.h>", "#include <acl/acl.h>", "#include <tuple>", "", dstmt, "",
        f"{rt} py_{safe_op}_forward({cparams}) {{",
        *(f'    TORCH_CHECK({t}.is_contiguous(), "Expected contiguous NPU tensor for {t}");' for t in ctens),
        stream_ln, f"    {launch_fn}({largs});", *(([rstmt] if rstmt else [])), "}", "",
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", f'    m.def("{safe_op}", &py_{safe_op}_forward, "Ascend 950 torch_npu extension binding for {safe_op}");', "}", "",
    ]
    return "\n".join(lines)


def _resolve_agent_gates_bin() -> str:
    if (eb := os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()) and Path(eb).is_file():
        return eb
    if found := shutil.which("agent-gates"):
        return found
    cands = ([Path(ef).expanduser().resolve().parent / ".venv/bin/agent-gates", Path(ef).expanduser().resolve().parent.parent / ".venv/bin/agent-gates"] if (ef := os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()) else [])
    cands.append(Path(__file__).resolve().parents[5].parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
    if hit := next((str(c) for c in cands if c.is_file()), None):
        return hit
    raise FileNotFoundError("Cannot locate 'agent-gates'. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or add to PATH.")


def verify_agent_gates_receipt(agent_gates_bin: str, run_id: str, *, source_root: str | Path, env: Mapping[str, str] | None = None, timeout_sec: float = 30.0) -> tuple[bool, str]:
    """Verify one AgentGates receipt by run ID and require an authoritative PASS."""
    try:
        proc = subprocess.run([agent_gates_bin, "verify", "--run-id", run_id, "--json"], text=True, capture_output=True, cwd=str(Path(source_root).resolve()), env=dict(env) if env is not None else None, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return False, f"agent-gates verify timed out after {timeout_sec:.0f}s"
    parsed: dict[str, object] = {}
    if (bi := proc.stdout.find("{")) >= 0:
        try:
            if isinstance(ld := json.loads(proc.stdout[bi:]), dict): parsed = ld
        except json.JSONDecodeError as exc:
            return False, f"agent-gates verify returned malformed JSON: {exc}"
    st, vrid, val = str(parsed.get("status", "")), str(parsed.get("runId", "")), parsed.get("valid") is True
    if proc.returncode == 0 and st == "PASS" and vrid == run_id and val:
        return True, ""
    return False, f"receipt_verification_failed: status={st or 'MISSING'} runId={vrid or 'MISSING'} valid={val} {(proc.stderr or proc.stdout or 'agent-gates verify failed').strip()[-1000:]}"


def run_stage_c_agent_gates(
    ledger: ObjectivePhysicalLedger, *, source_root: str | Path, device_id: str = "1",
    env_file: str | Path | None = None, device_test_rel: str | None = None,
    telemetry_report_path: str | Path | None = None, gate: str = "tilefoundry-npu-device", timeout_sec: int = 300,
) -> StageCResult:
    """Execute lease-locked NPU validation and msprof collection strictly via AgentGates."""
    ag_bin, src_p = _resolve_agent_gates_bin(), Path(source_root).resolve()
    cmd = [ag_bin, "run", "--adapter", "tilefoundry-npu", "--source", str(src_p), "--gate", gate, "--device-id", str(device_id), "--json"]
    with tempfile.TemporaryDirectory(prefix="tf_stage_c_env_") as tmp_dir:
        if device_test_rel and env_file and Path(env_file).is_file():
            te = Path(tmp_dir) / "env.sh"
            lines = [ln for ln in Path(env_file).read_text(encoding="utf-8").splitlines() if not ln.startswith("TILEFOUNDRY_NPU_DEVICE_TESTS=")]
            te.write_text("\n".join([*lines, f'TILEFOUNDRY_NPU_DEVICE_TESTS="{device_test_rel}"']) + "\n", encoding="utf-8")
            cmd.extend(["--env-file", str(te)])
        elif env_file and Path(env_file).is_file():
            cmd.extend(["--env-file", str(env_file)])
        proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(src_p), timeout=timeout_sec)
        parsed = {}
        if (bi := proc.stdout.find("{")) >= 0:
            with contextlib.suppress(json.JSONDecodeError):
                if isinstance(ld := json.loads(proc.stdout[bi:]), dict): parsed = ld
        status, run_id, rpath = str(parsed.get("status", "FAIL")), str(parsed.get("runId", "")).strip(), str(parsed.get("receiptPath", ""))
        passed = proc.returncode == 0 and status == "PASS"
        run_err = (proc.stderr or proc.stdout or "agent-gates failed")[-1000:]
        ver_err = ""
        if passed and not run_id: passed, run_err = False, "agent-gates run reported PASS without a runId"
        elif passed: passed, ver_err = verify_agent_gates_receipt(ag_bin, run_id, source_root=src_p)
    tel: dict[str, object] = {}
    if passed and telemetry_report_path and Path(telemetry_report_path).is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            tel = json.loads(Path(telemetry_report_path).read_text(encoding="utf-8"))
    return StageCResult(passed, passed, float(tel.get("dur_us", tel.get("measured_latency_us", 0.0))), compute_static_dynamic_residual(ledger, tel), tel, run_id, rpath, "" if passed else (ver_err or run_err)[-1000:])


class GateFunnel:
    """Orchestrates Stage A static -> Stage B bisheng -> Stage C AgentGates NPU verification."""

    def __init__(
        self, *, target: AscendTarget | None = None, skip_stage_b: bool = False,
        compile_runner: Callable[[str], StageBResult] | None = None,
        device_runner: Callable[[str, StageAResult], StageCResult] | None = None,
        enable_track_router: bool = False, artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._target, self._skip_stage_b, self._compile_runner, self._device_runner, self._enable_track_router, self._artifact_store = (
            target or AscendTarget("ascend.950pr"), skip_stage_b, compile_runner, device_runner, enable_track_router, artifact_store
        )

    def evaluate(
        self, dsl_source: str, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None,
        config_selector_name: str | None = None, pass_configs: dict[str, object] | None = None,
        compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
        var_bindings: Mapping[str, int] | None = None, backend_kind: str | None = None, freedom_mask: object | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
        regime_tag: str = "", spec_key: str = "",
    ) -> FunnelEvaluation:
        """Run Stage A, Stage B, and Stage C sequentially with early rejection."""
        sa = (
            evaluate_ascendc_candidate(dsl_source, freedom_mask=freedom_mask, target=self._target, enable_track_router=self._enable_track_router)
            if backend_kind in ("ascendc", "ascendc_template")
            else run_stage_a_static(dsl_source, factory_name=factory_name, factory_kwargs=factory_kwargs, target=self._target, pass_configs=pass_configs, compile_flags=compile_flags, config_selector_name=config_selector_name, dims=dims, var_bindings=var_bindings, enable_track_router=self._enable_track_router, ordered_tensor_shapes=ordered_tensor_shapes)
        )
        if not sa.passed or sa.ledger is None:
            return FunnelEvaluation(False, "stage_a", sa, error_message=sa.error_message)
        sb = (
            StageBResult(passed=True) if self._skip_stage_b
            else (self._compile_runner(dsl_source) if self._compile_runner is not None
                  else (run_stage_b_ascendc_compile(dsl_source, compile_flags=compile_flags, artifact_store=self._artifact_store, regime_tag=regime_tag, spec_key=spec_key) if backend_kind in ("ascendc", "ascendc_template")
                        else run_stage_b_compile(dsl_source, factory_name=factory_name, factory_kwargs=factory_kwargs, config_selector_name=config_selector_name, pass_configs=pass_configs, compile_flags=compile_flags, dims=dims, artifact_store=self._artifact_store, regime_tag=regime_tag, spec_key=spec_key)))
        )
        if not sb.passed:
            return FunnelEvaluation(False, "stage_b", sa, sb, error_message=sb.error_message)
        cs = sb.compiler_summary
        if self._device_runner is None:
            sc = StageCResult(True, True, 0.0, compute_static_dynamic_residual(sa.ledger, {}, compiler_summary=cs, roofline=roofline), {})
        else:
            _inject_cann_cpath_env()
            sc = self._device_runner(dsl_source, sa)
            if cs is not None and sc.residual is not None:
                r = sc.residual
                sc = replace(sc, residual=replace(
                    r, spill_bytes=max(r.spill_bytes, int(cs.spill_bytes)), vf_fusion_breaks=max(r.vf_fusion_breaks, int(cs.vf_fusion_breaks)),
                    alias_serialized_buffers=max(r.alias_serialized_buffers, int(cs.alias_serialized_buffers)),
                    dual_issue_ipc_collapse=max(r.dual_issue_ipc_collapse, float(cs.dual_issue_ipc_collapse)),
                    dcache_dcci_flushes=max(r.dcache_dcci_flushes, int(cs.dcache_dcci_flushes)),
                    has_local_membar_in_vf_call=r.has_local_membar_in_vf_call or bool(cs.has_local_membar_in_vf_call),
                    vloopv2_hit=r.vloopv2_hit and bool(sb.vloopv2_hit) and cs.vloopv2_loops > 0 and cs.software_loops == 0,
                    observation=sa.ledger.to_observation_5d(compiler_summary=cs, telemetry=sc.telemetry, roofline=roofline),
                ))
            elif sc.residual is not None and roofline is not None:
                sc = replace(sc, residual=replace(sc.residual, observation=sa.ledger.to_observation_5d(compiler_summary=cs, telemetry=sc.telemetry, roofline=roofline)))
        if not sc.passed or not sc.accuracy_passed:
            return FunnelEvaluation(False, "stage_c", sa, sb, sc, error_message=sc.error_message)
        return FunnelEvaluation(True, None, sa, sb, sc)


def prescreen_dsl_tile_kwargs(
    dsl_source: str, *, factory_name: str, candidate_kwargs_list: Sequence[Mapping[str, object]],
    target: AscendTarget | None = None, var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None, pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, object], StageAResult]:
    """Run Stage A across tile kwargs and pick the lowest-cost legal configuration."""
    if not candidate_kwargs_list:
        raise ValueError("candidate_kwargs_list must not be empty")
    legal, rejs = [], []
    for idx, rkw in enumerate(candidate_kwargs_list):
        kw = dict(rkw)
        res = run_stage_a_static(dsl_source, factory_name=factory_name, factory_kwargs=kw, target=target, pass_configs=pass_configs, compile_flags=compile_flags, config_selector_name=config_selector_name, dims=dims, var_bindings=var_bindings, ordered_tensor_shapes=ordered_tensor_shapes)
        if not res.passed or (lg := res.ledger) is None: rejs.append(f"{kw}: {res.error_message or 'stage_a_failed'}")
        elif not lg.is_legal or lg.max_overflow_bytes > 0: rejs.append(f"{kw}: overflow={lg.max_overflow_bytes}")
        elif deg := [b for wl in lg.waterlines.values() for b in wl.degraded_single_buffers]: rejs.append(f"{kw}: degraded_single_buffers={deg}")
        else:
            legal.append(((float(lg.steady_state_ii) if lg.steady_state_ii and lg.steady_state_ii > 0 else float("inf"), lg.pipe_cycles.get(lg.dominant_pipe, max((v for v in lg.pipe_cycles.values() if v > 0), default=0)), idx), kw, res))
    if not legal:
        raise ValueError(f"no legal tile kwargs passed Stage A prescreen: {'; '.join(rejs[:5])}")
    legal.sort(key=lambda item: item[0])
    return legal[0][1], legal[0][2]


def ledger_to_json(ledger: ObjectivePhysicalLedger | None) -> str:
    """Serialize an optional ledger safely for worker prompts."""
    return ledger.to_json() if ledger is not None else json.dumps(build_objective_physical_ledger(analysis_mode="readonly_envelope", is_legal=False, steady_state_ii=None, pipe_cycles={}, waterlines={}).to_dict(), indent=2)


def _extract_concrete_shape_list(raw_shapes: object) -> list[tuple[int, ...]]:
    if not isinstance(raw_shapes, (list, tuple)) or not raw_shapes:
        return []
    if all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in raw_shapes):
        return [tuple(int(d) for d in raw_shapes)]
    out: list[tuple[int, ...]] = []
    for it in raw_shapes:
        if isinstance(it, (list, tuple)) and it:
            out.extend([tuple(int(d) for d in it)] if all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in it) else _extract_concrete_shape_list(it))
    return out


def extract_ordered_tensor_shapes(
    *, case_entry: Mapping[str, object] | None = None, dims: Mapping[str, object] | None = None,
    proto: Mapping[str, object] | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Extract parameter-ordered concrete tensor shapes from case, dims, or proto specs."""
    if ordered_tensor_shapes is not None:
        return tuple(ex) if (ex := _extract_concrete_shape_list(ordered_tensor_shapes)) else None
    for c in (case_entry, dims):
        if isinstance(c, Mapping) and "ordered_tensor_shapes" in c and (ex := _extract_concrete_shape_list(c.get("ordered_tensor_shapes"))):
            return tuple(ex)
    ins, outs = [], []
    for c in (case_entry, dims):
        if isinstance(c, Mapping):
            if not ins: ins = _extract_concrete_shape_list(c.get("input_shapes") if "input_shapes" in c else c.get("input_shape"))
            if not outs: outs = _extract_concrete_shape_list(c.get("output_shapes") if "output_shapes" in c else c.get("output_shape"))
    if not outs and isinstance(proto, Mapping):
        attrs = case_entry.get("attrs") if isinstance(case_entry, Mapping) and isinstance(case_entry.get("attrs"), Mapping) else None
        op_b = proto["operator"] if "outputs" not in proto and isinstance(proto.get("operator"), Mapping) else proto
        routs = op_b.get("outputs") if isinstance(op_b, Mapping) else None
        if isinstance(routs, Sequence) and not isinstance(routs, (str, bytes)):
            dl = {str(k).strip().lower(): int(v) for s in (attrs, dims) if isinstance(s, Mapping) for k, v in s.items() if isinstance(v, int) and not isinstance(v, bool) and v >= 0}
            rlay = next((str(s[lk]).strip().upper() for s in (attrs, dims) if isinstance(s, Mapping) for lk in ("inputLayout", "input_layout", "layout") if isinstance(s.get(lk), str) and str(s[lk]).strip()), "")
            for it in routs:
                if isinstance(it, Mapping) and isinstance(sh_s := it.get("shape"), str) and "[" in sh_s:
                    ms = re.findall(r"(?:([A-Za-z0-9_]+)\s*:\s*)?\[([^\]]+)\]", sh_s)
                    ch = next((inn for tg, inn in ms if rlay and tg.strip().upper() == rlay), "") or (ms[0][1] if ms else "")
                    if ch:
                        sv = [int(t) if t.isdigit() else dl.get(t.lower()) for t in (x.strip() for x in ch.split(",") if x.strip())]
                        if sv and all(x is not None for x in sv):
                            outs.append(tuple(int(x) for x in sv if x is not None))
    return tuple(comb) if (comb := [*ins, *outs]) else None


__all__ = [
    "FunnelEvaluation", "GateFunnel", "StageAResult", "StageBResult", "StageCResult",
    "_resolve_artifact_store", "_resolve_cpp_consts", "ascend_compile_session",
    "build_stage_b_driver_source", "check_npu_entrypoint_purity", "check_repair_monotonicity",
    "compile_trackb_shared_so", "compute_trackb_shared_so_filename", "evaluate_ascendc_candidate",
    "extract_ordered_tensor_shapes", "has_unreserved_vf_stack_overflow", "ledger_to_json",
    "parse_cce_res_usage", "prescreen_dsl_tile_kwargs", "resolve_trackb_compile_flags",
    "run_stage_a_static", "run_stage_b_ascendc_compile", "run_stage_b_compile",
    "run_stage_c_agent_gates", "synthesize_pybind_extension_wrapper", "verify_agent_gates_receipt",
]
