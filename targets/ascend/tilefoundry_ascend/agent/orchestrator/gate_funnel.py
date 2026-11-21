"""Three-stage verification funnel: Stage A static ledger -> Stage B bisheng -> Stage C NPU."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    MicroarchProfile,
    extract_microarch_profile_from_cce,
)
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
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class StageAResult:
    """Output of millisecond Stage A static AST, ledger, and router inspection."""

    passed: bool
    prim_func: object | None
    ledger: ObjectivePhysicalLedger | None
    features: object | None = None
    route: object | None = None
    error_message: str = ""
    elapsed_ms: float = 0.0
    backend_kind: str = "tilelang"


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
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N_kv": "n_kv",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
    "M": "m_dim",
    "K": "k_dim",
    "H": "num_heads",
    "E": "num_experts",
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


def check_npu_entrypoint_purity(module_source: str, entrypoint: str) -> list[str]:
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
    queue = [entrypoint]
    while queue:
        fn_name = queue.pop(0)
        if fn_name in visited or fn_name not in funcs:
            continue
        visited.add(fn_name)
        fnode = funcs[fn_name]
        reachable_src.append(ast.get_source_segment(module_source, fnode) or "")
        for stmt in fnode.body:
            cond_txt = (
                (ast.get_source_segment(module_source, stmt.test) or "")
                if isinstance(stmt, ast.If)
                else ""
            )
            if isinstance(stmt, ast.If) and "device.type" in cond_txt and "!=" in cond_txt:
                continue
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    callee = (
                        sub.func.id
                        if isinstance(sub.func, ast.Name)
                        else (sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
                    )
                    if callee in _FORBIDDEN_NPU_HOST_CALLS:
                        errors.append(f"npu_host_fallback_forbidden:{fn_name}->{callee}")
                    if isinstance(sub.func, ast.Name) and sub.func.id in funcs:
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
    if backend == "tilelang_dsl" and "tilelang.compile(" not in "\n".join(reachable_src):
        errors.append(
            f"detached_decoy_primfunc:{entrypoint} never invokes tilelang.compile"
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


def _ascend_target_context():
    try:
        from tilelang import tvm  # noqa: PLC0415
        from tilelang.backend.module import create_backend_context  # noqa: PLC0415

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


_VLOOPV2_MISS_RE = re.compile(
    r"(?:vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|"
    r"vloop_?v2\s*[:=]\s*(?:false|0|no|miss)|cannot\s+use\s+vloop_?v2)",
    re.IGNORECASE,
)


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


def _check_custom_kernel_redlines(dsl_source: str) -> list[str]:
    errors: list[str] = []
    lower = dsl_source.lower()
    if "l1->gm" in lower or "gm->l0a" in lower or "gm->l0b" in lower:
        errors.append("forbidden_dav3510_dma_path")
    if ("int4b_t" in lower or "cube_int4" in lower) and "dist_unpack4" not in lower and "aiv_unpack" not in lower:
        errors.append("forbidden_dav3510_cube_int4_without_aiv_unpack")
    return errors


_CPP_DTYPE_BYTES: dict[str, int] = {
    "uint8_t": 1,
    "int8_t": 1,
    "char": 1,
    "half": 2,
    "bfloat16_t": 2,
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


def _safe_eval_cpp_int(expr: str, env: Mapping[str, int]) -> int | None:
    cleaned = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", str(expr or "").strip())
    if not cleaned:
        return None
    py_expr = _rewrite_cpp_ternary(_strip_outer_parens(cleaned))
    try:
        tree = ast.parse(py_expr, mode="eval")
        return _eval_cpp_ast_node(tree, env)
    except Exception:
        return None


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


def _resolve_cpp_consts(cpp_source: str) -> dict[str, int]:
    """Fold constexpr integers with template arguments, aliases, and sizeof dtype widths."""
    declarations = [
        (m.group(1), m.group(2))
        for m in re.finditer(
            r"\bconstexpr\s+(?:int|int32_t|uint32_t|size_t)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);",
            cpp_source,
        )
    ]
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
    return consts


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


def _build_ascendc_static_ledger(
    cpp_source: str,
    *,
    target: AscendTarget | None = None,
    compiler_summary: BishengResourceSummary | None = None,
    objdump_text: str = "",
    microarch_profile: MicroarchProfile | Mapping[str, object] | None = None,
) -> ObjectivePhysicalLedger:
    """Compute SRAM waterlines and pipe cycles from C++20 constexprs, calls, and objdump."""
    resolved_target = target or AscendTarget("ascend.950pr")
    from tilefoundry_ascend.ascend.target import AscendSpaceFacts  # noqa: PLC0415

    facts = resolved_target.get_facts(AscendSpaceFacts)
    lower = cpp_source.lower()

    consts = _resolve_cpp_consts(cpp_source)

    ub_offsets: list[tuple[str, int, int]] = []
    l1_offsets: list[tuple[str, int, int]] = []
    ptr_env: dict[str, tuple[str, int]] = {}

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
    mmad_src_cnt = len(re.findall(r"\basc_mmad\s*\(", cpp_source))
    vf_src_cnt = len(re.findall(r"\b(?:simd_inst|Reg)::\w+\s*(?:<[^>]*>)?\s*\(", cpp_source))

    mte2_cnt = max(mte2_src_cnt, dis_counts["mte2"])
    mte3_cnt = max(mte3_src_cnt, dis_counts["mte3"])
    fix_cnt = max(fix_src_cnt, dis_counts["fixpipe"])
    mmad_cnt = max(mmad_src_cnt, dis_counts["cube_mmad"])
    vf_cnt = max(vf_src_cnt, dis_counts["vec"])

    spill_penalty = int(compiler_summary.spill_bytes) * 2 if compiler_summary else 0
    sw_loop_mult = 2 if (compiler_summary and not compiler_summary.vloopv2_hit) else 1

    pipe_cycles: dict[str, int] = {}
    if mte2_cnt > 0 or not uses_cube:
        pipe_cycles["AIV0.PIPE_MTE2"] = max(1, mte2_cnt) * max(16, (tile_m * tile_n * 2) // 64)
    if mte3_cnt > 0:
        pipe_cycles["AIV0.PIPE_MTE3"] = mte3_cnt * max(16, (tile_m * tile_n * 2) // 64)
    if vf_cnt > 0 or not uses_cube:
        pipe_cycles["AIV0.PIPE_V"] = (
            max(1, vf_cnt) * max(8, (tile_m * tile_n) // 64) * sw_loop_mult + spill_penalty
        )
    if mmad_cnt > 0 or uses_cube:
        pipe_cycles["AIC.PIPE_M"] = max(1, mmad_cnt) * max(16, (tile_m * tile_n * tile_k) // 4096)
    if fix_cnt > 0 or uses_cube:
        pipe_cycles["AIC.PIPE_FIX"] = max(1, fix_cnt) * max(12, (tile_m * tile_n * 2) // 256)
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
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=is_legal,
        steady_state_ii=steady_ii,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        has_indirect_gather=uses_simt,
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
        )

    from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask  # noqa: PLC0415
    from tilefoundry_ascend.ascend.sil.verifier import (  # noqa: PLC0415
        SILVerifier,
        verify_source_against_facets,
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
    ledger = _build_ascendc_static_ledger(cpp_source, target=target)

    errors = [*redline_errs, *facet_report.errors]
    if ledger.max_overflow_bytes > 0:
        errors.append(f"max_overflow_bytes={ledger.max_overflow_bytes}")

    features: object | None = None
    route: object | None = None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            StructuralFeatures,
            route_execution_track,
        )

        lower = cpp_source.lower()
        uses_cube = "mmad" in lower or "l0c" in lower
        uses_simt = "__simt_vf__" in cpp_source or "Simt::" in cpp_source
        uses_ssbuf = "ssbuf" in lower or "crosscore" in lower
        static_affine_bounds = not uses_simt
        cube_to_vector_ratio = 1.0 if uses_cube else 0.0
        quant_granularity = "per_token_or_block" if "deq_scale" in lower else "none"
        features = (
            StructuralFeatures.from_case_ratios(
                case_ratios,
                static_affine_bounds=static_affine_bounds,
                cube_to_vector_ratio=cube_to_vector_ratio,
                quant_granularity=quant_granularity,
                indirect_sparse_gather=uses_simt,
                cross_iter_recurrence="none",
                dynamic_cross_core_handoff=uses_ssbuf,
            )
            if case_ratios is not None
            else StructuralFeatures(
                phi_1_static_affine_bounds=static_affine_bounds,
                phi_2_cube_to_vector_ratio=cube_to_vector_ratio,
                phi_3_quant_granularity=quant_granularity,
                phi_4_indirect_sparse_gather=uses_simt,
                phi_5_cross_iter_recurrence="none",
                phi_6_dynamic_cross_core_handoff=uses_ssbuf,
                phi_7_inner_contiguous_bytes=0,
                phi_8_persistent_wave_occupancy=0.0,
            )
        )
        route = route_execution_track(features, simt_dominant=uses_simt)

    if errors:
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=ledger,
            features=features,
            route=route,
            error_message="; ".join(errors),
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            backend_kind="ascendc",
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
        with _ascend_target_context():
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
) -> StageAResult:
    """Run millisecond Stage A AST, Two-Snapshot IR Diff, and optional track routing."""
    t0 = time.monotonic()
    try:
        parsed_tree = ast.parse(dsl_source)
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
                )
        with _ascend_target_context():
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
                )
            merged_pc, merged_cf = _merge_pass_configs_and_flags(
                sel_pc,
                sel_cf,
                pass_configs,
                compile_flags,
            )
            inspection: EnvelopeInspectionResult = inspect_tilelang_primfunc(
                prim_func,
                target=target,
                pass_configs=merged_pc or None,
                compile_flags=merged_cf or None,
                var_bindings=var_bindings,
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
        )

    ledger = inspection.ledger
    features: object | None = None
    route: object | None = None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (  # noqa: PLC0415
            extract_structural_features,
            route_execution_track,
        )

        structural = extract_structural_features(prim_func, ledger=ledger, target=target)
        if case_ratios is not None:
            structural = replace(
                structural,
                phi_7_inner_contiguous_bytes=case_ratios.inner_contiguous_bytes,
                phi_8_persistent_wave_occupancy=case_ratios.wave_occupancy_ratio,
            )
        features = structural
        route = route_execution_track(structural)
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
        }
    )
    return (
        "import contextlib, importlib.util, inspect, json, pathlib, tilelang\n"
        "from tilelang import tvm\n"
        "from tilelang.backend.module import create_backend_context\n"
        "from tvm import tirx, IRModule\n"
        "def _ascend_ctx():\n"
        "    if tvm.target.Target.current(allow_none=True) is None:\n"
        "        return create_backend_context('ascend').target\n"
        "    return contextlib.nullcontext()\n"
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
        "with _ascend_ctx():\n"
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
        "assert _fn is not None\n"
        "if not isinstance(_fn, (tirx.PrimFunc, IRModule)) and callable(_fn):\n"
        "    _compile_hook = getattr(_mod, 'compile_kernel', getattr(_mod, 'build_kernel', None))\n"
        "    if not callable(_compile_hook):\n"
        "        raise RuntimeError(\n"
        "            'stage_b_non_kernel_callable: plain Python function lacks @T.prim_func or compile_kernel'\n"
        "        )\n"
        "    _compile_hook()\n"
        "else:\n"
        "    _merged_pc, _merged_cf = {}, []\n"
        "    def _absorb_flags(items):\n"
        "        for _f in items:\n"
        "            if _f and _f not in _merged_cf:\n"
        "                _merged_cf.append(_f)\n"
        "    _attrs = getattr(_fn, 'attrs', None)\n"
        "    if _attrs and hasattr(_attrs, 'items'):\n"
        "        for _pk in ('tl.pass_configs', 'tilelang_pass_configs'):\n"
        "            if _pk in _attrs and hasattr(_attrs[_pk], 'items'):\n"
        "                for _ak, _av in _attrs[_pk].items():\n"
        "                    _val = getattr(_av, 'value', _av)\n"
        "                    if str(_ak) == 'tl.device_compile_flags':\n"
        "                        _absorb_flags(_norm_flags(_val))\n"
        "                    else:\n"
        "                        _merged_pc[str(_ak)] = _val\n"
        "        for _ck in ('tir.cce_flags', 'tilelang_compile_flags'):\n"
        "            if _ck in _attrs:\n"
        "                _absorb_flags(_norm_flags(_attrs[_ck]))\n"
        "    _merged_pc.update(_sel_pc)\n"
        "    _absorb_flags(_sel_cf)\n"
        "    _merged_pc.update(_cfg['pass_configs'])\n"
        "    _absorb_flags(_cfg['compile_flags'])\n"
        "    _absorb_flags(['--cce-res-usage'])\n"
        "    if _merged_cf:\n"
        "        _merged_pc['tl.device_compile_flags'] = list(_merged_cf)\n"
        "    with tvm.transform.PassContext(config=_merged_pc):\n"
        "        _art = tilelang.lower(\n"
        "            _fn, target='ascend', enable_device_compile=True\n"
        "        )\n"
        "    assert _art.kernel_source\n"
        "    if _cfg.get('kernel_dump_path'):\n"
        "        pathlib.Path(_cfg['kernel_dump_path']).write_text(str(_art.kernel_source), encoding='utf-8')\n"
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
) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    py_exe = _resolve_conda_python(python_bin)
    env = dict(os.environ)
    env["TILELANG_DISABLE_CACHE"] = "1"
    env.setdefault("USE_CUDA", "0")
    _inject_cann_cpath_env(env)
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_") as tmp_dir:
        cand_path = Path(tmp_dir) / "candidate_dsl.py"
        cand_path.write_text(dsl_source, encoding="utf-8")
        kernel_dump_path = Path(tmp_dir) / "lowered_kernel.cce"
        driver = build_stage_b_driver_source(
            cand_path,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=dims,
            kernel_dump_path=kernel_dump_path,
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
            return StageBResult(
                passed=False,
                error_message=f"stage_b_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
            )
        lowered_dump = (
            kernel_dump_path.read_text(encoding="utf-8", errors="ignore")
            if kernel_dump_path.is_file()
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
        detail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return StageBResult(
            passed=False,
            error_message=f"stage_b_compile_failed (rc={proc.returncode}): {detail}",
            elapsed_sec=time.monotonic() - t0,
            spill_bytes=spill_bytes,
            vloopv2_hit=vloopv2_hit,
            compiler_summary=summary,
            microarch_profile=ma_profile,
        )
    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
    )


_CANN_INCLUDE_SUBDIRS: tuple[str, ...] = (
    "include",
    "x86_64-linux/include",
    "aarch64-linux/include",
    "runtime/include",
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
    "l0c_to_ub": ("asc_copy_l0c2ub", 2, 3, "float"),
    "l0c_to_l1": ("asc_copy_l0c2l1", 2, 3, "float"),
    "l0c_to_gm": ("asc_copy_l0c2gm", 2, 3, "float"),
    "ub_to_gm": ("asc_copy_ub2gm", 2, 3, "bytes"),
}


def _strip_cpp_casts(expr: str) -> str:
    """Drop C-style scalar casts so call arguments can be folded numerically."""
    return re.sub(
        r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|unsigned|int|float|double|half)\s*\)",
        "",
        expr,
    )


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
    consts = _resolve_cpp_consts(cpp_source)
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


def _default_asc_devkit_include() -> str:
    env_devkit = os.environ.get("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", "").strip()
    if env_devkit:
        return str(Path(env_devkit).expanduser() / "include")
    sibling = Path(__file__).resolve().parents[5].parent / "asc-devkit" / "include"
    return str(sibling) if sibling.is_dir() else ""


def run_stage_b_ascendc_compile(
    cpp_source: str,
    *,
    arch: str = "dav-3510",
    compile_flags: Sequence[str] | str | None = None,
    bisheng_bin: str | Path | None = None,
    cann_root: str | Path | None = None,
    asc_devkit_include: str | Path | None = None,
    timeout_sec: int = 120,
    enable_device_compile: bool = True,
) -> StageBResult:
    """Verify native AscendC C++ translation unit compilation via bisheng in an isolated tempdir."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return StageBResult(
            passed=False,
            error_message="stage_b_ascendc_empty_source",
            elapsed_sec=time.monotonic() - t0,
        )

    resolved_bisheng = _resolve_bisheng_bin(bisheng_bin, cann_root=cann_root)
    if resolved_bisheng is None:
        return StageBResult(
            passed=False,
            error_message="stage_b_ascendc_bisheng_not_found",
            elapsed_sec=time.monotonic() - t0,
        )

    extra_flags = _normalize_flag_list(compile_flags)
    has_explicit_std = any(f.startswith("-std=") for f in extra_flags)
    has_explicit_x = "-x" in extra_flags
    has_tl_templates = "tl_templates/ascend/" in cpp_source
    has_kernel_op = bool(
        re.search(r'^\s*#\s*include\s+[<"]kernel_operator\.h[>"]', cpp_source, re.MULTILINE)
    )

    std_flags: list[str] = []
    if not has_explicit_std:
        std_flags = (
            ["-std=c++20"]
            if (has_tl_templates and not has_kernel_op)
            else ["-std=c++17"]
        )

    include_flags: list[str] = []
    seen_inc_flags = set(extra_flags)

    def _add_inc(inc_dir: str) -> None:
        flag = f"-I{inc_dir}"
        if inc_dir and Path(inc_dir).is_dir() and flag not in seen_inc_flags:
            seen_inc_flags.add(flag)
            include_flags.append(flag)

    asc_inc_str = (
        str(asc_devkit_include).strip()
        if asc_devkit_include is not None
        else _default_asc_devkit_include()
    )
    _add_inc(asc_inc_str)
    if has_tl_templates:
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
            cmd = [
                resolved_bisheng,
                *arch_flags,
                "--cce-aicore-only",
                "--cce-res-usage",
                "-O2",
                *std_flags,
                *include_flags,
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
            detail = (proc.stderr or proc.stdout or "bisheng failed without diagnostics").strip()[
                -1500:
            ]
            return StageBResult(
                passed=False,
                error_message=f"stage_b_ascendc_compile_failed (rc={proc.returncode}): {detail}",
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
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
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
            )

    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
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
    ) -> None:
        self._target = target or AscendTarget("ascend.950pr")
        self._skip_stage_b = skip_stage_b
        self._compile_runner = compile_runner
        self._device_runner = device_runner
        self._enable_track_router = enable_track_router

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
                    ),
                )
                stage_c = replace(stage_c, residual=merged_res)

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


__all__ = [
    "FunnelEvaluation",
    "GateFunnel",
    "StageAResult",
    "StageBResult",
    "StageCResult",
    "build_stage_b_driver_source",
    "check_npu_entrypoint_purity",
    "evaluate_ascendc_candidate",
    "ledger_to_json",
    "parse_cce_res_usage",
    "prescreen_dsl_tile_kwargs",
    "run_stage_a_static",
    "run_stage_b_ascendc_compile",
    "run_stage_b_compile",
    "run_stage_c_agent_gates",
    "synthesize_pybind_extension_wrapper",
    "verify_agent_gates_receipt",
]
