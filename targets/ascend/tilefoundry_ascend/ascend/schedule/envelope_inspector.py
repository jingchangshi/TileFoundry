"""Mode B read-only single-pass dynamic envelope accounting and Two-Snapshot IR Diff."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry.vendor.tilelang_z3 import Buffer, Capacity, Edge, Reuse, Task, tf_adapter
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    extract_microarch_profile_from_tirx,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    _assign_dag_stages,
    _capacities_from_facts,
    _resources_for_tasks,
    solve_pipeline_with_monotone_backoff,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask

_TILELANG_SCOPE_MAP: dict[str, str] = {
    "shared.dyn": "UB",
    "shared": "UB",
    "shared.l1": "L1",
    "shared.l1.dyn": "L1",
    "shared.l0a": "L0A",
    "shared.l0a.dyn": "L0A",
    "shared.l0b": "L0B",
    "shared.l0b.dyn": "L0B",
    "shared.l0c": "L0C",
    "shared.l0c.dyn": "L0C",
    "shared.ss": "SSBUF",
    "global": "GM",
}
_EXTERN_CALL_TAILS = frozenset(("call_extern", "call_pure_extern", "call_intrin", "call_llvm_intrin"))
_MAX_SAFE_SPEC_INT = 1 << 22
_UNBOUND_SCALAR_EXTENT_DEFAULT = 16
_UNBOUND_SCALAR_NAME_TOKENS = (
    "block", "tile", "split", "chunk", "step", "iter", "topk", "top_k", "sparse", "group", "stage",
)
_BUFFER_FREE_AXIS_CAP = 65536
_NUMEL_ALIASES = ("total_elements", "total_elems", "n_elems", "num_elements", "numel")
_DIM_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("b", "b_dim", "batch", "batch_size"),
    ("m", "m_dim", "m_rows", "q_total_rows", "num_tokens", "tokens"),
    ("n", "n_dim", "n_cols"),
    ("k", "k_dim"),
    ("h", "h_dim", "heads", "num_heads", "n_q", "n_heads"),
    ("l", "l_dim", "seq_len", "s", "s_dim"),
    ("c", "c_dim", "channels", "in_channels", "out_channels"),
    ("v", "v_dim", "vocab", "vocab_size"),
    ("g", "g_dim", "groups", "num_groups"),
    ("d", "d_dim", "head_dim", "d_nope"),
    ("s_kv", "kv_len", "s_kv_dim"),
    _NUMEL_ALIASES,
)


@dataclass(frozen=True)
class BufferSnapshot:
    """Captured physical buffer allocation size in one IR snapshot."""

    name: str
    scope: str
    bytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CornerShape:
    """Corner shape descriptor."""

    rows: int
    cols: int


def _instantiate_corner_shape_matrix(n: int = 3) -> tuple[CornerShape, ...]:
    return tuple(CornerShape(rows=32 * (i + 1), cols=32 * (i + 1)) for i in range(n))


@dataclass(frozen=True)
class EnvelopeInspectionResult:
    """Output of Mode B dynamic envelope and Two-Snapshot IR Diff inspection."""

    ledger: ObjectivePhysicalLedger
    explicit_snapshot: tuple[BufferSnapshot, ...]
    lowered_snapshot: tuple[BufferSnapshot, ...]
    merged_peak_bytes_by_scope: dict[str, int]
    lowered_passes_observed: tuple[str, ...]
    lowering_error: str | None = None


def _map_tilelang_scope(raw_scope: str) -> str | None:
    return _TILELANG_SCOPE_MAP.get(raw_scope.strip())


def _same_tirx_var(lhs: object, rhs: object) -> bool:
    return (hasattr(lhs, "same_as") and bool(lhs.same_as(rhs))) or lhs is rhs


def _op_name_str(op: object) -> str:
    return str(getattr(op, "name", op))


def _unwrapped_extern_symbol(call: object) -> str | None:
    op_name = _op_name_str(getattr(call, "op", ""))
    if op_name.rsplit(".", 1)[-1].lower() not in _EXTERN_CALL_TAILS:
        return None
    args = getattr(call, "args", ())
    if not args:
        return None
    first = args[0]
    if isinstance(first, str) and first:
        return first
    val = getattr(first, "value", None)
    return val if isinstance(val, str) and val else None


def _effective_tirx_call_name(call: object) -> str:
    sym = _unwrapped_extern_symbol(call)
    return sym if sym is not None else _op_name_str(getattr(call, "op", ""))


def _effective_tirx_call_args(call: object) -> list[object]:
    args = list(getattr(call, "args", ()))
    return args[1:] if (_unwrapped_extern_symbol(call) is not None and args) else args


def _static_tirx_int(expr) -> int | None:
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    val = getattr(expr, "value", None)
    return int(val) if isinstance(val, int) and not isinstance(val, bool) else None


def _static_shape_ints(shape: object) -> tuple[int, ...] | None:
    dims: list[int] = []
    for dim in shape:
        if (sv := _static_tirx_int(dim)) is None:
            return None
        dims.append(sv)
    return tuple(dims)


def _dtype_bytes(dtype_obj: object, DataTypeCls) -> float:
    dt = DataTypeCls(str(dtype_obj))
    return (dt.bits * dt.lanes) / 8.0


def _buffer_bytes(shape: tuple[int, ...] | None, dtype_obj: object, DataTypeCls) -> int:
    if not shape:
        return 0
    return int(math.ceil(math.prod(shape) * _dtype_bytes(dtype_obj, DataTypeCls)))


def _is_lowbit_dtype_name(dtype_str: str) -> bool:
    base = re.sub(r"x\d+$", "", dtype_str.strip().lower())
    return base in ("int8", "uint8", "i8", "u8", "int4", "uint4", "i4", "u4") or base.startswith(
        ("fp8", "float8", "f8", "fp4", "float4", "f4")
    )


def _compute_wave_occupancy(
    grid_blocks: int, spaces: AscendSpaceFacts, pipe_cycles: dict[str, int], aiv_split_factor: int
) -> float:
    uses_aic = pipe_cycles.get("AIC.PIPE_M", 0) > 0 or aiv_split_factor >= 2
    core_pool = spaces.num_aic_cores if uses_aic else spaces.num_aiv_cores
    return round(grid_blocks / float(max(1, core_pool)), 4)


def _select_dominant_contig_bytes(candidates: Sequence[tuple[int | None, int, int, int, int]]) -> int:
    non_scalars = [item for item in candidates if item[0] is None or item[0] != 1]
    if not non_scalars:
        return 0
    payloads = [item for item in non_scalars if item[1] > 1]
    return max(payloads or non_scalars, key=lambda item: (item[2], item[3], item[4]))[4]


def _attrs_contain_flag(attrs: object, flag: str) -> bool:
    if attrs is None:
        return False
    if isinstance(attrs, (str, bytes, bytearray)) or "String" in type(attrs).__name__:
        return flag in str(attrs)
    if hasattr(attrs, "items"):
        return any((flag in str(k)) or _attrs_contain_flag(v, flag) for k, v in attrs.items())
    if isinstance(attrs, (tuple, list)) or type(attrs).__name__ == "Array":
        return any(_attrs_contain_flag(item, flag) for item in attrs)
    return flag in str(attrs)


def _is_tilelang_exp_op(op_name: str) -> bool:
    low = op_name.lower()
    return "vexp" in low or "exp2" in low or low == "exp" or low.endswith(".exp")


def _is_reduce_or_sfu_call(op_name: str) -> bool:
    low = op_name.lower()
    return low == "exp" or any(
        t in low for t in ("vcadd", "vreduce", "reduce_", "vexp", "exp2", ".exp", "vsqrt", "rsqrt", "vdiv")
    )


def _is_sync_or_barrier_intrinsic(op_name: str) -> bool:
    low = op_name.lower()
    return any(
        t in low for t in ("barrier", "pipe_barrier", "cross_flag", "cross_core", "set_flag", "wait_flag", "mbarrier", "sync")
    )


def _is_disallowed_tirx_intrinsic(op_name: str, *, allow_sync: bool = False) -> bool:
    low = op_name.lower()
    if allow_sync and _is_sync_or_barrier_intrinsic(low):
        return False
    if any(
        t in low
        for t in (
            "copy", "dual_copy", "dma", "region", "access_ptr", "gemm", "mmad", "ascend_",
            "simd", "simt", "vld", "vsts", "pset", "vadd", "vmul", "vsub", "vexp", "vcadd",
            "vreduce", "reduce_", "barrier", "pipe_barrier", "cross_flag", "cross_core",
            "set_flag", "wait_flag", "mbarrier", "sync",
        )
    ):
        return True
    return bool(re.search(r"(?:^|[._])(?:mad|vec)(?:$|[._])", low))


def _is_tirx_math_call(op_name: str) -> bool:
    low = op_name.lower()
    if _is_disallowed_tirx_intrinsic(low):
        return False
    return low.startswith("tir.") or any(
        t in low
        for t in (
            "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan", "tanh", "erf", "abs", "fabs",
            "floor", "ceil", "round", "pow", "fma", "sigmoid", "max", "min", "div", "mod", "select", "cast",
        )
    )


def _collect_tilelang_expr_rw(
    expr,
    *,
    tirx,
    sfu_out: list[bool] | None = None,
) -> tuple[list[tuple[str, str, str]], bool, list[str], list[str]]:
    events: list[tuple[str, str, str]] = []
    has_exp = False
    ub_loads: list[str] = []
    ub_stores: list[str] = []

    def _absorb(sub) -> None:
        nonlocal has_exp
        sev, sexp, sul, sus = _collect_tilelang_expr_rw(sub, tirx=tirx, sfu_out=sfu_out)
        events.extend(sev)
        has_exp = has_exp or sexp
        ub_loads.extend(sul)
        ub_stores.extend(sus)

    def _visit_indices(indices) -> None:
        for idx in indices:
            if isinstance(idx, (tirx.And, tirx.Or, tirx.Not)):
                raise ValueError(
                    "Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / tvm.tirx.any instead"
                )
            _absorb(idx)

    def _pre(node):
        nonlocal has_exp
        if isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            call_args = _effective_tirx_call_args(node)
            if _is_tilelang_exp_op(op_name):
                has_exp = True
            if sfu_out is not None and _is_reduce_or_sfu_call(op_name):
                sfu_out[0] = True
            if ("vld" in op_name or "vsts" in op_name) and call_args:
                is_store = "vsts" in op_name
                sev, sexp, sul, sus = _collect_tilelang_expr_rw(call_args[0], tirx=tirx, sfu_out=sfu_out)
                events.extend(sev)
                has_exp = has_exp or sexp
                ub_loads.extend(sul)
                ub_stores.extend(sus)
                for bname, scope, _rw in sev:
                    if scope == "UB":
                        (ub_stores if is_store else ub_loads).append(bname)
                for rest in call_args[1:]:
                    _absorb(rest)
                return False
            if ("region" in op_name or "access_ptr" in op_name) and node.args and isinstance(node.args[0], tirx.BufferLoad):
                rw_idx = 1 if "region" in op_name else 2
                if len(node.args) > rw_idx:
                    bl = node.args[0]
                    _visit_indices(bl.indices)
                    scope = _map_tilelang_scope(bl.buffer.scope())
                    rw = _static_tirx_int(node.args[rw_idx]) or 1
                    if scope is not None:
                        if rw & 1:
                            events.append((bl.buffer.name, scope, "read"))
                        if rw & 2:
                            events.append((bl.buffer.name, scope, "write"))
                    return False
            if "copy" in op_name and len(call_args) >= 2:
                for arg, default_rw in ((call_args[0], "read"), (call_args[1], "write")):
                    if isinstance(arg, (tirx.BufferRegion, tirx.BufferLoad)):
                        if isinstance(arg, tirx.BufferLoad):
                            _visit_indices(arg.indices)
                        if (scope := _map_tilelang_scope(arg.buffer.scope())) is not None:
                            events.append((arg.buffer.name, scope, default_rw))
                    else:
                        _absorb(arg)
                return False
        elif isinstance(node, tirx.BufferLoad):
            _visit_indices(node.indices)
            if (scope := _map_tilelang_scope(node.buffer.scope())) is not None:
                events.append((node.buffer.name, scope, "read"))
                if scope == "UB":
                    ub_loads.append(node.buffer.name)
            return False
        elif isinstance(node, tirx.BufferRegion):
            if (scope := _map_tilelang_scope(node.buffer.scope())) is not None:
                events.append((node.buffer.name, scope, "read"))
            return False
        return True

    tirx.stmt_functor.pre_order_visit(expr, _pre)
    return events, has_exp, ub_loads, ub_stores


def _has_loop_carried_accumulator(events: Sequence[tuple[str, object, str]], local_keys: set[str]) -> bool:
    written_so_far: set[str] = set()
    read_before_write: set[str] = set()
    written_in_loop: set[str] = set()
    for buf_key, _scope, rw in events:
        if buf_key in local_keys:
            continue
        if rw == "read" and buf_key not in written_so_far:
            read_before_write.add(buf_key)
        elif rw == "write":
            written_so_far.add(buf_key)
            written_in_loop.add(buf_key)
    return bool(read_before_write & written_in_loop)


def _free_tirx_vars(expr, *, tirx) -> list[object]:
    if hasattr(tirx, "analysis") and hasattr(tirx.analysis, "undefined_vars"):
        try:
            return list(tirx.analysis.undefined_vars(expr))
        except Exception:
            pass
    found: list[object] = []
    seen_ids: set[int] = set()
    tirx.stmt_functor.post_order_visit(
        expr,
        lambda n: (found.append(n), seen_ids.add(id(n))) if isinstance(n, tirx.Var) and id(n) not in seen_ids else None,
    )
    return found


def _substitute_let_env(expr, let_env: Mapping[object, object], *, tirx):
    cur_expr = expr
    for _ in range(16):
        if not let_env:
            break
        expand = {v: let_env[v] for v in _free_tirx_vars(cur_expr, tirx=tirx) if v in let_env}
        if not expand:
            break
        nxt = tirx.stmt_functor.substitute(cur_expr, expand)
        if hasattr(nxt, "same_as") and nxt.same_as(cur_expr):
            break
        cur_expr = nxt
    return cur_expr


def _build_free_var_int_map(
    expr,
    var_bindings: Mapping[str, int] | None,
    var_obj_bindings: Sequence[tuple[object, int]] = (),
    *,
    tirx,
    exclude_var: object | None = None,
) -> dict[object, object]:
    vmap: dict[object, object] = {}
    if not var_bindings and not var_obj_bindings:
        return vmap
    lower_counts: dict[str, list[int]] = {}
    for rk, rv in (var_bindings or {}).items():
        if isinstance(rv, int) and not isinstance(rv, bool):
            lower_counts.setdefault(str(rk).lower(), []).append(int(rv))

    for fv in _free_tirx_vars(expr, tirx=tirx):
        if exclude_var is not None and _same_tirx_var(fv, exclude_var):
            continue
        bound = next((int(vv) for vo, vv in var_obj_bindings if _same_tirx_var(fv, vo)), None)
        exact = str(getattr(fv, "name", "") or "")
        hint = str(getattr(fv, "name_hint", "") or "")
        if bound is None and var_bindings:
            for cn in (exact, hint):
                if cn and cn in var_bindings and isinstance(rv := var_bindings[cn], int) and not isinstance(rv, bool):
                    bound = int(rv)
                    break
        if bound is None and (lookup := (exact or hint).lower()):
            if len(matches := lower_counts.get(lookup, [])) == 1:
                bound = matches[0]
        if bound is not None:
            vmap[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), bound)
    return vmap


def _lookup_stem_binding(cname: str, eff: Mapping[str, int], lower_eff: Mapping[str, int]) -> int | None:
    if cname and cname in eff:
        return eff[cname]
    lc = cname.lower()
    if lc and lc in lower_eff:
        return lower_eff[lc]
    for suffix in ("_dim", "_size", "_len", "_count", "_rows", "_cols"):
        if lc.endswith(suffix) and (stem := lc[: -len(suffix)]) in lower_eff:
            return lower_eff[stem]
        if lc and f"{lc}{suffix}" in lower_eff:
            return lower_eff[f"{lc}{suffix}"]
    for group in _DIM_ALIAS_GROUPS:
        if lc in group:
            for alias in group:
                if alias in lower_eff:
                    return lower_eff[alias]
    return None


def _match_binding_for_var(var, eff: Mapping[str, int], lower_eff: Mapping[str, int]) -> int | None:
    for cname in (str(getattr(var, "name", "") or ""), str(getattr(var, "name_hint", "") or "")):
        if (direct := _lookup_stem_binding(cname, eff, lower_eff)) is not None:
            return direct
        lc = cname.lower()
        pad_stem = lc[: lc.find("_pad")] if "_pad" in lc else (lc[4:] if lc.startswith("pad_") else "")
        if pad_stem and (base_val := _lookup_stem_binding(pad_stem, eff, lower_eff)) is not None and base_val > 0:
            return ((int(base_val) + 31) // 32) * 32
    return None


def _augment_var_bindings_from_buffers(
    entry_fn,
    var_bindings: Mapping[str, int] | None,
    *,
    tirx,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, int], list[tuple[object, int]]]:
    if not var_bindings and not ordered_tensor_shapes:
        return {}, []
    eff = {str(k): int(v) for k, v in (var_bindings or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
    lower_eff = {k.lower(): v for k, v in eff.items()}
    obj_bindings: list[tuple[object, int]] = []
    seen_ids: set[int] = set()

    def _bind_var(var, val: int) -> None:
        if id(var) not in seen_ids:
            seen_ids.add(id(var))
            obj_bindings.append((var, int(val)))
        for cn in (str(getattr(var, "name", "") or ""), str(getattr(var, "name_hint", "") or "")):
            if cn and cn not in eff:
                eff[cn] = int(val)
                lower_eff.setdefault(cn.lower(), int(val))

    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    tensor_idx = 0
    for param in getattr(entry_fn, "params", ()):
        buf = buffer_map.get(param)
        if buf is None:
            if isinstance(param, tirx.Var) and (m := _match_binding_for_var(param, eff, lower_eff)) is not None:
                _bind_var(param, m)
            continue
        shape = list(getattr(buf, "shape", ()))
        for dim_expr in (*shape, *(getattr(buf, "strides", ()) or ())):
            for fv in _free_tirx_vars(dim_expr, tirx=tirx):
                if (m := _match_binding_for_var(fv, eff, lower_eff)) is not None:
                    _bind_var(fv, m)
        if ordered_tensor_shapes is not None and tensor_idx < len(ordered_tensor_shapes):
            pos_shape = ordered_tensor_shapes[tensor_idx]
            if len(shape) == len(pos_shape):
                for de, dv in zip(shape, pos_shape):
                    if isinstance(de, tirx.Var) and id(de) not in seen_ids and isinstance(dv, int) and not isinstance(dv, bool):
                        _bind_var(de, max(1, int(dv)))
        tensor_idx += 1
        if len(shape) == 2 and isinstance(shape[0], tirx.Var) and id(shape[0]) not in seen_ids and _static_tirx_int(shape[1]) is not None:
            if (rv := lower_eff.get("q_total_rows", lower_eff.get("m_rows", lower_eff.get("m")))) is not None:
                _bind_var(shape[0], rv)
        elif len(shape) == 4:
            if isinstance(shape[0], tirx.Var) and id(shape[0]) not in seen_ids and (bv := lower_eff.get("b_dim", lower_eff.get("b"))) is not None:
                _bind_var(shape[0], bv)
            if (kv := lower_eff.get("s_kv")) is not None:
                if isinstance(shape[1], tirx.Var) and id(shape[1]) not in seen_ids and _static_tirx_int(shape[2]) is not None:
                    _bind_var(shape[1], kv)
                elif isinstance(shape[2], tirx.Var) and id(shape[2]) not in seen_ids and _static_tirx_int(shape[1]) is not None:
                    _bind_var(shape[2], kv)

    if not any(a in lower_eff for a in _NUMEL_ALIASES):
        synth_numel = 0
        for param in getattr(entry_fn, "params", ()):
            buf = buffer_map.get(param)
            if not buf or not (bshape := list(getattr(buf, "shape", ()) or ())):
                continue
            vals: list[int] = []
            for de in bshape:
                sv = _static_tirx_int(de)
                mv = sv if (sv is not None and sv > 0) else (_match_binding_for_var(de, eff, lower_eff) if isinstance(de, tirx.Var) else None)
                if mv is None or mv <= 0:
                    vals = []
                    break
                vals.append(int(mv))
            if vals:
                synth_numel = max(synth_numel, math.prod(vals))
        if synth_numel <= 0:
            b_v, h_v, s_v, m_v, n_v, d_v, k_v = (
                _lookup_stem_binding(x, eff, lower_eff) for x in ("b", "h", "s", "m", "n", "d", "k")
            )
            row_axes, has_axis = 1, False
            if m_v and m_v > 0:
                row_axes = m_v if (b_v and s_v and m_v >= b_v * s_v) else (b_v or 1) * (h_v or 1) * m_v
                has_axis = True
            elif s_v and s_v > 0:
                row_axes, has_axis = (b_v or 1) * (h_v or 1) * s_v, True
            elif b_v and b_v > 0:
                row_axes, has_axis = b_v * (h_v or 1), True
            col_v = n_v or d_v or k_v
            synth_numel = row_axes * col_v if (col_v and col_v > 0) else (row_axes if has_axis else 0)
        if synth_numel > 0:
            for alias in _NUMEL_ALIASES:
                eff.setdefault(alias, int(synth_numel))
                lower_eff.setdefault(alias, int(synth_numel))

    return eff, obj_bindings


def _resolve_symbolic_grid_extent(
    expr,
    let_env: Mapping[object, object],
    var_bindings: Mapping[str, int] | None,
    *,
    tirx,
    tvm_mod: object | None,
    var_obj_bindings: Sequence[tuple[object, int]] = (),
) -> tuple[int | None, bool]:
    if (static_val := _static_tirx_int(expr)) is not None:
        return static_val, False
    cur = _substitute_let_env(expr, let_env, tirx=tirx)
    if vmap := _build_free_var_int_map(cur, var_bindings, var_obj_bindings, tirx=tirx):
        cur = tirx.stmt_functor.substitute(cur, vmap)
    if tvm_mod is not None and hasattr(tvm_mod, "arith") and hasattr(tvm_mod.arith, "Analyzer"):
        try:
            cur = tvm_mod.arith.Analyzer().simplify(cur)
        except Exception:
            pass
    resolved = _static_tirx_int(cur)
    return (resolved, False) if resolved is not None else (None, True)


def _upper_bound_tirx_int(expr, *, tirx, let_env: Mapping[object, object] | None = None, _depth: int = 0) -> int | None:
    if expr is None or _depth > 12:
        return None
    cur = _substitute_let_env(expr, let_env, tirx=tirx) if let_env else expr
    if (sv := _static_tirx_int(cur)) is not None:
        return sv
    if isinstance(cur, (tirx.Min, tirx.Mul, tirx.Add)):
        lhs = _upper_bound_tirx_int(cur.a, tirx=tirx, _depth=_depth + 1)
        rhs = _upper_bound_tirx_int(cur.b, tirx=tirx, _depth=_depth + 1)
        if isinstance(cur, tirx.Min):
            if lhs and lhs > 0 and rhs and rhs > 0:
                return min(lhs, rhs)
            return lhs if (lhs and lhs > 0) else (rhs if (rhs and rhs > 0) else None)
        if isinstance(cur, tirx.Mul):
            return lhs * rhs if (lhs and lhs > 0 and rhs and rhs > 0) else None
        return lhs + rhs if (lhs is not None and lhs >= 0 and rhs is not None and rhs >= 0) else None
    return None


def _region_bytes(
    arg, tirx, DataTypeCls, *, let_env: Mapping[object, object] | None = None
) -> tuple[str | None, int, int]:
    buf = None
    raw_extents: list[object] = []
    if isinstance(arg, tirx.Call) and "region" in str(getattr(arg.op, "name", arg.op)):
        if len(arg.args) >= 2 and isinstance(arg.args[0], tirx.BufferLoad):
            buf, raw_extents = arg.args[0].buffer, list(arg.args[2:])
    elif isinstance(arg, tirx.BufferRegion):
        buf, raw_extents = arg.buffer, [r.extent for r in arg.region]
    elif isinstance(arg, tirx.BufferLoad):
        buf = arg.buffer
        eb = int(math.ceil(_dtype_bytes(buf.dtype, DataTypeCls)))
        return _map_tilelang_scope(buf.scope()), eb, eb
    if buf is None:
        return None, 0, 0
    scope = _map_tilelang_scope(buf.scope())
    if not raw_extents:
        return scope, 0, 0
    elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
    resolved = [_upper_bound_tirx_int(e, tirx=tirx, let_env=let_env) for e in raw_extents]
    last = resolved[-1]
    inner = int(math.ceil(last * elem_bytes)) if (last is not None and last > 0) else 0
    if any(v is None or v <= 0 for v in resolved):
        return scope, 0, inner
    return scope, int(math.ceil(math.prod(int(v) for v in resolved if v is not None) * elem_bytes)), inner


def _access_ptr_or_region_bytes(arg, *, tirx, DataTypeCls, let_env: Mapping[object, object] | None = None) -> int:
    if isinstance(arg, tirx.Call) and "access_ptr" in str(getattr(arg.op, "name", arg.op)).lower() and len(arg.args) >= 4:
        dtype_obj, ext = getattr(arg.args[0], "dtype", None), _static_tirx_int(arg.args[3])
        if dtype_obj is not None and ext is not None and ext > 0:
            return int(math.ceil(ext * _dtype_bytes(dtype_obj, DataTypeCls)))
    return _region_bytes(arg, tirx, DataTypeCls, let_env=let_env)[1]


def _contains_buffer_load(expr, tirx) -> bool:
    found = False

    def _visit(node) -> None:
        nonlocal found
        if not found and isinstance(node, tirx.BufferLoad):
            found = True

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found


def _is_sid_expr(expr, sid_vars: set[object], tirx) -> bool:
    if isinstance(expr, tirx.Var):
        name = str(getattr(expr, "name_hint", "")).lower()
        return any(expr.same_as(v) for v in sid_vars) or "subblock" in name or name == "sid"
    if isinstance(expr, tirx.Call):
        op_name = _effective_tirx_call_name(expr).lower()
        return "subblock" in op_name or "get_subblock_idx" in op_name
    return False


def _expr_references_sid(expr, sid_vars: set[object], tirx) -> bool:
    found = False

    def _visit(node) -> None:
        nonlocal found
        if not found and _is_sid_expr(node, sid_vars, tirx):
            found = True

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found


def _sid_guard_branches(cond, sid_vars: set[object], tirx, aiv_split_factor: int) -> tuple[str | None, str | None]:
    if isinstance(cond, (tirx.EQ, tirx.NE)):
        for lhs, rhs in ((cond.a, cond.b), (cond.b, cond.a)):
            if _is_sid_expr(lhs, sid_vars, tirx) and (val := _static_tirx_int(rhs)) is not None:
                matched = "aiv1" if val == 1 else "aiv0"
                other = ("aiv0" if val == 1 else "aiv1") if aiv_split_factor >= 2 else "aiv0"
                return (matched, other) if isinstance(cond, tirx.EQ) else (other, matched)
    if _expr_references_sid(cond, sid_vars, tirx):
        return "aiv0", ("aiv1" if aiv_split_factor >= 2 else "aiv0")
    return None, None


def _burst_scaled_copy_cycles(xfer_bytes: int, spaces: AscendSpaceFacts | None) -> int:
    base = max(1, (xfer_bytes + 63) // 64)
    if spaces is None or xfer_bytes <= 0:
        return base
    return int(math.ceil(float(base) / max(0.15, float(spaces.dma_burst_efficiency(xfer_bytes)))))


def _dav3510_macs_and_k0(dtype_str: str, spaces: AscendSpaceFacts | None) -> tuple[int, int]:
    base = re.sub(r"x\d+$", "", str(dtype_str).strip().lower())
    if base in ("fp4", "f4", "mxfp4", "f4e2m1", "f4e1m2", "float4_e2m1fn", "int4", "i4", "u4"):
        key, k0, fb = "f4e2m1", 64, 16384
    elif base in ("fp8", "f8", "fp8e4m3", "fp8e5m2", "float8_e4m3fn", "float8_e5m2", "hif8", "hifp8", "int8", "uint8", "i8", "u8"):
        key = "hif8" if "hif" in base else ("i8" if ("i8" in base or "int8" in base) else "fp8e4m3")
        k0, fb = 32, 8192
    elif base in ("hf32", "tf32"):
        key, k0, fb = "hf32", 8, 2048
    elif base in ("f32", "fp32", "float32"):
        key, k0, fb = "f32", 8, 256
    else:
        key, k0, fb = "bf16", 16, 4096
    if spaces is not None and hasattr(spaces, "cube_macs_per_cycle"):
        try:
            return int(spaces.cube_macs_per_cycle(key)), k0
        except ValueError:
            pass
    return fb, k0


def _arg_shape_and_dtype(arg, tirx) -> tuple[tuple[int, ...] | None, str | None]:
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op))
        if ("region" in op_name or "access_ptr" in op_name) and arg.args and isinstance(arg.args[0], tirx.BufferLoad):
            buf = arg.args[0].buffer
            if "region" in op_name and len(arg.args) >= 3 and (ext := _static_shape_ints(arg.args[2:])) is not None:
                return ext, str(buf.dtype)
            return _static_shape_ints(buf.shape), str(buf.dtype)
    if isinstance(arg, tirx.BufferRegion):
        ext = _static_shape_ints([r.extent for r in arg.region])
        return (ext if ext is not None else _static_shape_ints(arg.buffer.shape)), str(arg.buffer.dtype)
    if isinstance(arg, tirx.BufferLoad):
        return _static_shape_ints(arg.buffer.shape), str(arg.buffer.dtype)
    return None, None


def _estimate_dav3510_cube_cycles(args: Sequence[object], spaces: AscendSpaceFacts | None, tirx) -> int:
    ext_a, dt_a = _arg_shape_and_dtype(args[0], tirx) if len(args) > 0 else (None, None)
    ext_b, dt_b = _arg_shape_and_dtype(args[1], tirx) if len(args) > 1 else (None, None)
    ext_c, dt_c = _arg_shape_and_dtype(args[2], tirx) if len(args) > 2 else (None, None)
    ta = _static_tirx_int(args[3]) if len(args) > 3 else 0
    tb = _static_tirx_int(args[4]) if len(args) > 4 else 0
    m = _static_tirx_int(args[5]) if len(args) > 5 else None
    n = _static_tirx_int(args[6]) if len(args) > 6 else None
    k = _static_tirx_int(args[7]) if len(args) > 7 else None
    if ext_a and len(ext_a) >= 2:
        m = (ext_a[-1] if ta else ext_a[-2]) if m is None else m
        k = (ext_a[-2] if ta else ext_a[-1]) if k is None else k
    if ext_b and len(ext_b) >= 2:
        n = (ext_b[-2] if tb else ext_b[-1]) if n is None else n
        k = (ext_b[-1] if tb else ext_b[-2]) if k is None else k
    if ext_c and len(ext_c) >= 2:
        m = ext_c[-2] if m is None else m
        n = ext_c[-1] if n is None else n
    if m and n and k and m > 0 and n > 0 and k > 0:
        macs, k0 = _dav3510_macs_and_k0(dt_a or dt_b or dt_c or "bf16", spaces)
        mp, np_, kp = ((m + 15) // 16) * 16, ((n + 15) // 16) * 16, ((k + k0 - 1) // k0) * k0
        cyc = max(1, math.ceil((mp * np_ * kp) / float(max(1, macs))))
        return cyc + (6 if ((m + 15) // 16) * ((n + 15) // 16) < 10 else 0)
    return 4


def _estimate_dav3510_vec_cycles(op_name: str, args: Sequence[object], tirx, DataTypeCls) -> int:
    max_b = 0
    for arg in args:
        total_b = _region_bytes(arg, tirx, DataTypeCls)[1]
        if total_b > max_b:
            max_b = total_b
        elif total_b == 0 and isinstance(arg, tirx.Var):
            ds = str(getattr(arg, "dtype", ""))
            if ds and ds != "handle":
                try:
                    dt = DataTypeCls(ds)
                    max_b = max(max_b, int(math.ceil((dt.bits * dt.lanes) / 8.0)))
                except Exception:
                    pass
    base = max(1, math.ceil((max_b if max_b > 0 else 256) / 256.0))
    return base * (4 if any(t in op_name.lower() for t in ("exp", "log", "div", "sqrt", "rsqrt")) else 1)


def _tilelang_arg_buf(arg, tirx, DataTypeCls) -> tuple[str | None, str | None, int]:
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op))
        if ("region" in op_name or "access_ptr" in op_name) and arg.args and isinstance(arg.args[0], tirx.BufferLoad):
            buf = arg.args[0].buffer
            scope, full_b = _map_tilelang_scope(buf.scope()), _buffer_bytes(_static_shape_ints(buf.shape), buf.dtype, DataTypeCls)
            if "region" in op_name and len(arg.args) >= 3:
                part_b = _buffer_bytes(_static_shape_ints(arg.args[2:]), buf.dtype, DataTypeCls)
                return buf.name, scope, part_b if part_b > 0 else full_b
            return buf.name, scope, full_b
    if isinstance(arg, tirx.BufferRegion):
        buf = arg.buffer
        part_b = _buffer_bytes(_static_shape_ints([r.extent for r in arg.region]), buf.dtype, DataTypeCls)
        full_b = _buffer_bytes(_static_shape_ints(buf.shape), buf.dtype, DataTypeCls)
        return buf.name, _map_tilelang_scope(buf.scope()), part_b if part_b > 0 else full_b
    if isinstance(arg, tirx.BufferLoad):
        buf = arg.buffer
        return buf.name, _map_tilelang_scope(buf.scope()), _buffer_bytes(_static_shape_ints(buf.shape), buf.dtype, DataTypeCls)
    return None, None, 0


def _aiv_mask(pipe: str, *, aiv_split_factor: int, guarded_aiv: str | None) -> NodeResourceMask:
    if aiv_split_factor >= 2 and guarded_aiv is None:
        return NodeResourceMask.for_dual_aiv(pipe)
    return NodeResourceMask.for_core(guarded_aiv or "aiv0", pipe)


def _copy_resource_mask(src_s: str, dst_s: str, *, aiv_split_factor: int, guarded_aiv: str | None) -> NodeResourceMask:
    if src_s == "GM" and dst_s == "UB":
        return _aiv_mask("MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" and dst_s == "GM":
        return _aiv_mask("MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "GM" and dst_s == "L1":
        return NodeResourceMask.for_aic("MTE2")
    if src_s == "L1" and dst_s in ("L0A", "L0B"):
        return NodeResourceMask.for_aic("MTE1")
    if src_s == "L0C" and dst_s in ("GM", "L1"):
        return NodeResourceMask.for_aic("FIXPIPE")
    if src_s == "L0C" and dst_s == "UB":
        return NodeResourceMask.for_aic("FIXPIPE") | _aiv_mask("MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" and dst_s == "L1":
        return NodeResourceMask.for_aic("MTE1") | _aiv_mask("MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" or dst_s == "UB":
        return _aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    return NodeResourceMask.for_aic("MTE1")


def _extract_vf_accesses(stmt, *, tirx, DataTypeCls) -> tuple[list[str], list[str], int]:
    reads: list[str] = []
    writes: list[str] = []
    op_count = 0

    def _add(target: list[str], name: str | None, scope: str | None) -> None:
        if name is not None and scope is not None and scope != "GM" and name not in target:
            target.append(name)

    def _visit(node) -> None:
        nonlocal op_count
        if isinstance(node, tirx.BufferStore):
            op_count += 1
            _add(writes, node.buffer.name, _map_tilelang_scope(node.buffer.scope()))
        elif isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            if "simd" in op_name or "vec" in op_name:
                op_count += 1
            if ("access_ptr" in op_name and len(node.args) >= 3) or ("region" in op_name and len(node.args) >= 2):
                name, scope, _ = _tilelang_arg_buf(node, tirx, DataTypeCls)
                rw = _static_tirx_int(node.args[2 if "access_ptr" in op_name else 1]) or 1
                if rw & 1:
                    _add(reads, name, scope)
                if rw & 2:
                    _add(writes, name, scope)
        elif isinstance(node, tirx.BufferLoad):
            _add(reads, node.buffer.name, _map_tilelang_scope(node.buffer.scope()))

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return reads, writes, max(1, op_count)


@dataclass
class _SubtreeAcc:
    events: list[tuple[str, str, str]] = field(default_factory=list)
    has_exp: bool = False
    ub_loads: set[str] = field(default_factory=set)
    ub_stores: set[str] = field(default_factory=set)
    local_bufs: set[str] = field(default_factory=set)
    has_loop: bool = False
    has_vec_sub: bool = False
    has_disallowed_strict: bool = False
    has_disallowed_sync_ok: bool = False
    has_ub_arith_store: bool = False
    scope_direct_loads: set[str] = field(default_factory=set)
    scope_direct_stores: set[str] = field(default_factory=set)
    scope_direct_sfu: bool = False
    child_scopes: list[_ScopeRec] = field(default_factory=list)

    def _merge_base(self, child: _SubtreeAcc) -> None:
        self.events.extend(child.events)
        self.has_exp = self.has_exp or child.has_exp
        self.ub_loads.update(child.ub_loads)
        self.ub_stores.update(child.ub_stores)
        self.local_bufs.update(child.local_bufs)
        self.has_loop = self.has_loop or child.has_loop
        self.has_vec_sub = self.has_vec_sub or child.has_vec_sub
        self.has_disallowed_strict = self.has_disallowed_strict or child.has_disallowed_strict
        self.has_disallowed_sync_ok = self.has_disallowed_sync_ok or child.has_disallowed_sync_ok
        self.has_ub_arith_store = self.has_ub_arith_store or child.has_ub_arith_store

    def merge_non_scope(self, child: _SubtreeAcc) -> None:
        self._merge_base(child)
        self.scope_direct_loads.update(child.scope_direct_loads)
        self.scope_direct_stores.update(child.scope_direct_stores)
        self.scope_direct_sfu = self.scope_direct_sfu or child.scope_direct_sfu
        self.child_scopes.extend(child.child_scopes)

    def merge_from_child_scope(self, child: _SubtreeAcc, rec: _ScopeRec) -> None:
        self._merge_base(child)
        self.child_scopes.append(rec)


@dataclass
class _ScopeRec:
    direct_loads: set[str]
    direct_stores: set[str]
    direct_sfu: bool
    children: list[_ScopeRec]


@dataclass(frozen=True)
class _WalkCtx:
    sid_vars: frozenset[object] = frozenset()
    guarded_aiv: str | None = None
    let_env: Mapping[object, object] = field(default_factory=dict)
    in_loop: bool = False
    in_pipelined: bool = False
    in_vec_scope: bool = False
    in_block_scope: bool = False
    found_pipe_loop: bool = False
    loop_vars: tuple[object, ...] = ()

    def with_(self, **kw: object) -> _WalkCtx:
        return _WalkCtx(**{**{k: getattr(self, k) for k in self.__dataclass_fields__}, **kw})


@dataclass
class _UnifiedScanResult:
    pipe_cycles: dict[str, int]
    mte2_inflight: int
    padding_bytes: int
    opaque_regions: tuple[str, ...]
    has_simtvf: bool
    illegal_dma: bool
    aiv_split_factor: int
    explicit_snapshots: tuple[BufferSnapshot, ...]
    alloc_snapshots: tuple[BufferSnapshot, ...]
    requested_multibuffer: set[str]
    buf_meta: dict[str, tuple[str, int]]
    ub_buf_names: set[str]
    ub_static_bytes: dict[str, int]
    l1_static_bytes: dict[str, int]
    gm_static_bytes: dict[str, int]
    gm_scalar_names: set[str]
    dominant_contig_bytes: int
    effective_cube_m: int
    has_indirect_gather: bool
    has_lowbit_quant: bool
    has_online_rescale_loop: bool
    in_loop_ub_reload_stores: int
    gm_roundtrip_bytes: int
    min_gm_dma_bytes: int
    dma_burst_efficiency_ratio: float
    seen_in_loop_gm_dma: bool
    in_loop_pipe_all_barriers: int
    uses_dynamic_offset_pipelining: bool
    has_unlimit_memory_attr: bool
    has_coupled_reduce_sfu_accum: bool
    ub_bank_conflict_risk: float
    l1_bank_conflict_risk: float
    ub_scalar_compute_loops: int
    grid_blocks: int
    active_grid_blocks: int
    grid_is_symbolic: bool
    uses_unreserved_ubuf: bool
    pipelined_loops: list[tuple[object, set[object], str | None]]


def _is_pipelined_tirx_for(stmt, tirx) -> bool:
    if not isinstance(stmt, tirx.For):
        return False
    kind_val = int(stmt.kind) if hasattr(stmt.kind, "value") else stmt.kind
    return "num_stages" in (getattr(stmt, "annotations", None) or {}) or kind_val == 5


def _eval_coupled_scope(rec: _ScopeRec, wide_ub: set[str]) -> tuple[bool, bool, bool]:
    direct_rmw = bool((rec.direct_loads & wide_ub) & (rec.direct_stores & wide_ub))
    c_sfu = c_rmw = coupled = False
    for ch in rec.children:
        sf, rm, cp = _eval_coupled_scope(ch, wide_ub)
        c_sfu, c_rmw, coupled = c_sfu or sf, c_rmw or rm, coupled or cp
    if (rec.direct_sfu and (direct_rmw or c_rmw)) or (direct_rmw and c_sfu):
        coupled = True
    return (rec.direct_sfu or c_sfu), (direct_rmw or c_rmw), coupled


def _scan_primfunc_unified(
    entry_fn, mod_in=None, *, spaces: AscendSpaceFacts | None = None, tirx, DataTypeCls=None,
    aiv_split_factor: int | None = None, min_aiv_split: int = 1, unreserved_ubuf: bool = False,
    var_bindings: Mapping[str, int] | None = None, tvm_mod: object | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None, post_lower: bool = False,
    ub_buf_override: set[str] | None = None, allow_sync_in_scalar_loops: bool = False,
) -> _UnifiedScanResult:
    """Single-pass TIR statement walker collecting pipes, buffers, sync, DMA, and grid stats."""
    flag = "--cce-disable-vf-stack-reserved-ubuf"
    fn_attrs, mod_attrs = getattr(entry_fn, "attrs", None), getattr(mod_in, "attrs", None)
    effective_unreserved = bool(unreserved_ubuf or _attrs_contain_flag(fn_attrs, flag) or _attrs_contain_flag(mod_attrs, flag))
    has_unlimit_mem = _attrs_contain_flag(fn_attrs, "unlimit_memory") or _attrs_contain_flag(mod_attrs, "unlimit_memory")
    has_l1_half = _attrs_contain_flag(fn_attrs, "half_l1") or _attrs_contain_flag(mod_attrs, "half_l1")

    eff_bindings, var_obj_bindings = (
        _augment_var_bindings_from_buffers(entry_fn, var_bindings, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes)
        if (not post_lower and (var_bindings or ordered_tensor_shapes))
        else ({}, [])
    )

    detected_split, pipe_raw, opaque_regions = 1, {}, []
    has_simtvf = illegal_dma = has_lowbit_buf = has_indirect_gather = has_dyn_offset_pipe = False
    padding_bytes = max_mte2 = in_loop_pipe_all = ub_2d_total = ub_2d_conflict = ub_scalar_loops = 0
    any_symbolic_grid = seen_in_loop_gm_dma = has_online_rescale = False
    cube_m_values, block_dims, block_var_info, persistent_work_items, pipelined_loops = [], {}, [], [], []
    snap_entries, buf_meta, requested_mb = [], {}, set()
    ub_buf_names = set(ub_buf_override or ())
    ub_static_bytes, l1_static_bytes, gm_static_bytes = {}, {}, {}
    gm_scalar_names, contig_candidates, seen_gm_keys, seen_ub_bank = set(), [], set(), set()
    loop_dma_sizes, all_dma_sizes, gm_loop_scalars, ub_reload_store_bufs = [], [], set(), set()

    let_stmt_cls, bind_cls, alloc_cls = getattr(tirx, "LetStmt", None), getattr(tirx, "Bind", None), getattr(tirx, "Allocate", None)
    mod_classes = tuple(c for n in ("FloorMod", "Mod") if (c := getattr(tirx, n, None)) is not None)
    arith_types = tuple(c for n in ("Add", "Sub", "Mul", "Div", "FloorDiv", "Mod", "Max", "Min", "Select", "Cast") if (c := getattr(tirx, n, None)) is not None)

    def _add_opaque(tag: str) -> None:
        if tag not in opaque_regions:
            opaque_regions.append(tag)

    def _add_pipe(key: str, cyc: int) -> None:
        pipe_raw[key] = pipe_raw.get(key, 0) + max(1, cyc)

    def _emit_aiv(pipe_name: str, cyc: int, guarded_aiv: str | None) -> None:
        _add_pipe(f"{'UNSPLIT' if guarded_aiv is None else ('AIV1' if guarded_aiv == 'aiv1' else 'AIV0')}.{pipe_name}", cyc)

    def _record_snap(from_alloc: bool, var_obj, name: str, raw_scope: str, shape_obj, dtype_obj) -> None:
        if DataTypeCls is None or (scope := _map_tilelang_scope(raw_scope)) in (None, "GM") or (shape := _static_shape_ints(shape_obj)) is None:
            return
        snap = BufferSnapshot(name=str(name), scope=scope, bytes=_buffer_bytes(shape, dtype_obj, DataTypeCls), shape=shape, dtype=str(dtype_obj))
        for idx, (fa, pv, pk, ps) in enumerate(snap_entries):
            if fa == from_alloc and ((var_obj is not None and pv is not None and _same_tirx_var(var_obj, pv)) or (var_obj is None and pk == f"{scope}:{name}")):
                if snap.bytes > ps.bytes:
                    snap_entries[idx] = (fa, var_obj, pk, snap)
                return
        snap_entries.append((from_alloc, var_obj, f"{scope}:{name}", snap))

    def _register_buf(buf, *, from_alloc: bool) -> None:
        nonlocal has_lowbit_buf, ub_2d_total, ub_2d_conflict
        if _is_lowbit_dtype_name(str(buf.dtype)):
            has_lowbit_buf = True
        raw_scope = str(buf.scope()).strip()
        scope = _map_tilelang_scope(raw_scope)
        if scope == "UB" or raw_scope == "local.ub":
            ub_buf_names.add(buf.name)
        if DataTypeCls is None:
            return
        _record_snap(from_alloc, buf.data, buf.name, raw_scope, buf.shape, buf.dtype)
        static_shape = _static_shape_ints(buf.shape)
        nbytes = _buffer_bytes(static_shape, buf.dtype, DataTypeCls)
        if scope is not None and scope != "GM" and nbytes > 0:
            prev = buf_meta.get(buf.name)
            if prev is None or nbytes > prev[1]:
                buf_meta[buf.name] = (scope, nbytes)
        if scope == "UB" and static_shape is not None:
            ub_static_bytes[buf.name] = max(ub_static_bytes.get(buf.name, 0), nbytes)
            if buf.name not in seen_ub_bank and len(static_shape) >= 2 and math.prod(static_shape[:-1]) > 1 and nbytes > 0:
                seen_ub_bank.add(buf.name)
                eb = _dtype_bytes(buf.dtype, DataTypeCls)
                strides = getattr(buf, "strides", None) or ()
                rs = _static_tirx_int(strides[-2]) if len(strides) >= 2 else None
                row_elems = rs if (rs is not None and rs > 0) else static_shape[-1]
                ub_2d_total += nbytes
                if (sb := int(math.ceil(row_elems * eb))) >= 512 and sb % 512 == 0:
                    ub_2d_conflict += nbytes
        elif scope == "L1" and static_shape is not None and nbytes > 0:
            l1_static_bytes[buf.name] = max(l1_static_bytes.get(buf.name, 0), nbytes)
        elif scope == "GM" and buf.name not in seen_gm_keys:
            seen_gm_keys.add(buf.name)
            eb = _dtype_bytes(buf.dtype, DataTypeCls)
            if static_shape is not None:
                gm_static_bytes[buf.name] = nbytes
                if (tot := math.prod(static_shape) if static_shape else 0) == 1:
                    gm_scalar_names.add(buf.name)
                if static_shape and static_shape[-1] >= 1:
                    contig_candidates.append((tot, static_shape[-1], 0, nbytes, int(math.ceil(static_shape[-1] * eb))))
            else:
                gm_static_bytes[buf.name] = 0
                if buf.shape and (last_d := _static_tirx_int(buf.shape[-1])) is not None and last_d >= 1:
                    dyn_cnt = sum(1 for d in buf.shape if _static_tirx_int(d) is None)
                    st_dims = [v for d in buf.shape if (v := _static_tirx_int(d)) is not None]
                    st_prod = int(math.ceil(math.prod(st_dims) * eb)) if st_dims else 0
                    contig_candidates.append((None, last_d, dyn_cnt, st_prod, int(math.ceil(last_d * eb))))

    for param in getattr(entry_fn, "params", ()):
        if (buf := (getattr(entry_fn, "buffer_map", None) or {}).get(param)) is not None:
            _register_buf(buf, from_alloc=False)

    def _check_expr(expr, acc: _SubtreeAcc, *, check_arith: bool = False, loop_vars: tuple[object, ...] = ()) -> bool:
        nonlocal has_indirect_gather, has_dyn_offset_pipe
        if expr is None:
            return False
        sfu_flag = [False]
        ev, exp_f, uloads, ustores = _collect_tilelang_expr_rw(expr, tirx=tirx, sfu_out=sfu_flag)
        acc.events.extend(ev)
        acc.has_exp = acc.has_exp or exp_f
        acc.ub_loads.update(uloads)
        acc.ub_stores.update(ustores)
        acc.scope_direct_loads.update(uloads)
        acc.scope_direct_stores.update(ustores)
        if sfu_flag[0]:
            acc.scope_direct_sfu = True
        has_arith = has_load = False

        def _post(sub) -> None:
            nonlocal has_indirect_gather, has_dyn_offset_pipe, has_arith, has_load
            if isinstance(sub, tirx.Call):
                oname = _effective_tirx_call_name(sub).lower()
                if _is_disallowed_tirx_intrinsic(oname, allow_sync=False):
                    acc.has_disallowed_strict = True
                if _is_disallowed_tirx_intrinsic(oname, allow_sync=True):
                    acc.has_disallowed_sync_ok = True
                if check_arith and _is_tirx_math_call(oname):
                    has_arith = True
                if post_lower and loop_vars and "access_ptr" in str(getattr(sub.op, "name", sub.op)).lower() and len(sub.args) >= 3:
                    hit_mod = False

                    def _mvisit(mnode) -> None:
                        nonlocal hit_mod
                        if not hit_mod and mod_classes and isinstance(mnode, mod_classes):
                            if any(_same_tirx_var(fv, lv) for fv in _free_tirx_vars(mnode.a, tirx=tirx) for lv in loop_vars):
                                hit_mod = True

                    tirx.stmt_functor.post_order_visit(sub.args[2], _mvisit)
                    if hit_mod:
                        has_dyn_offset_pipe = True
            elif isinstance(sub, tirx.BufferLoad):
                if any(_contains_buffer_load(i, tirx) for i in sub.indices):
                    has_indirect_gather = True
                if check_arith:
                    rs = str(sub.buffer.scope()).strip()
                    if sub.buffer.name in ub_buf_names or _map_tilelang_scope(rs) in ("UB", "GM") or rs in ("local.ub", "global"):
                        has_load = True
            elif check_arith and arith_types and isinstance(sub, arith_types):
                has_arith = True

        tirx.stmt_functor.post_order_visit(expr, _post)
        return has_arith and has_load

    def _is_vec_sblock(node) -> bool:
        ann = getattr(node, "annotations", None) or {}
        hint = str(getattr(node, "name_hint", "")).lower()
        return "simd_vf" in hint or "simt_vf" in hint or "tl.simdvf_lanes" in ann or "tl.vf_source_index" in ann

    def _is_vec_attr(node) -> bool:
        key, nlow = str(getattr(node, "attr_key", "")).lower(), str(getattr(node, "node", "")).lower()
        if any(t in key for t in ("simdvf", "simtvf", "simt")) or "simt" in nlow or key == "pragma_parallel":
            return True
        if "thread_extent" in key:
            tag = str(getattr(node.node, "thread_tag", "")).lower()
            iv_n = str(getattr(getattr(node.node, "var", None), "name_hint", "")).lower()
            return all(t not in tag and t not in iv_n for t in ("blockidx", "cthread", "subblock"))
        return False

    def _record_block_dim(tag: str, expr, env: Mapping[object, object]) -> None:
        nonlocal any_symbolic_grid
        val, is_sym = _resolve_symbolic_grid_extent(expr, env, eff_bindings or var_bindings, tirx=tirx, tvm_mod=tvm_mod, var_obj_bindings=var_obj_bindings)
        if is_sym:
            any_symbolic_grid = True
        elif val is not None and val > 0:
            block_dims[tag] = val

    def _emit_bind_pipe_v(val_node, repeat: int, mte2_run: list[int], guarded_aiv: str | None) -> None:
        if isinstance(val_node, tirx.Call) and DataTypeCls is not None:
            b_op = _effective_tirx_call_name(val_node).lower()
            if ("simd" in b_op or "vec" in b_op) and "vld" not in b_op and "pset" not in b_op:
                mte2_run[0] = 0
                _emit_aiv("PIPE_V", repeat * _estimate_dav3510_vec_cycles(b_op, _effective_tirx_call_args(val_node), tirx, DataTypeCls), guarded_aiv)

    def _apply_thread_scope(tag: str, iv_name: str, extent, var_obj, ctx: _WalkCtx, *, is_attr: bool) -> tuple[frozenset[object], bool, bool]:
        nonlocal detected_split, has_simtvf
        next_sids, next_block = set(ctx.sid_vars), ctx.in_block_scope
        is_sub = "cthread" in tag or "subblock" in tag
        if "blockidx" in tag:
            _record_block_dim(tag, extent, ctx.let_env)
            if var_obj is not None and not block_var_info:
                block_var_info.append((var_obj, dict(ctx.let_env)))
            next_block = True
        if is_sub or "cthread" in iv_name or "subblock" in iv_name or (not is_attr and iv_name == "sid"):
            if var_obj is not None:
                next_sids.add(var_obj)
            if (sv := _static_tirx_int(extent)) is not None and sv >= 2:
                detected_split = 2
        elif is_attr and "blockidx" not in tag and "blockidx" not in iv_name:
            has_simtvf = True
            _add_opaque("simt_vf")
        return frozenset(next_sids), next_block, is_sub

    def _handle_call(call_node, ctx: _WalkCtx, *, repeat: int, mte2_run: list[int]) -> None:
        nonlocal detected_split, padding_bytes, max_mte2, illegal_dma, in_loop_pipe_all, seen_in_loop_gm_dma
        op_name, args, ann = _effective_tirx_call_name(call_node).lower(), _effective_tirx_call_args(call_node), getattr(call_node, "annotations", None) or {}
        if any(t in op_name for t in ("dual_copy", "subblock", "get_subblock_idx")) or "dual_dst_ctl" in ann:
            detected_split = 2
        if "pipe_barrier" in op_name and ctx.in_loop and any("pipe_all" in str(getattr(a, "value", a)).lower() for a in call_node.args):
            in_loop_pipe_all += 1
        if "copy" in op_name and len(args) >= 2 and DataTypeCls is not None and spaces is not None:
            src_s, xfer_b, inner_b = _region_bytes(args[0], tirx, DataTypeCls, let_env=ctx.let_env)
            dst_s, dst_b, dst_inner_b = _region_bytes(args[1], tirx, DataTypeCls, let_env=ctx.let_env)
            is_gm_name = post_lower and any(t in op_name for t in ("gm_to_", "_to_gm", "_gm_", "gm2", "2gm"))
            if is_gm_name or (src_s is not None and dst_s is not None and src_s != dst_s and "GM" in (src_s, dst_s)):
                if ctx.in_loop:
                    seen_in_loop_gm_dma = True
                if post_lower:
                    xfer_pl = max(_access_ptr_or_region_bytes(args[0], tirx=tirx, DataTypeCls=DataTypeCls, let_env=ctx.let_env),
                                  _access_ptr_or_region_bytes(args[1], tirx=tirx, DataTypeCls=DataTypeCls, let_env=ctx.let_env))
                    ok_sz = xfer_pl > 0
                else:
                    xfer_pl = max(xfer_b, dst_b)
                    ok_sz = xfer_pl >= 128
                if ok_sz:
                    all_dma_sizes.append(xfer_pl)
                    if ctx.in_loop:
                        loop_dma_sizes.append(xfer_pl)
            if src_s is not None and dst_s is not None:
                eff_inner = inner_b if inner_b > 0 else dst_inner_b
                xfer = max(xfer_b, dst_b) or eff_inner
                if src_s != dst_s and not spaces.is_dma_path_legal(src_s, dst_s):
                    illegal_dma = True
                if eff_inner > 0 and src_s == "GM" and dst_s == "UB":
                    if (rem := eff_inner % spaces.ub_align_bytes) != 0:
                        padding_bytes += (spaces.ub_align_bytes - rem) * repeat
                    mte2_run[0] += 1
                    max_mte2 = max(max_mte2, mte2_run[0])
                    _emit_aiv("PIPE_MTE2", repeat * _burst_scaled_copy_cycles(xfer, spaces), ctx.guarded_aiv)
                elif src_s == "UB" and dst_s == "GM":
                    mte2_run[0] = 0
                    _emit_aiv("PIPE_MTE3", repeat * _burst_scaled_copy_cycles(xfer, spaces), ctx.guarded_aiv)
                elif src_s == "UB" and dst_s == "L1":
                    mte2_run[0] = 0
                    _add_pipe("AIC.PIPE_MTE1", repeat * max(1, (xfer + 255) // 256))
                    _emit_aiv("PIPE_MTE3", repeat * max(1, (xfer + 63) // 64), ctx.guarded_aiv)
                elif src_s == "GM" and dst_s == "L1":
                    _add_pipe("AIC.PIPE_MTE2", repeat * _burst_scaled_copy_cycles(xfer, spaces))
                elif src_s == "L1" and dst_s in ("L0A", "L0B"):
                    _add_pipe("AIC.PIPE_MTE1", repeat * max(1, (xfer + 255) // 256))
                elif src_s == "L0C" and dst_s in ("GM", "L1", "UB"):
                    fr = max(1, spaces.fixpipe_bytes_per_cycle)
                    _add_pipe("AIC.PIPE_FIX", repeat * max(1, (xfer + fr - 1) // fr))
                    if dst_s == "UB":
                        _emit_aiv("PIPE_MTE2", repeat * max(1, (xfer + 63) // 64), ctx.guarded_aiv)
        elif ("gemm" in op_name or "mmad" in op_name) and len(args) >= 3:
            mte2_run[0] = 0
            if DataTypeCls is not None:
                m_val = _static_tirx_int(args[5]) if len(args) > 5 else None
                if m_val is None and isinstance(args[0], tirx.Call) and len(args[0].args) >= 4:
                    if (ext := _static_shape_ints(args[0].args[2:])) and len(ext) >= 2:
                        m_val = ext[-1] if (_static_tirx_int(args[3]) if len(args) > 3 else 0) else ext[-2]
                if m_val is not None and m_val > 0:
                    cube_m_values.append(m_val)
                _add_pipe("AIC.PIPE_M", repeat * _estimate_dav3510_cube_cycles(args, spaces, tirx))
        elif ("simd" in op_name or "vec" in op_name) and DataTypeCls is not None:
            mte2_run[0] = 0
            _emit_aiv("PIPE_V", repeat * _estimate_dav3510_vec_cycles(op_name, args, tirx, DataTypeCls), ctx.guarded_aiv)

    def _walk(stmt, ctx: _WalkCtx, *, repeat: int = 1, mte2_run: list[int] | None = None) -> _SubtreeAcc:
        nonlocal detected_split, has_simtvf, has_dyn_offset_pipe, has_unlimit_mem, max_mte2, has_online_rescale, ub_scalar_loops
        acc = _SubtreeAcc()
        if stmt is None:
            return acc
        mte2_run = mte2_run if mte2_run is not None else [0]
        split_hint = aiv_split_factor if aiv_split_factor is not None else max(min_aiv_split, detected_split)

        if isinstance(stmt, tirx.SeqStmt):
            seq_env = dict(ctx.let_env)
            for child in stmt.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    _check_expr(child.value, acc, loop_vars=ctx.loop_vars)
                    seq_env[child.var] = _substitute_let_env(child.value, seq_env, tirx=tirx)
                    _emit_bind_pipe_v(getattr(child, "value", None), repeat, mte2_run, ctx.guarded_aiv)
                    if hasattr(child, "body"):
                        acc.merge_non_scope(_walk(child.body, ctx.with_(let_env=dict(seq_env)), repeat=repeat, mte2_run=mte2_run))
                else:
                    acc.merge_non_scope(_walk(child, ctx.with_(let_env=dict(seq_env)), repeat=repeat, mte2_run=mte2_run))
            return acc

        if (let_stmt_cls is not None and isinstance(stmt, let_stmt_cls)) or (bind_cls is not None and isinstance(stmt, bind_cls)):
            _check_expr(stmt.value, acc, loop_vars=ctx.loop_vars)
            next_env = {**ctx.let_env, stmt.var: _substitute_let_env(stmt.value, ctx.let_env, tirx=tirx)}
            if isinstance(stmt, tirx.Bind):
                _emit_bind_pipe_v(getattr(stmt, "value", None), repeat, mte2_run, ctx.guarded_aiv)
            if hasattr(stmt, "body"):
                acc.merge_non_scope(_walk(stmt.body, ctx.with_(let_env=next_env), repeat=repeat, mte2_run=mte2_run))
            return acc

        if isinstance(stmt, tirx.While):
            _add_opaque("while_loop")
            if _contains_buffer_load(stmt.condition, tirx):
                _add_opaque("data_dependent_branch")
            _check_expr(stmt.condition, acc, loop_vars=ctx.loop_vars)
            sub = _walk(stmt.body, ctx.with_(in_loop=True), repeat=repeat, mte2_run=[0])
            acc.merge_non_scope(sub)
            acc.has_loop = True
            gm_loop_scalars.update(b for b, sc, _rw in sub.events if sc == "GM" and b in gm_scalar_names)
            if sub.has_exp and not has_online_rescale and _has_loop_carried_accumulator(sub.events, sub.local_bufs):
                has_online_rescale = True
            return acc

        if isinstance(stmt, tirx.For):
            if (tb := getattr(stmt, "thread_binding", None)) is not None:
                tb_tag = str(getattr(tb, "thread_tag", "")).lower()
                vname = str(getattr(stmt.loop_var, "name_hint", "")).lower()
                next_sids, next_block, is_sub = _apply_thread_scope(tb_tag, vname, stmt.extent, stmt.loop_var, ctx, is_attr=False)
                next_vec = ctx.in_vec_scope or ("blockidx" not in tb_tag and not is_sub)
                nctx = ctx.with_(sid_vars=next_sids, in_vec_scope=next_vec, in_block_scope=next_block, loop_vars=(*ctx.loop_vars, stmt.loop_var))
                acc.merge_non_scope(_walk(stmt.body, nctx, repeat=repeat, mte2_run=mte2_run))
                acc.has_loop = True
                return acc

            ann = getattr(stmt, "annotations", None) or {}
            kind_raw = getattr(stmt, "kind", 0)
            kv = getattr(kind_raw, "value", kind_raw)
            kind_val = int(kv) if isinstance(kv, int) else -1
            kind_str = str(kind_raw).lower()
            stages_val = _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else None
            is_pipe = kind_val == 5 or "pipelined" in kind_str or "num_stages" in ann or (stages_val is not None and stages_val >= 1) or bool(ann.get("enable_offset")) or (ann.get("software_pipeline_stage") is not None)
            if ann.get("enable_offset") or kind_val == 5 or (stages_val is not None and stages_val >= 2):
                has_dyn_offset_pipe = True
            next_found_pipe = ctx.found_pipe_loop
            if _is_pipelined_tirx_for(stmt, tirx) and not ctx.found_pipe_loop:
                pipelined_loops.append((stmt, set(ctx.sid_vars), ctx.guarded_aiv))
                next_found_pipe = True
            if ctx.in_block_scope and block_var_info:
                ext_sub = _substitute_let_env(stmt.extent, ctx.let_env, tirx=tirx)
                if any(_same_tirx_var(v, block_var_info[0][0]) for v in _free_tirx_vars(ext_sub, tirx=tirx)):
                    persistent_work_items.append(("loop", ext_sub))

            if (trip := _static_tirx_int(stmt.extent)) is None:
                _add_opaque(f"dynamic_for:{stmt.loop_var.name}")
            qualifies, sub_mte2 = (trip is None or trip > 1), [0]
            nctx = ctx.with_(in_loop=ctx.in_loop or qualifies, in_pipelined=ctx.in_pipelined or is_pipe, found_pipe_loop=next_found_pipe, loop_vars=(*ctx.loop_vars, stmt.loop_var))
            sub = _walk(stmt.body, nctx, repeat=(repeat * trip) if (trip is not None and trip > 0) else repeat, mte2_run=sub_mte2)
            if trip is not None and trip > 0 and sub_mte2[0] > 0:
                mte2_run[0] += sub_mte2[0] * trip
                max_mte2 = max(max_mte2, mte2_run[0])

            gm_loop_scalars.update(b for b, sc, _rw in sub.events if sc == "GM" and b in gm_scalar_names)
            if qualifies:
                ub_reload_store_bufs.update(sub.ub_loads & sub.ub_stores)
            if sub.has_exp and not has_online_rescale and _has_loop_carried_accumulator(sub.events, sub.local_bufs):
                has_online_rescale = True
            is_serial = kind_val == 0 and not any(t in kind_str for t in ("parallel", "vectorized", "unrolled", "thread")) and not is_pipe
            disallowed = sub.has_disallowed_sync_ok if allow_sync_in_scalar_loops else sub.has_disallowed_strict
            if not ctx.in_vec_scope and is_serial and (trip is None or trip >= 8) and not sub.has_loop and not sub.has_vec_sub and not disallowed and sub.has_ub_arith_store:
                ub_scalar_loops += 1

            if qualifies:
                acc.merge_from_child_scope(sub, _ScopeRec(set(sub.scope_direct_loads), set(sub.scope_direct_stores), sub.scope_direct_sfu, list(sub.child_scopes)))
            else:
                acc.merge_non_scope(sub)
            acc.has_loop = True
            return acc

        if isinstance(stmt, tirx.IfThenElse):
            if _contains_buffer_load(stmt.condition, tirx):
                _add_opaque("data_dependent_branch")
            _check_expr(stmt.condition, acc, loop_vars=ctx.loop_vars)
            if ctx.in_block_scope and block_var_info and stmt.else_case is None:
                cond_sub = _substitute_let_env(stmt.condition, ctx.let_env, tirx=tirx)
                if any(_same_tirx_var(v, block_var_info[0][0]) for v in _free_tirx_vars(cond_sub, tirx=tirx)):
                    persistent_work_items.append(("guard", cond_sub))
            t_g, e_g = _sid_guard_branches(stmt.condition, set(ctx.sid_vars), tirx, split_hint)
            b1, b2 = [mte2_run[0]], [mte2_run[0]]
            acc.merge_non_scope(_walk(stmt.then_case, ctx.with_(guarded_aiv=t_g if t_g is not None else ctx.guarded_aiv), repeat=repeat, mte2_run=b1))
            if stmt.else_case is not None:
                acc.merge_non_scope(_walk(stmt.else_case, ctx.with_(guarded_aiv=e_g if e_g is not None else ctx.guarded_aiv), repeat=repeat, mte2_run=b2))
            mte2_run[0] = max(b1[0], b2[0])
            return acc

        if isinstance(stmt, tirx.AttrStmt):
            key, klow = str(stmt.attr_key), str(stmt.attr_key).lower()
            next_sids, next_block = ctx.sid_vars, ctx.in_block_scope
            if "unlimit_memory" in klow:
                has_unlimit_mem = True
            if "buffer_version" in klow:
                if isinstance(stmt.node, tirx.Buffer):
                    requested_mb.add(stmt.node.name)
                elif hasattr(stmt.node, "items"):
                    for k, v in stmt.node.items():
                        if (int(v.value) if hasattr(v, "value") else 0) > 1 or str(v) == "auto":
                            requested_mb.add(getattr(k, "name", str(k)))
            if "thread_extent" in klow:
                tag = str(getattr(stmt.node, "thread_tag", "")).lower()
                iv = getattr(stmt.node, "var", None)
                next_sids, next_block, _ = _apply_thread_scope(tag, str(getattr(iv, "name_hint", "")).lower(), stmt.value, iv, ctx, is_attr=True)
            elif key == "pragma_parallel" or "simt" in klow:
                has_simtvf = True
            is_va = _is_vec_attr(stmt)
            acc.merge_non_scope(_walk(stmt.body, ctx.with_(sid_vars=next_sids, in_vec_scope=ctx.in_vec_scope or is_va, in_block_scope=next_block), repeat=repeat, mte2_run=mte2_run))
            if is_va:
                acc.has_vec_sub = True
            return acc

        if isinstance(stmt, tirx.SBlockRealize):
            return _walk(stmt.block, ctx, repeat=repeat, mte2_run=mte2_run)

        if isinstance(stmt, tirx.SBlock):
            ann = getattr(stmt, "annotations", None) or {}
            if _attrs_contain_flag(ann, "unlimit_memory"):
                has_unlimit_mem = True
            if "vector_count" in ann and (vc := _static_tirx_int(ann["vector_count"])) is not None and vc >= 2:
                detected_split = 2
            for k, v in ann.items():
                if "buffer_version" in str(k) and hasattr(v, "items"):
                    for bk, bv in v.items():
                        b_name = getattr(bk, "name", str(bk))
                        if (hasattr(bv, "__len__") and len(bv) > 0 and (int(bv[0].value) if hasattr(bv[0], "value") else 2) >= 2) or (hasattr(bv, "value") and int(bv.value) >= 2):
                            requested_mb.add(b_name)
            for buf in stmt.alloc_buffers:
                acc.local_bufs.add(buf.name)
                _register_buf(buf, from_alloc=False)
                if ctx.in_pipelined and _map_tilelang_scope(buf.scope()) in ("UB", "L1", "L0A", "L0B", "SSBUF"):
                    requested_mb.add(buf.name)
            is_vec = _is_vec_sblock(stmt)
            bctx = ctx.with_(in_vec_scope=ctx.in_vec_scope or is_vec)
            sub = _SubtreeAcc()
            if getattr(stmt, "init", None) is not None:
                sub.merge_non_scope(_walk(stmt.init, bctx, repeat=repeat, mte2_run=mte2_run))
            sub.merge_non_scope(_walk(stmt.body, bctx, repeat=repeat, mte2_run=mte2_run))
            if "simd_vf" in str(getattr(stmt, "name_hint", "")).lower() or "tl.simdvf_lanes" in ann:
                acc.merge_from_child_scope(sub, _ScopeRec(set(sub.scope_direct_loads), set(sub.scope_direct_stores), sub.scope_direct_sfu, list(sub.child_scopes)))
            else:
                acc.merge_non_scope(sub)
            if is_vec:
                acc.has_vec_sub = True
            return acc

        if isinstance(stmt, (tirx.AllocBuffer, tirx.DeclBuffer)) or (alloc_cls is not None and isinstance(stmt, alloc_cls)):
            if hasattr(stmt, "buffer"):
                acc.local_bufs.add(stmt.buffer.name)
                _register_buf(stmt.buffer, from_alloc=True)
            if hasattr(stmt, "body"):
                acc.merge_non_scope(_walk(stmt.body, ctx, repeat=repeat, mte2_run=mte2_run))
            return acc

        if isinstance(stmt, tirx.BufferStore):
            mte2_run[0] = 0
            _add_pipe(f"{'AIV1' if ctx.guarded_aiv == 'aiv1' else 'AIV0'}.PIPE_S", repeat)
            _check_expr(stmt.value, acc, loop_vars=ctx.loop_vars)
            for idx in stmt.indices:
                _check_expr(idx, acc, loop_vars=ctx.loop_vars)
            if (sc := _map_tilelang_scope(stmt.buffer.scope())) is not None:
                acc.events.append((stmt.buffer.name, sc, "write"))
                if sc == "UB":
                    acc.ub_stores.add(stmt.buffer.name)
                    acc.scope_direct_stores.add(stmt.buffer.name)
            rs = str(stmt.buffer.scope()).strip()
            if stmt.buffer.name in ub_buf_names or _map_tilelang_scope(rs) == "UB" or rs == "local.ub":
                if _check_expr(_substitute_let_env(stmt.value, ctx.let_env, tirx=tirx), _SubtreeAcc(), check_arith=True, loop_vars=ctx.loop_vars):
                    acc.has_ub_arith_store = True
            return acc

        if isinstance(stmt, tirx.Evaluate):
            _check_expr(stmt.value, acc, loop_vars=ctx.loop_vars)
            if isinstance(stmt.value, tirx.Call):
                _handle_call(stmt.value, ctx, repeat=repeat, mte2_run=mte2_run)
            return acc

        return acc

    root_acc = _walk(getattr(entry_fn, "body", None), _WalkCtx())
    eff_split = aiv_split_factor if aiv_split_factor is not None else max(min_aiv_split, detected_split)
    pipe_cycles: dict[str, int] = {}
    for pkey, cyc in pipe_raw.items():
        if pkey.startswith("UNSPLIT."):
            pname = pkey.split(".", 1)[1]
            if eff_split >= 2:
                per = max(1, cyc // 2)
                pipe_cycles[f"AIV0.{pname}"] = pipe_cycles.get(f"AIV0.{pname}", 0) + per
                pipe_cycles[f"AIV1.{pname}"] = pipe_cycles.get(f"AIV1.{pname}", 0) + per
            else:
                pipe_cycles[f"AIV0.{pname}"] = pipe_cycles.get(f"AIV0.{pname}", 0) + cyc
        else:
            pipe_cycles[pkey] = pipe_cycles.get(pkey, 0) + cyc

    explicit_snaps = tuple(sorted((e[3] for e in snap_entries if not e[0]), key=lambda s: (s.scope, s.name)))
    alloc_snaps = tuple(sorted((e[3] for e in snap_entries if e[0]), key=lambda s: (s.scope, s.name)))
    if not explicit_snaps:
        explicit_snaps = alloc_snaps

    uses_aic = pipe_cycles.get("AIC.PIPE_M", 0) > 0 or eff_split >= 2
    core_pool = (spaces.num_aic_cores if uses_aic else spaces.num_aiv_cores) if spaces is not None else 1
    if any_symbolic_grid:
        grid_is_symbolic, grid_blocks, active_grid_blocks = True, max(1, core_pool), max(1, core_pool)
    else:
        grid_is_symbolic = False
        grid_blocks = active_grid_blocks = math.prod(block_dims.values()) if block_dims else 1
        if not post_lower and 0 < grid_blocks <= 512 and block_var_info and persistent_work_items and (eff_bindings or var_obj_bindings) and tvm_mod is not None and hasattr(getattr(tvm_mod, "arith", None), "Analyzer"):
            bvar = block_var_info[0][0]
            bound_items, ok_bound = [], True
            for kind, iexpr in persistent_work_items:
                vmap = _build_free_var_int_map(iexpr, eff_bindings, var_obj_bindings, tirx=tirx, exclude_var=bvar)
                sub_e = tirx.stmt_functor.substitute(iexpr, vmap) if vmap else iexpr
                if any(not _same_tirx_var(v, bvar) for v in _free_tirx_vars(sub_e, tirx=tirx)):
                    ok_bound = False
                    break
                bound_items.append((kind, sub_e))
            if ok_bound:
                analyzer, bx_dtype = tvm_mod.arith.Analyzer(), str(getattr(bvar, "dtype", "int32") or "int32")
                act_cnt = work_cnt = 0
                for bx in range(grid_blocks):
                    bx_imm, bx_guard, bx_loops = tirx.IntImm(bx_dtype, bx), False, 0
                    for kind, sub_e in bound_items:
                        try:
                            sim = analyzer.simplify(tirx.stmt_functor.substitute(sub_e, {bvar: bx_imm}))
                        except Exception:
                            act_cnt = -1
                            break
                        val = _static_tirx_int(sim)
                        if val is None and isinstance(sim, tirx.IntImm):
                            val = int(sim.value)
                        if val is None:
                            act_cnt = -1
                            break
                        if kind == "guard" and val > 0:
                            bx_guard = True
                        elif kind == "loop" and val > 0:
                            bx_loops += int(val)
                    if act_cnt < 0:
                        break
                    if bx_guard or bx_loops > 0:
                        act_cnt += 1
                        work_cnt += bx_loops if bx_loops > 0 else 1
                if act_cnt >= 0:
                    active_grid_blocks, grid_blocks = act_cnt, max(act_cnt, work_cnt)

    gm_reads = {n for n, sc, rw in root_acc.events if sc == "GM" and rw == "read"}
    gm_writes = {n for n, sc, rw in root_acc.events if sc == "GM" and rw == "write"}
    gm_roundtrip_bytes = sum(gm_static_bytes.get(n, 0) for n in ((gm_reads & gm_writes) | gm_loop_scalars))
    dma_pool = [b for b in (loop_dma_sizes if (loop_dma_sizes or (post_lower and seen_in_loop_gm_dma)) else all_dma_sizes) if b > 0]
    min_gm_dma_bytes = min(dma_pool) if dma_pool else 0
    if spaces is not None and dma_pool:
        w_inv = sum(float(b) / max(1e-6, float(spaces.dma_burst_efficiency(int(b)))) for b in dma_pool)
        dma_burst_eff = round(min(1.0, float(sum(dma_pool)) / max(1e-6, w_inv)), 4)
    else:
        dma_burst_eff = 1.0

    wide_ub = {n for n, nb in ub_static_bytes.items() if nb >= 1024}
    coupled_sfu = bool(wide_ub) and any(_eval_coupled_scope(r, wide_ub)[2] for r in root_acc.child_scopes)
    ub_risk = round(min(1.0, ub_2d_conflict / float(ub_2d_total)), 4) if (ub_2d_total > 0 and ub_2d_conflict > 0) else 0.0
    l1_tot, l1_full = sum(l1_static_bytes.values()), max(1, spaces.l1_bytes if spaces is not None else 524288)
    l1_half = max(1, l1_full // 2)
    l1_risk = 0.0 if (l1_tot <= 0 or has_l1_half) else (round(min(1.0, l1_tot / float(l1_full)), 4) if l1_tot > l1_half else (round(min(0.5, l1_tot / float(l1_half)), 4) if len(l1_static_bytes) >= 2 else 0.0))

    return _UnifiedScanResult(
        pipe_cycles=pipe_cycles, mte2_inflight=max_mte2, padding_bytes=padding_bytes, opaque_regions=tuple(opaque_regions),
        has_simtvf=has_simtvf, illegal_dma=illegal_dma, aiv_split_factor=eff_split, explicit_snapshots=explicit_snaps,
        alloc_snapshots=alloc_snaps if alloc_snaps else explicit_snaps, requested_multibuffer=requested_mb,
        buf_meta=buf_meta, ub_buf_names=ub_buf_names, ub_static_bytes=ub_static_bytes, l1_static_bytes=l1_static_bytes,
        gm_static_bytes=gm_static_bytes, gm_scalar_names=gm_scalar_names,
        dominant_contig_bytes=_select_dominant_contig_bytes(contig_candidates),
        effective_cube_m=min(cube_m_values) if cube_m_values else 0, has_indirect_gather=has_indirect_gather,
        has_lowbit_quant=bool(has_lowbit_buf and pipe_cycles.get("AIC.PIPE_M", 0) > 0),
        has_online_rescale_loop=has_online_rescale, in_loop_ub_reload_stores=len(ub_reload_store_bufs),
        gm_roundtrip_bytes=gm_roundtrip_bytes, min_gm_dma_bytes=min_gm_dma_bytes, dma_burst_efficiency_ratio=dma_burst_eff,
        seen_in_loop_gm_dma=seen_in_loop_gm_dma, in_loop_pipe_all_barriers=in_loop_pipe_all,
        uses_dynamic_offset_pipelining=has_dyn_offset_pipe, has_unlimit_memory_attr=has_unlimit_mem,
        has_coupled_reduce_sfu_accum=coupled_sfu, ub_bank_conflict_risk=ub_risk, l1_bank_conflict_risk=l1_risk,
        ub_scalar_compute_loops=ub_scalar_loops, grid_blocks=grid_blocks, active_grid_blocks=active_grid_blocks,
        grid_is_symbolic=grid_is_symbolic, uses_unreserved_ubuf=effective_unreserved, pipelined_loops=pipelined_loops,
    )


def _detect_tilelang_aiv_split_factor(func, *, tirx) -> int:
    if func is None or getattr(func, "body", None) is None:
        return 1
    return _scan_primfunc_unified(func, tirx=tirx).aiv_split_factor


def _count_in_loop_pipe_all_barriers(stmt, *, tirx) -> int:
    fake_fn = type("_Fn", (), {"body": stmt, "params": (), "buffer_map": {}, "attrs": {}})()
    return _scan_primfunc_unified(fake_fn, tirx=tirx).in_loop_pipe_all_barriers


def _count_tirx_ub_scalar_compute_loops(
    stmt, ub_buf_names: set[str], *, tirx, allow_sync_intrinsics: bool = False
) -> int:
    fake_fn = type("_Fn", (), {"body": stmt, "params": (), "buffer_map": {}, "attrs": {}})()
    return _scan_primfunc_unified(
        fake_fn, tirx=tirx, ub_buf_override=ub_buf_names, allow_sync_in_scalar_loops=allow_sync_intrinsics
    ).ub_scalar_compute_loops


@dataclass(frozen=True)
class _TileLangLoopTask:
    label: str
    mask: NodeResourceMask
    latency: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _extract_pipelined_loop_tasks(
    loop_body, *, tirx, DataTypeCls, aiv_split_factor: int, sid_vars: set[object],
    guarded_aiv: str | None, spaces: AscendSpaceFacts | None = None,
    dynamic_inner_out: list[bool] | None = None,
) -> list[_TileLangLoopTask] | None:
    tasks: list[_TileLangLoopTask] = []
    aborted = False

    def _add_vf_task(body, scale: int, guard: str | None) -> None:
        reads, writes, vf_ops = _extract_vf_accesses(body, tirx=tirx, DataTypeCls=DataTypeCls)
        tasks.append(_TileLangLoopTask(
            label="simdvf", mask=_aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=guard),
            latency=min(8, max(1, (vf_ops + 3) // 4) * scale), reads=tuple(reads), writes=tuple(writes),
        ))

    def _dyn_or_abort(body, scale: int, cur_guard: str | None) -> None:
        nonlocal aborted
        if dynamic_inner_out is not None:
            dynamic_inner_out[0] = True
            if body is not None:
                _walk(body, scale=scale, cur_guard=cur_guard)
        else:
            aborted = True

    def _walk(node, *, scale: int, cur_guard: str | None) -> None:
        nonlocal aborted
        if node is None or aborted:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, tirx.While):
            _dyn_or_abort(node.body, scale, cur_guard)
        elif isinstance(node, tirx.For):
            if (trip := _static_tirx_int(node.extent)) is None:
                _dyn_or_abort(node.body, scale, cur_guard)
            elif trip > 0:
                _walk(node.body, scale=min(8, scale * trip), cur_guard=cur_guard)
        elif isinstance(node, tirx.IfThenElse):
            if _contains_buffer_load(node.condition, tirx):
                _dyn_or_abort(None, scale, cur_guard)
                if aborted:
                    return
            tg, eg = _sid_guard_branches(node.condition, sid_vars, tirx, aiv_split_factor)
            _walk(node.then_case, scale=scale, cur_guard=tg if tg is not None else cur_guard)
            if node.else_case is not None:
                _walk(node.else_case, scale=scale, cur_guard=eg if eg is not None else cur_guard)
        elif isinstance(node, tirx.SBlockRealize):
            _walk(node.block, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, tirx.SBlock):
            ann, hint = getattr(node, "annotations", None) or {}, str(getattr(node, "name_hint", "")).lower()
            _add_vf_task(node.body, scale, cur_guard) if ("simd_vf" in hint or "simt_vf" in hint or "tl.simdvf_lanes" in ann) else _walk(node.body, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, tirx.AttrStmt):
            key = str(node.attr_key).lower()
            _add_vf_task(node.body, scale, cur_guard) if ("simdvf_scope" in key or "simtvf_scope" in key) else _walk(node.body, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, (tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, tirx.BufferStore):
            sc = _map_tilelang_scope(node.buffer.scope())
            reads, _, _ = _extract_vf_accesses(node.value, tirx=tirx, DataTypeCls=DataTypeCls)
            tasks.append(_TileLangLoopTask(
                label="scalar_store", mask=NodeResourceMask.for_core(cur_guard or "aiv0", "SCALAR"),
                latency=1, reads=tuple(reads), writes=(node.buffer.name,) if sc not in (None, "GM") else (),
            ))
        elif isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name, args = _effective_tirx_call_name(node.value).lower(), _effective_tirx_call_args(node.value)
            if "copy" in op_name and len(args) >= 2:
                sn, ss, sb = _tilelang_arg_buf(args[0], tirx, DataTypeCls)
                dn, ds, db = _tilelang_arg_buf(args[1], tirx, DataTypeCls)
                if ss is not None and ds is not None:
                    xfer = max(sb, db)
                    eff_x = int(math.ceil(float(xfer) / max(0.15, float(spaces.dma_burst_efficiency(xfer))))) if (spaces is not None and xfer > 0 and "GM" in (ss, ds)) else xfer
                    tasks.append(_TileLangLoopTask(
                        label="copy", mask=_copy_resource_mask(ss, ds, aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard),
                        latency=min(8, max(1, (eff_x + 16383) // 16384) * scale),
                        reads=(sn,) if (sn is not None and ss != "GM") else (),
                        writes=(dn,) if (dn is not None and ds != "GM") else (),
                    ))
            elif ("gemm" in op_name or "mmad" in op_name) and len(args) >= 3:
                an, a_s, _ = _tilelang_arg_buf(args[0], tirx, DataTypeCls)
                bn, b_s, _ = _tilelang_arg_buf(args[1], tirx, DataTypeCls)
                cn, c_s, _ = _tilelang_arg_buf(args[2], tirx, DataTypeCls)
                clear_v = _static_tirx_int(args[9]) if len(args) > 9 else None
                reads_l = [n for n, s in ((an, a_s), (bn, b_s)) if n is not None and s not in (None, "GM")]
                if (clear_v is None or clear_v == 0) and cn is not None and c_s not in (None, "GM") and cn not in reads_l:
                    reads_l.append(cn)
                tasks.append(_TileLangLoopTask(
                    label="gemm", mask=NodeResourceMask.for_aic("CUBE"),
                    latency=min(8, max(1, _estimate_dav3510_cube_cycles(args, spaces, tirx)) * scale),
                    reads=tuple(reads_l), writes=(cn,) if (cn is not None and c_s not in (None, "GM")) else (),
                ))
            elif "simd" in op_name or "vec" in op_name:
                reads, writes, _ = _extract_vf_accesses(node.value, tirx=tirx, DataTypeCls=DataTypeCls)
                tasks.append(_TileLangLoopTask(
                    label="vec", mask=_aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard),
                    latency=min(8, _estimate_dav3510_vec_cycles(op_name, args, tirx, DataTypeCls) * scale),
                    reads=tuple(reads), writes=tuple(writes),
                ))

    _walk(loop_body, scale=1, cur_guard=guarded_aiv)
    return None if aborted else tasks


def _analytical_pipelined_ii(task_items: Sequence[_TileLangLoopTask]) -> int:
    if not task_items:
        return 1
    totals: dict[tuple[str, str], int] = {}
    for item in task_items:
        lat = max(1, int(item.latency))
        for ck, pn in item.mask.to_core_pipe_pairs():
            totals[(ck, pn)] = totals.get((ck, pn), 0) + lat
    return max(totals.values(), default=1)


def _solve_pipelined_loops_ii(
    pipelined_loops: Sequence[tuple[object, set[object], str | None]],
    buf_meta: Mapping[str, tuple[str, int]], spaces: AscendSpaceFacts, *, tirx, DataTypeCls,
    aiv_split_factor: int, has_simtvf: bool, unreserved_ubuf: bool, degraded_names: Sequence[str] = (),
) -> int | None:
    if not pipelined_loops:
        return None
    all_buf_names, degraded_set, solved_iis = set(buf_meta), set(degraded_names), []

    def _collect_local(stmt) -> set[str]:
        names: set[str] = set()
        tirx.stmt_functor.post_order_visit(
            stmt, lambda n: names.update(b.name for b in n.alloc_buffers) if isinstance(n, tirx.SBlock) else (names.add(n.buffer.name) if isinstance(n, (tirx.AllocBuffer, tirx.DeclBuffer)) else None),
        )
        return names

    for loop, sid_vars, loop_guard in pipelined_loops:
        trip = _static_tirx_int(loop.extent)
        if trip is None or trip < 1:
            return None
        ann = getattr(loop, "annotations", None) or {}
        sv = _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else 2
        num_stages = max(1, sv if sv is not None else 2)
        outer_bufs, dynamic_inner = all_buf_names - _collect_local(loop.body), [False]
        task_items = _extract_pipelined_loop_tasks(
            loop.body, tirx=tirx, DataTypeCls=DataTypeCls, aiv_split_factor=aiv_split_factor,
            sid_vars=sid_vars, guarded_aiv=loop_guard, spaces=spaces, dynamic_inner_out=dynamic_inner,
        )
        if task_items is None:
            return None
        if not task_items:
            continue
        if dynamic_inner[0]:
            solved_iis.append(_analytical_pipelined_ii(task_items))
            continue

        ref_bufs, last_w, readers_since, all_readers, first_readers = set(), {}, {}, {}, {}
        lat_by_name, task_names, task_specs = {}, [], []
        edges, reuses, edge_keys, reuse_keys = [], [], set(), set()

        def _add_edge(src: str, dst: str, dist: int, lat: int) -> None:
            if (ek := (src, dst, dist, lat)) not in edge_keys:
                edge_keys.add(ek)
                edges.append(Edge(src=src, dst=dst, distance=dist, latency=lat))

        def _add_reuse(bn: str, r: str, w: str) -> None:
            if r != w and (rk := (bn, r, w)) not in reuse_keys:
                reuse_keys.add(rk)
                reuses.append(Reuse(buffer=bn, last_reader=r, next_writer=w, delta=0, read_latency=1))

        for idx, item in enumerate(task_items):
            tname = f"t{idx}_{item.label}"
            lat_by_name[tname] = item.latency
            task_names.append(tname)
            task_specs.append((tname, item.mask.to_solver_resources(1, 0), item.latency))
            for bn in (b for b in item.reads if b in buf_meta):
                ref_bufs.add(bn)
                if bn in last_w:
                    if (w := last_w[bn]) != tname:
                        _add_edge(w, tname, 0, lat_by_name[w])
                        all_readers.setdefault(bn, []).append(tname)
                        readers_since.setdefault(bn, []).append(tname)
                elif bn in outer_bufs and tname not in (fr := first_readers.setdefault(bn, [])):
                    fr.append(tname)
            for bn in (b for b in item.writes if b in buf_meta):
                ref_bufs.add(bn)
                if bn in last_w:
                    for rn in readers_since.get(bn, ()):
                        _add_reuse(bn, rn, last_w[bn])
                    readers_since[bn] = []
                last_w[bn] = tname

        loop_carried: set[str] = set()
        for bn, writer in last_w.items():
            if bn in first_readers:
                loop_carried.add(bn)
                for rn in first_readers[bn]:
                    _add_edge(writer, rn, 1, lat_by_name[writer])
            else:
                for rn in all_readers.get(bn, ()):
                    _add_reuse(bn, rn, writer)

        max_stage = max(0, num_stages - 1)
        stages = _assign_dag_stages(task_names, edges, {}, max_stage) if max_stage > 0 else {n: 0 for n in task_names}
        manual = max_stage > 0
        tasks = [Task(name=n, resources=ru, latency=lt, stage=stages[n] if manual else None) for n, ru, lt in task_specs]
        reused_bufs, buffers, has_ssbuf = {r.buffer for r in reuses}, [], False
        for bn in sorted(ref_bufs):
            sp, sb = buf_meta[bn]
            has_ssbuf = has_ssbuf or sp == "SSBUF"
            owner = spaces.ssbuf_owner if sp == "SSBUF" else ("aiv" if sp == "UB" else "aic")
            if bn not in degraded_set and bn not in loop_carried and manual and num_stages > 1 and bn in reused_bufs:
                buffers.append(Buffer(name=bn, slot_bytes=sb, space=sp, owner=owner, auto_min=1, auto_max=num_stages))
            else:
                buffers.append(Buffer(name=bn, slot_bytes=sb, space=sp, owner=owner, fixed=1))

        caps = _capacities_from_facts(spaces, has_simtvf=has_simtvf, unreserved_ubuf=unreserved_ubuf)
        if has_ssbuf:
            caps = (*caps, Capacity("SSBUF", spaces.ssbuf_owner, spaces.ssbuf_bytes))
        problem = tf_adapter({
            "tasks": tuple(tasks), "edges": tuple(edges), "buffers": tuple(buffers), "reuses": tuple(reuses),
            "resources": _resources_for_tasks(tasks), "capacities": caps, "max_stage": max_stage, "manual": manual, "issue_after": (),
        })
        solved_iis.append(solve_pipeline_with_monotone_backoff(problem).assignment.ii)

    return max(solved_iis) if solved_iis else None


def _try_solve_tilelang_pipelined_ii(
    entry_fn,
    spaces: AscendSpaceFacts,
    *,
    tirx,
    DataTypeCls,
    aiv_split_factor: int,
    has_simtvf: bool,
    unreserved_ubuf: bool,
    degraded_names: Sequence[str] = (),
) -> int | None:
    if entry_fn is None or getattr(entry_fn, "body", None) is None:
        return None
    scan = _scan_primfunc_unified(
        entry_fn, spaces=spaces, tirx=tirx, DataTypeCls=DataTypeCls, aiv_split_factor=aiv_split_factor
    )
    return _solve_pipelined_loops_ii(
        scan.pipelined_loops, scan.buf_meta, spaces, tirx=tirx, DataTypeCls=DataTypeCls,
        aiv_split_factor=aiv_split_factor, has_simtvf=has_simtvf,
        unreserved_ubuf=unreserved_ubuf, degraded_names=degraded_names,
    )


def _scope_capacities(facts: AscendSpaceFacts, *, has_simtvf: bool, unreserved_ubuf: bool) -> dict[str, int]:
    ub_mode = "simtvf" if has_simtvf else "simdvf"
    return {
        "UB": facts.ub_limit_bytes(ub_mode, unreserved_ubuf=unreserved_ubuf),
        "L1": facts.l1_bytes,
        "L0A": facts.l0a_bytes,
        "L0B": facts.l0b_bytes,
        "L0C": facts.l0c_bytes,
        "SSBUF": facts.ssbuf_bytes,
    }


def _extract_tilelang_buffers_from_mod(mod, *, tirx, DataTypeCls, prefer_allocate: bool) -> tuple[BufferSnapshot, ...]:
    if mod is None:
        return ()
    snaps: list[BufferSnapshot] = []
    for _gvar, fn in mod.functions.items():
        if isinstance(fn, tirx.PrimFunc):
            scan = _scan_primfunc_unified(fn, tirx=tirx, DataTypeCls=DataTypeCls)
            snaps.extend(scan.alloc_snapshots if prefer_allocate else scan.explicit_snapshots)
    return tuple(snaps)


def _detect_degraded_buffers(
    explicit_snaps: tuple[BufferSnapshot, ...],
    materialized_snaps: tuple[BufferSnapshot, ...],
    requested_names: set[str],
) -> tuple[str, ...]:
    if not requested_names:
        return ()
    exp_map, mat_map = {s.name: s for s in explicit_snaps}, {s.name: s for s in materialized_snaps}
    degraded: list[str] = []
    for name in sorted(requested_names):
        exp, mat = exp_map.get(name), mat_map.get(name)
        if exp is None or mat is None:
            continue
        if (len(mat.shape) == len(exp.shape) and mat.bytes <= exp.bytes) or (
            len(mat.shape) > len(exp.shape) and mat.shape[0] <= 1
        ):
            degraded.append(name)
    return tuple(degraded)


def _diff_snapshots_by_scope(
    explicit_snaps: tuple[BufferSnapshot, ...],
    lowered_snaps: tuple[BufferSnapshot, ...],
    merged_peak_by_scope: dict[str, int],
    capacities: dict[str, int],
    degraded_names: tuple[str, ...],
) -> dict[str, ScopeWaterline]:
    scopes = sorted({s.scope for s in (*explicit_snaps, *lowered_snaps)} | set(merged_peak_by_scope))
    waterlines: dict[str, ScopeWaterline] = {}
    degraded_set = set(degraded_names)
    for scope in scopes:
        exp_map = {s.name: s.bytes for s in explicit_snaps if s.scope == scope}
        low_snaps = [s for s in lowered_snaps if s.scope == scope]
        imp_map: dict[str, int] = {}
        matched: set[str] = set()
        for snap in low_snaps:
            base = snap.name
            if base not in exp_map and base.endswith("_buf") and base[:-4] in exp_map:
                base = base[:-4]
            if base in exp_map and base not in matched:
                matched.add(base)
                if (delta := snap.bytes - exp_map[base]) > 0:
                    imp_map[f"{base}_expansion"] = delta
            else:
                imp_map[snap.name] = snap.bytes
        peak = merged_peak_by_scope.get(scope, sum(s.bytes for s in low_snaps) if low_snaps else sum(exp_map.values()))
        waterlines[scope] = build_scope_waterline(
            scope=scope,
            capacity_bytes=capacities.get(scope, 0),
            explicit_buffers=exp_map,
            implicit_buffers=imp_map,
            peak_allocated_bytes=peak,
            degraded_single_buffers=tuple(n for n in sorted(degraded_set) if n in exp_map),
        )
    return waterlines


def _to_python_attr_value(val: object) -> object:
    if isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "value") and isinstance(getattr(val, "value"), (bool, int, float, str)):
        return getattr(val, "value")
    if hasattr(val, "items"):
        return {str(k): _to_python_attr_value(v) for k, v in val.items()}
    if isinstance(val, (tuple, list)) or (hasattr(val, "__iter__") and not isinstance(val, (bytes, bytearray))):
        try:
            return [_to_python_attr_value(item) for item in val]
        except TypeError:
            return str(val)
    return str(val)


def _normalize_compile_flags(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        return raw_flags.strip().split() if raw_flags.strip() else []
    if isinstance(raw_flags, (tuple, list)) or (hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray))):
        return [s for item in raw_flags if (s := str(_to_python_attr_value(item)).strip())]
    return [t] if (t := str(raw_flags).strip()) else []


def _extract_primfunc_pass_configs_and_flags(
    entry_fn,
    mod_in,
    *,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
) -> tuple[dict[str, object], list[str]]:
    cfg: dict[str, object] = {}
    flags: list[str] = []

    def _append(items: Sequence[str]) -> None:
        for item in items:
            if item and item not in flags:
                flags.append(item)

    for owner in (mod_in, entry_fn):
        attrs = getattr(owner, "attrs", None)
        if not attrs or not hasattr(attrs, "items"):
            continue
        for pc_key in ("tl.pass_configs", "tilelang_pass_configs"):
            if pc_key in attrs and isinstance(raw_pc := _to_python_attr_value(attrs[pc_key]), dict):
                for k, v in raw_pc.items():
                    if k == "tl.device_compile_flags":
                        _append(_normalize_compile_flags(v))
                    else:
                        cfg[str(k)] = v
        for cf_key in ("tir.cce_flags", "tilelang_compile_flags"):
            if cf_key in attrs:
                _append(_normalize_compile_flags(attrs[cf_key]))

    for k, v in (pass_configs or {}).items():
        if str(k) == "tl.device_compile_flags":
            _append(_normalize_compile_flags(v))
        else:
            cfg[str(k)] = v
    _append(_normalize_compile_flags(compile_flags))
    if flags:
        cfg["tl.device_compile_flags"] = list(flags)
    return cfg, flags


def _add_unique_expr(target: list[object], expr) -> None:
    if expr is not None and not any(_same_tirx_var(expr, prev) for prev in target):
        target.append(expr)


def _collect_div_denominators(entry_fn, *, tirx=None) -> list[object]:
    if tirx is None:
        from tvm import tirx as _tirx  # noqa: PLC0415

        tirx = _tirx
    div_cls = tuple(c for n in ("FloorDiv", "FloorMod", "Div", "Mod") if (c := getattr(tirx, n, None)) is not None)
    if not div_cls or entry_fn is None:
        return []
    raw_denoms: list[object] = []
    let_env: dict[object, object] = {}
    let_cls = tuple(c for n in ("LetStmt", "Bind", "Let") if (c := getattr(tirx, n, None)) is not None)
    alloc_cls = tuple(c for n in ("AllocBuffer", "DeclBuffer") if (c := getattr(tirx, n, None)) is not None)
    sblock_cls = getattr(tirx, "SBlock", None)

    def _scan_buf(buf) -> None:
        for e in (*getattr(buf, "shape", ()), *(getattr(buf, "strides", ()) or ()), getattr(buf, "elem_offset", None)):
            if e is not None:
                tirx.stmt_functor.post_order_visit(e, _visit)

    def _visit(node) -> None:
        if let_cls and isinstance(node, let_cls):
            if (vo := getattr(node, "var", None)) is not None and (val := getattr(node, "value", None)) is not None:
                let_env[vo] = val
        if isinstance(node, div_cls):
            _add_unique_expr(raw_denoms, getattr(node, "b", None))
        if alloc_cls and isinstance(node, alloc_cls) and (buf := getattr(node, "buffer", None)) is not None:
            _scan_buf(buf)
        if sblock_cls is not None and isinstance(node, sblock_cls):
            for buf in getattr(node, "alloc_buffers", ()) or ():
                _scan_buf(buf)

    if (body := getattr(entry_fn, "body", None)) is not None:
        tirx.stmt_functor.post_order_visit(body, _visit)
    for buf in (getattr(entry_fn, "buffer_map", None) or {}).values():
        _scan_buf(buf)
    if not let_env:
        return raw_denoms
    expanded: list[object] = []
    for d in raw_denoms:
        _add_unique_expr(expanded, _substitute_let_env(d, let_env, tirx=tirx))
    return expanded


def _max_nonzero_const_magnitude(expr, *, tirx) -> int:
    max_val = 1
    int_imm_cls = getattr(tirx, "IntImm", None)
    if int_imm_cls is None or expr is None:
        return max_val

    def _visit(node) -> None:
        nonlocal max_val
        if isinstance(node, int_imm_cls) and (val := _static_tirx_int(node)) is not None and val != 0:
            max_val = max(max_val, abs(int(val)))

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return max_val


def _ensure_positive_denominators(
    denominators: Sequence[object], spec_var_map: dict[object, object], recorded_vars: list[object],
    allowed_vars: Sequence[object], *, is_handle_or_data: Callable[[object], bool],
    tirx, tvm_mod, max_rounds: int = 24,
) -> None:
    """Iteratively adjust free integer vars so every static denominator simplifies >= 1."""
    if not denominators:
        return

    def _find_key(vo):
        return next((k for k in spec_var_map if _same_tirx_var(vo, k)), None)

    def _is_allowed(vo) -> bool:
        ds = str(getattr(vo, "dtype", "int32") or "int32").lower()
        return isinstance(vo, tirx.Var) and not is_handle_or_data(vo) and ds != "handle" and "int" in ds and any(_same_tirx_var(vo, a) for a in allowed_vars)

    for denom in denominators:
        for fv in _free_tirx_vars(denom, tirx=tirx):
            if _is_allowed(fv) and _find_key(fv) is None:
                recorded_vars.append(fv)
                spec_var_map[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), 1)

    analyzer = tvm_mod.arith.Analyzer()

    def _simp(expr) -> tuple[int | None, bool]:
        try:
            return _static_tirx_int(analyzer.simplify(expr)), False
        except Exception:
            return None, True

    def _set_val(key, new_val: int) -> bool:
        if (cur_v := _static_tirx_int(spec_var_map[key])) is not None and int(cur_v) == int(new_val):
            return False
        spec_var_map[key] = tirx.IntImm(str(getattr(key, "dtype", "int32") or "int32"), int(new_val))
        return True

    for _ in range(max_rounds):
        changed = False
        for denom in denominators:
            sub_d = tirx.stmt_functor.substitute(denom, spec_var_map) if spec_var_map else denom
            simp_val, simp_err = _simp(sub_d)
            if not simp_err and (simp_val is None or simp_val > 0):
                continue
            if not (cands := [fv for fv in _free_tirx_vars(denom, tirx=tirx) if _is_allowed(fv)]):
                continue
            max_const = _max_nonzero_const_magnitude(denom, tirx=tirx)

            def _next_v(cur_v: int, mc: int = max_const) -> int:
                return min(_MAX_SAFE_SPEC_INT, max(mc, 1) if cur_v <= 0 else max(cur_v * 2, mc, 1))

            if nonpos := [(k, cv) for fv in cands if (k := _find_key(fv)) is not None and (cv := _static_tirx_int(spec_var_map[k])) is not None and cv <= 0]:
                for k, cv in nonpos:
                    changed = _set_val(k, _next_v(cv)) or changed
                continue
            if len(cands) == 1 and (k := _find_key(cands[0])) is not None:
                cv = _static_tirx_int(spec_var_map[k])
                changed = _set_val(k, _next_v(0 if cv is None else int(cv))) or changed
                continue

            cand_keys = [(k, 0 if (cv := _static_tirx_int(spec_var_map[k])) is None else int(cv)) for fv in cands if (k := _find_key(fv)) is not None]
            best_key, best_next, best_trial = None, 0, (None if (simp_err or simp_val is None) else int(simp_val))
            for key, cur_int in cand_keys:
                other_prod = 1
                for ok, oi in cand_keys:
                    if ok is not key:
                        other_prod = min(_MAX_SAFE_SPEC_INT, other_prod * max(1, oi))
                dbl = _next_v(cur_int)
                boost = min(_MAX_SAFE_SPEC_INT, max(dbl, cur_int * max(max_const, 2) * 2, other_prod * max(max_const, 1)))
                ds = str(getattr(key, "dtype", "int32") or "int32")
                for cn in ([dbl] if boost == dbl else [boost, dbl]):
                    if cn == cur_int:
                        continue
                    tv, terr = _simp(tirx.stmt_functor.substitute(denom, {**spec_var_map, key: tirx.IntImm(ds, cn)}))
                    if not terr and tv is not None and ((tv > 0 and (best_trial is None or tv > best_trial or best_trial <= 0)) or (best_trial is None or tv > best_trial)):
                        best_trial, best_key, best_next = tv, key, cn
                        if tv > 0:
                            break
            if best_key is not None and best_trial is not None and best_trial > 0:
                changed = _set_val(best_key, best_next) or changed
                continue
            for key, cur_int in cand_keys:
                if cur_int > 1:
                    tv, terr = _simp(tirx.stmt_functor.substitute(denom, {**spec_var_map, key: tirx.IntImm(str(getattr(key, "dtype", "int32") or "int32"), 1)}))
                    if not terr and tv is not None and tv > 0:
                        best_key, best_next, best_trial = key, 1, tv
                        break
            if best_key is not None:
                changed = _set_val(best_key, best_next) or changed
            elif grow := [(k, ci) for k, ci in cand_keys if ci < _MAX_SAFE_SPEC_INT]:
                mk, mc = min(grow, key=lambda x: x[1])
                changed = _set_val(mk, _next_v(mc)) or changed
        if not changed:
            break


def _specialize_primfunc_for_anchors(
    entry_fn,
    mod_in,
    var_bindings: Mapping[str, int] | None,
    *,
    tvm_mod,
    tirx,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
):
    if not var_bindings and not ordered_tensor_shapes:
        return entry_fn, mod_in
    eff_bindings, var_obj_bindings = _augment_var_bindings_from_buffers(
        entry_fn, var_bindings, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes
    )
    if not eff_bindings and not var_obj_bindings:
        return entry_fn, mod_in
    lower_eff = {k.lower(): v for k, v in eff_bindings.items()}
    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    handle_data = (
        *buffer_map.keys(),
        *(b.data for b in buffer_map.values() if getattr(b, "data", None) is not None),
    )

    def _is_handle_or_data(vo) -> bool:
        return any(_same_tirx_var(vo, h) for h in handle_data)

    spec_var_map: dict[object, object] = {}
    recorded_vars: list[object] = []

    def _maybe_bind(vo) -> None:
        if not isinstance(vo, tirx.Var) or _is_handle_or_data(vo) or any(_same_tirx_var(vo, p) for p in recorded_vars):
            return
        bound = next((int(vv) for vobj, vv in var_obj_bindings if _same_tirx_var(vo, vobj)), None)
        if bound is None:
            bound = _match_binding_for_var(vo, eff_bindings, lower_eff)
        if bound is None:
            return
        if (ds := str(getattr(vo, "dtype", "int32") or "int32")) == "handle":
            return
        recorded_vars.append(vo)
        spec_var_map[vo] = tirx.IntImm(ds, int(bound))

    for p in getattr(entry_fn, "params", ()):
        _maybe_bind(p)
    buf_free_vars: list[object] = []
    for buf in buffer_map.values():
        for expr in (*getattr(buf, "shape", ()), *(getattr(buf, "strides", ()) or ()), buf.elem_offset):
            for fv in _free_tirx_vars(expr, tirx=tirx):
                _maybe_bind(fv)
                _add_unique_expr(buf_free_vars, fv)
    body_free_vars = _free_tirx_vars(entry_fn.body, tirx=tirx)
    for fv in body_free_vars:
        _maybe_bind(fv)

    max_numel = max((int(lower_eff[k]) for k in _NUMEL_ALIASES if k in lower_eff and int(lower_eff[k]) > 0), default=0)
    for buf in buffer_map.values():
        axis_ints: list[int] = []
        for de in getattr(buf, "shape", ()) or ():
            sv = _static_tirx_int(de)
            if sv is not None and sv > 0:
                axis_ints.append(int(sv))
            elif de in spec_var_map and (bsv := _static_tirx_int(spec_var_map[de])) is not None and bsv > 0:
                axis_ints.append(int(bsv))
            else:
                axis_ints = []
                break
        if axis_ints:
            max_numel = max(max_numel, math.prod(axis_ints))
    capped_fb = min(max_numel if max_numel > 0 else 1024, _BUFFER_FREE_AXIS_CAP)

    rem_vars: list[object] = []

    def _collect_rem(expr) -> None:
        if expr is None:
            return
        for fv in _free_tirx_vars(expr, tirx=tirx):
            ds = str(getattr(fv, "dtype", "int32") or "int32").lower()
            if (
                isinstance(fv, tirx.Var)
                and not _is_handle_or_data(fv)
                and ds != "handle"
                and "int" in ds
                and not any(_same_tirx_var(fv, p) for p in recorded_vars)
                and any(_same_tirx_var(fv, bfv) for bfv in body_free_vars)
            ):
                _add_unique_expr(rem_vars, fv)

    def _visit_rem(node) -> None:
        if isinstance(node, tirx.For):
            _collect_rem(node.extent)
        elif isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            for idx_e in getattr(node, "indices", ()) or ():
                _collect_rem(idx_e)
        elif isinstance(node, tirx.Call):
            if any(t in str(getattr(node.op, "name", node.op)).lower() for t in ("copy", "region", "access_ptr")):
                for arg in getattr(node, "args", ()) or ():
                    _collect_rem(arg)

    tirx.stmt_functor.post_order_visit(entry_fn.body, _visit_rem)
    param_vars = [p for p in getattr(entry_fn, "params", ()) if isinstance(p, tirx.Var) and not _is_handle_or_data(p)]
    for fv in rem_vars:
        fv_name = str(getattr(fv, "name_hint", "") or str(fv)).lower()
        in_buf = any(_same_tirx_var(fv, bv) for bv in buf_free_vars)
        in_par = any(_same_tirx_var(fv, pv) for pv in param_vars)
        is_tok = any(tok in fv_name for tok in _UNBOUND_SCALAR_NAME_TOKENS)
        ext = _UNBOUND_SCALAR_EXTENT_DEFAULT if ((in_par and not in_buf) or is_tok or not in_buf) else capped_fb
        recorded_vars.append(fv)
        spec_var_map[fv] = tirx.IntImm(str(getattr(fv, "dtype", "int32") or "int32"), int(ext))

    params_list = list(getattr(entry_fn, "params", ()))
    _ensure_positive_denominators(
        _collect_div_denominators(entry_fn, tirx=tirx),
        spec_var_map,
        recorded_vars,
        [*params_list, *buf_free_vars, *body_free_vars],
        is_handle_or_data=_is_handle_or_data,
        tirx=tirx,
        tvm_mod=tvm_mod,
    )
    if not spec_var_map:
        return entry_fn, mod_in

    extra = [v for v in recorded_vars if not any(_same_tirx_var(v, p) for p in params_list)]
    tmp_fn = (
        tirx.PrimFunc(
            [*params_list, *extra], entry_fn.body, ret_type=entry_fn.ret_type,
            buffer_map=entry_fn.buffer_map, attrs=entry_fn.attrs,
        )
        if extra
        else entry_fn
    )
    spec_fn = tmp_fn.specialize(spec_var_map)
    target_gvar = next((gv for gv, fn in mod_in.functions.items() if _same_tirx_var(fn, entry_fn)), None)
    if target_gvar is None:
        target_gvar = str(entry_fn.attrs.get("global_symbol", "main")) if entry_fn.attrs else "main"
    updated = {k: (spec_fn if k == target_gvar else v) for k, v in mod_in.functions.items()}
    updated.setdefault(target_gvar, spec_fn)
    spec_mod = tvm_mod.IRModule(updated, attrs=getattr(mod_in, "attrs", None))
    if hasattr(tirx, "transform") and hasattr(tirx.transform, "Simplify"):
        spec_mod = tirx.transform.Simplify()(spec_mod)
        spec_fn = spec_mod.functions.get(
            target_gvar, next((f for f in spec_mod.functions.values() if isinstance(f, tirx.PrimFunc)), spec_fn)
        )
    return spec_fn, spec_mod


_PASS_TAP_CLASS_CACHE: dict[object, type] = {}


def _pass_instrument_if_available(cls: type) -> type:
    try:
        from tilelang import tvm as tvm_mod  # noqa: PLC0415
    except ImportError:
        return cls
    dec = getattr(getattr(getattr(tvm_mod, "ir", None), "instrument", None), "pass_instrument", None)
    return dec(cls) if callable(dec) else cls


def _build_pass_tap_instance(tvm_mod: object) -> ReadOnlyAscendPassTap:
    """Instantiate ReadOnlyAscendPassTap, lazily wrapping with PassInstrument if needed."""
    inst_mod = getattr(getattr(tvm_mod, "ir", None), "instrument", None)
    pi_base = getattr(inst_mod, "PassInstrument", None)
    if isinstance(pi_base, type) and issubclass(ReadOnlyAscendPassTap, pi_base):
        return ReadOnlyAscendPassTap()
    dec = getattr(inst_mod, "pass_instrument", None)
    cache_key: object = pi_base if isinstance(pi_base, type) else dec
    if cache_key is not None and (cached := _PASS_TAP_CLASS_CACHE.get(cache_key)) is not None:
        return cached()
    if callable(dec):
        wrapped = dec(ReadOnlyAscendPassTap)
        if cache_key is not None:
            _PASS_TAP_CLASS_CACHE[cache_key] = wrapped
        return wrapped()
    return ReadOnlyAscendPassTap()


@_pass_instrument_if_available
class ReadOnlyAscendPassTap:
    """Read-only 4-tap TVM PassContext instrument for Ascend TileLang lowering."""

    TARGET_PASSES: tuple[str, ...] = (
        "tl.MaterializeMultiBuffer",
        "tl.MergeUBAllocations",
        "tl.InsertNd2Nz",
        "tl.NormalizeAscendFractalStorage",
        "tl.LowerTileOp",
        "tl.AscendLowerTileOp",
    )

    def __init__(self) -> None:
        self.observed_passes: list[str] = []
        self.captured_mods: dict[str, object] = {}

    def run_before_pass(self, mod, info) -> None:
        """Capture pre-merge IRModule immediately before tl.MergeUBAllocations."""
        if str(info.name) == "tl.MergeUBAllocations":
            self.captured_mods["pre_merge"] = mod

    def run_after_pass(self, mod, info) -> None:
        """Record executed pass name and capture read-only IR snapshots at target taps."""
        name = str(info.name)
        self.observed_passes.append(name)
        if name in self.TARGET_PASSES:
            self.captured_mods[name] = mod


@contextlib.contextmanager
def _ascend_compile_session(
    ctx_target: object, *, pass_configs: Mapping[str, object] | None = None, instruments: Sequence[object] = (),
):
    from tilelang import tvm  # noqa: PLC0415

    cur = tvm.target.Target.current(allow_none=True)
    cur_kind = str(getattr(getattr(cur, "kind", None), "name", "") or "")
    target_ctx = ctx_target if (cur is None or cur_kind != "ascend") else contextlib.nullcontext()
    with target_ctx, tvm.transform.PassContext(config=dict(pass_configs or {}), instruments=list(instruments)):
        yield


def _reconcile_post_lower_device_mod(
    device_mod, *, spaces: AscendSpaceFacts, tirx, DataTypeCls, pre_pipe_cycles: dict[str, int],
    pre_barriers: int, pre_min_gm_dma_bytes: int, pre_dma_burst_eff: float,
    pre_ub_scalar_loops: int, pre_dyn_offset_pipe: bool,
    explicit_aiv_split: int | None, effective_aiv_split: int,
) -> tuple[dict[str, int], int, int, float, int, bool]:
    dev_funcs = [fn for fn in device_mod.functions.values() if isinstance(fn, tirx.PrimFunc)] if (device_mod is not None and hasattr(device_mod, "functions")) else []
    if not dev_funcs:
        return (dict(pre_pipe_cycles), pre_barriers, pre_min_gm_dma_bytes, pre_dma_burst_eff, pre_ub_scalar_loops, pre_dyn_offset_pipe)

    post_barriers = post_ub_scalar = 0
    post_dyn, post_pipe_totals, eff_pools = pre_dyn_offset_pipe, {}, []
    for dev_fn in dev_funcs:
        dscan = _scan_primfunc_unified(
            dev_fn, spaces=spaces, tirx=tirx, DataTypeCls=DataTypeCls, aiv_split_factor=explicit_aiv_split,
            min_aiv_split=effective_aiv_split, post_lower=True, allow_sync_in_scalar_loops=True,
        )
        post_barriers += dscan.in_loop_pipe_all_barriers
        post_ub_scalar += dscan.ub_scalar_compute_loops
        post_dyn = post_dyn or dscan.uses_dynamic_offset_pipelining
        if dscan.min_gm_dma_bytes > 0:
            eff_pools.append((dscan.min_gm_dma_bytes, dscan.dma_burst_efficiency_ratio))
        for pk, pv in dscan.pipe_cycles.items():
            post_pipe_totals[pk] = post_pipe_totals.get(pk, 0) + pv

    if eff_pools:
        post_min_dma, post_eff = min(m for m, _ in eff_pools), min(e for _, e in eff_pools)
        reconciled_min_dma = min(pre_min_gm_dma_bytes, post_min_dma) if pre_min_gm_dma_bytes > 0 else post_min_dma
        reconciled_burst_eff = min(pre_dma_burst_eff, post_eff) if pre_min_gm_dma_bytes > 0 else post_eff
    else:
        reconciled_min_dma, reconciled_burst_eff = pre_min_gm_dma_bytes, pre_dma_burst_eff

    reconciled_pipes = dict(pre_pipe_cycles)
    if explicit_aiv_split != 1:
        pre_a0, pre_a1 = pre_pipe_cycles.get("AIV0.PIPE_V", 0), pre_pipe_cycles.get("AIV1.PIPE_V", 0)
        post_a0, post_a1 = post_pipe_totals.get("AIV0.PIPE_V", 0), post_pipe_totals.get("AIV1.PIPE_V", 0)
        if (post_a0 != post_a1) or (pre_a0 == 0 and pre_a1 == 0 and (post_a0 > 0 or post_a1 > 0)):
            if post_a0 > 0:
                reconciled_pipes["AIV0.PIPE_V"] = max(pre_a0, post_a0)
            if post_a1 > 0:
                reconciled_pipes["AIV1.PIPE_V"] = max(pre_a1, post_a1)

    return (
        reconciled_pipes, max(pre_barriers, post_barriers), reconciled_min_dma,
        reconciled_burst_eff, max(pre_ub_scalar_loops, post_ub_scalar), post_dyn,
    )


def _entry_and_mod(prim_func, tvm, tirx):
    if isinstance(prim_func, tvm.IRModule):
        funcs = [fn for fn in prim_func.functions.values() if isinstance(fn, tirx.PrimFunc)]
        if not funcs:
            raise ValueError("IRModule contains no PrimFunc entries")
        return funcs[0], prim_func
    sym = str(prim_func.attrs.get("global_symbol", "main")) if prim_func.attrs else "main"
    return prim_func, tvm.IRModule({sym: prim_func})


def specialize_primfunc_for_anchor(
    prim_func, *, var_bindings: Mapping[str, int] | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
):
    """Specialize a TileLang PrimFunc or IRModule for one anchor shape without lowering."""
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    entry_fn, mod_in = _entry_and_mod(prim_func, tvm, tirx)
    return _specialize_primfunc_for_anchors(entry_fn, mod_in, var_bindings, tvm_mod=tvm, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes)


def inspect_tilelang_primfunc(
    prim_func, *, target: AscendTarget | None = None, pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, aiv_split_factor: int | None = None,
    unreserved_ubuf: bool = False, var_bindings: Mapping[str, int] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> EnvelopeInspectionResult:
    """Inspect a TileLang PrimFunc using single-pass IR scan and Two-Snapshot IR Diff."""
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415
    from tilelang.engine.lower import lower_to_host_device_ir  # noqa: PLC0415
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    spaces = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    entry_fn, mod_in = _entry_and_mod(prim_func, tvm, tirx)
    raw_cfg, _ = _extract_primfunc_pass_configs_and_flags(entry_fn, mod_in, pass_configs=pass_configs, compile_flags=compile_flags)
    ctx = create_backend_context("ascend", None, "auto")
    tap, lowering_error, device_mod = _build_pass_tap_instance(tvm), None, None

    with _ascend_compile_session(ctx.target, pass_configs=raw_cfg, instruments=[tap]):
        entry_fn, mod_in = _specialize_primfunc_for_anchors(entry_fn, mod_in, var_bindings, tvm_mod=tvm, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes)
        cfg, merged_flags = _extract_primfunc_pass_configs_and_flags(entry_fn, mod_in, pass_configs=pass_configs, compile_flags=compile_flags)
        ubuf_flag = "--cce-disable-vf-stack-reserved-ubuf"
        unreserved_hint = bool(unreserved_ubuf or any(ubuf_flag in f for f in merged_flags) or _attrs_contain_flag(cfg, ubuf_flag))
        scan = _scan_primfunc_unified(
            entry_fn, mod_in, spaces=spaces, tirx=tirx, DataTypeCls=TLDataType,
            aiv_split_factor=aiv_split_factor, unreserved_ubuf=unreserved_hint,
            var_bindings=var_bindings, tvm_mod=tvm, ordered_tensor_shapes=ordered_tensor_shapes,
        )
        capacities = _scope_capacities(spaces, has_simtvf=scan.has_simtvf, unreserved_ubuf=scan.uses_unreserved_ubuf)
        try:
            with tvm.transform.PassContext(config=cfg, instruments=[tap]):
                _host_mod, device_mod, _params, _tgt, _htgt = lower_to_host_device_ir(mod_in, ctx)
        except Exception as exc:
            lowering_error = f"{type(exc).__name__}: {exc}"

    captured, observed = tap.captured_mods, tap.observed_passes
    dev_snaps = _extract_tilelang_buffers_from_mod(device_mod, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=True)
    mat_snaps = _extract_tilelang_buffers_from_mod(captured.get("tl.MaterializeMultiBuffer"), tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=False) or dev_snaps
    pre_merge_mod = next((m for k in ("pre_merge", "tl.LowerTileOp", "tl.AscendLowerTileOp", "tl.InsertNd2Nz", "tl.NormalizeAscendFractalStorage", "tl.MaterializeMultiBuffer") if (m := captured.get(k)) is not None), device_mod)
    lowered_snaps = _extract_tilelang_buffers_from_mod(pre_merge_mod, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=True)
    post_merge_snaps = _extract_tilelang_buffers_from_mod(captured.get("tl.MergeUBAllocations") or device_mod, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=True)

    merged_peak: dict[str, int] = {}
    for snap in post_merge_snaps:
        merged_peak[snap.scope] = merged_peak.get(snap.scope, 0) + snap.bytes
    dev_totals: dict[str, int] = {}
    for snap in dev_snaps:
        dev_totals[snap.scope] = dev_totals.get(snap.scope, 0) + snap.bytes
    for scope, dbytes in dev_totals.items():
        merged_peak[scope] = max(merged_peak.get(scope, 0), dbytes)

    degraded = _detect_degraded_buffers(scan.explicit_snapshots, mat_snaps, scan.requested_multibuffer)
    waterlines = _diff_snapshots_by_scope(scan.explicit_snapshots, lowered_snaps, merged_peak, capacities, degraded)
    opaque = (*scan.opaque_regions, "lowering_rejected") if (lowering_error is not None and "lowering_rejected" not in scan.opaque_regions) else scan.opaque_regions

    pipe_cycles, rec_barriers, rec_min_dma, rec_burst_eff, rec_ub_scalar, rec_dyn_offset = _reconcile_post_lower_device_mod(
        device_mod, spaces=spaces, tirx=tirx, DataTypeCls=TLDataType, pre_pipe_cycles=scan.pipe_cycles,
        pre_barriers=scan.in_loop_pipe_all_barriers, pre_min_gm_dma_bytes=scan.min_gm_dma_bytes,
        pre_dma_burst_eff=scan.dma_burst_efficiency_ratio, pre_ub_scalar_loops=scan.ub_scalar_compute_loops,
        pre_dyn_offset_pipe=scan.uses_dynamic_offset_pipelining, explicit_aiv_split=aiv_split_factor,
        effective_aiv_split=scan.aiv_split_factor,
    )

    has_overflow = any(wl.overflow_bytes > 0 for wl in waterlines.values())
    solved_ii = _solve_pipelined_loops_ii(
        scan.pipelined_loops, scan.buf_meta, spaces, tirx=tirx, DataTypeCls=TLDataType,
        aiv_split_factor=scan.aiv_split_factor, has_simtvf=scan.has_simtvf,
        unreserved_ubuf=scan.uses_unreserved_ubuf, degraded_names=degraded,
    ) if (not scan.illegal_dma and not has_overflow and "lowering_rejected" not in opaque) else None
    analysis_mode = "exact_modulo_z3" if (solved_ii is not None and not opaque) else "readonly_envelope"
    wave_occ = 1.0 if scan.grid_is_symbolic else _compute_wave_occupancy(scan.grid_blocks, spaces, pipe_cycles, scan.aiv_split_factor)

    l1_wl, eff_l1_risk = waterlines.get("L1"), scan.l1_bank_conflict_risk
    if l1_wl is not None and l1_wl.peak_allocated_bytes > 0 and eff_l1_risk > 0.0:
        l1_full, l1_half = max(1, spaces.l1_bytes), max(1, spaces.l1_bytes // 2)
        if l1_wl.peak_allocated_bytes > l1_half:
            eff_l1_risk = max(eff_l1_risk, round(min(1.0, l1_wl.peak_allocated_bytes / float(l1_full)), 4))
        elif len(l1_wl.explicit_buffers) >= 2:
            eff_l1_risk = max(eff_l1_risk, round(min(0.5, l1_wl.peak_allocated_bytes / float(l1_half)), 4))

    ma_prof = extract_microarch_profile_from_tirx(entry_fn, var_bindings=var_bindings)
    ledger = build_objective_physical_ledger(
        analysis_mode=analysis_mode, is_legal=(not scan.illegal_dma) and (lowering_error is None),
        steady_state_ii=solved_ii, pipe_cycles=pipe_cycles, waterlines=waterlines,
        mte2_inflight_per_aiv=scan.mte2_inflight, unaligned_dma_padding_bytes=scan.padding_bytes,
        opaque_dynamic_regions=opaque, grid_blocks=scan.grid_blocks, wave_occupancy=wave_occ,
        gm_roundtrip_bytes=scan.gm_roundtrip_bytes, dominant_contig_bytes=scan.dominant_contig_bytes,
        effective_cube_m=scan.effective_cube_m, has_indirect_gather=scan.has_indirect_gather,
        has_lowbit_quant=scan.has_lowbit_quant, has_online_rescale_loop=scan.has_online_rescale_loop,
        in_loop_ub_reload_stores=scan.in_loop_ub_reload_stores, uses_unreserved_ubuf=scan.uses_unreserved_ubuf,
        grid_is_symbolic=scan.grid_is_symbolic, min_gm_dma_bytes=rec_min_dma,
        dma_burst_efficiency_ratio=rec_burst_eff, uses_dynamic_offset_pipelining=rec_dyn_offset,
        has_unlimit_memory_attr=scan.has_unlimit_memory_attr, has_coupled_reduce_sfu_accum=scan.has_coupled_reduce_sfu_accum,
        active_grid_blocks=scan.active_grid_blocks, in_loop_pipe_all_barriers=rec_barriers,
        ub_bank_conflict_risk=scan.ub_bank_conflict_risk, l1_bank_conflict_risk=eff_l1_risk,
        ub_scalar_compute_loops=rec_ub_scalar, microarch_profile=ma_prof.to_dict(),
    )
    return EnvelopeInspectionResult(
        ledger=ledger, explicit_snapshot=scan.explicit_snapshots, lowered_snapshot=lowered_snaps,
        merged_peak_bytes_by_scope=merged_peak, lowered_passes_observed=tuple(observed), lowering_error=lowering_error,
    )


__all__ = [
    "BufferSnapshot",
    "EnvelopeInspectionResult",
    "ReadOnlyAscendPassTap",
    "inspect_tilelang_primfunc",
    "specialize_primfunc_for_anchor",
]
