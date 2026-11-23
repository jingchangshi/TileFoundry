"""Mode B read-only dynamic envelope accounting and Two-Snapshot IR Diff."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Edge,
    Reuse,
    Task,
    tf_adapter,
)
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
from tilefoundry_ascend.ascend.target import (
    AscendSpaceFacts,
    AscendTarget,
    NodeResourceMask,
)

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
    """Instantiate 3-4 corner shapes for Stage A pre-check."""
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


_EXTERN_CALL_TAILS = frozenset(
    ("call_extern", "call_pure_extern", "call_intrin", "call_llvm_intrin")
)


def _op_name_str(op: object) -> str:
    """Return the raw string name of a TIR call op."""
    return str(getattr(op, "name", op))


def _unwrapped_extern_symbol(call: object) -> str | None:
    """Return target symbol name when *call* is call_extern/call_intrin with string args[0]."""
    op_name = _op_name_str(getattr(call, "op", ""))
    tail = op_name.rsplit(".", 1)[-1].lower()
    if tail not in _EXTERN_CALL_TAILS:
        return None
    args = getattr(call, "args", ())
    if not args:
        return None
    first = args[0]
    if isinstance(first, str) and first:
        return first
    val = getattr(first, "value", None)
    if isinstance(val, str) and val:
        return val
    return None


def _effective_tirx_call_name(call: object) -> str:
    """Resolve effective op name, unwrapping call_extern/call_intrin target symbols."""
    sym = _unwrapped_extern_symbol(call)
    if sym is not None:
        return sym
    return _op_name_str(getattr(call, "op", ""))


def _effective_tirx_call_args(call: object) -> list[object]:
    """Return target arguments of a TIR call, skipping extern symbol name if unwrapped."""
    args = list(getattr(call, "args", ()))
    if _unwrapped_extern_symbol(call) is not None and args:
        return args[1:]
    return args


def _static_shape_ints(shape: object) -> tuple[int, ...] | None:
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(dim)
        elif hasattr(dim, "value") and isinstance(getattr(dim, "value"), int):
            dims.append(int(dim.value))
        else:
            return None
    return tuple(dims)


def _dtype_bytes(dtype_obj: object, DataTypeCls) -> float:
    dt = DataTypeCls(str(dtype_obj))
    return (dt.bits * dt.lanes) / 8.0


def _buffer_bytes(shape: tuple[int, ...] | None, dtype_obj: object, DataTypeCls) -> int:
    if shape is None or not shape:
        return 0
    elem_bytes = _dtype_bytes(dtype_obj, DataTypeCls)
    return int(math.ceil(math.prod(shape) * elem_bytes))


def _is_lowbit_dtype_name(dtype_str: str) -> bool:
    """Return whether a dtype name denotes an 8-bit or 4-bit quantized format."""
    base = re.sub(r"x\d+$", "", dtype_str.strip().lower())
    return base in (
        "int8",
        "uint8",
        "i8",
        "u8",
        "int4",
        "uint4",
        "i4",
        "u4",
    ) or base.startswith(("fp8", "float8", "f8", "fp4", "float4", "f4"))


def _compute_wave_occupancy(
    grid_blocks: int,
    spaces: AscendSpaceFacts,
    pipe_cycles: dict[str, int],
    aiv_split_factor: int,
) -> float:
    """Compute grid wave occupancy against the active AIC or AIV core pool."""
    uses_aic = pipe_cycles.get("AIC.PIPE_M", 0) > 0 or aiv_split_factor >= 2
    core_pool = spaces.num_aic_cores if uses_aic else spaces.num_aiv_cores
    return round(grid_blocks / float(max(1, core_pool)), 4)


def _select_dominant_contig_bytes(
    candidates: Sequence[tuple[int | None, int, int, int, int]],
) -> int:
    """Return trailing contiguous dimension bytes of the largest non-scalar tensor."""
    non_scalars = [item for item in candidates if item[0] is None or item[0] != 1]
    if not non_scalars:
        return 0
    payloads = [item for item in non_scalars if item[1] > 1]
    pool = payloads if payloads else non_scalars
    best = max(pool, key=lambda item: (item[2], item[3], item[4]))
    return best[4]


def _attrs_contain_flag(attrs: object, flag: str) -> bool:
    """Recursively check whether TVM/TIR attrs contain *flag*."""
    if attrs is None:
        return False
    if isinstance(attrs, str):
        return flag in attrs
    if hasattr(attrs, "items"):
        return any(
            (flag in str(_k)) or _attrs_contain_flag(v, flag)
            for _k, v in attrs.items()
        )
    if isinstance(attrs, (tuple, list)) or (
        hasattr(attrs, "__iter__") and not isinstance(attrs, (bytes, bytearray))
    ):
        try:
            return any(_attrs_contain_flag(item, flag) for item in attrs)
        except TypeError:
            return flag in str(attrs)
    return flag in str(attrs)


def _is_tilelang_exp_op(op_name: str) -> bool:
    lower = op_name.lower()
    return "vexp" in lower or "exp2" in lower or lower == "exp" or lower.endswith(".exp")


def _collect_tilelang_expr_rw(
    expr,
    *,
    tirx,
) -> tuple[list[tuple[str, str, str]], bool, list[str], list[str]]:
    """Collect ordered buffer accesses, exp flags, and UB vld/vsts buffer names."""
    events: list[tuple[str, str, str]] = []
    has_exp = [False]
    ub_loads: list[str] = []
    ub_stores: list[str] = []

    def _visit_indices(indices) -> None:
        for idx in indices:
            if isinstance(idx, (tirx.And, tirx.Or, tirx.Not)):
                 raise ValueError("Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / tvm.tirx.any instead")
            sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(idx, tirx=tirx)
            events.extend(sub_ev)
            if sub_exp:
                has_exp[0] = True
            ub_loads.extend(sub_ul)
            ub_stores.extend(sub_us)

    def _record_ub_access(arg, *, is_store: bool) -> None:
        sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(arg, tirx=tirx)
        events.extend(sub_ev)
        if sub_exp:
            has_exp[0] = True
        ub_loads.extend(sub_ul)
        ub_stores.extend(sub_us)
        for buf_name, scope, _rw in sub_ev:
            if scope == "UB":
                if is_store:
                    ub_stores.append(buf_name)
                else:
                    ub_loads.append(buf_name)

    def _pre(node):
        if isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            call_args = _effective_tirx_call_args(node)
            if _is_tilelang_exp_op(op_name):
                has_exp[0] = True
            if "vld" in op_name and call_args:
                _record_ub_access(call_args[0], is_store=False)
                for rest in call_args[1:]:
                    sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(rest, tirx=tirx)
                    events.extend(sub_ev)
                    if sub_exp:
                        has_exp[0] = True
                    ub_loads.extend(sub_ul)
                    ub_stores.extend(sub_us)
                return False
            if "vsts" in op_name and call_args:
                _record_ub_access(call_args[0], is_store=True)
                for rest in call_args[1:]:
                    sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(rest, tirx=tirx)
                    events.extend(sub_ev)
                    if sub_exp:
                        has_exp[0] = True
                    ub_loads.extend(sub_ul)
                    ub_stores.extend(sub_us)
                return False
            if (
                "region" in op_name
                and len(node.args) >= 2
                and isinstance(node.args[0], tirx.BufferLoad)
            ):
                bl = node.args[0]
                _visit_indices(bl.indices)
                scope = _map_tilelang_scope(bl.buffer.scope())
                rw = _static_tirx_int(node.args[1]) or 1
                if scope is not None:
                    if rw & 1:
                        events.append((bl.buffer.name, scope, "read"))
                    if rw & 2:
                        events.append((bl.buffer.name, scope, "write"))
                return False
            if (
                "access_ptr" in op_name
                and len(node.args) >= 3
                and isinstance(node.args[0], tirx.BufferLoad)
            ):
                bl = node.args[0]
                _visit_indices(bl.indices)
                scope = _map_tilelang_scope(bl.buffer.scope())
                rw = _static_tirx_int(node.args[2]) or 1
                if scope is not None:
                    if rw & 1:
                        events.append((bl.buffer.name, scope, "read"))
                    if rw & 2:
                        events.append((bl.buffer.name, scope, "write"))
                return False
            if "copy" in op_name and len(call_args) >= 2:
                for arg, default_rw in ((call_args[0], "read"), (call_args[1], "write")):
                    if isinstance(arg, tirx.BufferRegion):
                        scope = _map_tilelang_scope(arg.buffer.scope())
                        if scope is not None:
                            events.append((arg.buffer.name, scope, default_rw))
                    elif isinstance(arg, tirx.BufferLoad):
                        _visit_indices(arg.indices)
                        scope = _map_tilelang_scope(arg.buffer.scope())
                        if scope is not None:
                            events.append((arg.buffer.name, scope, default_rw))
                    else:
                        sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(
                            arg, tirx=tirx
                        )
                        events.extend(sub_ev)
                        if sub_exp:
                            has_exp[0] = True
                        ub_loads.extend(sub_ul)
                        ub_stores.extend(sub_us)
                return False
        elif isinstance(node, tirx.BufferLoad):
            _visit_indices(node.indices)
            scope = _map_tilelang_scope(node.buffer.scope())
            if scope is not None:
                events.append((node.buffer.name, scope, "read"))
                if scope == "UB":
                    ub_loads.append(node.buffer.name)
            return False
        elif isinstance(node, tirx.BufferRegion):
            scope = _map_tilelang_scope(node.buffer.scope())
            if scope is not None:
                events.append((node.buffer.name, scope, "read"))
            return False
        return True

    tirx.stmt_functor.pre_order_visit(expr, _pre)
    return events, has_exp[0], ub_loads, ub_stores


def _collect_tilelang_stmt_events(
    stmt,
    *,
    tirx,
) -> tuple[list[tuple[str, str, str]], bool, set[str], set[str]]:
    """Collect ordered accesses, exp presence, and UB loads/stores across a TIR statement."""
    events: list[tuple[str, str, str]] = []
    has_exp = [False]
    ub_loads: set[str] = set()
    ub_stores: set[str] = set()

    def _absorb_expr(expr) -> None:
        if expr is None:
            return
        sub_ev, sub_exp, sub_ul, sub_us = _collect_tilelang_expr_rw(expr, tirx=tirx)
        events.extend(sub_ev)
        if sub_exp:
            has_exp[0] = True
        ub_loads.update(sub_ul)
        ub_stores.update(sub_us)

    def _walk(node) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child)
            return
        if isinstance(node, tirx.Bind):
            _absorb_expr(node.value)
            if hasattr(node, "body"):
                _walk(node.body)
            return
        if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body)
            return
        if isinstance(node, (tirx.SBlockRealize,)):
            _walk(node.block)
            return
        if isinstance(node, (tirx.SBlock, tirx.AttrStmt)):
            _walk(node.body)
            return
        if isinstance(node, (tirx.For, tirx.While)):
            if isinstance(node, tirx.While):
                _absorb_expr(node.condition)
            _walk(node.body)
            return
        if isinstance(node, tirx.IfThenElse):
            _absorb_expr(node.condition)
            _walk(node.then_case)
            if node.else_case is not None:
                _walk(node.else_case)
            return
        if isinstance(node, tirx.BufferStore):
            _absorb_expr(node.value)
            for idx in node.indices:
                _absorb_expr(idx)
            scope = _map_tilelang_scope(node.buffer.scope())
            if scope is not None:
                events.append((node.buffer.name, scope, "write"))
                if scope == "UB":
                    ub_stores.add(node.buffer.name)
            return
        if isinstance(node, tirx.Evaluate):
            _absorb_expr(node.value)

    _walk(stmt)
    return events, has_exp[0], ub_loads, ub_stores


def _has_loop_carried_accumulator(
    events: Sequence[tuple[str, object, str]],
    local_keys: set[str],
) -> bool:
    """Return True if any non-local buffer is read before write and written in *events*."""
    written_so_far: set[str] = set()
    read_before_write: set[str] = set()
    written_in_loop: set[str] = set()
    for buf_key, _scope, rw in events:
        if buf_key in local_keys:
            continue
        if rw == "read":
            if buf_key not in written_so_far:
                read_before_write.add(buf_key)
        elif rw == "write":
            written_so_far.add(buf_key)
            written_in_loop.add(buf_key)
    return bool(read_before_write & written_in_loop)


@dataclass(frozen=True)
class _TelemetryFeatures:
    grid_blocks: int
    gm_roundtrip_bytes: int
    dominant_contig_bytes: int
    effective_cube_m: int
    has_indirect_gather: bool
    has_lowbit_quant: bool
    has_online_rescale_loop: bool
    in_loop_ub_reload_stores: int
    uses_unreserved_ubuf: bool
    grid_is_symbolic: bool = False
    min_gm_dma_bytes: int = 0
    dma_burst_efficiency_ratio: float = 1.0
    uses_dynamic_offset_pipelining: bool = False
    has_unlimit_memory_attr: bool = False
    has_coupled_reduce_sfu_accum: bool = False
    active_grid_blocks: int = 0
    in_loop_pipe_all_barriers: int = 0
    ub_bank_conflict_risk: float = 0.0
    l1_bank_conflict_risk: float = 0.0
    ub_scalar_compute_loops: int = 0


def _collect_tirx_expr_vars(expr, *, tirx) -> list[object]:
    found: list[object] = []
    seen_ids: set[int] = set()

    def _visit(node) -> None:
        if isinstance(node, tirx.Var):
            nid = id(node)
            if nid not in seen_ids:
                seen_ids.add(nid)
                found.append(node)

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found


def _free_tirx_vars(expr, *, tirx) -> list[object]:
    if hasattr(tirx, "analysis") and hasattr(tirx.analysis, "undefined_vars"):
        try:
            return list(tirx.analysis.undefined_vars(expr))
        except Exception:
            pass
    return _collect_tirx_expr_vars(expr, tirx=tirx)


def _substitute_let_env(expr, let_env: Mapping[object, object], *, tirx):
    cur_expr = expr
    for _ in range(16):
        if not let_env:
            break
        free_vars = _free_tirx_vars(cur_expr, tirx=tirx)
        expand_map = {var: let_env[var] for var in free_vars if var in let_env}
        if not expand_map:
            break
        next_expr = tirx.stmt_functor.substitute(cur_expr, expand_map)
        if hasattr(next_expr, "same_as") and next_expr.same_as(cur_expr):
            break
        cur_expr = next_expr
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
    for raw_key, raw_val in (var_bindings or {}).items():
        if isinstance(raw_val, int) and not isinstance(raw_val, bool):
            lower_counts.setdefault(str(raw_key).lower(), []).append(int(raw_val))

    for fv in _free_tirx_vars(expr, tirx=tirx):
        if exclude_var is not None:
            if (hasattr(fv, "same_as") and fv.same_as(exclude_var)) or fv is exclude_var:
                continue
        bound: int | None = None
        for v_obj, v_val in var_obj_bindings:
            if (hasattr(fv, "same_as") and fv.same_as(v_obj)) or fv is v_obj:
                bound = int(v_val)
                break
        exact_name = str(getattr(fv, "name", "") or "")
        hint_name = str(getattr(fv, "name_hint", "") or "")
        if bound is None and var_bindings:
            for cand_name in (exact_name, hint_name):
                if cand_name and cand_name in var_bindings:
                    raw_val = var_bindings[cand_name]
                    if isinstance(raw_val, int) and not isinstance(raw_val, bool):
                        bound = int(raw_val)
                        break
        if bound is None:
            lookup = (exact_name or hint_name).lower()
            if lookup:
                matches = lower_counts.get(lookup, [])
                if len(matches) == 1:
                    bound = matches[0]
        if bound is not None:
            dtype_str = str(getattr(fv, "dtype", "int32") or "int32")
            vmap[fv] = tirx.IntImm(dtype_str, bound)
    return vmap


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
    ("total_elements", "total_elems", "n_elems", "num_elements", "numel"),
)


def _lookup_stem_binding(
    cname: str,
    eff: Mapping[str, int],
    lower_eff: Mapping[str, int],
) -> int | None:
    if cname and cname in eff:
        return eff[cname]
    lc = cname.lower()
    if lc and lc in lower_eff:
        return lower_eff[lc]
    for suffix in ("_dim", "_size", "_len", "_count", "_rows", "_cols"):
        if lc.endswith(suffix):
            stem = lc[: -len(suffix)]
            if stem in lower_eff:
                return lower_eff[stem]
        elif lc and f"{lc}{suffix}" in lower_eff:
            return lower_eff[f"{lc}{suffix}"]
    for group in _DIM_ALIAS_GROUPS:
        if lc in group:
            for alias in group:
                if alias in lower_eff:
                    return lower_eff[alias]
    return None


def _match_binding_for_var(
    var,
    eff: Mapping[str, int],
    lower_eff: Mapping[str, int],
) -> int | None:
    """Resolve an integer binding for a TIR Var by exact name, suffix strip, or canonical alias."""
    exact_name = str(getattr(var, "name", "") or "")
    hint_name = str(getattr(var, "name_hint", "") or "")
    for cname in (exact_name, hint_name):
        direct = _lookup_stem_binding(cname, eff, lower_eff)
        if direct is not None:
            return direct
        lc = cname.lower()
        pad_stem = ""
        if "_pad" in lc:
            pad_stem = lc[: lc.find("_pad")]
        elif lc.startswith("pad_"):
            pad_stem = lc[4:]
        if pad_stem:
            base_val = _lookup_stem_binding(pad_stem, eff, lower_eff)
            if base_val is not None and base_val > 0:
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
    eff: dict[str, int] = {
        str(k): int(v)
        for k, v in (var_bindings or {}).items()
        if isinstance(v, int) and not isinstance(v, bool)
    }
    lower_eff = {k.lower(): v for k, v in eff.items()}
    obj_bindings: list[tuple[object, int]] = []
    seen_ids: set[int] = set()

    def _is_already_bound(var) -> bool:
        return id(var) in seen_ids

    def _bind_var(var, val: int) -> None:
        if id(var) not in seen_ids:
            seen_ids.add(id(var))
            obj_bindings.append((var, int(val)))
        for cname in (
            str(getattr(var, "name", "") or ""),
            str(getattr(var, "name_hint", "") or ""),
        ):
            if cname and cname not in eff:
                eff[cname] = int(val)
                lower_eff.setdefault(cname.lower(), int(val))

    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    tensor_idx = 0
    for param in getattr(entry_fn, "params", ()):
        buf = buffer_map.get(param)
        if buf is None:
            if isinstance(param, tirx.Var):
                matched = _match_binding_for_var(param, eff, lower_eff)
                if matched is not None:
                    _bind_var(param, matched)
            continue
        shape = list(getattr(buf, "shape", ()))
        strides = list(getattr(buf, "strides", ()) or ())
        for dim_expr in (*shape, *strides):
            for fv in _free_tirx_vars(dim_expr, tirx=tirx):
                matched = _match_binding_for_var(fv, eff, lower_eff)
                if matched is not None:
                    _bind_var(fv, matched)
        if ordered_tensor_shapes is not None and tensor_idx < len(ordered_tensor_shapes):
            pos_shape = ordered_tensor_shapes[tensor_idx]
            if len(shape) == len(pos_shape):
                for dim_expr, dim_val in zip(shape, pos_shape):
                    if (
                        isinstance(dim_expr, tirx.Var)
                        and not _is_already_bound(dim_expr)
                        and isinstance(dim_val, int)
                        and not isinstance(dim_val, bool)
                    ):
                        _bind_var(dim_expr, max(1, int(dim_val)))
        tensor_idx += 1
        if len(shape) == 2:
            d0, d1 = shape
            if isinstance(d0, tirx.Var) and not _is_already_bound(d0) and _static_tirx_int(d1) is not None:
                row_val = lower_eff.get("q_total_rows", lower_eff.get("m_rows", lower_eff.get("m")))
                if row_val is not None:
                    _bind_var(d0, row_val)
        elif len(shape) == 4:
            d0, d1, d2, _d3 = shape
            if isinstance(d0, tirx.Var) and not _is_already_bound(d0):
                b_val = lower_eff.get("b_dim", lower_eff.get("b"))
                if b_val is not None:
                    _bind_var(d0, b_val)
            kv_val = lower_eff.get("s_kv")
            if kv_val is not None:
                if isinstance(d1, tirx.Var) and not _is_already_bound(d1) and _static_tirx_int(d2) is not None:
                    _bind_var(d1, kv_val)
                elif isinstance(d2, tirx.Var) and not _is_already_bound(d2) and _static_tirx_int(d1) is not None:
                    _bind_var(d2, kv_val)

    numel_aliases = ("total_elements", "total_elems", "n_elems", "num_elements", "numel")
    if not any(alias in lower_eff for alias in numel_aliases):
        synth_numel = 0
        for param in getattr(entry_fn, "params", ()):
            buf = buffer_map.get(param)
            if buf is None:
                continue
            buf_shape = list(getattr(buf, "shape", ()) or ())
            if not buf_shape:
                continue
            axis_vals: list[int] = []
            all_known = True
            for dim_expr in buf_shape:
                sval = _static_tirx_int(dim_expr)
                if sval is not None and sval > 0:
                    axis_vals.append(int(sval))
                elif isinstance(dim_expr, tirx.Var):
                    mval = _match_binding_for_var(dim_expr, eff, lower_eff)
                    if mval is not None and mval > 0:
                        axis_vals.append(int(mval))
                    else:
                        all_known = False
                        break
                else:
                    all_known = False
                    break
            if all_known and axis_vals:
                prod_val = math.prod(axis_vals)
                if prod_val > synth_numel:
                    synth_numel = prod_val
        if synth_numel <= 0:
            b_val = _lookup_stem_binding("b", eff, lower_eff)
            h_val = _lookup_stem_binding("h", eff, lower_eff)
            s_val = _lookup_stem_binding("s", eff, lower_eff)
            m_val = _lookup_stem_binding("m", eff, lower_eff)
            n_val = _lookup_stem_binding("n", eff, lower_eff)
            d_val = _lookup_stem_binding("d", eff, lower_eff)
            k_val = _lookup_stem_binding("k", eff, lower_eff)
            row_axes = 1
            has_axis = False
            if m_val is not None and m_val > 0:
                if b_val and s_val and m_val >= b_val * s_val:
                    row_axes = m_val
                else:
                    row_axes = (b_val or 1) * (h_val or 1) * m_val
                has_axis = True
            elif s_val is not None and s_val > 0:
                row_axes = (b_val or 1) * (h_val or 1) * s_val
                has_axis = True
            elif b_val is not None and b_val > 0:
                row_axes = b_val * (h_val or 1)
                has_axis = True
            col_val = n_val or d_val or k_val
            if col_val is not None and col_val > 0:
                synth_numel = row_axes * col_val
            elif has_axis:
                synth_numel = row_axes
        if synth_numel > 0:
            for alias in numel_aliases:
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
    static_val = _static_tirx_int(expr)
    if static_val is not None:
        return static_val, False

    cur_expr = _substitute_let_env(expr, let_env, tirx=tirx)
    vmap = _build_free_var_int_map(
        cur_expr,
        var_bindings,
        var_obj_bindings,
        tirx=tirx,
    )
    if vmap:
        cur_expr = tirx.stmt_functor.substitute(cur_expr, vmap)

    if tvm_mod is not None and hasattr(tvm_mod, "arith") and hasattr(tvm_mod.arith, "Analyzer"):
        try:
            cur_expr = tvm_mod.arith.Analyzer().simplify(cur_expr)
        except Exception:
            pass

    resolved = _static_tirx_int(cur_expr)
    if resolved is not None:
        return resolved, False
    return None, True


def _resolve_persistent_active_blocks(
    body,
    static_grid: int,
    var_bindings: Mapping[str, int] | None,
    var_obj_bindings: Sequence[tuple[object, int]],
    *,
    tirx,
    tvm_mod: object | None,
) -> tuple[int, int] | None:
    if static_grid <= 0 or static_grid > 512 or (not var_bindings and not var_obj_bindings):
        return None
    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)
    alloc_cls = getattr(tirx, "Allocate", None)

    def _same_var(a, b) -> bool:
        return bool((hasattr(a, "same_as") and a.same_as(b)) or a is b)

    def _find_blockidx_scope(node, env: dict[object, object]):
        if node is None:
            return None
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            if tb is not None and "blockidx" in str(getattr(tb, "thread_tag", "")).lower():
                return node.loop_var, node.body, dict(env)
            return _find_blockidx_scope(node.body, env)
        if isinstance(node, tirx.AttrStmt):
            if "thread_extent" in str(node.attr_key).lower():
                tag = str(getattr(node.node, "thread_tag", "")).lower()
                if "blockidx" in tag and getattr(node.node, "var", None) is not None:
                    return node.node.var, node.body, dict(env)
            return _find_blockidx_scope(node.body, env)
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(child.value, seq_env, tirx=tirx)
                    if hasattr(child, "body"):
                        found = _find_blockidx_scope(child.body, dict(seq_env))
                        if found is not None:
                            return found
                else:
                    found = _find_blockidx_scope(child, seq_env)
                    if found is not None:
                        return found
            return None
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(env)
            next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
            return _find_blockidx_scope(node.body, next_env)
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(env)
            next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
            if hasattr(node, "body"):
                return _find_blockidx_scope(node.body, next_env)
            return None
        if isinstance(node, tirx.SBlockRealize):
            return _find_blockidx_scope(node.block, env)
        if isinstance(node, tirx.SBlock):
            return _find_blockidx_scope(node.body, env)
        if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)) or (
            alloc_cls is not None and isinstance(node, alloc_cls)
        ):
            return _find_blockidx_scope(node.body, env) if hasattr(node, "body") else None
        return None

    scope = _find_blockidx_scope(body, {})
    if scope is None:
        return None
    block_var, inner_body, init_env = scope
    work_items: list[tuple[str, object]] = []

    def _depends_on_block_var(expr) -> bool:
        return any(_same_var(v, block_var) for v in _free_tirx_vars(expr, tirx=tirx))

    def _collect_dispatch(node, env: dict[object, object]) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(child.value, seq_env, tirx=tirx)
                    if hasattr(child, "body"):
                        _collect_dispatch(child.body, dict(seq_env))
                else:
                    _collect_dispatch(child, seq_env)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(env)
            next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
            _collect_dispatch(node.body, next_env)
            return
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(env)
            next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
            if hasattr(node, "body"):
                _collect_dispatch(node.body, next_env)
            return
        if isinstance(node, tirx.IfThenElse):
            cond_sub = _substitute_let_env(node.condition, env, tirx=tirx)
            if _depends_on_block_var(cond_sub) and node.else_case is None:
                work_items.append(("guard", cond_sub))
                return
            _collect_dispatch(node.then_case, env)
            if node.else_case is not None:
                _collect_dispatch(node.else_case, env)
            return
        if isinstance(node, tirx.For):
            if getattr(node, "thread_binding", None) is not None:
                _collect_dispatch(node.body, env)
                return
            ext_sub = _substitute_let_env(node.extent, env, tirx=tirx)
            if _depends_on_block_var(ext_sub):
                work_items.append(("loop", ext_sub))
                return
            _collect_dispatch(node.body, env)
            return
        if isinstance(node, tirx.AttrStmt):
            _collect_dispatch(node.body, env)
            return
        if isinstance(node, tirx.SBlockRealize):
            _collect_dispatch(node.block, env)
            return
        if isinstance(node, tirx.SBlock):
            _collect_dispatch(node.body, env)
            return
        if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)) or (
            alloc_cls is not None and isinstance(node, alloc_cls)
        ):
            if hasattr(node, "body"):
                _collect_dispatch(node.body, env)

    _collect_dispatch(inner_body, init_env)
    if not work_items:
        return None

    bound_items: list[tuple[str, object]] = []
    for kind, item_expr in work_items:
        vmap = _build_free_var_int_map(
            item_expr,
            var_bindings,
            var_obj_bindings,
            tirx=tirx,
            exclude_var=block_var,
        )
        sub_expr = tirx.stmt_functor.substitute(item_expr, vmap) if vmap else item_expr
        rem_vars = [
            v for v in _free_tirx_vars(sub_expr, tirx=tirx) if not _same_var(v, block_var)
        ]
        if rem_vars:
            return None
        bound_items.append((kind, sub_expr))

    if tvm_mod is None or not hasattr(tvm_mod, "arith") or not hasattr(tvm_mod.arith, "Analyzer"):
        return None
    analyzer = tvm_mod.arith.Analyzer()
    bx_dtype = str(getattr(block_var, "dtype", "int32") or "int32")
    active_blocks = 0
    total_work_units = 0
    for bx in range(static_grid):
        bx_imm = tirx.IntImm(bx_dtype, bx)
        bx_guard = False
        bx_loops = 0
        for kind, sub_expr in bound_items:
            try:
                sim = analyzer.simplify(
                    tirx.stmt_functor.substitute(sub_expr, {block_var: bx_imm})
                )
            except Exception:
                return None
            val = _static_tirx_int(sim)
            if val is None and isinstance(sim, tirx.IntImm):
                val = int(sim.value)
            if val is None:
                return None
            if kind == "guard":
                if val > 0:
                    bx_guard = True
            else:
                if val > 0:
                    bx_loops += int(val)
        if bx_guard or bx_loops > 0:
            active_blocks += 1
            total_work_units += bx_loops if bx_loops > 0 else 1
    return active_blocks, max(active_blocks, total_work_units)


def _inspect_tilelang_block_dims(
    body,
    var_bindings: Mapping[str, int] | None,
    *,
    tirx,
    tvm_mod: object | None,
    var_obj_bindings: Sequence[tuple[object, int]] = (),
) -> tuple[dict[str, int], bool]:
    block_dims: dict[str, int] = {}
    any_symbolic = [False]
    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)
    alloc_cls = getattr(tirx, "Allocate", None)

    def _record_extent(tag: str, expr, env: dict[object, object]) -> None:
        val, is_sym = _resolve_symbolic_grid_extent(
            expr,
            env,
            var_bindings,
            tirx=tirx,
            tvm_mod=tvm_mod,
            var_obj_bindings=var_obj_bindings,
        )
        if is_sym:
            any_symbolic[0] = True
        elif val is not None and val > 0:
            block_dims[tag] = val

    def _walk(node, env: dict[object, object]) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = child.value
                    if hasattr(child, "body"):
                        _walk(child.body, dict(seq_env))
                else:
                    _walk(child, seq_env)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(env)
            next_env[node.var] = node.value
            _walk(node.body, next_env)
            return
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(env)
            next_env[node.var] = node.value
            if hasattr(node, "body"):
                _walk(node.body, next_env)
            return
        if isinstance(node, tirx.AttrStmt):
            if "thread_extent" in str(node.attr_key).lower():
                tag = str(getattr(node.node, "thread_tag", "")).lower()
                if "blockidx" in tag:
                    _record_extent(tag, node.value, env)
            _walk(node.body, env)
            return
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            if tb is not None:
                tag = str(getattr(tb, "thread_tag", "")).lower()
                if "blockidx" in tag:
                    _record_extent(tag, node.extent, env)
            _walk(node.body, env)
            return
        if isinstance(node, tirx.While):
            _walk(node.body, env)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, env)
            if node.else_case is not None:
                _walk(node.else_case, env)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, env)
            return
        if isinstance(node, tirx.SBlock):
            if getattr(node, "init", None) is not None:
                _walk(node.init, env)
            _walk(node.body, env)
            return
        if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)) or (
            alloc_cls is not None and isinstance(node, alloc_cls)
        ):
            if hasattr(node, "body"):
                _walk(node.body, env)

    _walk(body, {})
    return block_dims, any_symbolic[0]


def _is_sync_or_barrier_intrinsic(op_name: str) -> bool:
    low = op_name.lower()
    return any(
        tok in low
        for tok in (
            "barrier",
            "pipe_barrier",
            "cross_flag",
            "cross_core",
            "set_flag",
            "wait_flag",
            "mbarrier",
            "sync",
        )
    )


def _is_disallowed_tirx_intrinsic(op_name: str, *, allow_sync: bool = False) -> bool:
    """Return True if *op_name* represents DMA, Cube, SIMD/SIMT, or barrier intrinsic."""
    low = op_name.lower()
    if allow_sync and _is_sync_or_barrier_intrinsic(low):
        return False
    if any(
        tok in low
        for tok in (
            "copy",
            "dual_copy",
            "dma",
            "region",
            "access_ptr",
            "gemm",
            "mmad",
            "ascend_",
            "simd",
            "simt",
            "vld",
            "vsts",
            "pset",
            "vadd",
            "vmul",
            "vsub",
            "vexp",
            "vcadd",
            "vreduce",
            "reduce_",
            "barrier",
            "pipe_barrier",
            "cross_flag",
            "cross_core",
            "set_flag",
            "wait_flag",
            "mbarrier",
            "sync",
        )
    ):
        return True
    return bool(re.search(r"(?:^|[._])(?:mad|vec)(?:$|[._])", low))


def _is_tirx_math_call(op_name: str) -> bool:
    """Return True if *op_name* is an elementwise/scalar math call rather than an intrinsic."""
    low = op_name.lower()
    if _is_disallowed_tirx_intrinsic(low):
        return False
    return low.startswith("tir.") or any(
        tok in low
        for tok in (
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "sin",
            "cos",
            "tan",
            "tanh",
            "erf",
            "abs",
            "fabs",
            "floor",
            "ceil",
            "round",
            "pow",
            "fma",
            "sigmoid",
            "max",
            "min",
            "div",
            "mod",
            "select",
            "cast",
        )
    )


def _count_tirx_ub_scalar_compute_loops(
    stmt,
    ub_buf_names: set[str],
    *,
    tirx,
    allow_sync_intrinsics: bool = False,
) -> int:
    """Count innermost serial UB scalar compute loops while filtering out outer/tiny/SIMD loops."""
    count = [0]
    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)
    alloc_cls = getattr(tirx, "Allocate", None)
    arith_types = tuple(
        cls
        for name in (
            "Add",
            "Sub",
            "Mul",
            "Div",
            "FloorDiv",
            "Mod",
            "Max",
            "Min",
            "Select",
            "Cast",
        )
        if (cls := getattr(tirx, name, None)) is not None
    )

    def _is_ub_buf(buf) -> bool:
        raw_scope = str(buf.scope()).strip()
        return (
            buf.name in ub_buf_names
            or _map_tilelang_scope(raw_scope) == "UB"
            or raw_scope == "local.ub"
        )

    def _is_ub_or_gm_buf(buf) -> bool:
        raw_scope = str(buf.scope()).strip()
        mapped = _map_tilelang_scope(raw_scope)
        return (
            buf.name in ub_buf_names
            or mapped in ("UB", "GM")
            or raw_scope in ("local.ub", "global")
        )

    def _is_vector_sblock(node) -> bool:
        ann = getattr(node, "annotations", None) or {}
        hint = str(getattr(node, "name_hint", "")).lower()
        return (
            "simd_vf" in hint
            or "simt_vf" in hint
            or "tl.simdvf_lanes" in ann
            or "tl.vf_source_index" in ann
        )

    def _is_vector_attr(node) -> bool:
        key = str(getattr(node, "attr_key", "")).lower()
        node_low = str(getattr(node, "node", "")).lower()
        if (
            "simdvf" in key
            or "simtvf" in key
            or "simt" in key
            or "simt" in node_low
            or key == "pragma_parallel"
        ):
            return True
        if "thread_extent" in key:
            tag = str(getattr(node.node, "thread_tag", "")).lower()
            iv = getattr(node.node, "var", None)
            iv_name = str(getattr(iv, "name_hint", "")).lower()
            if (
                "blockidx" not in tag
                and "blockidx" not in iv_name
                and "cthread" not in tag
                and "subblock" not in tag
                and "cthread" not in iv_name
                and "subblock" not in iv_name
            ):
                return True
        return False

    def _expr_has_disallowed_intrinsic(expr) -> bool:
        if expr is None:
            return False
        hit = [False]

        def _visit(sub) -> None:
            if isinstance(sub, tirx.Call):
                op_name = _effective_tirx_call_name(sub).lower()
                if _is_disallowed_tirx_intrinsic(op_name, allow_sync=allow_sync_intrinsics):
                    hit[0] = True

        tirx.stmt_functor.post_order_visit(expr, _visit)
        return hit[0]

    def _expr_has_arith_and_load(expr) -> bool:
        if expr is None:
            return False
        has_arith = [False]
        has_load = [False]

        def _visit(sub) -> None:
            if arith_types and isinstance(sub, arith_types):
                has_arith[0] = True
            elif isinstance(sub, tirx.Call):
                op_name = _effective_tirx_call_name(sub).lower()
                if _is_tirx_math_call(op_name):
                    has_arith[0] = True
            elif isinstance(sub, tirx.BufferLoad):
                if _is_ub_or_gm_buf(sub.buffer):
                    has_load[0] = True

        tirx.stmt_functor.post_order_visit(expr, _visit)
        return has_arith[0] and has_load[0]

    def _is_innermost_ub_scalar_body(loop_body, init_env: dict[object, object]) -> bool:
        aborted = [False]
        found_store = [False]

        def _scan_body(node, env: dict[object, object]) -> None:
            if node is None or aborted[0]:
                return
            if isinstance(node, (tirx.For, tirx.While)):
                aborted[0] = True
                return
            if isinstance(node, tirx.SeqStmt):
                seq_env = dict(env)
                for child in node.seq:
                    if bind_cls is not None and isinstance(child, bind_cls):
                        if _expr_has_disallowed_intrinsic(child.value):
                            aborted[0] = True
                            return
                        seq_env[child.var] = _substitute_let_env(
                            child.value, seq_env, tirx=tirx
                        )
                        if hasattr(child, "body"):
                            _scan_body(child.body, dict(seq_env))
                    else:
                        _scan_body(child, seq_env)
                return
            if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
                if _expr_has_disallowed_intrinsic(node.value):
                    aborted[0] = True
                    return
                next_env = dict(env)
                next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
                _scan_body(node.body, next_env)
                return
            if bind_cls is not None and isinstance(node, bind_cls):
                if _expr_has_disallowed_intrinsic(node.value):
                    aborted[0] = True
                    return
                next_env = dict(env)
                next_env[node.var] = _substitute_let_env(node.value, env, tirx=tirx)
                if hasattr(node, "body"):
                    _scan_body(node.body, next_env)
                return
            if isinstance(node, tirx.IfThenElse):
                if _expr_has_disallowed_intrinsic(node.condition):
                    aborted[0] = True
                    return
                _scan_body(node.then_case, env)
                if node.else_case is not None:
                    _scan_body(node.else_case, env)
                return
            if isinstance(node, tirx.SBlockRealize):
                _scan_body(node.block, env)
                return
            if isinstance(node, tirx.SBlock):
                if _is_vector_sblock(node):
                    aborted[0] = True
                    return
                if getattr(node, "init", None) is not None:
                    _scan_body(node.init, env)
                _scan_body(node.body, env)
                return
            if isinstance(node, tirx.AttrStmt):
                if _is_vector_attr(node):
                    aborted[0] = True
                    return
                _scan_body(node.body, env)
                return
            if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)) or (
                alloc_cls is not None and isinstance(node, alloc_cls)
            ):
                if hasattr(node, "body"):
                    _scan_body(node.body, env)
                return
            if isinstance(node, tirx.Evaluate):
                if _expr_has_disallowed_intrinsic(node.value):
                    aborted[0] = True
                return
            if isinstance(node, tirx.BufferStore):
                if _expr_has_disallowed_intrinsic(node.value) or any(
                    _expr_has_disallowed_intrinsic(idx) for idx in node.indices
                ):
                    aborted[0] = True
                    return
                if _is_ub_buf(node.buffer):
                    sub_val = _substitute_let_env(node.value, env, tirx=tirx)
                    if _expr_has_arith_and_load(sub_val):
                        found_store[0] = True

        _scan_body(loop_body, init_env)
        return (not aborted[0]) and found_store[0]

    def _walk(node, *, in_vector_scope: bool, let_env: dict[object, object]) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(let_env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(
                        child.value, seq_env, tirx=tirx
                    )
                    if hasattr(child, "body"):
                        _walk(
                            child.body,
                            in_vector_scope=in_vector_scope,
                            let_env=dict(seq_env),
                        )
                else:
                    _walk(child, in_vector_scope=in_vector_scope, let_env=seq_env)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            _walk(node.body, in_vector_scope=in_vector_scope, let_env=next_env)
            return
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            if hasattr(node, "body"):
                _walk(node.body, in_vector_scope=in_vector_scope, let_env=next_env)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, in_vector_scope=in_vector_scope, let_env=let_env)
            if node.else_case is not None:
                _walk(node.else_case, in_vector_scope=in_vector_scope, let_env=let_env)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, in_vector_scope=in_vector_scope, let_env=let_env)
            return
        if isinstance(node, tirx.SBlock):
            next_vec = in_vector_scope or _is_vector_sblock(node)
            if getattr(node, "init", None) is not None:
                _walk(node.init, in_vector_scope=next_vec, let_env=let_env)
            _walk(node.body, in_vector_scope=next_vec, let_env=let_env)
            return
        if isinstance(node, tirx.AttrStmt):
            next_vec = in_vector_scope or _is_vector_attr(node)
            _walk(node.body, in_vector_scope=next_vec, let_env=let_env)
            return
        if isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)) or (
            alloc_cls is not None and isinstance(node, alloc_cls)
        ):
            if hasattr(node, "body"):
                _walk(node.body, in_vector_scope=in_vector_scope, let_env=let_env)
            return
        if isinstance(node, tirx.While):
            _walk(node.body, in_vector_scope=in_vector_scope, let_env=let_env)
            return
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            if tb is not None:
                tag = str(getattr(tb, "thread_tag", "")).lower()
                next_vec = in_vector_scope or (
                    "blockidx" not in tag and "cthread" not in tag and "subblock" not in tag
                )
                _walk(node.body, in_vector_scope=next_vec, let_env=let_env)
                return
            kind_raw = getattr(node, "kind", 0)
            kind_val = (
                int(getattr(kind_raw, "value", kind_raw))
                if isinstance(getattr(kind_raw, "value", kind_raw), int)
                else -1
            )
            kind_str = str(kind_raw).lower()
            ann = getattr(node, "annotations", None) or {}
            stages_val = _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else None
            is_pipelined = (
                kind_val == 5
                or "pipelined" in kind_str
                or "num_stages" in ann
                or (stages_val is not None and stages_val >= 1)
                or bool(ann.get("enable_offset"))
                or (ann.get("software_pipeline_stage") is not None)
            )
            is_serial = (
                kind_val == 0
                and not any(
                    tok in kind_str
                    for tok in ("parallel", "vectorized", "unrolled", "thread")
                )
                and not is_pipelined
            )
            trip = _static_tirx_int(node.extent)
            valid_extent = trip is None or trip >= 8
            if not in_vector_scope and is_serial and valid_extent:
                if _is_innermost_ub_scalar_body(node.body, let_env):
                    count[0] += 1
                    return
            _walk(node.body, in_vector_scope=in_vector_scope, let_env=let_env)

    _walk(stmt, in_vector_scope=False, let_env={})
    return count[0]


def _collect_tilelang_telemetry(
    entry_fn,
    mod_in,
    *,
    tirx,
    DataTypeCls,
    pipe_cycles: dict[str, int],
    unreserved_ubuf: bool,
    spaces: AscendSpaceFacts | None = None,
    aiv_split_factor: int = 1,
    var_bindings: Mapping[str, int] | None = None,
    tvm_mod: object | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> _TelemetryFeatures:
    """Extract NP1 structural, quantization, and layout telemetry from a TVM PrimFunc."""
    flag = "--cce-disable-vf-stack-reserved-ubuf"
    effective_unreserved = bool(
        unreserved_ubuf
        or _attrs_contain_flag(getattr(entry_fn, "attrs", None), flag)
        or _attrs_contain_flag(getattr(mod_in, "attrs", None), flag)
    )

    block_dims: dict[str, int] = {}
    has_lowbit_buf = [False]
    has_indirect_gather = [False]
    has_unlimit_mem = [
        _attrs_contain_flag(getattr(entry_fn, "attrs", None), "unlimit_memory")
        or _attrs_contain_flag(getattr(mod_in, "attrs", None), "unlimit_memory")
    ]
    has_dyn_offset_pipe = [False]
    cube_m_values: list[int] = []
    gm_static_bytes: dict[str, int] = {}
    ub_static_bytes: dict[str, int] = {}
    ub_buf_names: set[str] = set()
    l1_static_bytes: dict[str, int] = {}
    ub_2d_total_bytes = [0]
    ub_2d_conflict_bytes = [0]
    has_l1_half_isolation = [
        _attrs_contain_flag(getattr(entry_fn, "attrs", None), "half_l1")
        or _attrs_contain_flag(getattr(mod_in, "attrs", None), "half_l1")
    ]
    gm_scalar_names: set[str] = set()
    contig_candidates: list[tuple[int | None, int, int, int, int]] = []
    seen_gm_keys: set[str] = set()
    seen_ub_bank_keys: set[str] = set()

    def _register_buffer(buf) -> None:
        dtype_str = str(buf.dtype)
        if _is_lowbit_dtype_name(dtype_str):
            has_lowbit_buf[0] = True
        raw_scope = str(buf.scope()).strip()
        scope = _map_tilelang_scope(raw_scope)
        if scope == "UB" or raw_scope == "local.ub":
            ub_buf_names.add(buf.name)
        static_shape = _static_shape_ints(buf.shape)
        if scope == "UB" and static_shape is not None:
            ub_bytes = _buffer_bytes(static_shape, buf.dtype, DataTypeCls)
            ub_static_bytes[buf.name] = max(ub_static_bytes.get(buf.name, 0), ub_bytes)
            if (
                buf.name not in seen_ub_bank_keys
                and len(static_shape) >= 2
                and math.prod(static_shape[:-1]) > 1
                and ub_bytes > 0
            ):
                seen_ub_bank_keys.add(buf.name)
                elem_b = _dtype_bytes(buf.dtype, DataTypeCls)
                strides = getattr(buf, "strides", None) or ()
                row_stride = (
                    _static_tirx_int(strides[-2])
                    if len(strides) >= 2
                    else None
                )
                row_elems = row_stride if (row_stride is not None and row_stride > 0) else static_shape[-1]
                stride_bytes = int(math.ceil(row_elems * elem_b))
                ub_2d_total_bytes[0] += ub_bytes
                if stride_bytes >= 512 and stride_bytes % 512 == 0:
                    ub_2d_conflict_bytes[0] += ub_bytes
        if scope == "L1" and static_shape is not None:
            l1_bytes = _buffer_bytes(static_shape, buf.dtype, DataTypeCls)
            if l1_bytes > 0:
                l1_static_bytes[buf.name] = max(l1_static_bytes.get(buf.name, 0), l1_bytes)
        if scope != "GM":
            return
        key = buf.name
        if key in seen_gm_keys:
            return
        seen_gm_keys.add(key)
        static_shape = _static_shape_ints(buf.shape)
        elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
        if static_shape is not None:
            nbytes = _buffer_bytes(static_shape, buf.dtype, DataTypeCls)
            gm_static_bytes[key] = nbytes
            total_elems = math.prod(static_shape) if static_shape else 0
            if total_elems == 1:
                gm_scalar_names.add(key)
            if static_shape and static_shape[-1] >= 1:
                last_dim = static_shape[-1]
                contig = int(math.ceil(last_dim * elem_bytes))
                contig_candidates.append((total_elems, last_dim, 0, nbytes, contig))
        else:
            gm_static_bytes[key] = 0
            if len(buf.shape) > 0:
                last_dim = _static_tirx_int(buf.shape[-1])
                if last_dim is not None and last_dim >= 1:
                    dyn_cnt = sum(1 for d in buf.shape if _static_tirx_int(d) is None)
                    static_dims = [
                        val for d in buf.shape if (val := _static_tirx_int(d)) is not None
                    ]
                    static_prod = (
                        int(math.ceil(math.prod(static_dims) * elem_bytes))
                        if static_dims
                        else 0
                    )
                    contig = int(math.ceil(last_dim * elem_bytes))
                    contig_candidates.append((None, last_dim, dyn_cnt, static_prod, contig))

    for param in getattr(entry_fn, "params", ()):
        buf = entry_fn.buffer_map.get(param) if hasattr(entry_fn, "buffer_map") else None
        if buf is not None:
            _register_buffer(buf)

    loops: list[tuple[object, bool, bool]] = []

    def _visit_all(node) -> None:
        if isinstance(node, tirx.SBlock):
            if _attrs_contain_flag(getattr(node, "annotations", None), "unlimit_memory"):
                has_unlimit_mem[0] = True
            for buf in node.alloc_buffers:
                _register_buffer(buf)
        elif isinstance(node, tirx.AttrStmt):
            if "unlimit_memory" in str(node.attr_key).lower():
                has_unlimit_mem[0] = True
        elif isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            _register_buffer(node.buffer)
        elif isinstance(node, tirx.BufferLoad):
            if any(_contains_buffer_load(idx, tirx) for idx in node.indices):
                has_indirect_gather[0] = True
        elif isinstance(node, tirx.For):
            ann = getattr(node, "annotations", None) or {}
            kind_val = int(node.kind) if hasattr(node.kind, "value") else node.kind
            stages_val = _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else None
            if ann.get("enable_offset") or kind_val == 5 or (stages_val is not None and stages_val >= 2):
                has_dyn_offset_pipe[0] = True
            tb = getattr(node, "thread_binding", None)
            if tb is None:
                trip = _static_tirx_int(node.extent)
                qualifies = trip is None or trip > 1
                loops.append((node.body, True, qualifies))
        elif isinstance(node, tirx.While):
            loops.append((node.body, False, False))
        elif isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            call_args = _effective_tirx_call_args(node)
            if ("gemm" in op_name or "mmad" in op_name) and len(call_args) >= 3:
                m_val = _static_tirx_int(call_args[5]) if len(call_args) > 5 else None
                if m_val is None:
                    _s, _t, _i = _region_bytes(call_args[0], tirx, DataTypeCls)
                    if isinstance(call_args[0], tirx.Call) and len(call_args[0].args) >= 4:
                        ext = _static_shape_ints(call_args[0].args[2:])
                        if ext and len(ext) >= 2:
                            trans_a = _static_tirx_int(call_args[3]) if len(call_args) > 3 else 0
                            m_val = ext[-1] if trans_a else ext[-2]
                if m_val is not None and m_val > 0:
                    cube_m_values.append(m_val)

    tirx.stmt_functor.post_order_visit(entry_fn.body, _visit_all)

    eff_bindings, var_obj_bindings = _augment_var_bindings_from_buffers(
        entry_fn,
        var_bindings,
        tirx=tirx,
        ordered_tensor_shapes=ordered_tensor_shapes,
    )
    resolved_dims, any_symbolic_grid = _inspect_tilelang_block_dims(
        entry_fn.body,
        eff_bindings or var_bindings,
        tirx=tirx,
        tvm_mod=tvm_mod,
        var_obj_bindings=var_obj_bindings,
    )
    block_dims.update(resolved_dims)
    uses_aic = pipe_cycles.get("AIC.PIPE_M", 0) > 0 or aiv_split_factor >= 2
    if spaces is not None:
        core_pool = spaces.num_aic_cores if uses_aic else spaces.num_aiv_cores
    else:
        core_pool = 1
    if any_symbolic_grid:
        grid_is_symbolic = True
        grid_blocks = max(1, core_pool)
        active_grid_blocks = grid_blocks
    else:
        grid_is_symbolic = False
        grid_blocks = math.prod(block_dims.values()) if block_dims else 1
        active_grid_blocks = grid_blocks
        resolved_persistent = _resolve_persistent_active_blocks(
            entry_fn.body,
            grid_blocks,
            eff_bindings,
            var_obj_bindings,
            tirx=tirx,
            tvm_mod=tvm_mod,
        )
        if resolved_persistent is not None:
            active_blks, eff_work_blks = resolved_persistent
            active_grid_blocks = active_blks
            grid_blocks = eff_work_blks
    dominant_contig_bytes = _select_dominant_contig_bytes(contig_candidates)
    effective_cube_m = min(cube_m_values) if cube_m_values else 0
    has_lowbit_quant = bool(has_lowbit_buf[0] and pipe_cycles.get("AIC.PIPE_M", 0) > 0)

    all_events, _exp, _ul, _us = _collect_tilelang_stmt_events(entry_fn.body, tirx=tirx)
    gm_reads = {name for name, scope, rw in all_events if scope == "GM" and rw == "read"}
    gm_writes = {name for name, scope, rw in all_events if scope == "GM" and rw == "write"}

    gm_loop_scalars: set[str] = set()
    has_online_rescale = False
    ub_reload_store_bufs: set[str] = set()

    for loop_body, is_for, qualifies_trip in loops:
        local_bufs = _collect_local_buf_names(loop_body, tirx=tirx)
        loop_events, loop_has_exp, loop_ub_loads, loop_ub_stores = _collect_tilelang_stmt_events(
            loop_body, tirx=tirx
        )
        for buf_name, scope, _rw in loop_events:
            if scope == "GM" and buf_name in gm_scalar_names:
                gm_loop_scalars.add(buf_name)

        if is_for and qualifies_trip:
            ub_reload_store_bufs.update(loop_ub_loads & loop_ub_stores)

        if loop_has_exp and not has_online_rescale:
            if _has_loop_carried_accumulator(loop_events, local_bufs):
                has_online_rescale = True

    gm_roundtrip_names = (gm_reads & gm_writes) | gm_loop_scalars
    gm_roundtrip_bytes = sum(gm_static_bytes.get(name, 0) for name in gm_roundtrip_names)
    gm_dma_loop, gm_dma_all = _collect_gm_dma_payloads(
        entry_fn.body,
        tirx=tirx,
        DataTypeCls=DataTypeCls,
    )
    dma_pool = [b for b in (gm_dma_loop if gm_dma_loop else gm_dma_all) if b > 0]
    min_gm_dma_bytes = min(dma_pool) if dma_pool else 0
    if spaces is not None and dma_pool:
        total_dma_bytes = float(sum(dma_pool))
        weighted_inv_eff = sum(
            float(b) / max(1e-6, float(spaces.dma_burst_efficiency(int(b))))
            for b in dma_pool
        )
        dma_burst_eff = round(min(1.0, total_dma_bytes / max(1e-6, weighted_inv_eff)), 4)
    else:
        dma_burst_eff = 1.0
    in_loop_pipe_all = _count_in_loop_pipe_all_barriers(entry_fn.body, tirx=tirx)
    wide_ub_names = {name for name, nbytes in ub_static_bytes.items() if nbytes >= 1024}
    coupled_reduce_sfu = _detect_coupled_reduce_sfu_accum(
        entry_fn.body,
        wide_ub_names,
        tirx=tirx,
    )
    ub_bank_conflict_risk = (
        round(min(1.0, ub_2d_conflict_bytes[0] / float(ub_2d_total_bytes[0])), 4)
        if ub_2d_total_bytes[0] > 0 and ub_2d_conflict_bytes[0] > 0
        else 0.0
    )
    l1_total = sum(l1_static_bytes.values())
    l1_full = max(1, spaces.l1_bytes if spaces is not None else 524288)
    l1_half = max(1, l1_full // 2)
    if l1_total <= 0 or has_l1_half_isolation[0]:
        l1_bank_conflict_risk = 0.0
    elif l1_total > l1_half:
        l1_bank_conflict_risk = round(min(1.0, l1_total / float(l1_full)), 4)
    elif len(l1_static_bytes) >= 2:
        l1_bank_conflict_risk = round(min(0.5, l1_total / float(l1_half)), 4)
    else:
        l1_bank_conflict_risk = 0.0

    ub_scalar_loops = _count_tirx_ub_scalar_compute_loops(
        entry_fn.body,
        ub_buf_names,
        tirx=tirx,
    )

    return _TelemetryFeatures(
        grid_blocks=grid_blocks,
        gm_roundtrip_bytes=gm_roundtrip_bytes,
        dominant_contig_bytes=dominant_contig_bytes,
        effective_cube_m=effective_cube_m,
        has_indirect_gather=has_indirect_gather[0],
        has_lowbit_quant=has_lowbit_quant,
        has_online_rescale_loop=has_online_rescale,
        in_loop_ub_reload_stores=len(ub_reload_store_bufs),
        uses_unreserved_ubuf=effective_unreserved,
        grid_is_symbolic=grid_is_symbolic,
        min_gm_dma_bytes=min_gm_dma_bytes,
        dma_burst_efficiency_ratio=dma_burst_eff,
        uses_dynamic_offset_pipelining=has_dyn_offset_pipe[0],
        has_unlimit_memory_attr=has_unlimit_mem[0],
        has_coupled_reduce_sfu_accum=coupled_reduce_sfu,
        active_grid_blocks=active_grid_blocks,
        in_loop_pipe_all_barriers=in_loop_pipe_all,
        ub_bank_conflict_risk=ub_bank_conflict_risk,
        l1_bank_conflict_risk=l1_bank_conflict_risk,
        ub_scalar_compute_loops=ub_scalar_loops,
    )


def _scope_capacities(
    facts: AscendSpaceFacts,
    *,
    has_simtvf: bool,
    unreserved_ubuf: bool,
) -> dict[str, int]:
    ub_mode = "simtvf" if has_simtvf else "simdvf"
    return {
        "UB": facts.ub_limit_bytes(ub_mode, unreserved_ubuf=unreserved_ubuf),
        "L1": facts.l1_bytes,
        "L0A": facts.l0a_bytes,
        "L0B": facts.l0b_bytes,
        "L0C": facts.l0c_bytes,
        "SSBUF": facts.ssbuf_bytes,
    }


def _extract_tilelang_buffers_from_func(
    func,
    *,
    tirx,
    DataTypeCls,
    prefer_allocate: bool = False,
) -> tuple[BufferSnapshot, ...]:
    """Extract static on-chip BufferSnapshot items from a TVM PrimFunc."""
    entries: list[tuple[object | None, str, BufferSnapshot]] = []

    def _record(var_obj, name: str, raw_scope: str, shape_obj, dtype_obj) -> None:
        scope = _map_tilelang_scope(raw_scope)
        if scope is None or scope == "GM":
            return
        shape = _static_shape_ints(shape_obj)
        nbytes = _buffer_bytes(shape, dtype_obj, DataTypeCls)
        if shape is None:
            return
        snap = BufferSnapshot(
            name=str(name),
            scope=scope,
            bytes=nbytes,
            shape=shape,
            dtype=str(dtype_obj),
        )
        for idx, (prev_var, prev_key, prev_snap) in enumerate(entries):
            same_var = (
                var_obj is not None
                and prev_var is not None
                and hasattr(var_obj, "same_as")
                and var_obj.same_as(prev_var)
            )
            if same_var or (var_obj is None and prev_key == f"{scope}:{name}"):
                if snap.bytes > prev_snap.bytes:
                    entries[idx] = (var_obj, prev_key, snap)
                return
        entries.append((var_obj, f"{scope}:{name}", snap))

    def _visit(node) -> None:
        if isinstance(node, tirx.SBlock) and not prefer_allocate:
            for buf in node.alloc_buffers:
                _record(buf.data, buf.name, buf.scope(), buf.shape, buf.dtype)
        elif isinstance(node, tirx.AllocBuffer):
            buf = node.buffer
            _record(buf.data, buf.name, buf.scope(), buf.shape, buf.dtype)
        elif isinstance(node, tirx.DeclBuffer):
            buf = node.buffer
            _record(buf.data, buf.name, buf.scope(), buf.shape, buf.dtype)
        elif isinstance(node, tirx.SBlock) and prefer_allocate:
            for buf in node.alloc_buffers:
                _record(buf.data, buf.name, buf.scope(), buf.shape, buf.dtype)

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    return tuple(sorted((e[2] for e in entries), key=lambda item: (item.scope, item.name)))


def _extract_tilelang_buffers_from_mod(mod, *, tirx, DataTypeCls, prefer_allocate: bool) -> tuple[BufferSnapshot, ...]:
    if mod is None:
        return ()
    all_snaps: list[BufferSnapshot] = []
    for _gvar, fn in mod.functions.items():
        if isinstance(fn, tirx.PrimFunc):
            all_snaps.extend(
                _extract_tilelang_buffers_from_func(
                    fn,
                    tirx=tirx,
                    DataTypeCls=DataTypeCls,
                    prefer_allocate=prefer_allocate,
                )
            )
    return tuple(all_snaps)


def _requested_multibuffer_names(func, *, tirx) -> set[str]:
    """Identify buffer names annotated for multi-buffering or inside Pipelined loops."""
    names: set[str] = set()
    in_pipelined = [False]

    def _pre(node) -> None:
        if isinstance(node, tirx.For):
            kind_val = int(node.kind) if hasattr(node.kind, "value") else node.kind
            ann = getattr(node, "annotations", {}) or {}
            if "num_stages" in ann or kind_val == 5:
                in_pipelined[0] = True
        elif isinstance(node, tirx.AttrStmt):
            if "buffer_version" in str(node.attr_key):
                if isinstance(node.node, tirx.Buffer):
                    names.add(node.node.name)
                elif hasattr(node.node, "items"):
                    for k, v in node.node.items():
                        v_int = int(v.value) if hasattr(v, "value") else 0
                        if v_int > 1 or str(v) == "auto":
                            names.add(getattr(k, "name", str(k)))
        elif isinstance(node, tirx.SBlock):
            ann = getattr(node, "annotations", {}) or {}
            for k, v in ann.items():
                if "buffer_version" in str(k) and hasattr(v, "items"):
                    for bk, bv in v.items():
                        b_name = getattr(bk, "name", str(bk))
                        if hasattr(bv, "__len__") and len(bv) > 0:
                            first = bv[0]
                            cnt = int(first.value) if hasattr(first, "value") else 2
                            if cnt >= 2:
                                names.add(b_name)
                        elif hasattr(bv, "value") and int(bv.value) >= 2:
                            names.add(b_name)
            if in_pipelined[0]:
                for buf in node.alloc_buffers:
                    if _map_tilelang_scope(buf.scope()) in ("UB", "L1", "L0A", "L0B", "SSBUF"):
                        names.add(buf.name)

    tirx.stmt_functor.post_order_visit(func.body, _pre)
    return names


def _detect_degraded_buffers(
    explicit_snaps: tuple[BufferSnapshot, ...],
    materialized_snaps: tuple[BufferSnapshot, ...],
    requested_names: set[str],
) -> tuple[str, ...]:
    """Return requested multi-buffers whose physical allocation stayed at 1x."""
    if not requested_names:
        return ()
    explicit_by_name = {s.name: s for s in explicit_snaps}
    mat_by_name = {s.name: s for s in materialized_snaps}
    degraded: list[str] = []
    for name in sorted(requested_names):
        exp = explicit_by_name.get(name)
        mat = mat_by_name.get(name)
        if exp is None or mat is None:
            continue
        if len(mat.shape) == len(exp.shape) and mat.bytes <= exp.bytes:
            degraded.append(name)
        elif len(mat.shape) > len(exp.shape) and mat.shape[0] <= 1:
            degraded.append(name)
    return tuple(degraded)


def _strip_version_suffix(name: str) -> str:
    if name.endswith("_buf"):
        return name[:-4]
    return name


def _diff_snapshots_by_scope(
    explicit_snaps: tuple[BufferSnapshot, ...],
    lowered_snaps: tuple[BufferSnapshot, ...],
    merged_peak_by_scope: dict[str, int],
    capacities: dict[str, int],
    degraded_names: tuple[str, ...],
) -> dict[str, ScopeWaterline]:
    """Compute B_lowered minus B_explicit bloat and ScopeWaterline per active scope."""
    scopes = sorted(
        {s.scope for s in explicit_snaps}
        | {s.scope for s in lowered_snaps}
        | set(merged_peak_by_scope.keys())
    )
    waterlines: dict[str, ScopeWaterline] = {}
    degraded_set = set(degraded_names)
    for scope in scopes:
        exp_map = {s.name: s.bytes for s in explicit_snaps if s.scope == scope}
        low_snaps = [s for s in lowered_snaps if s.scope == scope]
        imp_map: dict[str, int] = {}
        matched_explicit: set[str] = set()
        for snap in low_snaps:
            base = snap.name
            if base not in exp_map:
                stripped = _strip_version_suffix(base)
                if stripped in exp_map:
                    base = stripped
            if base in exp_map and base not in matched_explicit:
                matched_explicit.add(base)
                delta = snap.bytes - exp_map[base]
                if delta > 0:
                    imp_map[f"{base}_expansion"] = delta
            else:
                imp_map[snap.name] = snap.bytes
        lowered_sum = sum(s.bytes for s in low_snaps)
        peak = merged_peak_by_scope.get(scope, lowered_sum if low_snaps else sum(exp_map.values()))
        scoped_degraded = tuple(name for name in sorted(degraded_set) if name in exp_map)
        waterlines[scope] = build_scope_waterline(
            scope=scope,
            capacity_bytes=capacities.get(scope, 0),
            explicit_buffers=exp_map,
            implicit_buffers=imp_map,
            peak_allocated_bytes=peak,
            degraded_single_buffers=scoped_degraded,
        )
    return waterlines


def _upper_bound_tirx_int(
    expr,
    *,
    tirx,
    let_env: Mapping[object, object] | None = None,
    _depth: int = 0,
) -> int | None:
    """Return a tight static upper bound for a TIR extent expression when provable."""
    if expr is None or _depth > 12:
        return None
    cur = _substitute_let_env(expr, let_env, tirx=tirx) if let_env else expr
    static_val = _static_tirx_int(cur)
    if static_val is not None:
        return static_val
    if isinstance(cur, tirx.Min):
        lhs = _upper_bound_tirx_int(cur.a, tirx=tirx, _depth=_depth + 1)
        rhs = _upper_bound_tirx_int(cur.b, tirx=tirx, _depth=_depth + 1)
        if lhs is not None and lhs > 0 and rhs is not None and rhs > 0:
            return min(lhs, rhs)
        if lhs is not None and lhs > 0:
            return lhs
        if rhs is not None and rhs > 0:
            return rhs
        return None
    if isinstance(cur, tirx.Mul):
        lhs = _upper_bound_tirx_int(cur.a, tirx=tirx, _depth=_depth + 1)
        rhs = _upper_bound_tirx_int(cur.b, tirx=tirx, _depth=_depth + 1)
        if lhs is not None and lhs > 0 and rhs is not None and rhs > 0:
            return lhs * rhs
        return None
    if isinstance(cur, tirx.Add):
        lhs = _upper_bound_tirx_int(cur.a, tirx=tirx, _depth=_depth + 1)
        rhs = _upper_bound_tirx_int(cur.b, tirx=tirx, _depth=_depth + 1)
        if lhs is not None and lhs >= 0 and rhs is not None and rhs >= 0:
            return lhs + rhs
        return None
    return None


def _region_bytes(
    arg,
    tirx,
    DataTypeCls,
    *,
    let_env: Mapping[object, object] | None = None,
) -> tuple[str | None, int, int]:
    """Return (mapped_scope, total_bytes, inner_contiguous_bytes) for copy args."""
    if isinstance(arg, tirx.Call) and "region" in str(getattr(arg.op, "name", arg.op)):
        if len(arg.args) >= 2 and isinstance(arg.args[0], tirx.BufferLoad):
            buf = arg.args[0].buffer
            scope = _map_tilelang_scope(buf.scope())
            elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
            raw_extents = list(arg.args[2:])
            if not raw_extents:
                return scope, 0, 0
            resolved = [
                _upper_bound_tirx_int(ext, tirx=tirx, let_env=let_env)
                for ext in raw_extents
            ]
            last_dim = resolved[-1]
            inner = (
                int(math.ceil(last_dim * elem_bytes))
                if (last_dim is not None and last_dim > 0)
                else 0
            )
            if any(v is None or v <= 0 for v in resolved):
                return scope, 0, inner
            ext_ints = [int(v) for v in resolved if v is not None]
            total = int(math.ceil(math.prod(ext_ints) * elem_bytes))
            return scope, total, inner
    if isinstance(arg, tirx.BufferRegion):
        buf = arg.buffer
        scope = _map_tilelang_scope(buf.scope())
        elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
        raw_extents = [r.extent for r in arg.region]
        if not raw_extents:
            return scope, 0, 0
        resolved = [
            _upper_bound_tirx_int(ext, tirx=tirx, let_env=let_env)
            for ext in raw_extents
        ]
        last_dim = resolved[-1]
        inner = (
            int(math.ceil(last_dim * elem_bytes))
            if (last_dim is not None and last_dim > 0)
            else 0
        )
        if any(v is None or v <= 0 for v in resolved):
            return scope, 0, inner
        ext_ints = [int(v) for v in resolved if v is not None]
        total = int(math.ceil(math.prod(ext_ints) * elem_bytes))
        return scope, total, inner
    if isinstance(arg, tirx.BufferLoad):
        buf = arg.buffer
        scope = _map_tilelang_scope(buf.scope())
        elem_bytes = int(math.ceil(_dtype_bytes(buf.dtype, DataTypeCls)))
        return scope, elem_bytes, elem_bytes
    return None, 0, 0


def _contains_buffer_load(expr, tirx) -> bool:
    found = [False]

    def _visit(node) -> None:
        if isinstance(node, tirx.BufferLoad):
            found[0] = True

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found[0]


def _static_tirx_int(expr) -> int | None:
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    val = getattr(expr, "value", None)
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    return None


def _detect_tilelang_aiv_split_factor(func, *, tirx) -> int:
    """Detect coupled dual-AIV execution from cthread bindings or dual_copy ops."""
    if func is None or getattr(func, "body", None) is None:
        return 1
    detected = [1]

    def _visit(node) -> None:
        if detected[0] >= 2:
            return
        if isinstance(node, tirx.AttrStmt):
            key = str(node.attr_key).lower()
            if "thread_extent" in key:
                tag = str(getattr(node.node, "thread_tag", "")).lower()
                iv = getattr(node.node, "var", None)
                iv_name = str(getattr(iv, "name_hint", "")).lower()
                if "cthread" in tag or "subblock" in tag or "cthread" in iv_name or "subblock" in iv_name:
                    val = _static_tirx_int(node.value)
                    if val is not None and val >= 2:
                        detected[0] = 2
        elif isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            if tb is not None:
                tag = str(getattr(tb, "thread_tag", "")).lower()
                var_name = str(getattr(node.loop_var, "name_hint", "")).lower()
                if "cthread" in tag or "subblock" in tag or "cthread" in var_name or "subblock" in var_name:
                    val = _static_tirx_int(node.extent)
                    if val is not None and val >= 2:
                        detected[0] = 2
        elif isinstance(node, tirx.SBlock):
            ann = getattr(node, "annotations", None) or {}
            if "vector_count" in ann:
                val = _static_tirx_int(ann["vector_count"])
                if val is not None and val >= 2:
                    detected[0] = 2
        elif isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            ann = getattr(node, "annotations", None) or {}
            if (
                "dual_copy" in op_name
                or "subblock" in op_name
                or "get_subblock_idx" in op_name
                or "dual_dst_ctl" in ann
            ):
                detected[0] = 2

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    return detected[0]


def _is_sid_expr(expr, sid_vars: set[object], tirx) -> bool:
    if isinstance(expr, tirx.Var):
        name = str(getattr(expr, "name_hint", "")).lower()
        return any(expr.same_as(v) for v in sid_vars) or "subblock" in name or name == "sid"
    if isinstance(expr, tirx.Call):
        op_name = _effective_tirx_call_name(expr).lower()
        return "subblock" in op_name or "get_subblock_idx" in op_name
    return False


def _expr_references_sid(expr, sid_vars: set[object], tirx) -> bool:
    found = [False]

    def _visit(node) -> None:
        if not found[0] and _is_sid_expr(node, sid_vars, tirx):
            found[0] = True

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return found[0]


def _sid_guard_branches(
    cond,
    sid_vars: set[object],
    tirx,
    aiv_split_factor: int,
) -> tuple[str | None, str | None]:
    if isinstance(cond, (tirx.EQ, tirx.NE)):
        for lhs, rhs in ((cond.a, cond.b), (cond.b, cond.a)):
            if _is_sid_expr(lhs, sid_vars, tirx):
                val = _static_tirx_int(rhs)
                if val is not None:
                    matched_core = "aiv1" if val == 1 else "aiv0"
                    other_core = (
                        ("aiv0" if val == 1 else "aiv1")
                        if aiv_split_factor >= 2
                        else "aiv0"
                    )
                    if isinstance(cond, tirx.EQ):
                        return matched_core, other_core
                    return other_core, matched_core
    if _expr_references_sid(cond, sid_vars, tirx):
        return "aiv0", ("aiv1" if aiv_split_factor >= 2 else "aiv0")
    return None, None


def _burst_scaled_copy_cycles(xfer_bytes: int, spaces: AscendSpaceFacts | None) -> int:
    """Estimate DMA pipe cycles inflated by small-payload HBM burst inefficiency."""
    base = max(1, (xfer_bytes + 63) // 64)
    if spaces is None or xfer_bytes <= 0:
        return base
    eff = spaces.dma_burst_efficiency(xfer_bytes)
    return int(math.ceil(float(base) / max(0.15, float(eff))))


def _collect_gm_dma_payloads(
    stmt,
    *,
    tirx,
    DataTypeCls,
) -> tuple[list[int], list[int]]:
    """Collect GM<->UB/L1 copy byte payloads both inside loops and across the function."""
    loop_sizes: list[int] = []
    all_sizes: list[int] = []
    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)

    def _walk(node, *, in_loop: bool, let_env: dict[object, object]) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(let_env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(child.value, seq_env, tirx=tirx)
                    if hasattr(child, "body"):
                        _walk(child.body, in_loop=in_loop, let_env=dict(seq_env))
                else:
                    _walk(child, in_loop=in_loop, let_env=seq_env)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            _walk(node.body, in_loop=in_loop, let_env=next_env)
            return
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            if hasattr(node, "body"):
                _walk(node.body, in_loop=in_loop, let_env=next_env)
            return
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            trip = _static_tirx_int(node.extent) if tb is None else None
            sub_loop = in_loop or (tb is None and (trip is None or trip > 1))
            _walk(node.body, in_loop=sub_loop, let_env=let_env)
            return
        if isinstance(node, tirx.While):
            _walk(node.body, in_loop=True, let_env=let_env)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, in_loop=in_loop, let_env=let_env)
            if node.else_case is not None:
                _walk(node.else_case, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, (tirx.SBlock, tirx.AttrStmt, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name = _effective_tirx_call_name(node.value).lower()
            args = _effective_tirx_call_args(node.value)
            if "copy" in op_name and len(args) >= 2:
                src_s, xfer_b, _in1 = _region_bytes(
                    args[0], tirx, DataTypeCls, let_env=let_env
                )
                dst_s, dst_b, _in2 = _region_bytes(
                    args[1], tirx, DataTypeCls, let_env=let_env
                )
                if src_s is not None and dst_s is not None and src_s != dst_s and "GM" in (src_s, dst_s):
                    xfer = max(xfer_b, dst_b)
                    if xfer >= 128:
                        all_sizes.append(xfer)
                        if in_loop:
                            loop_sizes.append(xfer)

    _walk(stmt, in_loop=False, let_env={})
    return loop_sizes, all_sizes


def _count_in_loop_pipe_all_barriers(stmt, *, tirx) -> int:
    """Count pipe_barrier('PIPE_ALL') calls inside multi-trip or symbolic loops."""
    count = [0]
    let_stmt_cls = getattr(tirx, "LetStmt", None)

    def _walk(node, *, in_loop: bool) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child, in_loop=in_loop)
            return
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            trip = _static_tirx_int(node.extent) if tb is None else None
            sub_loop = in_loop or (tb is None and (trip is None or trip > 1))
            _walk(node.body, in_loop=sub_loop)
            return
        if isinstance(node, tirx.While):
            _walk(node.body, in_loop=True)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, in_loop=in_loop)
            if node.else_case is not None:
                _walk(node.else_case, in_loop=in_loop)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, in_loop=in_loop)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            _walk(node.body, in_loop=in_loop)
            return
        if isinstance(node, (tirx.SBlock, tirx.AttrStmt, tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, in_loop=in_loop)
            return
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name = _effective_tirx_call_name(node.value).lower()
            if "pipe_barrier" in op_name and in_loop:
                for arg in node.value.args:
                    val = getattr(arg, "value", arg)
                    if "pipe_all" in str(val).lower():
                        count[0] += 1
                        break

    _walk(stmt, in_loop=False)
    return count[0]


def _is_reduce_or_sfu_call(op_name: str) -> bool:
    lower = op_name.lower()
    return any(
        tok in lower
        for tok in ("vcadd", "vreduce", "reduce_", "vexp", "exp2", ".exp", "vsqrt", "rsqrt", "vdiv")
    ) or lower == "exp"


def _detect_coupled_reduce_sfu_accum(
    stmt,
    wide_ub_names: set[str],
    *,
    tirx,
) -> bool:
    """Return True if a single leaf loop/SimdVF couples reduce+SFU with wide UB RMW."""
    if not wide_ub_names:
        return False

    def _scan_expr_sfu_and_ub(expr, loads: set[str], stores: set[str], sfu_flag: list[bool]) -> None:
        if expr is None:
            return

        def _pre(node):
            if isinstance(node, tirx.Call):
                op_name = _effective_tirx_call_name(node).lower()
                if _is_reduce_or_sfu_call(op_name):
                    sfu_flag[0] = True
            return True

        tirx.stmt_functor.pre_order_visit(expr, _pre)
        ev, _exp, uloads, ustores = _collect_tilelang_expr_rw(expr, tirx=tirx)
        del ev
        loads.update(name for name in uloads if name in wide_ub_names)
        stores.update(name for name in ustores if name in wide_ub_names)

    def _is_scope_node(node) -> bool:
        if isinstance(node, tirx.For) and getattr(node, "thread_binding", None) is None:
            trip = _static_tirx_int(node.extent)
            return trip is None or trip > 1
        if isinstance(node, tirx.SBlock):
            ann = getattr(node, "annotations", None) or {}
            hint = str(getattr(node, "name_hint", "")).lower()
            return "simd_vf" in hint or "tl.simdvf_lanes" in ann
        return False

    def _inspect(node) -> tuple[bool, bool, bool]:
        if node is None:
            return False, False, False
        if _is_scope_node(node):
            direct_loads: set[str] = set()
            direct_stores: set[str] = set()
            direct_sfu = [False]
            child_sfu = False
            child_wide_rmw = False
            coupled = False

            def _walk_scope(sub) -> None:
                nonlocal child_sfu, child_wide_rmw, coupled
                if sub is None:
                    return
                if sub is not node and _is_scope_node(sub):
                    c_sfu, c_rmw, c_coupled = _inspect(sub)
                    child_sfu = child_sfu or c_sfu
                    child_wide_rmw = child_wide_rmw or c_rmw
                    coupled = coupled or c_coupled
                    return
                if isinstance(sub, tirx.SeqStmt):
                    for ch in sub.seq:
                        _walk_scope(ch)
                    return
                if isinstance(sub, tirx.While):
                    _scan_expr_sfu_and_ub(sub.condition, direct_loads, direct_stores, direct_sfu)
                    _walk_scope(sub.body)
                    return
                if isinstance(sub, tirx.For):
                    _walk_scope(sub.body)
                    return
                if isinstance(sub, tirx.IfThenElse):
                    _scan_expr_sfu_and_ub(sub.condition, direct_loads, direct_stores, direct_sfu)
                    _walk_scope(sub.then_case)
                    if sub.else_case is not None:
                        _walk_scope(sub.else_case)
                    return
                if isinstance(sub, tirx.SBlockRealize):
                    _walk_scope(sub.block)
                    return
                if isinstance(sub, tirx.Bind):
                    _scan_expr_sfu_and_ub(sub.value, direct_loads, direct_stores, direct_sfu)
                    if hasattr(sub, "body"):
                        _walk_scope(sub.body)
                    return
                if isinstance(sub, (tirx.SBlock, tirx.AttrStmt, tirx.AllocBuffer, tirx.DeclBuffer)):
                    if hasattr(sub, "body"):
                        _walk_scope(sub.body)
                    return
                if isinstance(sub, tirx.BufferStore):
                    _scan_expr_sfu_and_ub(sub.value, direct_loads, direct_stores, direct_sfu)
                    if (
                        _map_tilelang_scope(sub.buffer.scope()) == "UB"
                        and sub.buffer.name in wide_ub_names
                    ):
                        direct_stores.add(sub.buffer.name)
                    return
                if isinstance(sub, tirx.Evaluate):
                    _scan_expr_sfu_and_ub(sub.value, direct_loads, direct_stores, direct_sfu)

            _walk_scope(node.body)
            direct_rmw = bool(direct_loads & direct_stores)
            if (direct_sfu[0] and (direct_rmw or child_wide_rmw)) or (direct_rmw and child_sfu):
                coupled = True
            return bool(direct_sfu[0] or child_sfu), bool(direct_rmw or child_wide_rmw), coupled

        if isinstance(node, tirx.SeqStmt):
            results = [_inspect(ch) for ch in node.seq]
            return (
                any(r[0] for r in results),
                any(r[1] for r in results),
                any(r[2] for r in results),
            )
        if isinstance(node, (tirx.For, tirx.While, tirx.SBlock, tirx.AttrStmt, tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            return _inspect(node.body) if hasattr(node, "body") else (False, False, False)
        if isinstance(node, tirx.SBlockRealize):
            return _inspect(node.block)
        if isinstance(node, tirx.IfThenElse):
            t_res = _inspect(node.then_case)
            e_res = _inspect(node.else_case) if node.else_case is not None else (False, False, False)
            return (t_res[0] or e_res[0], t_res[1] or e_res[1], t_res[2] or e_res[2])
        return False, False, False

    return _inspect(stmt)[2]


def _dav3510_macs_and_k0(
    dtype_str: str,
    spaces: AscendSpaceFacts | None,
) -> tuple[int, int]:
    """Return (macs_per_cycle, fractal_k0) for Ascend 950 (dav-3510) Cube MMA."""
    base = re.sub(r"x\d+$", "", str(dtype_str).strip().lower())
    if base in ("fp4", "f4", "mxfp4", "f4e2m1", "f4e1m2", "float4_e2m1fn", "int4", "i4", "u4"):
        key, k0, fallback = "f4e2m1", 64, 16384
    elif base in (
        "fp8",
        "f8",
        "fp8e4m3",
        "fp8e5m2",
        "float8_e4m3fn",
        "float8_e5m2",
        "hif8",
        "hifp8",
        "int8",
        "uint8",
        "i8",
        "u8",
    ):
        key = "hif8" if "hif" in base else ("i8" if "i8" in base or "int8" in base else "fp8e4m3")
        k0, fallback = 32, 8192
    elif base in ("hf32", "tf32"):
        key, k0, fallback = "hf32", 8, 2048
    elif base in ("f32", "fp32", "float32"):
        key, k0, fallback = "f32", 8, 256
    else:
        key, k0, fallback = "bf16", 16, 4096
    if spaces is not None and hasattr(spaces, "cube_macs_per_cycle"):
        try:
            return int(spaces.cube_macs_per_cycle(key)), k0
        except ValueError:
            pass
    return fallback, k0


def _arg_shape_and_dtype(arg, tirx) -> tuple[tuple[int, ...] | None, str | None]:
    """Extract static tile extents and dtype string from a TileLang call argument."""
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op))
        if ("region" in op_name or "access_ptr" in op_name) and arg.args:
            first = arg.args[0]
            if isinstance(first, tirx.BufferLoad):
                buf = first.buffer
                if "region" in op_name and len(arg.args) >= 3:
                    ext = _static_shape_ints(arg.args[2:])
                    if ext is not None:
                        return ext, str(buf.dtype)
                return _static_shape_ints(buf.shape), str(buf.dtype)
    if isinstance(arg, tirx.BufferRegion):
        buf = arg.buffer
        ext = _static_shape_ints([r.extent for r in arg.region])
        return (ext if ext is not None else _static_shape_ints(buf.shape)), str(buf.dtype)
    if isinstance(arg, tirx.BufferLoad):
        buf = arg.buffer
        return _static_shape_ints(buf.shape), str(buf.dtype)
    return None, None


def _estimate_dav3510_cube_cycles(
    args: Sequence[object],
    spaces: AscendSpaceFacts | None,
    tirx,
) -> int:
    """Estimate Ascend 950 (dav-3510) Cube PIPE_M cycles with 16x16xK0 alignment & hazard."""
    ext_a, dt_a = _arg_shape_and_dtype(args[0], tirx) if len(args) > 0 else (None, None)
    ext_b, dt_b = _arg_shape_and_dtype(args[1], tirx) if len(args) > 1 else (None, None)
    ext_c, dt_c = _arg_shape_and_dtype(args[2], tirx) if len(args) > 2 else (None, None)
    trans_a = _static_tirx_int(args[3]) if len(args) > 3 else 0
    trans_b = _static_tirx_int(args[4]) if len(args) > 4 else 0
    m = _static_tirx_int(args[5]) if len(args) > 5 else None
    n = _static_tirx_int(args[6]) if len(args) > 6 else None
    k = _static_tirx_int(args[7]) if len(args) > 7 else None
    if ext_a and len(ext_a) >= 2:
        if m is None:
            m = ext_a[-1] if trans_a else ext_a[-2]
        if k is None:
            k = ext_a[-2] if trans_a else ext_a[-1]
    if ext_b and len(ext_b) >= 2:
        if n is None:
            n = ext_b[-2] if trans_b else ext_b[-1]
        if k is None:
            k = ext_b[-1] if trans_b else ext_b[-2]
    if ext_c and len(ext_c) >= 2:
        if m is None:
            m = ext_c[-2]
        if n is None:
            n = ext_c[-1]
    if m and n and k and m > 0 and n > 0 and k > 0:
        macs, k0 = _dav3510_macs_and_k0(dt_a or dt_b or dt_c or "bf16", spaces)
        m_pad = ((m + 15) // 16) * 16
        n_pad = ((n + 15) // 16) * 16
        k_pad = ((k + k0 - 1) // k0) * k0
        cycles = max(1, math.ceil((m_pad * n_pad * k_pad) / float(max(1, macs))))
        tile_blocks = ((m + 15) // 16) * ((n + 15) // 16)
        if tile_blocks < 10:
            cycles += 6
        return cycles
    return 4


def _estimate_dav3510_vec_cycles(
    op_name: str,
    args: Sequence[object],
    tirx,
    DataTypeCls,
) -> int:
    """Estimate Ascend 950 (dav-3510) Vector PIPE_V cycles for 256B RegTensor & SFU ops."""
    max_bytes = 0
    for arg in args:
        _sc, total_b, _in_b = _region_bytes(arg, tirx, DataTypeCls)
        if total_b > max_bytes:
            max_bytes = total_b
        elif total_b == 0 and isinstance(arg, tirx.Var):
            dtype_str = str(getattr(arg, "dtype", ""))
            if dtype_str and dtype_str != "handle":
                try:
                    dt = DataTypeCls(dtype_str)
                    vec_b = int(math.ceil((dt.bits * dt.lanes) / 8.0))
                    max_bytes = max(max_bytes, vec_b)
                except Exception:
                    pass
    elem_bytes = max_bytes if max_bytes > 0 else 256
    base_cycles = max(1, math.ceil(elem_bytes / 256.0))
    lower = op_name.lower()
    sfu_weight = (
        4
        if any(tok in lower for tok in ("exp", "log", "div", "sqrt", "rsqrt"))
        else 1
    )
    return base_cycles * sfu_weight


def _scan_tilelang_ast(
    func,
    spaces: AscendSpaceFacts,
    *,
    tirx,
    DataTypeCls,
    aiv_split_factor: int = 1,
) -> tuple[dict[str, int], int, int, tuple[str, ...], bool, bool]:
    """Walk TileLang PrimFunc AST to collect pipes, MTE2 queue, padding, and dynamic flags."""
    pipe_cycles: dict[str, int] = {}
    opaque_regions: list[str] = []
    has_simtvf = [False]
    illegal_dma = [False]
    padding_bytes = [0]
    max_mte2 = [0]

    def _add_pipe(key: str, cyc: int) -> None:
        pipe_cycles[key] = pipe_cycles.get(key, 0) + max(1, cyc)

    def _emit_aiv_pipe(pipe_name: str, cyc: int, *, guarded_aiv: str | None) -> None:
        if aiv_split_factor >= 2 and guarded_aiv is None:
            per_subcore_cyc = max(1, cyc // 2)
            _add_pipe(f"AIV0.{pipe_name}", per_subcore_cyc)
            _add_pipe(f"AIV1.{pipe_name}", per_subcore_cyc)
        else:
            core = "AIV1" if guarded_aiv == "aiv1" else "AIV0"
            _add_pipe(f"{core}.{pipe_name}", cyc)

    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)

    def _walk(
        stmt,
        *,
        repeat: int = 1,
        mte2_run: list[int] | None = None,
        sid_vars: set[object] | None = None,
        guarded_aiv: str | None = None,
        let_env: dict[object, object] | None = None,
    ) -> None:
        if mte2_run is None:
            mte2_run = [0]
        if sid_vars is None:
            sid_vars = set()
        if let_env is None:
            let_env = {}
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            seq_env = dict(let_env)
            for child in stmt.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(
                        child.value, seq_env, tirx=tirx
                    )
                _walk(
                    child,
                    repeat=repeat,
                    mte2_run=mte2_run,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
                    let_env=dict(seq_env),
                )
            return
        if let_stmt_cls is not None and isinstance(stmt, let_stmt_cls):
            next_env = dict(let_env)
            next_env[stmt.var] = _substitute_let_env(stmt.value, let_env, tirx=tirx)
            _walk(
                stmt.body,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
                let_env=next_env,
            )
            return
        if isinstance(stmt, tirx.While):
            if "while_loop" not in opaque_regions:
                opaque_regions.append("while_loop")
            if _contains_buffer_load(stmt.condition, tirx):
                if "data_dependent_branch" not in opaque_regions:
                    opaque_regions.append("data_dependent_branch")
            _walk(
                stmt.body,
                repeat=repeat,
                mte2_run=[0],
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
                let_env=let_env,
            )
            return
        if isinstance(stmt, tirx.For):
            tb = getattr(stmt, "thread_binding", None)
            if tb is not None:
                tb_tag = str(getattr(tb, "thread_tag", "")).lower()
                next_sids = set(sid_vars)
                var_name = str(getattr(stmt.loop_var, "name_hint", "")).lower()
                if "cthread" in tb_tag or "subblock" in tb_tag or var_name == "sid":
                    next_sids.add(stmt.loop_var)
                _walk(
                    stmt.body,
                    repeat=repeat,
                    mte2_run=mte2_run,
                    sid_vars=next_sids,
                    guarded_aiv=guarded_aiv,
                    let_env=let_env,
                )
                return
            trip = _static_tirx_int(stmt.extent)
            if trip is None:
                tag = f"dynamic_for:{stmt.loop_var.name}"
                if tag not in opaque_regions:
                    opaque_regions.append(tag)
                _walk(
                    stmt.body,
                    repeat=repeat,
                    mte2_run=[0],
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
                    let_env=let_env,
                )
            elif trip > 0:
                local_mte2 = [0]
                _walk(
                    stmt.body,
                    repeat=repeat * trip,
                    mte2_run=local_mte2,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
                    let_env=let_env,
                )
                if local_mte2[0] > 0:
                    mte2_run[0] += local_mte2[0] * trip
                    max_mte2[0] = max(max_mte2[0], mte2_run[0])
            return
        if isinstance(stmt, tirx.IfThenElse):
            if _contains_buffer_load(stmt.condition, tirx):
                if "data_dependent_branch" not in opaque_regions:
                    opaque_regions.append("data_dependent_branch")
            then_guard, else_guard = _sid_guard_branches(
                stmt.condition, sid_vars, tirx, aiv_split_factor
            )
            next_then = then_guard if then_guard is not None else guarded_aiv
            next_else = else_guard if else_guard is not None else guarded_aiv
            b1 = [mte2_run[0]]
            _walk(
                stmt.then_case,
                repeat=repeat,
                mte2_run=b1,
                sid_vars=sid_vars,
                guarded_aiv=next_then,
                let_env=let_env,
            )
            b2 = [mte2_run[0]]
            if stmt.else_case is not None:
                _walk(
                    stmt.else_case,
                    repeat=repeat,
                    mte2_run=b2,
                    sid_vars=sid_vars,
                    guarded_aiv=next_else,
                    let_env=let_env,
                )
            mte2_run[0] = max(b1[0], b2[0])
            return
        if isinstance(stmt, tirx.AttrStmt):
            key = str(stmt.attr_key)
            next_sids = set(sid_vars)
            if "thread_extent" in key:
                tag = str(getattr(stmt.node, "thread_tag", "")).lower()
                iv = getattr(stmt.node, "var", None)
                iv_name = str(getattr(iv, "name_hint", "")).lower()
                if "cthread" in tag or "subblock" in tag or "cthread" in iv_name or "subblock" in iv_name:
                    if iv is not None:
                        next_sids.add(iv)
                elif "blockidx" not in tag and "blockidx" not in iv_name:
                    has_simtvf[0] = True
                    if "simt_vf" not in opaque_regions:
                        opaque_regions.append("simt_vf")
            elif key == "pragma_parallel" or "simt" in key.lower():
                has_simtvf[0] = True
            _walk(
                stmt.body,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=next_sids,
                guarded_aiv=guarded_aiv,
                let_env=let_env,
            )
            return
        if isinstance(stmt, (tirx.SBlockRealize,)):
            _walk(
                stmt.block,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
                let_env=let_env,
            )
            return
        if isinstance(stmt, tirx.SBlock):
            _walk(
                stmt.body,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
                let_env=let_env,
            )
            return
        if isinstance(stmt, (tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            next_env = let_env
            if isinstance(stmt, tirx.Bind):
                next_env = dict(let_env)
                next_env[stmt.var] = _substitute_let_env(
                    stmt.value, let_env, tirx=tirx
                )
            if isinstance(stmt, tirx.Bind) and isinstance(getattr(stmt, "value", None), tirx.Call):
                b_op = _effective_tirx_call_name(stmt.value).lower()
                b_args = _effective_tirx_call_args(stmt.value)
                if (
                    ("simd" in b_op or "vec" in b_op)
                    and "vld" not in b_op
                    and "pset" not in b_op
                ):
                    mte2_run[0] = 0
                    vec_cyc = _estimate_dav3510_vec_cycles(
                        b_op, b_args, tirx, DataTypeCls
                    )
                    _emit_aiv_pipe("PIPE_V", repeat * vec_cyc, guarded_aiv=guarded_aiv)
            if hasattr(stmt, "body"):
                _walk(
                    stmt.body,
                    repeat=repeat,
                    mte2_run=mte2_run,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
                    let_env=next_env,
                )
            return
        if isinstance(stmt, tirx.BufferStore):
            if mte2_run[0] > 0:
                mte2_run[0] = 0
            core = "AIV1" if guarded_aiv == "aiv1" else "AIV0"
            _add_pipe(f"{core}.PIPE_S", repeat)
            return
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            op_name = _effective_tirx_call_name(stmt.value).lower()
            args = _effective_tirx_call_args(stmt.value)
            if "copy" in op_name and len(args) >= 2:
                src_s, xfer_b, inner_b = _region_bytes(
                    args[0], tirx, DataTypeCls, let_env=let_env
                )
                dst_s, dst_b, dst_inner_b = _region_bytes(
                    args[1], tirx, DataTypeCls, let_env=let_env
                )
                if src_s is not None and dst_s is not None:
                    eff_inner_b = inner_b if inner_b > 0 else dst_inner_b
                    xfer = max(xfer_b, dst_b)
                    if xfer == 0 and eff_inner_b > 0:
                        xfer = eff_inner_b
                    if src_s != dst_s and not spaces.is_dma_path_legal(src_s, dst_s):
                        illegal_dma[0] = True
                    if eff_inner_b > 0 and src_s == "GM" and dst_s == "UB":
                        rem = eff_inner_b % spaces.ub_align_bytes
                        if rem != 0:
                            padding_bytes[0] += (spaces.ub_align_bytes - rem) * repeat
                        mte2_run[0] += 1
                        max_mte2[0] = max(max_mte2[0], mte2_run[0])
                        _emit_aiv_pipe(
                            "PIPE_MTE2",
                            repeat * _burst_scaled_copy_cycles(xfer, spaces),
                            guarded_aiv=guarded_aiv,
                        )
                    elif src_s == "UB" and dst_s == "GM":
                        mte2_run[0] = 0
                        _emit_aiv_pipe(
                            "PIPE_MTE3",
                            repeat * _burst_scaled_copy_cycles(xfer, spaces),
                            guarded_aiv=guarded_aiv,
                        )
                    elif src_s == "UB" and dst_s == "L1":
                        mte2_run[0] = 0
                        _add_pipe("AIC.PIPE_MTE1", repeat * max(1, (xfer + 255) // 256))
                        _emit_aiv_pipe(
                            "PIPE_MTE3",
                            repeat * max(1, (xfer + 63) // 64),
                            guarded_aiv=guarded_aiv,
                        )
                    elif src_s == "GM" and dst_s == "L1":
                        _add_pipe("AIC.PIPE_MTE2", repeat * _burst_scaled_copy_cycles(xfer, spaces))
                    elif src_s == "L1" and dst_s in ("L0A", "L0B"):
                        _add_pipe("AIC.PIPE_MTE1", repeat * max(1, (xfer + 255) // 256))
                    elif src_s == "L0C" and dst_s in ("GM", "L1", "UB"):
                        fix_rate = max(1, spaces.fixpipe_bytes_per_cycle)
                        _add_pipe(
                            "AIC.PIPE_FIX",
                            repeat * max(1, (xfer + fix_rate - 1) // fix_rate),
                        )
                        if dst_s == "UB":
                            _emit_aiv_pipe(
                                "PIPE_MTE2",
                                repeat * max(1, (xfer + 63) // 64),
                                guarded_aiv=guarded_aiv,
                            )
            elif "gemm" in op_name or "mmad" in op_name:
                mte2_run[0] = 0
                gemm_cyc = _estimate_dav3510_cube_cycles(args, spaces, tirx)
                _add_pipe("AIC.PIPE_M", repeat * gemm_cyc)
            elif "simd" in op_name or "vec" in op_name:
                mte2_run[0] = 0
                vec_cyc = _estimate_dav3510_vec_cycles(op_name, args, tirx, DataTypeCls)
                _emit_aiv_pipe("PIPE_V", repeat * vec_cyc, guarded_aiv=guarded_aiv)

    _walk(func.body)
    return (
        pipe_cycles,
        max_mte2[0],
        padding_bytes[0],
        tuple(opaque_regions),
        has_simtvf[0],
        illegal_dma[0],
    )


@dataclass(frozen=True)
class _TileLangLoopTask:
    label: str
    mask: NodeResourceMask
    latency: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _is_pipelined_tirx_for(stmt, tirx) -> bool:
    if not isinstance(stmt, tirx.For):
        return False
    kind_val = int(stmt.kind) if hasattr(stmt.kind, "value") else stmt.kind
    ann = getattr(stmt, "annotations", None) or {}
    return "num_stages" in ann or kind_val == 5


def _collect_tilelang_buf_meta(
    func,
    *,
    tirx,
    DataTypeCls,
) -> dict[str, tuple[str, int]]:
    meta: dict[str, tuple[str, int]] = {}

    def _record(buf) -> None:
        scope = _map_tilelang_scope(buf.scope())
        if scope is None or scope == "GM":
            return
        shape = _static_shape_ints(buf.shape)
        nbytes = _buffer_bytes(shape, buf.dtype, DataTypeCls)
        if nbytes > 0:
            prev = meta.get(buf.name)
            if prev is None or nbytes > prev[1]:
                meta[buf.name] = (scope, nbytes)

    def _visit(node) -> None:
        if isinstance(node, tirx.SBlock):
            for buf in node.alloc_buffers:
                _record(buf)
        elif isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            _record(node.buffer)

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    return meta


def _collect_local_buf_names(stmt, *, tirx) -> set[str]:
    names: set[str] = set()

    def _visit(node) -> None:
        if isinstance(node, tirx.SBlock):
            for buf in node.alloc_buffers:
                names.add(buf.name)
        elif isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            names.add(node.buffer.name)

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return names


def _tilelang_arg_buf(arg, tirx, DataTypeCls) -> tuple[str | None, str | None, int]:
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op))
        if ("region" in op_name or "access_ptr" in op_name) and arg.args:
            first = arg.args[0]
            if isinstance(first, tirx.BufferLoad):
                buf = first.buffer
                scope = _map_tilelang_scope(buf.scope())
                shape = _static_shape_ints(buf.shape)
                full_b = _buffer_bytes(shape, buf.dtype, DataTypeCls)
                if "region" in op_name and len(arg.args) >= 3:
                    ext = _static_shape_ints(arg.args[2:])
                    part_b = _buffer_bytes(ext, buf.dtype, DataTypeCls)
                    return buf.name, scope, part_b if part_b > 0 else full_b
                return buf.name, scope, full_b
    if isinstance(arg, tirx.BufferRegion):
        buf = arg.buffer
        scope = _map_tilelang_scope(buf.scope())
        ext = _static_shape_ints([r.extent for r in arg.region])
        part_b = _buffer_bytes(ext, buf.dtype, DataTypeCls)
        full_b = _buffer_bytes(_static_shape_ints(buf.shape), buf.dtype, DataTypeCls)
        return buf.name, scope, part_b if part_b > 0 else full_b
    if isinstance(arg, tirx.BufferLoad):
        buf = arg.buffer
        scope = _map_tilelang_scope(buf.scope())
        full_b = _buffer_bytes(_static_shape_ints(buf.shape), buf.dtype, DataTypeCls)
        return buf.name, scope, full_b
    return None, None, 0


def _aiv_mask(pipe: str, *, aiv_split_factor: int, guarded_aiv: str | None) -> NodeResourceMask:
    if aiv_split_factor >= 2 and guarded_aiv is None:
        return NodeResourceMask.for_dual_aiv(pipe)
    return NodeResourceMask.for_core(guarded_aiv or "aiv0", pipe)


def _copy_resource_mask(
    src_scope: str,
    dst_scope: str,
    *,
    aiv_split_factor: int,
    guarded_aiv: str | None,
) -> NodeResourceMask:
    if src_scope == "GM" and dst_scope == "UB":
        return _aiv_mask("MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_scope == "UB" and dst_scope == "GM":
        return _aiv_mask("MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_scope == "GM" and dst_scope == "L1":
        return NodeResourceMask.for_aic("MTE2")
    if src_scope == "L1" and dst_scope in ("L0A", "L0B"):
        return NodeResourceMask.for_aic("MTE1")
    if src_scope == "L0C" and dst_scope in ("GM", "L1"):
        return NodeResourceMask.for_aic("FIXPIPE")
    if src_scope == "L0C" and dst_scope == "UB":
        return NodeResourceMask.for_aic("FIXPIPE") | _aiv_mask(
            "MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv
        )
    if src_scope == "UB" and dst_scope == "L1":
        return NodeResourceMask.for_aic("MTE1") | _aiv_mask(
            "MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv
        )
    if src_scope == "UB" or dst_scope == "UB":
        return _aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    return NodeResourceMask.for_aic("MTE1")


def _extract_vf_accesses(
    stmt,
    *,
    tirx,
    DataTypeCls,
) -> tuple[list[str], list[str], int]:
    reads: list[str] = []
    writes: list[str] = []
    op_count = [0]

    def _add_unique(target: list[str], name: str | None, scope: str | None) -> None:
        if name is not None and scope is not None and scope != "GM" and name not in target:
            target.append(name)

    def _visit(node) -> None:
        if isinstance(node, tirx.BufferStore):
            op_count[0] += 1
            scope = _map_tilelang_scope(node.buffer.scope())
            _add_unique(writes, node.buffer.name, scope)
        elif isinstance(node, tirx.Call):
            op_name = _effective_tirx_call_name(node).lower()
            if "simd" in op_name or "vec" in op_name:
                op_count[0] += 1
            if "access_ptr" in op_name and len(node.args) >= 3:
                name, scope, _ = _tilelang_arg_buf(node, tirx, DataTypeCls)
                rw = _static_tirx_int(node.args[2]) or 1
                if rw & 1:
                    _add_unique(reads, name, scope)
                if rw & 2:
                    _add_unique(writes, name, scope)
            elif "region" in op_name and len(node.args) >= 2:
                name, scope, _ = _tilelang_arg_buf(node, tirx, DataTypeCls)
                rw = _static_tirx_int(node.args[1]) or 1
                if rw & 1:
                    _add_unique(reads, name, scope)
                if rw & 2:
                    _add_unique(writes, name, scope)
        elif isinstance(node, tirx.BufferLoad):
            scope = _map_tilelang_scope(node.buffer.scope())
            _add_unique(reads, node.buffer.name, scope)

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return reads, writes, max(1, op_count[0])


def _extract_pipelined_loop_tasks(
    loop_body,
    *,
    tirx,
    DataTypeCls,
    aiv_split_factor: int,
    sid_vars: set[object],
    guarded_aiv: str | None,
    spaces: AscendSpaceFacts | None = None,
    dynamic_inner_out: list[bool] | None = None,
) -> list[_TileLangLoopTask] | None:
    tasks: list[_TileLangLoopTask] = []
    aborted = [False]

    def _walk(node, *, scale: int, cur_guard: str | None) -> None:
        if node is None or aborted[0]:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child, scale=scale, cur_guard=cur_guard)
            return
        if isinstance(node, tirx.While):
            if dynamic_inner_out is not None:
                dynamic_inner_out[0] = True
                _walk(node.body, scale=scale, cur_guard=cur_guard)
            else:
                aborted[0] = True
            return
        if isinstance(node, tirx.For):
            trip = _static_tirx_int(node.extent)
            if trip is None:
                if dynamic_inner_out is not None:
                    dynamic_inner_out[0] = True
                    _walk(node.body, scale=scale, cur_guard=cur_guard)
                else:
                    aborted[0] = True
                return
            if trip > 0:
                _walk(node.body, scale=min(8, scale * trip), cur_guard=cur_guard)
            return
        if isinstance(node, tirx.IfThenElse):
            if _contains_buffer_load(node.condition, tirx):
                if dynamic_inner_out is not None:
                    dynamic_inner_out[0] = True
                else:
                    aborted[0] = True
                    return
            t_guard, e_guard = _sid_guard_branches(
                node.condition, sid_vars, tirx, aiv_split_factor
            )
            _walk(
                node.then_case,
                scale=scale,
                cur_guard=t_guard if t_guard is not None else cur_guard,
            )
            if node.else_case is not None:
                _walk(
                    node.else_case,
                    scale=scale,
                    cur_guard=e_guard if e_guard is not None else cur_guard,
                )
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, scale=scale, cur_guard=cur_guard)
            return
        if isinstance(node, tirx.SBlock):
            ann = getattr(node, "annotations", None) or {}
            hint = str(getattr(node, "name_hint", "")).lower()
            if "simd_vf" in hint or "simt_vf" in hint or "tl.simdvf_lanes" in ann:
                reads, writes, vf_ops = _extract_vf_accesses(
                    node.body, tirx=tirx, DataTypeCls=DataTypeCls
                )
                lat = min(8, max(1, (vf_ops + 3) // 4) * scale)
                mask = _aiv_mask(
                    "VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard
                )
                tasks.append(
                    _TileLangLoopTask(
                        label="simdvf",
                        mask=mask,
                        latency=lat,
                        reads=tuple(reads),
                        writes=tuple(writes),
                    )
                )
                return
            _walk(node.body, scale=scale, cur_guard=cur_guard)
            return
        if isinstance(node, tirx.AttrStmt):
            key = str(node.attr_key).lower()
            if "simdvf_scope" in key or "simtvf_scope" in key:
                reads, writes, vf_ops = _extract_vf_accesses(
                    node.body, tirx=tirx, DataTypeCls=DataTypeCls
                )
                lat = min(8, max(1, (vf_ops + 3) // 4) * scale)
                mask = _aiv_mask(
                    "VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard
                )
                tasks.append(
                    _TileLangLoopTask(
                        label="simdvf",
                        mask=mask,
                        latency=lat,
                        reads=tuple(reads),
                        writes=tuple(writes),
                    )
                )
                return
            _walk(node.body, scale=scale, cur_guard=cur_guard)
            return
        if isinstance(node, (tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, scale=scale, cur_guard=cur_guard)
            return
        if isinstance(node, tirx.BufferStore):
            scope = _map_tilelang_scope(node.buffer.scope())
            writes = (node.buffer.name,) if scope is not None and scope != "GM" else ()
            reads, _, _ = _extract_vf_accesses(node.value, tirx=tirx, DataTypeCls=DataTypeCls)
            mask = NodeResourceMask.for_core(cur_guard or "aiv0", "SCALAR")
            tasks.append(
                _TileLangLoopTask(
                    label="scalar_store",
                    mask=mask,
                    latency=1,
                    reads=tuple(reads),
                    writes=writes,
                )
            )
            return
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name = _effective_tirx_call_name(node.value).lower()
            args = _effective_tirx_call_args(node.value)
            if "copy" in op_name and len(args) >= 2:
                src_n, src_s, src_b = _tilelang_arg_buf(args[0], tirx, DataTypeCls)
                dst_n, dst_s, dst_b = _tilelang_arg_buf(args[1], tirx, DataTypeCls)
                if src_s is not None and dst_s is not None:
                    mask = _copy_resource_mask(
                        src_s,
                        dst_s,
                        aiv_split_factor=aiv_split_factor,
                        guarded_aiv=cur_guard,
                    )
                    xfer = max(src_b, dst_b)
                    eff_xfer = xfer
                    if spaces is not None and xfer > 0 and ("GM" in (src_s, dst_s)):
                        eff_xfer = int(
                            math.ceil(float(xfer) / max(0.15, float(spaces.dma_burst_efficiency(xfer))))
                        )
                    lat = min(8, max(1, (eff_xfer + 16383) // 16384) * scale)
                    reads = (src_n,) if (src_n is not None and src_s != "GM") else ()
                    writes = (dst_n,) if (dst_n is not None and dst_s != "GM") else ()
                    tasks.append(
                        _TileLangLoopTask(
                            label="copy",
                            mask=mask,
                            latency=lat,
                            reads=reads,
                            writes=writes,
                        )
                    )
                return
            if ("gemm" in op_name or "mmad" in op_name) and len(args) >= 3:
                a_n, a_s, _ = _tilelang_arg_buf(args[0], tirx, DataTypeCls)
                b_n, b_s, _ = _tilelang_arg_buf(args[1], tirx, DataTypeCls)
                c_n, c_s, _ = _tilelang_arg_buf(args[2], tirx, DataTypeCls)
                gemm_cyc = _estimate_dav3510_cube_cycles(args, spaces, tirx)
                lat = min(8, max(1, gemm_cyc) * scale)
                clear_val = _static_tirx_int(args[9]) if len(args) > 9 else None
                c_is_read = clear_val is None or clear_val == 0
                reads_list: list[str] = []
                if a_n is not None and a_s not in (None, "GM"):
                    reads_list.append(a_n)
                if b_n is not None and b_s not in (None, "GM"):
                    reads_list.append(b_n)
                if c_is_read and c_n is not None and c_s not in (None, "GM") and c_n not in reads_list:
                    reads_list.append(c_n)
                writes = (c_n,) if (c_n is not None and c_s not in (None, "GM")) else ()
                tasks.append(
                    _TileLangLoopTask(
                        label="gemm",
                        mask=NodeResourceMask.for_aic("CUBE"),
                        latency=lat,
                        reads=tuple(reads_list),
                        writes=writes,
                    )
                )
                return
            if "simd" in op_name or "vec" in op_name:
                reads, writes, _ = _extract_vf_accesses(
                    node.value, tirx=tirx, DataTypeCls=DataTypeCls
                )
                mask = _aiv_mask(
                    "VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard
                )
                vec_cyc = _estimate_dav3510_vec_cycles(op_name, args, tirx, DataTypeCls)
                tasks.append(
                    _TileLangLoopTask(
                        label="vec",
                        mask=mask,
                        latency=min(8, vec_cyc * scale),
                        reads=tuple(reads),
                        writes=tuple(writes),
                    )
                )

    _walk(loop_body, scale=1, cur_guard=guarded_aiv)
    if aborted[0]:
        return None
    return tasks


def _analytical_pipelined_ii(task_items: Sequence[_TileLangLoopTask]) -> int:
    """Compute analytical steady-state II lower bound (max_pipe_cycle) across overlapped stages."""
    if not task_items:
        return 1
    pipe_totals: dict[tuple[str, str], int] = {}
    for item in task_items:
        lat = max(1, int(item.latency))
        for core_key, pipe_name in item.mask.to_core_pipe_pairs():
            pipe_totals[(core_key, pipe_name)] = pipe_totals.get((core_key, pipe_name), 0) + lat
    return max(pipe_totals.values(), default=1)


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
    """Solve Mode A Z3 steady-state II for static T.Pipelined TileLang loops."""
    if entry_fn is None or getattr(entry_fn, "body", None) is None:
        return None

    pipelined_loops: list[tuple[object, set[object], str | None]] = []

    def _find_loops(stmt, *, sid_vars: set[object], cur_guard: str | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                _find_loops(child, sid_vars=sid_vars, cur_guard=cur_guard)
            return
        if isinstance(stmt, tirx.For):
            if _is_pipelined_tirx_for(stmt, tirx):
                pipelined_loops.append((stmt, set(sid_vars), cur_guard))
                return
            tb = getattr(stmt, "thread_binding", None)
            next_sids = set(sid_vars)
            if tb is not None:
                tb_tag = str(getattr(tb, "thread_tag", "")).lower()
                var_name = str(getattr(stmt.loop_var, "name_hint", "")).lower()
                if "cthread" in tb_tag or "subblock" in tb_tag or var_name == "sid":
                    next_sids.add(stmt.loop_var)
            _find_loops(stmt.body, sid_vars=next_sids, cur_guard=cur_guard)
            return
        if isinstance(stmt, tirx.IfThenElse):
            t_guard, e_guard = _sid_guard_branches(
                stmt.condition, sid_vars, tirx, aiv_split_factor
            )
            _find_loops(
                stmt.then_case,
                sid_vars=sid_vars,
                cur_guard=t_guard if t_guard is not None else cur_guard,
            )
            if stmt.else_case is not None:
                _find_loops(
                    stmt.else_case,
                    sid_vars=sid_vars,
                    cur_guard=e_guard if e_guard is not None else cur_guard,
                )
            return
        if isinstance(stmt, tirx.AttrStmt):
            next_sids = set(sid_vars)
            if "thread_extent" in str(stmt.attr_key).lower():
                tag = str(getattr(stmt.node, "thread_tag", "")).lower()
                iv = getattr(stmt.node, "var", None)
                iv_name = str(getattr(iv, "name_hint", "")).lower()
                if (
                    ("cthread" in tag or "subblock" in tag or "cthread" in iv_name or "subblock" in iv_name)
                    and iv is not None
                ):
                    next_sids.add(iv)
            _find_loops(stmt.body, sid_vars=next_sids, cur_guard=cur_guard)
            return
        if isinstance(stmt, tirx.SBlockRealize):
            _find_loops(stmt.block, sid_vars=sid_vars, cur_guard=cur_guard)
            return
        if isinstance(stmt, (tirx.SBlock, tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(stmt, "body"):
                _find_loops(stmt.body, sid_vars=sid_vars, cur_guard=cur_guard)

    _find_loops(entry_fn.body, sid_vars=set(), cur_guard=None)
    if not pipelined_loops:
        return None

    buf_meta = _collect_tilelang_buf_meta(entry_fn, tirx=tirx, DataTypeCls=DataTypeCls)
    all_buf_names = set(buf_meta.keys())
    degraded_set = set(degraded_names)
    solved_iis: list[int] = []

    for loop, sid_vars, loop_guard in pipelined_loops:
        trip = _static_tirx_int(loop.extent)
        if trip is None or trip < 1:
            return None
        ann = getattr(loop, "annotations", None) or {}
        stages_val = _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else 2
        num_stages = max(1, stages_val if stages_val is not None else 2)
        local_bufs = _collect_local_buf_names(loop.body, tirx=tirx)
        outer_bufs = all_buf_names - local_bufs
        dynamic_inner = [False]
        task_items = _extract_pipelined_loop_tasks(
            loop.body,
            tirx=tirx,
            DataTypeCls=DataTypeCls,
            aiv_split_factor=aiv_split_factor,
            sid_vars=sid_vars,
            guarded_aiv=loop_guard,
            spaces=spaces,
            dynamic_inner_out=dynamic_inner,
        )
        if task_items is None:
            return None
        if not task_items:
            continue
        if dynamic_inner[0]:
            solved_iis.append(_analytical_pipelined_ii(task_items))
            continue

        referenced_bufs: set[str] = set()
        last_writer: dict[str, str] = {}
        readers_since_write: dict[str, list[str]] = {}
        all_readers: dict[str, list[str]] = {}
        first_readers_before_write: dict[str, list[str]] = {}
        task_latency_by_name: dict[str, int] = {}
        task_names: list[str] = []
        task_specs: list[tuple[str, tuple, int]] = []
        edges: list[Edge] = []
        reuses: list[Reuse] = []
        edge_keys: set[tuple[str, str, int, int]] = set()
        reuse_keys: set[tuple[str, str, str]] = set()

        for idx, item in enumerate(task_items):
            t_name = f"t{idx}_{item.label}"
            res_uses = item.mask.to_solver_resources(1, 0)
            task_latency_by_name[t_name] = item.latency
            task_names.append(t_name)
            task_specs.append((t_name, res_uses, item.latency))

            for buf_name in item.reads:
                if buf_name not in buf_meta:
                    continue
                referenced_bufs.add(buf_name)
                if buf_name in last_writer:
                    writer = last_writer[buf_name]
                    if writer != t_name:
                        w_lat = task_latency_by_name[writer]
                        e_key = (writer, t_name, 0, w_lat)
                        if e_key not in edge_keys:
                            edge_keys.add(e_key)
                            edges.append(Edge(src=writer, dst=t_name, distance=0, latency=w_lat))
                        all_readers.setdefault(buf_name, []).append(t_name)
                        readers_since_write.setdefault(buf_name, []).append(t_name)
                elif buf_name in outer_bufs:
                    first_readers = first_readers_before_write.setdefault(buf_name, [])
                    if t_name not in first_readers:
                        first_readers.append(t_name)

            for buf_name in item.writes:
                if buf_name not in buf_meta:
                    continue
                referenced_bufs.add(buf_name)
                if buf_name in last_writer:
                    prev_writer = last_writer[buf_name]
                    for r_name in readers_since_write.get(buf_name, ()):
                        if r_name != prev_writer:
                            r_key = (buf_name, r_name, prev_writer)
                            if r_key not in reuse_keys:
                                reuse_keys.add(r_key)
                                reuses.append(
                                    Reuse(
                                        buffer=buf_name,
                                        last_reader=r_name,
                                        next_writer=prev_writer,
                                        delta=0,
                                        read_latency=1,
                                    )
                                )
                    readers_since_write[buf_name] = []
                last_writer[buf_name] = t_name

        loop_carried_buffers: set[str] = set()
        for buf_name, writer in last_writer.items():
            if buf_name in first_readers_before_write:
                loop_carried_buffers.add(buf_name)
                w_lat = task_latency_by_name[writer]
                for r_name in first_readers_before_write[buf_name]:
                    e_key = (writer, r_name, 1, w_lat)
                    if e_key not in edge_keys:
                        edge_keys.add(e_key)
                        edges.append(Edge(src=writer, dst=r_name, distance=1, latency=w_lat))
                continue
            for r_name in all_readers.get(buf_name, ()):
                if r_name != writer:
                    r_key = (buf_name, r_name, writer)
                    if r_key not in reuse_keys:
                        reuse_keys.add(r_key)
                        reuses.append(
                            Reuse(
                                buffer=buf_name,
                                last_reader=r_name,
                                next_writer=writer,
                                delta=0,
                                read_latency=1,
                            )
                        )

        max_stage = max(0, num_stages - 1)
        if max_stage > 0:
            assigned_stages = _assign_dag_stages(task_names, edges, {}, max_stage)
            manual = True
        else:
            assigned_stages = {name: 0 for name in task_names}
            manual = False

        tasks = [
            Task(
                name=name,
                resources=res_uses,
                latency=lat,
                stage=assigned_stages[name] if manual else None,
            )
            for name, res_uses, lat in task_specs
        ]

        reused_buffers = {r.buffer for r in reuses}
        buffers: list[Buffer] = []
        has_ssbuf = False
        for buf_name in sorted(referenced_bufs):
            space, slot_bytes = buf_meta[buf_name]
            if space == "SSBUF":
                has_ssbuf = True
                owner = spaces.ssbuf_owner
            elif space == "UB":
                owner = "aiv"
            else:
                owner = "aic"
            if buf_name in degraded_set or buf_name in loop_carried_buffers:
                buffers.append(
                    Buffer(name=buf_name, slot_bytes=slot_bytes, space=space, owner=owner, fixed=1)
                )
            elif manual and num_stages > 1 and buf_name in reused_buffers:
                buffers.append(
                    Buffer(
                        name=buf_name,
                        slot_bytes=slot_bytes,
                        space=space,
                        owner=owner,
                        auto_min=1,
                        auto_max=num_stages,
                    )
                )
            else:
                buffers.append(
                    Buffer(name=buf_name, slot_bytes=slot_bytes, space=space, owner=owner, fixed=1)
                )

        capacities = _capacities_from_facts(
            spaces, has_simtvf=has_simtvf, unreserved_ubuf=unreserved_ubuf
        )
        if has_ssbuf:
            capacities = (
                *capacities,
                Capacity("SSBUF", spaces.ssbuf_owner, spaces.ssbuf_bytes),
            )
        problem = tf_adapter(
            {
                "tasks": tuple(tasks),
                "edges": tuple(edges),
                "buffers": tuple(buffers),
                "reuses": tuple(reuses),
                "resources": _resources_for_tasks(tasks),
                "capacities": capacities,
                "max_stage": max_stage,
                "manual": manual,
                "issue_after": (),
            }
        )
        res = solve_pipeline_with_monotone_backoff(problem)
        solved_iis.append(res.assignment.ii)

    return max(solved_iis) if solved_iis else None


def _to_python_attr_value(val: object) -> object:
    if isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "value") and isinstance(getattr(val, "value"), (bool, int, float, str)):
        return getattr(val, "value")
    if hasattr(val, "items"):
        return {str(k): _to_python_attr_value(v) for k, v in val.items()}
    if isinstance(val, (tuple, list)) or (
        hasattr(val, "__iter__") and not isinstance(val, (bytes, bytearray))
    ):
        try:
            return [_to_python_attr_value(item) for item in val]
        except TypeError:
            return str(val)
    return str(val)


def _normalize_compile_flags(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        text = raw_flags.strip()
        return text.split() if text else []
    if isinstance(raw_flags, (tuple, list)) or (
        hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray))
    ):
        out: list[str] = []
        for item in raw_flags:
            item_str = str(_to_python_attr_value(item)).strip()
            if item_str:
                out.append(item_str)
        return out
    text = str(raw_flags).strip()
    return [text] if text else []


def _extract_primfunc_pass_configs_and_flags(
    entry_fn,
    mod_in,
    *,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
) -> tuple[dict[str, object], list[str]]:
    cfg: dict[str, object] = {}
    flags: list[str] = []

    def _append_flags(items: Sequence[str]) -> None:
        for item in items:
            if item and item not in flags:
                flags.append(item)

    for owner in (mod_in, entry_fn):
        attrs = getattr(owner, "attrs", None)
        if not attrs or not hasattr(attrs, "items"):
            continue
        for pc_key in ("tl.pass_configs", "tilelang_pass_configs"):
            if pc_key in attrs:
                raw_pc = _to_python_attr_value(attrs[pc_key])
                if isinstance(raw_pc, dict):
                    for k, v in raw_pc.items():
                        if k == "tl.device_compile_flags":
                            _append_flags(_normalize_compile_flags(v))
                        else:
                            cfg[str(k)] = v
        for cf_key in ("tir.cce_flags", "tilelang_compile_flags"):
            if cf_key in attrs:
                _append_flags(_normalize_compile_flags(attrs[cf_key]))

    if pass_configs:
        for k, v in pass_configs.items():
            if str(k) == "tl.device_compile_flags":
                _append_flags(_normalize_compile_flags(v))
            else:
                cfg[str(k)] = v

    _append_flags(_normalize_compile_flags(compile_flags))
    if flags:
        cfg["tl.device_compile_flags"] = list(flags)
    return cfg, flags


def _collect_div_denominators(entry_fn, *, tirx=None) -> list[object]:
    """Collect denominator PrimExprs from division/modulo ops across body and buffers."""
    if tirx is None:
        from tvm import tirx as _tirx  # noqa: PLC0415

        tirx = _tirx
    div_classes = tuple(
        cls
        for name in ("FloorDiv", "FloorMod", "Div", "Mod")
        if (cls := getattr(tirx, name, None)) is not None
    )
    if not div_classes or entry_fn is None:
        return []
    raw_denoms: list[object] = []
    let_env: dict[object, object] = {}
    let_classes = tuple(
        cls
        for name in ("LetStmt", "Bind", "Let")
        if (cls := getattr(tirx, name, None)) is not None
    )
    alloc_classes = tuple(
        cls
        for name in ("AllocBuffer", "DeclBuffer")
        if (cls := getattr(tirx, name, None)) is not None
    )
    sblock_cls = getattr(tirx, "SBlock", None)

    def _add_denom(expr) -> None:
        if expr is None:
            return
        if not any(
            (hasattr(expr, "same_as") and expr.same_as(prev)) or expr is prev
            for prev in raw_denoms
        ):
            raw_denoms.append(expr)

    def _scan_obj(obj) -> None:
        if obj is not None:
            tirx.stmt_functor.post_order_visit(obj, _visit)

    def _scan_buffer(buf) -> None:
        for expr in (
            *getattr(buf, "shape", ()),
            *(getattr(buf, "strides", ()) or ()),
            getattr(buf, "elem_offset", None),
        ):
            _scan_obj(expr)

    def _visit(node) -> None:
        if let_classes and isinstance(node, let_classes):
            var_obj = getattr(node, "var", None)
            val_obj = getattr(node, "value", None)
            if var_obj is not None and val_obj is not None:
                let_env[var_obj] = val_obj
        if isinstance(node, div_classes):
            _add_denom(getattr(node, "b", None))
        if alloc_classes and isinstance(node, alloc_classes):
            buf = getattr(node, "buffer", None)
            if buf is not None:
                _scan_buffer(buf)
        if sblock_cls is not None and isinstance(node, sblock_cls):
            for buf in getattr(node, "alloc_buffers", ()) or ():
                _scan_buffer(buf)

    _scan_obj(getattr(entry_fn, "body", None))
    for buf in (getattr(entry_fn, "buffer_map", None) or {}).values():
        _scan_buffer(buf)

    if not let_env:
        return raw_denoms
    expanded: list[object] = []
    for denom in raw_denoms:
        sub = _substitute_let_env(denom, let_env, tirx=tirx)
        if not any(
            (hasattr(sub, "same_as") and sub.same_as(prev)) or sub is prev
            for prev in expanded
        ):
            expanded.append(sub)
    return expanded


def _max_nonzero_const_magnitude(expr, *, tirx) -> int:
    """Return the largest nonzero integer constant magnitude inside *expr*, else 1."""
    max_val = 1
    int_imm_cls = getattr(tirx, "IntImm", None)
    if int_imm_cls is None or expr is None:
        return max_val

    def _visit(node) -> None:
        nonlocal max_val
        if isinstance(node, int_imm_cls):
            val = _static_tirx_int(node)
            if val is not None and val != 0:
                max_val = max(max_val, abs(int(val)))

    tirx.stmt_functor.post_order_visit(expr, _visit)
    return max_val


_MAX_SAFE_SPEC_INT = 1 << 22


def _ensure_positive_denominators(
    denominators: Sequence[object],
    spec_var_map: dict[object, object],
    recorded_vars: list[object],
    allowed_vars: Sequence[object],
    *,
    is_handle_or_data: Callable[[object], bool],
    tirx,
    tvm_mod,
    max_rounds: int = 24,
) -> None:
    """Iteratively adjust free integer vars so every static denominator simplifies >= 1."""
    if not denominators:
        return

    def _same_var(lhs, rhs) -> bool:
        return (hasattr(lhs, "same_as") and lhs.same_as(rhs)) or lhs is rhs

    def _find_key(var_obj):
        for key in spec_var_map:
            if _same_var(var_obj, key):
                return key
        return None

    def _is_allowed_int_var(var_obj) -> bool:
        if not isinstance(var_obj, tirx.Var) or is_handle_or_data(var_obj):
            return False
        dtype_str = str(getattr(var_obj, "dtype", "int32") or "int32").lower()
        if dtype_str == "handle" or "int" not in dtype_str:
            return False
        return any(_same_var(var_obj, allowed) for allowed in allowed_vars)

    for denom in denominators:
        for fv in _free_tirx_vars(denom, tirx=tirx):
            if not _is_allowed_int_var(fv):
                continue
            if _find_key(fv) is None:
                dtype_str = str(getattr(fv, "dtype", "int32") or "int32")
                recorded_vars.append(fv)
                spec_var_map[fv] = tirx.IntImm(dtype_str, 1)

    analyzer = tvm_mod.arith.Analyzer()

    def _safe_simplify_int(expr) -> tuple[int | None, bool]:
        try:
            return _static_tirx_int(analyzer.simplify(expr)), False
        except Exception:
            return None, True

    def _set_var_val(key, new_val: int) -> bool:
        cur_v = _static_tirx_int(spec_var_map[key])
        if cur_v is not None and int(cur_v) == int(new_val):
            return False
        dtype_str = str(getattr(key, "dtype", "int32") or "int32")
        spec_var_map[key] = tirx.IntImm(dtype_str, int(new_val))
        return True

    for _ in range(max_rounds):
        changed = False
        for denom in denominators:
            sub_denom = (
                tirx.stmt_functor.substitute(denom, spec_var_map) if spec_var_map else denom
            )
            simp_val, simp_err = _safe_simplify_int(sub_denom)
            if not simp_err and (simp_val is None or simp_val > 0):
                continue
            candidates = [
                fv for fv in _free_tirx_vars(denom, tirx=tirx) if _is_allowed_int_var(fv)
            ]
            if not candidates:
                continue
            max_const = _max_nonzero_const_magnitude(denom, tirx=tirx)

            def _next_val(cur_v: int, floor_const: int = max_const) -> int:
                raw = max(floor_const, 1) if cur_v <= 0 else max(cur_v * 2, floor_const, 1)
                return min(_MAX_SAFE_SPEC_INT, raw)

            nonpos_keys = []
            for fv in candidates:
                key = _find_key(fv)
                if key is not None:
                    cur_v = _static_tirx_int(spec_var_map[key])
                    if cur_v is not None and cur_v <= 0:
                        nonpos_keys.append((key, cur_v))
            if nonpos_keys:
                for key, cur_v in nonpos_keys:
                    if _set_var_val(key, _next_val(cur_v)):
                        changed = True
                continue

            if len(candidates) == 1:
                key = _find_key(candidates[0])
                if key is not None:
                    cur_v = _static_tirx_int(spec_var_map[key])
                    cur_int = 0 if cur_v is None else int(cur_v)
                    if _set_var_val(key, _next_val(cur_int)):
                        changed = True
                continue

            cur_ints: dict[int, int] = {}
            cand_keys: list[tuple[object, int]] = []
            for fv in candidates:
                key = _find_key(fv)
                if key is None:
                    continue
                cur_v = _static_tirx_int(spec_var_map[key])
                cur_int = 0 if cur_v is None else int(cur_v)
                cur_ints[id(key)] = cur_int
                cand_keys.append((key, cur_int))

            best_key = None
            best_next = 0
            best_trial = None if (simp_err or simp_val is None) else int(simp_val)
            for key, cur_int in cand_keys:
                other_prod = 1
                for o_key, o_int in cand_keys:
                    if o_key is not key:
                        other_prod = min(_MAX_SAFE_SPEC_INT, other_prod * max(1, o_int))
                double_val = _next_val(cur_int)
                boost_val = min(
                    _MAX_SAFE_SPEC_INT,
                    max(
                        double_val,
                        cur_int * max(max_const, 2) * 2,
                        other_prod * max(max_const, 1),
                    ),
                )
                probe_vals = [double_val] if boost_val == double_val else [boost_val, double_val]
                dtype_str = str(getattr(key, "dtype", "int32") or "int32")
                for cand_next in probe_vals:
                    if cand_next == cur_int:
                        continue
                    trial_map = dict(spec_var_map)
                    trial_map[key] = tirx.IntImm(dtype_str, cand_next)
                    trial_sub = tirx.stmt_functor.substitute(denom, trial_map)
                    trial_val, trial_err = _safe_simplify_int(trial_sub)
                    if trial_err or trial_val is None:
                        continue
                    if trial_val > 0 and (best_trial is None or trial_val > best_trial or best_trial <= 0):
                        best_trial = trial_val
                        best_key = key
                        best_next = cand_next
                        break
                    if best_trial is None or trial_val > best_trial:
                        best_trial = trial_val
                        best_key = key
                        best_next = cand_next

            if best_key is not None and (best_trial is not None and best_trial > 0):
                if _set_var_val(best_key, best_next):
                    changed = True
                continue

            for key, cur_int in cand_keys:
                if cur_int <= 1:
                    continue
                dtype_str = str(getattr(key, "dtype", "int32") or "int32")
                trial_map = dict(spec_var_map)
                trial_map[key] = tirx.IntImm(dtype_str, 1)
                trial_sub = tirx.stmt_functor.substitute(denom, trial_map)
                trial_val, trial_err = _safe_simplify_int(trial_sub)
                if not trial_err and trial_val is not None and trial_val > 0:
                    best_key = key
                    best_next = 1
                    best_trial = trial_val
                    break

            if best_key is not None:
                if _set_var_val(best_key, best_next):
                    changed = True
            else:
                growable = [(key, cur_int) for key, cur_int in cand_keys if cur_int < _MAX_SAFE_SPEC_INT]
                if growable:
                    min_key, min_cur = min(growable, key=lambda item: item[1])
                    if _set_var_val(min_key, _next_val(min_cur)):
                        changed = True
        if not changed:
            break


def _collect_divisor_vars(stmt, tirx=None) -> list[object]:
    """Collect free TIR variables appearing in division or modulo denominators."""
    if tirx is None:
        from tvm import tirx as _tirx  # noqa: PLC0415

        tirx = _tirx
    div_classes = tuple(
        cls
        for name in ("FloorDiv", "FloorMod", "Div", "Mod")
        if (cls := getattr(tirx, name, None)) is not None
    )
    if not div_classes or stmt is None:
        return []
    found: list[object] = []

    def _add_var(var_obj) -> None:
        if not isinstance(var_obj, tirx.Var):
            return
        if not any(
            (hasattr(var_obj, "same_as") and var_obj.same_as(prev)) or var_obj is prev
            for prev in found
        ):
            found.append(var_obj)

    def _visit(node) -> None:
        if isinstance(node, div_classes):
            denom = getattr(node, "b", None)
            if denom is not None:
                for fv in _free_tirx_vars(denom, tirx=tirx):
                    _add_var(fv)

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return found


def _specialize_primfunc_for_anchors(
    entry_fn,
    mod_in,
    var_bindings: Mapping[str, int] | None,
    *,
    tvm_mod,
    tirx,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
):
    """Specialize an isolated PrimFunc copy using concrete anchor var_bindings."""
    if not var_bindings and not ordered_tensor_shapes:
        return entry_fn, mod_in

    eff_bindings, var_obj_bindings = _augment_var_bindings_from_buffers(
        entry_fn,
        var_bindings,
        tirx=tirx,
        ordered_tensor_shapes=ordered_tensor_shapes,
    )
    if not eff_bindings and not var_obj_bindings:
        return entry_fn, mod_in
    lower_eff = {k.lower(): v for k, v in eff_bindings.items()}

    buffer_map = getattr(entry_fn, "buffer_map", None) or {}
    handle_vars = list(buffer_map.keys())
    data_vars = [buf.data for buf in buffer_map.values() if getattr(buf, "data", None) is not None]

    def _is_handle_or_data(var_obj) -> bool:
        return any(
            (hasattr(var_obj, "same_as") and var_obj.same_as(h)) or var_obj is h
            for h in (*handle_vars, *data_vars)
        )

    spec_var_map: dict[object, object] = {}
    recorded_vars: list[object] = []

    def _maybe_bind(var_obj) -> None:
        if not isinstance(var_obj, tirx.Var) or _is_handle_or_data(var_obj):
            return
        if any(
            (hasattr(var_obj, "same_as") and var_obj.same_as(prev)) or var_obj is prev
            for prev in recorded_vars
        ):
            return
        bound: int | None = None
        for v_obj, v_val in var_obj_bindings:
            if (hasattr(var_obj, "same_as") and var_obj.same_as(v_obj)) or var_obj is v_obj:
                bound = int(v_val)
                break
        if bound is None:
            bound = _match_binding_for_var(var_obj, eff_bindings, lower_eff)
        if bound is None:
            return
        dtype_str = str(getattr(var_obj, "dtype", "int32") or "int32")
        if dtype_str == "handle":
            return
        recorded_vars.append(var_obj)
        spec_var_map[var_obj] = tirx.IntImm(dtype_str, int(bound))

    for param in getattr(entry_fn, "params", ()):
        _maybe_bind(param)
    buffer_free_vars: list[object] = []
    for buf in buffer_map.values():
        for expr in (*getattr(buf, "shape", ()), *(getattr(buf, "strides", ()) or ()), buf.elem_offset):
            for fv in _free_tirx_vars(expr, tirx=tirx):
                _maybe_bind(fv)
                if not any(
                    (hasattr(fv, "same_as") and fv.same_as(prev)) or fv is prev
                    for prev in buffer_free_vars
                ):
                    buffer_free_vars.append(fv)
    body_free_vars = _free_tirx_vars(entry_fn.body, tirx=tirx)
    for fv in body_free_vars:
        _maybe_bind(fv)

    max_known_numel = max(
        (
            int(lower_eff[k])
            for k in ("total_elements", "total_elems", "n_elems", "num_elements", "numel")
            if k in lower_eff and int(lower_eff[k]) > 0
        ),
        default=0,
    )
    for buf in buffer_map.values():
        axis_ints: list[int] = []
        ok = True
        for dim_expr in getattr(buf, "shape", ()) or ():
            sv = _static_tirx_int(dim_expr)
            if sv is not None and sv > 0:
                axis_ints.append(int(sv))
            elif dim_expr in spec_var_map:
                bound_sv = _static_tirx_int(spec_var_map[dim_expr])
                if bound_sv is not None and bound_sv > 0:
                    axis_ints.append(int(bound_sv))
                else:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok and axis_ints:
            max_known_numel = max(max_known_numel, math.prod(axis_ints))
    fallback_numel = max_known_numel if max_known_numel > 0 else 1024

    def _is_body_free_int_var(var_obj) -> bool:
        if not isinstance(var_obj, tirx.Var) or _is_handle_or_data(var_obj):
            return False
        dtype_str = str(getattr(var_obj, "dtype", "int32") or "int32").lower()
        if dtype_str == "handle" or "int" not in dtype_str:
            return False
        if any(
            (hasattr(var_obj, "same_as") and var_obj.same_as(prev)) or var_obj is prev
            for prev in recorded_vars
        ):
            return False
        return any(
            (hasattr(var_obj, "same_as") and var_obj.same_as(bfv)) or var_obj is bfv
            for bfv in body_free_vars
        )

    remaining_free_vars: list[object] = []

    def _collect_expr_free_vars(expr) -> None:
        if expr is None:
            return
        for fv in _free_tirx_vars(expr, tirx=tirx):
            if _is_body_free_int_var(fv) and not any(
                (hasattr(fv, "same_as") and fv.same_as(prev)) or fv is prev
                for prev in remaining_free_vars
            ):
                remaining_free_vars.append(fv)

    def _visit_remaining(node) -> None:
        if isinstance(node, tirx.For):
            _collect_expr_free_vars(node.extent)
        elif isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            for idx_expr in getattr(node, "indices", ()) or ():
                _collect_expr_free_vars(idx_expr)
        elif isinstance(node, tirx.Call):
            op_name = str(getattr(node.op, "name", node.op)).lower()
            if "copy" in op_name or "region" in op_name or "access_ptr" in op_name:
                for arg in getattr(node, "args", ()) or ():
                    _collect_expr_free_vars(arg)

    tirx.stmt_functor.post_order_visit(entry_fn.body, _visit_remaining)
    for fv in remaining_free_vars:
        dtype_str = str(getattr(fv, "dtype", "int32") or "int32")
        recorded_vars.append(fv)
        spec_var_map[fv] = tirx.IntImm(dtype_str, int(fallback_numel))

    params_list = list(getattr(entry_fn, "params", ()))
    denominators = _collect_div_denominators(entry_fn, tirx=tirx)
    _ensure_positive_denominators(
        denominators,
        spec_var_map,
        recorded_vars,
        [*params_list, *buffer_free_vars, *body_free_vars],
        is_handle_or_data=_is_handle_or_data,
        tirx=tirx,
        tvm_mod=tvm_mod,
    )

    if not spec_var_map:
        return entry_fn, mod_in

    extra_params = [
        v
        for v in recorded_vars
        if not any((hasattr(v, "same_as") and v.same_as(p)) or v is p for p in params_list)
    ]
    tmp_fn = (
        tirx.PrimFunc(
            [*params_list, *extra_params],
            entry_fn.body,
            ret_type=entry_fn.ret_type,
            buffer_map=entry_fn.buffer_map,
            attrs=entry_fn.attrs,
        )
        if extra_params
        else entry_fn
    )
    spec_fn = tmp_fn.specialize(spec_var_map)

    target_gvar = None
    for gvar, fn in mod_in.functions.items():
        if (hasattr(fn, "same_as") and fn.same_as(entry_fn)) or fn is entry_fn:
            target_gvar = gvar
            break
    if target_gvar is None:
        target_gvar = (
            str(entry_fn.attrs.get("global_symbol", "main"))
            if entry_fn.attrs
            else "main"
        )
    updated_funcs = {
        k: (spec_fn if k == target_gvar else v) for k, v in mod_in.functions.items()
    }
    if target_gvar not in updated_funcs:
        updated_funcs[target_gvar] = spec_fn
    spec_mod = tvm_mod.IRModule(updated_funcs, attrs=getattr(mod_in, "attrs", None))
    if hasattr(tirx, "transform") and hasattr(tirx.transform, "Simplify"):
        spec_mod = tirx.transform.Simplify()(spec_mod)
        if target_gvar in spec_mod.functions:
            spec_fn = spec_mod.functions[target_gvar]
        else:
            for fn in spec_mod.functions.values():
                if isinstance(fn, tirx.PrimFunc):
                    spec_fn = fn
                    break
    return spec_fn, spec_mod


def _pass_instrument_if_available(cls: type) -> type:
    """Attach tvm.ir.instrument.pass_instrument when TileLang runtime is available."""
    try:
        from tilelang import tvm as tvm_mod  # noqa: PLC0415
    except ImportError:
        return cls
    instr = getattr(getattr(tvm_mod, "ir", None), "instrument", None)
    decorator = getattr(instr, "pass_instrument", None)
    return decorator(cls) if callable(decorator) else cls


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
    ctx_target: object,
    *,
    pass_configs: Mapping[str, object] | None = None,
    instruments: Sequence[object] = (),
):
    from tilelang import tvm  # noqa: PLC0415

    cur_target = tvm.target.Target.current(allow_none=True)
    cur_kind = str(getattr(getattr(cur_target, "kind", None), "name", "") or "")
    target_ctx = (
        ctx_target
        if (cur_target is None or cur_kind != "ascend")
        else contextlib.nullcontext()
    )
    with target_ctx, tvm.transform.PassContext(
        config=dict(pass_configs or {}),
        instruments=list(instruments),
    ):
        yield


def _access_ptr_or_region_bytes(
    arg,
    *,
    tirx,
    DataTypeCls,
    let_env: Mapping[object, object] | None = None,
) -> int:
    if isinstance(arg, tirx.Call):
        op_name = str(getattr(arg.op, "name", arg.op)).lower()
        if "access_ptr" in op_name and len(arg.args) >= 4:
            dtype_obj = getattr(arg.args[0], "dtype", None)
            ext = _static_tirx_int(arg.args[3])
            if dtype_obj is not None and ext is not None and ext > 0:
                elem_b = _dtype_bytes(dtype_obj, DataTypeCls)
                return int(math.ceil(ext * elem_b))
    _scope, total_b, _inner_b = _region_bytes(arg, tirx, DataTypeCls, let_env=let_env)
    return total_b


def _collect_post_lower_gm_dma_payloads(
    stmt,
    *,
    tirx,
    DataTypeCls,
) -> tuple[list[int], list[int], bool]:
    """Collect lowered GM DMA byte payloads inside loops, function-wide, and loop presence."""
    loop_sizes: list[int] = []
    all_sizes: list[int] = []
    seen_in_loop_gm = [False]
    let_stmt_cls = getattr(tirx, "LetStmt", None)
    bind_cls = getattr(tirx, "Bind", None)

    def _walk(node, *, in_loop: bool, let_env: dict[object, object]) -> None:
        if node is None:
            return
        if isinstance(node, tirx.SeqStmt):
            seq_env = dict(let_env)
            for child in node.seq:
                if bind_cls is not None and isinstance(child, bind_cls):
                    seq_env[child.var] = _substitute_let_env(child.value, seq_env, tirx=tirx)
                    if hasattr(child, "body"):
                        _walk(child.body, in_loop=in_loop, let_env=dict(seq_env))
                else:
                    _walk(child, in_loop=in_loop, let_env=seq_env)
            return
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            _walk(node.body, in_loop=in_loop, let_env=next_env)
            return
        if bind_cls is not None and isinstance(node, bind_cls):
            next_env = dict(let_env)
            next_env[node.var] = _substitute_let_env(node.value, let_env, tirx=tirx)
            if hasattr(node, "body"):
                _walk(node.body, in_loop=in_loop, let_env=next_env)
            return
        if isinstance(node, tirx.For):
            tb = getattr(node, "thread_binding", None)
            trip = _static_tirx_int(node.extent) if tb is None else None
            sub_loop = in_loop or (tb is None and (trip is None or trip > 1))
            _walk(node.body, in_loop=sub_loop, let_env=let_env)
            return
        if isinstance(node, tirx.While):
            _walk(node.body, in_loop=True, let_env=let_env)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, in_loop=in_loop, let_env=let_env)
            if node.else_case is not None:
                _walk(node.else_case, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, (tirx.SBlock, tirx.AttrStmt, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, in_loop=in_loop, let_env=let_env)
            return
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name = _effective_tirx_call_name(node.value).lower()
            args = _effective_tirx_call_args(node.value)
            if "copy" in op_name and len(args) >= 2:
                is_gm_name = (
                    "gm_to_" in op_name
                    or "_to_gm" in op_name
                    or "_gm_" in op_name
                    or "gm2" in op_name
                    or "2gm" in op_name
                )
                src_s, _sb, _sin = _region_bytes(args[0], tirx, DataTypeCls, let_env=let_env)
                dst_s, _db, _din = _region_bytes(args[1], tirx, DataTypeCls, let_env=let_env)
                is_gm_region = (
                    src_s is not None
                    and dst_s is not None
                    and src_s != dst_s
                    and "GM" in (src_s, dst_s)
                )
                if is_gm_name or is_gm_region:
                    if in_loop:
                        seen_in_loop_gm[0] = True
                    b0 = _access_ptr_or_region_bytes(
                        args[0], tirx=tirx, DataTypeCls=DataTypeCls, let_env=let_env
                    )
                    b1 = _access_ptr_or_region_bytes(
                        args[1], tirx=tirx, DataTypeCls=DataTypeCls, let_env=let_env
                    )
                    xfer = max(b0, b1)
                    if xfer > 0:
                        all_sizes.append(xfer)
                        if in_loop:
                            loop_sizes.append(xfer)

    _walk(stmt, in_loop=False, let_env={})
    return loop_sizes, all_sizes, seen_in_loop_gm[0]


def _collect_func_ub_buf_names(func, *, tirx) -> set[str]:
    names: set[str] = set()

    def _check_buf(buf) -> None:
        raw_scope = str(buf.scope()).strip()
        if _map_tilelang_scope(raw_scope) == "UB" or raw_scope == "local.ub":
            names.add(buf.name)

    for param in getattr(func, "params", ()):
        buf = func.buffer_map.get(param) if hasattr(func, "buffer_map") else None
        if buf is not None:
            _check_buf(buf)

    def _visit(node) -> None:
        if isinstance(node, tirx.SBlock):
            for buf in node.alloc_buffers:
                _check_buf(buf)
        elif isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            _check_buf(node.buffer)

    tirx.stmt_functor.post_order_visit(func.body, _visit)
    return names


def _detect_post_lower_dynamic_offset(func, *, tirx) -> bool:
    mod_classes = tuple(
        cls
        for name in ("FloorMod", "Mod")
        if (cls := getattr(tirx, name, None)) is not None
    )
    found = [False]

    def _expr_has_loop_mod(expr, loop_vars: list[object]) -> bool:
        if not mod_classes or not loop_vars or expr is None:
            return False
        hit = [False]

        def _visit(sub) -> None:
            if isinstance(sub, mod_classes):
                fvs = _free_tirx_vars(sub.a, tirx=tirx)
                if any(
                    (hasattr(fv, "same_as") and fv.same_as(lv)) or fv is lv
                    for fv in fvs
                    for lv in loop_vars
                ):
                    hit[0] = True

        tirx.stmt_functor.post_order_visit(expr, _visit)
        return hit[0]

    def _walk(node, loop_vars: list[object]) -> None:
        if node is None or found[0]:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child, loop_vars)
            return
        if isinstance(node, tirx.For):
            ann = getattr(node, "annotations", None) or {}
            kind_val = int(node.kind) if hasattr(node.kind, "value") else node.kind
            stages_val = (
                _static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else None
            )
            if ann.get("enable_offset") or kind_val == 5 or (stages_val is not None and stages_val >= 2):
                found[0] = True
                return
            next_vars = [*loop_vars, node.loop_var]
            _walk(node.body, next_vars)
            return
        if isinstance(node, tirx.IfThenElse):
            _walk(node.then_case, loop_vars)
            if node.else_case is not None:
                _walk(node.else_case, loop_vars)
            return
        if isinstance(node, tirx.SBlockRealize):
            _walk(node.block, loop_vars)
            return
        if isinstance(node, (tirx.SBlock, tirx.AttrStmt, tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(node, "body"):
                _walk(node.body, loop_vars)
            return
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            for arg in node.value.args:
                if isinstance(arg, tirx.Call) and "access_ptr" in str(
                    getattr(arg.op, "name", arg.op)
                ).lower():
                    if len(arg.args) >= 3 and _expr_has_loop_mod(arg.args[2], loop_vars):
                        found[0] = True
                        return

    _walk(func.body, [])
    return found[0]


def _reconcile_post_lower_device_mod(
    device_mod,
    *,
    spaces: AscendSpaceFacts,
    tirx,
    DataTypeCls,
    pre_pipe_cycles: dict[str, int],
    pre_barriers: int,
    pre_min_gm_dma_bytes: int,
    pre_dma_burst_eff: float,
    pre_ub_scalar_loops: int,
    pre_dyn_offset_pipe: bool,
    explicit_aiv_split: int | None,
    effective_aiv_split: int,
) -> tuple[dict[str, int], int, int, float, int, bool]:
    """Read-only scan lowered device PrimFuncs to reconcile barriers, DMA, scalar, and AIV pipes."""
    if device_mod is None or not hasattr(device_mod, "functions"):
        return (
            dict(pre_pipe_cycles),
            pre_barriers,
            pre_min_gm_dma_bytes,
            pre_dma_burst_eff,
            pre_ub_scalar_loops,
            pre_dyn_offset_pipe,
        )
    dev_funcs = [fn for fn in device_mod.functions.values() if isinstance(fn, tirx.PrimFunc)]
    if not dev_funcs:
        return (
            dict(pre_pipe_cycles),
            pre_barriers,
            pre_min_gm_dma_bytes,
            pre_dma_burst_eff,
            pre_ub_scalar_loops,
            pre_dyn_offset_pipe,
        )

    post_barriers = 0
    post_loop_dma: list[int] = []
    post_all_dma: list[int] = []
    seen_any_in_loop_gm_dma = False
    post_ub_scalar_loops = 0
    post_dyn_offset = pre_dyn_offset_pipe
    post_pipe_totals: dict[str, int] = {}

    for dev_fn in dev_funcs:
        post_barriers += _count_in_loop_pipe_all_barriers(dev_fn.body, tirx=tirx)
        l_dma, a_dma, seen_loop = _collect_post_lower_gm_dma_payloads(
            dev_fn.body, tirx=tirx, DataTypeCls=DataTypeCls
        )
        seen_any_in_loop_gm_dma = seen_any_in_loop_gm_dma or seen_loop
        post_loop_dma.extend(l_dma)
        post_all_dma.extend(a_dma)
        ub_names = _collect_func_ub_buf_names(dev_fn, tirx=tirx)
        post_ub_scalar_loops += _count_tirx_ub_scalar_compute_loops(
            dev_fn.body,
            ub_names,
            tirx=tirx,
            allow_sync_intrinsics=True,
        )
        if not post_dyn_offset and _detect_post_lower_dynamic_offset(dev_fn, tirx=tirx):
            post_dyn_offset = True
        dev_split = _detect_tilelang_aiv_split_factor(dev_fn, tirx=tirx)
        scan_split = max(effective_aiv_split, dev_split) if explicit_aiv_split is None else explicit_aiv_split
        dev_pipes, _mte2, _pad, _opq, _simt, _ill = _scan_tilelang_ast(
            dev_fn,
            spaces,
            tirx=tirx,
            DataTypeCls=DataTypeCls,
            aiv_split_factor=scan_split,
        )
        for pkey, pval in dev_pipes.items():
            post_pipe_totals[pkey] = post_pipe_totals.get(pkey, 0) + pval

    reconciled_barriers = max(pre_barriers, post_barriers)
    reconciled_ub_scalar = max(pre_ub_scalar_loops, post_ub_scalar_loops)

    raw_pool = post_loop_dma if (post_loop_dma or seen_any_in_loop_gm_dma) else post_all_dma
    post_pool = [b for b in raw_pool if b > 0]
    if post_pool:
        post_min = min(post_pool)
        reconciled_min_dma = (
            min(pre_min_gm_dma_bytes, post_min) if pre_min_gm_dma_bytes > 0 else post_min
        )
        total_dma = float(sum(post_pool))
        weighted_inv = sum(
            float(b) / max(1e-6, float(spaces.dma_burst_efficiency(int(b))))
            for b in post_pool
        )
        post_eff = round(min(1.0, total_dma / max(1e-6, weighted_inv)), 4)
        reconciled_burst_eff = (
            min(pre_dma_burst_eff, post_eff) if pre_min_gm_dma_bytes > 0 else post_eff
        )
    else:
        reconciled_min_dma = pre_min_gm_dma_bytes
        reconciled_burst_eff = pre_dma_burst_eff

    reconciled_pipes = dict(pre_pipe_cycles)
    if explicit_aiv_split != 1:
        pre_a0 = pre_pipe_cycles.get("AIV0.PIPE_V", 0)
        pre_a1 = pre_pipe_cycles.get("AIV1.PIPE_V", 0)
        post_a0 = post_pipe_totals.get("AIV0.PIPE_V", 0)
        post_a1 = post_pipe_totals.get("AIV1.PIPE_V", 0)
        if (post_a0 != post_a1) or (pre_a0 == 0 and pre_a1 == 0 and (post_a0 > 0 or post_a1 > 0)):
            if post_a0 > 0:
                reconciled_pipes["AIV0.PIPE_V"] = max(pre_a0, post_a0)
            if post_a1 > 0:
                reconciled_pipes["AIV1.PIPE_V"] = max(pre_a1, post_a1)

    return (
        reconciled_pipes,
        reconciled_barriers,
        reconciled_min_dma,
        reconciled_burst_eff,
        reconciled_ub_scalar,
        post_dyn_offset,
    )


def specialize_primfunc_for_anchor(
    prim_func,
    *,
    var_bindings: Mapping[str, int] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
):
    """Specialize a TileLang PrimFunc or IRModule for one anchor shape without lowering."""
    from tilelang import tvm  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    if isinstance(prim_func, tvm.IRModule):
        funcs = [fn for fn in prim_func.functions.values() if isinstance(fn, tirx.PrimFunc)]
        if not funcs:
            raise ValueError("IRModule contains no PrimFunc entries")
        entry_fn = funcs[0]
        mod_in = prim_func
    else:
        entry_fn = prim_func
        sym = str(entry_fn.attrs.get("global_symbol", "main")) if entry_fn.attrs else "main"
        mod_in = tvm.IRModule({sym: entry_fn})

    return _specialize_primfunc_for_anchors(
        entry_fn,
        mod_in,
        var_bindings,
        tvm_mod=tvm,
        tirx=tirx,
        ordered_tensor_shapes=ordered_tensor_shapes,
    )


def inspect_tilelang_primfunc(
    prim_func,
    *,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    aiv_split_factor: int | None = None,
    unreserved_ubuf: bool = False,
    var_bindings: Mapping[str, int] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> EnvelopeInspectionResult:
    """Inspect a TileLang PrimFunc using Two-Snapshot IR Diff without bisheng compile."""
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415
    from tilelang.engine.lower import lower_to_host_device_ir  # noqa: PLC0415
    from tilelang.utils.language import DataType as TLDataType  # noqa: PLC0415
    from tvm import tirx  # noqa: PLC0415

    resolved_target = target or AscendTarget("ascend.950pr")
    spaces = resolved_target.get_facts(AscendSpaceFacts)

    if isinstance(prim_func, tvm.IRModule):
        funcs = [fn for fn in prim_func.functions.values() if isinstance(fn, tirx.PrimFunc)]
        if not funcs:
            raise ValueError("IRModule contains no PrimFunc entries")
        entry_fn = funcs[0]
        mod_in = prim_func
    else:
        entry_fn = prim_func
        sym = str(entry_fn.attrs.get("global_symbol", "main")) if entry_fn.attrs else "main"
        mod_in = tvm.IRModule({sym: entry_fn})

    raw_cfg, _raw_flags = _extract_primfunc_pass_configs_and_flags(
        entry_fn,
        mod_in,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    ctx = create_backend_context("ascend", None, "auto")
    tap = ReadOnlyAscendPassTap()
    lowering_error: str | None = None
    device_mod = None

    with _ascend_compile_session(ctx.target, pass_configs=raw_cfg, instruments=[tap]):
        entry_fn, mod_in = _specialize_primfunc_for_anchors(
            entry_fn,
            mod_in,
            var_bindings,
            tvm_mod=tvm,
            tirx=tirx,
            ordered_tensor_shapes=ordered_tensor_shapes,
        )
        cfg, merged_flags = _extract_primfunc_pass_configs_and_flags(
            entry_fn,
            mod_in,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        ubuf_flag = "--cce-disable-vf-stack-reserved-ubuf"
        unreserved_hint = bool(
            unreserved_ubuf
            or any(ubuf_flag in f for f in merged_flags)
            or _attrs_contain_flag(cfg, ubuf_flag)
        )
        effective_aiv_split = (
            aiv_split_factor
            if aiv_split_factor is not None
            else _detect_tilelang_aiv_split_factor(entry_fn, tirx=tirx)
        )
        explicit_snaps = _extract_tilelang_buffers_from_func(
            entry_fn, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=False
        )
        requested_mb = _requested_multibuffer_names(entry_fn, tirx=tirx)
        pipe_cycles, mte2_inflight, pad_bytes, opaque, has_simtvf, illegal_dma = (
            _scan_tilelang_ast(
                entry_fn,
                spaces,
                tirx=tirx,
                DataTypeCls=TLDataType,
                aiv_split_factor=effective_aiv_split,
            )
        )
        telemetry = _collect_tilelang_telemetry(
            entry_fn,
            mod_in,
            tirx=tirx,
            DataTypeCls=TLDataType,
            pipe_cycles=pipe_cycles,
            unreserved_ubuf=unreserved_hint,
            spaces=spaces,
            aiv_split_factor=effective_aiv_split,
            var_bindings=var_bindings,
            tvm_mod=tvm,
            ordered_tensor_shapes=ordered_tensor_shapes,
        )
        effective_unreserved = telemetry.uses_unreserved_ubuf
        capacities = _scope_capacities(
            spaces,
            has_simtvf=has_simtvf,
            unreserved_ubuf=effective_unreserved,
        )
        try:
            with tvm.transform.PassContext(config=cfg, instruments=[tap]):
                _host_mod, device_mod, _params, _tgt, _htgt = lower_to_host_device_ir(
                    mod_in, ctx
                )
        except Exception as exc:
            lowering_error = f"{type(exc).__name__}: {exc}"

    captured_mods = tap.captured_mods
    observed = tap.observed_passes

    dev_snaps = _extract_tilelang_buffers_from_mod(
        device_mod,
        tirx=tirx,
        DataTypeCls=TLDataType,
        prefer_allocate=True,
    )
    mat_snaps = _extract_tilelang_buffers_from_mod(
        captured_mods.get("tl.MaterializeMultiBuffer"),
        tirx=tirx,
        DataTypeCls=TLDataType,
        prefer_allocate=False,
    )
    if not mat_snaps and dev_snaps:
        mat_snaps = dev_snaps

    pre_merge_mod = (
        captured_mods.get("pre_merge")
        or captured_mods.get("tl.LowerTileOp")
        or captured_mods.get("tl.AscendLowerTileOp")
        or captured_mods.get("tl.InsertNd2Nz")
        or captured_mods.get("tl.NormalizeAscendFractalStorage")
        or captured_mods.get("tl.MaterializeMultiBuffer")
        or device_mod
    )
    lowered_snaps = _extract_tilelang_buffers_from_mod(
        pre_merge_mod,
        tirx=tirx,
        DataTypeCls=TLDataType,
        prefer_allocate=True,
    )
    post_merge_mod = captured_mods.get("tl.MergeUBAllocations") or device_mod
    post_merge_snaps = _extract_tilelang_buffers_from_mod(
        post_merge_mod,
        tirx=tirx,
        DataTypeCls=TLDataType,
        prefer_allocate=True,
    )
    merged_peak: dict[str, int] = {}
    for snap in post_merge_snaps:
        merged_peak[snap.scope] = merged_peak.get(snap.scope, 0) + snap.bytes
    dev_scope_totals: dict[str, int] = {}
    for snap in dev_snaps:
        dev_scope_totals[snap.scope] = dev_scope_totals.get(snap.scope, 0) + snap.bytes
    for scope, dev_bytes in dev_scope_totals.items():
        merged_peak[scope] = max(merged_peak.get(scope, 0), dev_bytes)

    degraded = _detect_degraded_buffers(explicit_snaps, mat_snaps, requested_mb)
    waterlines = _diff_snapshots_by_scope(
        explicit_snaps,
        lowered_snaps,
        merged_peak,
        capacities,
        degraded,
    )
    if lowering_error is not None and "lowering_rejected" not in opaque:
        opaque = (*opaque, "lowering_rejected")

    (
        pipe_cycles,
        reconciled_barriers,
        reconciled_min_dma,
        reconciled_burst_eff,
        reconciled_ub_scalar,
        reconciled_dyn_offset,
    ) = _reconcile_post_lower_device_mod(
        device_mod,
        spaces=spaces,
        tirx=tirx,
        DataTypeCls=TLDataType,
        pre_pipe_cycles=pipe_cycles,
        pre_barriers=telemetry.in_loop_pipe_all_barriers,
        pre_min_gm_dma_bytes=telemetry.min_gm_dma_bytes,
        pre_dma_burst_eff=telemetry.dma_burst_efficiency_ratio,
        pre_ub_scalar_loops=telemetry.ub_scalar_compute_loops,
        pre_dyn_offset_pipe=telemetry.uses_dynamic_offset_pipelining,
        explicit_aiv_split=aiv_split_factor,
        effective_aiv_split=effective_aiv_split,
    )

    has_overflow = any(wl.overflow_bytes > 0 for wl in waterlines.values())
    solved_ii: int | None = None
    if not illegal_dma and not has_overflow and "lowering_rejected" not in opaque:
        solved_ii = _try_solve_tilelang_pipelined_ii(
            entry_fn,
            spaces,
            tirx=tirx,
            DataTypeCls=TLDataType,
            aiv_split_factor=effective_aiv_split,
            has_simtvf=has_simtvf,
            unreserved_ubuf=effective_unreserved,
            degraded_names=degraded,
        )
    analysis_mode = (
        "exact_modulo_z3" if (solved_ii is not None and not opaque) else "readonly_envelope"
    )
    wave_occ = (
        1.0
        if telemetry.grid_is_symbolic
        else _compute_wave_occupancy(
            telemetry.grid_blocks, spaces, pipe_cycles, effective_aiv_split
        )
    )
    l1_wl = waterlines.get("L1")
    effective_l1_risk = telemetry.l1_bank_conflict_risk
    if l1_wl is not None and l1_wl.peak_allocated_bytes > 0:
        l1_full = max(1, spaces.l1_bytes)
        l1_half = max(1, l1_full // 2)
        if l1_wl.peak_allocated_bytes > l1_half and effective_l1_risk > 0.0:
            effective_l1_risk = max(
                effective_l1_risk,
                round(min(1.0, l1_wl.peak_allocated_bytes / float(l1_full)), 4),
            )
        elif len(l1_wl.explicit_buffers) >= 2 and effective_l1_risk > 0.0:
            effective_l1_risk = max(
                effective_l1_risk,
                round(min(0.5, l1_wl.peak_allocated_bytes / float(l1_half)), 4),
            )

    ma_prof = extract_microarch_profile_from_tirx(entry_fn, var_bindings=var_bindings)
    ledger = build_objective_physical_ledger(
        analysis_mode=analysis_mode,
        is_legal=(not illegal_dma) and (lowering_error is None),
        steady_state_ii=solved_ii,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        mte2_inflight_per_aiv=mte2_inflight,
        unaligned_dma_padding_bytes=pad_bytes,
        opaque_dynamic_regions=opaque,
        grid_blocks=telemetry.grid_blocks,
        wave_occupancy=wave_occ,
        gm_roundtrip_bytes=telemetry.gm_roundtrip_bytes,
        dominant_contig_bytes=telemetry.dominant_contig_bytes,
        effective_cube_m=telemetry.effective_cube_m,
        has_indirect_gather=telemetry.has_indirect_gather,
        has_lowbit_quant=telemetry.has_lowbit_quant,
        has_online_rescale_loop=telemetry.has_online_rescale_loop,
        in_loop_ub_reload_stores=telemetry.in_loop_ub_reload_stores,
        uses_unreserved_ubuf=effective_unreserved,
        grid_is_symbolic=telemetry.grid_is_symbolic,
        min_gm_dma_bytes=reconciled_min_dma,
        dma_burst_efficiency_ratio=reconciled_burst_eff,
        uses_dynamic_offset_pipelining=reconciled_dyn_offset,
        has_unlimit_memory_attr=telemetry.has_unlimit_memory_attr,
        has_coupled_reduce_sfu_accum=telemetry.has_coupled_reduce_sfu_accum,
        active_grid_blocks=telemetry.active_grid_blocks,
        in_loop_pipe_all_barriers=reconciled_barriers,
        ub_bank_conflict_risk=telemetry.ub_bank_conflict_risk,
        l1_bank_conflict_risk=effective_l1_risk,
        ub_scalar_compute_loops=reconciled_ub_scalar,
        microarch_profile=ma_prof.to_dict(),
    )
    return EnvelopeInspectionResult(
        ledger=ledger,
        explicit_snapshot=explicit_snaps,
        lowered_snapshot=lowered_snaps,
        merged_peak_bytes_by_scope=merged_peak,
        lowered_passes_observed=tuple(observed),
        lowering_error=lowering_error,
    )


__all__ = [
    "BufferSnapshot",
    "EnvelopeInspectionResult",
    "ReadOnlyAscendPassTap",
    "inspect_tilelang_primfunc",
    "specialize_primfunc_for_anchor",
]
