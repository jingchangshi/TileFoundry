"""TIR memory-arg unpacking, pipe cycle estimators, and Z3 pipelined-loop II solver."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tilefoundry.vendor.tilelang_z3 import Buffer, Capacity, Edge, Reuse, Task, tf_adapter
from tilefoundry_ascend.ascend.schedule.anchor_specializer import (
    any_tirx_node, static_shape_ints, static_tirx_int, substitute_let_env,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    _assign_dag_stages, _capacities_from_facts, _DTypeLike, _resources_for_tasks,
    dtype_bytes, solve_pipeline_with_monotone_backoff,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, NodeResourceMask

_TILELANG_SCOPE_MAP: dict[str, str] = {
    "shared.dyn": "UB", "shared": "UB", "shared.l1": "L1", "shared.l1.dyn": "L1",
    "shared.l0a": "L0A", "shared.l0a.dyn": "L0A", "shared.l0b": "L0B", "shared.l0b.dyn": "L0B",
    "shared.l0c": "L0C", "shared.l0c.dyn": "L0C", "shared.ss": "SSBUF", "global": "GM",
}
_EXTERN_CALL_TAILS = frozenset(("call_extern", "call_pure_extern", "call_intrin", "call_llvm_intrin"))


def map_tilelang_scope(raw_scope: str) -> str | None:
    """Map a TileLang TIR buffer scope string onto canonical Ascend memory spaces."""
    return _TILELANG_SCOPE_MAP.get(raw_scope.strip())


def op_name_str(op: object) -> str:
    """Return string name for a TIR op object."""
    return str(getattr(op, "name", op))


def unwrapped_extern_symbol(call: object) -> str | None:
    """Return target intrinsic symbol if *call* wraps an extern intrinsic call."""
    if op_name_str(getattr(call, "op", "")).rsplit(".", 1)[-1].lower() not in _EXTERN_CALL_TAILS:
        return None
    if not (args := getattr(call, "args", ())):
        return None
    first = args[0]
    if isinstance(first, str) and first:
        return first
    val = getattr(first, "value", None)
    return val if isinstance(val, str) and val else None


def effective_tirx_call_name(call: object) -> str:
    """Return effective intrinsic name for a TIR Call node."""
    sym = unwrapped_extern_symbol(call)
    return sym if sym is not None else op_name_str(getattr(call, "op", ""))


def effective_tirx_call_args(call: object) -> list[object]:
    """Return effective argument list for a TIR Call node."""
    args = list(getattr(call, "args", ()))
    return args[1:] if (unwrapped_extern_symbol(call) is not None and args) else args


def buffer_bytes(shape: tuple[int, ...] | None, dtype_obj: object, data_type_cls: Callable[[str], _DTypeLike]) -> int:
    """Compute ceiling byte size of a concrete buffer shape."""
    if not shape:
        return 0
    return int(math.ceil(math.prod(shape) * dtype_bytes(dtype_obj, data_type_cls)))


def _unwrap_mem_arg(arg: object, tirx: object) -> tuple[object | None, list[object] | None, bool]:
    if isinstance(arg, tirx.Call):
        op_name = op_name_str(arg.op)
        if ("region" in op_name or "access_ptr" in op_name) and arg.args and isinstance(arg.args[0], tirx.BufferLoad):
            exts = list(arg.args[2:]) if ("region" in op_name and len(arg.args) >= 2) else None
            return arg.args[0].buffer, exts, False
    elif isinstance(arg, tirx.BufferRegion):
        return arg.buffer, [r.extent for r in arg.region], False
    elif isinstance(arg, tirx.BufferLoad):
        return arg.buffer, None, True
    return None, None, False


def upper_bound_tirx_int(
    expr: object, *, tirx: object, let_env: Mapping[object, object] | None = None, _depth: int = 0
) -> int | None:
    """Compute a conservative static integer upper bound for *expr*."""
    if expr is None or _depth > 12:
        return None
    cur = substitute_let_env(expr, let_env, tirx=tirx) if let_env else expr
    if (sv := static_tirx_int(cur)) is not None:
        return sv
    if isinstance(cur, (tirx.Min, tirx.Mul, tirx.Add)):
        lhs = upper_bound_tirx_int(cur.a, tirx=tirx, _depth=_depth + 1)
        rhs = upper_bound_tirx_int(cur.b, tirx=tirx, _depth=_depth + 1)
        if isinstance(cur, tirx.Min):
            if lhs and lhs > 0 and rhs and rhs > 0:
                return min(lhs, rhs)
            return lhs if (lhs and lhs > 0) else (rhs if (rhs and rhs > 0) else None)
        if isinstance(cur, tirx.Mul):
            return lhs * rhs if (lhs and lhs > 0 and rhs and rhs > 0) else None
        return lhs + rhs if (lhs is not None and lhs >= 0 and rhs is not None and rhs >= 0) else None
    return None


def region_bytes(
    arg: object, tirx: object, data_type_cls: Callable[[str], _DTypeLike],
    *, let_env: Mapping[object, object] | None = None,
) -> tuple[str | None, int, int]:
    """Return (scope, total_bytes, inner_axis_bytes) for a TIR region or load arg."""
    buf, raw_extents, is_scalar_load = _unwrap_mem_arg(arg, tirx)
    if buf is None:
        return None, 0, 0
    scope = map_tilelang_scope(buf.scope())
    if is_scalar_load:
        eb = int(math.ceil(dtype_bytes(buf.dtype, data_type_cls)))
        return scope, eb, eb
    if not raw_extents:
        return scope, 0, 0
    elem_bytes = dtype_bytes(buf.dtype, data_type_cls)
    resolved = [upper_bound_tirx_int(e, tirx=tirx, let_env=let_env) for e in raw_extents]
    last = resolved[-1]
    inner = int(math.ceil(last * elem_bytes)) if (last is not None and last > 0) else 0
    if any(v is None or v <= 0 for v in resolved):
        return scope, 0, inner
    return scope, int(math.ceil(math.prod(int(v) for v in resolved if v is not None) * elem_bytes)), inner


def access_ptr_or_region_bytes(
    arg: object, *, tirx: object, data_type_cls: Callable[[str], _DTypeLike],
    let_env: Mapping[object, object] | None = None,
) -> int:
    """Return transfer byte size for an access_ptr Call or BufferRegion."""
    if isinstance(arg, tirx.Call) and "access_ptr" in op_name_str(arg.op).lower() and len(arg.args) >= 4:
        dtype_obj, ext = getattr(arg.args[0], "dtype", None), static_tirx_int(arg.args[3])
        if dtype_obj is not None and ext is not None and ext > 0:
            return int(math.ceil(ext * dtype_bytes(dtype_obj, data_type_cls)))
    return region_bytes(arg, tirx, data_type_cls, let_env=let_env)[1]


def contains_buffer_load(expr: object, tirx: object) -> bool:
    """Return True if *expr* contains any BufferLoad."""
    return any_tirx_node(expr, tirx, lambda n: isinstance(n, tirx.BufferLoad))


def _is_sid_expr(expr: object, sid_vars: set[object], tirx: object) -> bool:
    if isinstance(expr, tirx.Var):
        name = str(getattr(expr, "name_hint", "")).lower()
        return any(expr.same_as(v) for v in sid_vars) or "subblock" in name or name == "sid"
    if isinstance(expr, tirx.Call):
        op_name = effective_tirx_call_name(expr).lower()
        return "subblock" in op_name or "get_subblock_idx" in op_name
    return False


def sid_guard_branches(
    cond: object, sid_vars: set[object], tirx: object, aiv_split_factor: int
) -> tuple[str | None, str | None]:
    """Identify (then_aiv, else_aiv) sub-block core guards from an If condition."""
    if isinstance(cond, (tirx.EQ, tirx.NE)):
        for lhs, rhs in ((cond.a, cond.b), (cond.b, cond.a)):
            if _is_sid_expr(lhs, sid_vars, tirx) and (val := static_tirx_int(rhs)) is not None:
                matched = "aiv1" if val == 1 else "aiv0"
                other = ("aiv0" if val == 1 else "aiv1") if aiv_split_factor >= 2 else "aiv0"
                return (matched, other) if isinstance(cond, tirx.EQ) else (other, matched)
    if any_tirx_node(cond, tirx, lambda n: _is_sid_expr(n, sid_vars, tirx)):
        return "aiv0", ("aiv1" if aiv_split_factor >= 2 else "aiv0")
    return None, None


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


def _arg_shape_and_dtype(arg: object, tirx: object) -> tuple[tuple[int, ...] | None, str | None]:
    buf, raw_extents, _ = _unwrap_mem_arg(arg, tirx)
    if buf is None:
        return None, None
    if raw_extents and (ext := static_shape_ints(raw_extents)) is not None:
        return ext, str(buf.dtype)
    return static_shape_ints(buf.shape), str(buf.dtype)


def estimate_dav3510_cube_cycles(args: Sequence[object], spaces: AscendSpaceFacts | None, tirx: object) -> int:
    """Estimate dav-3510 Cube MMAD cycles from GEMM call arguments."""
    ext_a, dt_a = _arg_shape_and_dtype(args[0], tirx) if len(args) > 0 else (None, None)
    ext_b, dt_b = _arg_shape_and_dtype(args[1], tirx) if len(args) > 1 else (None, None)
    ext_c, dt_c = _arg_shape_and_dtype(args[2], tirx) if len(args) > 2 else (None, None)
    ta = static_tirx_int(args[3]) if len(args) > 3 else 0
    tb = static_tirx_int(args[4]) if len(args) > 4 else 0
    m = static_tirx_int(args[5]) if len(args) > 5 else None
    n = static_tirx_int(args[6]) if len(args) > 6 else None
    k = static_tirx_int(args[7]) if len(args) > 7 else None
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


def estimate_dav3510_vec_cycles(
    op_name: str, args: Sequence[object], tirx: object, data_type_cls: Callable[[str], _DTypeLike]
) -> int:
    """Estimate dav-3510 AIV vector cycles from call arguments."""
    max_b = 0
    for arg in args:
        total_b = region_bytes(arg, tirx, data_type_cls)[1]
        if total_b > max_b:
            max_b = total_b
        elif total_b == 0 and isinstance(arg, tirx.Var):
            ds = str(getattr(arg, "dtype", ""))
            if ds and ds != "handle":
                try:
                    dt = data_type_cls(ds)
                    max_b = max(max_b, int(math.ceil((dt.bits * dt.lanes) / 8.0)))
                except Exception:
                    pass
    base = max(1, math.ceil((max_b if max_b > 0 else 256) / 256.0))
    return base * (4 if any(t in op_name.lower() for t in ("exp", "log", "div", "sqrt", "rsqrt")) else 1)


def _tilelang_arg_buf(
    arg: object, tirx: object, data_type_cls: Callable[[str], _DTypeLike]
) -> tuple[str | None, str | None, int]:
    buf, raw_extents, _ = _unwrap_mem_arg(arg, tirx)
    if buf is None:
        return None, None, 0
    scope = map_tilelang_scope(buf.scope())
    full_b = buffer_bytes(static_shape_ints(buf.shape), buf.dtype, data_type_cls)
    if raw_extents:
        part_b = buffer_bytes(static_shape_ints(raw_extents), buf.dtype, data_type_cls)
        return buf.name, scope, part_b if part_b > 0 else full_b
    return buf.name, scope, full_b


def aiv_mask(pipe: str, *, aiv_split_factor: int, guarded_aiv: str | None) -> NodeResourceMask:
    """Return 24-bit NodeResourceMask for an AIV pipe under split/guard context."""
    if aiv_split_factor >= 2 and guarded_aiv is None:
        return NodeResourceMask.for_dual_aiv(pipe)
    return NodeResourceMask.for_core(guarded_aiv or "aiv0", pipe)


def copy_resource_mask(
    src_s: str, dst_s: str, *, aiv_split_factor: int, guarded_aiv: str | None
) -> NodeResourceMask:
    """Return 24-bit NodeResourceMask for a copy between memory spaces."""
    if src_s == "GM" and dst_s == "UB":
        return aiv_mask("MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" and dst_s == "GM":
        return aiv_mask("MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "GM" and dst_s == "L1":
        return NodeResourceMask.for_aic("MTE2")
    if src_s == "L1" and dst_s in ("L0A", "L0B"):
        return NodeResourceMask.for_aic("MTE1")
    if src_s == "L0C" and dst_s in ("GM", "L1"):
        return NodeResourceMask.for_aic("FIXPIPE")
    if src_s == "L0C" and dst_s == "UB":
        return NodeResourceMask.for_aic("FIXPIPE") | aiv_mask("MTE2", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" and dst_s == "L1":
        return NodeResourceMask.for_aic("MTE1") | aiv_mask("MTE3", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    if src_s == "UB" or dst_s == "UB":
        return aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=guarded_aiv)
    return NodeResourceMask.for_aic("MTE1")


def _extract_vf_accesses(
    stmt: object, *, tirx: object, data_type_cls: Callable[[str], _DTypeLike]
) -> tuple[list[str], list[str], int]:
    reads, writes, op_count = [], [], 0

    def _add(target: list[str], name: str | None, scope: str | None) -> None:
        if name is not None and scope is not None and scope != "GM" and name not in target:
            target.append(name)

    def _visit(node: object) -> None:
        nonlocal op_count
        if isinstance(node, tirx.BufferStore):
            op_count += 1
            _add(writes, node.buffer.name, map_tilelang_scope(node.buffer.scope()))
        elif isinstance(node, tirx.Call):
            op_name = effective_tirx_call_name(node).lower()
            if "simd" in op_name or "vec" in op_name:
                op_count += 1
            if ("access_ptr" in op_name and len(node.args) >= 3) or ("region" in op_name and len(node.args) >= 2):
                name, scope, _ = _tilelang_arg_buf(node, tirx, data_type_cls)
                rw = static_tirx_int(node.args[2 if "access_ptr" in op_name else 1]) or 1
                if rw & 1:
                    _add(reads, name, scope)
                if rw & 2:
                    _add(writes, name, scope)
        elif isinstance(node, tirx.BufferLoad):
            _add(reads, node.buffer.name, map_tilelang_scope(node.buffer.scope()))

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return reads, writes, max(1, op_count)


@dataclass(frozen=True)
class _TileLangLoopTask:
    label: str
    mask: NodeResourceMask
    latency: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _extract_pipelined_loop_tasks(
    loop_body: object, *, tirx: object, data_type_cls: Callable[[str], _DTypeLike],
    aiv_split_factor: int, sid_vars: set[object], guarded_aiv: str | None,
    spaces: AscendSpaceFacts | None = None, dynamic_inner_out: list[bool] | None = None,
) -> list[_TileLangLoopTask] | None:
    tasks: list[_TileLangLoopTask] = []
    aborted = False

    def _add_vf_task(body: object, scale: int, guard: str | None) -> None:
        reads, writes, vf_ops = _extract_vf_accesses(body, tirx=tirx, data_type_cls=data_type_cls)
        tasks.append(_TileLangLoopTask(
            label="simdvf", mask=aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=guard),
            latency=min(8, max(1, (vf_ops + 3) // 4) * scale), reads=tuple(reads), writes=tuple(writes),
        ))

    def _dyn_or_abort(body: object, scale: int, cur_guard: str | None) -> None:
        nonlocal aborted
        if dynamic_inner_out is not None:
            dynamic_inner_out[0] = True
            if body is not None:
                _walk(body, scale=scale, cur_guard=cur_guard)
        else:
            aborted = True

    def _walk(node: object, *, scale: int, cur_guard: str | None) -> None:
        nonlocal aborted
        if node is None or aborted:
            return
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                _walk(child, scale=scale, cur_guard=cur_guard)
        elif isinstance(node, tirx.While):
            _dyn_or_abort(node.body, scale, cur_guard)
        elif isinstance(node, tirx.For):
            if (trip := static_tirx_int(node.extent)) is None:
                _dyn_or_abort(node.body, scale, cur_guard)
            elif trip > 0:
                _walk(node.body, scale=min(8, scale * trip), cur_guard=cur_guard)
        elif isinstance(node, tirx.IfThenElse):
            if contains_buffer_load(node.condition, tirx):
                _dyn_or_abort(None, scale, cur_guard)
                if aborted:
                    return
            tg, eg = sid_guard_branches(node.condition, sid_vars, tirx, aiv_split_factor)
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
            sc = map_tilelang_scope(node.buffer.scope())
            reads, _, _ = _extract_vf_accesses(node.value, tirx=tirx, data_type_cls=data_type_cls)
            tasks.append(_TileLangLoopTask(
                label="scalar_store", mask=NodeResourceMask.for_core(cur_guard or "aiv0", "SCALAR"),
                latency=1, reads=tuple(reads), writes=(node.buffer.name,) if sc not in (None, "GM") else (),
            ))
        elif isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            op_name, args = effective_tirx_call_name(node.value).lower(), effective_tirx_call_args(node.value)
            if "copy" in op_name and len(args) >= 2:
                sn, ss, sb = _tilelang_arg_buf(args[0], tirx, data_type_cls)
                dn, ds, db = _tilelang_arg_buf(args[1], tirx, data_type_cls)
                if ss is not None and ds is not None:
                    xfer = max(sb, db)
                    eff_x = int(math.ceil(float(xfer) / max(0.15, float(spaces.dma_burst_efficiency(xfer))))) if (spaces is not None and xfer > 0 and "GM" in (ss, ds)) else xfer
                    tasks.append(_TileLangLoopTask(
                        label="copy", mask=copy_resource_mask(ss, ds, aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard),
                        latency=min(8, max(1, (eff_x + 16383) // 16384) * scale),
                        reads=(sn,) if (sn is not None and ss != "GM") else (),
                        writes=(dn,) if (dn is not None and ds != "GM") else (),
                    ))
            elif ("gemm" in op_name or "mmad" in op_name) and len(args) >= 3:
                an, a_s, _ = _tilelang_arg_buf(args[0], tirx, data_type_cls)
                bn, b_s, _ = _tilelang_arg_buf(args[1], tirx, data_type_cls)
                cn, c_s, _ = _tilelang_arg_buf(args[2], tirx, data_type_cls)
                clear_v = static_tirx_int(args[9]) if len(args) > 9 else None
                reads_l = [n for n, s in ((an, a_s), (bn, b_s)) if n is not None and s not in (None, "GM")]
                if (clear_v is None or clear_v == 0) and cn is not None and c_s not in (None, "GM") and cn not in reads_l:
                    reads_l.append(cn)
                tasks.append(_TileLangLoopTask(
                    label="gemm", mask=NodeResourceMask.for_aic("CUBE"),
                    latency=min(8, max(1, estimate_dav3510_cube_cycles(args, spaces, tirx)) * scale),
                    reads=tuple(reads_l), writes=(cn,) if (cn is not None and c_s not in (None, "GM")) else (),
                ))
            elif "simd" in op_name or "vec" in op_name:
                reads, writes, _ = _extract_vf_accesses(node.value, tirx=tirx, data_type_cls=data_type_cls)
                tasks.append(_TileLangLoopTask(
                    label="vec", mask=aiv_mask("VECTOR", aiv_split_factor=aiv_split_factor, guarded_aiv=cur_guard),
                    latency=min(8, estimate_dav3510_vec_cycles(op_name, args, tirx, data_type_cls) * scale),
                    reads=tuple(reads), writes=tuple(writes),
                ))

    _walk(loop_body, scale=1, cur_guard=guarded_aiv)
    return None if aborted else tasks


def _analytical_pipelined_ii(task_items: Sequence[_TileLangLoopTask]) -> int:
    totals: dict[tuple[str, str], int] = {}
    for item in task_items:
        lat = max(1, int(item.latency))
        for ck, pn in item.mask.to_core_pipe_pairs():
            totals[(ck, pn)] = totals.get((ck, pn), 0) + lat
    return max(totals.values(), default=1)


def solve_pipelined_loops_ii(
    pipelined_loops: Sequence[tuple[object, set[object], str | None]],
    buf_meta: Mapping[str, tuple[str, int]], spaces: AscendSpaceFacts, *, tirx: object,
    data_type_cls: Callable[[str], _DTypeLike], aiv_split_factor: int, has_simtvf: bool,
    unreserved_ubuf: bool, degraded_names: Sequence[str] = (),
) -> int | None:
    """Solve steady-state initiation interval across pipelined loops using Z3."""
    if not pipelined_loops:
        return None
    all_buf_names, degraded_set, solved_iis = set(buf_meta), set(degraded_names), []

    def _collect_local(stmt: object) -> set[str]:
        names: set[str] = set()
        tirx.stmt_functor.post_order_visit(
            stmt,
            lambda n: names.update(b.name for b in n.alloc_buffers) if isinstance(n, tirx.SBlock) else (names.add(n.buffer.name) if isinstance(n, (tirx.AllocBuffer, tirx.DeclBuffer)) else None),
        )
        return names

    for loop, sid_vars, loop_guard in pipelined_loops:
        trip = static_tirx_int(loop.extent)
        if trip is None or trip < 1:
            return None
        ann = getattr(loop, "annotations", None) or {}
        sv = static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else 2
        num_stages = max(1, sv if sv is not None else 2)
        outer_bufs, dynamic_inner = all_buf_names - _collect_local(loop.body), [False]
        task_items = _extract_pipelined_loop_tasks(
            loop.body, tirx=tirx, data_type_cls=data_type_cls, aiv_split_factor=aiv_split_factor,
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
