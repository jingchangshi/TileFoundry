"""Mode B read-only single-pass dynamic envelope accounting and Two-Snapshot IR Diff."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry_ascend.ascend.cost_model.microarch_profile import extract_microarch_profile_from_tirx
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger, ScopeWaterline, build_objective_physical_ledger, build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.anchor_specializer import (
    any_tirx_node, augment_var_bindings_from_buffers, build_free_var_int_map, entry_and_mod,
    free_tirx_vars, resolve_symbolic_grid_extent, same_tirx_var, specialize_primfunc_for_anchor,
    specialize_primfunc_for_anchors, static_shape_ints, static_tirx_int, substitute_let_env,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import _DTypeLike, dtype_bytes
from tilefoundry_ascend.ascend.schedule.tir_loop_ii import (
    access_ptr_or_region_bytes, buffer_bytes, contains_buffer_load, effective_tirx_call_args,
    effective_tirx_call_name, estimate_dav3510_cube_cycles, estimate_dav3510_vec_cycles,
    map_tilelang_scope, region_bytes, sid_guard_branches, solve_pipelined_loops_ii,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_effective_tirx_call_name = effective_tirx_call_name


@dataclass(frozen=True)
class BufferSnapshot:
    """Captured physical buffer allocation size in one IR snapshot."""

    name: str
    scope: str
    bytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class EnvelopeInspectionResult:
    """Output of Mode B dynamic envelope and Two-Snapshot IR Diff inspection."""

    ledger: ObjectivePhysicalLedger
    explicit_snapshot: tuple[BufferSnapshot, ...]
    lowered_snapshot: tuple[BufferSnapshot, ...]
    merged_peak_bytes_by_scope: dict[str, int]
    lowered_passes_observed: tuple[str, ...]
    lowering_error: str | None = None


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


def _is_disallowed_tirx_intrinsic(op_name: str, *, allow_sync: bool = False) -> bool:
    low = op_name.lower()
    if allow_sync and any(
        t in low for t in ("barrier", "pipe_barrier", "cross_flag", "cross_core", "set_flag", "wait_flag", "mbarrier", "sync")
    ):
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
    expr: object, *, tirx: object, sfu_out: list[bool] | None = None
) -> tuple[list[tuple[str, str, str]], bool, list[str], list[str]]:
    events: list[tuple[str, str, str]] = []
    has_exp = False
    ub_loads: list[str] = []
    ub_stores: list[str] = []

    def _absorb(sub: object) -> None:
        nonlocal has_exp
        sev, sexp, sul, sus = _collect_tilelang_expr_rw(sub, tirx=tirx, sfu_out=sfu_out)
        events.extend(sev)
        has_exp = has_exp or sexp
        ub_loads.extend(sul)
        ub_stores.extend(sus)

    def _visit_indices(indices: Sequence[object]) -> None:
        for idx in indices:
            if isinstance(idx, (tirx.And, tirx.Or, tirx.Not)):
                raise ValueError(
                    "Cannot use and / or / not operator to Expr, hint: use tvm.tirx.all / tvm.tirx.any instead"
                )
            _absorb(idx)

    def _pre(node: object) -> bool:
        nonlocal has_exp
        if isinstance(node, tirx.Call):
            op_name = effective_tirx_call_name(node).lower()
            call_args = effective_tirx_call_args(node)
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
                    scope = map_tilelang_scope(bl.buffer.scope())
                    rw = static_tirx_int(node.args[rw_idx]) or 1
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
                        if (scope := map_tilelang_scope(arg.buffer.scope())) is not None:
                            events.append((arg.buffer.name, scope, default_rw))
                    else:
                        _absorb(arg)
                return False
        elif isinstance(node, tirx.BufferLoad):
            _visit_indices(node.indices)
            if (scope := map_tilelang_scope(node.buffer.scope())) is not None:
                events.append((node.buffer.name, scope, "read"))
                if scope == "UB":
                    ub_loads.append(node.buffer.name)
            return False
        elif isinstance(node, tirx.BufferRegion):
            if (scope := map_tilelang_scope(node.buffer.scope())) is not None:
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


def _burst_scaled_copy_cycles(xfer_bytes: int, spaces: AscendSpaceFacts | None) -> int:
    base = max(1, (xfer_bytes + 63) // 64)
    if spaces is None or xfer_bytes <= 0:
        return base
    return int(math.ceil(float(base) / max(0.15, float(spaces.dma_burst_efficiency(xfer_bytes)))))


@dataclass
class _ScopeRec:
    direct_loads: set[str]
    direct_stores: set[str]
    direct_sfu: bool
    children: list[_ScopeRec]


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


def _is_pipelined_tirx_for(stmt: object, tirx: object) -> bool:
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
    entry_fn: object, mod_in: object | None = None, *, spaces: AscendSpaceFacts | None = None,
    tirx: object, data_type_cls: Callable[[str], _DTypeLike] | None = None,
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
        augment_var_bindings_from_buffers(entry_fn, var_bindings, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes)
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

    def _record_snap(from_alloc: bool, var_obj: object, name: str, raw_scope: str, shape_obj: object, dtype_obj: object) -> None:
        if data_type_cls is None or (scope := map_tilelang_scope(raw_scope)) in (None, "GM") or (shape := static_shape_ints(shape_obj)) is None:
            return
        snap = BufferSnapshot(name=str(name), scope=scope, bytes=buffer_bytes(shape, dtype_obj, data_type_cls), shape=shape, dtype=str(dtype_obj))
        for idx, (fa, pv, pk, ps) in enumerate(snap_entries):
            if fa == from_alloc and ((var_obj is not None and pv is not None and same_tirx_var(var_obj, pv)) or (var_obj is None and pk == f"{scope}:{name}")):
                if snap.bytes > ps.bytes:
                    snap_entries[idx] = (fa, var_obj, pk, snap)
                return
        snap_entries.append((from_alloc, var_obj, f"{scope}:{name}", snap))

    def _register_buf(buf: object, *, from_alloc: bool) -> None:
        nonlocal has_lowbit_buf, ub_2d_total, ub_2d_conflict
        if _is_lowbit_dtype_name(str(buf.dtype)):
            has_lowbit_buf = True
        raw_scope = str(buf.scope()).strip()
        scope = map_tilelang_scope(raw_scope)
        if scope == "UB" or raw_scope == "local.ub":
            ub_buf_names.add(buf.name)
        if data_type_cls is None:
            return
        _record_snap(from_alloc, buf.data, buf.name, raw_scope, buf.shape, buf.dtype)
        static_shape = static_shape_ints(buf.shape)
        nbytes = buffer_bytes(static_shape, buf.dtype, data_type_cls)
        if scope is not None and scope != "GM" and nbytes > 0:
            prev = buf_meta.get(buf.name)
            if prev is None or nbytes > prev[1]:
                buf_meta[buf.name] = (scope, nbytes)
        if scope == "UB" and static_shape is not None:
            ub_static_bytes[buf.name] = max(ub_static_bytes.get(buf.name, 0), nbytes)
            if buf.name not in seen_ub_bank and len(static_shape) >= 2 and math.prod(static_shape[:-1]) > 1 and nbytes > 0:
                seen_ub_bank.add(buf.name)
                eb = dtype_bytes(buf.dtype, data_type_cls)
                strides = getattr(buf, "strides", None) or ()
                rs = static_tirx_int(strides[-2]) if len(strides) >= 2 else None
                row_elems = rs if (rs is not None and rs > 0) else static_shape[-1]
                ub_2d_total += nbytes
                if (sb := int(math.ceil(row_elems * eb))) >= 512 and sb % 512 == 0:
                    ub_2d_conflict += nbytes
        elif scope == "L1" and static_shape is not None and nbytes > 0:
            l1_static_bytes[buf.name] = max(l1_static_bytes.get(buf.name, 0), nbytes)
        elif scope == "GM" and buf.name not in seen_gm_keys:
            seen_gm_keys.add(buf.name)
            eb = dtype_bytes(buf.dtype, data_type_cls)
            if static_shape is not None:
                gm_static_bytes[buf.name] = nbytes
                if (tot := math.prod(static_shape) if static_shape else 0) == 1:
                    gm_scalar_names.add(buf.name)
                if static_shape and static_shape[-1] >= 1:
                    contig_candidates.append((tot, static_shape[-1], 0, nbytes, int(math.ceil(static_shape[-1] * eb))))
            else:
                gm_static_bytes[buf.name] = 0
                if buf.shape and (last_d := static_tirx_int(buf.shape[-1])) is not None and last_d >= 1:
                    dyn_cnt = sum(1 for d in buf.shape if static_tirx_int(d) is None)
                    st_dims = [v for d in buf.shape if (v := static_tirx_int(d)) is not None]
                    st_prod = int(math.ceil(math.prod(st_dims) * eb)) if st_dims else 0
                    contig_candidates.append((None, last_d, dyn_cnt, st_prod, int(math.ceil(last_d * eb))))

    for param in getattr(entry_fn, "params", ()):
        if (buf := (getattr(entry_fn, "buffer_map", None) or {}).get(param)) is not None:
            _register_buf(buf, from_alloc=False)

    def _check_expr(expr: object, acc: _SubtreeAcc, *, check_arith: bool = False, loop_vars: tuple[object, ...] = ()) -> bool:
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

        def _post(sub: object) -> None:
            nonlocal has_indirect_gather, has_dyn_offset_pipe, has_arith, has_load
            if isinstance(sub, tirx.Call):
                oname = effective_tirx_call_name(sub).lower()
                if _is_disallowed_tirx_intrinsic(oname, allow_sync=False):
                    acc.has_disallowed_strict = True
                if _is_disallowed_tirx_intrinsic(oname, allow_sync=True):
                    acc.has_disallowed_sync_ok = True
                if check_arith and _is_tirx_math_call(oname):
                    has_arith = True
                if post_lower and loop_vars and "access_ptr" in str(getattr(sub.op, "name", sub.op)).lower() and len(sub.args) >= 3:
                    if any_tirx_node(
                        sub.args[2],
                        tirx,
                        lambda m: bool(mod_classes and isinstance(m, mod_classes) and any(same_tirx_var(fv, lv) for fv in free_tirx_vars(m.a, tirx=tirx) for lv in loop_vars)),
                    ):
                        has_dyn_offset_pipe = True
            elif isinstance(sub, tirx.BufferLoad):
                if any(contains_buffer_load(i, tirx) for i in sub.indices):
                    has_indirect_gather = True
                if check_arith:
                    rs = str(sub.buffer.scope()).strip()
                    if sub.buffer.name in ub_buf_names or map_tilelang_scope(rs) in ("UB", "GM") or rs in ("local.ub", "global"):
                        has_load = True
            elif check_arith and arith_types and isinstance(sub, arith_types):
                has_arith = True

        tirx.stmt_functor.post_order_visit(expr, _post)
        return has_arith and has_load

    def _is_vec_sblock(node: object) -> bool:
        ann = getattr(node, "annotations", None) or {}
        hint = str(getattr(node, "name_hint", "")).lower()
        return "simd_vf" in hint or "simt_vf" in hint or "tl.simdvf_lanes" in ann or "tl.vf_source_index" in ann

    def _is_vec_attr(node: object) -> bool:
        key, nlow = str(getattr(node, "attr_key", "")).lower(), str(getattr(node, "node", "")).lower()
        if any(t in key for t in ("simdvf", "simtvf", "simt")) or "simt" in nlow or key == "pragma_parallel":
            return True
        if "thread_extent" in key:
            tag = str(getattr(node.node, "thread_tag", "")).lower()
            iv_n = str(getattr(getattr(node.node, "var", None), "name_hint", "")).lower()
            return all(t not in tag and t not in iv_n for t in ("blockidx", "cthread", "subblock"))
        return False

    def _record_block_dim(tag: str, expr: object, env: Mapping[object, object]) -> None:
        nonlocal any_symbolic_grid
        val, is_sym = resolve_symbolic_grid_extent(expr, env, eff_bindings or var_bindings, tirx=tirx, tvm_mod=tvm_mod, var_obj_bindings=var_obj_bindings)
        if is_sym:
            any_symbolic_grid = True
        elif val is not None and val > 0:
            block_dims[tag] = val

    def _emit_bind_pipe_v(val_node: object, repeat: int, mte2_run: list[int], guarded_aiv: str | None) -> None:
        if isinstance(val_node, tirx.Call) and data_type_cls is not None:
            b_op = effective_tirx_call_name(val_node).lower()
            if ("simd" in b_op or "vec" in b_op) and "vld" not in b_op and "pset" not in b_op:
                mte2_run[0] = 0
                _emit_aiv("PIPE_V", repeat * estimate_dav3510_vec_cycles(b_op, effective_tirx_call_args(val_node), tirx, data_type_cls), guarded_aiv)

    def _apply_thread_scope(tag: str, iv_name: str, extent: object, var_obj: object, ctx: _WalkCtx, *, is_attr: bool) -> tuple[frozenset[object], bool, bool]:
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
            if (sv := static_tirx_int(extent)) is not None and sv >= 2:
                detected_split = 2
        elif is_attr and "blockidx" not in tag and "blockidx" not in iv_name:
            has_simtvf = True
            _add_opaque("simt_vf")
        return frozenset(next_sids), next_block, is_sub

    def _handle_call(call_node: object, ctx: _WalkCtx, *, repeat: int, mte2_run: list[int]) -> None:
        nonlocal detected_split, padding_bytes, max_mte2, illegal_dma, in_loop_pipe_all, seen_in_loop_gm_dma
        op_name, args, ann = effective_tirx_call_name(call_node).lower(), effective_tirx_call_args(call_node), getattr(call_node, "annotations", None) or {}
        if any(t in op_name for t in ("dual_copy", "subblock", "get_subblock_idx")) or "dual_dst_ctl" in ann:
            detected_split = 2
        if "pipe_barrier" in op_name and ctx.in_loop and any("pipe_all" in str(getattr(a, "value", a)).lower() for a in call_node.args):
            in_loop_pipe_all += 1
        if "copy" in op_name and len(args) >= 2 and data_type_cls is not None and spaces is not None:
            src_s, xfer_b, inner_b = region_bytes(args[0], tirx, data_type_cls, let_env=ctx.let_env)
            dst_s, dst_b, dst_inner_b = region_bytes(args[1], tirx, data_type_cls, let_env=ctx.let_env)
            is_gm_name = post_lower and any(t in op_name for t in ("gm_to_", "_to_gm", "_gm_", "gm2", "2gm"))
            if is_gm_name or (src_s is not None and dst_s is not None and src_s != dst_s and "GM" in (src_s, dst_s)):
                if ctx.in_loop:
                    seen_in_loop_gm_dma = True
                if post_lower:
                    xfer_pl = max(
                        access_ptr_or_region_bytes(args[0], tirx=tirx, data_type_cls=data_type_cls, let_env=ctx.let_env),
                        access_ptr_or_region_bytes(args[1], tirx=tirx, data_type_cls=data_type_cls, let_env=ctx.let_env),
                    )
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
            if data_type_cls is not None:
                m_val = static_tirx_int(args[5]) if len(args) > 5 else None
                if m_val is None and isinstance(args[0], tirx.Call) and len(args[0].args) >= 4:
                    if (ext := static_shape_ints(args[0].args[2:])) and len(ext) >= 2:
                        m_val = ext[-1] if (static_tirx_int(args[3]) if len(args) > 3 else 0) else ext[-2]
                if m_val is not None and m_val > 0:
                    cube_m_values.append(m_val)
                _add_pipe("AIC.PIPE_M", repeat * estimate_dav3510_cube_cycles(args, spaces, tirx))
        elif ("simd" in op_name or "vec" in op_name) and data_type_cls is not None:
            mte2_run[0] = 0
            _emit_aiv("PIPE_V", repeat * estimate_dav3510_vec_cycles(op_name, args, tirx, data_type_cls), ctx.guarded_aiv)

    def _walk(stmt: object, ctx: _WalkCtx, *, repeat: int = 1, mte2_run: list[int] | None = None) -> _SubtreeAcc:
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
                    seq_env[child.var] = substitute_let_env(child.value, seq_env, tirx=tirx)
                    _emit_bind_pipe_v(getattr(child, "value", None), repeat, mte2_run, ctx.guarded_aiv)
                    if hasattr(child, "body"):
                        acc.merge_non_scope(_walk(child.body, ctx.with_(let_env=dict(seq_env)), repeat=repeat, mte2_run=mte2_run))
                else:
                    acc.merge_non_scope(_walk(child, ctx.with_(let_env=dict(seq_env)), repeat=repeat, mte2_run=mte2_run))
            return acc

        if (let_stmt_cls is not None and isinstance(stmt, let_stmt_cls)) or (bind_cls is not None and isinstance(stmt, bind_cls)):
            _check_expr(stmt.value, acc, loop_vars=ctx.loop_vars)
            next_env = {**ctx.let_env, stmt.var: substitute_let_env(stmt.value, ctx.let_env, tirx=tirx)}
            if isinstance(stmt, tirx.Bind):
                _emit_bind_pipe_v(getattr(stmt, "value", None), repeat, mte2_run, ctx.guarded_aiv)
            if hasattr(stmt, "body"):
                acc.merge_non_scope(_walk(stmt.body, ctx.with_(let_env=next_env), repeat=repeat, mte2_run=mte2_run))
            return acc

        if isinstance(stmt, tirx.While):
            _add_opaque("while_loop")
            if contains_buffer_load(stmt.condition, tirx):
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
            stages_val = static_tirx_int(ann.get("num_stages")) if "num_stages" in ann else None
            is_pipe = kind_val == 5 or "pipelined" in kind_str or "num_stages" in ann or (stages_val is not None and stages_val >= 1) or bool(ann.get("enable_offset")) or (ann.get("software_pipeline_stage") is not None)
            if ann.get("enable_offset") or kind_val == 5 or (stages_val is not None and stages_val >= 2):
                has_dyn_offset_pipe = True
            next_found_pipe = ctx.found_pipe_loop
            if _is_pipelined_tirx_for(stmt, tirx) and not ctx.found_pipe_loop:
                pipelined_loops.append((stmt, set(ctx.sid_vars), ctx.guarded_aiv))
                next_found_pipe = True
            if ctx.in_block_scope and block_var_info:
                ext_sub = substitute_let_env(stmt.extent, ctx.let_env, tirx=tirx)
                if any(same_tirx_var(v, block_var_info[0][0]) for v in free_tirx_vars(ext_sub, tirx=tirx)):
                    persistent_work_items.append(("loop", ext_sub))

            if (trip := static_tirx_int(stmt.extent)) is None:
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
            if contains_buffer_load(stmt.condition, tirx):
                _add_opaque("data_dependent_branch")
            _check_expr(stmt.condition, acc, loop_vars=ctx.loop_vars)
            if ctx.in_block_scope and block_var_info and stmt.else_case is None:
                cond_sub = substitute_let_env(stmt.condition, ctx.let_env, tirx=tirx)
                if any(same_tirx_var(v, block_var_info[0][0]) for v in free_tirx_vars(cond_sub, tirx=tirx)):
                    persistent_work_items.append(("guard", cond_sub))
            t_g, e_g = sid_guard_branches(stmt.condition, set(ctx.sid_vars), tirx, split_hint)
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
            if "vector_count" in ann and (vc := static_tirx_int(ann["vector_count"])) is not None and vc >= 2:
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
                if ctx.in_pipelined and map_tilelang_scope(buf.scope()) in ("UB", "L1", "L0A", "L0B", "SSBUF"):
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
            if (sc := map_tilelang_scope(stmt.buffer.scope())) is not None:
                acc.events.append((stmt.buffer.name, sc, "write"))
                if sc == "UB":
                    acc.ub_stores.add(stmt.buffer.name)
                    acc.scope_direct_stores.add(stmt.buffer.name)
            rs = str(stmt.buffer.scope()).strip()
            if stmt.buffer.name in ub_buf_names or map_tilelang_scope(rs) == "UB" or rs == "local.ub":
                if _check_expr(substitute_let_env(stmt.value, ctx.let_env, tirx=tirx), _SubtreeAcc(), check_arith=True, loop_vars=ctx.loop_vars):
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
                vmap = build_free_var_int_map(iexpr, eff_bindings, var_obj_bindings, tirx=tirx, exclude_var=bvar)
                sub_e = tirx.stmt_functor.substitute(iexpr, vmap) if vmap else iexpr
                if any(not same_tirx_var(v, bvar) for v in free_tirx_vars(sub_e, tirx=tirx)):
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
                        val = static_tirx_int(sim)
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


def _detect_tilelang_aiv_split_factor(func: object, *, tirx: object) -> int:
    return 1 if (func is None or getattr(func, "body", None) is None) else _scan_primfunc_unified(func, tirx=tirx).aiv_split_factor


def _count_in_loop_pipe_all_barriers(stmt: object, *, tirx: object) -> int:
    return _scan_primfunc_unified(type("_Fn", (), {"body": stmt, "params": (), "buffer_map": {}, "attrs": {}})(), tirx=tirx).in_loop_pipe_all_barriers


def _count_tirx_ub_scalar_compute_loops(stmt: object, ub_buf_names: set[str], *, tirx: object, allow_sync_intrinsics: bool = False) -> int:
    return _scan_primfunc_unified(
        type("_Fn", (), {"body": stmt, "params": (), "buffer_map": {}, "attrs": {}})(),
        tirx=tirx, ub_buf_override=ub_buf_names, allow_sync_in_scalar_loops=allow_sync_intrinsics,
    ).ub_scalar_compute_loops


def _try_solve_tilelang_pipelined_ii(
    entry_fn: object, spaces: AscendSpaceFacts, *, tirx: object, DataTypeCls: Callable[[str], _DTypeLike],
    aiv_split_factor: int, has_simtvf: bool, unreserved_ubuf: bool, degraded_names: Sequence[str] = (),
) -> int | None:
    if entry_fn is None or getattr(entry_fn, "body", None) is None:
        return None
    scan = _scan_primfunc_unified(entry_fn, spaces=spaces, tirx=tirx, data_type_cls=DataTypeCls, aiv_split_factor=aiv_split_factor)
    return solve_pipelined_loops_ii(
        scan.pipelined_loops, scan.buf_meta, spaces, tirx=tirx, data_type_cls=DataTypeCls,
        aiv_split_factor=aiv_split_factor, has_simtvf=has_simtvf,
        unreserved_ubuf=unreserved_ubuf, degraded_names=degraded_names,
    )


def _scope_capacities(facts: AscendSpaceFacts, *, has_simtvf: bool, unreserved_ubuf: bool) -> dict[str, int]:
    ub_mode = "simtvf" if has_simtvf else "simdvf"
    return {
        "UB": facts.ub_limit_bytes(ub_mode, unreserved_ubuf=unreserved_ubuf),
        "L1": facts.l1_bytes, "L0A": facts.l0a_bytes, "L0B": facts.l0b_bytes,
        "L0C": facts.l0c_bytes, "SSBUF": facts.ssbuf_bytes,
    }


def _extract_tilelang_buffers_from_mod(
    mod: object, *, tirx: object, data_type_cls: Callable[[str], _DTypeLike], prefer_allocate: bool
) -> tuple[BufferSnapshot, ...]:
    if mod is None:
        return ()
    snaps: list[BufferSnapshot] = []
    for _gvar, fn in mod.functions.items():
        if isinstance(fn, tirx.PrimFunc):
            scan = _scan_primfunc_unified(fn, tirx=tirx, data_type_cls=data_type_cls)
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
        if (len(mat.shape) == len(exp.shape) and mat.bytes <= exp.bytes) or (len(mat.shape) > len(exp.shape) and mat.shape[0] <= 1):
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
            base = snap.name[:-4] if (snap.name not in exp_map and snap.name.endswith("_buf") and snap.name[:-4] in exp_map) else snap.name
            if base in exp_map and base not in matched:
                matched.add(base)
                if (delta := snap.bytes - exp_map[base]) > 0:
                    imp_map[f"{base}_expansion"] = delta
            else:
                imp_map[snap.name] = snap.bytes
        peak = merged_peak_by_scope.get(scope, sum(s.bytes for s in low_snaps) if low_snaps else sum(exp_map.values()))
        waterlines[scope] = build_scope_waterline(
            scope=scope, capacity_bytes=capacities.get(scope, 0), explicit_buffers=exp_map,
            implicit_buffers=imp_map, peak_allocated_bytes=peak,
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
    entry_fn: object,
    mod_in: object,
    *,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
) -> tuple[dict[str, object], list[str]]:
    cfg: dict[str, object] = {}
    flags: list[str] = []
    _append = lambda items: [flags.append(it) for it in items if it and it not in flags]

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


_PASS_TAP_CLASS_CACHE: dict[object, type] = {}


def _pass_instrument_if_available(cls: type) -> type:
    try:
        from tilelang import tvm as tvm_mod  # noqa: PLC0415
    except ImportError:
        return cls
    dec = getattr(getattr(getattr(tvm_mod, "ir", None), "instrument", None), "pass_instrument", None)
    return dec(cls) if callable(dec) else cls


def _build_pass_tap_instance(tvm_mod: object) -> ReadOnlyAscendPassTap:
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
        "tl.MaterializeMultiBuffer", "tl.MergeUBAllocations", "tl.InsertNd2Nz",
        "tl.NormalizeAscendFractalStorage", "tl.LowerTileOp", "tl.AscendLowerTileOp",
    )

    def __init__(self) -> None:
        self.observed_passes: list[str] = []
        self.captured_mods: dict[str, object] = {}

    def run_before_pass(self, mod: object, info: object) -> None:
        """Capture pre-merge IRModule immediately before tl.MergeUBAllocations."""
        if str(info.name) == "tl.MergeUBAllocations":
            self.captured_mods["pre_merge"] = mod

    def run_after_pass(self, mod: object, info: object) -> None:
        """Record executed pass name and capture read-only IR snapshots at target taps."""
        name = str(info.name)
        self.observed_passes.append(name)
        if name in self.TARGET_PASSES:
            self.captured_mods[name] = mod


@contextlib.contextmanager
def _ascend_compile_session(
    ctx_target: object, *, pass_configs: Mapping[str, object] | None = None, instruments: Sequence[object] = ()
):
    from tilelang import tvm  # noqa: PLC0415

    cur = tvm.target.Target.current(allow_none=True)
    cur_kind = str(getattr(getattr(cur, "kind", None), "name", "") or "")
    target_ctx = ctx_target if (cur is None or cur_kind != "ascend") else contextlib.nullcontext()
    with target_ctx, tvm.transform.PassContext(config=dict(pass_configs or {}), instruments=list(instruments)):
        yield


def _reconcile_post_lower_device_mod(
    device_mod: object, *, spaces: AscendSpaceFacts, tirx: object, data_type_cls: Callable[[str], _DTypeLike],
    pre_pipe_cycles: dict[str, int], pre_barriers: int, pre_min_gm_dma_bytes: int, pre_dma_burst_eff: float,
    pre_ub_scalar_loops: int, pre_dyn_offset_pipe: bool, explicit_aiv_split: int | None, effective_aiv_split: int,
) -> tuple[dict[str, int], int, int, float, int, bool]:
    dev_funcs = [fn for fn in device_mod.functions.values() if isinstance(fn, tirx.PrimFunc)] if (device_mod is not None and hasattr(device_mod, "functions")) else []
    if not dev_funcs:
        return (dict(pre_pipe_cycles), pre_barriers, pre_min_gm_dma_bytes, pre_dma_burst_eff, pre_ub_scalar_loops, pre_dyn_offset_pipe)

    post_barriers = post_ub_scalar = 0
    post_dyn, post_pipe_totals, eff_pools = pre_dyn_offset_pipe, {}, []
    for dev_fn in dev_funcs:
        dscan = _scan_primfunc_unified(
            dev_fn, spaces=spaces, tirx=tirx, data_type_cls=data_type_cls, aiv_split_factor=explicit_aiv_split,
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


def inspect_tilelang_primfunc(
    prim_func: object, *, target: AscendTarget | None = None, pass_configs: dict[str, object] | None = None,
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
    entry_fn, mod_in = entry_and_mod(prim_func, tvm, tirx)
    raw_cfg, _ = _extract_primfunc_pass_configs_and_flags(entry_fn, mod_in, pass_configs=pass_configs, compile_flags=compile_flags)
    ctx = create_backend_context("ascend", None, "auto")
    tap, lowering_error, device_mod = _build_pass_tap_instance(tvm), None, None

    with _ascend_compile_session(ctx.target, pass_configs=raw_cfg, instruments=[tap]):
        entry_fn, mod_in = specialize_primfunc_for_anchors(entry_fn, mod_in, var_bindings, tvm_mod=tvm, tirx=tirx, ordered_tensor_shapes=ordered_tensor_shapes)
        cfg, merged_flags = _extract_primfunc_pass_configs_and_flags(entry_fn, mod_in, pass_configs=pass_configs, compile_flags=compile_flags)
        ubuf_flag = "--cce-disable-vf-stack-reserved-ubuf"
        unreserved_hint = bool(unreserved_ubuf or any(ubuf_flag in f for f in merged_flags) or _attrs_contain_flag(cfg, ubuf_flag))
        scan = _scan_primfunc_unified(
            entry_fn, mod_in, spaces=spaces, tirx=tirx, data_type_cls=TLDataType,
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
    dev_snaps = _extract_tilelang_buffers_from_mod(device_mod, tirx=tirx, data_type_cls=TLDataType, prefer_allocate=True)
    mat_snaps = _extract_tilelang_buffers_from_mod(captured.get("tl.MaterializeMultiBuffer"), tirx=tirx, data_type_cls=TLDataType, prefer_allocate=False) or dev_snaps
    pre_merge_mod = next((m for k in ("pre_merge", "tl.LowerTileOp", "tl.AscendLowerTileOp", "tl.InsertNd2Nz", "tl.NormalizeAscendFractalStorage", "tl.MaterializeMultiBuffer") if (m := captured.get(k)) is not None), device_mod)
    lowered_snaps = _extract_tilelang_buffers_from_mod(pre_merge_mod, tirx=tirx, data_type_cls=TLDataType, prefer_allocate=True)
    post_merge_snaps = _extract_tilelang_buffers_from_mod(captured.get("tl.MergeUBAllocations") or device_mod, tirx=tirx, data_type_cls=TLDataType, prefer_allocate=True)

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
        device_mod, spaces=spaces, tirx=tirx, data_type_cls=TLDataType, pre_pipe_cycles=scan.pipe_cycles,
        pre_barriers=scan.in_loop_pipe_all_barriers, pre_min_gm_dma_bytes=scan.min_gm_dma_bytes,
        pre_dma_burst_eff=scan.dma_burst_efficiency_ratio, pre_ub_scalar_loops=scan.ub_scalar_compute_loops,
        pre_dyn_offset_pipe=scan.uses_dynamic_offset_pipelining, explicit_aiv_split=aiv_split_factor,
        effective_aiv_split=scan.aiv_split_factor,
    )

    has_overflow = any(wl.overflow_bytes > 0 for wl in waterlines.values())
    solved_ii = solve_pipelined_loops_ii(
        scan.pipelined_loops, scan.buf_meta, spaces, tirx=tirx, data_type_cls=TLDataType,
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
    "BufferSnapshot", "EnvelopeInspectionResult", "ReadOnlyAscendPassTap",
    "inspect_tilelang_primfunc", "specialize_primfunc_for_anchor",
]
