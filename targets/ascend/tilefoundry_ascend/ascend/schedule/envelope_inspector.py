"""Mode B read-only dynamic envelope accounting and Two-Snapshot IR Diff."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.hir.pipeline import BufferSchedule, PipelineError, PipelineSpec
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    Stmt,
    While,
)
from tilefoundry.ir.types import StorageKind, TensorType
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Edge,
    Reuse,
    Task,
    tf_adapter,
)
from tilefoundry_ascend.ascend.ir.execution import SIMDVF_KIND, SIMTVF_KIND
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    AscendDualCopy,
    AscendDualCopyL0cToUb,
    AscendDualCopyUbToL1,
    AscendGmToL1,
    AscendGmToL1Nz,
    AscendGmToUb,
    AscendGmToUbPad,
    AscendL0cToGm,
    AscendL0cToGmAtomic,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendUbToGm,
    AscendUbToGmAtomic,
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
    format_pipe_key,
    solve_ascend_pipeline,
    task_resource_mask,
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
    "shared.l0a": "L0A",
    "shared.l0b": "L0B",
    "shared.l0c": "L0C",
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


def _region_bytes(arg, tirx, DataTypeCls) -> tuple[str | None, int, int]:
    """Return (mapped_scope, total_bytes, inner_contiguous_bytes) for copy args."""
    if isinstance(arg, tirx.Call) and "region" in str(getattr(arg.op, "name", arg.op)):
        if len(arg.args) >= 2 and isinstance(arg.args[0], tirx.BufferLoad):
            buf = arg.args[0].buffer
            scope = _map_tilelang_scope(buf.scope())
            elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
            extents = _static_shape_ints(arg.args[2:])
            if extents is None or not extents:
                return scope, 0, 0
            total = int(math.ceil(math.prod(extents) * elem_bytes))
            inner = int(math.ceil(extents[-1] * elem_bytes))
            return scope, total, inner
    if isinstance(arg, tirx.BufferRegion):
        buf = arg.buffer
        scope = _map_tilelang_scope(buf.scope())
        elem_bytes = _dtype_bytes(buf.dtype, DataTypeCls)
        extents = _static_shape_ints([r.extent for r in arg.region])
        if extents is None or not extents:
            return scope, 0, 0
        total = int(math.ceil(math.prod(extents) * elem_bytes))
        inner = int(math.ceil(extents[-1] * elem_bytes))
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
            op_name = str(getattr(node.op, "name", node.op)).lower()
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
        op_name = str(getattr(expr.op, "name", expr.op)).lower()
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

    def _walk(
        stmt,
        *,
        repeat: int = 1,
        mte2_run: list[int] | None = None,
        sid_vars: set[object] | None = None,
        guarded_aiv: str | None = None,
    ) -> None:
        if mte2_run is None:
            mte2_run = [0]
        if sid_vars is None:
            sid_vars = set()
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                _walk(
                    child,
                    repeat=repeat,
                    mte2_run=mte2_run,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
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
                )
            elif trip > 0:
                local_mte2 = [0]
                _walk(
                    stmt.body,
                    repeat=repeat * trip,
                    mte2_run=local_mte2,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
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
            )
            b2 = [mte2_run[0]]
            if stmt.else_case is not None:
                _walk(
                    stmt.else_case,
                    repeat=repeat,
                    mte2_run=b2,
                    sid_vars=sid_vars,
                    guarded_aiv=next_else,
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
            )
            return
        if isinstance(stmt, (tirx.SBlockRealize,)):
            _walk(
                stmt.block,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
            )
            return
        if isinstance(stmt, tirx.SBlock):
            _walk(
                stmt.body,
                repeat=repeat,
                mte2_run=mte2_run,
                sid_vars=sid_vars,
                guarded_aiv=guarded_aiv,
            )
            return
        if isinstance(stmt, (tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(stmt, "body"):
                _walk(
                    stmt.body,
                    repeat=repeat,
                    mte2_run=mte2_run,
                    sid_vars=sid_vars,
                    guarded_aiv=guarded_aiv,
                )
            return
        if isinstance(stmt, tirx.BufferStore):
            if mte2_run[0] > 0:
                mte2_run[0] = 0
            core = "AIV1" if guarded_aiv == "aiv1" else "AIV0"
            _add_pipe(f"{core}.PIPE_S", repeat)
            return
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            op_name = str(getattr(stmt.value.op, "name", stmt.value.op))
            args = list(stmt.value.args)
            if "copy" in op_name and len(args) >= 2:
                src_s, xfer_b, inner_b = _region_bytes(args[0], tirx, DataTypeCls)
                dst_s, dst_b, _dst_in = _region_bytes(args[1], tirx, DataTypeCls)
                if src_s is not None and dst_s is not None:
                    xfer = max(xfer_b, dst_b)
                    if src_s != dst_s and not spaces.is_dma_path_legal(src_s, dst_s):
                        illegal_dma[0] = True
                    if inner_b > 0 and src_s == "GM" and dst_s == "UB":
                        rem = inner_b % spaces.ub_align_bytes
                        if rem != 0:
                            padding_bytes[0] += (spaces.ub_align_bytes - rem) * repeat
                        mte2_run[0] += 1
                        max_mte2[0] = max(max_mte2[0], mte2_run[0])
                        _emit_aiv_pipe(
                            "PIPE_MTE2",
                            repeat * max(1, (xfer + 63) // 64),
                            guarded_aiv=guarded_aiv,
                        )
                    elif src_s == "UB" and dst_s == "GM":
                        mte2_run[0] = 0
                        _emit_aiv_pipe(
                            "PIPE_MTE3",
                            repeat * max(1, (xfer + 63) // 64),
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
                        _add_pipe("AIC.PIPE_MTE2", repeat * max(1, (xfer + 63) // 64))
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
            elif "gemm" in op_name:
                mte2_run[0] = 0
                _add_pipe("AIC.PIPE_M", repeat * 4)
            elif "simd" in op_name or "vec" in op_name:
                mte2_run[0] = 0
                _emit_aiv_pipe("PIPE_V", repeat, guarded_aiv=guarded_aiv)

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
            op_name = str(getattr(node.op, "name", node.op)).lower()
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
            aborted[0] = True
            return
        if isinstance(node, tirx.For):
            trip = _static_tirx_int(node.extent)
            if trip is None:
                aborted[0] = True
                return
            if trip > 0:
                _walk(node.body, scale=min(8, scale * trip), cur_guard=cur_guard)
            return
        if isinstance(node, tirx.IfThenElse):
            if _contains_buffer_load(node.condition, tirx):
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
            op_name = str(getattr(node.value.op, "name", node.value.op)).lower()
            args = list(node.value.args)
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
                    lat = min(8, max(1, (xfer + 16383) // 16384) * scale)
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
                m = _static_tirx_int(args[5]) if len(args) > 5 else None
                n = _static_tirx_int(args[6]) if len(args) > 6 else None
                k = _static_tirx_int(args[7]) if len(args) > 7 else None
                if m and n and k:
                    lat = min(8, max(1, (2 * m * n * k + 65535) // 65536) * scale)
                else:
                    lat = min(8, 4 * scale)
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
                tasks.append(
                    _TileLangLoopTask(
                        label="vec",
                        mask=mask,
                        latency=min(8, scale),
                        reads=tuple(reads),
                        writes=tuple(writes),
                    )
                )

    _walk(loop_body, scale=1, cur_guard=guarded_aiv)
    if aborted[0]:
        return None
    return tasks


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
        task_items = _extract_pipelined_loop_tasks(
            loop.body,
            tirx=tirx,
            DataTypeCls=DataTypeCls,
            aiv_split_factor=aiv_split_factor,
            sid_vars=sid_vars,
            guarded_aiv=loop_guard,
        )
        if task_items is None:
            return None
        if not task_items:
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
        try:
            res = solve_ascend_pipeline(problem)
        except (PipelineError, ValueError):
            return None
        solved_iis.append(res.assignment.ii)

    return max(solved_iis) if solved_iis else None


def inspect_tilelang_primfunc(
    prim_func,
    *,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    aiv_split_factor: int | None = None,
    unreserved_ubuf: bool = False,
) -> EnvelopeInspectionResult:
    """Inspect a TileLang PrimFunc using Two-Snapshot IR Diff without bisheng compile."""
    import tilelang
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_to_host_device_ir
    from tilelang.utils.language import DataType as TLDataType
    from tilelang import tvm
    from tvm import tirx

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

    effective_aiv_split = (
        aiv_split_factor
        if aiv_split_factor is not None
        else _detect_tilelang_aiv_split_factor(entry_fn, tirx=tirx)
    )
    explicit_snaps = _extract_tilelang_buffers_from_func(
        entry_fn, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=False
    )
    requested_mb = _requested_multibuffer_names(entry_fn, tirx=tirx)
    pipe_cycles, mte2_inflight, pad_bytes, opaque, has_simtvf, illegal_dma = _scan_tilelang_ast(
        entry_fn,
        spaces,
        tirx=tirx,
        DataTypeCls=TLDataType,
        aiv_split_factor=effective_aiv_split,
    )
    capacities = _scope_capacities(
        spaces,
        has_simtvf=has_simtvf,
        unreserved_ubuf=unreserved_ubuf,
    )

    observed: list[str] = []
    captured_mods: dict[str, object] = {}

    @tvm.ir.instrument.pass_instrument
    class _SnapshotCapture:
        def run_before_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                captured_mods["pre_merge"] = mod

        def run_after_pass(self, mod, info):
            observed.append(str(info.name))
            if info.name in (
                "tl.MaterializeMultiBuffer",
                "tl.MergeUBAllocations",
                "tl.InsertNd2Nz",
                "tl.NormalizeAscendFractalStorage",
            ):
                captured_mods[str(info.name)] = mod

    lowering_error: str | None = None
    device_mod = None
    ctx = create_backend_context("ascend", None, "auto")
    cfg = dict(pass_configs or {})
    try:
        with tvm.transform.PassContext(config=cfg, instruments=[_SnapshotCapture()]):
            _host_mod, device_mod, _params, _tgt, _htgt = lower_to_host_device_ir(mod_in, ctx)
    except Exception as exc:
        lowering_error = f"{type(exc).__name__}: {exc}"

    mat_snaps = _extract_tilelang_buffers_from_mod(
        captured_mods.get("tl.MaterializeMultiBuffer"),
        tirx=tirx,
        DataTypeCls=TLDataType,
        prefer_allocate=False,
    )
    pre_merge_mod = (
        captured_mods.get("pre_merge")
        or captured_mods.get("tl.InsertNd2Nz")
        or captured_mods.get("tl.NormalizeAscendFractalStorage")
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

    has_overflow = any(wl.overflow_bytes > 0 for wl in waterlines.values())
    solved_ii: int | None = None
    if not opaque and not illegal_dma and not has_overflow:
        solved_ii = _try_solve_tilelang_pipelined_ii(
            entry_fn,
            spaces,
            tirx=tirx,
            DataTypeCls=TLDataType,
            aiv_split_factor=effective_aiv_split,
            has_simtvf=has_simtvf,
            unreserved_ubuf=unreserved_ubuf,
            degraded_names=degraded,
        )
    analysis_mode = "exact_modulo_z3" if solved_ii is not None else "readonly_envelope"

    ledger = build_objective_physical_ledger(
        analysis_mode=analysis_mode,
        is_legal=(not illegal_dma) and (lowering_error is None),
        steady_state_ii=solved_ii,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        mte2_inflight_per_aiv=mte2_inflight,
        unaligned_dma_padding_bytes=pad_bytes,
        opaque_dynamic_regions=opaque,
    )
    return EnvelopeInspectionResult(
        ledger=ledger,
        explicit_snapshot=explicit_snaps,
        lowered_snapshot=lowered_snaps,
        merged_peak_bytes_by_scope=merged_peak,
        lowered_passes_observed=tuple(observed),
        lowering_error=lowering_error,
    )


def _tir_tensor_bytes(tensor_type: TensorType) -> tuple[int, int]:
    elems = 1
    inner = 1
    for idx, dim in enumerate(tensor_type.shape):
        ext = static_dim_value(dim)
        if ext is None or ext < 1:
            return 0, 0
        elems *= ext
        if idx == len(tensor_type.shape) - 1:
            inner = ext
    total = (elems * tensor_type.dtype.bit_width + 7) // 8
    inner_bytes = (inner * tensor_type.dtype.bit_width + 7) // 8
    return total, inner_bytes


def _infer_tir_space(op: Op | None, tensor_type: TensorType) -> str:
    if tensor_type.storage is StorageKind.SMEM:
        return "UB"
    if isinstance(op, AscendL1ToL0A):
        return "L0A"
    if isinstance(op, AscendL1ToL0B):
        return "L0B"
    if isinstance(op, (AscendL0cToGm, AscendL0cToGmAtomic, AscendDualCopyL0cToUb)):
        return "L0C"
    return "L1"


def _tir_for_static_trip_count(stmt: For) -> int | None:
    """Compute static integer trip count for a TileFoundry TIR For loop."""
    start_v = static_dim_value(stmt.start)
    stop_v = static_dim_value(stmt.stop)
    step_v = static_dim_value(stmt.step)
    if start_v is None or stop_v is None or step_v is None or step_v <= 0:
        return None
    return max(0, (stop_v - start_v + step_v - 1) // step_v)


def inspect_tir_envelope(
    fn_or_stmt: PrimFunction | Stmt,
    *,
    target: AscendTarget | None = None,
    unreserved_ubuf: bool = False,
) -> EnvelopeInspectionResult:
    """Read-only Mode B inspector for TileFoundry TIR containing dynamic or opaque scopes."""
    resolved_target = target or AscendTarget("ascend.950pr")
    spaces = resolved_target.get_facts(AscendSpaceFacts)
    body = fn_or_stmt.body if isinstance(fn_or_stmt, PrimFunction) else fn_or_stmt

    explicit_by_scope: dict[str, dict[str, int]] = {}
    pipe_cycles: dict[str, int] = {}
    opaque: list[str] = []
    has_simtvf = [False]
    illegal_dma = [False]
    mte2_run = [0]
    max_mte2 = [0]
    pad_bytes = [0]

    def _record_buf(name: str, ttype: TensorType, op_hint: Op | None = None) -> None:
        if ttype.storage not in (StorageKind.SMEM, StorageKind.TMEM):
            return
        space = _infer_tir_space(op_hint, ttype)
        nbytes, _ = _tir_tensor_bytes(ttype)
        if nbytes > 0:
            scope_map = explicit_by_scope.setdefault(space, {})
            scope_map[name] = max(scope_map.get(name, 0), nbytes)

    def _walk(stmt: Stmt | None, *, repeat: int = 1) -> None:
        if stmt is None:
            return
        if isinstance(stmt, Sequential):
            for child in stmt.body:
                _walk(child, repeat=repeat)
            return
        if isinstance(stmt, LetStmt):
            if isinstance(stmt.var.type, TensorType):
                _record_buf(stmt.var.name, stmt.var.type)
            return
        if isinstance(stmt, While):
            if "while_loop" not in opaque:
                opaque.append("while_loop")
            _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, For):
            trip = _tir_for_static_trip_count(stmt)
            if trip is None:
                tag = f"dynamic_for:{stmt.induction_var.name}"
                if tag not in opaque:
                    opaque.append(tag)
                _walk(stmt.body, repeat=repeat)
            elif trip == 0:
                return
            else:
                _walk(stmt.body, repeat=repeat * trip)
            return
        if isinstance(stmt, If):
            if "conditional_branch" not in opaque:
                opaque.append("conditional_branch")
            _walk(stmt.then_body, repeat=repeat)
            _walk(stmt.else_body, repeat=repeat)
            return
        if isinstance(stmt, ExecutionScope):
            if getattr(stmt.kind, "name", stmt.kind) == SIMTVF_KIND.name:
                has_simtvf[0] = True
                if "simt_vf" not in opaque:
                    opaque.append("simt_vf")
            _walk(stmt.body, repeat=repeat)
            return
        if isinstance(stmt, MeshScope):
            _walk(stmt.body, repeat=repeat)
            return
        if hasattr(stmt, "callable") and isinstance(stmt.callable, Op):
            op = stmt.callable
            arg_types = tuple(arg.type if isinstance(arg.type, TensorType) else None for arg in stmt.args)
            res = getattr(op, "resource", None)
            mask = task_resource_mask(
                _tir_op_to_resources(op, arg_types)
            )
            for core_key, pipe_name in mask.to_core_pipe_pairs():
                key = format_pipe_key(core_key, pipe_name)
                pipe_cycles[key] = pipe_cycles.get(key, 0) + repeat
            if res == "mte2_engine":
                mte2_run[0] += repeat
                max_mte2[0] = max(max_mte2[0], mte2_run[0])
                for ty in arg_types:
                    if isinstance(ty, TensorType) and ty.storage is StorageKind.SMEM:
                        _total, inner_b = _tir_tensor_bytes(ty)
                        rem = inner_b % spaces.ub_align_bytes
                        if rem != 0:
                            pad_bytes[0] += (spaces.ub_align_bytes - rem) * repeat
            else:
                mte2_run[0] = 0
            if isinstance(op, (AscendGmToL1, AscendGmToL1Nz, AscendGmToUb, AscendGmToUbPad)):
                src_s = "GM"
                dst_s = "L1" if isinstance(op, (AscendGmToL1, AscendGmToL1Nz)) else "UB"
                if not spaces.is_dma_path_legal(src_s, dst_s):
                    illegal_dma[0] = True

    def _tir_op_to_resources(op: Op, arg_types: tuple[TensorType | None, ...]) -> tuple[tuple, ...]:
        from tilefoundry_ascend.ascend.schedule.pipeline_solver import _op_resources

        return _op_resources(op, arg_types)

    _walk(body)
    caps = _scope_capacities(spaces, has_simtvf=has_simtvf[0], unreserved_ubuf=unreserved_ubuf)
    waterlines = {
        scope: build_scope_waterline(
            scope=scope,
            capacity_bytes=caps.get(scope, 0),
            explicit_buffers=bufs,
            implicit_buffers={},
        )
        for scope, bufs in sorted(explicit_by_scope.items())
    }
    snaps = tuple(
        BufferSnapshot(name=name, scope=scope, bytes=nbytes, shape=(), dtype="tir")
        for scope, bufs in sorted(explicit_by_scope.items())
        for name, nbytes in sorted(bufs.items())
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=not illegal_dma[0],
        steady_state_ii=None,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        mte2_inflight_per_aiv=max_mte2[0],
        unaligned_dma_padding_bytes=pad_bytes[0],
        opaque_dynamic_regions=tuple(opaque),
    )
    return EnvelopeInspectionResult(
        ledger=ledger,
        explicit_snapshot=snaps,
        lowered_snapshot=snaps,
        merged_peak_bytes_by_scope={s: sum(b.values()) for s, b in explicit_by_scope.items()},
        lowered_passes_observed=(),
        lowering_error=None,
    )


__all__ = [
    "BufferSnapshot",
    "EnvelopeInspectionResult",
    "inspect_tilelang_primfunc",
    "inspect_tir_envelope",
]
