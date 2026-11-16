"""Mode B read-only dynamic envelope accounting and Two-Snapshot IR Diff."""

from __future__ import annotations

from dataclasses import dataclass
import math

from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.hir.pipeline import BufferSchedule, PipelineSpec
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
    format_pipe_key,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

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

    def _walk(stmt, *, repeat: int = 1, mte2_run: list[int] | None = None) -> None:
        if mte2_run is None:
            mte2_run = [0]
        if stmt is None:
            return
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                _walk(child, repeat=repeat, mte2_run=mte2_run)
            return
        if isinstance(stmt, tirx.While):
            if "while_loop" not in opaque_regions:
                opaque_regions.append("while_loop")
            if _contains_buffer_load(stmt.condition, tirx):
                if "data_dependent_branch" not in opaque_regions:
                    opaque_regions.append("data_dependent_branch")
            _walk(stmt.body, repeat=repeat, mte2_run=[0])
            return
        if isinstance(stmt, tirx.For):
            ext = stmt.extent
            trip = int(ext.value) if hasattr(ext, "value") and isinstance(ext.value, int) else None
            if trip is None:
                tag = f"dynamic_for:{stmt.loop_var.name}"
                if tag not in opaque_regions:
                    opaque_regions.append(tag)
                _walk(stmt.body, repeat=repeat, mte2_run=[0])
            else:
                local_mte2 = [0]
                _walk(stmt.body, repeat=repeat * trip, mte2_run=local_mte2)
                if local_mte2[0] > 0:
                    mte2_run[0] += local_mte2[0] * trip
                    max_mte2[0] = max(max_mte2[0], mte2_run[0])
            return
        if isinstance(stmt, tirx.IfThenElse):
            if _contains_buffer_load(stmt.condition, tirx):
                if "data_dependent_branch" not in opaque_regions:
                    opaque_regions.append("data_dependent_branch")
            b1 = [mte2_run[0]]
            _walk(stmt.then_case, repeat=repeat, mte2_run=b1)
            b2 = [mte2_run[0]]
            if stmt.else_case is not None:
                _walk(stmt.else_case, repeat=repeat, mte2_run=b2)
            mte2_run[0] = max(b1[0], b2[0])
            return
        if isinstance(stmt, tirx.AttrStmt):
            key = str(stmt.attr_key)
            if "thread_extent" in key and "cthread" not in key and "blockIdx" not in key:
                has_simtvf[0] = True
                if "simt_vf" not in opaque_regions:
                    opaque_regions.append("simt_vf")
            elif key == "pragma_parallel" or "simt" in key.lower():
                has_simtvf[0] = True
            _walk(stmt.body, repeat=repeat, mte2_run=mte2_run)
            return
        if isinstance(stmt, (tirx.SBlockRealize,)):
            _walk(stmt.block, repeat=repeat, mte2_run=mte2_run)
            return
        if isinstance(stmt, tirx.SBlock):
            _walk(stmt.body, repeat=repeat, mte2_run=mte2_run)
            return
        if isinstance(stmt, (tirx.Bind, tirx.AllocBuffer, tirx.DeclBuffer)):
            if hasattr(stmt, "body"):
                _walk(stmt.body, repeat=repeat, mte2_run=mte2_run)
            return
        if isinstance(stmt, tirx.BufferStore):
            if mte2_run[0] > 0:
                mte2_run[0] = 0
            _add_pipe("AIV0.PIPE_S", repeat)
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
                        _add_pipe("AIV0.PIPE_MTE2", repeat * max(1, (xfer + 63) // 64))
                    elif src_s == "UB" and dst_s == "GM":
                        mte2_run[0] = 0
                        _add_pipe("AIV0.PIPE_MTE3", repeat * max(1, (xfer + 63) // 64))
                    elif src_s == "GM" and dst_s == "L1":
                        _add_pipe("AIC.PIPE_MTE2", repeat * max(1, (xfer + 63) // 64))
                    elif src_s == "L1" and dst_s in ("L0A", "L0B"):
                        _add_pipe("AIC.PIPE_MTE1", repeat * max(1, (xfer + 255) // 256))
                    elif src_s == "L0C" and dst_s in ("GM", "L1", "UB"):
                        fix_rate = max(1, spaces.fixpipe_bytes_per_cycle)
                        _add_pipe("AIC.PIPE_FIX", repeat * max(1, (xfer + fix_rate - 1) // fix_rate))
                        if dst_s == "UB":
                            _add_pipe("AIV0.PIPE_MTE2", repeat * max(1, (xfer + 63) // 64))
                            if aiv_split_factor >= 2:
                                _add_pipe("AIV1.PIPE_MTE2", repeat * max(1, (xfer + 63) // 64))
            elif "gemm" in op_name:
                mte2_run[0] = 0
                _add_pipe("AIC.PIPE_M", repeat * 4)
            elif "simd" in op_name or "vec" in op_name:
                mte2_run[0] = 0
                _add_pipe("AIV0.PIPE_V", repeat)
                if aiv_split_factor >= 2:
                    _add_pipe("AIV1.PIPE_V", repeat)

    _walk(func.body)
    return (
        pipe_cycles,
        max_mte2[0],
        padding_bytes[0],
        tuple(opaque_regions),
        has_simtvf[0],
        illegal_dma[0],
    )


def inspect_tilelang_primfunc(
    prim_func,
    *,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    aiv_split_factor: int = 1,
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

    explicit_snaps = _extract_tilelang_buffers_from_func(
        entry_fn, tirx=tirx, DataTypeCls=TLDataType, prefer_allocate=False
    )
    requested_mb = _requested_multibuffer_names(entry_fn, tirx=tirx)
    pipe_cycles, mte2_inflight, pad_bytes, opaque, has_simtvf, illegal_dma = _scan_tilelang_ast(
        entry_fn,
        spaces,
        tirx=tirx,
        DataTypeCls=TLDataType,
        aiv_split_factor=aiv_split_factor,
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

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=(not illegal_dma) and (lowering_error is None),
        steady_state_ii=None,
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
            ext = static_dim_value(stmt.extent) if hasattr(stmt, "extent") else None
            if ext is None or ext < 1:
                tag = f"dynamic_for:{stmt.induction_var.name}"
                if tag not in opaque:
                    opaque.append(tag)
                _walk(stmt.body, repeat=repeat)
            else:
                _walk(stmt.body, repeat=repeat * ext)
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
