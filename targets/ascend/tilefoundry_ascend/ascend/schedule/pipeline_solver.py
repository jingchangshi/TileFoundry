"""Ascend950PR Z3 modulo pipeline constraint projection and solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path


from tilefoundry.ir.core import Call, Expr, Op, Var
from tilefoundry.ir.core.param_def import MemoryEffect, collect_param_defs
from tilefoundry.ir.hir.pipeline import BufferSchedule, PipelineError, PipelineSpec, TaskIntent
from tilefoundry.ir.tir.memory import AllocTensor, PtrOf, TensorView
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    Stmt,
    While,
)
from tilefoundry.ir.types import StorageKind, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from tilefoundry.ir.types.layout import size as layout_size
from tilefoundry.ir.types.mesh import levels as mesh_levels
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.vendor.tilelang_z3 import (
    SAT,
    Assignment,
    Buffer,
    Capacity,
    Edge,
    Problem,
    Resource,
    Reuse,
    Task,
    lower_bound_ii,
    solve,
    tf_adapter,
    verify_assignment,
)
from tilefoundry_ascend.ascend.ir.execution import SIMDVF_KIND, SIMTVF_KIND
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    AscendDualCopy,
    AscendDualCopyL0cToUb,
    AscendDualCopyUbToL1,
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendUbToGm,
)
from tilefoundry_ascend.ascend.cost_model.pipe_schedule_eval import (
    DIRECTION_SLOPE_MULT,
    MS_PIPE_INTERCEPT_US,
    MS_PIPE_SLOPE_US_PER_BYTE,
    normalize_ns_to_z3_ticks,
    resolve_cube_macs_safe,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask

_CORE_ALIASES: dict[str, str] = {
    "aic": "aic",
    "aic0": "aic",
    "aiv": "aiv0",
    "aiv0": "aiv0",
    "aiv1": "aiv1",
}

_PIPE_TO_LEDGER: dict[str, str] = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "CUBE": "PIPE_M",
    "M": "PIPE_M",
    "VECTOR": "PIPE_V",
    "V": "PIPE_V",
    "FIXPIPE": "PIPE_FIX",
    "FIX": "PIPE_FIX",
    "SCALAR": "PIPE_S",
    "S": "PIPE_S",
}

def read_from_sil(path: str | Path) -> dict | None:
    """Read and parse a SIL packet from a file, returning None if missing or invalid."""
    p = Path(path)
    if not p.is_file():
        logging.warning("SIL packet not found at %s", p)
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to read/parse SIL packet from %s: %s", p, exc)
        return None
    if not isinstance(payload, dict) or not payload.get("roofline"):
        return None
    return payload



def format_pipe_key(core: str, pipe: str) -> str:
    """Return canonical ledger pipe identifier such as 'AIC.PIPE_M' or 'AIV0.PIPE_V'."""
    norm_core = _CORE_ALIASES.get(core.strip().lower(), core.strip().lower()).upper()
    norm_pipe = pipe.strip().upper()
    if norm_pipe.startswith("PIPE_"):
        norm_pipe = norm_pipe[5:]
    ledger_pipe = _PIPE_TO_LEDGER.get(norm_pipe, f"PIPE_{norm_pipe}")
    return f"{norm_core}.{ledger_pipe}"


def task_resource_mask(task_or_resources: Task | Sequence[tuple]) -> NodeResourceMask:
    """Build a 24-bit NodeResourceMask from a Task or solver resource tuple sequence."""
    resources = (
        task_or_resources.resources
        if isinstance(task_or_resources, Task)
        else task_or_resources
    )
    mask = NodeResourceMask(0)
    for use in resources:
        core_raw = str(use[1]).strip().lower()
        pipe_raw = str(use[2]).strip()
        norm_core = _CORE_ALIASES.get(core_raw, core_raw)
        mask = mask | NodeResourceMask.for_core(norm_core, pipe_raw)
    return mask


def masks_conflict(lhs: NodeResourceMask, rhs: NodeResourceMask) -> bool:
    """Return whether two 24-bit core/pipe resource masks overlap."""
    return (lhs.bits & rhs.bits) != 0


def bitwise_conflict_pairs(tasks: Sequence[Task]) -> tuple[tuple[int, int], ...]:
    """Return sorted index pairs (i, j) whose 24-bit resource masks overlap."""
    masks = [task_resource_mask(task) for task in tasks]
    pairs: list[tuple[int, int]] = []
    for i, m_i in enumerate(masks):
        for j in range(i + 1, len(masks)):
            if m_i.conflicts_with(masks[j]):
                pairs.append((i, j))
    return tuple(pairs)


def pipe_cycles_from_problem(
    problem: Problem,
    assignment: Assignment | None = None,
) -> dict[str, int]:
    """Accumulate per-pipe cycle budgets from a Problem's 24-bit task masks."""
    cycles: dict[str, int] = {}
    for task in problem.tasks:
        mask = task_resource_mask(task)
        dur = max(1, int(task.latency))
        for core_key, pipe_name in mask.to_core_pipe_pairs():
            key = format_pipe_key(core_key, pipe_name)
            cycles[key] = cycles.get(key, 0) + dur
    return dict(sorted(cycles.items()))


@dataclass(frozen=True)
class AscendPipelineResult:
    """Solved Ascend950PR modulo pipeline assignment alongside the projected Z3 problem."""

    problem: Problem
    assignment: Assignment
    task_stages: dict[str, int]
    buffer_versions: dict[str, int]
    fallback_used: bool = False
    fallback_reason: str = ""
    applied_sil_policy: dict[str, object] | None = None

    @property
    def task_masks(self) -> dict[str, NodeResourceMask]:
        """Return the 24-bit NodeResourceMask per task name."""
        return {task.name: task_resource_mask(task) for task in self.problem.tasks}


@dataclass(frozen=True)
class _LoopTaskItem:
    label: str
    resources: tuple[tuple, ...]
    calls: tuple[tuple[Op, tuple[Expr, ...]], ...]


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _eval_dim_with_hints(
    dim: object,
    *,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> int | None:
    """Evaluate a static or symbolic dimension using optional regime dim hints."""
    static = static_dim_value(dim)
    if static is not None:
        return static
    if dim_hints is None and not allow_dynamic_fallback:
        return None
    if isinstance(dim, DimVar):
        if dim_hints and dim.name in dim_hints:
            return int(dim_hints[dim.name])
        if allow_dynamic_fallback:
            return max(1, int(dim.hi))
        return None
    if isinstance(dim, Var):
        if dim_hints and dim.name in dim_hints:
            return int(dim_hints[dim.name])
        if allow_dynamic_fallback:
            return 1
        return None
    if isinstance(dim, Call):
        target = dim.target
        if isinstance(target, DimConst):
            return int(target.value)
        if isinstance(target, DimVar):
            if dim_hints and target.name in dim_hints:
                return int(dim_hints[target.name])
            if allow_dynamic_fallback:
                return max(1, int(target.hi))
            return None
        if len(dim.args) == 2:
            lhs = _eval_dim_with_hints(
                dim.args[0], dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
            )
            rhs = _eval_dim_with_hints(
                dim.args[1], dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
            )
            if lhs is None or rhs is None:
                return None
            if isinstance(target, DimAdd):
                return lhs + rhs
            if isinstance(target, DimSub):
                return lhs - rhs
            if isinstance(target, DimMul):
                return lhs * rhs
            if isinstance(target, DimFloorDiv):
                return lhs // rhs if rhs != 0 else None
            if isinstance(target, DimMod):
                return lhs % rhs if rhs != 0 else None
            if isinstance(target, DimMin):
                return min(lhs, rhs)
            if isinstance(target, DimMax):
                return max(lhs, rhs)
    return None


def _tensor_byte_size(
    tensor_type: TensorType,
    space: str,
    facts: AscendSpaceFacts,
    *,
    aiv_factor: int = 1,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> int:
    total = 1
    for dim in tensor_type.shape:
        extent = _eval_dim_with_hints(
            dim, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
        )
        if extent is None or extent < 1:
            raise PipelineError(
                "ASCEND_DYNAMIC_PIPELINE_BUFFER",
                f"pipeline buffer shape {tensor_type.shape} requires static positive extents",
            )
        total *= extent
    raw_bytes = (total * tensor_type.dtype.bit_width + 7) // 8
    if space == "UB" and aiv_factor > 1:
        raw_bytes = (raw_bytes + aiv_factor - 1) // aiv_factor
    align = {
        "UB": facts.ub_align_bytes,
        "L1": facts.l1_align_bytes,
        "L0A": facts.l0a_align_bytes,
        "L0B": facts.l0b_align_bytes,
        "L0C": facts.l0c_align_bytes,
    }[space]
    return _align_up(raw_bytes, align)


def _static_tensor_elems(
    tensor_type: TensorType | None,
    *,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> int:
    if tensor_type is None:
        return 0
    total = 1
    for dim in tensor_type.shape:
        extent = _eval_dim_with_hints(
            dim, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
        )
        if extent is None or extent < 1:
            return 0
        total *= extent
    return total


def _static_tensor_bytes(
    tensor_type: TensorType | None,
    *,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> int:
    elems = _static_tensor_elems(
        tensor_type, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
    )
    if elems <= 0 or tensor_type is None:
        return 0
    return (elems * tensor_type.dtype.bit_width + 7) // 8


_DEFAULT_SOLVER_FACTS = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)


def _estimate_task_latency_ns(
    item: _LoopTaskItem,
    *,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> float:
    """Estimate continuous Ascend 950 nanosecond latency across 6-pipe + AVE op classes."""
    clock_ghz = float(getattr(_DEFAULT_SOLVER_FACTS, "clock_ghz", 1.65) or 1.65)
    if item.label in ("simdvf", "simtvf"):
        total_ns = MS_PIPE_INTERCEPT_US["VECTOR"] * 1000.0
        for op, args in item.calls:
            op_bytes = max(
                (
                    _static_tensor_bytes(
                        arg.type, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
                    )
                    for arg in args
                    if isinstance(arg.type, TensorType)
                ),
                default=0,
            )
            op_elems = max(
                (
                    _static_tensor_elems(
                        arg.type, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
                    )
                    for arg in args
                    if isinstance(arg.type, TensorType)
                ),
                default=0,
            )
            op_low = type(op).__name__.lower()
            is_f16 = any(
                isinstance(arg.type, TensorType) and arg.type.dtype.bit_width <= 16
                for arg in args
            )
            if any(tok in op_low for tok in ("exp", "log", "sqrt", "rsqrt", "div")):
                issue_w = 4.0 if is_f16 else 2.0
            elif "reduce" in op_low:
                issue_w = 3.0
            else:
                issue_w = 0.5
            beats = max(1.0, float(op_bytes) / 256.0 if op_bytes > 0 else float(op_elems) / 128.0)
            total_ns += (beats * issue_w) / max(0.1, clock_ghz)
        return max(26.0, total_ns)

    total_ns = 0.0
    for op, args in item.calls:
        arg_types = [arg.type if isinstance(arg.type, TensorType) else None for arg in args]
        if isinstance(op, AscendMmad) and len(arg_types) >= 3:
            lhs, rhs = arg_types[1], arg_types[2]
            if lhs is not None and rhs is not None and len(lhs.shape) >= 2 and len(rhs.shape) >= 2:
                m = _eval_dim_with_hints(
                    lhs.shape[-2] if op.a_layout == "MK" else lhs.shape[-1],
                    dim_hints=dim_hints,
                    allow_dynamic_fallback=allow_dynamic_fallback,
                )
                k = _eval_dim_with_hints(
                    lhs.shape[-1] if op.a_layout == "MK" else lhs.shape[-2],
                    dim_hints=dim_hints,
                    allow_dynamic_fallback=allow_dynamic_fallback,
                )
                n = _eval_dim_with_hints(
                    rhs.shape[-2] if op.b_layout == "NK" else rhs.shape[-1],
                    dim_hints=dim_hints,
                    allow_dynamic_fallback=allow_dynamic_fallback,
                )
                if m and k and n:
                    macs = float(resolve_cube_macs_safe(_DEFAULT_SOLVER_FACTS, lhs.dtype))
                    flops = float(2 * m * n * k)
                    compute_ns = flops / max(1e-6, macs * 2.0 * clock_ghz)
                    total_ns += compute_ns + MS_PIPE_INTERCEPT_US["CUBE"] * 1000.0
                    continue
        if isinstance(
            op,
            (
                AscendGmToUb,
                AscendUbToGm,
                AscendGmToL1,
                AscendL1ToL0A,
                AscendL1ToL0B,
                AscendL0cToGm,
                AscendDualCopy,
                AscendDualCopyL0cToUb,
                AscendDualCopyUbToL1,
            ),
        ):
            xfer_bytes = float(
                max(
                    (
                        _static_tensor_bytes(
                            t, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
                        )
                        for t in arg_types
                    ),
                    default=0,
                )
            )
            if isinstance(op, (AscendL1ToL0A, AscendL1ToL0B)):
                pipe, mult = "MTE1", 1.0
            elif isinstance(op, (AscendGmToL1, AscendGmToUb)):
                pipe, mult = "MTE2", 1.0
            elif isinstance(op, AscendDualCopyUbToL1):
                pipe, mult = "MTE3", DIRECTION_SLOPE_MULT["ub_to_l1"]
            elif isinstance(op, AscendUbToGm):
                pipe, mult = "MTE3", 1.0
            elif isinstance(op, AscendDualCopyL0cToUb):
                pipe, mult = "FIXPIPE", DIRECTION_SLOPE_MULT["l0c_to_ub"]
            else:
                pipe, mult = "FIXPIPE", 1.0
            total_ns += (
                xfer_bytes * MS_PIPE_SLOPE_US_PER_BYTE[pipe] * mult + MS_PIPE_INTERCEPT_US[pipe]
            ) * 1000.0
            continue
        elems = max(
            (
                _static_tensor_elems(
                    t, dim_hints=dim_hints, allow_dynamic_fallback=allow_dynamic_fallback
                )
                for t in arg_types
            ),
            default=0,
        )
        total_ns += (
            (max(1.0, float(elems) / 128.0) / max(0.1, clock_ghz))
            + MS_PIPE_INTERCEPT_US["VECTOR"] * 1000.0
        )
    return max(20.0, total_ns)


def _estimate_task_latency(
    item: _LoopTaskItem,
    *,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> int:
    """Estimate normalized hardware cycle quanta (1..12) from continuous Ascend 950 latencies."""
    ns = _estimate_task_latency_ns(
        item,
        dim_hints=dim_hints,
        allow_dynamic_fallback=allow_dynamic_fallback,
    )
    return min(12, max(1, int(round(ns / 400.0))))


def _op_resource_mask(
    op: Op,
    arg_types: tuple[TensorType | None, ...],
    *,
    aiv_core: str = "aiv0",
) -> NodeResourceMask:
    """Project an Ascend TIR Op onto a 24-bit AIV1|AIC|AIV0 NodeResourceMask."""
    resource = getattr(op, "resource", None)
    if resource == "mte2_engine":
        dst = arg_types[-1] if arg_types else None
        if dst is not None and dst.storage is StorageKind.TMEM:
            return NodeResourceMask.for_aic("MTE2")
        return NodeResourceMask.for_core(aiv_core, "MTE2")
    if resource == "mte3_engine":
        return NodeResourceMask.for_core(aiv_core, "MTE3")
    if resource == "mte1_engine":
        return NodeResourceMask.for_aic("MTE1")
    if resource == "cube_engine":
        return NodeResourceMask.for_aic("CUBE")
    if resource == "fixpipe_engine":
        return NodeResourceMask.for_aic("FIXPIPE")
    if resource == "dual_copy_engine":
        if isinstance(op, AscendDualCopyUbToL1):
            return NodeResourceMask.for_aic("MTE1") | NodeResourceMask.for_dual_aiv("MTE3")
        if isinstance(op, AscendDualCopyL0cToUb):
            return NodeResourceMask.for_aic("FIXPIPE")
        return NodeResourceMask.for_aic("FIXPIPE") | NodeResourceMask.for_dual_aiv("MTE2")
    if resource == "vector_engine":
        return NodeResourceMask.for_core(aiv_core, "VECTOR")
    if len(arg_types) >= 2 and all(ty is not None for ty in arg_types[:2]):
        src_s, dst_s = arg_types[0].storage, arg_types[1].storage
        if src_s is StorageKind.GMEM and dst_s is StorageKind.SMEM:
            return NodeResourceMask.for_core(aiv_core, "MTE2")
        if src_s is StorageKind.SMEM and dst_s is StorageKind.GMEM:
            return NodeResourceMask.for_core(aiv_core, "MTE3")
        if src_s is StorageKind.GMEM and dst_s is StorageKind.TMEM:
            return NodeResourceMask.for_aic("MTE2")
        if src_s is StorageKind.TMEM and dst_s is StorageKind.GMEM:
            return NodeResourceMask.for_aic("FIXPIPE")
    return NodeResourceMask.for_core(aiv_core, "VECTOR")


def _op_resources(
    op: Op,
    arg_types: tuple[TensorType | None, ...],
    *,
    aiv_core: str = "aiv0",
) -> tuple[tuple, ...]:
    return _op_resource_mask(op, arg_types, aiv_core=aiv_core).to_solver_resources(1, 0)


def _infer_tmem_space(roles: set[str]) -> str:
    if "l0a" in roles:
        return "L0A"
    if "l0b" in roles:
        return "L0B"
    if "l0c" in roles:
        return "L0C"
    return "L1"


def _space_owner(space: str) -> str:
    return "aiv" if space == "UB" else "aic"


def _capacities_from_facts(
    facts: AscendSpaceFacts,
    *,
    has_simtvf: bool = False,
    unreserved_ubuf: bool = False,
) -> tuple[Capacity, ...]:
    if has_simtvf:
        ub_bytes = facts.ub_limit_bytes("simtvf")
    else:
        ub_bytes = facts.ub_limit_bytes("simdvf", unreserved_ubuf=unreserved_ubuf)
    return (
        Capacity("UB", "aiv", ub_bytes),
        Capacity("L1", "aic", facts.l1_bytes),
        Capacity("L0A", "aic", facts.l0a_bytes),
        Capacity("L0B", "aic", facts.l0b_bytes),
        Capacity("L0C", "aic", facts.l0c_bytes),
    )


def _contains_simtvf(stmt: Stmt | None) -> bool:
    """Return True if ``stmt`` contains an ``ascend.simtvf`` execution scope."""
    if stmt is None:
        return False
    if isinstance(stmt, ExecutionScope):
        if getattr(stmt.kind, "name", stmt.kind) == SIMTVF_KIND.name:
            return True
        return _contains_simtvf(stmt.body)
    if isinstance(stmt, Sequential):
        return any(_contains_simtvf(child) for child in stmt.body)
    if isinstance(stmt, (MeshScope, For, While)):
        return _contains_simtvf(stmt.body)
    if isinstance(stmt, If):
        return _contains_simtvf(stmt.then_body) or _contains_simtvf(stmt.else_body)
    return False


def _resources_for_tasks(tasks: list[Task]) -> tuple[Resource, ...]:
    seen = {
        (owner, core, pipe)
        for task in tasks
        for owner, core, pipe, _dur, _off in task.resources
    }
    return tuple(Resource(owner, core, pipe, 1) for owner, core, pipe in sorted(seen))


def _resolve_root(expr: Expr, aliases: dict[int, tuple[str, TensorType]], env: dict[int, Expr]):
    seen: set[int] = set()
    cur: object = expr
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if id(cur) in aliases:
            return aliases[id(cur)]
        if isinstance(cur, Var):
            if id(cur) in env:
                cur = env[id(cur)]
                continue
            if isinstance(cur.type, TensorType):
                return cur.name, cur.type
            return None
        if isinstance(cur, Call):
            if isinstance(cur.target, (PtrOf, TensorView)) and cur.args:
                if isinstance(cur.args[0], Var) or isinstance(cur.args[0], Call):
                    cur = cur.args[0]
                    continue
            if isinstance(cur.type, TensorType):
                if cur.args and isinstance(cur.target, AllocTensor):
                    return None
                return f"buf_{id(cur)}", cur.type
            return None
        return None
    return None


def _collect_raw_calls(
    stmt: Stmt | None,
    env: dict[int, Expr],
    out: list[tuple[Op, tuple[Expr, ...]]],
) -> None:
    if stmt is None:
        return
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _collect_raw_calls(child, env, out)
        return
    if isinstance(stmt, LetStmt):
        env[id(stmt.var)] = stmt.value
        return
    if isinstance(stmt, (MeshScope, ExecutionScope)):
        _collect_raw_calls(stmt.body, env, out)
        return
    if isinstance(stmt, If):
        _collect_raw_calls(stmt.then_body, env, out)
        _collect_raw_calls(stmt.else_body, env, out)
        return
    if isinstance(stmt, (For, While)):
        _collect_raw_calls(stmt.body, env, out)
        return
    if isinstance(stmt, Evaluate):
        if isinstance(stmt.callable, Op):
            out.append((stmt.callable, stmt.args))


def _scope_aiv_override(kind: object, current: str) -> str:
    name = getattr(kind, "name", kind)
    if name in ("ascend.aiv0", "aiv0"):
        return "aiv0"
    if name in ("ascend.aiv1", "aiv1"):
        return "aiv1"
    return current


def _collect_stmts(
    stmt: Stmt | None,
    env: dict[int, Expr],
    out: list[_LoopTaskItem],
    *,
    aiv_core: str = "aiv0",
) -> None:
    if stmt is None:
        return
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _collect_stmts(child, env, out, aiv_core=aiv_core)
        return
    if isinstance(stmt, LetStmt):
        env[id(stmt.var)] = stmt.value
        return
    if isinstance(stmt, ExecutionScope) and stmt.kind in (SIMDVF_KIND.name, SIMTVF_KIND.name):
        inner: list[tuple[Op, tuple[Expr, ...]]] = []
        _collect_raw_calls(stmt.body, env, inner)
        if inner:
            vf_label = "simdvf" if stmt.kind == SIMDVF_KIND.name else "simtvf"
            vf_mask = NodeResourceMask.for_core(aiv_core, "VECTOR")
            out.append(
                _LoopTaskItem(
                    label=vf_label,
                    resources=vf_mask.to_solver_resources(1, 0),
                    calls=tuple(inner),
                )
            )
        return
    if isinstance(stmt, ExecutionScope):
        next_core = _scope_aiv_override(stmt.kind, aiv_core)
        _collect_stmts(stmt.body, env, out, aiv_core=next_core)
        return
    if isinstance(stmt, MeshScope):
        _collect_stmts(stmt.body, env, out, aiv_core=aiv_core)
        return
    if isinstance(stmt, If):
        _collect_stmts(stmt.then_body, env, out, aiv_core=aiv_core)
        _collect_stmts(stmt.else_body, env, out, aiv_core=aiv_core)
        return
    if isinstance(stmt, (For, While)):
        _collect_stmts(stmt.body, env, out, aiv_core=aiv_core)
        return
    if isinstance(stmt, Evaluate):
        if isinstance(stmt.callable, Op):
            op = stmt.callable
            schema_name = getattr(type(op), "_op_schema", None)
            op_label = schema_name.name if schema_name is not None else type(op).__name__
            arg_types = tuple(
                arg.type if isinstance(arg.type, TensorType) else None for arg in stmt.args
            )
            res_uses = _op_resources(op, arg_types, aiv_core=aiv_core)
            out.append(
                _LoopTaskItem(
                    label=op_label,
                    resources=res_uses,
                    calls=((op, stmt.args),),
                )
            )


def _collect_alloc_aliases(
    stmt: Stmt | None,
    aliases: dict[int, tuple[str, TensorType]],
    env: dict[int, Expr],
    *,
    skip_stmt: Stmt | None = None,
    names_out: set[str] | None = None,
) -> None:
    if stmt is None or stmt is skip_stmt:
        return
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _collect_alloc_aliases(child, aliases, env, skip_stmt=skip_stmt, names_out=names_out)
        return
    if isinstance(stmt, LetStmt):
        env[id(stmt.var)] = stmt.value
        if isinstance(stmt.var.type, TensorType) and stmt.var.type.storage in (
            StorageKind.SMEM,
            StorageKind.TMEM,
        ):
            aliases[id(stmt.var)] = (stmt.var.name, stmt.var.type)
            if names_out is not None:
                names_out.add(stmt.var.name)
        elif isinstance(stmt.var.type, TupleType):
            if stmt.var.type.fields and all(
                isinstance(f, TensorType) for f in stmt.var.type.fields
            ):
                aliases[id(stmt.var)] = (stmt.var.name, stmt.var.type.fields[0])
                if names_out is not None:
                    names_out.add(stmt.var.name)
        return
    if isinstance(stmt, (MeshScope, ExecutionScope, For, While)):
        _collect_alloc_aliases(stmt.body, aliases, env, skip_stmt=skip_stmt, names_out=names_out)
        return
    if isinstance(stmt, If):
        _collect_alloc_aliases(stmt.then_body, aliases, env, skip_stmt=skip_stmt, names_out=names_out)
        _collect_alloc_aliases(stmt.else_body, aliases, env, skip_stmt=skip_stmt, names_out=names_out)


def _mesh_aiv_extent(mesh) -> int:
    for topo, arr in zip(mesh.topologies, mesh_levels(mesh)):
        name = getattr(topo, "name", topo)
        if name == "aiv":
            ext = layout_size(arr)
            if isinstance(ext, int) and ext > 1:
                return ext
    return 1


def _max_aiv_in_stmt(stmt: Stmt | None) -> int:
    if stmt is None:
        return 1
    if isinstance(stmt, Sequential):
        return max((_max_aiv_in_stmt(c) for c in stmt.body), default=1)
    if isinstance(stmt, MeshScope):
        return max(_mesh_aiv_extent(stmt.mesh), _max_aiv_in_stmt(stmt.body))
    if isinstance(stmt, (ExecutionScope, For, While)):
        return _max_aiv_in_stmt(stmt.body)
    if isinstance(stmt, If):
        return max(_max_aiv_in_stmt(stmt.then_body), _max_aiv_in_stmt(stmt.else_body))
    return 1


def _active_aiv_factor(fn_stmt: Stmt | None, loop: For) -> int:
    """Return the active AIV split factor enclosing or inside ``loop``."""
    factor = 1

    def _walk(stmt: Stmt | None, cur_aiv: int) -> bool:
        nonlocal factor
        if stmt is None:
            return False
        if stmt is loop:
            factor = max(factor, cur_aiv, _max_aiv_in_stmt(loop.body))
            return True
        if isinstance(stmt, Sequential):
            return any(_walk(child, cur_aiv) for child in stmt.body)
        if isinstance(stmt, MeshScope):
            mesh_aiv = _mesh_aiv_extent(stmt.mesh)
            next_aiv = mesh_aiv if mesh_aiv > 1 else cur_aiv
            return _walk(stmt.body, next_aiv)
        if isinstance(stmt, (ExecutionScope, For, While)):
            return _walk(stmt.body, cur_aiv)
        if isinstance(stmt, If):
            return _walk(stmt.then_body, cur_aiv) or _walk(stmt.else_body, cur_aiv)
        return False

    if fn_stmt is not None and _walk(fn_stmt, 1):
        return factor
    return _max_aiv_in_stmt(loop.body)


def _recurrence_cycle_pairs(names: list[str], edges: list[Edge]) -> set[tuple[str, str]]:
    """Return ``(u, v)`` pairs sharing a loop-carried recurrence cycle."""
    fwd: dict[str, set[str]] = {n: set() for n in names}
    rev: dict[str, set[str]] = {n: set() for n in names}
    for e in edges:
        if e.distance == 0:
            fwd.setdefault(e.src, set()).add(e.dst)
            rev.setdefault(e.dst, set()).add(e.src)

    def _reach(start: str, adj: dict[str, set[str]]) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    same_cycle: set[tuple[str, str]] = set()
    for e in edges:
        if e.distance < 1:
            continue
        from_dst = _reach(e.dst, fwd)
        to_src = _reach(e.src, rev)
        cycle_nodes = from_dst & to_src
        for u in cycle_nodes:
            for v in cycle_nodes:
                same_cycle.add((u, v))
    return same_cycle


def _assign_dag_stages(
    names: list[str],
    edges: list[Edge],
    authored_stages: dict[str, int | None],
    max_stage: int,
) -> dict[str, int]:
    stages: dict[str, int] = {}
    for name in names:
        authored = authored_stages.get(name)
        stages[name] = authored if authored is not None else 0
    if any(stage is not None for stage in authored_stages.values()):
        return stages
    cycle_pairs = _recurrence_cycle_pairs(names, edges)
    for _ in names:
        updated = False
        for edge in edges:
            if edge.distance == 0:
                step = 0 if (edge.src, edge.dst) in cycle_pairs else 1
                candidate = min(max_stage, stages[edge.src] + step)
                if candidate > stages[edge.dst]:
                    stages[edge.dst] = candidate
                    updated = True
        if not updated:
            break
    return stages


def validate_pipeline_spec(spec: PipelineSpec, target: AscendTarget | None = None) -> None:
    """Validate declared pipeline bounds and buffer schedule entries for Ascend950PR."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    if spec.fixed_ii is not None and spec.fixed_ii < 1:
        raise PipelineError("PIPELINE_II_INVALID", f"fixed_ii must be >= 1, got {spec.fixed_ii}")
    if spec.max_buffer_versions < 1:
        raise PipelineError(
            "PIPELINE_BUFFER_VERSIONS_INVALID",
            f"max_buffer_versions must be >= 1, got {spec.max_buffer_versions}",
        )
    for intent in spec.task_intents:
        if intent.stage is not None and intent.stage > spec.max_stage:
            raise PipelineError(
                "AUTHORED_STAGE_EXCEEDS_MAX",
                f"task {intent.task_id!r} stage {intent.stage} exceeds max_stage {spec.max_stage}",
            )
    for entry in spec.buffer_schedules:
        if entry.space:
            limit = facts.limit_bytes(entry.space)
            if limit is None:
                raise PipelineError(
                    "ASCEND_UNKNOWN_BUFFER_SPACE",
                    f"buffer {entry.root!r} references unsupported space {entry.space!r}",
                )


def extract_tir_loop_problem(
    loop: For,
    *,
    target: AscendTarget | None = None,
    fn_stmt: Stmt | None = None,
    unreserved_ubuf: bool = False,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> tuple[Problem, dict[str, int]]:
    """Build a Z3 pipeline Problem and initial stage map from a TIR For loop."""
    spec = loop.pipeline
    if spec is None:
        raise PipelineError("PIPELINE_MISSING", "loop has no PipelineSpec declaration")
    resolved_target = target or AscendTarget("ascend.950pr")
    validate_pipeline_spec(spec, resolved_target)
    facts = resolved_target.get_facts(AscendSpaceFacts)
    policy_dict = _coerce_sil_policy(sil_policy)

    aliases: dict[int, tuple[str, TensorType]] = {}
    outer_buffers: set[str] = set()
    env: dict[int, Expr] = {}
    if fn_stmt is not None:
        _collect_alloc_aliases(fn_stmt, aliases, env, skip_stmt=loop, names_out=outer_buffers)
    _collect_alloc_aliases(loop.body, aliases, env)
    aiv_factor = _active_aiv_factor(fn_stmt, loop)

    task_items: list[_LoopTaskItem] = []
    _collect_stmts(loop.body, env, task_items)

    authored_by_id = {intent.task_id: intent for intent in spec.task_intents}
    authored_by_anchor = {intent.anchor: intent for intent in spec.task_intents}
    sched_by_root = {item.root: item for item in spec.buffer_schedules}

    buffer_types: dict[str, TensorType] = {}
    buffer_roles: dict[str, set[str]] = {}
    last_writer: dict[str, str] = {}
    readers_since_write: dict[str, list[str]] = {}
    all_readers: dict[str, list[str]] = {}
    first_readers_before_write: dict[str, list[str]] = {}
    task_latency_by_name: dict[str, int] = {}
    task_names: list[str] = []
    task_specs: list[tuple[str, tuple, int, int | None]] = []
    edges: list[Edge] = []
    reuses: list[Reuse] = []
    edge_keys: set[tuple[str, str, int, int]] = set()
    reuse_keys: set[tuple[str, str, str]] = set()

    raw_ns = [
        _estimate_task_latency_ns(
            item,
            dim_hints=dim_hints,
            allow_dynamic_fallback=allow_dynamic_fallback,
        )
        for item in task_items
    ]
    norm_latencies = normalize_ns_to_z3_ticks(raw_ns, max_ticks=12)

    for idx, item in enumerate(task_items):
        t_name = f"t{idx}_{item.label}"
        intent = authored_by_anchor.get(idx) or authored_by_id.get(t_name)
        res_uses = item.resources
        latency = (
            intent.cost_override
            if (intent and intent.cost_override is not None)
            else norm_latencies[idx]
        )
        authored_stage = intent.stage if intent else None
        task_latency_by_name[t_name] = latency
        task_names.append(t_name)
        task_specs.append((t_name, res_uses, latency, authored_stage))

        for op, args in item.calls:
            params = [p for p in collect_param_defs(type(op)) if p.kind == "input"]
            for p_idx, arg in enumerate(args):
                if p_idx >= len(params):
                    break
                param = params[p_idx]
                resolved = _resolve_root(arg, aliases, env)
                if resolved is None:
                    continue
                buf_name, buf_type = resolved
                if buf_type.storage not in (StorageKind.SMEM, StorageKind.TMEM):
                    continue
                buffer_types.setdefault(buf_name, buf_type)
                roles = buffer_roles.setdefault(buf_name, set())
                if isinstance(op, AscendL1ToL0A) and param.name == "dst":
                    roles.add("l0a")
                elif isinstance(op, AscendL1ToL0B) and param.name == "dst":
                    roles.add("l0b")
                elif isinstance(op, AscendMmad):
                    if param.name == "lhs":
                        roles.add("l0a")
                    elif param.name == "rhs":
                        roles.add("l0b")
                    elif param.name == "acc":
                        roles.add("l0c")
                elif isinstance(
                    op,
                    (AscendL0cToGm, AscendDualCopy, AscendDualCopyL0cToUb),
                ) and param.name == "src":
                    if buf_type.storage is StorageKind.TMEM:
                        roles.add("l0c")

                is_read = bool(param.effect and (param.effect & MemoryEffect.READ))
                is_write = bool(param.effect and (param.effect & MemoryEffect.WRITE))

                if is_read:
                    if buf_name in last_writer:
                        writer = last_writer[buf_name]
                        if writer != t_name:
                            w_lat = task_latency_by_name[writer]
                            e_key = (writer, t_name, 0, w_lat)
                            if e_key not in edge_keys:
                                edge_keys.add(e_key)
                                edges.append(
                                    Edge(src=writer, dst=t_name, distance=0, latency=w_lat)
                                )
                            all_readers.setdefault(buf_name, []).append(t_name)
                            readers_since_write.setdefault(buf_name, []).append(t_name)
                    elif buf_name in outer_buffers:
                        first_readers = first_readers_before_write.setdefault(buf_name, [])
                        if t_name not in first_readers:
                            first_readers.append(t_name)

                if is_write:
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

    if spec.enable_offset and spec.max_stage > 0:
        authored_map = {name: stage for name, _r, _lat, stage in task_specs}
        assigned_stages = _assign_dag_stages(task_names, edges, authored_map, spec.max_stage)
        manual = True
        max_stage = spec.max_stage
    else:
        assigned_stages = {name: 0 for name in task_names}
        manual = False
        max_stage = 0

    tasks = [
        Task(
            name=name,
            resources=res_uses,
            latency=latency,
            stage=assigned_stages[name] if manual else None,
        )
        for name, res_uses, latency, _st in task_specs
    ]

    reused_buffers = {r.buffer for r in reuses}
    buffers: list[Buffer] = []
    for buf_name, buf_type in buffer_types.items():
        sched = sched_by_root.get(buf_name)
        if sched is not None and sched.space:
            space = sched.space
        elif buf_type.storage is StorageKind.SMEM:
            space = "UB"
        else:
            space = _infer_tmem_space(buffer_roles.get(buf_name, set()))
        owner = (sched.owner if sched is not None and sched.owner else _space_owner(space))
        slot_bytes = _tensor_byte_size(
            buf_type,
            space,
            facts,
            aiv_factor=aiv_factor,
            dim_hints=dim_hints,
            allow_dynamic_fallback=allow_dynamic_fallback,
        )
        if sched is not None:
            if sched.versions[0] == "fixed":
                buffers.append(
                    Buffer(
                        name=buf_name,
                        slot_bytes=slot_bytes,
                        space=space,
                        owner=owner,
                        fixed=sched.versions[1],
                    )
                )
            else:
                lo, hi = sched.versions[1], sched.versions[2]
                if spec.max_buffer_versions > 0:
                    hi = min(hi, spec.max_buffer_versions)
                buffers.append(
                    Buffer(
                        name=buf_name,
                        slot_bytes=slot_bytes,
                        space=space,
                        owner=owner,
                        auto_min=lo,
                        auto_max=max(lo, hi),
                    )
                )
        elif buf_name in loop_carried_buffers:
            buffers.append(
                Buffer(
                    name=buf_name,
                    slot_bytes=slot_bytes,
                    space=space,
                    owner=owner,
                    fixed=1,
                )
            )
        elif manual and spec.max_buffer_versions > 1 and buf_name in reused_buffers:
            buffers.append(
                Buffer(
                    name=buf_name,
                    slot_bytes=slot_bytes,
                    space=space,
                    owner=owner,
                    auto_min=1,
                    auto_max=spec.max_buffer_versions,
                )
            )
        else:
            buffers.append(
                Buffer(
                    name=buf_name,
                    slot_bytes=slot_bytes,
                    space=space,
                    owner=owner,
                    fixed=1,
                )
            )

    issue_after = []
    for intent in spec.task_intents:
        for after_id in intent.issue_after:
            if after_id in authored_stages_set(task_names) and intent.task_id in authored_stages_set(
                task_names
            ):
                issue_after.append((after_id, intent.task_id))

    has_simtvf = _contains_simtvf(fn_stmt if fn_stmt is not None else loop.body)
    policy_ub_unlock = bool(policy_dict.get("ub_256kb_unlock", False)) if policy_dict else False
    unreserved = unreserved_ubuf or spec.origin == "simdvf_unreserved" or policy_ub_unlock
    problem = tf_adapter(
        {
            "tasks": tuple(tasks),
            "edges": tuple(edges),
            "buffers": tuple(buffers),
            "reuses": tuple(reuses),
            "resources": _resources_for_tasks(tasks),
            "capacities": _capacities_from_facts(
                facts,
                has_simtvf=has_simtvf,
                unreserved_ubuf=unreserved,
            ),
            "max_stage": max_stage,
            "manual": manual,
            "issue_after": tuple(issue_after),
        }
    )
    if policy_dict is not None:
        problem = _apply_sil_policy_to_problem(problem, policy_dict)
    if shadow_dump_path is not None:
        _write_shadow_dump(
            shadow_dump_path,
            {
                "kind": "extracted_problem",
                "tasks": [
                    {
                        "name": t.name,
                        "latency": t.latency,
                        "stage": t.stage,
                        "resources": [list(u) for u in t.resources],
                    }
                    for t in problem.tasks
                ],
                "buffers": [
                    {
                        "name": b.name,
                        "space": b.space,
                        "owner": b.owner,
                        "slot_bytes": b.slot_bytes,
                        "fixed": b.fixed,
                        "auto_min": b.auto_min,
                        "auto_max": b.auto_max,
                    }
                    for b in problem.buffers
                ],
                "capacities": [
                    {"space": c.space, "owner": c.owner, "bytes": c.bytes}
                    for c in problem.capacities
                ],
                "sil_policy": policy_dict,
            },
        )
    return problem, assigned_stages


def authored_stages_set(names: list[str]) -> set[str]:
    return set(names)


def _verify_bitwise_mask_schedule(problem: Problem, assignment: Assignment) -> list[str]:
    """Verify no two tasks with overlapping 24-bit resource masks share a modulo slot."""
    problems: list[str] = []
    ii = assignment.ii
    if ii < 1:
        return [f"INVALID_II {ii}"]
    for i, j in bitwise_conflict_pairs(problem.tasks):
        t_i = problem.tasks[i]
        t_j = problem.tasks[j]
        s_i = assignment.starts.get(t_i.name)
        s_j = assignment.starts.get(t_j.name)
        if s_i is None or s_j is None:
            continue
        if (s_i - s_j) % ii == 0:
            problems.append(f"BITWISE_MASK_OVERLAP {t_i.name}/{t_j.name} at II={ii}")
    return problems


def _coerce_sil_policy(sil_policy: Mapping[str, object] | object | None) -> dict[str, object] | None:
    """Normalize a SIL SchedulePolicy object or Mapping into a plain dictionary."""
    if sil_policy is None:
        return None
    if isinstance(sil_policy, Mapping):
        return dict(sil_policy)
    to_dict = getattr(sil_policy, "to_dict", None)
    if callable(to_dict):
        res = to_dict()
        if isinstance(res, Mapping):
            return dict(res)
    keys = (
        "regime_tag",
        "topology_axes",
        "tile_mnk",
        "l1_stages",
        "l1_half_bank_split",
        "l1_slot_aliasing",
        "l0c_buffers",
        "ub_256kb_unlock",
        "vector_mode",
    )
    out: dict[str, object] = {}
    for key in keys:
        if hasattr(sil_policy, key):
            out[key] = getattr(sil_policy, key)
    return out


def _write_shadow_dump(shadow_dump_path: str | Path | None, payload: dict[str, object]) -> None:
    if shadow_dump_path is None:
        return
    path = Path(shadow_dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_aliased_l1_name(query: str, l1_names: set[str]) -> str | None:
    if query in l1_names:
        return query
    q_lower = query.lower()
    exact_ci = [n for n in l1_names if n.lower() == q_lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    suffix = [n for n in l1_names if n.lower().endswith(f"_{q_lower}") or q_lower in n.lower()]
    if len(suffix) == 1:
        return suffix[0]
    return None


def _apply_sil_policy_to_problem(problem: Problem, policy: Mapping[str, object]) -> Problem:
    """Project SIL policy overrides (L1 stages, L0C buffers, 256KB UB, L1 aliasing) onto Problem."""
    l1_stages = policy.get("l1_stages")
    l0c_buffers = policy.get("l0c_buffers")
    ub_256kb_unlock = bool(policy.get("ub_256kb_unlock", False))
    raw_aliasing = policy.get("l1_slot_aliasing") or {}

    if l1_stages is not None:
        if not isinstance(l1_stages, int) or isinstance(l1_stages, bool) or l1_stages < 1:
            raise PipelineError("ASCEND_SIL_POLICY_INVALID", f"invalid l1_stages={l1_stages!r}")
    if l0c_buffers is not None:
        if not isinstance(l0c_buffers, int) or isinstance(l0c_buffers, bool) or l0c_buffers < 1:
            raise PipelineError("ASCEND_SIL_POLICY_INVALID", f"invalid l0c_buffers={l0c_buffers!r}")

    new_buffers: list[Buffer] = []
    for buf in problem.buffers:
        if buf.space == "L1" and l1_stages is not None:
            new_buffers.append(
                Buffer(
                    name=buf.name,
                    slot_bytes=buf.slot_bytes,
                    space=buf.space,
                    owner=buf.owner,
                    fixed=int(l1_stages),
                )
            )
        elif buf.space == "L0C" and l0c_buffers is not None:
            new_buffers.append(
                Buffer(
                    name=buf.name,
                    slot_bytes=buf.slot_bytes,
                    space=buf.space,
                    owner=buf.owner,
                    fixed=int(l0c_buffers),
                )
            )
        else:
            new_buffers.append(buf)

    buf_by_name = {b.name: b for b in new_buffers}
    l1_names = {b.name for b in new_buffers if b.space == "L1"}
    task_index = {t.name: idx for idx, t in enumerate(problem.tasks)}

    writers: dict[str, set[str]] = {}
    readers: dict[str, set[str]] = {}
    for reuse in problem.reuses:
        writers.setdefault(reuse.buffer, set()).add(reuse.next_writer)
        readers.setdefault(reuse.buffer, set()).add(reuse.last_reader)

    new_edges = list(problem.edges)
    edge_keys = {(e.src, e.dst, e.distance, e.latency) for e in new_edges}
    new_reuses = list(problem.reuses)
    reuse_keys = {(r.buffer, r.last_reader, r.next_writer, r.delta) for r in new_reuses}

    if isinstance(raw_aliasing, Mapping):
        for raw_a, raw_b in raw_aliasing.items():
            name_a = _resolve_aliased_l1_name(str(raw_a), l1_names)
            name_b = _resolve_aliased_l1_name(str(raw_b), l1_names)
            if not name_a or not name_b or name_a == name_b:
                continue
            order_a = min((task_index.get(w, 10**9) for w in writers.get(name_a, ())), default=10**9)
            order_b = min((task_index.get(w, 10**9) for w in writers.get(name_b, ())), default=10**9)
            if order_b < order_a:
                first_name, second_name = name_b, name_a
            else:
                first_name, second_name = name_a, name_b
            first_buf = buf_by_name[first_name]
            second_buf = buf_by_name[second_name]
            merged_bytes = max(first_buf.slot_bytes, second_buf.slot_bytes)
            buf_by_name[first_name] = replace(first_buf, slot_bytes=merged_bytes)
            buf_by_name[second_name] = replace(second_buf, slot_bytes=0)

            for r_task in sorted(readers.get(first_name, ())):
                for w_task in sorted(writers.get(second_name, ())):
                    if r_task != w_task:
                        ek = (r_task, w_task, 0, 1)
                        if ek not in edge_keys:
                            edge_keys.add(ek)
                            new_edges.append(Edge(src=r_task, dst=w_task, distance=0, latency=1))
            for r_task in sorted(readers.get(second_name, ())):
                for w_task in sorted(writers.get(first_name, ())):
                    if r_task != w_task:
                        rk = (first_name, r_task, w_task, 0)
                        if rk not in reuse_keys:
                            reuse_keys.add(rk)
                            new_reuses.append(
                                Reuse(
                                    buffer=first_name,
                                    last_reader=r_task,
                                    next_writer=w_task,
                                    delta=0,
                                    read_latency=1,
                                )
                            )

    ordered_buffers = tuple(buf_by_name[b.name] for b in new_buffers)
    new_capacities = []
    for cap in problem.capacities:
        if ub_256kb_unlock and cap.space == "UB" and cap.owner == "aiv" and cap.bytes < 256 * 1024:
            new_capacities.append(Capacity(space=cap.space, owner=cap.owner, bytes=256 * 1024))
        else:
            new_capacities.append(cap)

    return Problem(
        tasks=problem.tasks,
        edges=tuple(new_edges),
        buffers=ordered_buffers,
        reuses=tuple(new_reuses),
        resources=problem.resources,
        capacities=tuple(new_capacities),
        max_stage=problem.max_stage,
        manual=problem.manual,
        issue_after=problem.issue_after,
    )


def _solve_problem_core(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
) -> Assignment:
    lb = lower_bound_ii(problem)
    if fixed_ii is not None:
        if fixed_ii < lb:
            raise PipelineError(
                "ASCEND_PIPELINE_UNSAT",
                f"fixed_ii={fixed_ii} is below resource lower bound {lb}",
            )
        candidates = (fixed_ii,)
    else:
        seq_lat = sum(t.latency for t in problem.tasks) + 4
        upper = max(lb + max_ii, seq_lat)
        candidates = range(lb, upper + 1)

    last_detail = "UNSAT"
    unknown_streak = 0
    for ii in candidates:
        status, assignment, detail = solve(problem, ii, timeout_ms=timeout_ms)
        last_detail = detail
        if status == SAT and assignment is not None:
            problems = verify_assignment(problem, assignment)
            problems.extend(_verify_bitwise_mask_schedule(problem, assignment))
            if problems:
                raise PipelineError(
                    "ASCEND_PIPELINE_VERIFY_FAILED",
                    "; ".join(problems),
                )
            return assignment
        if status == "UNSAT":
            unknown_streak = 0
        else:
            unknown_streak += 1
            if unknown_streak >= 2:
                break
    raise PipelineError(
        "ASCEND_PIPELINE_UNSAT",
        f"no feasible Ascend modulo schedule found ({last_detail})",
    )


def solve_ascend_pipeline(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
) -> AscendPipelineResult:
    """Solve for the minimum feasible initiation interval II and verify the model."""
    policy_dict = _coerce_sil_policy(sil_policy)
    solved_problem = problem
    fallback_used = False
    fallback_reason = ""

    if policy_dict is not None:
        try:
            policy_problem = _apply_sil_policy_to_problem(problem, policy_dict)
            assignment = _solve_problem_core(
                policy_problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
            )
            solved_problem = policy_problem
        except PipelineError as exc:
            fallback_used = True
            fallback_reason = str(exc)
            assignment = _solve_problem_core(
                problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
            )
            solved_problem = problem
    else:
        assignment = _solve_problem_core(
            problem,
            fixed_ii=fixed_ii,
            max_ii=max_ii,
            timeout_ms=timeout_ms,
        )

    if shadow_dump_path is not None:
        _write_shadow_dump(
            shadow_dump_path,
            {
                "kind": "solved_pipeline",
                "ii": assignment.ii,
                "starts": assignment.starts,
                "stages": assignment.stages,
                "versions": assignment.versions,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "sil_policy": policy_dict,
                "tasks": [
                    {"name": t.name, "latency": t.latency, "resources": str(t.resources)}
                    for t in solved_problem.tasks
                ],
                "buffers": [
                    {"name": b.name, "space": b.space, "slot_bytes": b.slot_bytes}
                    for b in solved_problem.buffers
                ],
            },
        )

    return AscendPipelineResult(
        problem=solved_problem,
        assignment=assignment,
        task_stages=dict(assignment.stages),
        buffer_versions=dict(assignment.versions),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        applied_sil_policy=policy_dict,
    )


def solve_pipeline_with_monotone_backoff(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
    max_retries: int = 4,
) -> AscendPipelineResult:
    """Solve Z3 modulo schedule with monotone tile-byte backoff before raising PipelineError."""
    cur_problem = problem
    for step in range(max(0, int(max_retries)) + 1):
        try:
            return solve_ascend_pipeline(
                cur_problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
                sil_policy=sil_policy,
                shadow_dump_path=shadow_dump_path,
            )
        except PipelineError as exc:
            if step >= max_retries:
                raise exc
            shrunk_buffers: list[Buffer] = []
            changed = False
            for buf in cur_problem.buffers:
                if buf.slot_bytes > 32:
                    new_bytes = max(32, ((buf.slot_bytes // 2) // 32) * 32)
                    if new_bytes < buf.slot_bytes:
                        changed = True
                        shrunk_buffers.append(replace(buf, slot_bytes=new_bytes))
                        continue
                shrunk_buffers.append(buf)
            if not changed:
                single_ver_buffers: list[Buffer] = []
                for buf in shrunk_buffers:
                    if buf.fixed is None and (buf.auto_max is None or buf.auto_max > 1):
                        changed = True
                        single_ver_buffers.append(replace(buf, fixed=1, auto_min=1, auto_max=1))
                    else:
                        single_ver_buffers.append(buf)
                shrunk_buffers = single_ver_buffers
            if not changed:
                raise exc
            cur_problem = replace(cur_problem, buffers=tuple(shrunk_buffers))
    raise PipelineError("ASCEND_PIPELINE_UNSAT", "monotone backoff exhausted")


def solve_loop_with_ledger(
    loop: For,
    *,
    target: AscendTarget | None = None,
    fn_stmt: Stmt | None = None,
    fixed_ii: int | None = None,
    unreserved_ubuf: bool = False,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
    dim_hints: Mapping[str, int] | None = None,
    allow_dynamic_fallback: bool = False,
) -> tuple[AscendPipelineResult, ObjectivePhysicalLedger]:
    """Solve a static TIR loop via Mode A Z3 and project an ObjectivePhysicalLedger."""
    resolved_target = target or AscendTarget("ascend.950pr")
    problem, _stages = extract_tir_loop_problem(
        loop,
        target=resolved_target,
        fn_stmt=fn_stmt,
        unreserved_ubuf=unreserved_ubuf,
        dim_hints=dim_hints,
        allow_dynamic_fallback=allow_dynamic_fallback,
    )
    result = solve_ascend_pipeline(
        problem,
        fixed_ii=fixed_ii,
        sil_policy=sil_policy,
        shadow_dump_path=shadow_dump_path,
    )
    solved_problem = result.problem
    cap_map = {(c.space, c.owner): c.bytes for c in solved_problem.capacities}
    waterlines: dict[str, ScopeWaterline] = {}
    spaces = sorted({b.space for b in solved_problem.buffers if b.space})
    for space in spaces:
        bufs = [b for b in solved_problem.buffers if b.space == space]
        owner = bufs[0].owner
        cap_bytes = cap_map.get((space, owner), 0)
        explicit_bufs = {
            b.name: b.slot_bytes * result.buffer_versions.get(b.name, 1) for b in bufs
        }
        waterlines[space] = build_scope_waterline(
            scope=space,
            capacity_bytes=cap_bytes,
            explicit_buffers=explicit_bufs,
            implicit_buffers={},
        )
    pipe_cycles = pipe_cycles_from_problem(solved_problem, result.assignment)
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=result.assignment.ii,
        pipe_cycles=pipe_cycles,
        waterlines=waterlines,
        mte2_inflight_per_aiv=0,
        unaligned_dma_padding_bytes=0,
        opaque_dynamic_regions=(),
    )
    return result, ledger


def build_resolved_spec(spec: PipelineSpec, result: AscendPipelineResult) -> PipelineSpec:
    """Attach solved II, task stages, and resolved buffer version counts onto PipelineSpec."""
    existing_intents = {intent.task_id: intent for intent in spec.task_intents}
    resolved_intents: list[TaskIntent] = []
    for idx, task in enumerate(result.problem.tasks):
        prev = existing_intents.get(task.name)
        resolved_intents.append(
            TaskIntent(
                task_id=task.name,
                anchor=prev.anchor if prev is not None else idx,
                stage=result.task_stages.get(task.name, 0),
                issue_after=prev.issue_after if prev is not None else (),
                cost_override=prev.cost_override if prev is not None else task.latency,
            )
        )

    existing_bufs = {item.root: item for item in spec.buffer_schedules}
    resolved_bufs: list[BufferSchedule] = []
    for buf in result.problem.buffers:
        versions = result.buffer_versions.get(buf.name, 1)
        prev = existing_bufs.get(buf.name)
        resolved_bufs.append(
            BufferSchedule(
                root=buf.name,
                versions=("fixed", versions),
                owner=prev.owner if prev is not None and prev.owner else buf.owner,
                space=prev.space if prev is not None and prev.space else buf.space,
                layout=prev.layout if prev is not None else "",
            )
        )
    return PipelineSpec(
        max_stage=spec.max_stage,
        max_buffer_versions=spec.max_buffer_versions,
        fixed_ii=result.assignment.ii,
        enable_offset=spec.enable_offset,
        origin=spec.origin,
        task_intents=tuple(resolved_intents),
        buffer_schedules=tuple(resolved_bufs),
    )


__all__ = [
    "AscendPipelineResult",
    "bitwise_conflict_pairs",
    "build_resolved_spec",
    "extract_tir_loop_problem",
    "format_pipe_key",
    "masks_conflict",
    "pipe_cycles_from_problem",
    "read_from_sil",
    "solve_ascend_pipeline",
    "solve_loop_with_ledger",
    "solve_pipeline_with_monotone_backoff",
    "task_resource_mask",
    "validate_pipeline_spec",
]
