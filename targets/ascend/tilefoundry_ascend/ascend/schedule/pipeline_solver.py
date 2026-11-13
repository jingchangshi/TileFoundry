"""Ascend950PR Z3 modulo pipeline constraint projection and solver."""

from __future__ import annotations

from dataclasses import dataclass

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
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendUbToGm,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


@dataclass(frozen=True)
class AscendPipelineResult:
    """Solved Ascend950PR modulo pipeline assignment alongside the projected Z3 problem."""

    problem: Problem
    assignment: Assignment
    task_stages: dict[str, int]
    buffer_versions: dict[str, int]


@dataclass(frozen=True)
class _LoopTaskItem:
    label: str
    resources: tuple[tuple, ...]
    calls: tuple[tuple[Op, tuple[Expr, ...]], ...]


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_byte_size(
    tensor_type: TensorType,
    space: str,
    facts: AscendSpaceFacts,
    *,
    aiv_factor: int = 1,
) -> int:
    total = 1
    for dim in tensor_type.shape:
        extent = static_dim_value(dim)
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


def _static_tensor_elems(tensor_type: TensorType | None) -> int:
    if tensor_type is None:
        return 0
    total = 1
    for dim in tensor_type.shape:
        extent = static_dim_value(dim)
        if extent is None or extent < 1:
            return 0
        total *= extent
    return total


def _static_tensor_bytes(tensor_type: TensorType | None) -> int:
    elems = _static_tensor_elems(tensor_type)
    if elems <= 0 or tensor_type is None:
        return 0
    return (elems * tensor_type.dtype.bit_width + 7) // 8


def _estimate_task_latency(item: _LoopTaskItem) -> int:
    """Estimate normalized hardware cycle quanta (1..8) from DMA bytes, MMAD FLOPs, or VF ops."""
    if item.label in ("simdvf", "simtvf"):
        total_elems = 0
        for _op, args in item.calls:
            op_elems = max(
                (_static_tensor_elems(arg.type) for arg in args if isinstance(arg.type, TensorType)),
                default=0,
            )
            total_elems += op_elems
        return min(max(1, (total_elems + 4095) // 4096), 8)

    total_quanta = 0
    for op, args in item.calls:
        arg_types = [arg.type if isinstance(arg.type, TensorType) else None for arg in args]
        if isinstance(op, AscendMmad) and len(arg_types) >= 3:
            lhs, rhs = arg_types[1], arg_types[2]
            if lhs is not None and rhs is not None and len(lhs.shape) >= 2 and len(rhs.shape) >= 2:
                m = static_dim_value(lhs.shape[-2] if op.a_layout == "MK" else lhs.shape[-1])
                k = static_dim_value(lhs.shape[-1] if op.a_layout == "MK" else lhs.shape[-2])
                n = static_dim_value(rhs.shape[-2] if op.b_layout == "NK" else rhs.shape[-1])
                if m and k and n:
                    flops = 2 * m * n * k
                    total_quanta += max(1, (flops + 65535) // 65536)
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
            ),
        ):
            xfer_bytes = max((_static_tensor_bytes(t) for t in arg_types), default=0)
            total_quanta += max(1, (xfer_bytes + 16383) // 16384)
            continue
        elems = max((_static_tensor_elems(t) for t in arg_types), default=0)
        total_quanta += max(1, (elems + 4095) // 4096)
    return min(max(1, total_quanta), 8)


def _op_resources(op: Op, arg_types: tuple[TensorType | None, ...]) -> tuple[tuple, ...]:
    resource = getattr(op, "resource", None)
    if resource == "mte2_engine":
        dst = arg_types[-1] if arg_types else None
        if dst is not None and dst.storage is StorageKind.TMEM:
            return (("aic", "aic0", "mte2", 1),)
        return (("aiv", "aiv0", "mte2", 1),)
    if resource == "mte3_engine":
        return (("aiv", "aiv0", "mte3", 1),)
    if resource == "mte1_engine":
        return (("aic", "aic0", "mte1", 1),)
    if resource == "cube_engine":
        return (("aic", "aic0", "cube", 1),)
    if resource == "fixpipe_engine":
        return (("aic", "aic0", "fixpipe", 1),)
    if resource == "dual_copy_engine":
        return (("aic", "aic0", "fixpipe", 1), ("aiv", "aiv0", "mte2", 1))
    if resource == "vector_engine":
        return (("aiv", "aiv0", "vector", 1),)
    if len(arg_types) >= 2 and all(ty is not None for ty in arg_types[:2]):
        src_s, dst_s = arg_types[0].storage, arg_types[1].storage
        if src_s is StorageKind.GMEM and dst_s is StorageKind.SMEM:
            return (("aiv", "aiv0", "mte2", 1),)
        if src_s is StorageKind.SMEM and dst_s is StorageKind.GMEM:
            return (("aiv", "aiv0", "mte3", 1),)
        if src_s is StorageKind.GMEM and dst_s is StorageKind.TMEM:
            return (("aic", "aic0", "mte2", 1),)
        if src_s is StorageKind.TMEM and dst_s is StorageKind.GMEM:
            return (("aic", "aic0", "fixpipe", 1),)
    return (("aiv", "aiv0", "vector", 1),)


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


def _capacities_from_facts(facts: AscendSpaceFacts) -> tuple[Capacity, ...]:
    return (
        Capacity("UB", "aiv", facts.ub_usable_bytes),
        Capacity("L1", "aic", facts.l1_bytes),
        Capacity("L0A", "aic", facts.l0a_bytes),
        Capacity("L0B", "aic", facts.l0b_bytes),
        Capacity("L0C", "aic", facts.l0c_bytes),
    )


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


def _collect_stmts(stmt: Stmt | None, env: dict[int, Expr], out: list[_LoopTaskItem]) -> None:
    if stmt is None:
        return
    if isinstance(stmt, Sequential):
        for child in stmt.body:
            _collect_stmts(child, env, out)
        return
    if isinstance(stmt, LetStmt):
        env[id(stmt.var)] = stmt.value
        return
    if isinstance(stmt, ExecutionScope) and stmt.kind in (SIMDVF_KIND.name, SIMTVF_KIND.name):
        inner: list[tuple[Op, tuple[Expr, ...]]] = []
        _collect_raw_calls(stmt.body, env, inner)
        if inner:
            vf_label = "simdvf" if stmt.kind == SIMDVF_KIND.name else "simtvf"
            out.append(
                _LoopTaskItem(
                    label=vf_label,
                    resources=(("aiv", "aiv0", "vector", 1),),
                    calls=tuple(inner),
                )
            )
        return
    if isinstance(stmt, (MeshScope, ExecutionScope)):
        _collect_stmts(stmt.body, env, out)
        return
    if isinstance(stmt, If):
        _collect_stmts(stmt.then_body, env, out)
        _collect_stmts(stmt.else_body, env, out)
        return
    if isinstance(stmt, (For, While)):
        _collect_stmts(stmt.body, env, out)
        return
    if isinstance(stmt, Evaluate):
        if isinstance(stmt.callable, Op):
            op = stmt.callable
            schema_name = getattr(type(op), "_op_schema", None)
            op_label = schema_name.name if schema_name is not None else type(op).__name__
            arg_types = tuple(
                arg.type if isinstance(arg.type, TensorType) else None for arg in stmt.args
            )
            res_uses = _op_resources(op, arg_types)
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
) -> tuple[Problem, dict[str, int]]:
    """Build a Z3 pipeline Problem and initial stage map from a TIR For loop."""
    spec = loop.pipeline
    if spec is None:
        raise PipelineError("PIPELINE_MISSING", "loop has no PipelineSpec declaration")
    resolved_target = target or AscendTarget("ascend.950pr")
    validate_pipeline_spec(spec, resolved_target)
    facts = resolved_target.get_facts(AscendSpaceFacts)

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

    for idx, item in enumerate(task_items):
        t_name = f"t{idx}_{item.label}"
        intent = authored_by_anchor.get(idx) or authored_by_id.get(t_name)
        res_uses = item.resources
        latency = (
            intent.cost_override
            if (intent and intent.cost_override is not None)
            else _estimate_task_latency(item)
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
                elif isinstance(op, (AscendL0cToGm, AscendDualCopy)) and param.name == "src":
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
        slot_bytes = _tensor_byte_size(buf_type, space, facts, aiv_factor=aiv_factor)
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

    problem = tf_adapter(
        {
            "tasks": tuple(tasks),
            "edges": tuple(edges),
            "buffers": tuple(buffers),
            "reuses": tuple(reuses),
            "resources": _resources_for_tasks(tasks),
            "capacities": _capacities_from_facts(facts),
            "max_stage": max_stage,
            "manual": manual,
            "issue_after": tuple(issue_after),
        }
    )
    return problem, assigned_stages


def authored_stages_set(names: list[str]) -> set[str]:
    return set(names)


def solve_ascend_pipeline(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
) -> AscendPipelineResult:
    """Solve for the minimum feasible initiation interval II and verify the model."""
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
            if problems:
                raise PipelineError(
                    "ASCEND_PIPELINE_VERIFY_FAILED",
                    "; ".join(problems),
                )
            return AscendPipelineResult(
                problem=problem,
                assignment=assignment,
                task_stages=dict(assignment.stages),
                buffer_versions=dict(assignment.versions),
            )
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
    "build_resolved_spec",
    "extract_tir_loop_problem",
    "solve_ascend_pipeline",
    "validate_pipeline_spec",
]
