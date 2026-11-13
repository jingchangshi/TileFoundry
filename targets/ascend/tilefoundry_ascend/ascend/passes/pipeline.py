"""Ascend pipeline validation and Z3 resolution passes."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.loop_region import LoopRegion
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
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    build_resolved_spec,
    extract_tir_loop_problem,
    solve_ascend_pipeline,
    validate_pipeline_spec,
)
from tilefoundry_ascend.ascend.target import AscendTarget


@register_pass(AscendTarget, PassPoint.BEFORE_LOWER)
class ValidateAscendPipelines(ModulePass):
    """Validate HIR loop pipeline declarations against Ascend950PR constraints."""

    name = "validate_ascend_pipelines"

    def run(self, module: Module) -> Module:
        target = module.resolve_target()
        if not isinstance(target, AscendTarget):
            return module
        for fn in module.functions:
            if not isinstance(fn, HirFunction) or fn.body is None:
                continue
            for expr in collect_exprs(fn.body):
                if isinstance(expr, LoopRegion) and expr.pipeline is not None:
                    validate_pipeline_spec(expr.pipeline, target)
        return module


def _rewrite_tir_stmt(
    stmt: Stmt | None,
    *,
    fn_stmt: Stmt,
    target: AscendTarget,
    solved_records: dict[str, dict],
    fn_name: str,
) -> tuple[Stmt | None, bool]:
    if stmt is None:
        return None, False
    if isinstance(stmt, Sequential):
        children = []
        changed = False
        for child in stmt.body:
            new_child, child_changed = _rewrite_tir_stmt(
                child,
                fn_stmt=fn_stmt,
                target=target,
                solved_records=solved_records,
                fn_name=fn_name,
            )
            children.append(new_child)
            changed = changed or child_changed
        return (replace(stmt, body=tuple(children)), True) if changed else (stmt, False)
    if isinstance(stmt, LetStmt):
        return stmt, False
    if isinstance(stmt, (MeshScope, ExecutionScope)):
        new_body, changed = _rewrite_tir_stmt(
            stmt.body,
            fn_stmt=fn_stmt,
            target=target,
            solved_records=solved_records,
            fn_name=fn_name,
        )
        return (replace(stmt, body=new_body), True) if changed else (stmt, False)
    if isinstance(stmt, While):
        new_body, changed = _rewrite_tir_stmt(
            stmt.body,
            fn_stmt=fn_stmt,
            target=target,
            solved_records=solved_records,
            fn_name=fn_name,
        )
        return (replace(stmt, body=new_body), True) if changed else (stmt, False)
    if isinstance(stmt, If):
        then_body, c1 = _rewrite_tir_stmt(
            stmt.then_body,
            fn_stmt=fn_stmt,
            target=target,
            solved_records=solved_records,
            fn_name=fn_name,
        )
        else_body, c2 = _rewrite_tir_stmt(
            stmt.else_body,
            fn_stmt=fn_stmt,
            target=target,
            solved_records=solved_records,
            fn_name=fn_name,
        )
        if c1 or c2:
            return replace(stmt, then_body=then_body, else_body=else_body), True
        return stmt, False
    if isinstance(stmt, For):
        new_body, body_changed = _rewrite_tir_stmt(
            stmt.body,
            fn_stmt=fn_stmt,
            target=target,
            solved_records=solved_records,
            fn_name=fn_name,
        )
        updated = replace(stmt, body=new_body) if body_changed else stmt
        if updated.pipeline is None:
            return updated, body_changed
        problem, _initial_stages = extract_tir_loop_problem(
            updated,
            target=target,
            fn_stmt=fn_stmt,
        )
        result = solve_ascend_pipeline(problem, fixed_ii=updated.pipeline.fixed_ii)
        resolved_spec = build_resolved_spec(updated.pipeline, result)
        loop_key = f"{fn_name}:{updated.induction_var.name}"
        solved_records[loop_key] = {
            "ii": result.assignment.ii,
            "stages": dict(result.task_stages),
            "phases": dict(result.assignment.phases),
            "versions": dict(result.buffer_versions),
            "issue_order": tuple(result.assignment.issue_order),
        }
        return replace(updated, pipeline=resolved_spec), True
    return stmt, False


@register_pass(AscendTarget, PassPoint.AFTER_LOWER)
class ResolveAscendPipelines(ModulePass):
    """Resolve pipelined TIR For loops using the Z3 modulo scheduler."""

    name = "resolve_ascend_pipelines"

    def run(self, module: Module) -> Module:
        target = module.resolve_target()
        if not isinstance(target, AscendTarget):
            return module
        solved_records: dict[str, dict] = {}
        new_fns = []
        changed = False
        for fn in module.functions:
            if isinstance(fn, PrimFunction) and fn.body is not None:
                new_body, body_changed = _rewrite_tir_stmt(
                    fn.body,
                    fn_stmt=fn.body,
                    target=target,
                    solved_records=solved_records,
                    fn_name=fn.name,
                )
                if body_changed:
                    new_fns.append(replace(fn, body=new_body))
                    changed = True
                else:
                    new_fns.append(fn)
            else:
                new_fns.append(fn)
        if isinstance(module.metadata, dict) and solved_records:
            module.metadata["ascend.pipelines"] = solved_records
        if not changed:
            return module
        return replace(module, functions=tuple(new_fns))


__all__ = ["ResolveAscendPipelines", "ValidateAscendPipelines"]
