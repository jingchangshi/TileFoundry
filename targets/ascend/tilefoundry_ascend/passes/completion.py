"""Ascend completion-protocol injection at PassPoint.AFTER_LOWER (S2).

STRUCTURAL LIMITATION (recorded, not a detail): the completion protocol is introduced by
a TARGET PASS *AFTER* lowering, not derived from authored-HIR semantics. It is therefore
provable only from the pass plus the plan, NOT from the source program's semantics. The
authored HIR says nothing about flags; the flag nodes appear because this pass runs.

The plan travels on Module.metadata (ConvertHIRToTIR mutates and returns the SAME Module,
so metadata survives finalize). The pass is registered ONLY for Ascend950PRTarget, so a
CUDA module can never acquire it (target_passes looks the pass up by target type).

This file is the S2 mechanism increment: companion helpers + registration + a safe, valid
Drain append. Stage-relative FlagSet/FlagWait placement and the plan->HIR emitter are the
next increments; this pass does not pretend to be stage-correct yet.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, MeshScope, Sequential
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass

from ..tf_target import Ascend950PRTarget

COMPLETION_PLAN_KEY = "ascend.completion_plan"


def carry_completion_plan(module: Module, plan: dict) -> Module:
    """Attach the resolved plan to the module so AFTER_LOWER can see it.

    Module.metadata is a declared field and ConvertHIRToTIR returns the same module
    object, so this survives finalize without a private side channel.
    """
    module.metadata[COMPLETION_PLAN_KEY] = plan
    return module


def drain_stmt() -> Evaluate:
    return Evaluate(callable=Drain(), args=())


def _has_drain(body: Sequential) -> bool:
    for s in body.body:
        if isinstance(s, Evaluate) and isinstance(s.callable, Drain):
            return True
        if isinstance(s, (MeshScope, ExecutionScope)) and isinstance(s.body, Sequential):
            if _has_drain(s.body):
                return True
    return False


def append_drain(body: Sequential) -> Sequential:
    """Append the final completion barrier INSIDE the innermost mesh/execution scope.

    A drain is a mesh-scoped statement, so it must sit where the executing threads are;
    appending it at function top level would be outside every MeshScope.
    """
    if _has_drain(body):
        return body
    if body.body:
        last = body.body[-1]
        if isinstance(last, (MeshScope, ExecutionScope)) and isinstance(last.body, Sequential):
            inner = append_drain(last.body)
            if inner is not last.body:
                return Sequential(body=tuple(body.body[:-1]) + (replace(last, body=inner),))
    return Sequential(body=tuple(body.body) + (drain_stmt(),))


def inject_into_function(fn: PrimFunction, plan: dict | None) -> PrimFunction:
    """Return *fn* with completion nodes injected (currently: the drain barrier)."""
    if plan is None:
        return fn
    return replace(fn, body=append_drain(fn.body))


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class InjectCompletionPass(ModulePass):
    """Inject the completion protocol into a lowered Ascend module."""

    name = "ascend_inject_completion"

    def run(self, module: Module) -> Module:
        plan = module.metadata.get(COMPLETION_PLAN_KEY)
        if plan is None:
            return module
        fn = module.entry_function()
        if not isinstance(fn, PrimFunction):
            return module
        new_fn = inject_into_function(fn, plan)
        if new_fn is fn:
            return module
        return replace(module, functions=tuple(new_fn if f is fn else f for f in module.functions))


__all__ = ["COMPLETION_PLAN_KEY", "carry_completion_plan", "inject_into_function",
           "InjectCompletionPass"]
