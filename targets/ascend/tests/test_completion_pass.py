"""S2: the completion-injection pass is target-scoped and injects a valid drain."""
from __future__ import annotations

from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, MeshScope, Sequential
from tilefoundry.ir.types import Layout, Mesh, Topology, make_tensor_type
from tilefoundry.passes.pass_manager import _TARGET_PASSES, PassPoint
from tilefoundry.visitor_registry.verify import verify_prim_function

from tilefoundry_ascend.passes.completion import InjectCompletionPass, inject_into_function
from tilefoundry_ascend.tf_target import Ascend950PRTarget

_MESH = Mesh((Topology("thread", 1),), Layout((1,), (1,)))
_BIND = Var(type=make_tensor_type((), storage="rmem"), name="m")


def _fn() -> PrimFunction:
    return PrimFunction(name="k", params=(), body=Sequential(body=(
        MeshScope(mesh=_MESH, binding=_BIND, body=Sequential(body=())),)))


def _ndrain(body: Sequential) -> int:
    n = 0
    for s in body.body:
        if isinstance(s, Evaluate) and isinstance(s.callable, Drain):
            n += 1
        elif isinstance(s, (MeshScope, ExecutionScope)) and isinstance(s.body, Sequential):
            n += _ndrain(s.body)
    return n


def test_pass_is_registered_only_for_ascend() -> None:
    reg = _TARGET_PASSES.get((Ascend950PRTarget, PassPoint.AFTER_LOWER), [])
    assert InjectCompletionPass in reg
    for (target, _point), passes in _TARGET_PASSES.items():
        if target is Ascend950PRTarget:
            continue
        assert InjectCompletionPass not in passes, target


def test_injects_one_drain_inside_the_mesh_scope() -> None:
    fn = inject_into_function(_fn(), {"flag_pool": {}})
    assert _ndrain(fn.body) == 1
    verify_prim_function(fn)
    assert _ndrain(inject_into_function(fn, {"flag_pool": {}}).body) == 1  # idempotent


def test_no_plan_leaves_function_unchanged() -> None:
    assert _ndrain(inject_into_function(_fn(), None).body) == 0
