"""D0 TIR pipeline reader: walk the REAL statement structure.

The reader used to read `Sequential.stmts`, but `Sequential`'s field is `body`
(docs/spec/tir.md §1.2, ir/tir/stmts.py), so actual was ALWAYS empty. A lowered
pipelined loop sits inside `MeshScope.body` and usually `ExecutionScope.body`, so a
reader that only knew a top-level container could never see it either. The result:
`VerifyPipelinePreservation` raised PIPELINE_DECLARATION_LOST on a loop whose
`For.pipeline` was genuinely not None -- and the 22 function-level D0 controls never
noticed, because they exercise the predicates directly, not the public finalize path.

Three cases, all built as direct TIR so they pin the READER rather than the builder:

  1. a For carrying `pipeline` inside MeshScope->ExecutionScope is FOUND (recursion);
  2. the same For with `pipeline=None` is refused as PIPELINE_DECLARATION_LOST;
  3. a loop with no pipeline at all reads as {} and is not misreported.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import tilefoundry_ascend  # noqa: F401  (registers the SimdVF surface / target)
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.hir.pipeline import PipelineError, PipelineSpec
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import ExecutionScope, For, MeshScope, Sequential
from tilefoundry.ir.types import DType, Mesh, Topology, make_tensor_type
from tilefoundry_ascend.passes.pipeline import (
    PIPELINE_MANIFEST_KEY,
    PipelineManifest,
    VerifyPipelinePreservation,
    collect_tir_pipeline_declarations,
    verify_pipeline_preservation,
)

AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))
_INDEX = make_tensor_type((), dtype=DType.i64, storage="umat")


def _for(pipeline):
    induction = Var(name="i", type=_INDEX)
    return For(
        induction,
        Constant(0, type=_INDEX),
        Constant(4, type=_INDEX),
        Constant(1, type=_INDEX),
        Sequential(body=()),
        pipeline=pipeline,
    )


def _module_with(for_stmt, *, wrap: bool = True):
    """One PrimFunction whose body holds *for_stmt*, optionally under Mesh/Execution."""
    body = Sequential(body=(for_stmt,))
    if wrap:
        body = Sequential(
            body=(
                MeshScope(
                    AIV,
                    Var(name="aiv", type=_INDEX),
                    Sequential(body=(ExecutionScope(kind="ascend.simdvf", body=body),)),
                ),
            )
        )
    function = PrimFunction(name="k", params=(), body=body, output_count=0)

    class _Module:
        functions = (function,)

    return _Module(), function


def _manifest(loops=()):
    return PipelineManifest(schema="ascend.pipeline_manifest/1", loops=tuple(loops),
                            buffer_declarations=(), requirements=())


def test_pipeline_inside_mesh_and_execution_scope_is_found():
    """CASE 1 (positive): recursion through MeshScope/ExecutionScope.body works."""
    spec = PipelineSpec(num_stages=2, origin=(1, 1))
    module, _ = _module_with(_for(spec), wrap=True)
    actual = collect_tir_pipeline_declarations(module)
    assert actual == {"loop[0]:i": spec}, (
        "the reader did not find the For.pipeline nested in MeshScope->ExecutionScope; "
        "actual=%r. This is the empty-actual defect the fix closes." % (actual,)
    )


def test_dropped_pipeline_is_refused_as_lost():
    """CASE 2 (negative): same nesting, For.pipeline None -> DECLARATION_LOST."""
    spec = PipelineSpec(num_stages=2, origin=(1, 1))
    module, _ = _module_with(_for(None), wrap=True)
    actual = collect_tir_pipeline_declarations(module)
    assert actual == {}, actual  # the loop is there, the declaration is not
    with pytest_raises(PipelineError) as caught:
        verify_pipeline_preservation(
            {"loop": "loop[0]:i", "pipeline": spec},
            {"loop": "loop[0]:i", "pipeline": actual.get("loop[0]:i")},
        )
    assert caught.value.code == "PIPELINE_DECLARATION_LOST", caught.value


def test_program_without_any_pipeline_is_silent():
    """CASE 3 (control): no pipeline -> {} and the AFTER pass does not misreport."""
    module, _ = _module_with(_for(None), wrap=True)
    assert collect_tir_pipeline_declarations(module) == {}
    module.metadata = {PIPELINE_MANIFEST_KEY: _manifest()}  # type: ignore[attr-defined]
    assert VerifyPipelinePreservation().run(module) is module


class pytest_raises:
    """Local stand-in so this file runs without pytest in the env of record."""

    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("expected %s; nothing was raised" % self.exc.__name__)
        if not issubclass(exc_type, self.exc):
            return False
        self.value = exc
        return True


def main() -> int:
    cases = (
        test_pipeline_inside_mesh_and_execution_scope_is_found,
        test_dropped_pipeline_is_refused_as_lost,
        test_program_without_any_pipeline_is_silent,
    )
    failed = 0
    for case in cases:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(cases) - failed, len(cases)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
