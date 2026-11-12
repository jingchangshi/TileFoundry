"""F0: an authored `tf.pipelined` declaration may not vanish in parse-time DCE.

The bug this pins: `tf.pipelined` is a DECLARATION, but the HIR body is built by
reachability. A loop whose value nothing consumes never reaches the returned
expression, so the `LoopRegion` -- and with it `LoopRegion.pipeline` -- is gone,
while finalize still succeeds. That "all zero" result is about a program with no
pipeline in it, which is a vacuous success, not evidence.

F0 records the intent from the authored AST before the body is walked, then
requires each one to be a typed `LoopRegion.pipeline` on the HIR that survived.
These tests are red for the right reason only if the guard is wired; the
`test_..._raised...` case is the hostile program, and deleting the guard makes it
load successfully instead of refusing.
"""

from __future__ import annotations

import ast
import pathlib
import tempfile

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline import PipelineError
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.parser.pipeline_intents import (
    PIPELINE_INTENT_SCHEMA,
    register_pipeline_intents,
    verify_pipeline_intents,
)

HEADER = (
    "from __future__ import annotations\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry.target import CpuTarget\n"
)


def _load(program: str):
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_f0_"))
    path = directory / "prog.py"
    path.write_text(program)
    return load_authored_ir("%s:M" % path)


def _program(*, returned: str, loop: str = "tf.pipelined(4, num_stages=2)") -> str:
    return "\n".join(
        [
            HEADER,
            '@module(entry="k", target=CpuTarget())',
            "class M:",
            "    @func",
            '    def k(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:',
            "        acc = x",
            "        for i in %s:" % loop,
            "            acc = acc + 1.0",
            "        return %s" % returned,
        ]
    ) + "\n"


def _loops(module):
    found = []
    for item in module.functions:
        body = getattr(item, "body", None)
        if body is not None:
            found.extend(e for e in collect_exprs(body) if isinstance(e, LoopRegion))
    return found


def test_consumed_loop_keeps_its_pipeline_declaration():
    """CONTROL: the loop's value is returned, so the LoopRegion and its spec live."""
    module = _load(_program(returned="acc"))
    loops = _loops(module)
    assert len(loops) == 1, "expected exactly one surviving loop, got %d" % len(loops)
    assert loops[0].pipeline is not None
    assert loops[0].pipeline.num_stages == 2


def test_unconsumed_loop_is_refused_as_dropped_by_dce():
    """BAD: `return x` leaves the loop's value unused -> the declaration disappears.

    Without the guard this program loads cleanly and finalizes with zero loops --
    the exact vacuous success F0 exists to forbid. So this case is the mutation
    detector: remove the guard and it stops raising.
    """
    with pytest.raises(PipelineError) as caught:
        _load(_program(returned="x"))
    assert caught.value.code == "PIPELINE_DECLARATION_DROPPED_BY_DCE", caught.value
    # The refusal names the authored source, not a rebuilt empty set.
    assert "num_stages" in str(caught.value)
    assert "tf.pipelined" in str(caught.value)


def test_ordinary_program_without_a_pipeline_is_unaffected():
    """No declaration means no expectation: an ordinary loop loads and survives."""
    module = _load(_program(returned="acc", loop="range(4)"))
    loops = _loops(module)
    assert len(loops) == 1
    assert loops[0].pipeline is None


def test_empty_intent_set_is_a_noop_not_a_vacuous_failure():
    """An empty authored set is legal (nothing was declared); it must not refuse."""
    module = _load(_program(returned="acc", loop="range(4)"))
    verify_pipeline_intents((), module.functions[0])  # must not raise


def test_intent_is_located_from_the_authored_ast_before_any_body():
    """Registration reads the source, so a dropped loop still has a recorded intent."""
    source = (
        "def k(x):\n"
        "    acc = x\n"
        "    for i in tf.pipelined(4, num_stages=3, annotations={'enable_offset': True}):\n"
        "        acc = acc + 1.0\n"
        "    return x\n"
    )
    node = ast.parse(source).body[0]
    intents = register_pipeline_intents(node, function_name="k", dialect="hir")
    assert len(intents) == 1
    intent = intents[0]
    assert intent.schema == PIPELINE_INTENT_SCHEMA
    assert intent.num_stages == 3
    assert intent.enable_offset is True
    assert intent.source_key == (3, 13)  # the tf.pipelined call position


def test_duplicate_source_intent_is_refused(monkeypatch):
    """Two intents at one source is a source error, never silently collapsed."""
    from tilefoundry.parser import pipeline_intents as module  # noqa: PLC0415

    source = (
        "def k(x):\n"
        "    acc = x\n"
        "    for i in tf.pipelined(4, num_stages=2):\n"
        "        acc = acc + 1.0\n"
        "    return acc\n"
    )
    node = ast.parse(source).body[0]
    one = register_pipeline_intents(node, function_name="k", dialect="hir")
    parsed = _load(_program(returned="acc")).functions[0]
    with pytest.raises(PipelineError) as caught:
        module.verify_pipeline_intents(one + one, parsed)
    assert caught.value.code == "PIPELINE_SOURCE_DUPLICATE", caught.value
