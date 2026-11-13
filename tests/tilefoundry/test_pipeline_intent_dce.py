"""Authored ``tf.pipelined`` declarations may not vanish in parse-time DCE."""

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
    PipelineIntent,
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
    path.write_text(program, encoding="utf-8")
    return load_authored_ir(f"{path}:M")


def _program(
    *, returned: str, loop: str = "tf.pipelined(4, max_stage=2, max_buffer_versions=2)"
) -> str:
    return (
        "\n".join(
            [
                HEADER,
                '@module(entry="k", target=CpuTarget())',
                "class M:",
                "    @func",
                '    def k(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:',
                "        acc = x",
                f"        for i in {loop}:",
                "            acc = acc + 1.0",
                f"        return {returned}",
            ]
        )
        + "\n"
    )


def _loops(module):
    found = []
    for item in module.functions:
        body = getattr(item, "body", None)
        if body is not None:
            found.extend(e for e in collect_exprs(body) if isinstance(e, LoopRegion))
    return found


def test_consumed_loop_keeps_its_pipeline_declaration():
    module = _load(_program(returned="acc"))
    loops = _loops(module)
    assert len(loops) == 1, f"expected one loop, got {len(loops)}"
    assert loops[0].pipeline is not None
    assert loops[0].pipeline.max_stage == 2
    assert loops[0].pipeline.max_buffer_versions == 2


def test_unconsumed_loop_is_refused_as_dropped_by_dce():
    with pytest.raises(PipelineError) as caught:
        _load(_program(returned="x"))
    assert caught.value.code == "PIPELINE_DECLARATION_DROPPED_BY_DCE", caught.value
    assert "tf.pipelined" in str(caught.value)


def test_ordinary_program_without_a_pipeline_is_unaffected():
    module = _load(_program(returned="acc", loop="range(4)"))
    loops = _loops(module)
    assert len(loops) == 1
    assert loops[0].pipeline is None


def test_empty_intent_set_is_a_noop_not_a_vacuous_failure():
    module = _load(_program(returned="acc", loop="range(4)"))
    verify_pipeline_intents((), module.functions[0])


def test_intent_is_located_from_the_authored_ast_before_any_body():
    source = (
        "def k(x):\n"
        "    acc = x\n"
        "    for i in tf.pipelined(4, max_stage=3, max_buffer_versions=2, enable_offset=True):\n"
        "        acc = acc + 1.0\n"
        "    return x\n"
    )
    node = ast.parse(source).body[0]
    intents = register_pipeline_intents(node, function_name="k", dialect="hir")
    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, PipelineIntent)
    assert intent.max_stage == 3
    assert intent.max_buffer_versions == 2
    assert intent.enable_offset is True
    assert intent.source_key == (3, 13)


def test_duplicate_source_intent_is_refused():
    from tilefoundry.parser import pipeline_intents as module  # noqa: PLC0415

    source = (
        "def k(x):\n"
        "    acc = x\n"
        "    for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2):\n"
        "        acc = acc + 1.0\n"
        "    return acc\n"
    )
    node = ast.parse(source).body[0]
    one = register_pipeline_intents(node, function_name="k", dialect="hir")
    parsed = _load(_program(returned="acc")).functions[0]
    with pytest.raises(PipelineError) as caught:
        module.verify_pipeline_intents(one + one, parsed)
    assert caught.value.code == "PIPELINE_SOURCE_DUPLICATE", caught.value
