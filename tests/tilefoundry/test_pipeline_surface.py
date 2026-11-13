"""Pipeline surface: declaration carried, printed, cloned, never invented."""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from tilefoundry.analysis.pipeline_declarations import (
    verify_buffer_versions_unchanged,
    verify_pipeline_preserved,
)
from tilefoundry.analysis.pipeline_reader import collect_buffer_versions
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline import PipelineError, PipelineSpec
from tilefoundry.ir.visitor import ExprCloner, collect_exprs

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry.target import CpuTarget\n"
)


def build(loop_line: str, *, alloc: str = "", entry: str = "k", header_extra: str = ""):
    rows = [
        HEADER + header_extra,
        f'@module(entry="{entry}", target=CpuTarget())',
        "class M:",
        "    @func",
        f'    def {entry}(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:',
        "        acc = x",
    ]
    if alloc:
        rows.append("        " + alloc)
    rows.append("        " + loop_line)
    rows.append("            acc = acc + 1.0")
    rows.append("        return acc + y" if alloc else "        return acc")
    source = "\n".join(rows) + "\n"
    directory = pathlib.Path(tempfile.mkdtemp())
    path = directory / "prog.py"
    path.write_text(source, encoding="utf-8")
    return load_authored_ir(f"{path}:M"), directory


def loops_of(module):
    found = []
    for item in module.functions:
        body = getattr(item, "body", None)
        if body is not None:
            found.extend(e for e in collect_exprs(body) if isinstance(e, LoopRegion))
    return found


def only_loop(module) -> LoopRegion:
    loops = loops_of(module)
    assert len(loops) == 1, f"expected one loop, got {len(loops)}"
    return loops[0]


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "2", None, [], object()],
    ids=["true", "false", "zero", "negative", "float", "str", "none", "list", "object"],
)
def test_max_stage_must_be_a_plain_int(value):
    with pytest.raises(PipelineError) as caught:
        PipelineSpec(max_stage=value, max_buffer_versions=2)
    assert caught.value.code == "PIPELINE_MAX_STAGE_INVALID", caught.value


def test_enable_offset_must_be_a_bool():
    with pytest.raises(PipelineError) as caught:
        PipelineSpec(max_stage=2, max_buffer_versions=2, enable_offset=1)
    assert caught.value.code == "PIPELINE_ENABLE_OFFSET_INVALID", caught.value


def test_unknown_annotation_key_is_named_not_ignored():
    with pytest.raises(PipelineError) as caught:
        build("for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, unknown_flag=1):")
    assert caught.value.code == "PIPELINE_ANNOTATION_UNSUPPORTED", caught.value


def test_ordinary_range_loop_declares_no_pipeline():
    module, _ = build("for i in range(4):")
    assert only_loop(module).pipeline is None


def test_ordinary_loop_prints_as_range_never_as_pipelined():
    module, _ = build("for i in range(4):")
    printed = as_script(module)
    assert "tf.pipelined(" not in printed
    assert "range(4)" in printed


def test_pipelined_loop_carries_a_typed_spec():
    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=3, enable_offset=False):"
    )
    spec = only_loop(module).pipeline
    assert isinstance(spec, PipelineSpec)
    assert spec.max_stage == 2
    assert spec.max_buffer_versions == 3
    assert spec.enable_offset is False


def test_printer_emits_pipelined_and_never_falls_back_to_range():
    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):"
    )
    printed = as_script(module)
    assert "tf.pipelined(" in printed, printed
    assert "max_stage=2" in printed, printed
    assert "max_buffer_versions=2" in printed, printed
    assert "enable_offset=True" in printed, printed


def test_print_parse_roundtrip_preserves_the_declaration():
    module, directory = build(
        "for i in tf.pipelined(4, max_stage=3, max_buffer_versions=2, enable_offset=True):"
    )
    printed = as_script(module)
    path = directory / "roundtrip.py"
    path.write_text(printed, encoding="utf-8")
    again = load_authored_ir(f"{path}:M")
    spec = only_loop(again).pipeline
    assert spec.max_stage == 3
    assert spec.max_buffer_versions == 2
    assert spec.enable_offset is True


def test_clone_preserves_the_declaration():
    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=False):"
    )
    before = only_loop(module).pipeline
    cloned = ExprCloner().visit(only_loop(module))
    assert cloned is not None
    assert cloned.pipeline == before


def test_specialize_preserves_the_declaration():
    from tilefoundry.ir.hir.specialize import InstantiateDimensionCloner

    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=False):"
    )
    before = only_loop(module).pipeline
    function = module.functions[0]
    specialised = InstantiateDimensionCloner({}).function(function)
    after = [e for e in collect_exprs(specialised.body) if isinstance(e, LoopRegion)]
    assert after, "specialization lost the loop entirely"
    assert after[0].pipeline == before


def test_bad_control_lost_pipeline_declaration_is_caught():
    import dataclasses

    module, _ = build("for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2):")
    loop = only_loop(module)
    assert loop.pipeline is not None

    verify_pipeline_preserved(module, module)

    damaged = dataclasses.replace(loop, pipeline=None)
    assert damaged.pipeline is None

    with pytest.raises(PipelineError) as caught:
        verify_pipeline_preserved(module, _ModuleImage(damaged))
    assert caught.value.code == "PIPELINE_DECLARATION_LOST", caught.value


def test_bad_control_empty_expectation_is_refused_not_accepted():
    module, _ = build("for i in range(4):")
    with pytest.raises(PipelineError) as caught:
        verify_pipeline_preserved(module, module)
    assert caught.value.code == "PIPELINE_EXPECTATION_MISSING", caught.value


def test_bad_control_budget_applied_as_buffer_version_is_caught():
    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, "
        'buffer_schedules=(("y", "fixed", 1),)):',
        alloc='y = tf.empty(Tensor[(64,), "f32"])',
    )
    folded, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, "
        'buffer_schedules=(("y", "fixed", 2),)):',
        alloc='y = tf.empty(Tensor[(64,), "f32"])',
    )
    versions = collect_buffer_versions(module)
    assert versions
    assert all(v == 1 for v in versions.values()), versions

    with pytest.raises(PipelineError) as caught:
        verify_buffer_versions_unchanged(module, folded)
    assert caught.value.code == "BUFFER_VERSION_DECLARATION_CHANGED", caught.value


class _ModuleImage:
    def __init__(self, *bodies) -> None:
        self.functions = tuple(_FunctionImage(body) for body in bodies)


class _FunctionImage:
    def __init__(self, body) -> None:
        self.body = body


def test_verify_buffer_versions_needs_a_real_reader():
    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, "
        'buffer_schedules=(("y", "fixed", 1),)):',
        alloc='y = tf.empty(Tensor[(64,), "f32"])',
    )
    seen = collect_buffer_versions(module)
    assert seen
    for key, count in seen.items():
        assert count == 1, (key, count)


def test_tir_for_carries_pipeline_through_stmt_rebuild():
    from tilefoundry.ir.tir.stmts import For, Sequential
    from tilefoundry.ir.visitor import _rebuild_stmt_children

    module, _ = build(
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):"
    )
    spec = only_loop(module).pipeline
    loop = only_loop(module)
    rebuilt = _rebuild_stmt_children(
        For(loop.induction_var, loop.start, loop.extent, loop.step, Sequential(()), pipeline=spec),
        (Sequential(()),),
    )
    assert isinstance(rebuilt, For)
    assert rebuilt.pipeline == spec


def test_tir_for_without_pipeline_is_distinguishable_from_one_with_it():
    from tilefoundry.ir.tir.stmts import For, Sequential

    module, _ = build("for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2):")
    loop = only_loop(module)
    bare = For(loop.induction_var, loop.start, loop.extent, loop.step, Sequential(()))
    carrying = For(
        loop.induction_var,
        loop.start,
        loop.extent,
        loop.step,
        Sequential(()),
        pipeline=loop.pipeline,
    )
    assert bare.pipeline is None
    assert carrying.pipeline is not None
    assert bare != carrying


def test_symbolic_extent_is_kept_not_normalized_to_zero():
    module, _ = build(
        "for i in tf.pipelined(N, max_stage=2, max_buffer_versions=2):",
        header_extra='N = DimVar("n", 0, 64)\n',
    )
    loop = only_loop(module)
    from tilefoundry.ir.core import Constant

    extent = loop.extent
    assert not isinstance(extent, Constant) or extent.value != 0
