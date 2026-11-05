"""D0 pipeline SURFACE: declaration carried, printed, cloned -- never invented."""

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

HEADER = 'from __future__ import annotations\n\nfrom tilefoundry.dsl import *\nfrom tilefoundry.target import CpuTarget\n'


def build(loop_line: str, *, alloc: str = "", entry: str = "k", header_extra: str = ""):
    """Parse a one-loop program and return (module, directory)."""
    rows = [
        HEADER + header_extra,
        '@module(entry="%s", target=CpuTarget())' % entry,
        "class M:",
        "    @func",
        '    def %s(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:' % entry,
        "        acc = x",
    ]
    if alloc:
        rows.append("        " + alloc)
    rows.append("        " + loop_line)
    rows.append("            acc = acc + 1.0")
    # The allocation must be LIVE or parse-time dead-code elimination removes it,
    # and a reader that finds nothing would make the buffer control vacuous.
    rows.append("        return acc + y" if alloc else "        return acc")
    source = "\n".join(rows) + "\n"
    directory = pathlib.Path(tempfile.mkdtemp())
    path = directory / "prog.py"
    path.write_text(source)
    return load_authored_ir("%s:M" % path), directory


def loops_of(module):
    found = []
    for item in module.functions:
        body = getattr(item, "body", None)
        if body is not None:
            found.extend(e for e in collect_exprs(body) if isinstance(e, LoopRegion))
    return found


def only_loop(module) -> LoopRegion:
    loops = loops_of(module)
    assert len(loops) == 1, "expected exactly one loop, got %d" % len(loops)
    return loops[0]


# --- the spec refuses what D0 cannot represent -------------------------------


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "2", None, [], object()],
    ids=["true", "false", "zero", "negative", "float", "str", "none", "list", "object"],
)
def test_num_stages_must_be_a_plain_int(value):
    """bool is an int subclass, so `True` would sail through as 1 without this."""
    with pytest.raises(PipelineError) as caught:
        PipelineSpec(num_stages=value)
    assert caught.value.code == "PIPELINE_NUM_STAGES_INVALID", caught.value


def test_enable_offset_must_be_a_bool():
    with pytest.raises(PipelineError) as caught:
        PipelineSpec(num_stages=2, enable_offset=1)
    assert caught.value.code == "PIPELINE_ENABLE_OFFSET_INVALID", caught.value


def test_unknown_annotation_key_is_named_not_ignored():
    with pytest.raises(PipelineError) as caught:
        PipelineSpec.from_surface(2, {"num_stages": 4})
    assert caught.value.code == "PIPELINE_ANNOTATION_UNSUPPORTED", caught.value


# --- a plain loop stays plain (the regression that must never move) ----------


def test_ordinary_range_loop_declares_no_pipeline():
    module, _ = build("for i in range(4):")
    assert only_loop(module).pipeline is None


def test_ordinary_loop_prints_as_range_never_as_pipelined():
    module, _ = build("for i in range(4):")
    printed = as_script(module)
    assert "tf.pipelined(" not in printed
    assert "range(4)" in printed


# --- the declaration is typed, printed and re-parsed -------------------------


def test_pipelined_loop_carries_a_typed_spec():
    module, _ = build('for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):')
    spec = only_loop(module).pipeline
    assert isinstance(spec, PipelineSpec)
    assert spec.num_stages == 2
    assert spec.enable_offset is False
    assert spec.schema_version == "pipeline/1"


def test_printer_emits_pipelined_and_never_falls_back_to_range():
    module, _ = build('for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": True}):')
    printed = as_script(module)
    assert "tf.pipelined(" in printed, printed
    assert "num_stages=2" in printed, printed
    assert "enable_offset" in printed, "False must be printed explicitly too"


def test_print_parse_roundtrip_preserves_the_declaration():
    module, directory = build('for i in tf.pipelined(4, num_stages=3, annotations={"enable_offset": True}):')
    printed = as_script(module)
    path = directory / "roundtrip.py"
    path.write_text(printed)
    again = load_authored_ir("%s:M" % path)
    spec = only_loop(again).pipeline
    assert spec.num_stages == 3
    assert spec.enable_offset is True


# --- clone and specialize must not drop it -----------------------------------


def test_clone_preserves_the_declaration():
    module, _ = build('for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):')
    before = only_loop(module).pipeline
    cloned = ExprCloner().visit(only_loop(module))
    assert cloned is not None
    assert cloned.pipeline == before


def test_specialize_preserves_the_declaration():
    from tilefoundry.ir.hir.specialize import InstantiateDimensionCloner

    module, _ = build('for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):')
    before = only_loop(module).pipeline
    function = module.functions[0]
    specialised = InstantiateDimensionCloner({}).function(function)
    after = [e for e in collect_exprs(specialised.body) if isinstance(e, LoopRegion)]
    assert after, "specialization lost the loop entirely"
    assert after[0].pipeline == before


# --- BAD CONTROL 1: a dropped declaration must be caught by name -------------


def test_bad_control_lost_pipeline_declaration_is_caught():
    """A copy that drops pipeline MUST fail as PIPELINE_DECLARATION_LOST.

    This is the control the D0 surface exists to make possible. Without it the
    preservation test above could pass on a program that never had a pipeline.
    """
    import dataclasses

    module, _ = build('for i in tf.pipelined(4, num_stages=2):')
    loop = only_loop(module)
    assert loop.pipeline is not None

    # Positive sanity first: the undamaged image satisfies the check.
    verify_pipeline_preserved(module, module)

    damaged = dataclasses.replace(loop, pipeline=None)
    assert damaged.pipeline is None, "the control did not actually drop the spec"

    with pytest.raises(PipelineError) as caught:
        verify_pipeline_preserved(module, _ModuleImage(damaged))
    assert caught.value.code == "PIPELINE_DECLARATION_LOST", caught.value


def test_bad_control_empty_expectation_is_refused_not_accepted():
    """An expectation measured off the checked side would pass by shrinking."""
    module, _ = build("for i in range(4):")  # no pipeline at all
    with pytest.raises(PipelineError) as caught:
        verify_pipeline_preserved(module, module)
    assert caught.value.code == "PIPELINE_EXPECTATION_MISSING", caught.value


# --- BAD CONTROL 2: budget must not become a buffer version count ------------


def test_bad_control_budget_applied_as_buffer_version_is_caught():
    """budget 2 must NOT silently become "every buffer has 2 versions"."""
    module, _ = build(
        'for i in tf.pipelined(4, num_stages=2):',
        alloc="y = tf.empty(Tensor[(64,), \"f32\"], buffers=1)",
    )
    versions = collect_buffer_versions(module)
    assert versions, "the fixture declares no buffer, so the control would be vacuous"
    assert all(v == 1 for v in versions.values()), versions

    # Simulate the defect: the budget was folded into the buffer declarations.
    folded = _VersionsAsBudget(versions, budget=2)
    with pytest.raises(PipelineError) as caught:
        verify_buffer_versions_unchanged(module, folded)
    assert caught.value.code == "BUFFER_VERSION_DECLARATION_CHANGED", caught.value


class _ModuleImage:
    """A module-shaped view holding one body, so the verifier can compare sides.

    Deliberately a VIEW and not a clone: the point of the control is that the
    expected side comes from the authored program, so a check that measured
    expectations off the image being checked could never notice a loss.
    """

    def __init__(self, *bodies) -> None:
        self.functions = tuple(_FunctionImage(body) for body in bodies)


class _FunctionImage:
    def __init__(self, body) -> None:
        self.body = body


class _VersionsAsBudget:
    """Stands in for a lowering that re-read every buffer as `budget` versions."""

    def __init__(self, versions, budget: int) -> None:
        self.functions = ()
        self._versions = {key: budget for key in versions}


def test_verify_buffer_versions_needs_a_real_reader(monkeypatch):
    """Guard the plumbing: the reader is what makes the control non-vacuous."""
    module, _ = build(
        "for i in range(4):",
        alloc="y = tf.empty(Tensor[(64,), \"f32\"], buffers=1)",
    )
    seen = collect_buffer_versions(module)
    assert seen, "the reader found no buffer declarations in a program that has one"
    for key, count in seen.items():
        assert count == 1, (key, count)

# --- TIR For carries the declaration through rebuild ------------------------


def test_tir_for_carries_pipeline_through_stmt_rebuild():
    """The lowering path builds TIR For with the spec; a rebuild must keep it.

    _rebuild_stmt_children is how every TIR clone/rewrite rebuilds a statement. If
    it dropped the field the declaration would vanish on the first rewrite and
    nothing downstream would notice, because the For would still be well formed.
    """
    from tilefoundry.ir.tir.stmts import For, Sequential
    from tilefoundry.ir.visitor import _rebuild_stmt_children

    module, _ = build('for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": True}):')
    spec = only_loop(module).pipeline
    loop = only_loop(module)
    rebuilt = _rebuild_stmt_children(
        For(loop.induction_var, loop.start, loop.extent, loop.step, Sequential(()), pipeline=spec),
        (Sequential(()),),
    )
    assert isinstance(rebuilt, For)
    assert rebuilt.pipeline == spec, "the rebuild dropped the pipeline declaration"


def test_tir_for_without_pipeline_is_distinguishable_from_one_with_it():
    """The field must actually differ, or every loss check would be vacuous."""
    from tilefoundry.ir.tir.stmts import For, Sequential

    module, _ = build('for i in tf.pipelined(4, num_stages=2):')
    loop = only_loop(module)
    bare = For(loop.induction_var, loop.start, loop.extent, loop.step, Sequential(()))
    carrying = For(
        loop.induction_var, loop.start, loop.extent, loop.step, Sequential(()),
        pipeline=loop.pipeline,
    )
    assert bare.pipeline is None
    assert carrying.pipeline is not None
    assert bare != carrying, "a dropped declaration must not compare equal to a kept one"


# --- a symbolic bound is NOT a zero -----------------------------------------


def test_symbolic_extent_is_kept_not_normalized_to_zero():
    """UNKNOWN must never be resolved into a concrete value, least of all 0.

    A bound silently read as 0 makes a pipelined loop do nothing while the
    declaration still claims a budget, so the run and the record disagree.
    """
    module, _ = build(
        'for i in tf.pipelined(N, num_stages=2):',
        header_extra="N = DimVar(\"n\", 0, 64)\n",
    )
    loop = only_loop(module)
    from tilefoundry.ir.core import Constant

    extent = loop.extent
    assert not isinstance(extent, Constant) or extent.value != 0, (
        "the symbolic extent collapsed to a constant (0), so UNKNOWN was defaulted: %r"
        % (extent,),
    )
