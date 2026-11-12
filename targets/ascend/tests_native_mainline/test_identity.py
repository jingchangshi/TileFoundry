"""Identity must read ACTUAL Op attributes and Var binders.

R137-pre1: _function_identity used to store only {kind, name} for a real,
non-dataclass Op, so every instruction attribute was invisible. Two legal
instruction bodies differing only in Vadds(value=1.0 / 7.0) hashed identically
even though both verify. These tests are written against the real AR native
builder (not a hand-built fake IR), because the collision was reproduced there.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Sequential
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_ascend import Ascend950PRTarget
from tilefoundry_ascend.ir.ops import Pset, Vadds, Vcvt, Vld, Vmul
from tilefoundry_ascend.plan.identity import function_identity, module_identity
from tilefoundry_ascend.passes.p3_chain import ir_identity

ROOT = Path(__file__).resolve().parents[1] / "tilefoundry_ascend"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ar = _load("_identity_attnres_build", "attnres/build.py")

SPEC = dict(n=6, tokens=333, d=2047, token_rows=8, block_n=4, v_versions=2,
            family=ar.BLOCKED_ONLINE, score_mode=ar.FACTORED_GW)


def _replace_vmul(node, value):
    """Replace the AR builder's Vmul(x, y, mask, dst) with Vadds(x, mask, dst)."""
    if isinstance(node, Evaluate) and isinstance(node.callable, Vmul):
        a, _b, mask, dst = node.args
        return Evaluate(callable=Vadds(value=value), args=(a, mask, dst))
    if isinstance(node, tuple):
        return tuple(_replace_vmul(item, value) for item in node)
    if not is_dataclass(node):
        return node
    changes = {}
    for field in fields(node):
        if field.name in ("type", "metadata", "annotation", "origin"):
            continue
        before = getattr(node, field.name, None)
        after = _replace_vmul(before, value)
        if after is not before:
            changes[field.name] = after
    return dataclasses.replace(node, **changes) if changes else node


def _ar_module(value):
    module = ar.build_attnres_module(ar.AttnResSpec(**SPEC))
    function = dataclasses.replace(
        module.entry_function(), body=_replace_vmul(module.entry_function().body, value)
    )
    rewritten = dataclasses.replace(module, functions=(function,))
    verify_prim_function(function, module_fns=rewritten)
    return rewritten


def _lane():
    return make_tensor_type((64,), DType.f32, "rmem")


def _vadds_function(name, params, body):
    return PrimFunction(
        name,
        tuple(params),
        Sequential(tuple(body)),
        output_count=1,
        target=Ascend950PRTarget(),
    )


def test_two_legal_vadds_values_have_different_identity():
    one = _ar_module(1.0)
    seven = _ar_module(7.0)
    assert function_identity(one.entry_function()) != function_identity(seven.entry_function())
    assert module_identity(one) != module_identity(seven)
    assert ir_identity(one) != ir_identity(seven)


def test_same_vadds_value_is_the_same_identity():
    left = _ar_module(1.0)
    right = _ar_module(1.0)
    assert function_identity(left.entry_function()) == function_identity(right.entry_function())
    assert ir_identity(left) == ir_identity(right)


def test_op_attributes_of_conversion_and_predicate_are_covered():
    a, m, d = Var("a", type=_lane()), Var("m", type=_lane()), Var("d", type=_lane())
    body_one = (
        Evaluate(callable=Vld(dist="unit"), args=(a, d)),
        Evaluate(callable=Pset(elem_width=32), args=(m,)),
        Evaluate(callable=Vcvt(direction="unpk_b16", part=0, elem_width=16), args=(a, m, d)),
    )
    body_two = (
        Evaluate(callable=Vld(dist="brc"), args=(a, d)),
        Evaluate(callable=Pset(elem_width=16), args=(m,)),
        Evaluate(callable=Vcvt(direction="unpk_b16", part=1, elem_width=16), args=(a, m, d)),
    )
    one = _vadds_function("k", (a, m, d), body_one)
    two = _vadds_function("k", (a, m, d), body_two)
    assert function_identity(one) != function_identity(two)


def test_same_name_different_binder_is_not_the_same_identity():
    """Spelling is not a binder: a param v and a let v are different programs."""
    a = Var("a", type=_lane())
    v_param = Var("v", type=_lane())
    mask = Var("mask", type=_lane())

    def with_param():
        return _vadds_function(
            "k",
            (a, v_param, mask),
            (Evaluate(callable=Vadds(value=1.0), args=(a, v_param, mask)),),
        )

    v_let = Var("v", type=_lane())
    b = Var("b", type=_lane())

    def with_let():
        return _vadds_function(
            "k",
            (a, b, mask),
            (
                LetStmt(var=v_let, value=b),
                Evaluate(callable=Vadds(value=1.0), args=(a, v_let, mask)),
            ),
        )

    assert function_identity(with_param()) != function_identity(with_let())


def test_plain_dataclass_content_is_still_in_the_identity():
    """A change that is not an Op attribute must still change the identity."""
    a = Var("a", type=_lane())
    left = _vadds_function(
        "k", (a,), (Evaluate(callable=Vadds(value=1.0), args=(a, a, a)),)
    )
    right = _vadds_function(
        "j", (a,), (Evaluate(callable=Vadds(value=1.0), args=(a, a, a)),)
    )
    assert function_identity(left) != function_identity(right)

# ------------------------------------------------------------ S1 encoder tests
import json

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.core.module import Module
from tilefoundry.visitor_registry.access_relation import AccessRelation
from tilefoundry_ascend.plan.identity import (
    IDENTITY_ENCODING_VERSION,
    IDENTITY_UNSUPPORTED_VALUE,
    IdentityRefusal,
    source_bytes,
    target_bytes,
    _encode,
)

FIXTURES = [
    ("targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py",
     "MinimalAivPipeline"),
    ("targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py",
     "IndependentUB"),
]


def _fixture_module(path: str, name: str):
    program = checked_program("%s:%s" % (path, name), None, command="test identity")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    return finalize(analyzed, authored=program.authored)


def test_source_bytes_are_repeatedly_identical_for_real_fixtures():
    for path, name in FIXTURES:
        module = _fixture_module(path, name)
        entry = module.entry_function()
        first = source_bytes(entry)
        second = source_bytes(entry)
        assert first == second, path
        assert json.loads(first)["enc"] == IDENTITY_ENCODING_VERSION
        assert source_bytes(entry) == module_bytes_1(module)


def module_bytes_1(module):
    return source_bytes(module.entry_function())


def test_a_real_content_change_changes_the_source_bytes():
    module = _fixture_module(FIXTURES[0][0], FIXTURES[0][1])
    entry = module.entry_function()
    original = source_bytes(entry)
    changed = source_bytes(dataclasses.replace(entry, output_count=entry.output_count + 1))
    assert original != changed


def test_target_bytes_come_from_the_actual_instance_facts():
    left = Ascend950PRTarget()
    right = Ascend950PRTarget()
    assert target_bytes(left) == target_bytes(right)
    payload = json.dumps(json.loads(target_bytes(left)))
    # the declared smem capacity is present as an exact integer, not a repr
    assert "262144" in payload
    shrunk = Ascend950PRTarget(aiv_count=1, aic_count=1)
    assert target_bytes(shrunk) != target_bytes(left)


def test_dangling_var_is_refused_and_never_falls_back_to_its_name():
    a = Var("a", type=_lane())
    ghost = Var("ghost", type=_lane())
    function = _vadds_function(
        "k", (a,), (Evaluate(callable=Vadds(value=1.0), args=(a, ghost, a)),)
    )
    with pytest.raises(IdentityRefusal) as caught:
        source_bytes(function)
    assert caught.value.code == IDENTITY_UNSUPPORTED_VALUE
    assert "lexical scope" in str(caught.value)
    assert "ghost" not in str(caught.value)


def test_float_bit_patterns_are_distinguished():
    a = Var("a", type=_lane())

    def body(value):
        return _vadds_function(
            "k", (a,), (Evaluate(callable=Vadds(value=value), args=(a, a, a)),)
        )

    assert source_bytes(body(0.0)) != source_bytes(body(-0.0))
    assert source_bytes(body(1.0)) != source_bytes(body(1.0 + 2 ** -52))


def test_isl_relation_is_read_positionally_not_by_parameter_name():
    import isl

    from tilefoundry.ir.isl_interop import shape_to_isl_set
    from tilefoundry.ir.types.dim import DimVar
    from tilefoundry.visitor_registry.access_relation import _parameter_names

    extent = DimVar(name="n", lo=0, hi=64)
    values = {}
    box = shape_to_isl_set((extent,), values)
    relation = isl.map.identity(box.get_space().map_from_set()).intersect_domain(box)
    name = _parameter_names(relation)[0]
    bound = AccessRelation(relation, {name: extent})
    renamed = relation.set_dim_name(isl.dim_type.PARAM, 0, "zz9")
    assert _parameter_names(renamed) == ["zz9"]
    renamed_bound = AccessRelation(renamed, {"zz9": extent})
    # Same position, same binding: an alpha-renamed parameter is the same content.
    assert _encode(bound, {}, "$.a", 0) == _encode(renamed_bound, {}, "$.a", 0)
    # Same position, different binding: the identity must move.
    other = DimVar(name="m", lo=0, hi=128)
    rebound = AccessRelation(renamed, {"zz9": other})
    assert _encode(bound, {}, "$.a", 0) != _encode(rebound, {}, "$.a", 0)


def test_unsupported_value_refuses_with_a_field_path():
    class Opaque:
        pass

    with pytest.raises(IdentityRefusal) as caught:
        _encode(Opaque(), {}, "$.opaque", 0)
    assert caught.value.code == IDENTITY_UNSUPPORTED_VALUE
    assert caught.value.path == "$.opaque"
    assert "no canonical encoding" in str(caught.value)



def test_a_declined_fact_is_recorded_structurally_not_as_prose():
    """Wording is not meaning: rewording a refusal must not move identity.

    A target that answers "I have no projection for this fact class" states a
    fact, but the sentence is human-readable text. Identity bytes carry the
    refusal kind and the fact class name, so editing the message changes
    nothing - while a target that genuinely GAINS a projection still changes
    the identity, because that is a real difference.
    """
    import tilefoundry.target.base as base

    module = _fixture_module(FIXTURES[0][0], FIXTURES[0][1])
    entry = module.entry_function()
    source = source_bytes(entry)
    target = target_bytes(entry.target)
    assert b"no projection" not in source and b"no projection" not in target

    real = base.UnsupportedCapabilityError

    class Reworded(real):
        def __init__(self, *args):
            super().__init__("an entirely different human sentence about "
                             "the same refusal")

    import tilefoundry_ascend.tf_target as tf
    base.UnsupportedCapabilityError = Reworded
    tf.UnsupportedCapabilityError = Reworded
    try:
        again_source = source_bytes(entry)
        again_target = target_bytes(entry.target)
    finally:
        base.UnsupportedCapabilityError = real
        tf.UnsupportedCapabilityError = real
    assert again_source == source, "a reworded refusal moved the source identity"
    assert again_target == target, "a reworded refusal moved the target identity"
    assert b"declined" in target


def test_an_unexpected_fact_failure_is_never_swallowed_as_declined():
    """Only the declared refusal kind becomes a record; a real bug propagates."""
    module = _fixture_module(FIXTURES[0][0], FIXTURES[0][1])
    entry = module.entry_function()
    before = target_bytes(entry.target)
    cls = type(entry.target)
    real = cls.get_facts

    def breaking(self, facts_cls):
        raise AssertionError("a different failure is not a declined fact")

    cls.get_facts = breaking
    try:
        with pytest.raises(AssertionError):
            target_bytes(entry.target)
    finally:
        cls.get_facts = real
    assert target_bytes(entry.target) == before


def test_the_model_surface_does_not_enter_source_identity():
    """A cost model is not program content.

    Editing the ISA latency of an instruction this program uses must not
    change the program's source bytes. It does change the problem and plan
    identities, where the model belongs (see test_delays.py and
    test_resolved.py).
    """
    import dataclasses

    from tilefoundry_ascend.ir.isa import ISA
    from tilefoundry_ascend.plan.delays import delay_table_bytes

    module = _fixture_module(FIXTURES[0][0], FIXTURES[0][1])
    entry = module.entry_function()
    baseline = source_bytes(entry)
    row = ISA["gm_to_ub"]
    ISA["gm_to_ub"] = dataclasses.replace(row, latency_model=row.latency_model + 7)
    try:
        assert source_bytes(entry) == baseline
        assert delay_table_bytes() != baseline  # the table is model, not program
    finally:
        ISA["gm_to_ub"] = row


def test_event_contracts_are_declared_without_entering_program_identity():
    """Step-1 scope: a contract row is a stated target fact, not program text.

    Adding or removing an event-domain row must not change an existing
    program's source bytes - those rows describe ops the program does not
    contain yet. When completion injects real flag/drain statements, the
    program's OWN content changes and so does its identity, through the
    statements, not through the table.
    """
    from tilefoundry_ascend.ir.isa import EVENT_CONTRACT_NAMES, ISA

    module = _fixture_module(FIXTURES[0][0], FIXTURES[0][1])
    entry = module.entry_function()
    baseline = source_bytes(entry)
    saved = {name: ISA.pop(name) for name in EVENT_CONTRACT_NAMES}
    try:
        assert source_bytes(entry) == baseline
    finally:
        ISA.update(saved)
    assert all(name in ISA for name in EVENT_CONTRACT_NAMES)


def test_a_target_without_the_capabilities_refuses_by_name():
    from tilefoundry.target import CpuTarget

    with pytest.raises(IdentityRefusal) as caught:
        target_bytes(CpuTarget())
    assert caught.value.code == IDENTITY_UNSUPPORTED_VALUE
    assert "does not publish" in str(caught.value)
    # a PrimFunction over such a target refuses through the source boundary too
    lane = make_tensor_type((64,), DType.f32, "rmem")
    a = Var("a", type=lane)
    function = PrimFunction("k", (a,), Sequential((
        Evaluate(callable=Vadds(value=1.0), args=(a, a, a)),)),
        output_count=1, target=CpuTarget())
    with pytest.raises(IdentityRefusal) as caught2:
        source_bytes(function)
    assert "does not publish" in str(caught2.value)



def test_isa_facts_reach_the_problem_but_never_the_program_identity():
    """The separation is bidirectional.

    _isa_rows reads the target instruction table for the instructions the
    program actually uses, so editing an ISA row moves model_version and
    therefore the problem and plan identities. It must NOT move source_bytes,
    which describe the program, nor target_bytes, which describe the machine.
    """
    import dataclasses

    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.ir.isa import ISA
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    def resolved():
        program = checked_program(FIXTURES[1][0] + ":" + FIXTURES[1][1],
                                 None, command="test model separation")
        analyzed = analyze(program.module, program.entry,
                           analysis=("compute-cost", "memory"))
        module = finalize(analyzed, authored=program.authored)
        problem = analyze_native_problem(
            build_native_unit_tree(module, authored=program.authored), None, None,
            module.entry_function().target)
        answers = {region.region_ref.region_id:
                   solve_region(to_schedule_problem(region))
                   for region in problem.regions}
        return module, resolve_schedule(problem, answers)

    module, plan = resolved()
    program_bytes = source_bytes(module.entry_function())
    machine_bytes = target_bytes(module.entry_function().target)
    row = ISA["gm_to_ub"]
    ISA["gm_to_ub"] = dataclasses.replace(row, latency_model=999)
    try:
        module2, plan2 = resolved()
        assert source_bytes(module2.entry_function()) == program_bytes
        assert target_bytes(module2.entry_function().target) == machine_bytes
        assert plan2.model_version != plan.model_version
        assert plan2.problem_identity != plan.problem_identity
        assert plan2.plan_identity != plan.plan_identity
    finally:
        ISA["gm_to_ub"] = row
    module3, plan3 = resolved()
    assert plan3.plan_identity == plan.plan_identity, "restoring must restore it"
