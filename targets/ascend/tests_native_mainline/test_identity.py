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
