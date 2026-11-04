"""Unit tests for the core completion ops (tir.flag).

Each invalid form must be rejected AT IR LEVEL by verify_prim_function, not by a text
matcher: unknown channel/kind, bad generation, bad slot, bad flag id, bad pipe pair.
"""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.flag import CompletionKind, Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Sequential
from tilefoundry.ir.types import Layout, Mesh, Topology, make_tensor_type
from tilefoundry.visitor_registry.verify import verify_prim_function

_MESH = Mesh((Topology("thread", 1),), Layout((1,), (1,)))
_BIND = Var(type=make_tensor_type((), storage="rmem"), name="m")


def _pf(op) -> PrimFunction:
    body = MeshScope(mesh=_MESH, binding=_BIND,
                     body=Sequential(body=(Evaluate(callable=op, args=()),)))
    return PrimFunction(name="fn", params=(), body=Sequential(body=(body,)))


def _flag(**over) -> FlagSet:
    kw = dict(channel_name="x_ready", kind="ready", slot=0, generation=0,
              versions=2, flag_id=3, producer_pipe="mte2", consumer_pipe="vector")
    kw.update(over)
    return FlagSet(**kw)


def test_valid_flag_and_wait_pass() -> None:
    verify_prim_function(_pf(_flag()))
    verify_prim_function(_pf(FlagSet(**{**_flag().__dict__, "channel_name": "x_free"})))


def test_unknown_channel_name_rejected() -> None:
    with pytest.raises(VerifyError, match="channel name"):
        verify_prim_function(_pf(_flag(channel_name="")))


def test_unknown_kind_rejected() -> None:
    with pytest.raises(VerifyError, match="unknown completion kind"):
        verify_prim_function(_pf(_flag(kind="bogus")))


def test_bad_generation_rejected() -> None:
    with pytest.raises(VerifyError, match="generation must be a non-negative"):
        verify_prim_function(_pf(_flag(generation=-1)))


def test_bad_slot_rejected() -> None:
    with pytest.raises(VerifyError, match="generation"):
        verify_prim_function(_pf(_flag(generation=4, versions=2, slot=1)))


def test_bad_flag_id_rejected() -> None:
    with pytest.raises(VerifyError, match="flag id must be a non-negative"):
        verify_prim_function(_pf(_flag(flag_id=-2)))


def test_bad_pipe_pair_rejected() -> None:
    with pytest.raises(VerifyError, match="cross-engine edge"):
        verify_prim_function(_pf(_flag(producer_pipe="vector", consumer_pipe="vector")))


def test_same_pipe_allowed_for_once() -> None:
    verify_prim_function(_pf(_flag(kind=CompletionKind.ONCE.value,
                                   producer_pipe="vector", consumer_pipe="vector")))


def test_drain_is_inert() -> None:
    verify_prim_function(_pf(Drain()))
