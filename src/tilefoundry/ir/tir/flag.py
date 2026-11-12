"""Effect-form TIR Ops for async completion: per-buffer flag set/wait and a drain.

A completion edge is generic: a producer SETs a flag for a buffer slot at a generation,
and a consumer WAITs for it before touching that slot. Ring depth ("versions") is an
attribute, so the rule "slot == generation % versions" is target-agnostic.

TARGET-AGNOSTIC BY DESIGN. The channel NAME, pipe names and flag id are opaque attributes
supplied by the target (the Ascend side owns their vocabulary). The core validates only
structure: a non-empty channel name, a known completion KIND, non-negative generation,
positive versions, slot == generation % versions, non-negative flag id, and a legal
pipe pair. Per-target name/scoping checks live in the target verifier.

This module deliberately registers NO candidate and changes NO default, so targets that do
not emit these ops are unaffected.
"""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


class CompletionKind(str, Enum):
    """What a flag edge orders (target-agnostic)."""

    READY = "ready"   # producer finished writing a slot; consumer may read
    FREE = "free"     # consumer finished using a slot; producer may overwrite
    ONCE = "once"     # a one-shot ordering edge (not ring-slotted)


_ALLOWED_KINDS = {k.value for k in CompletionKind}


def _check_common(call, *, need_pipes: bool) -> None:
    t = call.target
    name = t.channel_name
    if not isinstance(name, str) or not name.strip():
        raise VerifyError("completion flag: channel name must be a non-empty string")
    if t.kind not in _ALLOWED_KINDS:
        raise VerifyError(
            "completion flag: unknown completion kind %r (expected one of %s)"
            % (t.kind, sorted(_ALLOWED_KINDS))
        )
    if not isinstance(t.generation, int) or t.generation < 0:
        raise VerifyError("completion flag: generation must be a non-negative int, got %r" % (t.generation,))
    if not isinstance(t.versions, int) or t.versions < 1:
        raise VerifyError("completion flag: versions must be a positive int, got %r" % (t.versions,))
    if not isinstance(t.slot, int) or t.slot < 0:
        raise VerifyError("completion flag: slot must be a non-negative int, got %r" % (t.slot,))
    expect = t.generation % t.versions
    if t.slot != expect:
        raise VerifyError(
            "completion flag: slot %d != generation %d %% versions %d (= %d)"
            % (t.slot, t.generation, t.versions, expect)
        )
    if not isinstance(t.flag_id, int) or t.flag_id < 0:
        raise VerifyError("completion flag: flag id must be a non-negative int, got %r" % (t.flag_id,))
    if need_pipes:
        for pname in ("producer_pipe", "consumer_pipe"):
            v = getattr(t, pname)
            if not isinstance(v, str) or not v.strip():
                raise VerifyError("completion flag: %s must be a non-empty string" % pname)
        if t.producer_pipe == t.consumer_pipe and t.kind != CompletionKind.ONCE.value:
            raise VerifyError(
                "completion flag: pipe pair %r -> %r is not a cross-engine edge (equal pipes "
                "require kind='once')" % (t.producer_pipe, t.consumer_pipe)
            )


_ATTRS = dict(
    channel_name=ParamDef(kind="attribute", annotation=str),
    kind=ParamDef(kind="attribute", annotation=str),
    slot=ParamDef(kind="attribute", annotation=int),
    generation=ParamDef(kind="attribute", annotation=int),
    versions=ParamDef(kind="attribute", annotation=int),
    flag_id=ParamDef(kind="attribute", annotation=int),
    producer_pipe=ParamDef(kind="attribute", annotation=str),
    consumer_pipe=ParamDef(kind="attribute", annotation=str),
)


@register_op(dialect="T", category="sync", name="flag_set")
class FlagSet(Op):
    """Producer SETs a completion flag for a buffer slot at a generation."""

    # The execution domain is the TARGET's, resolved from its own
    # MemoryExecutionFacts. thread_execution_mesh() hard-coded the topology name
    # "thread", the one place this module broke the target-agnostic contract its
    # own docstring states; on Ascend that made every event op fail the kernel
    # frame's aiv mesh with a declaration error rather than a program error.
    execution_mesh = utils.memory_execution_mesh()

    channel_name = _ATTRS["channel_name"]
    kind = _ATTRS["kind"]
    slot = _ATTRS["slot"]
    generation = _ATTRS["generation"]
    versions = _ATTRS["versions"]
    flag_id = _ATTRS["flag_id"]
    producer_pipe = _ATTRS["producer_pipe"]
    consumer_pipe = _ATTRS["consumer_pipe"]


@register_op(dialect="T", category="sync", name="flag_wait")
class FlagWait(Op):
    """Consumer WAITs for a completion flag on a buffer slot at a generation."""

    execution_mesh = utils.memory_execution_mesh()

    channel_name = _ATTRS["channel_name"]
    kind = _ATTRS["kind"]
    slot = _ATTRS["slot"]
    generation = _ATTRS["generation"]
    versions = _ATTRS["versions"]
    flag_id = _ATTRS["flag_id"]
    producer_pipe = _ATTRS["producer_pipe"]
    consumer_pipe = _ATTRS["consumer_pipe"]


@register_op(dialect="T", category="sync", name="drain")
class Drain(Op):
    """Final completion barrier: nothing may be outstanding past this point."""

    execution_mesh = utils.memory_execution_mesh()


@register_access_relation(FlagSet)
@register_access_relation(FlagWait)
@register_access_relation(Drain)
def _(call, ctx):
    """The completion ops carry no tensor operands, so they induce no access edges."""
    return identity_relations(call, ctx)


@register_typeinfer(FlagSet)
@register_typeinfer(FlagWait)
@register_typeinfer(Drain)
def _(call, ctx) -> UnitType:
    return UnitType()


@register_verify_stmt(FlagSet)
def _verify_flag_set(call, ctx) -> None:
    _check_common(call, need_pipes=True)


@register_verify_stmt(FlagWait)
def _verify_flag_wait(call, ctx) -> None:
    _check_common(call, need_pipes=True)


@register_verify_stmt(Drain)
def _verify_drain(call, ctx) -> None:
    # A drain carries no operands or attributes; its position is checked at function
    # level by the target verifier (it must be the final statement and must follow all
    # flag waits).
    return None


__all__ = ["CompletionKind", "FlagSet", "FlagWait", "Drain"]
