"""AIV completion ops, bound to the native execution mesh.

This module contains only the three wrapped core ops the target needs. The old
static finite-domain completion trace and its materializer adapter are archived.
"""
from __future__ import annotations

from tilefoundry.ir.pattern import utils as _pattern_utils
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait, _check_common as _flag_check_common
from tilefoundry.ir.types import UnitType as _UnitType
from tilefoundry.visitor_registry import (
    register_typeinfer as _register_typeinfer,
    register_verify_stmt as _register_verify_stmt,
)
from tilefoundry.visitor_registry.access_relation import (
    identity_relations as _identity_relations,
    register_access_relation as _register_access_relation,
)

__all__ = ["AivDrain", "AivFlagSet", "AivFlagWait"]


class AivFlagSet(FlagSet):
    execution_mesh = _pattern_utils.memory_execution_mesh()


class AivFlagWait(FlagWait):
    execution_mesh = _pattern_utils.memory_execution_mesh()


class AivDrain(Drain):
    execution_mesh = _pattern_utils.memory_execution_mesh()


@_register_access_relation(AivFlagSet)
@_register_access_relation(AivFlagWait)
@_register_access_relation(AivDrain)
def _aiv_access(call, ctx):
    return _identity_relations(call, ctx)


@_register_typeinfer(AivFlagSet)
@_register_typeinfer(AivFlagWait)
@_register_typeinfer(AivDrain)
def _aiv_type(call, ctx):
    return _UnitType()


@_register_verify_stmt(AivFlagSet)
@_register_verify_stmt(AivFlagWait)
def _aiv_verify(call, ctx):
    _flag_check_common(call, need_pipes=True)


@_register_verify_stmt(AivDrain)
def _aiv_drain_verify(call, ctx):
    return None
