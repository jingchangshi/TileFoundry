"""Define structural TIR statements.

Parser assignment sugar lowers to ``LetStmt``; allocations are expression Ops
anchored as its value. ``Sequential`` packs a tuple into one Stmt so every body
uses a uniform statement slot while remaining iterable.

See [tir §1.2](docs/spec/tir.md#12-structural-stmts-tirstmts).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.types.mesh import Mesh

if TYPE_CHECKING:  # the HIR package auto-imports, so a runtime import here cycles
    from tilefoundry.ir.hir.pipeline import PipelineSpec


@dataclass(unsafe_hash=True)
class Sequential(Stmt):
    """Wrap a ``tuple[Stmt, ...]`` as one TIR statement."""

    body: tuple[Stmt, ...]

    def __iter__(self):
        return iter(self.body)

    def __len__(self) -> int:
        return len(self.body)

    def __getitem__(self, idx):
        return self.body[idx]


@dataclass(unsafe_hash=True)
class LetStmt(Stmt):
    """Bind var to value for the following statements of the enclosing Sequential."""

    var: Var
    value: Expr


@dataclass(unsafe_hash=True)
class For(Stmt):
    """A counted loop.

    pipeline is a CORE, target-neutral optional field recording that the loop was
    AUTHORED as pipelined. A D0 for still runs serially: the field is carried, not
    consumed, and it must never be read as "a solver produced a schedule". Keeping
    it on the statement (rather than as debug sidecar metadata) means a TIR
    roundtrip rebuilds it explicitly instead of losing it and reinstating a
    default.
    """

    induction_var: Var
    start: Expr
    stop: Expr
    step: Expr
    body: Sequential
    pipeline: PipelineSpec | None = None


@dataclass(unsafe_hash=True)
class While(Stmt):
    cond: Expr
    body: Sequential


@dataclass(unsafe_hash=True)
class If(Stmt):
    cond: Expr
    then_body: Sequential
    else_body: Sequential


@dataclass(unsafe_hash=True)
class MeshScope(Stmt):
    mesh: Mesh
    binding: Var
    body: Sequential


@dataclass(unsafe_hash=True)
class ExecutionScope(Stmt):
    """Run body in one declared execution mode; not a mesh, opens no participants.

    The scope is the TIR spelling of an HIR execution region: it fixes the
    execution mode that the operations inside it are checked against, and its
    boundary is where register-private values must stop.
    """

    kind: str

    body: Sequential

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("ExecutionScope: kind must be a non-empty string")
        if not isinstance(self.body, Sequential):
            raise TypeError(
                f"ExecutionScope {self.kind!r}: body must be a Sequential, "
                f"got {type(self.body).__name__}"
            )


@dataclass(unsafe_hash=True)
class Return(Stmt):
    """Empty return; tir functions have no value return."""


@dataclass(unsafe_hash=True)
class Evaluate(Stmt):
    """Place a value-less effect Op or retained function invocation in statement position."""

    callable: "Op | SymbolRef | PrimFunction | ModuleList | Call"  # noqa: A003 -- spec field name
    args: tuple[Expr, ...]


__all__ = [
    "Sequential",
    "LetStmt",
    "For",
    "While",
    "If",
    "MeshScope",
    "ExecutionScope",
    "Return",
    "Evaluate",
]
