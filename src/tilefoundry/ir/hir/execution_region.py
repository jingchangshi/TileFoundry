"""HIR execution region: a body computed in one declared execution mode.

Not a mesh, and not a set of participants. A MeshRegion states *who* runs a
piece of work; an ExecutionRegion states *in which execution mode* it runs, and
therefore what may cross its boundary at all. The values the region captures are
evaluated outside it and are named by params; what the body computes leaves
through body only. That single exit is what makes register non-escape a
structural property instead of a convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.execution import ExecutionKind, execution_kind


@dataclass(unsafe_hash=True)
class ExecutionRegion(Expr):
    """A region of work in one declared execution mode.

    kind names a registered ExecutionKind; params and args are the captured
    values in the same shape every other region states them, so isolation is
    checked the same way. The region types as its body does: which mode ran the
    work is not a fact about the shape of its result.
    """

    kind: str
    params: tuple[Var, ...]
    args: tuple[Expr, ...]
    body: Expr

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("ExecutionRegion: kind must be a non-empty string")
        if len(self.params) != len(self.args):
            raise ValueError(
                f"ExecutionRegion {self.kind!r}: {len(self.params)} params but "
                f"{len(self.args)} args"
            )

    def captures(self) -> tuple[tuple[Var, Expr], ...]:
        """Pair every region parameter with the argument it stands for."""
        return tuple(zip(self.params, self.args, strict=True))

    def declaration(self) -> ExecutionKind | None:
        """The registered declaration of this region's kind, if any."""
        return execution_kind(self.kind)


__all__ = ["ExecutionRegion"]
