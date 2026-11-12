from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.hir.pipeline import PipelineError, PipelineSpec
from tilefoundry.ir.types.tensor_type import ShapeDim


@dataclass(unsafe_hash=True)
class LoopRegion(Expr):
    """Loop-phi-shaped structured SSA folding a tile-style loop into one region.

    A region rather than a scope: the values the loop carries are its own state,
    stated once here, not something every expression inside holds a copy of.

    The first len(yield_values) params carry state; remaining params capture
    invariant values. All matching args are evaluated outside the loop.

    pipeline records authored pipelining INTENT (D0 surface only). It is None for
    an ordinary range/tf.tile loop, and nothing about it is a schedule: no offset
    plan is generated here and no solver is implied.
    """

    induction_var: Var
    params: tuple[Var, ...]
    args: tuple[Expr, ...]
    body: Expr
    yield_values: tuple[Expr, ...]
    extent: ShapeDim
    step: ShapeDim
    start: ShapeDim = 0
    pipeline: PipelineSpec | None = None

    def __post_init__(self) -> None:
        # The bounds are the SINGLE authority: a pipeline spec must never carry a
        # second copy of stop, because two copies of one expression drift apart
        # under clone and substitute. Time is bought by refusing the mismatch by
        # name here instead of debugging it after a lowering.
        if self.pipeline is not None and not isinstance(self.pipeline, PipelineSpec):
            raise PipelineError(
                "PIPELINE_DECLARATION_INVALID",
                "LoopRegion.pipeline must be a PipelineSpec or None, got %s"
                % (type(self.pipeline).__name__,),
            )

    def carries(self):
        return tuple(zip(self.params, self.args, self.yield_values))

    def captures(self) -> tuple[tuple[Var, Expr], ...]:
        """Pair invariant parameters with their arguments, excluding carries."""
        carried = len(self.yield_values)
        return tuple(zip(self.params[carried:], self.args[carried:], strict=True))


__all__ = ["LoopRegion"]
