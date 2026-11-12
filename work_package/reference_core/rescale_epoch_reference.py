"""Tiny logical completion model for deferred-rescale anchor boundaries.

Port the invariants into existing TF state effects/completion/version passes.
This is not production IR, a solver, a device simulator, or hardware proof.
The caller supplies the COMPLETE PV producer set, including output columns
and probability hi/lo components. ``complete_pv`` asserts actual completion;
calling it does not prove a device has completed anything.

Each declared column represents one output scalar per row for tiny tests.
Epochs here have finite initialized anchors. Empty-row initialization belongs
to online_softmax.py. Precomputation may run ahead on hardware, but this model
commits new-epoch mass/PV only after the old-scale boundary has retired.
Python float is mathematical reference arithmetic, not FP32 emulation.
"""
from dataclasses import dataclass
from math import exp, isclose, isfinite


@dataclass(frozen=True)
class Producer:
    block: int
    column: int
    component: str = "single"


@dataclass(frozen=True)
class Epoch:
    ordinal: int
    anchors: tuple[float, ...]
    producers: frozenset[Producer]


@dataclass(frozen=True)
class Boundary:
    next_epoch: Epoch
    predicate: bool
    factors: tuple[float, ...]


class ProtocolError(ValueError):
    pass


class EpochProtocol:
    """One group: pending L0C, prefix UB, denominator, immutable control payload.

    Metadata consumers are logical boundary users, e.g. AIC drain control and
    both AIV factor readers. Retirement means their REAL final use completed.
    It is deliberately conservative to retire them only after drain_and_clear.
    A no-change boundary is allowed as a conservative drain with all factors 1.
    """
    def __init__(self, epoch, columns, *, metadata_consumers=("aic", "aiv0", "aiv1")):
        if type(columns) is not int or columns <= 0:
            raise ProtocolError("positive output-column count required")
        self.columns, self.rows = columns, len(epoch.anchors)
        self._check_epoch(epoch)
        self.consumers = frozenset(metadata_consumers)
        if not self.consumers or any(not isinstance(c, str) or not c for c in self.consumers):
            raise ProtocolError("declare every boundary metadata consumer")
        self.epoch = epoch
        self.prefix = [[0.0] * columns for _ in range(self.rows)]
        self.pending = [[0.0] * columns for _ in range(self.rows)]
        self.denominator = [0.0] * self.rows
        self.completed, self.mass_blocks = set(), set()
        self.boundary, self.drained, self.retired = None, False, set()

    def _check_epoch(self, epoch):
        if (type(epoch.ordinal) is not int or epoch.ordinal < 0 or self.rows == 0
                or type(epoch.anchors) is not tuple or len(epoch.anchors) != self.rows
                or not all(isfinite(a) for a in epoch.anchors)
                or type(epoch.producers) is not frozenset):
            raise ProtocolError("invalid immutable epoch")
        for p in epoch.producers:
            if (not isinstance(p, Producer) or type(p.block) is not int or p.block < 0
                    or type(p.column) is not int or not 0 <= p.column < self.columns
                    or not isinstance(p.component, str) or not p.component):
                raise ProtocolError("invalid PV producer identity")

    def _active(self, ordinal):
        if ordinal != self.epoch.ordinal:
            raise ProtocolError("stale or not-yet-active anchor epoch")
        if self.drained:
            raise ProtocolError("old-epoch writes after drain/clear are forbidden")

    def add_mass(self, ordinal, block, values):
        """Commit this block's unquantized probability sum at the current anchor."""
        self._active(ordinal)
        if block not in {p.block for p in self.epoch.producers} or block in self.mass_blocks:
            raise ProtocolError("unknown/duplicate probability block")
        values = tuple(values)
        if len(values) != self.rows or any(not isfinite(v) or v < 0 for v in values):
            raise ProtocolError("invalid denominator increment")
        self.denominator = [a + b for a, b in zip(self.denominator, values)]
        self.mass_blocks.add(block)

    def complete_pv(self, ordinal, producer, contribution):
        """Record completion, with one contribution per row to producer.column."""
        self._active(ordinal)
        if producer not in self.epoch.producers or producer in self.completed:
            raise ProtocolError("unknown/duplicate PV producer")
        if producer.block not in self.mass_blocks:
            raise ProtocolError("PV has no matching probability block")
        contribution = tuple(contribution)
        if len(contribution) != self.rows or not all(isfinite(v) for v in contribution):
            raise ProtocolError("invalid PV contribution")
        for row, value in enumerate(contribution):
            self.pending[row][producer.column] += value
        self.completed.add(producer)

    def publish_boundary(self, next_epoch, *, predicate, factors):
        """Publish one immutable decision; never overwrite a live factor/predicate."""
        if self.boundary is not None:
            raise ProtocolError("predicate/factor payload still has boundary consumers")
        self._check_epoch(next_epoch)
        if next_epoch.ordinal != self.epoch.ordinal + 1:
            raise ProtocolError("next epoch must follow the current epoch")
        factors = tuple(factors)
        changes = [b > a for a, b in zip(self.epoch.anchors, next_epoch.anchors)]
        if (any(b < a for a, b in zip(self.epoch.anchors, next_epoch.anchors))
                or type(predicate) is not bool or predicate != any(changes)
                or len(factors) != self.rows):
            raise ProtocolError("predicate/anchor transition mismatch")
        for old, new, factor in zip(self.epoch.anchors, next_epoch.anchors, factors):
            expected = 1.0 if old == new else exp(old - new)
            if (not isfinite(factor) or (old == new and factor != 1.0)
                    or not isclose(factor, expected, rel_tol=1e-12, abs_tol=0.0)):
                raise ProtocolError("factor must rescale this row; unchanged rows require 1")
        self.boundary = Boundary(next_epoch, predicate, factors)

    def _all_pv_complete(self):
        if (self.completed != set(self.epoch.producers)
                or self.mass_blocks != {p.block for p in self.epoch.producers}):
            raise ProtocolError("all old-anchor PV producers must complete before drain")

    def drain_and_clear(self):
        if self.boundary is None or self.drained:
            raise ProtocolError("one pending boundary is required")
        self._all_pv_complete()
        for row, factor in enumerate(self.boundary.factors):
            self.prefix[row] = [(a + b) * factor
                                for a, b in zip(self.prefix[row], self.pending[row])]
            self.pending[row] = [0.0] * self.columns
            self.denominator[row] *= factor
        self.drained = True

    def retire_metadata_consumer(self, consumer):
        if not self.drained or consumer not in self.consumers or consumer in self.retired:
            raise ProtocolError("invalid/premature metadata consumer retirement")
        self.retired.add(consumer)

    def activate_next_epoch(self):
        if not self.drained or self.retired != set(self.consumers):
            raise ProtocolError("drain/clear and all metadata uses must complete first")
        self.epoch = self.boundary.next_epoch
        self.boundary, self.drained = None, False
        self.retired, self.completed, self.mass_blocks = set(), set(), set()

    def output(self):
        """Logical final drain/normalization; not a hardware finalization proof."""
        if self.boundary is not None:
            raise ProtocolError("unfinished anchor boundary")
        self._all_pv_complete()
        return tuple(tuple((a + b) / self.denominator[row] if self.denominator[row] else 0.0
                           for a, b in zip(self.prefix[row], self.pending[row]))
                     for row in range(self.rows))
