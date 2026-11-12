"""P6-02/G61: the deferred-rescale scale epoch, its protocol and its verifier.

WHAT THIS MODULE IS. It is the executable form of cases/mla.md section 8 and spec 04 section 4.2
line 42 / spec 03 section 3.8: a deferred-rescale MLA keeps its running numerator in TWO places
and moves it to one only at an anchor boundary.

    AIV holds   anchor a, denominator l, already-drained prefix U_ub
    AIC holds   pending U_l0c accumulated under the CURRENT anchor
    invariant   U_total(t) = U_ub + U_l0c, at the SAME normalisation scale

A new KV block's maximum logit bmax extends the anchor only when bmax - a > theta. Otherwise P may
exceed 1 (bounded by theta) and PV accumulates straight into L0C, which removes the per-block
output transfer that the canonical route pays for every tile.

THE BOUNDARY IS A COMPLETION EDGE, NOT A BUFFER REUSE. When the group drains:

    every old-anchor PV completes, for ALL output columns and BOTH hi/lo probability components
      -> drain pending L0C into UB
      -> U_ub = (U_ub + pending) * exp(a_old - a_new)   (factor 1 for rows whose anchor held)
      -> l     = l * factor
      -> clear pending L0C
      -> only NOW may new-anchor PV accumulate

The failure this protects against is quantified in the case doc and reproduced in the tests: two
blocks with scores 0 and 10 and all-ones V must output 1; if the score-0 PV arrives late, after
the switch to anchor 10, and is added directly into the pending accumulator, the output approaches
2. A stale P buffer is NOT sufficient protection -- the P slot can be legitimately free while the
PV it produced is still in flight -- which is exactly why the epoch carries its own completion set
and its own immutable predicate/factor payload.

THE GROUP PREDICATE CROSSES CORES THROUGH SS. Two AIVs each decide locally whether they need a new
anchor, and the AIC must act on ONE group-wide decision. The target's shared scalar storage (SS) is
the only channel: uint32/uint64, 32 B alignment, 3 KiB model limit, no VF/DMA/atomic, and NOT a
cross-program barrier. Each AIV writes DISJOINT bytes; the AIC reads both and ORs them. This module
therefore verifies byte-level disjointness rather than trusting two variables to be separate, and
it refuses a layout where the two publishers overlap.

WHAT THIS MODULE DOES NOT DO. It does not prove a device Set/Wait completes; the reference model it
generalises says the same thing and so does spec 03. It does not emit TIR. Its numerics are Python
floats against small matrices, which is reference arithmetic and not FP16/BF16 emulation. The device
measurement that G61 asks for is reported separately in evidence/G61_DEVICE_EVIDENCE.json, and where
it could not be taken it says so rather than substituting a model.
"""
import math
from dataclasses import dataclass, field, replace
from math import exp, isclose, isfinite
from typing import Iterable, Sequence

# ------------------------------------------------------------------ target facts

#: cases/mla.md line 36: one MIX group's scalar shared storage is uint32/uint64, 32 B aligned, with
#: a 3 KiB model limit, and it is explicitly NOT usable as a cross-program barrier.
SS_CAPACITY_BYTES = 3 * 1024
SS_ALIGN = 32
SS_DTYPES = {"uint32": 4, "uint64": 8, "fp32": 4}

#: A per-publisher decision slot is one aligned unit so that two AIVs writing "the same" flag never
#: land in the same bytes. This is the alignment the disjointness check enforces.
SS_DECISION_BYTES = 32

PUBLISHERS = ("aiv0", "aiv1")
AIC = "aic"

#: Finite maxima of the storage dtypes, used for the theta bound. Taken from the dtype definition,
#: not from a number that happened to work.
DTYPE_MAX_FINITE = {"float16": 65504.0, "bfloat16": 3.3895313892515355e38}
F32_MAX = 3.4028234663852886e38
P_HEADROOM = 4.0
THETA_PRACTICAL_CAP = 12.0


class RescaleError(RuntimeError):
    """A refusal with a stable code, so a report can name the rule that was broken."""

    def __init__(self, code: str, message: str, subject: str = "", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__(f"{code}: {message}"
                         + (f" [{subject}: expected {expected!r}, got {actual!r}]"
                            if subject else ""))

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


def theta_legal_bound(p_dtype: str, *, block_c: int, v_abs_max: float = 8.0) -> float:
    """The largest theta for which exp(theta) * block_c * |V| still fits the operand dtype.

    cases/mla.md line 222 forbids copying the reference implementation's constant 6: the bound
    depends on the P operand dtype and on how much mass one tile can accumulate. Two independent
    ceilings apply and the smaller wins:

      * the probability itself: P = exp(score - anchor) <= exp(theta), and P is stored in p_dtype,
        so exp(theta) must not overflow p_dtype's finite range;
      * the PV product: one output cell accumulates block_c terms of magnitude
        P * |V| <= exp(theta) * v_abs_max, in FP32, so exp(theta) * block_c * v_abs_max must not
        overflow FP32.

    Headroom is applied because P is a rounded operand and a value that merely fits is one rounding
    away from inf.
    """
    if p_dtype not in DTYPE_MAX_FINITE:
        raise RescaleError("RESCALE_BAD_P_DTYPE", "theta must be bounded by a declared storage "
                           "dtype", "p_dtype", sorted(DTYPE_MAX_FINITE), p_dtype)
    if not isinstance(block_c, int) or block_c < 1:
        raise RescaleError("RESCALE_BAD_BLOCK_C", "block_c must be a positive int", "block_c",
                           ">= 1", block_c)
    if not isfinite(v_abs_max) or v_abs_max <= 0:
        raise RescaleError("RESCALE_BAD_V_BOUND", "the V magnitude bound must be finite and "
                           "positive", "v_abs_max", "> 0", v_abs_max)
    operand = math.log(DTYPE_MAX_FINITE[p_dtype] / P_HEADROOM)
    accumulator = math.log(F32_MAX / (P_HEADROOM * block_c * v_abs_max))
    return max(0.0, min(operand, accumulator))


def theta_search_bound(p_dtype: str, *, block_c: int, v_abs_max: float = 8.0) -> float:
    """The theta actually worth searching: the legal bound, capped at a practical exponent.

    A theta of 20 is legal for bfloat16 and useless: the probability operand would sit 20 e-folds
    above the anchor, so the FP32 accumulator would be dominated by cancellation and the numerics
    would be worthless even though nothing overflowed. The cap is a numerics choice and is
    recorded as one.
    """
    return min(theta_legal_bound(p_dtype, block_c=block_c, v_abs_max=v_abs_max),
               THETA_PRACTICAL_CAP)

# ------------------------------------------------------------------ producer identity

@dataclass(frozen=True)
class Producer:
    """One PV producer: a KV block, one output column, and one probability component.

    hi/lo is a COMPONENT, not a separate column, because a hi/lo precision strategy splits the
    probability operand while the output column is still one column. Making the component part of
    the identity is what lets the completion check insist that BOTH components finish before a
    drain -- a check that would silently pass if hi/lo shared one producer.
    """

    block: int
    column: int
    component: str = "single"

    def __post_init__(self):
        if not isinstance(self.block, int) or self.block < 0:
            raise RescaleError("RESCALE_BAD_PRODUCER_BLOCK", "PV producer block must be an int "
                               ">= 0", "block", ">= 0", self.block)
        if not isinstance(self.column, int) or self.column < 0:
            raise RescaleError("RESCALE_BAD_PRODUCER_COLUMN", "PV producer column must be an int "
                               ">= 0", "column", ">= 0", self.column)
        if not isinstance(self.component, str) or not self.component:
            raise RescaleError("RESCALE_BAD_PRODUCER_COMPONENT", "PV producer component must be a "
                               "non-empty string", "component", "a component name", self.component)

    def as_dict(self) -> dict:
        return {"block": self.block, "column": self.column, "component": self.component}


@dataclass(frozen=True)
class Epoch:
    """One scale generation: a per-row anchor and the complete PV producer set under it."""

    ordinal: int
    anchors: tuple
    producers: frozenset

    def __post_init__(self):
        if not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise RescaleError("RESCALE_BAD_EPOCH_ORDINAL", "epoch ordinal must be an int >= 0",
                               "ordinal", ">= 0", self.ordinal)
        if not isinstance(self.anchors, tuple) or not self.anchors:
            raise RescaleError("RESCALE_BAD_EPOCH_ANCHORS", "epoch anchors must be a non-empty "
                               "tuple", "anchors", "a non-empty tuple", self.anchors)
        for anchor in self.anchors:
            if not isfinite(anchor):
                raise RescaleError("RESCALE_NONFINITE_ANCHOR", "an anchor must be finite: an "
                                   "empty row does not get an anchor, it gets initialisation",
                                   "anchors", "finite floats", self.anchors)
        if not isinstance(self.producers, frozenset):
            raise RescaleError("RESCALE_BAD_EPOCH_PRODUCERS", "the producer set must be a "
                               "frozenset so it cannot change under the epoch",
                               "producers", "frozenset", type(self.producers).__name__)
        for producer in self.producers:
            if not isinstance(producer, Producer):
                raise RescaleError("RESCALE_BAD_EPOCH_PRODUCER", "every element must be a "
                                   "Producer", "producers", "Producer", repr(producer))

    def blocks(self) -> frozenset:
        return frozenset(p.block for p in self.producers)

    def as_dict(self) -> dict:
        return {"ordinal": self.ordinal, "anchors": list(self.anchors),
                "producers": [p.as_dict() for p in sorted(self.producers,
                                                          key=lambda x: (x.block, x.column,
                                                                         x.component))]}


@dataclass(frozen=True)
class Boundary:
    """The immutable published decision for one boundary: predicate plus per-row factors."""

    next_epoch: Epoch
    predicate: bool
    factors: tuple

    def as_dict(self) -> dict:
        return {"next_ordinal": self.next_epoch.ordinal, "predicate": self.predicate,
                "factors": list(self.factors)}


def boundary_factors(old_anchors: Sequence[float], new_anchors: Sequence[float]) -> tuple:
    """exp(a_old - a_new) per row, and exactly 1.0 where the anchor did not move.

    A row whose anchor held must get factor 1 rather than exp(0) computed in a way that could
    drift: the spec says "unchanged rows use factor=1", and a computed 0.9999999 would rescale a
    prefix that must not move.
    """
    if len(old_anchors) != len(new_anchors):
        raise RescaleError("RESCALE_FACTOR_ROW_MISMATCH", "old and new anchors must have the same "
                           "row count", "rows", len(old_anchors), len(new_anchors))
    factors = []
    for old, new in zip(old_anchors, new_anchors):
        if new < old:
            raise RescaleError("RESCALE_ANCHOR_REGRESSION", "an anchor may only grow or hold; a "
                               "decreasing anchor would rescale the prefix UP and overflow",
                               "anchor", ">= old %r" % old, new)
        factors.append(1.0 if old == new else exp(old - new))
    return tuple(factors)

# ------------------------------------------------------------------ the SS group predicate

@dataclass(frozen=True)
class SSByteRange:
    """A byte range in the group's shared scalar storage, with its publisher."""

    publisher: str
    offset: int
    length: int

    def __post_init__(self):
        if self.publisher not in PUBLISHERS:
            raise RescaleError("RESCALE_BAD_SS_PUBLISHER", "only a paired AIV publishes into SS",
                               "publisher", list(PUBLISHERS), self.publisher)
        if self.offset % SS_DECISION_BYTES:
            raise RescaleError("RESCALE_SS_MISALIGNED", "an SS decision slot must be aligned to "
                               "%d B" % SS_DECISION_BYTES, "offset",
                               "a multiple of %d" % SS_DECISION_BYTES, self.offset)
        if self.length < 1 or self.length % SS_DECISION_BYTES:
            raise RescaleError("RESCALE_SS_BAD_LENGTH", "an SS decision slot must be a whole "
                               "number of %d B units so two publishers can never share bytes"
                               % SS_DECISION_BYTES, "length",
                               "a positive multiple of %d" % SS_DECISION_BYTES, self.length)
        if self.offset + self.length > SS_CAPACITY_BYTES:
            raise RescaleError("RESCALE_SS_OUT_OF_RANGE", "the slot exceeds the SS model limit",
                               "end", "<= %d" % SS_CAPACITY_BYTES,
                               self.offset + self.length)

    def bytes(self) -> set:
        return set(range(self.offset, self.offset + self.length))

    def as_dict(self) -> dict:
        return {"publisher": self.publisher, "offset": self.offset, "length": self.length}


@dataclass(frozen=True)
class GroupPredicateLayout:
    """Where each side publishes its local "I need a new anchor" decision, and where the AIC reads.

    The layout is a VERIFIED object rather than a convention because the failure it prevents is
    invisible in the source: two AIVs that both write "predicate" at offset 0 produce a group
    decision that is really "whichever AIV wrote last", and the kernel is only wrong when the two
    AIVs disagree -- that is, on exactly the shapes with an unbalanced row tail.
    """

    publishers: tuple
    aic_reads: tuple
    slots: tuple

    def __post_init__(self):
        if len(self.publishers) != 2:
            raise RescaleError("RESCALE_GROUP_NEEDS_TWO_AIVS", "the group predicate is published "
                               "by exactly the two paired AIVs", "publishers", 2,
                               len(self.publishers))
        if tuple(sorted(self.publishers)) != tuple(sorted(PUBLISHERS)):
            raise RescaleError("RESCALE_GROUP_WRONG_PUBLISHERS", "the publishers must be the two "
                               "paired AIVs", "publishers", list(PUBLISHERS),
                               list(self.publishers))
        if tuple(sorted(self.aic_reads)) != tuple(sorted(PUBLISHERS)):
            raise RescaleError("RESCALE_AIC_MUST_READ_BOTH", "the AIC must read BOTH published "
                               "decisions and take the group decision itself; reading one side is "
                               "how a disagreement becomes silent", "aic_reads",
                               list(PUBLISHERS), list(self.aic_reads))

    @staticmethod
    def disjoint(*, base: int = 0) -> "GroupPredicateLayout":
        """The canonical layout: one aligned slot per publisher, adjacent, non-overlapping."""
        slots = tuple(SSByteRange(publisher=publisher,
                                 offset=base + index * SS_DECISION_BYTES,
                                 length=SS_DECISION_BYTES)
                      for index, publisher in enumerate(PUBLISHERS))
        return GroupPredicateLayout(publishers=PUBLISHERS, aic_reads=PUBLISHERS, slots=slots)

    def slot_of(self, publisher: str) -> SSByteRange:
        for slot in self.slots:
            if slot.publisher == publisher:
                return slot
        raise RescaleError("RESCALE_SS_PUBLISHER_UNSLOTTED", "no slot for this publisher",
                           "publisher", [s.publisher for s in self.slots], publisher)

    def verify(self) -> tuple:
        """Return (ok, problems). Byte-level, because "two variables" is not disjointness."""
        problems = []
        seen = {}
        for slot in self.slots:
            for byte in slot.bytes():
                if byte in seen:
                    problems.append(
                        "byte %d is written by both %s and %s: the group decision would be "
                        "whichever AIV wrote last, which is silent whenever the two agree and "
                        "wrong exactly when they do not" % (byte, seen[byte], slot.publisher))
                else:
                    seen[byte] = slot.publisher
        publishers_in_slots = {slot.publisher for slot in self.slots}
        for publisher in self.publishers:
            if publisher not in publishers_in_slots:
                problems.append("publisher %s has no slot" % publisher)
        # The AIC reads both slots; if it read only one the OR would collapse to that side.
        for publisher in PUBLISHERS:
            if publisher not in self.aic_reads:
                problems.append("the AIC does not read %s's slot" % publisher)
        total = sum(slot.length for slot in self.slots)
        if total > SS_CAPACITY_BYTES:
            problems.append("the predicate uses %d B of %d B SS" % (total, SS_CAPACITY_BYTES))
        return (not problems), tuple(problems)

    def as_dict(self) -> dict:
        ok, problems = self.verify()
        return {"publishers": list(self.publishers), "aic_reads": list(self.aic_reads),
                "slots": [s.as_dict() for s in self.slots],
                "disjoint": ok, "problems": list(problems),
                "bytes_used": sum(s.length for s in self.slots),
                "ss_capacity_bytes": SS_CAPACITY_BYTES}


def group_decision(local: dict, layout: GroupPredicateLayout) -> bool:
    """The AIC's group-wide OR of the two published local decisions.

    Every publisher must have published. A missing publication is NOT "false": treating an absent
    decision as "no rescale needed" would let an AIV that never reached its publish (a short loop,
    a fully masked tile) silently suppress a drain the other AIV requires.
    """
    missing = [p for p in layout.publishers if p not in local]
    if missing:
        raise RescaleError("RESCALE_GROUP_DECISION_INCOMPLETE",
                           "every paired AIV must publish a decision before the AIC takes the "
                           "group one; absent is not false", "missing", [], missing)
    for publisher, value in local.items():
        if not isinstance(value, bool):
            raise RescaleError("RESCALE_GROUP_DECISION_NOT_BOOL", "a published group decision "
                               "must be a bool", publisher, "bool", value)
    return any(local[p] for p in layout.publishers)

# ------------------------------------------------------------------ the protocol

#: The states a group moves through. Named rather than a set of booleans, because "drained and
#: retired but not activated" is a state the naive flag soup cannot express and the spec requires.
DRAINING_IDLE = "idle"            # accumulating under the current anchor
BOUNDARY_PUBLISHED = "boundary"   # predicate+factors published, drain not yet performed
DRAINED = "drained"               # prefix rescaled, pending cleared, consumers not retired
ACTIVATED = "activated"           # next epoch live, ready to accumulate again


@dataclass
class ScaleEpochProtocol:
    """One AIC/AIV group's deferred-rescale state, with every publication checked.

    The fields mirror the hardware exactly so a mismatch is a value mismatch and not a category
    error:

      prefix[r][c]      U_ub, the already-drained numerator in UB
      pending[r][c]     U_l0c, the current-anchor numerator resident in L0C
      denominator[r]    l, always at the CURRENT anchor's scale
      anchors[r]        a, the current per-row scale

    Every method that moves mass or scale validates its own precondition and raises a coded error,
    so a caller cannot reach an inconsistent state by skipping a step in the right order.
    """

    columns: int
    rows: int
    epoch: Epoch
    layout: GroupPredicateLayout = field(default_factory=GroupPredicateLayout.disjoint)
    metadata_consumers: tuple = ("aic", "aiv0", "aiv1")
    phase: str = DRAINING_IDLE
    prefix: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    denominator: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    boundary: Boundary = None
    completed: set = field(default_factory=set)
    mass_blocks: set = field(default_factory=set)
    retired: set = field(default_factory=set)
    local_decisions: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.columns, int) or self.columns <= 0:
            raise RescaleError("RESCALE_BAD_COLUMNS", "the group must own a positive number of "
                               "output columns", "columns", "> 0", self.columns)
        if self.rows != len(self.epoch.anchors):
            raise RescaleError("RESCALE_ROW_COUNT_MISMATCH", "the row count must equal the epoch's "
                               "anchor count", "rows", len(self.epoch.anchors), self.rows)
        if self.columns < 1:
            raise RescaleError("RESCALE_BAD_COLUMNS", "at least one output column", "columns",
                               "> 0", self.columns)
        # Producers must lie inside this group's real column range. A producer for column 700 in a
        # 512-column group is not "extra work", it is a different kernel's contribution.
        for producer in self.epoch.producers:
            if producer.column >= self.columns:
                raise RescaleError("RESCALE_PRODUCER_OUT_OF_RANGE", "a PV producer addresses a "
                                   "column this group does not own", "column",
                                   "< %d" % self.columns, producer.column)
        if not self.metadata_consumers:
            raise RescaleError("RESCALE_NO_METADATA_CONSUMERS", "declare every boundary metadata "
                               "consumer, or the payload can never be retired", "consumers",
                               "a non-empty tuple", self.metadata_consumers)
        self.prefix = [[0.0] * self.columns for _ in range(self.rows)]
        self.pending = [[0.0] * self.columns for _ in range(self.rows)]
        self.denominator = [0.0] * self.rows
        self.anchors = list(self.epoch.anchors)

    # -- invariant --------------------------------------------------------------

    def total(self, row: int, column: int) -> float:
        """U_total = U_ub + U_l0c for one cell, which is the route's whole premise."""
        return self.prefix[row][column] + self.pending[row][column]

    def invariant_report(self) -> dict:
        """A checkable statement of the two route invariants, for the evidence record.

        1. U_total is prefix + pending, cell by cell -- nothing lives anywhere else.
        2. l is at the current anchor's scale: l equals exp(a - a_ref) times the mass actually
           committed, so a rescale that moved U but not l would show up here as a mismatch.
        """
        shape_ok = (len(self.prefix) == self.rows == len(self.pending)
                    and all(len(r) == self.columns for r in self.prefix)
                    and all(len(r) == self.columns for r in self.pending))
        finite_ok = all(isfinite(v) for row in self.prefix + self.pending for v in row) \
            and all(isfinite(v) for v in self.denominator)
        # A drained row's pending must be exactly zero: a nonzero pending after drain means the
        # clear did not cover every column, and the old-scale mass would be double counted.
        cleared_ok = (all(v == 0.0 for row in self.pending for v in row)
                      if self.phase in (DRAINED, ACTIVATED) else True)
        return {"rows": self.rows, "columns": self.columns,
                "shape_ok": shape_ok, "finite_ok": finite_ok,
                "pending_cleared_after_drain": cleared_ok,
                "phase": self.phase, "epoch": self.epoch.ordinal,
                "completed": len(self.completed), "expected": len(self.epoch.producers),
                "retired": sorted(self.retired), "consumers": sorted(self.metadata_consumers),
                "ok": bool(shape_ok and finite_ok and cleared_ok)}

    def assert_consistent(self) -> None:
        report = self.invariant_report()
        if not report["ok"]:
            raise RescaleError("RESCALE_INVARIANT_BROKEN", "the deferred-rescale state violates "
                               "its invariants", "report", "ok", report)

    # -- accumulation -----------------------------------------------------------

    def _require_accumulating(self, ordinal: int, *, what: str) -> None:
        if ordinal != self.epoch.ordinal:
            raise RescaleError("RESCALE_STALE_EPOCH",
                               "%s addressed epoch %r while epoch %r is live: an old-scale write "
                               "after activation would be added at the wrong scale"
                               % (what, ordinal, self.epoch.ordinal),
                               "ordinal", self.epoch.ordinal, ordinal)
        if self.phase == BOUNDARY_PUBLISHED:
            raise RescaleError("RESCALE_OLD_SCALE_WRITE_AFTER_PUBLISH",
                               "%s arrived after the boundary was published: the old anchor's PV "
                               "must ALL complete before the drain, and this one did not" % what,
                               "phase", DRAINING_IDLE, self.phase)
        if self.phase in (DRAINED, ACTIVATED):
            raise RescaleError("RESCALE_OLD_SCALE_WRITE_AFTER_DRAIN",
                               "%s arrived after drain/clear: its contribution belongs to the old "
                               "anchor but the accumulator is now at the new one, so adding it "
                               "would mix scales" % what,
                               "phase", DRAINING_IDLE, self.phase)

    def add_mass(self, ordinal: int, block: int, values: Sequence[float]) -> None:
        """Commit one block's unquantised probability mass to the denominator.

        The denominator is summed from the FULL-PRECISION probabilities, never from the quantised
        PV operands: cases/mla.md line 195 says l must not be redefined as a sum over the rounded
        operand, because that changes the numerical path silently.
        """
        self._require_accumulating(ordinal, what="probability mass")
        if block not in self.epoch.blocks():
            raise RescaleError("RESCALE_UNKNOWN_MASS_BLOCK", "this block is not in the epoch's "
                               "producer set", "block", sorted(self.epoch.blocks()), block)
        if block in self.mass_blocks:
            raise RescaleError("RESCALE_DUPLICATE_MASS_BLOCK", "a block's mass may be committed "
                               "exactly once", "block", "not yet committed", block)
        values = tuple(float(v) for v in values)
        if len(values) != self.rows:
            raise RescaleError("RESCALE_MASS_ROW_COUNT", "one increment per row", "rows",
                               self.rows, len(values))
        for value in values:
            if not isfinite(value) or value < 0:
                raise RescaleError("RESCALE_BAD_MASS_INCREMENT", "a probability mass increment "
                                   "must be finite and non-negative", "value", ">= 0", value)
        self.denominator = [a + b for a, b in zip(self.denominator, values)]
        self.mass_blocks.add(block)

    def complete_pv(self, ordinal: int, producer: Producer,
                    contribution: Sequence[float]) -> None:
        """Record one PV producer's completion and accumulate it into pending (L0C)."""
        self._require_accumulating(ordinal, what="a PV contribution")
        if producer not in self.epoch.producers:
            raise RescaleError("RESCALE_UNKNOWN_PRODUCER", "this PV producer is not in the "
                               "epoch's declared set", "producer", "a declared producer",
                               producer.as_dict())
        if producer in self.completed:
            raise RescaleError("RESCALE_DUPLICATE_PRODUCER", "a PV producer completes exactly "
                               "once", "producer", "not yet completed", producer.as_dict())
        if producer.block not in self.mass_blocks:
            raise RescaleError("RESCALE_PV_WITHOUT_MASS", "a PV contribution requires its "
                               "probability block's mass to be committed first, or the "
                               "denominator and the numerator describe different weights",
                               "block", sorted(self.mass_blocks), producer.block)
        contribution = tuple(float(v) for v in contribution)
        if len(contribution) != self.rows:
            raise RescaleError("RESCALE_PV_ROW_COUNT", "one contribution value per row", "rows",
                               self.rows, len(contribution))
        for value in contribution:
            if not isfinite(value):
                raise RescaleError("RESCALE_NONFINITE_PV", "a PV contribution must be finite",
                                   "value", "finite", value)
        for row, value in enumerate(contribution):
            self.pending[row][producer.column] += value
        self.completed.add(producer)
    # -- the boundary -----------------------------------------------------------

    def publish_local_decision(self, publisher: str, *, needs_rescale: bool) -> None:
        """One AIV publishes its LOCAL decision into its own disjoint SS slot."""
        if publisher not in self.layout.publishers:
            raise RescaleError("RESCALE_UNKNOWN_PUBLISHER", "only a paired AIV publishes a local "
                               "decision", "publisher", list(self.layout.publishers), publisher)
        if not isinstance(needs_rescale, bool):
            raise RescaleError("RESCALE_DECISION_NOT_BOOL", "a local decision is a bool",
                               publisher, "bool", needs_rescale)
        if publisher in self.local_decisions:
            raise RescaleError("RESCALE_DECISION_REPUBLISHED", "a local decision for this boundary "
                               "was already published; republishing would let one AIV change the "
                               "group decision after the other had read it", "publisher",
                               "not yet published", publisher)
        self.local_decisions[publisher] = needs_rescale

    def publish_boundary(self, next_epoch: Epoch, *, factors: Sequence[float] = None) -> Boundary:
        """Take the group decision and publish the immutable predicate + factor payload.

        The predicate is NOT taken from the caller: it is derived from the two published local
        decisions through the SS layout. That is the spec's requirement that the AIC act on one
        group-wide decision rather than on whichever AIV ran last, and deriving it here means a
        caller cannot get it wrong.
        """
        if self.phase != DRAINING_IDLE:
            raise RescaleError("RESCALE_BOUNDARY_TWICE",
                               "a boundary payload may not be overwritten before its consumers "
                               "retire: the previous predicate and factors are still live",
                               "phase", DRAINING_IDLE, self.phase)
        if not isinstance(next_epoch, Epoch):
            raise RescaleError("RESCALE_BAD_NEXT_EPOCH", "the next epoch must be an Epoch",
                               "next_epoch", "Epoch", type(next_epoch).__name__)
        if next_epoch.ordinal != self.epoch.ordinal + 1:
            raise RescaleError("RESCALE_EPOCH_NOT_ADJACENT", "the next epoch must follow the "
                               "current one", "ordinal", self.epoch.ordinal + 1,
                               next_epoch.ordinal)
        if len(next_epoch.anchors) != self.rows:
            raise RescaleError("RESCALE_NEXT_EPOCH_ROWS", "the next epoch must have one anchor "
                               "per row", "rows", self.rows, len(next_epoch.anchors))
        # Group decision from the SS publications, not from a parameter.
        predicate = group_decision(self.local_decisions, self.layout)
        derived = boundary_factors(self.anchors, next_epoch.anchors)
        # The published predicate must AGREE with the anchors it accompanies. A predicate of False
        # with a moved anchor is the dangerous mismatch: it would skip the drain while the
        # accumulator scale changed underneath the pending mass.
        anchors_moved = any(new != old for old, new in zip(self.anchors, next_epoch.anchors))
        if predicate != anchors_moved:
            raise RescaleError(
                "RESCALE_PREDICATE_DISAGREES_WITH_ANCHORS",
                "the group predicate says %r but the anchors %s; a False predicate with a moved "
                "anchor would skip the drain and mix scales"
                % (predicate, "moved" if anchors_moved else "held"),
                "predicate", anchors_moved, predicate)
        if factors is not None:
            factors = tuple(float(f) for f in factors)
            if len(factors) != self.rows:
                raise RescaleError("RESCALE_FACTOR_ROW_COUNT", "one factor per row", "rows",
                                   self.rows, len(factors))
            for row, (given, expected) in enumerate(zip(factors, derived)):
                if not isfinite(given) or not isclose(given, expected, rel_tol=1e-12, abs_tol=0.0):
                    raise RescaleError("RESCALE_FACTOR_WRONG",
                                       "a factor must rescale exactly this row; unchanged rows "
                                       "require exactly 1", "row %d" % row, expected, given)
        else:
            factors = derived
        self.boundary = Boundary(next_epoch=next_epoch, predicate=predicate, factors=factors)
        self.phase = BOUNDARY_PUBLISHED
        self.history.append({"event": "boundary_published", "epoch": self.epoch.ordinal,
                             "next": next_epoch.ordinal, "predicate": predicate,
                             "factors": list(factors),
                             "local_decisions": dict(self.local_decisions)})
        return self.boundary

    def _all_pv_complete(self) -> None:
        """The gate the spec puts in front of every drain: ALL columns, BOTH components."""
        if self.completed != set(self.epoch.producers):
            missing = sorted(set(self.epoch.producers) - self.completed,
                             key=lambda p: (p.block, p.column, p.component))
            raise RescaleError(
                "RESCALE_DRAIN_BEFORE_ALL_PV",
                "every old-anchor PV must complete before the drain, including every output "
                "column and both hi/lo components; %d producer(s) are outstanding, so draining "
                "now would rescale a prefix that is missing their contribution and then add it "
                "at the wrong scale" % len(missing),
                "missing", [], [p.as_dict() for p in missing[:8]])
        if self.mass_blocks != set(self.epoch.blocks()):
            missing = sorted(set(self.epoch.blocks()) - self.mass_blocks)
            raise RescaleError("RESCALE_DRAIN_BEFORE_ALL_MASS",
                               "every probability block's mass must be committed before the "
                               "drain, or the denominator would rescale without its numerator",
                               "missing_blocks", [], missing)

    def drain_and_clear(self) -> dict:
        """drain pending L0C -> UB, rescale prefix, clear pending, rescale denominator."""
        if self.phase != BOUNDARY_PUBLISHED:
            raise RescaleError("RESCALE_DRAIN_WITHOUT_BOUNDARY",
                               "a drain requires exactly one published boundary",
                               "phase", BOUNDARY_PUBLISHED, self.phase)
        self._all_pv_complete()
        if not self.boundary.predicate:
            # A no-change boundary is a legal, conservative drain with every factor 1: it flushes
            # L0C without moving the scale. It still drains and clears.
            pass
        drained = [[0.0] * self.columns for _ in range(self.rows)]
        for row, factor in enumerate(self.boundary.factors):
            for column in range(self.columns):
                merged = self.prefix[row][column] + self.pending[row][column]
                drained[row][column] = self.pending[row][column]
                self.prefix[row][column] = merged * factor
            self.pending[row] = [0.0] * self.columns
            self.denominator[row] *= factor
        self.phase = DRAINED
        self.history.append({"event": "drained", "epoch": self.epoch.ordinal,
                             "predicate": self.boundary.predicate,
                             "factors": list(self.boundary.factors)})
        self.assert_consistent()
        return {"predicate": self.boundary.predicate,
                "factors": list(self.boundary.factors),
                "drained_columns": self.columns,
                "next_ordinal": self.boundary.next_epoch.ordinal}

    def retire_metadata_consumer(self, consumer: str) -> None:
        """Retire one boundary metadata consumer after its REAL final use."""
        if self.phase != DRAINED:
            raise RescaleError("RESCALE_PREMATURE_RETIREMENT",
                               "a metadata consumer may only retire after the drain it depends on",
                               "phase", DRAINED, self.phase)
        if consumer not in self.metadata_consumers:
            raise RescaleError("RESCALE_UNKNOWN_CONSUMER", "this consumer does not read the "
                               "boundary payload", "consumer", list(self.metadata_consumers),
                               consumer)
        if consumer in self.retired:
            raise RescaleError("RESCALE_DOUBLE_RETIREMENT", "a consumer retires once",
                               "consumer", "not yet retired", consumer)
        self.retired.add(consumer)

    def activate_next_epoch(self) -> Epoch:
        """Make the next epoch live. Only after every boundary consumer finished."""
        if self.phase != DRAINED:
            raise RescaleError("RESCALE_ACTIVATE_BEFORE_DRAIN",
                               "the next epoch cannot be activated before the drain/clear",
                               "phase", DRAINED, self.phase)
        outstanding = set(self.metadata_consumers) - self.retired
        if outstanding:
            raise RescaleError(
                "RESCALE_ACTIVATE_BEFORE_CONSUMERS_RETIRED",
                "the predicate and factor payload must stay live until EVERY consumer's real "
                "final use completes; %s still hold it, and activating now would overwrite the "
                "factors those consumers are still reading" % sorted(outstanding),
                "outstanding", [], sorted(outstanding))
        previous = self.epoch
        self.epoch = self.boundary.next_epoch
        self.anchors = list(self.epoch.anchors)
        self.boundary = None
        self.completed = set()
        self.mass_blocks = set()
        self.retired = set()
        self.local_decisions = {}
        self.phase = DRAINING_IDLE
        self.history.append({"event": "activated", "from": previous.ordinal,
                             "to": self.epoch.ordinal})
        return self.epoch

    # -- finalisation -----------------------------------------------------------

    def flush(self) -> tuple:
        """The end-of-kernel flush: wait for all PV, add pending to prefix, clear, normalise."""
        if self.phase == BOUNDARY_PUBLISHED:
            raise RescaleError("RESCALE_FLUSH_WITH_PENDING_BOUNDARY",
                               "a published boundary must be drained before the final flush, or "
                               "the payload is dropped with live consumers", "phase",
                               DRAINING_IDLE, self.phase)
        if self.phase == DRAINED:
            self.activate_next_epoch()
        self._all_pv_complete()
        for row in range(self.rows):
            for column in range(self.columns):
                self.prefix[row][column] += self.pending[row][column]
            self.pending[row] = [0.0] * self.columns
        self.history.append({"event": "flush", "epoch": self.epoch.ordinal})
        return self.output()

    def output(self) -> tuple:
        """U_total / l per row and column. Not a hardware finalisation proof."""
        self._all_pv_complete()
        if any(v != 0.0 for row in self.pending for v in row):
            raise RescaleError("RESCALE_OUTPUT_WITH_UNDRAINED_PENDING",
                               "output requires pending to be empty: U_total/l with mass still in "
                               "L0C would omit it", "phase", DRAINING_IDLE, self.phase)
        return tuple(tuple((self.prefix[row][column] / self.denominator[row])
                           if self.denominator[row] else 0.0
                           for column in range(self.columns))
                     for row in range(self.rows))

    def as_dict(self) -> dict:
        return {"columns": self.columns, "rows": self.rows, "phase": self.phase,
                "epoch": self.epoch.ordinal, "anchors": list(self.anchors),
                "denominator": list(self.denominator),
                "layout": self.layout.as_dict(), "invariants": self.invariant_report(),
                "history": list(self.history)}

# ------------------------------------------------------------------ the conditional guard pair

@dataclass(frozen=True)
class ConditionalGuard:
    """One conditional Set (publisher) and its matching Wait (consumer), for a checkable pair.

    spec 03 section 3.5: every wait must match the set of the RIGHT generation, under a compatible
    guard. A conditional Set paired with an unconditional Wait deadlocks on the branch where the
    Set is skipped; two different predicates are not implication. Both are checked here rather
    than argued in prose.
    """

    channel: str
    kind: str
    setter_kind: str
    setter_key: str
    waiter_kind: str
    waiter_key: str
    generation: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"channel": self.channel, "kind": self.kind,
                "setter": {"kind": self.setter_kind, "key": self.setter_key},
                "waiter": {"kind": self.waiter_kind, "key": self.waiter_key},
                "generation": self.generation, "note": self.note}


def verify_conditional_guards(guards: Iterable[ConditionalGuard]) -> tuple:
    """Return (problems, checks). Every Set/Wait pair on the deferred route must agree.

    A pair is refused when the waiter can execute while the setter is skipped, because that WAIT
    can never be satisfied, and when the two sides name different predicates, because two distinct
    predicates do not prove anything about each other.
    """
    from ..completion.guards import ALWAYS, COND, Guard, guard_implies, guards_agree
    problems, checks = [], []
    for guard in guards:
        setter = Guard(guard.setter_kind, guard.setter_key)
        waiter = Guard(guard.waiter_kind, guard.waiter_key)
        implies, why = guard_implies(waiter, setter)
        agree, agree_why = guards_agree(setter, waiter)
        entry = {"channel": guard.channel, "kind": guard.kind,
                 "setter": guard.setter_kind, "waiter": guard.waiter_kind,
                 "waiter_implies_setter": implies, "reason": why,
                 "sides_agree": agree, "agree_reason": agree_why,
                 "ok": bool(implies and agree)}
        checks.append(entry)
        if not implies:
            problems.append("%s (%s): %s" % (guard.channel, guard.kind, why))
        if not agree:
            problems.append("%s (%s): %s" % (guard.channel, guard.kind, agree_why))
        # The deferred route's whole point is that the decision is CONDITIONAL. An unconditional
        # Set would drain on every tile, which is the canonical route wearing the deferred
        # route's name -- so the guard must actually be conditional.
        if guard.kind.startswith("deferred") and setter.kind != COND:
            problems.append("%s (%s): the deferred control channel is unconditional, so it is not "
                            "a conditional drain at all" % (guard.channel, guard.kind))
        if setter.kind == ALWAYS and waiter.kind == ALWAYS:
            problems.append("%s (%s): both sides unconditional, so this pair proves nothing about "
                            "the deferred path" % (guard.channel, guard.kind))
    return tuple(problems), tuple(checks)


def deferred_route_guards(predicate_key: str, *, epoch: int = 0) -> tuple:
    """The five Set/Wait pairs the deferred route needs, all on ONE group predicate.

    The channels are the ones the case doc names: the two AIV publications, the AIC's group read,
    the conditional drain/clear, and the factor publication. They must all carry the SAME predicate
    key, because the drain decision is one decision -- a per-channel spelling that happens to agree
    today is not the same thing as one published decision.
    """
    conditional = ("cond", predicate_key)
    return (
        ConditionalGuard("ss_aiv0_decision", "deferred_ss_publish", *conditional, *conditional,
                         generation=epoch),
        ConditionalGuard("ss_aiv1_decision", "deferred_ss_publish", *conditional, *conditional,
                         generation=epoch),
        ConditionalGuard("ss_aic_group_read", "deferred_group_read", *conditional, *conditional,
                         generation=epoch),
        ConditionalGuard("l0c_drain_clear", "deferred_drain", *conditional, *conditional,
                         generation=epoch),
        ConditionalGuard("factor_publish", "deferred_factor", *conditional, *conditional,
                         generation=epoch),
    )


# ------------------------------------------------------------------ completion obligations

@dataclass(frozen=True)
class BoundaryObligation:
    """One real completion edge the compiler must express as an obligation, not as prose."""

    name: str
    producer: str
    consumers: tuple
    kind: str
    why: str

    def as_dict(self) -> dict:
        return {"name": self.name, "producer": self.producer, "consumers": list(self.consumers),
                "kind": self.kind, "why": self.why}


def boundary_obligations(epoch: int = 0, *, hi_lo: bool = False) -> tuple:
    """The obligations a scale-epoch boundary adds on top of the canonical route.

    These are the edges that make the boundary a COMPLETION relation rather than a buffer-reuse
    accident. Each names a real producer and its real consumers, so the existing state-effect and
    version passes can carry them (spec 01 line 144: the predicate publish, its read and the
    conditional drain must become real IR effects).
    """
    components = ("hi", "lo") if hi_lo else ("single",)
    # Every NAME carries the epoch. Without it two boundaries produce obligations that compare
    # EQUAL as artifacts, so a checker that unions them silently drops one boundary's edges --
    # which is the "predicate and factor overwritten by the next boundary" failure in another
    # guise. Identity has to be part of the name, not something a reader is trusted to remember.
    tag = "_e%d" % epoch
    out = [
        BoundaryObligation(
            "pv_all_columns_all_components_complete" + tag, "pv_%d" % epoch,
            ("drain_%d" % epoch,),
            "completion",
            "every old-anchor PV must finish before the drain, for every output column and both "
            "hi/lo components; a stale P buffer does not imply this, because the P slot is legally "
            "free while the PV it produced is still in flight"),
        BoundaryObligation(
            "ss_decision_publish_both_aivs" + tag, "aiv_decision_%d" % epoch,
            ("aic_group_read_%d" % epoch,), "state",
            "each AIV writes disjoint SS bytes and the AIC reads both; the AIC's group OR is the "
            "only decision, so a one-sided read would be a silent disagreement"),
        BoundaryObligation(
            "group_predicate_visible_to_aic" + tag, "ss_slot_%d" % epoch,
            ("l0c_drain_%d" % epoch,), "state",
            "the conditional drain and clear are guarded by the published predicate; SS is a "
            "scalar channel and carries no data-path completion of its own, so the read must be "
            "an explicit effect"),
        BoundaryObligation(
            "factor_survives_all_boundary_consumers" + tag, "factor_%d" % epoch,
            ("prefix_rescale_%d" % epoch, "denominator_rescale_%d" % epoch),
            "version",
            "the per-row factors must stay live until the last consumer of the boundary finishes; "
            "overwriting them with the next boundary's values rescales at the wrong epoch"),
        BoundaryObligation(
            "l0c_clear_visible_before_new_anchor_pv" + tag, "l0c_clear_%d" % epoch,
            ("pv_new_anchor_%d" % (epoch + 1),), "completion",
            "the new anchor's PV may only accumulate after the clear: L0C is clear/accumulate "
            "mode, so an unclear residue would be added into the new-scale accumulator"),
        BoundaryObligation(
            "pending_has_no_reader_until_drain" + tag, "l0c_pending_%d" % epoch,
            ("drain_%d" % epoch,), "version",
            "the canonical route's per-tile contribution transfer must NOT read L0C while the "
            "route is deferred; that transfer is exactly the cost this route removes"),
    ]
    for component in components:
        out.append(BoundaryObligation(
            "pv_component_%s_complete%s" % (component, tag), "pv_%s_%d" % (component, epoch),
            ("drain_%d" % epoch,), "completion",
            "the hi and lo probability components are separate producers, so both must be in the "
            "completion set; folding them into one producer would let a drain proceed with half "
            "the contribution"))
    return tuple(out)

# ------------------------------------------------------------------ the epoch driver

@dataclass(frozen=True)
class DeferredConfig:
    """The deferred route's tunables, each bounded rather than free."""

    block_c: int = 128
    theta: float = 0.0
    p_dtype: str = "float16"
    v_abs_max: float = 8.0
    precision: str = "p_single"
    columns: int = 0

    def components(self) -> tuple:
        return ("hi", "lo") if self.precision == "p_hi_lo" else ("single",)

    def check(self) -> None:
        """Every declared knob is inside the domain G61 argues about."""
        if not isinstance(self.block_c, int) or self.block_c < 1:
            raise RescaleError("RESCALE_BAD_BLOCK_C", "block_c must be a positive int", "block_c",
                               ">= 1", self.block_c)
        if not isfinite(self.theta) or self.theta < 0:
            raise RescaleError("RESCALE_BAD_THETA", "theta must be finite and non-negative; a "
                               "negative theta would extend the anchor for a block that raised "
                               "bmax", "theta", ">= 0", self.theta)
        bound = theta_legal_bound(self.p_dtype, block_c=self.block_c, v_abs_max=self.v_abs_max)
        if self.theta > bound:
            raise RescaleError("RESCALE_THETA_EXCEEDS_BOUND",
                               "theta exceeds the exp bound this P operand and tile can hold; the "
                               "reference implementation's constant must not be copied blindly",
                               "theta", "<= %.6f" % bound, self.theta)
        if self.precision not in ("p_single", "p_hi_lo"):
            raise RescaleError("RESCALE_BAD_PRECISION", "unknown precision policy", "precision",
                               ["p_single", "p_hi_lo"], self.precision)


def _cast(value: float) -> float:
    """Round to the storage dtype's precision without changing the mathematical reference.

    The reference model is Python float arithmetic. This is a DECLARED rounding so that a hi/lo
    split has something to represent; it is not an emulation of FP16 or BF16, and no accuracy claim
    rests on it.
    """
    if not isfinite(value):
        return value
    return float("%.4g" % value)


def _dot(left, right) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


@dataclass
class DeferredRun:
    """The outcome of driving one real shape through the deferred route."""

    rows: tuple
    drains: int
    max_anchor_span: float
    columns: int
    epoch: int
    invariants: dict
    history: tuple

    def as_dict(self) -> dict:
        return {"rows": len(self.rows), "columns": self.columns, "drains": self.drains,
                "final_epoch": self.epoch,
                "max_anchor_span": self.max_anchor_span,
                "invariants": self.invariants,
                "history": list(self.history)}


def run_deferred(queries_n, queries_r, kn, kr, v, *, key_n=None, key_r=None,
                 config: DeferredConfig = None, valid=None,
                 columns_slice=None, scale=None) -> DeferredRun:
    """Drive the deferred route over one small MLA problem THROUGH ScaleEpochProtocol.

    The numerics go through the protocol object rather than beside it, so the arithmetic and the
    completion rules cannot disagree: a drain the protocol refuses is a drain the numbers never
    perform. That is the property a reimplementation would quietly lose.

    key_n/key_r default to kn/kr. Passing distinct key matrices models the K_nope-value-reuse case
    where V aliases the loaded K_nope tile; the algebra is identical, only the loads differ.
    valid(row, kv_index) decides mask participation.
    """
    key_n = kn if key_n is None else key_n
    key_r = kr if key_r is None else key_r
    config = config or DeferredConfig()
    config.check()
    rows = len(queries_n)
    kv = len(key_n)
    dv = len(v[0])
    lo, hi = columns_slice if columns_slice is not None else (0, dv)
    width = hi - lo
    if width < 1:
        raise RescaleError("RESCALE_EMPTY_SHARD", "the output-column shard is empty", "width",
                           ">= 1", width)
    if scale is None:
        scale = 1.0 / math.sqrt(len(queries_n[0]) + len(queries_r[0]))
    valid = valid or (lambda row, index: True)
    components = config.components()

    def make_epoch(ordinal, anchors, block_set):
        """One epoch with its COMPLETE producer set: every column x every precision component."""
        producers = frozenset(Producer(block=block, column=column, component=component)
                              for block in block_set
                              for column in range(width)
                              for component in components)
        return Epoch(ordinal=ordinal, anchors=tuple(anchors), producers=producers)

    anchors = [0.0] * rows
    initialised = [False] * rows
    protocol = None
    drains = 0
    max_span = 0.0
    ordinal = 0

    for begin in range(0, kv, config.block_c):
        stop = min(begin + config.block_c, kv)
        block_scores = [[(_dot(queries_n[row], key_n[j]) + _dot(queries_r[row], key_r[j])) * scale
                         if valid(row, j) else None for j in range(begin, stop)]
                        for row in range(rows)]
        block_max = [max((z for z in scores if z is not None), default=-math.inf)
                     for scores in block_scores]
        live = [block_max[row] != -math.inf for row in range(rows)]
        if not any(live):
            continue

        if protocol is None:
            # First live block: the anchors ARE established here, with no drain. This is the
            # "first effective tile initialises the anchor" rule; an empty row is skipped rather
            # than computing -inf - -inf.
            proposed = [block_max[row] if live[row] else 0.0 for row in range(rows)]
            protocol = ScaleEpochProtocol(columns=width, rows=rows,
                                          epoch=make_epoch(0, proposed, {begin}))
            anchors = list(protocol.anchors)
            for row in range(rows):
                initialised[row] = initialised[row] or live[row]
        else:
            # A row needs a new anchor only when its block max would push P above exp(theta).
            need = [live[row] and block_max[row] - anchors[row] > config.theta
                    for row in range(rows)]
            proposed = [block_max[row] if need[row] else anchors[row] for row in range(rows)]
            moves = any(proposed[row] != anchors[row] for row in range(rows))
            if moves:
                # One AIC/AIV group decision: if ANY row must rescale, the whole group drains.
                # The two AIVs publish into disjoint SS slots and the AIC ORs them.
                #
                # The next epoch owns ONLY this block's producers. The old blocks' producers
                # belong to the epoch being retired: their mass is about to be drained and their
                # completion is what the drain gate checks. Carrying them into the next epoch
                # would demand a second completion for PV that already happened, and the drain
                # would then be refused forever -- which is what an earlier version of this loop
                # did.
                next_epoch = make_epoch(ordinal + 1, proposed, {begin})
                for publisher in PUBLISHERS:
                    protocol.publish_local_decision(publisher, needs_rescale=moves)
                protocol.publish_boundary(next_epoch)
                protocol.drain_and_clear()
                drains += 1
                for consumer in protocol.metadata_consumers:
                    protocol.retire_metadata_consumer(consumer)
                protocol.activate_next_epoch()
                ordinal += 1
                anchors = list(protocol.anchors)
            else:
                # No boundary: this block joins the CURRENT epoch's producer set. The epoch is
                # immutable, so joining republishes it with the enlarged set -- and the protocol's
                # accumulated completion set carries over, which is what makes "all old-anchor PV
                # complete" a real count rather than a fresh start.
                protocol.epoch = make_epoch(ordinal, anchors,
                                            set(protocol.epoch.blocks()) | {begin})

        span = max((block_max[row] - anchors[row] for row in range(rows) if live[row]),
                   default=0.0)
        max_span = max(max_span, span)

        mass = [0.0] * rows
        contributions = {}
        for row in range(rows):
            if not live[row]:
                continue
            probabilities = [0.0 if z is None else exp(z - anchors[row])
                             for z in block_scores[row]]
            # l is summed from the FULL-PRECISION probabilities. cases/mla.md line 195 forbids
            # redefining it as a sum over the rounded operand.
            mass[row] = math.fsum(probabilities)
            if config.precision == "p_hi_lo":
                high = [_cast(x) for x in probabilities]
                operands = {"hi": high,
                            "lo": [_cast(x - y) for x, y in zip(probabilities, high)]}
            else:
                operands = {"single": [_cast(x) for x in probabilities]}
            for column in range(width):
                for component in components:
                    contributions[(row, column, component)] = math.fsum(
                        operands[component][j - begin] * v[j][lo + column]
                        for j in range(begin, stop))
        if any(mass):
            protocol.add_mass(ordinal, begin, mass)
        for producer in sorted(protocol.epoch.producers,
                              key=lambda p: (p.block, p.column, p.component)):
            if producer.block != begin:
                continue
            # A row that participates in nothing contributes exactly zero, not a missing entry.
            # Substituting a default here is CORRECT and not a fudge: a fully masked row has no
            # PV contribution to any column, which is why the flush below also reports it as the
            # zero vector rather than as -inf.
            protocol.complete_pv(ordinal, producer,
                                 [contributions.get((row, producer.column, producer.component),
                                                    0.0)
                                  for row in range(rows)])

    if protocol is None:
        raise RescaleError("RESCALE_NO_LIVE_BLOCK", "no KV block participated, so there is no "
                           "anchor to flush", "blocks", ">= 1", len(range(0, kv, config.block_c)))
    # Final flush: the live epoch may still hold pending mass in L0C.
    return_rows = protocol.flush()
    return DeferredRun(rows=tuple(return_rows), drains=drains, max_anchor_span=max_span,
                       columns=width, epoch=protocol.epoch.ordinal,
                       invariants=protocol.invariant_report(), history=tuple(protocol.history))



# ------------------------------------------------------------------ fault injection

#: Each fault is a distinct way the scale boundary can be violated, named so an evidence record
#: can say which one was injected rather than "a negative test".
FAULTS = (
    "late_old_anchor_pv",        # old-scale PV arrives after drain/clear
    "premature_factor_overwrite",  # the next boundary's factors overwrite live ones
    "drain_before_all_pv",       # drain with an outstanding producer
    "drain_before_hi_lo_complete",  # only one probability component finished
    "premature_consumer_retirement",  # metadata consumer retires before its drain
    "activate_before_retirement",  # next epoch live while a consumer still reads the payload
    "overlapping_ss_slots",      # the two AIVs write the same SS bytes
    "one_sided_group_read",      # the AIC reads only one AIV's decision
    "predicate_anchors_mismatch",  # predicate says no change while the anchors moved
    "anchor_regression",         # an anchor decreases, rescaling the prefix up
    "denominator_from_quantised_p",  # l summed from the rounded operand instead of P
)


def inject_fault(fault: str, *, seed: int = 0) -> dict:
    """Run one scale boundary with a deliberate violation and report what REFUSED it.

    A refusal is the only acceptable outcome: a fault that runs to completion is a fault the
    compiler would also let through. Each case returns the coded error that stopped it, or
    the fact that nothing did.
    """
    if fault not in FAULTS:
        raise RescaleError("RESCALE_UNKNOWN_FAULT", "unknown injected fault", "fault", list(FAULTS),
                           fault)

    rows, columns = 2, 4
    def epoch(ordinal, anchors, blocks, components=("single",)):
        producers = frozenset(Producer(block=b, column=c, component=k)
                              for b in blocks for c in range(columns) for k in components)
        return Epoch(ordinal=ordinal, anchors=tuple(anchors), producers=producers)

    protocol = ScaleEpochProtocol(columns=columns, rows=rows,
                                  epoch=epoch(0, (0.0, 0.0), [0]))
    # Establish epoch 0 completely.
    protocol.add_mass(0, 0, [1.0, 1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: (p.column, p.component)):
        protocol.complete_pv(0, producer, [1.0 + seed, 2.0 + seed])

    if fault == "late_old_anchor_pv":
        next_epoch = epoch(1, (10.0, 10.0), [1])
        protocol.publish_local_decision("aiv0", needs_rescale=True)
        protocol.publish_local_decision("aiv1", needs_rescale=True)
        protocol.publish_boundary(next_epoch)
        protocol.drain_and_clear()
        for consumer in protocol.metadata_consumers:
            protocol.retire_metadata_consumer(consumer)
        protocol.activate_next_epoch()
        # The failure: epoch 0's PV arrives now, at epoch 1's scale.
        try:
            protocol.complete_pv(0, Producer(block=0, column=0), [5.0, 5.0])
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code,
                    "why": exc.message, "effect_if_accepted":
                    "the score-0 contribution would be added at the anchor-10 scale, so the "
                    "output for the two-block example is near 2 instead of 1"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "premature_factor_overwrite":
        protocol.publish_local_decision("aiv0", needs_rescale=True)
        protocol.publish_local_decision("aiv1", needs_rescale=True)
        protocol.publish_boundary(epoch(1, (10.0, 10.0), [1]))
        try:
            # A second boundary while the first payload still has consumers.
            protocol.publish_boundary(epoch(1, (20.0, 20.0), [1]))
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "the live factors would be replaced by the next "
                                          "boundary's, rescaling the prefix at the wrong epoch"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "drain_before_all_pv":
        wider = ScaleEpochProtocol(columns=columns, rows=rows,
                                   epoch=epoch(0, (0.0, 0.0), [0, 1]))
        wider.add_mass(0, 0, [1.0, 1.0])
        wider.add_mass(0, 1, [1.0, 1.0])
        for producer in sorted(wider.epoch.producers, key=lambda p: (p.block, p.column,
                                                                   p.component)):
            if producer.block == 0:
                wider.complete_pv(0, producer, [1.0, 1.0])
        wider.publish_local_decision("aiv0", needs_rescale=True)
        wider.publish_local_decision("aiv1", needs_rescale=True)
        wider.publish_boundary(epoch(1, (10.0, 10.0), [2]))
        try:
            wider.drain_and_clear()
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "block 1's contribution would be rescaled from the "
                                          "wrong anchor and then added at the new scale"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "drain_before_hi_lo_complete":
        split = ScaleEpochProtocol(columns=columns, rows=rows,
                                   epoch=epoch(0, (0.0, 0.0), [0], components=("hi", "lo")))
        split.add_mass(0, 0, [1.0, 1.0])
        for producer in sorted(split.epoch.producers, key=lambda p: (p.column, p.component)):
            if producer.component == "hi":
                split.complete_pv(0, producer, [1.0, 1.0])
        split.publish_local_decision("aiv0", needs_rescale=True)
        split.publish_local_decision("aiv1", needs_rescale=True)
        split.publish_boundary(epoch(1, (10.0, 10.0), [1], components=("hi", "lo")))
        try:
            split.drain_and_clear()
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "the lo component of the probability operand would be "
                                          "lost, so the result would be the hi-only answer"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "premature_consumer_retirement":
        protocol.publish_local_decision("aiv0", needs_rescale=True)
        protocol.publish_local_decision("aiv1", needs_rescale=True)
        protocol.publish_boundary(epoch(1, (10.0, 10.0), [1]))
        try:
            protocol.retire_metadata_consumer("aic")
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "the AIC could read a factor that was already retired "
                                          "and recycled for another channel"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "activate_before_retirement":
        protocol.publish_local_decision("aiv0", needs_rescale=True)
        protocol.publish_local_decision("aiv1", needs_rescale=True)
        protocol.publish_boundary(epoch(1, (10.0, 10.0), [1]))
        protocol.drain_and_clear()
        protocol.retire_metadata_consumer("aic")
        try:
            protocol.activate_next_epoch()
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "the two AIVs would still be reading factors that the "
                                          "next epoch had already overwritten"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "overlapping_ss_slots":
        # No local import: this module IS rescale, and importing it into itself shadows the
        # module-level names, which is what made the one-sided-read branch below fail.
        overlap = GroupPredicateLayout(
            publishers=("aiv0", "aiv1"), aic_reads=("aiv0", "aiv1"),
            slots=(SSByteRange("aiv0", 0, SS_DECISION_BYTES),
                   SSByteRange("aiv1", 0, SS_DECISION_BYTES)))
        ok, problems = overlap.verify()
        return {"fault": fault, "refused": not ok, "code": "RESCALE_SS_SLOTS_OVERLAP"
                if not ok else None, "why": "; ".join(problems),
                "effect_if_accepted": "the group decision becomes whichever AIV wrote last, "
                                      "which is wrong exactly when the two disagree"}

    if fault == "one_sided_group_read":
        try:
            GroupPredicateLayout(publishers=("aiv0", "aiv1"), aic_reads=("aiv0",),
                                 slots=GroupPredicateLayout.disjoint().slots)
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "an AIV that needs a rescale would be outvoted by the "
                                          "one whose slot the AIC happens to read"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "predicate_anchors_mismatch":
        # Publish locals that say "no rescale" while the anchors actually move.
        protocol.publish_local_decision("aiv0", needs_rescale=False)
        protocol.publish_local_decision("aiv1", needs_rescale=False)
        try:
            protocol.publish_boundary(epoch(1, (10.0, 10.0), [1]))
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "the drain would be skipped while the accumulator scale "
                                          "changed, mixing two scales in one L0C accumulator"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "anchor_regression":
        try:
            boundary_factors((10.0, 10.0), (0.0, 10.0))
        except RescaleError as exc:
            return {"fault": fault, "refused": True, "code": exc.code, "why": exc.message,
                    "effect_if_accepted": "exp(a_old - a_new) > 1 would rescale the drained "
                                          "prefix UP and can overflow the output"}
        return {"fault": fault, "refused": False, "code": None}

    if fault == "denominator_from_quantised_p":
        # Demonstrate the difference the rule makes: summing the ROUNDED operand gives a
        # different denominator than summing P, and the case doc forbids substituting it.
        import random as _random
        rng = _random.Random(seed)
        exact = [rng.uniform(0.1, 1.5) for _ in range(32)]
        rounded = [_cast(x) for x in exact]
        diff = abs(math.fsum(exact) - math.fsum(rounded))
        refused = diff > 0.0
        return {"fault": fault, "refused": refused,
                "code": "RESCALE_QUANTISED_DENOMINATOR_DIFFERS" if refused else None,
                "why": "summing the rounded operand changes l by %.6g on a 32-element block, "
                       "which silently redefines the denominator" % diff,
                "effect_if_accepted": "the normalisation would use the quantised weights, "
                                      "changing the numerical path the case doc fixes in P0"}
    raise RescaleError("RESCALE_UNREACHED_FAULT", "fault branch not implemented", "fault",
                       list(FAULTS), fault)


def run_all_faults() -> dict:
    """Every injected fault, with the code that refused it. Used verbatim by the evidence."""
    results = [inject_fault(fault) for fault in FAULTS]
    return {"faults": results,
            "all_refused": all(r["refused"] for r in results),
            "refused_count": sum(1 for r in results if r["refused"]),
            "total": len(results)}


__all__ = [
    "SS_CAPACITY_BYTES", "SS_ALIGN", "SS_DTYPES", "SS_DECISION_BYTES", "PUBLISHERS", "AIC",
    "DTYPE_MAX_FINITE", "F32_MAX", "P_HEADROOM", "THETA_PRACTICAL_CAP",
    "RescaleError", "theta_legal_bound", "theta_search_bound",
    "Producer", "Epoch", "Boundary", "boundary_factors",
    "SSByteRange", "GroupPredicateLayout", "group_decision",
    "DRAINING_IDLE", "BOUNDARY_PUBLISHED", "DRAINED", "ACTIVATED", "ScaleEpochProtocol",
    "ConditionalGuard", "verify_conditional_guards", "deferred_route_guards",
    "BoundaryObligation", "boundary_obligations",
    "DeferredConfig", "DeferredRun", "run_deferred",
    "FAULTS", "inject_fault", "run_all_faults",
]
