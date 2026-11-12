"""P5-03/G51: buffer lifetime, D-tail and invalid-row safety as CHECKABLE clauses.

WHY THIS IS A SEPARATE MODULE WITH NEGATIVE TESTS. A lifetime bug does not announce itself: a V
buffer released one task too early produces a plausible number, not a crash, and on a shapes
matrix it may only bite on one tail case. So the rules here are stated as predicates over a
declared event sequence, and the tests feed them DELIBERATELY WRONG sequences and require a
rejection. A checker that only ever sees correct input proves nothing about its own power.

THE FIVE RULES, each named after the failure it prevents:

  v_free_after_last_read     FREE(V) must come after the WEIGHTED SUM, not after the score.
                             Releasing at the score lets the next LOAD_V overwrite a tile the sum
                             still has to read -- the spec repeats this twice.
  x_out_reuse_after_store    x / output staging may be reused only after the MTE3 store has read
                             them. Reusing earlier overwrites data still in flight to GM.
  d_tail_denominator         the RMS denominator and the 1/sqrt(D) temperature use the REAL D.
                             A padded DP rescales every score and is a different operator.
  d_tail_no_overread         no raw load may read past the allocated bytes. A masked load is
                             fine; an unmasked full-width load on a padded tail is not.
  invalid_row_no_nan         an all-invalid token row gets score = -inf for every block. The
                             softmax shift then computes -inf - (-inf) = NaN, so the guard must
                             be a FINITE floor applied before the subtraction.
  masked_fma_no_wipe         a masked FMA in MODE_ZEROING on the LAST D chunk clears inactive
                             lanes, wiping the partial sum earlier chunks accumulated. The last
                             chunk must merge (MODE_MERGING), not zero.

WHY invalid_row_no_nan USES A FINITE FLOOR RATHER THAN -inf. With m = -inf and score = -inf,
max(m, score) is -inf and rho = exp(-inf - -inf) = exp(NaN) = NaN, which propagates to l and the
output. A finite floor (the measured kernel uses -1e30) makes the subtraction finite, so the guard
is arithmetic rather than a special case, and a fully-invalid row yields a well-defined 0/0-free
result instead of NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import AttentionResidualError

# ---------------------------------------------------------------- event vocabulary

LOAD_V = "LOAD_V"
SCORE = "SCORE"
WEIGHTED_SUM = "WEIGHTED_SUM"
FREE_V = "FREE_V"
LOAD_X = "LOAD_X"
FINALIZE = "FINALIZE"
STORE_Y = "STORE_Y"
REUSE_X = "REUSE_X"

#: The order the kernel must impose. WEIGHTED_SUM before FREE_V, and FREE_V before the next
#: LOAD_V, are the two that a review is most likely to get wrong.
REQUIRED_ORDER = (
    LOAD_V,
    SCORE,
    WEIGHTED_SUM,
    FREE_V,
    LOAD_X,
    FINALIZE,
    STORE_Y,
    REUSE_X,
)

#: A finite floor. -inf here would make rho = exp(-inf - -inf) = NaN.
FINITE_FLOOR = -1.0e30


@dataclass(frozen=True)
class LifetimeEvent:
    """One event in a declared execution sequence, scoped to a slot and an epoch.

    owner/epoch are explicit because a prefetch scheme reuses a slot across token groups; without
    them a FREE in epoch 0 could be mistaken for a FREE in epoch 1 and the checker would pass a
    sequence that is actually racing.
    """

    kind: str
    slot: str = "v0"
    epoch: int = 0
    group: int = 0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "slot": self.slot, "epoch": self.epoch,
                "group": self.group}


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rule": self.rule, "message": self.message, "detail": dict(self.detail)}


def check_lifetime(events) -> tuple:
    """Return the violations in a declared event sequence; empty means the sequence is safe.

    The check is deliberately per-slot and per-epoch: a FREE_V for slot v0 in epoch 0 does not
    authorise a LOAD_V into v0 in epoch 1, because the epoch change is exactly where a prefetch
    scheme's race would live.
    """
    violations = []
    seen = []  # (kind, slot, epoch, group) in order

    for ev in events:
        if not isinstance(ev, LifetimeEvent):
            raise AttentionResidualError(
                "G51_NOT_AN_EVENT", "expected LifetimeEvent instances", "event",
                "LifetimeEvent", type(ev).__name__)

        if ev.kind == FREE_V:
            # Every earlier read of this slot, same epoch, must be complete -- and the last read
            # is the WEIGHTED_SUM, not the SCORE.
            reads = [e for e in seen
                     if e.slot == ev.slot and e.epoch == ev.epoch
                     and e.kind in (SCORE, WEIGHTED_SUM)]
            if not reads:
                violations.append(Violation(
                    "v_free_before_last_read",
                    "FREE_V on slot %r epoch %d with no preceding read" % (ev.slot, ev.epoch),
                    {"slot": ev.slot, "epoch": ev.epoch}))
            elif not any(e.kind == WEIGHTED_SUM for e in reads):
                violations.append(Violation(
                    "v_free_after_score_not_sum",
                    "FREE_V on slot %r epoch %d follows the SCORE but not the WEIGHTED_SUM; the "
                    "weighted sum is V's last consumer, so releasing here lets a reload "
                    "overwrite a tile still needed" % (ev.slot, ev.epoch),
                    {"slot": ev.slot, "epoch": ev.epoch}))

        if ev.kind == LOAD_V:
            # A reload into a slot is only legal after that slot's previous epoch was freed.
            previous = [e for e in seen if e.slot == ev.slot and e.epoch < ev.epoch]
            if previous and not any(e.kind == FREE_V for e in previous):
                violations.append(Violation(
                    "v_reload_without_free",
                    "LOAD_V into slot %r epoch %d while an earlier epoch was never freed"
                    % (ev.slot, ev.epoch), {"slot": ev.slot, "epoch": ev.epoch}))

        if ev.kind == REUSE_X:
            if not any(e.kind == STORE_Y and e.group == ev.group for e in seen):
                violations.append(Violation(
                    "x_out_reuse_before_store",
                    "REUSE_X for group %d before its STORE_Y completed" % ev.group,
                    {"group": ev.group}))

        seen.append(ev)

    return tuple(violations)


# ---------------------------------------------------------------- D-tail rules

def check_d_tail(*, d: int, padded_d: int, denominator: int, mask_last_chunk_merges: bool,
                 raw_load_bytes_allocated: int, raw_load_bytes_requested: int) -> tuple:
    """Return the D-tail violations. Each clause is a defect seen or named in the spec."""
    violations = []

    # The real D must be the denominator. Using padded_d silently rescales every score.
    if denominator != d:
        violations.append(Violation(
            "d_tail_denominator",
            "the RMS denominator is %d but the real D is %d; using the padded width rescales "
            "every score and describes a different operator" % (denominator, d),
            {"denominator": denominator, "d": d}))

    # The last chunk must MERGE, not zero: MODE_ZEROING would wipe the partial sum that earlier
    # chunks accumulated in the inactive lanes.
    if d != padded_d and not mask_last_chunk_merges:
        violations.append(Violation(
            "masked_fma_wipes_partial_sum",
            "D=%d is a tail of %d and the last chunk uses MODE_ZEROING; inactive lanes would "
            "have their accumulated partial sum cleared" % (d, padded_d),
            {"d": d, "padded_d": padded_d}))

    # A raw full-width load must not read past what was allocated.
    if raw_load_bytes_requested > raw_load_bytes_allocated:
        violations.append(Violation(
            "raw_load_overreads_allocation",
            "a raw load requests %d bytes but only %d are allocated"
            % (raw_load_bytes_requested, raw_load_bytes_allocated),
            {"requested": raw_load_bytes_requested, "allocated": raw_load_bytes_allocated}))

    return tuple(violations)


# ---------------------------------------------------------------- invalid-row rules

def _safe_exp(x: float) -> float:
    """exp(x) with exp(-inf) = 0 and NaN propagating.

    This distinction is the whole point: -inf and +inf are NOT the same failure. exp(-inf) is a
    perfectly good 0.0 -- it is exactly what a guard SHOULD produce for a block that is entirely
    invalid -- whereas a NaN input means the subtraction itself was undefined and nothing
    downstream can be trusted. Treating them alike would either hide a real NaN or reject a
    correct guard.
    """
    if math.isnan(x):
        return float("nan")
    if x == float("-inf"):
        return 0.0
    return math.exp(x)


def online_softmax_step(m, l, score, floor=FINITE_FLOOR):
    """One guarded online-softmax merge, used by the tests as the arithmetic oracle.

    Written out so the -inf - -inf hazard is testable without a device: passing score = -inf and
    m = -inf with floor=None reproduces the NaN, and the finite floor prevents it.
    """
    base = m if floor is None else max(m, floor)
    new_m = max(base, score)
    if floor is not None:
        new_m = max(new_m, floor)
    rho = _safe_exp(base - new_m)
    bet = _safe_exp(score - new_m)
    return new_m, rho, bet


def check_invalid_row(*, all_invalid: bool, floor: float, scores, initial_m: float,
                      invalid_lane_policy: str = "masked_zero") -> tuple:
    """Return the invalid-row violations for a declared row state.

    THERE ARE TWO DIFFERENT 'ALL-INVALID ROW' CASES and conflating them was a real mistake in
    this module's first version:

      masked_zero  every D lane is masked to 0. Then v = 0, so sum(v^2) = 0, q = 0,
                   rms = sqrt(0/D + eps) = sqrt(eps), the key is 0, and the SCORE IS 0 -- a
                   perfectly finite number. The softmax over N is then uniform and agg = 0, so
                   the output is the residual branch (1-alpha)*x. No -inf is involved at all.
                   This is the case a device actually meets, and it is SAFE.

      neg_inf      the score is FORCED to -inf for every block (e.g. a validity mask applied at
                   the score rather than at the data). Then the -inf - -inf hazard is real and
                   needs a finite floor, and a row that is -inf everywhere additionally leaves
                   l = 0 so the final division needs its own guard.

    Reporting the masked_zero case as if it were neg_inf would flag a correct kernel; reporting
    neg_inf as masked_zero would let a real NaN through.
    """
    if invalid_lane_policy not in ("masked_zero", "neg_inf"):
        raise AttentionResidualError(
            "G51_BAD_INVALID_POLICY", "unknown invalid-lane policy", "invalid_lane_policy",
            ("masked_zero", "neg_inf"), invalid_lane_policy)
    violations = []

    # -inf as the floor is the trap: max(-inf,-inf) is -inf and -inf - -inf is NaN.
    if not math.isfinite(floor):
        violations.append(Violation(
            "invalid_row_nan_guard_missing",
            "the softmax floor is not finite (%r); an all-invalid row would compute "
            "exp(-inf - -inf) = NaN" % (floor,), {"floor": floor}))

    if not math.isfinite(initial_m):
        violations.append(Violation(
            "initial_m_not_finite",
            "the initial running max is %r; it must be a finite floor so the first "
            "subtraction is finite" % (initial_m,), {"initial_m": initial_m}))

    # Only the neg_inf policy requires every score to be -inf. For masked_zero a FINITE score is
    # the CORRECT outcome (it is exactly 0), so flagging it would reject a safe row.
    if all_invalid and invalid_lane_policy == "neg_inf":
        for s in scores:
            if math.isfinite(s):
                violations.append(Violation(
                    "invalid_row_has_finite_score",
                    "the row uses the neg_inf policy but block score %r is finite, so the "
                    "validity mask is not applied to every block" % (s,), {"score": s}))

    # Reproduce the arithmetic so a NaN shows up here, not only on device.
    m, l = initial_m, 0.0
    for s in scores:
        m, rho, bet = online_softmax_step(m, l, s, floor=floor)
        if math.isnan(rho) or math.isnan(bet):
            violations.append(Violation(
                "invalid_row_nan_produced",
                "the merge produced rho=%r bet=%r, so the result would be NaN" % (rho, bet),
                {"rho": rho, "bet": bet}))
            break
        l = rho * l + bet

    # SECOND-ORDER HAZARD, found by this module's own test rather than by reading the spec:
    # the finite floor makes the merge coefficients finite, but if EVERY block scores -inf then
    # bet is 0 at every step, l stays exactly 0, and the final acc/l is 0/0. A finite floor is
    # necessary but not sufficient -- the denominator needs its own guard.
    if l == 0.0:
        violations.append(Violation(
            "invalid_row_zero_denominator",
            "the running softmax denominator is 0 after the guard, so the final acc/l is 0/0; "
            "the finite floor prevents the NaN in the merge but not the one in the division",
            {"l": l, "m": m}))

    return tuple(violations)


def nan_demonstration() -> dict:
    """Show the hazard concretely, for the evidence record rather than as an assertion.

    Three states are recorded, because there are three distinct outcomes and conflating any two
    of them is how a guard looks correct while still producing NaN:

      broken   m = -inf and score = -inf with NO floor  -> rho = exp(NaN) = NaN
      floored  a finite floor                            -> coefficients finite, bet = 0
      zero_l   every block -inf, so l = 0                -> the DIVISION is 0/0
    """
    broken_m, broken_rho, broken_bet = online_softmax_step(
        float("-inf"), 0.0, float("-inf"), floor=None)
    floored_m, floored_rho, floored_bet = online_softmax_step(
        FINITE_FLOOR, 0.0, float("-inf"), floor=FINITE_FLOOR)
    # accumulate across four all-invalid blocks to show l collapsing to zero
    m, l = FINITE_FLOOR, 0.0
    for _ in range(4):
        m, rho, bet = online_softmax_step(m, l, float("-inf"), floor=FINITE_FLOOR)
        l = rho * l + bet
    return {
        "broken": {"m": broken_m, "rho": broken_rho, "bet": broken_bet,
                   "nan": math.isnan(broken_rho) or math.isnan(broken_bet)},
        "floored": {"m": floored_m, "rho": floored_rho, "bet": floored_bet,
                    "finite": math.isfinite(floored_rho) and math.isfinite(floored_bet)},
        "zero_l": {"l_after_four_invalid_blocks": l,
                   "division_defined": l != 0.0},
        "rule": ("a FINITE floor makes m - score finite; a separate denominator guard is still "
                 "required because a row that is entirely -inf leaves l = 0 and acc/l is 0/0"),
    }


# ---------------------------------------------------------------- prefetch rules

@dataclass(frozen=True)
class PrefetchPlan:
    """A V prefetch scheme: how many slots, and which owner/epoch owns each."""

    slots: int
    group_prefetch: bool
    v_versions: int

    def as_dict(self) -> dict:
        return {"slots": self.slots, "group_prefetch": self.group_prefetch,
                "v_versions": self.v_versions}


def check_prefetch(plan: PrefetchPlan, *, n_blocks: int, ub_bytes_needed: int,
                   ub_bytes_available: int) -> tuple:
    """Return the prefetch violations.

    Two distinct requirements: every slot must have a DISTINCT owner/epoch (otherwise two
    in-flight loads share a slot), and the plan must fit in UB -- a deeper prefetch that does not
    fit is not a faster plan, it is a compile failure.
    """
    violations = []
    if plan.slots < 1:
        violations.append(Violation(
            "prefetch_no_slot", "a prefetch plan needs at least one slot", {"slots": plan.slots}))
    if plan.slots > n_blocks:
        violations.append(Violation(
            "prefetch_more_slots_than_blocks",
            "%d prefetch slots for only %d V blocks; the extra slots can never hold distinct "
            "blocks" % (plan.slots, n_blocks), {"slots": plan.slots, "n_blocks": n_blocks}))
    if ub_bytes_needed > ub_bytes_available:
        violations.append(Violation(
            "prefetch_ub_overflow",
            "the prefetch plan needs %d UB bytes but %d are available"
            % (ub_bytes_needed, ub_bytes_available),
            {"needed": ub_bytes_needed, "available": ub_bytes_available}))
    # Each in-flight slot must be versioned separately, or two prefetches alias.
    if plan.slots > 1 and plan.v_versions < 2:
        violations.append(Violation(
            "prefetch_versions_insufficient",
            "%d prefetch slots but only %d buffer version(s): two in-flight loads would share "
            "one buffer" % (plan.slots, plan.v_versions),
            {"slots": plan.slots, "v_versions": plan.v_versions}))
    return tuple(violations)


def group_prefetch_target(*, group: int, programs: int) -> int:
    """The NEXT group's token index for the outer prefetch: g + P.

    Stated as a function because 'g + P' is the whole content of the requirement and an
    off-by-one here would prefetch the group currently being computed.
    """
    return group + programs