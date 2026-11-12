"""P7-01/G69: the candidate space the performance search is allowed to walk.

WHY A GENERATOR AND NOT A HAND-WRITTEN LIST OF WINNERS. G70 asks whether the search compared
candidates that MEAN something, and the only way to answer that from artifacts is if the space
was enumerated and then pruned for a stated reason. A list of winners proves nothing: it cannot
show that a family was reachable, that a tile was considered, or that a rejection was arithmetic
rather than taste. So this module enumerates OVER THE DISCRETE DECISIONS the specs name, gives
every candidate a stable identity, and hands the space to search.filter for legality/resource
judgement.

THE THREE ORIGINS, AND WHY THEY ARE SEPARATE FIELDS. Spec 8.3 budgets a case as
12 seeds + 20 neighbours + 8 advanced candidates. Those are different KINDS of candidate and a
report has to keep them apart: a seed is a stratified probe of the space, a neighbour changes
ONE primary factor around an incumbent (spec 8.4), and an advanced candidate is a route the
performance phase owes (deferred rescale, output-column sharding, hi/lo P). Marking the origin
means "we tested 8 family-diverse seeds and 20 one-factor neighbours" is checkable rather than
countable.

WHY THE FAMILY IS DERIVED. A report must not be able to label a candidate it did not configure.
family() is computed from the decisions (CN for attention_residual, rescale mode and shard width
for MLA), so a canonical candidate cannot be reported as a deferred one, exactly as
attnres/model.py derives the AR family from CN instead of storing it.

WHAT IS DELIBERATELY NOT HERE. No cost model, no ranking, no "this one should win". Estimating
latency is explicitly allowed only for pruning (spec 8.1), and the ranking is device
measurement in harness.py. This module only says which configurations EXIST.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

from ..attnres.model import (
    BLOCKED_ONLINE,
    FACTORED_GW,
    LITERAL_FP32,
    ONLINE,
    RESIDENT_N,
    AttentionResidualError,
    build_attention_residual,
)
from ..mla.model import BFLOAT16, BSND, BNSD, FLOAT16, MLAError, MLASpec, build_mla

# ------------------------------------------------------------------ vocabulary

ORIGIN_SEED = "seed"
ORIGIN_NEIGHBOR = "neighbor"
ORIGIN_ADVANCED = "advanced"
ORIGINS = (ORIGIN_SEED, ORIGIN_NEIGHBOR, ORIGIN_ADVANCED)

RESCALE_CANONICAL = "canonical"
RESCALE_DEFERRED = "deferred"
RESCALE_MODES = (RESCALE_CANONICAL, RESCALE_DEFERRED)

PRECISION_SINGLE = "p_single"
PRECISION_HI_LO = "p_hi_lo"
PRECISION_POLICIES = (PRECISION_SINGLE, PRECISION_HI_LO)

MLA_CANONICAL = "mla_canonical"
MLA_DEFERRED = "mla_deferred"
MLA_OUTPUT_SHARD = "mla_output_shard"

#: Query-tile heights the specs name (spec 8.6: BQ in {16,32,64}). 16 is generated ON PURPOSE so
#: the filter can reject it for the real reason (the 128-column P packing path needs two 64-lane
#: vector halves, i.e. >= 32 rows) instead of the generator quietly never offering it.
BQ_CHOICES = (16, 32, 64)
#: KV positions per tile (spec 8.6: BC in {64,128}).
BC_CHOICES = (64, 128)
#: Output-column panel widths (spec 8.6: BD in {64,128,256}).
BD_CHOICES = (64, 128, 256)
#: Ring version counts to start from (spec 8.6: 1..3 per buffer, not forced equal).
VERSION_CHOICES = (1, 2, 3)
#: Pipeline recipes: 1 sequential, 2 shallow, 3 deep.
STAGE_CHOICES = (1, 2, 3)
#: Output-column shard counts for the small-decode family (spec 6.2/8.7).
SHARD_CHOICES = (1, 2, 4)

#: attention_residual token rows per program (spec 8.5: R in {1,2,4,8,16}).
R_CHOICES = (1, 2, 4, 8)
#: Prefetch recipes for attention_residual (spec 8.5).
PREFETCH_NONE = "none"
PREFETCH_X = "prefetch_x"
PREFETCH_NEXT_GROUP = "prefetch_next_group_first_block"
PREFETCHES = (PREFETCH_NONE, PREFETCH_X, PREFETCH_NEXT_GROUP)
#: Vector engine flavours the target exposes (spec 8.5: SimtVF / SimdVF).
SIMD_VF = "simdvf"
SIMT_VF = "simtvf"
VF_MODES = (SIMD_VF, SIMT_VF)

F32_MAX = 3.4028234663852886e38
#: Largest finite value each P-operand dtype can hold.
DTYPE_MAX_FINITE = {FLOAT16: 65504.0, BFLOAT16: 3.3895313892515355e38}
#: Two bits of mantissa headroom: a P value within a factor of 4 of the dtype's maximum has
#: already lost the low mantissa bits to rounding, which is exactly what a precision failure is.
P_HEADROOM = 4.0
#: Beyond ~12 the anchor slack exceeds any realistic logit spread, so a deferred candidate would
#: never drain and would only pay the control cost. That is a SEARCH-SPACE boundary, not a
#: legality bound: filter.py rejects only above theta_legal_bound(), and records when a generated
#: seed uses the practical cap.
THETA_PRACTICAL_CAP = 12.0


class GeneratorError(RuntimeError):
    """A violated contract in candidate generation."""

    def __init__(self, code, message, subject="", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ theta bounds

def theta_legal_bound(p_dtype: str, *, block_c: int, v_abs_max: float = 8.0) -> float:
    """The largest anchor slack that is NUMERICALLY LEGAL for this P operand and tile.

    Deferred accumulation keeps P = exp(z - anchor) in the P-operand dtype and accumulates
    P @ V in an FP32 L0C tile. Both constrain theta = max(z) - anchor:

      * the operand:   exp(theta) must stay clear of the dtype's finite maximum, or the operand
                       becomes inf and every PV contribution from that tile is inf*0;
      * the accumulator: a tile of BC columns contributes at most exp(theta) * BC * max|V| to
                       one FP32 accumulator, which must stay clear of FP32's maximum.

    Both are computed from declared constants (dtype maxima, tile width, an explicit |V| bound),
    never from a remembered magic number -- the case doc calls out copying the reference
    implementation's constant 6 as a defect. The result is floored at 0: a negative bound means
    the candidate cannot be expressed at all, which the generator reports by returning 0.0 and
    letting the filter/seed logic treat it as "only theta=0 is legal".
    """
    if p_dtype not in DTYPE_MAX_FINITE:
        raise GeneratorError("GEN_UNKNOWN_P_DTYPE",
                             "no declared finite maximum for the P operand dtype",
                             "p_dtype", sorted(DTYPE_MAX_FINITE), p_dtype)
    if block_c < 1:
        raise GeneratorError("GEN_BAD_BLOCK_C", "block_c must be positive", "block_c", ">= 1",
                             block_c)
    operand = math.log(DTYPE_MAX_FINITE[p_dtype] / P_HEADROOM)
    accumulator = math.log(F32_MAX / (P_HEADROOM * block_c * max(v_abs_max, 1e-30)))
    return max(0.0, min(operand, accumulator))


def theta_search_bound(p_dtype: str, *, block_c: int, v_abs_max: float = 8.0) -> float:
    """The largest anchor slack worth GENERATING as a positive-theta seed.

    min(legal bound, THETA_PRACTICAL_CAP). Below the cap a deferred candidate still drains on
    realistic logit spreads; above it the candidate is legal but indistinguishable from
    canonical except for its control cost, so generating it as a stratified seed would waste a
    slot that a genuinely different configuration could use. Neighbour generation may still
    reach the legal bound, and filter.py never rejects a legal theta.
    """
    return min(theta_legal_bound(p_dtype, block_c=block_c, v_abs_max=v_abs_max),
               THETA_PRACTICAL_CAP)


# ------------------------------------------------------------------ MLA candidates

@dataclass(frozen=True)
class MLACandidate:
    """One point in the MLA schedule space. Immutable so a report cannot be edited after the fact.

    The knobs are exactly the ones specs 8.6/8.7 make searchable: query-tile height, KV tile,
    output-column panel, the two independent ring version counts, the pipeline recipe, the P
    precision policy, the rescale route with its theta, the output-column shard count, and
    whether the loaded K_nope value is reused as V. Everything else (dtype, layout, causal,
    scale) belongs to the CASE specialization and is bound by the harness identity, not by the
    candidate -- otherwise the same schedule would hash differently per case and the compile
    cache could never be shared.
    """

    block_q: int
    block_c: int
    output_panel: int
    kv_versions: int
    p_versions: int
    stages: int
    precision: str
    rescale: str
    theta: float
    output_shards: int
    kv_reuse: bool
    origin: str = ORIGIN_SEED
    parent: str = ""
    changed: str = ""
    rationale: str = ""

    def __post_init__(self):
        if self.precision not in PRECISION_POLICIES:
            raise GeneratorError("GEN_BAD_PRECISION", "unknown P precision policy", "precision",
                                 PRECISION_POLICIES, self.precision)
        if self.rescale not in RESCALE_MODES:
            raise GeneratorError("GEN_BAD_RESCALE", "unknown rescale route", "rescale",
                                 RESCALE_MODES, self.rescale)
        if self.origin not in ORIGINS:
            raise GeneratorError("GEN_BAD_ORIGIN", "unknown candidate origin", "origin", ORIGINS,
                                 self.origin)
        if self.rescale == RESCALE_CANONICAL and self.theta != 0.0:
            # A canonical candidate has no anchor slack: it rescales every tile. Carrying a theta
            # would make two identical schedules hash differently.
            raise GeneratorError(
                "GEN_THETA_ON_CANONICAL",
                "canonical accumulation rescales every tile, so theta must be 0; a nonzero theta "
                "here would silently describe the deferred route", "theta", 0.0, self.theta)
        if self.rescale == RESCALE_DEFERRED and self.theta < 0.0:
            raise GeneratorError("GEN_NEGATIVE_THETA", "theta must be >= 0", "theta", ">= 0",
                                 self.theta)
        if self.origin == ORIGIN_NEIGHBOR and not self.changed:
            # A neighbour that does not name the ONE factor it changed is just a second seed, and
            # spec 8.4 requires neighbours to change one primary factor at a time.
            raise GeneratorError("GEN_UNLABELLED_NEIGHBOUR",
                                 "a neighbour must name the single factor it changed", "changed",
                                 "a knob name", self.changed)

    # -------------------------------------------------------------- identity

    def knobs(self) -> tuple:
        """The decisions, in a fixed order, with no provenance fields.

        Provenance (origin/parent/rationale) is deliberately excluded: the same configuration
        reached as a seed and as a neighbour is ONE candidate, and deduplication must collapse
        them rather than letting the search count it twice.
        """
        return (("block_q", self.block_q), ("block_c", self.block_c),
                ("output_panel", self.output_panel), ("kv_versions", self.kv_versions),
                ("p_versions", self.p_versions), ("stages", self.stages),
                ("precision", self.precision), ("rescale", self.rescale),
                ("theta", round(float(self.theta), 9)),
                ("output_shards", self.output_shards), ("kv_reuse", bool(self.kv_reuse)))

    def key(self) -> str:
        return _digest(dict(self.knobs()))

    @property
    def family(self) -> str:
        """Derived, never stored: a report cannot label a route that was not configured."""
        if self.output_shards > 1:
            return MLA_OUTPUT_SHARD
        if self.rescale == RESCALE_DEFERRED:
            return MLA_DEFERRED
        return MLA_CANONICAL

    def with_(self, **changes) -> "MLACandidate":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        out = dict(self.knobs())
        out.update({"key": self.key(), "family": self.family, "origin": self.origin,
                    "parent": self.parent, "changed": self.changed,
                    "rationale": self.rationale})
        return out


# ------------------------------------------------------------------ AR candidates

@dataclass(frozen=True)
class AttentionResidualCandidate:
    """One point in the attention_residual space (spec 8.5).

    The shape fields are part of the candidate because build_attention_residual REFUSES illegal
    family/CN combinations, and that refusal is the family definition. Carrying them here means a
    candidate can be built and judged without the caller re-deriving the shape.
    """

    n: int
    tokens: int
    d: int
    dtype: str
    block_n: int
    score_mode: str
    token_rows: int
    d_chunk: int
    v_versions: int
    #: The family a caller CLAIMS. It is not the family the candidate has: family is derived from
    #: CN and N below, and a claim that disagrees with the derivation is refused. That is what
    #: stops a report from labelling a CN=4 schedule resident_n.
    declared_family: str = ""
    prefetch: str = PREFETCH_NONE
    vf_mode: str = SIMD_VF
    origin: str = ORIGIN_SEED
    parent: str = ""
    changed: str = ""
    rationale: str = ""

    def __post_init__(self):
        if self.declared_family and self.declared_family != self.family:
            raise GeneratorError(
                "GEN_DECLARED_FAMILY_CONTRADICTS_CN",
                "the declared family disagrees with the family CN and N derive; the CN decides "
                "and the name follows it", "declared_family", self.family, self.declared_family)
        if self.prefetch not in PREFETCHES:
            raise GeneratorError("GEN_BAD_PREFETCH", "unknown prefetch recipe", "prefetch",
                                 PREFETCHES, self.prefetch)
        if self.vf_mode not in VF_MODES:
            raise GeneratorError("GEN_BAD_VF_MODE", "unknown vector flavour", "vf_mode",
                                 VF_MODES, self.vf_mode)
        if self.origin not in ORIGINS:
            raise GeneratorError("GEN_BAD_ORIGIN", "unknown candidate origin", "origin", ORIGINS,
                                 self.origin)
        if self.origin == ORIGIN_NEIGHBOR and not self.changed:
            raise GeneratorError("GEN_UNLABELLED_NEIGHBOUR",
                                 "a neighbour must name the single factor it changed", "changed",
                                 "a knob name", self.changed)

    def knobs(self) -> tuple:
        return (("n", self.n), ("tokens", self.tokens), ("d", self.d),
                ("dtype", self.dtype), ("block_n", self.block_n),
                ("score_mode", self.score_mode), ("token_rows", self.token_rows),
                ("d_chunk", self.d_chunk), ("v_versions", self.v_versions),
                ("prefetch", self.prefetch), ("vf_mode", self.vf_mode))

    def key(self) -> str:
        return _digest(dict(self.knobs()))

    @property
    def family(self) -> str:
        """Derived from CN and N, matching attnres/model.py's own derivation."""
        if self.block_n == 1:
            return ONLINE
        if self.block_n == self.n:
            return RESIDENT_N
        return BLOCKED_ONLINE

    def build_spec(self):
        """Build the attention_residual spec this candidate describes.

        Raises AttentionResidualError for combinations the operator's own domain forbids (for
        example CN == 1 labelled resident_n); the filter turns that into a named rejection
        instead of guessing.
        """
        return build_attention_residual(
            n=self.n, tokens=self.tokens, d=self.d, dtype=self.dtype, family=self.family,
            score_mode=self.score_mode, token_rows=self.token_rows, block_n=self.block_n,
            d_chunk=self.d_chunk, v_versions=self.v_versions)

    def with_(self, **changes) -> "AttentionResidualCandidate":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        out = dict(self.knobs())
        out.update({"key": self.key(), "family": self.family, "origin": self.origin,
                    "parent": self.parent, "changed": self.changed,
                    "rationale": self.rationale})
        return out


# ------------------------------------------------------------------ dedup

def deduplicate(candidates: Iterable) -> tuple:
    """Collapse candidates that configure the SAME schedule, keeping the first.

    Spec 8.3 forbids counting a re-labelled configuration as a new candidate. Two candidates
    with the same knobs are the same schedule even when they were reached by different routes,
    so the FIRST one wins (seeds before neighbours before advanced, by construction of the
    caller) and the dropped duplicates are reported by the caller through len().
    """
    seen = {}
    out = []
    for candidate in candidates:
        key = candidate.key()
        if key in seen:
            continue
        seen[key] = candidate
        out.append(candidate)
    return tuple(out)


def _dedup_tagged(candidates: Iterable) -> tuple:
    return deduplicate(candidates)


# ------------------------------------------------------------------ MLA seeds

def mla_seeds(spec: MLASpec, *, limit: int = 12, p_dtype: Optional[str] = None,
              v_abs_max: float = 8.0) -> tuple:
    """Twelve stratified MLA seeds, each probing a different axis (spec 8.6/8.7).

    Stratification is the point: a seed set that is twelve variations of one tile measures one
    configuration twelve times. Each entry below names the axis it probes, and the set contains
    at least two resource-different probes of every route G61/G62 require (canonical, deferred
    with theta=0, deferred with a positive theta, output-column sharding) plus precision, KV
    reuse, deep pipeline and panel-width probes. limit truncates deterministically so the caller
    can honour a smaller configured budget without changing WHICH seeds survive.
    """
    p_dtype = p_dtype or spec.dtype
    theta_pos = round(theta_search_bound(p_dtype, block_c=128, v_abs_max=v_abs_max), 6)
    d_nope = spec.shape.d_nope
    seeds = [
        dict(block_q=64, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="canonical baseline: full-width tile, double-buffered KV and P, shallow "
                       "pipeline, low-precision P, K_nope reused as V"),
        dict(block_q=32, block_c=128, output_panel=128, kv_versions=1, p_versions=1, stages=1,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="minimum-resource canonical probe: single-version sequential loop, the "
                       "ceiling for shapes whose UB/L1 is nearly exhausted"),
        dict(block_q=64, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_DEFERRED, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="deferred rescale at theta=0: drains on every anchor change, so it is the "
                       "correctness-equivalent deferred form with maximum drain count"),
        dict(block_q=64, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_DEFERRED, theta=theta_pos,
             output_shards=1, kv_reuse=True,
             rationale="deferred rescale at the largest theta the P operand and FP32 accumulator "
                       "allow with headroom: the route's actual performance claim"),
        dict(block_q=64, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_HI_LO, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="hi/lo P: two-component operand doubling P traffic and MMA count, the "
                       "precision fallback if single-component P fails the original comparator"),
        dict(block_q=32, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=2,
             kv_reuse=True,
             rationale="2-way output-column shard: repeats QK/softmax but doubles independent "
                       "jobs, the small-decode parallelism probe"),
        dict(block_q=32, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=4,
             kv_reuse=True,
             rationale="4-way output-column shard: the widest shard the panel width permits, so "
                       "the cost of repeated QK/softmax is measured rather than assumed"),
        dict(block_q=64, block_c=128, output_panel=256, kv_versions=2, p_versions=2, stages=3,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="deep pipeline with a wide PV panel: halves the number of PV/FIX calls per "
                       "KV tile at the cost of L0/Ring capacity"),
        dict(block_q=64, block_c=128, output_panel=64, kv_versions=2, p_versions=2, stages=3,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="narrow PV panel: more PV/FIX calls but much smaller UB/U footprint, and "
                       "the only legal panel width for D_nope=448 shapes"),
        dict(block_q=64, block_c=64, output_panel=128, kv_versions=3, p_versions=2, stages=3,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="half-width KV tile with triple-buffered K and a deep pipeline: more "
                       "versions in flight, more tiles per sequence"),
        dict(block_q=64, block_c=128, output_panel=128, kv_versions=2, p_versions=2, stages=2,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=False,
             rationale="independent V read: charges a second V tile on L1 and changes the last "
                       "reader of the K slot, the value-reuse comparison the spec requires"),
        dict(block_q=32, block_c=128, output_panel=128, kv_versions=3, p_versions=3, stages=3,
             precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0, output_shards=1,
             kv_reuse=True,
             rationale="maximum versions (3/3): the deepest prefetch distance the version bound "
                       "allows, to test whether more in-flight tiles pay for their capacity"),
    ]
    d_nope = spec.shape.d_nope
    out = []
    for entry in seeds:
        payload = dict(entry)
        # Seed panels and shard counts are adapted onto values that TILE this D_nope; a seed that
        # cannot compute the operator is not a stratified probe of the space.
        payload["output_panel"] = _adapt_panel(payload["output_panel"], d_nope)
        payload["output_shards"] = _adapt_shards(payload["output_shards"], d_nope)
        out.append(MLACandidate(**payload))
    # A case whose D_nope forbids some panels or shards yields fewer DISTINCT seeds after
    # adaptation (two seeds can collapse onto the same panel). Top up with further distinct
    # configurations so the stratified coverage the budget promises is actually delivered, and
    # record that the top-up happened rather than silently reporting a short seed set.
    out.extend(_top_up_mla_seeds(out, spec=spec, p_dtype=p_dtype, theta_pos=theta_pos,
                                v_abs_max=v_abs_max))
    return _truncate(_dedup_tagged(out), limit)


def _top_up_mla_seeds(existing, *, spec, p_dtype: str, theta_pos: float,
                      v_abs_max: float) -> list:
    """Extra distinct seeds for cases whose D_nope collapses part of the stratified set.

    These are not filler: each is a configuration the specs name that the main list could not
    express for this shape (a narrow panel, a wide panel, a single-version deep pipeline), so the
    top-up preserves the coverage rather than padding the count.
    """
    d_nope = spec.shape.d_nope
    legal = legal_panels(d_nope)
    seen = {c.key() for c in existing}
    extras = [
        dict(block_q=64, block_c=128, output_panel=min(legal) if legal else 64, kv_versions=2,
             p_versions=2, stages=3, precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL,
             theta=0.0, output_shards=1, kv_reuse=True,
             rationale="narrow-panel canonical: the widest panel this D_nope allows may still be "
                       "too large for the UB footprint, so the narrow one is a real alternative"),
        dict(block_q=32, block_c=128, output_panel=max(legal) if legal else 64, kv_versions=1,
             p_versions=1, stages=3, precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL,
             theta=0.0, output_shards=1, kv_reuse=True,
             rationale="deep pipeline with a single version: isolates stage count from ring "
                       "capacity, which the paired seeds cannot"),
        dict(block_q=64, block_c=128, output_panel=max(legal) if legal else 64, kv_versions=2,
             p_versions=2, stages=2, precision=PRECISION_SINGLE, rescale=RESCALE_DEFERRED,
             theta=round(theta_pos, 6), output_shards=1, kv_reuse=False,
             rationale="deferred rescale with an independent V read: the route's cost when V is "
                       "not aliased onto the loaded K_nope tile"),
    ]
    out = []
    for entry in extras:
        candidate = MLACandidate(**entry)
        if candidate.key() in seen:
            continue
        seen.add(candidate.key())
        out.append(candidate)
    return out


def legal_panels(d_nope: int) -> tuple:
    """The output-panel widths from the declared set that actually TILE this D_nope.

    A panel that does not divide D_nope silently computes fewer columns (the 448-versus-128 bug
    kernel.py documents), so a seed whose panel does not tile the case's D_nope is not a probe of
    a different configuration -- it is a configuration that cannot compute the operator. Seeds are
    therefore adapted onto a panel that tiles, while neighbours are left free to propose the
    declared widths so the LEGAL_PANEL_NOT_TILING_D_NOPE rejection stays reachable and visible.
    """
    return tuple(p for p in sorted(set(BD_CHOICES) | {64})
                 if p % 64 == 0 and d_nope % p == 0)


def legal_shards(d_nope: int) -> tuple:
    """Shard counts whose shard width is a whole number of 64-wide MMA panels.

    448 is the clearest case: neither 2 nor 4 shards give a 64-multiple width, so output-column
    sharding is genuinely unavailable there. Reporting that as an absent candidate is honest;
    generating a 224-column shard and calling it a shard would not be.
    """
    return tuple(s for s in sorted(set(SHARD_CHOICES))
                 if d_nope % (s * 64) == 0)


def _adapt_panel(preferred: int, d_nope: int) -> int:
    """The largest legal panel not exceeding the preferred one, else the smallest legal."""
    legal = legal_panels(d_nope)
    if not legal:
        raise GeneratorError("GEN_NO_LEGAL_PANEL",
                             "no declared panel width tiles this D_nope", "d_nope",
                             "a multiple of 64", d_nope)
    below = [p for p in legal if p <= preferred]
    return max(below) if below else min(legal)


def _adapt_shards(preferred: int, d_nope: int) -> int:
    legal = legal_shards(d_nope)
    if not legal:
        return 1
    below = [s for s in legal if s <= preferred]
    return max(below) if below else 1


def _truncate(items: Sequence, limit: int) -> tuple:
    if limit is None:
        return tuple(items)
    if limit < 1:
        raise GeneratorError("GEN_BAD_LIMIT", "limit must be positive", "limit", ">= 1", limit)
    return tuple(items[:limit])


# ------------------------------------------------------------------ AR seeds

def attention_residual_seeds(*, n: int, tokens: int, d: int, dtype: str = "bfloat16",
                             limit: int = 12) -> tuple:
    """Stratified attention_residual seeds spanning all three families (spec 8.5).

    The three families are one algebra parameterised by CN, so the seed set must cover the
    EXTREMES of CN (1, N, and a strict partial block) and then vary one factor at a time. A seed
    set that only explores CN=1 would make the later family comparison vacuous.
    """
    seeds = []
    base = dict(score_mode=LITERAL_FP32, token_rows=1, d_chunk=d, v_versions=2,
                prefetch=PREFETCH_NONE, vf_mode=SIMD_VF)

    def add(block_n, rationale, **overrides):
        payload = dict(base)
        payload.update(overrides)
        seeds.append(dict(block_n=block_n, rationale=rationale, **payload))

    # CN=1 IS the online family and CN=N IS resident_n, so those two are always expressible.
    add(1, "online CN=1 baseline: one state update per V block, the G50 kernel's schedule")
    add(1, "online CN=1 single-version: the smallest possible working set", v_versions=1)
    # A STRICT partial block only exists when 1 < CN < N, i.e. when N >= 3. At N == 2 there is no
    # integer between 1 and N, so blocked_online is not merely a bad candidate there -- it is not
    # a configuration at all, and the generator says so instead of emitting an illegal one.
    partial_cns = [cn for cn in (2, 4, 8, 16) if cn < n]
    if partial_cns:
        cn = partial_cns[0]
        add(cn, "blocked_online at CN=%d: merges a block before one state update, paying a "
                "second V read" % cn)
        add(cn, "blocked_online at CN=%d with one version: isolates the merge cost from the ring "
                "capacity" % cn, v_versions=1)
    else:
        # Record WHY the family is absent rather than quietly delivering eleven seeds.
        seeds.append(dict(
            block_n=1, rationale="blocked_online is UNREACHABLE at N=%d: the family requires a "
                                 "strict partial block 1 < CN < N and no such integer exists, so "
                                 "this slot probes online with a deeper V ring instead" % n,
            score_mode=LITERAL_FP32, token_rows=1, d_chunk=d, v_versions=3,
            prefetch=PREFETCH_NONE, vf_mode=SIMD_VF))
    add(n, "resident_n: every block on chip, one state update, hard UB wall", v_versions=1)
    add(n, "resident_n with a double-buffered V ring: the residency family's "
           "capacity-versus-overlap trade")
    add(1, "online with factored_gw: hoists the per-element division out of the reduction, a "
           "distinct precision policy", score_mode=FACTORED_GW)
    add(1, "online with x prefetch: overlap the next token group's x load", prefetch=PREFETCH_X)
    add(1, "online prefetching the next group's first block: the g+P reuse pattern",
        prefetch=PREFETCH_NEXT_GROUP)
    add(1, "online with two token rows per program: fewer groups, more per-group state, "
           "generated as UNMEASURED territory", token_rows=2)
    if min(d, 1024) != d:
        add(1, "chunked D: the chunk is the unit that must fit, so this is the seed that can "
               "survive a D that does not", d_chunk=min(d, 1024))
    else:
        add(1, "online with a triple-buffered V ring: the deepest prefetch the version bound "
               "allows for an aligned D whose chunk cannot be narrowed", v_versions=3)
    add(1, "online on the SIMT vector flavour: a different instruction selection for the same "
           "algebra", vf_mode=SIMT_VF)
    out = []
    for entry in seeds:
        out.append(AttentionResidualCandidate(n=n, tokens=tokens, d=d, dtype=dtype, **entry))
    return _truncate(_dedup_tagged(out), limit)


# ------------------------------------------------------------------ neighbours

def _clamp_choice(value, choices):
    """The next value in choices, or None when value is already the last one."""
    ordered = sorted(choices)
    if value not in ordered:
        # Not in the declared set: step to the smallest member, which is a legal neighbour.
        return ordered[0]
    index = ordered.index(value)
    if index + 1 < len(ordered):
        return ordered[index + 1]
    return None


def _prev_choice(value, choices):
    ordered = sorted(choices)
    if value not in ordered:
        return None
    index = ordered.index(value)
    return ordered[index - 1] if index > 0 else None


def mla_neighbours(candidate: MLACandidate, *, p_dtype: Optional[str] = None,
                   limit: int = 20, v_abs_max: float = 8.0) -> tuple:
    """One-factor-at-a-time neighbours of a candidate (spec 8.4: 每次只改一个主因素).

    Each neighbour names the knob it changed, so a report can show that the neighbourhood walk
    was one-factor and not a second random sample. Neighbours that would leave the declared
    choice sets are not produced; neighbours that are ILLEGAL are produced and left to the
    filter, because a legality rejection is evidence about the space.
    """
    base = dict(block_q=candidate.block_q, block_c=candidate.block_c,
                output_panel=candidate.output_panel, kv_versions=candidate.kv_versions,
                p_versions=candidate.p_versions, stages=candidate.stages,
                precision=candidate.precision, rescale=candidate.rescale,
                theta=candidate.theta, output_shards=candidate.output_shards,
                kv_reuse=candidate.kv_reuse)
    out = []

    def add(changed, **changes):
        payload = dict(base)
        payload.update(changes)
        out.append(MLACandidate(origin=ORIGIN_NEIGHBOR, parent=candidate.key(),
                                changed=changed, **payload))

    for name, choices in (("block_q", BQ_CHOICES), ("block_c", BC_CHOICES),
                          ("output_panel", BD_CHOICES), ("kv_versions", VERSION_CHOICES),
                          ("p_versions", VERSION_CHOICES), ("stages", STAGE_CHOICES),
                          ("output_shards", SHARD_CHOICES)):
        current = base[name]
        nxt = _clamp_choice(current, choices)
        if nxt is not None and nxt != current:
            add(name, **{name: nxt})
        prev = _prev_choice(current, choices)
        if prev is not None and prev != current:
            add(name, **{name: prev})

    other_precision = (PRECISION_HI_LO if candidate.precision == PRECISION_SINGLE
                       else PRECISION_SINGLE)
    add("precision", precision=other_precision)
    add("kv_reuse", kv_reuse=not candidate.kv_reuse)

    # The rescale route is one factor with two sub-decisions (mode and theta), so each mutation
    # changes exactly one of them: flipping the mode, or moving theta between 0 and the bound.
    if candidate.rescale == RESCALE_CANONICAL:
        add("rescale", rescale=RESCALE_DEFERRED, theta=0.0)
    else:
        add("rescale", rescale=RESCALE_CANONICAL, theta=0.0)
        bound = theta_search_bound(p_dtype or BFLOAT16, block_c=candidate.block_c,
                                   v_abs_max=v_abs_max)
        if candidate.theta == 0.0 and bound > 0.0:
            add("theta", theta=round(bound, 6))
        elif candidate.theta != 0.0:
            add("theta", theta=0.0)
    return _truncate(_dedup_tagged(out), limit)


def attention_residual_neighbours(candidate: AttentionResidualCandidate, *, limit: int = 20) -> tuple:
    """One-factor-at-a-time neighbours in the attention_residual space."""
    base = dict(n=candidate.n, tokens=candidate.tokens, d=candidate.d, dtype=candidate.dtype,
                block_n=candidate.block_n, score_mode=candidate.score_mode,
                token_rows=candidate.token_rows, d_chunk=candidate.d_chunk,
                v_versions=candidate.v_versions, prefetch=candidate.prefetch,
                vf_mode=candidate.vf_mode)
    out = []

    def add(changed, **changes):
        payload = dict(base)
        payload.update(changes)
        out.append(AttentionResidualCandidate(origin=ORIGIN_NEIGHBOR,
                                              parent=candidate.key(), changed=changed,
                                              **payload))

    # CN moves within the block family only: CN=1 is online and CN=N is resident_n, so moving
    # across those boundaries would change the FAMILY, which is a different factor.
    if candidate.family == BLOCKED_ONLINE:
        next_cn = _clamp_choice(candidate.block_n, (2, 4, 8, 16))
        if next_cn and next_cn < candidate.n:
            add("block_n", block_n=next_cn)
        prev_cn = _prev_choice(candidate.block_n, (2, 4, 8, 16))
        if prev_cn:
            add("block_n", block_n=prev_cn)
    add("score_mode", score_mode=(FACTORED_GW if candidate.score_mode == LITERAL_FP32
                                  else LITERAL_FP32))
    for name, choices in (("token_rows", R_CHOICES), ("v_versions", VERSION_CHOICES)):
        current = base[name]
        nxt = _clamp_choice(current, choices)
        if nxt is not None and nxt != current:
            add(name, **{name: nxt})
        prev = _prev_choice(current, choices)
        if prev is not None and prev != current:
            add(name, **{name: prev})
    for prefetch in PREFETCHES:
        if prefetch != candidate.prefetch:
            add("prefetch", prefetch=prefetch)
    add("vf_mode", vf_mode=(SIMT_VF if candidate.vf_mode == SIMD_VF else SIMD_VF))
    if candidate.d_chunk == candidate.d:
        smaller = min(candidate.d, 1024)
        if smaller != candidate.d:
            add("d_chunk", d_chunk=smaller)
    else:
        add("d_chunk", d_chunk=candidate.d)
    return _truncate(_dedup_tagged(out), limit)


# ------------------------------------------------------------------ advanced

def mla_advanced_around(incumbent: MLACandidate, *, p_dtype: Optional[str] = None,
                        limit: int = 8, v_abs_max: float = 8.0) -> tuple:
    """The advanced routes owed by G61/G62, anchored on the current incumbent's tiling.

    These are not neighbours: they switch to a DIFFERENT algorithmic route (deferred rescale,
    output-column sharding, hi/lo P) while keeping the incumbent's tile so the comparison isolates
    the route. Spec 8.3 reserves 8 slots for exactly this.
    """
    bound = theta_search_bound(p_dtype or BFLOAT16, block_c=incumbent.block_c,
                               v_abs_max=v_abs_max)
    recipes = [
        (dict(rescale=RESCALE_DEFERRED, theta=0.0),
         "G61 deferred rescale at theta=0 on the incumbent's tiling"),
        (dict(rescale=RESCALE_DEFERRED, theta=round(bound, 6)),
         "G61 deferred rescale at the legal theta bound on the incumbent's tiling"),
        (dict(precision=PRECISION_HI_LO),
         "hi/lo two-component P on the incumbent's tiling"),
        (dict(output_shards=2),
         "G62 2-way output-column shard on the incumbent's tiling"),
        (dict(output_shards=4),
         "G62 4-way output-column shard on the incumbent's tiling"),
        (dict(kv_reuse=False),
         "independent V read instead of reusing the loaded K_nope value"),
        (dict(rescale=RESCALE_DEFERRED, theta=round(bound, 6), precision=PRECISION_HI_LO),
         "deferred rescale with hi/lo P: the precision-safe deferred route"),
        (dict(rescale=RESCALE_DEFERRED, theta=round(bound, 6), output_shards=2),
         "deferred rescale sharded over output columns: small-decode route"),
    ]
    base = dict(block_q=incumbent.block_q, block_c=incumbent.block_c,
                output_panel=incumbent.output_panel, kv_versions=incumbent.kv_versions,
                p_versions=incumbent.p_versions, stages=incumbent.stages,
                precision=incumbent.precision, rescale=incumbent.rescale,
                theta=incumbent.theta, output_shards=incumbent.output_shards,
                kv_reuse=incumbent.kv_reuse)
    out = []
    for changes, rationale in recipes:
        payload = dict(base)
        payload.update(changes)
        candidate = MLACandidate(origin=ORIGIN_ADVANCED, parent=incumbent.key(),
                                 changed=",".join(sorted(changes)), rationale=rationale,
                                 **payload)
        if candidate.key() == incumbent.key():
            continue
        out.append(candidate)
    return _truncate(_dedup_tagged(out), limit)


__all__ = [
    "ORIGIN_SEED", "ORIGIN_NEIGHBOR", "ORIGIN_ADVANCED", "ORIGINS",
    "RESCALE_CANONICAL", "RESCALE_DEFERRED", "RESCALE_MODES",
    "PRECISION_SINGLE", "PRECISION_HI_LO", "PRECISION_POLICIES",
    "MLA_CANONICAL", "MLA_DEFERRED", "MLA_OUTPUT_SHARD",
    "BQ_CHOICES", "BC_CHOICES", "BD_CHOICES", "VERSION_CHOICES", "STAGE_CHOICES",
    "SHARD_CHOICES", "R_CHOICES", "PREFETCHES", "PREFETCH_NONE", "PREFETCH_X",
    "PREFETCH_NEXT_GROUP", "SIMD_VF", "SIMT_VF", "VF_MODES",
    "THETA_PRACTICAL_CAP", "DTYPE_MAX_FINITE", "P_HEADROOM", "F32_MAX",
    "GeneratorError", "MLACandidate", "AttentionResidualCandidate",
    "theta_legal_bound", "theta_search_bound", "mla_seeds", "attention_residual_seeds",
    "mla_neighbours", "attention_residual_neighbours", "mla_advanced_around",
    "deduplicate",
]
