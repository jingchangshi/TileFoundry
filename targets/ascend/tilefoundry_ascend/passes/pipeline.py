"""D0 pipeline surface for the Ascend target: declarations, tail predicates, part.

SCOPE, stated once and not overstated. D0 carries the AUTHORED intent that a loop is
pipelined, plus the structural predicates that intent implies. It does NOT solve a
schedule, does not generate an offset plan, and does not materialise buffer versions:
a D0 for still runs serially and the typed fields say so. Nothing here is evidence
that a pipeline is feasible, that stages/II were resolved, or that synchronisation is
correct. Those are D1/D2 and are marked UNRESOLVED_D0 rather than implied.

WHY THE RULES ARE PLAIN FUNCTIONS AND THE PASSES ARE THIN. Every refusal in the
design table is a predicate over typed values ("budget 3 with x=4", "tail of 70 over
64 claimed FULL_TILE"). Keeping those predicates callable directly makes each bad
control falsifiable in isolation, and this module is explicit about which code it
owes. The passes add the ordering and the wiring, and call the SAME functions, so a
rule tested directly is the rule that actually runs.

THE MANIFEST TRAVELS ON Module.metadata. BEFORE_LOWER sees the authored module and
AFTER_LOWER sees the lowered one; the only declared carrier between them is metadata
(ConvertHIRToTIR returns the same Module object). A side channel keyed by id() would
break on any interned copy and would be invisible to review.

THE PRESERVATION CHECK NEVER SHRINKS ITS EXPECTATION. The expected side is built from
the authored manifest captured BEFORE lowering. An empty expectation is refused
(PIPELINE_EXPECTATION_MISSING) rather than accepted vacuously: a check whose expected
set is derived from the thing being checked can always pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from tilefoundry.ir.hir.pipeline import PipelineError
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass

from ..tf_target import Ascend950PRTarget

__all__ = [
    "TAIL_FULL_TILE",
    "TAIL_MASK_ACTIVE",
    "TailPolicy",
    "UNKNOWN",
    "BufferDeclaration",
    "PipelineManifest",
    "PredicateRequirement",
    "PipelineSurfaceError",
    "validate_pipeline_declarations",
    "verify_buffer_versions",
    "normalize_tail_requirement",
    "required_active_for_issue",
    "select_part",
    "normalize_part_selection",
    "verify_tir_part",
    "verify_predicate_coverage",
    "verify_issue_predicate_coverage",
    "verify_pipeline_preservation",
    "build_pipeline_manifest",
    "collect_tir_pipeline_declarations",
    "ValidatePipelineDeclarations",
    "NormalizeSchedulePredicates",
    "VerifyPredicateCoverage",
    "VerifyPipelinePreservation",
]


class TailPolicy(Enum):
    """The typed tail strategy. Not a free string, and not a default.

    A string annotation would let a typo silently mean "whatever the consumer decided",
    which is exactly how a full-tile assumption gets in without anyone declaring it.
    """

    FULL_TILE = "FULL_TILE"
    MASK_ACTIVE = "MASK_ACTIVE"


TAIL_FULL_TILE = TailPolicy.FULL_TILE
TAIL_MASK_ACTIVE = TailPolicy.MASK_ACTIVE


class _Unknown:
    """The UNKNOWN bound. A distinct object so it cannot be mistaken for 0 or None.

    None already means "no value supplied" at several call sites; reusing it for
    UNKNOWN would collapse two different facts into one, and the whole point of the
    rule is that UNKNOWN must not become a number.
    """

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise TypeError(
            "UNKNOWN is not truthy/falsy and must not be used in a condition; it is a "
            "missing bound, and a branch on it would silently pick one of the two "
            "answers the design forbids picking"
        )


UNKNOWN = _Unknown()


#: A refusal raised by the Ascend D0 rules. PipelineError carries a stable .code, so
#: reusing it keeps one refusal vocabulary across Core and target instead of inventing
#: a second one that callers would have to learn.
PipelineSurfaceError = PipelineError


@dataclass(frozen=True)
class BufferDeclaration:
    """One buffer version declaration, with where it came from.

    buffer_key is produced by the reader from declaration order and the value, never
    from id() or a display name: two parsed copies of the same source must agree, or a
    comparison across two views of one program could never match.
    """

    buffer_key: str
    versions: int
    kind: str = "FIXED"
    origin: Any | None = None


@dataclass(frozen=True)
class PredicateRequirement:
    """An immutable predicate obligation, derived from bounds rather than asserted.

    status is NORMALIZED only when the requirement was derived from known bounds.
    An UNKNOWN requirement must NOT carry a concrete active set: a concrete set on an
    UNKNOWN bound is the "UNKNOWN became 0" defect, so it is refused at construction
    instead of being trusted downstream.
    """

    issue_source_id: str
    scope: str
    required_active: frozenset
    granularity_bits: int
    producer_kind: str
    bounds_evidence: tuple
    policy: TailPolicy
    part_selection: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in ("NORMALIZED", "UNKNOWN"):
            raise PipelineError(
                "PREDICATE_PLAN_UNSUPPORTED",
                "requirement status must be NORMALIZED or UNKNOWN, got %r"
                % (self.status,),
            )
        if self.status == "UNKNOWN" and self.required_active:
            raise PipelineError(
                "DEFAULTED_UNKNOWN_VALUE",
                "an UNKNOWN requirement carries a concrete active set %r; that is an "
                "unknown bound silently turned into specific lanes"
                % (sorted(self.required_active),),
            )


@dataclass(frozen=True)
class PipelineManifest:
    """The derived, read-only record BEFORE_LOWER hands to AFTER_LOWER.

    Deliberately derived and typed: it is not a second copy of the source, and it is
    not widened after the fact. loops holds (loop_key, PipelineSpec) pairs exactly as
    the authored side declared them.
    """

    schema: str
    loops: tuple
    buffer_declarations: tuple
    requirements: tuple = ()

    def specs(self) -> dict:
        return {key: spec for key, spec in self.loops}


# ---------------------------------------------------------------------------
# Declaration checks (BEFORE_LOWER).
# ---------------------------------------------------------------------------

def validate_pipeline_declarations(spec, buffer_versions) -> None:
    """Refuse a version count that the loop budget cannot hold.

    The budget is the NUMBER OF STAGES of this loop; versions are PER BUFFER. They are
    different levels, so a budget of 2 does not re-read every buffer as 2 versions, and
    a declared version count larger than the budget is refused rather than clamped:
    clamping would change what runs while the declaration still reads as authored.
    """
    if spec is None:
        raise PipelineError(
            "PIPELINE_DECLARATION_INVALID",
            "no PipelineSpec given, so there is no budget to validate against",
        )
    budget = getattr(spec, "num_stages", None)
    if type(budget) is not int or budget < 1:
        raise PipelineError(
            "PIPELINE_NUM_STAGES_INVALID",
            "the stage budget must be a positive int, got %r" % (budget,),
        )
    for key, versions in sorted((buffer_versions or {}).items()):
        if type(versions) is not int or versions < 1:
            raise PipelineError(
                "BUFFER_VERSION_DECLARATION_INVALID",
                "buffer %s declares versions=%r; a version count must be a positive "
                "int (bool is not an int here)" % (key, versions),
            )
        if versions > budget:
            raise PipelineError(
                "BUFFER_VERSION_EXCEEDS_PIPELINE_BUDGET",
                "buffer %s declares %d versions but the loop stage budget is %d; the "
                "budget is not raised to fit and the count is not clamped down"
                % (key, versions, budget),
            )


def verify_buffer_versions(declared, actual) -> None:
    """Refuse a lowered buffer whose version count differs from its declaration.

    Distinct from EXCEEDS on purpose: EXCEEDS is an authored input error caught before
    lowering, while this is a DRIFT between two views of the same program. Reporting
    one as the other would tell an operator to fix the wrong thing.
    """
    declared = dict(declared or {})
    actual = dict(actual or {})
    changed = sorted(k for k in declared if k in actual and declared[k] != actual[k])
    if changed:
        raise PipelineError(
            "BUFFER_VERSION_DECLARATION_CHANGED",
            "these buffers changed version count after lowering: %s"
            % ([(k, declared[k], actual[k]) for k in changed],),
        )
    missing = sorted(k for k in declared if k not in actual)
    if missing:
        raise PipelineError(
            "BUFFER_VERSION_DECLARATION_CHANGED",
            "these buffer version declarations disappeared: %s" % (missing,),
        )


# ---------------------------------------------------------------------------
# Tail predicates (BEFORE_LOWER normalisation).
# ---------------------------------------------------------------------------

def _is_unknown(value) -> bool:
    return value is UNKNOWN


def _require_plain_int(value, name: str, *, minimum: int | None = None) -> int:
    """A plain int with an optional lower bound; a bool is never an int here."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "%s must be a plain int, got %r" % (name, value),
        )
    if minimum is not None and value < minimum:
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "%s must be >= %d, got %r" % (name, minimum, value),
        )
    return value


def _default_issue_origin(logical_len: int, lane_count: int) -> int:
    """Start of the LAST issue window covering a 1-D logical length.

    This is a window START, not a remainder: it is where the final issue begins, so the
    active set is derived per lane from origin + l < logical_len, rather than from a bare
    remainder count. For N=70/lane=64 the last issue starts at 64; for N=64/64 it starts
    at 0 (a full issue), which a remainder-only formula cannot express.
    """
    remainder = logical_len % lane_count
    return logical_len - lane_count if remainder == 0 else logical_len - remainder


def required_active_for_issue(logical_len, lane_count, origin):
    """S = {l | 0 <= l < lane_count and 0 <= origin + l < logical_len}, from bounds only.

    An empty S is UNPROVABLE (PREDICATE_PROOF_UNAVAILABLE), never "no lanes needed": an
    empty requirement would make any emitted predicate look sufficient. UNKNOWN bounds
    are UNPROVABLE too -- UNKNOWN is not 0 and not a full tile.
    """
    if _is_unknown(logical_len) or _is_unknown(origin) or _is_unknown(lane_count):
        raise PipelineError(
            "PREDICATE_PROOF_UNAVAILABLE",
            "a bound is UNKNOWN (logical_len=%r, lane_count=%r, origin=%r), so no active "
            "set can be proved; UNKNOWN is neither 0 nor a full tile"
            % (logical_len, lane_count, origin),
        )
    _require_plain_int(logical_len, "logical_len", minimum=0)
    lane_count = _require_plain_int(lane_count, "lane_count", minimum=1)
    origin = _require_plain_int(origin, "origin", minimum=0)
    active = frozenset(lane for lane in range(lane_count) if 0 <= origin + lane < logical_len)
    if not active:
        raise PipelineError(
            "PREDICATE_PROOF_UNAVAILABLE",
            "issue window origin=%d lane_count=%d covers no lane of logical length %d, "
            "so coverage cannot be proved; an empty required set is not nothing-needed"
            % (origin, lane_count, logical_len),
        )
    return active


def normalize_tail_requirement(logical_len, base, policy, *,
                              origin=None, lane_count=None, extent=None,
                              granularity_bits: int = 32,
                              producer_kind: str = "VF",
                              scope: str = "root",
                              issue_source_id: str = "tail.issue0",
                              part_selection: int = 0,
                              bounds_evidence: tuple = ()) -> PredicateRequirement:
    """Derive the active set ONE ISSUE WINDOW needs, or refuse to guess.

    'base' is the LOGICAL lane base of one issue (the element granularity of a full
    issue), NOT a physical tile width. A packed bf16 tile may be a different number of
    bytes/elements, so reading 'base' as a tile width would silently compute the wrong
    lane count. 'lane_count' defaults to 'base'; 'origin' is the window start and
    defaults to the LAST issue covering logical_len.

    The required set comes from the BOUNDS -- {l | 0 <= l < lane_count and
    0 <= origin + l < logical_len} -- never from an authored active count. FULL_TILE is
    accepted only when the window really covers every lane; on N=70/lane=64 the last
    issue needs six lanes, and nothing here proves a full 64.

    UNKNOWN is refused rather than defaulted, and logical_len 0 is refused rather than
    read as a full tile through 0 % base == 0.
    """
    if _is_unknown(logical_len) or _is_unknown(extent):
        raise PipelineError(
            "PIPELINE_BOUND_UNKNOWN",
            "the tail bound is UNKNOWN (logical_len=%r, extent=%r), so no active set "
            "can be derived. UNKNOWN is not 0 and is not a full tile"
            % (logical_len, extent),
        )
    base = _require_plain_int(base, "base", minimum=1)
    if type(logical_len) is not int or logical_len < 0:
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "logical_len must be a non-negative int, got %r" % (logical_len,),
        )
    if not isinstance(policy, TailPolicy):
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "the tail policy must be a TailPolicy, got %s. A string here would let a "
            "typo silently select the consumer default" % (type(policy).__name__,),
        )
    lanes = base if lane_count is None else _require_plain_int(lane_count, "lane_count", minimum=1)
    if logical_len == 0:
        raise PipelineError(
            "PREDICATE_PROOF_UNAVAILABLE",
            "logical length 0 has no issue to satisfy; 0 %% base == 0 must not be read "
            "as a full tile",
        )
    start = (
        _default_issue_origin(logical_len, lanes)
        if origin is None
        else _require_plain_int(origin, "origin", minimum=0)
    )
    active = required_active_for_issue(logical_len, lanes, start)
    if policy is TailPolicy.FULL_TILE and active != frozenset(range(lanes)):
        raise PipelineError(
            "UNPROVEN_FULL_TILE",
            "FULL_TILE was claimed for issue window [%d, %d) of a logical length %d. It "
            "needs lanes %r, not a full %d" % (start, start + lanes, logical_len,
                                               sorted(active), lanes),
        )
    evidence = tuple(bounds_evidence) or (
        ("logical_len", logical_len),
        ("lane_count", lanes),
        ("origin", start),
    )
    return PredicateRequirement(
        issue_source_id=issue_source_id,
        scope=scope,
        required_active=active,
        granularity_bits=granularity_bits,
        producer_kind=producer_kind,
        bounds_evidence=evidence,
        policy=policy,
        part_selection=part_selection,
        status="NORMALIZED",
    )


# ---------------------------------------------------------------------------
# part selection.
# ---------------------------------------------------------------------------

def select_part(projection) -> int:
    """Write part 0 only when the projection UNIQUELY supports it.

    part is not a free choice and is not implied by tail length or dtype: it follows
    from the transport/dtype projection. Writing 0 by default would be the same class
    of defect as defaulting an unknown bound, so anything that is not provably part0 is
    refused by name.
    """
    if projection is None:
        raise PipelineError(
            "PART_SELECTION_UNSUPPORTED",
            "no projection given, so part cannot be selected; a default of 0 would be "
            "a guess about the transport path",
        )
    unique = getattr(projection, "unique_part", None)
    part = unique() if callable(unique) else None
    if part != 0:
        raise PipelineError(
            "PART_SELECTION_UNSUPPORTED",
            "the projection does not uniquely select part0 (it reports %r), so D0 does "
            "not write a part" % (part,),
        )
    return 0


def normalize_part_selection(projection) -> int:
    """The part a NORMALIZE step records: 0 only through a unique projection.

    Kept as its own entry point so a caller cannot write part 0 by omission: the only
    way to get 0 out is a projection that reports it uniquely via select_part.
    """
    return select_part(projection)


def verify_tir_part(part) -> None:
    """Refuse a TIR-constructed part that is not part0.

    A raw TIR attribute is not a projection and carries no uniqueness proof, so only an
    absent part or the int 0 is admissible; a directly built part1 is refused by name
    instead of being treated as a legal schedule choice.
    """
    if part is None:
        return
    if isinstance(part, bool) or not isinstance(part, int) or part != 0:
        raise PipelineError(
            "PART_SELECTION_UNSUPPORTED",
            "a TIR operation states part=%r; D0 writes part 0 only through a unique "
            "projection and does not accept a directly constructed part1" % (part,),
        )


# ---------------------------------------------------------------------------
# Coverage verification (AFTER_LOWER).
# ---------------------------------------------------------------------------

def verify_predicate_coverage(required=None, actual=None, *, offset_value=None,
                              offset_status=None, producer_is_fresh=None,
                              dma_window=None, granularity_bits=None,
                              actual_granularity_bits=None, scope=None,
                              actual_scope=None, source_bound=None,
                              logical_len=None, base=None, lane_count=None,
                              origin=None) -> None:
    """Compare the active set the BOUNDS require with what was actually emitted.

    The required set is either supplied (as a set already re-derived from bounds by the
    caller) or re-derived HERE from (logical_len, lane_count/base, origin). It is never
    read back from an authored active_count, and a caller that passes one set into both
    slots cannot make a wrong predicate certify itself: 'actual' is a distinct value.

    Checks are ordered most-specific first, so the code names the real defect instead of
    collapsing everything into a coverage mismatch.
    """
    if offset_status is UNKNOWN and offset_value == 0:
        raise PipelineError(
            "DEFAULTED_UNKNOWN_VALUE",
            "an UNKNOWN offset status was recorded together with the value 0. 0 is a "
            "number: it would compare as a real offset for the rest of the pipeline",
        )
    if producer_is_fresh is False:
        raise PipelineError(
            "PREDICATE_STALE_OR_OVERWRITTEN",
            "the predicate producer is stale: it was issued in an older region or was "
            "overwritten before this consumer, so the mask read here is not the mask "
            "the bounds require",
        )
    if dma_window is False:
        raise PipelineError(
            "DMA_TAIL_ACCESS_UNSUPPORTED",
            "the tail is legal for the VF predicate but its DMA access window cannot "
            "be expressed; a VF mask does not bound a DMA transfer, so the DMA access "
            "is NOT excused",
        )
    if source_bound is False:
        raise PipelineError(
            "PREDICATE_SOURCE_UNBOUND",
            "no real source loop/operand is bound to this predicate requirement",
        )
    if scope is not None and actual_scope is not None and scope != actual_scope:
        raise PipelineError(
            "PREDICATE_SCOPE_MISMATCH",
            "the producer lives in scope %r but the consumer is in %r"
            % (actual_scope, scope),
        )
    if (granularity_bits is not None and actual_granularity_bits is not None
            and granularity_bits != actual_granularity_bits):
        raise PipelineError(
            "MASK_GRANULARITY_MISMATCH",
            "the producer applies %d-bit granularity but the consumer needs %d bits"
            % (actual_granularity_bits, granularity_bits),
        )
    # Independent recomputation: if bounds are given, recompute S here and ignore any
    # supplied 'required' (an authored count must not certify itself).
    if logical_len is not None or base is not None or lane_count is not None:
        lanes = lane_count if lane_count is not None else base
        start = origin
        if start is None and isinstance(logical_len, int) and isinstance(lanes, int) and lanes > 0:
            start = _default_issue_origin(logical_len, lanes)
        needed = required_active_for_issue(logical_len, lanes, start)
    else:
        if _is_unknown(required):
            raise PipelineError(
                "PREDICATE_PROOF_UNAVAILABLE",
                "the required active set is UNKNOWN, so coverage cannot be proved; "
                "UNKNOWN is not treated as satisfied",
            )
        needed = set(required or ())
    if not needed:
        raise PipelineError(
            "PREDICATE_PROOF_UNAVAILABLE",
            "the required active set is empty, so coverage cannot be proved. An empty "
            "requirement would make any predicate look sufficient",
        )
    if actual is None:
        raise PipelineError(
            "MISSING_TAIL_PREDICATE",
            "the consumer needs the active set %r but no predicate was emitted at all"
            % (sorted(needed),),
        )
    emitted = set(actual or ())
    if not emitted:
        raise PipelineError(
            "MISSING_TAIL_PREDICATE",
            "the consumer needs %r but the emitted predicate is empty" % (sorted(needed),),
        )
    if emitted != needed:
        raise PipelineError(
            "MASK_COVERAGE_MISMATCH",
            "the emitted predicate covers %r but the bounds require %r (extra=%r, "
            "missing=%r). Coverage is recomputed from bounds, not read from the "
            "authored active_count"
            % (sorted(emitted), sorted(needed), sorted(emitted - needed),
               sorted(needed - emitted)),
        )


def verify_issue_predicate_coverage(logical_len, base, actual, *, origin=None,
                                    lane_count=None, **kwargs) -> None:
    """Recompute S from bounds and verify it against the emitted predicate.

    This is the entry the AFTER pass calls: the requirement side is derived here from
    bounds/domain, and 'actual' is whatever the lowering emitted. Neither side reads the
    other, so a wrong predicate cannot certify itself.
    """
    lanes = lane_count if lane_count is not None else base
    start = origin
    if start is None and isinstance(logical_len, int) and isinstance(lanes, int) and lanes > 0:
        start = _default_issue_origin(logical_len, lanes)
    verify_predicate_coverage(
        actual=actual, logical_len=logical_len, base=base, lane_count=lanes,
        origin=start, **kwargs,
    )


def verify_pipeline_preservation(expected, actual) -> None:
    """Refuse a loop that declared a pipeline and lost it, or changed it.

    Both sides are typed descriptors {"loop": key, "pipeline": spec}. expected comes
    from the authored manifest and is NEVER narrowed to match actual: a comparison
    whose expectation follows the thing being checked always passes.
    """
    if expected is None:
        raise PipelineError(
            "PIPELINE_EXPECTATION_MISSING",
            "no authored expectation was supplied, so this check would pass vacuously",
        )
    def _spec_of(side):
        if isinstance(side, dict):
            return side.get("pipeline")
        return getattr(side, "pipeline", None)
    def _loop_of(side):
        if isinstance(side, dict):
            return side.get("loop")
        return getattr(side, "loop", None)
    expected_spec = _spec_of(expected)
    actual_spec = _spec_of(actual)
    key = _loop_of(expected)
    if expected_spec is None:
        # The authored side declared nothing for this loop: there is no obligation.
        return
    if actual_spec is None:
        raise PipelineError(
            "PIPELINE_DECLARATION_LOST",
            "loop %r declared a pipeline (%r) and no longer carries one after "
            "cloning/lowering, while the authored source still declares it"
            % (key, expected_spec),
        )
    if expected_spec != actual_spec:
        raise PipelineError(
            "PIPELINE_DECLARATION_CHANGED",
            "loop %r declared %r but now carries %r"
            % (key, expected_spec, actual_spec),
        )


# ---------------------------------------------------------------------------
# Manifest construction and the TIR-side reader.
# ---------------------------------------------------------------------------

PIPELINE_MANIFEST_KEY = "ascend.pipeline_manifest"
MANIFEST_SCHEMA = "ascend.pipeline_manifest/1"


def build_pipeline_manifest(module) -> PipelineManifest:
    """Derive the read-only manifest from the AUTHORED module. Never from the TIR."""
    from tilefoundry.analysis.pipeline_reader import (
        collect_buffer_versions,
        collect_pipeline_declarations,
    )
    specs = collect_pipeline_declarations(module)
    versions = collect_buffer_versions(module)
    declarations = tuple(
        BufferDeclaration(buffer_key=key, versions=value)
        for key, value in sorted(versions.items())
    )
    return PipelineManifest(
        schema=MANIFEST_SCHEMA,
        loops=tuple(sorted(specs.items())),
        buffer_declarations=declarations,
    )


def collect_tir_pipeline_declarations(module) -> dict:
    """{loop position key: PipelineSpec} read from the LOWERED TIR loops.

    The Core reader walks HIR expressions; a lowered module holds TIR statements, so
    this walks the statement tree instead. Keys are positions in document order, the
    same shape the authored reader produces, because a key based on id() would differ
    between two views of one program and could never match.

    Every statement that HOLDS a body must be descended through by its REAL field:
    `Sequential.body`, `For.body`, `MeshScope.body`, `ExecutionScope.body`, and both
    arms of an `If`. A lowered pipelined loop sits inside a MeshScope and usually an
    ExecutionScope, so a reader that only knew the top-level container found nothing
    and its verdict was an empty actual -- the false PIPELINE_DECLARATION_LOST this
    function exists to avoid.
    """
    from tilefoundry.ir.tir.stmts import (  # noqa: PLC0415
        ExecutionScope,
        For,
        If,
        MeshScope,
        Sequential,
        While,
    )

    found: list = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, For):
            found.append(node)
            walk(node.body)
        elif isinstance(node, While):
            walk(node.body)
        elif isinstance(node, If):
            walk(node.then_body)
            walk(node.else_body)
        elif isinstance(node, (Sequential, MeshScope, ExecutionScope)):
            walk(node.body)

    for item in getattr(module, "functions", ()) or ():
        walk(getattr(item, "body", None))
    out: dict = {}
    for index, loop in enumerate(found):
        spec = getattr(loop, "pipeline", None)
        if spec is not None:
            iv = getattr(loop, "induction_var", None)
            out["loop[%d]:%s" % (index, getattr(iv, "name", "?"))] = spec
    return out


def _collect_actual_predicates(module) -> dict:
    """{issue_source_id: frozenset(lanes)} of predicates actually emitted in the TIR.

    D0 materialises no pset/pge predicate yet, so the lowered program has no producer to
    read and this returns {}. It exists so the AFTER pass compares a REAL actual side
    (absent today, hence MISSING_TAIL_PREDICATE) instead of feeding the required set back
    in, and so a later materialisation step has exactly one place to fill.
    """
    return {}


# ---------------------------------------------------------------------------
# The ordered BEFORE_LOWER / AFTER_LOWER entries.
# ---------------------------------------------------------------------------

def _verify_hir_part_attributes(module) -> None:
    """Refuse an authored HIR part that is not 0 (a directly constructed part1).

    Runs in the FIRST BEFORE_LOWER pass so the refusal is PART_SELECTION_UNSUPPORTED instead
    of a later generic verifier error: a directly built part1 must be named by the P chain.
    """
    from tilefoundry.ir.core import Call  # noqa: PLC0415

    def visit(node):
        if node is None:
            return
        if isinstance(node, (tuple, list)):
            for child in node:
                visit(child)
            return
        if isinstance(node, Call):
            op = getattr(node.target, "op", None)
            if op is not None:
                verify_tir_part(getattr(op, "part", None))
            for arg in node.args:
                visit(arg)
            return
        for attr in ("body", "then_body", "else_body", "yield_values", "args", "elements"):
            visit(getattr(node, attr, None))

    for function in getattr(module, "functions", ()) or ():
        visit(getattr(function, "body", None))


@register_pass(Ascend950PRTarget, PassPoint.BEFORE_LOWER)
class ValidatePipelineDeclarations(ModulePass):
    """BEFORE 1/2: validate authored declarations and capture the manifest.

    READ-ONLY with respect to the program: it does not rewrite SSA, allocations or
    accesses, so the AnalysisResult the caller already holds stays valid. It only
    records a derived typed manifest for the AFTER passes to compare against.
    """

    name = "d0.validate_pipeline_declarations"

    def run(self, module):
        _verify_hir_part_attributes(module)
        manifest = build_pipeline_manifest(module)
        for key, spec in manifest.loops:
            versions = {decl.buffer_key: decl.versions
                        for decl in manifest.buffer_declarations}
            validate_pipeline_declarations(spec, versions)
        module.metadata[PIPELINE_MANIFEST_KEY] = manifest
        return module


@register_pass(Ascend950PRTarget, PassPoint.BEFORE_LOWER)
class NormalizeSchedulePredicates(ModulePass):
    """BEFORE 2/2: derive immutable predicate requirements from bounds.

    D0 derives requirements for the pipeline loops it was given, from the loop extent
    in the authored source. Anything the D0 domain cannot express is refused as
    PREDICATE_PLAN_UNSUPPORTED rather than turned into a dummy predicate.
    """

    name = "d0.normalize_schedule_predicates"

    def run(self, module):
        manifest = module.metadata.get(PIPELINE_MANIFEST_KEY)
        if manifest is None:
            # ValidatePipelineDeclarations is ordered before this pass; reaching here
            # without a manifest means the ordering was broken, not that there is
            # nothing to do.
            raise PipelineError(
                "PIPELINE_EXPECTATION_MISSING",
                "NormalizeSchedulePredicates ran before the manifest existed, so the "
                "BEFORE ordering is broken",
            )
        if not manifest.loops:
            return module
        # D0 carries the loop BUDGET (PipelineSpec.num_stages) but NOT a per-issue source
        # access window. Deriving a requirement from the loop stop would be the design's
        # "loop stop == lane length" mistake, and fabricating an UNKNOWN/empty requirement
        # per loop (then skipping it in the AFTER pass) is exactly the wiring this change
        # removes. With no source bounds artifact there is nothing provable, so the
        # requirement set stays EMPTY: the public path must not invent a requirement it
        # cannot derive. The predicate RULES are exercised at the helper level, and the
        # AFTER pass refuses anything it is actually handed but cannot verify.
        #
        # This is a stated gap, not a silent pass: producing a real per-issue requirement
        # needs F/L source access windows, which the current typed structure does not have.
        module.metadata[PIPELINE_MANIFEST_KEY] = dataclass_replace(
            manifest, requirements=())
        return module


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class VerifyPredicateCoverage(ModulePass):
    """AFTER 1/2: coverage, recomputed rather than read back.

    D0 has no materialised tail predicates yet, so this pass refuses nothing it cannot
    prove and asserts nothing it has not derived. What it does enforce is the rule that
    matters structurally: if a requirement exists and no predicate was emitted for it,
    that is reported, and an UNKNOWN requirement is never treated as satisfied.
    """

    name = "d0.verify_predicate_coverage"

    def run(self, module):
        manifest = module.metadata.get(PIPELINE_MANIFEST_KEY)
        if manifest is None:
            return module
        actual_by_issue = _collect_actual_predicates(module)
        for requirement in manifest.requirements:
            if requirement.status == "UNKNOWN":
                # An UNKNOWN requirement cannot be proved. It must be refused, never
                # skipped: a skipped requirement is indistinguishable from a satisfied one.
                raise PipelineError(
                    "PREDICATE_PROOF_UNAVAILABLE",
                    "requirement %r is UNKNOWN, so its coverage cannot be proved; an "
                    "UNKNOWN requirement is not treated as satisfied"
                    % (requirement.issue_source_id,),
                )
            evidence = dict(requirement.bounds_evidence)
            missing = [key for key in ("logical_len", "lane_count", "origin") if key not in evidence]
            if missing:
                raise PipelineError(
                    "PREDICATE_SOURCE_UNBOUND",
                    "requirement %r carries no %s, so its active set cannot be re-derived"
                    % (requirement.issue_source_id, missing),
                )
            # Recompute S independently from the recorded bounds. NEVER feed
            # requirement.required_active back in as the actual side: that is the
            # required-vs-required self-comparison this wiring replaces.
            needed = required_active_for_issue(
                evidence["logical_len"], evidence["lane_count"], evidence["origin"]
            )
            actual = actual_by_issue.get(requirement.issue_source_id)
            verify_predicate_coverage(
                needed,
                actual,
                granularity_bits=requirement.granularity_bits,
                scope=requirement.scope,
            )
        return module


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class VerifyPipelinePreservation(ModulePass):
    """AFTER 2/2: the authored declaration must survive lowering.

    The expected side is the authored manifest captured BEFORE lowering. It is never
    rebuilt from the lowered module, because that would let a lost declaration shrink
    its own expectation. A missing manifest is refused rather than skipped: an AFTER
    check that silently does nothing is indistinguishable from one that passed.
    """

    name = "d0.verify_pipeline_preservation"

    def run(self, module):
        manifest = module.metadata.get(PIPELINE_MANIFEST_KEY)
        if manifest is None:
            raise PipelineError(
                "PIPELINE_EXPECTATION_MISSING",
                "no authored pipeline manifest reached AFTER_LOWER. Either the BEFORE "
                "entry did not run or its record was dropped across lowering; either "
                "way this check would pass vacuously",
            )
        if not manifest.loops:
            return module
        actual = collect_tir_pipeline_declarations(module)
        for key, spec in manifest.loops:
            verify_pipeline_preservation(
                {"loop": key, "pipeline": spec},
                {"loop": key, "pipeline": actual.get(key)},
            )
        return module


def dataclass_replace(instance, **changes):
    """Local import wrapper so the module does not pull dataclasses into __all__."""
    from dataclasses import replace as _replace
    return _replace(instance, **changes)



# ---------------------------------------------------------------------------
# task-77: part-selection wiring (BEFORE normalize, AFTER TIR verify).
# ---------------------------------------------------------------------------

PART_PROJECTION_KEY = "ascend.part_projection"
PART_SELECTION_KEY = "ascend.part_selection"
PART_SELECTION_STATUS_KEY = "ascend.part_selection_status"


@register_pass(Ascend950PRTarget, PassPoint.BEFORE_LOWER)
class NormalizePartSelection(ModulePass):
    """BEFORE: record the part ONLY through a unique transport projection.

    No projection artifact means NO part is recorded (status NO_PROJECTION) -- never a
    default of 0, which would be a guess about the transport path.
    """

    name = "d0.normalize_part_selection"

    def run(self, module):
        projection = module.metadata.get(PART_PROJECTION_KEY)
        if projection is None:
            module.metadata[PART_SELECTION_STATUS_KEY] = "NO_PROJECTION"
            return module
        module.metadata[PART_SELECTION_KEY] = normalize_part_selection(projection)
        module.metadata[PART_SELECTION_STATUS_KEY] = "NORMALIZED"
        return module


def _walk_tir_statements(node, visit):
    from tilefoundry.ir.tir.stmts import (  # noqa: PLC0415
        ExecutionScope,
        For,
        If,
        MeshScope,
        Sequential,
        While,
    )

    if isinstance(node, (list, tuple)):
        for item in node:
            _walk_tir_statements(item, visit)
    elif isinstance(node, For):
        visit(node)
        _walk_tir_statements(node.body, visit)
    elif isinstance(node, While):
        _walk_tir_statements(node.body, visit)
    elif isinstance(node, If):
        _walk_tir_statements(node.then_body, visit)
        _walk_tir_statements(node.else_body, visit)
    elif isinstance(node, (Sequential, MeshScope, ExecutionScope)):
        _walk_tir_statements(node.body, visit)
    else:
        visit(node)


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class VerifyTirPartSelection(ModulePass):
    """AFTER: a directly constructed TIR part (e.g. part=1) is refused by name."""

    name = "d0.verify_tir_part_selection"

    def run(self, module):
        from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415

        def visit(stmt):
            if isinstance(stmt, Evaluate):
                verify_tir_part(getattr(stmt.callable, "part", None))

        for function in getattr(module, "functions", ()) or ():
            _walk_tir_statements(getattr(function, "body", None), visit)
        return module




PREDICATE_PROVENANCE_KEY = "d0.predicate_provenance"


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class RecordPredicateProvenance(ModulePass):
    """AFTER: record ONLY the materialised actual predicate provenance.

    required is deliberately NOT recorded here -- L re-derives required independently from
    its own ledger facts. A missing pset/mask node writes nothing, so the predicate stays
    UNKNOWN (never a fabricated WIRED).
    """

    name = "d0.record_predicate_provenance"

    def run(self, module):
        from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415

        actual = None
        counter = 0

        def lanes_of(stmt):
            dst = stmt.args[-1] if stmt.args else None
            shape = getattr(getattr(dst, "type", None), "shape", ()) or ()
            count = 1
            for dim in shape:
                count *= int(dim)
            return count

        def visit(stmt):
            nonlocal actual, counter
            if not isinstance(stmt, Evaluate):
                return
            op = getattr(stmt, "callable", None)
            if type(op).__name__ not in ("Pset", "Pset16"):
                return
            ref = "%s:pset[%d]" % (getattr(stmt, "instruction", "tir"), counter)
            counter += 1
            if actual is None:
                actual = {"kind": "materialised", "ref": ref, "active": list(range(lanes_of(stmt)))}

        for function in getattr(module, "functions", ()) or ():
            _walk_tir_statements(getattr(function, "body", None), visit)
        if actual is not None:
            module.metadata[PREDICATE_PROVENANCE_KEY] = {"actual": actual}
        return module

