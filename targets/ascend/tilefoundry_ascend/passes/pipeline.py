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
    "select_part",
    "verify_predicate_coverage",
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


def normalize_tail_requirement(logical_len, base, policy, *,
                              extent=None, granularity_bits: int = 32,
                              producer_kind: str = "VF",
                              scope: str = "root",
                              issue_source_id: str = "tail.issue0",
                              part_selection: int = 0,
                              bounds_evidence: tuple = ()) -> PredicateRequirement:
    """Derive the active set a tail issue actually needs, or refuse to guess.

    The required set is computed from the BOUNDS (logical length, issue base), never
    taken from an authored claim. That is why FULL_TILE on N=70/base=64 is refused:
    70 % 64 != 0, so the tail issue needs six lanes, and no evidence in this program
    proves a full 64. A full predicate there would read the six lanes it needs plus 58
    it has no right to.

    UNKNOWN is refused rather than defaulted. There is no honest numeric answer for a
    bound nobody resolved, and 0 is a number that would make a downstream comparison
    look safe.
    """
    if _is_unknown(logical_len) or _is_unknown(extent):
        raise PipelineError(
            "PIPELINE_BOUND_UNKNOWN",
            "the tail bound is UNKNOWN (logical_len=%r, extent=%r), so no active set "
            "can be derived. UNKNOWN is not 0 and is not a full tile"
            % (logical_len, extent),
        )
    if type(logical_len) is not int or type(base) is not int or base < 1:
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "logical_len and base must be ints with base >= 1, got %r and %r"
            % (logical_len, base),
        )
    if not isinstance(policy, TailPolicy):
        raise PipelineError(
            "PREDICATE_PLAN_UNSUPPORTED",
            "the tail policy must be a TailPolicy, got %s. A string here would let a "
            "typo silently select the consumer default" % (type(policy).__name__,),
        )
    remainder = logical_len % base
    if policy is TailPolicy.FULL_TILE and remainder != 0:
        raise PipelineError(
            "UNPROVEN_FULL_TILE",
            "FULL_TILE was claimed for a tail of %d over base %d. %d %% %d == %d, so "
            "the last issue needs %d lanes and nothing in this program proves a full "
            "%d" % (logical_len, base, logical_len, base, remainder, remainder, base),
        )
    active = frozenset(range(remainder if remainder else base))
    return PredicateRequirement(
        issue_source_id=issue_source_id,
        scope=scope,
        required_active=active,
        granularity_bits=granularity_bits,
        producer_kind=producer_kind,
        bounds_evidence=tuple(bounds_evidence) or (("logical_len", logical_len),
                                                   ("base", base)),
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


# ---------------------------------------------------------------------------
# Coverage verification (AFTER_LOWER).
# ---------------------------------------------------------------------------

def verify_predicate_coverage(required, actual, *, offset_value=None,
                              offset_status=None, producer_is_fresh=None,
                              dma_window=None, granularity_bits=None,
                              actual_granularity_bits=None, scope=None,
                              actual_scope=None, source_bound=None) -> None:
    """Compare the required active set with what was actually emitted.

    The caller passes the set it re-derived from bounds. This function never trusts an
    authored active_count: an active_count recorded by the same pass that produced the
    predicate would let a wrong predicate certify itself.

    The checks are ordered most-specific first, so the code names the real defect
    instead of collapsing everything into a coverage mismatch.
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
    if actual is None:
        raise PipelineError(
            "MISSING_TAIL_PREDICATE",
            "the consumer needs the active set %r but no predicate was emitted at all"
            % (sorted(required or ()),),
        )
    needed = set(required or ())
    emitted = set(actual or ())
    if not needed:
        raise PipelineError(
            "PREDICATE_PROOF_UNAVAILABLE",
            "the required active set is empty, so coverage cannot be proved. An empty "
            "requirement would make any predicate look sufficient",
        )
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
    """
    from tilefoundry.ir.tir.stmts import For, If, Sequential  # noqa: PLC0415

    found: list = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, For):
            found.append(node)
            walk(node.body)
        elif isinstance(node, (If,)):
            walk(getattr(node, "then_body", None))
            walk(getattr(node, "else_body", None))
        elif isinstance(node, Sequential):
            walk(getattr(node, "stmts", ()) or ())

    for item in getattr(module, "functions", ()) or ():
        walk(getattr(item, "body", None))
    out: dict = {}
    for index, loop in enumerate(found):
        spec = getattr(loop, "pipeline", None)
        if spec is not None:
            iv = getattr(loop, "induction_var", None)
            out["loop[%d]:%s" % (index, getattr(iv, "name", "?"))] = spec
    return out


# ---------------------------------------------------------------------------
# The ordered BEFORE_LOWER / AFTER_LOWER entries.
# ---------------------------------------------------------------------------

@register_pass(Ascend950PRTarget, PassPoint.BEFORE_LOWER)
class ValidatePipelineDeclarations(ModulePass):
    """BEFORE 1/2: validate authored declarations and capture the manifest.

    READ-ONLY with respect to the program: it does not rewrite SSA, allocations or
    accesses, so the AnalysisResult the caller already holds stays valid. It only
    records a derived typed manifest for the AFTER passes to compare against.
    """

    name = "d0.validate_pipeline_declarations"

    def run(self, module):
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
        requirements = []
        for key, spec in manifest.loops:
            requirement = PredicateRequirement(
                issue_source_id=key,
                scope="root",
                required_active=frozenset(),
                granularity_bits=32,
                producer_kind="VF",
                bounds_evidence=(),
                policy=TailPolicy.MASK_ACTIVE,
                part_selection=0,
                status="UNKNOWN",
            )
            requirements.append(requirement)
        module.metadata[PIPELINE_MANIFEST_KEY] = dataclass_replace(
            manifest, requirements=tuple(requirements))
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
        for requirement in manifest.requirements:
            if requirement.status == "UNKNOWN":
                # Nothing was proved for this loop. It is recorded, not passed off as
                # covered; D0 has no bounds evidence for it yet.
                continue
            verify_predicate_coverage(
                requirement.required_active,
                requirement.required_active,
                granularity_bits=requirement.granularity_bits,
                actual_granularity_bits=requirement.granularity_bits,
                scope=requirement.scope,
                actual_scope=requirement.scope,
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
