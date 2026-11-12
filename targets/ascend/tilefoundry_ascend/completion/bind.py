"""Binding the completion template to the plan's ACTUAL problem.

Two jobs, both about independence:

1. BIND. The obligations completion protects are the ones the solver and the
   plan were built against. So the only source of structural truth is
   ResolvedPlan.problem_ref, and the effects argument must be that problem's own
   Effect records, in the problem's own order, object for object (identity, not
   equality: two coordinate records with different ids prove nothing about which
   one the schedule used).

2. RE-ENUMERATE. verify_completion must be able to say what MUST exist without
   reading the template it is checking. This module derives that set straight
   from the problem's real records: every RAW/STATE dependence, every reuse
   edge, and one FREE obligation per MAXIMAL TERMINAL READER. If the template
   builder and this reader disagree, the template is refused - a shared bug in
   the builder is not a proof.

WHY "TERMINAL", NOT "LAST IN SOURCE ORDER" AND NOT "HIGHEST ISSUE_RANK". A slot
may be overwritten once every read of the generation it carries has LANDED. The
problem records which accesses terminate a buffer's reads
(LogicalBuffer.terminal_readers). plan/problem.py computes that set as the final
reader in access order for one root inside one region, which is a per-instance
statement: it is the right basis for the per-generation FREE publication, and it
is NOT a proof of ordering across two readers that live in different issue slots
of the same generation. So:

- one FREE channel per DISTINCT reader task appearing in terminal_readers, on
  that reader's own pipe, with the writer as consumer;
- readers are never merged because they share a pipe string, and never collapsed
  to "the one with the highest issue_rank". Merging needs a completion partial
  order this slice cannot state; it is refused by name instead (the spec's own
  condition: only readers with a PROVEN completion order on the same pipe/domain
  may share a flag).
"""
from __future__ import annotations

from .model import CompletionError

__all__ = ["bound_regions", "bound_effects", "obligations_of", "terminal_readers"]


def plan_content_verified(plan) -> None:
    """Recompute the plan's own content where it claims one, and refuse a mismatch.

    WHY A CONSUMER MUST RECOMPUTE. plan.plan_identity and plan.plan_canonical_bytes
    are STORED fields, and dataclasses.replace() carries stored fields across
    untouched. Measured (R20, fifth repeat of this shape): replacing one
    RegionPlan's lower/stop/step with (0, 0, 1) leaves plan_identity
    byte-identical to the original - the tampering did not even have to recompute
    anything - while the object the consumer holds now says the region never
    executes. A static zero trip count then silences every obligation this module
    derives, which is the coherently-deleted-obligation hole reached from the
    INPUT side instead.

    So this function refuses a plan whose recorded anchor disagrees with the
    bytes recomputed from the objects it actually holds, and treats "the plan
    declares an anchor but the anchor cannot be recomputed" as the same named
    refusal rather than a pass. plan_canonical_bytes/plan_identity_of are called
    from plan/resolved.py rather than re-implemented here: two recipes hashing one
    plan is how an honest recompute starts disagreeing with the recorded value.
    Code reused, not invented: COMPLETION_PLAN_STALE already means "this plan is
    not the content that was certified".

    A plan with NO recorded anchor at all (a hand-assembled ResolvedPlan, which
    the ring fixture is) has nothing to compare against. It is not silently
    trusted either: region_bounds() below is the dependency-free check - it ties
    each RegionPlan's bound Exprs to the problem's own Region objects by IDENTITY,
    and that is what a replace() on the RegionPlan breaks regardless of any
    digest - so the attack is closed on both kinds of plan.
    """
    from ..plan.resolved import plan_canonical_bytes

    stored = bytes(getattr(plan, "plan_canonical_bytes", b"") or b"")
    if not stored:
        return
    problem = getattr(plan, "problem_ref", None)
    try:
        canonical = plan_canonical_bytes(plan)
    except Exception as error:  # noqa: BLE001 - a claimed anchor must recompute
        raise CompletionError(
            "COMPLETION_PLAN_STALE",
            "the plan records %d bytes of canonical content that cannot be recomputed "
            "from the objects it now holds (%s), so its recorded identity is not "
            "evidence" % (len(stored), type(error).__name__),
            expected="recomputable plan content", actual=str(error)[:120]) from None
    if stored != canonical:
        raise CompletionError(
            "COMPLETION_PLAN_STALE",
            "the plan's recorded canonical bytes (%d) do not match the %d bytes "
            "recomputed from the objects it now holds, so its bounds/schedule are not "
            "the ones that were solved" % (len(stored), len(canonical)),
            expected=stored[:32].decode("utf-8", "replace"),
            actual=canonical[:32].decode("utf-8", "replace"))
    # And the DIGEST is a second, independent stored field: comparing only the
    # bytes would leave it unverified, which is the same defect one field over.
    # Measured (R21, lead's probe): replacing plan_identity with a fabricated
    # sha256 while leaving plan_canonical_bytes honest was ACCEPTED by build and
    # then VERIFIED, carrying the fabricated value into completion.plan_identity.
    # Not an authorization hole on its own - require_completion_verified goes
    # through the content registry and never reads that string - but a stored
    # field that is read and never recomputed is exactly the R21 shape, so the
    # digest is recomputed here from the same single recipe.
    from ..plan.resolved import plan_identity_of

    _canonical, recomputed_digest = plan_identity_of(plan)
    reported = str(getattr(plan, "plan_identity", "") or "")
    if reported and reported != recomputed_digest:
        raise CompletionError(
            "COMPLETION_PLAN_STALE",
            "the plan self-reports %s while its live content recomputes to %s, so the "
            "recorded identity is not the content that was solved"
            % (reported, recomputed_digest),
            expected=reported, actual=recomputed_digest)


def region_bounds(plan_region, region_problem) -> tuple:
    """The region's bound Exprs, after checking they ARE the problem's own.

    A static trip count of 0 is a legitimate answer for a region that really
    never executes (the ring at N=0 has no obligations and no events). It is NOT
    a license to report no obligations, because "0 trips" is a conclusion about
    the bounds and bounds live in two places: the RegionProblem's own Region and
    this RegionPlan's private copies. Object identity is what ties them: if the
    RegionPlan's Exprs are not the same objects the problem was built from, the
    zero came from someone's edit, not from the program.

    This is the input-trust question the fifth repeat of this shape demands:
    every "empty means none" branch has to be able to say WHERE its emptiness
    comes from. Compared by identity, not equality - equal content reached
    through different objects is exactly what replace() produces.
    """
    # region_problem is one RegionProblem, whose region_ref IS the Region the
    # plan's region_ref should be; NormalizedProblem.regions is the list above it.
    source = getattr(region_problem, "region_ref", None)
    if source is None:
        raise CompletionError(
            "COMPLETION_REGION_UNRESOLVED",
            "a RegionProblem declares no region_ref, so no plan region can be tied "
            "back to the bounds it was analysed from",
            getattr(getattr(plan_region, "region_ref", None), "region_id", ""))
    if plan_region.region_ref is not source:
        raise CompletionError(
            "COMPLETION_REGION_BINDING_MISMATCH",
            "region %r's plan does not carry this RegionProblem's own Region object, "
            "so its bounds are not the analysed ones"
            % (source.region_id,),
            expected=id(source), actual=id(plan_region.region_ref))
    region_id = source.region_id
    for name in ("lower_expr", "stop_expr", "step_expr"):
        planned = getattr(plan_region, name, None)
        declared = getattr(source, name, None)
        # Compared by CONTENT, matching what plan/resolved.py's own bounds check
        # does, so an Expr that was legitimately rebuilt into the same value is
        # not refused. The disagreement this has to catch is a value
        # disagreement: R20's tamper replaces the RegionPlan's copy with 0/0/1
        # while the problem's Region still says 0/1/1, and no content comparison
        # can make that pass.
        if planned is not declared and not _bounds_equal(planned, declared):
            raise CompletionError(
                "COMPLETION_BOUNDS_MISMATCH",
                "region %r's plan %s is not the bound the problem declares, so this "
                "region's trip count was read from an edited copy rather than from the "
                "program" % (region_id, name),
                expected=repr(declared)[:80], actual=repr(planned)[:80])
    return (plan_region.lower_expr, plan_region.stop_expr, plan_region.step_expr)


def _bounds_equal(claimed, declared) -> bool:
    """Whether two bound Exprs denote the same thing, without raising."""
    if claimed is declared:
        return True
    from .plan import expr_equal

    try:
        return bool(expr_equal(claimed, declared))
    except Exception:  # noqa: BLE001 - uncomparable bounds are a disagreement
        return False


def bound_regions(plan, effects) -> tuple:
    """The plan's own problem regions, with the effects argument bound to them."""
    problem = getattr(plan, "problem_ref", None)
    from ..plan.problem import NormalizedProblem

    if not isinstance(problem, NormalizedProblem):
        raise CompletionError(
            "COMPLETION_PROBLEM_BINDING_UNAVAILABLE",
            "ResolvedPlan.problem_ref is %s, so completion has no RegionProblem to read "
            "accesses, effects, edges or reuse from; region(s) %s of this plan are "
            "therefore unbound" % (
                "None" if problem is None else "a %s" % type(problem).__name__,
                [region_plan.region_ref.region_id for region_plan in plan.regions]),
            expected="NormalizedProblem",
            actual="None" if problem is None else type(problem).__name__)
    if not problem.regions:
        raise CompletionError(
            "COMPLETION_EFFECTS_MISSING",
            "the bound problem declares no RegionProblem at all, so no effect, edge or "
            "reuse obligation exists to build a template from",
            expected="NormalizedProblem.regions", actual=())

    plan_content_verified(plan)
    recorded = str(getattr(plan, "problem_identity", "") or "")
    live = str(getattr(problem, "problem_identity", "") or "")
    if recorded and recorded != live:
        raise CompletionError(
            "COMPLETION_PROBLEM_STALE",
            "ResolvedPlan.problem_identity is %s but the bound problem now reports %s, so "
            "the plan was resolved against a different problem content" % (recorded, live),
            expected=recorded, actual=live)
    bound_effects(plan, effects)
    return problem.regions


def bound_effects(plan, effects) -> tuple:
    """Require effects to BE the problem's own Effect tuple, in problem order.

    One input shape, no polymorphic dispatch. The previous version accepted a
    NormalizedProblem, a RegionProblem, or a tuple of regions here, which meant
    the argument's TYPE decided which obligations existed - the exact thing a
    frozen boundary must not do.
    """
    problem = plan.problem_ref
    if not isinstance(effects, tuple):
        raise CompletionError(
            "COMPLETION_EFFECTS_UNBOUND",
            "effects must be the plan problem's real Effect tuple", "tuple",
            type(effects).__name__)
    expected = tuple(effect for region in problem.regions for effect in region.effects)
    if len(effects) != len(expected):
        raise CompletionError(
            "COMPLETION_EFFECTS_UNBOUND",
            "completion was given %d effect record(s) while the plan's problem declares "
            "%d" % (len(effects), len(expected)), len(expected), len(effects))
    for index, (offered, own) in enumerate(zip(effects, expected)):
        if offered is not own:
            raise CompletionError(
                "COMPLETION_EFFECT_NOT_FROM_PROBLEM",
                "effects[%d] is not the problem's own Effect record; obligations are "
                "built from the effect table the plan was resolved against" % index,
                getattr(own, "effect_id", "?"), getattr(offered, "effect_id", "?"))
    return expected


def terminal_readers(region_problem, buffer_logical) -> tuple:
    """The reader task ids that terminate this buffer's reads, per pipe.

    Read from the problem's own terminal_readers fact (access ids), resolved
    through AccessFact to (task_id, pipe). An access named there that is not a
    READ of this buffer is a refusal: the fact would be describing a different
    program than the one being scheduled.
    """
    access_by_id = {access.access_id: access for access in region_problem.accesses}
    if len(access_by_id) != len(region_problem.accesses):
        raise CompletionError("COMPLETION_ACCESS_DUPLICATE",
                              "two access facts share one access id",
                              region_problem.region_ref.region_id)
    pipe_of_access = {}
    for effect in region_problem.effects:
        for access_id in effect.access_ids:
            existing = pipe_of_access.get(access_id)
            if existing is not None and existing != effect.pipe:
                raise CompletionError("COMPLETION_ACCESS_PIPE_CONFLICT",
                                      "one access is claimed by two pipes", access_id,
                                      existing, effect.pipe)
            pipe_of_access[access_id] = effect.pipe
    roots = {id(buffer_logical.root_ref)}
    readers = []
    for access_id in buffer_logical.terminal_readers:
        access = access_by_id.get(access_id)
        if access is None:
            raise CompletionError("COMPLETION_TERMINAL_READER_UNKNOWN",
                                  "a terminal reader names an access this region does not "
                                  "declare", buffer_logical.buffer_id, access_id)
        if id(access.root_ref) not in roots or access.mode not in _READ_MODES:
            raise CompletionError("COMPLETION_TERMINAL_READER_MISMATCH",
                                  "a terminal reader of %r is not a read of that buffer"
                                  % (buffer_logical.buffer_id,), access_id, access.mode)
        pipe = pipe_of_access.get(access_id)
        if pipe is None:
            raise CompletionError("COMPLETION_ACCESS_UNSOURCED",
                                  "a terminal reader is named by no effect, so its pipe "
                                  "and completion are unknown", access_id)
        readers.append((pipe, access.task_id, access_id))
    return tuple(readers)


_READ_MODES = frozenset({"READ", "READ_WRITE"})


def exit_obligations(region_problem, plan_region) -> tuple:
    """("EXIT", region_id, task_id) for every task owing a region-exit drain.

    Derived here from the problem's own records, independently of the builder's
    _drains(): the two must agree, and neither reads the completion being
    verified. A task that WRITES a GMEM buffer whose effect declares
    WRITE_VISIBLE has an asynchronous store; program order cannot retire it, so
    only a drain at the region exit can - UNLESS some declared RAW edge from
    that writer to a reader of the same buffer takes over the ordering, because
    then a READY channel protects the hand-off instead.

    WHY THE EDGE AND NOT AN ACCESS IS THE TEST (a hardening the challenger's
    attack forced). The first version dropped the drain whenever any READ access
    on that root existed. Measured consequence (independent_ub, unit:2 writes
    'out' with WRITE_VISIBLE): adding one fabricated READ access on 'out' with no
    effect-edge and no dependence removed ("EXIT", region:0, unit:2) from this
    derivation while adding NO READY obligation - a bare access deleted a
    synchronisation and nothing took its place. The two halves were keyed on
    different objects: the drain on an ACCESS, the READY on an EDGE. Now both
    read the same relation, so a forged access changes nothing and the only way
    to trade a drain for a flag is to declare the edge - which ADDS an
    obligation to the required set instead of silently removing one. The
    remaining route (forge an edge too) is the source audit's and the plan
    gate's duty (task-46/47), not this derivation's.

    This rule is deliberately STRICTER than the builder's. Where they disagree -
    a boundary write with in-region readers but no proving edge - this derivation
    still demands the drain, and the completion is refused with a named
    obligation rather than passing on a weaker argument. On both S0 programs the
    two agree, so no legitimate program is refused for this reason today.

    A region the plan bounds statically to zero trips executes nothing, so it
    owes no drain - but only if those bounds are the problem's own Expr objects.
    region_trip_count reads the RegionPlan's PRIVATE copies, and a region whose
    stop was replaced reads as zero trips while the program still runs the loop,
    which silently deletes every obligation (R20, fifth repeat of this shape:
    "0 / empty / consistent" treated as "none" without asking whether the input
    is trustworthy). So region_bounds() is called FIRST and the zero is honoured
    only when the Exprs are identical objects to the problem's Region's.
    """
    from .plan import _static_int, region_trip_count

    region_id = plan_region.region_ref.region_id
    region_bounds(plan_region, region_problem)          # bounds provenance
    if _static_int(region_trip_count(plan_region)) == 0:
        return ()

    completion_of = {}
    for effect in region_problem.effects:
        for access_id in effect.access_ids:
            completion_of[access_id] = effect.completion
    logical_by_root = {id(buffer.root_ref): buffer for buffer in region_problem.buffers}
    planned = {buffer.buffer_id for buffer in plan_region.buffers}
    task_ids = {task.task_id for task in region_problem.tasks}

    # Which (writer, buffer) pairs a declared RAW edge already covers: the edge
    # must point at a real task of this region that READS that same root.
    readers_by_task = {}
    for access in region_problem.accesses:
        if access.mode in _READ_MODES:
            readers_by_task.setdefault(access.task_id, set()).add(id(access.root_ref))
    covered = set()
    for edge in region_problem.edges:
        if edge.kind != "RAW" or edge.producer not in task_ids:
            continue
        if edge.consumer not in task_ids:
            # an edge to a task this region does not declare cannot discharge an
            # obligation in it
            continue
        for root in readers_by_task.get(edge.consumer, ()):
            covered.add((edge.producer, root))

    owed = []
    for task in region_problem.tasks:
        boundary = []
        for access in region_problem.accesses:
            if access.task_id != task.task_id or access.mode not in _WRITE_MODES:
                continue
            if completion_of.get(access.access_id) != "WRITE_VISIBLE":
                continue
            logical = logical_by_root.get(id(access.root_ref))
            if logical is None or logical.space != "gmem":
                continue
            if logical.buffer_id not in planned:
                continue
            if (task.task_id, id(access.root_ref)) in covered:
                continue
            boundary.append(logical.buffer_id)
        if boundary:
            owed.append(("EXIT", region_id, task.task_id))
    return tuple(owed)


def obligations_of(region_problem, plan_region) -> tuple:
    """(ready_keys, free_keys, reuse_by_buffer, exit_keys) for this region, derived.

    Derived from the problem's own records only, never from the template:

      ready_keys - one ("READY", region_id, producer_task, consumer_task) per
        declared RAW dependence. WAW/WAR/anything else is refused by name: this
        slice protects data-availability with events, and an ordering this table
        cannot state is not silently promoted to a flag.
      free_keys  - one ("FREE", region_id, buffer_id, reader_task) per maximal
        terminal reader of a buffer this looped region rewrites.
      reuse_by_buffer - the problem's own reuse record per buffer, so verify can
        compare a channel's distance_offset and generation delta against the
        obligation instead of against the channel itself.
    """
    region_id = plan_region.region_ref.region_id

    # A region the plan bounds statically to zero trips executes nothing, so it
    # owes no event: there is no generation whose read must land. This is the one
    # rule the builder and this enumeration share, and it is read from the plan's
    # own bounds on both sides rather than being assumed.
    from .plan import _static_int, region_trip_count

    region_bounds(plan_region, region_problem)          # bounds provenance
    if _static_int(region_trip_count(plan_region)) == 0:
        # Four values, the same shape every other return of this function has.
        # This branch used to return three, so a tampered or genuinely static-zero
        # plan reached _obligation_index's 4-way unpack as a bare ValueError -
        # an unnamed crash on the one path that must name what is missing. It was
        # never exercised because no test fed obligations_of a zero-trip region
        # until the lead's R20 probe did; the arity mismatch is now pinned by
        # test_zero_trip_region_is_refused_by_input_provenance_not_by_crash.
        return (), (), {}, ()

    ready_keys = []
    for edge in region_problem.edges:
        if edge.kind != "RAW":
            raise CompletionError(
                "COMPLETION_EDGE_KIND_UNSUPPORTED",
                "this slice protects RAW dependences with events; a %s edge needs a "
                "different completion argument" % (edge.kind,),
                "%s->%s in %s" % (edge.producer, edge.consumer, region_id))
        if int(edge.distance) != 0:
            raise CompletionError(
                "COMPLETION_GENERATION_DELTA_UNSUPPORTED",
                "a dependence across generations needs a value-generation map this "
                "slice cannot state", "%s->%s" % (edge.producer, edge.consumer))
        ready_keys.append(("READY", region_id, edge.producer, edge.consumer))

    reuse_by_buffer = {}
    access_by_id = {access.access_id: access for access in region_problem.accesses}
    logical_by_root = {id(buffer.root_ref): buffer for buffer in region_problem.buffers}
    for reuse in region_problem.reuse:
        writer = access_by_id.get(reuse.next_writer)
        reader = access_by_id.get(reuse.last_reader)
        if writer is None or reader is None:
            raise CompletionError("COMPLETION_REUSE_ENDPOINT_UNKNOWN",
                                  "a reuse edge names an access this region does not "
                                  "declare", reuse.buffer)
        buffer = logical_by_root.get(id(writer.root_ref))
        if buffer is None or buffer.buffer_id != reuse.buffer:
            raise CompletionError("COMPLETION_REUSE_BUFFER_UNKNOWN",
                                  "a reuse edge does not point at a buffer of this region",
                                  reuse.buffer)
        if reuse.buffer in reuse_by_buffer:
            raise CompletionError("COMPLETION_MULTIPLE_REUSE_UNSUPPORTED",
                                  "two reuse edges protect one buffer id", reuse.buffer)
        reuse_by_buffer[reuse.buffer] = reuse

    free_keys = []
    looped = getattr(plan_region.region_ref, "induction_var_ref", None) is not None
    if looped:
        plan_buffers = {buffer.buffer_id for buffer in plan_region.buffers}
        for buffer in region_problem.buffers:
            if buffer.buffer_id not in plan_buffers:
                continue
            writers = [access for access in region_problem.accesses
                       if access.root_ref is buffer.root_ref
                       and access.mode in _WRITE_MODES]
            if not writers:
                continue
            readers = terminal_readers(region_problem, buffer)
            if not readers:
                raise CompletionError(
                    "COMPLETION_TERMINAL_READERS_MISSING",
                    "a rewritten loop buffer declares no terminal reader, so no FREE "
                    "publication is owed and its reuse is unproven", buffer.buffer_id)
            if buffer.buffer_id not in reuse_by_buffer:
                raise CompletionError(
                    "COMPLETION_REUSE_UNPROVEN",
                    "a rewritten loop buffer carries no reuse obligation, so the slot's "
                    "reuse delay has no source", buffer.buffer_id)
            for _pipe, task_id, _access_id in readers:
                free_keys.append(("FREE", region_id, buffer.buffer_id, task_id))
    exit_keys = tuple(exit_obligations(region_problem, plan_region))
    return tuple(ready_keys), tuple(free_keys), reuse_by_buffer, exit_keys


# Access modes, matching plan/problem.py so "reader" and "writer" mean the same
# thing on both sides of the boundary.
_READ_MODES = frozenset({"READ", "READ_WRITE"})
_WRITE_MODES = frozenset({"WRITE", "READ_WRITE"})
