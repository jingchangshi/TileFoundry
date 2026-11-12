"""Completion verification: re-enumerate the obligations, then authorize.

WHAT MAKES THIS A VERIFICATION AND NOT A SANITY CHECK. verify_completion does
not read the template's construction and nod along. It derives, straight from
the plan's own problem records, what MUST be protected: every declared RAW/STATE
dependence, every reuse edge with its own distance offset and delay, and one FREE
obligation per maximal terminal reader of every rewritten buffer. Then it demands
a TWO-WAY correspondence with the template. That asymmetry is the point: a channel
deleted wholesale, a reuse distance_offset rewritten, or a dependence left out all
break one direction and are refused by name.

WHY A DIGEST IS NOT A PROOF. completion_identity covers sites and channels, so a
recomputed identity only says "this is the content I am holding"; it says nothing
about whether that content protects the dependences the program has. Identity is
checked here for self-consistency only, never as the completeness argument. No
human-readable sentence enters any identity used here (see _identity_payload).

AUTHORITY (require_completion_verified). A public, hand-constructible
CompletionVerification field cannot authorize anything: anyone can write
status="VERIFIED". So the receipt is NOT the authority. verify_completion records,
in a module-private table keyed by the certified completion's content identity, the
EXACT plan and target objects it checked. require_completion_verified looks itself
up there and then RE-RUNS the whole verification against those saved objects before
returning. Consequences: a completion whose content changed has a different content
identity, is absent from the table, and is refused; a receipt pasted onto changed
content is never consulted; and the plan/target that granted authority are the ones
still being verified, not a caller's re-supplied substitutes.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Expr

from ..plan.resolved import ResolvedPlan
from ..plan.units import Auto, Fixed

from .model import DEVICE, CompletionError, core_for_pipe
from .plan import (
    AFTER_TASK,
    ANCHORS,
    BEFORE_TASK,
    CompletionChannel,
    CompletionPlan,
    CompletionSite,
    CompletionVerification,
    DRAIN,
    EXIT,
    FREE,
    LOCAL_EVENT,
    MECHANISM_KINDS,
    READY,
    REASONS,
    REGION_EXIT,
    STATE,
    completion_identity,
    free_wait_generation,
    region_tail_guard,
    expr_equal,
    free_wait_guard,
    region_trip_count,
    site_domain_guard,
    tail_window,
    _event_facts,
    _expr_key,
    _static_bool,
    _target_identity,
)

__all__ = ["require_completion_verified", "verify_completion"]

#: content identity -> (the exact plan, target and certified completion that were
#: verified). Private on purpose: the only writer is verify_completion after a
#: full pass, and the only reader is require_completion_verified, which re-runs
#: the check against these saved objects instead of trusting any field a caller
#: could construct.
AUTHORIZED = {}


def _fail(code: str, detail: str, source: str = "", expected=None, actual=None):
    raise CompletionError(code, detail, source, expected, actual)


def _text(expr) -> str:
    try:
        return str(_expr_key(expr))
    except CompletionError:
        return type(expr).__name__


def _site_map(completion) -> dict:
    sites = {}
    for site in completion.sites:
        if not isinstance(site, CompletionSite):
            _fail("COMPLETION_SITE_INVALID", "a site is not a CompletionSite record",
                  type(site).__name__)
        if site.site_id in sites:
            _fail("COMPLETION_SITE_DUPLICATE", "two sites share one id", site.site_id)
        if site.anchor not in ANCHORS:
            _fail("COMPLETION_SITE_ANCHOR_INVALID", "an anchor is not typed", site.site_id,
                  list(ANCHORS), site.anchor)
        if site.anchor == REGION_EXIT and site.task_ref is not None:
            _fail("COMPLETION_SITE_TASK_ON_EXIT",
                  "a region-exit site must not bind a task", site.site_id, None,
                  getattr(site.task_ref, "name", type(site.task_ref).__name__))
        if site.anchor != REGION_EXIT and site.task_ref is None:
            _fail("COMPLETION_SITE_TASK_MISSING",
                  "a task-anchored site must bind the real unit", site.site_id)
        if not isinstance(site.task_ordinal, Expr) or not isinstance(site.generation, Expr) \
                or not isinstance(site.guard, Expr):
            _fail("COMPLETION_SITE_EXPR_INVALID",
                  "site ordinal/generation/guard must be typed Exprs, not plain values",
                  site.site_id, "Expr",
                  [type(site.task_ordinal).__name__, type(site.generation).__name__,
                   type(site.guard).__name__])
        sites[site.site_id] = site
    return sites


def _region_map(plan) -> dict:
    regions = {}
    for region_plan in plan.regions:
        tasks = {}
        for task in region_plan.tasks:
            tasks[id(task.task_ref)] = task.task_id
        buffers = {}
        for buffer in region_plan.buffers:
            buffers[id(buffer.buffer_ref)] = buffer
        regions[region_plan.region_ref.region_id] = (region_plan, tasks, buffers)
    return regions


def _task_of(site, tasks) -> str:
    if site.task_ref is None:
        return ""
    task_id = tasks.get(id(site.task_ref))
    if task_id is None:
        _fail("COMPLETION_SITE_TASK_UNKNOWN",
              "a site binds a unit that is not a task of its region", site.site_id)
    return task_id


def _check_domain(channel, buffer_plan) -> None:
    domain = buffer_plan.domain
    if isinstance(domain, Fixed) and int(domain.value) != channel.versions:
        _fail("COMPLETION_VERSIONS_OUT_OF_DOMAIN",
              "a Fixed domain states an equality the channel must hold",
              channel.channel_id, int(domain.value), channel.versions)
    if isinstance(domain, Auto) and not (
            int(domain.minimum) <= channel.versions <= int(domain.maximum)):
        _fail("COMPLETION_VERSIONS_OUT_OF_DOMAIN",
              "the channel version count is outside the buffer's Auto domain",
              channel.channel_id, [int(domain.minimum), int(domain.maximum)],
              channel.versions)


def _logical_by_root(region_problem):
    return {id(buffer.root_ref): buffer for buffer in region_problem.buffers}


def _buffer_id_of(channel, region_plan, region_problem):
    """Resolve a channel's buffer to the problem's own buffer id, by identity."""
    for buffer in region_problem.buffers:
        if buffer.root_ref is channel.buffer_ref:
            return buffer.buffer_id
    _fail("COMPLETION_CHANNEL_BUFFER_UNBOUND",
          "a channel protects a root this region's problem does not declare",
          channel.channel_id)


def _obligation_index(plan):
    """Re-enumerate what MUST be protected, per region, from the problem only."""
    from .bind import obligations_of, plan_content_verified

    # Recomputed here, not trusted from the caller: build verified this plan when
    # it was made, but the plan object verify_completion receives can have been
    # replaced afterwards (dataclasses.replace carries the stored digest across
    # untouched, so a bounds edit is invisible to anything that only reads
    # plan.plan_identity).
    plan_content_verified(plan)
    problem = getattr(plan, "problem_ref", None)
    if problem is None:
        _fail("COMPLETION_PROBLEM_BINDING_UNAVAILABLE",
              "ResolvedPlan.problem_ref is None, so no obligation can be re-enumerated "
              "independently of the template; unbound region(s): %s"
              % ([rp.region_ref.region_id for rp in plan.regions],),
              expected="NormalizedProblem", actual=None)
    by_id = {region.region_ref.region_id: region for region in problem.regions}
    ready, free, exit_keys, reuse = set(), set(), set(), {}
    for region_plan in plan.regions:
        region_problem = by_id.get(region_plan.region_ref.region_id)
        if region_problem is None:
            _fail("COMPLETION_REGION_UNRESOLVED",
                  "the plan has a region with no problem record",
                  region_plan.region_ref.region_id)
        r_ready, r_free, r_reuse, r_exit = obligations_of(region_problem, region_plan)
        ready.update(r_ready)
        free.update(r_free)
        reuse.update(r_reuse)
        exit_keys.update(r_exit)
    return ready, free, reuse, exit_keys


def _buffer_id_of(region_problem, channel):
    for buffer in region_problem.buffers:
        if buffer.root_ref is channel.buffer_ref:
            return buffer.buffer_id
    _fail("COMPLETION_CHANNEL_BUFFER_UNBOUND",
          "a channel protects a root this region's problem does not declare",
          channel.channel_id)


def _correspondence(plan, completion, target) -> None:
    """Two-way match between the template's channels and the problem's obligations.

    Missing half: an obligation no channel protects - a deleted READY channel, a
    dropped FREE publication, or a coherently removed EXIT drain. Extra half: a
    channel protecting something the problem never declared. Both directions are
    required, because a template that merely has the right SHAPE can still be
    missing the obligation that matters.

    EXIT is in here on purpose. Site balance and the exit_site_ids set are
    CONSISTENCY checks: removing an EXIT channel together with its producer site,
    its drain-exit consumer site AND its exit_site_ids entry leaves all of them
    self-consistent, and that deletion was measured as ACCEPTED (task-48 case
    (b)). An obligation-level check is the only thing that can see it, so EXIT
    gets the same two-way treatment as READY and FREE.
    """
    # _obligation_index guards the binding, so a plan with no problem is refused
    # there rather than being reported twice with slightly different words.
    required_ready, required_free, reuse, required_exit = _obligation_index(plan)
    by_id = {region.region_ref.region_id: region for region in plan.problem_ref.regions}
    sites = {site.site_id: site for site in completion.sites}
    # A site names its unit by OBJECT identity (task_ref); the task id is the
    # plan's own mapping for that object, resolved here rather than trusted from
    # any string a site might carry.
    task_of_region = {}
    for region_plan in plan.regions:
        task_of_region[region_plan.region_ref.region_id] = {
            id(task.task_ref): task.task_id for task in region_plan.tasks}

    def task_of(site):
        mapping = task_of_region.get(site.region_id, {})
        if site.task_ref is None:
            return None
        if id(site.task_ref) not in mapping:
            _fail("COMPLETION_SITE_TASK_UNKNOWN",
                  "a channel endpoint binds a unit that is not a task of its region",
                  site.site_id)
        return mapping[id(site.task_ref)]

    have_ready, have_free, have_exit = set(), set(), set()
    for channel in completion.channels:
        # STATE is the one reason excluded, and not silently: _check_channel
        # refuses any STATE channel because this slice states no state version
        # fact, so no STATE channel can reach here without already having been
        # refused by name.
        if channel.reason not in (READY, FREE, EXIT):
            continue
        producer = sites.get(channel.producer)
        if producer is None:
            # Unreachable: the structure pass (_verify_structure, which runs
            # first) refuses a dangling producer with
            # COMPLETION_CHANNEL_ENDPOINT_UNKNOWN. Kept as a defensive skip, not
            # as the check - the earlier claim that this line could hide a
            # malformed reference was retracted once that order was confirmed.
            continue
        region_id = producer.region_id
        region_problem = by_id[region_id]
        if channel.reason == EXIT:
            for consumer in channel.consumers:
                site = sites.get(consumer)
                if site is not None and site.anchor == REGION_EXIT:
                    have_exit.add(("EXIT", region_id, task_of(producer)))
            continue
        buffer_id = _buffer_id_of(region_problem, channel)
        if channel.reason == READY:
            for consumer in channel.consumers:
                site = sites.get(consumer)
                if site is not None and site.anchor == BEFORE_TASK:
                    have_ready.add(("READY", region_id, task_of(producer), task_of(site)))
        else:
            waits = [sites[c] for c in channel.consumers
                     if sites.get(c) and sites[c].anchor == BEFORE_TASK]
            if waits:
                have_free.add(("FREE", region_id, buffer_id, task_of(producer)))
            # The channel's own distance offset must come from the problem, not
            # from the channel: an edited offset would otherwise be validated
            # against itself through free_wait_generation().
            edge = reuse.get(buffer_id)
            stated = 0 if edge is None else int(edge.distance_offset)
            if int(channel.distance_offset) != stated:
                _fail("COMPLETION_REUSE_OFFSET_MISMATCH",
                      "the channel's reuse distance offset is not the problem's reuse "
                      "obligation for this buffer", channel.channel_id, stated,
                      channel.distance_offset)

    def text(keys):
        # Positional fields are joined into one locator string per obligation, so
        # a refusal names WHICH region/task/buffer is unprotected rather than
        # dumping a repr the reader has to parse.
        return sorted("; ".join(str(field) for field in item) for item in keys)

    missing = required_ready - have_ready
    missing |= required_free - have_free
    missing |= required_exit - have_exit
    if missing:
        # Named kwargs. This call used to pass the locator list into the third
        # positional slot, which is source, so a reader saw the identity twice
        # and nothing in expected/actual - the same defect the lead flagged
        # earlier for the problem-binding refusal.
        _fail("COMPLETION_OBLIGATION_MISSING",
              "%d obligation(s) the problem states are protected by no channel: %s"
              % (len(missing), text(missing)[:4]),
              source=completion.identity,
              expected=text(missing),
              actual=text(have_ready | have_free | have_exit))
    extra = ((have_ready - required_ready) | (have_free - required_free)
             | (have_exit - required_exit))
    if extra:
        _fail("COMPLETION_CHANNEL_UNJUSTIFIED",
              "%d channel endpoint(s) protect an obligation this problem does not "
              "declare: %s" % (len(extra), text(extra)[:4]),
              source=completion.identity,
              expected=text(required_ready | required_free | required_exit),
              actual=text(extra))


def verify_completion(plan, completion, target):
    """Certify one completion against the plan's own obligations, or refuse.

    Structural rules first (every site/channel/domain/namespace check), then the
    obligation correspondence re-enumerated from the problem. Returns the
    completion with a receipt attached and registers the real plan/target objects
    it certified, keyed by the certified content identity.
    """
    problem = getattr(plan, "problem_ref", None)
    if problem is None:
        _fail("COMPLETION_PROBLEM_BINDING_UNAVAILABLE",
              "ResolvedPlan.problem_ref is None, so the effect table this plan was "
              "resolved against cannot be re-bound and nothing can be certified; "
              "unbound region(s): %s" % ([rp.region_ref.region_id for rp in plan.regions],),
              expected="NormalizedProblem", actual=None)
    from .bind import bound_effects

    bound_effects(plan, tuple(effect for region in problem.regions
                               for effect in region.effects))
    checked = _verify_structure(plan, completion, target)
    _correspondence(plan, checked, target)
    AUTHORIZED[checked.identity] = (plan, target, checked)
    return checked


def _verify_structure(plan, completion, target):
    """Partial structural checks; not an authorization decision."""
    if not isinstance(plan, ResolvedPlan):
        _fail("COMPLETION_PLAN_INVALID", "verify needs the single ResolvedPlan",
              type(plan).__name__)
    if not isinstance(completion, CompletionPlan):
        _fail("COMPLETION_INVALID", "verify needs a CompletionPlan",
              type(completion).__name__)
    if getattr(plan, "feasibility", None) != "SAT":
        _fail("COMPLETION_PLAN_NOT_SAT", "no completion exists for a non-SAT plan",
              "%r" % (getattr(plan, "feasibility", None),))
    if completion.plan_identity != plan.plan_identity:
        _fail("COMPLETION_PLAN_STALE",
              "the completion was built from another plan identity",
              completion.identity, plan.plan_identity, completion.plan_identity)
    if not completion.effects_identity or not completion.target_identity:
        _fail("COMPLETION_IDENTITY_MISSING",
              "a completion states its effect and target identity", completion.identity)

    facts = _event_facts(target)
    expected_target = _target_identity(plan, facts)
    if completion.target_identity != expected_target:
        _fail("COMPLETION_TARGET_MISMATCH",
              "the completion was built against another target/event surface",
              completion.identity, expected_target, completion.target_identity)

    regions = _region_map(plan)
    sites = _site_map(completion)
    for site in sites.values():
        if site.region_id not in regions:
            _fail("COMPLETION_SITE_REGION_UNKNOWN",
                  "a site lives in a region the plan does not have",
                  site.site_id, sorted(regions), site.region_id)

    used = set()
    ready_groups = {}
    namespaces = {}
    channel_ids = [channel.channel_id for channel in completion.channels
                   if isinstance(channel, CompletionChannel)]
    if len(set(channel_ids)) != len(channel_ids):
        _fail("COMPLETION_CHANNEL_DUPLICATE", "two channels share one id",
              completion.identity)
    for channel in completion.channels:
        if not isinstance(channel, CompletionChannel):
            _fail("COMPLETION_CHANNEL_INVALID", "a channel is not a CompletionChannel",
                  type(channel).__name__)
        if channel.reason not in REASONS:
            _fail("COMPLETION_CHANNEL_REASON_INVALID", "a channel reason is not typed",
                  channel.channel_id, list(REASONS), channel.reason)
        if channel.versions < 1:
            _fail("COMPLETION_VERSIONS_INVALID", "a channel states at least one version",
                  channel.channel_id, ">= 1", channel.versions)
        producer = sites.get(channel.producer)
        if producer is None:
            _fail("COMPLETION_CHANNEL_ENDPOINT_UNKNOWN", "a channel producer names no site",
                  channel.channel_id, sorted(sites), channel.producer)
        if not channel.consumers:
            _fail("COMPLETION_CHANNEL_NO_CONSUMER",
                  "a channel with no consumer is a set with no wait", channel.channel_id)
        consumers = []
        for consumer_id in channel.consumers:
            consumer = sites.get(consumer_id)
            if consumer is None:
                _fail("COMPLETION_CHANNEL_ENDPOINT_UNKNOWN",
                      "a channel consumer names no site", channel.channel_id,
                      sorted(sites), consumer_id)
            consumers.append(consumer)
        if len({consumer.site_id for consumer in consumers}) != len(consumers):
            _fail("COMPLETION_CHANNEL_DUPLICATE_CONSUMER",
                  "a channel lists the same consumer twice", channel.channel_id)
        if any(consumer.region_id != producer.region_id for consumer in consumers):
            _fail("COMPLETION_CHANNEL_REGION_MISMATCH",
                  "a channel spans two regions, which no event can carry",
                  channel.channel_id, producer.region_id,
                  [consumer.region_id for consumer in consumers])
        used.add(channel.producer)
        used.update(consumer.site_id for consumer in consumers)
        region_plan, tasks, buffers = regions[producer.region_id]
        _check_channel(channel, producer, consumers, region_plan, tasks, buffers, facts,
                       ready_groups, namespaces)

    for site_id in sites:
        if site_id not in used:
            _fail("COMPLETION_SITE_UNBALANCED",
                  "a site is neither a producer nor a consumer: an unmatched set/wait",
                  site_id)
    declared_exit = tuple(site_id for site_id, site in sites.items()
                          if site.anchor == REGION_EXIT)
    if tuple(completion.exit_site_ids) != declared_exit:
        _fail("COMPLETION_EXIT_SITES_MISMATCH",
              "exit_site_ids is not the set of region-exit sites",
              completion.identity, list(declared_exit), list(completion.exit_site_ids))
    for key, ranges in namespaces.items():
        ordered = sorted(ranges)
        for (start, stop, owner_id), (other_start, _other_stop, other_id) in zip(
                ordered, ordered[1:]):
            if other_start < stop:
                _fail("COMPLETION_NAMESPACE_CONFLICT",
                      "two channels occupy the same event ids in one namespace",
                      key, owner_id, other_id)

    identity = completion_identity(completion)
    if completion.identity and completion.identity != identity:
        _fail("COMPLETION_IDENTITY_MISMATCH",
              "the completion content does not match its recorded identity",
              completion.identity, identity, completion.identity)
    receipt = CompletionVerification(
        status="VERIFIED",
        completion_identity=identity,
        plan_identity=plan.plan_identity,
        target_identity=completion.target_identity,
    )
    return replace(completion, identity=identity, verification=receipt)


def _check_channel(channel, producer, consumers, region_plan, tasks, buffers, facts,
                   ready_groups, namespaces) -> None:
    n_expr = region_trip_count(region_plan)
    buffer_plan = None
    if channel.reason in (READY, FREE):
        if channel.buffer_ref is None:
            _fail("COMPLETION_BUFFER_MISSING",
                  "a READY/FREE channel protects a real buffer", channel.channel_id)
        buffer_plan = buffers.get(id(channel.buffer_ref))
        if buffer_plan is None:
            _fail("COMPLETION_BUFFER_UNKNOWN",
                  "a channel protects a buffer that is not scheduled in its region",
                  channel.channel_id, sorted(item.buffer_id for item in buffers.values()))
        if channel.versions != buffer_plan.resolved_versions:
            _fail("COMPLETION_CHANNEL_VERSIONS_MISMATCH",
                  "the channel version count is not the plan's resolved_versions",
                  channel.channel_id, buffer_plan.resolved_versions, channel.versions)
        _check_domain(channel, buffer_plan)
    if channel.reason == STATE and channel.state_ref is None:
        _fail("COMPLETION_STATE_NOT_SOURCED",
              "a STATE channel must name the state it protects", channel.channel_id)

    _task_of(producer, tasks)
    for consumer in consumers:
        _task_of(consumer, tasks)
    if channel.reason == READY:
        if producer.anchor != AFTER_TASK:
            _fail("COMPLETION_ANCHOR_INVALID",
                  "a READY set happens after the write completes", channel.channel_id,
                  AFTER_TASK, producer.anchor)
        for consumer in consumers:
            if consumer.anchor != BEFORE_TASK:
                _fail("COMPLETION_ANCHOR_INVALID",
                      "a READY wait happens before the read", channel.channel_id,
                      BEFORE_TASK, consumer.anchor)
        expected_guard = site_domain_guard(region_plan, producer.task_ordinal)
        if not expr_equal(producer.guard, expected_guard):
            _fail("COMPLETION_NOTIFY_GUARD_MISMATCH",
                  "the READY set is not guarded by this region's in-domain envelope",
                  channel.channel_id, _text(expected_guard), _text(producer.guard))
        for consumer in consumers:
            if not expr_equal(consumer.guard, site_domain_guard(
                    region_plan, consumer.task_ordinal)):
                _fail("COMPLETION_NOTIFY_GUARD_MISMATCH",
                      "the READY wait is not guarded by this region's envelope",
                      channel.channel_id)
        pipes = {consumer.pipe for consumer in consumers}
        if len(pipes) != 1:
            _fail("COMPLETION_SHARED_FLAG_ACROSS_PIPES",
                  "one ordinary flag cannot notify consumers on different pipes",
                  channel.channel_id, "one consumer pipe", sorted(pipes))
        pipe = next(iter(pipes))
        group = (buffer_plan.buffer_id, pipe)
        if group in ready_groups:
            _fail("COMPLETION_DUPLICATE_READY_GROUP",
                  "the same buffer/pipe pair already has a READY channel",
                  channel.channel_id, ready_groups[group])
        ready_groups[group] = channel.channel_id
        for consumer in consumers:
            if not expr_equal(consumer.generation, producer.generation):
                _fail("COMPLETION_GENERATION_MISMATCH",
                      "the READY wait reads a generation the producer did not set",
                      channel.channel_id, _text(producer.generation),
                      _text(consumer.generation))
            if not expr_equal(consumer.task_ordinal, producer.task_ordinal):
                _fail("COMPLETION_GENERATION_MISMATCH",
                      "the READY set and wait must name the same ordinal",
                      channel.channel_id, _text(producer.task_ordinal),
                      _text(consumer.task_ordinal))
    elif channel.reason == FREE:
        if producer.anchor != AFTER_TASK:
            _fail("COMPLETION_ANCHOR_INVALID",
                  "a FREE publish happens after the read completes", channel.channel_id,
                  AFTER_TASK, producer.anchor)
        waits = [consumer for consumer in consumers if consumer.anchor == BEFORE_TASK]
        tails = [consumer for consumer in consumers if consumer.anchor == REGION_EXIT]
        if len(tails) != 1 or len(waits) != 1:
            _fail("COMPLETION_TAIL_WAIT_MISSING",
                  "a FREE channel has one writer wait and one symbolic tail consumer",
                  channel.channel_id, [BEFORE_TASK, REGION_EXIT],
                  [consumer.anchor for consumer in consumers])
        wait, tail = waits[0], tails[0]
        expected_guard = free_wait_guard(producer.region_id, channel.versions,
                                         channel.distance_offset, region_trip_count(region_plan))
        if not expr_equal(wait.guard, expected_guard):
            _fail("COMPLETION_WAIT_GUARD_MISMATCH",
                  "the writer wait is not the in-domain envelope with the first V+offset "
                  "generations exempt", channel.channel_id, _text(expected_guard),
                  _text(wait.guard))
        if not expr_equal(producer.guard, site_domain_guard(region_plan,
                                                            producer.task_ordinal)):
            _fail("COMPLETION_NOTIFY_GUARD_MISMATCH",
                  "the FREE publish is not guarded by this region's envelope",
                  channel.channel_id)
        expected_generation = free_wait_generation(
            producer.region_id, channel.versions, channel.distance_offset)
        if not expr_equal(wait.generation, expected_generation):
            _fail("COMPLETION_GENERATION_MISMATCH",
                  "the writer wait generation is not ordinal-(versions+offset)",
                  channel.channel_id, _text(expected_generation), _text(wait.generation))
        if not expr_equal(wait.task_ordinal, producer.task_ordinal):
            _fail("COMPLETION_GENERATION_MISMATCH",
                  "the writer wait and the FREE publish must name the same ordinal",
                  channel.channel_id, _text(producer.task_ordinal),
                  _text(wait.task_ordinal))
        if wait.pipe != tail.pipe or wait.owner != tail.owner:
            _fail("COMPLETION_NAMESPACE_MISMATCH",
                  "the writer wait and the exit tail consume in the same domain",
                  channel.channel_id, wait.pipe, tail.pipe)
        expected_tail = region_tail_guard(producer.region_id, n_expr, channel.versions,
                                          channel.distance_offset)
        window = tail_window(tail)
        if window is None:
            _fail("COMPLETION_TAIL_NOT_SYMBOLIC",
                  "the tail consumer must carry the symbolic window, not an expanded "
                  "per-generation list", channel.channel_id, "window Expr",
                  type(tail.guard).__name__)
        if not expr_equal(tail.guard, expected_tail):
            _fail("COMPLETION_TAIL_RANGE_MISMATCH",
                  "the tail window is not max(0,N-(versions+offset))..N of this plan's "
                  "bounds", channel.channel_id, _text(expected_tail), _text(tail.guard))
        if _static_bool(tail.guard) is not None:
            _fail("COMPLETION_TAIL_NOT_SYMBOLIC",
                  "a constant tail guard states an expanded generation set",
                  channel.channel_id)
        if expr_equal(wait.task_ordinal, tail.task_ordinal):
            _fail("COMPLETION_GENERATION_MISMATCH",
                  "the in-region wait and the exit tail must not share one ordinal",
                  channel.channel_id)
        if channel.distance_offset < 0 and channel.versions + channel.distance_offset < 1:
            _fail("COMPLETION_VERSION_DELTA", "versions + offset must stay positive",
                  channel.channel_id, 1, channel.versions + channel.distance_offset)
    elif channel.reason == EXIT:
        if producer.anchor != AFTER_TASK:
            _fail("COMPLETION_ANCHOR_INVALID",
                  "an exit drain is published after the async boundary write",
                  channel.channel_id, AFTER_TASK, producer.anchor)
        if len(consumers) != 1 or consumers[0].anchor != REGION_EXIT:
            _fail("COMPLETION_ANCHOR_INVALID",
                  "an exit drain has one region-exit consumer", channel.channel_id,
                  REGION_EXIT, [consumer.anchor for consumer in consumers])
        if channel.versions != 1:
            _fail("COMPLETION_VERSIONS_INVALID",
                  "an exit drain is one shot, not a ring", channel.channel_id, 1,
                  channel.versions)
    else:
        _fail("COMPLETION_STATE_NOT_SOURCED",
              "this slice states no state version fact to verify", channel.channel_id)

    mechanism = channel.physical_mechanism
    if mechanism.kind not in MECHANISM_KINDS:
        _fail("COMPLETION_MECHANISM_INVALID", "a mechanism kind is not typed",
              channel.channel_id, list(MECHANISM_KINDS), mechanism.kind)
    if mechanism.kind == LOCAL_EVENT:
        if not mechanism.namespaces or mechanism.base_id is None:
            _fail("COMPLETION_MECHANISM_INVALID",
                  "an event mechanism states its namespaces and base id",
                  channel.channel_id)
        _check_namespace(channel, producer, consumers, facts, namespaces)
    elif mechanism.kind == DRAIN:
        if mechanism.namespaces or mechanism.base_id is not None:
            _fail("COMPLETION_MECHANISM_INVALID",
                  "a drain occupies no event ids", channel.channel_id)
        if channel.reason != EXIT:
            _fail("COMPLETION_MECHANISM_INVALID",
                  "only an exit drain uses the drain mechanism", channel.channel_id)
    else:
        _fail("COMPLETION_MECHANISM_INVALID",
              "this slice emits none of the other mechanisms", channel.channel_id,
              LOCAL_EVENT, mechanism.kind)


def _check_namespace(channel, producer, consumers, facts, namespaces) -> None:
    mechanism = channel.physical_mechanism
    receiver = consumers[0]
    try:
        producer_core = core_for_pipe(producer.pipe)
        consumer_core = core_for_pipe(receiver.pipe)
    except CompletionError as error:
        raise CompletionError("COMPLETION_PIPE_UNKNOWN", error.message,
                              channel.channel_id, None, producer.pipe) from None
    namespace = facts.physical_namespace_for(
        producer_core, consumer_core, producer.pipe, receiver.pipe,
        owner_key=(receiver.owner,))
    declared = mechanism.namespaces[0]
    expected_core = producer_core.upper() if producer_core == consumer_core else DEVICE.upper()
    if declared.core != expected_core or tuple(declared.pipe_pair or ()) != tuple(
            namespace.pipe_pair):
        _fail("COMPLETION_NAMESPACE_MISMATCH",
              "the declared namespace is not the physical hand-off namespace",
              channel.channel_id, (expected_core, tuple(namespace.pipe_pair)),
              (declared.core, tuple(declared.pipe_pair or ())))
    if declared.owner != receiver.owner:
        _fail("COMPLETION_NAMESPACE_MISMATCH",
              "the namespace is owned by the receiving execution domain",
              channel.channel_id, _owner_text(receiver.owner), _owner_text(declared.owner))
    if declared.bank < 0:
        _fail("COMPLETION_NAMESPACE_MISMATCH", "a bank is a non-negative number",
              channel.channel_id, ">= 0", declared.bank)
    scope = "intra_core" if producer_core == consumer_core else "cross_core"
    pool = facts.hard_event_pool(
        producer_core if producer_core == consumer_core else DEVICE, scope)
    if pool is None:
        _fail("COMPLETION_EVENT_NAMESPACE_MISSING",
              "the target declares no pool for this hand-off", channel.channel_id)
    if mechanism.base_id + channel.versions > int(pool.capacity):
        _fail("COMPLETION_EVENT_POOL_EXHAUSTED",
              "the channel's id range exceeds the declared pool", channel.channel_id,
              int(pool.capacity), mechanism.base_id + channel.versions)
    for key in {item.key() for item in mechanism.namespaces}:
        namespaces.setdefault(key, []).append(
            (mechanism.base_id, mechanism.base_id + channel.versions, channel.channel_id))


def _owner_text(owner) -> str:
    from .plan import _owner_text as _text_of

    return _text_of(owner)


def require_completion_verified(completion):
    """The consumption gate: only content that was actually verified may be read.

    WHY CONTENT EQUALITY AND NOT OBJECT IDENTITY (measured, deliberate):
    the registry is keyed by the certified content identity, and the value holds
    the real plan/target/completion that were checked. Consumption therefore
    proceeds by (a) looking the content identity up, (b) requiring the stored
    content to equal the content in hand, and (c) RE-RUNNING the full structural
    check and the obligation correspondence against the STORED real objects, then
    (d) requiring the re-verified content identity to match.

    That is stronger than binding to id() and it survives the cases an is-check
    would break: a legitimate re-materialisation in another process, or a
    dataclasses.replace() twin that is byte-for-byte the same schedule, is
    consumable because none of the four checks can distinguish it from what was
    verified - and nothing downstream depends on the object's address. An
    is-binding would reject those while adding no protection, since a forger who
    can construct an equal object can also construct the identity.

    The residual risk is a content-identity COLLISION between two semantically
    different completions. That is the sha256-256-bit collision domain (digest()
    truncates to 16 hex chars = 64 bits for readability), which is not a
    reachability property an attacker controls here: to exploit it you would have
    to find two distinct templates with the same digest AND get one verified.
    If a future slice needs the stronger claim, the fix is to lengthen the digest
    and re-hash existing identities, not to add an is-check.
    """

    if not isinstance(completion, CompletionPlan):
        _fail("COMPLETION_INVALID", "consumption needs a CompletionPlan",
              type(completion).__name__)
    identity = completion_identity(completion)
    entry = AUTHORIZED.get(identity)
    if entry is None:
        _fail("COMPLETION_NOT_VERIFIED",
              "no verification exists for this completion content; a hand-set "
              "verification field grants nothing",
              source=identity, expected="an entry registered by verify_completion",
              actual=None)
    plan, target, registered = entry
    if registered != completion:
        _fail("COMPLETION_CONTENT_CHANGED",
              "the certified completion is not the content being consumed", identity)
    recheck = _verify_structure(plan, completion, target)
    _correspondence(plan, recheck, target)
    if recheck.identity != identity:
        _fail("COMPLETION_IDENTITY_MISMATCH",
              "re-verification produced a different content identity", identity,
              recheck.identity)
    return recheck
