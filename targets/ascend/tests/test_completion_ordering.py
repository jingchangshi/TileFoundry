"""C completion ORDERING controls (CPU only).

These are the bad controls for the "47/47 VERIFIED" false positive: the coverage count
was an EXISTENCE count, not a semantic one.

  * ``COMPLETION_WAIT_BEFORE_SET`` was unreachable: it compared the TIR identity PATH of
    a Set with the PATH of a Wait for equality, and a Set statement is never a Wait
    statement, so the conjunct was always False. Fixed to compare real tree ordinals.
  * the per-generation dict comprehension keyed by generation silently merged several
    Set/Wait nodes of one generation into the last one.
  * the injector collapsed every generation of a multi-instance action onto
    ``paths[0]``, so 18 cross-iteration edges never got a Set that precedes their Wait.

The positive control is the real POC carrier (R's materialised, unrolled pipeline), which
must still be VERIFIED 47/47 with a REAL "wait after set" ordering for every checked
obligation. No criterion is relaxed here: the unrolled carrier orders correctly, and the
loop-nested carrier is refused by name because it has no prologue to cover iteration 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for entry in (
    REPO / "src",
    REPO / "targets" / "ascend",
    REPO / "targets" / "ascend" / "tests",
    REPO / "examples" / "rmsnorm_950pr",
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import dataclasses as _dc

from tilefoundry.ir.tir.flag import FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.visitor_registry.verify import verify_prim_function

from tilefoundry_ascend import sync as C
from tilefoundry_ascend.schedule.ledger import extract_ledger

_CACHE = {}


def _build():
    """Run the real L->S->R chain and return everything C needs. Cached per session."""
    if _CACHE:
        return _CACHE
    import unittest.mock as _mock

    import test_task83_vertical as T
    from poc_vertical_adapter import nominal_solve_result, project_problem
    from tilefoundry_ascend.plan import materialize as MAT
    from tilefoundry_ascend.plan import resolved_plan as RP
    from tilefoundry_ascend.solver import bridge as B

    sources, tir, problem, expansion, expected, _sp = T._l()
    r_problem, _ = project_problem(problem, tir, expansion.count, sources)
    model_facts = {
        "model_id": "m",
        "model_version": "v1",
        "memory_groups": [{
            "id": "g0",
            "capacity_bytes": max(1, sum(int(b["bytes"]) for b in r_problem["buffers"])),
            "buffers": [b["root"] for b in r_problem["buffers"]],
            "cost_model": "MODEL",
        }],
        "cost_model": "MODEL",
    }
    core = B.solve_real_problem(r_problem, model_facts, {})
    nominal = nominal_solve_result(core, r_problem)
    original_status = RP._solve_status
    with _mock.patch.object(
        RP, "_solve_status",
        lambda r: RP.SAT_POSTCHECKED if getattr(r, "status", "") == "NOMINAL_SAT" else original_status(r),
    ):
        plan = RP.build_resolved_plan(r_problem, nominal, {"L": "l", "S": "s", "C": "c", "F": "f", "P": "p"})
        base = 0
        facts = {}
        for buf in r_problem["buffers"]:
            facts[buf["root"]] = {"versions": 1, "alignment": 64, "arena_id": "arena0",
                                  "base_offset_bytes": base}
            base += ((buf["bytes"] + 63) // 64) * 64
        plan = RP.resolve_buffer_layout(plan, facts)
    rewritten, matmap = MAT.materialize_plan(tir, plan)
    ledger = extract_ledger(tir)
    required = C.derive_completion_obligations_from_ledger(
        ledger, expansion, problem=problem, expected=expected)
    caps = {"supported_pairs": [["vector", "DMA"], ["DMA", "vector"]],
            "same_pipe_pipe_order": False, "flag_pool_size": 256, "cost_model": "MODEL"}
    actionable = tuple(o for o in required if o.kind in ("RAW", "WAR", "FREE"))
    channels, _ = C.choose_completion_mechanisms(actionable, {}, caps)
    allocation = C.allocate_completion_resources(channels, [], caps)
    _CACHE.update(dict(
        tir=tir, rewritten=rewritten, matmap=matmap, ledger=ledger, problem=problem,
        expansion=expansion, expected=expected, required=required, caps=caps,
        actionable=actionable, channels=channels, allocation=allocation,
    ))
    return _CACHE


def _inject(**kwargs):
    ctx = _build()
    return C.inject_completion_anchored(
        ctx["rewritten"], None, ctx["actionable"], ctx["allocation"],
        channels=ctx["channels"], ledger=ctx["ledger"],
        materialization=ctx["matmap"], **kwargs)


def _verify(actual, placement):
    ctx = _build()
    return C.verify_completion_coverage_from_ledger(
        ctx["ledger"], ctx["expansion"], problem=ctx["problem"], actual_tir=actual,
        placement_map=placement, capabilities=ctx["caps"], expected=ctx["expected"])


def _channel_index():
    ctx = _build()
    out = {}
    for channel in ctx["channels"]:
        for oid in channel.covered_obligation_ids:
            out[oid] = channel
    return out


def _ordinals(tir):
    """Ordinals in the SAME path scheme _actual_nodes() reports (C._walk)."""
    return C._actual_path_ordinals(tir)


# --------------------------------------------------------------------------------------
# Positive control: the real carrier is VERIFIED *and* genuinely ordered.
# --------------------------------------------------------------------------------------

def test_baseline_is_verified_and_every_checked_edge_is_really_ordered():
    actual, placement = _inject()
    report = _verify(actual, placement)
    assert report.status == "VERIFIED", report.as_dict()
    assert report.covered == 47 and report.required == 47
    assert report.covered_by_kind == {"RAW": 12, "SSA": 16, "WAR_FREE": 18, "FINAL": 1}

    nodes = C._actual_nodes(actual)
    ordinals = _ordinals(actual)
    by_channel = _channel_index()
    checked = 0
    for obligation in _build()["actionable"]:
        channel = by_channel.get(obligation.obligation_id)
        assert channel is not None
        gens = {int(g) for g in obligation.generation_domain}
        sets = [p for p, n in nodes["set"]
                if n.channel_name == channel.channel_id and n.kind == C._flag_kind(obligation.kind)
                and int(n.generation) in gens]
        waits = [p for p, n in nodes["wait"]
                 if n.channel_name == channel.channel_id and n.kind == C._flag_kind(obligation.kind)
                 and int(n.generation) in gens]
        assert sets and waits, obligation.obligation_id
        checked += 1
        # The invariant the dead code claimed to check, now actually true.
        assert min(ordinals[p] for p in sets) < max(ordinals[p] for p in waits), (
            obligation.obligation_id, "wait is not after its set")
    assert checked == 30, checked


def test_cross_iteration_pipeline_is_not_misread_as_wait_before_set():
    """Forward control: Set@i0 / Wait@i1 is legal pipelining, not a violation."""
    ctx = _build()
    cross = [o for o in ctx["actionable"] if C._iteration_offset(o) >= 1]
    assert len(cross) == 18, len(cross)
    actual, placement = _inject()
    report = _verify(actual, placement)
    assert report.code is None and report.status == "VERIFIED", report.as_dict()
    assert all(C._iteration_offset(o) == 1 for o in cross)

    nodes = C._actual_nodes(actual)
    ordinals = _ordinals(actual)
    by_channel = _channel_index()
    for obligation in cross:
        channel = by_channel[obligation.obligation_id]
        gens = {int(g) for g in obligation.generation_domain}
        sets = [(ordinals[p], int(n.generation)) for p, n in nodes["set"]
                if n.channel_name == channel.channel_id and int(n.generation) in gens]
        waits = [(ordinals[p], int(n.generation)) for p, n in nodes["wait"]
                 if n.channel_name == channel.channel_id and int(n.generation) in gens]
        # distance is real: generation g's set is at instance g-1, its wait at g.
        for generation in sorted(gens):
            set_ord = [o for o, g in sets if g == generation]
            wait_ord = [o for o, g in waits if g == generation]
            assert set_ord and wait_ord
            assert min(set_ord) < max(wait_ord), (obligation.obligation_id, generation)


# --------------------------------------------------------------------------------------
# Bad control 1: typed-legal swap of a matched Set/Wait pair must be REFUSED.
# --------------------------------------------------------------------------------------

def _swap_pair(tir, want_set, want_wait):
    """Swap two sibling statements in the innermost body that holds both.

    Returns (mutant, swapped) where swapped is False when no body holds both.
    """
    if isinstance(tir, Sequential):
        set_index = None
        wait_index = None
        for index, child in enumerate(tir.body):
            if want_set(child):
                set_index = index
            if want_wait(child):
                wait_index = index
        if set_index is not None and wait_index is not None:
            body = list(tir.body)
            body[set_index], body[wait_index] = body[wait_index], body[set_index]
            return _dc.replace(tir, body=tuple(body)), True
        for index, child in enumerate(tir.body):
            new_child, swapped = _swap_pair(child, want_set, want_wait)
            if swapped:
                body = list(tir.body)
                body[index] = new_child
                return _dc.replace(tir, body=tuple(body)), True
        return tir, False
    body = getattr(tir, "body", None)
    if isinstance(body, Sequential):
        new_body, swapped = _swap_pair(body, want_set, want_wait)
        if swapped:
            return _dc.replace(tir, body=new_body), True
    return tir, False


def _rebuild_placements(placement, tir):
    """Re-derive every SET/WAIT placement from the REAL node paths of the mutant TIR.

    Without this the refusal could be an artefact of a stale placement map. Rebinding the
    map to the mutated tree means the ONLY thing that can reject the mutant is ordering.
    """
    nodes = C._actual_nodes(tir)
    by_role = {}
    for path, node in nodes["set"]:
        by_role[(node.channel_name, "SET", int(node.generation))] = path
    for path, node in nodes["wait"]:
        by_role[(node.channel_name, "WAIT", int(node.generation))] = path
    placements = []
    for item in placement["placements"]:
        if item.role in ("SET", "WAIT"):
            path = by_role.get((item.channel_id, item.role, int(item.generation)))
            assert path is not None, (item.role, item.generation)
            placements.append(_dc.replace(item, actual_stmt_ref=path))
        else:
            placements.append(item)
    return dict(placement, placements=tuple(placements))


def test_typed_legal_set_wait_swap_is_refused_by_name():
    actual, placement = _inject()
    nodes = C._actual_nodes(actual)
    by_channel = _channel_index()

    # Pick a same-iteration (RAW) obligation: its set and wait must share one body sweep.
    target = next(o for o in _build()["actionable"]
                  if o.kind == "RAW" and C._iteration_offset(o) == 0)
    channel = by_channel[target.obligation_id]
    generation = int(target.generation_domain[0])
    set_paths = {p for p, n in nodes["set"]
                 if n.channel_name == channel.channel_id and int(n.generation) == generation}
    wait_paths = {p for p, n in nodes["wait"]
                  if n.channel_name == channel.channel_id and int(n.generation) == generation}

    def want_set(node):
        return isinstance(node, Evaluate) and isinstance(node.callable, FlagSet) and any(
            p in set_paths for p in _paths_of(actual, node))

    def want_wait(node):
        return isinstance(node, Evaluate) and isinstance(node.callable, FlagWait) and any(
            p in wait_paths for p in _paths_of(actual, node))

    mutant, swapped = _swap_pair(actual.body, want_set, want_wait)
    assert swapped, "no body held both the Set and the Wait"
    mutant = PrimFunction(name=actual.name, params=actual.params, body=mutant,
                          output_count=actual.output_count, target=actual.target)
    # Typed legality must not be broken, or the test would be measuring a different error.
    assert verify_prim_function(mutant) is None, verify_prim_function(mutant)

    fresh_placement = _rebuild_placements(placement, mutant)
    report = _verify(mutant, fresh_placement)
    assert report.status == "FAILED", report.as_dict()
    assert report.code == "COMPLETION_WAIT_BEFORE_SET", report.as_dict()


def _paths_of(tir, target_node):
    """Paths of one statement object, in the SAME scheme _actual_nodes() uses."""
    return [path for path, node in C._walk(tir) if node is target_node]


# --------------------------------------------------------------------------------------
# Bad control 2: a duplicated Set for one generation must be refused.
# --------------------------------------------------------------------------------------

def _duplicate_first(node, predicate):
    if isinstance(node, Sequential):
        for index, child in enumerate(node.body):
            if predicate(child):
                body = list(node.body)
                body.insert(index, child)
                return _dc.replace(node, body=tuple(body)), True
            new_child, hit = _duplicate_first(child, predicate)
            if hit:
                body = list(node.body)
                body[index] = new_child
                return _dc.replace(node, body=tuple(body)), True
        return node, False
    body = getattr(node, "body", None)
    if isinstance(body, Sequential):
        new_body, hit = _duplicate_first(body, predicate)
        if hit:
            return _dc.replace(node, body=new_body), True
    return node, False


def test_duplicate_set_for_one_generation_is_refused():
    actual, placement = _inject()
    nodes = C._actual_nodes(actual)
    by_channel = _channel_index()
    target = next(o for o in _build()["actionable"] if o.kind == "RAW")
    channel = by_channel[target.obligation_id]
    generation = int(target.generation_domain[0])
    set_paths = {p for p, n in nodes["set"]
                 if n.channel_name == channel.channel_id and int(n.generation) == generation}
    assert len(set_paths) == 1

    def want(node):
        return (isinstance(node, Evaluate) and isinstance(node.callable, FlagSet)
                and any(p in set_paths for p in _paths_of(actual, node)))

    mutant, duplicated = _duplicate_first(actual.body, want)
    assert duplicated
    mutant = PrimFunction(name=actual.name, params=actual.params, body=mutant,
                          output_count=actual.output_count, target=actual.target)
    assert verify_prim_function(mutant) is None, verify_prim_function(mutant)
    report = _verify(mutant, _rebuild_placements(placement, mutant))
    assert report.status == "FAILED", report.as_dict()
    assert report.code == "COMPLETION_DUPLICATE_FLAG_NODE", report.as_dict()


# --------------------------------------------------------------------------------------
# Bad control 3: a deleted Wait must be refused (and stay a NOT_VERIFIED count).
# --------------------------------------------------------------------------------------

def test_removed_wait_is_refused_with_the_existing_not_verified_code():
    ctx = _build()
    victim = ctx["actionable"][0]
    actual, placement = _inject(drop_wait_ids=(victim.obligation_id,))
    report = _verify(actual, placement)
    assert report.status == "FAILED", report.as_dict()
    assert report.code in ("COMPLETION_NOT_VERIFIED", "COMPLETION_DUPLICATE_FLAG_NODE"), report.as_dict()
    assert report.covered == 46, report.covered


# --------------------------------------------------------------------------------------
# Reachability: the loop-nested carrier cannot honour i0's wait, and says so by name.
# --------------------------------------------------------------------------------------

def test_loop_nested_carrier_without_prologue_is_named_startup_unsatisfied():
    """A non-peeled carrier has one instance per action, so every generation lands in one
    body sweep and the first iteration waits on a generation the producer has not set.

    That is a REAL gap of the carrier + injector, not a checker defect, so it must be
    refused by name rather than counted as coverage.
    """
    ctx = _build()
    actual, placement = C.inject_completion_anchored(
        ctx["tir"], None, ctx["actionable"], ctx["allocation"],
        channels=ctx["channels"], ledger=ctx["ledger"])
    assert verify_prim_function(actual) is None, verify_prim_function(actual)
    report = _verify(actual, placement)
    assert report.status == "FAILED", report.as_dict()
    assert report.code == "COMPLETION_STARTUP_UNSATISFIED", report.as_dict()


def test_wait_before_set_branch_is_reachable_at_all():
    """The branch used to be dead code; prove it fires on a TYPED-LEGAL minimal module.

    The probe keeps the carrier's real MeshScope (flag ops are mesh-scoped), so the only
    defect it carries is the statement order. A module that fails verify_prim_function
    would be testing some other error and would prove nothing about this branch.
    """
    ctx = _build()
    obligation = next(o for o in ctx["actionable"] if o.kind == "RAW")
    channel = _channel_index()[obligation.obligation_id]
    allocation_item = {item.channel_id: item for item in ctx["allocation"]}[channel.channel_id]
    generation = int(obligation.generation_domain[0])
    # Aiv* subclasses carry the mesh pattern; the bare Core FlagSet/FlagWait do not, and a
    # bare one would be refused for MESH reasons instead of ordering.
    wait_node = Evaluate(callable=C._flag_node(C.AivFlagWait, obligation, channel,
                                               allocation_item, generation), args=())
    set_node = Evaluate(callable=C._flag_node(C.AivFlagSet, obligation, channel,
                                              allocation_item, generation), args=())
    from tilefoundry.ir.tir.flag import Drain

    actual, _placement = _inject()
    from tilefoundry.ir.tir.stmts import For, MeshScope, Sequential as _Seq
    scope = actual.body
    while not isinstance(scope, MeshScope):
        scope = scope.body if isinstance(scope.body, _Seq) else scope.body[0]
    loop = next(child for child in scope.body.body if isinstance(child, For))
    inner = Sequential(body=(wait_node, set_node,
                            Evaluate(callable=C.AivDrain(), args=())))
    probe = PrimFunction(name="c_order_probe", params=(), 
                         body=_dc.replace(scope, body=Sequential(body=(
                             _dc.replace(loop, body=inner),))),
                         output_count=0)
    assert verify_prim_function(probe) is None, verify_prim_function(probe)
    # Placements must point at the probe's REAL node paths; a stale ref would be refused
    # as COMPLETION_MAP_MISMATCH before ordering is ever looked at.
    probe_nodes = C._actual_nodes(probe)
    set_path = next(p for p, n in probe_nodes["set"]
                    if n.channel_name == channel.channel_id and int(n.generation) == generation)
    wait_path = next(p for p, n in probe_nodes["wait"]
                     if n.channel_name == channel.channel_id and int(n.generation) == generation)
    placements = (
        C.ActionPlacement("wait", wait_path, "WAIT", channel.channel_id,
                          allocation_item.flag_id, generation, generation % obligation.versions,
                          obligation.versions, obligation.consumer_task,
                          obligation.consumer_action, (obligation.obligation_id,)),
        C.ActionPlacement("set", set_path, "SET", channel.channel_id,
                          allocation_item.flag_id, generation, generation % obligation.versions,
                          obligation.versions, obligation.producer_task,
                          obligation.producer_action, (obligation.obligation_id,)),
    )
    # Scope the requirement to the ONE obligation this probe materialises; otherwise the
    # other 29 obligations are genuinely absent and COMPLETION_NOT_VERIFIED (which now
    # correctly outranks ordering) would be reported instead.
    report = C.verify_completion_coverage_from_ledger(
        ctx["ledger"], ctx["expansion"], problem=ctx["problem"], actual_tir=probe,
        placement_map={"placements": placements, "channels": ctx["channels"],
                       "allocations": ctx["allocation"]},
        capabilities=ctx["caps"], expected=ctx["expected"], obligations=(obligation,))
    assert report.status == "FAILED", report.as_dict()
    assert report.code == "COMPLETION_WAIT_BEFORE_SET", report.as_dict()


# --------------------------------------------------------------------------------------
# RAW branch strictness: same rules as the reuse branch, identical result on 1:1 roots.
# --------------------------------------------------------------------------------------

def test_raw_branch_is_bit_identical_for_the_single_read_single_write_carrier():
    ctx = _build()
    raw = [o for o in ctx["required"] if o.kind == "RAW"]
    assert len(raw) == 12
    assert [o.obligation_id for o in raw] == [
        "RAW:X-in:i0", "RAW:X-in:i1", "RAW:X-in:i2", "RAW:X-in:i3",
        "RAW:Z-in:i0", "RAW:Z-in:i1", "RAW:Z-in:i2", "RAW:Z-in:i3",
        "RAW:Y-out:i0", "RAW:Y-out:i1", "RAW:Y-out:i2", "RAW:Y-out:i3",
    ]
    produced = {o.obligation_id: (o.producer_action, o.consumer_action) for o in raw}
    assert produced["RAW:X-in:i0"] == (
        "k/s0/MeshScope/s0/for:i/s2/GmToUb:1",
        "k/s0/MeshScope/s0/for:i/s6/ExecutionScope/s2/Vld:0")


def test_raw_branch_refuses_multi_read_multi_write_and_expands_one_to_many():
    ctx = _build()
    ledger = ctx["ledger"]
    entries = list(ledger.entries)

    class _FakeLedger:
        def __init__(self, rows, alias=None):
            self.entries = rows
            # root_alias is what maps the ExecutionScope-local roots onto the allocation
            # root the RAW family is keyed by; dropping it would test a different error.
            self.root_alias = dict(alias or {})

    # A second read entry on the same RAW family root: one-to-many must EXPAND, not pick [0].
    raw_root = next(str(e.root) for e in entries
                    if str(getattr(e, "space", "")) == "SMEM"
                    and str(getattr(e, "mode", "")) == "read"
                    and str(e.root).endswith("/s1"))
    extra_read = _dc.replace(next(e for e in entries
                                  if str(getattr(e, "space", "")) == "SMEM"
                                  and str(getattr(e, "mode", "")) == "read"
                                  and str(e.root) == raw_root),
                             action_id="probe/extra-read:0",
                             path="probe/extra-read")
    expanded = C._raw_obligations_from_ledger(
        _FakeLedger(entries + [extra_read], ledger.root_alias),
        ctx["expansion"], ctx["expected"])
    x_in = [o for o in expanded if o.obligation_id.split("#")[0] == "RAW:X-in:i0"]
    assert len(x_in) == 2, [o.obligation_id for o in x_in]
    assert len({o.obligation_id for o in x_in}) == 2
    assert {o.consumer_action for o in x_in} == {"probe/extra-read:0",
                                                 "k/s0/MeshScope/s0/for:i/s6/ExecutionScope/s2/Vld:0"}

    # Several reads AND several writes: refuse by name instead of silently pairing.
    extra_write = _dc.replace(next(e for e in entries
                                   if str(getattr(e, "space", "")) == "SMEM"
                                   and str(getattr(e, "mode", "")) == "write"
                                   and str(e.root).endswith("/s1")),
                              action_id="probe/extra-write:1",
                              path="probe/extra-write")
    try:
        C._raw_obligations_from_ledger(
            _FakeLedger(entries + [extra_read, extra_write], ledger.root_alias),
            ctx["expansion"], ctx["expected"])
    except C.CompletionError as error:
        assert error.code == "COMPLETION_RELATION_AMBIGUOUS", error.code
    else:
        raise AssertionError("multi-read/multi-write RAW root was silently paired")
