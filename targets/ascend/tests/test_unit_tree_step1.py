"""structured-l/1 — ScheduleUnitTree + single-region lift.

Covers §5 step 1 of artifacts/run03/w45_vadd_pair/RESOLVEDPLAN_BLOCKED_DESIGN.md:
the two measurable trees (vadd and decode_dyn) and the lift that keeps identity.

Run:
  PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_unit_tree_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry.cli.source import checked_program
from tilefoundry.analysis.api import analyze
from tilefoundry.schedule.finalize import finalize
from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger
from tilefoundry_ascend.plan.unit_tree import (
    KIND_DMA, KIND_LOOP, KIND_VF, UnitTreeError, build_unit_tree, lift_region,
)

VADD = str(REPO / "targets" / "ascend" / "tests" / "fixtures" / "d0_pipeline"
           / "pipe_vadd_carrier.py") + ":PipeVaddCarrier"
DECODE = str(REPO / "targets" / "ascend" / "tests" / "fixtures" / "rmsnorm"
             / "rmsnorm_decode_56aiv_dyn.py") + ":RMSNormD"
PREFILL = str(REPO / "targets" / "ascend" / "tests" / "fixtures" / "rmsnorm"
              / "rmsnorm_prefill_56aiv_dyn.py") + ":RMSNormD"


def _load(fixture):
    """The one approved chain; deviating from it measures a different program."""
    prog = checked_program(fixture, None, command="structured-l")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    function = module.entry_function()
    ledger = extract_ledger(function)
    return ledger, function, build_problem(ledger)


def _vadd():
    ledger, function, problem = _load(VADD)
    return build_unit_tree(ledger, function), ledger, problem


def _decode():
    ledger, function, problem = _load(DECODE)
    return build_unit_tree(ledger, function), ledger, problem


# --------------------------------------------------------------- vadd: n=1 -> 4
def test_vadd_root_is_the_loop_alone():
    """Acceptance 1a: RootRegion(n=1), and that one child is the loop unit."""
    tree, ledger, _ = _vadd()
    root = tree.region_children("root")
    assert len(root) == 1, "root region must hold exactly the loop, got %d" % len(root)
    assert root[0].kind == KIND_LOOP
    assert root[0].loop_var == "i"


def test_vadd_loop_domain_is_zero_to_four():
    """Acceptance 1d: the loop domain survives as 0..4, read from the TIR node."""
    tree, _, _ = _vadd()
    loop = tree.loop_units()[0]
    assert loop.loop_domain["start"] == 0
    assert loop.loop_domain["stop"] == 4
    assert loop.loop_domain["step"] == 1


def test_vadd_loop_body_has_four_leaves_in_source_order():
    """Acceptance 1b: LoopBodyRegion(n=4) = LoadX, LoadZ, WholeVF, StoreY."""
    tree, ledger, _ = _vadd()
    loop = tree.loop_units()[0]
    body = tree.region_children(loop.unit_id)
    assert [u.kind for u in body] == [KIND_DMA, KIND_DMA, KIND_VF, KIND_DMA]

    ops = []
    for unit in body:
        ops.append(sorted({e.op for e in ledger.entries
                           if e.action_id in set(unit.leaf_action_ids)}))
    assert ops[0] == ["GmToUb"], "first leaf is LoadX, got %s" % ops[0]
    assert ops[1] == ["GmToUb"], "second leaf is LoadZ, got %s" % ops[1]
    assert ops[2] == ["Pset", "Vadd", "Vld", "Vsts"], "third leaf is WholeVF, got %s" % ops[2]
    assert ops[3] == ["copy"], "fourth leaf is StoreY, got %s" % ops[3]


def test_vadd_vf_actions_bind_to_the_whole_vf_exactly_once():
    """Acceptance 1c: the VF's internal actions belong to one leaf, once."""
    tree, ledger, _ = _vadd()
    vf_leaves = [u for u in tree.units.values() if u.kind == KIND_VF]
    assert len(vf_leaves) == 1
    whole_vf = vf_leaves[0]
    assert whole_vf.action_count == 11

    appearances = {}
    for unit in tree.leaves():
        for aid in unit.leaf_action_ids:
            appearances[aid] = appearances.get(aid, 0) + 1
    assert max(appearances.values()) == 1, "an action binds to more than one leaf"
    assert set(whole_vf.leaf_action_ids) <= set(appearances)


def test_vadd_source_inventory_is_not_lost():
    """Acceptance 1e: all 17 entries and all 8 instruction tasks survive the shrink."""
    tree, ledger, problem = _vadd()
    assert len(ledger.entries) == 17
    assert len(problem.tasks) == 8
    assert tree.unbound_entries(ledger) == [], "every ledger action must have a leaf owner"
    assert len(tree.bound_action_ids()) == 17


# ------------------------------------------------------ decode_dyn: 129 / 129
def test_decode_root_region_is_129():
    """Acceptance 2a: root n=129 = 64 DMA + 64 VF + 1 For."""
    tree, _, _ = _decode()
    root = tree.region_children("root")
    assert len(root) == 129, "root direct children must be 129, got %d" % len(root)
    counts = {}
    for unit in root:
        counts[unit.kind] = counts.get(unit.kind, 0) + 1
    assert counts == {KIND_DMA: 64, KIND_VF: 64, KIND_LOOP: 1}, counts
    assert root[-1].kind == KIND_LOOP, "the 129th root child is the For"


def test_decode_row_body_region_is_129():
    """Acceptance 2b: the row loop body is 128 DMA + 1 MainVF."""
    tree, _, _ = _decode()
    loop = tree.loop_units()[0]
    assert loop.loop_var == "r"
    body = tree.region_children(loop.unit_id)
    assert len(body) == 129, "row body direct children must be 129, got %d" % len(body)
    counts = {}
    for unit in body:
        counts[unit.kind] = counts.get(unit.kind, 0) + 1
    assert counts == {KIND_DMA: 128, KIND_VF: 1}, counts


def test_decode_all_65_vf_leaves_have_a_unique_owner():
    """Acceptance 3a: 65 whole-VF leaves, every action owned once."""
    tree, ledger, _ = _decode()
    vf_leaves = [u for u in tree.units.values() if u.kind == KIND_VF]
    assert len(vf_leaves) == 65, "expected 65 VF leaves, got %d" % len(vf_leaves)

    appearances = {}
    for unit in tree.leaves():
        for aid in unit.leaf_action_ids:
            appearances[aid] = appearances.get(aid, 0) + 1
    assert max(appearances.values()) == 1, "an action binds to more than one leaf"
    assert tree.unbound_entries(ledger) == []


def test_decode_main_vf_is_only_in_the_row_layer():
    """Acceptance 3b: 64 VF leaves at root, and the 65th (MainVF) inside the row loop.

    Judged from the tree relationship, not from a hardcoded statement index.
    """
    tree, _, _ = _decode()
    loop = tree.loop_units()[0]
    at_root = [u for u in tree.region_children("root") if u.kind == KIND_VF]
    in_row = [u for u in tree.region_children(loop.unit_id) if u.kind == KIND_VF]
    assert len(at_root) == 64
    assert len(in_row) == 1, "MainVF belongs to the row layer only"
    # the loop path is what marks the row layer; the root VFs must not carry it
    marker = "/for:%s/" % loop.loop_var
    assert marker in in_row[0].scope_path
    assert all(marker not in u.scope_path for u in at_root)


def test_prefill_keeps_both_loop_levels():
    """§3.2: prefill is a dynamic outer s with a static inner rr=4, not one flat loop."""
    ledger, function, _ = _load(PREFILL)
    tree = build_unit_tree(ledger, function)
    loops = {u.loop_var: u for u in tree.loop_units()}
    assert set(loops) == {"s", "rr"}, loops
    assert loops["rr"].loop_domain == {"start": 0, "stop": 4, "step": 1}
    root = tree.region_children("root")
    assert len(root) == 129
    outer = [u for u in root if u.kind == KIND_LOOP]
    assert len(outer) == 1 and outer[0].loop_var == "s"
    # s's body wraps rr rather than repeating the row list
    s_body = tree.region_children(outer[0].unit_id)
    assert len(s_body) == 1 and s_body[0].kind == KIND_LOOP
    assert len(tree.region_children(s_body[0].unit_id)) == 129


# ------------------------------------------------------------- the lift
def test_vadd_lift_uses_units_not_instruction_tasks():
    """Acceptance 4: the region schedules 4 units, not the original 8 tasks."""
    tree, _, problem = _vadd()
    loop = tree.loop_units()[0]
    lifted = lift_region(tree, problem, loop.unit_id)

    assert len(lifted.optimised_variable_ids) == 4, (
        "the loop body must be 4 optimisation variables, not %d"
        % len(lifted.optimised_variable_ids))
    assert len(lifted.optimised_variable_ids) != len(problem.tasks)
    # identity is preserved rather than renumbered
    assert lifted.source_task_count == 8
    assert lifted.source_access_count == 17
    assert len(lifted.task_bindings) == 8
    assert {b.source_task_id for b in lifted.task_bindings} == {
        str(t.stable_id) for t in problem.tasks}
    # the four VF instructions all land on the WholeVF leaf
    vf_bindings = [b for b in lifted.task_bindings if b.unit_id.endswith("#VF_LEAF")]
    assert len(vf_bindings) == 5, "5 of the 8 flat tasks are inside the whole VF"
    assert all(b.unit_id == lifted.optimised_variable_ids[2] for b in vf_bindings)


def test_vadd_root_lift_is_a_single_variable():
    """The root region has one child, so it needs no root-level solve."""
    tree, _, problem = _vadd()
    lifted = lift_region(tree, problem, "root")
    assert len(lifted.optimised_variable_ids) == 1
    assert tree.unit(lifted.optimised_variable_ids[0]).kind == KIND_LOOP


def test_decode_lift_regions_have_the_measured_sizes():
    tree, _, problem = _decode()
    root = lift_region(tree, problem, "root")
    assert len(root.optimised_variable_ids) == 129
    loop = tree.loop_units()[0]
    row = lift_region(tree, problem, loop.unit_id)
    assert len(row.optimised_variable_ids) == 129
    # both regions still account for the whole inventory
    assert root.source_task_count == len(problem.tasks)
    assert row.source_task_count == len(problem.tasks)


def test_lift_refuses_an_unknown_region():
    tree, _, problem = _vadd()
    try:
        lift_region(tree, problem, "no-such-region")
    except UnitTreeError as exc:
        assert exc.code == "REGION_UNKNOWN"
    else:
        raise AssertionError("an unknown region must be refused, not silently accepted")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((test.__name__, exc))
            print("FAIL %s: %s" % (test.__name__, exc))
        else:
            print("ok   %s" % test.__name__)
    print("\n%d passed, %d failed" % (len(tests) - len(failures), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
