from dataclasses import replace
from itertools import product
import unittest

from implementation.modulo_kernel import (
    Assignment, Buffer, Capacity, Edge, Problem, ResourceUse, Reuse, Task,
    solve, verify_assignment,
)


def task(name, pipe, stage=None, latency=1, order=0):
    return Task(name, (ResourceUse(pipe, 1),), latency, order, stage)


class ModuloTests(unittest.TestCase):
    def test_fixed_versions_survive_model_extraction(self):
        p = Problem((task("load", "dma", 0, 2), task("vf", "v", 1, 2, 1)),
                    buffers=(Buffer("x", 64, 1, 3, 2),),
                    edges=(Edge("load", "vf", 0, 2),))
        a = solve(p).assignment
        self.assertEqual(a.versions["x"], 2)
        self.assertEqual(a.stages, {"load": 0, "vf": 1})

    def test_nonzero_authored_origin_is_preserved(self):
        p = Problem((task("load", "dma", 2, 2), task("vf", "v", 3, 2, 1)),
                    edges=(Edge("load", "vf", 0, 2),))
        a = solve(p).assignment
        self.assertEqual(a.stages, {"load": 2, "vf": 3})
        self.assertTrue(all(a.starts[n] == a.stages[n]*a.ii+a.phases[n] for n in a.starts))

    def test_partial_manual_stage_defaults_to_zero(self):
        p = Problem((task("load", "dma", None, 2), task("vf", "v", 1, 2, 1)),
                    edges=(Edge("load", "vf", 0, 2),))
        self.assertEqual(solve(p).assignment.stages, {"load": 0, "vf": 1})

    def test_no_offset_is_a_solver_constraint(self):
        p = Problem((task("a", "a", latency=2), task("b", "b", order=1)),
                    edges=(Edge("a", "b", 0, 2),), enable_offset=False)
        a = solve(p).assignment
        self.assertEqual(a.ii, 3)
        self.assertEqual(set(a.stages.values()), {0})

    def test_no_offset_rejects_nonzero_author_stage(self):
        with self.assertRaises(ValueError):
            solve(Problem((task("a", "a", 1),), enable_offset=False))

    def test_last_reader_reuse_determines_version(self):
        p = Problem((task("load", "dma", 0), task("consume", "v", 2, 1, 1)),
                    buffers=(Buffer("x", 64, 1, 4),),
                    edges=(Edge("load", "consume", 0, 1),),
                    reuse=(Reuse("consume", "load", "x", 1),), ii_max=1)
        a = solve(p).assignment
        self.assertEqual((a.ii, a.versions["x"]), (1, 3))

    def test_all_consumers_constrain_reuse(self):
        p = Problem((task("load", "dma", 0), task("early", "v0", 1, 1, 1),
                     task("late", "v1", 3, 1, 2)),
                    buffers=(Buffer("x", 64, 1, 5),),
                    reuse=(Reuse("early", "load", "x", 1), Reuse("late", "load", "x", 1)),
                    ii_max=1)
        self.assertEqual(solve(p).assignment.versions["x"], 4)

    def test_fixed_too_few_versions_is_unsat(self):
        p = Problem((task("load", "dma", 0), task("consume", "v", 2, 1, 1)),
                    buffers=(Buffer("x", 64, 1, 4, 1),),
                    reuse=(Reuse("consume", "load", "x", 1),), ii_max=3)
        self.assertEqual(solve(p).status, "UNSAT")

    def test_capacity_uses_padded_bytes_and_reserve(self):
        p = Problem((task("a", "p", 0),),
                    buffers=(Buffer("x", 128, 1, 4, 2),),
                    capacities=(Capacity("aiv0/UB", ("x",), 271, 16),))
        self.assertEqual(solve(p).status, "UNSAT")

    def test_resources_are_qualified_by_owner(self):
        p = Problem((task("a", "aiv0/V", 0), task("b", "aiv1/V", 0, order=1)))
        self.assertEqual(solve(p).assignment.ii, 1)
        p2 = replace(p, tasks=(p.tasks[0], replace(p.tasks[1], uses=(ResourceUse("aiv0/V", 1),))))
        self.assertEqual(solve(p2).assignment.ii, 2)

    def test_cyclic_resource_intervals_include_wrapping_overlap(self):
        p = Problem((Task("a", (ResourceUse("p", 2),), 2, 0, 0),
                     Task("b", (ResourceUse("p", 2),), 2, 1, 0)), ii_max=4)
        a = solve(p).assignment
        self.assertEqual(a.ii, 4)
        self.assertEqual((a.phases["b"]-a.phases["a"]) % 4, 2)

    def test_recurrence_sets_ii_lower_bound(self):
        p = Problem((task("u", "v", latency=7),), edges=(Edge("u", "u", 1, 7, "STATE"),))
        self.assertEqual(solve(p).assignment.ii, 7)

    def test_manual_same_pipe_source_order_uses_phase(self):
        p = Problem((task("first", "p", 2), task("second", "p", 0, order=1)),
                    manual_pipe_order=(("first", "second"),))
        a = solve(p).assignment
        self.assertLess(a.phases["first"], a.phases["second"])
        self.assertEqual(a.issue_order, ("first", "second"))

    def test_issue_after_does_not_create_a_completion_edge(self):
        p = Problem((task("a", "a", 0, 20), task("b", "b", 0, order=1)),
                    issue_after=(("a", "b"),), ii_max=1)
        a = solve(p).assignment
        self.assertEqual(a.ii, 1)
        self.assertEqual(a.issue_order, ("a", "b"))

    def test_cyclic_issue_order_is_unsat(self):
        p = Problem((task("a", "a"), task("b", "b", order=1)),
                    issue_after=(("a", "b"), ("b", "a")), ii_max=2)
        self.assertEqual(solve(p).status, "UNSAT")

    def test_deadline_is_unknown_not_unsat(self):
        r = solve(Problem((task("a", "a"),)), timeout_s=0)
        self.assertEqual(r.status, "UNKNOWN")
        self.assertIsNone(r.assignment)

    def test_forged_operand_schedule_assignment_is_rejected(self):
        p = Problem((task("a", "a", 0),), buffers=(Buffer("x", 32, 1, 4, 3),))
        a = solve(p).assignment
        with self.assertRaisesRegex(ValueError, "version contract"):
            verify_assignment(p, replace(a, versions={"x": 1}))

    def test_stage_bound_and_version_bound_are_independent(self):
        p = Problem((task("a", "a", 7),), buffers=(Buffer("x", 32, 1, 1, 1),), max_stage=7)
        a = solve(p).assignment
        self.assertEqual((a.stages["a"], a.versions["x"]), (7, 1))

    def test_positive_distance_war_not_reversed_raw(self):
        p = Problem((task("reader", "v", 0, 2), task("writer", "dma", 0, 1, 1)),
                    edges=(Edge("reader", "writer", 0, 2, "WAR"),), enable_offset=False)
        a = solve(p).assignment
        self.assertGreaterEqual(a.starts["writer"], a.starts["reader"]+2)

    def test_small_instances_against_exhaustive_phase_oracle(self):
        # The oracle enumerates only a tiny fixed-stage two-pipe family and does
        # not reuse the implementation's postcheck or constraint builder.
        for sa, sb, latency, distance in product(range(2), range(2), (1, 3), (0, 1)):
            p = Problem((task("a", "a", sa, latency), task("b", "b", sb, 1, 1)),
                        edges=(Edge("a", "b", distance, latency),), ii_max=6)
            feasible = []
            for ii in range(1, 7):
                if any(sb*ii+pb+distance*ii >= sa*ii+pa+latency
                       for pa, pb in product(range(ii), repeat=2)):
                    feasible.append(ii)
            result = solve(p)
            self.assertEqual(result.status, "SAT" if feasible else "UNSAT")
            if feasible:
                self.assertEqual(result.assignment.ii, min(feasible))


if __name__ == "__main__":
    unittest.main()
