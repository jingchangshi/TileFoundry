from copy import deepcopy
from types import SimpleNamespace
import unittest

from implementation.compile_transaction import CompileRefused, ReplanRequired, run_native_chain


class Hooks:
    def __init__(self):
        self.mode = "ok"
        self.solve_calls = []
        self.emitted = 0
    def identity(self, m): return repr(m)
    def entry_identity(self, m): return repr(m["entry"])
    def form_units(self, m): return [m["entry"]]
    def analyze(self, m, units): return {"native": units[0]}
    def solve(self, p, cuts, *, timeout_s):
        self.solve_calls.append(cuts)
        return SimpleNamespace(status="UNKNOWN" if self.mode == "unknown" else "SAT")
    def resolve(self, m, units, problem, solution): return {"versions": 3, "ii": 2}
    def complete(self, m, plan, *, timeout_s): return {**plan, "completion": "symbolic"}
    def place(self, m, plan, *, timeout_s):
        if self.mode == "repeat_cut" or self.mode == "one_cut" and len(self.solve_calls) == 1:
            raise ReplanRequired(("limit_x_versions", 2))
        return {**plan, "storage": "physical"}
    def apply(self, m, plan):
        if self.mode == "metadata_only":
            return {**m, "metadata": {"symbolic_body": "new"}}
        if self.mode == "mutate_source":
            m["entry"] = "changed_source"
        return {"entry": ("symbolic_body", plan["ii"], plan["versions"])}
    def verify_ir(self, source, result, plan):
        assert result["entry"][1:] == (plan["ii"], plan["versions"])
    def emit(self, result, plan, *, timeout_s):
        self.emitted += 1
        if self.mode == "mutate_backend": result["entry"] = "wrong"
        return (result["entry"], "explicit_compile_config")
    def verify_emission(self, result, plan, artifact):
        if self.mode == "bad_conformance": raise CompileRefused("emitted IR mismatch")
        assert artifact[0] == result["entry"]


class TransactionTests(unittest.TestCase):
    def test_actual_entry_consumes_same_ii_and_versions(self):
        m = {"entry": "original"}
        result = run_native_chain(m, Hooks())
        self.assertEqual(result.module["entry"], ("symbolic_body", 2, 3))
        self.assertEqual(m, {"entry": "original"})

    def test_metadata_only_result_is_refused_before_emission(self):
        h = Hooks(); h.mode = "metadata_only"
        with self.assertRaisesRegex(CompileRefused, "not installed"):
            run_native_chain({"entry": "original"}, h)
        self.assertEqual(h.emitted, 0)

    def test_input_mutation_is_refused(self):
        h = Hooks(); h.mode = "mutate_source"
        with self.assertRaisesRegex(CompileRefused, "mutated the input"):
            run_native_chain({"entry": "original"}, h)

    def test_backend_mutation_is_refused(self):
        h = Hooks(); h.mode = "mutate_backend"
        with self.assertRaisesRegex(CompileRefused, "backend mutated"):
            run_native_chain({"entry": "original"}, h)

    def test_replan_uses_remaining_shared_budget_and_new_cut(self):
        h = Hooks(); h.mode = "one_cut"
        result = run_native_chain({"entry": "original"}, h)
        self.assertEqual(h.solve_calls, [(), (("limit_x_versions", 2),)])
        self.assertEqual(h.emitted, 1)

    def test_repeated_cut_cannot_loop(self):
        h = Hooks(); h.mode = "repeat_cut"
        with self.assertRaisesRegex(CompileRefused, "repeated"):
            run_native_chain({"entry": "original"}, h)
        self.assertEqual(len(h.solve_calls), 2)

    def test_unknown_is_not_runnable(self):
        h = Hooks(); h.mode = "unknown"
        with self.assertRaisesRegex(CompileRefused, "UNKNOWN"):
            run_native_chain({"entry": "original"}, h)
        self.assertEqual(h.emitted, 0)

    def test_failed_conformance_never_returns_ready(self):
        h = Hooks(); h.mode = "bad_conformance"
        with self.assertRaisesRegex(CompileRefused, "mismatch"):
            run_native_chain({"entry": "original"}, h)

    def test_expired_deadline_never_emits(self):
        h = Hooks()
        with self.assertRaisesRegex(CompileRefused, "deadline"):
            run_native_chain({"entry": "original"}, h, timeout_s=0)
        self.assertEqual(h.emitted, 0)


if __name__ == "__main__": unittest.main()
