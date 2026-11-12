import ast
from dataclasses import replace
import unittest

from implementation.modulo_kernel import Assignment
from implementation.symbolic_template import build_loop, final_free_generations, free_wait_generation


class PythonASTBuilder:
    """Diagnostic backend only. Production uses native TF builders."""
    def __init__(self): self.count = 0
    def const(self, value): return ast.Constant(value)
    def fresh_index(self, hint):
        self.count += 1
        return ast.Name(f"{hint}_{self.count}", ast.Load())
    def add(self, a, b): return ast.BinOp(a, ast.Add(), b)
    def sub(self, a, b): return ast.BinOp(a, ast.Sub(), b)
    def mul(self, a, b): return ast.BinOp(a, ast.Mult(), b)
    def floordiv(self, a, b): return ast.BinOp(a, ast.FloorDiv(), b)
    def maximum(self, a, b): return ast.Call(ast.Name("max", ast.Load()), [a, b], [])
    def mod(self, a, b): return ast.BinOp(a, ast.Mod(), b)
    def le(self, a, b): return ast.Compare(a, [ast.LtE()], [b])
    def lt(self, a, b): return ast.Compare(a, [ast.Lt()], [b])
    def logical_and(self, a, b): return ast.BoolOp(ast.And(), [a, b])
    def let(self, variable, value): return ast.Assign([ast.Name(variable.id, ast.Store())], value)
    def sequence(self, statements): return list(statements)
    def if_then(self, condition, body): return ast.If(condition, body, [])
    def for_range(self, variable, start, stop, step, body):
        return ast.For(ast.Name(variable.id, ast.Store()),
                       ast.Call(ast.Name("range", ast.Load()), [start, stop, step], []), body, [])


def template(lower=5, step=3, stages=(2, 4)):
    b = PythonASTBuilder()
    a = Assignment(2, {"load": stages[0]*2, "vf": stages[1]*2+1},
                   dict(zip(("load", "vf"), stages)), {"load": 0, "vf": 1},
                   {"x": 3}, ("load", "vf"))
    def task_body(name, inst):
        row = ast.Tuple([ast.Constant(name), inst.ordinal, inst.source_index,
                         inst.slot(b, 3), inst.slot(b, 3, 1)], ast.Load())
        return [ast.Expr(ast.Call(ast.Attribute(ast.Name("trace", ast.Load()), "append", ast.Load()), [row], []))]
    body = build_loop(b, lower=b.const(lower), stop=ast.Name("stop", ast.Load()), step=step,
                      tasks_by_id={"load": "load", "vf": "vf"}, assignment=a,
                      instantiate_whole_task=task_body)
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


class SymbolicTests(unittest.TestCase):
    def test_empty_short_and_long_loops_exact_instance_coverage(self):
        code = compile(template(), "<diagnostic-template>", "exec")
        for n in (0, 1, 2, 3, 4, 9, 101):
            env = {"stop": 5+3*n, "trace": []}
            exec(code, env)
            trace = env["trace"]
            self.assertEqual(len(trace), n*2)
            for name in ("load", "vf"):
                self.assertEqual([k for task, k, *_ in trace if task == name], list(range(n)))
            for _, k, i, slot, nextslot in trace:
                self.assertEqual((i, slot, nextslot), (5+3*k, k % 3, (k+1) % 3))

    def test_ragged_bound_uses_ceildiv(self):
        env = {"stop": 10, "trace": []}
        exec(compile(template(), "<template>", "exec"), env)
        self.assertEqual(sorted(set(r[2] for r in env["trace"])), [5, 8])

    def test_stop_before_start_is_empty(self):
        env = {"stop": -20, "trace": []}
        exec(compile(template(), "<template>", "exec"), env)
        self.assertEqual(env["trace"], [])

    def test_compiled_body_size_is_independent_of_runtime_n(self):
        tree = template()
        # N is one runtime binding, not a constructor input or an unroll count.
        self.assertEqual(sum(isinstance(n, ast.For) for n in ast.walk(tree)), 1)
        self.assertEqual(sum(isinstance(n, ast.If) for n in ast.walk(tree)), 2)
        self.assertLess(len(list(ast.walk(tree))), 180)

    def test_free_credit_initial_and_final_counts(self):
        for n in range(12):
            for v in range(1, 6):
                waits = [free_wait_generation(k, v) for k in range(n)]
                consumed = [g for g in waits if g is not None] + list(final_free_generations(n, v))
                self.assertEqual(sorted(consumed), list(range(n)))
                self.assertLessEqual(len(final_free_generations(n, v)), v)

    def test_free_credit_rejects_zero_versions(self):
        with self.assertRaises(ValueError):
            free_wait_generation(0, 0)


if __name__ == "__main__": unittest.main()
