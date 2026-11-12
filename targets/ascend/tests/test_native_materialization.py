"""Address semantics of the real public native materialization path."""
from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.materialize import index_statements_ledger

SOURCE = Path(__file__).resolve().parent / "fixtures/d0_pipeline/pipe_vadd_carrier.py"


def native_module(source=SOURCE):
    program = checked_program(str(source), None, command="native-aiv")
    return finalize(analyze(program.module, program.entry, analysis=("memory",)),
                    authored=program.authored)


def scalar(expr):
    kind = type(expr).__name__
    if kind == "Constant":
        return expr.value
    if kind == "Call" and type(expr.target).__name__ == "DimMul":
        return scalar(expr.args[0]) * scalar(expr.args[1])
    raise AssertionError("unsubstituted index: %r" % (expr,))


def input_offsets(module):
    offsets = []
    def visit(expr):
        if type(expr).__name__ == "Call":
            if type(expr.target).__name__ == "Slice" and getattr(expr.args[0], "name", "") == "x":
                offsets.append(scalar(expr.args[1].elements[0]))
            for arg in expr.args:
                visit(arg)
    for stmt in index_statements_ledger(module.entry_function()).values():
        if type(stmt).__name__ == "LetStmt" and stmt.var.name.startswith("value__"):
            visit(stmt.value)
    return offsets


def test_each_materialized_instance_uses_its_own_gm_input_slice():
    assert input_offsets(native_module()) == [0, 64, 128, 192]


def test_guard_substitution_and_nested_control_refusal():
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.stmts import If, While, Sequential
    from tilefoundry_ascend.plan.materialize import _rw_rewrite_body
    from tilefoundry_ascend.plan.resolved_plan import PlanError
    original = native_module().metadata["ascend.p3_pre_completion"]
    loop = next(n for n in index_statements_ledger(original).values()
                if type(n).__name__ == "For")
    empty = Sequential(body=())
    mapping = {loop.induction_var.name: Constant(type=loop.induction_var.type, value=2)}
    for guard in (If(cond=loop.induction_var, then_body=empty, else_body=empty),
                  While(cond=loop.induction_var, body=empty)):
        result = _rw_rewrite_body(guard, (), {}, {}, None, {"k": 0}, mapping)
        assert scalar(result.cond) == 2
    with pytest.raises(PlanError, match="nested For"):
        _rw_rewrite_body(loop, (), {}, {}, None, {"k": 0}, mapping)
    with pytest.raises(PlanError, match="unsupported statement"):
        _rw_rewrite_body(object(), (), {}, {}, None, {"k": 0}, mapping)


def test_nonzero_lower_and_step_keep_all_static_instances_executable():
    # The frontend D0 syntax intentionally allows only stop. Exercise the native
    # materializer contract with a typed nonzero-loop TIR and matching plan.
    from tilefoundry_ascend.plan.materialize import materialize_instruction_plan_v2
    module = native_module()
    original = module.metadata["ascend.p3_pre_completion"]
    loop = next(n for n in index_statements_ledger(original).values()
                if type(n).__name__ == "For")
    loop.start = replace(loop.start, value=2)
    loop.stop = replace(loop.stop, value=10)
    loop.step = replace(loop.step, value=2)
    plan = module.metadata["ascend.p3_plan"]
    plan = replace(plan, loop=replace(plan.loop, domain={"start": 2, "stop": 10, "step": 2}))
    rewritten, _ = materialize_instruction_plan_v2(original, plan)
    class View:
        def entry_function(self):
            return rewritten
    assert input_offsets(View()) == [128, 256, 384, 512]
    loops = [n for n in index_statements_ledger(rewritten).values()
             if type(n).__name__ == "For"]
    assert len(loops) == 1
    assert (scalar(loops[0].start), scalar(loops[0].stop), scalar(loops[0].step)) == (0, 1, 1)

