"""Restricted native arithmetic emission and actual-TIR fault rejection."""
from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.codegen.native_aiv import NativeAIVBinding
from tilefoundry_ascend.codegen.x_runtime import CodegenError, emit_for_codegen
from tilefoundry_ascend.plan.materialize import index_statements_ledger

ROOT = Path(__file__).resolve().parents[1]


def build(source=None):
    source = source or ROOT / "tilefoundry_ascend/attnres/native_square.py"
    p = checked_program(str(source), None, command="native-aiv")
    return finalize(analyze(p.module, p.entry, analysis=("memory",)), authored=p.authored)


def emit(module):
    return emit_for_codegen(module.metadata["ascend.p3_plan"], NativeAIVBinding(module))


def nodes(module):
    return list(index_statements_ledger(module.entry_function()).values())


def test_native_mul_and_add_emit_different_real_arithmetic():
    mul = emit(build())
    add = emit(build(ROOT / "tests/fixtures/d0_pipeline/pipe_vadd_carrier.py"))
    assert "S.vmul(" in mul.source and "S.vadd(" in add.source
    assert mul.source_sha256 != add.source_sha256
    assert mul.pass_configs == (("tl.enable_auto_schedule", False),)
    assert "T.no_op" not in mul.source
    assert "T.copy(ub[128:192], out[192:256])" in mul.source
    assert 'T.ascend_wait_flag("V_MTE3", 2)' in mul.source
    assert 'T.ascend_wait_flag("MTE3_V", 2)' in mul.source


@pytest.mark.parametrize("output_count", [0, 2])
def test_invalid_native_output_count_is_refused(output_count):
    module = build()
    module.entry_function().output_count = output_count
    with pytest.raises(CodegenError):
        emit(module)


def test_extra_copy_writing_readonly_input_is_refused():
    from tilefoundry.ir.tir.stmts import Evaluate, Sequential
    from tilefoundry.ir.tir.memory.copy import Copy
    module = build()
    loop = next(n for n in nodes(module) if type(n).__name__ == "For")
    source_load = next(n for n in nodes(module)
                       if type(getattr(n, "callable", None)).__name__ == "GmToUb")
    store = next(n for n in nodes(module)
                 if type(getattr(n, "callable", None)).__name__ == "Copy")
    extra = Evaluate(callable=Copy(), args=(store.args[0], source_load.args[0]))
    loop.body = Sequential(body=tuple(loop.body.body[:-1]) + (extra, loop.body.body[-1]))
    with pytest.raises(CodegenError):
        emit(module)


def test_tf_author_product_uses_both_distinct_inputs():
    import torch
    from tilefoundry.evaluator import evaluate
    p = checked_program(str(ROOT / "tilefoundry_ascend/attnres/native_square.py"),
                        None, command="native-oracle")
    x = (torch.arange(256, dtype=torch.float32) - 113) / 16
    z = (torch.arange(256, dtype=torch.float32).remainder(17) - 7) / 8
    actual = evaluate(p.authored, x, z)
    assert torch.equal(actual, x * z)
    assert not torch.equal(actual, x * x)


def test_noncontiguous_native_parameter_is_refused():
    from tilefoundry.ir.types import Layout
    module = build()
    param = module.entry_function().params[0]
    param.type = replace(param.type, layout=Layout((256,), (2,)))
    with pytest.raises(CodegenError, match="contiguous"):
        emit(module)


def test_store_copy_is_in_the_unique_plan_with_last_reader_completion():
    module = build()
    plan = module.metadata["ascend.p3_plan"]
    assert any("/copy:0" in a.action_id and a.mode == "read" for a in plan.accesses)
    assert module.metadata["ascend.p3_completion"].verified
    for n in nodes(module):
        if type(n).__name__ == "LetStmt" and n.var.name.startswith("stored__"):
            assert n.value.args[0].value == 512


@pytest.mark.parametrize("field,value", [("slot", 1), ("flag_id", 7), ("generation", 99)])
def test_actual_wait_endpoint_mutations_are_rejected(field, value):
    module = build()
    wait = next(n for n in nodes(module) if type(getattr(n, "callable", None)).__name__ == "AivFlagWait")
    fields = ("channel_name", "kind", "slot", "generation", "versions", "flag_id",
              "producer_pipe", "consumer_pipe")
    kwargs = {name: getattr(wait.callable, name) for name in fields}
    kwargs[field] = value
    wait.callable = type(wait.callable)(**kwargs)
    with pytest.raises(CodegenError):
        emit(module)


def test_actual_guard_hiding_computation_is_rejected():
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.stmts import If, Sequential
    from tilefoundry.ir.types import make_tensor_type, DType
    module = build()
    loop = next(n for n in nodes(module) if type(n).__name__ == "For")
    loop.body = Sequential(body=(If(cond=Constant(type=make_tensor_type((), dtype=DType.bool), value=False), then_body=loop.body,
                                     else_body=Sequential(body=())),))
    with pytest.raises(CodegenError):
        emit(module)


def test_all_rebound_store_lets_are_rejected():
    module = build()
    for store in nodes(module):
        if type(store).__name__ == "LetStmt" and store.var.name.startswith("stored__"):
            store.value = replace(store.value, args=(replace(store.value.args[0], value=0),))
    with pytest.raises(CodegenError):
        emit(module)


def test_rebound_store_let_is_rejected():
    module = build()
    store = [n for n in nodes(module) if type(n).__name__ == "LetStmt"
             and n.var.name == "stored__k0"][-1]
    store.value = replace(store.value, args=(replace(store.value.args[0], value=0),))
    with pytest.raises(CodegenError):
        emit(module)
