"""The single CPU execution entry for REAL lowered native TIR.

One interpreter, in the framework, for numerical acceptance of S0/S1/S2. It runs
the actual PrimFunction produced by the normal entry, statement by statement, and
delegates every INSTRUCTION's value semantics to the existing
register_schedule_eval handlers (tilefoundry.evaluator.registry). This module
never re-implements an instruction's math; when a required handler is absent it
refuses by name and lists what is missing.

SUPPORTED STATEMENTS: Sequential, LetStmt, For, If, Evaluate, ExecutionScope,
MeshScope. Any other statement is a named refusal, never a silent skip.

MEMORY MODEL: the lowered program states non-GM windows as address recipes
(T.tensor_view(pointer-constant)); a TensorView binding maps to the storage tensor
at that (space, address), so two spellings of one window observe one physical
buffer, exactly as the TIR does. AllocTensor bindings are private tensors. An
instruction's write operand is updated IN PLACE, which is what makes a following
view of the same storage observe the store.

INPUTS/OUTPUTS: run_entry takes the public ABI names. Leading parameters are
inputs; the trailing output_count parameters are allocated outputs. Input arrays
are copied, so executing never mutates a caller's input. The return value is a
numpy dict keyed by output parameter name.
"""
from __future__ import annotations

import numpy as np
import torch

from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.registry import eval_registry, schedule_eval_registry
from tilefoundry.evaluator.value import EvalError, TensorValue, to_torch_dtype
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.hir.schedule import instruction_schema
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
from tilefoundry.ir.types import TensorType

__all__ = ["ExecutionRefusal", "run_entry", "supported_instructions"]

_SUPPORTED_STATEMENTS = (
    "Sequential", "LetStmt", "For", "If", "Evaluate", "ExecutionScope", "MeshScope",
)


class ExecutionRefusal(RuntimeError):
    """A named refusal from the CPU execution boundary."""

    def __init__(self, code: str, detail: str, subject: str = "") -> None:
        self.code = code
        self.detail = detail
        self.subject = subject
        super().__init__("%s: %s (subject=%r)" % (code, detail, subject))


def supported_instructions() -> tuple:
    """Registered schedule-eval instruction classes, by name."""
    return tuple(sorted(cls.__name__ for cls in schedule_eval_registry._map))


def _node(node) -> str:
    return type(node).__name__


def _static_shape(tensor_type) -> tuple:
    from tilefoundry.ir.types.utils import static_dim_value

    shape = []
    for dim in tuple(getattr(tensor_type, "shape", ()) or ()):
        value = static_dim_value(dim)
        if value is None:
            raise ExecutionRefusal(
                "EXEC_SHAPE_UNSUPPORTED",
                "a tensor shape dimension is not a static integer",
                str(tensor_type),
            )
        shape.append(int(value))
    return tuple(shape)


def _torch_dtype(tensor_type):
    try:
        return to_torch_dtype(tensor_type.dtype)
    except (EvalError, AttributeError):
        raise ExecutionRefusal(
            "EXEC_DTYPE_UNSUPPORTED", "no torch dtype for this declared type",
            str(getattr(tensor_type, "dtype", tensor_type)),
        ) from None


def _numpy_dtype(tensor_type):
    torch_dtype = _torch_dtype(tensor_type)
    return torch.empty((), dtype=torch_dtype).numpy().dtype


def _space_of(tensor_type) -> str:
    storage = getattr(tensor_type, "storage", None)
    if storage is None:
        raise ExecutionRefusal("EXEC_VIEW_UNRESOLVED",
                               "a viewed tensor carries no storage space", str(tensor_type))
    return getattr(storage, "name", str(storage)).lower()


class _Memory:
    """Shared storage for address-tagged windows, plus the Var window index."""

    def __init__(self) -> None:
        self.storages = {}
        self.windows = {}

    def storage(self, space: str, address: int, tensor_type: TensorType) -> torch.Tensor:
        key = (space, address)
        shape = _static_shape(tensor_type)
        current = self.storages.get(key)
        if current is None:
            current = torch.zeros(shape, dtype=_torch_dtype(tensor_type))
            self.storages[key] = current
            return current
        if tuple(current.shape) != shape:
            raise ExecutionRefusal(
                "EXEC_VIEW_SHAPE_MISMATCH",
                "two windows at one %s address state different shapes %s and %s"
                % (space, tuple(current.shape), shape),
                str(address),
            )
        return current


def _allocate(tensor_type: TensorType) -> torch.Tensor:
    return torch.zeros(_static_shape(tensor_type), dtype=_torch_dtype(tensor_type))


def _input_array(value, tensor_type: TensorType, name: str) -> torch.Tensor:
    shape = _static_shape(tensor_type)
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=_numpy_dtype(tensor_type)))
    except (TypeError, ValueError) as error:
        raise ExecutionRefusal("EXEC_INPUT_DTYPE", "input cannot be read as the declared "
                               "dtype: %s" % (error,), name) from None
    if array.shape != shape:
        raise ExecutionRefusal(
            "EXEC_INPUT_SHAPE", "input shape does not match the declared parameter",
            name,
        )
    return torch.from_numpy(array.copy())


def _scalar_param(value, tensor_type: TensorType, name: str):
    dtype = tensor_type.dtype
    try:
        if getattr(dtype, "name", "") == "f32":
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        raise ExecutionRefusal(
            "EXEC_INPUT_SCALAR", "scalar input is not a number", name) from None


def _resolve(module_or_function):
    if isinstance(module_or_function, PrimFunction):
        return module_or_function
    entry = getattr(module_or_function, "entry_function", None)
    if callable(entry):
        function = entry()
        if not isinstance(function, PrimFunction):
            raise ExecutionRefusal(
                "EXEC_ENTRY_UNSUPPORTED",
                "module entry is not a lowered PrimFunction", type(function).__name__)
        return function
    raise ExecutionRefusal(
        "EXEC_ENTRY_UNSUPPORTED",
        "run_entry takes a Module or a PrimFunction", type(module_or_function).__name__)


def run_entry(module_or_function, inputs):
    """Execute the REAL entry function once and return numpy outputs by name."""
    if not isinstance(inputs, dict):
        raise ExecutionRefusal("EXEC_INPUT_CONTAINER",
                               "inputs must be a dict keyed by public parameter name",
                               type(inputs).__name__)
    function = _resolve(module_or_function)
    output_count = int(getattr(function, "output_count", 0) or 0)
    parameters = tuple(function.params)
    if not 0 < output_count < len(parameters):
        raise ExecutionRefusal(
            "EXEC_OUTPUT_COUNT",
            "the entry must have a non-empty input list and at least one trailing output",
            "%d/%d" % (output_count, len(parameters)),
        )
    split = len(parameters) - output_count
    env = {}
    outputs = {}
    known = set()
    for index, parameter in enumerate(parameters):
        name = parameter.name
        known.add(name)
        tensor_type = parameter.type
        if not isinstance(tensor_type, TensorType):
            raise ExecutionRefusal("EXEC_PARAMETER_TYPE",
                                   "a parameter does not carry a tensor type", name)
        if index < split:
            if name not in inputs:
                raise ExecutionRefusal("EXEC_INPUT_MISSING",
                                       "a leading parameter has no input value", name)
            if tuple(getattr(tensor_type, "shape", ()) or ()) == ():
                env[id(parameter)] = _scalar_param(inputs[name], tensor_type, name)
            else:
                env[id(parameter)] = _input_array(inputs[name], tensor_type, name)
        else:
            tensor = _allocate(tensor_type)
            env[id(parameter)] = tensor
            outputs[name] = tensor
    extra = sorted(set(inputs) - known)
    if extra:
        raise ExecutionRefusal(
            "EXEC_INPUT_UNKNOWN", "inputs name no parameter of the entry", ", ".join(extra))
    memory = _Memory()
    _exec(function.body, env, memory)
    result = {}
    for name, tensor in outputs.items():
        result[name] = tensor.detach().cpu().numpy().copy()
    return result


def _exec(stmt, env, memory) -> None:
    kind = _node(stmt)
    if kind == "Sequential":
        for item in stmt.body:
            _exec(item, env, memory)
        return
    if kind in ("ExecutionScope", "MeshScope"):
        _exec(stmt.body, env, memory)
        return
    if kind == "LetStmt":
        _bind_let(stmt, env, memory)
        return
    if kind == "For":
        _exec_for(stmt, env, memory)
        return
    if kind == "If":
        branch = stmt.then_body if bool(_eval_scalar(stmt.cond, env)) else stmt.else_body
        _exec(branch, dict(env), memory)
        return
    if kind == "Evaluate":
        _exec_instruction(stmt, env, memory)
        return
    if kind == "While":
        raise ExecutionRefusal(
            "EXEC_STATEMENT_UNSUPPORTED",
            "a While statement is outside the native loop domain",
            kind,
        )
    raise ExecutionRefusal(
        "EXEC_STATEMENT_UNSUPPORTED",
        "statement is outside the executor's supported domain (%s)"
        % (", ".join(_SUPPORTED_STATEMENTS),),
        kind,
    )


def _bind_let(stmt: LetStmt, env, memory) -> None:
    value = stmt.value
    var = stmt.var
    tensor_type = var.type
    target = getattr(value, "target", None) if isinstance(value, type(stmt.value)) else None
    kind = _node(value) if not isinstance(value, (int, float, bool)) else "scalar"
    if kind == "Call":
        target = value.target
        target_name = _node(target) if not isinstance(target, type) else target.__name__
        if target_name == "AllocTensor":
            declared = getattr(target, "tensor_type", None) or tensor_type
            env[id(var)] = _allocate(declared)
            return
        if target_name == "TensorView":
            _bind_view(var, value, env, memory)
            return
        scalar = _eval_scalar(value, env)
        env[id(var)] = scalar
        return
    if kind == "Constant":
        env[id(var)] = value.value
        return
    if kind == "Var":
        source = env.get(id(value))
        if source is None:
            raise ExecutionRefusal("EXEC_UNBOUND_VAR",
                                   "a LetStmt reads a Var with no value", value.name)
        env[id(var)] = source
        return
    raise ExecutionRefusal(
        "EXEC_STATEMENT_UNSUPPORTED",
        "a LetStmt binds neither an allocation, a window nor a scalar expression",
        kind,
    )


def _bind_view(var: Var, view, env, memory) -> None:
    base = view.args[0] if view.args else None
    if isinstance(base, Constant):
        storage = getattr(base.type, "storage", None)
        if storage is None:
            raise ExecutionRefusal("EXEC_VIEW_UNRESOLVED",
                                   "a window base carries no storage space", var.name)
        space = getattr(storage, "name", str(storage)).lower()
        tensor = memory.storage(space, int(base.value), var.type)
        env[id(var)] = tensor
        memory.windows[id(var)] = (space, int(base.value))
        return
    if isinstance(base, Var):
        window = memory.windows.get(id(base))
        if window is not None:
            tensor = memory.storage(window[0], window[1], var.type)
            env[id(var)] = tensor
            memory.windows[id(var)] = window
            return
        source = env.get(id(base))
        if isinstance(source, torch.Tensor):
            if tuple(source.shape) != _static_shape(var.type):
                raise ExecutionRefusal("EXEC_VIEW_SHAPE_MISMATCH",
                                       "a window aliases a tensor of a different shape",
                                       var.name)
            env[id(var)] = source
            return
    raise ExecutionRefusal(
        "EXEC_VIEW_UNRESOLVED",
        "a window base is neither a pointer constant nor a bound tensor", var.name,
    )


def _exec_for(stmt: For, env, memory) -> None:
    start = int(_eval_scalar(stmt.start, env))
    stop = int(_eval_scalar(stmt.stop, env))
    step = int(_eval_scalar(stmt.step, env))
    if step == 0:
        raise ExecutionRefusal("EXEC_LOOP_STEP", "a loop step is zero", stmt.induction_var.name)
    for index in range(start, stop, step):
        inner = dict(env)
        inner[id(stmt.induction_var)] = index
        _exec(stmt.body, inner, memory)


def _operand(arg, env) -> TensorValue:
    if isinstance(arg, Var):
        value = env.get(id(arg))
        if value is None:
            raise ExecutionRefusal("EXEC_UNBOUND_VAR",
                                   "an instruction operand has no value", arg.name)
        if isinstance(value, torch.Tensor):
            return TensorValue(data=value, type=arg.type)
        return TensorValue(
            data=torch.as_tensor(value, dtype=_torch_dtype(arg.type)), type=arg.type)
    if isinstance(arg, Constant):
        tensor_type = arg.type
        if not isinstance(tensor_type, TensorType):
            raise ExecutionRefusal("EXEC_CONSTANT_UNSUPPORTED",
                                   "a constant operand carries no tensor type", _node(arg))
        return TensorValue(
            data=torch.as_tensor(arg.value, dtype=_torch_dtype(tensor_type)),
            type=tensor_type,
        )
    raise ExecutionRefusal(
        "EXEC_OPERAND_UNSUPPORTED",
        "an instruction operand is neither a Var nor a Constant", _node(arg))


def _exec_instruction(stmt: Evaluate, env, memory) -> None:
    op = stmt.callable
    cls = op if isinstance(op, type) else type(op)
    handler = schedule_eval_registry.lookup(cls)
    if handler is None:
        raise ExecutionRefusal(
            "EXEC_INSTRUCTION_UNSUPPORTED",
            "no @register_schedule_eval handler; supported instructions are %s"
            % (", ".join(supported_instructions()),),
            cls.__name__,
        )
    try:
        parameters, writes = instruction_schema(op, len(stmt.args))
    except ValueError as error:
        raise ExecutionRefusal("EXEC_INSTRUCTION_SCHEMA", str(error), cls.__name__) from None
    write_parameter = writes[-1]
    index = parameters.index(write_parameter)
    write_arg = stmt.args[index]
    if not isinstance(write_arg, Var):
        raise ExecutionRefusal(
            "EXEC_WRITE_TARGET",
            "an instruction's write operand is not a bound Var", write_parameter.name)
    destination = env.get(id(write_arg))
    if not isinstance(destination, torch.Tensor):
        raise ExecutionRefusal("EXEC_WRITE_TARGET",
                               "the write operand has no computed storage",
                               write_arg.name)
    values = tuple(_operand(arg, env) for arg in stmt.args)
    result_type = getattr(write_arg, "type", None)
    try:
        result = handler(EvaluateContext(op=op, args=values, result_type=result_type))
    except EvalError as error:
        raise ExecutionRefusal("EXEC_EVALUATOR_REFUSED", str(error), cls.__name__) from None
    if not isinstance(result, TensorValue) or not isinstance(result.data, torch.Tensor):
        raise ExecutionRefusal(
            "EXEC_EVALUATOR_REFUSED",
            "the registered evaluator returned no tensor value", cls.__name__)
    destination.copy_(result.data.to(dtype=destination.dtype).reshape(destination.shape))


def _eval_scalar(expr, env):
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, (int, float)):
        return expr
    kind = _node(expr)
    if kind == "Constant":
        return expr.value
    if kind == "Var":
        value = env.get(id(expr))
        if value is None:
            raise ExecutionRefusal("EXEC_UNBOUND_VAR",
                                   "a scalar expression reads a Var with no value", expr.name)
        return value
    if kind == "Call":
        return _call_scalar(expr, env)
    raise ExecutionRefusal(
        "EXEC_EXPR_UNSUPPORTED",
        "expression is outside the executor's scalar domain", kind)


def _call_scalar(call, env):
    target = call.target
    cls = target if isinstance(target, type) else type(target)
    values = []
    for arg in call.args:
        raw = _eval_scalar(arg, env)
        tensor_type = getattr(arg, "type", None)
        if not isinstance(tensor_type, TensorType):
            raise ExecutionRefusal("EXEC_EXPR_UNSUPPORTED",
                                   "a scalar operand carries no tensor type", _node(arg))
        values.append(TensorValue(
            data=torch.as_tensor(raw, dtype=_torch_dtype(tensor_type)), type=tensor_type))
    handler = eval_registry.lookup(cls) or schedule_eval_registry.lookup(cls)
    if handler is None:
        raise ExecutionRefusal(
            "EXEC_EXPR_UNSUPPORTED",
            "no registered evaluator for expression op %r" % (cls.__name__,),
            cls.__name__,
        )
    try:
        result = handler(EvaluateContext(
            op=target if not isinstance(target, type) else None,
            args=tuple(values),
            result_type=getattr(call, "type", None),
        ))
    except EvalError as error:
        raise ExecutionRefusal("EXEC_EVALUATOR_REFUSED", str(error), cls.__name__) from None
    if not isinstance(result, TensorValue) or result.data.numel() != 1:
        raise ExecutionRefusal(
            "EXEC_EXPR_UNSUPPORTED",
            "a scalar expression evaluated to a non-scalar value", cls.__name__)
    return result.data.reshape(()).item()
