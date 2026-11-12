"""Independent CPU evaluator for the supported native template subset.

This is a TEST ORACLE, not a compiler stage and not a device model. It executes the real
native TIR body the builders produce (For/Sequential/ExecutionScope/MeshScope/LetStmt with
AllocTensor, plus the transport instructions GmToUb/UbToGm/Pset/Vld/Vsts/Vmul and the
completion ops) against numpy arrays, so an ABI claim can be checked by a VALUE instead of
by a metadata flag.

It deliberately does not import the emitter: a verifier that re-runs the generator proves
nothing about the artifact.
"""
from __future__ import annotations

import numpy as np

F32 = "float32"
NP_DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "u8": np.uint8,
    "bool": np.bool_,
}

DIM_OPS = {
    "DimAdd": lambda a, b: a + b,
    "DimSub": lambda a, b: a - b,
    "DimMul": lambda a, b: a * b,
    "DimFloorDiv": lambda a, b: a // b,
    "DimCeilDiv": lambda a, b: -(-a // b),
}

#: Ops that only move predicate/data between the declared spaces.
TRANSPORT_OPS = ("GmToUb", "UbToGm", "Pset", "Vld", "Vsts", "Vmul")


class ReferenceRefusal(RuntimeError):
    """The evaluator refuses rather than approximating an unsupported node."""

    def __init__(self, code, message, subject="", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (subject=%r)" % (code, message, subject))


def _dtype_of(tensor_type):
    name = getattr(getattr(tensor_type, "dtype", None), "name", None)
    if name not in NP_DTYPES:
        raise ReferenceRefusal("REF_DTYPE_UNSUPPORTED",
                               "no numpy dtype for this native dtype", name,
                               sorted(NP_DTYPES), name)
    return NP_DTYPES[name]


def _node(node):
    return type(node).__name__


def eval_expr(expr, env):
    kind = _node(expr)
    if kind == "Constant":
        return expr.value
    if kind == "Var":
        if expr.name not in env:
            raise ReferenceRefusal("REF_UNBOUND_VAR", "variable has no bound value",
                                   expr.name, "bound", None)
        return env[expr.name]
    if kind == "Call":
        op = _node(expr.target)
        args = [eval_expr(item, env) for item in expr.args]
        handler = DIM_OPS.get(op)
        if handler is None:
            raise ReferenceRefusal("REF_EXPR_UNSUPPORTED",
                                   "expression op is outside the evaluator's supported domain",
                                   op, sorted(DIM_OPS), op)
        if len(args) != 2:
            raise ReferenceRefusal("REF_EXPR_ARITY", "binary dimension op arity mismatch",
                                   op, 2, len(args))
        return handler(int(args[0]), int(args[1]))
    raise ReferenceRefusal("REF_EXPR_UNSUPPORTED",
                           "node is outside the evaluator's supported expression domain",
                           kind, ("Constant", "Var", "Call"), kind)


def _alloc(stmt, env):
    value = stmt.value
    if _node(value) == "Call" and _node(value.target) == "AllocTensor":
        tensor_type = getattr(value.target, "tensor_type", None) or getattr(stmt.var, "type", None)
        shape = tuple(int(dim) for dim in getattr(tensor_type, "shape", ()) or ())
        env[stmt.var.name] = np.zeros(shape, dtype=_dtype_of(tensor_type))
        return True
    return False


def _op_name(callable_):
    return _node(callable_)


def exec_op(op, args, env, log):
    """Execute one Evaluate'd instruction. Unknown ops are refused by name."""
    name = _op_name(op)
    values = [env[item.name] if _node(item) == "Var" else item for item in args]
    if name == "GmToUb":
        src, dst = values
        dst[...] = src
        return
    if name == "UbToGm":
        src, dst = values
        dst[...] = src
        return
    if name == "Pset":
        (mask,) = values
        mask[...] = True
        return
    if name == "Vld":
        src, dst = values
        dst[...] = src
        return
    if name == "Vsts":
        src, dst = values
        dst[...] = src
        return
    if name == "Vmul":
        a, b, mask, dst = values
        np.copyto(dst, np.where(mask, a * b, dst))
        return
    if name in ("AivFlagSet", "AivFlagWait", "AivDrain"):
        log.append((name, tuple(
            getattr(item, "name", item) for item in args if _node(item) == "Var")))
        return
    raise ReferenceRefusal("REF_OP_UNSUPPORTED",
                           "instruction is outside the evaluator's supported domain",
                           name, tuple(TRANSPORT_OPS), name)


def exec_stmt(stmt, env, log):
    kind = _node(stmt)
    if kind == "Sequential":
        for item in stmt.body:
            exec_stmt(item, env, log)
        return
    if kind == "LetStmt":
        if _alloc(stmt, env):
            return
        env[stmt.var.name] = eval_expr(stmt.value, env)
        return
    if kind == "For":
        start = int(eval_expr(stmt.start, env))
        stop = int(eval_expr(stmt.stop, env))
        step = int(eval_expr(stmt.step, env))
        if step == 0:
            raise ReferenceRefusal("REF_LOOP_STEP", "loop step is zero", stmt.induction_var.name,
                                   "nonzero", 0)
        for index in range(start, stop, step):
            inner = dict(env)
            inner[stmt.induction_var.name] = index
            exec_stmt(stmt.body, inner, log)
        return
    if kind in ("ExecutionScope", "MeshScope"):
        exec_stmt(stmt.body, env, log)
        return
    if kind == "If":
        branch = stmt.then_body if bool(eval_expr(stmt.cond, env)) else stmt.else_body
        exec_stmt(branch, env, log)
        return
    if kind == "Evaluate":
        exec_op(stmt.callable, stmt.args, env, log)
        return
    raise ReferenceRefusal("REF_STMT_UNSUPPORTED",
                           "statement is outside the evaluator's supported domain", kind,
                           ("Sequential", "LetStmt", "For", "ExecutionScope", "MeshScope",
                            "If", "Evaluate"), kind)


def run_module(module, tensors, scalars):
    """Run one native module on real values; returns {output_name: array} and the event log."""
    from tilefoundry_ascend.codegen.abi import abi_from_module

    abi = abi_from_module(module)
    from tf_ascend_runtime.abi import validate_bindings

    tensor_specs = abi.tensor_inputs
    scalar_specs = abi.scalars
    if len(tensors) != len(tensor_specs):
        raise ReferenceRefusal("REF_INPUT_COUNT", "wrong number of input tensors", abi.entry,
                               len(tensor_specs), len(tensors))
    if len(scalars) != len(scalar_specs):
        raise ReferenceRefusal("REF_SCALAR_COUNT", "wrong number of scalars", abi.entry,
                               len(scalar_specs), len(scalars))
    env = {}
    arguments = []
    for spec, value in zip(tensor_specs, tensors):
        array = np.ascontiguousarray(np.asarray(value, dtype=abi.parameter(spec.name).runtime_dtype))
        if array.shape != tuple(spec.shape):
            raise ReferenceRefusal("REF_INPUT_SHAPE", "input shape does not match the ABI",
                                   spec.name, tuple(spec.shape), array.shape)
        env[spec.name] = array
        arguments.append(array)
    for spec, value in zip(scalar_specs, scalars):
        env[spec.name] = int(value)
        arguments.append(int(value))
    outputs = {}
    for spec in abi.outputs:
        outputs[spec.name] = np.zeros(tuple(spec.shape), dtype=spec.runtime_dtype)
        env[spec.name] = outputs[spec.name]
    entry = module.entry_function()
    log = []
    exec_stmt(entry.body, env, log)
    return {name: np.array(value) for name, value in outputs.items()}, log
