"""Derive the ordered callable ABI from a REAL native PrimFunction.

The compiler side owns this adapter so the runtime package stays pure: a PrimFunction's
parameter list becomes an OrderedABI whose outputs are its TRAILING output_count params.
Nothing here guesses an ABI from metadata strings -- the parameters, their storage, dtype
and shape come from the native types themselves.
"""
from __future__ import annotations

from tf_ascend_runtime.abi import (
    DEVICE_STORAGES,
    ROLE_SCALAR,
    ROLE_TENSOR,
    AbiRefusal,
    OrderedABI,
    ParameterSpec,
)

__all__ = ["abi_from_prim_function", "abi_from_module", "entry_function"]


def entry_function(module):
    """The module's entry PrimFunction, or a refusal naming the mismatch."""
    entry = getattr(module, "entry", None)
    if not entry:
        raise AbiRefusal("ABI_ENTRY_MISSING", "module has no entry name")
    for function in tuple(getattr(module, "functions", ()) or ()):
        if getattr(function, "name", None) == entry:
            return function
    raise AbiRefusal("ABI_ENTRY_UNKNOWN", "module entry is not one of its functions",
                     entry, [getattr(f, "name", None) for f in getattr(module, "functions", ())],
                     entry)


def abi_from_prim_function(function, *, entry=None):
    """The ordered ABI of one PrimFunction (trailing output_count params are outputs)."""
    name = entry or getattr(function, "name", "")
    if not name:
        raise AbiRefusal("ABI_ENTRY_MISSING", "PrimFunction has no name")
    output_count = getattr(function, "output_count", None)
    if output_count is None:
        raise AbiRefusal("ABI_OUTPUT_COUNT_MISSING", "PrimFunction declares no output_count",
                         name, "an int", None)
    specs = []
    for index, parameter in enumerate(tuple(getattr(function, "params", ()) or ())):
        tensor_type = getattr(parameter, "type", None)
        if tensor_type is None or not hasattr(tensor_type, "storage"):
            raise AbiRefusal("ABI_PARAMETER_TYPE",
                             "parameter does not carry a tensor type", name, "TensorType",
                             type(tensor_type).__name__)
        shape = tuple(int(dim) for dim in getattr(tensor_type, "shape", ()) or ())
        dtype = getattr(getattr(tensor_type, "dtype", None), "name", None)
        if dtype is None:
            raise AbiRefusal("ABI_PARAMETER_DTYPE", "parameter type carries no dtype",
                             getattr(parameter, "name", str(index)), "DType", None)
        storage = getattr(tensor_type, "storage", None)
        storage_name = getattr(storage, "name", str(storage)).lower()
        if storage_name in DEVICE_STORAGES:
            role = ROLE_TENSOR
        elif not shape:
            role = ROLE_SCALAR
            storage_name = ROLE_SCALAR
        else:
            raise AbiRefusal(
                "ABI_PARAMETER_STORAGE",
                "a non-scalar boundary parameter must live in device storage; a rank>=1"
                " parameter in a local/register space cannot cross the callable boundary",
                getattr(parameter, "name", str(index)), list(DEVICE_STORAGES), storage_name)
        specs.append(ParameterSpec(
            name=getattr(parameter, "name", str(index)), index=index, role=role,
            storage=storage_name, dtype=dtype, shape=shape))
    return OrderedABI(entry=name, parameters=tuple(specs), output_count=output_count)


def abi_from_module(module):
    """The ordered ABI of a module's real entry function."""
    return abi_from_prim_function(entry_function(module), entry=getattr(module, "entry", None))
