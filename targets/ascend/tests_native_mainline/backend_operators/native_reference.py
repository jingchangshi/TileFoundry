"""Thin test shell over the ONE framework CPU executor.

This file used to hold a second TIR interpreter. It no longer does: execution is
tilefoundry_ascend.verify.executor.run_entry. What stays here is fixture
assembly from the ordered ABI (tensor inputs, scalar extents) and the small
return shape the existing value tests expect. No statement or instruction is
interpreted here.
"""
from __future__ import annotations

import numpy as np

from tilefoundry_ascend.codegen.abi import abi_from_module
from tilefoundry_ascend.verify.executor import ExecutionRefusal, run_entry

__all__ = ["ReferenceRefusal", "run_module"]


class ReferenceRefusal(ExecutionRefusal):
    """Backwards-compatible name for a refused fixture execution."""


def run_module(module, tensors, scalars):
    """Run one native module on real values; returns (outputs by name, no event log)."""
    abi = abi_from_module(module)
    if len(tensors) != len(abi.tensor_inputs):
        raise ReferenceRefusal(
            "REF_INPUT_COUNT", "wrong number of input tensors", abi.entry)
    if len(scalars) != len(abi.scalars):
        raise ReferenceRefusal(
            "REF_SCALAR_COUNT", "wrong number of scalars", abi.entry)
    inputs = {}
    for spec, value in zip(abi.tensor_inputs, tensors):
        inputs[spec.name] = np.asarray(value, dtype=spec.runtime_dtype)
    for spec, value in zip(abi.scalars, scalars):
        inputs[spec.name] = int(value)
    outputs = run_entry(module, inputs)
    return outputs, []
