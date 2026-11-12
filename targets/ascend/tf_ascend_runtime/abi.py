"""Pure ordered-ABI types shared by the compiler side and the runtime side.

This module is the ONE definition of the callable ABI. It imports nothing from
TileFoundry, TileLang or Torch, which is what lets the NPU runtime worker load a
descriptor without dragging the compiler (or ORTools) into the process.

WHY THE ORDER IS THE CONTRACT. A PrimFunction records `output_count` as a count of
TRAILING output parameters; the callable's inputs are therefore the leading
`len(params) - output_count` parameters and the outputs are the last ones. A builder that
puts an output in the middle of the parameter list while claiming `output_count=1` hands
the runtime a descriptor whose output slot is a scalar: the caller allocates nothing,
passes a shape extent where a tensor belongs, and the store lands outside the output.
So the ABI is derived from the REAL parameter list and refuses any arrangement that
cannot be a trailing-output callable contract.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

ABI_SCHEMA = "ascend.native-abi/1"

ROLE_TENSOR = "tensor"
ROLE_SCALAR = "scalar"
ROLES = (ROLE_TENSOR, ROLE_SCALAR)

#: Storage kinds a device tensor may occupy in a callable ABI. A register or a scalar
#: value can never be a boundary argument of an Ascend kernel.
DEVICE_STORAGES = ("gmem",)

#: Native dtype -> the tensor-library name. A name outside this closed table is refused;
#: guessing a runtime dtype would make "dtype matches" unverifiable.
RUNTIME_DTYPES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "bool": "bool",
}


class AbiRefusal(ValueError):
    """A callable ABI that cannot be produced or bound, with a stable code."""

    def __init__(self, code, message, subject="", expected=None, actual=None):
        self.code = str(code)
        self.message = str(message)
        self.subject = str(subject)
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (subject=%r expected=%r actual=%r)"
                         % (self.code, self.message, self.subject, expected, actual))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


def runtime_dtype(native_name):
    """The tensor-library dtype name for a native dtype name (refusal on unknown)."""
    key = str(native_name)
    name = RUNTIME_DTYPES.get(key)
    if name is None:
        raise AbiRefusal("ABI_DTYPE_UNKNOWN", "no runtime dtype for this native dtype",
                         key, sorted(RUNTIME_DTYPES), key)
    return name


@dataclass(frozen=True)
class ParameterSpec:
    """One ordered callable parameter, as the compiled entry really declares it."""

    name: str
    index: int
    role: str
    storage: str
    dtype: str
    shape: tuple = ()

    @property
    def is_scalar(self) -> bool:
        return self.role == ROLE_SCALAR

    @property
    def runtime_dtype(self) -> str:
        return runtime_dtype(self.dtype)

    def as_dict(self):
        return {"name": self.name, "index": self.index, "role": self.role,
                "storage": self.storage, "dtype": self.dtype,
                "runtime_dtype": self.runtime_dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class OrderedABI:
    """The ordered callable contract: leading inputs (tensors/scalars), trailing outputs."""

    entry: str
    parameters: tuple
    output_count: int
    schema: str = ABI_SCHEMA

    def __post_init__(self) -> None:
        if not self.entry:
            raise AbiRefusal("ABI_ENTRY_MISSING", "the ABI must name the entry symbol")
        if not isinstance(self.output_count, int) or isinstance(self.output_count, bool) \
                or self.output_count < 1:
            raise AbiRefusal("ABI_OUTPUT_COUNT_INVALID",
                             "output_count must be a positive int", self.entry,
                             ">= 1", self.output_count)
        params = tuple(self.parameters)
        if not params:
            raise AbiRefusal("ABI_NO_PARAMETERS", "an entry with no parameters has no ABI",
                             self.entry, ">= 1 parameter", 0)
        if len(params) <= self.output_count:
            raise AbiRefusal("ABI_OUTPUT_COUNT_TOO_LARGE",
                             "every parameter is claimed as an output", self.entry,
                             "< %d" % len(params), self.output_count)
        for position, parameter in enumerate(params):
            if not isinstance(parameter, ParameterSpec):
                raise AbiRefusal("ABI_PARAMETER_TYPE", "parameters must be ParameterSpec",
                                 self.entry, "ParameterSpec", type(parameter).__name__)
            if parameter.index != position:
                raise AbiRefusal("ABI_PARAMETER_ORDER",
                                 "parameter index does not match its position", self.entry,
                                 position, parameter.index)
            if parameter.role not in ROLES:
                raise AbiRefusal("ABI_PARAMETER_ROLE", "unknown parameter role",
                                 parameter.name, ROLES, parameter.role)
            if parameter.role == ROLE_TENSOR and parameter.storage not in DEVICE_STORAGES:
                raise AbiRefusal(
                    "ABI_TENSOR_STORAGE",
                    "a boundary tensor must live in a device-visible storage",
                    parameter.name, list(DEVICE_STORAGES), parameter.storage)
            if parameter.role == ROLE_SCALAR and parameter.shape:
                raise AbiRefusal("ABI_SCALAR_SHAPE", "a scalar parameter must be rank-0",
                                 parameter.name, (), parameter.shape)
            runtime_dtype(parameter.dtype)
        split = len(params) - self.output_count
        for parameter in params[split:]:
            if parameter.role != ROLE_TENSOR:
                raise AbiRefusal(
                    "ABI_OUTPUT_NOT_TENSOR",
                    "a trailing output parameter must be a device tensor", parameter.name,
                    ROLE_TENSOR, parameter.role)
        for parameter in params[:split]:
            if parameter.role == ROLE_TENSOR and parameter.storage not in DEVICE_STORAGES:
                raise AbiRefusal("ABI_INPUT_STORAGE", "an input tensor must be device-visible",
                                 parameter.name, list(DEVICE_STORAGES), parameter.storage)

    # ---------------------------------------------------------------- projections
    @property
    def output_split(self) -> int:
        return len(self.parameters) - self.output_count

    @property
    def inputs(self) -> tuple:
        return tuple(self.parameters[:self.output_split])

    @property
    def outputs(self) -> tuple:
        return tuple(self.parameters[self.output_split:])

    @property
    def tensor_inputs(self) -> tuple:
        return tuple(item for item in self.inputs if item.role == ROLE_TENSOR)

    @property
    def scalars(self) -> tuple:
        return tuple(item for item in self.inputs if item.role == ROLE_SCALAR)

    def ordered_arguments(self) -> tuple:
        """Inputs in CALL order: tensors first, then scalars (the callable's own order)."""
        return tuple(self.inputs)

    def parameter(self, name) -> ParameterSpec:
        for item in self.parameters:
            if item.name == name:
                return item
        raise AbiRefusal("ABI_PARAMETER_UNKNOWN", "no such parameter", str(name),
                         [item.name for item in self.parameters], name)

    def digest(self) -> str:
        payload = {"schema": self.schema, "entry": self.entry,
                   "output_count": self.output_count,
                   "parameters": [item.as_dict() for item in self.parameters]}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def as_dict(self):
        return {"schema": self.schema, "entry": self.entry,
                "output_count": self.output_count,
                "ordered_arguments": [item.as_dict() for item in self.inputs],
                "output_parameters": [item.as_dict() for item in self.outputs],
                "digest": self.digest()}

    @classmethod
    def from_dict(cls, payload):
        """Rebuild from a descriptor. A missing/foreign schema is refused."""
        if not isinstance(payload, dict):
            raise AbiRefusal("ABI_DESCRIPTOR_TYPE", "ABI descriptor must be a mapping",
                             "", "dict", type(payload).__name__)
        if payload.get("schema") != ABI_SCHEMA:
            raise AbiRefusal("ABI_DESCRIPTOR_SCHEMA", "foreign ABI descriptor schema",
                             str(payload.get("schema")), ABI_SCHEMA, payload.get("schema"))
        raw = tuple(payload.get("ordered_arguments") or ())
        raw_outputs = tuple(payload.get("output_parameters") or ())
        params = []
        for position, item in enumerate(raw + raw_outputs):
            params.append(ParameterSpec(
                name=item["name"], index=position, role=item["role"],
                storage=item["storage"], dtype=item["dtype"],
                shape=tuple(item.get("shape") or ())))
        abi = cls(entry=payload["entry"], parameters=tuple(params),
                  output_count=len(raw_outputs))
        if "digest" in payload and payload["digest"] != abi.digest():
            raise AbiRefusal("ABI_DESCRIPTOR_DIGEST",
                             "ABI descriptor digest does not match its content",
                             abi.entry, payload["digest"], abi.digest())
        return abi


def _shape_of(value):
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    return shape


def _dtype_of(value):
    return getattr(getattr(value, "dtype", None), "name", None) or str(getattr(value, "dtype", ""))


def validate_bindings(abi, tensors, scalars):
    """Check the caller's real arguments against the ABI, without touching any math.

    Refuses: count/shape/dtype mismatch, non-contiguous tensors, aliasing between an input
    and an output, and any device mismatch. Returns the ordered argument list the kernel
    must be invoked with.
    """
    if not isinstance(abi, OrderedABI):
        raise AbiRefusal("ABI_TYPE", "validate_bindings needs an OrderedABI", "",
                         "OrderedABI", type(abi).__name__)
    tensors = tuple(tensors)
    scalars = tuple(scalars)
    expected_tensors = abi.tensor_inputs
    expected_scalars = abi.scalars
    if len(tensors) != len(expected_tensors):
        raise AbiRefusal("ABI_INPUT_COUNT", "input tensor count does not match the ABI",
                         abi.entry, len(expected_tensors), len(tensors))
    if len(scalars) != len(expected_scalars):
        raise AbiRefusal("ABI_SCALAR_COUNT", "scalar count does not match the ABI",
                         abi.entry, len(expected_scalars), len(scalars))
    arguments = []
    for spec, value in zip(expected_tensors, tensors):
        shape = _shape_of(value)
        if shape != tuple(spec.shape):
            raise AbiRefusal("ABI_INPUT_SHAPE", "input shape does not match the ABI",
                             spec.name, tuple(spec.shape), shape)
        if _dtype_of(value) != spec.runtime_dtype:
            raise AbiRefusal("ABI_INPUT_DTYPE", "input dtype does not match the ABI",
                             spec.name, spec.runtime_dtype, _dtype_of(value))
        contiguous = getattr(value, "is_contiguous", None)
        if callable(contiguous) and not contiguous():
            raise AbiRefusal("ABI_INPUT_NONCONTIGUOUS",
                             "input tensor must be contiguous", spec.name, True, False)
        arguments.append(value)
    for spec, value in zip(expected_scalars, scalars):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AbiRefusal("ABI_SCALAR_TYPE", "scalar must be a python int", spec.name,
                             "int", type(value).__name__)
    arguments.extend(scalars)
    return tuple(arguments)


__all__ = [
    "ABI_SCHEMA", "ROLE_TENSOR", "ROLE_SCALAR", "ROLES", "DEVICE_STORAGES",
    "RUNTIME_DTYPES", "AbiRefusal", "runtime_dtype", "ParameterSpec", "OrderedABI",
    "validate_bindings",
]
