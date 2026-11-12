"""Pure Ascend runtime package: artifact loading, ABI, bind/invoke and TL reading.

NOTHING HERE IMPORTS THE TILEFOUNDRY COMPILER. The NPU runtime worker must be able to load
an already-decided artifact without pulling in TileFoundry, ORTools or the analysis layer;
torch/TileLang are imported lazily by the actions that really need them.
"""
from .abi import (
    ABI_SCHEMA,
    DEVICE_STORAGES,
    ROLE_SCALAR,
    ROLE_TENSOR,
    ROLES,
    RUNTIME_DTYPES,
    AbiRefusal,
    OrderedABI,
    ParameterSpec,
    runtime_dtype,
    validate_bindings,
)

__all__ = [
    "ABI_SCHEMA", "DEVICE_STORAGES", "ROLE_SCALAR", "ROLE_TENSOR", "ROLES",
    "RUNTIME_DTYPES", "AbiRefusal", "OrderedABI", "ParameterSpec", "runtime_dtype",
    "validate_bindings",
]
