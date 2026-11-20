"""Ascend execution mode declarations.

``ExecutionKind.__post_init__`` enforces a 32-bit mask width for element dtypes,
while ``SUPPORTED_VECTOR_DTYPES`` also exposes non-32-bit vector register
transport and compute types.
"""

from __future__ import annotations

from tilefoundry.ir.core.execution import ExecutionKind, register_execution_kind
from tilefoundry.ir.types import DType, StorageKind

_PREFERRED_VECTOR_DTYPE_NAMES: tuple[str, ...] = (
    "f32",
    "f16",
    "bf16",
    "i32",
    "i64",
    "fp8e4m3",
    "fp8e5m2",
    "hif8",
    "f8e8m0",
    "f4e2m1",
    "f4e1m2",
    "i4",
    "u4",
    "i8",
    "u8",
    "i16",
    "u16",
    "u32",
    "u64",
    "bool",
)


def _supported_vector_dtypes(*, mask_width_bits: int | None = None) -> tuple[DType, ...]:
    """Collect supported AIV vector dtypes via getattr to avoid import-order fragility."""
    dtypes: list[DType] = []
    for name in _PREFERRED_VECTOR_DTYPE_NAMES:
        dt = getattr(DType, name, None)
        if isinstance(dt, DType) and dt not in dtypes:
            if mask_width_bits is None or dt.bit_width == mask_width_bits:
                dtypes.append(dt)
    for dt in DType._members().values():
        if isinstance(dt, DType) and dt not in dtypes:
            if mask_width_bits is None or dt.bit_width == mask_width_bits:
                dtypes.append(dt)
    return tuple(dtypes)


SUPPORTED_VECTOR_DTYPES: tuple[DType, ...] = _supported_vector_dtypes(mask_width_bits=None)
_MASK_COMPATIBLE_ELEMENT_DTYPES: tuple[DType, ...] = _supported_vector_dtypes(mask_width_bits=32)

SIMDVF_KIND = ExecutionKind(
    name="ascend.simdvf",
    surface="SimdVF",
    register_storages=(StorageKind.RMEM, StorageKind.CARRIER),
    mask_width_bits=32,
    conversion_mask_width_bits=16,
    lane_count=None,
    element_dtypes=_MASK_COMPATIBLE_ELEMENT_DTYPES,
    required_topologies=("aiv",),
)

SIMTVF_KIND = ExecutionKind(
    name="ascend.simtvf",
    surface="SimtVF",
    register_storages=(StorageKind.RMEM, StorageKind.CARRIER),
    mask_width_bits=32,
    conversion_mask_width_bits=16,
    lane_count=None,
    element_dtypes=_MASK_COMPATIBLE_ELEMENT_DTYPES,
    required_topologies=("aiv",),
)

SimdVF = register_execution_kind(SIMDVF_KIND, dialect="T")
SimtVF = register_execution_kind(SIMTVF_KIND, dialect="T")

__all__ = [
    "SIMDVF_KIND",
    "SIMTVF_KIND",
    "SUPPORTED_VECTOR_DTYPES",
    "SimdVF",
    "SimtVF",
]
