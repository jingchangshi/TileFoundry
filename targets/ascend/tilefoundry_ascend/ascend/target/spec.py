"""Typed Ascend hardware schema builders."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.facts import TARGET_MEMORY_OWNER
from tilefoundry.target.hardware.envelope import HardwareDocument, SchemaValidationError
from tilefoundry.target.hardware.schema import SchemaReader
from tilefoundry_ascend.ascend.target.architecture import DaVinci950
from tilefoundry_ascend.ascend.target.device import Ascend950PR

ARCHITECTURE_SCHEMA = "tilefoundry.ascend.architecture/v2"
DEVICE_SCHEMA = "tilefoundry.ascend.device/v2"

DAVINCI_950_ID = "ascend.davinci_950"
ASCEND_950PR_ID = "ascend.950pr"

_ALLOWED_OWNERS = (TARGET_MEMORY_OWNER, "block", "aic", "aiv", "lane")


def _memory_owner(reader: SchemaReader, path: str) -> str:
    owner = reader.text(path)
    if owner not in _ALLOWED_OWNERS:
        raise SchemaValidationError(
            f"{reader.document.id}: memory owner {owner!r} at {path!r} must be "
            f"one of {list(_ALLOWED_OWNERS)}"
        )
    return owner


def _dtypes(names: tuple[str, ...], document: HardwareDocument) -> tuple[DType, ...]:
    resolved = []
    for name in names:
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise SchemaValidationError(f"{document.id}: unknown compute dtype {name!r}")
        resolved.append(dtype)
    return tuple(resolved)


def build_davinci_950(document: HardwareDocument) -> DaVinci950:
    """Build the immutable DaVinci 950 architecture value from its document."""
    reader = SchemaReader(document)
    architecture = DaVinci950(
        name=reader.text("identity.name"),
        supported_compute_dtypes=_dtypes(reader.names("instruction.compute_dtypes"), document),
        capabilities=reader.names("capabilities"),
        aic_per_block=reader.integer("compute.aic_per_block", unit="count"),
        aiv_per_block=reader.integer("compute.aiv_per_block", unit="count"),
        vector_lanes_f32=reader.integer("compute.vector_lanes_f32", unit="count"),
        max_threads_per_cta=reader.integer("compute.max_simt_threads_per_aiv", unit="count"),
        ub_align_bytes=reader.integer("memory.ub.align_bytes", unit="byte"),
        l1_align_bytes=reader.integer("memory.l1.align_bytes", unit="byte"),
        l0a_align_bytes=reader.integer("memory.l0a.align_bytes", unit="byte"),
        l0b_align_bytes=reader.integer("memory.l0b.align_bytes", unit="byte"),
        l0c_align_bytes=reader.integer("memory.l0c.align_bytes", unit="byte"),
        tmem_owner=_memory_owner(reader, "memory.tmem.owner"),
        smem_owner=_memory_owner(reader, "memory.smem.owner"),
        rmem_owner=_memory_owner(reader, "memory.rmem.owner"),
        cube_mmad_atom=reader.text("geometry.cube_mmad_atom"),
    )
    reader.close()
    return architecture


def build_ascend_950pr(document: HardwareDocument) -> Ascend950PR:
    """Build the immutable Ascend 950PR device value from its document."""
    reader = SchemaReader(document)
    unit_flops = (
        (
            "aic",
            (
                (DType.f16, reader.integer("throughput.aic.f16", unit="flop/s")),
                (DType.bf16, reader.integer("throughput.aic.bf16", unit="flop/s")),
            ),
        ),
        (
            "aiv",
            ((DType.f32, reader.integer("throughput.aiv.f32", unit="flop/s")),),
        ),
    )
    device = Ascend950PR(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.block_count", unit="count"),
        aic_count=reader.integer("compute.aic_count", unit="count"),
        aiv_count=reader.integer("compute.aiv_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        hbm_owner=_memory_owner(reader, "memory.hbm.owner"),
        hbm_bandwidth_bytes_per_second=reader.integer("memory.hbm.bandwidth", unit="byte/s"),
        l1_bytes_per_aic=reader.integer("memory.l1.per_aic", unit="byte"),
        l0a_bytes_per_aic=reader.integer("memory.l0a.per_aic", unit="byte"),
        l0b_bytes_per_aic=reader.integer("memory.l0b.per_aic", unit="byte"),
        l0c_bytes_per_aic=reader.integer("memory.l0c.per_aic", unit="byte"),
        ub_nominal_bytes_per_aiv=reader.integer("memory.ub.nominal_per_aiv", unit="byte"),
        ub_simdvf_usable_bytes_per_aiv=reader.integer(
            "memory.ub.simdvf_usable_per_aiv", unit="byte"
        ),
        ub_usable_bytes_per_aiv=reader.integer("memory.ub.usable_per_aiv", unit="byte"),
        rmem_bytes_per_lane=reader.integer("memory.rmem.per_lane", unit="byte"),
        _unit_flops=unit_flops,
        aiv_lane_events_per_second=reader.integer("performance.aiv.lane_events", unit="op/s"),
    )
    reader.close()
    if device.ub_simdvf_usable_bytes_per_aiv > device.ub_nominal_bytes_per_aiv:
        raise SchemaValidationError(
            f"{document.id}: SimdVF usable UB ({device.ub_simdvf_usable_bytes_per_aiv}) cannot "
            f"exceed nominal UB ({device.ub_nominal_bytes_per_aiv})"
        )
    if device.ub_usable_bytes_per_aiv > device.ub_simdvf_usable_bytes_per_aiv:
        raise SchemaValidationError(
            f"{document.id}: SimtVF usable UB ({device.ub_usable_bytes_per_aiv}) cannot exceed "
            f"SimdVF usable UB ({device.ub_simdvf_usable_bytes_per_aiv})"
        )
    return device


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "ASCEND_950PR_ID",
    "DAVINCI_950_ID",
    "DEVICE_SCHEMA",
    "build_ascend_950pr",
    "build_davinci_950",
]
