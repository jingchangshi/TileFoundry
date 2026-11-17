"""Typed Ascend hardware schema builders."""

from __future__ import annotations

from dataclasses import dataclass

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
        ssbuf_align_bytes=reader.integer("memory.ssbuf.align_bytes", unit="byte"),
        ssbuf_owner=_memory_owner(reader, "memory.ssbuf.owner"),
        valid_dma_paths=reader.names("datapath.valid_dma_paths"),
        forbidden_dma_paths=reader.names("datapath.forbidden_dma_paths"),
        max_mte2_inflight_per_aiv=reader.integer("microarch.max_mte2_inflight_per_aiv", unit="count"),
        fixpipe_bytes_per_cycle=reader.integer(
            "microarch.fixpipe_bytes_per_cycle", unit="byte/cycle"
        ),
    )
    reader.close()
    overlap = set(architecture.valid_dma_paths) & set(architecture.forbidden_dma_paths)
    if overlap:
        raise SchemaValidationError(
            f"{document.id}: DMA paths cannot be both valid and forbidden: {sorted(overlap)}"
        )
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
    burst_bw_table = (
        (4096, reader.integer("memory.hbm.burst_bandwidth.b4096", unit="byte/s")),
        (8192, reader.integer("memory.hbm.burst_bandwidth.b8192", unit="byte/s")),
        (16384, reader.integer("memory.hbm.burst_bandwidth.b16384", unit="byte/s")),
        (32768, reader.integer("memory.hbm.burst_bandwidth.b32768", unit="byte/s")),
        (65536, reader.integer("memory.hbm.burst_bandwidth.b65536", unit="byte/s")),
    )
    device = Ascend950PR(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.block_count", unit="count"),
        aic_count=reader.integer("compute.aic_count", unit="count"),
        aiv_count=reader.integer("compute.aiv_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        hbm_owner=_memory_owner(reader, "memory.hbm.owner"),
        hbm_bandwidth_bytes_per_second=reader.integer("memory.hbm.bandwidth", unit="byte/s"),
        hbm_burst_bandwidth_table=burst_bw_table,
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
        ssbuf_bytes_per_block=reader.integer("memory.ssbuf.per_block", unit="byte"),
        ssbuf_latency_cycles=reader.integer("memory.ssbuf.latency_cycles", unit="cycle"),
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


CORE_BIT_SHIFT: dict[str, int] = {"aiv0": 0, "aic": 8, "aiv1": 16}

PIPE_BIT_INDEX: dict[str, int] = {
    "MTE1": 0,
    "MTE2": 1,
    "MTE3": 2,
    "CUBE": 3,
    "VECTOR": 4,
    "FIXPIPE": 5,
    "SCALAR": 6,
}

_PIPE_ALIASES: dict[str, str] = {
    "M": "CUBE",
    "V": "VECTOR",
    "FIX": "FIXPIPE",
    "S": "SCALAR",
}

_ALLOWED_PIPES_BY_CORE: dict[str, frozenset[str]] = {
    "aiv0": frozenset({"MTE1", "MTE2", "MTE3", "VECTOR", "SCALAR"}),
    "aic": frozenset({"MTE1", "MTE2", "MTE3", "CUBE", "FIXPIPE", "SCALAR"}),
    "aiv1": frozenset({"MTE1", "MTE2", "MTE3", "VECTOR", "SCALAR"}),
}

_AIV_INVALID_BYTE = (1 << 3) | (1 << 5) | (1 << 7)
_AIC_INVALID_BYTE = (1 << 4) | (1 << 7)
_INVALID_BITS_MASK = _AIV_INVALID_BYTE | (_AIC_INVALID_BYTE << 8) | (_AIV_INVALID_BYTE << 16)


def _normalize_pipe(pipe: str) -> str:
    key = pipe.strip().upper()
    if key.startswith("PIPE_"):
        key = key[5:]
    key = _PIPE_ALIASES.get(key, key)
    if key not in PIPE_BIT_INDEX:
        raise ValueError(f"unknown Ascend resource pipe {pipe!r}")
    return key


@dataclass(frozen=True)
class NodeResourceMask:
    """24-bit coupled core resource mask encoding AIV0, AIC, and AIV1 pipes."""

    bits: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or isinstance(self.bits, bool):
            raise TypeError(f"mask bits must be int, got {type(self.bits).__name__}")
        if self.bits < 0 or self.bits > 0xFFFFFF:
            raise ValueError(f"24-bit mask out of range: {hex(self.bits)}")
        if self.bits & _INVALID_BITS_MASK:
            raise ValueError(
                f"invalid pipe bits in 24-bit mask {hex(self.bits)}; check core capabilities"
            )

    @property
    def aiv0_bits(self) -> int:
        """Return 8-bit pipe mask for AIV0."""
        return self.bits & 0xFF

    @property
    def aic_bits(self) -> int:
        """Return 8-bit pipe mask for AIC."""
        return (self.bits >> 8) & 0xFF

    @property
    def aiv1_bits(self) -> int:
        """Return 8-bit pipe mask for AIV1."""
        return (self.bits >> 16) & 0xFF

    def __or__(self, other: NodeResourceMask) -> NodeResourceMask:
        return NodeResourceMask(self.bits | other.bits)

    def __and__(self, other: NodeResourceMask) -> NodeResourceMask:
        return NodeResourceMask(self.bits & other.bits)

    def __bool__(self) -> bool:
        return self.bits != 0

    def __int__(self) -> int:
        return self.bits

    def conflicts_with(self, other: NodeResourceMask) -> bool:
        """Return whether two resource masks share at least one hardware pipe."""
        return (self.bits & other.bits) != 0

    @classmethod
    def for_core(cls, core: str, *pipes: str) -> NodeResourceMask:
        """Construct a mask binding one core slot to one or more named pipes."""
        core_key = core.strip().lower()
        if core_key not in CORE_BIT_SHIFT:
            raise ValueError(f"unknown coupled core {core!r}")
        shift = CORE_BIT_SHIFT[core_key]
        allowed = _ALLOWED_PIPES_BY_CORE[core_key]
        bits = 0
        for raw in pipes:
            pipe_name = _normalize_pipe(raw)
            if pipe_name not in allowed:
                raise ValueError(f"core {core_key!r} does not support pipe {pipe_name!r}")
            bits |= 1 << (shift + PIPE_BIT_INDEX[pipe_name])
        return cls(bits)

    @classmethod
    def for_aiv0(cls, *pipes: str) -> NodeResourceMask:
        """Construct an AIV0 resource mask for the given pipe names."""
        return cls.for_core("aiv0", *pipes)

    @classmethod
    def for_aic(cls, *pipes: str) -> NodeResourceMask:
        """Construct an AIC resource mask for the given pipe names."""
        return cls.for_core("aic", *pipes)

    @classmethod
    def for_aiv1(cls, *pipes: str) -> NodeResourceMask:
        """Construct an AIV1 resource mask for the given pipe names."""
        return cls.for_core("aiv1", *pipes)

    @classmethod
    def for_dual_aiv(cls, *pipes: str) -> NodeResourceMask:
        """Construct a symmetric dual-AIV mask setting bits on both AIV0 and AIV1."""
        return cls.for_core("aiv0", *pipes) | cls.for_core("aiv1", *pipes)

    def to_core_pipe_pairs(self) -> tuple[tuple[str, str], ...]:
        """Expand active bits into ordered (core, pipe_name) tuples."""
        pairs: list[tuple[str, str]] = []
        for core_key in ("aiv0", "aic", "aiv1"):
            shift = CORE_BIT_SHIFT[core_key]
            for pipe_name, bit_idx in PIPE_BIT_INDEX.items():
                if self.bits & (1 << (shift + bit_idx)):
                    pairs.append((core_key, pipe_name))
        return tuple(pairs)

    def to_solver_resources(
        self, duration: int = 1, offset: int = 0
    ) -> tuple[tuple[str, str, str, int, int], ...]:
        """Project active bits into pipeline solver resource tuples."""
        return tuple(
            ("aic" if core == "aic" else "aiv", core, pipe_name, duration, offset)
            for core, pipe_name in self.to_core_pipe_pairs()
        )


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "ASCEND_950PR_ID",
    "CORE_BIT_SHIFT",
    "DAVINCI_950_ID",
    "DEVICE_SCHEMA",
    "PIPE_BIT_INDEX",
    "NodeResourceMask",
    "build_ascend_950pr",
    "build_davinci_950",
]
