"""Ascend compilation target composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry.target.base import (
    Architecture,
    Device,
    HardwareSpec,
    Target,
    architecture_of,
    available_device_ids,
    check_compatible,
    register_target,
    select,
)
from tilefoundry.target.facts import (
    MemoryExecutionFacts,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.utils.python_source import PythonExpr
from tilefoundry_ascend.ascend.target.architecture import DaVinci950
from tilefoundry_ascend.ascend.target.device import Ascend950PR
from tilefoundry_ascend.ascend.target.spec import (
    ARCHITECTURE_SCHEMA,
    ASCEND_950PR_ID,
    DEVICE_SCHEMA,
    build_ascend_950pr,
    build_davinci_950,
)


@register_target
@dataclass(frozen=True, init=False)
class AscendTarget(Target):
    """Ascend target composed from one DaVinci architecture and one device."""

    name: ClassVar[str] = "ascend"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry_ascend.ascend.target.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_davinci_950,
            DEVICE_SCHEMA: build_ascend_950pr,
        },
    )
    architecture: Architecture = field(init=False)
    device: Device = field(init=False)
    architecture_id: str | None = field(default=None, init=False, compare=False)
    device_id: str | None = field(default=None, init=False, compare=False)
    architecture_digest: str | None = field(default=None, init=False, compare=False)
    device_digest: str | None = field(default=None, init=False, compare=False)
    _architecture_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _device_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @property
    def identity(self) -> str:
        return self.device_id or self.name

    @classmethod
    def available(cls) -> tuple[AscendTarget, ...]:
        return tuple(cls(device_id) for device_id in available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path | None = None,
        architecture: Architecture | str | Path | None = None,
    ) -> None:
        device = ASCEND_950PR_ID if device is None else device
        if architecture is None:
            architecture = architecture_of(
                device,
                device_type=Ascend950PR,
                role="AscendTarget.device",
                hardware=self.hardware,
            )
        architecture = select(
            architecture,
            DaVinci950,
            role="AscendTarget.architecture",
            hardware=self.hardware,
        )
        device = select(
            device,
            Ascend950PR,
            role="AscendTarget.device",
            hardware=self.hardware,
        )
        if architecture.id is not None and device.id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "_architecture_document", architecture.document)
        object.__setattr__(self, "_device_document", device.document)

    def _topology_facts(self) -> TopologyFacts:
        return TopologyFacts(
            (
                TopologyLevelFacts(
                    "block",
                    self.device.sm_count,
                    self.device.sm_count,
                ),
                TopologyLevelFacts(
                    "aic",
                    self.architecture.topology_limit("aic"),
                    self.device.aic_count,
                ),
                TopologyLevelFacts(
                    "aiv",
                    self.architecture.topology_limit("aiv"),
                    self.device.aiv_count,
                ),
                TopologyLevelFacts(
                    "lane",
                    self.architecture.topology_limit("lane"),
                    self.device.aiv_count * self.architecture.vector_lanes_f32,
                ),
            ),
            parallel_level="block",
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project Ascend hardware facts through the requested aggregate type."""
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, self._topology_facts())
        if facts_type is MemoryExecutionFacts and query is None:
            return facts_result(
                self,
                facts_type,
                MemoryExecutionFacts(
                    tuple(level.name for level in self._topology_facts().topologies)
                ),
            )
        if facts_type is TopologyLevelFacts:
            level = self._topology_facts().level(
                query if isinstance(query, str) else None
            )
            if level is not None:
                return facts_result(self, facts_type, level)
            return super().get_facts(facts_type, query)

        from tilefoundry.analysis.facts import (  # noqa: PLC0415
            MemoryHierarchyFacts,
            PerformanceServiceFacts,
            ThroughputFacts,
        )
        from tilefoundry_ascend.ascend.target.facts import (  # noqa: PLC0415
            AscendSpaceFacts,
            ascend_space_facts,
            memory_hierarchy,
            performance_service,
            throughput,
        )

        if facts_type is AscendSpaceFacts and query is None:
            return facts_result(self, facts_type, ascend_space_facts(self))
        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, memory_hierarchy(self, query))
        if facts_type is PerformanceServiceFacts:
            return facts_result(self, facts_type, performance_service(self, query))
        if facts_type is ThroughputFacts:
            return facts_result(self, facts_type, throughput(self, query))
        return super().get_facts(facts_type, query)

    def to_python(self) -> PythonExpr:
        if type(self) is AscendTarget and self.architecture_id and self.device_id:
            return PythonExpr(
                ("from tilefoundry_ascend import AscendTarget",),
                f'AscendTarget("{self.device_id}")',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the compiler architecture triple name."""
        return self.architecture.name


__all__ = ["AscendTarget"]
