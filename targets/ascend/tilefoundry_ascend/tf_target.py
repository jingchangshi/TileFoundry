"""TileFoundry-native Ascend950PR Target.

Registered as a first-class Target so normal spec/target discovery, fact
projection and the schedule-finalize path can consume it. Storage names reuse
the existing StorageKind vocabulary (gmem/smem/rmem) so TF type and allocation
analyses keep working without a parallel space enum.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
)
from tilefoundry.ir.types.storage import TARGET_MEMORY_OWNER
from tilefoundry.target.base import Target, register_target
from tilefoundry.target.facts import (
    MemoryExecutionFacts,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from tilefoundry.utils.python_source import PythonExpr

UB_BYTES = 256 * 1024
VREG_BYTES = 256


@register_target
@dataclass(frozen=True)
class Ascend950PRTarget(Target):
    """Ascend950PR device: 56 AIV / 28 AIC, GM + UB + VREG/PREG."""

    name = "ascend950pr"
    aiv_count: int = 56
    aic_count: int = 28

    def _topology(self) -> TopologyFacts:
        return TopologyFacts(
            (
                TopologyLevelFacts("aiv", self.aiv_count, self.aiv_count, from_target=True),
                TopologyLevelFacts("aic", self.aic_count, self.aic_count, from_target=True),
            ),
            parallel_level="aiv",
        )

    def _memory(self) -> MemoryHierarchyFacts:
        return MemoryHierarchyFacts(
            explicit_levels=(
                ExplicitMemoryLevelFacts("gmem", None, "target", TARGET_MEMORY_OWNER),
                ExplicitMemoryLevelFacts("smem", UB_BYTES, "aiv", "aiv"),
                ExplicitMemoryLevelFacts("rmem", VREG_BYTES, "aiv", "aiv"),
            ),
            implicit_levels=(
                ImplicitMemoryLevelFacts("l2", None, "target"),
            ),
            relations=(
                MemoryLevelRelation(MemoryRelationKind.CACHES, near="l2", far="gmem"),
            ),
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, self._topology())
        if facts_type is MemoryExecutionFacts and query is None:
            return facts_result(self, facts_type, MemoryExecutionFacts(("aiv",)))
        if facts_type is MemoryHierarchyFacts:
            return facts_result(self, facts_type, self._memory())
        return super().get_facts(facts_type, query)

    def torch_device_type(self) -> str:
        return "npu"

    def to_python(self) -> PythonExpr:
        if type(self) is Ascend950PRTarget:
            return PythonExpr(
                ("from tilefoundry_ascend.tf_target import Ascend950PRTarget",),
                "Ascend950PRTarget()",
            )
        return super().to_python()


__all__ = ["Ascend950PRTarget"]
