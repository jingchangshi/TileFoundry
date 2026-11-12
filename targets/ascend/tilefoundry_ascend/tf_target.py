"""TileFoundry-native Ascend950PR Target.

Registered as a first-class Target so normal spec/target discovery, fact
projection and the schedule-finalize path can consume it. Storage names reuse
the existing StorageKind vocabulary (gmem/smem/rmem) so TF type and allocation
analyses keep working without a parallel space enum.

Scope of this declaration: it states AIV-only synchronous execution and coarse
spatial ownership. It does not model AIC reachability, VF register classes,
PREG/VREG escape rules, or MTE issue/completion; those are separate P1/P2
obligations.
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

DEFAULT_AIV = 56
DEFAULT_AIC = 28
UB_BYTES = 256 * 1024


@register_target
@dataclass(frozen=True)
class Ascend950PRTarget(Target):
    """Ascend950PR device: AIV execution domain, GM + UB + register file."""

    name = "ascend950pr"
    aiv_count: int = DEFAULT_AIV
    aic_count: int = DEFAULT_AIC

    def __post_init__(self) -> None:
        for field_name in ("aiv_count", "aic_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Ascend950PRTarget: {field_name} must be a positive int, got {value!r}"
                )

    def _topology(self) -> TopologyFacts:
        # AIV first because it is the only admitted execution level here; AIC is
        # a sibling engine, not a nested level under AIV.
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
                # Aggregate register-file capacity is not sourced yet. 256 B is
                # one vector register's width, not the per-AIV register budget.
                ExplicitMemoryLevelFacts("rmem", None, "aiv", "aiv"),
                # CARRIER is a DISTINCT register class (bf16 transport windows), not
                # smem/UB and not an f32 element register. Declaring it here is what
                # lets the allocation analysis give a private carrier value a lifetime
                # without charging it to the 256 KiB UB budget. Aggregate capacity is
                # UNSOURCED (None), exactly as for rmem; 256 B/window is a register
                # width, never a total budget.
                ExplicitMemoryLevelFacts("carrier", None, "aiv", "aiv"),
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

    def completion_ops(self) -> dict:
        """The completion ops this target admits, bound to its own execution mesh.

        The core FlagSet/FlagWait/Drain declare thread_execution_mesh(); an Ascend kernel runs
        in an ("aiv",) mesh, so the standard verifier rejects the core ops with "does not
        match Mesh(('thread',)...)". The Aiv* subclasses already exist in sync.py for exactly
        this reason -- this hook is what makes the completion emitter USE them instead of the
        core classes. Imported lazily: sync.py imports the op registries, and this module is
        imported while the target itself is being defined.
        """
        from .sync import AivDrain, AivFlagSet, AivFlagWait

        return {"set": AivFlagSet, "wait": AivFlagWait, "drain": AivDrain}

    def torch_device_type(self) -> str:
        return "npu"

    def to_python(self) -> PythonExpr:
        if type(self) is Ascend950PRTarget:
            fields = []
            if self.aiv_count != DEFAULT_AIV:
                fields.append(f"aiv_count={self.aiv_count}")
            if self.aic_count != DEFAULT_AIC:
                fields.append(f"aic_count={self.aic_count}")
            args = ", ".join(fields)
            return PythonExpr(
                ("from tilefoundry_ascend.tf_target import Ascend950PRTarget",),
                f"Ascend950PRTarget({args})",
            )
        return super().to_python()


__all__ = ["Ascend950PRTarget"]
