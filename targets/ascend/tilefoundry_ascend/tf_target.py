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

#: UB->L1 ND2NZ staging contract (task-27 C1). Scratch is (rows+1)*cols ELEMENTS of
#: the destination dtype in shared.dyn; the extra row is the UB bank-conflict padding.
ND2NZ_SCRATCH_SCOPE = "shared.dyn"
ND2NZ_SCRATCH_ALIGNMENT = 32
ND2NZ_SCRATCH_STATUS = "REFERENCE_IMPLEMENTATION"


def nd2nz_scratch_extent(rows: int, cols: int) -> int:
    """Scratch extent in ELEMENTS: exactly (rows + 1) * cols (never rows*cols)."""
    rows = int(rows)
    cols = int(cols)
    if rows < 0 or cols < 0:
        raise ValueError("nd2nz scratch rows/cols must be non-negative")
    return (rows + 1) * cols


#: dual_copy lowering paths (task-27 C3): L0C->UB stays a hardware dual-destination
#: copy on AIC; GM<->UB and UB->L1 lower to per-AIV software copies indexed by cthread.
DUAL_COPY_PATHS = {
    "L0C->UB": {"mechanism": "hardware_dual_destination", "owner": "aic",
                "same_dtype_required": True, "min_rank": 2,
                "status": "REFERENCE_IMPLEMENTATION"},
    "GM->UB": {"mechanism": "software_per_aiv_copy", "owner": "aiv",
               "status": "REFERENCE_IMPLEMENTATION"},
    "UB->GM": {"mechanism": "software_per_aiv_copy", "owner": "aiv",
               "status": "REFERENCE_IMPLEMENTATION"},
    "UB->L1": {"mechanism": "software_per_aiv_copy_then_nd2nz", "owner": "aiv",
               "status": "REFERENCE_IMPLEMENTATION"},
}

#: Post-schedule refinement order after auto_schedule=False (task-27 D1). auto_schedule
#: controls only the scheduling branch; these passes still run.
POST_SCHEDULE_PASSES = (
    "InferBufferAliases",
    "MergeUBAllocations(align_bytes=32)",
    "RewriteFlagToBuf",
    "AscendThreadSync('shared')",
    "AscendThreadSync('shared.dyn')",
)

#: AscendThreadSync scope facts (task-27 A6): only SIMT_VF blocks are planned;
#: global scope returns unchanged. Real device effect is UNRESOLVED.
THREAD_SYNC_FACTS = {
    "pass": "tl.AscendThreadSync",
    "processed_block": "SIMT_VF",
    "global_scope": "RETURN_UNCHANGED",
    "status": "REFERENCE_IMPLEMENTATION",
    "device_effect": "UNRESOLVED",
}


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

    def sibling_cores(self) -> dict:
        """AIC and AIV are sibling execution engines, not nested levels.

        A hand-off between them is a cross-core mechanism; neither core owns the other's
        local storage. This mapping is the explicit owner relation used by target facts.
        """
        return {"aiv": "aic", "aic": "aiv"}

    def core_owners(self) -> dict:
        return {"aiv": "aiv", "aic": "aic", "device": "device"}

    def space_facts(self) -> dict:
        """Every declared memory space (L1/L0A/B/C/SS/UB included), with provenance.

        Kept separate from MemoryHierarchyFacts because the generic TF hierarchy
        vocabulary only admits gmem/smem/rmem/carrier names; adding "l1"/"l0a" there
        would make parser storage resolution fail. This is the authoritative Ascend
        space-fact table.
        """
        from .memory import space_facts_table

        return space_facts_table()

    def target_event_facts(self):
        """The declared physical + logical event surface of this target."""
        from .completion.model import ASCEND_EVENT_FACTS

        return ASCEND_EVENT_FACTS

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


__all__ = ["Ascend950PRTarget", "nd2nz_scratch_extent", "DUAL_COPY_PATHS",
           "POST_SCHEDULE_PASSES", "THREAD_SYNC_FACTS", "ND2NZ_SCRATCH_SCOPE",
           "ND2NZ_SCRATCH_ALIGNMENT", "ND2NZ_SCRATCH_STATUS"]
