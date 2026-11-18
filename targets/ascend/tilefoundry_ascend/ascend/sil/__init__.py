"""Scheduling-Interface-Layer (SIL) package for Ascend 950 (dav-3510)."""

from tilefoundry_ascend.ascend.sil.op_analyzer import (
    DEFAULT_OPS_ROOT,
    OpAdvisoryExtractor,
    get_default_extractor,
    resolve_canonical_topology_axes,
    synthesize_default_policy,
)
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_BACKEND_TIERS,
    VALID_COMPUTE_ENGINES,
    VALID_FREEDOM_FACETS,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    VALID_VECTOR_MODES,
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    SchedulingEnvelope,
    TelemetryFeedback,
    TopologyAxes,
    VerificationReport,
)
from tilefoundry_ascend.ascend.sil.verifier import (
    SILVerifier,
    verify_policy_envelope,
    verify_source_against_facets,
)

__all__ = [
    "DEFAULT_OPS_ROOT",
    "VALID_BACKEND_TIERS",
    "VALID_COMPUTE_ENGINES",
    "VALID_FREEDOM_FACETS",
    "VALID_ITERATION_REGULARITIES",
    "VALID_MEMORY_RINGS",
    "VALID_VECTOR_MODES",
    "AdvisoryEnvelope",
    "ConstraintRelaxationMask",
    "HardwareLegalityBounds",
    "OpAdvisoryExtractor",
    "PolicyEnvelope",
    "RegimeSlice",
    "SILVerifier",
    "SchedulePolicy",
    "SchedulingEnvelope",
    "TelemetryFeedback",
    "TopologyAxes",
    "VerificationReport",
    "get_default_extractor",
    "resolve_canonical_topology_axes",
    "synthesize_default_policy",
    "verify_policy_envelope",
    "verify_source_against_facets",
]
