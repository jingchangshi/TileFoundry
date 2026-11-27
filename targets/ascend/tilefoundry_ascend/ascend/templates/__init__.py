"""Dual-Track SIL Parameterized Template Engine for Ascend 950 (dav-3510)."""

from tilefoundry_ascend.ascend.templates.ascendc_templates import ASCENDC_RENDERERS
from tilefoundry_ascend.ascend.templates.registry import (
    ComposableComputeSlotSpec,
    ConstraintRelaxationMask,
    CoordinationShellSpec,
    RenderedTemplate,
    TemplateRegistry,
    TopologyAxes,
    get_template_registry,
    materialize_top_translation_unit,
    relax_facets_for_bottleneck,
    render_template,
)
from tilefoundry_ascend.ascend.templates.shells import (
    CANONICAL_TOPOLOGY_PRESETS,
    DEFAULT_POLICIES_BY_AXES,
    SCHEDULE_POLICY_KEYS,
    SHELL_AIC_GEMM_FIXPIPE,
    SHELL_AIV_SIMD_STREAM,
    SHELL_AIV_SIMT_ATOMIC,
    SHELL_AIV_STRIPED_VM,
    SHELL_CV_PERSISTENT_STATE,
    SHELL_CV_RAGGED_EPILOGUE,
    SHELL_CV_RING_PIPELINE,
    SchedulePolicy,
    coerce_topology_axes,
    normalize_schedule_policy,
    resolve_canonical_shell_axes,
)
from tilefoundry_ascend.ascend.sil.schema import VALID_FREEDOM_FACETS
from tilefoundry_ascend.ascend.templates.tilelang_templates import TILELANG_BUILDERS

__all__ = [
    "ASCENDC_RENDERERS",
    "CANONICAL_TOPOLOGY_PRESETS",
    "ComposableComputeSlotSpec",
    "ConstraintRelaxationMask",
    "CoordinationShellSpec",
    "DEFAULT_POLICIES_BY_AXES",
    "RenderedTemplate",
    "SCHEDULE_POLICY_KEYS",
    "SHELL_AIC_GEMM_FIXPIPE",
    "SHELL_AIV_SIMD_STREAM",
    "SHELL_AIV_SIMT_ATOMIC",
    "SHELL_AIV_STRIPED_VM",
    "SHELL_CV_PERSISTENT_STATE",
    "SHELL_CV_RAGGED_EPILOGUE",
    "SHELL_CV_RING_PIPELINE",
    "SchedulePolicy",
    "TILELANG_BUILDERS",
    "TemplateRegistry",
    "TopologyAxes",
    "VALID_FREEDOM_FACETS",
    "coerce_topology_axes",
    "get_template_registry",
    "materialize_top_translation_unit",
    "normalize_schedule_policy",
    "relax_facets_for_bottleneck",
    "render_template",
    "resolve_canonical_shell_axes",
]
