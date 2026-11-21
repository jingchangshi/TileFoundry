"""SchedulePolicy contract, MECE 7-Facet Lattice, and Dual-Track Template Registry."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    ConstraintRelaxationMask,
    TopologyAxes,
)

SHELL_AIV_SIMD_STREAM = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
SHELL_AIV_SIMT_ATOMIC = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
SHELL_AIC_GEMM_FIXPIPE = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
SHELL_CV_RAGGED_EPILOGUE = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
SHELL_CV_RING_PIPELINE = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
SHELL_CV_PERSISTENT_STATE = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
SHELL_AIV_STRIPED_VM = TopologyAxes(
    "aiv_only", "persistent_onchip_state", "dynamic_opcode_stream"
)

CANONICAL_TOPOLOGY_PRESETS: tuple[TopologyAxes, ...] = (
    SHELL_AIV_SIMD_STREAM,
    SHELL_AIV_SIMT_ATOMIC,
    SHELL_AIC_GEMM_FIXPIPE,
    SHELL_CV_RAGGED_EPILOGUE,
    SHELL_CV_RING_PIPELINE,
    SHELL_CV_PERSISTENT_STATE,
    SHELL_AIV_STRIPED_VM,
)


def _read_attr_or_key(obj: object | None, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _observation_from_residual(residual: object | None) -> ObservationVector5D | None:
    """Extract an embedded 5D observation from a residual dataclass or serialized mapping."""
    raw_observation = _read_attr_or_key(residual, "observation", None)
    if isinstance(raw_observation, ObservationVector5D):
        return raw_observation
    if isinstance(raw_observation, Mapping):
        return ObservationVector5D.from_dict(raw_observation)
    return None


_DOMINANT_BOUND_FACET_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "exec_mode_and_spr",
        ("spill", "vloopv2", "exec_mode", "regbase", "simt", "spr"),
    ),
    (
        "sram_layout_and_aliasing",
        ("l1_overflow", "ub_overflow", "capacity_overflow", "bank", "sram"),
    ),
    ("dma_and_cache_hierarchy", ("dma", "mte2", "hbm", "l2", "cache")),
    (
        "pipeline_and_sync_graph",
        (
            "sync",
            "crosscore",
            "ring_sync",
            "stage_dag",
            "pipeline",
            "multi_mmad",
            "mmad",
            "alias",
        ),
    ),
    (
        "grid_and_workspace_topology",
        ("grid", "wave", "swat", "streamk", "workspace", "gm_atomic"),
    ),
)


def _dominant_bound_facets(dominant_bound: str) -> frozenset[str]:
    """Map a free-form dominant-bound label onto the causal facets its tokens imply."""
    return frozenset(
        facet
        for facet, tokens in _DOMINANT_BOUND_FACET_TOKENS
        if any(token in dominant_bound for token in tokens)
    )


def coerce_topology_axes(axes_or_mapping: TopologyAxes | Mapping[str, Any]) -> TopologyAxes:
    """Coerce TopologyAxes or axis-dict into a validated TopologyAxes instance."""
    if isinstance(axes_or_mapping, TopologyAxes):
        return axes_or_mapping
    if isinstance(axes_or_mapping, Mapping):
        return TopologyAxes.from_dict(axes_or_mapping)
    raise TypeError(
        f"Expected TopologyAxes or Mapping[str, Any], got {type(axes_or_mapping).__name__}"
    )


def resolve_canonical_shell_axes(
    axes: TopologyAxes | Mapping[str, Any],
    available: Mapping[TopologyAxes, Any] | None = None,
) -> TopologyAxes:
    """Resolve an arbitrary 3-axis TopologyAxes onto its registered coordination shell preset."""
    target = coerce_topology_axes(axes)
    if available is not None and target in available:
        return target
    if target in CANONICAL_TOPOLOGY_PRESETS and available is None:
        return target

    if target.iteration_regularity == "dynamic_opcode_stream":
        cand = SHELL_AIV_STRIPED_VM
    elif target.memory_ring == "persistent_onchip_state":
        cand = SHELL_CV_PERSISTENT_STATE
    elif target.memory_ring == "bidirectional_cv_ring":
        cand = SHELL_CV_RING_PIPELINE
    elif target.memory_ring == "l0c_ub_epilogue" or (
        target.uses_cube and target.iteration_regularity == "ragged_group"
    ):
        cand = SHELL_CV_RAGGED_EPILOGUE
    elif target.memory_ring == "gm_l1_l0_fixpipe" or target.uses_cube:
        cand = SHELL_AIC_GEMM_FIXPIPE
    elif target.memory_ring == "ub_random_atomic" or target.iteration_regularity == "irregular_simt":
        cand = SHELL_AIV_SIMT_ATOMIC
    else:
        cand = SHELL_AIV_SIMD_STREAM

    if available is not None and cand not in available:
        raise KeyError(
            f"No coordination shell registered for {target.key()} (mapped to {cand.key()})"
        )
    return cand


def relax_facets_for_bottleneck(
    mask: ConstraintRelaxationMask,
    *,
    dominant_bound: str = "",
    residual: object | None = None,
    compiler_summary: object | None = None,
    l1_overflow: bool = False,
    ub_overflow: bool = False,
) -> ConstraintRelaxationMask:
    """Unlock minimal causal MECE facets, with dominant-bound fallback when 5D is quiet."""
    dom = str(dominant_bound or "").strip().lower()
    observation = _observation_from_residual(residual)
    if observation is not None:
        relaxed = observation.compute_axis_residuals().unlock_causal_facets(mask)
        if relaxed != mask:
            return relaxed
        fallback = _dominant_bound_facets(dom)
        if l1_overflow or ub_overflow:
            fallback |= frozenset({"sram_layout_and_aliasing"})
        return relaxed.unlock(*sorted(fallback)) if fallback else relaxed

    unlock_exec_mode_and_spr = mask.exec_mode_and_spr
    unlock_sram_layout = mask.sram_layout_and_aliasing
    unlock_dma_cache = mask.dma_and_cache_hierarchy
    unlock_pipeline_sync = mask.pipeline_and_sync_graph
    unlock_grid_workspace = mask.grid_and_workspace_topology

    dom_facets = _dominant_bound_facets(dom)

    sync_stall = float(_read_attr_or_key(residual, "measured_sync_stall_ratio", 0.0) or 0.0)
    mte2_wait = float(_read_attr_or_key(residual, "measured_mte2_wait_ratio", 0.0) or 0.0)
    raw_l2 = _read_attr_or_key(residual, "l2_hit_rate", None)
    l2_hit = 1.0 if raw_l2 is None else float(raw_l2)
    bank_conflict = float(_read_attr_or_key(residual, "bank_conflict_ratio", 0.0) or 0.0)
    pseudo_ser = bool(_read_attr_or_key(residual, "pseudo_double_buffer_serialized", False))
    wave_ratio = float(_read_attr_or_key(residual, "wave_quantization_ratio", 1.0) or 1.0)
    dual_issue_collapse = float(_read_attr_or_key(residual, "dual_issue_ipc_collapse", 0.0) or 0.0)

    vf_breaks = max(
        int(_read_attr_or_key(residual, "vf_fusion_breaks", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "vf_fusion_breaks", 0) or 0),
    )
    alias_ser = max(
        int(_read_attr_or_key(residual, "alias_serialized_buffers", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "alias_serialized_buffers", 0) or 0),
    )
    dcci_flushes = max(
        int(_read_attr_or_key(residual, "dcache_dcci_flushes", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "dcache_dcci_flushes", 0) or 0),
    )
    spill_bytes = max(
        int(_read_attr_or_key(residual, "reg_spill_bytes", 0) or 0),
        int(_read_attr_or_key(residual, "spill_bytes", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "reg_spill_bytes", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "spill_bytes", 0) or 0),
    )
    software_loops = max(
        int(_read_attr_or_key(compiler_summary, "software_loops_emitted", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "software_loops", 0) or 0),
    )
    vloopv2_loops = max(
        int(_read_attr_or_key(compiler_summary, "vloopv2_loops_emitted", 0) or 0),
        int(_read_attr_or_key(compiler_summary, "vloopv2_loops", 0) or 0),
    )
    local_membar = bool(
        _read_attr_or_key(residual, "has_local_membar_in_vf_call", False)
        or _read_attr_or_key(compiler_summary, "has_local_membar_in_vf_call", False)
    )

    if (
        spill_bytes > 0
        or (software_loops > 0 and vloopv2_loops == 0)
        or dual_issue_collapse >= 0.4
        or "exec_mode_and_spr" in dom_facets
    ):
        unlock_exec_mode_and_spr = True

    if (
        l1_overflow
        or ub_overflow
        or bank_conflict > 0.10
        or "sram_layout_and_aliasing" in dom_facets
    ):
        unlock_sram_layout = True

    if (
        mte2_wait > 0.30
        or l2_hit < 0.70
        or dcci_flushes > 0
        or "dma_and_cache_hierarchy" in dom_facets
    ):
        unlock_dma_cache = True

    if (
        sync_stall > 0.25
        or local_membar
        or alias_ser > 0
        or vf_breaks > 0
        or pseudo_ser
        or "pipeline_and_sync_graph" in dom_facets
    ):
        unlock_pipeline_sync = True

    if wave_ratio > 1.25 or "grid_and_workspace_topology" in dom_facets:
        unlock_grid_workspace = True

    return ConstraintRelaxationMask(
        alg_compute_and_monoid=mask.alg_compute_and_monoid,
        tile_and_stage_knobs=mask.tile_and_stage_knobs,
        exec_mode_and_spr=unlock_exec_mode_and_spr,
        sram_layout_and_aliasing=unlock_sram_layout,
        dma_and_cache_hierarchy=unlock_dma_cache,
        pipeline_and_sync_graph=unlock_pipeline_sync,
        grid_and_workspace_topology=unlock_grid_workspace,
    )


class SchedulePolicy(TypedDict, total=False):
    """Parameterization contract shared by SIL, TileLang, and AscendC shell templates."""

    regime_tag: str
    topology_axes: dict[str, str] | TopologyAxes
    backend_tier: str
    tile_mnk: tuple[int, int, int]
    row_pack_factor: int
    num_kv_splits: int
    l1_stages: int
    l1_half_bank_split: bool
    l1_slot_aliasing: dict[str, str]
    l0c_buffers: int
    ub_256kb_unlock: bool
    ub_estimated_bytes: int
    l1_estimated_bytes: int
    ub_bank_pad_datablocks: int
    ub_rw_shift_datablocks: int
    ub_row_stride_bytes: int
    cv_datapath: str
    sync_mode: str
    preload_depth: int
    reverse_ack_period: int
    ensure_empty_subblock_sync: bool
    swizzle_window: int
    tail_wave_split: bool
    streamk_mode: str
    l2_bypass_operands: tuple[str, ...]
    vector_mode: str
    direct_no_fusion_vf: bool
    multi_row_unroll: int
    reduce_accumulators: int
    simt_threads: int
    hf32_allowed: bool
    fast_unshifted_exp: bool
    w4_mode: str
    dma_paths_used: tuple[str, ...]
    cube_operand_dtype: str
    dtype: str
    dim_size: int
    freedom_mask: dict[str, bool] | ConstraintRelaxationMask


SCHEDULE_POLICY_KEYS: tuple[str, ...] = (
    "tile_mnk",
    "row_pack_factor",
    "l1_stages",
    "l1_half_bank_split",
    "ub_256kb_unlock",
    "ub_bank_pad_datablocks",
    "cv_datapath",
    "sync_mode",
    "swizzle_window",
    "simt_threads",
    "multi_row_unroll",
    "w4_mode",
)

DEFAULT_POLICIES_BY_AXES: dict[TopologyAxes, dict[str, Any]] = {
    SHELL_AIV_SIMD_STREAM: {
        "tile_mnk": (8, 128, 1),
        "row_pack_factor": 4,
        "l1_stages": 1,
        "l1_half_bank_split": False,
        "ub_256kb_unlock": True,
        "ub_bank_pad_datablocks": 1,
        "cv_datapath": "none",
        "sync_mode": "local_mem_bar",
        "swizzle_window": 1,
        "simt_threads": 0,
        "multi_row_unroll": 2,
        "w4_mode": "none",
        "dtype": "bfloat16",
        "dim_size": 128,
    },
    SHELL_AIV_SIMT_ATOMIC: {
        "tile_mnk": (16, 64, 1),
        "row_pack_factor": 1,
        "l1_stages": 1,
        "l1_half_bank_split": False,
        "ub_256kb_unlock": False,
        "ub_bank_pad_datablocks": 8,
        "cv_datapath": "none",
        "sync_mode": "warp_shfl_cas",
        "swizzle_window": 1,
        "simt_threads": 512,
        "multi_row_unroll": 1,
        "w4_mode": "none",
        "dtype": "float32",
        "dim_size": 64,
    },
    SHELL_AIC_GEMM_FIXPIPE: {
        "tile_mnk": (128, 128, 64),
        "row_pack_factor": 1,
        "l1_stages": 2,
        "l1_half_bank_split": True,
        "ub_256kb_unlock": False,
        "ub_bank_pad_datablocks": 0,
        "cv_datapath": "l0c_to_gm_fixpipe",
        "sync_mode": "unit_flag",
        "swizzle_window": 4,
        "simt_threads": 0,
        "multi_row_unroll": 1,
        "w4_mode": "w4a16",
        "dtype": "float16",
        "dim_size": 128,
    },
    SHELL_CV_RAGGED_EPILOGUE: {
        "tile_mnk": (128, 128, 64),
        "row_pack_factor": 1,
        "l1_stages": 2,
        "l1_half_bank_split": True,
        "ub_256kb_unlock": False,
        "ub_bank_pad_datablocks": 1,
        "cv_datapath": "direct_ub",
        "sync_mode": "cross_core_flag",
        "swizzle_window": 4,
        "simt_threads": 0,
        "multi_row_unroll": 2,
        "w4_mode": "none",
        "dtype": "bfloat16",
        "dim_size": 128,
    },
    SHELL_CV_RING_PIPELINE: {
        "tile_mnk": (96, 112, 64),
        "row_pack_factor": 1,
        "l1_stages": 3,
        "l1_half_bank_split": True,
        "ub_256kb_unlock": True,
        "ub_bank_pad_datablocks": 1,
        "cv_datapath": "ub_l1_ring",
        "sync_mode": "cross_core_flag",
        "swizzle_window": 4,
        "simt_threads": 0,
        "multi_row_unroll": 2,
        "w4_mode": "none",
        "dtype": "float16",
        "dim_size": 128,
    },
    SHELL_CV_PERSISTENT_STATE: {
        "tile_mnk": (64, 64, 64),
        "row_pack_factor": 2,
        "l1_stages": 2,
        "l1_half_bank_split": False,
        "ub_256kb_unlock": True,
        "ub_bank_pad_datablocks": 1,
        "cv_datapath": "direct_ub",
        "sync_mode": "cross_core_flag",
        "swizzle_window": 1,
        "simt_threads": 0,
        "multi_row_unroll": 2,
        "w4_mode": "none",
        "dtype": "float32",
        "dim_size": 64,
    },
    SHELL_AIV_STRIPED_VM: {
        "tile_mnk": (16, 128, 1),
        "row_pack_factor": 1,
        "l1_stages": 1,
        "l1_half_bank_split": False,
        "ub_256kb_unlock": True,
        "ub_bank_pad_datablocks": 1,
        "cv_datapath": "none",
        "sync_mode": "local_mem_bar",
        "swizzle_window": 1,
        "simt_threads": 0,
        "multi_row_unroll": 2,
        "w4_mode": "none",
        "dtype": "float32",
        "dim_size": 128,
    },
}

_DTYPE_BRIDGE_ALIASES: dict[str, str] = {
    "f16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "f32": "float32",
    "float32": "float32",
}


def normalize_schedule_policy(
    topology_axes: TopologyAxes | Mapping[str, Any],
    policy: Mapping[str, Any] | object | None = None,
) -> dict[str, Any]:
    """Resolve *topology_axes* to its canonical shell and merge SchedulePolicy overrides."""
    shell_axes = resolve_canonical_shell_axes(topology_axes, DEFAULT_POLICIES_BY_AXES)
    merged = dict(DEFAULT_POLICIES_BY_AXES[shell_axes])
    merged["topology_axes"] = shell_axes.to_dict()
    policy_map: Mapping[str, Any] | None = None
    if policy is not None:
        if isinstance(policy, Mapping):
            policy_map = policy
        elif hasattr(policy, "to_dict") and callable(getattr(policy, "to_dict")):
            raw_dict = policy.to_dict()
            if isinstance(raw_dict, Mapping):
                policy_map = raw_dict
        if policy_map is None:
            raise TypeError(f"Expected Mapping or SIL SchedulePolicy, got {type(policy).__name__}")
        for k, v in policy_map.items():
            if v is not None:
                merged[str(k)] = v
        if "dtype" not in policy_map and policy_map.get("cube_operand_dtype"):
            raw_dt = str(policy_map["cube_operand_dtype"]).strip().lower()
            if raw_dt in _DTYPE_BRIDGE_ALIASES:
                merged["dtype"] = _DTYPE_BRIDGE_ALIASES[raw_dt]
        if "dim_size" not in policy_map and "tile_mnk" in policy_map:
            raw_t = policy_map["tile_mnk"]
            if len(raw_t) == 3 and int(raw_t[1]) > 0:
                merged["dim_size"] = min(2048, int(raw_t[1]))

    raw_mnk = merged["tile_mnk"]
    if len(raw_mnk) != 3:
        raise ValueError(f"tile_mnk must have length 3, got {raw_mnk!r}")
    merged["tile_mnk"] = (max(1, int(raw_mnk[0])), max(1, int(raw_mnk[1])), max(1, int(raw_mnk[2])))
    merged["row_pack_factor"] = max(1, int(merged["row_pack_factor"]))
    merged["l1_stages"] = max(1, int(merged["l1_stages"]))
    merged["l1_half_bank_split"] = bool(merged["l1_half_bank_split"])
    merged["ub_256kb_unlock"] = bool(merged["ub_256kb_unlock"])
    merged["ub_bank_pad_datablocks"] = max(0, int(merged["ub_bank_pad_datablocks"]))
    merged["cv_datapath"] = str(merged["cv_datapath"])
    merged["sync_mode"] = str(merged["sync_mode"])
    merged["swizzle_window"] = max(1, int(merged["swizzle_window"]))
    merged["simt_threads"] = max(0, int(merged["simt_threads"]))
    merged["multi_row_unroll"] = max(1, int(merged["multi_row_unroll"]))
    merged["w4_mode"] = str(merged["w4_mode"])
    merged["dtype"] = str(merged.get("dtype", "float16"))
    merged["dim_size"] = max(1, int(merged.get("dim_size", 128)))
    if "freedom_mask" in merged and merged["freedom_mask"] is not None:
        raw_mask = merged["freedom_mask"]
        if isinstance(raw_mask, ConstraintRelaxationMask):
            merged["freedom_mask"] = raw_mask.to_dict()
        elif isinstance(raw_mask, Mapping):
            merged["freedom_mask"] = ConstraintRelaxationMask.from_dict(raw_mask).to_dict()
        else:
            raise TypeError(
                f"freedom_mask must be ConstraintRelaxationMask or Mapping, got {type(raw_mask).__name__}"
            )
    return merged


@dataclass(frozen=True)
class ComposableComputeSlotSpec:
    """Algorithmic synthesis contract injecting a compute body into a 3-axis coordination shell."""

    op_name: str
    freedom_mask: ConstraintRelaxationMask = field(default_factory=ConstraintRelaxationMask.bottom)
    coordination_shell: TopologyAxes = field(default_factory=lambda: SHELL_AIV_SIMD_STREAM)
    slot_name: str = "compute_body"
    compute_body_cpp: str = ""
    atom_sequence: tuple[str, ...] = ()
    l1_alias_map: dict[str, str] = field(default_factory=dict)
    custom_ring_sync: str = ""
    invariants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op_name, str) or not self.op_name.strip():
            raise ValueError("ComposableComputeSlotSpec.op_name must be a non-empty string")
        if isinstance(self.freedom_mask, Mapping):
            object.__setattr__(
                self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask)
            )
        elif not isinstance(self.freedom_mask, ConstraintRelaxationMask):
            raise TypeError(
                "ComposableComputeSlotSpec.freedom_mask must be ConstraintRelaxationMask"
            )
        object.__setattr__(
            self, "coordination_shell", coerce_topology_axes(self.coordination_shell)
        )
        object.__setattr__(self, "atom_sequence", tuple(str(a) for a in self.atom_sequence))
        object.__setattr__(
            self,
            "l1_alias_map",
            {str(k): str(v) for k, v in dict(self.l1_alias_map).items()},
        )
        object.__setattr__(self, "custom_ring_sync", str(self.custom_ring_sync))
        object.__setattr__(self, "invariants", tuple(str(inv) for inv in self.invariants))

        if not self.freedom_mask.is_unrestricted_top:
            if self.l1_alias_map and not self.freedom_mask.sram_layout_and_aliasing:
                raise ValueError(
                    "sram_layout_and_aliasing facet must be unlocked to customize l1_alias_map"
                )
            if self.custom_ring_sync and not self.freedom_mask.pipeline_and_sync_graph:
                raise ValueError(
                    "pipeline_and_sync_graph facet must be unlocked to customize custom_ring_sync"
                )
            if self.atom_sequence and not (
                self.freedom_mask.pipeline_and_sync_graph
                or self.freedom_mask.dma_and_cache_hierarchy
                or self.freedom_mask.exec_mode_and_spr
            ):
                raise ValueError(
                    "atom_sequence composition requires unlocking pipeline_and_sync_graph, dma_and_cache_hierarchy, or exec_mode_and_spr"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "freedom_mask": self.freedom_mask.to_dict(),
            "coordination_shell": self.coordination_shell.to_dict(),
            "slot_name": self.slot_name,
            "compute_body_cpp": self.compute_body_cpp,
            "atom_sequence": list(self.atom_sequence),
            "l1_alias_map": dict(self.l1_alias_map),
            "custom_ring_sync": self.custom_ring_sync,
            "invariants": list(self.invariants),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComposableComputeSlotSpec:
        raw_mask = payload.get("freedom_mask")
        mask = (
            ConstraintRelaxationMask.from_dict(raw_mask)
            if isinstance(raw_mask, Mapping)
            else (
                raw_mask
                if isinstance(raw_mask, ConstraintRelaxationMask)
                else ConstraintRelaxationMask.bottom()
            )
        )
        raw_shell = payload.get("coordination_shell", SHELL_AIV_SIMD_STREAM)
        shell = (
            raw_shell
            if isinstance(raw_shell, TopologyAxes)
            else (
                TopologyAxes.from_dict(raw_shell)
                if isinstance(raw_shell, Mapping)
                else SHELL_AIV_SIMD_STREAM
            )
        )
        return cls(
            op_name=str(payload["op_name"]),
            freedom_mask=mask,
            coordination_shell=shell,
            slot_name=str(payload.get("slot_name", "compute_body")),
            compute_body_cpp=str(payload.get("compute_body_cpp", "")),
            atom_sequence=tuple(str(x) for x in (payload.get("atom_sequence") or ())),
            l1_alias_map={
                str(k): str(v) for k, v in (payload.get("l1_alias_map") or {}).items()
            },
            custom_ring_sync=str(payload.get("custom_ring_sync", "")),
            invariants=tuple(str(x) for x in (payload.get("invariants") or ())),
        )

    def render_contract(self, policy: Mapping[str, Any] | None = None) -> str:
        """Render [Coordination Shell + SchedulePolicy + Compute Graph] AscendC artifact."""
        norm = normalize_schedule_policy(self.coordination_shell, policy)
        norm["freedom_mask"] = self.freedom_mask.to_dict()
        atoms_line = ", ".join(self.atom_sequence) if self.atom_sequence else "default_shell_atoms"
        alias_line = ", ".join(f"{k}->{v}" for k, v in self.l1_alias_map.items()) or "none"
        sync_line = self.custom_ring_sync or "shell_default"
        unlocked_line = ",".join(self.freedom_mask.unlocked_facets())
        inv_lines = "\n".join(f"//   - invariant: {inv}" for inv in self.invariants)
        body = self.compute_body_cpp.strip() or f"// default {self.slot_name} functor body"
        header = [
            "// Algorithmic Synthesis Contract: [Coordination Shell + SchedulePolicy Knobs + Synthesized Compute Graph]",
            f"// op_name={self.op_name} | unrestricted_top={self.freedom_mask.is_unrestricted_top} | unlocked=[{unlocked_line}] | shell={self.coordination_shell.key()} | slot={self.slot_name}",
            f"// atom_sequence=[{atoms_line}] | l1_alias_map={{{alias_line}}} | custom_ring_sync={sync_line}",
        ]
        if inv_lines:
            header.append(inv_lines)
        slot_block = (
            f"\n// === Synthesized Compute Slot ({self.slot_name}) ===\n"
            f"{body}\n"
            f"// === End Synthesized Compute Slot ===\n"
        )
        if self.freedom_mask.is_unrestricted_top and self.compute_body_cpp.strip():
            return "\n".join(header) + "\n" + slot_block
        shell_cpp = render_template(self.coordination_shell, norm, track="ascendc").ascendc_cpp
        return "\n".join(header) + "\n" + slot_block + "\n" + shell_cpp


@dataclass(frozen=True)
class CoordinationShellSpec:
    """Descriptor for one canonical 3-axis Ascend 950 coordination shell."""

    topology_axes: TopologyAxes
    title: str
    description: str
    core_domain: str
    tilelang_builder: Callable[[dict[str, Any]], dict[str, Any]]
    ascendc_renderer: Callable[[dict[str, Any]], str]
    default_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedTemplate:
    """Dual-track template rendering output for a TopologyAxes shell and SchedulePolicy."""

    topology_axes: TopologyAxes
    policy: dict[str, Any]
    prim_func: object | None = None
    pass_configs: dict[str, object] = field(default_factory=dict)
    compile_flags: tuple[str, ...] = ()
    ascendc_cpp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TemplateRegistry:
    """Registry mapping canonical 3-axis TopologyAxes shells to TileLang and AscendC templates."""

    def __init__(self) -> None:
        self._specs: dict[TopologyAxes, CoordinationShellSpec] = {}

    def register(self, spec: CoordinationShellSpec) -> None:
        axes = coerce_topology_axes(spec.topology_axes)
        self._specs[axes] = spec

    def list_topology_axes(self) -> tuple[TopologyAxes, ...]:
        ordered = [axes for axes in CANONICAL_TOPOLOGY_PRESETS if axes in self._specs]
        extras = [axes for axes in self._specs if axes not in CANONICAL_TOPOLOGY_PRESETS]
        return tuple(ordered + extras)

    def resolve_shell_axes(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> TopologyAxes:
        return resolve_canonical_shell_axes(topology_axes, self._specs)

    def get_spec(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> CoordinationShellSpec:
        shell_axes = self.resolve_shell_axes(topology_axes)
        return self._specs[shell_axes]

    def default_policy(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> dict[str, Any]:
        shell_axes = self.resolve_shell_axes(topology_axes)
        return normalize_schedule_policy(shell_axes, self.get_spec(shell_axes).default_policy)

    def render(
        self,
        topology_axes: TopologyAxes | Mapping[str, Any],
        policy: Mapping[str, Any] | object | None = None,
        *,
        track: Literal["tilelang", "ascendc", "both"] = "both",
    ) -> RenderedTemplate:
        shell_axes = self.resolve_shell_axes(topology_axes)
        spec = self._specs[shell_axes]
        norm_policy = normalize_schedule_policy(shell_axes, policy)
        prim_func = None
        pass_configs: dict[str, object] = {}
        compile_flags: tuple[str, ...] = ()
        metadata: dict[str, Any] = {}
        ascendc_cpp = ""

        if track in ("tilelang", "both"):
            tl_out = spec.tilelang_builder(norm_policy)
            prim_func = tl_out["prim_func"]
            pass_configs = dict(tl_out.get("pass_configs") or {})
            compile_flags = tuple(str(f) for f in (tl_out.get("compile_flags") or ()))
            metadata = dict(tl_out.get("metadata") or {})

        if track in ("ascendc", "both"):
            ascendc_cpp = spec.ascendc_renderer(norm_policy)

        return RenderedTemplate(
            topology_axes=shell_axes,
            policy=norm_policy,
            prim_func=prim_func,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            ascendc_cpp=ascendc_cpp,
            metadata=metadata,
        )

    def render_template(
        self,
        topology_axes: TopologyAxes | Mapping[str, Any],
        policy: Mapping[str, Any] | object | None = None,
        *,
        track: Literal["tilelang", "ascendc", "both"] = "both",
    ) -> RenderedTemplate:
        """Alias for render() matching TemplateRegistry.render_template contract."""
        return self.render(topology_axes, policy, track=track)

    def render_with_compute_slot(
        self,
        slot_spec: ComposableComputeSlotSpec,
        policy: Mapping[str, Any] | object | None = None,
    ) -> RenderedTemplate:
        """Render a coordination shell infused with a ComposableComputeSlotSpec compute body."""
        base_policy = normalize_schedule_policy(slot_spec.coordination_shell, policy)
        base_policy["freedom_mask"] = slot_spec.freedom_mask.to_dict()
        if slot_spec.l1_alias_map:
            base_policy["l1_slot_aliasing"] = dict(slot_spec.l1_alias_map)
        synthesized_cpp = slot_spec.render_contract(base_policy)
        shell_axes = self.resolve_shell_axes(slot_spec.coordination_shell)
        return RenderedTemplate(
            topology_axes=shell_axes,
            policy=base_policy,
            ascendc_cpp=synthesized_cpp,
            metadata={
                "op_name": slot_spec.op_name,
                "freedom_mask": slot_spec.freedom_mask.to_dict(),
                "unrestricted_top": slot_spec.freedom_mask.is_unrestricted_top,
                "slot_name": slot_spec.slot_name,
                "atom_sequence": list(slot_spec.atom_sequence),
            },
        )


_GLOBAL_REGISTRY: TemplateRegistry | None = None


def get_template_registry() -> TemplateRegistry:
    """Return the singleton TemplateRegistry with all canonical Ascend 950 shells."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        from tilefoundry_ascend.ascend.templates.ascendc_templates import (  # noqa: PLC0415
            ASCENDC_RENDERERS,
        )
        from tilefoundry_ascend.ascend.templates.tilelang_templates import (  # noqa: PLC0415
            TILELANG_BUILDERS,
        )

        reg = TemplateRegistry()
        descriptions = {
            SHELL_AIV_SIMD_STREAM: (
                "Pure RegBase SIMD VF",
                "AIV-only",
                "256KB UB unlock, 16-32KB burst row packing, 2-row unroll, predicated prime tails, inf/nan-safe math",
            ),
            SHELL_AIV_SIMT_ATOMIC: (
                "SIMT + Hybrid SIMD/SIMT UB",
                "AIV-SIMT/SIMD",
                "512-thread __simt_vf__, +8-padded UB atomics, leader-only asc_atomic_cas + __shfl, UintDiv, <=512B SIMT vs >512B SIMD",
            ),
            SHELL_AIC_GEMM_FIXPIPE: (
                "Dense & Quantized GEMM",
                "AIC/AIV-Mix",
                "WINDOW_LEN SWAT serpentine swizzle, Half-L1 isolation, unit_flag gating, W4A16/MXA8W4 unpack/FixPipe deq_scale, L2 disable",
            ),
            SHELL_CV_RAGGED_EPILOGUE: (
                "Grouped GEMM + Fused Epilogue",
                "AIC/AIV-Mix",
                "Rolling cross-group core assignment, m==0 fast skip with subblock sync parity, BaseMBalance, L0C->UB SPLIT_M + SwiGLU",
            ),
            SHELL_CV_RING_PIPELINE: (
                "Heterogeneous 3-4 Stage CV Ring",
                "AIC/AIV-Mix",
                "M>=82 Cube-bound tile sizing (e.g. 96x112), L0C->UB SPLIT_M + UB->L1 Nd2Nz, in-place K_rope->P L1 aliasing",
            ),
            SHELL_CV_PERSISTENT_STATE: (
                "Persistent Chunk/Recurrent State",
                "AIC/AIV-Mix",
                "Persistent (B,H) ownership, FP32 on-chip state residency, anchor=G_tail/2 exp range halving, initC=false L0C acc, 5-step NS",
            ),
            SHELL_AIV_STRIPED_VM: (
                "Striped Register-Resident VM",
                "AIV-only",
                "Horizontal W-dimension AIV striping, 100% UB-resident regs[R, W_slice], zero cross-core sync, LocalMemBar hazard tracking",
            ),
        }
        for axes in CANONICAL_TOPOLOGY_PRESETS:
            title, domain, desc = descriptions[axes]
            reg.register(
                CoordinationShellSpec(
                    topology_axes=axes,
                    title=title,
                    description=desc,
                    core_domain=domain,
                    tilelang_builder=TILELANG_BUILDERS[axes],
                    ascendc_renderer=ASCENDC_RENDERERS[axes],
                    default_policy=DEFAULT_POLICIES_BY_AXES[axes],
                )
            )
        _GLOBAL_REGISTRY = reg
    return _GLOBAL_REGISTRY


def render_template(
    topology_axes: TopologyAxes | Mapping[str, Any],
    policy: Mapping[str, Any] | object | None = None,
    *,
    track: Literal["tilelang", "ascendc", "both"] = "both",
) -> RenderedTemplate:
    """Convenience wrapper rendering a coordination shell template from the global registry."""
    return get_template_registry().render(topology_axes, policy, track=track)


_SHELL_CONTRACT_LINE_RE = re.compile(
    r"^(?://\s*Algorithmic Synthesis Contract:.*|"
    r"//\s*op_name=.*unrestricted_top=.*|"
    r"//\s*atom_sequence=.*l1_alias_map=.*|"
    r"//\s*-\s*invariant:.*|"
    r"//\s*===\s*(?:End\s+)?Synthesized Compute Slot.*)\n?",
    re.MULTILINE,
)


def _infer_entry_core_attr(
    body: str,
    *,
    axes: TopologyAxes | None = None,
) -> str:
    if axes is not None:
        if axes.compute_engine == "coupled_1c2v":
            return "__mix__(1, 2)"
        if axes.compute_engine == "aic_only":
            return "__cube__"
        return "__vector__"
    has_aic = "ASCEND_IS_AIC" in body
    has_aiv = "ASCEND_IS_AIV" in body
    if has_aic and has_aiv:
        return "__mix__(1, 2)"
    if ("Mmad(" in body or "FixpipeParams" in body) and not has_aiv and "Reg::" not in body:
        return "__cube__"
    return "__vector__"


def _sanitize_c_ident(name: str, default: str = "ascend_top_kernel") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    if not cleaned or cleaned[0].isdigit():
        return default
    return cleaned


_C_TYPE_KEYWORDS = frozenset(
    {
        "auto",
        "bool",
        "char",
        "const",
        "double",
        "float",
        "int",
        "long",
        "short",
        "signed",
        "struct",
        "unsigned",
        "void",
        "volatile",
    }
)


def _split_top_level_csv(params_text: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in params_text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            item = "".join(cur).strip()
            if item:
                parts.append(item)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_c_param_name(param_decl: str) -> str:
    depth = 0
    eq_idx = -1
    for idx, ch in enumerate(param_decl):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]" and depth > 0:
            depth -= 1
        elif ch == "=" and depth == 0:
            eq_idx = idx
            break
    lhs = param_decl[:eq_idx].strip() if eq_idx >= 0 else param_decl.strip()
    lhs = re.sub(r"(?:\s*\[[^\]]*\])+\s*$", "", lhs).strip()
    fn_ptr = re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", lhs)
    if fn_ptr:
        return fn_ptr.group(1)
    ident = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs)
    if ident and ident.group(1) not in _C_TYPE_KEYWORDS:
        return ident.group(1)
    return ""


def _format_host_launcher_stub(
    host_signature: str | Mapping[str, Any],
    entry_fn: str,
) -> str:
    if isinstance(host_signature, Mapping):
        fn_name = _sanitize_c_ident(
            str(host_signature.get("name") or host_signature.get("op_name") or f"launch_{entry_fn}"),
            default=f"launch_{entry_fn}",
        )
        raw_params = host_signature.get("params") or host_signature.get("args") or ()
        param_parts: list[str] = ["void* stream"]
        call_args: list[str] = []
        for item in raw_params:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                ptype = str(item[0]).strip()
                pname = _sanitize_c_ident(str(item[1]), "arg")
                if pname == "stream":
                    param_parts[0] = f"{ptype} stream"
                else:
                    param_parts.append(f"{ptype} {pname}")
                    call_args.append(pname)
            else:
                pstr = str(item).strip()
                if not pstr or pstr == "void":
                    continue
                pname = _extract_c_param_name(pstr)
                if pname == "stream":
                    param_parts[0] = pstr
                else:
                    param_parts.append(pstr)
                    if pname:
                        call_args.append(pname)
        if len(param_parts) == 1:
            param_parts.extend(["void* x", "void* y"])
            call_args.extend(["x", "y"])
        sig_decl = f"inline void {fn_name}({', '.join(param_parts)})"
        return (
            "#ifndef __CCE_AICORE__\n"
            f'extern "C" {sig_decl} {{\n'
            f"    {entry_fn}<<<1, nullptr, stream>>>({', '.join(call_args)});\n"
            "}\n"
            "#endif"
        )

    sig_text = str(host_signature).strip().rstrip(";").strip()
    if "(" in sig_text and ")" in sig_text:
        if "{" in sig_text:
            return f"#ifndef __CCE_AICORE__\n{sig_text}\n#endif"
        l_paren = sig_text.find("(")
        r_paren = sig_text.rfind(")")
        params_inner = sig_text[l_paren + 1 : r_paren].strip()
        call_args = []
        stream_expr = ""
        for p_item in _split_top_level_csv(params_inner):
            if p_item in ("void", "..."):
                continue
            pname = _extract_c_param_name(p_item)
            if not pname:
                continue
            if not stream_expr and (
                pname == "stream" or re.search(r"\b(?:aclrtStream|rtStream_t)\b", p_item)
            ):
                stream_expr = pname
            else:
                call_args.append(pname)
        sig_decl = re.sub(r'^extern\s+"C"\s+', "", sig_text)
        if stream_expr == "stream":
            stream_setup = ""
        elif stream_expr:
            stream_setup = f"    void* stream = {stream_expr};\n"
        else:
            stream_setup = "    void* stream = nullptr;\n"
        return (
            "#ifndef __CCE_AICORE__\n"
            f'extern "C" {sig_decl} {{\n'
            f"{stream_setup}"
            f"    {entry_fn}<<<1, nullptr, stream>>>({', '.join(call_args)});\n"
            "}\n"
            "#endif"
        )

    fn_name = _sanitize_c_ident(sig_text, f"launch_{entry_fn}")
    sig_decl = f"inline void {fn_name}(void* stream, void* x, void* y)"
    return (
        "#ifndef __CCE_AICORE__\n"
        f'extern "C" {sig_decl} {{\n'
        f"    {entry_fn}<<<1, nullptr, stream>>>(x, y);\n"
        "}\n"
        "#endif"
    )


def _extract_body_from_input(
    source_or_slot: object | None,
    *,
    policy: Mapping[str, Any] | object | None,
    lowered_kernel_cpp: str | None,
) -> tuple[str, str, TopologyAxes | None]:
    """Extract (raw_cpp, op_name, topology_axes) from slot, shell, PrimFunc, or C++ source."""
    op_name = "ascend_top_kernel"
    axes: TopologyAxes | None = None

    if policy is not None:
        raw_axes = _read_attr_or_key(policy, "topology_axes", None)
        if raw_axes is not None:
            try:
                axes = coerce_topology_axes(raw_axes)
            except Exception:
                axes = None

    if isinstance(source_or_slot, ComposableComputeSlotSpec):
        op_name = source_or_slot.op_name
        axes = axes or source_or_slot.coordination_shell
        if lowered_kernel_cpp is not None and lowered_kernel_cpp.strip():
            return lowered_kernel_cpp, op_name, axes
        return source_or_slot.compute_body_cpp, op_name, axes

    if isinstance(source_or_slot, TopologyAxes):
        axes = source_or_slot
        if lowered_kernel_cpp is not None and lowered_kernel_cpp.strip():
            return lowered_kernel_cpp, op_name, axes
        rendered = render_template(axes, policy, track="ascendc")
        return rendered.ascendc_cpp, op_name, axes

    if isinstance(source_or_slot, RenderedTemplate):
        axes = axes or source_or_slot.topology_axes
        meta_name = source_or_slot.metadata.get("op_name") if source_or_slot.metadata else None
        if meta_name:
            op_name = str(meta_name)
        if lowered_kernel_cpp is not None and lowered_kernel_cpp.strip():
            return lowered_kernel_cpp, op_name, axes
        return source_or_slot.ascendc_cpp, op_name, axes

    if lowered_kernel_cpp is not None and lowered_kernel_cpp.strip():
        if isinstance(source_or_slot, str) and source_or_slot.strip() and "\n" not in source_or_slot and ";" not in source_or_slot:
            op_name = source_or_slot.strip()
        return lowered_kernel_cpp, op_name, axes

    if source_or_slot is None:
        return "", op_name, axes

    if hasattr(source_or_slot, "kernel_source"):
        ks = getattr(source_or_slot, "kernel_source", "")
        return str(ks or ""), op_name, axes

    if isinstance(source_or_slot, str):
        return source_or_slot, op_name, axes

    try:
        import tilelang  # noqa: PLC0415

        art = tilelang.lower(source_or_slot, target="ascend")
        ks = getattr(art, "kernel_source", "")
        if ks:
            return str(ks), op_name, axes
    except Exception:
        pass

    return str(source_or_slot), op_name, axes


def _looks_like_file_scope_cpp(body: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\n)\s*(?:"
            r"extern\s+\"C\"|"
            r"template\s*<|"
            r"(?:struct|class|enum|namespace)\s+[A-Za-z_]\w*|"
            r"(?:constexpr|static|inline|__aicore__|__global__|__simd_vf__|__simt_vf__|__no_simd_vf_fusion__)\b|"
            r"(?:void|int|uint32_t|int32_t|float|half|bfloat16_t)\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{"
            r")",
            body,
        )
    )


def materialize_top_translation_unit(
    source_or_slot: object | None,
    *,
    policy: Mapping[str, Any] | object | None = None,
    lowered_kernel_cpp: str | None = None,
    host_signature: str | Mapping[str, Any] | None = None,
) -> str:
    """Materialize a self-contained AscendC translation unit at ConstraintRelaxationMask.top().

    Transitions either a constrained coordination shell / ComposableComputeSlotSpec or a
    lowered Track-A TileLang PrimFunc (``tilelang.lower(...).kernel_source``) into a
    standalone native C++ translation unit with ``#include "kernel_operator.h"``,
    an ``extern "C" __global__ __aicore__`` entry point, optional Host launcher stub,
    and ZERO locked shell wrappers.
    """
    raw_cpp, op_name, axes = _extract_body_from_input(
        source_or_slot,
        policy=policy,
        lowered_kernel_cpp=lowered_kernel_cpp,
    )
    cleaned = _SHELL_CONTRACT_LINE_RE.sub("", raw_cpp).strip()

    has_tl_templates = "tl_templates/ascend/" in cleaned
    has_kernel_op_inc = bool(
        re.search(r'^\s*#\s*include\s+[<"]kernel_operator\.h[>"]', cleaned, re.MULTILINE)
    )
    has_using_ns = bool(re.search(r"^\s*using\s+namespace\s+AscendC\s*;", cleaned, re.MULTILINE))

    core_attr = _infer_entry_core_attr(cleaned, axes=axes)
    entry_fn = _sanitize_c_ident(f"{op_name}_kernel", "ascend_top_kernel")

    has_global_entry = bool(re.search(r"\b__global__\b", cleaned))
    if has_global_entry:
        def _upgrade_global(m: re.Match[str]) -> str:
            prefix = m.group(0)
            if "__aicore__" in prefix:
                return prefix
            return prefix.replace("__global__", "__global__ __aicore__", 1)

        cleaned = re.sub(
            r'(?:extern\s+"C"\s+)?__global__(?:\s+__[A-Za-z0-9_]+__(?:\([^)]*\))?)*',
            _upgrade_global,
            cleaned,
        )
        if not re.search(r'extern\s+"C"\s+__global__', cleaned):
            cleaned = re.sub(r"\b__global__\b", 'extern "C" __global__', cleaned, count=1)

    header_lines: list[str] = [
        "// TileFoundry Top-Element Standalone AscendC Translation Unit (ConstraintRelaxationMask.top())",
        f"// op_name={op_name} | unrestricted_top=True | shell_locks=none",
    ]
    if axes is not None:
        header_lines.append(f"// topology_axes={axes.key()}")

    if not has_kernel_op_inc:
        if has_tl_templates:
            pass
        else:
            header_lines.append('#include "kernel_operator.h"')
            if not has_using_ns:
                header_lines.append("using namespace AscendC;")
    elif not has_using_ns and not has_tl_templates:
        cleaned = "using namespace AscendC;\n\n" + cleaned

    parts: list[str] = ["\n".join(header_lines)]

    if has_global_entry:
        if cleaned:
            parts.append(cleaned)
    elif _looks_like_file_scope_cpp(cleaned):
        if cleaned:
            parts.append(cleaned)
        if core_attr == "__cube__":
            entry_body = "    if ASCEND_IS_AIC {\n        (void)x;\n        (void)y;\n    }"
        elif core_attr.startswith("__mix__"):
            entry_body = (
                "    if ASCEND_IS_AIC {\n        (void)x;\n    }\n"
                "    if ASCEND_IS_AIV {\n        (void)y;\n    }"
            )
        else:
            entry_body = "    if ASCEND_IS_AIV {\n        (void)x;\n        (void)y;\n    }"
        parts.append(
            f'extern "C" __global__ __aicore__ {core_attr} void {entry_fn}('
            "__gm__ float* x, __gm__ float* y) {\n"
            f"{entry_body}\n"
            "}"
        )
    else:
        indented_body = (
            "\n".join(f"    {ln}" if ln.strip() else "" for ln in cleaned.splitlines())
            if cleaned
            else "    if ASCEND_IS_AIV {\n        (void)x;\n        (void)y;\n    }"
        )
        parts.append(
            f'extern "C" __global__ __aicore__ {core_attr} void {entry_fn}('
            "__gm__ float* x, __gm__ float* y) {\n"
            "    (void)x;\n"
            "    (void)y;\n"
            f"{indented_body}\n"
            "}"
        )

    if host_signature:
        parts.append(_format_host_launcher_stub(host_signature, entry_fn))

    return "\n\n".join(p for p in parts if p.strip()) + "\n"


__all__ = [
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
