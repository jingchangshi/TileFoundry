"""Pure-data coordination shell presets and SchedulePolicy normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, TypeVar

_T = TypeVar("_T")

from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes

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


def _read_attr_or_key(obj: object | None, key: str, default: _T = None) -> object | _T:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def coerce_topology_axes(axes_or_mapping: TopologyAxes | Mapping[str, object]) -> TopologyAxes:
    """Coerce TopologyAxes or axis-dict into a validated TopologyAxes instance."""
    if isinstance(axes_or_mapping, TopologyAxes):
        return axes_or_mapping
    if isinstance(axes_or_mapping, Mapping):
        return TopologyAxes.from_dict(axes_or_mapping)
    raise TypeError(
        f"Expected TopologyAxes or Mapping[str, object], got {type(axes_or_mapping).__name__}"
    )


def resolve_canonical_shell_axes(
    axes: TopologyAxes | Mapping[str, object],
    available: Mapping[TopologyAxes, object] | None = None,
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

DEFAULT_POLICIES_BY_AXES: dict[TopologyAxes, dict[str, object]] = {
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
    topology_axes: TopologyAxes | Mapping[str, object],
    policy: Mapping[str, object] | object | None = None,
) -> dict[str, object]:
    """Resolve *topology_axes* to its canonical shell and merge SchedulePolicy overrides."""
    shell_axes = resolve_canonical_shell_axes(topology_axes, DEFAULT_POLICIES_BY_AXES)
    merged: dict[str, object] = dict(DEFAULT_POLICIES_BY_AXES[shell_axes])
    merged["topology_axes"] = shell_axes.to_dict()
    policy_map: Mapping[str, object] | None = None
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
                "freedom_mask must be ConstraintRelaxationMask or Mapping, "
                f"got {type(raw_mask).__name__}"
            )
    return merged


__all__ = [
    "CANONICAL_TOPOLOGY_PRESETS",
    "DEFAULT_POLICIES_BY_AXES",
    "SCHEDULE_POLICY_KEYS",
    "SHELL_AIC_GEMM_FIXPIPE",
    "SHELL_AIV_SIMD_STREAM",
    "SHELL_AIV_SIMT_ATOMIC",
    "SHELL_AIV_STRIPED_VM",
    "SHELL_CV_PERSISTENT_STATE",
    "SHELL_CV_RAGGED_EPILOGUE",
    "SHELL_CV_RING_PIPELINE",
    "SchedulePolicy",
    "_DTYPE_BRIDGE_ALIASES",
    "_read_attr_or_key",
    "coerce_topology_axes",
    "normalize_schedule_policy",
    "resolve_canonical_shell_axes",
]
