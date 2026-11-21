"""Static-dynamic residual modeling and two-lane (Exploit + Stepping-Stone) beam search."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry_ascend.agent.router.capability_gate import StructuralFeatures
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.observation_space import (
    CompilerSummaryLike,
    ObservationVector5D,
    RooflineSummaryLike,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_NOVELTY_TOKENS = (
    "T.alloc_ss",
    "T.dual_copy",
    "T.Persistent",
    "T.SimtVF",
    "T.Scope",
    "unit_flag_ctrl",
    "vgather2",
    "T.MixedKernel",
    "T.Pipelined",
    "T.SimdVF",
    "PURE_SIMDVF_UBUF_FLAGS",
    "T.set_atomic",
)


def _read_float(payload: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    """Read the first present numeric telemetry key, otherwise return the neutral default."""
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _read_int(payload: Mapping[str, object], *keys: str, default: int = 0) -> int:
    """Read the first present integral telemetry key, otherwise return the neutral default."""
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return default


def _read_bool(payload: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Read a boolean telemetry flag with Python truthiness, otherwise return the default."""
    if key not in payload:
        return default
    return bool(payload[key])


@dataclass(frozen=True)
class StaticDynamicResidual:
    """Gap between predicted ledger utilization and measured NPU msprof counters."""

    measured_latency_us: float
    predicted_bottleneck_util: float
    measured_aic_mac_ratio: float
    measured_aiv_vec_ratio: float
    measured_mte2_wait_ratio: float
    measured_sync_stall_ratio: float
    unexplained_gap_ratio: float
    pseudo_double_buffer_serialized: bool = False
    spill_bytes: int = 0
    bank_conflict_ratio: float = 0.0
    l2_hit_rate: float = 1.0
    vf_fusion_breaks: int = 0
    alias_serialized_buffers: int = 0
    dual_issue_ipc_collapse: float = 0.0
    dcache_dcci_flushes: int = 0
    has_local_membar_in_vf_call: bool = False
    vloopv2_hit: bool = True
    observation: ObservationVector5D | None = None
    transfer_inflation_ratio: float = 0.0
    effective_compute_eta: float = 0.0

    @property
    def has_l1_compiler_summary(self) -> bool:
        """Report whether L1 penalties came from a real bisheng compiler summary."""
        return self.observation is not None and self.observation.has_l1_compiler_summary

    @property
    def measured_l2_fields(self) -> tuple[str, ...]:
        """Report which L2 fields were actually observed in msprof telemetry."""
        return self.observation.measured_l2_fields if self.observation is not None else ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation including the 5D observation."""
        return {
            "measured_latency_us": self.measured_latency_us,
            "predicted_bottleneck_util": self.predicted_bottleneck_util,
            "measured_aic_mac_ratio": self.measured_aic_mac_ratio,
            "measured_aiv_vec_ratio": self.measured_aiv_vec_ratio,
            "measured_mte2_wait_ratio": self.measured_mte2_wait_ratio,
            "measured_sync_stall_ratio": self.measured_sync_stall_ratio,
            "unexplained_gap_ratio": self.unexplained_gap_ratio,
            "pseudo_double_buffer_serialized": self.pseudo_double_buffer_serialized,
            "spill_bytes": self.spill_bytes,
            "bank_conflict_ratio": self.bank_conflict_ratio,
            "l2_hit_rate": self.l2_hit_rate,
            "vf_fusion_breaks": self.vf_fusion_breaks,
            "alias_serialized_buffers": self.alias_serialized_buffers,
            "dual_issue_ipc_collapse": self.dual_issue_ipc_collapse,
            "dcache_dcci_flushes": self.dcache_dcci_flushes,
            "has_local_membar_in_vf_call": self.has_local_membar_in_vf_call,
            "vloopv2_hit": self.vloopv2_hit,
            "transfer_inflation_ratio": self.transfer_inflation_ratio,
            "effective_compute_eta": self.effective_compute_eta,
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StaticDynamicResidual:
        """Reconstruct StaticDynamicResidual from a serialized dictionary."""
        raw_observation = payload.get("observation")
        if raw_observation is None:
            observation = None
        elif isinstance(raw_observation, ObservationVector5D):
            observation = raw_observation
        elif isinstance(raw_observation, Mapping):
            observation = ObservationVector5D.from_dict(raw_observation)
        else:
            raise ValueError(
                f"residual field 'observation' must be a mapping, got "
                f"{type(raw_observation).__name__}"
            )
        default_inflation = (
            observation.transfer_inflation_ratio if observation is not None else 0.0
        )
        default_eff_eta = observation.effective_compute_eta if observation is not None else 0.0
        return cls(
            measured_latency_us=float(payload.get("measured_latency_us", 0.0)),
            predicted_bottleneck_util=float(payload.get("predicted_bottleneck_util", 0.0)),
            measured_aic_mac_ratio=float(payload.get("measured_aic_mac_ratio", 0.0)),
            measured_aiv_vec_ratio=float(payload.get("measured_aiv_vec_ratio", 0.0)),
            measured_mte2_wait_ratio=float(payload.get("measured_mte2_wait_ratio", 0.0)),
            measured_sync_stall_ratio=float(payload.get("measured_sync_stall_ratio", 0.0)),
            unexplained_gap_ratio=float(payload.get("unexplained_gap_ratio", 0.0)),
            pseudo_double_buffer_serialized=bool(
                payload.get("pseudo_double_buffer_serialized", False)
            ),
            spill_bytes=int(payload.get("spill_bytes", 0)),
            bank_conflict_ratio=float(payload.get("bank_conflict_ratio", 0.0)),
            l2_hit_rate=float(payload.get("l2_hit_rate", 1.0)),
            vf_fusion_breaks=max(0, int(payload.get("vf_fusion_breaks", 0) or 0)),
            alias_serialized_buffers=max(0, int(payload.get("alias_serialized_buffers", 0) or 0)),
            dual_issue_ipc_collapse=max(
                0.0, min(1.0, float(payload.get("dual_issue_ipc_collapse", 0.0) or 0.0))
            ),
            dcache_dcci_flushes=max(0, int(payload.get("dcache_dcci_flushes", 0) or 0)),
            has_local_membar_in_vf_call=bool(payload.get("has_local_membar_in_vf_call", False)),
            vloopv2_hit=bool(payload.get("vloopv2_hit", True)),
            observation=observation,
            transfer_inflation_ratio=max(
                0.0, _read_float(payload, "transfer_inflation_ratio", default=default_inflation)
            ),
            effective_compute_eta=max(
                0.0, _read_float(payload, "effective_compute_eta", default=default_eff_eta)
            ),
        )


def compute_static_dynamic_residual(
    ledger: ObjectivePhysicalLedger,
    telemetry: Mapping[str, object],
    *,
    compiler_summary: CompilerSummaryLike | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
) -> StaticDynamicResidual:
    """Compute StaticDynamicResidual from ledger, msprof telemetry, compiler, and Roofline."""
    measured_us = _read_float(telemetry, "measured_latency_us", "dur_us")
    aic_mac = _read_float(telemetry, "aic_mac_ratio", "aic_mad")
    aiv_vec = _read_float(telemetry, "aiv_vec_ratio", "aiv_vec")
    mte2_wait = _read_float(telemetry, "mte2_wait_ratio", "aiv_mte2", "aic_mte2")
    sync_stall = _read_float(telemetry, "sync_stall_ratio")

    positive_cycles = [v for v in ledger.pipe_cycles.values() if v > 0]
    if positive_cycles and ledger.steady_state_ii is not None and ledger.steady_state_ii > 0:
        dom = max(positive_cycles)
        predicted_util = min(1.0, float(dom) / float(max(dom, ledger.steady_state_ii)))
    elif positive_cycles:
        dom = max(positive_cycles)
        non_dom = sum(positive_cycles) - dom
        predicted_util = min(1.0, float(dom) / float(max(1.0, dom + 0.25 * non_dom)))
    else:
        predicted_util = 0.0

    pmu_keys = (
        "aic_mac_ratio",
        "aic_mad",
        "aiv_vec_ratio",
        "aiv_vec",
        "mte2_wait_ratio",
        "aiv_mte2",
        "aic_fixpipe",
    )
    if any(key in telemetry for key in pmu_keys):
        aic_fix = _read_float(telemetry, "aic_fixpipe")
        observed_util = max(aic_mac, aiv_vec, mte2_wait, aic_fix)
        unexplained = min(
            1.0,
            max(max(0.0, predicted_util - observed_util), sync_stall * 0.5),
        )
    elif "roofline_eta" in telemetry:
        eta = min(1.0, max(0.0, _read_float(telemetry, "roofline_eta")))
        unexplained = min(1.0, max(max(0.0, predicted_util - eta), sync_stall * 0.5))
    else:
        unexplained = min(1.0, sync_stall * 0.5)

    steady_ii = ledger.steady_state_ii
    ii_looks_pipelined = steady_ii is None or steady_ii <= 2
    pseudo_serialized = bool(
        ledger.uses_dynamic_offset_pipelining
        and ii_looks_pipelined
        and (mte2_wait >= 0.32 or unexplained >= 0.25 or sync_stall >= 0.20)
    )
    comp_spill = compiler_summary.spill_bytes if compiler_summary is not None else 0
    spill_bytes = max(0, comp_spill, _read_int(telemetry, "spill_bytes"))
    bank_conflict_ratio = max(
        0.0,
        min(
            1.0,
            _read_float(
                telemetry,
                "bank_conflict_ratio",
                "ub_bank_conflict_ratio",
                "resource_conflict_ratio",
            ),
        ),
    )
    l2_hit_rate = max(
        0.0,
        min(
            1.0,
            _read_float(telemetry, "l2_hit_rate", "l2_cache_hit_rate", default=1.0),
        ),
    )

    vf_fusion_breaks = max(
        0,
        _read_int(telemetry, "vf_fusion_breaks"),
        compiler_summary.vf_fusion_breaks if compiler_summary is not None else 0,
    )
    alias_serialized_buffers = max(
        0,
        _read_int(telemetry, "alias_serialized_buffers"),
        compiler_summary.alias_serialized_buffers if compiler_summary is not None else 0,
    )
    dcache_dcci_flushes = max(
        0,
        _read_int(telemetry, "dcache_dcci_flushes"),
        compiler_summary.dcache_dcci_flushes if compiler_summary is not None else 0,
    )
    dual_issue_ipc_collapse = max(
        0.0,
        min(
            1.0,
            max(
                _read_float(telemetry, "dual_issue_ipc_collapse"),
                (
                    compiler_summary.dual_issue_ipc_collapse
                    if compiler_summary is not None
                    else 0.0
                ),
            ),
        ),
    )
    has_local_membar_in_vf_call = _read_bool(
        telemetry, "has_local_membar_in_vf_call"
    ) or (
        compiler_summary.has_local_membar_in_vf_call
        if compiler_summary is not None
        else False
    )
    if "vloopv2_hit" in telemetry:
        vloopv2_hit = _read_bool(telemetry, "vloopv2_hit")
    elif compiler_summary is not None:
        vloopv2_hit = not (
            compiler_summary.software_loops > 0 and compiler_summary.vloopv2_loops == 0
        )
    else:
        vloopv2_hit = True

    observation = ObservationVector5D.from_tiers(
        ledger,
        compiler_summary=compiler_summary,
        telemetry=telemetry,
        roofline=roofline,
    )
    return StaticDynamicResidual(
        measured_latency_us=measured_us,
        predicted_bottleneck_util=round(predicted_util, 6),
        measured_aic_mac_ratio=aic_mac,
        measured_aiv_vec_ratio=aiv_vec,
        measured_mte2_wait_ratio=mte2_wait,
        measured_sync_stall_ratio=sync_stall,
        unexplained_gap_ratio=round(unexplained, 6),
        pseudo_double_buffer_serialized=pseudo_serialized,
        spill_bytes=spill_bytes,
        bank_conflict_ratio=round(bank_conflict_ratio, 6),
        l2_hit_rate=round(l2_hit_rate, 6),
        vf_fusion_breaks=vf_fusion_breaks,
        alias_serialized_buffers=alias_serialized_buffers,
        dual_issue_ipc_collapse=round(dual_issue_ipc_collapse, 6),
        dcache_dcci_flushes=dcache_dcci_flushes,
        has_local_membar_in_vf_call=has_local_membar_in_vf_call,
        vloopv2_hit=vloopv2_hit,
        observation=observation,
        transfer_inflation_ratio=observation.transfer_inflation_ratio,
        effective_compute_eta=observation.effective_compute_eta,
    )


def compiler_summary_from_residual(
    residual: StaticDynamicResidual,
) -> BishengResourceSummary | None:
    """Recover the genuine L1 compiler summary captured in a residual's observation.

    Returns ``None`` when the residual was built without a real bisheng summary so that
    callers rebuilding a residual from cached L2 telemetry never fabricate zero penalties.
    """
    observation = residual.observation
    if observation is None or not observation.has_l1_compiler_summary:
        return None
    return BishengResourceSummary(
        vec_regs=observation.onchip_sram_reg.l1_vec_regs,
        scalar_regs=observation.onchip_sram_reg.l1_scalar_regs,
        spill_bytes=observation.onchip_sram_reg.l1_spill_bytes,
        stack_bytes=observation.onchip_sram_reg.l1_stack_bytes,
        vloopv2_loops=observation.compute_issue.l1_vloopv2_loops,
        software_loops=observation.compute_issue.l1_software_loops,
        has_local_membar_in_vf_call=observation.pipeline_sync.l1_local_membar_in_vf,
        vf_fusion_breaks=observation.compute_issue.l1_vf_fusion_breaks,
        alias_serialized_buffers=observation.pipeline_sync.l1_alias_serialized_buffers,
        dual_issue_ipc_collapse=observation.compute_issue.l1_dual_issue_collapse,
        dcache_dcci_flushes=observation.offchip_dma.l1_dcache_dcci_flushes,
    )


def compute_eq3_hap_objective(
    measured_latency_us: float,
    *,
    ref_perf_us: float = 0.0,
    t_hw_us: float = 0.0,
    spill_bytes: int = 0,
    bank_conflict_ratio: float = 0.0,
    l2_hit_rate: float = 1.0,
    residual: StaticDynamicResidual | None = None,
) -> float:
    """Compute CANNBench Eq.3 performance score discounted by HAP hardware penalties."""
    if measured_latency_us <= 0.0:
        return 0.0
    eff_spill = max(0, int(spill_bytes))
    eff_bank = max(0.0, min(1.0, float(bank_conflict_ratio)))
    eff_l2 = max(0.0, min(1.0, float(l2_hit_rate)))
    if residual is not None:
        eff_spill = max(eff_spill, int(residual.spill_bytes))
        eff_bank = max(eff_bank, float(residual.bank_conflict_ratio))
        if l2_hit_rate == 1.0:
            eff_l2 = max(0.0, min(1.0, float(residual.l2_hit_rate)))

    if t_hw_us > 0.0:
        baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(t_hw_us * 10.0, 10.0)
        denom = (measured_latency_us - t_hw_us) + (baseline - t_hw_us)
        eq3 = max(0.0, (baseline - t_hw_us) / denom) if denom > 0.0 else 0.0
    elif ref_perf_us > 0.0:
        eq3 = float(ref_perf_us) / float(measured_latency_us)
    else:
        eq3 = 1.0 / float(measured_latency_us)

    spill_penalty = 1.0 + min(1.0, float(eff_spill) / 1024.0) * 0.25
    bank_penalty = 1.0 + eff_bank * 0.30
    l2_penalty = 1.0 + (1.0 - eff_l2) * 0.15
    return round(eq3 / (spill_penalty * bank_penalty * l2_penalty), 6)


def _is_int_gt_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 1
    )


def _extract_target_names(stmt: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        names.append(stmt.target.id)
    return names


def _has_static_pingpong_buffers(tree: ast.AST, dsl_source: str) -> bool:
    if "enable_offset=True" in dsl_source:
        return False
    shape_counts: dict[str, int] = {}
    two_slot_vars: set[str] = set()
    for stmt in ast.walk(tree):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
            continue
        call = stmt.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in ("alloc_shared", "alloc_ub", "alloc_fragment"):
            continue
        if not call.args:
            continue
        shape_node = call.args[0]
        shape_key = ast.dump(shape_node)
        target_names = _extract_target_names(stmt)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + max(1, len(target_names))
        if isinstance(shape_node, (ast.Tuple, ast.List)) and len(shape_node.elts) >= 2:
            lead = shape_node.elts[0]
            if isinstance(lead, ast.Constant) and lead.value == 2:
                two_slot_vars.update(target_names)
    if any(count >= 2 for count in shape_counts.values()):
        return True
    if not two_slot_vars:
        return False
    seen_slots: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        var_name = node.value.id
        if var_name not in two_slot_vars:
            continue
        lead_slice = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) and node.slice.elts else node.slice
        if isinstance(lead_slice, ast.Constant) and isinstance(lead_slice.value, int) and lead_slice.value in (0, 1):
            seen_slots.setdefault(var_name, set()).add(lead_slice.value)
    return any(len(slots) >= 2 for slots in seen_slots.values())


def _has_loop_fission_multi_pass(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor)):
            sibling_fors = sum(1 for stmt in node.body if isinstance(stmt, ast.For))
            if sibling_fors >= 2:
                return True
    return False


def _has_pipe_all_barrier(tree: ast.AST, dsl_source: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("ascend_pipe_barrier", "pipe_barrier"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == "PIPE_ALL":
                        return True
    return "ascend_pipe_barrier" in dsl_source and "PIPE_ALL" in dsl_source


def _node_has_add_mult_fma(loop_node: ast.For) -> bool:
    has_exp = False
    has_fma_expr = False
    for inner in ast.walk(loop_node):
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            if inner.func.attr in ("exp", "exp2"):
                has_exp = True
            elif inner.func.attr in ("fma", "vfma"):
                has_fma_expr = True
        elif isinstance(inner, ast.AugAssign) and isinstance(inner.op, ast.Add):
            if any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult) for sub in ast.walk(inner.value)):
                has_fma_expr = True
        elif isinstance(inner, ast.Assign) and isinstance(inner.value, ast.BinOp) and isinstance(inner.value.op, ast.Add):
            if any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult) for sub in ast.walk(inner.value)):
                has_fma_expr = True
    return has_fma_expr and not has_exp


def _has_vfma_unshifted(tree: ast.AST, dsl_source: str) -> bool:
    if "vfma" in dsl_source.lower():
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("fma", "vfma"):
                return True
        if isinstance(node, ast.For) and _node_has_add_mult_fma(node):
            return True
    return False


def _has_split_kv_multi(tree: ast.AST, dsl_source: str) -> bool:
    if "partial_out" in dsl_source:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "num_kv_splits" and _is_int_gt_one(node.value):
            return True
        if isinstance(node, ast.Dict):
            for key_node, val_node in zip(node.keys, node.values):
                if isinstance(key_node, ast.Constant) and key_node.value == "num_kv_splits" and _is_int_gt_one(val_node):
                    return True
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "num_kv_splits" for t in node.targets) and _is_int_gt_one(node.value):
                return True
    return False


_SYNC_NOTIFY_EVENT_RE = re.compile(
    r"asc_sync_notify\s*\([^,]+,[^,]+,\s*((?:[^()]+|\([^()]*\))+)\)"
)


def extract_cpp_trackb_signature_tags(cpp_source: str) -> set[str]:
    """Extract structural mechanism tags from tl_templates/ascend C++20 kernel bodies."""
    tags: set[str] = set()
    if "asc_mmad(" in cpp_source:
        tags.add("cpp:cube_mmad")
    if "asc_copy_gm2l1_nd2nz(" in cpp_source:
        tags.add("cpp:gm2l1_nd2nz")
    if "asc_copy_l0c2ub(" in cpp_source:
        tags.add("cpp:l0c2ub_ring")
    if "asc_lock(" in cpp_source and "asc_unlock(" in cpp_source:
        tags.add("cpp:slot_lock_pipeline")
    events = {ev.strip() for ev in _SYNC_NOTIFY_EVENT_RE.findall(cpp_source) if ev.strip()}
    if len(events) >= 2 or any("%" in ev or "& 1" in ev or "&1" in ev for ev in events):
        tags.add("cpp:pingpong_event_parity")
    if "simd_inst::vsts_pack_b32" in cpp_source:
        tags.add("cpp:vsts_pack_b32")
    if "+ 16" in cpp_source or "+ 32" in cpp_source:
        tags.add("cpp:ub_bank_pad")
    return tags


def extract_structural_signature(
    dsl_source: str,
    *,
    track: str = "",
    residual: StaticDynamicResidual | None = None,
    stage_b: object | None = None,
) -> tuple[str, ...]:
    """Extract sorted architectural primitive signature for stepping-stone diversity."""
    tags: list[str] = []
    if track and track != "unconstrained":
        tags.append(f"track:{track}")
    for token in _NOVELTY_TOKENS:
        if token in dsl_source:
            tags.append(token)
    for hw_token in (
        "SetL2CacheHint",
        "CACHE_MODE_DISABLE",
        "LocalMemBar",
        "DATA_BLOCK_COPY",
        "FixpipeParamsArch3510",
        "__simd_vf__",
        "__simt_vf__",
        "__launch_bounds__",
    ):
        if hw_token in dsl_source:
            tags.append(hw_token)
    eff_spill = 0
    eff_vf_breaks = 0
    eff_alias_ser = 0
    eff_ipc_collapse = 0.0
    eff_dcci = 0
    eff_local_membar = False
    eff_vloop_miss = False
    if residual is not None:
        eff_spill = max(eff_spill, int(residual.spill_bytes))
        eff_vf_breaks = max(eff_vf_breaks, int(residual.vf_fusion_breaks))
        eff_alias_ser = max(eff_alias_ser, int(residual.alias_serialized_buffers))
        eff_ipc_collapse = max(eff_ipc_collapse, float(residual.dual_issue_ipc_collapse))
        eff_dcci = max(eff_dcci, int(residual.dcache_dcci_flushes))
        eff_local_membar = eff_local_membar or bool(residual.has_local_membar_in_vf_call)
        if residual.vloopv2_hit is False:
            eff_vloop_miss = True
        if residual.bank_conflict_ratio > 0.05:
            tags.append("hw:bank_conflict")
        if residual.l2_hit_rate < 0.70:
            tags.append("hw:low_l2_hit")
    if stage_b is not None:
        eff_spill = max(eff_spill, int(getattr(stage_b, "spill_bytes", 0) or 0))
        if getattr(stage_b, "vloopv2_hit", True) is False:
            eff_vloop_miss = True
        comp = getattr(stage_b, "compiler_summary", None)
        if comp is not None:
            eff_spill = max(eff_spill, int(getattr(comp, "spill_bytes", 0) or 0))
            eff_vf_breaks = max(eff_vf_breaks, int(getattr(comp, "vf_fusion_breaks", 0) or 0))
            eff_alias_ser = max(eff_alias_ser, int(getattr(comp, "alias_serialized_buffers", 0) or 0))
            eff_ipc_collapse = max(eff_ipc_collapse, float(getattr(comp, "dual_issue_ipc_collapse", 0.0) or 0.0))
            eff_dcci = max(eff_dcci, int(getattr(comp, "dcache_dcci_flushes", 0) or 0))
            eff_local_membar = eff_local_membar or bool(getattr(comp, "has_local_membar_in_vf_call", False))
            if int(getattr(comp, "software_loops", 0) or 0) > 0 and int(getattr(comp, "vloopv2_loops", 0) or 0) == 0:
                eff_vloop_miss = True
    if eff_vloop_miss:
        tags.append("hw:vloopv2_miss")
    if eff_spill > 0:
        tags.append("hw:register_spill")
    if eff_vf_breaks > 0:
        tags.append("hw:vf_fusion_break")
    if eff_alias_ser > 0:
        tags.append("hw:alias_serialized")
    if eff_ipc_collapse > 0.0:
        tags.append("hw:ipc_collapse")
    if eff_dcci > 0:
        tags.append("hw:dcache_dcci")
    if eff_local_membar:
        tags.append("hw:local_membar_vf")
    tree = ast.parse(dsl_source)
    for node in tree.body:
        val_node = None
        if isinstance(node, ast.Assign):
            val_node = node.value
        elif isinstance(node, ast.AnnAssign):
            val_node = node.value
        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
            cpp_text = val_node.value
            if (
                "tl_templates/ascend/" in cpp_text
                or "__aicore__" in cpp_text
                or "__global__" in cpp_text
                or "__simd_vf__" in cpp_text
            ):
                tags.extend(extract_cpp_trackb_signature_tags(cpp_text))

    if _has_static_pingpong_buffers(tree, dsl_source):
        tags.append("static_pingpong_2buf")
    if _has_loop_fission_multi_pass(tree):
        tags.append("loop_fission_multi_pass")
    if _has_pipe_all_barrier(tree, dsl_source):
        tags.append("pipe_all_barrier")
    if _has_vfma_unshifted(tree, dsl_source):
        tags.append("vfma_unshifted")
    if _has_split_kv_multi(tree, dsl_source):
        tags.append("split_kv_multi")
    return tuple(sorted(set(tags)))


@dataclass(frozen=True)
class BeamCandidate:
    """Single evaluated kernel candidate tracked by the two-lane beam search."""

    candidate_id: str
    round_idx: int
    worker_kind: str
    rule_id: str | None
    dsl_source: str
    track: str
    ledger: ObjectivePhysicalLedger
    residual: StaticDynamicResidual
    latency_us: float
    accuracy_passed: bool
    structural_signature: tuple[str, ...] = ()
    hypothesis: str = ""
    parent_id: str | None = None
    features: StructuralFeatures | None = None
    objective_score: float | None = None

    def __post_init__(self) -> None:
        if not self.track:
            object.__setattr__(self, "track", "unconstrained")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "candidate_id": self.candidate_id,
            "round_idx": self.round_idx,
            "worker_kind": self.worker_kind,
            "rule_id": self.rule_id,
            "dsl_source": self.dsl_source,
            "track": self.track,
            "ledger": self.ledger.to_dict(),
            "residual": self.residual.to_dict(),
            "latency_us": self.latency_us,
            "accuracy_passed": self.accuracy_passed,
            "structural_signature": list(self.structural_signature),
            "hypothesis": self.hypothesis,
            "parent_id": self.parent_id,
            "features": self.features.to_dict() if self.features is not None else None,
            "objective_score": self.objective_score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BeamCandidate:
        """Reconstruct a BeamCandidate from a serialized dictionary."""
        raw_ledger = payload["ledger"]
        ledger = (
            raw_ledger
            if isinstance(raw_ledger, ObjectivePhysicalLedger)
            else ObjectivePhysicalLedger.from_dict(raw_ledger)
        )
        raw_residual = payload["residual"]
        residual = (
            raw_residual
            if isinstance(raw_residual, StaticDynamicResidual)
            else StaticDynamicResidual.from_dict(raw_residual)
        )
        raw_features = payload.get("features")
        if raw_features is None:
            features = None
        elif isinstance(raw_features, StructuralFeatures):
            features = raw_features
        else:
            features = StructuralFeatures.from_dict(raw_features)
        raw_rule = payload.get("rule_id")
        raw_parent = payload.get("parent_id")
        raw_sig = payload.get("structural_signature") or ()
        raw_obj = payload.get("objective_score")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            round_idx=int(payload["round_idx"]),
            worker_kind=str(payload["worker_kind"]),
            rule_id=str(raw_rule) if raw_rule is not None else None,
            dsl_source=str(payload["dsl_source"]),
            track=str(payload.get("track") or "unconstrained"),
            ledger=ledger,
            residual=residual,
            latency_us=float(payload["latency_us"]),
            accuracy_passed=bool(payload["accuracy_passed"]),
            structural_signature=tuple(str(item) for item in raw_sig),
            hypothesis=str(payload.get("hypothesis", "")),
            parent_id=str(raw_parent) if raw_parent is not None else None,
            features=features,
            objective_score=float(raw_obj) if raw_obj is not None else None,
        )


@dataclass
class TwoLaneBeamSearch:
    """Maintains greedy Exploit beam and diversity-preserving Stepping-Stone beam."""

    exploit_width: int = 2
    stepping_stone_width: int = 2
    exploit_beam: list[BeamCandidate] = field(default_factory=list)
    stepping_stone_beam: list[BeamCandidate] = field(default_factory=list)
    exploited_signatures: set[tuple[str, ...]] = field(default_factory=set)

    def mark_parent_selected(self, candidate: BeamCandidate) -> None:
        """Record a parent candidate signature as exploited once expanded."""
        self.exploited_signatures.add(candidate.structural_signature)

    def update(
        self,
        new_candidates: Sequence[BeamCandidate],
    ) -> tuple[list[BeamCandidate], list[BeamCandidate]]:
        """Update Exploit and Stepping-Stone lanes from legal, accurate candidates."""
        valid = [
            c
            for c in new_candidates
            if c.accuracy_passed and c.ledger.is_legal and c.latency_us > 0.0
        ]
        if not valid:
            return list(self.exploit_beam), list(self.stepping_stone_beam)

        exploit_pool = {c.candidate_id: c for c in self.exploit_beam}
        for cand in valid:
            exploit_pool[cand.candidate_id] = cand
        use_objective = any(c.objective_score is not None for c in exploit_pool.values())
        sorted_exploit = sorted(
            exploit_pool.values(),
            key=lambda c: (
                (
                    0 if c.objective_score is not None else 1,
                    -(c.objective_score if c.objective_score is not None else 0.0),
                    c.latency_us,
                    c.candidate_id,
                )
                if use_objective
                else (c.latency_us, c.candidate_id)
            ),
        )
        self.exploit_beam = sorted_exploit[: self.exploit_width]
        exploit_ids = {c.candidate_id for c in self.exploit_beam}
        for c in self.exploit_beam:
            self.exploited_signatures.add(c.structural_signature)

        stone_pool = [
            c
            for c in (*self.stepping_stone_beam, *valid)
            if c.candidate_id not in exploit_ids
        ]
        deduped_stones: dict[tuple[str, ...], BeamCandidate] = {}
        for cand in stone_pool:
            sig = cand.structural_signature or (cand.candidate_id,)
            prev = deduped_stones.get(sig)
            if prev is None or cand.latency_us < prev.latency_us:
                deduped_stones[sig] = cand

        novel_ranked = sorted(
            deduped_stones.values(),
            key=lambda c: (
                0 if c.structural_signature not in self.exploited_signatures else 1,
                -len(c.structural_signature),
                c.latency_us,
                c.candidate_id,
            ),
        )
        self.stepping_stone_beam = novel_ranked[: self.stepping_stone_width]
        return list(self.exploit_beam), list(self.stepping_stone_beam)

    def parent_pool(self) -> list[BeamCandidate]:
        """Return combined unique parent candidates across both beams."""
        merged: dict[str, BeamCandidate] = {}
        for c in (*self.exploit_beam, *self.stepping_stone_beam):
            merged[c.candidate_id] = c
        return list(merged.values())

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of both beam lanes and exploited signatures."""
        return {
            "exploit_width": self.exploit_width,
            "stepping_stone_width": self.stepping_stone_width,
            "exploit_beam": [c.to_dict() for c in self.exploit_beam],
            "stepping_stone_beam": [c.to_dict() for c in self.stepping_stone_beam],
            "exploited_signatures": [list(sig) for sig in sorted(self.exploited_signatures)],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TwoLaneBeamSearch:
        """Reconstruct a TwoLaneBeamSearch instance from a serialized snapshot."""
        raw_exploit = payload.get("exploit_beam") or ()
        raw_stones = payload.get("stepping_stone_beam") or ()
        raw_sigs = payload.get("exploited_signatures") or ()
        return cls(
            exploit_width=int(payload.get("exploit_width", 2)),
            stepping_stone_width=int(payload.get("stepping_stone_width", 2)),
            exploit_beam=[
                c if isinstance(c, BeamCandidate) else BeamCandidate.from_dict(c)
                for c in raw_exploit
            ],
            stepping_stone_beam=[
                c if isinstance(c, BeamCandidate) else BeamCandidate.from_dict(c)
                for c in raw_stones
            ],
            exploited_signatures={tuple(str(item) for item in sig) for sig in raw_sigs},
        )


__all__ = [
    "BeamCandidate",
    "StaticDynamicResidual",
    "TwoLaneBeamSearch",
    "compute_eq3_hap_objective",
    "compute_static_dynamic_residual",
    "compiler_summary_from_residual",
    "extract_cpp_trackb_signature_tags",
    "extract_structural_signature",
]
