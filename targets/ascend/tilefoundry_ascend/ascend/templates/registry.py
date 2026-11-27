"""SchedulePolicy contract, MECE 7-Facet Lattice, and Dual-Track Template Registry."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.templates.ascendc_templates import ASCENDC_RENDERERS
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
    _DTYPE_BRIDGE_ALIASES,
    _read_attr_or_key,
    coerce_topology_axes,
    normalize_schedule_policy,
    resolve_canonical_shell_axes,
)
from tilefoundry_ascend.ascend.templates.tilelang_templates import TILELANG_BUILDERS

_DOMINANT_BOUND_FACET_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exec_mode_and_spr", ("spill", "vloopv2", "exec_mode", "regbase", "simt", "spr")),
    ("sram_layout_and_aliasing", ("l1_overflow", "ub_overflow", "capacity_overflow", "bank", "sram")),
    ("dma_and_cache_hierarchy", ("dma", "mte2", "hbm", "l2", "cache")),
    ("pipeline_and_sync_graph", ("sync", "crosscore", "ring_sync", "stage_dag", "pipeline", "multi_mmad", "mmad", "alias")),
    ("grid_and_workspace_topology", ("grid", "wave", "swat", "streamk", "workspace", "gm_atomic")),
)
_SHELL_DESCRIPTIONS: dict[TopologyAxes, tuple[str, str, str]] = {
    SHELL_AIV_SIMD_STREAM: ("Pure RegBase SIMD VF", "AIV-only", "256KB UB unlock, 16-32KB burst row packing, 2-row unroll, predicated prime tails, inf/nan-safe math"),
    SHELL_AIV_SIMT_ATOMIC: ("SIMT + Hybrid SIMD/SIMT UB", "AIV-SIMT/SIMD", "512-thread __simt_vf__, +8-padded UB atomics, leader-only asc_atomic_cas + __shfl, UintDiv, <=512B SIMT vs >512B SIMD"),
    SHELL_AIC_GEMM_FIXPIPE: ("Dense & Quantized GEMM", "AIC/AIV-Mix", "WINDOW_LEN SWAT serpentine swizzle, Half-L1 isolation, unit_flag gating, W4A16/MXA8W4 unpack/FixPipe deq_scale, L2 disable"),
    SHELL_CV_RAGGED_EPILOGUE: ("Grouped GEMM + Fused Epilogue", "AIC/AIV-Mix", "Rolling cross-group core assignment, m==0 fast skip with subblock sync parity, BaseMBalance, L0C->UB SPLIT_M + SwiGLU"),
    SHELL_CV_RING_PIPELINE: ("Heterogeneous 3-4 Stage CV Ring", "AIC/AIV-Mix", "M>=82 Cube-bound tile sizing (e.g. 96x112), L0C->UB SPLIT_M + UB->L1 Nd2Nz, in-place K_rope->P L1 aliasing"),
    SHELL_CV_PERSISTENT_STATE: ("Persistent Chunk/Recurrent State", "AIC/AIV-Mix", "Persistent (B,H) ownership, FP32 on-chip state residency, anchor=G_tail/2 exp range halving, initC=false L0C acc, 5-step NS"),
    SHELL_AIV_STRIPED_VM: ("Striped Register-Resident VM", "AIV-only", "Horizontal W-dimension AIV striping, 100% UB-resident regs[R, W_slice], zero cross-core sync, LocalMemBar hazard tracking"),
}
_SHELL_CONTRACT_LINE_RE = re.compile(
    r"^(?://\s*Algorithmic Synthesis Contract:.*|//\s*op_name=.*unrestricted_top=.*|"
    r"//\s*atom_sequence=.*l1_alias_map=.*|//\s*-\s*invariant:.*|"
    r"//\s*===\s*(?:End\s+)?Synthesized Compute Slot.*)\n?",
    re.MULTILINE,
)
_C_TYPE_KEYWORDS = frozenset(
    {"auto", "bool", "char", "const", "double", "float", "int", "long", "short", "signed", "struct", "unsigned", "void", "volatile"}
)


def _observation_from_residual(residual: object | None) -> ObservationVector5D | None:
    """Extract an embedded 5D observation from a residual dataclass or serialized mapping."""
    raw = _read_attr_or_key(residual, "observation", None)
    if isinstance(raw, ObservationVector5D):
        return raw
    return ObservationVector5D.from_dict(raw) if isinstance(raw, Mapping) else None


def _dominant_bound_facets(dominant_bound: str) -> frozenset[str]:
    """Map a free-form dominant-bound label onto the causal facets its tokens imply."""
    return frozenset(f for f, toks in _DOMINANT_BOUND_FACET_TOKENS if any(t in dominant_bound for t in toks))


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
    if (obs := _observation_from_residual(residual)) is not None:
        relaxed = obs.compute_axis_residuals().unlock_causal_facets(mask)
        if relaxed != mask:
            return relaxed
        fb = _dominant_bound_facets(dom) | (frozenset({"sram_layout_and_aliasing"}) if (l1_overflow or ub_overflow) else frozenset())
        return relaxed.unlock(*sorted(fb)) if fb else relaxed

    def _f(obj: object | None, key: str, default: float = 0.0) -> float:
        v = _read_attr_or_key(obj, key, None)
        return default if v is None else float(v)

    def _imax(*pairs: tuple[object | None, str]) -> int:
        return max(int(_read_attr_or_key(obj, k, 0) or 0) for obj, k in pairs)

    dom_facets = _dominant_bound_facets(dom)
    spill_bytes = _imax((residual, "reg_spill_bytes"), (residual, "spill_bytes"), (compiler_summary, "reg_spill_bytes"), (compiler_summary, "spill_bytes"))
    sw_loops = _imax((compiler_summary, "software_loops_emitted"), (compiler_summary, "software_loops"))
    vloopv2 = _imax((compiler_summary, "vloopv2_loops_emitted"), (compiler_summary, "vloopv2_loops"))
    local_membar = bool(_read_attr_or_key(residual, "has_local_membar_in_vf_call", False) or _read_attr_or_key(compiler_summary, "has_local_membar_in_vf_call", False))

    unlocks: list[str] = []
    if spill_bytes > 0 or (sw_loops > 0 and vloopv2 == 0) or _f(residual, "dual_issue_ipc_collapse") >= 0.4 or "exec_mode_and_spr" in dom_facets:
        unlocks.append("exec_mode_and_spr")
    if l1_overflow or ub_overflow or _f(residual, "bank_conflict_ratio") > 0.10 or "sram_layout_and_aliasing" in dom_facets:
        unlocks.append("sram_layout_and_aliasing")
    if _f(residual, "measured_mte2_wait_ratio") > 0.30 or _f(residual, "l2_hit_rate", 1.0) < 0.70 or _imax((residual, "dcache_dcci_flushes"), (compiler_summary, "dcache_dcci_flushes")) > 0 or "dma_and_cache_hierarchy" in dom_facets:
        unlocks.append("dma_and_cache_hierarchy")
    if (
        _f(residual, "measured_sync_stall_ratio") > 0.25
        or local_membar
        or _imax((residual, "alias_serialized_buffers"), (compiler_summary, "alias_serialized_buffers")) > 0
        or _imax((residual, "vf_fusion_breaks"), (compiler_summary, "vf_fusion_breaks")) > 0
        or bool(_read_attr_or_key(residual, "pseudo_double_buffer_serialized", False))
        or "pipeline_and_sync_graph" in dom_facets
    ):
        unlocks.append("pipeline_and_sync_graph")
    if _f(residual, "wave_quantization_ratio", 1.0) > 1.25 or "grid_and_workspace_topology" in dom_facets:
        unlocks.append("grid_and_workspace_topology")
    return mask.unlock(*unlocks) if unlocks else mask


def unlock_cross_topology_paradigm_shift(mask: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
    """Unlock all 7 MECE AST freedom facets when SIL authorizes cross-topology promotion."""
    _ = mask
    return ConstraintRelaxationMask.top()


@dataclass(frozen=True)
class ComposableComputeSlotSpec(JsonRecordMixin):
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
            object.__setattr__(self, "freedom_mask", ConstraintRelaxationMask.from_dict(self.freedom_mask))
        elif not isinstance(self.freedom_mask, ConstraintRelaxationMask):
            raise TypeError("ComposableComputeSlotSpec.freedom_mask must be ConstraintRelaxationMask")
        object.__setattr__(self, "coordination_shell", coerce_topology_axes(self.coordination_shell))
        object.__setattr__(self, "atom_sequence", tuple(str(a) for a in self.atom_sequence))
        object.__setattr__(self, "l1_alias_map", {str(k): str(v) for k, v in dict(self.l1_alias_map).items()})
        object.__setattr__(self, "custom_ring_sync", str(self.custom_ring_sync))
        object.__setattr__(self, "invariants", tuple(str(inv) for inv in self.invariants))
        if not self.freedom_mask.is_unrestricted_top:
            if self.l1_alias_map and not self.freedom_mask.sram_layout_and_aliasing:
                raise ValueError("sram_layout_and_aliasing facet must be unlocked to customize l1_alias_map")
            if self.custom_ring_sync and not self.freedom_mask.pipeline_and_sync_graph:
                raise ValueError("pipeline_and_sync_graph facet must be unlocked to customize custom_ring_sync")
            if self.atom_sequence and not (
                self.freedom_mask.pipeline_and_sync_graph
                or self.freedom_mask.dma_and_cache_hierarchy
                or self.freedom_mask.exec_mode_and_spr
            ):
                raise ValueError("atom_sequence composition requires unlocking pipeline_and_sync_graph, dma_and_cache_hierarchy, or exec_mode_and_spr")

    def render_contract(self, policy: Mapping[str, Any] | None = None) -> str:
        """Render [Coordination Shell + SchedulePolicy + Compute Graph] AscendC artifact."""
        norm = normalize_schedule_policy(self.coordination_shell, policy)
        norm["freedom_mask"] = self.freedom_mask.to_dict()
        atoms_line = ", ".join(self.atom_sequence) if self.atom_sequence else "default_shell_atoms"
        alias_line = ", ".join(f"{k}->{v}" for k, v in self.l1_alias_map.items()) or "none"
        body = self.compute_body_cpp.strip() or f"// default {self.slot_name} functor body"
        header = [
            "// Algorithmic Synthesis Contract: [Coordination Shell + SchedulePolicy Knobs + Synthesized Compute Graph]",
            f"// op_name={self.op_name} | unrestricted_top={self.freedom_mask.is_unrestricted_top} | unlocked=[{','.join(self.freedom_mask.unlocked_facets())}] | shell={self.coordination_shell.key()} | slot={self.slot_name}",
            f"// atom_sequence=[{atoms_line}] | l1_alias_map={{{alias_line}}} | custom_ring_sync={self.custom_ring_sync or 'shell_default'}",
        ]
        if self.invariants:
            header.extend(f"//   - invariant: {inv}" for inv in self.invariants)
        slot_block = f"\n// === Synthesized Compute Slot ({self.slot_name}) ===\n{body}\n// === End Synthesized Compute Slot ===\n"
        out = "\n".join(header) + "\n" + slot_block
        if self.freedom_mask.is_unrestricted_top and self.compute_body_cpp.strip():
            return out
        return out + "\n" + render_template(self.coordination_shell, norm, track="ascendc").ascendc_cpp


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
        self._specs[coerce_topology_axes(spec.topology_axes)] = spec

    def list_topology_axes(self) -> tuple[TopologyAxes, ...]:
        ordered = [ax for ax in CANONICAL_TOPOLOGY_PRESETS if ax in self._specs]
        return tuple(ordered + [ax for ax in self._specs if ax not in CANONICAL_TOPOLOGY_PRESETS])

    def resolve_shell_axes(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> TopologyAxes:
        return resolve_canonical_shell_axes(topology_axes, self._specs)

    def get_spec(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> CoordinationShellSpec:
        return self._specs[self.resolve_shell_axes(topology_axes)]

    def default_policy(self, topology_axes: TopologyAxes | Mapping[str, Any]) -> dict[str, Any]:
        shell_axes = self.resolve_shell_axes(topology_axes)
        return normalize_schedule_policy(shell_axes, self._specs[shell_axes].default_policy)

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
        prim_func, pass_configs, compile_flags, metadata, ascendc_cpp = None, {}, (), {}, ""
        if track in ("tilelang", "both"):
            tl_out = spec.tilelang_builder(norm_policy)
            prim_func = tl_out["prim_func"]
            pass_configs = dict(tl_out.get("pass_configs") or {})
            compile_flags = tuple(str(f) for f in (tl_out.get("compile_flags") or ()))
            metadata = dict(tl_out.get("metadata") or {})
        if track in ("ascendc", "both"):
            ascendc_cpp = spec.ascendc_renderer(norm_policy)
        return RenderedTemplate(shell_axes, norm_policy, prim_func, pass_configs, compile_flags, ascendc_cpp, metadata)

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
        return RenderedTemplate(
            topology_axes=self.resolve_shell_axes(slot_spec.coordination_shell),
            policy=base_policy,
            ascendc_cpp=slot_spec.render_contract(base_policy),
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
        reg = TemplateRegistry()
        for axes in CANONICAL_TOPOLOGY_PRESETS:
            title, domain, desc = _SHELL_DESCRIPTIONS[axes]
            reg.register(
                CoordinationShellSpec(
                    axes, title, desc, domain,
                    TILELANG_BUILDERS[axes], ASCENDC_RENDERERS[axes], DEFAULT_POLICIES_BY_AXES[axes],
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


def _infer_entry_core_attr(body: str, *, axes: TopologyAxes | None = None) -> str:
    if axes is not None:
        return {"coupled_1c2v": "__mix__(1, 2)", "aic_only": "__cube__"}.get(axes.compute_engine, "__vector__")
    has_aic, has_aiv = "ASCEND_IS_AIC" in body, "ASCEND_IS_AIV" in body
    if has_aic and has_aiv:
        return "__mix__(1, 2)"
    if any(k in body for k in ("Mmad(", "FixpipeParams", "asc_mmad(", "ascend_gemm_l1", "asc_copy_l0c2")) and not has_aiv and "Reg::" not in body:
        return "__cube__"
    return "__vector__"


def _sanitize_c_ident(name: str, default: str = "ascend_top_kernel") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    return default if (not cleaned or cleaned[0].isdigit()) else cleaned


def _split_top_level_csv(params_text: str) -> list[str]:
    parts, cur, depth = [], [], 0
    for ch in params_text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            if item := "".join(cur).strip():
                parts.append(item)
            cur = []
            continue
        cur.append(ch)
    if tail := "".join(cur).strip():
        parts.append(tail)
    return parts


def _extract_c_param_name(param_decl: str) -> str:
    depth, eq_idx = 0, -1
    for idx, ch in enumerate(param_decl):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]" and depth > 0:
            depth -= 1
        elif ch == "=" and depth == 0:
            eq_idx = idx
            break
    lhs = re.sub(r"(?:\s*\[[^\]]*\])+\s*$", "", (param_decl[:eq_idx] if eq_idx >= 0 else param_decl).strip()).strip()
    if fn_ptr := re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", lhs):
        return fn_ptr.group(1)
    ident = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs)
    return ident.group(1) if ident and ident.group(1) not in _C_TYPE_KEYWORDS else ""


def _format_host_launcher_stub(host_signature: str | Mapping[str, Any], entry_fn: str) -> str:
    if isinstance(host_signature, Mapping):
        fn_name = _sanitize_c_ident(str(host_signature.get("name") or host_signature.get("op_name") or f"launch_{entry_fn}"), f"launch_{entry_fn}")
        param_parts, call_args = ["void* stream"], []
        for item in host_signature.get("params") or host_signature.get("args") or ():
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                ptype, pname = str(item[0]).strip(), _sanitize_c_ident(str(item[1]), "arg")
                pstr = f"{ptype} {pname}"
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
        return f'#ifndef __CCE_AICORE__\nextern "C" inline void {fn_name}({", ".join(param_parts)}) {{\n    {entry_fn}<<<1, nullptr, stream>>>({", ".join(call_args)});\n}}\n#endif'

    sig_text = str(host_signature).strip().rstrip(";").strip()
    if "(" in sig_text and ")" in sig_text:
        if "{" in sig_text:
            return f"#ifndef __CCE_AICORE__\n{sig_text}\n#endif"
        l_paren, r_paren = sig_text.find("("), sig_text.rfind(")")
        call_args, stream_expr = [], ""
        for p_item in _split_top_level_csv(sig_text[l_paren + 1 : r_paren].strip()):
            if p_item in ("void", "...") or not (pname := _extract_c_param_name(p_item)):
                continue
            if not stream_expr and (pname == "stream" or re.search(r"\b(?:aclrtStream|rtStream_t)\b", p_item)):
                stream_expr = pname
            else:
                call_args.append(pname)
        sig_decl = re.sub(r'^extern\s+"C"\s+', "", sig_text)
        stream_setup = "" if stream_expr == "stream" else f"    void* stream = {stream_expr or 'nullptr'};\n"
        return f'#ifndef __CCE_AICORE__\nextern "C" {sig_decl} {{\n{stream_setup}    {entry_fn}<<<1, nullptr, stream>>>({", ".join(call_args)});\n}}\n#endif'

    fn_name = _sanitize_c_ident(sig_text, f"launch_{entry_fn}")
    return f'#ifndef __CCE_AICORE__\nextern "C" inline void {fn_name}(void* stream, void* x, void* y) {{\n    {entry_fn}<<<1, nullptr, stream>>>(x, y);\n}}\n#endif'


def _extract_body_from_input(
    source_or_slot: object | None,
    *,
    policy: Mapping[str, Any] | object | None,
    lowered_kernel_cpp: str | None,
) -> tuple[str, str, TopologyAxes | None]:
    """Extract (raw_cpp, op_name, topology_axes) from slot, shell, PrimFunc, or C++ source."""
    op_name, axes = "ascend_top_kernel", None
    if policy is not None and (raw_axes := _read_attr_or_key(policy, "topology_axes", None)) is not None:
        with contextlib.suppress(Exception):
            axes = coerce_topology_axes(raw_axes)
    has_lowered = lowered_kernel_cpp is not None and bool(lowered_kernel_cpp.strip())
    if isinstance(source_or_slot, ComposableComputeSlotSpec):
        return (lowered_kernel_cpp if has_lowered else source_or_slot.compute_body_cpp), source_or_slot.op_name, axes or source_or_slot.coordination_shell
    if isinstance(source_or_slot, TopologyAxes):
        return (lowered_kernel_cpp if has_lowered else render_template(source_or_slot, policy, track="ascendc").ascendc_cpp), op_name, source_or_slot
    if isinstance(source_or_slot, RenderedTemplate):
        meta_name = source_or_slot.metadata.get("op_name") if source_or_slot.metadata else None
        return (lowered_kernel_cpp if has_lowered else source_or_slot.ascendc_cpp), (str(meta_name) if meta_name else op_name), axes or source_or_slot.topology_axes
    if has_lowered:
        if isinstance(source_or_slot, str) and source_or_slot.strip() and "\n" not in source_or_slot and ";" not in source_or_slot:
            op_name = source_or_slot.strip()
        return lowered_kernel_cpp or "", op_name, axes
    if source_or_slot is None:
        return "", op_name, axes
    if hasattr(source_or_slot, "kernel_source"):
        return str(getattr(source_or_slot, "kernel_source", "") or ""), op_name, axes
    if isinstance(source_or_slot, str):
        return source_or_slot, op_name, axes
    with contextlib.suppress(Exception):
        import tilelang  # noqa: PLC0415
        from tilelang import tvm  # noqa: PLC0415
        from tilelang.backend.module import create_backend_context  # noqa: PLC0415

        ctx = create_backend_context("ascend").target if tvm.target.Target.current(allow_none=True) is None else contextlib.nullcontext()
        with ctx:
            if ks := getattr(tilelang.lower(source_or_slot, target="ascend"), "kernel_source", ""):
                return str(ks), op_name, axes
    return str(source_or_slot), op_name, axes


def _looks_like_file_scope_cpp(body: str) -> bool:
    return bool(
        re.search(
            r'(?:^|\n)\s*(?:extern\s+"C"|template\s*<|(?:struct|class|enum|namespace)\s+[A-Za-z_]\w*|'
            r"(?:constexpr|static|inline|__aicore__|__global__|__simd_vf__|__simt_vf__|__no_simd_vf_fusion__)\b|"
            r"(?:void|int|uint32_t|int32_t|float|half|bfloat16_t)\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{)",
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
    """Materialize a self-contained AscendC translation unit at ConstraintRelaxationMask.top()."""
    raw_cpp, op_name, axes = _extract_body_from_input(source_or_slot, policy=policy, lowered_kernel_cpp=lowered_kernel_cpp)
    cleaned = _SHELL_CONTRACT_LINE_RE.sub("", raw_cpp).strip()
    if re.search(r'^\s*#\s*include\s+[<"]kernel_operator\.h[>"]', cleaned, re.MULTILINE):
        raise ValueError("Legacy kernel_operator.h is not allowed in dav-3510 translation units.")
    if re.search(r"^\s*using\s+namespace\s+AscendC\s*;", cleaned, re.MULTILINE):
        raise ValueError("Legacy 'using namespace AscendC;' is not allowed in dav-3510 translation units.")

    core_attr = _infer_entry_core_attr(cleaned, axes=axes)
    entry_fn = _sanitize_c_ident(f"{op_name}_kernel", "ascend_top_kernel")
    has_global_entry = bool(re.search(r"\b__global__\b", cleaned))
    if has_global_entry:
        cleaned = re.sub(
            r'(?:extern\s+"C"\s+)?__global__(?:\s+__[A-Za-z0-9_]+__(?:\([^)]*\))?)*',
            lambda m: m.group(0) if "__aicore__" in m.group(0) else m.group(0).replace("__global__", "__global__ __aicore__", 1),
            cleaned,
        )
        if not re.search(r'extern\s+"C"\s+__global__', cleaned):
            cleaned = re.sub(r"\b__global__\b", 'extern "C" __global__', cleaned, count=1)

    header_lines = [
        "// TileFoundry Top-Element Standalone AscendC Translation Unit (ConstraintRelaxationMask.top())",
        f"// op_name={op_name} | unrestricted_top=True | shell_locks=none",
    ]
    if axes is not None:
        header_lines.append(f"// topology_axes={axes.key()}")
    if "tl_templates/ascend/" not in cleaned and "c_api/asc_" not in cleaned:
        header_lines.extend(["#include <tl_templates/ascend/common.h>", "#include <c_api/asc_simd.h>"])
    parts = ["\n".join(header_lines)]

    if has_global_entry:
        if cleaned:
            parts.append(cleaned)
    elif _looks_like_file_scope_cpp(cleaned):
        if cleaned:
            parts.append(cleaned)
        if core_attr == "__cube__":
            entry_body = "    if ASCEND_IS_AIC {\n        (void)x;\n        (void)y;\n    }"
        elif core_attr.startswith("__mix__"):
            entry_body = "    if ASCEND_IS_AIC {\n        (void)x;\n    }\n    if ASCEND_IS_AIV {\n        (void)y;\n    }"
        else:
            entry_body = "    if ASCEND_IS_AIV {\n        (void)x;\n        (void)y;\n    }"
        parts.append(f'extern "C" __global__ __aicore__ {core_attr} void {entry_fn}(__gm__ float* x, __gm__ float* y) {{\n{entry_body}\n}}')
    else:
        indented_body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in cleaned.splitlines()) if cleaned else "    if ASCEND_IS_AIV {\n        (void)x;\n        (void)y;\n    }"
        parts.append(f'extern "C" __global__ __aicore__ {core_attr} void {entry_fn}(__gm__ float* x, __gm__ float* y) {{\n    (void)x;\n    (void)y;\n{indented_body}\n}}')

    if host_signature:
        parts.append(_format_host_launcher_stub(host_signature, entry_fn))
    return "\n\n".join(p for p in parts if p.strip()) + "\n"


__all__ = [
    "CANONICAL_TOPOLOGY_PRESETS", "ComposableComputeSlotSpec", "ConstraintRelaxationMask",
    "CoordinationShellSpec", "DEFAULT_POLICIES_BY_AXES", "RenderedTemplate", "SCHEDULE_POLICY_KEYS",
    "SHELL_AIC_GEMM_FIXPIPE", "SHELL_AIV_SIMD_STREAM", "SHELL_AIV_SIMT_ATOMIC", "SHELL_AIV_STRIPED_VM",
    "SHELL_CV_PERSISTENT_STATE", "SHELL_CV_RAGGED_EPILOGUE", "SHELL_CV_RING_PIPELINE", "SchedulePolicy",
    "TemplateRegistry", "TopologyAxes", "VALID_FREEDOM_FACETS", "coerce_topology_axes",
    "get_template_registry", "materialize_top_translation_unit", "normalize_schedule_policy",
    "relax_facets_for_bottleneck", "render_template", "resolve_canonical_shell_axes",
    "unlock_cross_topology_paradigm_shift",
]
