"""Static-dynamic residual modeling and two-lane (Exploit + Stepping-Stone) beam search."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tilefoundry_ascend.agent.specs.latency_algebra import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, TopologyAxes

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

_HW_TOKENS = (
    "SetL2CacheHint",
    "CACHE_MODE_DISABLE",
    "LocalMemBar",
    "DATA_BLOCK_COPY",
    "FixpipeParamsArch3510",
    "__simd_vf__",
    "__simt_vf__",
    "__launch_bounds__",
)

_SYNC_NOTIFY_EVENT_RE = re.compile(
    r"asc_sync_notify\s*\([^,]+,[^,]+,\s*((?:[^()]+|\([^()]*\))+)\)"
)
_CPP_CUBE_MMAD_RE = re.compile(
    r"\b(?:asc_mmad(?:_mx|_sparse)?|ascend_gemm_l1)\s*(?:<[^;{}()]*>)?\s*\("
)


def compiler_summary_from_residual(
    residual: StaticDynamicResidual,
) -> BishengResourceSummary | None:
    """Recover the genuine L1 compiler summary captured in a residual's observation."""
    observation = residual.observation
    if observation is None or not observation.has_l1_compiler_summary:
        return None
    extra_signals: dict[str, float] = {}
    if residual.scalar_compute_loop_fallback > 0.0:
        extra_signals["scalar_compute_loop_fallback"] = float(residual.scalar_compute_loop_fallback)
    if residual.dcache_bypass_narrow_store > 0.0:
        extra_signals["dcache_bypass_narrow_store"] = float(residual.dcache_bypass_narrow_store)
    return BishengResourceSummary(
        vec_regs=observation.onchip_sram_reg.l1_vec_regs,
        scalar_regs=observation.onchip_sram_reg.l1_scalar_regs,
        spill_bytes=observation.onchip_sram_reg.l1_spill_bytes,
        stack_bytes=max(observation.onchip_sram_reg.l1_stack_bytes, residual.stack_bytes),
        vloopv2_loops=observation.compute_issue.l1_vloopv2_loops,
        software_loops=observation.compute_issue.l1_software_loops,
        has_local_membar_in_vf_call=observation.pipeline_sync.l1_local_membar_in_vf,
        vf_fusion_breaks=observation.compute_issue.l1_vf_fusion_breaks,
        alias_serialized_buffers=observation.pipeline_sync.l1_alias_serialized_buffers,
        dual_issue_ipc_collapse=observation.compute_issue.l1_dual_issue_collapse,
        dcache_dcci_flushes=observation.offchip_dma.l1_dcache_dcci_flushes,
        extra_signals=extra_signals,
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
    _ = spill_bytes
    if measured_latency_us <= 0.0:
        return 0.0
    eff_bank = max(0.0, min(1.0, float(bank_conflict_ratio)))
    eff_l2 = max(0.0, min(1.0, float(l2_hit_rate)))
    if residual is not None:
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

    bank_penalty = 1.0 + eff_bank * 0.30
    l2_penalty = 1.0 + (1.0 - eff_l2) * 0.15
    return round(eq3 / (bank_penalty * l2_penalty), 6)


def _is_int_gt_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 1
    )


def _has_static_pingpong_buffers(tree: ast.AST, dsl_source: str) -> bool:
    if "enable_offset=True" in dsl_source:
        return False
    shape_counts: dict[str, int] = {}
    two_slot_vars: set[str] = set()
    for stmt in ast.walk(tree):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
            continue
        call = stmt.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr not in ("alloc_shared", "alloc_ub", "alloc_fragment")
            or not call.args
        ):
            continue
        shape_node = call.args[0]
        shape_key = ast.dump(shape_node)
        names = (
            [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if isinstance(stmt, ast.Assign)
            else ([stmt.target.id] if isinstance(stmt.target, ast.Name) else [])
        )
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + max(1, len(names))
        if (
            isinstance(shape_node, (ast.Tuple, ast.List))
            and len(shape_node.elts) >= 2
            and isinstance(shape_node.elts[0], ast.Constant)
            and shape_node.elts[0].value == 2
        ):
            two_slot_vars.update(names)
    if any(count >= 2 for count in shape_counts.values()):
        return True
    if not two_slot_vars:
        return False
    seen_slots: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Subscript)
            or not isinstance(node.value, ast.Name)
            or node.value.id not in two_slot_vars
        ):
            continue
        lead_slice = (
            node.slice.elts[0]
            if isinstance(node.slice, ast.Tuple) and node.slice.elts
            else node.slice
        )
        if (
            isinstance(lead_slice, ast.Constant)
            and isinstance(lead_slice.value, int)
            and lead_slice.value in (0, 1)
        ):
            seen_slots.setdefault(node.value.id, set()).add(lead_slice.value)
    return any(len(slots) >= 2 for slots in seen_slots.values())


def _has_loop_fission_multi_pass(tree: ast.AST) -> bool:
    containers = (ast.FunctionDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor)
    return any(
        isinstance(node, containers) and sum(1 for stmt in node.body if isinstance(stmt, ast.For)) >= 2
        for node in ast.walk(tree)
    )


def _has_pipe_all_barrier(tree: ast.AST, dsl_source: str) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("ascend_pipe_barrier", "pipe_barrier")
            and any(isinstance(arg, ast.Constant) and arg.value == "PIPE_ALL" for arg in node.args)
        ):
            return True
    return "ascend_pipe_barrier" in dsl_source and "PIPE_ALL" in dsl_source


def _has_vfma_unshifted(tree: ast.AST, dsl_source: str) -> bool:
    if "vfma" in dsl_source.lower():
        return True
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("fma", "vfma")
        ):
            return True
        if isinstance(node, ast.For):
            has_exp = False
            has_fma_expr = False
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    if inner.func.attr in ("exp", "exp2"):
                        has_exp = True
                    elif inner.func.attr in ("fma", "vfma"):
                        has_fma_expr = True
                elif (
                    (isinstance(inner, ast.AugAssign) and isinstance(inner.op, ast.Add))
                    or (
                        isinstance(inner, ast.Assign)
                        and isinstance(inner.value, ast.BinOp)
                        and isinstance(inner.value.op, ast.Add)
                    )
                ):
                    if any(
                        isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult)
                        for sub in ast.walk(inner.value)
                    ):
                        has_fma_expr = True
            if has_fma_expr and not has_exp:
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


def extract_cpp_trackb_signature_tags(cpp_source: str) -> set[str]:
    """Extract structural mechanism tags from tl_templates/ascend C++20 kernel bodies."""
    tags: set[str] = set()
    if _CPP_CUBE_MMAD_RE.search(cpp_source):
        tags.add("cpp:cube_mmad")
    for needle, tag in (
        ("asc_copy_gm2l1_nd2nz(", "cpp:gm2l1_nd2nz"),
        ("asc_copy_l0c2ub(", "cpp:l0c2ub_ring"),
        ("simd_inst::vsts_pack_b32", "cpp:vsts_pack_b32"),
    ):
        if needle in cpp_source:
            tags.add(tag)
    if "asc_lock(" in cpp_source and "asc_unlock(" in cpp_source:
        tags.add("cpp:slot_lock_pipeline")
    events = {ev.strip() for ev in _SYNC_NOTIFY_EVENT_RE.findall(cpp_source) if ev.strip()}
    if len(events) >= 2 or any("%" in ev or "& 1" in ev or "&1" in ev for ev in events):
        tags.add("cpp:pingpong_event_parity")
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
    tags.extend(tok for tok in (*_NOVELTY_TOKENS, *_HW_TOKENS) if tok in dsl_source)

    comp = getattr(stage_b, "compiler_summary", None) if stage_b is not None else None
    sources = [s for s in (residual, comp) if s is not None]
    eff_spill = max((int(getattr(s, "spill_bytes", 0) or 0) for s in sources), default=0)
    if stage_b is not None:
        eff_spill = max(eff_spill, int(getattr(stage_b, "spill_bytes", 0) or 0))
    eff_vf_breaks = max((int(getattr(s, "vf_fusion_breaks", 0) or 0) for s in sources), default=0)
    eff_alias_ser = max(
        (int(getattr(s, "alias_serialized_buffers", 0) or 0) for s in sources), default=0
    )
    eff_ipc_collapse = max(
        (float(getattr(s, "dual_issue_ipc_collapse", 0.0) or 0.0) for s in sources), default=0.0
    )
    eff_dcci = max((int(getattr(s, "dcache_dcci_flushes", 0) or 0) for s in sources), default=0)
    eff_local_membar = any(bool(getattr(s, "has_local_membar_in_vf_call", False)) for s in sources)
    eff_vloop_miss = (
        (residual is not None and residual.vloopv2_hit is False)
        or (stage_b is not None and getattr(stage_b, "vloopv2_hit", True) is False)
        or (
            comp is not None
            and int(getattr(comp, "software_loops", 0) or 0) > 0
            and int(getattr(comp, "vloopv2_loops", 0) or 0) == 0
        )
    )
    if residual is not None:
        if residual.bank_conflict_ratio > 0.05:
            tags.append("hw:bank_conflict")
        if residual.l2_hit_rate < 0.70:
            tags.append("hw:low_l2_hit")
    for cond, hw_tag in (
        (eff_vloop_miss, "hw:vloopv2_miss"),
        (eff_spill > 0, "hw:register_spill"),
        (eff_vf_breaks > 0, "hw:vf_fusion_break"),
        (eff_alias_ser > 0, "hw:alias_serialized"),
        (eff_ipc_collapse > 0.0, "hw:ipc_collapse"),
        (eff_dcci > 0, "hw:dcache_dcci"),
        (eff_local_membar, "hw:local_membar_vf"),
    ):
        if cond:
            tags.append(hw_tag)

    tree = ast.parse(dsl_source)
    for node in tree.body:
        val_node = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
            cpp_text = val_node.value
            if any(
                tok in cpp_text
                for tok in ("tl_templates/ascend/", "__aicore__", "__global__", "__simd_vf__")
            ):
                tags.extend(extract_cpp_trackb_signature_tags(cpp_text))

    for pred, ast_tag in (
        (_has_static_pingpong_buffers(tree, dsl_source), "static_pingpong_2buf"),
        (_has_loop_fission_multi_pass(tree), "loop_fission_multi_pass"),
        (_has_pipe_all_barrier(tree, dsl_source), "pipe_all_barrier"),
        (_has_vfma_unshifted(tree, dsl_source), "vfma_unshifted"),
        (_has_split_kv_multi(tree, dsl_source), "split_kv_multi"),
    ):
        if pred:
            tags.append(ast_tag)
    return tuple(sorted(set(tags)))


@dataclass(frozen=True)
class BeamCandidate(JsonRecordMixin):
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
    features: dict[str, object] | None = None
    topology_axes: TopologyAxes | None = None
    case_ratios: CasePhysicsRatios | None = None
    objective_score: float | None = None
    case_latencies_us: dict[int, float] = field(default_factory=dict)
    policy: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track:
            object.__setattr__(self, "track", "unconstrained")
        if self.features is not None:
            if not isinstance(self.features, Mapping):
                raise TypeError(
                    f"BeamCandidate.features must be Mapping or None, "
                    f"got {type(self.features).__name__}"
                )
            object.__setattr__(self, "features", {str(k): v for k, v in self.features.items()})
        for field_name, cls in (("topology_axes", TopologyAxes), ("case_ratios", CasePhysicsRatios)):
            val = getattr(self, field_name)
            if val is None or isinstance(val, cls):
                continue
            if isinstance(val, Mapping):
                object.__setattr__(self, field_name, cls.from_dict(val))
            else:
                raise TypeError(
                    f"BeamCandidate.{field_name} must be {cls.__name__} or None, "
                    f"got {type(val).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        out = super().to_dict()
        out["case_latencies_us"] = {
            int(k): float(v) for k, v in self.case_latencies_us.items()
        }
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BeamCandidate:
        """Reconstruct a BeamCandidate from a serialized dictionary."""
        cleaned = dict(payload)
        raw_lats = cleaned.get("case_latencies_us")
        clean_lats: dict[int, float] = {}
        if isinstance(raw_lats, Mapping):
            for k, v in raw_lats.items():
                if isinstance(k, bool) or isinstance(v, bool):
                    continue
                try:
                    clean_lats[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        cleaned["case_latencies_us"] = clean_lats
        if not isinstance(cleaned.get("features"), Mapping):
            cleaned["features"] = None
        if not isinstance(cleaned.get("policy"), Mapping):
            cleaned["policy"] = {}
        return super().from_dict(cleaned)


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

        exploit_pool = {c.candidate_id: c for c in (*self.exploit_beam, *valid)}
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

        deduped_stones: dict[tuple[str, ...], BeamCandidate] = {}
        for cand in (*self.stepping_stone_beam, *valid):
            if cand.candidate_id in exploit_ids:
                continue
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
        merged = {c.candidate_id: c for c in (*self.exploit_beam, *self.stepping_stone_beam)}
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
