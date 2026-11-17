"""Static-dynamic residual modeling and two-lane (Exploit + Stepping-Stone) beam search."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry_ascend.agent.router.capability_gate import StructuralFeatures
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

    def to_dict(self) -> dict[str, float | bool]:
        """Return float and flag dictionary representation."""
        return {
            "measured_latency_us": self.measured_latency_us,
            "predicted_bottleneck_util": self.predicted_bottleneck_util,
            "measured_aic_mac_ratio": self.measured_aic_mac_ratio,
            "measured_aiv_vec_ratio": self.measured_aiv_vec_ratio,
            "measured_mte2_wait_ratio": self.measured_mte2_wait_ratio,
            "measured_sync_stall_ratio": self.measured_sync_stall_ratio,
            "unexplained_gap_ratio": self.unexplained_gap_ratio,
            "pseudo_double_buffer_serialized": self.pseudo_double_buffer_serialized,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StaticDynamicResidual:
        """Reconstruct StaticDynamicResidual from a serialized dictionary."""
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
        )


def compute_static_dynamic_residual(
    ledger: ObjectivePhysicalLedger,
    telemetry: Mapping[str, object],
) -> StaticDynamicResidual:
    """Compute StaticDynamicResidual from static ledger and msprof telemetry counters."""
    measured_us = float(telemetry.get("measured_latency_us", telemetry.get("dur_us", 0.0)))
    aic_mac = float(telemetry.get("aic_mac_ratio", telemetry.get("aic_mad", 0.0)))
    aiv_vec = float(telemetry.get("aiv_vec_ratio", telemetry.get("aiv_vec", 0.0)))
    mte2_wait = float(telemetry.get("mte2_wait_ratio", telemetry.get("aiv_mte2", 0.0)))
    sync_stall = float(telemetry.get("sync_stall_ratio", 0.0))

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
        aic_fix = float(telemetry.get("aic_fixpipe", 0.0))
        observed_util = max(aic_mac, aiv_vec, mte2_wait, aic_fix)
        unexplained = min(
            1.0,
            max(max(0.0, predicted_util - observed_util), sync_stall * 0.5),
        )
    elif "roofline_eta" in telemetry:
        eta = min(1.0, max(0.0, float(telemetry["roofline_eta"])))
        unexplained = min(1.0, max(max(0.0, predicted_util - eta), sync_stall * 0.5))
    else:
        unexplained = min(1.0, sync_stall * 0.5)

    steady_ii = getattr(ledger, "steady_state_ii", None)
    ii_looks_pipelined = steady_ii is None or int(steady_ii) <= 2
    pseudo_serialized = bool(
        getattr(ledger, "uses_dynamic_offset_pipelining", False)
        and ii_looks_pipelined
        and (mte2_wait >= 0.32 or unexplained >= 0.25 or sync_stall >= 0.20)
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
    )


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


def extract_structural_signature(dsl_source: str, *, track: str = "") -> tuple[str, ...]:
    """Extract sorted architectural primitive signature for stepping-stone diversity."""
    tags: list[str] = []
    if track and track != "unconstrained":
        tags.append(f"track:{track}")
    for token in _NOVELTY_TOKENS:
        if token in dsl_source:
            tags.append(token)
    try:
        tree = ast.parse(dsl_source)
    except SyntaxError:
        if "ascend_pipe_barrier" in dsl_source and "PIPE_ALL" in dsl_source:
            tags.append("pipe_all_barrier")
        if "vfma" in dsl_source.lower():
            tags.append("vfma_unshifted")
        if "partial_out" in dsl_source:
            tags.append("split_kv_multi")
        return tuple(sorted(set(tags)))

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
    "compute_static_dynamic_residual",
    "extract_structural_signature",
]
