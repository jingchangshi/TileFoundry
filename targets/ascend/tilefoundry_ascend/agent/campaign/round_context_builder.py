"""Immutable champion snapshot and per-round prompt/context synthesis builder."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    DeterministicSearchBoundaryLedger, is_clean_calibration_sample,
)
from tilefoundry_ascend.agent.campaign.helpers import (
    OperatorSemanticsCache, derive_topology_axes, dispatch_l0_gate, lookup_case_dims_safe,
    regime_mean_recoverable_excess_us, resolve_active_trackb_kernel,
    resolve_cannbench_tasks_root as _helpers_tasks_root, resolve_operator_entrypoint_safe,
    resolve_operator_semantics_for_gates, resolve_trackb_freedom_mask, run_l0_gate,
    select_active_regime, slice_contract_instruction, source_sha256,
)
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor
from tilefoundry_ascend.agent.gates.cpp_slice_protocol import (
    build_trackb_context_view, escalate_primfunc_to_trackb_tu,
)
from tilefoundry_ascend.agent.gates.gate_funnel import StageAResult, extract_ordered_tensor_shapes
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    SYMPTOM_FAMILY_TO_FACETS, AscendCausalRule, SymptomFamily,
    build_predicate_context, normalize_runtime_backend,
    rank_symptom_families, should_escalate_diagnostic_or_creative,
)
from tilefoundry_ascend.agent.npu import CandidateSurrogateResult, bottleneck_anchor_for_regime
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, sanitize_effort
from tilefoundry_ascend.agent.orchestrator.llm_constraints import build_progressive_llm_constraints
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    CREATIVE_FOCUS_HINTS, build_prescreen_summary_from_kwargs, summarize_excluded_mechanisms,
)
from tilefoundry_ascend.agent.orchestrator.regime_slice_patcher import (
    DEFAULT_READONLY_HOST_EXPORTS, _active_trackb_kernel, _case_ratio_predicate_entries,
    _compact_trackb_snippet_source, _extract_p2_focus_lines_and_diag,
    _extract_regime_abi_context, _is_real_tir_primfunc, extract_regime_snippet,
    patch_regime_into_module, resolve_regime_config, split_readonly_host_tail,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate, StaticDynamicResidual, TwoLaneBeamSearch,
    compiler_summary_from_residual, compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.orchestrator.seed_synthesizer import (
    extract_golden_entrypoint_contract, format_golden_entrypoint_prompt_block,
)
from tilefoundry_ascend.agent.router.capability_gate import evaluate_track_escalation
from tilefoundry_ascend.agent.router.isa_gap_scanner import scan_tilelang_pipe_gaps
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, CannbenchOpSpec, CannbenchOpsRegistry, PerfTargetSpec, RegimeTargetSpec,
    build_anchor_var_bindings, compute_case_aligned_eta, get_input_domain_envelope,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary, CostFeatures, build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import CostGateDecision
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver, RegimePlanResult, recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics, PyTorchSemanticAnalyzer, RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, RegimeSlice, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.templates.registry import (
    relax_facets_for_bottleneck, unlock_cross_topology_paradigm_shift,
)

logger = logging.getLogger(__name__)


class _CampaignConfigLike(Protocol):
    """Structural config subset consumed by RoundContextBuilder."""
    @property
    def operator_name(self) -> str: ...
    @property
    def pure_golden(self) -> bool: ...
    @property
    def stagnation_escalate_rounds(self) -> int: ...
    @property
    def k_rule_workers(self) -> int: ...
    @property
    def enable_tilesize_prescreen(self) -> bool: ...
    @property
    def exploit_effort(self) -> str: ...
    @property
    def allow_unreviewed_distilled_rules(self) -> bool: ...


def _format_anchor_telemetry(result: CandidateSurrogateResult, regime_tag: str) -> str:
    rows = [
        {
            "case_num": a.case_num, "regime_tag": a.regime_tag,
            "target_regime": a.regime_tag == regime_tag, "accuracy_passed": a.accuracy_passed,
            "cand_us": round(a.cand_us, 4), "ref_perf_us": round(a.ref_perf_us, 4),
            "t_hw_us": round(a.t_hw_us, 4), "speedup": round(a.speedup, 4),
            "roofline_eta": round(a.roofline_eta, 4), "eq3_score": round(a.eq3_score, 4),
        }
        for a in result.anchors
    ]
    return json.dumps({
        "candidate_id": result.candidate_id, "all_passed": result.all_passed,
        "avg_speedup": round(result.avg_speedup, 4), "avg_roofline_eta": round(result.avg_roofline_eta, 4),
        "surrogate_score": round(result.surrogate_score, 4), "anchors": rows,
    }, indent=2)


@dataclass(frozen=True)
class ChampionState:
    """Immutable snapshot of champion source, results, anchors, and telemetry caches."""

    incumbent_source: str
    best_result: CandidateSurrogateResult
    active_anchors: tuple[AnchorCaseSpec, ...]
    regime_specs: tuple[RegimeTargetSpec, ...]
    stagnation_rounds: int = 0
    stagnation_by_regime: Mapping[str, int] = field(default_factory=dict)
    compile_wall_streak_by_regime: Mapping[str, int] = field(default_factory=dict)
    last_round_promoted_by_regime: Mapping[str, bool] = field(default_factory=dict)
    last_round_promoted: bool = True
    last_local_rel_gain_by_regime: Mapping[str, float] = field(default_factory=dict)
    freedom_masks: Mapping[str, ConstraintRelaxationMask] = field(default_factory=dict)
    msprof_by_case: Mapping[tuple[str, int], Mapping[str, float]] = field(default_factory=dict)
    p2_cache: Mapping[tuple[str, str], Mapping[str, object]] = field(default_factory=dict)
    calibrated_dispatch_floor_us: float = 0.0
    boundary_ledger: DeterministicSearchBoundaryLedger | None = None
    beams: Mapping[str, TwoLaneBeamSearch] = field(default_factory=dict)
    current_regime: str | None = None
    frozen_regimes: frozenset[str] = frozenset()
    active_kernel_cache: Mapping[tuple[str, str], str | None] = field(default_factory=dict)
    live_regime_plans: Mapping[str, RegimePlanResult] = field(default_factory=dict)
    operator_proto_cache: Mapping[str, object] | None = None
    sil_advisory: object | None = None
    topology_axes: TopologyAxes | None = None
    semantic_topology_axes: TopologyAxes | None = None
    regime_blueprints: Mapping[str, HirBlueprintPacket] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundSynthesisBundle:
    """Immutable per-round synthesis inputs and worker GenerationRequests."""

    regime_spec: RegimeTargetSpec
    func_name: str
    parent_module_source: str
    parent_snippet: str
    parent_lane: str
    selected_stone_id: str | None
    selected_stone_hypothesis: str
    parent_ledger: ObjectivePhysicalLedger
    parent_residual: StaticDynamicResidual
    parent_axes: TopologyAxes
    ranked_families: tuple[tuple[SymptomFamily, float], ...]
    ranked_rules: tuple[AscendCausalRule, ...]
    worker_specs: tuple[tuple[str, str, AscendCausalRule | None, GenerationRequest], ...]
    prescreen_summary: str
    gate_decision: CostGateDecision
    live_plan: RegimePlanResult
    blueprint: HirBlueprintPacket | None
    parent_p2_diag: Mapping[str, object] | None
    track: str
    active_parent_id: str = ""
    parent_sha: str = ""
    parent_lat_us: float = 0.0
    parent_obj_score: float = 0.0
    parent_l1_summary: BishengResourceSummary | None = None
    parent_l1_summary_source: str = "absent"
    parent_features: Mapping[str, object] | None = None
    parent_workload: CostFeatures | None = None
    parent_cost_pred: object | None = None
    parent_dominant_bound: str = ""
    active_mask: ConstraintRelaxationMask | None = None
    escalated: bool = False
    paradigm_shift_active: bool = False
    p2_escalated: bool = False
    mean_recoverable_excess_us: float | None = None
    prev_local_rel_gain: float | None = None
    local_best: CandidateSurrogateResult | None = None
    creative_effort: str = "high"
    stagnation_rounds: int = 0


class RoundContextBuilder:
    """Construct immutable round context bundles, telemetry prompts, and worker requests."""

    def __init__(
        self, *, config: _CampaignConfigLike, regime_specs: Sequence[RegimeTargetSpec],
        target: AscendTarget, tracker: BayesianRuleTracker, cost_advisor: CostModelAdvisor,
        regime_solver: RegimePiecewiseSolver, source_root: Path,
        perf_target: PerfTargetSpec | None = None, env_file: Path | None = None,
        cannbench_repo: Path | None = None, enable_track_router: bool = False, run_id: str = "",
        experience_lake: ExperienceLake | None = None, experience_lake_dir: Path | None = None,
        sil_advisory: object | None = None, topology_axes: TopologyAxes | None = None,
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        freedom_masks: Mapping[str, ConstraintRelaxationMask] | None = None,
        boundary_ledger: DeterministicSearchBoundaryLedger | None = None,
        run_l0_fn: Callable[..., StageAResult] | None = None,
        l1_summary_fn: Callable[[str, RegimeTargetSpec], BishengResourceSummary | None] | None = None,
        collect_msprof_fn: Callable[..., Mapping[tuple[str, int], Mapping[str, float]] | None] | None = None,
        collect_p2_fn: Callable[..., Mapping[str, object] | None] | None = None,
        resolve_op_semantics_fn: Callable[[], PyTorchOpSemantics | None] | None = None,
        resolve_active_trackb_kernel_fn: Callable[[str, RegimeTargetSpec], str | None] | None = None,
        lookup_case_dims_fn: Callable[[int, str], dict[str, object]] | None = None,
        resolve_cannbench_tasks_root_fn: Callable[[], Path | None] | None = None,
        resolve_trackb_freedom_mask_fn: Callable[[RegimeTargetSpec], ConstraintRelaxationMask] | None = None,
    ) -> None:
        self._config, self._regime_specs, self._target = config, tuple(regime_specs), target
        self._tracker, self._cost_advisor, self._regime_solver = tracker, cost_advisor, regime_solver
        self._source_root, self._perf_target = Path(source_root).resolve(), perf_target
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._enable_track_router, self._run_id = bool(enable_track_router), run_id
        self._experience_lake, self._experience_lake_dir = experience_lake, experience_lake_dir
        self._sil_advisory, self._topology_axes = sil_advisory, topology_axes
        self._semantic_topology_axes: TopologyAxes | None = None
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = freedom_masks if isinstance(freedom_masks, dict) else dict(freedom_masks or {})
        self._boundary_ledger = boundary_ledger or DeterministicSearchBoundaryLedger()
        self._run_l0_fn, self._l1_summary_fn = run_l0_fn, l1_summary_fn
        self._collect_msprof_fn, self._collect_p2_fn = collect_msprof_fn, collect_p2_fn
        self._resolve_op_semantics_fn = resolve_op_semantics_fn
        self._resolve_active_trackb_kernel_fn = resolve_active_trackb_kernel_fn
        self._lookup_case_dims_fn = lookup_case_dims_fn
        self._resolve_cannbench_tasks_root_fn = resolve_cannbench_tasks_root_fn
        self._resolve_trackb_freedom_mask_fn = resolve_trackb_freedom_mask_fn
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        self._golden_abi_block_cache: str | None = None
        self._blueprint_provenance: dict[str, object] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._semantics_cache = OperatorSemanticsCache()
        self._update_blueprint_provenance(None)
        self._regime_blueprints: dict[str, HirBlueprintPacket] = dict(regime_blueprints) if regime_blueprints is not None else {}

    def _resolve_cannbench_tasks_root(self) -> Path | None:
        if self._resolve_cannbench_tasks_root_fn is not None:
            return self._resolve_cannbench_tasks_root_fn()
        return _helpers_tasks_root(cannbench_repo=self._cannbench_repo, env_file=self._env_file, source_root=self._source_root)

    def _update_blueprint_provenance(self, spec: CannbenchOpSpec | None) -> None:
        self._blueprint_provenance = {
            "pure_golden_mode": bool(self._config.pure_golden), "operator_name": self._config.operator_name,
            "golden_path": str(spec.golden_path) if spec is not None else "", "source_mode": "pure_golden",
        }

    def _resolve_operator_spec(self) -> CannbenchOpSpec | None:
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is None:
            return None
        reg = CannbenchOpsRegistry(tasks_root=tasks_root, source_root=self._source_root, pure_golden=bool(self._config.pure_golden))
        return reg.resolve(self._config.operator_name) if reg.has_operator(self._config.operator_name) else None

    def _validated_regime_blueprints(self, provided: Mapping[str, HirBlueprintPacket] | None) -> dict[str, HirBlueprintPacket]:
        if provided is None:
            return self._build_semantic_first_regime_blueprints()
        missing = sorted(s.regime_tag for s in self._regime_specs if s.regime_tag not in provided)
        if missing:
            raise ValueError(f"regime_blueprints missing regime tags {missing}; got {sorted(provided)}")
        try:
            spec = self._resolve_operator_spec()
        except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError):
            spec = None
        self._update_blueprint_provenance(spec)
        return dict(provided)

    def _build_semantic_first_regime_blueprints(self) -> dict[str, HirBlueprintPacket]:
        spec = self._resolve_operator_spec()
        self._update_blueprint_provenance(spec)
        if spec is None:
            return {}
        if not spec.golden_path.is_file():
            raise FileNotFoundError(f"official PyTorch golden reference missing for {spec.op_id} at {spec.golden_path}")
        try:
            golden_src = spec.golden_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read PyTorch golden reference for {spec.op_id} at {spec.golden_path}: {exc}") from exc
        analyzer = PyTorchSemanticAnalyzer(self._target)
        try:
            sem = analyzer.analyze_source(golden_src, entry_name=spec.entrypoint, proto=spec.proto)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"cannot analyze PyTorch golden semantics for {spec.op_id}: {exc}") from exc
        self._semantic_topology_axes = sem.base_axes
        packets: dict[str, HirBlueprintPacket] = {}
        for rspec in self._regime_specs:
            adims = rspec.representative_anchor_dims or rspec.representative_hir_dims
            rcase = RepresentativeCase.from_case_entry(rspec.representative_case_entry, proto=spec.proto) if rspec.representative_case_entry else None
            try:
                packets[rspec.regime_tag] = analyzer.build_universal_blueprint(sem, adims, representative_case=rcase)
            except (FileNotFoundError, OSError, SyntaxError, ValueError, KeyError) as exc:
                raise RuntimeError(f"failed to synthesize semantic-first blueprint for {spec.op_id} regime {rspec.regime_tag}: {type(exc).__name__}: {exc}") from exc
        return packets

    def _build_golden_entrypoint_abi_block(self) -> str:
        if self._golden_abi_block_cache is not None:
            return self._golden_abi_block_cache
        try:
            spec = self._resolve_operator_spec()
        except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError):
            spec = None
        if spec is None or not spec.golden_path.is_file():
            self._golden_abi_block_cache = ""
            return ""
        try:
            gsrc = spec.golden_path.read_text(encoding="utf-8")
            dtxt = spec.desc_path.read_text(encoding="utf-8") if spec.desc_path.is_file() else ""
            block = format_golden_entrypoint_prompt_block(extract_golden_entrypoint_contract(gsrc, spec.entrypoint, proto=spec.proto, desc_text=dtxt))
        except (OSError, SyntaxError, ValueError, KeyError, TypeError):
            block = ""
        self._golden_abi_block_cache = block
        return block

    def _resolve_active_trackb_kernel(self, module_source: str, regime_spec: RegimeTargetSpec) -> str | None:
        if self._resolve_active_trackb_kernel_fn is not None:
            return self._resolve_active_trackb_kernel_fn(module_source, regime_spec)
        return resolve_active_trackb_kernel(module_source, regime_spec, cache=self._active_kernel_cache)

    def _resolve_op_sem(self) -> PyTorchOpSemantics | None:
        return resolve_operator_semantics_for_gates(
            self._config.operator_name, source_root=self._source_root,
            pure_golden=bool(self._config.pure_golden), tasks_root=self._resolve_cannbench_tasks_root(),
            cannbench_repo=self._cannbench_repo, env_file=self._env_file,
            cache=self._semantics_cache, override_fn=self._resolve_op_semantics_fn,
        )

    def _exec_l0(
        self, module_source: str, regime_spec: RegimeTargetSpec, *, anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        return dispatch_l0_gate(
            module_source, regime_spec, l0_checker=self._run_l0_fn,
            default_fn=lambda ms, rs, anchor_dims_override=None: run_l0_gate(
                ms, rs, target=self._target, enable_track_router=self._enable_track_router,
                freedom_masks=self._freedom_masks, anchor_dims_override=anchor_dims_override,
                op_semantics=self._resolve_op_sem(), op_proto=self._semantics_cache.proto,
            ),
            anchor_dims_override=anchor_dims_override,
        )

    def _axes(
        self, ledger: ObjectivePhysicalLedger | None, *, features: object | None = None, route: object | None = None
    ) -> TopologyAxes:
        return derive_topology_axes(
            self._config.operator_name, ledger, topology_axes=self._topology_axes,
            semantic_topology_axes=self._semantic_topology_axes, sil_advisory=self._sil_advisory,
            features=features, route=route,
        )

    def _resolve_round_parent(
        self, *, incumbent_source: str, regime_spec: RegimeTargetSpec, regime_beam: TwoLaneBeamSearch,
        escalated: bool, sibling_specs: Sequence[RegimeTargetSpec] | None = None,
    ) -> tuple[str, str, str, BeamCandidate | None, str]:
        eff_sibs = tuple(sibling_specs) if sibling_specs is not None else self._regime_specs
        op_entry = resolve_operator_entrypoint_safe(self._config.operator_name)
        _, default_snip = extract_regime_snippet(incumbent_source, regime_spec, operator_entrypoint=op_entry, sibling_specs=eff_sibs)
        if not escalated:
            return incumbent_source, default_snip, "exploit", None, ""
        for stone in regime_beam.stepping_stone_beam:
            try:
                patched = patch_regime_into_module(
                    incumbent_source, regime_spec, stone.dsl_source, operator_entrypoint=op_entry,
                    sibling_specs=eff_sibs, active_kernel=self._resolve_active_trackb_kernel(incumbent_source, regime_spec),
                )
                l0 = self._exec_l0(patched, regime_spec)
                if not l0.passed:
                    logger.warning("Skipping invalid stepping-stone %s for regime %s: %s", stone.candidate_id, regime_spec.regime_tag, l0.error_message[:160])
                    continue
                _, hsnip = extract_regime_snippet(patched, regime_spec, operator_entrypoint=op_entry, sibling_specs=eff_sibs)
                regime_beam.mark_parent_selected(stone)
                return patched, hsnip, "stepping_stone", stone, (stone.hypothesis if stone.worker_kind == "creative_explore" else "")
            except (SyntaxError, ValueError, KeyError) as exc:
                logger.warning("Failed to patch stepping-stone slice %s for regime %s: %s", stone.candidate_id, regime_spec.regime_tag, exc)
        return incumbent_source, default_snip, "stepping_stone", None, ""

    @staticmethod
    def _format_msprof_section(
        residual: StaticDynamicResidual | None, *, p2_diag: Mapping[str, object] | None = None
    ) -> str:
        has_pmu = bool(
            residual is not None and (
                residual.measured_aic_mac_ratio > 0.0 or residual.measured_aiv_vec_ratio > 0.0
                or residual.measured_mte2_wait_ratio > 0.0 or residual.measured_sync_stall_ratio > 0.0
            )
        )
        has_haz = bool(
            residual is not None and (
                residual.spill_bytes > 0 or residual.stack_bytes > 0 or not residual.vloopv2_hit
                or residual.vf_fusion_breaks > 0 or residual.alias_serialized_buffers > 0
                or residual.dual_issue_ipc_collapse > 0.0 or residual.dcache_dcci_flushes > 0
                or residual.has_local_membar_in_vf_call or residual.scalar_compute_loop_fallback > 0.0
                or residual.dcache_bypass_narrow_store > 0.0
            )
        )
        p2_lines: list[str] = []
        if p2_diag:
            raw_spots = p2_diag.get("p2_hotspots") or p2_diag.get("source_line_hotspots") or ()
            if isinstance(raw_spots, Sequence) and not isinstance(raw_spots, (str, bytes)):
                spots = []
                for s in list(raw_spots)[:3]:
                    if isinstance(s, Mapping) or hasattr(s, "source_location"):
                        loc = str(s.get("source_location") if isinstance(s, Mapping) else getattr(s, "source_location", "?")) or "?"
                        cnt = s.get("instruction_count", 0) if isinstance(s, Mapping) else getattr(s, "instruction_count", 0)
                        sh = s.get("instruction_share") if isinstance(s, Mapping) else getattr(s, "instruction_share", None)
                        spots.append(f"{loc} (instrs={cnt}, share={float(sh):.3f})" if isinstance(sh, (int, float)) and not isinstance(sh, bool) else f"{loc} (instrs={cnt})")
                    else:
                        spots.append(str(s))
                if spots:
                    p2_lines.append("p2_top_hotspots: " + "; ".join(spots))
            wf = p2_diag.get("p2_mte_waitflag_share") if p2_diag.get("p2_mte_waitflag_share") is not None else p2_diag.get("mte_waitflag_share")
            if isinstance(wf, (int, float)) and not isinstance(wf, bool):
                p2_lines.append(f"p2_mte_waitflag_share: {float(wf):.4f}")
            ri = p2_diag.get("p2_issue_interval_us") if p2_diag.get("p2_issue_interval_us") is not None else p2_diag.get("issue_interval_us")
            if isinstance(ri, Mapping) and ri:
                parts = []
                for pipe in sorted(ri):
                    st = ri[pipe]
                    if isinstance(st, Mapping) or (hasattr(st, "p50_us") and hasattr(st, "p90_us")):
                        p50 = float(st.get("p50_us", 0.0) if isinstance(st, Mapping) else getattr(st, "p50_us", 0.0) or 0.0)
                        p90 = float(st.get("p90_us", 0.0) if isinstance(st, Mapping) else getattr(st, "p90_us", 0.0) or 0.0)
                        parts.append(f"{pipe}(p50={p50:.3f}us, p90={p90:.3f}us)")
                    elif isinstance(st, (int, float)) and not isinstance(st, bool):
                        parts.append(f"{pipe}={float(st):.4f}us")
                if parts:
                    p2_lines.append("p2_issue_interval_us: " + ", ".join(parts))
            elif isinstance(ri, (int, float)) and not isinstance(ri, bool):
                p2_lines.append(f"p2_issue_interval_us: {float(ri):.4f}")
        if not has_pmu and not has_haz and not p2_lines:
            return ""
        lines = ["### HARDWARE MSPROF TELEMETRY"]
        if has_pmu and residual is not None:
            rmap = residual.to_dict()
            lines.extend([
                f"aic_mac_ratio: {residual.measured_aic_mac_ratio:.4f}",
                f"aiv_vec_ratio: {residual.measured_aiv_vec_ratio:.4f}",
                f"mte2_wait_ratio: {residual.measured_mte2_wait_ratio:.4f}",
                f"sync_stall_ratio: {residual.measured_sync_stall_ratio:.4f}",
                f"unexplained_gap_ratio: {residual.unexplained_gap_ratio:.4f}",
            ])
            for key in ("transfer_inflation_ratio", "effective_compute_eta"):
                val = rmap.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    lines.append(f"{key}: {float(val):.4f}")
        if has_haz and residual is not None:
            lines.extend([
                f"spill_bytes: {residual.spill_bytes}", f"stack_bytes: {residual.stack_bytes}",
                f"vloopv2_hit: {residual.vloopv2_hit}", f"vf_fusion_breaks: {residual.vf_fusion_breaks}",
                f"alias_serialized_buffers: {residual.alias_serialized_buffers}",
                f"dual_issue_ipc_collapse: {residual.dual_issue_ipc_collapse:.4f}",
                f"dcache_dcci_flushes: {residual.dcache_dcci_flushes}",
                f"has_local_membar_in_vf_call: {residual.has_local_membar_in_vf_call}",
                f"scalar_compute_loop_fallback: {residual.scalar_compute_loop_fallback:.1f}",
                f"dcache_bypass_narrow_store: {residual.dcache_bypass_narrow_store:.1f}",
            ])
        lines.extend(p2_lines)
        return "\n".join(lines) + "\n\n"

    def _format_sil_live_policy_section(
        self, *, live_plan: RegimePlanResult | None, active_mask: ConstraintRelaxationMask | None,
        escalation: object | None = None, pipe_gaps: object | None = None,
        cost_pred: object | None = None, escalated_trackb_tu: str = "",
        parent_ledger: ObjectivePhysicalLedger | None = None, parent_residual: StaticDynamicResidual | None = None,
    ) -> str:
        if live_plan is None and active_mask is None:
            return ""
        lines = ["### SIL LIVE POLICY & 7-FACET CONSTRAINT ENVELOPE"]
        if live_plan is not None:
            pol = live_plan.policy
            lines.append(f"- topology_axes: {live_plan.topology_axes.key()} (solver_mode={live_plan.solver_mode}, backend_tier={pol.get('backend_tier', 'tilelang_dsl')})")
            lines.append(f"- live_solved_knobs: tile_mnk={pol.get('tile_mnk')}, row_pack_factor={pol.get('row_pack_factor', 1)}, l1_stages={pol.get('l1_stages', 1)}, ub_bytes={pol.get('ub_estimated_bytes', 0)}, l1_bytes={pol.get('l1_estimated_bytes', 0)}")
            if live_plan.notes:
                lines.append(f"- solver_note: {live_plan.notes[-1]}")
        if active_mask is not None:
            locked = active_mask.locked_facets()
            lines.append(f"- unlocked_facets: {', '.join(active_mask.unlocked_facets())}")
            lines.append(f"- locked_facets: {', '.join(locked) if locked else 'none (top)'}")
        rbound = str(self._cost_advisor.resolve_dominant_bound(parent_ledger, parent_residual, fallback_cost_pred=cost_pred) or "") if (parent_ledger or parent_residual or cost_pred is not None) else ""
        if cost_pred is not None:
            pv = float(cost_pred) if isinstance(cost_pred, (int, float)) and not isinstance(cost_pred, bool) else float(getattr(cost_pred, "pred_us", 0.0) or 0.0)
            bv = rbound or str(getattr(cost_pred, "dominant_bound", "") or "")
            lines.append(f"- cost_model_pred_us: {pv:.2f} (dominant_bound={bv}, tier1_penalty_us={float(getattr(cost_pred, 'tier1_penalty_us', 0.0) or 0.0):.2f})")
        elif rbound:
            lines.append(f"- dominant_bound: {rbound}")
        if escalation is not None:
            lines.append(f"- track_escalation: should_escalate={bool(getattr(escalation, 'should_escalate', False))}, to_tier={getattr(escalation, 'to_tier', '')}, reasons={list(getattr(escalation, 'trigger_reasons', ()) or ())}")
        if escalated_trackb_tu.strip():
            lines.extend(["- escalated_trackb_seed_cpp:", "```cpp", escalated_trackb_tu.strip(), "```"])
        if pipe_gaps is not None:
            missing = tuple(getattr(pipe_gaps, "missing_primitives", ()) or ())
            lines.append(f"- tilelang_7pipe_isa_coverage: {float(getattr(pipe_gaps, 'coverage_ratio', 1.0)):.2%}, missing_count={len(missing)}, sample_gaps={list(missing[:6])}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _format_trackb_hotpath_section(regime_spec: RegimeTargetSpec, active_kernel: str | None = None) -> tuple[str, str]:
        backend = normalize_runtime_backend(regime_spec.runtime_backend)
        esc_tier = "ascendc_template" if backend in ("ascendc_trackb", "hybrid_trackb") else "tilelang_dsl"
        if not regime_spec.trackb_symbols:
            return esc_tier, ""
        aline = f"- active_kernel: {active_kernel}\n" if active_kernel else ""
        sig_rule = (
            "- _render_trackb_tu is editable: keep its callable ABI ``_render_trackb_tu(**dims)`` "
            "and the launcher dims it receives consistent with the launched kernel.\n"
            if "_render_trackb_tu" in regime_spec.editable_symbols
            else "- _render_trackb_tu is immutable: keep the launched kernel's ``template <...>`` "
            "and parameter signature unchanged so instantiation still binds.\n"
        )
        section = (
            "### ACTIVE RUNTIME HOT PATH\n"
            f"- runtime_backend: {regime_spec.runtime_backend}\n"
            f"- active_trackb_symbols: {list(regime_spec.trackb_symbols)}\n{aline}"
            "- CRITICAL: On NPU this regime executes the Track-B CCE (C++20) kernel string "
            "and config selector shown in TARGET FUNCTION SNIPPET, NOT dead TileLang fallback "
            "builders. Edit the active Track-B CCE (C++20) kernel string / selector directly.\n\n"
            "### TRACK-B REPLY CONTRACT\n"
            "- Reply with `# target_bottleneck:` and `# structural_hypothesis:` comments followed by one fenced block in one of these forms:\n"
            "  1. Complete bare C++ unit per changed function (preferred): either the ``__global__`` kernel or one ``__simd_vf__`` helper, keeping its exact name and signature.\n"
            "  2. SEARCH/REPLACE hunks with ``<<<<<<< SEARCH [Lstart-Lend]`` line anchors against the launched kernel and the helpers it calls.\n"
            "  3. A Python slice that reassigns one authorized Track-B C++ constant in full.\n"
            "- Never echo ``[READONLY_SIBLING_DECLARATION]`` or read-only helper markers.\n"
            "- A full C++ constant must still contain every sibling ``__global__`` kernel present in the original string; a replacement that drops one is rejected.\n"
            f"{sig_rule}- Put any selector change outside the hunks as a Python ``_select_*_config`` definition in the same fenced block.\n\n"
        )
        return esc_tier, section

    @staticmethod
    def _prepare_prompt_snippet_and_tail(
        *, snippet_source: str, incumbent_source: str, regime_spec: RegimeTargetSpec,
        op_entry: str, active_kernel: str | None, focus_lines: Sequence[int], diag_text: str,
    ) -> tuple[str, str, str]:
        if regime_spec.trackb_symbols:
            comp = _compact_trackb_snippet_source(
                snippet_source, regime_spec, incumbent_source=incumbent_source,
                active_kernel=active_kernel, focus_lines=focus_lines, diagnostic_text=diag_text,
            )
            return comp, "", ""
        prot = DEFAULT_READONLY_HOST_EXPORTS
        if op_entry and op_entry not in prot and op_entry not in regime_spec.editable_symbols:
            prot = (*DEFAULT_READONLY_HOST_EXPORTS, op_entry)
        editable, tail = split_readonly_host_tail(snippet_source, prot)
        if not tail and incumbent_source.strip():
            _, tail = split_readonly_host_tail(incumbent_source, prot)
        fenced = editable if editable.strip() else snippet_source
        tail_sec = f"<read_only_host_tail>\n{tail}\n</read_only_host_tail>\n\n" if tail else ""
        tir_sec = (
            "### TIR SCOPE CONTRACT\n- Never define a nested `def` inside `@T.prim_func`; keep all helpers at module top level.\n"
            "- Do not redefine or alter `<read_only_host_tail>` bindings.\n\n"
        )
        return fenced, tail_sec, tir_sec

    def _build_exploit_regime_request(
        self, *, snippet_source: str, func_name: str, regime_spec: RegimeTargetSpec,
        rule: AscendCausalRule, best_result: CandidateSurrogateResult,
        blueprint: HirBlueprintPacket | None = None, parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None, track: str = "unconstrained",
        incumbent_source: str = "", sil_policy_section: str = "", effort: str | None = None,
        p2_diag: Mapping[str, object] | None = None,
    ) -> GenerationRequest:
        """Build a rule-exploit GenerationRequest using header comments and a fenced block."""
        mod_src = incumbent_source or snippet_source
        akernel = self._resolve_active_trackb_kernel(mod_src, regime_spec)
        flines, dtext = _extract_p2_focus_lines_and_diag(p2_diag)
        op_entry = resolve_operator_entrypoint_safe(str(getattr(self._config, "operator_name", "") or ""))
        fenced, tail_sec, tir_sec = self._prepare_prompt_snippet_and_tail(
            snippet_source=snippet_source, incumbent_source=incumbent_source, regime_spec=regime_spec,
            op_entry=op_entry, active_kernel=akernel, focus_lines=flines, diag_text=dtext,
        )
        sdesc = slice_contract_instruction(func_name, regime_spec, operator_entrypoint=op_entry)
        esc_tier, hot_sec = self._format_trackb_hotpath_section(regime_spec, active_kernel=akernel)
        reply_tail = "ONLY the reply form required by the TRACK-B REPLY CONTRACT." if regime_spec.trackb_symbols else "ONLY Python code for this regime slice (never nest `def` inside `@T.prim_func`)."
        sys_prompt = (
            "You are an Ascend 950PR kernel performance engineer. Apply the selected causal optimization rule cleanly "
            f"while preserving exact numerical semantics, dynamic shape compatibility, and {sdesc}. "
            "Begin with two single-line header comments:\n# target_bottleneck: <bottleneck metric>\n"
            f"# structural_hypothesis: <concise optimization hypothesis>\nfollowed by one fenced code block containing {reply_tail}"
        )
        ctext = build_progressive_llm_constraints(
            escalation_tier=esc_tier, track=track, enable_track_router=bool(self._enable_track_router),
            sil_advisory=self._sil_advisory, freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint, current_ledger=parent_ledger, topology_axes=self._axes(parent_ledger),
            active_symptom=rule.symptom_family.value if hasattr(rule.symptom_family, "value") else str(rule.symptom_family),
        )
        op_name = str(getattr(self._config, "operator_name", "") or getattr(self._config, "operator", "") or "")
        dom = get_input_domain_envelope(op_name)
        gblock = self._build_golden_entrypoint_abi_block()
        abi_sec = _extract_regime_abi_context(incumbent_source, regime_spec, golden_abi_block=gblock) if incumbent_source else gblock
        eff_rule = rule.render_for_backend(regime_spec.runtime_backend) if hasattr(rule, "render_for_backend") else rule
        bblock = self._boundary_ledger.format_regime_boundary_prompt(regime_spec.regime_tag)
        uprompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n{hot_sec}{tir_sec}{ctext}\n\n"
            f"{sil_policy_section}{f'{bblock}\n\n' if bblock else ''}"
            f"### SELECTED CAUSAL RULE\nRule ID: {eff_rule.rule_id}\nPrecondition: {eff_rule.precondition}\n"
            f"Transform Template:\n{eff_rule.transform_template.strip()}\nExpected Postcondition: {eff_rule.expected_postcondition}\n\n"
            f"{self._format_msprof_section(parent_residual, p2_diag=p2_diag)}"
            f"{f'{dom.format_prompt_block()}\n\n' if dom is not None else ''}"
            f"### SURROGATE ANCHOR TELEMETRY\n{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_sec}### TARGET FUNCTION SNIPPET\n```python\n{fenced}\n```\n\n{tail_sec}"
        )
        eff = sanitize_effort(effort if effort is not None else getattr(self._config, "exploit_effort", "medium"), default="medium")
        return GenerationRequest(role="exploit_rule", system_prompt=sys_prompt, user_prompt=uprompt, effort=eff)

    def _build_creative_regime_request(
        self, *, snippet_source: str, func_name: str, regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult, effort: str, excluded_mechanisms: Sequence[str] = (),
        focus_hint: str = "", prescreen_summary: str = "", blueprint: HirBlueprintPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None, parent_residual: StaticDynamicResidual | None = None,
        track: str = "unconstrained", incumbent_source: str = "", parent_hypothesis: str = "",
        sil_policy_section: str = "", p2_diag: Mapping[str, object] | None = None, paradigm_shift: bool = False,
    ) -> GenerationRequest:
        """Build a creative GenerationRequest using header comments and a fenced block."""
        mod_src = incumbent_source or snippet_source
        akernel = self._resolve_active_trackb_kernel(mod_src, regime_spec)
        flines, dtext = _extract_p2_focus_lines_and_diag(p2_diag)
        op_entry = resolve_operator_entrypoint_safe(str(getattr(self._config, "operator_name", "") or ""))
        fenced, tail_sec, tir_sec = self._prepare_prompt_snippet_and_tail(
            snippet_source=snippet_source, incumbent_source=incumbent_source, regime_spec=regime_spec,
            op_entry=op_entry, active_kernel=akernel, focus_lines=flines, diag_text=dtext,
        )
        sdesc = slice_contract_instruction(func_name, regime_spec, operator_entrypoint=op_entry)
        esc_tier, hot_sec = self._format_trackb_hotpath_section(regime_spec, active_kernel=akernel)
        reply_desc = (
            "the TRACK-B REPLY CONTRACT form (SEARCH/REPLACE hunks, one bare C++ unit per changed function, or a Python slice reassigning one authorized constant)"
            if regime_spec.trackb_symbols
            else "Python code for this regime slice only (never nest `def` inside `@T.prim_func`)"
        )
        sys_prompt = (
            "You are an Ascend 950PR (dav-3510) systems architect. Propose an architectural optimization grounded in "
            "Roofline limits (56 AIV / 28 MixCore blocks, 3 KiB SSBUF, L0C->UB direct path, UB->L1 direct path, 64-elem UB alignment, Split-KV parallelism). "
            f"Preserve exact numerical semantics, signatures, and {sdesc}. Respond ONLY with two header comments:\n"
            "# target_bottleneck: <bottleneck metric>\n# structural_hypothesis: <concise architectural hypothesis>\n"
            f"followed by a native fenced code block containing {reply_desc}."
        )
        clines = "\n".join(f"- {i}" for i in excluded_mechanisms) if excluded_mechanisms else "- None recorded yet; explore any high-impact structural axis."
        neg_sec = f"### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)\n{clines}\nPropose a structurally orthogonal transformation that does NOT duplicate the mechanisms listed above.\n\n"
        ctext = build_progressive_llm_constraints(
            escalation_tier=esc_tier, track=track, enable_track_router=bool(self._enable_track_router),
            sil_advisory=self._sil_advisory, freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint, current_ledger=parent_ledger, excluded_mechanisms=excluded_mechanisms,
            topology_axes=self._axes(parent_ledger),
        )
        op_name = str(getattr(self._config, "operator_name", "") or getattr(self._config, "operator", "") or "")
        dom = get_input_domain_envelope(op_name)
        gblock = self._build_golden_entrypoint_abi_block()
        abi_sec = _extract_regime_abi_context(incumbent_source, regime_spec, golden_abi_block=gblock) if incumbent_source else gblock
        shift_sec = (
            "### PARADIGM SHIFT CROSS-TOPOLOGY UNLOCK\nCross-topology paradigm shift active (full_slice_rewrite authorized): "
            "CostModel prescreen is disengaged; explore coupled_1c2v AIC+AIV chunked GEMM, multi-stage GM workspace, or multi-core tiling across the regime slice.\n\n"
            if paradigm_shift else ""
        )
        bblock = self._boundary_ledger.format_regime_boundary_prompt(regime_spec.regime_tag)
        uprompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n{shift_sec}{hot_sec}{tir_sec}{ctext}\n\n"
            f"{sil_policy_section}{f'{bblock}\n\n' if bblock else ''}"
            f"{f'### ARCHITECTURAL FOCUS HINT\n{focus_hint}\n\n' if focus_hint else ''}"
            f"{f'### STEPPING-STONE PARENT HYPOTHESIS\n{parent_hypothesis}\n\n' if parent_hypothesis else ''}"
            f"{neg_sec}{f'### TILE PRESCREEN GUIDANCE\n{prescreen_summary}\n\n' if prescreen_summary else ''}"
            f"{self._format_msprof_section(parent_residual, p2_diag=p2_diag)}"
            f"{f'{dom.format_prompt_block()}\n\n' if dom is not None else ''}"
            f"### SURROGATE ANCHOR TELEMETRY\n{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_sec}### TARGET FUNCTION SNIPPET\n```python\n{fenced}\n```\n\n{tail_sec}"
        )
        return GenerationRequest(role="explore_creative", system_prompt=sys_prompt, user_prompt=uprompt, effort=effort, paradigm_shift=bool(paradigm_shift))

    def build_round_bundle(
        self, state: ChampionState, *, round_idx: int, regime_spec: RegimeTargetSpec | None = None,
        working_anchors: Sequence[AnchorCaseSpec] | None = None, exclude_regime: str | None = None,
        creative_effort: str = "high",
    ) -> RoundSynthesisBundle:
        """Assemble the complete RoundSynthesisBundle from an immutable ChampionState."""
        if state.boundary_ledger is not None:
            self._boundary_ledger = state.boundary_ledger
        if state.sil_advisory is not None:
            self._sil_advisory = state.sil_advisory
        if state.topology_axes is not None:
            self._topology_axes = state.topology_axes
        if state.semantic_topology_axes is not None:
            self._semantic_topology_axes = state.semantic_topology_axes
        if state.operator_proto_cache is not None:
            self._semantics_cache.proto = state.operator_proto_cache
        if state.regime_blueprints:
            self._regime_blueprints = dict(state.regime_blueprints)
        for k, v in state.freedom_masks.items():
            self._freedom_masks.setdefault(k, v)
        self._active_kernel_cache.update(state.active_kernel_cache)
        eff_specs = state.regime_specs or self._regime_specs
        eff_anchors = tuple(working_anchors if working_anchors is not None else state.active_anchors)
        selected_spec = regime_spec or select_active_regime(
            eff_specs, self._perf_target, state.current_regime, set(state.frozen_regimes),
            state.best_result, eff_anchors, allow_satisfied_fallback=True, exclude_regime=exclude_regime,
            stagnation_by_regime=state.stagnation_by_regime, compile_wall_streak_by_regime=state.compile_wall_streak_by_regime,
            last_round_promoted_by_regime=state.last_round_promoted_by_regime,
            last_round_promoted=state.last_round_promoted, dispatch_floor_us=state.calibrated_dispatch_floor_us,
        )
        if selected_spec is None:
            raise RuntimeError("No active unfrozen regime available for round synthesis.")
        op_entry = resolve_operator_entrypoint_safe(self._config.operator_name)
        func_name, _ = extract_regime_snippet(state.incumbent_source, selected_spec, operator_entrypoint=op_entry, sibling_specs=eff_specs)
        rbeam = state.beams.get(selected_spec.regime_tag) or TwoLaneBeamSearch()
        stag = int(state.stagnation_by_regime.get(selected_spec.regime_tag, state.stagnation_rounds))
        escalated = stag >= int(self._config.stagnation_escalate_rounds)
        parent_src, parent_snip, parent_lane, stone, stone_hyp = self._resolve_round_parent(
            incumbent_source=state.incumbent_source, regime_spec=selected_spec,
            regime_beam=rbeam, escalated=escalated, sibling_specs=eff_specs,
        )
        stone_id = stone.candidate_id if stone is not None else None
        active_pid = stone_id or (rbeam.exploit_beam[0].candidate_id if rbeam.exploit_beam else f"incumbent_r{max(0, round_idx - 1)}")
        bn_anchor = bottleneck_anchor_for_regime(state.best_result, selected_spec.regime_tag)
        bn_dims = None
        if bn_anchor is not None:
            rbn = lookup_case_dims_safe(
                self._config.operator_name, bn_anchor.case_num, regime_tag=selected_spec.regime_tag,
                regime_specs=self._regime_specs, cache=self._case_dims_cache, cannbench_repo=self._cannbench_repo,
                env_file=self._env_file, override_lookup_fn=self._lookup_case_dims_fn,
            )
            if rbn:
                bn_dims = {str(k): int(v) for k, v in rbn.items() if isinstance(v, int) and not isinstance(v, bool)}
        base_sa = self._exec_l0(parent_src, selected_spec, anchor_dims_override=bn_dims or None)
        pledger = base_sa.ledger if base_sa.ledger is not None else (stone.ledger if stone else None)
        if pledger is None:
            raise RuntimeError(f"Stage-A L0 produced no ledger for regime {selected_spec.regime_tag!r}: {base_sa.error_message}")
        ptrack = ((stone.track if stone and stone.track is not None else (base_sa.route.track if base_sa.route else "unconstrained")) if self._enable_track_router else "unconstrained")
        pfeat = base_sa.features if self._enable_track_router else None
        rbest = state.best_result.for_regime(selected_spec.regime_tag)
        local_best = rbest if rbest.anchors else state.best_result
        prev_rel = state.last_local_rel_gain_by_regime.get(selected_spec.regime_tag)
        mean_rec = regime_mean_recoverable_excess_us(state.best_result, selected_spec, dispatch_floor_us=state.calibrated_dispatch_floor_us)
        paxes = self._axes(pledger, features=pfeat, route=base_sa.route)
        bkey = f"{selected_spec.regime_tag}:{paxes.key()}"
        shist = self._boundary_ledger.spearman_history.get(bkey) or []
        clean_n = 0
        if self._experience_lake is not None:
            try:
                _, all_obs = self._experience_lake.load_records()
                clean_n = sum(1 for o in all_obs if o.operator == self._config.operator_name and o.regime_tag == selected_spec.regime_tag and o.accuracy_passed and is_clean_calibration_sample(o.burst_batch_cv))
            except (OSError, ValueError):
                clean_n = 0
        gdec = self._cost_advisor.evaluate_round_policy(
            enable_prescreen_flag=bool(self._config.enable_tilesize_prescreen),
            bucket_clean_samples=clean_n, out_of_sample_spearman_rho=shist[-1] if shist else None,
        )
        if self._boundary_ledger.is_cost_prescreen_quarantined(bkey):
            gdec = replace(gdec, allow_candidate_pruning=False, reason="spearman_rank_inverted")
        rep_dims = {str(k): int(v) for k, v in (selected_spec.representative_anchor_dims or selected_spec.representative_hir_dims).items() if isinstance(v, int) and not isinstance(v, bool)}
        bp = self._regime_blueprints.get(selected_spec.regime_tag)
        bp_rf = bp.roofline if bp is not None else None
        sil_reg = next((r for r in (getattr(self._sil_advisory, "regimes", ()) or ()) if getattr(r, "regime_tag", None) == selected_spec.regime_tag), None)
        if sil_reg is None:
            sil_reg = RegimeSlice(regime_tag=selected_spec.regime_tag, case_nums=(), representative_dims=rep_dims, has_unaligned_tail=("unaligned" in selected_spec.regime_tag or "tail" in selected_spec.regime_tag))
        adv_dt = tuple(getattr(self._sil_advisory, "input_dtypes", ()) or ())
        if not adv_dt:
            rcd = selected_spec.representative_case_entry.get("dtype")
            adv_dt = (rcd.strip(),) if isinstance(rcd, str) and rcd.strip() else (tuple(str(i).strip() for i in rcd if str(i).strip()) if isinstance(rcd, (list, tuple)) else ())
        live_plan = recompute_regime_for_live_buffers(
            self._regime_solver, sil_reg, pledger, op_name=self._config.operator_name, topology_axes=paxes,
            roofline=bp_rf, input_dtypes=adv_dt or ("float16",), has_nan_inf_cases=bool(getattr(self._sil_advisory, "has_nan_inf_cases", False)),
            has_indirect_gather=(paxes.iteration_regularity == "irregular_simt"),
            has_recurrence=(paxes.memory_ring in ("persistent_onchip_state", "bidirectional_cv_ring")), uses_cube=paxes.uses_cube,
        )
        msprof_by_case = {(str(s), int(c)): dict(m) for (s, c), m in state.msprof_by_case.items()}
        if stag >= 1 and self._collect_msprof_fn is not None:
            extra_mp = self._collect_msprof_fn(parent_src, state.best_result, eff_anchors, [selected_spec])
            if isinstance(extra_mp, Mapping):
                for k, m in extra_mp.items():
                    if isinstance(k, tuple) and len(k) == 2 and isinstance(m, Mapping):
                        msprof_by_case[(str(k[0]), int(k[1]))] = {str(mk): float(mv) for mk, mv in m.items() if isinstance(mv, (int, float)) and not isinstance(mv, bool)}
        psha = source_sha256(parent_src)
        cached_mp = msprof_by_case.get((psha, int(bn_anchor.case_num))) if bn_anchor is not None else None
        plat = stone.latency_us if (stone is not None and stone.latency_us > 0.0) else local_best.mean_latency_us()
        pscore = float(stone.objective_score if (stone is not None and stone.objective_score is not None) else local_best.surrogate_score)
        pl1 = compiler_summary_from_residual(stone.residual) if stone is not None else (self._l1_summary_fn(parent_src, selected_spec) if self._l1_summary_fn else None)
        pl1_src = "stepping_stone_residual" if (stone is not None and pl1 is not None) else ("bisheng_l1" if pl1 is not None else "absent")
        raw_dur = float(cached_mp.get("dur_us", 0.0) or 0.0) if cached_mp else 0.0
        cdur = raw_dur if raw_dur > 0.0 else (float(bn_anchor.cand_us) if bn_anchor is not None and bn_anchor.cand_us > 0.0 else plat)
        same_roof = bn_anchor is not None and bn_anchor.t_hw_us > 0.0 and cdur > 0.0
        eta_case = compute_case_aligned_eta(bn_anchor, cdur) if (same_roof and bn_anchor) else (float(bn_anchor.roofline_eta) if bn_anchor else float(local_best.avg_roofline_eta))
        tel_pay: dict[str, object] = {"dur_us": cdur, "roofline_eta": eta_case}
        if bn_anchor is not None:
            tel_pay["anchor_case_idx"] = int(bn_anchor.case_num)
            if bn_anchor.t_hw_us > 0.0:
                tel_pay["case_t_hw_us"] = float(bn_anchor.t_hw_us)
        if same_roof:
            tel_pay["effective_compute_eta"] = eta_case
        if cached_mp:
            for k, v in cached_mp.items():
                if not ((k == "dur_us" and float(v) <= 0.0) or (k in ("roofline_eta", "effective_compute_eta") and same_roof)):
                    tel_pay[str(k)] = v
            pres = compute_static_dynamic_residual(pledger, tel_pay, compiler_summary=pl1, roofline=bp_rf)
        elif stone is not None and any([
            stone.residual.measured_aiv_vec_ratio > 0.0, stone.residual.measured_aic_mac_ratio > 0.0,
            stone.residual.measured_mte2_wait_ratio > 0.0, stone.residual.measured_sync_stall_ratio > 0.0,
            stone.residual.spill_bytes > 0, stone.residual.stack_bytes > 0, not stone.residual.vloopv2_hit,
            stone.residual.vf_fusion_breaks > 0, stone.residual.alias_serialized_buffers > 0,
            stone.residual.dual_issue_ipc_collapse > 0.0, stone.residual.dcache_dcci_flushes > 0,
            bool(stone.residual.has_local_membar_in_vf_call), stone.residual.scalar_compute_loop_fallback > 0.0,
            stone.residual.dcache_bypass_narrow_store > 0.0,
        ]):
            pres = stone.residual
        else:
            pres = compute_static_dynamic_residual(pledger, tel_pay, compiler_summary=pl1, roofline=bp_rf)
        ctx = build_predicate_context(pledger, features=pfeat, residual=pres, compiler_summary=pl1, roofline=bp_rf, extra={"topology_axes": paxes, **_case_ratio_predicate_entries(selected_spec.representative_case_ratios)})
        if not gdec.engaged:
            ctx["use_uncalibrated_analytical"] = False
            if pl1 is None and not cached_mp:
                ctx["scalar_compute_loop_fallback"] = 0.0
                if isinstance(ctx.get("extra_signals"), dict):
                    ctx["extra_signals"].pop("scalar_compute_loop_fallback", None)
        rfams = tuple(rank_symptom_families(ctx))
        p2_esc = should_escalate_diagnostic_or_creative(list(rfams), stagnation_rounds=stag)
        p2_diag = state.p2_cache.get((psha, selected_spec.regime_tag))
        if p2_esc and p2_diag is None and self._collect_p2_fn is not None:
            p2_diag = self._collect_p2_fn(parent_src, state.best_result, eff_anchors, selected_spec)
        pcomp = compiler_summary_from_residual(pres)
        wls = getattr(pledger, "waterlines", {}) or {}
        l1_ov = bool(wls.get("L1") is not None and int(getattr(wls["L1"], "overflow_bytes", 0) or 0) > 0)
        ub_ov = bool(wls.get("UB") is not None and int(getattr(wls["UB"], "overflow_bytes", 0) or 0) > 0)
        pwl = build_cost_features_from_ledger(pledger, live_plan=live_plan, axes=paxes, blueprint=bp, sil_advisory=self._sil_advisory, target=self._target)
        if self._experience_lake_dir is not None:
            try:
                self._tracker.sync_from_experience_lake(self._experience_lake_dir, exclude_run_id=self._run_id, allow_unreviewed_distilled_rules=self._config.allow_unreviewed_distilled_rules)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round KB sync skipped: %s", exc)
            try:
                self._cost_advisor.sync_experience_lake(self._experience_lake_dir, persist_calibration=False, exclude_run_id=self._run_id)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round cost replay skipped: %s", exc)
        pcost = self._cost_advisor.predict_candidate_us(pwl, bisheng_res=pcomp)
        pbound = str(self._cost_advisor.resolve_dominant_bound(pledger, pres, fallback_cost_pred=pcost) or "")
        op_sem = self._resolve_op_sem()
        has_rec = bool(getattr(getattr(bp, "semantics", None), "has_associative_outer_recurrence", False) or getattr(op_sem, "has_associative_outer_recurrence", False))
        pshift = bool(stag >= 2 and float(eta_case if bn_anchor else local_best.avg_roofline_eta) < 0.15 and has_rec and paxes.compute_engine == "aiv_only")
        amask = (
            self._resolve_trackb_freedom_mask_fn(selected_spec)
            if self._resolve_trackb_freedom_mask_fn is not None
            else resolve_trackb_freedom_mask(selected_spec, freedom_masks=self._freedom_masks, baseline_source=state.incumbent_source)
        )
        esc_dec, pgaps, esc_tu = None, None, ""
        if escalated:
            amask = relax_facets_for_bottleneck(amask, dominant_bound=pbound, residual=pres, compiler_summary=pcomp, l1_overflow=l1_ov, ub_overflow=ub_ov)
            for fam, _ in rfams:
                if not self._tracker.is_family_tabued(regime_tag=selected_spec.regime_tag, symptom_family=fam, parent_sha=psha, current_round=round_idx):
                    if SYMPTOM_FAMILY_TO_FACETS.get(fam):
                        amask = amask.unlock(*SYMPTOM_FAMILY_TO_FACETS[fam])
                    break
            self._freedom_masks[selected_spec.regime_tag] = amask
            ctier = "ascendc_template" if normalize_runtime_backend(selected_spec.runtime_backend) in ("ascendc_trackb", "hybrid_trackb") else normalize_runtime_backend(str(live_plan.policy.get("backend_tier", "tilelang_dsl")))
            ctier = "ascendc_template" if ctier in ("ascendc_trackb", "hybrid_trackb") else (ctier if ctier == "tilelang_dsl" else "tilelang_dsl")
            sf = dict(pfeat) if isinstance(pfeat, Mapping) else {}
            sf.update({"l1_overflow": l1_ov, "ub_overflow": ub_ov, "freedom_mask": amask})
            sf.setdefault("uses_cube", paxes.uses_cube)
            if selected_spec.representative_case_ratios is not None:
                cr = selected_spec.representative_case_ratios
                sf.update({"representative_case_id": cr.case_id, "inner_contiguous_bytes": cr.inner_contiguous_bytes, "persistent_wave_occupancy": cr.wave_occupancy_ratio})
            esc_dec = evaluate_track_escalation(
                current_tier=ctier, roofline_eta=float(eta_case if bn_anchor else local_best.avg_roofline_eta),
                target_eta=float(self._perf_target.target_roofline_eta) if self._perf_target else 0.85,
                stagnation_rounds=int(stag), dominant_bound=pbound, compiler_summary=pcomp, structural_features=sf,
            )
            pgaps = scan_tilelang_pipe_gaps()
            if esc_dec and bool(getattr(esc_dec, "should_escalate", False)) and str(getattr(esc_dec, "to_tier", "")) == "ascendc_template" and _is_real_tir_primfunc(base_sa.prim_func):
                try:
                    raw_tu = escalate_primfunc_to_trackb_tu(base_sa.prim_func)
                    esc_tu = build_trackb_context_view(raw_tu, regime_tag=selected_spec.regime_tag, module_source=parent_src, active_kernel=_active_trackb_kernel(raw_tu, raw_tu, regime_tag=selected_spec.regime_tag))
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Track-B PrimFunc escalation lowering skipped: %s", exc)
        else:
            self._freedom_masks.setdefault(selected_spec.regime_tag, amask)
        if pshift:
            amask = unlock_cross_topology_paradigm_shift(amask)
            self._freedom_masks[selected_spec.regime_tag] = amask
        sil_sec = self._format_sil_live_policy_section(
            live_plan=live_plan, active_mask=amask, escalation=esc_dec, pipe_gaps=pgaps,
            cost_pred=pcost, escalated_trackb_tu=esc_tu, parent_ledger=pledger, parent_residual=pres,
        )
        rrules = tuple(self._tracker.rank_applicable_rules(
            ctx, top_k=self._config.k_rule_workers, topology_axes=paxes,
            operator_name=self._config.operator_name, runtime_backend=selected_spec.runtime_backend,
            regime_tag=selected_spec.regime_tag, parent_sha=psha, current_round=round_idx, symptom_ranking=list(rfams),
        ))
        wspecs: list[tuple[str, str, AscendCausalRule | None, GenerationRequest]] = [
            (
                f"r{round_idx}_rule_{idx}", "rule_exploit", rule,
                self._build_exploit_regime_request(
                    snippet_source=parent_snip, func_name=func_name, regime_spec=selected_spec, rule=rule,
                    best_result=local_best, blueprint=bp, parent_ledger=pledger, parent_residual=pres,
                    track=ptrack, incumbent_source=parent_src, sil_policy_section=sil_sec,
                    effort=self._config.exploit_effort, p2_diag=p2_diag,
                ),
            )
            for idx, rule in enumerate(rrules[: self._config.k_rule_workers])
        ]
        excl = summarize_excluded_mechanisms(list(rrules), rbeam.exploited_signatures)
        ps_sum = ""
        if gdec.engaged and gdec.allow_prompt_guidance:
            adims = selected_spec.representative_anchor_dims or selected_spec.representative_hir_dims
            pk, pf, ppc, pcf = dict(selected_spec.sample_builder_kwargs), func_name, {}, ()
            try:
                rc = resolve_regime_config(parent_src, selected_spec, dims=adims)
                pk.update(rc.builder_kwargs)
                pf, ppc, pcf = rc.builder_name, dict(rc.pass_configs), rc.compile_flags
            except (SyntaxError, AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Failed to resolve prescreen config for regime %s: %s", selected_spec.regime_tag, exc)
            ps_sum = build_prescreen_summary_from_kwargs(
                pk, dsl_source=parent_src, factory_name=pf, var_bindings=build_anchor_var_bindings(adims, kwargs=pk) or None,
                config_selector_name=selected_spec.config_selector_fn or None, dims=adims, pass_configs=ppc or None,
                compile_flags=pcf or None, cost_model=self._cost_advisor.underlying_cost_model, axes=paxes,
                live_plan=live_plan, blueprint=bp, sil_advisory=self._sil_advisory,
                ordered_tensor_shapes=extract_ordered_tensor_shapes(case_entry=selected_spec.representative_case_entry, dims=adims, proto=self._semantics_cache.proto),
            ) or str(self._cost_advisor.build_prompt_guidance(decision=gdec, base_cfg=pk, dims=adims, regime_tag=selected_spec.regime_tag, top_k=4) or "")
        deficit = max(0, self._config.k_rule_workers - len(rrules))
        tot_cr = max(2, 1 + deficit) if (pshift or p2_esc) else (1 + deficit)
        for ci in range(tot_cr):
            wspecs.append((
                f"r{round_idx}_creative" if ci == 0 else f"r{round_idx}_creative_{ci}", "creative_explore", None,
                self._build_creative_regime_request(
                    snippet_source=parent_snip, func_name=func_name, regime_spec=selected_spec, best_result=local_best,
                    effort=creative_effort, excluded_mechanisms=excl,
                    focus_hint=CREATIVE_FOCUS_HINTS[ci % len(CREATIVE_FOCUS_HINTS)] if tot_cr > 1 else "",
                    prescreen_summary=ps_sum, blueprint=bp, parent_ledger=pledger, parent_residual=pres,
                    track=ptrack, incumbent_source=parent_src, parent_hypothesis=stone_hyp,
                    sil_policy_section=sil_sec, p2_diag=p2_diag, paradigm_shift=pshift,
                ),
            ))
        return RoundSynthesisBundle(
            regime_spec=selected_spec, func_name=func_name, parent_module_source=parent_src, parent_snippet=parent_snip,
            parent_lane=parent_lane, selected_stone_id=stone_id, selected_stone_hypothesis=stone_hyp,
            parent_ledger=pledger, parent_residual=pres, parent_axes=paxes, ranked_families=rfams, ranked_rules=rrules,
            worker_specs=tuple(wspecs), prescreen_summary=ps_sum, gate_decision=gdec, live_plan=live_plan,
            blueprint=bp, parent_p2_diag=p2_diag, track=ptrack, active_parent_id=active_pid, parent_sha=psha,
            parent_lat_us=plat, parent_obj_score=pscore, parent_l1_summary=pl1, parent_l1_summary_source=pl1_src,
            parent_features=pfeat if isinstance(pfeat, Mapping) else None, parent_workload=pwl,
            parent_cost_pred=pcost, parent_dominant_bound=pbound, active_mask=amask, escalated=escalated,
            paradigm_shift_active=pshift, p2_escalated=p2_esc, mean_recoverable_excess_us=mean_rec,
            prev_local_rel_gain=prev_rel, local_best=local_best, creative_effort=creative_effort, stagnation_rounds=stag,
        )


__all__ = ["ChampionState", "RoundContextBuilder", "RoundSynthesisBundle"]
