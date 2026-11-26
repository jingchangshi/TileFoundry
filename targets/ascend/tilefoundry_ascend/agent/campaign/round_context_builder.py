"""Immutable champion snapshot and per-round prompt/context synthesis builder."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, TypeVar

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpSpec,
    CannbenchOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    _has_flat_operator_layout,
    build_anchor_var_bindings,
    get_input_domain_envelope,
    lookup_case_dims,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    DeterministicSearchBoundaryLedger,
    is_clean_calibration_sample,
)
from tilefoundry_ascend.agent.campaign.latency_algebra import (
    AnchorConservationPoint,
    is_regime_saturated,
    regime_bandit_ucb_us,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import to_conservation_point
from tilefoundry_ascend.agent.campaign.regime_slice_patcher import (
    DEFAULT_READONLY_HOST_EXPORTS,
    _active_trackb_kernel,
    _case_ratio_predicate_entries,
    _compact_trackb_snippet_source,
    _extract_p2_focus_lines_and_diag,
    _extract_regime_abi_context,
    _extract_trackb_cpp_source,
    _has_prim_func_builder,
    _is_real_tir_primfunc,
    _ordered_top_level_trackb_kernels,
    _render_trackb_tu_from_module,
    _resolve_operator_entrypoint_safe,
    _trackb_kernel_names,
    extract_regime_snippet,
    patch_regime_into_module,
    resolve_regime_config,
    split_readonly_host_tail,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    extract_golden_entrypoint_contract,
    format_golden_entrypoint_prompt_block,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    bottleneck_anchor_for_regime,
    extract_ordered_tensor_shapes,
)
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    SYMPTOM_FAMILY_TO_FACETS,
    AscendCausalRule,
    SymptomFamily,
    build_predicate_context,
    normalize_runtime_backend,
    rank_symptom_families,
    should_escalate_diagnostic_or_creative,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    build_trackb_context_view,
    escalate_primfunc_to_trackb_tu,
    select_active_trackb_kernel,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    evaluate_ascendc_candidate,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    sanitize_effort,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    CREATIVE_FOCUS_HINTS,
    build_prescreen_summary_from_kwargs,
    summarize_excluded_mechanisms,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    TwoLaneBeamSearch,
    compiler_summary_from_residual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.router.capability_gate import evaluate_track_escalation
from tilefoundry_ascend.agent.router.isa_gap_scanner import scan_tilelang_pipe_gaps
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    CostFeatures,
    build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import CostGateDecision
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver,
    RegimePlanResult,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import (
    ConstraintRelaxationMask,
    RegimeSlice,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.templates.registry import (
    relax_facets_for_bottleneck,
    unlock_cross_topology_paradigm_shift,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class _CampaignConfigLike(Protocol):
    """Structural config subset consumed by RoundContextBuilder."""

    @property
    def operator_name(self) -> str:
        """Operator identifier."""
        ...

    @property
    def pure_golden(self) -> bool:
        """Whether pure-golden mode is enabled."""
        ...

    @property
    def stagnation_escalate_rounds(self) -> int:
        """Round threshold before escalating to stepping-stone lane."""
        ...

    @property
    def k_rule_workers(self) -> int:
        """Number of rule-exploit workers per round."""
        ...

    @property
    def enable_tilesize_prescreen(self) -> bool:
        """Whether tile-size prescreen guidance is enabled."""
        ...

    @property
    def exploit_effort(self) -> str:
        """Effort level for exploit workers."""
        ...

    @property
    def allow_unreviewed_distilled_rules(self) -> bool:
        """Whether unreviewed distilled rules may sync from ExperienceLake."""
        ...


def _format_anchor_telemetry(
    result: CandidateSurrogateResult,
    regime_tag: str,
) -> str:
    """Serialize surrogate anchor telemetry rows as formatted JSON."""
    rows = []
    for anchor in result.anchors:
        rows.append(
            {
                "case_num": anchor.case_num,
                "regime_tag": anchor.regime_tag,
                "target_regime": anchor.regime_tag == regime_tag,
                "accuracy_passed": anchor.accuracy_passed,
                "cand_us": round(anchor.cand_us, 4),
                "ref_perf_us": round(anchor.ref_perf_us, 4),
                "t_hw_us": round(anchor.t_hw_us, 4),
                "speedup": round(anchor.speedup, 4),
                "roofline_eta": round(anchor.roofline_eta, 4),
                "headroom": round(max(0.0, 1.0 - anchor.roofline_eta), 4),
            }
        )
    return json.dumps(
        {
            "operator": result.operator_name,
            "avg_speedup": round(result.avg_speedup, 4),
            "min_case_speedup": round(result.min_case_speedup, 4),
            "avg_roofline_eta": round(result.avg_roofline_eta, 4),
            "surrogate_score": round(result.surrogate_score, 4),
            "anchors": rows,
        },
        indent=2,
    )


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
        self,
        *,
        config: _CampaignConfigLike,
        regime_specs: Sequence[RegimeTargetSpec],
        target: AscendTarget,
        tracker: BayesianRuleTracker,
        cost_advisor: CostModelAdvisor,
        regime_solver: RegimePiecewiseSolver,
        source_root: Path,
        perf_target: PerfTargetSpec | None = None,
        env_file: Path | None = None,
        cannbench_repo: Path | None = None,
        enable_track_router: bool = False,
        run_id: str = "",
        experience_lake: ExperienceLake | None = None,
        experience_lake_dir: Path | None = None,
        sil_advisory: object | None = None,
        topology_axes: TopologyAxes | None = None,
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        freedom_masks: Mapping[str, ConstraintRelaxationMask] | None = None,
        boundary_ledger: DeterministicSearchBoundaryLedger | None = None,
        run_l0_fn: Callable[..., StageAResult] | None = None,
        l1_summary_fn: Callable[[str, RegimeTargetSpec], BishengResourceSummary | None]
        | None = None,
        collect_msprof_fn: Callable[
            [str, CandidateSurrogateResult, Sequence[AnchorCaseSpec], Sequence[RegimeTargetSpec]],
            Mapping[tuple[str, int], Mapping[str, float]] | None,
        ]
        | None = None,
        collect_p2_fn: Callable[
            [str, CandidateSurrogateResult, Sequence[AnchorCaseSpec], RegimeTargetSpec],
            Mapping[str, object] | None,
        ]
        | None = None,
        resolve_op_semantics_fn: Callable[[], PyTorchOpSemantics | None] | None = None,
        resolve_active_trackb_kernel_fn: Callable[[str, RegimeTargetSpec], str | None]
        | None = None,
    ) -> None:
        self._config = config
        self._regime_specs = tuple(regime_specs)
        self._target = target
        self._tracker = tracker
        self._cost_advisor = cost_advisor
        self._regime_solver = regime_solver
        self._source_root = Path(source_root).resolve()
        self._perf_target = perf_target
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._enable_track_router = bool(enable_track_router)
        self._run_id = run_id
        self._experience_lake = experience_lake
        self._experience_lake_dir = experience_lake_dir
        self._sil_advisory = sil_advisory
        self._topology_axes = topology_axes
        self._semantic_topology_axes: TopologyAxes | None = None
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = dict(freedom_masks or {})
        self._boundary_ledger = boundary_ledger or DeterministicSearchBoundaryLedger()
        self._run_l0_fn = run_l0_fn
        self._l1_summary_fn = l1_summary_fn
        self._collect_msprof_fn = collect_msprof_fn
        self._collect_p2_fn = collect_p2_fn
        self._resolve_op_semantics_fn = resolve_op_semantics_fn
        self._resolve_active_trackb_kernel_fn = resolve_active_trackb_kernel_fn

        self._case_dims_cache: dict[int, dict[str, object]] = {}
        self._golden_abi_block_cache: str | None = None
        self._blueprint_provenance: dict[str, object] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._operator_proto_cache: Mapping[str, object] | None = None
        self._operator_semantics_cache: PyTorchOpSemantics | None = None
        self._operator_semantics_loaded: bool = False
        self._calibrated_dispatch_floor_us: float = 0.0
        self._stagnation_by_regime: dict[str, int] = {}
        self._compile_wall_streak_by_regime: dict[str, int] = {}
        self._last_round_promoted: bool = True
        self._last_round_promoted_by_regime: dict[str, bool] = {}

        self._update_blueprint_provenance(None)
        self._regime_blueprints: dict[str, HirBlueprintPacket] = (
            dict(regime_blueprints) if regime_blueprints is not None else {}
        )

    @staticmethod
    def _source_sha256(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def _invoke_with_optional_kwargs(
        runner: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> _T:
        """Pass only the optional keywords an injected runner actually accepts."""
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return runner(*args)
        accepts_any = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        accepted = {
            name: value
            for name, value in kwargs.items()
            if accepts_any or name in signature.parameters
        }
        return runner(*args, **accepted)

    def _resolve_cannbench_tasks_root(self) -> Path | None:
        """Resolve the cann-bench tasks root, falling back to the in-repo mirror."""
        try:
            repo = resolve_cannbench_repo(
                cannbench_repo=self._cannbench_repo,
                env_file=self._env_file,
            )
        except FileNotFoundError:
            repo = None
        if repo is not None:
            return resolve_cannbench_tasks_root(repo)
        in_repo = self._source_root / "tests" / "ascend" / "benchmarks"
        if _has_flat_operator_layout(in_repo):
            return in_repo
        return None

    def _update_blueprint_provenance(self, spec: CannbenchOpSpec | None) -> None:
        pure_mode = bool(self._config.pure_golden)
        self._blueprint_provenance = {
            "pure_golden_mode": pure_mode,
            "operator_name": self._config.operator_name,
            "golden_path": str(spec.golden_path) if spec is not None else "",
            "source_mode": "pure_golden",
        }

    def _resolve_operator_spec(self) -> CannbenchOpSpec | None:
        """Resolve the official cann-bench task spec for the campaign operator."""
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is None:
            return None
        registry = CannbenchOpsRegistry(
            tasks_root=tasks_root,
            source_root=self._source_root,
            pure_golden=bool(self._config.pure_golden),
        )
        if not registry.has_operator(self._config.operator_name):
            return None
        return registry.resolve(self._config.operator_name)

    def _validated_regime_blueprints(
        self,
        provided: Mapping[str, HirBlueprintPacket] | None,
    ) -> dict[str, HirBlueprintPacket]:
        """Validate caller-supplied per-regime blueprints or build semantic-first packets."""
        if provided is None:
            return self._build_semantic_first_regime_blueprints()
        missing = sorted(
            spec.regime_tag for spec in self._regime_specs if spec.regime_tag not in provided
        )
        if missing:
            raise ValueError(
                f"regime_blueprints missing regime tags {missing}; got {sorted(provided)}"
            )
        try:
            spec = self._resolve_operator_spec()
        except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError):
            spec = None
        self._update_blueprint_provenance(spec)
        return dict(provided)

    def _build_semantic_first_regime_blueprints(self) -> dict[str, HirBlueprintPacket]:
        """Build per-regime blueprints from PyTorch golden semantics and anchor dims."""
        spec = self._resolve_operator_spec()
        self._update_blueprint_provenance(spec)
        if spec is None:
            return {}
        if not spec.golden_path.is_file():
            raise FileNotFoundError(
                f"official PyTorch golden reference missing for {spec.op_id} at "
                f"{spec.golden_path}"
            )
        try:
            golden_source = spec.golden_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"cannot read PyTorch golden reference for {spec.op_id} at "
                f"{spec.golden_path}: {exc}"
            ) from exc
        analyzer = PyTorchSemanticAnalyzer(self._target)
        try:
            semantics = analyzer.analyze_source(
                golden_source,
                entry_name=spec.entrypoint,
                proto=spec.proto,
            )
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(
                f"cannot analyze PyTorch golden semantics for {spec.op_id}: {exc}"
            ) from exc
        self._semantic_topology_axes = semantics.base_axes
        packets: dict[str, HirBlueprintPacket] = {}
        for regime_spec in self._regime_specs:
            anchor_dims = (
                regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
            )
            rep_case = (
                RepresentativeCase.from_case_entry(
                    regime_spec.representative_case_entry,
                    proto=spec.proto,
                )
                if regime_spec.representative_case_entry
                else None
            )
            try:
                packets[regime_spec.regime_tag] = analyzer.build_universal_blueprint(
                    semantics,
                    anchor_dims,
                    representative_case=rep_case,
                )
            except (FileNotFoundError, OSError, SyntaxError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"failed to synthesize semantic-first blueprint for {spec.op_id} "
                    f"regime {regime_spec.regime_tag}: {type(exc).__name__}: {exc}"
                ) from exc
        return packets

    def _build_golden_entrypoint_abi_block(self) -> str:
        """Resolve and cache the ParamBuilder-checked golden entrypoint ABI prompt block."""
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
            golden_source = spec.golden_path.read_text(encoding="utf-8")
            desc_text = (
                spec.desc_path.read_text(encoding="utf-8") if spec.desc_path.is_file() else ""
            )
            contract = extract_golden_entrypoint_contract(
                golden_source,
                spec.entrypoint,
                proto=spec.proto,
                desc_text=desc_text,
            )
            block = format_golden_entrypoint_prompt_block(contract)
        except (OSError, SyntaxError, ValueError, KeyError, TypeError):
            block = ""
        self._golden_abi_block_cache = block
        return block

    def _compute_residual_with_roofline(
        self,
        ledger: ObjectivePhysicalLedger,
        telemetry: Mapping[str, object],
        *,
        compiler_summary: BishengResourceSummary | None = None,
        roofline: HirRooflineSummary | None = None,
    ) -> StaticDynamicResidual:
        """Compute StaticDynamicResidual while forwarding Roofline."""
        return compute_static_dynamic_residual(
            ledger,
            telemetry,
            compiler_summary=compiler_summary,
            roofline=roofline,
        )

    def _observation_with_roofline(
        self,
        ledger: ObjectivePhysicalLedger,
        *,
        compiler_summary: BishengResourceSummary | None = None,
        telemetry: Mapping[str, object] | None = None,
        roofline: HirRooflineSummary | None = None,
    ) -> ObservationVector5D:
        """Build ObservationVector5D while forwarding Roofline."""
        return ledger.to_observation_5d(
            compiler_summary=compiler_summary,
            telemetry=telemetry,
            roofline=roofline,
        )

    def _roofline_arg_for_solver(
        self,
        roofline: HirRooflineSummary | None,
    ) -> HirRooflineSummary | None:
        """Forward HirRooflineSummary directly to the regime solver."""
        return roofline

    def _resolve_active_trackb_kernel(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> str | None:
        """Resolve launched Track-B kernel from instantiation truth or raw TU."""
        if self._resolve_active_trackb_kernel_fn is not None:
            return self._resolve_active_trackb_kernel_fn(module_source, regime_spec)
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return None
        extracted = _extract_trackb_cpp_source(module_source, regime_spec)
        if extracted is None:
            return None
        cache_key = (self._source_sha256(module_source), regime_spec.regime_tag)
        if cache_key not in self._active_kernel_cache:
            rendered = _render_trackb_tu_from_module(
                module_source,
                extracted[1],
                regime_spec,
                require_renderer=False,
            )
            if rendered == extracted[1]:
                fallback = select_active_trackb_kernel(
                    extracted[1],
                    regime_tag=regime_spec.regime_tag,
                    module_source=module_source,
                )
                if fallback is None:
                    raise ValueError(
                        f"Track-B source for regime {regime_spec.regime_tag!r} "
                        "contains no __global__ kernel"
                    )
                self._active_kernel_cache[cache_key] = fallback
            else:
                try:
                    self._active_kernel_cache[cache_key] = _active_trackb_kernel(
                        rendered,
                        extracted[1],
                        regime_tag=regime_spec.regime_tag,
                        module_source=module_source,
                    )
                except ValueError as exc:
                    if "no unique top-level launch target" not in str(exc):
                        raise
                    kernels = _trackb_kernel_names(extracted[1])
                    ordered = _ordered_top_level_trackb_kernels(rendered, extracted[1], kernels)
                    if not ordered:
                        raise
                    tagged = select_active_trackb_kernel(
                        extracted[1],
                        regime_tag=regime_spec.regime_tag,
                        module_source=module_source,
                    )
                    self._active_kernel_cache[cache_key] = (
                        tagged if tagged in ordered else ordered[0]
                    )
        return self._active_kernel_cache[cache_key]

    def _resolve_operator_semantics_for_gates(self) -> PyTorchOpSemantics | None:
        if self._resolve_op_semantics_fn is not None:
            return self._resolve_op_semantics_fn()
        if self._operator_semantics_loaded:
            return self._operator_semantics_cache
        self._operator_semantics_loaded = True
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is None:
            return None
        try:
            spec = CannbenchOpsRegistry(
                tasks_root=tasks_root,
                source_root=self._source_root,
                pure_golden=bool(self._config.pure_golden),
            ).resolve(self._config.operator_name)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        self._operator_proto_cache = spec.proto if isinstance(spec.proto, Mapping) else None
        if not spec.golden_path.is_file():
            return None
        try:
            golden_src = spec.golden_path.read_text(encoding="utf-8")
            self._operator_semantics_cache = PyTorchSemanticAnalyzer().analyze_source(
                golden_src,
                entry_name=spec.entrypoint,
                proto=spec.proto,
            )
        except (OSError, SyntaxError, ValueError):
            self._operator_semantics_cache = None
        return self._operator_semantics_cache

    def _run_l0(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        if self._run_l0_fn is not None:
            return self._invoke_with_optional_kwargs(
                self._run_l0_fn,
                module_source,
                regime_spec,
                anchor_dims_override=anchor_dims_override,
            )
        op_semantics = self._resolve_operator_semantics_for_gates()
        req_dma = bool(op_semantics is not None and not op_semantics.has_indirect_index)
        req_cube = bool(op_semantics is not None and op_semantics.has_gemm_contraction)
        is_hybrid = regime_spec.runtime_backend == "hybrid_trackb"
        has_prim_func = _has_prim_func_builder(module_source, regime_spec.builder_candidates)
        tb_extracted = (
            _extract_trackb_cpp_source(module_source, regime_spec)
            if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            else None
        )
        mask = self._freedom_masks.get(regime_spec.regime_tag, ConstraintRelaxationMask.bottom())
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_extracted is not None:
            instantiated_tu = _render_trackb_tu_from_module(
                module_source,
                tb_extracted[1],
                regime_spec,
                require_renderer=False,
                anchor_dims_override=anchor_dims_override,
            )
            return evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
                require_hardware_dma=req_dma,
                require_cube_engine=req_cube,
                module_source=module_source,
                regime_tag=regime_spec.regime_tag,
            )
        anchor_dims = dict(
            anchor_dims_override
            or regime_spec.representative_anchor_dims
            or regime_spec.representative_hir_dims
        )
        resolved = resolve_regime_config(module_source, regime_spec, dims=anchor_dims)
        var_bindings = build_anchor_var_bindings(anchor_dims, kwargs=resolved.builder_kwargs)
        ordered_tensor_shapes = extract_ordered_tensor_shapes(
            case_entry=regime_spec.representative_case_entry,
            dims=anchor_dims,
            proto=self._operator_proto_cache,
        )
        primary_res = run_stage_a_static(
            module_source,
            factory_name=resolved.builder_name,
            factory_kwargs=resolved.builder_kwargs,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            target=self._target,
            var_bindings=var_bindings or None,
            enable_track_router=self._enable_track_router,
            case_ratios=regime_spec.representative_case_ratios,
            require_hardware_dma=req_dma,
            require_cube_engine=bool(req_cube and not (is_hybrid and tb_extracted is not None)),
            ordered_tensor_shapes=ordered_tensor_shapes,
        )
        if not primary_res.passed and is_hybrid and tb_extracted is not None and not has_prim_func:
            instantiated_tu = _render_trackb_tu_from_module(
                module_source,
                tb_extracted[1],
                regime_spec,
                require_renderer=False,
                anchor_dims_override=anchor_dims_override,
            )
            return evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
                require_hardware_dma=req_dma,
                require_cube_engine=req_cube,
                module_source=module_source,
                regime_tag=regime_spec.regime_tag,
            )
        return primary_res

    def _derive_topology_axes(
        self,
        ledger: ObjectivePhysicalLedger | None,
        *,
        features: object | None = None,
        route: object | None = None,
    ) -> TopologyAxes:
        if self._topology_axes is not None:
            return self._topology_axes
        if self._semantic_topology_axes is not None:
            return self._semantic_topology_axes
        if self._sil_advisory is not None:
            return resolve_canonical_topology_axes(self._sil_advisory, features=features)
        if route is not None:
            r_axes = (
                route.get("topology_axes")
                if isinstance(route, Mapping)
                else getattr(route, "topology_axes", None)
            )
            if isinstance(r_axes, TopologyAxes):
                return r_axes
            if isinstance(r_axes, Mapping) and all(
                k in r_axes for k in ("compute_engine", "memory_ring", "iteration_regularity")
            ):
                return TopologyAxes.from_dict(r_axes)
        uses_cube = bool(
            ledger is not None
            and (
                ledger.pipe_cycles.get("AIC.PIPE_M", 0) > 0
                or ledger.pipe_cycles.get("AIC.CUBE", 0) > 0
            )
        )
        return resolve_canonical_topology_axes(
            {"op_name": self._config.operator_name, "uses_cube": uses_cube},
            features=features,
        )

    def _resolve_round_parent(
        self,
        *,
        incumbent_source: str,
        regime_spec: RegimeTargetSpec,
        regime_beam: TwoLaneBeamSearch,
        escalated: bool,
        sibling_specs: Sequence[RegimeTargetSpec] | None = None,
    ) -> tuple[str, str, str, BeamCandidate | None, str]:
        """Select parent module, full slice snippet, lane, and stepping stone cleanly."""
        eff_siblings = tuple(sibling_specs) if sibling_specs is not None else self._regime_specs
        op_entry = _resolve_operator_entrypoint_safe(self._config.operator_name)
        _, default_snippet = extract_regime_snippet(
            incumbent_source,
            regime_spec,
            operator_entrypoint=op_entry,
            sibling_specs=eff_siblings,
        )
        if not escalated:
            return incumbent_source, default_snippet, "exploit", None, ""
        for stone in regime_beam.stepping_stone_beam:
            try:
                stone_patched = patch_regime_into_module(
                    incumbent_source,
                    regime_spec,
                    stone.dsl_source,
                    operator_entrypoint=op_entry,
                    sibling_specs=eff_siblings,
                    active_kernel=self._resolve_active_trackb_kernel(
                        incumbent_source,
                        regime_spec,
                    ),
                )
                l0_probe = self._run_l0(stone_patched, regime_spec)
                if not l0_probe.passed:
                    logger.warning(
                        "Skipping invalid stepping-stone %s for regime %s: %s",
                        stone.candidate_id,
                        regime_spec.regime_tag,
                        l0_probe.error_message[:160],
                    )
                    continue
                _, hydrated_snippet = extract_regime_snippet(
                    stone_patched,
                    regime_spec,
                    operator_entrypoint=op_entry,
                    sibling_specs=eff_siblings,
                )
                regime_beam.mark_parent_selected(stone)
                hyp = stone.hypothesis if stone.worker_kind == "creative_explore" else ""
                return stone_patched, hydrated_snippet, "stepping_stone", stone, hyp
            except (SyntaxError, ValueError, KeyError) as exc:
                logger.warning(
                    "Failed to patch stepping-stone slice %s for regime %s: %s",
                    stone.candidate_id,
                    regime_spec.regime_tag,
                    exc,
                )
        return incumbent_source, default_snippet, "stepping_stone", None, ""

    def _is_regime_satisfied(
        self,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> bool:
        if self._perf_target is None:
            return False
        anchor_nums = {
            a.case_num for a in active_anchors if a.regime_tag == regime_spec.regime_tag
        }
        matched = [a for a in best_result.anchors if a.case_num in anchor_nums]
        if not matched:
            return False
        for metric in matched:
            if not metric.accuracy_passed:
                return False
            if metric.roofline_eta < self._perf_target.target_roofline_eta:
                return False
            if metric.speedup < self._perf_target.target_min_case_speedup:
                return False
            max_us = self._perf_target.per_anchor_max_us.get(metric.case_num)
            if max_us is not None and metric.cand_us > max_us:
                return False
        return True

    def _regime_mean_recoverable_excess_us(
        self,
        best_result: CandidateSurrogateResult,
        regime_spec: RegimeTargetSpec,
    ) -> float | None:
        """Compute mean recoverable silicon excess (us) above calibrated dispatch floor."""
        matched = [
            a
            for a in best_result.anchors
            if a.regime_tag == regime_spec.regime_tag and a.cand_us > 0.0
        ]
        if not matched:
            return None
        floor_us = float(self._calibrated_dispatch_floor_us or 0.0)
        excesses = [to_conservation_point(a, a).recoverable_sched_us(floor_us) for a in matched]
        return sum(excesses) / len(excesses)

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None,
        stagnation_by_regime: Mapping[str, int] | None = None,
        last_round_promoted: bool | None = None,
        regime_specs: Sequence[RegimeTargetSpec] | None = None,
    ) -> RegimeTargetSpec | None:
        """Select the highest-urgency unfrozen regime using bandit UCB and headroom."""
        specs = tuple(regime_specs) if regime_specs is not None else self._regime_specs
        by_tag = {r.regime_tag: r for r in specs}
        wall_map = self._compile_wall_streak_by_regime
        stag_map = (
            stagnation_by_regime
            if stagnation_by_regime is not None
            else self._stagnation_by_regime
        )
        active_unfrozen_count = sum(
            1 for spec in specs if spec.regime_tag not in frozen_regimes
        )
        candidates = [
            spec
            for spec in specs
            if spec.regime_tag not in frozen_regimes
            and (active_unfrozen_count <= 1 or int(wall_map.get(spec.regime_tag, 0) or 0) < 3)
            and not self._is_regime_satisfied(spec, best_result, active_anchors)
        ]
        if not candidates and allow_satisfied_fallback:
            candidates = [
                spec
                for spec in specs
                if spec.regime_tag not in frozen_regimes
                and (active_unfrozen_count <= 1 or int(wall_map.get(spec.regime_tag, 0) or 0) < 3)
            ]
        if not candidates:
            return None
        if exclude_regime and any(spec.regime_tag != exclude_regime for spec in candidates):
            candidates = [spec for spec in candidates if spec.regime_tag != exclude_regime]

        dispatch_floor = float(self._calibrated_dispatch_floor_us or 0.0)
        regime_points_map: dict[str, list[AnchorConservationPoint]] = {}
        floor_saturated_map: dict[str, bool] = {}
        has_unsat_other_map: dict[str, bool] = {}
        for spec in candidates:
            sb_nums = {
                int(a.case_num) for a in active_anchors if a.regime_tag == spec.regime_tag
            }
            if sb_nums:
                matched = [
                    a
                    for a in best_result.anchors
                    if int(a.case_num) in sb_nums and a.cand_us > 0.0
                ]
            else:
                matched = [
                    a
                    for a in best_result.anchors
                    if a.regime_tag == spec.regime_tag and a.cand_us > 0.0
                ]
            pts = [to_conservation_point(a, a) for a in matched]
            regime_points_map[spec.regime_tag] = pts
            floor_saturated_map[spec.regime_tag] = is_regime_saturated(
                pts, dispatch_floor, sigma_mult=2.0
            )
            has_unsat_other_map[spec.regime_tag] = (
                bool(pts) and not floor_saturated_map[spec.regime_tag]
            )

        def _is_floor_deprioritized(spec: RegimeTargetSpec) -> int:
            if not floor_saturated_map.get(spec.regime_tag, False):
                return 0
            other_unsat = any(
                other_tag != spec.regime_tag and is_unsat
                for other_tag, is_unsat in has_unsat_other_map.items()
            )
            return 1 if other_unsat else 0

        def _urgency_key(
            spec: RegimeTargetSpec,
        ) -> tuple[int, int, int, float, float, float, str]:
            pts = regime_points_map.get(spec.regime_tag, ())
            stag = max(0, int(stag_map.get(spec.regime_tag, 0) or 0))
            rec = self._regime_mean_recoverable_excess_us(best_result, spec)
            has_excess = 1 if (rec is None or rec > 0.0) else 0
            rec_val = float(rec) if rec is not None else 0.0
            headroom = max(0.0, float(best_result.regime_headroom(spec.regime_tag)))
            bandit_score = (
                regime_bandit_ucb_us(pts, dispatch_floor, stag)
                if pts
                else (headroom / math.sqrt(1.0 + float(stag)))
            )
            wall_yield = 1 if int(wall_map.get(spec.regime_tag, 0) or 0) >= 2 else 0
            floor_dep = _is_floor_deprioritized(spec)
            return (
                wall_yield,
                floor_dep,
                -has_excess,
                -bandit_score,
                -headroom,
                -rec_val,
                spec.regime_tag,
            )

        candidates.sort(key=_urgency_key)
        top_spec = candidates[0]
        candidate_tags = {spec.regime_tag for spec in candidates}
        promoted_map = self._last_round_promoted_by_regime
        was_promoted = (
            promoted_map.get(current_regime, self._last_round_promoted)
            if (current_regime and last_round_promoted is None)
            else bool(last_round_promoted)
        )
        if (
            current_regime
            and current_regime in candidate_tags
            and current_regime in by_tag
            and int(wall_map.get(current_regime, 0) or 0) < 2
            and was_promoted
            and _is_floor_deprioritized(by_tag[current_regime]) == 0
        ):
            top_pts = regime_points_map.get(top_spec.regime_tag, ())
            cur_pts = regime_points_map.get(current_regime, ())
            if top_pts and cur_pts:
                top_ucb = regime_bandit_ucb_us(top_pts, dispatch_floor, 0)
                cur_ucb = regime_bandit_ucb_us(cur_pts, dispatch_floor, 0)
                cur_sigma = math.sqrt(sum(p.sigma_paired_us() ** 2 for p in cur_pts))
                if top_ucb - cur_ucb <= 2.0 * cur_sigma:
                    return by_tag[current_regime]
        return top_spec

    @staticmethod
    def _slice_contract_instruction(
        func_name: str,
        regime_spec: RegimeTargetSpec,
        *,
        operator_entrypoint: str = "",
    ) -> str:
        """Format slice scope instructions for *func_name* and *regime_spec*."""
        kernel_name_warning = ""
        if operator_entrypoint == "kernel":
            kernel_name_warning = (
                " CRITICAL: the public operator entrypoint is named 'kernel', so you MUST NOT "
                "define a top-level '@T.prim_func def kernel' or 'def kernel'; name any inner "
                "prim_func differently (e.g., '_ar_kernel_impl') and return it from the builder."
            )
        if not regime_spec.editable_symbols:
            return f"function '{func_name}'{kernel_name_warning}"
        syms = ", ".join(regime_spec.editable_symbols)
        helper_note = (
            f" or '{regime_spec.allowed_helper_prefix}*' helpers"
            if regime_spec.allowed_helper_prefix
            else ""
        )
        return (
            f"authorized slice symbols ({syms}){helper_note} for entrypoint '{func_name}'. "
            "You may output ONLY the subset of functions or prefixed macros you modified "
            "(e.g., just _select_*_config, a single helper macro, or one builder); omitted "
            "sibling symbols in this regime automatically inherit their parent implementation. "
            f"Preserve exact public entrypoint '{func_name}' and `kernel = {func_name}` without "
            f"renaming or modifying symbols from other regimes{kernel_name_warning}"
        )

    @staticmethod
    def _format_msprof_section(
        residual: StaticDynamicResidual | None,
        *,
        p2_diag: Mapping[str, object] | None = None,
    ) -> str:
        """Format measured MSPROF PMU counters, compiler hazards, and P2 hotspots."""
        has_pmu = bool(
            residual is not None
            and (
                residual.measured_aic_mac_ratio > 0.0
                or residual.measured_aiv_vec_ratio > 0.0
                or residual.measured_mte2_wait_ratio > 0.0
                or residual.measured_sync_stall_ratio > 0.0
            )
        )
        has_compiler_hazard = bool(
            residual is not None
            and (
                residual.spill_bytes > 0
                or residual.stack_bytes > 0
                or (not residual.vloopv2_hit)
                or residual.vf_fusion_breaks > 0
                or residual.alias_serialized_buffers > 0
                or residual.dual_issue_ipc_collapse > 0.0
                or residual.dcache_dcci_flushes > 0
                or bool(residual.has_local_membar_in_vf_call)
                or residual.scalar_compute_loop_fallback > 0.0
                or residual.dcache_bypass_narrow_store > 0.0
            )
        )
        p2_lines: list[str] = []
        if p2_diag:
            raw_hotspots = (
                p2_diag.get("p2_hotspots") or p2_diag.get("source_line_hotspots") or ()
            )
            if isinstance(raw_hotspots, Sequence) and not isinstance(raw_hotspots, (str, bytes)):
                formatted_spots: list[str] = []
                for spot in list(raw_hotspots)[:3]:
                    if isinstance(spot, Mapping):
                        loc = str(spot.get("source_location") or "?")
                        cnt = spot.get("instruction_count", 0)
                        share = spot.get("instruction_share")
                        if isinstance(share, (int, float)) and not isinstance(share, bool):
                            formatted_spots.append(
                                f"{loc} (instrs={cnt}, share={float(share):.3f})"
                            )
                        else:
                            formatted_spots.append(f"{loc} (instrs={cnt})")
                    elif hasattr(spot, "source_location"):
                        loc = str(getattr(spot, "source_location", "?"))
                        cnt = getattr(spot, "instruction_count", 0)
                        share = getattr(spot, "instruction_share", None)
                        if isinstance(share, (int, float)) and not isinstance(share, bool):
                            formatted_spots.append(
                                f"{loc} (instrs={cnt}, share={float(share):.3f})"
                            )
                        else:
                            formatted_spots.append(f"{loc} (instrs={cnt})")
                    else:
                        formatted_spots.append(str(spot))
                if formatted_spots:
                    p2_lines.append("p2_top_hotspots: " + "; ".join(formatted_spots))
            raw_waitflag = p2_diag.get("p2_mte_waitflag_share")
            if raw_waitflag is None:
                raw_waitflag = p2_diag.get("mte_waitflag_share")
            if isinstance(raw_waitflag, (int, float)) and not isinstance(raw_waitflag, bool):
                p2_lines.append(f"p2_mte_waitflag_share: {float(raw_waitflag):.4f}")
            raw_issue = p2_diag.get("p2_issue_interval_us")
            if raw_issue is None:
                raw_issue = p2_diag.get("issue_interval_us")
            if isinstance(raw_issue, Mapping) and raw_issue:
                parts: list[str] = []
                for pipe in sorted(raw_issue):
                    stats = raw_issue[pipe]
                    if isinstance(stats, Mapping):
                        p50 = float(stats.get("p50_us", 0.0) or 0.0)
                        p90 = float(stats.get("p90_us", 0.0) or 0.0)
                        parts.append(f"{pipe}(p50={p50:.3f}us, p90={p90:.3f}us)")
                    elif hasattr(stats, "p50_us") and hasattr(stats, "p90_us"):
                        p50 = float(getattr(stats, "p50_us", 0.0) or 0.0)
                        p90 = float(getattr(stats, "p90_us", 0.0) or 0.0)
                        parts.append(f"{pipe}(p50={p50:.3f}us, p90={p90:.3f}us)")
                    elif isinstance(stats, (int, float)) and not isinstance(stats, bool):
                        parts.append(f"{pipe}={float(stats):.4f}us")
                if parts:
                    p2_lines.append("p2_issue_interval_us: " + ", ".join(parts))
            elif isinstance(raw_issue, (int, float)) and not isinstance(raw_issue, bool):
                p2_lines.append(f"p2_issue_interval_us: {float(raw_issue):.4f}")
        has_p2 = bool(p2_lines)
        if not has_pmu and not has_compiler_hazard and not has_p2:
            return ""
        lines = ["### HARDWARE MSPROF TELEMETRY"]
        if has_pmu and residual is not None:
            res_map = residual.to_dict()
            lines.extend(
                [
                    f"aic_mac_ratio: {residual.measured_aic_mac_ratio:.4f}",
                    f"aiv_vec_ratio: {residual.measured_aiv_vec_ratio:.4f}",
                    f"mte2_wait_ratio: {residual.measured_mte2_wait_ratio:.4f}",
                    f"sync_stall_ratio: {residual.measured_sync_stall_ratio:.4f}",
                    f"unexplained_gap_ratio: {residual.unexplained_gap_ratio:.4f}",
                ]
            )
            raw_infl = res_map.get("transfer_inflation_ratio")
            if isinstance(raw_infl, (int, float)) and not isinstance(raw_infl, bool):
                lines.append(f"transfer_inflation_ratio: {float(raw_infl):.4f}")
            raw_eta = res_map.get("effective_compute_eta")
            if isinstance(raw_eta, (int, float)) and not isinstance(raw_eta, bool):
                lines.append(f"effective_compute_eta: {float(raw_eta):.4f}")
        if has_compiler_hazard and residual is not None:
            lines.extend(
                [
                    f"spill_bytes: {residual.spill_bytes}",
                    f"stack_bytes: {residual.stack_bytes}",
                    f"vloopv2_hit: {residual.vloopv2_hit}",
                    f"vf_fusion_breaks: {residual.vf_fusion_breaks}",
                    f"alias_serialized_buffers: {residual.alias_serialized_buffers}",
                    f"dual_issue_ipc_collapse: {residual.dual_issue_ipc_collapse:.4f}",
                    f"dcache_dcci_flushes: {residual.dcache_dcci_flushes}",
                    f"has_local_membar_in_vf_call: {residual.has_local_membar_in_vf_call}",
                    f"scalar_compute_loop_fallback: {residual.scalar_compute_loop_fallback:.1f}",
                    f"dcache_bypass_narrow_store: {residual.dcache_bypass_narrow_store:.1f}",
                ]
            )
        if p2_lines:
            lines.extend(p2_lines)
        return "\n".join(lines) + "\n\n"

    def _format_sil_live_policy_section(
        self,
        *,
        live_plan: RegimePlanResult | None,
        active_mask: ConstraintRelaxationMask | None,
        escalation: object | None = None,
        pipe_gaps: object | None = None,
        cost_pred: object | None = None,
        escalated_trackb_tu: str = "",
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
    ) -> str:
        """Format live SIL solver knobs, 7-facet relaxation envelope, and escalation state."""
        if live_plan is None and active_mask is None:
            return ""
        lines = ["### SIL LIVE POLICY & 7-FACET CONSTRAINT ENVELOPE"]
        if live_plan is not None:
            pol = live_plan.policy
            lines.append(
                f"- topology_axes: {live_plan.topology_axes.key()} "
                f"(solver_mode={live_plan.solver_mode}, "
                f"backend_tier={pol.get('backend_tier', 'tilelang_dsl')})"
            )
            lines.append(
                f"- live_solved_knobs: tile_mnk={pol.get('tile_mnk')}, "
                f"row_pack_factor={pol.get('row_pack_factor', 1)}, "
                f"l1_stages={pol.get('l1_stages', 1)}, "
                f"ub_bytes={pol.get('ub_estimated_bytes', 0)}, "
                f"l1_bytes={pol.get('l1_estimated_bytes', 0)}"
            )
            if live_plan.notes:
                lines.append(f"- solver_note: {live_plan.notes[-1]}")
        if active_mask is not None:
            lines.append(f"- unlocked_facets: {', '.join(active_mask.unlocked_facets())}")
            locked = active_mask.locked_facets()
            lines.append(f"- locked_facets: {', '.join(locked) if locked else 'none (top)'}")
        resolved_bound = ""
        if parent_ledger is not None or parent_residual is not None or cost_pred is not None:
            resolved_bound = str(
                self._cost_advisor.resolve_dominant_bound(
                    parent_ledger,
                    parent_residual,
                    fallback_cost_pred=cost_pred,
                )
                or ""
            )
        if cost_pred is not None:
            pred_val = (
                float(cost_pred)
                if isinstance(cost_pred, (int, float)) and not isinstance(cost_pred, bool)
                else float(getattr(cost_pred, "pred_us", 0.0) or 0.0)
            )
            bound_val = resolved_bound or str(getattr(cost_pred, "dominant_bound", "") or "")
            penalty_val = float(getattr(cost_pred, "tier1_penalty_us", 0.0) or 0.0)
            lines.append(
                f"- cost_model_pred_us: {pred_val:.2f} "
                f"(dominant_bound={bound_val}, "
                f"tier1_penalty_us={penalty_val:.2f})"
            )
        elif resolved_bound:
            lines.append(f"- dominant_bound: {resolved_bound}")
        if escalation is not None:
            should_esc = bool(getattr(escalation, "should_escalate", False))
            to_tier = getattr(escalation, "to_tier", "")
            reasons = tuple(getattr(escalation, "trigger_reasons", ()) or ())
            lines.append(
                f"- track_escalation: should_escalate={should_esc}, to_tier={to_tier}, "
                f"reasons={list(reasons)}"
            )
        if escalated_trackb_tu.strip():
            lines.append("- escalated_trackb_seed_cpp:")
            lines.append("```cpp")
            lines.append(escalated_trackb_tu.strip())
            lines.append("```")
        if pipe_gaps is not None:
            cov = float(getattr(pipe_gaps, "coverage_ratio", 1.0))
            missing = tuple(getattr(pipe_gaps, "missing_primitives", ()) or ())
            lines.append(
                f"- tilelang_7pipe_isa_coverage: {cov:.2%}, missing_count={len(missing)}, "
                f"sample_gaps={list(missing[:6])}"
            )
        return "\n".join(lines) + "\n\n"

    def _build_cost_workload(
        self,
        ledger: ObjectivePhysicalLedger,
        *,
        live_plan: RegimePlanResult | None = None,
        axes: TopologyAxes | None = None,
        blueprint: HirBlueprintPacket | None = None,
        stage_b_profile: Mapping[str, object] | None = None,
    ) -> CostFeatures:
        return build_cost_features_from_ledger(
            ledger,
            live_plan=live_plan,
            axes=axes,
            blueprint=blueprint,
            sil_advisory=self._sil_advisory,
            target=self._target,
            stage_b_profile=stage_b_profile,
        )

    def _lookup_case_dims_safe(
        self,
        case_num: int,
        regime_tag: str = "",
    ) -> dict[str, object]:
        """Look up case dimensions from CANNBench or fall back to representative dims."""
        cnum = int(case_num)
        if cnum in self._case_dims_cache:
            return dict(self._case_dims_cache[cnum])
        dims: dict[str, object] = {}
        try:
            dims = dict(
                lookup_case_dims(
                    self._config.operator_name,
                    cnum,
                    cannbench_repo=self._cannbench_repo,
                    env_file=self._env_file,
                )
            )
        except (FileNotFoundError, OSError, KeyError, ValueError):
            for spec in self._regime_specs:
                if regime_tag and spec.regime_tag == regime_tag:
                    dims = dict(spec.representative_anchor_dims or spec.representative_hir_dims)
                    break
            if not dims and self._regime_specs:
                first = self._regime_specs[0]
                dims = dict(first.representative_anchor_dims or first.representative_hir_dims)
        self._case_dims_cache[cnum] = dict(dims)
        return dims

    @staticmethod
    def _format_trackb_hotpath_section(
        regime_spec: RegimeTargetSpec,
        active_kernel: str | None = None,
    ) -> tuple[str, str]:
        """Return `(escalation_tier, hotpath_prompt_section)` for *regime_spec*."""
        backend = normalize_runtime_backend(regime_spec.runtime_backend)
        esc_tier = (
            "ascendc_template"
            if backend in ("ascendc_trackb", "hybrid_trackb")
            else "tilelang_dsl"
        )
        if not regime_spec.trackb_symbols:
            return esc_tier, ""
        active_line = f"- active_kernel: {active_kernel}\n" if active_kernel else ""
        if "_render_trackb_tu" in regime_spec.editable_symbols:
            signature_rule = (
                "- _render_trackb_tu is editable: keep its callable ABI "
                "``_render_trackb_tu(**dims)`` and the launcher dims it receives consistent "
                "with the launched kernel.\n"
            )
        else:
            signature_rule = (
                "- _render_trackb_tu is immutable: keep the launched kernel's ``template <...>`` "
                "and parameter signature unchanged so instantiation still binds.\n"
            )
        section = (
            "### ACTIVE RUNTIME HOT PATH\n"
            f"- runtime_backend: {regime_spec.runtime_backend}\n"
            f"- active_trackb_symbols: {list(regime_spec.trackb_symbols)}\n"
            f"{active_line}"
            "- CRITICAL: On NPU this regime executes the Track-B CCE (C++20) kernel string "
            "and config selector shown in TARGET FUNCTION SNIPPET, NOT dead TileLang fallback "
            "builders. Edit the active Track-B CCE (C++20) kernel string / selector directly.\n\n"
            "### TRACK-B REPLY CONTRACT\n"
            "- Reply with `# target_bottleneck:` and `# structural_hypothesis:` comments "
            "followed by one fenced block in one of these forms:\n"
            "  1. Complete bare C++ unit per changed function (preferred): either the "
            "``__global__`` kernel or one ``__simd_vf__`` helper, keeping its exact name and "
            "signature.\n"
            "  2. SEARCH/REPLACE hunks with ``<<<<<<< SEARCH [Lstart-Lend]`` line anchors against "
            "the launched kernel and the helpers it calls.\n"
            "  3. A Python slice that reassigns one authorized Track-B C++ constant in full.\n"
            "- Never echo ``[READONLY_SIBLING_DECLARATION]`` or read-only helper markers.\n"
            "- A full C++ constant must still contain every sibling ``__global__`` kernel "
            "present in the original string; a replacement that drops one is rejected.\n"
            f"{signature_rule}"
            "- Put any selector change outside the hunks as a Python ``_select_*_config`` "
            "definition in the same fenced block.\n\n"
        )
        return esc_tier, section

    @staticmethod
    def _prepare_prompt_snippet_and_tail(
        *,
        snippet_source: str,
        incumbent_source: str,
        regime_spec: RegimeTargetSpec,
        op_entry: str,
        active_kernel: str | None,
        focus_lines: Sequence[int],
        diag_text: str,
    ) -> tuple[str, str, str]:
        """Prepare fenced snippet, read-only host tail tag, and Track-A TIR scope section."""
        if regime_spec.trackb_symbols:
            compacted = _compact_trackb_snippet_source(
                snippet_source,
                regime_spec,
                incumbent_source=incumbent_source,
                active_kernel=active_kernel,
                focus_lines=focus_lines,
                diagnostic_text=diag_text,
            )
            return compacted, "", ""
        protected_exports: tuple[str, ...] = DEFAULT_READONLY_HOST_EXPORTS
        if (
            op_entry
            and op_entry not in protected_exports
            and op_entry not in regime_spec.editable_symbols
        ):
            protected_exports = (*DEFAULT_READONLY_HOST_EXPORTS, op_entry)
        editable_snippet, readonly_tail = split_readonly_host_tail(
            snippet_source,
            protected_exports,
        )
        if not readonly_tail and incumbent_source.strip():
            _, readonly_tail = split_readonly_host_tail(
                incumbent_source,
                protected_exports,
            )
        fenced_snippet = editable_snippet if editable_snippet.strip() else snippet_source
        host_tail_section = (
            f"<read_only_host_tail>\n{readonly_tail}\n</read_only_host_tail>\n\n"
            if readonly_tail
            else ""
        )
        tir_scope_section = (
            "### TIR SCOPE CONTRACT\n"
            "- Never define a nested `def` inside `@T.prim_func`; keep all helpers at "
            "module top level.\n"
            "- Do not redefine or alter `<read_only_host_tail>` bindings.\n\n"
        )
        return fenced_snippet, host_tail_section, tir_scope_section

    def _build_exploit_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        rule: AscendCausalRule,
        best_result: CandidateSurrogateResult,
        blueprint: HirBlueprintPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        sil_policy_section: str = "",
        effort: str | None = None,
        p2_diag: Mapping[str, object] | None = None,
    ) -> GenerationRequest:
        """Build a rule-exploit GenerationRequest using header comments and a fenced block."""
        module_source = incumbent_source or snippet_source
        active_kernel = self._resolve_active_trackb_kernel(module_source, regime_spec)
        focus_lines, diag_text = _extract_p2_focus_lines_and_diag(p2_diag)
        op_entry = _resolve_operator_entrypoint_safe(
            str(getattr(self._config, "operator_name", "") or "")
        )
        fenced_snippet, host_tail_section, tir_scope_section = (
            self._prepare_prompt_snippet_and_tail(
                snippet_source=snippet_source,
                incumbent_source=incumbent_source,
                regime_spec=regime_spec,
                op_entry=op_entry,
                active_kernel=active_kernel,
                focus_lines=focus_lines,
                diag_text=diag_text,
            )
        )
        slice_desc = self._slice_contract_instruction(
            func_name,
            regime_spec,
            operator_entrypoint=op_entry,
        )
        esc_tier, hotpath_section = self._format_trackb_hotpath_section(
            regime_spec,
            active_kernel=active_kernel,
        )
        header_proto = (
            "Begin with two single-line header comments:\n"
            "# target_bottleneck: <bottleneck metric>\n"
            "# structural_hypothesis: <concise optimization hypothesis>\n"
            "followed by one fenced code block containing "
        )
        if regime_spec.trackb_symbols:
            sys_prompt = (
                "You are an Ascend 950PR kernel performance engineer. Apply the selected causal "
                "optimization rule cleanly while preserving exact numerical semantics, dynamic "
                f"shape compatibility, and {slice_desc}. {header_proto}"
                "ONLY the reply form required by the TRACK-B REPLY CONTRACT."
            )
        else:
            sys_prompt = (
                "You are an Ascend 950PR kernel performance engineer. Apply the selected causal "
                "optimization rule cleanly while preserving exact numerical semantics, dynamic "
                f"shape compatibility, and {slice_desc}. {header_proto}"
                "ONLY Python code for this regime slice (never nest `def` inside `@T.prim_func`)."
            )
        constraint_text = build_progressive_llm_constraints(
            escalation_tier=esc_tier,
            track=track,
            enable_track_router=bool(self._enable_track_router),
            sil_advisory=self._sil_advisory,
            freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint,
            current_ledger=parent_ledger,
            topology_axes=self._derive_topology_axes(parent_ledger),
            active_symptom=(
                rule.symptom_family.value
                if hasattr(rule.symptom_family, "value")
                else str(rule.symptom_family)
            ),
        )
        track_section = f"{constraint_text}\n\n"
        msprof_section = self._format_msprof_section(parent_residual, p2_diag=p2_diag)
        op_name = str(
            getattr(self._config, "operator_name", "") or getattr(self._config, "operator", "") or ""
        )
        domain_env = get_input_domain_envelope(op_name)
        domain_section = (
            f"{domain_env.format_prompt_block()}\n\n" if domain_env is not None else ""
        )
        golden_abi_block = self._build_golden_entrypoint_abi_block()
        abi_section = (
            _extract_regime_abi_context(
                incumbent_source,
                regime_spec,
                golden_abi_block=golden_abi_block,
            )
            if incumbent_source
            else golden_abi_block
        )
        eff_rule = (
            rule.render_for_backend(regime_spec.runtime_backend)
            if hasattr(rule, "render_for_backend")
            else rule
        )
        boundary_block = self._boundary_ledger.format_regime_boundary_prompt(
            regime_spec.regime_tag
        )
        boundary_section = f"{boundary_block}\n\n" if boundary_block else ""
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{hotpath_section}"
            f"{tir_scope_section}"
            f"{track_section}"
            f"{sil_policy_section}"
            f"{boundary_section}"
            f"### SELECTED CAUSAL RULE\n"
            f"Rule ID: {eff_rule.rule_id}\n"
            f"Precondition: {eff_rule.precondition}\n"
            f"Transform Template:\n{eff_rule.transform_template.strip()}\n"
            f"Expected Postcondition: {eff_rule.expected_postcondition}\n\n"
            f"{msprof_section}"
            f"{domain_section}"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_section}"
            f"### TARGET FUNCTION SNIPPET\n```python\n{fenced_snippet}\n```\n\n"
            f"{host_tail_section}"
        )
        cfg_effort = getattr(self._config, "exploit_effort", "medium")
        eff = sanitize_effort(effort if effort is not None else cfg_effort, default="medium")
        return GenerationRequest(
            role="exploit_rule",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            effort=eff,
        )

    def _build_creative_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        effort: str,
        excluded_mechanisms: Sequence[str] = (),
        focus_hint: str = "",
        prescreen_summary: str = "",
        blueprint: HirBlueprintPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        parent_hypothesis: str = "",
        sil_policy_section: str = "",
        p2_diag: Mapping[str, object] | None = None,
        paradigm_shift: bool = False,
    ) -> GenerationRequest:
        """Build a creative GenerationRequest using header comments and a fenced block."""
        module_source = incumbent_source or snippet_source
        active_kernel = self._resolve_active_trackb_kernel(module_source, regime_spec)
        focus_lines, diag_text = _extract_p2_focus_lines_and_diag(p2_diag)
        op_entry = _resolve_operator_entrypoint_safe(
            str(getattr(self._config, "operator_name", "") or "")
        )
        fenced_snippet, host_tail_section, tir_scope_section = (
            self._prepare_prompt_snippet_and_tail(
                snippet_source=snippet_source,
                incumbent_source=incumbent_source,
                regime_spec=regime_spec,
                op_entry=op_entry,
                active_kernel=active_kernel,
                focus_lines=focus_lines,
                diag_text=diag_text,
            )
        )
        slice_desc = self._slice_contract_instruction(
            func_name,
            regime_spec,
            operator_entrypoint=op_entry,
        )
        esc_tier, hotpath_section = self._format_trackb_hotpath_section(
            regime_spec,
            active_kernel=active_kernel,
        )
        if regime_spec.trackb_symbols:
            reply_desc = (
                "the TRACK-B REPLY CONTRACT form (SEARCH/REPLACE hunks, one bare C++ unit per "
                "changed function, or a Python slice reassigning one authorized constant)"
            )
        else:
            reply_desc = (
                "Python code for this regime slice only (never nest `def` inside `@T.prim_func`)"
            )
        sys_prompt = (
            "You are an Ascend 950PR (dav-3510) systems architect. Propose an architectural "
            "optimization grounded in Roofline limits (56 AIV / 28 MixCore blocks, 3 KiB SSBUF, "
            "L0C->UB direct path, UB->L1 direct path, 64-elem UB alignment, Split-KV parallelism). "
            f"Preserve exact numerical semantics, signatures, and {slice_desc}. "
            "Respond ONLY with two header comments:\n"
            "# target_bottleneck: <bottleneck metric>\n"
            "# structural_hypothesis: <concise architectural hypothesis>\n"
            f"followed by a native fenced code block containing {reply_desc}."
        )
        if excluded_mechanisms:
            constraint_lines = "\n".join(f"- {item}" for item in excluded_mechanisms)
        else:
            constraint_lines = "- None recorded yet; explore any high-impact structural axis."
        neg_section = (
            "### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)\n"
            f"{constraint_lines}\n"
            "Propose a structurally orthogonal transformation that does NOT duplicate the "
            "mechanisms listed above.\n\n"
        )
        constraint_text = build_progressive_llm_constraints(
            escalation_tier=esc_tier,
            track=track,
            enable_track_router=bool(self._enable_track_router),
            sil_advisory=self._sil_advisory,
            freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint,
            current_ledger=parent_ledger,
            excluded_mechanisms=excluded_mechanisms,
            topology_axes=self._derive_topology_axes(parent_ledger),
        )
        track_section = f"{constraint_text}\n\n"
        focus_section = f"### ARCHITECTURAL FOCUS HINT\n{focus_hint}\n\n" if focus_hint else ""
        parent_hyp_section = (
            f"### STEPPING-STONE PARENT HYPOTHESIS\n{parent_hypothesis}\n\n"
            if parent_hypothesis
            else ""
        )
        prescreen_section = (
            f"### TILE PRESCREEN GUIDANCE\n{prescreen_summary}\n\n"
            if prescreen_summary
            else ""
        )
        msprof_section = self._format_msprof_section(parent_residual, p2_diag=p2_diag)
        op_name = str(
            getattr(self._config, "operator_name", "") or getattr(self._config, "operator", "") or ""
        )
        domain_env = get_input_domain_envelope(op_name)
        domain_section = (
            f"{domain_env.format_prompt_block()}\n\n" if domain_env is not None else ""
        )
        golden_abi_block = self._build_golden_entrypoint_abi_block()
        abi_section = (
            _extract_regime_abi_context(
                incumbent_source,
                regime_spec,
                golden_abi_block=golden_abi_block,
            )
            if incumbent_source
            else golden_abi_block
        )
        shift_section = (
            "### PARADIGM SHIFT CROSS-TOPOLOGY UNLOCK\n"
            "Cross-topology paradigm shift active (full_slice_rewrite authorized): "
            "CostModel prescreen is disengaged; explore coupled_1c2v AIC+AIV chunked "
            "GEMM, multi-stage GM workspace, or multi-core tiling across the regime slice.\n\n"
            if paradigm_shift
            else ""
        )
        boundary_block = self._boundary_ledger.format_regime_boundary_prompt(
            regime_spec.regime_tag
        )
        boundary_section = f"{boundary_block}\n\n" if boundary_block else ""
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{shift_section}"
            f"{hotpath_section}"
            f"{tir_scope_section}"
            f"{track_section}"
            f"{sil_policy_section}"
            f"{boundary_section}"
            f"{focus_section}"
            f"{parent_hyp_section}"
            f"{neg_section}"
            f"{prescreen_section}"
            f"{msprof_section}"
            f"{domain_section}"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_section}"
            f"### TARGET FUNCTION SNIPPET\n```python\n{fenced_snippet}\n```\n\n"
            f"{host_tail_section}"
        )
        return GenerationRequest(
            role="explore_creative",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            effort=effort,
            paradigm_shift=bool(paradigm_shift),
        )

    def _resolve_trackb_freedom_mask(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> ConstraintRelaxationMask:
        existing = self._freedom_masks.get(regime_spec.regime_tag)
        if existing is not None:
            return existing
        return ConstraintRelaxationMask.bottom()

    def build_round_bundle(
        self,
        state: ChampionState,
        *,
        round_idx: int,
        regime_spec: RegimeTargetSpec | None = None,
        working_anchors: Sequence[AnchorCaseSpec] | None = None,
        exclude_regime: str | None = None,
        creative_effort: str = "high",
    ) -> RoundSynthesisBundle:
        """Assemble the complete RoundSynthesisBundle from an immutable ChampionState."""
        self._calibrated_dispatch_floor_us = float(state.calibrated_dispatch_floor_us)
        self._stagnation_by_regime = dict(state.stagnation_by_regime)
        self._compile_wall_streak_by_regime = dict(state.compile_wall_streak_by_regime)
        self._last_round_promoted = bool(state.last_round_promoted)
        self._last_round_promoted_by_regime = dict(state.last_round_promoted_by_regime)
        if state.boundary_ledger is not None:
            self._boundary_ledger = state.boundary_ledger
        if state.sil_advisory is not None:
            self._sil_advisory = state.sil_advisory
        if state.topology_axes is not None:
            self._topology_axes = state.topology_axes
        if state.semantic_topology_axes is not None:
            self._semantic_topology_axes = state.semantic_topology_axes
        if state.operator_proto_cache is not None:
            self._operator_proto_cache = state.operator_proto_cache
        if state.regime_blueprints:
            self._regime_blueprints = dict(state.regime_blueprints)
        for k, v in state.freedom_masks.items():
            self._freedom_masks.setdefault(k, v)
        self._active_kernel_cache.update(state.active_kernel_cache)

        eff_specs = state.regime_specs or self._regime_specs
        eff_anchors = tuple(
            working_anchors if working_anchors is not None else state.active_anchors
        )
        selected_spec = regime_spec or self._select_active_regime(
            state.current_regime,
            set(state.frozen_regimes),
            state.best_result,
            eff_anchors,
            allow_satisfied_fallback=True,
            exclude_regime=exclude_regime,
            stagnation_by_regime=state.stagnation_by_regime,
            regime_specs=eff_specs,
        )
        if selected_spec is None:
            raise RuntimeError("No active unfrozen regime available for round synthesis.")

        op_entry = _resolve_operator_entrypoint_safe(self._config.operator_name)
        func_name, _default_snippet = extract_regime_snippet(
            state.incumbent_source,
            selected_spec,
            operator_entrypoint=op_entry,
            sibling_specs=eff_specs,
        )
        regime_beam = state.beams.get(selected_spec.regime_tag) or TwoLaneBeamSearch()
        stagnation_rounds = int(
            state.stagnation_by_regime.get(selected_spec.regime_tag, state.stagnation_rounds)
        )
        escalated = stagnation_rounds >= int(self._config.stagnation_escalate_rounds)

        (
            parent_module_source,
            parent_snippet,
            parent_lane,
            selected_stone,
            selected_stone_hypothesis,
        ) = self._resolve_round_parent(
            incumbent_source=state.incumbent_source,
            regime_spec=selected_spec,
            regime_beam=regime_beam,
            escalated=escalated,
            sibling_specs=eff_specs,
        )
        stone_ledger = selected_stone.ledger if selected_stone is not None else None
        stone_track = selected_stone.track if selected_stone is not None else None
        selected_stone_id = selected_stone.candidate_id if selected_stone is not None else None
        active_parent_id = (
            selected_stone_id
            if selected_stone_id
            else (
                regime_beam.exploit_beam[0].candidate_id
                if regime_beam.exploit_beam
                else f"incumbent_r{max(0, round_idx - 1)}"
            )
        )

        bn_anchor = bottleneck_anchor_for_regime(state.best_result, selected_spec.regime_tag)
        bn_dims: dict[str, int] | None = None
        if bn_anchor is not None:
            resolved_bn = self._lookup_case_dims_safe(
                bn_anchor.case_num, selected_spec.regime_tag
            )
            if resolved_bn:
                bn_dims = {
                    str(k): int(v)
                    for k, v in resolved_bn.items()
                    if isinstance(v, int) and not isinstance(v, bool)
                }

        base_stage_a = self._run_l0(
            parent_module_source,
            selected_spec,
            anchor_dims_override=bn_dims or None,
        )
        parent_ledger = base_stage_a.ledger if base_stage_a.ledger is not None else stone_ledger
        if parent_ledger is None:
            raise RuntimeError(
                f"Stage-A L0 produced no ledger for regime {selected_spec.regime_tag!r}: "
                f"{base_stage_a.error_message}"
            )
        if self._enable_track_router:
            parent_track = (
                stone_track
                if stone_track is not None
                else (
                    base_stage_a.route.track
                    if base_stage_a.route is not None
                    else "unconstrained"
                )
            )
            parent_features = base_stage_a.features
        else:
            parent_track = "unconstrained"
            parent_features = None

        regime_best = state.best_result.for_regime(selected_spec.regime_tag)
        local_best = regime_best if regime_best.anchors else state.best_result
        prev_local_rel_gain = state.last_local_rel_gain_by_regime.get(selected_spec.regime_tag)
        mean_recoverable_excess_us = self._regime_mean_recoverable_excess_us(
            state.best_result, selected_spec
        )
        parent_axes = self._derive_topology_axes(
            parent_ledger,
            features=parent_features,
            route=base_stage_a.route,
        )
        bucket_key = f"{selected_spec.regime_tag}:{parent_axes.key()}"
        spearman_hist = self._boundary_ledger.spearman_history.get(bucket_key) or []
        latest_rho = spearman_hist[-1] if spearman_hist else None
        clean_samples = 0
        if self._experience_lake is not None:
            try:
                _cands, all_obs = self._experience_lake.load_records()
                clean_samples = sum(
                    1
                    for obs in all_obs
                    if obs.operator == self._config.operator_name
                    and obs.regime_tag == selected_spec.regime_tag
                    and obs.accuracy_passed
                    and is_clean_calibration_sample(obs.burst_batch_cv)
                )
            except (OSError, ValueError):
                clean_samples = 0
        gate_decision = self._invoke_with_optional_kwargs(
            self._cost_advisor.evaluate_round_policy,
            enable_prescreen_flag=bool(self._config.enable_tilesize_prescreen),
            bucket_clean_samples=clean_samples,
            out_of_sample_spearman_rho=latest_rho,
        )
        if self._boundary_ledger.is_cost_prescreen_quarantined(bucket_key):
            gate_decision = replace(
                gate_decision,
                allow_candidate_pruning=False,
                reason="spearman_rank_inverted",
            )

        rep_dims = {
            str(k): int(v)
            for k, v in (
                selected_spec.representative_anchor_dims or selected_spec.representative_hir_dims
            ).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        blueprint = self._regime_blueprints.get(selected_spec.regime_tag)
        blueprint_roofline = blueprint.roofline if blueprint is not None else None
        sil_regime = None
        if self._sil_advisory is not None:
            for reg in getattr(self._sil_advisory, "regimes", ()) or ():
                if getattr(reg, "regime_tag", None) == selected_spec.regime_tag:
                    sil_regime = reg
                    break
        if sil_regime is None:
            sil_regime = RegimeSlice(
                regime_tag=selected_spec.regime_tag,
                case_nums=(),
                representative_dims=rep_dims,
                has_unaligned_tail=(
                    "unaligned" in selected_spec.regime_tag or "tail" in selected_spec.regime_tag
                ),
            )
        adv_dtypes = tuple(getattr(self._sil_advisory, "input_dtypes", ()) or ())
        if not adv_dtypes:
            raw_case_dtype = selected_spec.representative_case_entry.get("dtype")
            if isinstance(raw_case_dtype, str) and raw_case_dtype.strip():
                adv_dtypes = (raw_case_dtype.strip(),)
            elif isinstance(raw_case_dtype, (list, tuple)):
                adv_dtypes = tuple(
                    str(item).strip() for item in raw_case_dtype if str(item).strip()
                )
        adv_nan_inf = bool(getattr(self._sil_advisory, "has_nan_inf_cases", False))
        live_plan = recompute_regime_for_live_buffers(
            self._regime_solver,
            sil_regime,
            parent_ledger,
            op_name=self._config.operator_name,
            topology_axes=parent_axes,
            roofline=self._roofline_arg_for_solver(blueprint_roofline),
            input_dtypes=adv_dtypes or ("float16",),
            has_nan_inf_cases=adv_nan_inf,
            has_indirect_gather=(parent_axes.iteration_regularity == "irregular_simt"),
            has_recurrence=(
                parent_axes.memory_ring in ("persistent_onchip_state", "bidirectional_cv_ring")
            ),
            uses_cube=parent_axes.uses_cube,
        )

        msprof_by_case = {
            (str(sha), int(cnum)): dict(metrics)
            for (sha, cnum), metrics in state.msprof_by_case.items()
        }
        if stagnation_rounds >= 1 and self._collect_msprof_fn is not None:
            extra_msprof = self._collect_msprof_fn(
                parent_module_source,
                state.best_result,
                eff_anchors,
                [selected_spec],
            )
            if isinstance(extra_msprof, Mapping):
                for key, metrics in extra_msprof.items():
                    if isinstance(key, tuple) and len(key) == 2 and isinstance(metrics, Mapping):
                        msprof_by_case[(str(key[0]), int(key[1]))] = {
                            str(mk): float(mv)
                            for mk, mv in metrics.items()
                            if isinstance(mv, (int, float)) and not isinstance(mv, bool)
                        }

        parent_sha = self._source_sha256(parent_module_source)
        cached_parent_msprof = (
            msprof_by_case.get((parent_sha, int(bn_anchor.case_num)))
            if bn_anchor is not None
            else None
        )
        parent_lat_us = (
            selected_stone.latency_us
            if (selected_stone is not None and selected_stone.latency_us > 0.0)
            else local_best.mean_latency_us()
        )
        parent_obj_score = float(
            selected_stone.objective_score
            if (selected_stone is not None and selected_stone.objective_score is not None)
            else local_best.surrogate_score
        )
        parent_l1_summary = (
            compiler_summary_from_residual(selected_stone.residual)
            if selected_stone is not None
            else (
                self._l1_summary_fn(parent_module_source, selected_spec)
                if self._l1_summary_fn is not None
                else None
            )
        )
        parent_l1_summary_source = (
            "stepping_stone_residual"
            if selected_stone is not None and parent_l1_summary is not None
            else ("bisheng_l1" if parent_l1_summary is not None else "absent")
        )

        raw_msprof_dur = (
            float(cached_parent_msprof.get("dur_us", 0.0) or 0.0)
            if cached_parent_msprof
            else 0.0
        )
        if raw_msprof_dur > 0.0:
            case_dur_us = raw_msprof_dur
        elif bn_anchor is not None and bn_anchor.cand_us > 0.0:
            case_dur_us = float(bn_anchor.cand_us)
        else:
            case_dur_us = parent_lat_us

        has_same_case_roof = bn_anchor is not None and bn_anchor.t_hw_us > 0.0 and case_dur_us > 0.0
        if has_same_case_roof and bn_anchor is not None:
            eta_case = min(1.0, max(0.0, float(bn_anchor.t_hw_us) / case_dur_us))
        elif bn_anchor is not None:
            eta_case = float(bn_anchor.roofline_eta)
        else:
            eta_case = float(local_best.avg_roofline_eta)

        residual_roofline = None if has_same_case_roof else blueprint_roofline
        if cached_parent_msprof:
            telemetry_payload: dict[str, object] = {
                "dur_us": case_dur_us,
                "roofline_eta": eta_case,
            }
            if has_same_case_roof:
                telemetry_payload["effective_compute_eta"] = eta_case
            for k, v in cached_parent_msprof.items():
                if k == "dur_us" and float(v) <= 0.0:
                    continue
                if k in ("roofline_eta", "effective_compute_eta") and has_same_case_roof:
                    continue
                telemetry_payload[str(k)] = v
            telemetry_payload["dur_us"] = case_dur_us
            telemetry_payload["roofline_eta"] = eta_case
            parent_residual = self._compute_residual_with_roofline(
                parent_ledger,
                telemetry_payload,
                compiler_summary=parent_l1_summary,
                roofline=residual_roofline,
            )
        elif selected_stone is not None and (
            selected_stone.residual.measured_aiv_vec_ratio > 0.0
            or selected_stone.residual.measured_aic_mac_ratio > 0.0
            or selected_stone.residual.measured_mte2_wait_ratio > 0.0
            or selected_stone.residual.measured_sync_stall_ratio > 0.0
            or selected_stone.residual.spill_bytes > 0
            or selected_stone.residual.stack_bytes > 0
            or (not selected_stone.residual.vloopv2_hit)
            or selected_stone.residual.vf_fusion_breaks > 0
            or selected_stone.residual.alias_serialized_buffers > 0
            or selected_stone.residual.dual_issue_ipc_collapse > 0.0
            or selected_stone.residual.dcache_dcci_flushes > 0
            or bool(selected_stone.residual.has_local_membar_in_vf_call)
            or selected_stone.residual.scalar_compute_loop_fallback > 0.0
            or selected_stone.residual.dcache_bypass_narrow_store > 0.0
        ):
            parent_residual = selected_stone.residual
        else:
            telemetry_payload = {
                "dur_us": case_dur_us,
                "roofline_eta": eta_case,
            }
            if has_same_case_roof:
                telemetry_payload["effective_compute_eta"] = eta_case
            parent_residual = self._compute_residual_with_roofline(
                parent_ledger,
                telemetry_payload,
                compiler_summary=parent_l1_summary,
                roofline=residual_roofline,
            )

        ctx = build_predicate_context(
            parent_ledger,
            features=parent_features,
            residual=parent_residual,
            compiler_summary=parent_l1_summary,
            roofline=residual_roofline,
            extra={
                "topology_axes": parent_axes,
                **_case_ratio_predicate_entries(selected_spec.representative_case_ratios),
            },
        )
        if not gate_decision.engaged:
            ctx["use_uncalibrated_analytical"] = False
            if parent_l1_summary is None and not cached_parent_msprof:
                ctx["scalar_compute_loop_fallback"] = 0.0
                extra_sig_map = ctx.get("extra_signals")
                if isinstance(extra_sig_map, dict):
                    extra_sig_map.pop("scalar_compute_loop_fallback", None)

        ranked_families = tuple(rank_symptom_families(ctx))
        p2_escalated = should_escalate_diagnostic_or_creative(
            list(ranked_families),
            stagnation_rounds=stagnation_rounds,
        )
        parent_p2_diag: Mapping[str, object] | None = state.p2_cache.get(
            (parent_sha, selected_spec.regime_tag)
        )
        if p2_escalated and parent_p2_diag is None and self._collect_p2_fn is not None:
            parent_p2_diag = self._collect_p2_fn(
                parent_module_source,
                state.best_result,
                eff_anchors,
                selected_spec,
            )

        parent_comp_summary = compiler_summary_from_residual(parent_residual)
        parent_waterlines = getattr(parent_ledger, "waterlines", {}) or {}
        l1_wl = parent_waterlines.get("L1")
        ub_wl = parent_waterlines.get("UB")
        l1_overflow = bool(
            l1_wl is not None and int(getattr(l1_wl, "overflow_bytes", 0) or 0) > 0
        )
        ub_overflow = bool(
            ub_wl is not None and int(getattr(ub_wl, "overflow_bytes", 0) or 0) > 0
        )
        parent_workload = self._build_cost_workload(
            parent_ledger,
            live_plan=live_plan,
            axes=parent_axes,
            blueprint=blueprint,
        )

        if self._experience_lake_dir is not None:
            try:
                self._tracker.sync_from_experience_lake(
                    self._experience_lake_dir,
                    exclude_run_id=self._run_id,
                    allow_unreviewed_distilled_rules=(
                        self._config.allow_unreviewed_distilled_rules
                    ),
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round KB sync skipped: %s", exc)
            try:
                self._invoke_with_optional_kwargs(
                    self._cost_advisor.sync_experience_lake,
                    self._experience_lake_dir,
                    persist_calibration=False,
                    exclude_run_id=self._run_id,
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round cost replay skipped: %s", exc)

        parent_cost_pred = self._cost_advisor.predict_candidate_us(
            parent_workload,
            bisheng_res=parent_comp_summary,
        )
        parent_dominant_bound = str(
            self._cost_advisor.resolve_dominant_bound(
                parent_ledger,
                parent_residual,
                fallback_cost_pred=parent_cost_pred,
            )
            or ""
        )
        bp_semantics = getattr(blueprint, "semantics", None)
        op_semantics_for_shift = self._resolve_operator_semantics_for_gates()
        has_promotable_recurrence = bool(
            getattr(bp_semantics, "has_associative_outer_recurrence", False)
            or getattr(op_semantics_for_shift, "has_associative_outer_recurrence", False)
        )
        paradigm_shift_active = bool(
            stagnation_rounds >= 2
            and float(eta_case if bn_anchor is not None else local_best.avg_roofline_eta) < 0.15
            and has_promotable_recurrence
            and parent_axes.compute_engine == "aiv_only"
        )
        active_mask = self._resolve_trackb_freedom_mask(selected_spec)
        escalation_decision = None
        pipe_gap_report = None
        escalated_trackb_tu = ""
        if escalated:
            active_mask = relax_facets_for_bottleneck(
                active_mask,
                dominant_bound=parent_dominant_bound,
                residual=parent_residual,
                compiler_summary=parent_comp_summary,
                l1_overflow=l1_overflow,
                ub_overflow=ub_overflow,
            )
            for fam, _score in ranked_families:
                if not self._tracker.is_family_tabued(
                    regime_tag=selected_spec.regime_tag,
                    symptom_family=fam,
                    parent_sha=parent_sha,
                    current_round=round_idx,
                ):
                    fam_facets = SYMPTOM_FAMILY_TO_FACETS.get(fam, ())
                    if fam_facets:
                        active_mask = active_mask.unlock(*fam_facets)
                    break
            self._freedom_masks[selected_spec.regime_tag] = active_mask
            if normalize_runtime_backend(selected_spec.runtime_backend) in (
                "ascendc_trackb",
                "hybrid_trackb",
            ):
                current_tier = "ascendc_template"
            else:
                current_tier = normalize_runtime_backend(
                    str(live_plan.policy.get("backend_tier", "tilelang_dsl"))
                )
                if current_tier not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
                    current_tier = "tilelang_dsl"
                elif current_tier in ("ascendc_trackb", "hybrid_trackb"):
                    current_tier = "ascendc_template"
            sf_ctx = dict(parent_features) if isinstance(parent_features, Mapping) else {}
            sf_ctx["l1_overflow"] = l1_overflow
            sf_ctx["ub_overflow"] = ub_overflow
            sf_ctx["freedom_mask"] = active_mask
            sf_ctx.setdefault("uses_cube", parent_axes.uses_cube)
            case_ratios = selected_spec.representative_case_ratios
            if case_ratios is not None:
                sf_ctx["representative_case_id"] = case_ratios.case_id
                sf_ctx["inner_contiguous_bytes"] = case_ratios.inner_contiguous_bytes
                sf_ctx["persistent_wave_occupancy"] = case_ratios.wave_occupancy_ratio
            target_eta = (
                float(self._perf_target.target_roofline_eta)
                if self._perf_target is not None
                else 0.85
            )
            escalation_decision = evaluate_track_escalation(
                current_tier=current_tier,
                roofline_eta=float(
                    eta_case if bn_anchor is not None else local_best.avg_roofline_eta
                ),
                target_eta=target_eta,
                stagnation_rounds=int(stagnation_rounds),
                dominant_bound=parent_dominant_bound,
                compiler_summary=parent_comp_summary,
                structural_features=sf_ctx,
            )
            pipe_gap_report = scan_tilelang_pipe_gaps()
            if (
                escalation_decision is not None
                and bool(getattr(escalation_decision, "should_escalate", False))
                and str(getattr(escalation_decision, "to_tier", "")) == "ascendc_template"
                and _is_real_tir_primfunc(base_stage_a.prim_func)
            ):
                try:
                    raw_tu = escalate_primfunc_to_trackb_tu(base_stage_a.prim_func)
                    escalated_trackb_tu = build_trackb_context_view(
                        raw_tu,
                        regime_tag=selected_spec.regime_tag,
                        module_source=parent_module_source,
                        active_kernel=_active_trackb_kernel(
                            raw_tu,
                            raw_tu,
                            regime_tag=selected_spec.regime_tag,
                        ),
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Track-B PrimFunc escalation lowering skipped: %s", exc)
        else:
            self._freedom_masks.setdefault(selected_spec.regime_tag, active_mask)

        if paradigm_shift_active:
            active_mask = unlock_cross_topology_paradigm_shift(active_mask)
            self._freedom_masks[selected_spec.regime_tag] = active_mask

        sil_policy_section = self._format_sil_live_policy_section(
            live_plan=live_plan,
            active_mask=active_mask,
            escalation=escalation_decision,
            pipe_gaps=pipe_gap_report,
            cost_pred=parent_cost_pred,
            escalated_trackb_tu=escalated_trackb_tu,
            parent_ledger=parent_ledger,
            parent_residual=parent_residual,
        )
        ranked_rules = tuple(
            self._tracker.rank_applicable_rules(
                ctx,
                top_k=self._config.k_rule_workers,
                topology_axes=parent_axes,
                operator_name=self._config.operator_name,
                runtime_backend=selected_spec.runtime_backend,
                regime_tag=selected_spec.regime_tag,
                parent_sha=parent_sha,
                current_round=round_idx,
                symptom_ranking=list(ranked_families),
            )
        )
        selected_rules = ranked_rules[: self._config.k_rule_workers]
        worker_specs: list[tuple[str, str, AscendCausalRule | None, GenerationRequest]] = []
        for idx, rule in enumerate(selected_rules):
            worker_specs.append(
                (
                    f"r{round_idx}_rule_{idx}",
                    "rule_exploit",
                    rule,
                    self._build_exploit_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=selected_spec,
                        rule=rule,
                        best_result=local_best,
                        blueprint=blueprint,
                        parent_ledger=parent_ledger,
                        parent_residual=parent_residual,
                        track=parent_track,
                        incumbent_source=parent_module_source,
                        sil_policy_section=sil_policy_section,
                        effort=self._config.exploit_effort,
                        p2_diag=parent_p2_diag,
                    ),
                )
            )
        excluded = summarize_excluded_mechanisms(
            list(ranked_rules),
            regime_beam.exploited_signatures,
        )
        if gate_decision.engaged and gate_decision.allow_prompt_guidance:
            anchor_dims = (
                selected_spec.representative_anchor_dims or selected_spec.representative_hir_dims
            )
            prescreen_kwargs: dict[str, object] = dict(selected_spec.sample_builder_kwargs)
            prescreen_factory = func_name
            prescreen_pc: dict[str, object] = {}
            prescreen_cf: tuple[str, ...] = ()
            try:
                resolved_cfg = resolve_regime_config(
                    parent_module_source,
                    selected_spec,
                    dims=anchor_dims,
                )
                prescreen_kwargs.update(resolved_cfg.builder_kwargs)
                prescreen_factory = resolved_cfg.builder_name
                prescreen_pc = dict(resolved_cfg.pass_configs)
                prescreen_cf = resolved_cfg.compile_flags
            except (
                SyntaxError,
                AttributeError,
                ImportError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Failed to resolve prescreen config for regime %s: %s",
                    selected_spec.regime_tag,
                    exc,
                )
            var_bindings = build_anchor_var_bindings(anchor_dims, kwargs=prescreen_kwargs)
            prescreen_shapes = extract_ordered_tensor_shapes(
                case_entry=selected_spec.representative_case_entry,
                dims=anchor_dims,
                proto=self._operator_proto_cache,
            )
            prescreen_summary = build_prescreen_summary_from_kwargs(
                prescreen_kwargs,
                dsl_source=parent_module_source,
                factory_name=prescreen_factory,
                var_bindings=var_bindings or None,
                config_selector_name=selected_spec.config_selector_fn or None,
                dims=anchor_dims,
                pass_configs=prescreen_pc or None,
                compile_flags=prescreen_cf or None,
                cost_model=self._cost_advisor.underlying_cost_model,
                axes=parent_axes,
                live_plan=live_plan,
                blueprint=blueprint,
                sil_advisory=self._sil_advisory,
                ordered_tensor_shapes=prescreen_shapes,
            )
            if not prescreen_summary:
                advisor_guidance = self._cost_advisor.build_prompt_guidance(
                    decision=gate_decision,
                    base_cfg=prescreen_kwargs,
                    dims=anchor_dims,
                    regime_tag=selected_spec.regime_tag,
                    top_k=4,
                )
                if advisor_guidance:
                    prescreen_summary = str(advisor_guidance)
        else:
            prescreen_summary = ""

        deficit = max(0, self._config.k_rule_workers - len(ranked_rules))
        total_creative = (
            max(2, 1 + deficit)
            if (paradigm_shift_active or p2_escalated)
            else (1 + deficit)
        )
        for c_idx in range(total_creative):
            wid = f"r{round_idx}_creative" if c_idx == 0 else f"r{round_idx}_creative_{c_idx}"
            hint = (
                CREATIVE_FOCUS_HINTS[c_idx % len(CREATIVE_FOCUS_HINTS)]
                if total_creative > 1
                else ""
            )
            worker_specs.append(
                (
                    wid,
                    "creative_explore",
                    None,
                    self._build_creative_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=selected_spec,
                        best_result=local_best,
                        effort=creative_effort,
                        excluded_mechanisms=excluded,
                        focus_hint=hint,
                        prescreen_summary=prescreen_summary,
                        blueprint=blueprint,
                        parent_ledger=parent_ledger,
                        parent_residual=parent_residual,
                        track=parent_track,
                        incumbent_source=parent_module_source,
                        parent_hypothesis=selected_stone_hypothesis,
                        sil_policy_section=sil_policy_section,
                        p2_diag=parent_p2_diag,
                        paradigm_shift=paradigm_shift_active,
                    ),
                )
            )

        return RoundSynthesisBundle(
            regime_spec=selected_spec,
            func_name=func_name,
            parent_module_source=parent_module_source,
            parent_snippet=parent_snippet,
            parent_lane=parent_lane,
            selected_stone_id=selected_stone_id,
            selected_stone_hypothesis=selected_stone_hypothesis,
            parent_ledger=parent_ledger,
            parent_residual=parent_residual,
            parent_axes=parent_axes,
            ranked_families=ranked_families,
            ranked_rules=ranked_rules,
            worker_specs=tuple(worker_specs),
            prescreen_summary=prescreen_summary,
            gate_decision=gate_decision,
            live_plan=live_plan,
            blueprint=blueprint,
            parent_p2_diag=parent_p2_diag,
            track=parent_track,
            active_parent_id=active_parent_id,
            parent_sha=parent_sha,
            parent_lat_us=parent_lat_us,
            parent_obj_score=parent_obj_score,
            parent_l1_summary=parent_l1_summary,
            parent_l1_summary_source=parent_l1_summary_source,
            parent_features=parent_features if isinstance(parent_features, Mapping) else None,
            parent_workload=parent_workload,
            parent_cost_pred=parent_cost_pred,
            parent_dominant_bound=parent_dominant_bound,
            active_mask=active_mask,
            escalated=escalated,
            paradigm_shift_active=paradigm_shift_active,
            p2_escalated=p2_escalated,
            mean_recoverable_excess_us=mean_recoverable_excess_us,
            prev_local_rel_gain=prev_local_rel_gain,
            local_best=local_best,
            creative_effort=creative_effort,
            stagnation_rounds=stagnation_rounds,
        )


__all__ = [
    "ChampionState",
    "RoundContextBuilder",
    "RoundSynthesisBundle",
]
