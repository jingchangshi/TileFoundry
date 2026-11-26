"""Stage-gate coordinator for round LLM generation, L0/L1 filtering, L2 eval, and tail guards."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, TypeVar

import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    GuardrailCaseSpec,
    RegimeTargetSpec,
    _has_flat_operator_layout,
    build_anchor_var_bindings,
    compute_case_eq3_score,
    default_source_root,
    extract_trackb_cpp_source,
    get_operator_entrypoint,
    get_operator_task,
    lookup_case_dims,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.fault_types import (
    FaultPlane,
    TypedFault,
    coerce_fault_plane,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import DeterministicSearchBoundaryLedger
from tilefoundry_ascend.agent.campaign.promotion_coordinator import evaluate_promotion_gate
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AgentGatesDeviceFaultError,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L2BatchResults,
    aggregate_surrogate_result,
    extract_ordered_tensor_shapes,
    run_l0_l1_round_filter,
    run_l1_compile_check,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor, create_cost_advisor
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker,
    append_kb_event_shard,
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import select_active_trackb_kernel
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    evaluate_ascendc_candidate,
    resolve_trackb_compile_flags,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    GenerationRequest,
    KernelLLMBackend,
    OpenAICompatHTTPBackend,
    SubprocessCliTemplateBackend,
    create_llm_backend_from_env,
    extract_creative_payload,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    CostFeatures,
    build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import CostGateDecision
from tilefoundry_ascend.ascend.cost_model.experience_lake import compute_source_sha256
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePlanResult
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.sil.verifier import verify_source_against_facets
from tilefoundry_ascend.ascend.target import AscendTarget

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

_SHAPE_DIM_KEYS: frozenset[str] = frozenset(
    {
        "b",
        "b_dim",
        "case_id",
        "case_num",
        "d",
        "d_dim",
        "d_nope",
        "d_rope",
        "half_dim",
        "hidden_dim",
        "inner",
        "inner_dim",
        "k",
        "m",
        "n",
        "n_depth",
        "n_kv",
        "n_q",
        "s",
        "s_dim",
        "s_kv",
        "seq_len",
    }
)


class OccupancyExhaustedError(RuntimeError):
    """Typed infrastructure error raised when NPU card occupancy is exhausted."""

    fault_plane: str = "INFRA_OCCUPANCY"


class StageGateConfig(Protocol):
    """Configuration surface consumed by StageGateCoordinator."""

    operator_name: str
    device_id: str
    warmup: int
    repeat: int
    max_syntax_repairs: int
    cost_prescreen_top_k: int | None
    pure_golden: bool


class _ResolvedConfigLike(Protocol):
    builder_name: str
    builder_kwargs: Mapping[str, object]
    pass_configs: Mapping[str, object]
    compile_flags: tuple[str, ...]
    selector_name: str
    combine_kwargs: Mapping[str, object]


@dataclass(frozen=True)
class _DefaultStageGateConfig:
    operator_name: str = ""
    device_id: str = "1"
    warmup: int = 3
    repeat: int = 5
    max_syntax_repairs: int = 1
    cost_prescreen_top_k: int | None = None
    pure_golden: bool = False


@dataclass(frozen=True)
class CandidateBatch:
    """Inputs for one round of candidate generation and L0/L1/L2 stage gates."""

    round_idx: int
    regime_spec: RegimeTargetSpec
    worker_specs: Sequence[tuple[str, str, AscendCausalRule | None, GenerationRequest]]
    parent_module_source: str
    parent_snippet: str
    parent_ledger: ObjectivePhysicalLedger | None = None
    parent_axes: TopologyAxes | None = None
    active_mask: ConstraintRelaxationMask | None = None
    bn_dims: Mapping[str, int] | None = None
    live_plan: RegimePlanResult | None = None
    blueprint: HirBlueprintPacket | None = None
    gate_decision: CostGateDecision | None = None
    parent_p2_diag: Mapping[str, object] | None = None
    creative_candidate_ids: Collection[str] = ()
    func_name: str = ""


@dataclass(frozen=True)
class SpeculativeCreativeResult:
    """Outcome produced by a non-blocking speculative explore_creative task."""

    origin_round_idx: int
    worker_id: str
    snippet: str | None
    hypothesis: str = ""
    error: str = ""
    is_transport_error: bool = False


@dataclass(frozen=True)
class StageGateOutcome:
    """Aggregated results and telemetry from one round's L0/L1/L2 gate execution."""

    raw_candidates: list[tuple[str, str]]
    meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]]
    failed_entries: list[tuple[str, str, str]]
    l0_l1_checks: list[L0L1CandidateCheck]
    passed_checks: list[L0L1CandidateCheck]
    l0_l1_faults_by_cid: dict[str, TypedFault]
    l2_results: dict[str, CandidateSurrogateResult]
    l2_run_ids: dict[str, str]
    l2_infra_fault: TypedFault | None
    llm_gen_sec: float
    l0_l1_sec: float
    syntax_repair_sec: float
    l2_prewarm_sec: float
    l2_npu_sec: float
    round_prewarm_stats: dict[str, int]
    round_active_kernel: str
    infeasible_boundaries_added: int
    prescreen_pruned_list: list[str]
    knob_sig_by_cid: dict[str, str]
    ub_bytes_by_cid: dict[str, int]
    pred_by_cid: dict[str, float]
    cand_axes_by_cid: dict[str, TopologyAxes]


class StageGateCoordinator:
    """Coordinate round generation, speculative creative tasks, L0/L1/L2 gates, and tail checks."""

    def __init__(
        self,
        *,
        operator_name: str = "",
        target_module_path: Path | None = None,
        regime_specs: Sequence[RegimeTargetSpec] = (),
        initial_anchors: Sequence[AnchorCaseSpec] = (),
        scoreboard_anchors: Sequence[AnchorCaseSpec] = (),
        guardrail_cases: Sequence[GuardrailCaseSpec] = (),
        backend: KernelLLMBackend | None = None,
        tracker: BayesianRuleTracker | None = None,
        cost_advisor: CostModelAdvisor | None = None,
        boundary_ledger: DeterministicSearchBoundaryLedger | None = None,
        target: AscendTarget | None = None,
        source_root: Path | None = None,
        env_file: Path | None = None,
        cannbench_repo: Path | None = None,
        checkpoint_path: Path | None = None,
        experience_lake_dir: Path | None = None,
        run_id: str = "",
        device_id: str = "1",
        l4_device_pool: Sequence[str] = (),
        warmup: int = 3,
        repeat: int = 5,
        max_syntax_repairs: int = 1,
        cost_prescreen_top_k: int | None = None,
        pure_golden: bool = False,
        enable_track_router: bool = False,
        sil_advisory: object | None = None,
        topology_axes: TopologyAxes | None = None,
        semantic_topology_axes: TopologyAxes | None = None,
        l0_checker: Callable[..., StageAResult] | None = None,
        l1_compiler: Callable[..., StageBResult] | None = None,
        l2_batch_evaluator: Callable[
            [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
            Mapping[str, CandidateSurrogateResult],
        ]
        | None = None,
        static_width_dims_fn: Callable[[RegimeTargetSpec], Sequence[Mapping[str, object]]]
        | None = None,
        config: StageGateConfig | None = None,
        active_incumbent_source: str = "",
    ) -> None:
        self._config: StageGateConfig = (
            config
            if config is not None
            else _DefaultStageGateConfig(
                operator_name=operator_name,
                device_id=device_id,
                warmup=warmup,
                repeat=repeat,
                max_syntax_repairs=max_syntax_repairs,
                cost_prescreen_top_k=cost_prescreen_top_k,
                pure_golden=pure_golden,
            )
        )
        self._target = target or AscendTarget("ascend.950pr")
        self._source_root = (
            Path(source_root).resolve() if source_root is not None else default_source_root()
        )
        self._target_module_path = (
            Path(target_module_path).resolve()
            if target_module_path is not None
            else (self._source_root / "kernel.py")
        )
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self._experience_lake_dir = (
            Path(experience_lake_dir).resolve() if experience_lake_dir is not None else None
        )
        self._run_id = run_id
        self._regime_specs = tuple(regime_specs)
        self._initial_anchors = tuple(initial_anchors)
        self._scoreboard_anchors = (
            tuple(scoreboard_anchors) if scoreboard_anchors else tuple(initial_anchors)
        )
        self._guardrail_cases = list(guardrail_cases)
        self._backend = backend
        self._tracker = tracker or BayesianRuleTracker.from_rules(())
        self._cost_advisor = cost_advisor or create_cost_advisor(
            "off",
            target=self._target,
            operator_name=self._config.operator_name,
        )
        self._boundary_ledger = boundary_ledger or DeterministicSearchBoundaryLedger()
        self._l4_device_pool = tuple(l4_device_pool)
        self._enable_track_router = bool(enable_track_router)
        self._sil_advisory = sil_advisory
        self._topology_axes = topology_axes
        self._semantic_topology_axes = semantic_topology_axes
        self._l0_checker = l0_checker
        self._l1_compiler = l1_compiler
        self._l2_batch_evaluator = l2_batch_evaluator
        self._static_width_dims_fn = static_width_dims_fn

        self._active_incumbent_source: str = active_incumbent_source
        self._incumbent_anchor_us_by_case: dict[int, float] = {}
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._l1_summary_cache: dict[tuple[str, str], BishengResourceSummary | None] = {}
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        self._operator_semantics_cache: PyTorchOpSemantics | None = None
        self._operator_proto_cache: Mapping[str, object] | None = None
        self._operator_semantics_loaded: bool = False
        self._last_guardrail_metrics_by_cid: dict[str, dict[int, AnchorEvalMetrics]] = {}
        self._last_l2_timing_meta: dict[str, object] = {}
        self._last_tail_guard_reject_reason: str = ""
        self._pending_speculative: list[
            tuple[int, str, object, asyncio.Task[SpeculativeCreativeResult]]
        ] = []

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

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
        by_name = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in names:
            if name in by_name:
                return by_name[name]
        return None

    @staticmethod
    def _function_ast_dump(module_source: str, func_name: str) -> str:
        tree = ast.parse(module_source)
        node = StageGateCoordinator._find_top_level_func(tree, (func_name,))
        return ast.dump(node) if node is not None else ""

    @staticmethod
    def _resolve_cann_root_from_env_file(env_file: str | Path | None) -> Path | None:
        if env_file is None:
            return None
        p = Path(env_file)
        if not p.is_file():
            return None
        vals = parse_simple_env_file(p)
        setup_raw = vals.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
        if not setup_raw:
            return None
        parent = Path(setup_raw).expanduser().parent
        return parent if parent.is_dir() else None

    @staticmethod
    def _resolve_operator_entrypoint_safe(operator_name: str) -> str:
        try:
            return get_operator_entrypoint(operator_name)
        except (KeyError, FileNotFoundError):
            return ""

    @staticmethod
    def _ledger_has_cube_engine(ledger: ObjectivePhysicalLedger | None) -> bool:
        if ledger is None:
            return False
        return (
            int(ledger.pipe_cycles.get("AIC.PIPE_M", 0))
            + int(ledger.pipe_cycles.get("AIC.CUBE", 0))
        ) > 0

    @staticmethod
    def _ledger_has_hardware_dma(ledger: ObjectivePhysicalLedger | None) -> bool:
        if ledger is None:
            return False
        return (
            int(ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0))
            + int(ledger.pipe_cycles.get("AIV1.PIPE_MTE2", 0))
            + int(ledger.pipe_cycles.get("AIC.PIPE_MTE2", 0))
        ) > 0

    @staticmethod
    def _has_vf_stack_overflow(stage_b: StageBResult) -> bool:
        """Detect deterministic VF stack overflow from diagnostics or stack bytes."""
        if not stage_b.passed:
            diag = f"{stage_b.error_message}\n{stage_b.full_diagnostic}".lower()
            if (
                "exceeded vf stack size" in diag
                or "vf_stack" in diag
                or "stack frame size" in diag
                or "stack overflow" in diag
            ):
                return True
        if stage_b.compiler_summary is not None and int(stage_b.compiler_summary.stack_bytes) > 0:
            return True
        return False

    @staticmethod
    def _slice_contract_instruction(
        func_name: str,
        regime_spec: RegimeTargetSpec,
        *,
        operator_entrypoint: str = "",
    ) -> str:
        kernel_name_warning = ""
        if operator_entrypoint == "kernel":
            kernel_name_warning = (
                " CRITICAL: the public operator entrypoint is named 'kernel', so you must not "
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

    def _get_backend(self) -> KernelLLMBackend:
        if self._backend is None:
            self._backend = create_llm_backend_from_env()
        return self._backend

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

    def _lookup_case_dims_safe(
        self,
        case_num: int,
        regime_tag: str = "",
    ) -> dict[str, object]:
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

    def _belongs_to_regime(
        self,
        dims: Mapping[str, object],
        regime_spec: RegimeTargetSpec,
    ) -> bool:
        if len(self._regime_specs) <= 1:
            return True
        try:
            matched = resolve_regime_for_case(
                self._config.operator_name,
                dims,
                self._regime_specs,
            )
            return matched.regime_tag == regime_spec.regime_tag
        except (KeyError, ValueError, TypeError):
            return False

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

    def _extract_knob_signature(
        self,
        mod_src: str,
        regime_spec: RegimeTargetSpec,
    ) -> str:
        """Extract canonical sorted knob signature for deterministic boundary ledger."""
        if not mod_src.strip():
            return ""
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            resolve_regime_config,
        )

        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        try:
            resolved = resolve_regime_config(mod_src, regime_spec, dims=anchor_dims)
        except (
            SyntaxError,
            AttributeError,
            ImportError,
            KeyError,
            NameError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return ""
        parts: list[str] = []
        if resolved.selector_name:
            for key in sorted(resolved.builder_kwargs):
                if str(key).lower() in _SHAPE_DIM_KEYS:
                    continue
                parts.append(f"{key}={resolved.builder_kwargs[key]!r}")
        for key in sorted(resolved.pass_configs):
            parts.append(f"pc.{key}={resolved.pass_configs[key]!r}")
        if resolved.compile_flags:
            parts.append(
                f"flags={' '.join(sorted(str(flag) for flag in resolved.compile_flags))}"
            )
        return ",".join(parts)

    def _persist_round_candidate_artifact(
        self,
        check: L0L1CandidateCheck,
        *,
        round_idx: int,
    ) -> None:
        """Write generated source and untruncated L0/L1 diagnostics beside the checkpoint."""
        if self._checkpoint_path is None:
            return
        artifact_dir = self._checkpoint_path.parent / "candidates"
        stem = f"r{round_idx}_{check.candidate_id.removeprefix(f'r{round_idx}_')}"
        self._atomic_write_text(
            artifact_dir / f"{stem}.py",
            check.patched_module_source or check.snippet_source,
        )
        diag_lines = [
            f"candidate_id: {check.candidate_id}",
            f"round: {round_idx}",
            f"passed: {check.passed}",
            f"failed_tier: {check.failed_tier or ''}",
            f"repaired: {check.repaired}",
            f"error_message: {check.error_message}",
        ]
        if check.stage_a is not None:
            diag_lines.extend(
                [
                    f"stage_a_passed: {check.stage_a.passed}",
                    f"stage_a_backend_kind: {check.stage_a.backend_kind}",
                    f"stage_a_error: {check.stage_a.error_message}",
                ]
            )
        if check.stage_b is not None:
            diag_lines.extend(
                [
                    f"stage_b_passed: {check.stage_b.passed}",
                    f"stage_b_error: {check.stage_b.error_message}",
                    f"stage_b_spill_bytes: {check.stage_b.spill_bytes}",
                    f"stage_b_vloopv2_hit: {check.stage_b.vloopv2_hit}",
                ]
            )
            if check.stage_b.full_diagnostic:
                diag_lines.extend(
                    [
                        "stage_b_full_diagnostic:",
                        check.stage_b.full_diagnostic,
                    ]
                )
        self._atomic_write_text(
            artifact_dir / f"{stem}.diag.txt",
            "\n".join(diag_lines) + "\n",
        )

    def _record_kb_event(
        self,
        *,
        rule_id: str,
        postcondition_realized: bool,
        speed_gain: float = 0.0,
        topology_axes: object | None = None,
        round_idx: int = 0,
    ) -> None:
        if self._experience_lake_dir is None:
            return
        try:
            append_kb_event_shard(
                self._experience_lake_dir,
                run_id=self._run_id,
                rule_id=rule_id,
                postcondition_realized=postcondition_realized,
                speed_gain=speed_gain,
                topology_axes=topology_axes,
                operator_name=self._config.operator_name,
                round_idx=round_idx,
            )
        except (OSError, ValueError) as exc:
            logger.debug("KB event shard append skipped: %s", exc)

    def _resolve_combine_gate_spec(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        resolved: _ResolvedConfigLike,
    ) -> tuple[str, dict[str, object]] | None:
        """Return combine builder symbol and kwargs when combine gate applies."""
        combine_fn = next(
            (
                sym
                for sym in regime_spec.editable_symbols
                if sym.startswith("_build_") and "combine" in sym
            ),
            "",
        )
        if not combine_fn:
            return None
        cand_dump = self._function_ast_dump(module_source, combine_fn)
        if not cand_dump:
            return None
        splits = int(resolved.builder_kwargs.get("num_kv_splits", 1) or 1)
        modified = False
        baseline_src = self._active_incumbent_source
        if not baseline_src and self._target_module_path.is_file():
            try:
                baseline_src = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                baseline_src = ""
        if not baseline_src:
            modified = True
        else:
            base_dump = self._function_ast_dump(baseline_src, combine_fn)
            modified = cand_dump != base_dump
        if splits <= 1 and not modified:
            return None
        combine_splits = splits if splits > 1 else 4
        merged_kwargs: dict[str, object] = {
            **regime_spec.sample_builder_kwargs,
            **resolved.builder_kwargs,
            "num_kv_splits": combine_splits,
            **resolved.combine_kwargs,
        }
        mod_tree = ast.parse(module_source)
        fn_node = self._find_top_level_func(mod_tree, (combine_fn,))
        if fn_node is not None and fn_node.args.kwarg is None:
            accepted = {
                *(a.arg for a in fn_node.args.posonlyargs),
                *(a.arg for a in fn_node.args.args),
                *(a.arg for a in fn_node.args.kwonlyargs),
            }
            combine_kwargs = {k: v for k, v in merged_kwargs.items() if k in accepted}
        else:
            combine_kwargs = merged_kwargs
        return combine_fn, combine_kwargs

    def _instantiate_trackb_tu(
        self,
        module_source: str,
        cpp_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> str:
        """Instantiate Track-B C++20 TU via module _render_trackb_tu if present."""
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            _render_trackb_tu_from_module,
        )

        return _render_trackb_tu_from_module(
            module_source,
            cpp_source,
            regime_spec,
            require_renderer=False,
            anchor_dims_override=anchor_dims_override,
        )

    def _resolve_active_trackb_kernels(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> tuple[str, ...]:
        """Resolve ordered active Track-B ``__global__`` kernels for single or multi-stage TUs."""
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return ()
        extracted = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)
        if extracted is None:
            return ()
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            _ordered_top_level_trackb_kernels,
            _trackb_kernel_names,
        )

        rendered = self._instantiate_trackb_tu(module_source, extracted[1], regime_spec)
        kernels = _trackb_kernel_names(extracted[1])
        ordered = _ordered_top_level_trackb_kernels(rendered, extracted[1], kernels)
        if ordered:
            return ordered
        single = self._resolve_active_trackb_kernel(module_source, regime_spec)
        return (single,) if single else ()

    def _resolve_active_trackb_kernel(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> str | None:
        """Resolve the launched Track-B kernel from instantiation truth, else the raw TU."""
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return None
        extracted = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)
        if extracted is None:
            return None
        cache_key = (self._source_sha256(module_source), regime_spec.regime_tag)
        if cache_key not in self._active_kernel_cache:
            from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
                _active_trackb_kernel,
                _ordered_top_level_trackb_kernels,
                _trackb_kernel_names,
            )

            rendered = self._instantiate_trackb_tu(module_source, extracted[1], regime_spec)
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

    def _resolve_trackb_freedom_mask(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> ConstraintRelaxationMask:
        """Return active 7-facet mask for *regime_spec*, seeding Track-B baseline facets once."""
        existing = self._freedom_masks.get(regime_spec.regime_tag)
        if existing is not None:
            return existing
        baseline_src = self._active_incumbent_source
        if not baseline_src and self._target_module_path.is_file():
            baseline_src = self._target_module_path.read_text(encoding="utf-8")
        if baseline_src and regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb"):
            base_tb = extract_trackb_cpp_source(baseline_src, regime_spec.trackb_symbols)
            if base_tb is not None:
                base_tu = self._instantiate_trackb_tu(baseline_src, base_tb[1], regime_spec)
                rep = verify_source_against_facets(base_tu, ConstraintRelaxationMask.bottom())
                fixes = rep.suggested_fixes
                if not rep.passed and fixes:
                    return ConstraintRelaxationMask(
                        alg_compute_and_monoid=True,
                        tile_and_stage_knobs=True,
                        exec_mode_and_spr=bool(fixes.get("unlock_exec_mode_and_spr", False)),
                        sram_layout_and_aliasing=bool(
                            fixes.get("unlock_sram_layout_and_aliasing", False)
                        ),
                        dma_and_cache_hierarchy=bool(
                            fixes.get("unlock_dma_and_cache_hierarchy", False)
                        ),
                        pipeline_and_sync_graph=bool(
                            fixes.get("unlock_pipeline_and_sync_graph", False)
                        ),
                        grid_and_workspace_topology=bool(
                            fixes.get("unlock_grid_and_workspace_topology", False)
                        ),
                    )
        return ConstraintRelaxationMask.bottom()

    def _resolve_trackb_baseline_tu(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> str | None:
        """Return baseline Track-B TU for differential forbidden-token checks."""
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return None
        baseline_src = self._active_incumbent_source
        if not baseline_src and self._target_module_path.is_file():
            try:
                baseline_src = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                baseline_src = ""
        if not baseline_src:
            return None
        base_tb = extract_trackb_cpp_source(baseline_src, regime_spec.trackb_symbols)
        if base_tb is None:
            return None
        return self._instantiate_trackb_tu(baseline_src, base_tb[1], regime_spec)

    def _compile_trackb_stage_b(
        self,
        cpp_source: str,
        selector_flags: Sequence[str] | str | None = None,
    ) -> StageBResult:
        cann_root = self._resolve_cann_root_from_env_file(self._env_file)
        bisheng_bin: str | None = None
        flags = list(resolve_trackb_compile_flags(cpp_source, selector_flags))
        if cann_root is not None:
            cand_bin = cann_root / "bin" / "bisheng"
            if cand_bin.is_file():
                bisheng_bin = str(cand_bin)
            if (cann_root / "include").is_dir():
                flags.append(f"-I{cann_root / 'include'}")
        return run_stage_b_ascendc_compile(
            cpp_source,
            compile_flags=flags,
            bisheng_bin=bisheng_bin,
            cann_root=cann_root,
        )

    def _resolve_trackb_selector_flags(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> tuple[str, ...]:
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            _resolve_selector_compile_flags,
        )

        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        return _resolve_selector_compile_flags(module_source, regime_spec, dims=anchor_dims)

    def _resolve_operator_semantics_for_gates(self) -> PyTorchOpSemantics | None:
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

    def _default_l0_checker(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            _has_prim_func_builder,
            resolve_regime_config,
        )

        op_semantics = self._resolve_operator_semantics_for_gates()
        req_dma = bool(op_semantics is not None and not op_semantics.has_indirect_index)
        req_cube = bool(op_semantics is not None and op_semantics.has_gemm_contraction)
        is_hybrid = regime_spec.runtime_backend == "hybrid_trackb"
        has_prim_func = _has_prim_func_builder(module_source, regime_spec.builder_candidates)
        tb_extracted = (
            extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)
            if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            else None
        )
        baseline_tb_tu = (
            self._resolve_trackb_baseline_tu(regime_spec) if tb_extracted is not None else None
        )
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_extracted is not None:
            active_mask = self._resolve_trackb_freedom_mask(regime_spec)
            instantiated_tu = self._invoke_with_optional_kwargs(
                self._instantiate_trackb_tu,
                module_source,
                tb_extracted[1],
                regime_spec,
                anchor_dims_override=anchor_dims_override,
            )
            return evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=active_mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
                baseline_cpp_source=baseline_tb_tu,
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
        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=anchor_dims)
        except (
            SyntaxError,
            AttributeError,
            ImportError,
            KeyError,
            NameError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=f"stage_a_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        var_bindings = build_anchor_var_bindings(
            anchor_dims,
            kwargs=resolved.builder_kwargs,
        )
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
        if not primary_res.passed:
            if is_hybrid and tb_extracted is not None and not has_prim_func:
                active_mask = self._resolve_trackb_freedom_mask(regime_spec)
                instantiated_tu = self._invoke_with_optional_kwargs(
                    self._instantiate_trackb_tu,
                    module_source,
                    tb_extracted[1],
                    regime_spec,
                    anchor_dims_override=anchor_dims_override,
                )
                return evaluate_ascendc_candidate(
                    instantiated_tu,
                    freedom_mask=active_mask,
                    target=self._target,
                    enable_track_router=self._enable_track_router,
                    case_ratios=regime_spec.representative_case_ratios,
                    baseline_cpp_source=baseline_tb_tu,
                    require_hardware_dma=req_dma,
                    require_cube_engine=req_cube,
                    module_source=module_source,
                    regime_tag=regime_spec.regime_tag,
                )
            return primary_res
        combine_spec = self._resolve_combine_gate_spec(module_source, regime_spec, resolved)
        if combine_spec is not None:
            combine_fn, combine_kwargs = combine_spec
            combine_res = run_stage_a_static(
                module_source,
                factory_name=combine_fn,
                factory_kwargs=combine_kwargs,
                pass_configs=resolved.pass_configs or None,
                compile_flags=resolved.compile_flags or None,
                target=self._target,
                var_bindings=var_bindings or None,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
                require_hardware_dma=req_dma,
                require_cube_engine=False,
                ordered_tensor_shapes=ordered_tensor_shapes,
            )
            if not combine_res.passed:
                return StageAResult(
                    passed=False,
                    prim_func=primary_res.prim_func,
                    ledger=primary_res.ledger,
                    features=primary_res.features,
                    route=primary_res.route,
                    error_message=f"combine_kernel({combine_fn}): {combine_res.error_message}",
                    elapsed_ms=primary_res.elapsed_ms + combine_res.elapsed_ms,
                )
        if is_hybrid and tb_extracted is not None:
            tb_sym, tb_cpp = tb_extracted
            active_mask = self._resolve_trackb_freedom_mask(regime_spec)
            instantiated_tu = self._invoke_with_optional_kwargs(
                self._instantiate_trackb_tu,
                module_source,
                tb_cpp,
                regime_spec,
                anchor_dims_override=anchor_dims_override,
            )
            tb_res = evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=active_mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
                baseline_cpp_source=baseline_tb_tu,
                require_hardware_dma=req_dma,
                require_cube_engine=False,
                module_source=module_source,
                regime_tag=regime_spec.regime_tag,
            )
            total_ms = primary_res.elapsed_ms + tb_res.elapsed_ms
            if not tb_res.passed:
                return replace(
                    tb_res,
                    error_message=f"trackb_ascendc({tb_sym}): {tb_res.error_message}",
                    elapsed_ms=total_ms,
                )
            if req_cube and not (
                self._ledger_has_cube_engine(primary_res.ledger)
                or self._ledger_has_cube_engine(tb_res.ledger)
            ):
                return replace(
                    tb_res,
                    passed=False,
                    error_message=(
                        f"trackb_ascendc({tb_sym}): missing_cube_engine_primitives: "
                        "aggregate hybrid_trackb candidate has zero AIC.PIPE_M cycles"
                    ),
                    elapsed_ms=total_ms,
                )
            if req_dma and not (
                self._ledger_has_hardware_dma(primary_res.ledger)
                or self._ledger_has_hardware_dma(tb_res.ledger)
            ):
                return replace(
                    tb_res,
                    passed=False,
                    error_message=(
                        f"trackb_ascendc({tb_sym}): missing_hardware_dma_staging: "
                        "aggregate hybrid_trackb candidate has zero PIPE_MTE2 cycles"
                    ),
                    elapsed_ms=total_ms,
                )
            return replace(
                tb_res,
                elapsed_ms=total_ms,
            )
        return primary_res

    def _run_l0(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        if self._l0_checker is not None:
            return self._invoke_with_optional_kwargs(
                self._l0_checker,
                module_source,
                regime_spec,
                anchor_dims_override=anchor_dims_override,
            )
        return self._default_l0_checker(
            module_source,
            regime_spec,
            anchor_dims_override=anchor_dims_override,
        )

    def _resolve_max_row_width_dims(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> dict[str, object] | None:
        """Return the largest-row-width case dims routed to ``regime_spec``."""
        candidates: list[dict[str, object]] = []
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is not None:
            try:
                task = get_operator_task(self._config.operator_name)
                cases_path = tasks_root / task / "cases.yaml"
                if cases_path.is_file():
                    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
                    for row in parsed.get("cases") or ():
                        if isinstance(row, Mapping) and row.get("case_id") is not None:
                            cnum = int(row["case_id"])
                            dims = self._lookup_case_dims_safe(cnum, regime_spec.regime_tag)
                            if dims and self._belongs_to_regime(dims, regime_spec):
                                candidates.append(dict(dims))
            except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
                pass
        for anc in self._initial_anchors:
            if len(self._regime_specs) > 1 and anc.regime_tag != regime_spec.regime_tag:
                continue
            dims = self._lookup_case_dims_safe(int(anc.case_num), anc.regime_tag)
            if dims and self._belongs_to_regime(dims, regime_spec):
                candidates.append(dict(dims))
        own_rep = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        if own_rep:
            candidates.append(dict(own_rep))
        if not candidates:
            return None

        row_width_keys = (
            "D",
            "d_dim",
            "K",
            "N",
            "hidden_dim",
            "d_nope",
            "d_rope",
            "inner",
            "inner_dim",
            "S",
            "s_dim",
            "seq_len",
        )

        def _row_width_score(dims: Mapping[str, object]) -> tuple[int, int]:
            width = 1
            has_width_key = False
            for key in row_width_keys:
                val = dims.get(key)
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    width *= val
                    has_width_key = True
            total_prod = 1
            for val in dims.values():
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    total_prod *= val
            return (width if has_width_key else total_prod, total_prod)

        return max(candidates, key=_row_width_score)

    def _resolve_tail_anchor_dims(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> dict[str, object] | None:
        """Return the smallest-element case dims routed to ``regime_spec`` when distinct."""
        primary_dims = dict(
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        candidates: list[dict[str, object]] = []
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is not None:
            try:
                task = get_operator_task(self._config.operator_name)
                cases_path = tasks_root / task / "cases.yaml"
                if cases_path.is_file():
                    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
                    for row in parsed.get("cases") or ():
                        if isinstance(row, Mapping) and row.get("case_id") is not None:
                            cnum = int(row["case_id"])
                            dims = self._lookup_case_dims_safe(cnum, regime_spec.regime_tag)
                            if dims and self._belongs_to_regime(dims, regime_spec):
                                candidates.append(dict(dims))
            except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
                pass
        for anc in self._scoreboard_anchors:
            if len(self._regime_specs) > 1 and anc.regime_tag != regime_spec.regime_tag:
                continue
            dims = self._lookup_case_dims_safe(int(anc.case_num), anc.regime_tag)
            if dims and self._belongs_to_regime(dims, regime_spec):
                candidates.append(dict(dims))
        if regime_spec.representative_hir_dims:
            candidates.append(dict(regime_spec.representative_hir_dims))
        if not candidates:
            return None

        def _elem_score(dims: Mapping[str, object]) -> tuple[int, int]:
            prod = 1
            count = 0
            for val in dims.values():
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    prod *= val
                    count += 1
            return (prod if count > 0 else 0, len(dims))

        smallest = min(candidates, key=_elem_score)
        if smallest == primary_dims:
            return None
        return smallest

    def _collect_distinct_width_dims(
        self,
        regime_spec: RegimeTargetSpec,
        anchor_dims: Mapping[str, object] | None,
        uses_unreserved_ubuf: bool,
        static_width_dims_fn: Callable[[RegimeTargetSpec], Sequence[Mapping[str, object]]]
        | None,
    ) -> list[dict[str, object]]:
        seen: list[dict[str, object]] = [dict(anchor_dims)] if anchor_dims else []
        targets: list[dict[str, object]] = []
        if static_width_dims_fn is not None:
            for item in static_width_dims_fn(regime_spec):
                cand = dict(item)
                if cand and cand not in seen:
                    seen.append(cand)
                    targets.append(cand)
        if uses_unreserved_ubuf:
            max_dims = self._resolve_max_row_width_dims(regime_spec)
            if max_dims and dict(max_dims) not in seen:
                seen.append(dict(max_dims))
                targets.append(dict(max_dims))
        return targets

    def _build_width_failure_result(
        self,
        width_res: StageBResult,
        total_elapsed: float,
    ) -> StageBResult:
        if self._has_vf_stack_overflow(width_res):
            stack_bytes = (
                int(width_res.compiler_summary.stack_bytes)
                if width_res.compiler_summary is not None
                else 0
            )
            err_msg = (
                f"stage_b_max_row_width_vf_stack_overflow: stack_bytes={stack_bytes} "
                f"({width_res.error_message})"
            )
            return StageBResult(
                passed=False,
                error_message=err_msg,
                full_diagnostic=width_res.full_diagnostic,
                compiler_summary=width_res.compiler_summary,
                elapsed_sec=total_elapsed,
                fault_plane="STAGE_B_COMPILE",
                fault_code="VF_STACK_OVERFLOW",
                failed_tier="stage_b",
            )
        return StageBResult(
            passed=False,
            error_message=f"stage_b_max_row_width_compile_failed: {width_res.error_message}",
            full_diagnostic=width_res.full_diagnostic,
            compiler_summary=width_res.compiler_summary,
            elapsed_sec=total_elapsed,
            fault_plane=width_res.fault_plane or "STAGE_B_COMPILE",
            fault_code=width_res.fault_code or "STAGE_B_COMPILE_FAILED",
            failed_tier=width_res.failed_tier or "stage_b",
        )

    def _normalize_primary_vf_overflow(self, res: StageBResult) -> StageBResult:
        if not res.passed and self._has_vf_stack_overflow(res):
            return replace(
                res,
                fault_plane=res.fault_plane or "STAGE_B_COMPILE",
                fault_code="VF_STACK_OVERFLOW",
                failed_tier=res.failed_tier or "stage_b",
            )
        return res

    def _run_l1(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        static_width_dims_fn: Callable[[RegimeTargetSpec], Sequence[Mapping[str, object]]]
        | None = None,
    ) -> StageBResult:
        """Run Stage-B L1 compilation across anchor, tail, and distinct static width dims."""
        eff_width_fn = (
            static_width_dims_fn
            if static_width_dims_fn is not None
            else self._static_width_dims_fn
        )
        tail_anchor_dims = self._resolve_tail_anchor_dims(regime_spec)
        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        if self._l1_compiler is not None:
            primary_b = self._invoke_with_optional_kwargs(
                self._l1_compiler,
                module_source,
                regime_spec,
                tail_anchor_dims=tail_anchor_dims,
            )
            if not primary_b.passed:
                return self._normalize_primary_vf_overflow(primary_b)
            if self._has_vf_stack_overflow(primary_b):
                return self._build_width_failure_result(primary_b, primary_b.elapsed_sec)
            total_elapsed = primary_b.elapsed_sec
            for w_dims in self._collect_distinct_width_dims(
                regime_spec, anchor_dims, False, eff_width_fn
            ):
                w_spec = replace(regime_spec, representative_anchor_dims=dict(w_dims))
                w_b = self._invoke_with_optional_kwargs(
                    self._l1_compiler,
                    module_source,
                    w_spec,
                    dims=dict(w_dims),
                    tail_anchor_dims=tail_anchor_dims,
                )
                total_elapsed += w_b.elapsed_sec
                if not w_b.passed or self._has_vf_stack_overflow(w_b):
                    return self._build_width_failure_result(w_b, total_elapsed)
            if total_elapsed != primary_b.elapsed_sec:
                return replace(primary_b, elapsed_sec=total_elapsed)
            return primary_b

        tb_extracted = (
            extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)
            if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            else None
        )
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_extracted is not None:
            instantiated_tu = self._instantiate_trackb_tu(
                module_source, tb_extracted[1], regime_spec
            )
            sel_flags = self._resolve_trackb_selector_flags(module_source, regime_spec)
            effective_flags = resolve_trackb_compile_flags(instantiated_tu, sel_flags)
            tb_res = (
                self._compile_trackb_stage_b(instantiated_tu, selector_flags=sel_flags)
                if sel_flags
                else self._compile_trackb_stage_b(instantiated_tu)
            )
            if not tb_res.passed:
                return self._normalize_primary_vf_overflow(tb_res)
            uses_unreserved_ubuf = any(
                "--cce-disable-vf-stack-reserved-ubuf" in str(fl) for fl in effective_flags
            )
            if uses_unreserved_ubuf and self._has_vf_stack_overflow(tb_res):
                return self._build_width_failure_result(tb_res, tb_res.elapsed_sec)
            total_elapsed = tb_res.elapsed_sec
            for w_dims in self._collect_distinct_width_dims(
                regime_spec, anchor_dims, uses_unreserved_ubuf, eff_width_fn
            ):
                w_spec = replace(regime_spec, representative_anchor_dims=dict(w_dims))
                w_tu = self._instantiate_trackb_tu(module_source, tb_extracted[1], w_spec)
                w_flags = self._resolve_trackb_selector_flags(module_source, w_spec)
                w_eff_flags = resolve_trackb_compile_flags(w_tu, w_flags)
                if w_tu != instantiated_tu or w_eff_flags != effective_flags:
                    w_b = (
                        self._compile_trackb_stage_b(w_tu, selector_flags=w_flags)
                        if w_flags
                        else self._compile_trackb_stage_b(w_tu)
                    )
                    total_elapsed += w_b.elapsed_sec
                    if not w_b.passed or self._has_vf_stack_overflow(w_b):
                        return self._build_width_failure_result(w_b, total_elapsed)
            if total_elapsed != tb_res.elapsed_sec:
                return replace(tb_res, elapsed_sec=total_elapsed)
            return tb_res

        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            _has_prim_func_builder,
            resolve_regime_config,
        )

        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=anchor_dims)
        except (
            SyntaxError,
            AttributeError,
            ImportError,
            KeyError,
            NameError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return StageBResult(
                passed=False,
                error_message=f"l1_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        primary_b = self._invoke_with_optional_kwargs(
            run_l1_compile_check,
            module_source,
            builder_name=resolved.builder_name,
            builder_kwargs=resolved.builder_kwargs,
            config_selector_name=regime_spec.config_selector_fn or None,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            source_root=self._source_root,
            env_file=self._env_file,
            tail_anchor_dims=tail_anchor_dims,
        )
        if not primary_b.passed:
            if (
                regime_spec.runtime_backend == "hybrid_trackb"
                and tb_extracted is not None
                and not _has_prim_func_builder(module_source, regime_spec.builder_candidates)
            ):
                instantiated_tu = self._instantiate_trackb_tu(
                    module_source, tb_extracted[1], regime_spec
                )
                tb_fb = (
                    self._compile_trackb_stage_b(
                        instantiated_tu,
                        selector_flags=resolved.compile_flags,
                    )
                    if resolved.compile_flags
                    else self._compile_trackb_stage_b(instantiated_tu)
                )
                return self._normalize_primary_vf_overflow(tb_fb)
            return self._normalize_primary_vf_overflow(primary_b)

        final_b = primary_b
        uses_unreserved_ubuf = any(
            "--cce-disable-vf-stack-reserved-ubuf" in str(fl) for fl in resolved.compile_flags
        )
        if uses_unreserved_ubuf and self._has_vf_stack_overflow(final_b):
            return self._build_width_failure_result(final_b, final_b.elapsed_sec)
        total_elapsed = final_b.elapsed_sec
        for w_dims in self._collect_distinct_width_dims(
            regime_spec, anchor_dims, uses_unreserved_ubuf, eff_width_fn
        ):
            try:
                w_resolved = resolve_regime_config(module_source, regime_spec, dims=w_dims)
            except (
                SyntaxError,
                AttributeError,
                ImportError,
                KeyError,
                NameError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                w_resolved = resolved
            if w_resolved != resolved or eff_width_fn is not None:
                w_b = run_l1_compile_check(
                    module_source,
                    builder_name=w_resolved.builder_name,
                    builder_kwargs=w_resolved.builder_kwargs,
                    config_selector_name=regime_spec.config_selector_fn or None,
                    pass_configs=w_resolved.pass_configs or None,
                    compile_flags=w_resolved.compile_flags or None,
                    dims=w_dims,
                    source_root=self._source_root,
                    env_file=self._env_file,
                )
                total_elapsed += w_b.elapsed_sec
                if not w_b.passed or self._has_vf_stack_overflow(w_b):
                    return self._build_width_failure_result(w_b, total_elapsed)

        combine_elapsed_sec = 0.0
        combine_spec = self._resolve_combine_gate_spec(module_source, regime_spec, resolved)
        if combine_spec is not None:
            combine_fn, combine_kwargs = combine_spec
            combine_b = run_l1_compile_check(
                module_source,
                builder_name=combine_fn,
                builder_kwargs=combine_kwargs,
                config_selector_name=None,
                pass_configs=resolved.pass_configs or None,
                compile_flags=resolved.compile_flags or None,
                source_root=self._source_root,
                env_file=self._env_file,
            )
            if not combine_b.passed:
                return StageBResult(
                    passed=False,
                    error_message=f"combine_kernel({combine_fn}): {combine_b.error_message}",
                    full_diagnostic=combine_b.full_diagnostic,
                    elapsed_sec=total_elapsed + combine_b.elapsed_sec,
                )
            combine_elapsed_sec = combine_b.elapsed_sec
        if regime_spec.runtime_backend == "hybrid_trackb" and tb_extracted is not None:
            tb_sym, tb_cpp = tb_extracted
            instantiated_tu = self._instantiate_trackb_tu(module_source, tb_cpp, regime_spec)
            tb_b = (
                self._compile_trackb_stage_b(
                    instantiated_tu,
                    selector_flags=resolved.compile_flags,
                )
                if resolved.compile_flags
                else self._compile_trackb_stage_b(instantiated_tu)
            )
            agg_elapsed_sec = total_elapsed + combine_elapsed_sec + tb_b.elapsed_sec
            if not tb_b.passed:
                return self._normalize_primary_vf_overflow(
                    replace(
                        tb_b,
                        error_message=f"trackb_ascendc({tb_sym}): {tb_b.error_message}",
                        elapsed_sec=agg_elapsed_sec,
                    )
                )
            return replace(tb_b, elapsed_sec=agg_elapsed_sec)
        if total_elapsed != final_b.elapsed_sec:
            return replace(final_b, elapsed_sec=total_elapsed)
        return final_b

    def _l1_summary_for_first_round(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> BishengResourceSummary | None:
        """Compile the parent once per source when no promoted stone carries L1 data."""
        cache_key = (self._source_sha256(module_source), regime_spec.regime_tag)
        if cache_key not in self._l1_summary_cache:
            result = self._run_l1(module_source, regime_spec)
            if not result.passed:
                logger.warning(
                    "First-round parent L1 for regime %s failed: %s",
                    regime_spec.regime_tag,
                    result.error_message,
                )
            self._l1_summary_cache[cache_key] = (
                result.compiler_summary if result.passed else None
            )
        return self._l1_summary_cache[cache_key]

    def _normalize_paired_l2_results(
        self,
        raw_results: Mapping[str, CandidateSurrogateResult],
        incumbent_slot_result: CandidateSurrogateResult | None,
    ) -> dict[str, CandidateSurrogateResult]:
        """Re-anchor candidate latencies via paired_speedup_vs_incumbent when available."""
        inc_slot_by_case = (
            {
                int(a.case_num): float(a.cand_us)
                for a in incumbent_slot_result.anchors
                if a.cand_us > 0.0
            }
            if incumbent_slot_result is not None
            else {}
        )
        normalized: dict[str, CandidateSurrogateResult] = {}
        for cid, cand_res in raw_results.items():
            changed = False
            updated_anchors: list[AnchorEvalMetrics] = []
            for anc in cand_res.anchors:
                paired_sp = float(getattr(anc, "paired_speedup_vs_incumbent", 0.0) or 0.0)
                inc_base_us = self._incumbent_anchor_us_by_case.get(
                    int(anc.case_num),
                    inc_slot_by_case.get(int(anc.case_num), 0.0),
                )
                if paired_sp > 0.0 and inc_base_us > 0.0 and anc.accuracy_passed:
                    eff_us = inc_base_us / paired_sp
                    eff_speedup = (
                        (anc.ref_perf_us / eff_us) if anc.ref_perf_us > 0.0 else anc.speedup
                    )
                    eff_eta = (
                        (anc.t_hw_us / eff_us) if anc.t_hw_us > 0.0 else anc.roofline_eta
                    )
                    eff_eq3 = compute_case_eq3_score(anc.ref_perf_us, eff_us, anc.t_hw_us)
                    updated_anchors.append(
                        replace(
                            anc,
                            cand_us=eff_us,
                            speedup=eff_speedup,
                            roofline_eta=eff_eta,
                            eq3_score=eff_eq3,
                        )
                    )
                    changed = True
                else:
                    updated_anchors.append(anc)
            if changed:
                normalized[cid] = aggregate_surrogate_result(
                    cand_res.candidate_id,
                    cand_res.operator_name,
                    tuple(updated_anchors),
                    error_message=cand_res.error_message,
                )
            else:
                normalized[cid] = cand_res
        return normalized

    def _run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        include_guardrails: bool = True,
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        """Evaluate one batch and return results plus each measured candidate's run id."""
        eff_inc_src = self._active_incumbent_source
        if (
            not eff_inc_src
            and self._l2_batch_evaluator is None
            and self._target_module_path.is_file()
        ):
            try:
                eff_inc_src = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                eff_inc_src = ""
        cand_ids = {cid for cid, _path in candidates}
        should_prepend_incumbent = bool(
            len(candidates) >= 1
            and eff_inc_src.strip()
            and "incumbent_slot" not in cand_ids
            and "__incumbent__" not in cand_ids
            and "incumbent_baseline" not in cand_ids
        )

        scoreboard_case_nums = {int(a.case_num) for a in active_anchors}
        extra_guardrails: list[AnchorCaseSpec] = []
        guardrail_case_nums: set[int] = set()
        if include_guardrails and self._guardrail_cases and active_anchors:
            for g in self._guardrail_cases:
                g_cnum = int(g.case_num)
                if g_cnum not in scoreboard_case_nums and g_cnum not in guardrail_case_nums:
                    extra_guardrails.append(g.to_anchor_spec())
                    guardrail_case_nums.add(g_cnum)
        eval_anchors = (
            [*active_anchors, *extra_guardrails] if extra_guardrails else list(active_anchors)
        )

        l2_meta: dict[str, object] = {
            "prewarm_stats": {},
            "l2_prewarm_sec": 0.0,
            "l2_npu_sec": None,
        }

        def _capture_l2_meta(evaluated: object) -> None:
            raw_prewarm = getattr(evaluated, "prewarm_stats", None)
            if isinstance(raw_prewarm, Mapping):
                l2_meta["prewarm_stats"] = {
                    str(k): int(v)
                    for k, v in raw_prewarm.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            raw_pw_sec = getattr(evaluated, "l2_prewarm_sec", None)
            if isinstance(raw_pw_sec, (int, float)) and not isinstance(raw_pw_sec, bool):
                l2_meta["l2_prewarm_sec"] = float(raw_pw_sec)
            raw_npu_sec = getattr(evaluated, "l2_npu_sec", None)
            if isinstance(raw_npu_sec, (int, float)) and not isinstance(raw_npu_sec, bool):
                l2_meta["l2_npu_sec"] = float(raw_npu_sec)

        def _dispatch(
            batch_cands: Sequence[tuple[str, Path]],
        ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
            if self._l2_batch_evaluator is not None:
                evaluated = self._l2_batch_evaluator(batch_cands, eval_anchors)
                _capture_l2_meta(evaluated)
                if isinstance(evaluated, L2BatchResults):
                    return dict(evaluated), dict(evaluated.run_ids)
                return dict(evaluated), {}
            results, run_id, _receipt = self._invoke_with_optional_kwargs(
                run_surrogate_batch_via_agent_gates,
                self._config.operator_name,
                batch_cands,
                eval_anchors,
                source_root=self._source_root,
                env_file=self._env_file,
                device_id=self._config.device_id,
                device_pool=self._l4_device_pool,
                warmup=self._config.warmup,
                repeat=self._config.repeat,
                require_all_passed=False,
            )
            _capture_l2_meta(results)
            return results, {candidate_id: run_id for candidate_id in results}

        if should_prepend_incumbent:
            with tempfile.TemporaryDirectory(prefix="tf_camp_inc_") as inc_dir:
                inc_path = Path(inc_dir) / "incumbent_slot.py"
                inc_path.write_text(eff_inc_src, encoding="utf-8")
                results, run_ids = _dispatch([("incumbent_slot", inc_path), *candidates])
        else:
            results, run_ids = _dispatch(candidates)

        self._last_l2_timing_meta = l2_meta
        inc_slot_result = results.pop("incumbent_slot", None)
        run_ids.pop("incumbent_slot", None)
        normalized = self._normalize_paired_l2_results(results, inc_slot_result)
        if guardrail_case_nums:
            stripped_results: dict[str, CandidateSurrogateResult] = {}
            for cid, cand_res in normalized.items():
                kept_anchors = tuple(
                    a for a in cand_res.anchors if int(a.case_num) not in guardrail_case_nums
                )
                self._last_guardrail_metrics_by_cid[cid] = {
                    int(a.case_num): a
                    for a in cand_res.anchors
                    if int(a.case_num) in guardrail_case_nums
                }
                if kept_anchors:
                    stripped_results[cid] = aggregate_surrogate_result(
                        cand_res.candidate_id,
                        cand_res.operator_name,
                        kept_anchors,
                        error_message=cand_res.error_message,
                    )
                else:
                    stripped_results[cid] = replace(cand_res, anchors=())
            return stripped_results, run_ids
        return normalized, run_ids

    def _recheck_single_sibling_anchor(
        self,
        candidate_id: str,
        candidate_source: str,
        dipped_anchor: AnchorCaseSpec,
    ) -> CandidateSurrogateResult | None:
        """Re-evaluate one candidate on a single dipped sibling anchor."""
        with tempfile.TemporaryDirectory(prefix="tf_camp_recheck_") as tmp_dir:
            cand_file = Path(tmp_dir) / f"{candidate_id}.py"
            cand_file.write_text(candidate_source, encoding="utf-8")
            try:
                l2_results, _ = self._run_l2_batch(
                    [(candidate_id, cand_file)],
                    [dipped_anchor],
                    include_guardrails=False,
                )
            except (
                OSError,
                subprocess.SubprocessError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                logger.warning(
                    "Single-point sibling recheck failed for %s case %d: %s",
                    candidate_id,
                    dipped_anchor.case_num,
                    exc,
                )
                return None
        return l2_results.get(candidate_id)

    @staticmethod
    def _regime_closure_ast_hash(
        module_source: str,
        spec: RegimeTargetSpec,
        *,
        operator_entrypoint: str = "",
        sibling_specs: Sequence[RegimeTargetSpec] = (),
    ) -> str | None:
        """Compute canonical AST+Track-B-TU hash for one regime's reachable closure."""
        from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
            extract_regime_snippet,
        )
        from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
            _normalized_module_ast_signature,
        )

        try:
            _primary, snippet = extract_regime_snippet(
                module_source,
                spec,
                operator_entrypoint=operator_entrypoint,
                sibling_specs=sibling_specs,
            )
            ast.parse(snippet)
        except (SyntaxError, ValueError):
            return None
        return _normalized_module_ast_signature(snippet)

    def _unchanged_sibling_regimes(
        self,
        incumbent_src: str,
        candidate_src: str,
        active_regime_spec: RegimeTargetSpec,
    ) -> set[str]:
        """Return sibling regime tags whose Python AST + Track-B TU closure is unchanged."""
        if not incumbent_src.strip() or not candidate_src.strip():
            return set()
        op_entry = self._resolve_operator_entrypoint_safe(self._config.operator_name)
        unchanged: set[str] = set()
        for spec in self._regime_specs:
            if spec.regime_tag == active_regime_spec.regime_tag:
                continue
            inc_hash = self._regime_closure_ast_hash(
                incumbent_src,
                spec,
                operator_entrypoint=op_entry,
                sibling_specs=self._regime_specs,
            )
            cand_hash = self._regime_closure_ast_hash(
                candidate_src,
                spec,
                operator_entrypoint=op_entry,
                sibling_specs=self._regime_specs,
            )
            if inc_hash is not None and cand_hash is not None and inc_hash == cand_hash:
                unchanged.add(spec.regime_tag)
        return unchanged

    def _retest_noisy_tail_anchors(
        self,
        *,
        candidate_id: str,
        candidate_source: str,
        cur_res: CandidateSurrogateResult,
        regressed_anchors: Sequence[AnchorEvalMetrics],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> CandidateSurrogateResult | None:
        """Re-evaluate noisy regressed tail anchors once and merge into CandidateSurrogateResult."""
        anchor_spec_by_case = {int(a.case_num): a for a in active_anchors}
        retest_specs: list[AnchorCaseSpec] = []
        seen_cases: set[int] = set()
        for anc in regressed_anchors:
            cnum = int(anc.case_num)
            if cnum in seen_cases:
                continue
            seen_cases.add(cnum)
            spec = anchor_spec_by_case.get(cnum)
            if spec is None:
                spec = AnchorCaseSpec(
                    task=f"operator/{cur_res.operator_name}",
                    case_num=cnum,
                    regime_tag=anc.regime_tag,
                    ref_perf_us=anc.ref_perf_us,
                    t_hw_us=anc.t_hw_us,
                )
            retest_specs.append(spec)
        if not retest_specs:
            return None
        with tempfile.TemporaryDirectory(prefix="tf_camp_tail_retest_") as tmp_dir:
            cand_file = Path(tmp_dir) / f"{candidate_id}.py"
            cand_file.write_text(candidate_source, encoding="utf-8")
            try:
                l2_results, _ = self._run_l2_batch(
                    [(candidate_id, cand_file)],
                    retest_specs,
                    include_guardrails=False,
                )
            except (
                OSError,
                subprocess.SubprocessError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                logger.warning(
                    "Noisy tail retest failed for %s: %s",
                    candidate_id,
                    exc,
                )
                return None
        retest_res = l2_results.get(candidate_id)
        if retest_res is None or not retest_res.anchors:
            return None
        retest_by_case = {int(a.case_num): a for a in retest_res.anchors}
        merged_anchors = tuple(
            retest_by_case.get(int(a.case_num), a) for a in cur_res.anchors
        )
        return aggregate_surrogate_result(
            cur_res.candidate_id,
            cur_res.operator_name,
            merged_anchors,
            error_message=cur_res.error_message,
        )

    def _evaluate_dual_layer_tail_guard(
        self,
        *,
        candidate_id: str,
        candidate_source: str,
        top_res: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult,
        regime_spec: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec],
        incumbent_source: str = "",
    ) -> tuple[bool, CandidateSurrogateResult]:
        """Delegate CEGAR guardrails, 3-sigma non-regression, and 2-sigma gain checks."""
        eff_incumbent = incumbent_source or self._active_incumbent_source
        unchanged_siblings = (
            self._unchanged_sibling_regimes(eff_incumbent, candidate_source, regime_spec)
            if eff_incumbent and eff_incumbent != candidate_source
            else set()
        )
        ok, updated_res, reject_reason = evaluate_promotion_gate(
            candidate_id=candidate_id,
            cur_res=top_res,
            best_result=best_result,
            regime_spec=regime_spec,
            guardrail_cases=self._guardrail_cases,
            cand_guardrail_metrics=self._last_guardrail_metrics_by_cid.get(candidate_id, {}),
            unchanged_siblings=unchanged_siblings,
            active_anchors=active_anchors,
            retest_noisy_fn=lambda regressed, cur: self._retest_noisy_tail_anchors(
                candidate_id=candidate_id,
                candidate_source=candidate_source,
                cur_res=cur,
                regressed_anchors=regressed,
                active_anchors=active_anchors,
            ),
            recheck_single_sibling_fn=lambda spec: self._recheck_single_sibling_anchor(
                candidate_id,
                candidate_source,
                spec,
            ),
        )
        self._last_tail_guard_reject_reason = reject_reason
        return ok, updated_res

    def _record_llm_failure(
        self,
        *,
        wid: str,
        rule: AscendCausalRule | None,
        err: str,
        is_transport_error: bool,
        regime_spec: RegimeTargetSpec,
        parent_sha: str,
        parent_axes: TopologyAxes,
        round_idx: int,
        failed_entries: list[tuple[str, str, str]],
    ) -> None:
        stage_tag = "llm_transport" if is_transport_error else "llm_generate"
        failed_entries.append((wid, stage_tag, err[:1500]))
        if rule is not None and not is_transport_error:
            self._tracker.record_outcome(
                rule.rule_id,
                postcondition_realized=False,
                topology_axes=parent_axes,
            )
            self._tracker.record_scoped_failure(
                regime_tag=regime_spec.regime_tag,
                rule_id=rule.rule_id,
                parent_sha=parent_sha,
                current_round=round_idx,
                reason="llm_generate_parse_failed",
            )
            self._record_kb_event(
                rule_id=rule.rule_id,
                postcondition_realized=False,
                topology_axes=parent_axes,
                round_idx=round_idx,
            )

    def _absorb_creative_task(
        self,
        task: asyncio.Task[SpeculativeCreativeResult],
        *,
        origin_round_idx: int,
        wid: str,
        rule_obj: object,
        batch: CandidateBatch,
        parent_sha: str,
        parent_axes: TopologyAxes,
        raw_candidates: list[tuple[str, str]],
        meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]],
        failed_entries: list[tuple[str, str, str]],
    ) -> None:
        _ = origin_round_idx
        rule = rule_obj if isinstance(rule_obj, AscendCausalRule) else None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            meta_by_id[wid] = ("creative_explore", rule, "")
            self._record_llm_failure(
                wid=wid,
                rule=rule,
                err=f"llm_transport:{type(exc).__name__}:{exc}",
                is_transport_error=True,
                regime_spec=batch.regime_spec,
                parent_sha=parent_sha,
                parent_axes=parent_axes,
                round_idx=batch.round_idx,
                failed_entries=failed_entries,
            )
            return
        res = task.result()
        meta_by_id[res.worker_id] = ("creative_explore", rule, res.hypothesis)
        if res.snippet is None:
            self._record_llm_failure(
                wid=res.worker_id,
                rule=rule,
                err=res.error,
                is_transport_error=res.is_transport_error,
                regime_spec=batch.regime_spec,
                parent_sha=parent_sha,
                parent_axes=parent_axes,
                round_idx=batch.round_idx,
                failed_entries=failed_entries,
            )
            return
        raw_candidates.append((res.worker_id, res.snippet))

    @staticmethod
    def _classify_l2_exception(exc: BaseException) -> TypedFault | None:
        is_occupancy = (
            isinstance(exc, OccupancyExhaustedError)
            or type(exc).__name__ == "OccupancyExhaustedError"
            or str(getattr(exc, "fault_plane", "")) == "INFRA_OCCUPANCY"
        )
        if is_occupancy:
            return TypedFault(
                plane=FaultPlane.INFRA_OCCUPANCY,
                code="L2_INFRA_OCCUPANCY",
                failed_tier="l2_infra_occupancy",
                detail=str(exc)[:500],
                retriable=True,
            )
        msg_low = str(exc).lower()
        if isinstance(exc, AgentGatesDeviceFaultError) or "device_fault" in msg_low:
            exc_plane = (
                coerce_fault_plane(getattr(exc, "fault_plane", ""))
                if isinstance(exc, AgentGatesDeviceFaultError)
                else FaultPlane.INFRA_DEVICE_FAULT
            )
            if exc_plane == FaultPlane.UNMODELED_FAULT:
                exc_plane = FaultPlane.INFRA_DEVICE_FAULT
            exc_code = (
                str(getattr(exc, "fault_code", "") or "")
                if isinstance(exc, AgentGatesDeviceFaultError)
                else ""
            ) or (
                "L2_INFRA_OCCUPANCY"
                if exc_plane == FaultPlane.INFRA_OCCUPANCY
                else "L2_INFRA_DEVICE_FAULT"
            )
            exc_tier = (
                "l2_infra_occupancy"
                if exc_plane == FaultPlane.INFRA_OCCUPANCY
                else "l2_infra_device_fault"
            )
            return TypedFault(
                plane=exc_plane,
                code=exc_code,
                failed_tier=exc_tier,
                detail=str(exc)[:500],
                retriable=True,
            )
        if isinstance(exc, (TimeoutError, OSError, subprocess.SubprocessError)) or any(
            tok in msg_low
            for tok in ("occupancy", "resource busy", "device_busy", "card_busy")
        ):
            return TypedFault(
                plane=FaultPlane.INFRA_OCCUPANCY,
                code="L2_INFRA_OCCUPANCY",
                failed_tier="l2_infra_occupancy",
                detail=str(exc)[:500],
                retriable=True,
            )
        return None

    async def cancel_pending_speculative(self) -> None:
        """Cancel all in-flight speculative creative tasks cleanly."""
        pending = list(self._pending_speculative)
        self._pending_speculative.clear()
        tasks = [task for _orig_round, _wid, _rule, task in pending if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_round_gates(
        self,
        batch: CandidateBatch,
        working_anchors: Sequence[AnchorCaseSpec] = (),
        *,
        backend: KernelLLMBackend | None = None,
        parent_sha: str = "",
        parent_workload: CostFeatures | Mapping[str, object] | None = None,
    ) -> StageGateOutcome:
        """Execute LLM generation, speculative creative harvesting, and L0->L1->L2 gates."""
        eff_backend = backend or self._backend
        if eff_backend is None and batch.worker_specs:
            eff_backend = self._get_backend()
        eff_parent_sha = parent_sha or self._source_sha256(batch.parent_module_source)
        eff_parent_axes = batch.parent_axes or self._derive_topology_axes(batch.parent_ledger)
        eff_active_mask = batch.active_mask or self._resolve_trackb_freedom_mask(batch.regime_spec)

        raw_candidates: list[tuple[str, str]] = []
        failed_entries: list[tuple[str, str, str]] = []
        meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]] = {}

        still_pending: list[
            tuple[int, str, object, asyncio.Task[SpeculativeCreativeResult]]
        ] = []
        for origin_round_idx, wid, rule_obj, task in self._pending_speculative:
            if not task.done():
                still_pending.append((origin_round_idx, wid, rule_obj, task))
                continue
            self._absorb_creative_task(
                task,
                origin_round_idx=origin_round_idx,
                wid=wid,
                rule_obj=rule_obj,
                batch=batch,
                parent_sha=eff_parent_sha,
                parent_axes=eff_parent_axes,
                raw_candidates=raw_candidates,
                meta_by_id=meta_by_id,
                failed_entries=failed_entries,
            )
        self._pending_speculative = still_pending

        stagger_sec = (
            1.5
            if isinstance(
                eff_backend,
                (AgyCliLLMBackend, OpenAICompatHTTPBackend, SubprocessCliTemplateBackend),
            )
            else 0.0
        )

        async def _call_exploit_worker(
            slot_idx: int,
            wid: str,
            wkind: str,
            rule: AscendCausalRule | None,
            req: GenerationRequest,
        ) -> tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]:
            assert eff_backend is not None
            if stagger_sec > 0.0 and slot_idx > 0:
                await asyncio.sleep(float(slot_idx) * stagger_sec)
            try:
                raw = await eff_backend.generate(req)
            except (
                OSError,
                subprocess.SubprocessError,
                TimeoutError,
                RuntimeError,
                ConnectionError,
            ) as exc:
                return wid, wkind, rule, None, "", f"llm_transport:{type(exc).__name__}:{exc}", True
            try:
                snippet = extract_python_dsl(raw)
                return wid, wkind, rule, snippet, "", "", False
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return wid, wkind, rule, None, "", f"llm_parse:{type(exc).__name__}:{exc}", False

        async def _call_creative_worker(
            slot_idx: int,
            wid: str,
            req: GenerationRequest,
        ) -> SpeculativeCreativeResult:
            assert eff_backend is not None
            if stagger_sec > 0.0 and slot_idx > 0:
                await asyncio.sleep(float(slot_idx) * stagger_sec)
            try:
                raw = await eff_backend.generate(req)
            except (
                OSError,
                subprocess.SubprocessError,
                TimeoutError,
                RuntimeError,
                ConnectionError,
            ) as exc:
                return SpeculativeCreativeResult(
                    origin_round_idx=batch.round_idx,
                    worker_id=wid,
                    snippet=None,
                    hypothesis="",
                    error=f"llm_transport:{type(exc).__name__}:{exc}",
                    is_transport_error=True,
                )
            try:
                creative = extract_creative_payload(raw)
                snippet = creative["dsl_source"]
                raw_hyp = str(creative.get("physical_hypothesis", "")).strip()
                raw_metric = str(creative.get("target_local_metric", "")).strip()
                if raw_hyp and raw_metric:
                    hypothesis_text = f"{raw_hyp} [target_metric={raw_metric}]"
                else:
                    hypothesis_text = raw_hyp or raw_metric
                return SpeculativeCreativeResult(
                    origin_round_idx=batch.round_idx,
                    worker_id=wid,
                    snippet=snippet,
                    hypothesis=hypothesis_text,
                    error="",
                    is_transport_error=False,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return SpeculativeCreativeResult(
                    origin_round_idx=batch.round_idx,
                    worker_id=wid,
                    snippet=None,
                    hypothesis="",
                    error=f"llm_parse:{type(exc).__name__}:{exc}",
                    is_transport_error=False,
                )

        exploit_specs: list[
            tuple[int, str, str, AscendCausalRule | None, GenerationRequest]
        ] = []
        creative_specs: list[
            tuple[int, str, AscendCausalRule | None, GenerationRequest]
        ] = []
        for slot_idx, (wid, wkind, rule, req) in enumerate(batch.worker_specs):
            if wkind == "creative_explore":
                creative_specs.append((slot_idx, wid, rule, req))
            else:
                exploit_specs.append((slot_idx, wid, wkind, rule, req))

        llm_t0 = time.monotonic()
        current_creative_tasks: list[
            tuple[int, str, AscendCausalRule | None, asyncio.Task[SpeculativeCreativeResult]]
        ] = []
        for slot_idx, wid, rule, req in creative_specs:
            task = asyncio.create_task(
                _call_creative_worker(slot_idx, wid, req),
                name=f"speculative_creative_r{batch.round_idx}_{wid}",
            )
            current_creative_tasks.append((batch.round_idx, wid, rule, task))

        exploit_replies: list[
            tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]
        ] = []
        if exploit_specs:
            exploit_replies = list(
                await asyncio.gather(
                    *(
                        _call_exploit_worker(slot_idx, wid, wkind, rule, req)
                        for slot_idx, wid, wkind, rule, req in exploit_specs
                    )
                )
            )
        elif current_creative_tasks:
            await asyncio.gather(*(task for _, _, _, task in current_creative_tasks))
        llm_gen_sec = max(0.0, time.monotonic() - llm_t0)

        for wid, wkind, rule, snippet, cand_hypothesis, err, is_transport_error in exploit_replies:
            meta_by_id[wid] = (wkind, rule, cand_hypothesis)
            if snippet is None:
                self._record_llm_failure(
                    wid=wid,
                    rule=rule,
                    err=err,
                    is_transport_error=is_transport_error,
                    regime_spec=batch.regime_spec,
                    parent_sha=eff_parent_sha,
                    parent_axes=eff_parent_axes,
                    round_idx=batch.round_idx,
                    failed_entries=failed_entries,
                )
            else:
                raw_candidates.append((wid, snippet))

        for origin_round_idx, wid, rule, task in current_creative_tasks:
            if task.done():
                self._absorb_creative_task(
                    task,
                    origin_round_idx=origin_round_idx,
                    wid=wid,
                    rule_obj=rule,
                    batch=batch,
                    parent_sha=eff_parent_sha,
                    parent_axes=eff_parent_axes,
                    raw_candidates=raw_candidates,
                    meta_by_id=meta_by_id,
                    failed_entries=failed_entries,
                )
            else:
                self._pending_speculative.append((origin_round_idx, wid, rule, task))

        l0_l1_checks: list[L0L1CandidateCheck] = []
        round_active_kernel = ""
        l0_l1_sec = 0.0
        syntax_repair_sec = 0.0

        if raw_candidates:
            from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: PLC0415
                _compact_trackb_snippet_source,
                _detect_trackb_dead_code_edit,
                _extract_p2_focus_lines_and_diag,
                patch_regime_into_module,
            )

            prescreen_top_k: int | None = None
            stage_a_ranker = None
            parent_analytical_us: float | None = None
            score_stage_a_analytical_fn: (
                Callable[[str, str, StageAResult], float] | None
            ) = None
            gate_decision = batch.gate_decision
            if (
                gate_decision is not None
                and gate_decision.engaged
                and gate_decision.allow_candidate_pruning
            ):
                if (
                    self._config.cost_prescreen_top_k is not None
                    and int(self._config.cost_prescreen_top_k) > 0
                ):
                    prescreen_top_k = int(self._config.cost_prescreen_top_k)
                elif self._config.cost_prescreen_top_k is None:
                    non_creative_count = sum(
                        1
                        for cid, _ in raw_candidates
                        if meta_by_id.get(cid, ("", None, ""))[0] != "creative_explore"
                    )
                    if non_creative_count >= 2:
                        prescreen_top_k = max(1, non_creative_count - 1)

                def _rank_stage_a_passers(
                    passers: Sequence[tuple[str, str, StageAResult]],
                ) -> list[tuple[str, float]]:
                    if self._cost_advisor.underlying_cost_model is not None:
                        return [
                            (item.candidate_id, item.pred_us)
                            for item in rank_stage_a_candidates_with_cost_model(
                                passers,
                                cost_model=self._cost_advisor.underlying_cost_model,
                                live_plan=batch.live_plan,
                                axes=eff_parent_axes,
                                blueprint=batch.blueprint,
                                sil_advisory=self._sil_advisory,
                                parent_ledger=batch.parent_ledger,
                            )
                        ]
                    filtered = self._cost_advisor.filter_stage_a_passers(
                        passers,
                        decision=gate_decision,
                        live_plan=batch.live_plan,
                        axes=eff_parent_axes,
                        blueprint=batch.blueprint,
                        sil_advisory=self._sil_advisory,
                        parent_ledger=batch.parent_ledger,
                        top_k=prescreen_top_k,
                    )
                    out_pairs: list[tuple[str, float]] = []
                    for item in filtered:
                        if hasattr(item, "candidate_id") and hasattr(item, "pred_us"):
                            out_pairs.append((str(item.candidate_id), float(item.pred_us)))
                        elif isinstance(item, (tuple, list)) and len(item) >= 2:
                            out_pairs.append((str(item[0]), float(item[1])))
                    return out_pairs

                stage_a_ranker = _rank_stage_a_passers if prescreen_top_k is not None else None
                eff_parent_workload = parent_workload
                if eff_parent_workload is None and batch.parent_ledger is not None:
                    eff_parent_workload = build_cost_features_from_ledger(
                        batch.parent_ledger,
                        live_plan=batch.live_plan,
                        axes=eff_parent_axes,
                        blueprint=batch.blueprint,
                        sil_advisory=self._sil_advisory,
                        target=self._target,
                    )
                parent_l0_pred = (
                    self._cost_advisor.predict_candidate_us(
                        eff_parent_workload,
                        bisheng_res=None,
                    )
                    if eff_parent_workload is not None
                    else None
                )
                if isinstance(parent_l0_pred, (int, float)) and not isinstance(
                    parent_l0_pred, bool
                ):
                    eff_parent_analytical_us = float(parent_l0_pred)
                elif parent_l0_pred is not None:
                    calibrated_parent_us = float(getattr(parent_l0_pred, "pred_us", 0.0) or 0.0)
                    raw_parent_analytical_us = float(
                        getattr(parent_l0_pred, "tier0_physics_us", 0.0) or 0.0
                    ) + float(getattr(parent_l0_pred, "tier1_penalty_us", 0.0) or 0.0)
                    eff_parent_analytical_us = (
                        calibrated_parent_us
                        if calibrated_parent_us > 0.0
                        else raw_parent_analytical_us
                    )
                else:
                    eff_parent_analytical_us = 0.0
                parent_analytical_us = (
                    eff_parent_analytical_us if eff_parent_analytical_us > 0.0 else None
                )

                def _score_stage_a_analytical(
                    _cid: str,
                    _patched_src: str,
                    stage_a: StageAResult,
                ) -> float:
                    cand_axes = (
                        self._derive_topology_axes(
                            stage_a.ledger,
                            features=stage_a.features,
                            route=stage_a.route,
                        )
                        if stage_a.ledger is not None
                        else eff_parent_axes
                    )
                    features = build_cost_features_from_ledger(
                        stage_a.ledger,
                        live_plan=batch.live_plan,
                        axes=cand_axes,
                        blueprint=batch.blueprint,
                        sil_advisory=self._sil_advisory,
                        target=self._cost_advisor.target,
                    )
                    pred = self._cost_advisor.predict_candidate_us(features)
                    if isinstance(pred, (int, float)) and not isinstance(pred, bool):
                        return float(pred)
                    if pred is None:
                        return 0.0
                    calibrated_us = float(getattr(pred, "pred_us", 0.0) or 0.0)
                    if calibrated_us > 0.0:
                        return calibrated_us
                    return float(getattr(pred, "tier0_physics_us", 0.0) or 0.0) + float(
                        getattr(pred, "tier1_penalty_us", 0.0) or 0.0
                    )

                score_stage_a_analytical_fn = _score_stage_a_analytical

            creative_candidate_ids = {
                cid
                for cid, _ in raw_candidates
                if meta_by_id.get(cid, ("", None, ""))[0] == "creative_explore"
            } | set(batch.creative_candidate_ids)

            def _candidate_l0_check(mod_src: str) -> StageAResult:
                cand_sha_in = compute_source_sha256(mod_src)
                knob_sig_in = self._extract_knob_signature(mod_src, batch.regime_spec)
                infeasible_reason = self._boundary_ledger.precheck_known_infeasible(
                    regime_tag=batch.regime_spec.regime_tag,
                    cand_sha=cand_sha_in,
                    knob_signature=knob_sig_in,
                )
                if infeasible_reason is not None:
                    return StageAResult(
                        passed=False,
                        prim_func=None,
                        ledger=None,
                        error_message=infeasible_reason,
                        backend_kind=batch.regime_spec.runtime_backend,
                    )
                dead_err = _detect_trackb_dead_code_edit(
                    batch.parent_module_source, mod_src, batch.regime_spec
                )
                if dead_err is not None:
                    return StageAResult(
                        passed=False,
                        prim_func=None,
                        ledger=None,
                        error_message=dead_err,
                        backend_kind="ascendc",
                    )
                return self._invoke_with_optional_kwargs(
                    self._run_l0,
                    mod_src,
                    batch.regime_spec,
                    anchor_dims_override=batch.bn_dims or None,
                )

            round_active_kernel = (
                self._resolve_active_trackb_kernel(
                    batch.parent_module_source,
                    batch.regime_spec,
                )
                or ""
            )
            parent_focus_lines, parent_diag_text = _extract_p2_focus_lines_and_diag(
                batch.parent_p2_diag
            )
            repair_parent_source = _compact_trackb_snippet_source(
                batch.parent_snippet,
                batch.regime_spec,
                incumbent_source=batch.parent_module_source,
                active_kernel=round_active_kernel,
                focus_lines=parent_focus_lines,
                diagnostic_text=parent_diag_text,
            )
            trackb_prologue_line_offset = 0
            if batch.regime_spec.runtime_backend in (
                "ascendc_trackb",
                "cce_trackb",
                "hybrid_trackb",
            ):
                sentinel = "__TF_TRACKB_SENTINEL__"
                try:
                    rendered_probe = self._instantiate_trackb_tu(
                        batch.parent_module_source,
                        sentinel,
                        batch.regime_spec,
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    rendered_probe = ""
                if isinstance(rendered_probe, str) and sentinel in rendered_probe:
                    trackb_prologue_line_offset = rendered_probe.split(sentinel, 1)[0].count("\n")

            op_entry = self._resolve_operator_entrypoint_safe(self._config.operator_name)

            def _patch_against(base_mod_src: str, snip: str) -> str:
                return patch_regime_into_module(
                    base_mod_src,
                    batch.regime_spec,
                    snip,
                    operator_entrypoint=op_entry,
                    sibling_specs=self._regime_specs,
                    active_kernel=round_active_kernel,
                )

            class _TimedRepairBackend:
                def __init__(self, inner: KernelLLMBackend) -> None:
                    self._inner = inner
                    self.repair_sec = 0.0

                async def generate(
                    self,
                    request: GenerationRequest,
                    *,
                    effort_override: object = None,
                ) -> str:
                    t0 = time.monotonic()
                    try:
                        if effort_override is not None:
                            return await self._inner.generate(
                                request, effort_override=effort_override
                            )
                        return await self._inner.generate(request)
                    finally:
                        self.repair_sec += max(0.0, time.monotonic() - t0)

            timed_repair_backend = (
                _TimedRepairBackend(eff_backend)
                if eff_backend is not None and self._config.max_syntax_repairs > 0
                else None
            )
            l0_l1_t0 = time.monotonic()
            l0_l1_checks = await run_l0_l1_round_filter(
                raw_candidates,
                patch_fn=lambda snip: _patch_against(batch.parent_module_source, snip),
                repatch_from_current_fn=lambda cur_mod_src, snip: _patch_against(
                    cur_mod_src, snip
                ),
                l0_checker=_candidate_l0_check,
                l1_compiler=lambda mod_src: self._run_l1(mod_src, batch.regime_spec),
                repair_backend=timed_repair_backend,
                max_syntax_repairs=self._config.max_syntax_repairs,
                slice_hint=self._slice_contract_instruction(
                    batch.func_name,
                    batch.regime_spec,
                    operator_entrypoint=op_entry,
                ),
                backend_kind=batch.regime_spec.runtime_backend,
                parent_source=repair_parent_source,
                parent_module_source=batch.parent_module_source,
                trackb_prologue_line_offset=trackb_prologue_line_offset,
                stage_a_cost_ranker=stage_a_ranker,
                cost_prescreen_top_k=prescreen_top_k,
                sil_advisory=self._sil_advisory,
                freedom_mask=eff_active_mask,
                ub_256kb_unlock=(
                    bool(batch.parent_ledger.uses_unreserved_ubuf)
                    if batch.parent_ledger
                    else False
                ),
                creative_candidate_ids=creative_candidate_ids,
                parent_analytical_us=parent_analytical_us,
                stage_a_analytical_scorer=score_stage_a_analytical_fn,
                parent_topology_axes=eff_parent_axes,
                candidate_topology_axes_fn=lambda _mod_src, sa: self._derive_topology_axes(
                    sa.ledger,
                    features=sa.features,
                    route=sa.route,
                )
                if sa.ledger is not None
                else None,
            )
            l0_l1_total_sec = max(0.0, time.monotonic() - l0_l1_t0)
            syntax_repair_sec = (
                timed_repair_backend.repair_sec if timed_repair_backend is not None else 0.0
            )
            l0_l1_sec = max(0.0, l0_l1_total_sec - syntax_repair_sec)

        passed_checks = [chk for chk in l0_l1_checks if chk.passed]
        infeasible_boundaries_added = 0
        prescreen_pruned_list: list[str] = []
        knob_sig_by_cid: dict[str, str] = {}
        ub_bytes_by_cid: dict[str, int] = {}
        pred_by_cid: dict[str, float] = {}
        cand_axes_by_cid: dict[str, TopologyAxes] = {}
        for chk in l0_l1_checks:
            self._persist_round_candidate_artifact(chk, round_idx=batch.round_idx)
            knob_sig = (
                self._extract_knob_signature(chk.patched_module_source, batch.regime_spec)
                if chk.patched_module_source
                else ""
            )
            knob_sig_by_cid[chk.candidate_id] = knob_sig
            ub_bytes = 0
            if chk.stage_a is not None and chk.stage_a.ledger is not None:
                ub_wl = (getattr(chk.stage_a.ledger, "waterlines", {}) or {}).get("UB")
                if ub_wl is not None:
                    ub_bytes = int(getattr(ub_wl, "peak_allocated_bytes", 0) or 0)
            ub_bytes_by_cid[chk.candidate_id] = ub_bytes
            if chk.stage_a is not None and (chk.stage_a.passed or chk.stage_a.ledger is not None):
                c_axes = self._derive_topology_axes(
                    chk.stage_a.ledger,
                    features=chk.stage_a.features,
                    route=chk.stage_a.route,
                )
                cand_axes_by_cid[chk.candidate_id] = c_axes
                try:
                    c_feat = build_cost_features_from_ledger(
                        chk.stage_a.ledger,
                        live_plan=batch.live_plan,
                        axes=c_axes,
                        blueprint=batch.blueprint,
                        sil_advisory=self._sil_advisory,
                        target=self._cost_advisor.target,
                    )
                    c_pred = self._cost_advisor.predict_candidate_us(c_feat)
                    if isinstance(c_pred, (int, float)) and not isinstance(c_pred, bool):
                        c_pred_val = float(c_pred)
                    elif c_pred is not None:
                        c_pred_val = float(getattr(c_pred, "pred_us", 0.0) or 0.0)
                        if c_pred_val <= 0.0:
                            c_pred_val = float(
                                getattr(c_pred, "tier0_physics_us", 0.0) or 0.0
                            ) + float(getattr(c_pred, "tier1_penalty_us", 0.0) or 0.0)
                    else:
                        c_pred_val = 0.0
                    if c_pred_val > 0.0:
                        pred_by_cid[chk.candidate_id] = c_pred_val
                except (TypeError, ValueError):
                    pass

        l0_l1_faults_by_cid: dict[str, TypedFault] = {}
        for chk in l0_l1_checks:
            if not chk.passed:
                failed_entries.append(
                    (chk.candidate_id, chk.failed_tier or "l0_l1", chk.error_message[:1500])
                )
                if chk.failed_tier == "cost_prescreen":
                    prescreen_pruned_list.append(chk.candidate_id)
                cand_sha_chk = (
                    compute_source_sha256(chk.patched_module_source)
                    if chk.patched_module_source
                    else ""
                )
                raw_fp = (
                    getattr(chk, "fault_plane", "")
                    or (getattr(chk.stage_b, "fault_plane", "") if chk.stage_b else "")
                    or (getattr(chk.stage_a, "fault_plane", "") if chk.stage_a else "")
                    or (
                        "LLM_TRANSPORT"
                        if chk.failed_tier == "llm_transport"
                        else (
                            "COMPILER_LOWERING"
                            if chk.failed_tier == "stage_b"
                            else "STATIC_CONTRACT"
                        )
                    )
                )
                raw_fc = (
                    getattr(chk, "fault_code", "")
                    or (getattr(chk.stage_b, "fault_code", "") if chk.stage_b else "")
                    or (getattr(chk.stage_a, "fault_code", "") if chk.stage_a else "")
                    or f"{(chk.failed_tier or 'L0_L1').upper()}_FAILED"
                )
                chk_fault = TypedFault(
                    plane=coerce_fault_plane(raw_fp),
                    code=str(raw_fc),
                    failed_tier=str(chk.failed_tier or "l0_l1"),
                    detail=str(chk.error_message or "")[:500],
                )
                l0_l1_faults_by_cid[chk.candidate_id] = chk_fault
                stack_bytes = (
                    int(chk.stage_b.compiler_summary.stack_bytes)
                    if chk.stage_b is not None and chk.stage_b.compiler_summary is not None
                    else 0
                )
                if self._boundary_ledger.record_compile_wall(
                    regime_tag=batch.regime_spec.regime_tag,
                    cand_sha=cand_sha_chk,
                    knob_signature=knob_sig_by_cid.get(chk.candidate_id, ""),
                    error_code_norm=chk_fault.code,
                    ub_bytes=ub_bytes_by_cid.get(chk.candidate_id, 0),
                    stack_bytes=stack_bytes,
                ):
                    infeasible_boundaries_added += 1
                _wkind, rule, _hyp = meta_by_id.get(chk.candidate_id, ("unknown", None, ""))
                if (
                    rule is not None
                    and chk_fault.plane.penalizes_bandit()
                    and chk.failed_tier != "cost_prescreen"
                    and "[repair_timeout]" not in chk.error_message
                ):
                    self._tracker.record_outcome(
                        rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=eff_parent_axes,
                    )
                    self._tracker.record_scoped_failure(
                        regime_tag=batch.regime_spec.regime_tag,
                        rule_id=rule.rule_id,
                        parent_sha=eff_parent_sha,
                        current_round=batch.round_idx,
                        reason=f"l0_l1_failed:{chk.failed_tier or 'l0_l1'}",
                    )
                    self._record_kb_event(
                        rule_id=rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=eff_parent_axes,
                        round_idx=batch.round_idx,
                    )

        l2_results: dict[str, CandidateSurrogateResult] = {}
        l2_run_ids: dict[str, str] = {}
        l2_infra_fault: TypedFault | None = None
        l2_prewarm_sec = 0.0
        l2_npu_sec = 0.0
        round_prewarm_stats: dict[str, int] = {
            "total_tasks": 0,
            "cache_hits": 0,
            "compiled_tasks": 0,
            "failed_candidates": 0,
        }

        if passed_checks:
            with tempfile.TemporaryDirectory(prefix=f"tf_camp_r{batch.round_idx}_") as tmp_dir:
                cand_paths: list[tuple[str, Path]] = []
                for chk in passed_checks:
                    cand_file = Path(tmp_dir) / f"{chk.candidate_id}.py"
                    cand_file.write_text(chk.patched_module_source, encoding="utf-8")
                    cand_paths.append((chk.candidate_id, cand_file))
                l2_t0 = time.monotonic()
                try:
                    l2_results, l2_run_ids = self._run_l2_batch(
                        cand_paths,
                        working_anchors,
                        include_guardrails=True,
                    )
                except Exception as exc:
                    classified = self._classify_l2_exception(exc)
                    if classified is None and not isinstance(
                        exc,
                        (
                            AgentGatesDeviceFaultError,
                            OSError,
                            subprocess.SubprocessError,
                            TimeoutError,
                            json.JSONDecodeError,
                            RuntimeError,
                        ),
                    ):
                        raise
                    logger.warning(
                        "Round %d L2 batch failed after retries: %s",
                        batch.round_idx,
                        exc,
                    )
                    l2_results = {}
                    l2_run_ids = {}
                    l2_infra_fault = classified
                    fail_tier_l2 = (
                        l2_infra_fault.failed_tier if l2_infra_fault is not None else "l2_error"
                    )
                    for chk in passed_checks:
                        failed_entries.append((chk.candidate_id, fail_tier_l2, str(exc)[:1500]))
                l2_wall_sec = max(0.0, time.monotonic() - l2_t0)
                raw_pw_stats = self._last_l2_timing_meta.get("prewarm_stats")
                if isinstance(raw_pw_stats, Mapping):
                    for pw_key in (
                        "total_tasks",
                        "cache_hits",
                        "compiled_tasks",
                        "failed_candidates",
                    ):
                        if pw_key in raw_pw_stats:
                            round_prewarm_stats[pw_key] = int(raw_pw_stats[pw_key])
                l2_prewarm_sec = float(self._last_l2_timing_meta.get("l2_prewarm_sec") or 0.0)
                raw_npu_sec = self._last_l2_timing_meta.get("l2_npu_sec")
                l2_npu_sec = (
                    float(raw_npu_sec)
                    if isinstance(raw_npu_sec, (int, float)) and not isinstance(raw_npu_sec, bool)
                    else max(0.0, l2_wall_sec - l2_prewarm_sec)
                )

        return StageGateOutcome(
            raw_candidates=raw_candidates,
            meta_by_id=meta_by_id,
            failed_entries=failed_entries,
            l0_l1_checks=l0_l1_checks,
            passed_checks=passed_checks,
            l0_l1_faults_by_cid=l0_l1_faults_by_cid,
            l2_results=l2_results,
            l2_run_ids=l2_run_ids,
            l2_infra_fault=l2_infra_fault,
            llm_gen_sec=llm_gen_sec,
            l0_l1_sec=l0_l1_sec,
            syntax_repair_sec=syntax_repair_sec,
            l2_prewarm_sec=l2_prewarm_sec,
            l2_npu_sec=l2_npu_sec,
            round_prewarm_stats=round_prewarm_stats,
            round_active_kernel=round_active_kernel,
            infeasible_boundaries_added=infeasible_boundaries_added,
            prescreen_pruned_list=prescreen_pruned_list,
            knob_sig_by_cid=knob_sig_by_cid,
            ub_bytes_by_cid=ub_bytes_by_cid,
            pred_by_cid=pred_by_cid,
            cand_axes_by_cid=cand_axes_by_cid,
        )


__all__ = [
    "CandidateBatch",
    "OccupancyExhaustedError",
    "SpeculativeCreativeResult",
    "StageGateConfig",
    "StageGateCoordinator",
    "StageGateOutcome",
]
