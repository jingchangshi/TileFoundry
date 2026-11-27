"""Stage-gate coordinator for round LLM generation, L0/L1 filtering, L2 eval, and tail guards."""

from __future__ import annotations

import ast, asyncio, hashlib, json, logging, os, subprocess, tempfile, threading, time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar

import yaml

from tilefoundry_ascend.agent.campaign.funnel_telemetry import DeterministicSearchBoundaryLedger
from tilefoundry_ascend.agent.campaign.helpers import (
    OperatorSemanticsCache, build_width_failure_result, compile_trackb_stage_b,
    count_trackb_workspace_stages, derive_topology_axes, dispatch_l0_gate,
    extract_candidate_topology_axes, extract_cost_pred_us, extract_knob_signature,
    is_noop_identical_failure, ledger_has_cube_engine, ledger_has_hardware_dma,
    lookup_case_dims_safe, merge_interval_wall_sec, normalize_primary_vf_overflow,
    persist_round_candidate_artifact, resolve_active_trackb_kernel, resolve_active_trackb_kernels,
    resolve_cannbench_tasks_root, resolve_cases_yaml_sha256_for_operator,
    resolve_operator_entrypoint_safe, resolve_operator_semantics_for_gates,
    resolve_trackb_freedom_mask, run_hybrid_trackb_stage_b, run_l0_gate,
    slice_contract_instruction, source_sha256, stage_b_has_vf_overflow,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import evaluate_promotion_gate
from tilefoundry_ascend.agent.contracts import (
    CampaignStateSnapshot, NpuExecutionPort, RegimeCompilePlan,
)
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor, create_cost_advisor
from tilefoundry_ascend.agent.gates.gate_funnel import (
    StageAResult, StageBResult, evaluate_ascendc_candidate, extract_ordered_tensor_shapes,
    resolve_trackb_compile_flags, run_stage_a_static,
)
from tilefoundry_ascend.agent.gates.stage_b_gate import (
    ModuleSelectorPlanExtractor, StageBCompileGate, build_post_selector_regime_plan,
    run_l1_compile_check,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker, append_kb_event_shard
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule
from tilefoundry_ascend.agent.npu import (
    AgentGatesDeviceFaultError, AnchorEvalMetrics, CandidateSurrogateResult, L0L1CandidateCheck,
    L0L1RoundResults, L2BatchResults, OccupancyExhaustedError, _normalized_module_ast_signature,
    aggregate_surrogate_result, lookup_case_static_spec, run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest, KernelLLMBackend, create_llm_backend_from_env,
    extract_creative_payload, extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_syntax_repair_request, check_repair_monotonicity, classify_stage_failure_for_repair,
    extract_lowered_compiler_excerpt_with_lang, parse_scoped_search_replace_blocks,
)
from tilefoundry_ascend.agent.orchestrator.regime_slice_patcher import (
    _compact_trackb_snippet_source, _detect_trackb_dead_code_edit,
    _extract_p2_focus_lines_and_diag, _has_prim_func_builder, _render_trackb_tu_from_module,
    _resolve_selector_compile_flags, patch_regime_into_module, resolve_regime_config,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, GuardrailCaseSpec, RegimeTargetSpec, build_anchor_var_bindings,
    compute_case_eq3_score, default_source_root, extract_trackb_cpp_source, get_operator_task,
    parse_simple_env_file, resolve_regime_for_case,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, TypedFault, coerce_fault_plane
from tilefoundry_ascend.ascend.compile_cache.artifact_store import ArtifactStore, resolve_unified_tl_cache_dir
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary, CostFeatures, build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    CostGateDecision, is_paradigm_shift_exempt,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import compute_source_sha256
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePlanResult
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

logger = logging.getLogger(__name__)
_MAX_L1_COMPILE_WORKERS = 8
_ORIG_RUN_L0_L1_ROUND_FILTER: object | None = None
_ItemT = TypeVar("_ItemT")
_ResT = TypeVar("_ResT")


def _pool_map(items: Sequence[_ItemT], fn: Callable[[_ItemT], _ResT]) -> list[_ResT]:
    if len(items) <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=min(_MAX_L1_COMPILE_WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


def _repair_timeout_message(error_message: str, is_transport: bool) -> str:
    return f"[repair_timeout] {error_message}" if is_transport else error_message


async def run_l0_l1_round_filter(
    candidates: Sequence[tuple[str, str]], *, patch_fn: Callable[[str], str],
    l0_checker: Callable[[str], StageAResult], l1_compiler: Callable[[str], StageBResult],
    repatch_from_current_fn: Callable[[str, str], str] | None = None,
    repair_backend: KernelLLMBackend | None = None, max_syntax_repairs: int = 1,
    slice_hint: str = "", backend_kind: str = "tilelang_dsl", parent_source: str = "",
    parent_module_source: str = "", trackb_prologue_line_offset: int = 0,
    stage_a_cost_ranker: Callable[[Sequence[tuple[str, str, StageAResult]]], Sequence[tuple[str, float]]] | None = None,
    cost_prescreen_top_k: int | None = None, sil_advisory: object | None = None,
    sil_verification: object | None = None, freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf", ub_256kb_unlock: bool = False,
    creative_candidate_ids: Collection[str] = (), parent_analytical_us: float | None = None,
    stage_a_analytical_scorer: Callable[[str, str, StageAResult], float] | None = None,
    parent_topology_axes: TopologyAxes | None = None,
    candidate_topology_axes_fn: Callable[[str, StageAResult], TopologyAxes | None] | None = None,
    parent_workspace_stages: int | None = None,
    candidate_workspace_stages_fn: Callable[[str, StageAResult], int] | None = None,
    skip_repair_when_survivors_exist: bool = False,
) -> L0L1RoundResults:
    """Filter candidates concurrently via noop guard, L0 static, prescreen, and L1 compile."""
    _ = trackb_prologue_line_offset
    max_attempts = max(0, max_syntax_repairs)
    bisheng_sem = asyncio.Semaphore(min(8, os.cpu_count() or 4))
    repair_intervals: list[tuple[float, float]] = []
    cum_repair_sec = 0.0
    base_src = parent_module_source if parent_module_source.strip() else parent_source
    base_sig = _normalized_module_ast_signature(base_src) if base_src.strip() else None
    eff_pws = int(parent_workspace_stages) if parent_workspace_stages is not None else (count_trackb_workspace_stages(base_src) if base_src.strip() else 0)

    def _fail(cid: str, tier: str, patched: str, snip: str, rep: bool, err: str, fp: str, fc: str, sa: StageAResult | None = None, sb: StageBResult | None = None) -> L0L1CandidateCheck:
        return L0L1CandidateCheck(cid, False, tier, patched, snip, rep, stage_a=sa, stage_b=sb, error_message=err, fault_plane=fp, fault_code=fc)

    async def _try_repair(
        cur_src: str, diag: str, *, lowered_excerpt: str = "", lowered_excerpt_lang: str | None = None
    ) -> tuple[str | None, str, bool]:
        nonlocal cum_repair_sec
        assert repair_backend is not None
        req = build_syntax_repair_request(
            cur_src, diag, slice_hint=slice_hint, sil_advisory=sil_advisory,
            sil_verification=sil_verification, backend_kind=backend_kind, parent_source=parent_source,
            freedom_mask=freedom_mask, vector_mode=vector_mode, ub_256kb_unlock=ub_256kb_unlock,
            lowered_excerpt=lowered_excerpt, lowered_excerpt_lang=lowered_excerpt_lang,
        )
        t0 = time.monotonic()
        try:
            raw = await repair_backend.generate(req)
        except Exception as exc:
            t1 = time.monotonic()
            repair_intervals.append((t0, t1)); cum_repair_sec += max(0.0, t1 - t0)
            return None, f"{diag} | repair_transport: {type(exc).__name__}: {exc}", True
        t1 = time.monotonic()
        repair_intervals.append((t0, t1)); cum_repair_sec += max(0.0, t1 - t0)
        try:
            rep = extract_python_dsl(raw)
        except Exception as exc:
            return None, f"{diag} | repair_parse: {type(exc).__name__}: {exc}", False
        if parent_source.strip() and (v := check_repair_monotonicity(parent_source, rep, backend_kind=backend_kind)):
            return None, f"{diag} | repair_monotonicity_violation: {'; '.join(v)}", False
        return rep, "", False

    async def _run_stage_a_phase(
        cid: str, cur_snip: str, *, start_attempt: int = 0, repaired: bool = False,
        initial_patch_fn: Callable[[str], str] | None = None, attempt_limit: int | None = None,
    ) -> tuple[L0L1CandidateCheck | None, tuple[str, str, StageAResult, bool, int] | None]:
        snip, is_rep, apf = cur_snip, repaired, (initial_patch_fn or patch_fn)
        eff_max = max_attempts if attempt_limit is None else max(0, attempt_limit)
        for att in range(start_attempt, eff_max + 1):
            try:
                patched = apf(snip)
            except Exception as exc:
                err = f"ast_patch_error: {type(exc).__name__}: {exc}"
                if is_noop_identical_failure(error_message=err):
                    return _fail(cid, "noop_identical", "", snip, is_rep, err, "NOOP_IDENTICAL", "NOOP_IDENTICAL"), None
                if repair_backend is None or att >= eff_max:
                    return _fail(cid, "stage_a", "", snip, is_rep, err, "STAGE_A_STATIC", "AST_PATCH_ERROR"), None
                ns, re_err, tr = await _try_repair(snip, err)
                if ns is None:
                    return _fail(cid, "stage_a", "", snip, is_rep, _repair_timeout_message(re_err, tr), "STAGE_A_STATIC", "SYNTAX_REPAIR_FAILED"), None
                snip, is_rep, apf = ns, True, patch_fn
                continue
            apf = patch_fn
            if base_sig is not None and (psig := _normalized_module_ast_signature(patched)) is not None and psig == base_sig:
                return _fail(cid, "noop_identical", patched, snip, is_rep, "noop_identical_to_parent", "NOOP_IDENTICAL", "NOOP_IDENTICAL"), None
            sa = await asyncio.to_thread(l0_checker, patched)
            if not sa.passed:
                err = sa.error_message or "l0_static_failed"
                if is_noop_identical_failure(error_message=err, result=sa):
                    return _fail(cid, "noop_identical", patched, snip, is_rep, err, "NOOP_IDENTICAL", "NOOP_IDENTICAL", sa=sa), None
                fp, fc = str(getattr(sa, "fault_plane", "") or "STAGE_A_STATIC"), str(getattr(sa, "fault_code", "") or "STAGE_A_FAILED")
                if repair_backend is None or att >= eff_max or not classify_stage_failure_for_repair(fc):
                    return _fail(cid, "stage_a", patched, snip, is_rep, err, fp, fc, sa=sa), None
                ns, re_err, tr = await _try_repair(snip, err)
                if ns is None:
                    return _fail(cid, "stage_a", patched, snip, is_rep, _repair_timeout_message(re_err, tr), fp, fc, sa=sa), None
                snip, is_rep = ns, True
                continue
            return None, (snip, patched, sa, is_rep, att)
        raise RuntimeError("unreachable repair termination")

    def _sb_repair_patch_fn(base_patched: str) -> Callable[[str], str]:
        if repatch_from_current_fn is None or not base_patched.strip():
            return patch_fn

        def _patch_rep(s: str) -> str:
            try:
                return repatch_from_current_fn(base_patched, s)
            except Exception:
                if not parse_scoped_search_replace_blocks(s):
                    return patch_fn(s)
                raise

        return _patch_rep

    async def _run_stage_b_and_repairs(
        cid: str, snip: str, patched: str, sa: StageAResult, rep: bool, att: int, *, attempt_limit: int | None = None
    ) -> L0L1CandidateCheck:
        cs, cp, csa, crep, catt = snip, patched, sa, rep, att
        eff_max = max_attempts if attempt_limit is None else max(0, attempt_limit)
        while True:
            async with bisheng_sem:
                sb = await asyncio.to_thread(l1_compiler, cp)
            if sb.passed:
                return L0L1CandidateCheck(cid, True, None, cp, cs, crep, stage_a=csa, stage_b=sb)
            err = sb.error_message or "l1_compile_failed"
            if is_noop_identical_failure(error_message=err, result=sb):
                return _fail(cid, "noop_identical", cp, cs, crep, err, "NOOP_IDENTICAL", "NOOP_IDENTICAL", sa=csa, sb=sb)
            fp, fc = str(getattr(sb, "fault_plane", "") or "STAGE_B_COMPILE"), str(getattr(sb, "fault_code", "") or "STAGE_B_FAILED")
            if repair_backend is None or catt >= eff_max or not classify_stage_failure_for_repair(fc):
                return _fail(cid, "stage_b", cp, cs, crep, err, fp, fc, sa=csa, sb=sb)
            ex, ex_lang = extract_lowered_compiler_excerpt_with_lang(sb)
            ns, re_err, tr = await _try_repair(cs, err, lowered_excerpt=ex, lowered_excerpt_lang=ex_lang)
            if ns is None:
                return _fail(cid, "stage_b", cp, cs, crep, _repair_timeout_message(re_err, tr), fp, fc, sa=csa, sb=sb)
            fchk, pstate = await _run_stage_a_phase(cid, ns, start_attempt=catt + 1, repaired=True, initial_patch_fn=_sb_repair_patch_fn(cp))
            if fchk is not None:
                return fchk
            assert pstate is not None
            cs, cp, csa, crep, catt = pstate

    async def _check_one(cid: str, snip: str, *, attempt_limit: int | None = None) -> L0L1CandidateCheck:
        fchk, pstate = await _run_stage_a_phase(cid, snip, attempt_limit=attempt_limit)
        if fchk is not None:
            return fchk
        assert pstate is not None
        return await _run_stage_b_and_repairs(cid, *pstate, attempt_limit=attempt_limit)

    async def _phase2_repair(chk: L0L1CandidateCheck) -> L0L1CandidateCheck:
        if chk.passed or chk.failed_tier == "noop_identical" or not classify_stage_failure_for_repair(str(chk.fault_code or "")):
            return chk
        fc, fp, err = str(chk.fault_code or ""), str(chk.fault_plane or ""), chk.error_message or "l1_compile_failed"
        if chk.failed_tier == "stage_b" and chk.stage_b is not None:
            ex, ex_lang = extract_lowered_compiler_excerpt_with_lang(chk.stage_b)
            ns, re_err, tr = await _try_repair(chk.snippet_source, err, lowered_excerpt=ex, lowered_excerpt_lang=ex_lang)
            if ns is None:
                return replace(chk, error_message=_repair_timeout_message(re_err, tr))
            ipf: Callable[[str], str] | None = _sb_repair_patch_fn(chk.patched_module_source)
        else:
            ns, re_err, tr = await _try_repair(chk.snippet_source, err)
            if ns is None:
                return _fail(chk.candidate_id, "stage_a", chk.patched_module_source, chk.snippet_source, chk.repaired, _repair_timeout_message(re_err, tr), fp or "STAGE_A_STATIC", ("SYNTAX_REPAIR_FAILED" if fc == "AST_PATCH_ERROR" else fc) or "STAGE_A_FAILED", sa=chk.stage_a)
            ipf = None
        fchk, pstate = await _run_stage_a_phase(chk.candidate_id, ns, start_attempt=1, repaired=True, initial_patch_fn=ipf)
        if fchk is not None:
            return fchk
        assert pstate is not None
        return await _run_stage_b_and_repairs(chk.candidate_id, *pstate)

    def _finalize(items: Sequence[L0L1CandidateCheck]) -> L0L1RoundResults:
        ws, cs = merge_interval_wall_sec(repair_intervals), float(cum_repair_sec)
        return L0L1RoundResults([replace(c, wall_repair_sec=ws, cum_repair_worker_sec=cs) for c in items], wall_repair_sec=ws, cum_repair_worker_sec=cs)

    if not (stage_a_cost_ranker is not None and cost_prescreen_top_k is not None and int(cost_prescreen_top_k) > 0):
        if skip_repair_when_survivors_exist and repair_backend is not None and max_attempts > 0:
            p1 = list(await asyncio.gather(*(_check_one(cid, s, attempt_limit=0) for cid, s in candidates)))
            return _finalize(p1 if any(c.passed for c in p1) else list(await asyncio.gather(*(_phase2_repair(c) for c in p1))))
        return _finalize(list(await asyncio.gather(*(_check_one(cid, s) for cid, s in candidates))))

    assert stage_a_cost_ranker is not None
    top_k = int(cost_prescreen_top_k or 0)
    sa_out = list(await asyncio.gather(*(_run_stage_a_phase(cid, s) for cid, s in candidates)))
    by_idx: dict[int, L0L1CandidateCheck] = {}
    passers: list[tuple[int, str, str, str, StageAResult, bool, int]] = []
    for idx, ((cid, _), (fchk, pstate)) in enumerate(zip(candidates, sa_out)):
        if fchk is not None:
            by_idx[idx] = fchk
        else:
            assert pstate is not None
            passers.append((idx, cid, *pstate))
    if not passers:
        return _finalize([by_idx[i] for i in range(len(candidates))])
    cids = {str(cid) for cid in creative_candidate_ids}
    exempt, same = [], []
    for item in passers:
        _i, cid, _sn, patched, sa, _r, _a = item
        ca = candidate_topology_axes_fn(patched, sa) if candidate_topology_axes_fn else extract_candidate_topology_axes(sa)
        cws = int(candidate_workspace_stages_fn(patched, sa)) if candidate_workspace_stages_fn else count_trackb_workspace_stages(patched)
        (exempt if is_paradigm_shift_exempt(ca, parent_topology_axes, cand_workspace_stages=cws, parent_workspace_stages=eff_pws, is_creative=(cid in cids)) else same).append(item)
    if len(same) > top_k:
        pbid, obid = {}, {}
        for ridx, (rcid, rsc) in enumerate(stage_a_cost_ranker([(cid, patched, sa) for _i, cid, _sn, patched, sa, _r, _a in same])):
            if (sid := str(rcid)) not in pbid:
                pbid[sid], obid[sid] = float(rsc), ridx
        ssame = sorted(same, key=lambda it: (pbid.get(it[1], float("inf")), obid.get(it[1], len(same) + it[0]), it[0]))
        kept_same, reserve_same = ssame[:top_k], ssame[top_k:]
    else:
        pbid, kept_same, reserve_same = {}, same, []
    kept, ks_ids, passed_l1 = [*exempt, *kept_same], {it[1] for it in kept_same}, 0
    if kept:
        for (idx, cid, *_), chk in zip(kept, await asyncio.gather(*(_run_stage_b_and_repairs(cid, cs, p, sa, r, a) for _i, cid, cs, p, sa, r, a in kept))):
            by_idx[idx] = chk
            if chk.passed and cid in ks_ids:
                passed_l1 += 1
    _ = (parent_analytical_us, stage_a_analytical_scorer)
    for idx, cid, cs, p, sa, r, a in reserve_same:
        if passed_l1 < top_k:
            chk = await _run_stage_b_and_repairs(cid, cs, p, sa, r, a)
            by_idx[idx] = chk
            if chk.passed:
                passed_l1 += 1
        else:
            by_idx[idx] = _fail(cid, "cost_prescreen", p, cs, r, f"cost_prescreen_pruned:pred_us={pbid.get(cid, float('inf')):.2f}", "COST_PRESCREEN", "COST_PRESCREEN_PRUNED", sa=sa)
    return _finalize([by_idx[i] for i in range(len(candidates))])


class StageGateConfig(Protocol):
    """Configuration surface consumed by StageGateCoordinator."""

    operator_name: str
    device_id: str
    warmup: int
    repeat: int
    max_syntax_repairs: int
    cost_prescreen_top_k: int | None
    pure_golden: bool


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


_ORIG_RUN_L0_L1_ROUND_FILTER = run_l0_l1_round_filter


class StageGateCoordinator:
    """Orchestrate candidate LLM generation, L0/L1 filtering, L2 surrogate eval, and guards."""

    def __init__(
        self, *, operator_name: str = "", target_module_path: Path | None = None,
        regime_specs: Sequence[RegimeTargetSpec] = (), initial_anchors: Sequence[AnchorCaseSpec] = (),
        scoreboard_anchors: Sequence[AnchorCaseSpec] = (), guardrail_cases: list[GuardrailCaseSpec] | None = None,
        backend: KernelLLMBackend | None = None, tracker: BayesianRuleTracker | None = None,
        cost_advisor: CostModelAdvisor | None = None,
        boundary_ledger: DeterministicSearchBoundaryLedger | None = None,
        target: AscendTarget | None = None, source_root: Path | None = None,
        env_file: Path | None = None, cannbench_repo: Path | None = None,
        checkpoint_path: Path | None = None, experience_lake_dir: Path | None = None,
        run_id: str = "", device_id: str = "1", l4_device_pool: Sequence[str] = (),
        warmup: int = 3, repeat: int = 5, max_syntax_repairs: int = 1,
        cost_prescreen_top_k: int | None = None, pure_golden: bool = False,
        enable_track_router: bool = False, sil_advisory: object | None = None,
        topology_axes: TopologyAxes | None = None, semantic_topology_axes: TopologyAxes | None = None,
        npu_port: NpuExecutionPort | None = None,
        l0_checker: Callable[..., StageAResult] | None = None,
        l1_compiler: Callable[[str, RegimeTargetSpec], StageBResult] | None = None,
        l2_batch_evaluator: Callable[[Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]], Mapping[str, CandidateSurrogateResult]] | None = None,
        static_width_dims_fn: Callable[[RegimeTargetSpec], Sequence[Mapping[str, object]]] | None = None,
        resolve_op_semantics_fn: Callable[[], PyTorchOpSemantics | None] | None = None,
        persist_artifact_fn: Callable[[L0L1CandidateCheck, int], None] | None = None,
        record_kb_event_fn: Callable[..., None] | None = None,
        config: StageGateConfig | None = None, active_incumbent_source: str = "",
        incumbent_anchor_us_by_case: Mapping[int, float] | None = None,
        freedom_masks: Mapping[str, ConstraintRelaxationMask] | None = None,
    ) -> None:
        self._config: StageGateConfig = config or _DefaultStageGateConfig(operator_name=operator_name, device_id=device_id, warmup=warmup, repeat=repeat, max_syntax_repairs=max_syntax_repairs, cost_prescreen_top_k=cost_prescreen_top_k, pure_golden=pure_golden)
        self._target_module_path = Path(target_module_path).resolve() if target_module_path else Path("module.py")
        self._regime_specs, self._initial_anchors = tuple(regime_specs), tuple(initial_anchors)
        self._scoreboard_anchors = tuple(scoreboard_anchors or initial_anchors)
        self._guardrail_cases: list[GuardrailCaseSpec] = guardrail_cases if guardrail_cases is not None else []
        self._backend, self._tracker = backend, (tracker or BayesianRuleTracker())
        self._target = target or AscendTarget("ascend.950pr")
        self._source_root = Path(source_root).resolve() if source_root else default_source_root()
        self._cost_advisor = cost_advisor or create_cost_advisor(mode="off", target=self._target, operator_name=self._config.operator_name)
        self._boundary_ledger = boundary_ledger or DeterministicSearchBoundaryLedger()
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        self._experience_lake_dir = Path(experience_lake_dir).resolve() if experience_lake_dir is not None else None
        self._run_id, self._l4_device_pool, self._enable_track_router = run_id, tuple(l4_device_pool), bool(enable_track_router)
        self._sil_advisory, self._topology_axes, self._semantic_topology_axes = sil_advisory, topology_axes, semantic_topology_axes
        self._npu_port, self._l0_checker, self._l1_compiler, self._l2_batch_evaluator = npu_port, l0_checker, l1_compiler, l2_batch_evaluator
        self._static_width_dims_fn, self._resolve_op_semantics_fn = static_width_dims_fn, resolve_op_semantics_fn
        self._persist_artifact_fn, self._record_kb_event_fn = persist_artifact_fn, record_kb_event_fn
        self._active_incumbent_source = active_incumbent_source
        self._incumbent_anchor_us_by_case: dict[int, float] = dict(incumbent_anchor_us_by_case or {})
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = dict(freedom_masks or {})
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        self._case_static_spec_cache: dict[int, dict[str, object]] = {}
        self._cases_yaml_sha256_cache: str | None = None
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._semantics_cache = OperatorSemanticsCache()
        self._l1_summary_cache: dict[tuple[str, str], BishengResourceSummary | None] = {}
        self._last_l2_timing_meta: dict[str, object] = {}
        self._last_guardrail_metrics_by_cid: dict[str, dict[int, AnchorEvalMetrics]] = {}
        self._last_tail_guard_reject_reason: str = ""
        self._pending_speculative: list[tuple[int, str, object, asyncio.Task[SpeculativeCreativeResult]]] = []

    def _get_backend(self) -> KernelLLMBackend:
        if self._backend is None:
            self._backend = create_llm_backend_from_env()
        return self._backend

    def _lookup_case_dims_safe(self, case_num: int, regime_tag: str = "") -> dict[str, object]:
        return lookup_case_dims_safe(
            self._config.operator_name, case_num, regime_tag=regime_tag, regime_specs=self._regime_specs,
            cache=self._case_dims_cache, cannbench_repo=self._cannbench_repo, env_file=self._env_file,
        )

    def _resolve_cases_yaml_sha256(self) -> str:
        if self._cases_yaml_sha256_cache is None:
            self._cases_yaml_sha256_cache = resolve_cases_yaml_sha256_for_operator(
                self._config.operator_name, cannbench_repo=self._cannbench_repo,
                env_file=self._env_file, source_root=self._source_root,
            )
        return self._cases_yaml_sha256_cache

    def _lookup_case_static_spec_safe(self, case_num: int) -> dict[str, object] | None:
        if not self._config.operator_name:
            return None
        cnum = int(case_num)
        if cnum in self._case_static_spec_cache:
            return dict(self._case_static_spec_cache[cnum])
        try:
            spec = dict(lookup_case_static_spec(self._config.operator_name, cnum, cannbench_repo=self._cannbench_repo, env_file=self._env_file))
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return None
        self._case_static_spec_cache[cnum] = dict(spec)
        return spec

    def _resolve_stage_b_anchor_case_specs(self) -> list[tuple[int, str, dict[str, object]]]:
        if not self._config.operator_name:
            return []
        specs, seen = [], set()
        for item in (*self._scoreboard_anchors, *self._guardrail_cases):
            cnum = int(item.case_num)
            if cnum not in seen:
                seen.add(cnum)
                if (ss := self._lookup_case_static_spec_safe(cnum)) is not None:
                    specs.append((cnum, str(item.regime_tag or ""), ss))
        return specs

    def _belongs_to_regime(self, dims: Mapping[str, object], regime_spec: RegimeTargetSpec) -> bool:
        if len(self._regime_specs) <= 1:
            return True
        try:
            return resolve_regime_for_case(self._config.operator_name, dims, self._regime_specs).regime_tag == regime_spec.regime_tag
        except (KeyError, ValueError, TypeError):
            return False

    def _derive_topology_axes(
        self, ledger: ObjectivePhysicalLedger | None, *, features: object | None = None, route: object | None = None
    ) -> TopologyAxes:
        return derive_topology_axes(
            self._config.operator_name, ledger, topology_axes=self._topology_axes,
            semantic_topology_axes=self._semantic_topology_axes, sil_advisory=self._sil_advisory,
            features=features, route=route,
        )

    def _persist_round_candidate_artifact(self, check: L0L1CandidateCheck, *, round_idx: int) -> None:
        if self._persist_artifact_fn is not None:
            self._persist_artifact_fn(check, round_idx)
        else:
            persist_round_candidate_artifact(self._checkpoint_path, check, round_idx=round_idx)

    def _record_kb_event(
        self, *, rule_id: str, postcondition_realized: bool, speed_gain: float = 0.0,
        topology_axes: object | None = None, round_idx: int = 0,
    ) -> None:
        if self._record_kb_event_fn is not None:
            self._record_kb_event_fn(rule_id=rule_id, postcondition_realized=postcondition_realized, speed_gain=speed_gain, topology_axes=topology_axes, round_idx=round_idx)
            return
        if self._experience_lake_dir is None:
            return
        try:
            append_kb_event_shard(
                self._experience_lake_dir, run_id=self._run_id, rule_id=rule_id,
                postcondition_realized=postcondition_realized, speed_gain=speed_gain,
                topology_axes=topology_axes.to_dict() if hasattr(topology_axes, "to_dict") else (dict(topology_axes) if isinstance(topology_axes, Mapping) else {}),
                round_idx=round_idx,
            )
        except (OSError, ValueError) as exc:
            logger.debug("KB event shard append skipped: %s", exc)

    def _compile_trackb_stage_b(
        self, cpp_source: str, selector_flags: Sequence[str] | str | None = None, *,
        artifact_store: ArtifactStore | None = None, regime_tag: str = "",
        verify_shared_link: bool = False, emit_shared_so_cache_dir: Path | None = None,
    ) -> StageBResult:
        return compile_trackb_stage_b(
            cpp_source, selector_flags, env_file=self._env_file, source_root=self._source_root,
            artifact_store=artifact_store, regime_tag=regime_tag,
            verify_shared_link=verify_shared_link, emit_shared_so_cache_dir=emit_shared_so_cache_dir,
        )

    def _resolve_operator_semantics_for_gates(self) -> PyTorchOpSemantics | None:
        return resolve_operator_semantics_for_gates(
            self._config.operator_name, source_root=self._source_root,
            pure_golden=bool(self._config.pure_golden), cannbench_repo=self._cannbench_repo,
            env_file=self._env_file, cache=self._semantics_cache, override_fn=self._resolve_op_semantics_fn,
        )

    def _resolve_trackb_baseline_tu(self, regime_spec: RegimeTargetSpec, *, incumbent_source: str = "") -> str | None:
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return None
        bsrc = incumbent_source or self._active_incumbent_source
        if not bsrc and self._target_module_path.is_file():
            try:
                bsrc = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                bsrc = ""
        btb = extract_trackb_cpp_source(bsrc, regime_spec.trackb_symbols) if bsrc else None
        return _render_trackb_tu_from_module(bsrc, btb[1], regime_spec, require_renderer=False) if btb else None

    def _default_l0_checker(
        self, module_source: str, regime_spec: RegimeTargetSpec, *, anchor_dims_override: Mapping[str, object] | None = None
    ) -> StageAResult:
        op_sem = self._resolve_operator_semantics_for_gates()
        btb_tu = self._resolve_trackb_baseline_tu(regime_spec)
        primary = run_l0_gate(
            module_source, regime_spec, target=self._target, enable_track_router=self._enable_track_router,
            freedom_masks=self._freedom_masks, active_incumbent_source=self._active_incumbent_source,
            target_module_path=self._target_module_path, anchor_dims_override=anchor_dims_override,
            op_semantics=op_sem, op_proto=self._semantics_cache.proto, baseline_tb_tu=btb_tu,
        )
        if not primary.passed or regime_spec.runtime_backend == "ascendc_trackb":
            return primary
        dims = dict(anchor_dims_override or regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims)
        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
        except (SyntaxError, AttributeError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError):
            return primary
        combine_fn = next((s for s in regime_spec.editable_symbols if s.startswith("_build_") and "combine" in s), "")
        if combine_fn:
            try:
                tree = ast.parse(module_source)
                cnode = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == combine_fn), None)
            except SyntaxError:
                cnode = None
            cdump = ast.dump(cnode, include_attributes=False) if cnode is not None else ""
            if cdump:
                bsrc = self._active_incumbent_source or (self._target_module_path.read_text(encoding="utf-8") if self._target_module_path.is_file() else "")
                bdump = ""
                if bsrc:
                    try:
                        bnode = next((n for n in ast.parse(bsrc).body if isinstance(n, ast.FunctionDef) and n.name == combine_fn), None)
                        bdump = ast.dump(bnode, include_attributes=False) if bnode is not None else ""
                    except SyntaxError:
                        bdump = ""
                splits = int(resolved.builder_kwargs.get("num_kv_splits", 1) or 1)
                if splits > 1 or not bsrc or cdump != bdump:
                    mkw: dict[str, object] = {**regime_spec.sample_builder_kwargs, **resolved.builder_kwargs, "num_kv_splits": splits if splits > 1 else 4, **resolved.combine_kwargs}
                    if cnode is not None and cnode.args.kwarg is None:
                        acc = {*(a.arg for a in cnode.args.posonlyargs), *(a.arg for a in cnode.args.args), *(a.arg for a in cnode.args.kwonlyargs)}
                        mkw = {k: v for k, v in mkw.items() if k in acc}
                    cres = run_stage_a_static(
                        module_source, factory_name=combine_fn, factory_kwargs=mkw,
                        pass_configs=resolved.pass_configs or None, compile_flags=resolved.compile_flags or None,
                        target=self._target, var_bindings=build_anchor_var_bindings(dims, kwargs=resolved.builder_kwargs) or None,
                        enable_track_router=self._enable_track_router, case_ratios=regime_spec.representative_case_ratios,
                        require_hardware_dma=bool(op_sem and not op_sem.has_indirect_index), require_cube_engine=False,
                        ordered_tensor_shapes=extract_ordered_tensor_shapes(case_entry=regime_spec.representative_case_entry, dims=dims, proto=self._semantics_cache.proto),
                    )
                    if not cres.passed:
                        return StageAResult(passed=False, prim_func=primary.prim_func, ledger=primary.ledger, features=primary.features, route=primary.route, error_message=f"combine_kernel({combine_fn}): {cres.error_message}", elapsed_ms=primary.elapsed_ms + cres.elapsed_ms)
        tb_ext = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols) if regime_spec.runtime_backend == "hybrid_trackb" else None
        if regime_spec.runtime_backend == "hybrid_trackb" and tb_ext is not None:
            mask = resolve_trackb_freedom_mask(regime_spec, freedom_masks=self._freedom_masks, baseline_source=self._active_incumbent_source, target_module_path=self._target_module_path)
            tu = _render_trackb_tu_from_module(module_source, tb_ext[1], regime_spec, require_renderer=False, anchor_dims_override=anchor_dims_override)
            tb_res = evaluate_ascendc_candidate(
                tu, freedom_mask=mask, target=self._target, enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios, baseline_cpp_source=btb_tu,
                require_hardware_dma=bool(op_sem and not op_sem.has_indirect_index), require_cube_engine=False,
                module_source=module_source, regime_tag=regime_spec.regime_tag,
            )
            tms = primary.elapsed_ms + tb_res.elapsed_ms
            if not tb_res.passed:
                return replace(tb_res, error_message=f"trackb_ascendc({tb_ext[0]}): {tb_res.error_message}", elapsed_ms=tms)
            if op_sem and op_sem.has_gemm_contraction and not (ledger_has_cube_engine(primary.ledger) or ledger_has_cube_engine(tb_res.ledger)):
                return replace(tb_res, passed=False, error_message=f"trackb_ascendc({tb_ext[0]}): missing_cube_engine_primitives: zero AIC.PIPE_M cycles", elapsed_ms=tms)
            if op_sem and not op_sem.has_indirect_index and not (ledger_has_hardware_dma(primary.ledger) or ledger_has_hardware_dma(tb_res.ledger)):
                return replace(tb_res, passed=False, error_message=f"trackb_ascendc({tb_ext[0]}): missing_hardware_dma_staging: zero PIPE_MTE2 cycles", elapsed_ms=tms)
            return replace(tb_res, elapsed_ms=tms)
        return primary

    def _run_l0(
        self, module_source: str, regime_spec: RegimeTargetSpec, *, anchor_dims_override: Mapping[str, object] | None = None
    ) -> StageAResult:
        return dispatch_l0_gate(
            module_source, regime_spec, l0_checker=self._l0_checker,
            default_fn=self._default_l0_checker, anchor_dims_override=anchor_dims_override,
        )

    def _collect_regime_cases_by_id(
        self, regime_spec: RegimeTargetSpec, *, extra_dims: Sequence[Mapping[str, object]] = ()
    ) -> dict[str, dict[str, object]]:
        cases_by_id: dict[str, dict[str, object]] = {}
        seen: set[int] = set()
        if (troot := resolve_cannbench_tasks_root(cannbench_repo=self._cannbench_repo, env_file=self._env_file, source_root=self._source_root)) and self._config.operator_name:
            try:
                cp = troot / get_operator_task(self._config.operator_name) / "cases.yaml"
                if cp.is_file():
                    for row in (yaml.safe_load(cp.read_text(encoding="utf-8")) or {}).get("cases") or ():
                        if isinstance(row, Mapping) and row.get("case_id") is not None:
                            cn = int(row["case_id"])
                            dims = self._lookup_case_static_spec_safe(cn) or self._lookup_case_dims_safe(cn, regime_spec.regime_tag)
                            if dims and self._belongs_to_regime(dims, regime_spec):
                                cases_by_id[str(cn)] = dict(dims)
                                seen.add(cn)
            except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
                pass
        for item in (*self._scoreboard_anchors, *self._guardrail_cases, *self._initial_anchors):
            if len(self._regime_specs) > 1 and item.regime_tag and item.regime_tag != regime_spec.regime_tag:
                continue
            cn = int(item.case_num)
            if cn in seen:
                continue
            dims = self._lookup_case_static_spec_safe(cn) or self._lookup_case_dims_safe(cn, str(item.regime_tag or regime_spec.regime_tag))
            if dims and self._belongs_to_regime(dims, regime_spec):
                cases_by_id[str(cn)] = dict(dims)
                seen.add(cn)
        for idx, ed in enumerate(extra_dims):
            if ed:
                cases_by_id[f"extra_{idx}"] = dict(ed)
        adims = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        if adims and (dict(adims) not in cases_by_id.values() or not cases_by_id):
            cases_by_id["__primary__"] = dict(adims)
        return cases_by_id

    def _run_l1(
        self, module_source: str, regime_spec: RegimeTargetSpec, *,
        static_width_dims_fn: Callable[[RegimeTargetSpec], Sequence[Mapping[str, object]]] | None = None,
    ) -> StageBResult:
        """Run Stage-B L1 compilation over unique post-selector CompileTaskSpecs."""
        eff_wfn = static_width_dims_fn if static_width_dims_fn is not None else self._static_width_dims_fn
        adims = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        ewd = list(eff_wfn(regime_spec)) if eff_wfn is not None else []
        if self._l1_compiler is not None:
            pb = self._l1_compiler(module_source, regime_spec)
            if not pb.passed:
                return normalize_primary_vf_overflow(pb)
            if stage_b_has_vf_overflow(pb):
                return build_width_failure_result(pb, pb.elapsed_sec)
            tot, seen = pb.elapsed_sec, ([dict(adims)] if adims else [])
            for wd_raw in ewd:
                if (wd := dict(wd_raw)) and wd not in seen:
                    seen.append(wd)
                    wb = self._l1_compiler(module_source, replace(regime_spec, representative_anchor_dims=wd))
                    tot += wb.elapsed_sec
                    if not wb.passed or stage_b_has_vf_overflow(wb):
                        return build_width_failure_result(wb, tot)
            return replace(pb, elapsed_sec=tot) if tot != pb.elapsed_sec else pb
        env_vals = parse_simple_env_file(self._env_file) if self._env_file else None
        utl_cache = resolve_unified_tl_cache_dir(env_vals=env_vals, source_root=self._source_root)
        store = ArtifactStore(utl_cache)
        tb_ext = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols) if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb") else None
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_ext is not None:
            tu = _render_trackb_tu_from_module(module_source, tb_ext[1], regime_spec, require_renderer=False)
            sfl = _resolve_selector_compile_flags(module_source, regime_spec)
            efl = tuple(resolve_trackb_compile_flags(tu, sfl))
            tb_res = self._compile_trackb_stage_b(tu, selector_flags=sfl or None, artifact_store=store, regime_tag=regime_spec.regime_tag)
            if not tb_res.passed:
                return normalize_primary_vf_overflow(tb_res, compile_flags=efl, builder_name="_render_trackb_tu")
            if stage_b_has_vf_overflow(tb_res, compile_flags=efl, builder_name="_render_trackb_tu"):
                return build_width_failure_result(tb_res, tb_res.elapsed_sec, compile_flags=efl, builder_name="_render_trackb_tu")
            seen_k, pend = {(tu, efl)}, []
            for cd in self._collect_regime_cases_by_id(regime_spec, extra_dims=ewd).values():
                if not cd:
                    continue
                cspec = replace(regime_spec, representative_anchor_dims=dict(cd))
                ctu = _render_trackb_tu_from_module(module_source, tb_ext[1], cspec, require_renderer=False, anchor_dims_override=dict(cd))
                cfl = _resolve_selector_compile_flags(module_source, cspec, dims=dict(cd))
                ceff = tuple(resolve_trackb_compile_flags(ctu, cfl))
                if (ctu, ceff) not in seen_k:
                    seen_k.add((ctu, ceff)); pend.append((ctu, cfl, ceff))
            xres = _pool_map(pend, lambda it: (it[2], self._compile_trackb_stage_b(it[0], selector_flags=it[1] or None, artifact_store=store, regime_tag=regime_spec.regime_tag)))
            tot = tb_res.elapsed_sec + sum(r.elapsed_sec for _, r in xres)
            for efl_i, wb in xres:
                if not wb.passed or stage_b_has_vf_overflow(wb, compile_flags=efl_i, builder_name="_render_trackb_tu"):
                    return build_width_failure_result(wb, tot, compile_flags=efl_i, builder_name="_render_trackb_tu")
            return replace(tb_res, elapsed_sec=tot) if tot != tb_res.elapsed_sec else tb_res
        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=adims)
        except (SyntaxError, AttributeError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError) as exc:
            return StageBResult(passed=False, error_message=f"l1_config_resolve_error: {type(exc).__name__}: {exc}")
        tb_b, cached_tb_rec, tb_sec = None, None, 0.0
        if regime_spec.runtime_backend == "hybrid_trackb" and tb_ext is not None:
            tb_b, cached_tb_rec, tb_sec, fres = run_hybrid_trackb_stage_b(
                module_source, tb_ext, regime_spec, resolved, utl_cache, store,
                env_file=self._env_file, source_root=self._source_root,
                scoreboard_anchors=self._scoreboard_anchors, guardrail_cases=self._guardrail_cases,
                lookup_dims_fn=lambda cn, rt: self._lookup_case_static_spec_safe(cn) or self._lookup_case_dims_safe(cn, rt),
            )
            if fres is not None:
                return fres
        anchor_case_specs, cases_sha = self._resolve_stage_b_anchor_case_specs(), self._resolve_cases_yaml_sha256()
        cases_by_id = self._collect_regime_cases_by_id(regime_spec, extra_dims=ewd)
        plan: RegimeCompilePlan | None = None
        if cases_by_id:
            try:
                ext = ModuleSelectorPlanExtractor(
                    target=self._target, builder_name=resolved.builder_name, config_selector_name=regime_spec.config_selector_fn or "",
                    default_builder_kwargs=dict(resolved.builder_kwargs), default_pass_configs=dict(resolved.pass_configs or {}),
                    default_compile_flags=tuple(resolved.compile_flags or ()),
                )
                plan = build_post_selector_regime_plan(module_source, regime_spec.regime_tag or "default", cases_by_id, ext, env_file=self._env_file, source_root=self._source_root)
            except (SyntaxError, AttributeError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError):
                plan = None

        def _check_l1(bname: str, bkw: Mapping[str, object], pc: Mapping[str, object] | None, cf: Sequence[str] | None, d: dict[str, object] | None, acs: Sequence[tuple[int, str, dict[str, object]]] | None) -> StageBResult:
            return run_l1_compile_check(
                module_source, builder_name=bname, builder_kwargs=bkw, config_selector_name=regime_spec.config_selector_fn or None,
                pass_configs=pc or None, compile_flags=cf or None, dims=d, anchor_case_specs=acs, operator_name=self._config.operator_name,
                cases_yaml_sha256=cases_sha, source_root=self._source_root, env_file=self._env_file, regime_tag=regime_spec.regime_tag,
                tl_cache_dir=utl_cache, artifact_store=store,
            )

        cann_setup = ((env_vals or {}).get("TILEFOUNDRY_NPU_CANN_SETUP") or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
        tl_src = ((env_vals or {}).get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()
        pb: StageBResult | None = None
        tot = 0.0
        if (bool(cann_setup) or bool(tl_src)) and plan is not None and plan.unique_specs_by_key:
            rc = StageBCompileGate(plan, store, env_file=self._env_file, source_root=self._source_root).run(module_source)
            tot, bsum = rc.elapsed_sec, BishengResourceSummary(stack_bytes=rc.max_stack_bytes)
            if not rc.passed:
                fc = rc.fault_code or "L1_COMPILE_FAILED"
                pb = StageBResult(passed=False, error_message=f"{'l1_vf_stack_overflow' if fc == 'VF_STACK_OVERFLOW' else 'l1_compile_failed'}: {rc.diagnostic}", full_diagnostic=rc.diagnostic, compiler_summary=bsum, elapsed_sec=tot, fault_plane="STAGE_B_COMPILE", fault_code=fc, failed_tier="stage_b")
            else:
                fk = next(iter(plan.unique_specs_by_key))
                art = store.get_verified_artifact(fk)
                pb = StageBResult(passed=True, elapsed_sec=tot, compiler_summary=bsum, artifact_id=art.artifact_id if art else fk, binary_paths=art.binary_paths if art else ())
        elif plan is not None and plan.unique_specs_by_key:
            fdims = {csp.task_key: dict(cases_by_id.get(cid) or adims or {}) for cid, cspecs in plan.tasks_by_case_id.items() for csp in cspecs}
            sitems = list(plan.unique_specs_by_key.values())
            _run_spec = lambda sp: (sp, _check_l1(sp.builder_name, dict(sp.static_kwargs), dict(sp.pass_configs), list(sp.compile_flags), fdims.get(sp.task_key) or (dict(adims) if adims else None), anchor_case_specs or None))
            psp = sitems[0]
            _, pb = _run_spec(psp)
            tot = pb.elapsed_sec
            if pb.passed:
                if stage_b_has_vf_overflow(pb, compile_flags=psp.compile_flags, builder_name=psp.builder_name):
                    return build_width_failure_result(pb, tot, compile_flags=psp.compile_flags, builder_name=psp.builder_name)
                exres = _pool_map(sitems[1:], _run_spec)
                tot += sum(wb.elapsed_sec for _, wb in exres)
                for sp, wb in exres:
                    if not wb.passed or stage_b_has_vf_overflow(wb, compile_flags=sp.compile_flags, builder_name=sp.builder_name):
                        return build_width_failure_result(wb, tot, compile_flags=sp.compile_flags, builder_name=sp.builder_name)
            else:
                pb = normalize_primary_vf_overflow(pb, compile_flags=psp.compile_flags, builder_name=psp.builder_name)
        else:
            pb = _check_l1(resolved.builder_name, resolved.builder_kwargs, resolved.pass_configs, resolved.compile_flags, dict(adims) if adims else None, anchor_case_specs or None)
            tot = pb.elapsed_sec
            if pb.passed:
                if stage_b_has_vf_overflow(pb, compile_flags=resolved.compile_flags, builder_name=resolved.builder_name):
                    return build_width_failure_result(pb, tot, compile_flags=resolved.compile_flags, builder_name=resolved.builder_name)
                seen_fb = [dict(adims)] if adims else []
                for wd_raw in ewd:
                    if not (wd := dict(wd_raw)) or wd in seen_fb:
                        continue
                    seen_fb.append(wd)
                    try:
                        wr = resolve_regime_config(module_source, regime_spec, dims=wd)
                    except (SyntaxError, AttributeError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError):
                        wr = resolved
                    wb = _check_l1(wr.builder_name, wr.builder_kwargs, wr.pass_configs, wr.compile_flags, wd, None)
                    tot += wb.elapsed_sec
                    if not wb.passed or stage_b_has_vf_overflow(wb, compile_flags=wr.compile_flags, builder_name=wr.builder_name):
                        return build_width_failure_result(wb, tot, compile_flags=wr.compile_flags, builder_name=wr.builder_name)
            else:
                pb = normalize_primary_vf_overflow(pb, compile_flags=resolved.compile_flags, builder_name=resolved.builder_name)
        assert pb is not None
        if not pb.passed and regime_spec.runtime_backend == "hybrid_trackb" and tb_ext is not None and not _has_prim_func_builder(module_source, regime_spec.builder_candidates):
            if tb_b is not None:
                return normalize_primary_vf_overflow(replace(tb_b, elapsed_sec=tb_sec) if tb_sec != tb_b.elapsed_sec else tb_b, compile_flags=resolved.compile_flags, builder_name="_render_trackb_tu")
            if cached_tb_rec is not None:
                return StageBResult(passed=True, elapsed_sec=tb_sec, binary_paths=cached_tb_rec.binary_paths, artifact_id=cached_tb_rec.artifact_id)
        if not pb.passed:
            return pb
        ntot = tot + tb_sec
        return replace(pb, elapsed_sec=ntot) if ntot != pb.elapsed_sec else pb

    def _l1_summary_reuse_key(self, module_source: str, regime_spec: RegimeTargetSpec) -> tuple[str, str]:
        src_sha = source_sha256(module_source)
        adims = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        try:
            res = resolve_regime_config(module_source, regime_spec, dims=adims)
            tb_ext = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols) if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb") else None
            tb_hash = source_sha256(tb_ext[1]) if tb_ext is not None else ""
        except (ValueError, SyntaxError, KeyError, AttributeError, TypeError, NameError, ImportError, RuntimeError, OSError):
            return (src_sha, f"regime:{regime_spec.regime_tag}")
        sig = hashlib.sha256("|".join((str(res.builder_name), repr(tuple(res.compile_flags)), str(regime_spec.runtime_backend), tb_hash)).encode("utf-8")).hexdigest()
        return (src_sha, f"build:{sig}")

    def _l1_summary_for_first_round(self, module_source: str, regime_spec: RegimeTargetSpec) -> BishengResourceSummary | None:
        ckey = (source_sha256(module_source), regime_spec.regime_tag)
        if ckey in self._l1_summary_cache:
            return self._l1_summary_cache[ckey]
        rkey = self._l1_summary_reuse_key(module_source, regime_spec)
        if rkey in self._l1_summary_cache:
            self._l1_summary_cache[ckey] = self._l1_summary_cache[rkey]
            return self._l1_summary_cache[rkey]
        res = self._run_l1(module_source, regime_spec)
        if not res.passed:
            logger.warning("First-round parent L1 for regime %s failed: %s", regime_spec.regime_tag, res.error_message)
        summary = res.compiler_summary if res.passed else None
        self._l1_summary_cache[ckey] = self._l1_summary_cache[rkey] = summary
        return summary

    def _normalize_paired_l2_results(
        self, raw_results: Mapping[str, CandidateSurrogateResult], incumbent_slot_result: CandidateSurrogateResult | None,
        *, incumbent_anchor_us_by_case: Mapping[int, float] | None = None,
    ) -> dict[str, CandidateSurrogateResult]:
        slot_by_c = {int(a.case_num): float(a.cand_us) for a in incumbent_slot_result.anchors if a.cand_us > 0.0} if incumbent_slot_result else {}
        inc_map = incumbent_anchor_us_by_case if incumbent_anchor_us_by_case is not None else self._incumbent_anchor_us_by_case
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, cres in raw_results.items():
            changed, upd = False, []
            for anc in cres.anchors:
                psp = float(getattr(anc, "paired_speedup_vs_incumbent", 0.0) or 0.0)
                ibus = inc_map.get(int(anc.case_num), slot_by_c.get(int(anc.case_num), 0.0))
                if psp > 0.0 and ibus > 0.0 and anc.accuracy_passed:
                    eus = ibus / psp
                    upd.append(replace(
                        anc, cand_us=eus, speedup=(anc.ref_perf_us / eus) if anc.ref_perf_us > 0.0 else anc.speedup,
                        roofline_eta=min(1.0, max(0.0, anc.t_hw_us / eus)) if anc.t_hw_us > 0.0 else anc.roofline_eta,
                        eq3_score=compute_case_eq3_score(anc.ref_perf_us, eus, anc.t_hw_us),
                    ))
                    changed = True
                else:
                    upd.append(anc)
            out[cid] = aggregate_surrogate_result(cres.candidate_id, cres.operator_name, tuple(upd), error_message=cres.error_message) if changed else cres
        return out

    def _run_l2_batch(
        self, candidates: Sequence[tuple[str, Path]], active_anchors: Sequence[AnchorCaseSpec], *,
        include_guardrails: bool = True, incumbent_source: str = "",
        incumbent_anchor_us_by_case: Mapping[int, float] | None = None,
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        self._last_l2_timing_meta = {}
        eff_inc = incumbent_source or self._active_incumbent_source
        if not eff_inc and self._l2_batch_evaluator is None and self._npu_port is None and self._target_module_path.is_file():
            try:
                eff_inc = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                eff_inc = ""
        cids = {cid for cid, _ in candidates}
        prepend_inc = bool(candidates and eff_inc.strip() and not ({"incumbent_slot", "__incumbent__", "incumbent_baseline"} & cids))
        sb_cnums = {int(a.case_num) for a in active_anchors}
        ex_gr, gr_cnums = [], set()
        if include_guardrails and self._guardrail_cases and active_anchors:
            for g in self._guardrail_cases:
                if (gn := int(g.case_num)) not in sb_cnums and gn not in gr_cnums:
                    ex_gr.append(g.to_anchor_spec())
                    gr_cnums.add(gn)
        eval_anchors = [*active_anchors, *ex_gr] if ex_gr else list(active_anchors)
        l2_meta: dict[str, object] = {"prewarm_stats": {}, "l2_prewarm_sec": 0.0, "l2_npu_sec": None, "null_launch_us": None}

        def _cap(ev: object) -> None:
            if isinstance(rp := getattr(ev, "prewarm_stats", None), Mapping):
                l2_meta["prewarm_stats"] = {str(k): int(v) for k, v in rp.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for k in ("l2_prewarm_sec", "l2_npu_sec", "null_launch_us"):
                if isinstance(rv := getattr(ev, k, None), (int, float)) and not isinstance(rv, bool):
                    l2_meta[k] = float(rv)

        def _dispatch(bcands: Sequence[tuple[str, Path]]) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
            if self._npu_port is not None:
                ev = self._npu_port.run_l2_batch(bcands, eval_anchors)
                _cap(ev)
                return (dict(ev), dict(ev.run_ids)) if isinstance(ev, L2BatchResults) else (dict(ev), {})
            if self._l2_batch_evaluator is not None:
                ev = self._l2_batch_evaluator(bcands, eval_anchors)
                _cap(ev)
                return (dict(ev), dict(ev.run_ids)) if isinstance(ev, L2BatchResults) else (dict(ev), {})
            results, run_id, _ = run_surrogate_batch_via_agent_gates(
                self._config.operator_name, bcands, eval_anchors, source_root=self._source_root,
                env_file=self._env_file, device_id=self._config.device_id, device_pool=self._l4_device_pool,
                warmup=self._config.warmup, repeat=self._config.repeat, require_all_passed=False,
            )
            _cap(results)
            return results, {cid: run_id for cid in results}

        if prepend_inc:
            with tempfile.TemporaryDirectory(prefix="tf_camp_inc_") as idir:
                ipath = Path(idir) / "incumbent_slot.py"
                ipath.write_text(eff_inc, encoding="utf-8")
                results, run_ids = _dispatch([("incumbent_slot", ipath), *candidates])
        else:
            results, run_ids = _dispatch(candidates)
        self._last_l2_timing_meta = l2_meta
        inc_slot = results.pop("incumbent_slot", None)
        run_ids.pop("incumbent_slot", None)
        norm = self._normalize_paired_l2_results(results, inc_slot, incumbent_anchor_us_by_case=incumbent_anchor_us_by_case)
        if gr_cnums:
            stripped: dict[str, CandidateSurrogateResult] = {}
            for cid, cr in norm.items():
                self._last_guardrail_metrics_by_cid[cid] = {int(a.case_num): a for a in cr.anchors if int(a.case_num) in gr_cnums}
                kept = tuple(a for a in cr.anchors if int(a.case_num) not in gr_cnums)
                stripped[cid] = aggregate_surrogate_result(cr.candidate_id, cr.operator_name, kept, error_message=cr.error_message) if kept else replace(cr, anchors=())
            final_map = stripped
        else:
            final_map = norm
        pw = l2_meta.get("prewarm_stats")
        wrapped = L2BatchResults(
            final_map, run_ids, prewarm_stats=pw if isinstance(pw, Mapping) else None,
            l2_prewarm_sec=float(l2_meta.get("l2_prewarm_sec") or 0.0), l2_npu_sec=float(l2_meta.get("l2_npu_sec") or 0.0),
            null_launch_us=float(l2_meta["null_launch_us"]) if isinstance(l2_meta.get("null_launch_us"), (int, float)) else None,
        )
        return wrapped, run_ids

    def _unchanged_sibling_regimes(self, incumbent_src: str, candidate_src: str, active_spec: RegimeTargetSpec) -> set[str]:
        if not incumbent_src.strip() or not candidate_src.strip():
            return set()
        op_entry = resolve_operator_entrypoint_safe(self._config.operator_name)
        unchanged: set[str] = set()
        for spec in self._regime_specs:
            if spec.regime_tag == active_spec.regime_tag:
                continue
            try:
                _, isnip = extract_regime_snippet(incumbent_src, spec, operator_entrypoint=op_entry, sibling_specs=self._regime_specs)
                _, csnip = extract_regime_snippet(candidate_src, spec, operator_entrypoint=op_entry, sibling_specs=self._regime_specs)
                ast.parse(isnip)
                ast.parse(csnip)
                ih, ch = _normalized_module_ast_signature(isnip), _normalized_module_ast_signature(csnip)
                if ih is not None and ch is not None and ih == ch:
                    unchanged.add(spec.regime_tag)
            except (SyntaxError, ValueError):
                continue
        return unchanged

    def _retest_single_or_tail(
        self, candidate_id: str, candidate_source: str, anchors_in: Sequence[AnchorCaseSpec]
    ) -> CandidateSurrogateResult | None:
        if not anchors_in:
            return None
        with tempfile.TemporaryDirectory(prefix="tf_camp_retest_") as tdir:
            cf = Path(tdir) / f"{candidate_id}.py"
            cf.write_text(candidate_source, encoding="utf-8")
            try:
                l2_res, _ = self._run_l2_batch([(candidate_id, cf)], list(anchors_in), include_guardrails=False)
            except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                logger.warning("Anchor recheck failed for %s: %s", candidate_id, exc)
                return None
        return l2_res.get(candidate_id)

    def _retest_noisy_tail_anchors(
        self, *, candidate_id: str, candidate_source: str, cur_res: CandidateSurrogateResult,
        regressed_anchors: Sequence[AnchorEvalMetrics], active_anchors: Sequence[AnchorCaseSpec],
    ) -> CandidateSurrogateResult | None:
        by_c = {int(a.case_num): a for a in active_anchors}
        specs, seen = [], set()
        for anc in regressed_anchors:
            if (cn := int(anc.case_num)) in seen:
                continue
            seen.add(cn)
            specs.append(by_c.get(cn) or AnchorCaseSpec(task=f"operator/{cur_res.operator_name}", case_num=cn, regime_tag=anc.regime_tag, ref_perf_us=anc.ref_perf_us, t_hw_us=anc.t_hw_us))
        ret = self._retest_single_or_tail(candidate_id, candidate_source, specs)
        if ret is None or not ret.anchors:
            return None
        rb = {int(a.case_num): a for a in ret.anchors}
        return aggregate_surrogate_result(cur_res.candidate_id, cur_res.operator_name, tuple(rb.get(int(a.case_num), a) for a in cur_res.anchors), error_message=cur_res.error_message)

    def _evaluate_dual_layer_tail_guard(
        self, *, candidate_id: str, candidate_source: str, top_res: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult, regime_spec: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec], incumbent_source: str = "",
        guardrail_cases: Sequence[GuardrailCaseSpec] | None = None,
    ) -> tuple[bool, CandidateSurrogateResult]:
        eff_inc = incumbent_source or self._active_incumbent_source
        unchanged = self._unchanged_sibling_regimes(eff_inc, candidate_source, regime_spec) if (eff_inc and eff_inc != candidate_source) else set()
        eff_gr = list(guardrail_cases) if guardrail_cases is not None else self._guardrail_cases
        ok, updated, reason = evaluate_promotion_gate(
            candidate_id=candidate_id, cur_res=top_res, best_result=best_result, regime_spec=regime_spec,
            guardrail_cases=eff_gr, cand_guardrail_metrics=self._last_guardrail_metrics_by_cid.get(candidate_id, {}),
            unchanged_siblings=unchanged, active_anchors=active_anchors,
            retest_noisy_fn=lambda reg, cur: self._retest_noisy_tail_anchors(candidate_id=candidate_id, candidate_source=candidate_source, cur_res=cur, regressed_anchors=reg, active_anchors=active_anchors),
            recheck_single_sibling_fn=lambda sp: self._retest_single_or_tail(candidate_id, candidate_source, [sp]),
        )
        self._last_tail_guard_reject_reason = reason
        return ok, updated

    def _record_llm_failure(
        self, *, wid: str, rule: AscendCausalRule | None, err: str, is_transport_error: bool,
        regime_spec: RegimeTargetSpec, parent_sha: str, parent_axes: TopologyAxes,
        round_idx: int, failed_entries: list[tuple[str, str, str]],
    ) -> None:
        failed_entries.append((wid, "llm_transport" if is_transport_error else "llm_generate", err[:1500]))
        if rule is not None and not is_transport_error:
            self._tracker.record_outcome(rule.rule_id, postcondition_realized=False, topology_axes=parent_axes)
            self._tracker.record_scoped_failure(regime_tag=regime_spec.regime_tag, rule_id=rule.rule_id, parent_sha=parent_sha, current_round=round_idx, reason="llm_generate_parse_failed")
            self._record_kb_event(rule_id=rule.rule_id, postcondition_realized=False, topology_axes=parent_axes, round_idx=round_idx)

    def _absorb_creative_task(
        self, task: asyncio.Task[SpeculativeCreativeResult], *, origin_round_idx: int, wid: str,
        rule_obj: object, batch: CandidateBatch, parent_sha: str, parent_axes: TopologyAxes,
        raw_candidates: list[tuple[str, str]], meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]],
        failed_entries: list[tuple[str, str, str]],
    ) -> None:
        _ = origin_round_idx
        rule = rule_obj if isinstance(rule_obj, AscendCausalRule) else None
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:
            meta_by_id[wid] = ("creative_explore", rule, "")
            self._record_llm_failure(wid=wid, rule=rule, err=f"llm_transport:{type(exc).__name__}:{exc}", is_transport_error=True, regime_spec=batch.regime_spec, parent_sha=parent_sha, parent_axes=parent_axes, round_idx=batch.round_idx, failed_entries=failed_entries)
            return
        res = task.result()
        meta_by_id[res.worker_id] = ("creative_explore", rule, res.hypothesis)
        if res.snippet is None:
            self._record_llm_failure(wid=res.worker_id, rule=rule, err=res.error, is_transport_error=res.is_transport_error, regime_spec=batch.regime_spec, parent_sha=parent_sha, parent_axes=parent_axes, round_idx=batch.round_idx, failed_entries=failed_entries)
            return
        raw_candidates.append((res.worker_id, res.snippet))

    @staticmethod
    def _classify_l2_exception(exc: BaseException) -> TypedFault | None:
        if isinstance(exc, OccupancyExhaustedError) or type(exc).__name__ == "OccupancyExhaustedError" or str(getattr(exc, "fault_plane", "")) == "INFRA_OCCUPANCY":
            return TypedFault(plane=FaultPlane.INFRA_OCCUPANCY, code="L2_INFRA_OCCUPANCY", failed_tier="l2_infra_occupancy", detail=str(exc)[:500], retriable=True)
        msg_low = str(exc).lower()
        if isinstance(exc, AgentGatesDeviceFaultError) or "device_fault" in msg_low:
            ep = coerce_fault_plane(getattr(exc, "fault_plane", "")) if isinstance(exc, AgentGatesDeviceFaultError) else FaultPlane.INFRA_DEVICE_FAULT
            ep = FaultPlane.INFRA_DEVICE_FAULT if ep == FaultPlane.UNMODELED_FAULT else ep
            ec = (str(getattr(exc, "fault_code", "") or "") if isinstance(exc, AgentGatesDeviceFaultError) else "") or ("L2_INFRA_OCCUPANCY" if ep == FaultPlane.INFRA_OCCUPANCY else "L2_INFRA_DEVICE_FAULT")
            et = "l2_infra_occupancy" if ep == FaultPlane.INFRA_OCCUPANCY else "l2_infra_device_fault"
            return TypedFault(plane=ep, code=ec, failed_tier=et, detail=str(exc)[:500], retriable=True)
        if isinstance(exc, (TimeoutError, OSError, subprocess.SubprocessError)) or any(t in msg_low for t in ("occupancy", "resource busy", "device_busy", "card_busy")):
            return TypedFault(plane=FaultPlane.INFRA_OCCUPANCY, code="L2_INFRA_OCCUPANCY", failed_tier="l2_infra_occupancy", detail=str(exc)[:500], retriable=True)
        return None

    async def cancel_pending_speculative(self) -> None:
        """Cancel all in-flight speculative creative tasks cleanly."""
        pending = list(self._pending_speculative)
        self._pending_speculative.clear()
        tasks = [t for _, _, _, t in pending if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def screen_and_score_round(self, snapshot: CampaignStateSnapshot, batch: CandidateBatch) -> StageGateOutcome:
        """Execute StageGatePort protocol screening against an immutable CampaignStateSnapshot."""
        return await self.execute_round_gates(batch, snapshot=snapshot)

    async def execute_round_gates(
        self, batch: CandidateBatch, working_anchors: Sequence[AnchorCaseSpec] = (), *,
        backend: KernelLLMBackend | None = None, parent_sha: str = "",
        parent_workload: CostFeatures | Mapping[str, object] | None = None,
        snapshot: CampaignStateSnapshot | None = None,
    ) -> StageGateOutcome:
        """Execute LLM generation, speculative creative harvesting, and L0->L1->L2 gates."""
        eff_inc = snapshot.incumbent_source if snapshot is not None else self._active_incumbent_source
        inc_anchor_map = dict(snapshot.incumbent_anchor_us_by_case) if snapshot is not None else self._incumbent_anchor_us_by_case
        eff_backend = backend or (self._get_backend() if (self._backend is None and batch.worker_specs) else self._backend)
        eff_psha = parent_sha or source_sha256(batch.parent_module_source)
        eff_paxes = batch.parent_axes or self._derive_topology_axes(batch.parent_ledger)
        eff_mask = batch.active_mask or resolve_trackb_freedom_mask(batch.regime_spec, freedom_masks=self._freedom_masks, baseline_source=eff_inc, target_module_path=self._target_module_path)
        raw_candidates: list[tuple[str, str]] = []
        failed_entries: list[tuple[str, str, str]] = []
        meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]] = {}
        still_pending = []
        for or_idx, wid, robj, task in self._pending_speculative:
            if not task.done():
                still_pending.append((or_idx, wid, robj, task))
            else:
                self._absorb_creative_task(task, origin_round_idx=or_idx, wid=wid, rule_obj=robj, batch=batch, parent_sha=eff_psha, parent_axes=eff_paxes, raw_candidates=raw_candidates, meta_by_id=meta_by_id, failed_entries=failed_entries)
        self._pending_speculative = still_pending

        async def _call_exploit(wid: str, wkind: str, rule: AscendCausalRule | None, req: GenerationRequest) -> tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]:
            assert eff_backend is not None
            try:
                return wid, wkind, rule, extract_python_dsl(await eff_backend.generate(req)), "", "", False
            except (OSError, subprocess.SubprocessError, TimeoutError, RuntimeError, ConnectionError) as exc:
                return wid, wkind, rule, None, "", f"llm_transport:{type(exc).__name__}:{exc}", True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return wid, wkind, rule, None, "", f"llm_parse:{type(exc).__name__}:{exc}", False

        async def _call_creative(wid: str, req: GenerationRequest) -> SpeculativeCreativeResult:
            assert eff_backend is not None
            try:
                cr = extract_creative_payload(await eff_backend.generate(req))
                rh, rm = str(cr.get("physical_hypothesis", "")).strip(), str(cr.get("target_local_metric", "")).strip()
                return SpeculativeCreativeResult(batch.round_idx, wid, cr["dsl_source"], f"{rh} [target_metric={rm}]" if (rh and rm) else (rh or rm), "", False)
            except (OSError, subprocess.SubprocessError, TimeoutError, RuntimeError, ConnectionError) as exc:
                return SpeculativeCreativeResult(batch.round_idx, wid, None, "", f"llm_transport:{type(exc).__name__}:{exc}", True)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return SpeculativeCreativeResult(batch.round_idx, wid, None, "", f"llm_parse:{type(exc).__name__}:{exc}", False)

        ex_specs = [(wid, wk, r, req) for wid, wk, r, req in batch.worker_specs if wk != "creative_explore"]
        cr_specs = [(wid, r, req) for wid, wk, r, req in batch.worker_specs if wk == "creative_explore"]

        class _TimedRepair:
            def __init__(self, inner: KernelLLMBackend) -> None:
                self._inner, self.repair_sec = inner, 0.0

            async def generate(self, request: GenerationRequest, *, effort_override: object = None) -> str:
                t0 = time.monotonic()
                try:
                    return await (self._inner.generate(request, effort_override=effort_override) if effort_override is not None else self._inner.generate(request))
                finally:
                    self.repair_sec += max(0.0, time.monotonic() - t0)

        def _make_filter_runner(non_cr: int) -> tuple[object | None, str, _TimedRepair | None, Callable[..., object]]:
            pk, ranker, paus, scorer = None, None, None, None
            gd = batch.gate_decision
            if gd is not None and gd.engaged and gd.allow_candidate_pruning:
                if self._config.cost_prescreen_top_k is not None and int(self._config.cost_prescreen_top_k) > 0:
                    pk = int(self._config.cost_prescreen_top_k)
                elif self._config.cost_prescreen_top_k is None and non_cr >= 2:
                    pk = max(1, non_cr - 1)

                def _rank(passers: Sequence[tuple[str, str, StageAResult]]) -> list[tuple[str, float]]:
                    if self._cost_advisor.underlying_cost_model is not None:
                        return [(it.candidate_id, it.pred_us) for it in rank_stage_a_candidates_with_cost_model(passers, cost_model=self._cost_advisor.underlying_cost_model, live_plan=batch.live_plan, axes=eff_paxes, blueprint=batch.blueprint, sil_advisory=self._sil_advisory, parent_ledger=batch.parent_ledger)]
                    filt = self._cost_advisor.filter_stage_a_passers(passers, decision=gd, live_plan=batch.live_plan, axes=eff_paxes, blueprint=batch.blueprint, sil_advisory=self._sil_advisory, parent_ledger=batch.parent_ledger, top_k=pk)
                    return [(str(it.candidate_id), float(it.pred_us)) if hasattr(it, "candidate_id") else (str(it[0]), float(it[1])) for it in filt]

                ranker = _rank if pk is not None else None
                epw = parent_workload or (build_cost_features_from_ledger(batch.parent_ledger, live_plan=batch.live_plan, axes=eff_paxes, blueprint=batch.blueprint, sil_advisory=self._sil_advisory, target=self._target) if batch.parent_ledger else None)
                pl0 = extract_cost_pred_us(self._cost_advisor.predict_candidate_us(epw, bisheng_res=None) if epw is not None else None)
                paus = pl0 if pl0 > 0.0 else None

                def _score(_c: str, _s: str, sa: StageAResult) -> float:
                    ca = self._derive_topology_axes(sa.ledger, features=sa.features, route=sa.route) if sa.ledger else eff_paxes
                    ft = build_cost_features_from_ledger(sa.ledger, live_plan=batch.live_plan, axes=ca, blueprint=batch.blueprint, sil_advisory=self._sil_advisory, target=self._cost_advisor.target)
                    return extract_cost_pred_us(self._cost_advisor.predict_candidate_us(ft))

                scorer = _score

            def _cand_l0(mod_src: str) -> StageAResult:
                ksig = extract_knob_signature(mod_src, batch.regime_spec)
                if (ir := self._boundary_ledger.precheck_known_infeasible(regime_tag=batch.regime_spec.regime_tag, cand_sha=compute_source_sha256(mod_src), knob_signature=ksig)) is not None:
                    return StageAResult(passed=False, prim_func=None, ledger=None, error_message=ir, backend_kind=batch.regime_spec.runtime_backend)
                if (de := _detect_trackb_dead_code_edit(batch.parent_module_source, mod_src, batch.regime_spec)) is not None:
                    return StageAResult(passed=False, prim_func=None, ledger=None, error_message=de, backend_kind="ascendc")
                return self._run_l0(mod_src, batch.regime_spec, anchor_dims_override=batch.bn_dims or None)

            ak = resolve_active_trackb_kernel(batch.parent_module_source, batch.regime_spec, cache=self._active_kernel_cache) or ""
            flines, dtext = _extract_p2_focus_lines_and_diag(batch.parent_p2_diag)
            repsrc = _compact_trackb_snippet_source(batch.parent_snippet, batch.regime_spec, incumbent_source=batch.parent_module_source, active_kernel=ak, focus_lines=flines, diagnostic_text=dtext)
            off = 0
            if batch.regime_spec.runtime_backend in ("ascendc_trackb", "cce_trackb", "hybrid_trackb"):
                try:
                    rprobe = _render_trackb_tu_from_module(batch.parent_module_source, "__TF_TRACKB_SENTINEL__", batch.regime_spec, require_renderer=False)
                    if "__TF_TRACKB_SENTINEL__" in rprobe:
                        off = rprobe.split("__TF_TRACKB_SENTINEL__", 1)[0].count("\n")
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    pass
            op_entry = resolve_operator_entrypoint_safe(self._config.operator_name)
            _pa = lambda bsrc, snip: patch_regime_into_module(bsrc, batch.regime_spec, snip, operator_entrypoint=op_entry, sibling_specs=self._regime_specs, active_kernel=ak)
            trep = _TimedRepair(eff_backend) if (eff_backend is not None and self._config.max_syntax_repairs > 0) else None
            ub_unlock = bool(batch.parent_ledger.uses_unreserved_ubuf) if batch.parent_ledger else False
            shint = slice_contract_instruction(batch.func_name, batch.regime_spec, operator_entrypoint=op_entry)

            async def _run_filter(
                cands: Sequence[tuple[str, str]], *, l1_fn: Callable[[str], StageBResult] | None = None,
                cr_ids: Collection[str] | None = None, keep_repair: bool = True, keep_ranker: bool = True,
            ) -> L0L1RoundResults:
                return await run_l0_l1_round_filter(
                    cands, patch_fn=lambda s: _pa(batch.parent_module_source, s),
                    repatch_from_current_fn=lambda cs, s: _pa(cs, s), l0_checker=_cand_l0,
                    l1_compiler=l1_fn or (lambda ms: self._run_l1(ms, batch.regime_spec)),
                    repair_backend=trep if keep_repair else None,
                    max_syntax_repairs=self._config.max_syntax_repairs if keep_repair else 0,
                    slice_hint=shint, backend_kind=batch.regime_spec.runtime_backend, parent_source=repsrc,
                    parent_module_source=batch.parent_module_source, trackb_prologue_line_offset=off,
                    stage_a_cost_ranker=ranker if keep_ranker else None, cost_prescreen_top_k=pk,
                    sil_advisory=self._sil_advisory, freedom_mask=eff_mask, ub_256kb_unlock=ub_unlock,
                    creative_candidate_ids=cr_ids if cr_ids is not None else set(batch.creative_candidate_ids),
                    parent_analytical_us=paus, stage_a_analytical_scorer=scorer, parent_topology_axes=eff_paxes,
                    candidate_topology_axes_fn=lambda _ms, sa: self._derive_topology_axes(sa.ledger, features=sa.features, route=sa.route) if sa.ledger else None,
                    skip_repair_when_survivors_exist=True,
                )

            return ranker, ak, trep, _run_filter

        def _spawn_cr() -> list[tuple[int, str, AscendCausalRule | None, asyncio.Task[SpeculativeCreativeResult]]]:
            if any(not t.done() for _, _, _, t in self._pending_speculative) and cr_specs:
                logger.info("Round %d skipping %d creative worker(s): pending speculative", batch.round_idx, len(cr_specs))
                return []
            return [(batch.round_idx, wid, r, asyncio.create_task(_call_creative(wid, req), name=f"spec_cr_r{batch.round_idx}_{wid}")) for wid, r, req in cr_specs]

        def _rec_ex(replies: Sequence[tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]]) -> None:
            for wid, wk, r, sn, ch, err, tr in replies:
                meta_by_id[wid] = (wk, r, ch)
                if sn is None:
                    self._record_llm_failure(wid=wid, rule=r, err=err, is_transport_error=tr, regime_spec=batch.regime_spec, parent_sha=eff_psha, parent_axes=eff_paxes, round_idx=batch.round_idx, failed_entries=failed_entries)
                else:
                    raw_candidates.append((wid, sn))

        def _harv_cr(tasks: Sequence[tuple[int, str, AscendCausalRule | None, asyncio.Task[SpeculativeCreativeResult]]]) -> list[tuple[str, str]]:
            abs_out = []
            for or_idx, wid, r, t in tasks:
                if t.done():
                    prev = len(raw_candidates)
                    self._absorb_creative_task(t, origin_round_idx=or_idx, wid=wid, rule_obj=r, batch=batch, parent_sha=eff_psha, parent_axes=eff_paxes, raw_candidates=raw_candidates, meta_by_id=meta_by_id, failed_entries=failed_entries)
                    if len(raw_candidates) > prev:
                        abs_out.append(raw_candidates[-1])
                else:
                    self._pending_speculative.append((or_idx, wid, r, t))
            return abs_out

        l0_l1_checks: list[L0L1CandidateCheck] = []
        round_ak, l0_l1_sec, repair_sec, llm_sec = "", 0.0, 0.0, 0.0
        exp_non_cr = sum(1 for cid, _ in raw_candidates if meta_by_id.get(cid, ("", None, ""))[0] != "creative_explore") + len(ex_specs)
        can_stream = run_l0_l1_round_filter is _ORIG_RUN_L0_L1_ROUND_FILTER and bool(ex_specs)
        pre_ctx = _make_filter_runner(exp_non_cr) if can_stream else None
        if pre_ctx is not None and pre_ctx[0] is None:
            _, ak_val, trep, run_filter = pre_ctx
            sl1_sem = threading.Semaphore(2)

            async def _stream_one(wid: str, snip: str, *, is_cr: bool) -> L0L1CandidateCheck:
                res = await run_filter(
                    [(wid, snip)], l1_fn=lambda ms: _with_sem(sl1_sem, lambda: self._run_l1(ms, batch.regime_spec)),
                    cr_ids=({wid} if (is_cr or wid in batch.creative_candidate_ids) else ()),
                    keep_repair=False, keep_ranker=False,
                )
                return res[0]

            stasks = {wid: asyncio.create_task(_stream_one(wid, sn, is_cr=True)) for wid, sn in raw_candidates}
            lt0 = time.monotonic()
            ctasks = _spawn_cr()
            etasks = [asyncio.create_task(_call_exploit(wid, wk, r, req)) for wid, wk, r, req in ex_specs]
            erep_map: dict[str, tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]] = {}
            for fut in asyncio.as_completed(etasks):
                rep = await fut
                erep_map[rep[0]] = rep
                if rep[3] is not None and rep[0] not in stasks:
                    stasks[rep[0]] = asyncio.create_task(_stream_one(rep[0], rep[3], is_cr=False))
            ldone = time.monotonic()
            llm_sec = max(0.0, ldone - lt0)
            _rec_ex([erep_map[wid] for wid, *_ in ex_specs])
            for wid, sn in _harv_cr(ctasks):
                if wid not in stasks:
                    stasks[wid] = asyncio.create_task(_stream_one(wid, sn, is_cr=True))
            if raw_candidates:
                round_ak = ak_val
                l0_l1_checks = list(await asyncio.gather(*(stasks[wid] for wid, _ in raw_candidates)))
                if not any(c.passed for c in l0_l1_checks) and self._config.max_syntax_repairs > 0 and trep is not None:
                    repairable = [(w, s) for (w, s), c in zip(raw_candidates, l0_l1_checks) if c.failed_tier != "noop_identical"]
                    if repairable:
                        cr_ids = {c for c, _ in raw_candidates if meta_by_id.get(c, ("", None, ""))[0] == "creative_explore"} | set(batch.creative_candidate_ids)
                        rep_res = await run_filter(repairable, cr_ids=cr_ids)
                        rby_id = {c.candidate_id: c for c in rep_res}
                        l0_l1_checks = [rby_id.get(c.candidate_id, c) for c in l0_l1_checks]
                repair_sec = trep.repair_sec if trep else 0.0
                l0_l1_sec = max(0.0, max(0.0, time.monotonic() - ldone) - repair_sec)
        else:
            lt0 = time.monotonic()
            ctasks = _spawn_cr()
            ereplies = list(await asyncio.gather(*(_call_exploit(wid, wk, r, req) for wid, wk, r, req in ex_specs))) if ex_specs else []
            if not ex_specs and ctasks:
                await asyncio.gather(*(t for _, _, _, t in ctasks))
            llm_sec = max(0.0, time.monotonic() - lt0)
            _rec_ex(ereplies)
            _harv_cr(ctasks)
            if raw_candidates:
                non_cr = sum(1 for c, _ in raw_candidates if meta_by_id.get(c, ("", None, ""))[0] != "creative_explore")
                _, round_ak, trep, run_filter = _make_filter_runner(non_cr)
                cr_ids = {c for c, _ in raw_candidates if meta_by_id.get(c, ("", None, ""))[0] == "creative_explore"} | set(batch.creative_candidate_ids)
                ft0 = time.monotonic()
                l0_l1_checks = list(await run_filter(raw_candidates, cr_ids=cr_ids))
                repair_sec = trep.repair_sec if trep else 0.0
                l0_l1_sec = max(0.0, max(0.0, time.monotonic() - ft0) - repair_sec)

        passed_checks = [c for c in l0_l1_checks if c.passed]
        inf_added, pruned_list, ksig_cid, ub_cid, pred_cid, axes_cid, faults_cid = 0, [], {}, {}, {}, {}, {}
        for chk in l0_l1_checks:
            self._persist_round_candidate_artifact(chk, round_idx=batch.round_idx)
            ksig_cid[chk.candidate_id] = extract_knob_signature(chk.patched_module_source, batch.regime_spec) if chk.patched_module_source else ""
            ub_wl = (getattr(chk.stage_a.ledger, "waterlines", {}) or {}).get("UB") if (chk.stage_a and chk.stage_a.ledger) else None
            ub_cid[chk.candidate_id] = int(getattr(ub_wl, "peak_allocated_bytes", 0) or 0) if ub_wl else 0
            if chk.stage_a is not None and (chk.stage_a.passed or chk.stage_a.ledger is not None):
                ca = self._derive_topology_axes(chk.stage_a.ledger, features=chk.stage_a.features, route=chk.stage_a.route)
                axes_cid[chk.candidate_id] = ca
                try:
                    cp = extract_cost_pred_us(self._cost_advisor.predict_candidate_us(build_cost_features_from_ledger(chk.stage_a.ledger, live_plan=batch.live_plan, axes=ca, blueprint=batch.blueprint, sil_advisory=self._sil_advisory, target=self._cost_advisor.target)))
                    if cp > 0.0:
                        pred_cid[chk.candidate_id] = cp
                except (TypeError, ValueError):
                    pass
            if chk.passed and chk.patched_module_source and chk.stage_b and chk.stage_b.passed and chk.stage_b.compiler_summary:
                self._l1_summary_cache[(source_sha256(chk.patched_module_source), batch.regime_spec.regime_tag)] = chk.stage_b.compiler_summary
                self._l1_summary_cache[self._l1_summary_reuse_key(chk.patched_module_source, batch.regime_spec)] = chk.stage_b.compiler_summary
            if not chk.passed:
                failed_entries.append((chk.candidate_id, chk.failed_tier or "l0_l1", chk.error_message[:1500]))
                if chk.failed_tier == "cost_prescreen":
                    pruned_list.append(chk.candidate_id)
                csha = compute_source_sha256(chk.patched_module_source) if chk.patched_module_source else ""
                rfp = getattr(chk, "fault_plane", "") or (getattr(chk.stage_b, "fault_plane", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_plane", "") if chk.stage_a else "") or ("LLM_TRANSPORT" if chk.failed_tier == "llm_transport" else ("COMPILER_LOWERING" if chk.failed_tier == "stage_b" else "STATIC_CONTRACT"))
                rfc = getattr(chk, "fault_code", "") or (getattr(chk.stage_b, "fault_code", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_code", "") if chk.stage_a else "") or f"{(chk.failed_tier or 'L0_L1').upper()}_FAILED"
                cfault = TypedFault(plane=coerce_fault_plane(rfp), code=str(rfc), failed_tier=str(chk.failed_tier or "l0_l1"), detail=str(chk.error_message or "")[:500])
                faults_cid[chk.candidate_id] = cfault
                sb = int(chk.stage_b.compiler_summary.stack_bytes) if (chk.stage_b and chk.stage_b.compiler_summary) else 0
                if self._boundary_ledger.record_compile_wall(regime_tag=batch.regime_spec.regime_tag, cand_sha=csha, knob_signature=ksig_cid[chk.candidate_id], error_code_norm=cfault.code, ub_bytes=ub_cid[chk.candidate_id], stack_bytes=sb):
                    inf_added += 1
                _, rule, _ = meta_by_id.get(chk.candidate_id, ("unknown", None, ""))
                if rule is not None and cfault.plane.penalizes_bandit() and chk.failed_tier != "cost_prescreen" and "[repair_timeout]" not in chk.error_message:
                    self._tracker.record_outcome(rule.rule_id, postcondition_realized=False, topology_axes=eff_paxes)
                    self._tracker.record_scoped_failure(regime_tag=batch.regime_spec.regime_tag, rule_id=rule.rule_id, parent_sha=eff_psha, current_round=batch.round_idx, reason=f"l0_l1_failed:{chk.failed_tier or 'l0_l1'}")
                    self._record_kb_event(rule_id=rule.rule_id, postcondition_realized=False, topology_axes=eff_paxes, round_idx=batch.round_idx)

        l2_results: dict[str, CandidateSurrogateResult] = {}
        l2_run_ids: dict[str, str] = {}
        l2_infra_fault: TypedFault | None = None
        l2_pw_sec, l2_npu_sec = 0.0, 0.0
        pw_stats = {"total_tasks": 0, "cache_hits": 0, "compiled_tasks": 0, "failed_candidates": 0}
        if passed_checks:
            with tempfile.TemporaryDirectory(prefix=f"tf_camp_r{batch.round_idx}_") as tdir:
                cpaths = []
                for chk in passed_checks:
                    cf = Path(tdir) / f"{chk.candidate_id}.py"
                    cf.write_text(chk.patched_module_source, encoding="utf-8")
                    cpaths.append((chk.candidate_id, cf))
                lt0 = time.monotonic()
                self._last_l2_timing_meta = {}
                failed_l2 = False
                try:
                    l2_results, l2_run_ids = self._run_l2_batch(cpaths, working_anchors, include_guardrails=True, incumbent_source=eff_inc, incumbent_anchor_us_by_case=inc_anchor_map)
                except Exception as exc:
                    failed_l2 = True
                    l2_infra_fault = self._classify_l2_exception(exc)
                    if l2_infra_fault is None and not isinstance(exc, (AgentGatesDeviceFaultError, OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError)):
                        raise
                    logger.warning("Round %d L2 batch failed after retries: %s", batch.round_idx, exc)
                    l2_results, l2_run_ids = {}, {}
                    ftier = l2_infra_fault.failed_tier if l2_infra_fault else "l2_error"
                    for chk in passed_checks:
                        failed_entries.append((chk.candidate_id, ftier, str(exc)[:1500]))
                l2_wsec = max(0.0, time.monotonic() - lt0)
                if not failed_l2:
                    if isinstance(rpw := self._last_l2_timing_meta.get("prewarm_stats"), Mapping):
                        for pk_k in pw_stats:
                            if pk_k in rpw:
                                pw_stats[pk_k] = int(rpw[pk_k])
                    l2_pw_sec = float(self._last_l2_timing_meta.get("l2_prewarm_sec") or 0.0)
                    rnpu = self._last_l2_timing_meta.get("l2_npu_sec")
                    l2_npu_sec = float(rnpu) if isinstance(rnpu, (int, float)) and not isinstance(rnpu, bool) else max(0.0, l2_wsec - l2_pw_sec)
        return StageGateOutcome(
            raw_candidates=raw_candidates, meta_by_id=meta_by_id, failed_entries=failed_entries,
            l0_l1_checks=l0_l1_checks, passed_checks=passed_checks, l0_l1_faults_by_cid=faults_cid,
            l2_results=l2_results, l2_run_ids=l2_run_ids, l2_infra_fault=l2_infra_fault,
            llm_gen_sec=llm_sec, l0_l1_sec=l0_l1_sec, syntax_repair_sec=repair_sec,
            l2_prewarm_sec=l2_pw_sec, l2_npu_sec=l2_npu_sec, round_prewarm_stats=pw_stats,
            round_active_kernel=round_ak, infeasible_boundaries_added=inf_added,
            prescreen_pruned_list=pruned_list, knob_sig_by_cid=ksig_cid, ub_bytes_by_cid=ub_cid,
            pred_by_cid=pred_cid, cand_axes_by_cid=axes_cid,
        )


def _with_sem(sem: threading.Semaphore, fn: Callable[[], StageBResult]) -> StageBResult:
    with sem:
        return fn()


__all__ = [
    "CandidateBatch", "OccupancyExhaustedError", "SpeculativeCreativeResult", "StageGateConfig",
    "StageGateCoordinator", "StageGateOutcome", "resolve_active_trackb_kernel",
    "resolve_active_trackb_kernels", "run_l0_l1_round_filter",
]
