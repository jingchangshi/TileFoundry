"""Standalone zero-config score-sprint marathon runner and event watcher."""

from __future__ import annotations

import argparse
import ast
import asyncio
import ctypes
import hashlib
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    default_source_root,
    discover_module_trackb_symbols,
    extract_trackb_cpp_source,
    get_default_anchors,
    get_default_regime_specs,
    get_operator_module_relpath,
    get_operator_task,
    parse_simple_env_file,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY,
    CampaignConfig,
    PerfTargetCampaignRunner,
    _render_trackb_tu_from_module,
    parse_device_ids,
    patch_regime_into_module,
    resolve_campaign_device_plan,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    GoldenEntrypointContract,
    OracleAccuracyResult,
    SeedSynthesisOutcome,
    extract_golden_entrypoint_contract,
    prepare_operator_seed_or_takeover,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    _LLM_API_KEY_ENV,
    _NON_AGENT_GATES_ENV_KEYS,
    DEFAULT_L4_CANNBENCH_WAIT_SEC,
    CandidateSurrogateResult,
    L2BatchResults,
    L3AcceptanceResult,
    _format_env_line,
    _validate_llm_api_key_env,
    collect_bottleneck_msprof_via_agent_gates,
    run_l1_compile_check,
    run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates,
    run_micro_probe_via_agent_gates,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    check_npu_entrypoint_purity,
    evaluate_ascendc_candidate,
    resolve_trackb_compile_flags,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    KernelLLMBackend,
    create_llm_backend_from_env,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    compute_source_sha256,
    resolve_default_experience_lake_dir,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

logger = logging.getLogger("tilefoundry_ascend.marathon")

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_WATCH_MASK = _IN_CLOSE_WRITE | _IN_MOVED_TO
_LLM_ENV_PREFIX = "TILEFOUNDRY_NPU_LLM_"
_LLM_CLI_OPTIONS = {
    "llm_backend": ("BACKEND", "Generation backend."),
    "exploit_model": ("EXPLOIT_MODEL", "Model for rule-guided exploitation."),
    "explore_model": ("EXPLORE_MODEL", "Model for creative exploration."),
    "escalate_model": ("ESCALATE_MODEL", "Model for escalated exploration."),
    "llm_api_base": ("API_BASE", "Base URL for the HTTP generation backend."),
}
_LEGACY_AGY_ENV_KEYS = frozenset(
    {
        "TILEFOUNDRY_NPU_AGY_BIN",
        "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
        "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
        "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL",
        "TILEFOUNDRY_NPU_AGY_COMMON_ARGS",
        "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
    }
)
_SUPPORTED_OPERATORS: tuple[str, ...] = ()
_AGENT_GATES_ALLOWED_ENV_KEYS = frozenset(
    {
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS",
        "TILEFOUNDRY_NPU_CANN_SETUP",
        "TILEFOUNDRY_NPU_CONDA_SH",
        "TILEFOUNDRY_NPU_CONDA_ENV",
        "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
        "TILEFOUNDRY_NPU_PYTHON",
        "TILEFOUNDRY_NPU_TILELANG_SOURCE",
        "TILEFOUNDRY_NPU_TILELANG_BUILD",
        "TILEFOUNDRY_NPU_CPU_TESTS",
        "TILEFOUNDRY_NPU_DEVICE_TESTS",
        "TILEFOUNDRY_NPU_BUILD_SUBDIR",
        "TILEFOUNDRY_NPU_CANNBENCH_REPO",
        "TILEFOUNDRY_NPU_WORK_ROOT",
        "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
        "TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE",
        "TILEFOUNDRY_NPU_CANNBENCH_TARGETS",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
        "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION",
        "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID",
        "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN",
        "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT",
        "TILEFOUNDRY_NPU_TILELANG_REPO",
        "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
        "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
        "TILEFOUNDRY_NPU_ASC_DEVKIT_REPO",
    }
)
_ADAPTER_TOOLCHAIN_ENV_KEYS: tuple[str, ...] = (
    "TILEFOUNDRY_NPU_CANN_SETUP",
    "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
)
_ADAPTER_PATH_ENV_KEYS: frozenset[str] = frozenset(
    {"TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH"}
)


def _is_llm_env_key(key: str) -> bool:
    return key.startswith(_LLM_ENV_PREFIX) or key in _LEGACY_AGY_ENV_KEYS


def _is_agent_gates_allowed_env_key(key: str) -> bool:
    """Return whether the environment key belongs in an AgentGates file."""
    return key not in _NON_AGENT_GATES_ENV_KEYS and (
        _is_llm_env_key(key)
        or key in _AGENT_GATES_ALLOWED_ENV_KEYS
        or key.startswith("TILEFOUNDRY_NPU_CANNBENCH_")
    )


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    for name, (_, help_text) in _LLM_CLI_OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", help=help_text)


def _llm_backend_overrides(
    env_values: Mapping[str, str], options: argparse.Namespace | MarathonConfig
) -> dict[str, str]:
    overrides = {
        key: value
        for key, value in env_values.items()
        if _is_llm_env_key(key) and key not in _NON_AGENT_GATES_ENV_KEYS
    }
    overrides.update(
        {
            f"{_LLM_ENV_PREFIX}{key}": value
            for name, (key, _) in _LLM_CLI_OPTIONS.items()
            if (value := getattr(options, name)) is not None
        }
    )
    return overrides


@dataclass(frozen=True)
class MarathonConfig:
    """Configuration for a zero-config score-sprint marathon campaign."""

    operator_name: str
    device_id: str = ""
    retry_device_id: str = ""
    device_ids: tuple[str, ...] = ()
    source_root: Path | None = None
    worktree: Path | None = None
    isolate_worktree: bool = False
    state_dir: Path | None = None
    target_module_path: Path | None = None
    env_file: Path | None = None
    checkpoint_path: Path | None = None
    report_path: Path | None = None
    campaign_kb_path: Path | None = None
    experience_lake_dir: Path | None = None
    log_path: Path | None = None
    max_rounds: int = 80
    round_regime_tags: tuple[str, ...] = ()
    max_wall_clock_min: float = 720.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_l2_gain_for_l3: float = 0.005
    min_surrogate_gain: float = 0.08
    min_flywheel_gain: float = 0.03
    stagnation_escalate_rounds: int = 1
    stagnation_freeze_rounds: int = 12
    stagnation_discount: float = 0.45
    target_avg_speedup: float = 2.985
    target_min_case_speedup: float = 2.95
    target_roofline_eta: float = 0.995
    target_surrogate_score: float = 99.8
    min_optimization_rounds: int = 15
    disable_early_goal_stop: bool = False
    run_final_l4: bool = True
    l4_regression_ratio: float = 1.0
    warmup: int = 3
    repeat: int = 5
    enable_tilesize_prescreen: bool = True
    cost_prescreen_top_k: int | None = None
    prewarm_l3: bool = True
    salvage_beam_candidates: bool = True
    resume: bool | None = None
    escalate_model: str | None = None
    exploit_effort: str = "high"
    auto_bootstrap_seed: bool = True
    cold_start: bool = False
    pure_golden: bool = False
    llm_backend: str | None = None
    exploit_model: str | None = None
    explore_model: str | None = None
    llm_api_base: str | None = None


def short_operator_tag(operator_name: str) -> str:
    """Return short filesystem tag for an operator name or task id."""
    norm = operator_name.rsplit("/", 1)[-1] if "/" in operator_name else operator_name
    return norm.replace("_", "-")


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via mkstemp, fsync, and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _resolve_cann_root_from_env(env_file: Path | None) -> Path | None:
    """Return the CANN install root derived from the adapter env file setup script."""
    if env_file is None or not env_file.is_file():
        return None
    setup_raw = parse_simple_env_file(env_file).get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
    if not setup_raw:
        return None
    root = Path(setup_raw).expanduser().parent
    return root if root.is_dir() else None


def _module_binds_kernel_alias(tree: ast.Module, entrypoint: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if (
            any(isinstance(target, ast.Name) and target.id == "kernel" for target in targets)
            and isinstance(value, ast.Name)
            and value.id == entrypoint
        ):
            return True
    return False


def _stage0_entrypoint_contract_error(
    module_source: str,
    entrypoint: str,
    track: str,
    *,
    contract: GoldenEntrypointContract | None = None,
) -> str:
    """Return a rejection reason when the public entrypoint or its dispatch chain is absent."""
    if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
        return f"unknown Stage-0 track {track!r}"
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == entrypoint for node in tree.body
    ):
        return f"seed module does not define the public entrypoint {entrypoint}"
    if not _module_binds_kernel_alias(tree, entrypoint):
        return f"seed module does not bind `kernel = {entrypoint}`"
    if contract is not None:
        sig_error = validate_entrypoint_signature_compatibility(module_source, contract)
        if sig_error:
            return sig_error
    purity_errors = check_npu_entrypoint_purity(
        module_source,
        entrypoint,
        track=track,
        require_kernel_cache=True,
    )
    if purity_errors:
        return "entrypoint purity rejection: " + "; ".join(purity_errors)
    if track in ("cce_trackb", "hybrid_trackb"):
        offending = {
            token: count
            for token, count in CCE_CPP20_SURFACE.count_forbidden_tokens(module_source).items()
            if count
        }
        if offending:
            rendered = ", ".join(f"{token}={count}" for token, count in offending.items())
            return f"CCE C++20 seed references forbidden tokens: {rendered}"
    return ""


def _has_prim_func_builder(
    module_source: str,
    builder_candidates: Sequence[str],
) -> bool:
    if not builder_candidates:
        return False
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return False
    candidate_set = set(builder_candidates)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in candidate_set:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) or sub is node:
                continue
            for dec in sub.decorator_list:
                dec_src = ast.get_source_segment(module_source, dec) or ""
                if "prim_func" in dec_src:
                    return True
    return False


def _instantiate_stage0_trackb_tu(
    module_source: str,
    cpp_source: str,
    regime_spec: RegimeTargetSpec,
) -> str:
    """Render a seed's Track-B launcher for *regime_spec* before Stage-0 gating."""
    return _render_trackb_tu_from_module(
        module_source,
        cpp_source,
        regime_spec,
        require_renderer=True,
    )


def _aggregate_hardware_engine_error(
    ledgers: Sequence[ObjectivePhysicalLedger | None],
    semantics: PyTorchOpSemantics | None,
) -> str:
    """Enforce aggregate Cube and DMA hardware-engine invariants across module ledgers."""
    if semantics is None:
        return ""
    active_ledgers = [ledger for ledger in ledgers if ledger is not None]
    if not active_ledgers:
        return ""
    if semantics.has_matmul:
        has_cube = any(int(lg.pipe_cycles.get("AIC.PIPE_M", 0)) > 0 for lg in active_ledgers)
        if not has_cube:
            return (
                "aggregate L0 hardware-engine rejection: missing_cube_engine_primitives "
                "(module AIC.PIPE_M == 0 across Track-A and Track-B ledgers)"
            )
    if not semantics.has_indirect_index:
        has_dma = any(
            lg.has_indirect_gather
            or any(k.endswith("PIPE_MTE2") and int(v) > 0 for k, v in lg.pipe_cycles.items())
            for lg in active_ledgers
        )
        if not has_dma:
            return (
                "aggregate L0 hardware-engine rejection: missing_hardware_dma_staging "
                "(module PIPE_MTE2 == 0 across Track-A and Track-B ledgers)"
            )
    return ""


def _run_stage0_track_a_gates(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    entrypoint: str,
    worktree: Path,
    env_file: Path | None,
    contract: GoldenEntrypointContract | None = None,
    semantics: PyTorchOpSemantics | None = None,
    enforce_hardware_engines: bool = True,
) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
    contract_error = _stage0_entrypoint_contract_error(
        module_source,
        entrypoint,
        "tilelang_dsl",
        contract=contract,
    )
    if contract_error:
        return False, contract_error, None
    dims = dict(regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims)
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
    except (
        SyntaxError,
        ImportError,
        KeyError,
        NameError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return False, f"seed config resolution failed: {type(exc).__name__}: {exc}", None
    req_dma = bool(
        enforce_hardware_engines and semantics is not None and not semantics.has_indirect_index
    )
    req_cube = bool(enforce_hardware_engines and semantics is not None and semantics.has_matmul)
    static = run_stage_a_static(
        module_source,
        factory_name=resolved.builder_name,
        factory_kwargs=resolved.builder_kwargs,
        pass_configs=resolved.pass_configs or None,
        compile_flags=resolved.compile_flags or None,
        case_ratios=regime_spec.representative_case_ratios,
        require_hardware_dma=req_dma,
        require_cube_engine=req_cube,
    )
    if not static.passed:
        return False, f"L0 static/waterline rejection: {static.error_message}", static.ledger
    compiled = run_l1_compile_check(
        module_source,
        builder_name=resolved.builder_name,
        builder_kwargs=resolved.builder_kwargs,
        config_selector_name=resolved.selector_name or None,
        pass_configs=resolved.pass_configs or None,
        compile_flags=resolved.compile_flags or None,
        source_root=worktree,
        env_file=env_file,
    )
    if not compiled.passed:
        return False, f"L1 device compilation rejection: {compiled.error_message}", static.ledger
    return True, "", static.ledger


def _validate_stage0_track_a(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    entrypoint: str,
    worktree: Path,
    env_file: Path | None,
    contract: GoldenEntrypointContract | None = None,
    semantics: PyTorchOpSemantics | None = None,
) -> tuple[bool, str]:
    """Run L0 static purity/waterline and L1 device compilation for a Track-A seed."""
    passed, err, _ = _run_stage0_track_a_gates(
        module_source,
        regime_spec,
        entrypoint=entrypoint,
        worktree=worktree,
        env_file=env_file,
        contract=contract,
        semantics=semantics,
        enforce_hardware_engines=True,
    )
    return passed, err


def _resolve_stage0_trackb_selector_flags(
    module_source: str,
    regime_spec: RegimeTargetSpec,
) -> tuple[str, ...]:
    if not regime_spec.config_selector_fn:
        return ()
    dims = dict(regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims)
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
    except (
        SyntaxError,
        ImportError,
        KeyError,
        NameError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return ()
    return tuple(resolved.compile_flags)


def _run_stage0_track_b_gates(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    entrypoint: str,
    env_file: Path | None,
    contract: GoldenEntrypointContract | None = None,
    track: str = "cce_trackb",
    semantics: PyTorchOpSemantics | None = None,
    enforce_hardware_engines: bool = True,
) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
    contract_error = _stage0_entrypoint_contract_error(
        module_source,
        entrypoint,
        track,
        contract=contract,
    )
    if contract_error:
        return False, contract_error, None
    trackb_symbols = regime_spec.trackb_symbols or discover_module_trackb_symbols(module_source)
    extracted = extract_trackb_cpp_source(module_source, trackb_symbols)
    if extracted is None:
        return False, "seed exposes no Track-B C++20 translation-unit string constant", None
    symbol, cpp_source = extracted
    try:
        instantiated_tu = _instantiate_stage0_trackb_tu(module_source, cpp_source, regime_spec)
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
    ) as exc:
        return False, f"Track-B _render_trackb_tu instantiation failed ({symbol}): {exc}", None
    req_dma = bool(
        enforce_hardware_engines and semantics is not None and not semantics.has_indirect_index
    )
    req_cube = bool(enforce_hardware_engines and semantics is not None and semantics.has_matmul)
    verdict = evaluate_ascendc_candidate(
        instantiated_tu,
        freedom_mask=ConstraintRelaxationMask.top(),
        case_ratios=regime_spec.representative_case_ratios,
        require_hardware_dma=req_dma,
        require_cube_engine=req_cube,
    )
    if not verdict.passed:
        return False, f"Track-B L0 token purity ({symbol}): {verdict.error_message}", verdict.ledger
    cann_root = _resolve_cann_root_from_env(env_file)
    selector_flags = _resolve_stage0_trackb_selector_flags(module_source, regime_spec)
    flags = list(resolve_trackb_compile_flags(instantiated_tu, selector_flags))
    bisheng_bin: str | None = None
    if cann_root is not None:
        candidate_bin = cann_root / "bin" / "bisheng"
        if candidate_bin.is_file():
            bisheng_bin = str(candidate_bin)
        if (cann_root / "include").is_dir():
            flags.append(f"-I{cann_root / 'include'}")
    compiled = run_stage_b_ascendc_compile(
        instantiated_tu,
        compile_flags=flags,
        bisheng_bin=bisheng_bin,
        cann_root=cann_root,
        verify_shared_link=True,
    )
    if not compiled.passed:
        return (
            False,
            f"Track-B L1 bisheng compilation ({symbol}): {compiled.error_message}",
            verdict.ledger,
        )
    return True, "", verdict.ledger


def _validate_stage0_track_b(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    entrypoint: str,
    env_file: Path | None,
    contract: GoldenEntrypointContract | None = None,
    track: str = "cce_trackb",
    semantics: PyTorchOpSemantics | None = None,
) -> tuple[bool, str]:
    """Run L0 token purity and L1 bisheng compilation for a Track-B CCE seed."""
    passed, err, _ = _run_stage0_track_b_gates(
        module_source,
        regime_spec,
        entrypoint=entrypoint,
        env_file=env_file,
        contract=contract,
        track=track,
        semantics=semantics,
        enforce_hardware_engines=True,
    )
    return passed, err


def _validate_stage0_hybrid_track_b(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    entrypoint: str,
    worktree: Path,
    env_file: Path | None,
    contract: GoldenEntrypointContract | None = None,
    semantics: PyTorchOpSemantics | None = None,
) -> tuple[bool, str]:
    """Validate hybrid Track-B TU and optional TileLang @T.prim_func builder."""
    has_prim = _has_prim_func_builder(module_source, regime_spec.builder_candidates)
    passed_tb, error_tb, ledger_tb = _run_stage0_track_b_gates(
        module_source,
        regime_spec,
        entrypoint=entrypoint,
        env_file=env_file,
        contract=contract,
        track="hybrid_trackb",
        semantics=semantics,
        enforce_hardware_engines=not has_prim,
    )
    if not passed_tb:
        return False, error_tb
    if not has_prim:
        return True, ""
    passed_ta, error_ta, ledger_ta = _run_stage0_track_a_gates(
        module_source,
        regime_spec,
        entrypoint=entrypoint,
        worktree=worktree,
        env_file=env_file,
        contract=contract,
        semantics=semantics,
        enforce_hardware_engines=False,
    )
    if not passed_ta:
        return False, error_ta
    agg_err = _aggregate_hardware_engine_error((ledger_tb, ledger_ta), semantics)
    if agg_err:
        return False, agg_err
    return True, ""


def build_stage0_static_validator(
    regime_specs: Sequence[RegimeTargetSpec],
    *,
    entrypoint: str,
    worktree: Path,
    env_file: Path | None,
    entrypoint_contract: GoldenEntrypointContract | None = None,
    semantics: PyTorchOpSemantics | None = None,
) -> Callable[[str, str], tuple[bool, str]]:
    """Build the L0/L1 gate pair the bounded Stage-0 seed loop invokes per attempt."""
    assert regime_specs, "Stage-0 seed synthesis requires at least one regime specification"

    def _validate(module_source: str, track: str) -> tuple[bool, str]:
        if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
            err = _stage0_entrypoint_contract_error(
                module_source,
                entrypoint,
                track,
                contract=entrypoint_contract,
            )
            return False, f"[{regime_specs[0].regime_tag}] {err}"
        failures: list[str] = []
        for regime_spec in regime_specs:
            if track == "cce_trackb":
                passed, error = _validate_stage0_track_b(
                    module_source,
                    regime_spec,
                    entrypoint=entrypoint,
                    env_file=env_file,
                    contract=entrypoint_contract,
                    semantics=semantics,
                )
            elif track == "hybrid_trackb":
                passed, error = _validate_stage0_hybrid_track_b(
                    module_source,
                    regime_spec,
                    entrypoint=entrypoint,
                    worktree=worktree,
                    env_file=env_file,
                    contract=entrypoint_contract,
                    semantics=semantics,
                )
            else:
                passed, error = _validate_stage0_track_a(
                    module_source,
                    regime_spec,
                    entrypoint=entrypoint,
                    worktree=worktree,
                    env_file=env_file,
                    contract=entrypoint_contract,
                    semantics=semantics,
                )
            if not passed:
                failures.append(f"[{regime_spec.regime_tag}] {error}")
        if failures:
            return False, "; ".join(failures)
        return True, ""

    return _validate


def build_stage0_oracle_validator(
    *,
    operator_name: str,
    staging_dir: Path,
    worktree: Path,
    device_id: str,
    op_logger: logging.Logger,
    blocked_env_max_retries: int = 12,
    blocked_env_backoff_sec: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0),
) -> Callable[[str], OracleAccuracyResult]:
    """Build the NPU accuracy validator that stages seed bytes by digest before L3 verification."""

    def _validate(module_source: str) -> OracleAccuracyResult:
        digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
        staged_module = staging_dir / f"{digest}.py"
        atomic_write_text(staged_module, module_source)
        max_retries = max(0, int(blocked_env_max_retries))
        result: L3AcceptanceResult | None = None
        is_transient_blocked_env = False
        for retry_idx in range(max_retries + 1):
            result = run_l3_full_contract_via_agent_gates(
                operator_name=operator_name,
                candidate_module_path=staged_module,
                source_root=worktree,
                device_id=str(device_id),
            )
            if result.passed:
                is_transient_blocked_env = False
                break
            err_low = (result.error_message or "").lower()
            is_transient_blocked_env = result.failure_kind != "candidate" and (
                "blocked_env" in err_low or "resource busy" in err_low
            )
            if not is_transient_blocked_env or retry_idx >= max_retries:
                break
            wait_sec = (
                float(blocked_env_backoff_sec[min(retry_idx, len(blocked_env_backoff_sec) - 1)])
                if blocked_env_backoff_sec
                else 10.0
            )
            op_logger.info(
                "STAGE0_ORACLE_RETRY_BLOCKED_ENV op=%s retry=%d/%d wait=%.1fs detail=%s",
                operator_name,
                retry_idx + 1,
                max_retries,
                wait_sec,
                result.error_message[:240],
            )
            time.sleep(wait_sec)
        assert result is not None
        oracle_failure_kind = (
            "environment"
            if (not result.passed and is_transient_blocked_env)
            else (result.failure_kind or "candidate")
        )
        op_logger.info(
            "STAGE0_ORACLE op=%s sha256=%s passed=%s run_id=%s state=%s elapsed=%.1fs",
            operator_name,
            digest,
            result.passed,
            result.run_id,
            result.state,
            result.elapsed_sec,
        )
        return OracleAccuracyResult(
            passed=result.passed,
            error_message=result.error_message,
            candidate_sha256=result.candidate_sha256,
            l3_run_id=result.run_id,
            l3_receipt_path=result.receipt_path,
            l3_state=result.state,
            elapsed_sec=result.elapsed_sec,
            failure_kind=oracle_failure_kind,
        )

    return _validate


def _render_seed_package_init(op_id: str, entrypoint: str) -> str:
    """Render the package initializer for a freshly cold-started operator directory."""
    return (
        f'"""Cold-start Ascend 950PR operator package for {op_id}."""\n\n'
        f"from .kernels import {entrypoint}, kernel\n\n"
        f'__all__ = [{entrypoint!r}, "kernel"]\n'
    )


def prepare_marathon_seed(
    *,
    operator_name: str,
    target_module: Path,
    staging_dir: Path,
    worktree: Path,
    env_file: Path | None,
    cannbench_repo: Path | None,
    llm_backend: KernelLLMBackend,
    cold_start: bool,
    device_id: str,
    op_logger: logging.Logger,
    pure_golden: bool = False,
) -> SeedSynthesisOutcome:
    """Run bounded Stage-0 seed synthesis or explicit incumbent takeover before the campaign."""
    effective_cold_start = bool(cold_start or pure_golden)
    registry = CannbenchOpsRegistry(
        tasks_root=cannbench_repo,
        source_root=worktree,
        pure_golden=pure_golden,
    )
    spec = registry.resolve(operator_name)
    if not spec.golden_path.is_file():
        raise FileNotFoundError(
            f"operator {operator_name} has no PyTorch golden reference at {spec.golden_path}"
        )
    golden_source = spec.golden_path.read_text(encoding="utf-8")
    desc_text = (
        spec.desc_path.read_text(encoding="utf-8")
        if spec.desc_path.is_file()
        else ""
    )
    golden_contract = extract_golden_entrypoint_contract(
        golden_source,
        spec.entrypoint,
        proto=spec.proto,
        desc_text=desc_text,
    )
    take_over_incumbent = not effective_cold_start and target_module.is_file()
    legacy_regime_specs = (
        registry.get_regime_specs(operator_name) if take_over_incumbent else None
    )

    def _stage0_static_validator_factory(
        regime_specs: tuple[RegimeTargetSpec, ...],
        *,
        semantics: PyTorchOpSemantics | None = None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        return build_stage0_static_validator(
            regime_specs,
            entrypoint=spec.entrypoint,
            worktree=worktree,
            env_file=env_file,
            entrypoint_contract=golden_contract,
            semantics=semantics,
        )

    init_path = target_module.parent / "__init__.py"
    created_init = not take_over_incumbent and not init_path.exists()
    if created_init:
        target_module.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(init_path, _render_seed_package_init(spec.op_id, spec.entrypoint))
    try:
        outcome = asyncio.run(
            prepare_operator_seed_or_takeover(
                operator_name=operator_name,
                entrypoint=spec.entrypoint,
                pytorch_golden_source=golden_source,
                proto=spec.proto,
                cases=spec.cases,
                legacy_regime_specs=legacy_regime_specs,
                llm_backend=llm_backend,
                l0_l1_validator_factory=_stage0_static_validator_factory,
                oracle_accuracy_validator=build_stage0_oracle_validator(
                    operator_name=operator_name,
                    staging_dir=staging_dir,
                    worktree=worktree,
                    device_id=device_id,
                    op_logger=op_logger,
                ),
                existing_module_path=None if effective_cold_start else target_module,
                authored_hir_source=None,
                scheduled_hir_source=None,
                force_cold_start=effective_cold_start,
                max_seed_attempts=6,
                desc_text=desc_text,
                golden_contract=golden_contract,
            )
        )
    except (OSError, RuntimeError, ValueError):
        if created_init:
            init_path.unlink(missing_ok=True)
        raise
    if outcome.mode == "cold_start":
        digest = hashlib.sha256(outcome.module_source.encode("utf-8")).hexdigest()
        assert outcome.verification_state == "l3_verified"
        if outcome.candidate_sha256 != digest:
            raise RuntimeError(
                f"Stage-0 seed digest mismatch for {operator_name}: outcome carries "
                f"{outcome.candidate_sha256!r}, module source hashes to {digest}"
            )
        atomic_write_text(target_module, outcome.module_source)
        op_logger.info(
            "STAGE0_SEED_PUBLISH op=%s sha256=%s run_id=%s receipt=%s",
            operator_name,
            digest,
            outcome.l3_run_id,
            outcome.l3_receipt_path,
        )
    return outcome


def _ensure_sys_paths(worktree: Path) -> None:
    for entry in (
        str(worktree / "targets" / "ascend"),
        str(worktree / "src"),
        str(worktree),
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _has_intact_cann_opp(set_env_path: Path) -> bool:
    """Return True when the CANN install has intact built-in OPP kernel configs."""
    tbe_kernel_cfg = (
        set_env_path.parent
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
    )
    return tbe_kernel_cfg.is_dir()


def _find_cann_setup_script(seeded: Mapping[str, str]) -> str:
    direct = seeded.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
    if direct and Path(direct).is_file():
        return str(Path(direct).resolve())

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add_candidate(path: Path) -> None:
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    base_roots: list[Path] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        base = Path(raw).expanduser()
        base_roots.append(base)
        _add_candidate(base / "set_env.sh")
        _add_candidate(base.parent / "set_env.sh")

    for base in base_roots:
        for v_root in (base.parent, base.parent.parent):
            if not v_root.is_dir():
                continue
            _add_candidate(v_root / "Ascend" / "9.3.0-weekly" / "cann-9.3.0" / "set_env.sh")
            for pattern in (
                "Ascend/*/cann-*/set_env.sh",
                "*/cann-*/set_env.sh",
                "cann-*/set_env.sh",
            ):
                for match in sorted(v_root.glob(pattern)):
                    _add_candidate(match)

    bisheng = shutil.which("bisheng")
    if bisheng:
        b_path = Path(bisheng).resolve()
        for parent in b_path.parents[:5]:
            _add_candidate(parent / "set_env.sh")
            for v_root in (parent.parent, parent.parent.parent):
                if v_root.is_dir():
                    for pattern in (
                        "Ascend/*/cann-*/set_env.sh",
                        "*/cann-*/set_env.sh",
                        "cann-*/set_env.sh",
                    ):
                        for match in sorted(v_root.glob(pattern)):
                            _add_candidate(match)

    for cand in candidates:
        if _has_intact_cann_opp(cand):
            return str(cand)
    return str(candidates[0]) if candidates else ""


def _infer_conda_settings(python_bin: str, seeded: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    py_path = Path(python_bin).expanduser()
    resolved = py_path.resolve() if py_path.exists() else py_path
    parts = resolved.parents
    if len(parts) >= 4 and parts[2].name == "envs":
        out["TILEFOUNDRY_NPU_CONDA_ENV"] = parts[1].name
        out["TILEFOUNDRY_NPU_CONDA_ENVS_PATH"] = str(parts[2])
        conda_sh = parts[3] / "etc" / "profile.d" / "conda.sh"
        if conda_sh.is_file():
            out["TILEFOUNDRY_NPU_CONDA_SH"] = str(conda_sh)

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix and "TILEFOUNDRY_NPU_CONDA_ENV" not in out:
        cp = Path(conda_prefix)
        out["TILEFOUNDRY_NPU_CONDA_ENV"] = cp.name
        if cp.parent.name == "envs":
            out["TILEFOUNDRY_NPU_CONDA_ENVS_PATH"] = str(cp.parent)
            cand_sh = cp.parent.parent / "etc" / "profile.d" / "conda.sh"
            if cand_sh.is_file():
                out["TILEFOUNDRY_NPU_CONDA_SH"] = str(cand_sh)

    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe and "TILEFOUNDRY_NPU_CONDA_SH" not in out:
        ce = Path(conda_exe).resolve()
        if len(ce.parents) >= 2:
            cand_sh = ce.parents[1] / "etc" / "profile.d" / "conda.sh"
            if cand_sh.is_file():
                out["TILEFOUNDRY_NPU_CONDA_SH"] = str(cand_sh)

    for key in (
        "TILEFOUNDRY_NPU_CONDA_SH",
        "TILEFOUNDRY_NPU_CONDA_ENV",
        "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    ):
        val = seeded.get(key, "").strip() or os.environ.get(key, "").strip()
        if val:
            out[key] = val
    return out


def discover_marathon_environment(
    *,
    operator_name: str,
    worktree: Path,
    state_dir: Path,
    base_env_file: str | Path | None = None,
    source_root: Path | None = None,
) -> dict[str, str]:
    """Discover runtime toolchain and workspace paths with no hardcoded machine paths."""
    env_hint = base_env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    env_path = Path(env_hint).expanduser() if env_hint else None
    if env_path is not None and not env_path.is_file():
        env_path = None
    seeded = parse_simple_env_file(env_path) if env_path is not None else {}

    for key, value in os.environ.items():
        if env_path is not None and _is_llm_env_key(key):
            continue
        if key.startswith("TILEFOUNDRY_NPU_") and value.strip() and key not in seeded:
            seeded[key] = value.strip()

    ws_parents: list[Path] = [worktree.resolve().parent]
    if source_root is not None:
        src_parent = source_root.resolve().parent
        if src_parent not in ws_parents:
            ws_parents.append(src_parent)

    if env_path is None:
        for parent in ws_parents:
            adapter_env = (
                parent / "AgentGates" / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh"
            )
            if adapter_env.is_file():
                adapter_values = parse_simple_env_file(adapter_env)
                for key in _ADAPTER_TOOLCHAIN_ENV_KEYS:
                    if key in seeded:
                        continue
                    val = adapter_values.get(key, "").strip()
                    if not val:
                        continue
                    if key in _ADAPTER_PATH_ENV_KEYS and not Path(val).expanduser().is_file():
                        continue
                    seeded[key] = val
                break

    python_bin = (
        seeded.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "").strip()
        or sys.executable
    )
    values: dict[str, str] = dict(seeded)
    values["TILEFOUNDRY_NPU_PYTHON"] = python_bin
    values.update(_infer_conda_settings(python_bin, seeded))

    cann_setup = _find_cann_setup_script(seeded)
    if cann_setup:
        values["TILEFOUNDRY_NPU_CANN_SETUP"] = cann_setup

    tl_dir = next(
        (parent / "tilelang" for parent in ws_parents if (parent / "tilelang").is_dir()),
        ws_parents[0] / "tilelang",
    )
    if "TILEFOUNDRY_NPU_TILELANG_SOURCE" not in values and tl_dir.is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_SOURCE"] = str(tl_dir)
    if "TILEFOUNDRY_NPU_TILELANG_REPO" not in values and tl_dir.is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_REPO"] = str(tl_dir)
    tl_src = (
        Path(values["TILEFOUNDRY_NPU_TILELANG_SOURCE"])
        if values.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        else tl_dir
    )
    if "TILEFOUNDRY_NPU_TILELANG_BUILD" not in values and (tl_src / "build").is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_BUILD"] = str(tl_src / "build")

    if not values.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"):
        for parent in ws_parents:
            cand = parent / "cann-bench"
            if cand.is_dir() and (cand / "tasks").is_dir():
                values["TILEFOUNDRY_NPU_CANNBENCH_REPO"] = str(cand)
                break

    sibling_keys = (
        ("TILEFOUNDRY_NPU_CANNBOT_DSL_REPO", "cannbot-dsl"),
        ("TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "ops-transformer"),
        ("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", "asc-devkit"),
    )
    for key, folder_name in sibling_keys:
        if not values.get(key):
            for parent in ws_parents:
                cand = parent / folder_name
                if cand.is_dir():
                    values[key] = str(cand)
                    break

    if not values.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN"):
        for parent in ws_parents:
            ag_cand = parent / "AgentGates" / ".venv" / "bin" / "agent-gates"
            if ag_cand.is_file():
                values["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = str(ag_cand)
                break
        if not values.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN"):
            ag_which = shutil.which("agent-gates")
            if ag_which:
                values["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag_which

    values.setdefault("TILEFOUNDRY_NPU_CPU_TESTS", "")
    values.setdefault("TILEFOUNDRY_NPU_DEVICE_TESTS", "")
    values.setdefault("TILEFOUNDRY_NPU_BUILD_SUBDIR", "build")
    values["TILEFOUNDRY_NPU_WORK_ROOT"] = str((state_dir / "cannbench-work").resolve())
    values.setdefault("TILEFOUNDRY_NPU_CANNBENCH_ARCH", "ascend950")
    values["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] = (
        f"{get_operator_task(operator_name)}:{get_operator_module_relpath(operator_name)}"
    )
    return values


def ensure_marathon_env_file(
    *,
    operator_name: str,
    worktree: Path,
    state_dir: Path,
    env_file: str | Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """Reuse an existing env.sh or materialize a discovered zero-config env.sh."""
    target = Path(env_file).expanduser() if env_file is not None else (state_dir / "env.sh")
    target = target.resolve()
    discovered = discover_marathon_environment(
        operator_name=operator_name,
        worktree=worktree,
        state_dir=state_dir,
        base_env_file=target if target.is_file() else None,
        source_root=source_root,
    )
    allowed_discovered = {
        key: val for key, val in discovered.items() if _is_agent_gates_allowed_env_key(key)
    }
    _validate_llm_api_key_env(allowed_discovered.get(_LLM_API_KEY_ENV, "").strip())
    if target.is_file():
        existing = parse_simple_env_file(target)
        disallowed_existing = [
            key for key in existing if not _is_agent_gates_allowed_env_key(key)
        ]
        missing = [
            key
            for key, val in allowed_discovered.items()
            if val and not existing.get(key)
        ]
        if not missing and not disallowed_existing:
            return target
        merged = dict(allowed_discovered)
        merged.update(
            {
                key: val
                for key, val in existing.items()
                if val and _is_agent_gates_allowed_env_key(key)
            }
        )
    else:
        merged = allowed_discovered

    lines = [
        f"# Isolated marathon environment for {operator_name}",
        "# Plain KEY=VALUE assignments; safe for env parsers and shell sourcing.",
        "",
    ]
    for key in sorted(merged):
        lines.append(_format_env_line(key, merged[key]))
    lines.append("")
    atomic_write_text(target, "\n".join(lines))
    return target


def configure_marathon_process_env(
    *,
    operator_name: str,
    env_file: Path,
    worktree: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, str]:
    """Configure coordinator process env vars cleanly for isolated worker subprocesses."""
    short_op = short_operator_tag(operator_name)
    tmp_dir = Path(tempfile.gettempdir()) / f"tf_marathon_{short_op}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["USE_CUDA"] = "0"
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    os.environ["TILEFOUNDRY_NPU_ENV_FILE"] = str(env_file)

    parsed = parse_simple_env_file(env_file)
    for key in (
        "TILEFOUNDRY_NPU_AGENT_GATES_BIN",
        "TILEFOUNDRY_NPU_CANNBENCH_REPO",
        "TILEFOUNDRY_NPU_TILELANG_SOURCE",
        "TILEFOUNDRY_NPU_TILELANG_BUILD",
    ):
        val = parsed.get(key, "").strip()
        if val and not os.environ.get(key):
            os.environ[key] = val

    if not os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip():
        ag_which = shutil.which("agent-gates")
        if ag_which:
            os.environ["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag_which
    return parsed


def _resolve_worktree(cfg: MarathonConfig) -> tuple[Path, Path]:
    source_root = (
        Path(cfg.source_root).expanduser().resolve()
        if cfg.source_root is not None
        else default_source_root()
    )
    if cfg.worktree is not None:
        return source_root, Path(cfg.worktree).expanduser().resolve()
    if not cfg.isolate_worktree:
        return source_root, source_root

    short_op = short_operator_tag(cfg.operator_name)
    cand = source_root.parent / f"tilefoundry-marathon-{short_op}"
    if cand.is_dir() and (cand / ".git").exists():
        return source_root, cand.resolve()
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(cand), "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return source_root, cand.resolve()


FastCreativeAgyBackend = AgyCliLLMBackend


class ExtremeCampaignRunner(PerfTargetCampaignRunner):
    """Runner with stagnation-discounted regime rotation and per-round telemetry."""

    def __init__(
        self,
        *args,
        stagnation_discount: float = 0.45,
        op_logger: logging.Logger | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_stag_by_regime: dict[str, int] = {}
        self._stagnation_discount = max(0.0, float(stagnation_discount))
        self._op_logger = op_logger or logger

    def _restore_from_checkpoint(self):  # type: ignore[override]
        restored = super()._restore_from_checkpoint()
        self._live_stag_by_regime = dict(restored[12])
        return restored

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None,
    ) -> RegimeTargetSpec | None:
        by_tag = {r.regime_tag: r for r in self._regime_specs}
        if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
            spec = by_tag[current_regime]
            if not self._is_regime_satisfied(spec, best_result, active_anchors):
                return spec

        candidates = [
            spec
            for spec in self._regime_specs
            if spec.regime_tag not in frozen_regimes
            and not self._is_regime_satisfied(spec, best_result, active_anchors)
        ]
        if not candidates and allow_satisfied_fallback:
            if (
                current_regime
                and current_regime not in frozen_regimes
                and current_regime in by_tag
            ):
                return by_tag[current_regime]
            candidates = [
                spec for spec in self._regime_specs if spec.regime_tag not in frozen_regimes
            ]
        if not candidates:
            return None
        if exclude_regime and any(spec.regime_tag != exclude_regime for spec in candidates):
            candidates = [spec for spec in candidates if spec.regime_tag != exclude_regime]
        candidates.sort(
            key=lambda s: best_result.regime_headroom(s.regime_tag)
            / (1.0 + self._stagnation_discount * self._live_stag_by_regime.get(s.regime_tag, 0)),
            reverse=True,
        )
        return candidates[0]

    def _save_checkpoint(self, **kwargs) -> None:  # type: ignore[override]
        self._live_stag_by_regime = {
            str(k): int(v) for k, v in (kwargs.get("stagnation_by_regime") or {}).items()
        }
        super()._save_checkpoint(**kwargs)
        rounds = kwargs.get("rounds") or ()
        best_result = kwargs.get("best_result")
        frozen_regimes = sorted(str(x) for x in (kwargs.get("frozen_regimes") or ()))
        stop_reason = str(kwargs.get("stop_reason", ""))
        op_name = self._config.operator_name
        if rounds:
            last = rounds[-1]
            self._op_logger.info(
                "ROUND_TELEMETRY op=%s round=%d regime=%s promoted=%s "
                "best_score=%.6f best_speedup=%.6f stag=%s frozen=%s stop=%s",
                op_name,
                last.round_idx,
                last.regime_tag,
                last.promoted,
                best_result.surrogate_score if best_result else 0.0,
                best_result.avg_speedup if best_result else 0.0,
                self._live_stag_by_regime,
                frozen_regimes,
                stop_reason,
            )
        else:
            self._op_logger.info(
                "BASELINE_TELEMETRY op=%s best_score=%.6f best_speedup=%.6f stop=%s",
                op_name,
                best_result.surrogate_score if best_result else 0.0,
                best_result.avg_speedup if best_result else 0.0,
                stop_reason,
            )


def build_prewarmed_l3_runner(
    *,
    operator_name: str,
    worktree: Path,
    env_file: Path,
    device_id: str,
    prewarm_l3: bool = True,
    op_logger: logging.Logger | None = None,
) -> Callable[..., L3AcceptanceResult]:
    """Build the L3 runner; the adapter owns environment, cache and prompt loading.

    ``prewarm_l3`` remains accepted so existing marathon configurations keep parsing,
    but promotion evidence now comes from the adapter-owned V2 gate, which prepares
    its own interpreter, cache policy and compare bridge inside the workspace.
    """
    _ = env_file, prewarm_l3
    log = op_logger or logger
    default_device_id = str(device_id)

    def _run_l3(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        result = run_l3_full_contract_via_agent_gates(
            operator_name=operator_name,
            candidate_module_path=module_path,
            source_root=worktree,
            device_id=str(device_id or default_device_id),
            resume_execution_id=resume_execution_id,
            resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )
        log.info(
            "L3_CONTRACT_DONE op=%s passed=%s elapsed=%.1fs run_id=%s execution_id=%s "
            "failure_kind=%s",
            operator_name,
            result.passed,
            result.elapsed_sec,
            result.run_id,
            result.execution_id,
            result.failure_kind,
        )
        return result

    return _run_l3


def build_resilient_l2_evaluator(
    *,
    operator_name: str,
    worktree: Path,
    env_file: Path,
    device_id: str,
    warmup: int = 3,
    repeat: int = 5,
    batch_fn: Callable[..., tuple[dict[str, CandidateSurrogateResult], str, str]] | None = None,
    op_logger: logging.Logger | None = None,
) -> Callable[[Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]], L2BatchResults]:
    """Build batched L2 evaluator with single-candidate isolation fallback on failures."""
    log = op_logger or logger
    eval_fn = batch_fn or run_surrogate_batch_via_agent_gates

    def _eval(
        cands: Sequence[tuple[str, Path]],
        anc_seq: Sequence[AnchorCaseSpec],
    ) -> L2BatchResults:
        cand_list = list(cands)
        try:
            res, run_id, _rcp = eval_fn(
                operator_name,
                cand_list,
                anc_seq,
                source_root=worktree,
                env_file=env_file,
                device_id=device_id,
                warmup=warmup,
                repeat=repeat,
                require_all_passed=False,
                timeout_sec=660,
            )
            return L2BatchResults(res, {cid: run_id for cid in res})
        except Exception as exc:
            if len(cand_list) <= 1:
                raise
            log.warning(
                "Batched L2 failed (%s: %s); falling back to single-candidate evaluation for %s",
                type(exc).__name__,
                exc,
                [cid for cid, _ in cand_list],
            )
            merged: dict[str, CandidateSurrogateResult] = {}
            run_ids: dict[str, str] = {}
            last_exc: Exception | None = None
            for cid, mod_path in cand_list:
                try:
                    one_res, run_id, _rcp = eval_fn(
                        operator_name,
                        [(cid, mod_path)],
                        anc_seq,
                        source_root=worktree,
                        env_file=env_file,
                        device_id=device_id,
                        warmup=warmup,
                        repeat=repeat,
                        require_all_passed=False,
                        timeout_sec=360,
                    )
                    merged.update(one_res)
                    run_ids.update({cand_id: run_id for cand_id in one_res})
                except Exception as sub_exc:
                    last_exc = sub_exc
                    log.warning(
                        "Single-candidate L2 fallback failed for %s: %s: %s",
                        cid,
                        type(sub_exc).__name__,
                        sub_exc,
                    )
            if merged:
                return L2BatchResults(merged, run_ids)
            assert last_exc is not None
            raise last_exc

    return _eval


def salvage_unpromoted_beam_winner(
    *,
    operator_name: str,
    checkpoint_path: Path,
    target_module_path: Path,
    worktree: Path,
    env_file: Path,
    device_id: str,
    min_surrogate_gain: float,
    l2_evaluator: Callable[
        [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
        dict[str, CandidateSurrogateResult],
    ],
    l3_runner: Callable[[Path], L3AcceptanceResult],
    regime_specs: Sequence[RegimeTargetSpec] | None = None,
    op_logger: logging.Logger | None = None,
    experience_lake_dir: str | Path | None = None,
) -> bool:
    """Re-verify and promote high-scoring unpromoted beam entries across restarts."""
    if not checkpoint_path.is_file():
        return False
    log = op_logger or logger
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("pending_l4"), Mapping):
        log.info(
            "SALVAGE_SKIPPED pending L4 checkpoint must resume before any target mutation"
        )
        return False
    best_dict = payload.get("best_result") or {}
    best_res = CandidateSurrogateResult.from_dict(best_dict)
    if not payload.get("incumbent_source"):
        return False
    incumbent_src = str(payload["incumbent_source"])
    promoted_set = {int(x) for x in (payload.get("promoted_rounds") or ())}
    tried_cache_path = checkpoint_path.parent / ".salvage_tried.json"
    tried_keys: set[str] = set()
    if tried_cache_path.is_file():
        try:
            tried_keys = set(json.loads(tried_cache_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            tried_keys = set()

    beam_states = payload.get("beam_states") or {}
    specs_by_tag = {
        spec.regime_tag: spec
        for spec in (
            regime_specs
            if regime_specs is not None
            else get_default_regime_specs(operator_name)
        )
    }
    candidates: list[tuple[float, str, dict[str, object], RegimeTargetSpec]] = []
    for rtag, bmap in beam_states.items():
        spec = specs_by_tag.get(str(rtag))
        if spec is None or not isinstance(bmap, Mapping):
            continue
        for row in bmap.get("exploit_beam", ()):
            if not isinstance(row, Mapping):
                continue
            r_idx = int(row.get("round_idx", -1))
            if r_idx in promoted_set or not row.get("dsl_source"):
                continue
            if not row.get("accuracy_passed", False):
                continue
            obj = float(row.get("objective_score", 0.0) or 0.0)
            candidates.append((obj, str(rtag), dict(row), spec))

    if not candidates:
        return False
    candidates.sort(key=lambda item: item[0], reverse=True)

    active_anchors = [
        AnchorCaseSpec.from_dict(item) for item in (payload.get("active_anchors") or ())
    ] or list(get_default_anchors(operator_name))

    for _obj, rtag, row, regime_spec in candidates[:2]:
        cid = str(row.get("candidate_id", ""))
        r_idx = int(row.get("round_idx", -1))
        try:
            patched_src = patch_regime_into_module(
                incumbent_src,
                regime_spec,
                str(row["dsl_source"]),
            )
        except Exception as exc:
            log.warning("SALVAGE_SKIPPED %s patch failed: %s", cid, exc)
            continue

        sha = hashlib.sha256(patched_src.encode("utf-8")).hexdigest()
        cache_key = f"{cid}:{sha}"
        if cache_key in tried_keys:
            continue

        with tempfile.TemporaryDirectory(prefix="tf_marathon_salvage_") as tmp_dir:
            cand_file = Path(tmp_dir) / f"{cid or 'cand'}.py"
            cand_file.write_text(patched_src, encoding="utf-8")
            try:
                res_map = l2_evaluator([(cid, cand_file)], active_anchors)
            except Exception as exc:
                log.warning("SALVAGE_SKIPPED %s L2 eval failed: %s", cid, exc)
                continue
        cand_res = res_map.get(cid)
        salvage_lake = (
            ExperienceLake(experience_lake_dir)
            if experience_lake_dir is not None and str(experience_lake_dir).strip()
            else None
        )

        def _record_salvage(msprof_pmu: Mapping[str, float] | None = None) -> None:
            if salvage_lake is None or cand_res is None:
                return
            try:
                cand_sha = compute_source_sha256(patched_src)
                parent_sha = compute_source_sha256(incumbent_src)
                salvage_lake.register_candidate(
                    CandidateArtifactRecord(
                        cand_sha=cand_sha,
                        operator=operator_name,
                        regime_tag=rtag,
                        backend_track=str(row.get("track") or "tilelang_dsl"),
                        parent_sha=parent_sha,
                        worker_kind=str(row.get("worker_kind") or "salvage"),
                        rule_id=(str(row["rule_id"]) if row.get("rule_id") else None),
                        dsl_line_count=len(patched_src.splitlines()),
                    )
                )
                run_id = f"salvage_{operator_name.replace('/', '_')}_{int(time.time())}_{os.getpid()}"
                for a in cand_res.anchors:
                    if a.cand_us <= 0.0:
                        continue
                    salvage_lake.append_observation(
                        CaseObservationRecord(
                            run_id=run_id,
                            operator=operator_name,
                            regime_tag=a.regime_tag,
                            cand_sha=cand_sha,
                            eval_tier="salvage",
                            case_num=int(a.case_num),
                            measured_us=float(a.cand_us),
                            accuracy_passed=bool(a.accuracy_passed),
                            parent_sha=parent_sha,
                            device_id=str(device_id),
                            t_hw_us=float(a.t_hw_us),
                            ref_perf_us=float(a.ref_perf_us),
                            speedup=float(a.speedup),
                            eq3_score=float(a.eq3_score),
                            msprof_pmu=(
                                {str(k): float(v) for k, v in msprof_pmu.items()}
                                if (msprof_pmu and a.regime_tag == rtag)
                                else {}
                            ),
                        )
                    )
            except Exception as exc:
                log.debug("Salvage ExperienceLake write skipped: %s", exc)

        if (
            cand_res is None
            or not cand_res.all_passed
            or cand_res.surrogate_score <= best_res.surrogate_score + min_surrogate_gain
        ):
            _record_salvage()
            tried_keys.add(cache_key)
            atomic_write_text(tried_cache_path, json.dumps(sorted(tried_keys)) + "\n")
            continue

        atomic_write_text(target_module_path, patched_src)
        l3_res = l3_runner(target_module_path)
        if not l3_res.passed:
            _record_salvage()
            atomic_write_text(target_module_path, incumbent_src)
            tried_keys.add(cache_key)
            atomic_write_text(tried_cache_path, json.dumps(sorted(tried_keys)) + "\n")
            log.warning(
                "SALVAGE_SKIPPED %s L3 failed: %s",
                cid,
                l3_res.error_message[-300:],
            )
            continue

        try:
            msprof_map = collect_bottleneck_msprof_via_agent_gates(
                operator_name,
                target_module_path,
                [a for a in active_anchors if a.regime_tag == rtag],
                source_root=worktree,
                env_file=env_file,
                device_id=device_id,
                timeout_sec=600,
            )
        except Exception as exc:
            log.warning("SALVAGE msprof refresh failed (non-fatal): %s", exc)
            msprof_map = {}

        _record_salvage(msprof_map.get(rtag))

        msprof_cache = dict(payload.get("msprof_cache") or {})
        if rtag in msprof_map:
            msprof_cache[f"{sha}:{rtag}"] = msprof_map[rtag]

        rounds_raw = list(payload.get("rounds") or [])
        for r in rounds_raw:
            cur_idx = int(r.get("round_idx", -1))
            if cur_idx == r_idx:
                r["promoted"] = True
                r["rolled_back_l3"] = False
                r["stagnation_rounds_after"] = 0
                r["best_round_surrogate_score"] = round(cand_res.surrogate_score, 6)
                r["best_so_far_surrogate_score"] = round(cand_res.surrogate_score, 6)
            if (
                cur_idx > r_idx
                and float(r.get("best_so_far_surrogate_score", 0.0)) < cand_res.surrogate_score
            ):
                r["best_so_far_surrogate_score"] = round(cand_res.surrogate_score, 6)
                r["best_so_far_speedup"] = round(cand_res.avg_speedup, 6)

        prom_rounds = sorted(promoted_set | {r_idx}) if r_idx >= 0 else sorted(promoted_set)
        rb_rounds = [int(x) for x in payload.get("rolled_back_rounds", ()) if int(x) != r_idx]
        stag_map = {str(k): int(v) for k, v in (payload.get("stagnation_by_regime") or {}).items()}
        stag_map[rtag] = 0

        payload["incumbent_source"] = patched_src
        payload["best_result"] = cand_res.to_dict()
        payload["promoted_rounds"] = prom_rounds
        payload["rolled_back_rounds"] = rb_rounds
        payload["stagnation_by_regime"] = stag_map
        payload["stagnation_rounds"] = 0
        payload["current_regime"] = rtag
        payload["last_unpromoted_regime"] = None
        payload["rounds"] = rounds_raw
        payload["msprof_cache"] = msprof_cache
        atomic_write_text(checkpoint_path, json.dumps(payload, indent=2) + "\n")
        log.info(
            "SALVAGE_PROMOTED op=%s %s score=%.6f -> %.6f speedup=%.6f run_id=%s",
            operator_name,
            cid,
            best_res.surrogate_score,
            cand_res.surrogate_score,
            cand_res.avg_speedup,
            l3_res.run_id,
        )
        return True
    return False


def _configure_marathon_logging(log_path: Path | None) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == resolved
            for h in root_logger.handlers
        )
        if not already:
            fh = logging.FileHandler(resolved, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            root_logger.addHandler(fh)


def run_operator_marathon(
    config: MarathonConfig | None = None,
    **overrides: object,
) -> int:
    """Configure workspace, environment, and run ExtremeCampaignRunner.

    Capture the parent's physical visibility before configure_marathon_process_env
    removes it, so it may only narrow the declared device pool.
    """
    if config is None:
        cfg = MarathonConfig(**overrides)  # type: ignore[arg-type]
    elif overrides:
        data = {k: getattr(config, k) for k in config.__dataclass_fields__}
        data.update(overrides)
        cfg = MarathonConfig(**data)  # type: ignore[arg-type]
    else:
        cfg = config

    _source_root, worktree = _resolve_worktree(cfg)
    _ensure_sys_paths(worktree)

    short_op = short_operator_tag(cfg.operator_name)
    state_dir = (
        Path(cfg.state_dir).expanduser().resolve()
        if cfg.state_dir is not None
        else (worktree / ".tilefoundry-marathon" / cfg.operator_name).resolve()
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    op_tag = cfg.operator_name.rsplit("/", 1)[-1]
    external_pure_golden_mirror: Path | None = None
    if state_dir.is_relative_to(worktree):
        staging_dir = (state_dir / "stage0_candidates").resolve()
    else:
        state_key = hashlib.sha256(str(state_dir).encode("utf-8")).hexdigest()[:12]
        staging_dir = (
            worktree / ".tilefoundry-marathon" / "_stage0_staging" / state_key / short_op
        ).resolve()

    if cfg.target_module_path is not None:
        target_module = Path(cfg.target_module_path).expanduser().resolve()
    elif cfg.pure_golden:
        if state_dir.is_relative_to(worktree):
            target_module = (state_dir / "pure_golden_ops" / op_tag / "kernels.py").resolve()
        else:
            state_key = hashlib.sha256(str(state_dir).encode("utf-8")).hexdigest()[:12]
            target_module = (
                worktree / ".tilefoundry-marathon" / "pure_golden_ops" / state_key / op_tag / "kernels.py"
            ).resolve()
            external_pure_golden_mirror = (
                state_dir / "pure_golden_ops" / op_tag / "kernels.py"
            ).resolve()
    else:
        target_module = (worktree / get_operator_module_relpath(cfg.operator_name)).resolve()
    checkpoint_path = (
        Path(cfg.checkpoint_path).expanduser().resolve()
        if cfg.checkpoint_path is not None
        else (state_dir / "checkpoint.json")
    )
    report_path = (
        Path(cfg.report_path).expanduser().resolve()
        if cfg.report_path is not None
        else (state_dir / "report.json")
    )
    campaign_kb_path = (
        Path(cfg.campaign_kb_path).expanduser().resolve()
        if cfg.campaign_kb_path is not None
        else (state_dir / "campaign_kb.yaml")
    )

    resolved_log_path = (
        Path(cfg.log_path).expanduser().resolve()
        if cfg.log_path is not None
        else (state_dir / "marathon.log")
    )
    _configure_marathon_logging(resolved_log_path)
    op_logger = logging.getLogger(f"tilefoundry_ascend.marathon.{short_op}")

    env_file = ensure_marathon_env_file(
        operator_name=cfg.operator_name,
        worktree=worktree,
        state_dir=state_dir,
        env_file=cfg.env_file,
        source_root=_source_root,
    )
    visible_device_ids = parse_device_ids(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    )
    env_values = configure_marathon_process_env(
        operator_name=cfg.operator_name,
        env_file=env_file,
        worktree=worktree,
        source_root=_source_root,
    )
    primary_device_id, device_pool = resolve_campaign_device_plan(
        device_id=str(cfg.device_id),
        device_ids=tuple(cfg.device_ids),
        env_device_ids=parse_device_ids(
            env_values.get(CAMPAIGN_DEVICE_IDS_ENV_KEY, "")
        ),
        visible_device_ids=visible_device_ids,
    )
    llm_backend = create_llm_backend_from_env(overrides=_llm_backend_overrides(env_values, cfg))
    raw_cannbench_repo = (
        env_values.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
        or os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    )
    cannbench_repo = Path(raw_cannbench_repo) if raw_cannbench_repo else None
    op_logger.info(
        "MARATHON_BOOT op=%s device=%s pool=%s worktree=%s env_file=%s",
        cfg.operator_name,
        primary_device_id,
        ",".join(device_pool),
        worktree,
        env_file,
    )
    resume_flag = checkpoint_path.is_file() if cfg.resume is None else bool(cfg.resume)
    seeded_regime_specs: Sequence[RegimeTargetSpec] | None = None
    seed_outcome: SeedSynthesisOutcome | None = None
    if cfg.auto_bootstrap_seed:
        if resume_flag and checkpoint_path.is_file() and target_module.is_file():
            op_logger.info(
                "STAGE0_SEED_SKIPPED_RESUME op=%s checkpoint=%s target=%s",
                cfg.operator_name,
                checkpoint_path,
                target_module,
            )
        else:
            try:
                seed_outcome = prepare_marathon_seed(
                    operator_name=cfg.operator_name,
                    target_module=target_module,
                    staging_dir=staging_dir,
                    worktree=worktree,
                    env_file=env_file,
                    cannbench_repo=cannbench_repo,
                    llm_backend=llm_backend,
                    cold_start=cfg.cold_start,
                    device_id=primary_device_id,
                    op_logger=op_logger,
                    pure_golden=cfg.pure_golden,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                op_logger.error("STAGE0_SEED_FAILED op=%s error=%s", cfg.operator_name, exc)
                return 2
            seeded_regime_specs = seed_outcome.regime_specs
            op_logger.info(
                "STAGE0_SEED op=%s mode=%s track=%s attempts=%d verification=%s sha256=%s "
                "run_id=%s receipt=%s l0_l1=%s oracle_accuracy=%s",
                cfg.operator_name,
                seed_outcome.mode,
                seed_outcome.initial_track,
                seed_outcome.attempts_used,
                seed_outcome.verification_state,
                seed_outcome.candidate_sha256,
                seed_outcome.l3_run_id,
                seed_outcome.l3_receipt_path,
                seed_outcome.l0_l1_passed,
                seed_outcome.oracle_accuracy_passed,
            )

    l2_evaluator = build_resilient_l2_evaluator(
        operator_name=cfg.operator_name,
        worktree=worktree,
        env_file=env_file,
        device_id=primary_device_id,
        warmup=cfg.warmup,
        repeat=cfg.repeat,
        op_logger=op_logger,
    )
    l3_runner = build_prewarmed_l3_runner(
        operator_name=cfg.operator_name,
        worktree=worktree,
        env_file=env_file,
        device_id=primary_device_id,
        prewarm_l3=cfg.prewarm_l3,
        op_logger=op_logger,
    )

    lake_dir = resolve_default_experience_lake_dir(
        worktree=worktree,
        checkpoint_path=checkpoint_path,
        explicit_dir=cfg.experience_lake_dir,
    )

    if cfg.salvage_beam_candidates:
        salvage_unpromoted_beam_winner(
            operator_name=cfg.operator_name,
            checkpoint_path=checkpoint_path,
            target_module_path=target_module,
            worktree=worktree,
            env_file=env_file,
            device_id=primary_device_id,
            min_surrogate_gain=cfg.min_surrogate_gain,
            l2_evaluator=l2_evaluator,
            l3_runner=l3_runner,
            regime_specs=seeded_regime_specs,
            op_logger=op_logger,
            experience_lake_dir=lake_dir,
        )

    perf_target = PerfTargetSpec(
        operator_name=cfg.operator_name,
        target_avg_speedup=cfg.target_avg_speedup,
        target_min_case_speedup=cfg.target_min_case_speedup,
        target_roofline_eta=cfg.target_roofline_eta,
        target_surrogate_score=cfg.target_surrogate_score,
        min_optimization_rounds=cfg.min_optimization_rounds,
        per_anchor_max_us={},
    )
    campaign_cfg = CampaignConfig(
        operator_name=cfg.operator_name,
        max_rounds=cfg.max_rounds,
        max_wall_clock_min=cfg.max_wall_clock_min,
        k_rule_workers=cfg.k_rule_workers,
        max_syntax_repairs=cfg.max_syntax_repairs,
        min_l2_gain_for_l3=cfg.min_l2_gain_for_l3,
        min_surrogate_gain=cfg.min_surrogate_gain,
        min_flywheel_gain=cfg.min_flywheel_gain,
        stagnation_escalate_rounds=cfg.stagnation_escalate_rounds,
        stagnation_freeze_rounds=cfg.stagnation_freeze_rounds,
        disable_early_goal_stop=cfg.disable_early_goal_stop,
        run_final_l4=cfg.run_final_l4,
        l4_regression_ratio=cfg.l4_regression_ratio,
        device_id=primary_device_id,
        retry_device_id=str(cfg.retry_device_id),
        device_ids=tuple(device for device in device_pool if device != primary_device_id),
        warmup=cfg.warmup,
        repeat=cfg.repeat,
        enable_tilesize_prescreen=cfg.enable_tilesize_prescreen,
        cost_prescreen_top_k=cfg.cost_prescreen_top_k,
        exploit_effort=cfg.exploit_effort,
        round_regime_tags=tuple(cfg.round_regime_tags),
        pure_golden=cfg.pure_golden,
    )
    tracker = BayesianRuleTracker.from_experience_lake(lake_dir)
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_runner=lambda spec: run_micro_probe_via_agent_gates(
            spec,
            source_root=worktree,
            env_file=env_file,
            device_id=primary_device_id,
            min_speedup=1.08,
            timeout_sec=360,
        ),
    )
    pure_anchors: tuple[AnchorCaseSpec, ...] | None = None
    if cfg.pure_golden:
        pure_registry = CannbenchOpsRegistry(
            tasks_root=cannbench_repo,
            source_root=worktree,
            pure_golden=True,
        )
        pure_anchors = pure_registry.get_anchors(cfg.operator_name)
        if seeded_regime_specs is None:
            seeded_regime_specs = pure_registry.get_regime_specs(cfg.operator_name)

    def _msprof_fn(
        op_name: str,
        cand_path: Path,
        anc_seq: Sequence[AnchorCaseSpec],
    ) -> Mapping[str, Mapping[str, float]]:
        return collect_bottleneck_msprof_via_agent_gates(
            op_name,
            cand_path,
            anc_seq,
            source_root=worktree,
            env_file=env_file,
            device_id=primary_device_id,
            timeout_sec=600,
        )

    def _l4_fn(mod_path: Path, **kwargs: object) -> object:
        return run_l4_official_cannbench_via_agent_gates(
            operator_name=cfg.operator_name,
            candidate_module_path=mod_path,
            source_root=worktree,
            device_id=str(kwargs.get("device_id") or primary_device_id),
            timeout_sec=DEFAULT_L4_CANNBENCH_WAIT_SEC,
            resume_execution_id=str(kwargs.get("resume_execution_id") or ""),
            resume_run_id=str(kwargs.get("resume_run_id") or ""),
            on_execution_started=kwargs.get("on_execution_started"),
        )

    if cfg.pure_golden:
        runner = ExtremeCampaignRunner(
            config=campaign_cfg,
            target_module_path=target_module,
            anchors=pure_anchors,
            regime_specs=seeded_regime_specs,
            perf_target=perf_target,
            llm_backend=llm_backend,
            tracker=tracker,
            flywheel_distiller=flywheel,
            l2_batch_evaluator=l2_evaluator,
            msprof_collector=_msprof_fn,
            l3_contract_runner=l3_runner,
            l4_cannbench_runner=_l4_fn,
            checkpoint_path=checkpoint_path,
            campaign_kb_path=campaign_kb_path,
            cost_calibration_path=state_dir / "cost_model_calibration.jsonl",
            experience_lake_dir=lake_dir,
            source_root=worktree,
            env_file=env_file,
            cannbench_repo=cannbench_repo,
            authored_hir_path=None,
            scheduled_hir_path=None,
            regime_blueprints=(
                seed_outcome.blueprints_by_regime if seed_outcome is not None else None
            ),
            topology_axes=(
                seed_outcome.semantics.base_axes if seed_outcome is not None else None
            ),
            stagnation_discount=cfg.stagnation_discount,
            op_logger=op_logger,
        )
    else:
        runner = ExtremeCampaignRunner(
            config=campaign_cfg,
            target_module_path=target_module,
            regime_specs=seeded_regime_specs,
            perf_target=perf_target,
            llm_backend=llm_backend,
            tracker=tracker,
            flywheel_distiller=flywheel,
            l2_batch_evaluator=l2_evaluator,
            msprof_collector=_msprof_fn,
            l3_contract_runner=l3_runner,
            l4_cannbench_runner=_l4_fn,
            checkpoint_path=checkpoint_path,
            campaign_kb_path=campaign_kb_path,
            cost_calibration_path=state_dir / "cost_model_calibration.jsonl",
            experience_lake_dir=lake_dir,
            source_root=worktree,
            env_file=env_file,
            cannbench_repo=cannbench_repo,
            regime_blueprints=(
                seed_outcome.blueprints_by_regime if seed_outcome is not None else None
            ),
            topology_axes=(
                seed_outcome.semantics.base_axes if seed_outcome is not None else None
            ),
            stagnation_discount=cfg.stagnation_discount,
            op_logger=op_logger,
        )
    op_logger.info(
        "Launching ExtremeCampaignRunner op=%s device=%s resume=%s max_rounds=%d "
        "max_wall_min=%.1f target_score=%.1f target_eta=%.3f worktree=%s",
        cfg.operator_name,
        primary_device_id,
        resume_flag,
        campaign_cfg.max_rounds,
        campaign_cfg.max_wall_clock_min,
        perf_target.target_surrogate_score,
        perf_target.target_roofline_eta,
        worktree,
    )
    report = asyncio.run(runner.run(resume=resume_flag))
    if external_pure_golden_mirror is not None and target_module.is_file():
        external_pure_golden_mirror.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(external_pure_golden_mirror, target_module.read_text(encoding="utf-8"))
        src_init = target_module.parent / "__init__.py"
        if src_init.is_file():
            atomic_write_text(
                external_pure_golden_mirror.parent / "__init__.py",
                src_init.read_text(encoding="utf-8"),
            )
    rendered = json.dumps(report.to_dict(), indent=2)
    atomic_write_text(report_path, rendered + "\n")
    print(rendered)
    return 0


def _atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _is_runner_alive(op: str, op_dir: Path, operator_name: str = "") -> bool:
    my_pid = os.getpid()
    dir_str = str(op_dir)
    needles = [f"/{op}/run_extreme.py", dir_str]
    op_tokens = [f"--operator {op}", f"--operator={op}"]
    if operator_name:
        op_tokens.extend([f"--operator {operator_name}", f"--operator={operator_name}"])
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (1, my_pid):
            continue
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
            )
        except OSError:
            continue
        if "wait_next_round" in cmd or "watch" in cmd:
            continue
        if any(needle in cmd for needle in needles):
            return True
        is_cli = "tilefoundry_ascend.agent.campaign" in cmd or "tilefoundry-ascend-marathon" in cmd
        if is_cli and any(tok in cmd for tok in op_tokens):
            return True
    return False


def _anchor_comparison(ckpt: Mapping[str, object]) -> list[dict[str, object]]:
    base_anc = {
        int(a["case_num"]): a
        for a in (ckpt.get("baseline_result") or {}).get("anchors", ())
        if isinstance(a, dict) and "case_num" in a
    }
    rows: list[dict[str, object]] = []
    for a in (ckpt.get("best_result") or {}).get("anchors", ()):
        if not isinstance(a, dict) or "case_num" not in a:
            continue
        cnum = int(a["case_num"])
        ba = base_anc.get(cnum, {})
        rows.append(
            {
                "case_num": cnum,
                "regime_tag": a.get("regime_tag", ""),
                "base_us": ba.get("cand_us"),
                "best_us": a.get("cand_us"),
                "base_sp": ba.get("speedup"),
                "best_sp": a.get("speedup"),
                "best_eta": a.get("roofline_eta"),
                "best_eq3": a.get("eq3_score"),
            }
        )
    return rows


def _read_watch_state(op: str, op_dir: Path) -> dict[str, object] | None:
    ckpt_path = op_dir / "checkpoint.json"
    rep_path = op_dir / "report.json"
    if not ckpt_path.is_file():
        return None
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rep = None
    report_done = False
    if rep_path.is_file():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            report_done = isinstance(rep, dict) and "stop_reason" in rep
        except (OSError, ValueError):
            report_done = False

    best = ckpt.get("best_result") or {}
    base = ckpt.get("baseline_result") or {}
    rounds = list(ckpt.get("rounds") or ())
    operator_name = str(ckpt.get("operator_name", op))
    return {
        "op": op,
        "operator_name": operator_name,
        "rounds_len": len(rounds),
        "rounds": rounds,
        "promoted_rounds": [int(x) for x in (ckpt.get("promoted_rounds") or ())],
        "rolled_back_rounds": [int(x) for x in (ckpt.get("rolled_back_rounds") or ())],
        "base_score": round(float(base.get("surrogate_score", 0.0)), 6),
        "best_score": round(float(best.get("surrogate_score", 0.0)), 6),
        "best_speedup": round(float(best.get("avg_speedup", 0.0)), 6),
        "min_case_speedup": round(float(best.get("min_case_speedup", 0.0)), 6),
        "avg_roofline_eta": round(float(best.get("avg_roofline_eta", 0.0)), 6),
        "stagnation_by_regime": dict(ckpt.get("stagnation_by_regime") or {}),
        "frozen_regimes": list(ckpt.get("frozen_regimes") or ()),
        "stop_reason": str(ckpt.get("stop_reason", "in_progress")),
        "report_done": report_done,
        "l4_result": (rep.get("l4_result") if isinstance(rep, dict) else None)
        or ckpt.get("l4_result"),
        "anchors": _anchor_comparison(ckpt),
        "runner_alive": _is_runner_alive(op, op_dir, operator_name=operator_name),
    }


def _snapshot_key(state: Mapping[str, object]) -> tuple:
    return (
        state["rounds_len"],
        tuple(state["promoted_rounds"]),
        state["best_score"],
        state["stop_reason"],
        state["report_done"],
    )


def _prev_key(prev: Mapping[str, object]) -> tuple:
    return (
        int(prev.get("rounds_len", 0)),
        tuple(int(x) for x in (prev.get("promoted_rounds") or ())),
        round(float(prev.get("best_score", 0.0)), 6),
        str(prev.get("stop_reason", "in_progress")),
        bool(prev.get("report_done", False)),
    )


def _tail_log(op_dir: Path, lines: int = 35) -> str:
    log_path = op_dir / "campaign.log"
    if not log_path.is_file():
        return ""
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(all_lines[-lines:])


def _detect_watch_changes(
    state_dirs: Mapping[str, Path],
    last_reported: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    states: dict[str, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    for op, op_dir in state_dirs.items():
        st = _read_watch_state(op, op_dir)
        if st is None:
            continue
        states[op] = st
        prev = last_reported.get(op) or {}
        if not isinstance(prev, Mapping):
            prev = {}
        if _snapshot_key(st) == _prev_key(prev):
            continue

        prev_len = int(prev.get("rounds_len", 0))
        rounds_list = st["rounds"]
        assert isinstance(rounds_list, list)
        if int(st["rounds_len"]) > prev_len:
            for r in rounds_list[prev_len:]:
                ev: dict[str, object] = {
                    "event_type": "round_completed",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "round_idx": r.get("round_idx"),
                    "regime_tag": r.get("regime_tag"),
                    "parent_lane": r.get("parent_lane"),
                    "creative_effort": r.get("creative_effort"),
                    "l0_l1_passed_ids": r.get("l0_l1_passed_ids", []),
                    "l0_l1_failed": r.get("l0_l1_failed", []),
                    "best_round_surrogate_score": r.get("best_round_surrogate_score"),
                    "best_so_far_surrogate_score": r.get("best_so_far_surrogate_score"),
                    "best_round_speedup": r.get("best_round_speedup"),
                    "best_so_far_speedup": r.get("best_so_far_speedup"),
                    "promoted": bool(r.get("promoted", False)),
                    "rolled_back_l3": bool(r.get("rolled_back_l3", False)),
                    "stagnation_by_regime": st["stagnation_by_regime"],
                    "frozen_regimes": st["frozen_regimes"],
                }
                if ev["promoted"]:
                    ev["anchors"] = st["anchors"]
                events.append(ev)
        elif float(st["best_score"]) > round(float(prev.get("best_score", 0.0)), 6) or (
            st["promoted_rounds"] != [int(x) for x in (prev.get("promoted_rounds") or ())]
        ):
            events.append(
                {
                    "event_type": "salvage_promoted",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "best_score": st["best_score"],
                    "best_speedup": st["best_speedup"],
                    "min_case_speedup": st["min_case_speedup"],
                    "avg_roofline_eta": st["avg_roofline_eta"],
                    "promoted_rounds": st["promoted_rounds"],
                    "stagnation_by_regime": st["stagnation_by_regime"],
                    "anchors": st["anchors"],
                }
            )

        if (bool(st["report_done"]) and not bool(prev.get("report_done", False))) or (
            str(st["stop_reason"]) != "in_progress"
            and str(prev.get("stop_reason", "in_progress")) == "in_progress"
        ):
            events.append(
                {
                    "event_type": "campaign_finished",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "stop_reason": st["stop_reason"],
                    "report_done": st["report_done"],
                    "total_rounds": st["rounds_len"],
                    "base_score": st["base_score"],
                    "best_score": st["best_score"],
                    "best_speedup": st["best_speedup"],
                    "min_case_speedup": st["min_case_speedup"],
                    "avg_roofline_eta": st["avg_roofline_eta"],
                    "promoted_rounds": st["promoted_rounds"],
                    "l4_result": st["l4_result"],
                    "anchors": st["anchors"],
                }
            )
    return events, states


def _build_watch_output(
    state_dirs: Mapping[str, Path],
    events: list[dict[str, object]],
    states: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scoreboard = {}
    for op, st in states.items():
        scoreboard[op] = {
            "operator_name": st["operator_name"],
            "rounds_len": st["rounds_len"],
            "promoted_rounds": st["promoted_rounds"],
            "base_score": st["base_score"],
            "best_score": st["best_score"],
            "best_speedup": st["best_speedup"],
            "min_case_speedup": st["min_case_speedup"],
            "avg_roofline_eta": st["avg_roofline_eta"],
            "stagnation_by_regime": st["stagnation_by_regime"],
            "frozen_regimes": st["frozen_regimes"],
            "stop_reason": st["stop_reason"],
            "report_done": st["report_done"],
            "runner_alive": st["runner_alive"],
        }
    both_finished = len(states) == len(state_dirs) and len(states) > 0 and all(
        bool(st["report_done"]) and str(st["stop_reason"]) != "in_progress"
        for st in states.values()
    )
    return {
        "events": events,
        "scoreboard": scoreboard,
        "both_finished": both_finished,
    }


def _persist_watch_states(state_path: Path, states: Mapping[str, Mapping[str, object]]) -> None:
    saved = {
        op: {
            "rounds_len": st["rounds_len"],
            "promoted_rounds": st["promoted_rounds"],
            "best_score": st["best_score"],
            "stop_reason": st["stop_reason"],
            "report_done": st["report_done"],
        }
        for op, st in states.items()
    }
    _atomic_write_json(state_path, saved)


def _setup_inotify(state_dirs: Mapping[str, Path]) -> int | None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            return None
        for op_dir in state_dirs.values():
            if not op_dir.is_dir():
                continue
            wd = libc.inotify_add_watch(fd, str(op_dir).encode("utf-8"), _WATCH_MASK)
            if wd < 0:
                os.close(fd)
                return None
        return fd
    except OSError:
        return None


def watch_marathon_rounds(
    *,
    state_dirs: Mapping[str, Path],
    state_file: Path,
    poll_interval_sec: float = 30.0,
    once: bool = False,
) -> dict[str, object]:
    """Watch checkpoint directories for round completions or unexpected exits."""
    last_reported: dict[str, object] = {}
    if state_file.is_file():
        try:
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                last_reported = loaded
        except (OSError, ValueError):
            last_reported = {}

    events, states = _detect_watch_changes(state_dirs, last_reported)
    if events or once:
        _persist_watch_states(state_file, states)
        return _build_watch_output(state_dirs, events, states)

    fd = _setup_inotify(state_dirs)
    dead_streak = {op: 0 for op in state_dirs}
    try:
        while True:
            if fd is not None:
                readable, _, _ = select.select([fd], [], [], poll_interval_sec)
                if readable:
                    while True:
                        try:
                            chunk = os.read(fd, 4096)
                            if not chunk:
                                break
                        except BlockingIOError:
                            break
            else:
                time.sleep(poll_interval_sec)

            events, states = _detect_watch_changes(state_dirs, last_reported)
            if events:
                _persist_watch_states(state_file, states)
                return _build_watch_output(state_dirs, events, states)

            unexpected: list[dict[str, object]] = []
            for op, st in states.items():
                still_running = st["stop_reason"] == "in_progress" and not st["report_done"]
                if still_running and not st["runner_alive"]:
                    dead_streak[op] += 1
                    if dead_streak[op] >= 2:
                        unexpected.append(
                            {
                                "event_type": "runner_exited_unexpectedly",
                                "op": op,
                                "operator_name": st["operator_name"],
                                "rounds_len": st["rounds_len"],
                                "best_score": st["best_score"],
                                "log_tail": _tail_log(state_dirs[op], 40),
                            }
                        )
                else:
                    dead_streak[op] = 0
            if unexpected:
                return _build_watch_output(state_dirs, unexpected, states)
    finally:
        if fd is not None:
            os.close(fd)


def _discover_watch_state_dirs(
    base_dir: Path | None,
    explicit_pairs: Sequence[str],
) -> tuple[dict[str, Path], Path]:
    if explicit_pairs:
        resolved: dict[str, Path] = {}
        first_parent: Path | None = None
        for item in explicit_pairs:
            if "=" in item:
                label, raw_dir = item.split("=", 1)
            else:
                p = Path(item).expanduser().resolve()
                label, raw_dir = p.name, str(p)
            p = Path(raw_dir).expanduser().resolve()
            resolved[label.strip()] = p
            if first_parent is None:
                first_parent = p.parent
        assert first_parent is not None
        anchor = Path(base_dir).expanduser().resolve() if base_dir else first_parent
        return resolved, anchor / "last_reported.json"

    anchor = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else (default_source_root() / ".tilefoundry-marathon").resolve()
    )
    resolved = {}
    if anchor.is_dir():
        for child in sorted(anchor.iterdir()):
            if child.is_dir() and (child / "checkpoint.json").is_file():
                resolved[child.name] = child
    if not resolved and anchor.is_dir():
        for child in sorted(anchor.iterdir()):
            if child.is_dir():
                resolved[child.name] = child
    return resolved, anchor / "last_reported.json"


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator",
        required=True,
        help="Target operator to optimize (short name e.g. rms_norm or task id e.g. level2/rms_norm).",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help=(
            "Authorized NPU device ID; defaults to the first card declared by "
            "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS, else 1."
        ),
    )
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=(),
        help=(
            "Extra explicitly authorized NPU cards for whole-L4 attempts, "
            "comma-separated; a blocked-env preemption moves to the next "
            "untried pool card."
        ),
    )
    parser.add_argument(
        "--retry-device-id",
        default="",
        help=(
            "A second explicitly authorized NPU card for at most one fresh "
            "retry after a proven-terminal infrastructure failure."
        ),
    )
    parser.add_argument("--source-root", default=None, help="TileFoundry repository root.")
    parser.add_argument("--worktree", default=None, help="Optional isolated worktree root.")
    parser.add_argument(
        "--isolate-worktree",
        action="store_true",
        help="Create or reuse an isolated sibling git worktree automatically.",
    )
    parser.add_argument("--state-dir", default=None, help="Directory for checkpoint/logs/env.")
    parser.add_argument("--target-file", default=None, help="Override target kernels.py path.")
    parser.add_argument("--env-file", default=None, help="Optional pre-existing env.sh path.")
    _add_llm_arguments(parser)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint JSON path.")
    parser.add_argument("--report-json", default=None, help="Override final report JSON path.")
    parser.add_argument("--campaign-kb", default=None, help="Override distilled KB YAML path.")
    parser.add_argument("--log-file", default=None, help="Optional file path to tee runner logs.")
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument(
        "--round-regime",
        action="append",
        default=None,
        metavar="TAG",
        help=(
            "Force the regime of the next round; repeat in round order. "
            "Later rounds keep adaptive regime rotation."
        ),
    )
    parser.add_argument("--max-wall-clock-min", type=float, default=720.0)
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument("--no-final-l4", action="store_true", help="Skip final L4 CANNBench gate.")
    parser.add_argument("--fresh", action="store_true", help="Ignore prior checkpoint and restart.")
    parser.add_argument("--no-salvage", action="store_true", help="Disable beam salvage check.")
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Force bounded Stage-0 cold-start seed synthesis over an existing kernels.py.",
    )
    parser.add_argument(
        "--pure-golden",
        dest="pure_golden",
        action="store_true",
        default=None,
        help="Isolate synthesis from authored HIR, DEFAULT_REGIME_SPECS, and repo incumbents.",
    )
    parser.add_argument(
        "--no-pure-golden",
        dest="pure_golden",
        action="store_false",
        help="Disable pure-golden isolation even when --cold-start is set.",
    )
    parser.add_argument(
        "--experience-lake-dir",
        default=None,
        help="Override experience lake directory for Bayesian tracker and distilled rules.",
    )
    parser.add_argument(
        "--enable-tilesize-prescreen",
        dest="enable_tilesize_prescreen",
        action="store_true",
        default=True,
        help="Enable tile-size prescreen guidance table in creative prompts (default: enabled).",
    )
    parser.add_argument(
        "--disable-tilesize-prescreen",
        dest="enable_tilesize_prescreen",
        action="store_false",
        help="Disable tile-size prescreen guidance table in creative prompts.",
    )
    parser.add_argument(
        "--cost-prescreen-top-k",
        type=int,
        default=None,
        help="Keep only the top-K cost-model candidates during creative prescreen.",
    )


def build_marathon_arg_parser() -> argparse.ArgumentParser:
    """Construct standalone CLI parser for zero-config marathon runs and watching."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-marathon",
        description="Zero-config extreme score-sprint runner and round watcher.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run or auto-resume a score-sprint campaign.")
    _add_run_arguments(run_parser)

    watch_parser = sub.add_parser("watch", help="Watch marathon rounds and emit JSON events.")
    watch_parser.add_argument("--base-dir", default=None, help="Base directory containing runs.")
    watch_parser.add_argument(
        "--state-dir",
        action="append",
        default=[],
        help="Explicit label=path pair to watch (repeatable).",
    )
    watch_parser.add_argument("--state-file", default=None, help="Override last_reported JSON.")
    watch_parser.add_argument("--poll-interval-sec", type=float, default=30.0)
    watch_parser.add_argument("--once", action="store_true", help="Snapshot once and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute marathon CLI subcommands (defaults to 'run' when --operator is given)."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] not in ("run", "watch", "-h", "--help"):
        raw_argv = ["run", *raw_argv]
    args = build_marathon_arg_parser().parse_args(raw_argv)

    if args.command == "watch":
        state_dirs, default_state_file = _discover_watch_state_dirs(
            Path(args.base_dir) if args.base_dir else None,
            args.state_dir,
        )
        state_file = (
            Path(args.state_file).expanduser().resolve()
            if args.state_file
            else default_state_file
        )
        output = watch_marathon_rounds(
            state_dirs=state_dirs,
            state_file=state_file,
            poll_interval_sec=float(args.poll_interval_sec),
            once=bool(args.once),
        )
        print(json.dumps(output, indent=2), flush=True)
        return 0

    pure_golden = bool(args.cold_start) if args.pure_golden is None else bool(args.pure_golden)
    cfg = MarathonConfig(
        operator_name=str(args.operator),
        device_id=str(args.device_id),
        retry_device_id=str(args.retry_device_id),
        device_ids=tuple(args.device_ids),
        source_root=Path(args.source_root) if args.source_root else None,
        worktree=Path(args.worktree) if args.worktree else None,
        isolate_worktree=bool(args.isolate_worktree),
        state_dir=Path(args.state_dir) if args.state_dir else None,
        target_module_path=Path(args.target_file) if args.target_file else None,
        env_file=Path(args.env_file) if args.env_file else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        report_path=Path(args.report_json) if args.report_json else None,
        campaign_kb_path=Path(args.campaign_kb) if args.campaign_kb else None,
        experience_lake_dir=(
            Path(args.experience_lake_dir) if args.experience_lake_dir else None
        ),
        log_path=Path(args.log_file) if args.log_file else None,
        max_rounds=int(args.max_rounds),
        round_regime_tags=tuple(args.round_regime or ()),
        max_wall_clock_min=float(args.max_wall_clock_min),
        k_rule_workers=int(args.k_rule_workers),
        run_final_l4=not bool(args.no_final_l4),
        resume=False if args.fresh else None,
        salvage_beam_candidates=not bool(args.no_salvage),
        cold_start=bool(args.cold_start),
        pure_golden=pure_golden,
        enable_tilesize_prescreen=bool(args.enable_tilesize_prescreen),
        cost_prescreen_top_k=(
            int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None
        ),
        llm_backend=args.llm_backend,
        exploit_model=args.exploit_model,
        explore_model=args.explore_model,
        escalate_model=args.escalate_model,
        llm_api_base=args.llm_api_base,
    )
    return run_operator_marathon(cfg)


__all__ = [
    "ExtremeCampaignRunner",
    "FastCreativeAgyBackend",
    "MarathonConfig",
    "atomic_write_text",
    "build_marathon_arg_parser",
    "build_prewarmed_l3_runner",
    "build_resilient_l2_evaluator",
    "build_stage0_oracle_validator",
    "build_stage0_static_validator",
    "configure_marathon_process_env",
    "discover_marathon_environment",
    "ensure_marathon_env_file",
    "main",
    "prepare_marathon_seed",
    "run_operator_marathon",
    "salvage_unpromoted_beam_winner",
    "short_operator_tag",
    "watch_marathon_rounds",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
