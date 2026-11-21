"""Offline end-to-end tests for WP4 cold-start seed synthesis and universal blueprints."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent import (
    AnchorCaseSpec,
    CampaignConfig,
    MockLLMBackend,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign import campaign_runner as campaign_runner_module
from tilefoundry_ascend.agent.campaign import marathon
from tilefoundry_ascend.agent.campaign import seed_synthesizer as seed_synthesizer_module
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    CannbenchOpsRegistry,
    build_semantic_regime_specs,
    campaign_case_regime_tag,
    extract_trackb_cpp_source,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignReport,
    CampaignRoundRecord,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    build_arg_parser as build_campaign_arg_parser,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    build_marathon_arg_parser,
    build_stage0_static_validator,
    prepare_marathon_seed,
    run_operator_marathon,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    SeedSynthesisOutcome,
    prepare_operator_seed_or_takeover,
    select_initial_track_from_semantics,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L3AcceptanceResult,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
)
from tilefoundry_ascend.agent.router.capability_gate import RouteDecision, StructuralFeatures
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger, ScopeWaterline
from tilefoundry_ascend.ascend.sil import CasePhysicsRatios
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

_EXP_GOLDEN = """
import torch

def exp(x, base=-1.0, scale=1.0, shift=0.0):
    temp = scale * x.to(torch.float32) + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base))
    return torch.exp(temp).to(x.dtype)
"""

_GATHER_GOLDEN = """
import torch

def gather(x, index):
    return torch.gather(x, dim=1, index=index)
"""

_WIDE_REDUCE_GOLDEN = """
import torch

def reduce_sum(x):
    return torch.sum(x, dim=1)
"""

_EXP_CASES: tuple[dict[str, object], ...] = (
    {
        "case_id": 1,
        "input_shape": [[1024, 1024]],
        "dtype": ["float16"],
        "attrs": {"base": -1.0, "scale": 1.0, "shift": 0.0},
    },
    {
        "case_id": 2,
        "input_shape": [[8192, 8192]],
        "dtype": ["float32"],
        "attrs": {"base": -1.0, "scale": 1.5, "shift": 0.0},
    },
    {
        "case_id": 3,
        "input_shape": [[1023, 1023]],
        "dtype": ["bfloat16"],
        "attrs": {"base": 2.0, "scale": 1.0, "shift": 0.0},
    },
    {
        "case_id": 4,
        "input_shape": [[2, 3, 17, 1024, 101]],
        "dtype": ["float16"],
        "attrs": {"base": 3.0, "scale": 0.25, "shift": -2.0},
    },
)

_EXP_PROTO: dict[str, object] = {
    "name": "Exp",
    "inputs": [{"name": "x", "dtype": ["float16", "float32", "bfloat16"]}],
    "outputs": [{"name": "y", "dtype": ["float16"]}],
    "schema": "exp(Tensor x, float base=-1.0, float scale=1.0, float shift=0.0) -> Tensor y",
}

_EXP_SEMANTIC_REGIME_TAGS = {"vector", "vector_twopass"}

_SEED_MODULE = '''
"""Synthesized Ascend 950PR exp kernel."""

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _select_exp_config(
    M: int = 1024,
    D: int = 1024,
    tl_dtype: str = "float16",
) -> dict[str, object]:
    """Select the elementwise tile plan for one regime."""
    return {"tile_m": 64, "pass_configs": {}, "compile_flags": []}


def _build_exp_prim_func(
    M: int = 1024,
    D: int = 1024,
    tile_m: int = 64,
    tl_dtype: str = "float16",
):
    """Build the elementwise exp @T.prim_func for one regime."""
    import tilelang.ascend.language as T

    @T.prim_func
    def main(X: T.Tensor((M, D), tl_dtype), Y: T.Tensor((M, D), tl_dtype)):
        with T.Kernel(1) as _cid:
            for i, j in T.Parallel(M, D):
                Y[i, j] = T.exp(X[i, j])

    return main


def _get_kernel(
    M: int = 1024,
    D: int = 1024,
    tl_dtype: str = "float16",
):
    """Compile and cache the exp kernel for one regime."""
    import tilelang

    key = (M, D, tl_dtype)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        cfg = _select_exp_config(M, D, tl_dtype)
        prog = _build_exp_prim_func(M, D, int(cfg["tile_m"]), tl_dtype)
        compiled = tilelang.compile(
            prog,
            target="ascend",
            pass_configs=cfg["pass_configs"] or None,
            compile_flags=cfg["compile_flags"] or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    """Compute elementwise exp on NPU tensors through the compiled kernel."""
    flat = x.reshape(-1, x.shape[-1])
    out = _get_kernel(int(flat.shape[0]), int(flat.shape[1]), "float16")(flat)
    return out.view(x.shape)


kernel = exp
'''

_TRACKB_SEED_MODULE = '''
"""Synthesized Track-B CCE C++20 gather kernel."""

EXP_TRACKB_CU = """
#include <tl_templates/ascend/kernel_entry.h>
__global__ void gather_kernel() {}
"""


def _select_exp_config(M: int = 1024, D: int = 1024, tl_dtype: str = "float16") -> dict[str, object]:
    """Select the Track-B translation unit for one regime."""
    return {"pass_configs": {}, "compile_flags": [], "trackb_symbol": "EXP_TRACKB_CU"}


def exp(x, index):
    """Dispatch gather through the bisheng-compiled Track-B library."""
    raise RuntimeError("offline test fixture has no compiled Track-B library")


kernel = exp
'''

_INCUMBENT_MODULE = """
import torch

def exp(x):
    return x

kernel = exp
"""


class _ScriptedBackend:
    """Deterministic async LLM backend replaying a fixed reply sequence."""

    def __init__(self, replies: Sequence[str | TimeoutError]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> str:
        self.requests.append(req)
        reply = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        if isinstance(reply, TimeoutError):
            raise reply
        return reply


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _accept_oracle(module_source: str) -> OracleAccuracyResult:
    return OracleAccuracyResult(
        passed=True,
        candidate_sha256=_sha256(module_source),
        l3_run_id="run-stage0-0001",
        l3_receipt_path="/tmp/tilefoundry-l3-receipt.json",
        l3_state="passed",
        elapsed_sec=1.5,
    )


def _always_accept(source: str, track: str) -> tuple[bool, str]:
    return True, ""


def _accept_any_stage0(
    regime_specs: Sequence[RegimeTargetSpec],
) -> Callable[[str, str], tuple[bool, str]]:
    return _always_accept


def _legacy_regime_specs() -> tuple[RegimeTargetSpec, ...]:
    return (
        RegimeTargetSpec(
            regime_tag="vector",
            builder_candidates=("_build_exp_prim_func",),
            sample_builder_kwargs={"M": 1024, "D": 1024},
            description="legacy vector regime",
            representative_hir_dims={"M": 1024, "D": 1024},
            representative_anchor_dims={"M": 1024, "D": 1024},
            config_selector_fn="_select_exp_config",
            editable_symbols=("_select_exp_config", "_build_exp_prim_func"),
            allowed_helper_prefix="_exp_vector_",
        ),
        RegimeTargetSpec(
            regime_tag="vector_twopass",
            builder_candidates=("_build_exp_prim_func",),
            sample_builder_kwargs={"M": 8192, "D": 8192},
            description="legacy vector_twopass regime",
            representative_hir_dims={"M": 8192, "D": 8192},
            representative_anchor_dims={"M": 8192, "D": 8192},
            config_selector_fn="_select_exp_config",
            editable_symbols=("_select_exp_config", "_build_exp_prim_func"),
            allowed_helper_prefix="_exp_vector_twopass_",
        ),
    )


def _prepare(
    *,
    golden_source: str,
    backend: KernelLLMBackend,
    cases: Sequence[Mapping[str, object]] = _EXP_CASES,
    existing_module_path: Path | None = None,
    force_cold_start: bool = False,
    max_seed_attempts: int = 4,
    l0_l1_validator_factory: Callable[
        [tuple[RegimeTargetSpec, ...]], Callable[[str, str], tuple[bool, str]]
    ] = _accept_any_stage0,
    oracle_accuracy_validator: Callable[[str], OracleAccuracyResult] = _accept_oracle,
) -> SeedSynthesisOutcome:
    return asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="exp",
            entrypoint="exp",
            pytorch_golden_source=golden_source,
            proto=_EXP_PROTO,
            cases=cases,
            legacy_regime_specs=_legacy_regime_specs(),
            llm_backend=backend,
            l0_l1_validator_factory=l0_l1_validator_factory,
            oracle_accuracy_validator=oracle_accuracy_validator,
            existing_module_path=existing_module_path,
            force_cold_start=force_cold_start,
            max_seed_attempts=max_seed_attempts,
        )
    )


def _write_fixture_repo(root: Path, *, include_golden: bool = True) -> Path:
    op_dir = root / "tasks" / "level1" / "exp"
    op_dir.mkdir(parents=True)
    if include_golden:
        (op_dir / "golden.py").write_text(_EXP_GOLDEN, encoding="utf-8")
    (op_dir / "cases.yaml").write_text(
        yaml.safe_dump({"cases": list(_EXP_CASES)}),
        encoding="utf-8",
    )
    (op_dir / "proto.yaml").write_text(
        yaml.safe_dump({"operator": _EXP_PROTO}),
        encoding="utf-8",
    )
    return root


def _target_module(root: Path) -> Path:
    return root / "examples" / "ascend950pr_ops" / "exp" / "kernels.py"


def _install_accepting_gates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    oracle_passed: bool = True,
    oracle_error: str = "accuracy=0.912",
) -> list[str]:
    oracle_calls: list[str] = []

    def _static_factory(
        regime_specs: Sequence[RegimeTargetSpec],
        *,
        entrypoint: str,
        worktree: Path,
        env_file: Path | None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        assert regime_specs
        assert entrypoint == "exp"

        def _validate(module_source: str, track: str) -> tuple[bool, str]:
            return True, ""

        return _validate

    def _oracle_factory(
        *,
        operator_name: str,
        staging_dir: Path,
        worktree: Path,
        device_id: str,
        op_logger: logging.Logger,
    ) -> Callable[[str], OracleAccuracyResult]:
        def _validate(module_source: str) -> OracleAccuracyResult:
            oracle_calls.append(module_source)
            if not oracle_passed:
                return OracleAccuracyResult(passed=False, error_message=oracle_error)
            return OracleAccuracyResult(
                passed=True,
                candidate_sha256=_sha256(module_source),
                l3_run_id="run-stage0-0001",
                l3_receipt_path=str(staging_dir / "receipt.json"),
                l3_state="passed",
                elapsed_sec=1.5,
            )

        return _validate

    monkeypatch.setattr(marathon, "build_stage0_static_validator", _static_factory)
    monkeypatch.setattr(marathon, "build_stage0_oracle_validator", _oracle_factory)
    return oracle_calls


def test_select_initial_track_prefers_trackb_for_indirect_index() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_GATHER_GOLDEN, entry_name="gather")
    ratios = tuple(analyzer.compute_case_ratios(semantics, c) for c in _EXP_CASES[:1])
    assert semantics.has_indirect_index is True
    assert select_initial_track_from_semantics(semantics, ratios) == "cce_trackb"


def test_select_initial_track_prefers_trackb_for_wide_stride_reduction() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_WIDE_REDUCE_GOLDEN, entry_name="reduce_sum")
    case = {
        "case_id": 1,
        "input_shape": [[1, 1024, 8192]],
        "dtype": ["float16"],
        "attrs": {"dim": 1},
    }
    ratios = (analyzer.compute_case_ratios(semantics, case),)
    assert ratios[0].stride_ratio > 1.0
    assert ratios[0].working_set_sram_ratio > 1.0
    assert select_initial_track_from_semantics(semantics, ratios) == "cce_trackb"


def test_select_initial_track_keeps_tilelang_for_elementwise_exp() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_EXP_GOLDEN, entry_name="exp")
    ratios = tuple(analyzer.compute_case_ratios(semantics, c) for c in _EXP_CASES)
    assert select_initial_track_from_semantics(semantics, ratios) == "tilelang_dsl"


def test_prepare_seed_cold_start_synthesizes_validated_module() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    factory_regime_tags: list[set[str]] = []

    def _recording_factory(
        regime_specs: Sequence[RegimeTargetSpec],
    ) -> Callable[[str, str], tuple[bool, str]]:
        factory_regime_tags.append({regime.regime_tag for regime in regime_specs})
        return _always_accept

    outcome = _prepare(
        golden_source=_EXP_GOLDEN,
        backend=backend,
        l0_l1_validator_factory=_recording_factory,
    )
    assert outcome.mode == "cold_start"
    assert outcome.initial_track == "tilelang_dsl"
    assert outcome.module_source.strip() == _SEED_MODULE.strip()
    assert outcome.attempts_used == 1
    assert outcome.verification_state == "l3_verified"
    assert outcome.l0_l1_passed is True
    assert outcome.oracle_accuracy_passed is True
    assert outcome.candidate_sha256 == _sha256(outcome.module_source)
    assert factory_regime_tags == [_EXP_SEMANTIC_REGIME_TAGS]
    assert outcome.l3_run_id == "run-stage0-0001"
    assert outcome.l3_receipt_path == "/tmp/tilefoundry-l3-receipt.json"
    assert len(backend.requests) == 1
    assert backend.requests[0].role == "explore_creative"
    assert "PYTORCH MATHEMATICAL REFERENCE" in backend.requests[0].user_prompt
    assert "dialect=tilelang_dsl" in backend.requests[0].user_prompt
    assert "ideal_roofline_us:" in backend.requests[0].user_prompt


def test_prepare_seed_trackb_uses_cce_cpp20_surface() -> None:
    backend = _ScriptedBackend([_TRACKB_SEED_MODULE])
    outcome = _prepare(golden_source=_GATHER_GOLDEN, backend=backend)
    assert outcome.initial_track == "cce_trackb"
    assert "dialect=cce_cpp20" in backend.requests[0].user_prompt
    assert "CCE C++20 (tl_templates/ascend)" in backend.requests[0].system_prompt
    assert outcome.regime_specs
    assert all(regime.runtime_backend == "ascendc_trackb" for regime in outcome.regime_specs)
    assert all(regime.trackb_symbols == ("EXP_TRACKB_CU",) for regime in outcome.regime_specs)


def test_seed_prompt_declares_campaign_module_abi() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    _prepare(golden_source=_EXP_GOLDEN, backend=backend)
    system_prompt = backend.requests[0].system_prompt
    for symbol in (
        "_select_exp_config(",
        "_build_exp_prim_func(",
        "_get_kernel(",
        'tilelang.compile(prog, target="ascend")',
        "kernel = exp",
    ):
        assert symbol in system_prompt
    assert "M: int = 8192" in system_prompt
    assert "D: int = 101" in system_prompt
    assert "base: float = 3.0" in system_prompt
    assert "tl_dtype: str = 'float16'" in system_prompt


def test_seed_prompt_carries_official_contract_and_regime_instantiations() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    _prepare(golden_source=_EXP_GOLDEN, backend=backend)
    user_prompt = backend.requests[0].user_prompt
    assert "### OFFICIAL PROTO CONTRACT" in user_prompt
    assert "### OFFICIAL CASES" in user_prompt
    assert "### CAMPAIGN REGIME INSTANTIATIONS" in user_prompt
    for tag in _EXP_SEMANTIC_REGIME_TAGS:
        assert f"regime `{tag}`" in user_prompt
        assert "_select_exp_config(" in user_prompt
    assert '"bfloat16"' in user_prompt
    assert '"base"' in user_prompt


def test_seed_prompt_trackb_requires_translation_unit_constant() -> None:
    backend = _ScriptedBackend([_TRACKB_SEED_MODULE])
    _prepare(golden_source=_GATHER_GOLDEN, backend=backend)
    system_prompt = backend.requests[0].system_prompt
    assert "top-level C++20 translation-unit string constant" in system_prompt
    assert "#include <tl_templates/ascend/" in system_prompt
    assert "EXP_TRACKB_CU" in system_prompt
    assert "_select_exp_config" in system_prompt
    assert "_build_exp_prim_func" in system_prompt
    assert "_get_kernel" in system_prompt
    assert "kernel = exp" in system_prompt
    for forbidden in CCE_CPP20_SURFACE.forbidden_tokens:
        assert forbidden in system_prompt


def test_prepare_seed_incumbent_takeover_skips_generation(tmp_path: Path) -> None:
    incumbent = tmp_path / "kernels.py"
    incumbent.write_text(_INCUMBENT_MODULE, encoding="utf-8")
    backend = _ScriptedBackend([_SEED_MODULE])

    def _forbid_static_factory(
        regime_specs: Sequence[RegimeTargetSpec],
    ) -> Callable[[str, str], tuple[bool, str]]:
        raise AssertionError("incumbent takeover must not build Stage-0 gates")

    def _forbid_oracle(module_source: str) -> OracleAccuracyResult:
        raise AssertionError("incumbent takeover must not run NPU accuracy checks")

    outcome = _prepare(
        golden_source=_EXP_GOLDEN,
        backend=backend,
        existing_module_path=incumbent,
        l0_l1_validator_factory=_forbid_static_factory,
        oracle_accuracy_validator=_forbid_oracle,
    )
    assert outcome.mode == "incumbent_takeover"
    assert outcome.initial_track == "incumbent_detected"
    assert outcome.module_source == _INCUMBENT_MODULE
    assert outcome.attempts_used == 0
    assert outcome.verification_state == "unverified_incumbent"
    assert outcome.l0_l1_passed is False
    assert outcome.oracle_accuracy_passed is False
    assert outcome.candidate_sha256 == _sha256(_INCUMBENT_MODULE)
    assert outcome.l3_run_id == ""
    assert outcome.l3_receipt_path == ""
    assert {r.regime_tag for r in outcome.regime_specs} == _EXP_SEMANTIC_REGIME_TAGS
    assert all(r.runtime_backend == "tilelang_dsl" for r in outcome.regime_specs)
    assert backend.requests == []
    assert incumbent.read_text(encoding="utf-8") == _INCUMBENT_MODULE


def test_prepare_seed_force_cold_start_overrides_incumbent(tmp_path: Path) -> None:
    incumbent = tmp_path / "kernels.py"
    incumbent.write_text(_INCUMBENT_MODULE, encoding="utf-8")
    backend = _ScriptedBackend([_SEED_MODULE])
    outcome = _prepare(
        golden_source=_EXP_GOLDEN,
        backend=backend,
        existing_module_path=incumbent,
        force_cold_start=True,
    )
    assert outcome.mode == "cold_start"
    assert outcome.verification_state == "l3_verified"
    assert outcome.module_source.strip() == _SEED_MODULE.strip()
    assert {r.regime_tag for r in outcome.regime_specs} == _EXP_SEMANTIC_REGIME_TAGS
    assert len(backend.requests) == 1


def test_prepare_seed_bounded_loop_reports_failed_attempts() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    diagnostics: list[str] = []

    def _reject_static(module_source: str, track: str) -> tuple[bool, str]:
        diagnostics.append(track)
        return False, "missing @T.prim_func entry"

    def _reject_factory(
        regime_specs: Sequence[RegimeTargetSpec],
    ) -> Callable[[str, str], tuple[bool, str]]:
        assert {regime.regime_tag for regime in regime_specs} == _EXP_SEMANTIC_REGIME_TAGS
        return _reject_static

    with pytest.raises(RuntimeError, match="after 2 attempts") as excinfo:
        _prepare(
            golden_source=_EXP_GOLDEN,
            backend=backend,
            max_seed_attempts=2,
            l0_l1_validator_factory=_reject_factory,
        )
    assert "missing @T.prim_func entry" in str(excinfo.value)
    assert diagnostics == ["tilelang_dsl", "tilelang_dsl"]
    assert len(backend.requests) == 2
    assert backend.requests[1].role == "syntax_repair"
    prompt = backend.requests[1].user_prompt
    assert "PREVIOUS ATTEMPT FAILURE DIAGNOSTIC" in prompt
    assert "missing @T.prim_func entry" in prompt
    assert "### LAST REJECTED CANDIDATE SOURCE" in prompt
    assert _SEED_MODULE.strip() in prompt
    assert "### LAST REJECTED CANDIDATE DIAGNOSTIC" not in prompt


def test_prepare_seed_retries_transport_timeout_without_consuming_attempt() -> None:
    backend = _ScriptedBackend([TimeoutError("LLM HTTP request timed out"), _SEED_MODULE])
    outcome = _prepare(golden_source=_EXP_GOLDEN, backend=backend)
    assert outcome.attempts_used == 1
    assert outcome.verification_state == "l3_verified"
    assert outcome.module_source.strip() == _SEED_MODULE.strip()
    assert len(backend.requests) == 2
    assert backend.requests[0].role == "explore_creative"
    assert backend.requests[1].role == "explore_creative"
    prompt = backend.requests[1].user_prompt
    assert "LLM generation transport failure: TimeoutError" in prompt
    assert "LLM HTTP request timed out" in prompt
    assert "### LAST REJECTED CANDIDATE SOURCE" not in prompt


def test_prepare_seed_stops_after_transport_retry_budget_exhausted() -> None:
    backend = _ScriptedBackend(
        [
            TimeoutError("LLM HTTP request timed out"),
            TimeoutError("LLM HTTP request timed out"),
        ]
    )
    with pytest.raises(RuntimeError, match="after 2 retries") as excinfo:
        _prepare(golden_source=_EXP_GOLDEN, backend=backend, max_seed_attempts=2)
    message = str(excinfo.value)
    assert "LLM generation transport failure: TimeoutError" in message
    assert "LLM HTTP request timed out" in message
    assert len(backend.requests) == 3


def test_prepare_seed_timeout_after_rejection_keeps_last_candidate() -> None:
    rejected_source = "def exp(x):\n    return x\n\nkernel = exp\n"
    backend = _ScriptedBackend(
        [
            rejected_source,
            TimeoutError("LLM HTTP request timed out"),
            _SEED_MODULE,
        ]
    )

    def _reject_once(module_source: str, track: str) -> tuple[bool, str]:
        if module_source == rejected_source.strip():
            return False, "missing @T.prim_func entry"
        return True, ""

    def _factory(
        regime_specs: Sequence[RegimeTargetSpec],
    ) -> Callable[[str, str], tuple[bool, str]]:
        return _reject_once

    outcome = _prepare(
        golden_source=_EXP_GOLDEN,
        backend=backend,
        l0_l1_validator_factory=_factory,
    )
    assert outcome.attempts_used == 2
    assert outcome.verification_state == "l3_verified"
    prompt = backend.requests[2].user_prompt
    timeout_index = prompt.index("LLM generation transport failure: TimeoutError")
    source_header_index = prompt.index("### LAST REJECTED CANDIDATE SOURCE")
    source_index = prompt.index(rejected_source.strip())
    matched_header_index = prompt.index("### LAST REJECTED CANDIDATE DIAGNOSTIC")
    rejection_index = prompt.index("missing @T.prim_func entry")
    assert (
        timeout_index
        < source_header_index
        < source_index
        < matched_header_index
        < rejection_index
    )
    assert _SEED_MODULE.strip() not in prompt


def test_prepare_seed_retries_after_oracle_mismatch() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    oracle_calls = 0

    def _flaky_oracle(module_source: str) -> OracleAccuracyResult:
        nonlocal oracle_calls
        oracle_calls += 1
        if oracle_calls == 1:
            return OracleAccuracyResult(
                passed=False,
                error_message="max_abs_err=0.0312 exceeds 1e-3",
                l3_state="failed",
            )
        return _accept_oracle(module_source)

    outcome = _prepare(
        golden_source=_EXP_GOLDEN,
        backend=backend,
        oracle_accuracy_validator=_flaky_oracle,
    )
    assert outcome.attempts_used == 2
    assert outcome.oracle_accuracy_passed is True
    assert outcome.l3_run_id == "run-stage0-0001"
    assert len(backend.requests) == 2
    prompt = backend.requests[1].user_prompt
    assert "NPU numerical mismatch" in prompt
    assert "max_abs_err=0.0312" in prompt
    assert "### LAST REJECTED CANDIDATE SOURCE" in prompt
    assert _SEED_MODULE.strip() in prompt
    assert "### LAST REJECTED CANDIDATE DIAGNOSTIC" not in prompt


def test_prepare_seed_rejects_oracle_evidence_for_a_different_candidate() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])

    def _wrong_digest(module_source: str) -> OracleAccuracyResult:
        return OracleAccuracyResult(
            passed=True,
            candidate_sha256="0" * 64,
            l3_run_id="run-stage0-0001",
            l3_receipt_path="/tmp/receipt.json",
            l3_state="passed",
        )

    with pytest.raises(RuntimeError, match="oracle evidence digest mismatch"):
        _prepare(
            golden_source=_EXP_GOLDEN,
            backend=backend,
            oracle_accuracy_validator=_wrong_digest,
        )


def test_prepare_seed_rejects_oracle_success_without_l3_receipt() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])

    def _no_receipt(module_source: str) -> OracleAccuracyResult:
        return OracleAccuracyResult(
            passed=True,
            candidate_sha256=_sha256(module_source),
            l3_run_id="",
            l3_receipt_path="",
            l3_state="passed",
        )

    with pytest.raises(RuntimeError, match="lacks an L3 run_id/receipt"):
        _prepare(
            golden_source=_EXP_GOLDEN,
            backend=backend,
            oracle_accuracy_validator=_no_receipt,
        )


def test_prepare_seed_blueprints_cover_every_semantic_regime() -> None:
    backend = _ScriptedBackend([_SEED_MODULE])
    outcome = _prepare(golden_source=_EXP_GOLDEN, backend=backend)
    assert {r.regime_tag for r in outcome.regime_specs} == _EXP_SEMANTIC_REGIME_TAGS
    assert set(outcome.blueprints_by_regime) == _EXP_SEMANTIC_REGIME_TAGS
    expected_tags = {
        campaign_case_regime_tag(
            ratio,
            next(case for case in _EXP_CASES if case["case_id"] == ratio.case_id),
            base_axes=outcome.semantics.base_axes,
            has_reduction=outcome.semantics.has_reduction,
        )
        for ratio in outcome.case_ratios
    }
    assert expected_tags == _EXP_SEMANTIC_REGIME_TAGS
    assert {r.case_id for r in outcome.case_ratios} == {1, 2, 3, 4}
    for regime in outcome.regime_specs:
        packet = outcome.blueprints_by_regime[regime.regime_tag]
        assert packet.entry_name == "exp"
        assert packet.roofline.min_gm_bytes > 0
        assert packet.roofline.ideal_roofline_us > 0.0
        assert packet.z3_pipeline_summary["source"] == "pytorch_semantic_frontend"
        assert packet.requested_anchor_dims == regime.representative_hir_dims


def test_stage0_static_validator_rejects_missing_public_entrypoint(tmp_path: Path) -> None:
    validator = build_stage0_static_validator(
        _legacy_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    source = "def other(x):\n    return x\n\nkernel = other\n"
    passed, error = validator(source, "tilelang_dsl")
    assert passed is False
    assert "does not define the public entrypoint exp" in error


def test_stage0_static_validator_rejects_detached_kernel_alias(tmp_path: Path) -> None:
    validator = build_stage0_static_validator(
        _legacy_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    source = "def exp(x):\n    return x\n\ndef other(x):\n    return x\n\nkernel = other\n"
    passed, error = validator(source, "tilelang_dsl")
    assert passed is False
    assert "does not bind `kernel = exp`" in error


def test_stage0_static_validator_rejects_outer_torch_fallback(tmp_path: Path) -> None:
    validator = build_stage0_static_validator(
        _legacy_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    source = "import tilelang\n\ndef exp(x, y):\n    return torch.matmul(x, y)\n\nkernel = exp\n"
    passed, error = validator(source, "tilelang_dsl")
    assert passed is False
    assert "entrypoint purity rejection" in error
    assert "npu_host_fallback_forbidden:exp->matmul" in error


def test_stage0_static_validator_rejects_cce_forbidden_tokens(tmp_path: Path) -> None:
    validator = build_stage0_static_validator(
        _legacy_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    source = (
        '_TU = """\n#include <tl_templates/ascend/kernel_entry.h>\n__global__ void k() {}\n"""\n'
        "_DEAD = 'TPipe'\n\n"
        "def exp(x):\n"
        "    return x\n\n"
        "kernel = exp\n"
    )
    passed, error = validator(source, "cce_trackb")
    assert passed is False
    assert "CCE C++20 seed references forbidden tokens" in error
    assert "TPipe=" in error


def test_stage0_oracle_validator_stages_candidate_by_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir = tmp_path / "stage0_candidates"
    captured: dict[str, object] = {}

    def _fake_l3(
        *,
        operator_name: str,
        candidate_module_path: Path,
        source_root: Path,
        device_id: str,
    ) -> L3AcceptanceResult:
        captured["operator_name"] = operator_name
        captured["candidate_module_path"] = candidate_module_path
        captured["source_root"] = source_root
        captured["device_id"] = device_id
        digest = _sha256(candidate_module_path.read_text(encoding="utf-8"))
        return L3AcceptanceResult(
            passed=True,
            run_id="run-stage0-0042",
            receipt_path=str(tmp_path / "receipt.json"),
            elapsed_sec=2.5,
            candidate_sha256=digest,
            state="passed",
        )

    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", _fake_l3)
    validator = marathon.build_stage0_oracle_validator(
        operator_name="exp",
        staging_dir=staging_dir,
        worktree=tmp_path,
        device_id="1",
        op_logger=logging.getLogger("test.wp4.oracle_staging"),
    )
    result = validator(_SEED_MODULE)
    digest = _sha256(_SEED_MODULE)
    staged = staging_dir / f"{digest}.py"
    assert staged.read_text(encoding="utf-8") == _SEED_MODULE
    assert captured["candidate_module_path"] == staged
    assert captured["source_root"] == tmp_path
    assert result.passed is True
    assert result.candidate_sha256 == digest
    assert result.l3_run_id == "run-stage0-0042"
    assert result.l3_receipt_path == str(tmp_path / "receipt.json")


def test_prepare_marathon_seed_cold_start_writes_module_and_package_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    target = _target_module(tmp_path)
    staging_dir = tmp_path / ".tilefoundry-marathon" / "exp" / "stage0_candidates"
    oracle_calls = _install_accepting_gates(monkeypatch)
    backend = _ScriptedBackend([_SEED_MODULE])

    outcome = prepare_marathon_seed(
        operator_name="exp",
        target_module=target,
        staging_dir=staging_dir,
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=repo,
        llm_backend=backend,
        cold_start=False,
        device_id="1",
        op_logger=logging.getLogger("test.wp4.cold_start"),
    )

    assert outcome.mode == "cold_start"
    assert outcome.verification_state == "l3_verified"
    assert outcome.candidate_sha256 == _sha256(outcome.module_source)
    assert _sha256(oracle_calls[0]) == outcome.candidate_sha256
    assert outcome.l3_run_id == "run-stage0-0001"
    assert target.read_text(encoding="utf-8").strip() == _SEED_MODULE.strip()
    init_source = (target.parent / "__init__.py").read_text(encoding="utf-8")
    assert "from .kernels import exp, kernel" in init_source
    assert len(oracle_calls) == 1
    assert oracle_calls[0].strip() == _SEED_MODULE.strip()
    assert {r.regime_tag for r in outcome.regime_specs} == _EXP_SEMANTIC_REGIME_TAGS
    assert set(outcome.blueprints_by_regime) == _EXP_SEMANTIC_REGIME_TAGS


def test_prepare_marathon_seed_cold_start_never_reads_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    target = _target_module(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(_INCUMBENT_MODULE, encoding="utf-8")
    _install_accepting_gates(monkeypatch)
    real_read_text = Path.read_text
    resolved_target = target.resolve()

    def _guarded_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.resolve() == resolved_target:
            raise AssertionError("explicit cold start must not read the incumbent module")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read_text)
    outcome = prepare_marathon_seed(
        operator_name="exp",
        target_module=target,
        staging_dir=tmp_path / "stage0_candidates",
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=repo,
        llm_backend=_ScriptedBackend([_SEED_MODULE]),
        cold_start=True,
        device_id="1",
        op_logger=logging.getLogger("test.wp4.no_incumbent_read"),
    )
    monkeypatch.undo()
    assert outcome.mode == "cold_start"
    assert target.read_text(encoding="utf-8").strip() == _SEED_MODULE.strip()


def test_prepare_marathon_seed_incumbent_keeps_existing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    target = _target_module(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(_INCUMBENT_MODULE, encoding="utf-8")
    oracle_calls = _install_accepting_gates(monkeypatch)
    backend = _ScriptedBackend([_SEED_MODULE])

    outcome = prepare_marathon_seed(
        operator_name="exp",
        target_module=target,
        staging_dir=tmp_path / "stage0_candidates",
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=repo,
        llm_backend=backend,
        cold_start=False,
        device_id="1",
        op_logger=logging.getLogger("test.wp4.incumbent"),
    )

    assert outcome.mode == "incumbent_takeover"
    assert outcome.verification_state == "unverified_incumbent"
    assert outcome.l0_l1_passed is False
    assert outcome.oracle_accuracy_passed is False
    assert outcome.candidate_sha256 == _sha256(_INCUMBENT_MODULE)
    assert target.read_text(encoding="utf-8") == _INCUMBENT_MODULE
    assert not (target.parent / "__init__.py").exists()
    assert backend.requests == []
    assert oracle_calls == []


def test_prepare_marathon_seed_cold_start_keeps_prior_module_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    target = _target_module(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(_INCUMBENT_MODULE, encoding="utf-8")
    _install_accepting_gates(monkeypatch, oracle_passed=False)

    with pytest.raises(RuntimeError, match="Stage-0 cold-start seed synthesis failed"):
        prepare_marathon_seed(
            operator_name="exp",
            target_module=target,
            staging_dir=tmp_path / "stage0_candidates",
            worktree=tmp_path,
            env_file=None,
            cannbench_repo=repo,
            llm_backend=_ScriptedBackend([_SEED_MODULE]),
            cold_start=True,
            device_id="1",
            op_logger=logging.getLogger("test.wp4.keep_prior"),
        )

    assert target.read_text(encoding="utf-8") == _INCUMBENT_MODULE
    assert not (target.parent / "__init__.py").exists()


def test_prepare_marathon_seed_removes_fresh_module_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    target = _target_module(tmp_path)
    _install_accepting_gates(monkeypatch, oracle_passed=False, oracle_error="accuracy=0.871")

    with pytest.raises(RuntimeError, match="Stage-0 cold-start seed synthesis failed"):
        prepare_marathon_seed(
            operator_name="exp",
            target_module=target,
            staging_dir=tmp_path / "stage0_candidates",
            worktree=tmp_path,
            env_file=None,
            cannbench_repo=repo,
            llm_backend=_ScriptedBackend([_SEED_MODULE]),
            cold_start=False,
            device_id="1",
            op_logger=logging.getLogger("test.wp4.fresh_failure"),
        )

    assert not target.exists()
    assert not (target.parent / "__init__.py").exists()


def test_prepare_marathon_seed_requires_golden_reference(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench", include_golden=False)
    with pytest.raises(FileNotFoundError, match="no PyTorch golden reference"):
        prepare_marathon_seed(
            operator_name="exp",
            target_module=_target_module(tmp_path),
            staging_dir=tmp_path / "stage0_candidates",
            worktree=tmp_path,
            env_file=None,
            cannbench_repo=repo,
            llm_backend=_ScriptedBackend([_SEED_MODULE]),
            cold_start=True,
            device_id="1",
            op_logger=logging.getLogger("test.wp4.no_golden"),
        )


def test_extract_trackb_cpp_source_scans_top_level_string_constants() -> None:
    module_source = (
        '"""Track-B seed."""\n\n'
        '_TRACKB_TU = """\n__global__ void exp_kernel() {}\n"""\n'
        '_OTHER = "ignored"\n'
    )
    assert extract_trackb_cpp_source(module_source, ("_TRACKB_TU",)) == (
        "_TRACKB_TU",
        "\n__global__ void exp_kernel() {}\n",
    )
    assert extract_trackb_cpp_source(module_source, ("_MISSING",)) is None
    assert extract_trackb_cpp_source(module_source, ()) is None
    assert extract_trackb_cpp_source('_TRACKB_TU = "   "\n', ("_TRACKB_TU",)) is None


def test_cold_start_seed_module_resolves_campaign_regime_config(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    regimes = CannbenchOpsRegistry(tasks_root=repo).get_regime_specs("exp")
    resolved = resolve_regime_config(_SEED_MODULE, regimes[0], dims={"M": 1024, "D": 1024})
    assert resolved.builder_name == "_build_exp_prim_func"
    assert resolved.selector_name == "_select_exp_config"
    assert resolved.builder_kwargs["tile_m"] == 64


def _exp_blueprint(tmp_path: Path) -> HirBlueprintPacket:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    spec = CannbenchOpsRegistry(tasks_root=repo).resolve("exp")
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(
        spec.golden_path.read_text(encoding="utf-8"),
        entry_name=spec.entrypoint,
        proto=spec.proto,
    )
    return analyzer.build_universal_blueprint(semantics, {"M": 1024, "D": 1024})


def test_campaign_runner_synthesizes_blueprints_without_authored_model(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    registry = CannbenchOpsRegistry(tasks_root=repo)
    spec = registry.resolve("exp")
    regimes = registry.get_regime_specs("exp")
    assert len(regimes) >= 2
    target = _target_module(tmp_path)

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="exp", max_rounds=1),
        target_module_path=target,
        perf_target=PerfTargetSpec("exp", 1.5, 1.2, 0.9),
        anchors=tuple(
            AnchorCaseSpec(spec.task, a.case_num, a.regime_tag, 0.0, 0.0) for a in spec.anchors
        ),
        regime_specs=regimes,
        checkpoint_path=tmp_path / "ckpt.json",
        cannbench_repo=repo,
    )

    assert runner._authored_hir_path is None
    assert set(runner._regime_blueprints) == {regime.regime_tag for regime in regimes}
    for packet in runner._regime_blueprints.values():
        assert packet.entry_name == "exp"
        assert packet.roofline.min_gm_bytes > 0
        assert packet.roofline.ideal_roofline_us > 0.0
        assert packet.z3_pipeline_summary["source"] == "pytorch_semantic_frontend"


def test_campaign_runner_raises_when_official_golden_is_missing(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench", include_golden=False)
    with pytest.raises(FileNotFoundError, match="golden reference"):
        PerfTargetCampaignRunner(
            config=CampaignConfig(operator_name="exp", max_rounds=1),
            target_module_path=_target_module(tmp_path),
            perf_target=PerfTargetSpec("exp", 1.5, 1.2, 0.9),
            regime_specs=_legacy_regime_specs(),
            checkpoint_path=tmp_path / "ckpt.json",
            cannbench_repo=repo,
        )


def test_campaign_runner_raises_on_authored_hir_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    authored_hir = tmp_path / "model.py"
    authored_hir.write_text("# authored HIR\n", encoding="utf-8")
    resolved_authored = authored_hir.resolve()
    real_read_text = Path.read_text

    def _guarded_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.resolve() == resolved_authored:
            raise OSError("permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read_text)
    with pytest.raises(RuntimeError, match="cannot read authored HIR source"):
        PerfTargetCampaignRunner(
            config=CampaignConfig(operator_name="exp", max_rounds=1),
            target_module_path=_target_module(tmp_path),
            perf_target=PerfTargetSpec("exp", 1.5, 1.2, 0.9),
            regime_specs=_legacy_regime_specs(),
            authored_hir_path=authored_hir,
            checkpoint_path=tmp_path / "ckpt.json",
            cannbench_repo=repo,
        )


def test_campaign_runner_raises_when_one_regime_blueprint_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    authored_hir = tmp_path / "model.py"
    authored_hir.write_text("# authored HIR\n", encoding="utf-8")

    def _fail_compile(*args: object, **kwargs: object) -> HirBlueprintPacket:
        raise ValueError("unsupported operator construct")

    monkeypatch.setattr(campaign_runner_module, "compile_hir_blueprint", _fail_compile)
    with pytest.raises(
        RuntimeError, match="failed to compile authored HIR blueprint for regime vector"
    ):
        PerfTargetCampaignRunner(
            config=CampaignConfig(operator_name="exp", max_rounds=1),
            target_module_path=_target_module(tmp_path),
            perf_target=PerfTargetSpec("exp", 1.5, 1.2, 0.9),
            regime_specs=_legacy_regime_specs(),
            authored_hir_path=authored_hir,
            checkpoint_path=tmp_path / "ckpt.json",
            cannbench_repo=repo,
        )


def test_layer3_roofline_reaches_both_track_prompts(tmp_path: Path) -> None:
    packet = _exp_blueprint(tmp_path)
    for tier in ("tilelang_dsl", "cce_cpp20"):
        prompt = build_progressive_llm_constraints(escalation_tier=tier, blueprint=packet)
        assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in prompt
        assert f"entry: {packet.entry_name}" in prompt
        assert f"ideal_roofline_us: {packet.roofline.ideal_roofline_us!r}" in prompt
    sub_us_packet = replace(
        packet,
        roofline=replace(packet.roofline, ideal_roofline_us=1.7e-05),
    )
    sub_us_prompt = build_progressive_llm_constraints(
        escalation_tier="tilelang_dsl",
        blueprint=sub_us_packet,
    )
    assert "ideal_roofline_us: 1.7e-05" in sub_us_prompt
    cce_prompt = build_progressive_llm_constraints(
        escalation_tier="cce_cpp20",
        blueprint=packet,
    )
    assert "CCE C++20 (tl_templates/ascend)" in cce_prompt


def test_marathon_cli_exposes_cold_start_prescreen_and_target_file_flags() -> None:
    parser = build_marathon_arg_parser()
    args = parser.parse_args(
        [
            "run",
            "--operator",
            "exp",
            "--cold-start",
            "--enable-tilesize-prescreen",
            "--cost-prescreen-top-k",
            "5",
            "--target-file",
            "/tmp/exp-kernels.py",
        ]
    )
    assert args.cold_start is True
    assert args.enable_tilesize_prescreen is True
    assert args.cost_prescreen_top_k == 5
    assert args.target_file == "/tmp/exp-kernels.py"

    defaults = parser.parse_args(["run", "--operator", "exp"])
    assert defaults.cold_start is False
    assert defaults.enable_tilesize_prescreen is False
    assert defaults.cost_prescreen_top_k is None
    assert defaults.target_file is None


def test_campaign_runner_cli_exposes_prescreen_top_k() -> None:
    args = build_campaign_arg_parser().parse_args(
        ["--operator", "exp", "--enable-tilesize-prescreen", "--cost-prescreen-top-k", "7"]
    )
    assert args.enable_tilesize_prescreen is True
    assert args.cost_prescreen_top_k == 7


def _real_cannbench_repo() -> Path | None:
    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    candidates = (
        Path(env_repo).expanduser() if env_repo else None,
        Path.home() / "workspace" / "cann-bench",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "tasks" / "level1").is_dir():
            return candidate
    return None


@pytest.mark.skipif(
    _real_cannbench_repo() is None,
    reason="real cann-bench tasks repository is not available",
)
def test_every_level12_task_generates_every_semantic_regime_blueprint() -> None:
    repo = _real_cannbench_repo()
    assert repo is not None
    registry = CannbenchOpsRegistry(tasks_root=repo)
    analyzer = PyTorchSemanticAnalyzer()
    checked = 0
    for operator_name in registry.list_operators():
        spec = registry.resolve(operator_name)
        if spec.level_num > 2:
            continue
        semantics = analyzer.analyze_source(
            spec.golden_path.read_text(encoding="utf-8"),
            entry_name=spec.entrypoint,
            proto=spec.proto,
        )
        ratios = tuple(analyzer.compute_case_ratios(semantics, case) for case in spec.cases)
        regimes = build_semantic_regime_specs(
            operator_name=operator_name,
            entrypoint=spec.entrypoint,
            cases=spec.cases,
            proto=spec.proto,
            base_axes=semantics.base_axes,
            has_reduction=semantics.has_reduction,
            case_ratios=ratios,
            runtime_backend="tilelang_dsl",
        )
        expected_tags = {
            campaign_case_regime_tag(
                ratio,
                case,
                base_axes=semantics.base_axes,
                has_reduction=semantics.has_reduction,
            )
            for ratio, case in zip(ratios, spec.cases)
        }
        actual_tags = {regime.regime_tag for regime in regimes}
        assert 1 <= len(regimes) <= 4, operator_name
        if len(expected_tags) == 1 and len(spec.cases) >= 2:
            base_tag = next(iter(expected_tags))
            assert len(regimes) == 2, operator_name
            assert all(tag == base_tag or tag.startswith(f"{base_tag}_") for tag in actual_tags), (
                operator_name
            )
        else:
            assert actual_tags == expected_tags, operator_name
        for regime in regimes:
            packet = analyzer.build_universal_blueprint(
                semantics,
                regime.representative_hir_dims,
            )
            assert packet.entry_name == spec.entrypoint, (operator_name, regime.regime_tag)
            assert packet.roofline.ideal_roofline_us > 0.0, (operator_name, regime.regime_tag)
            assert packet.z3_pipeline_summary["source"] == "pytorch_semantic_frontend", (
                operator_name,
                regime.regime_tag,
            )
        checked += 1
    assert checked == 15


@pytest.mark.skipif(
    _real_cannbench_repo() is None,
    reason="real cann-bench tasks repository is not available",
)
def test_fifteen_production_operators_use_capped_wp1_campaign_partitions() -> None:
    repo = _real_cannbench_repo()
    assert repo is not None
    registry = CannbenchOpsRegistry(tasks_root=repo)
    analyzer = PyTorchSemanticAnalyzer()
    examples = Path(__file__).resolve().parents[3] / "examples" / "ascend950pr_ops"
    operators = sorted(path.parent.name for path in examples.glob("*/kernels.py"))
    assert len(operators) == 15
    expected_partitions = {
        "exp": 2,
        "mla": 4,
        "q4_0_dequant_matmul": 3,
        "softmax": 3,
    }
    checked = 0
    for operator_name in operators:
        spec = registry.resolve(operator_name)
        semantics = analyzer.analyze_source(
            spec.golden_path.read_text(encoding="utf-8"),
            entry_name=spec.entrypoint,
            proto=spec.proto,
        )
        ratios = tuple(analyzer.compute_case_ratios(semantics, case) for case in spec.cases)
        regimes = build_semantic_regime_specs(
            operator_name=operator_name,
            entrypoint=spec.entrypoint,
            cases=spec.cases,
            proto=spec.proto,
            base_axes=semantics.base_axes,
            has_reduction=semantics.has_reduction,
            case_ratios=ratios,
            runtime_backend="tilelang_dsl",
        )
        assert 1 <= len(regimes) <= 4, operator_name
        if operator_name in expected_partitions:
            assert len(regimes) == expected_partitions[operator_name], operator_name
        wp1_tags = {
            campaign_case_regime_tag(
                ratio,
                case,
                base_axes=semantics.base_axes,
                has_reduction=semantics.has_reduction,
            )
            for ratio, case in zip(ratios, spec.cases)
        }
        if len(wp1_tags) > 4:
            wp1_tags = {tag.removesuffix("_split") for tag in wp1_tags}
        actual_tags = {regime.regime_tag for regime in regimes}
        if len(wp1_tags) == 1 and len(spec.cases) >= 2:
            base_tag = next(iter(wp1_tags))
            assert len(regimes) == 2, operator_name
            assert all(tag == base_tag or tag.startswith(f"{base_tag}_") for tag in actual_tags), (
                operator_name
            )
        else:
            assert actual_tags == wp1_tags, operator_name
        for regime in regimes:
            packet = analyzer.build_universal_blueprint(
                semantics,
                regime.representative_hir_dims,
            )
            assert packet.entry_name == spec.entrypoint, (operator_name, regime.regime_tag)
            assert packet.roofline.ideal_roofline_us > 0.0, (operator_name, regime.regime_tag)
            assert packet.z3_pipeline_summary["source"] == "pytorch_semantic_frontend", (
                operator_name,
                regime.regime_tag,
            )
        checked += 1
    assert checked == 15


def test_empty_shell_seed_renderers_are_removed() -> None:
    assert not hasattr(marathon, "render_direct_tilelang_seed_module")
    assert not hasattr(marathon, "ensure_operator_seed_module")
    assert "render_direct_tilelang_seed_module" not in marathon.__all__
    assert "ensure_operator_seed_module" not in marathon.__all__
    assert "prepare_marathon_seed" in marathon.__all__
    assert "build_stage0_oracle_validator" in marathon.__all__
    assert "build_stage0_static_validator" in marathon.__all__


def _semantic_runner(
    tmp_path: Path,
    *,
    enable_track_router: bool = False,
) -> tuple[PerfTargetCampaignRunner, tuple[RegimeTargetSpec, ...]]:
    repo = _write_fixture_repo(tmp_path / "cann-bench")
    registry = CannbenchOpsRegistry(tasks_root=repo)
    spec = registry.resolve("exp")
    regimes = registry.get_regime_specs("exp")
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="exp", max_rounds=1),
        target_module_path=_target_module(tmp_path),
        perf_target=PerfTargetSpec("exp", 1.5, 1.2, 0.9),
        anchors=tuple(
            AnchorCaseSpec(spec.task, a.case_num, a.regime_tag, 0.0, 0.0) for a in spec.anchors
        ),
        regime_specs=regimes,
        checkpoint_path=tmp_path / "ckpt.json",
        cannbench_repo=repo,
        enable_track_router=enable_track_router,
    )
    return runner, regimes


def _evidence_ledger() -> ObjectivePhysicalLedger:
    return ObjectivePhysicalLedger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=1,
        pipe_cycles={
            "AIC.PIPE_M": 200,
            "AIC.CUBE": 100,
            "AIV0.PIPE_V": 400,
            "AIV1.PIPE_V": 400,
        },
        dominant_pipe="AIV0.PIPE_V",
        waterlines={
            "L1": ScopeWaterline(
                scope="L1",
                capacity_bytes=524288,
                explicit_bytes=262144,
                implicit_bloat_bytes=0,
                peak_allocated_bytes=262144,
                overflow_bytes=0,
                explicit_buffers={},
                implicit_buffers={},
            ),
            "UB": ScopeWaterline(
                scope="UB",
                capacity_bytes=196608,
                explicit_bytes=131072,
                implicit_bloat_bytes=0,
                peak_allocated_bytes=131072,
                overflow_bytes=0,
                explicit_buffers={},
                implicit_buffers={},
            ),
        },
        mte2_inflight_per_aiv=2,
        unaligned_dma_padding_bytes=0,
        gm_roundtrip_bytes=4 * 1024 * 1024,
        dominant_contig_bytes=2 * 1024 * 1024,
        dma_burst_efficiency_ratio=0.94,
        wave_occupancy=0.375,
        microarch_profile={"subkernel_count": 1, "mte2_bytes_hint": 1048576},
    )


def _trackb_route(regime: RegimeTargetSpec) -> RouteDecision:
    ratios = regime.representative_case_ratios
    assert ratios is not None
    return RouteDecision(
        track="developer",
        features=StructuralFeatures(
            phi_1_static_affine_bounds=True,
            phi_2_cube_to_vector_ratio=0.0,
            phi_3_quant_granularity="none",
            phi_4_indirect_sparse_gather=False,
            phi_5_cross_iter_recurrence="none",
            phi_6_dynamic_cross_core_handoff=False,
            phi_7_inner_contiguous_bytes=ratios.inner_contiguous_bytes,
            phi_8_persistent_wave_occupancy=ratios.wave_occupancy_ratio,
        ),
        rationale=("evidence fixture",),
        recommended_primitives=(),
        recommended_backend_tier="ascendc_template",
    )


def _surrogate_result(operator_name: str) -> CandidateSurrogateResult:
    return CandidateSurrogateResult(
        candidate_id="baseline",
        operator_name=operator_name,
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.5,
        surrogate_score=0.5,
    )


def test_semantic_regime_specs_roundtrip_official_representative_cases() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_EXP_GOLDEN, entry_name="exp")
    ratios = tuple(analyzer.compute_case_ratios(semantics, case) for case in _EXP_CASES)
    regimes = build_semantic_regime_specs(
        operator_name="exp",
        entrypoint="exp",
        cases=_EXP_CASES,
        proto=_EXP_PROTO,
        base_axes=semantics.base_axes,
        has_reduction=semantics.has_reduction,
        case_ratios=ratios,
        runtime_backend="tilelang_dsl",
    )
    cases_by_id = {int(case["case_id"]): case for case in _EXP_CASES}
    assert len(regimes) >= 2
    for regime in regimes:
        representative_case = regime.representative_case_entry
        assert representative_case
        case_id = int(representative_case["case_id"])
        assert representative_case == cases_by_id[case_id]
        assert regime.representative_case_ratios is not None
        assert regime.representative_case_ratios.case_id == case_id
        assert RepresentativeCase.from_case_entry(representative_case).input_shapes
        restored = RegimeTargetSpec.from_dict(regime.to_dict())
        assert restored.representative_case_entry == representative_case
        assert restored.representative_case_ratios == regime.representative_case_ratios


def test_cold_start_blueprints_use_each_regimes_official_representative_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _constraints(**kwargs: object) -> str:
        return "constraints"

    monkeypatch.setattr(
        seed_synthesizer_module,
        "build_progressive_llm_constraints",
        _constraints,
    )
    backend = _ScriptedBackend([_SEED_MODULE])
    outcome = _prepare(golden_source=_EXP_GOLDEN, backend=backend)
    cases_by_id = {int(case["case_id"]): case for case in _EXP_CASES}
    assert outcome.regime_specs
    for regime in outcome.regime_specs:
        packet = outcome.blueprints_by_regime[regime.regime_tag]
        case_id = int(regime.representative_case_entry["case_id"])
        assert packet.z3_pipeline_summary["representative_case_id"] == case_id
        representative = RepresentativeCase.from_case_entry(cases_by_id[case_id])
        assert representative.input_shapes
        raw_dtypes = cases_by_id[case_id]["dtype"]
        assert isinstance(raw_dtypes, list)
        assert representative.input_dtypes == tuple(str(dtype) for dtype in raw_dtypes)


def test_campaign_runner_passes_representative_case_ratios_to_static_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, regimes = _semantic_runner(tmp_path)
    regime = regimes[0]
    ratios = regime.representative_case_ratios
    assert ratios is not None
    stage_a_ratios: list[object] = []
    trackb_ratios: list[object] = []

    def _fake_stage_a(*args: object, **kwargs: object) -> StageAResult:
        stage_a_ratios.append(kwargs.get("case_ratios"))
        return StageAResult(passed=True, prim_func=None, ledger=_evidence_ledger())

    def _fake_trackb(*args: object, **kwargs: object) -> StageAResult:
        trackb_ratios.append(kwargs.get("case_ratios"))
        return StageAResult(
            passed=True,
            prim_func=None,
            ledger=_evidence_ledger(),
            backend_kind="ascendc",
        )

    monkeypatch.setattr(campaign_runner_module, "run_stage_a_static", _fake_stage_a)
    monkeypatch.setattr(campaign_runner_module, "evaluate_ascendc_candidate", _fake_trackb)
    monkeypatch.setattr(
        PerfTargetCampaignRunner,
        "_resolve_combine_gate_spec",
        lambda self, module_source, regime_spec, resolved: None,
    )
    monkeypatch.setattr(
        PerfTargetCampaignRunner,
        "_instantiate_trackb_tu",
        lambda self, module_source, cpp_source, regime_spec: cpp_source,
    )

    dsl_result = runner._default_l0_checker(_SEED_MODULE, regime)
    assert dsl_result.passed is True
    assert stage_a_ratios and all(item == ratios for item in stage_a_ratios)

    trackb_regime = replace(
        regime,
        runtime_backend="ascendc_trackb",
        trackb_symbols=("EXP_TRACKB_CU",),
    )
    trackb_result = runner._default_l0_checker(_TRACKB_SEED_MODULE, trackb_regime)
    assert trackb_result.passed is True
    assert trackb_result.backend_kind == "ascendc"
    assert trackb_ratios == [ratios]


def test_trackb_round_evidence_persists_blueprint_observation_and_provenance(
    tmp_path: Path,
) -> None:
    runner, regimes = _semantic_runner(tmp_path, enable_track_router=True)
    regime = replace(regimes[0], runtime_backend="ascendc_trackb")
    blueprint = runner._regime_blueprints[regime.regime_tag]
    route = _trackb_route(regime)
    compiler_summary = BishengResourceSummary(
        vec_regs=64,
        spill_bytes=512,
        vloopv2_loops=1,
        vf_fusion_breaks=1,
    )
    check = L0L1CandidateCheck(
        candidate_id="r2_trackb",
        passed=True,
        failed_tier=None,
        patched_module_source=_TRACKB_SEED_MODULE,
        snippet_source="",
        stage_a=StageAResult(
            passed=True,
            prim_func=None,
            ledger=_evidence_ledger(),
            features=route.features,
            route=route,
            backend_kind="ascendc",
        ),
        stage_b=StageBResult(passed=True, compiler_summary=compiler_summary),
    )
    candidate_sha = runner._source_sha256(check.patched_module_source)
    unsubmitted_check = L0L1CandidateCheck(
        candidate_id="r2_trackb_l1_failed",
        passed=False,
        failed_tier="stage_b",
        patched_module_source=_TRACKB_SEED_MODULE + "\n# rejected before fast eval\n",
        snippet_source="",
        stage_a=StageAResult(
            passed=True,
            prim_func=None,
            ledger=_evidence_ledger(),
            features=route.features,
            route=route,
            backend_kind="ascendc",
        ),
        stage_b=StageBResult(passed=False, error_message="bisheng compile failed"),
        error_message="bisheng compile failed",
    )
    parent_sha = runner._source_sha256(_SEED_MODULE)
    case_entry = regime.representative_case_entry
    raw_case_id = case_entry.get("case_id")
    assert isinstance(raw_case_id, int)
    raw_dtype = case_entry.get("dtype")
    expected_dtype: str | list[str]
    if isinstance(raw_dtype, str):
        expected_dtype = raw_dtype
    elif isinstance(raw_dtype, (list, tuple)):
        expected_dtype = [str(item) for item in raw_dtype]
    else:
        expected_dtype = ""
    runner._msprof_cache[(candidate_sha, regime.regime_tag)] = {
        "dur_us": 41.5,
        "aiv_vec_ratio": 0.62,
        "aic_mac_ratio": 0.05,
        "mte2_wait_ratio": 0.18,
    }
    evidence = runner._collect_trackb_round_evidence(
        round_idx=2,
        regime_spec=regime,
        blueprint=blueprint,
        live_plan=None,
        active_mask=runner._resolve_trackb_freedom_mask(regime),
        checks=(check, unsubmitted_check),
        l2_results={"r2_trackb": _surrogate_result("exp")},
        active_kernel="gather_kernel",
        l2_run_ids={"r2_trackb": "run-fast-42"},
        parent_sha256=parent_sha,
        parent_l1_summary=compiler_summary,
        parent_l1_summary_source="bisheng_l1",
    )
    assert len(evidence) == 2
    item = evidence[0]
    assert item["round_idx"] == 2
    assert item["regime_tag"] == regime.regime_tag
    assert item["candidate_id"] == "r2_trackb"
    assert item["candidate_sha256"] == candidate_sha
    assert item["runtime_backend"] == "ascendc_trackb"
    assert item["backend_tier"] == "ascendc_template"
    assert item["track"] == "developer"
    assert item["active_kernel"] == "gather_kernel"
    assert item["l2_evaluated"] is True
    assert item["l2_run_id"] == "run-fast-42"
    assert item["parent_candidate_sha256"] == parent_sha
    parent_summary = item["parent_compiler_summary"]
    assert isinstance(parent_summary, Mapping)
    assert parent_summary == compiler_summary.to_dict()
    assert item["parent_compiler_summary_source"] == "bisheng_l1"
    assert item["representative_case_id"] == raw_case_id
    assert item["representative_input_dtype"] == expected_dtype
    unsubmitted = evidence[1]
    assert unsubmitted["candidate_id"] == "r2_trackb_l1_failed"
    assert unsubmitted["active_kernel"] == "gather_kernel"
    assert unsubmitted["l2_evaluated"] is False
    assert unsubmitted["l2_run_id"] == ""
    assert unsubmitted["latency_us"] == 0.0
    unsubmitted_provenance = unsubmitted["field_provenance"]
    assert isinstance(unsubmitted_provenance, Mapping)
    assert unsubmitted_provenance["latency_source"] == "none"
    assert unsubmitted_provenance["observation_source"] == "roofline_static"
    blueprint_summary = item["blueprint"]
    assert isinstance(blueprint_summary, Mapping)
    assert blueprint_summary["ideal_roofline_us"] == blueprint.roofline.ideal_roofline_us
    observation = item["observation"]
    assert isinstance(observation, Mapping)
    assert observation["has_l2_telemetry"] is True
    residuals = observation["residuals"]
    assert isinstance(residuals, Mapping)
    assert residuals["compute_issue_gap"] >= 0.25
    assert item["dominant_axis"] == "onchip_sram_reg"
    unlocked = item["unlocked_facets"]
    assert isinstance(unlocked, list)
    assert "exec_mode_and_spr" in unlocked
    provenance = item["field_provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["latency_source"] == "none"
    assert provenance["observation_source"] == "msprof_l2"
    assert provenance["compiler_summary_source"] == "bisheng_l1"
    assert provenance["has_l1_compiler_summary"] is True
    measured_fields = provenance["measured_l2_fields"]
    assert isinstance(measured_fields, list)
    assert "l2_measured_latency_us" in measured_fields

    record = CampaignRoundRecord(
        round_idx=2,
        regime_tag=regime.regime_tag,
        builder_name="_build_exp_prim_func",
        parent_lane="exploit",
        creative_effort="medium",
        l0_l1_passed_ids=("r2_trackb",),
        l0_l1_failed=(),
        best_round_speedup=1.0,
        best_so_far_speedup=1.0,
        promoted=False,
        rolled_back_l3=False,
        stagnation_rounds_after=0,
        trackb_evidence=evidence,
    )
    best = _surrogate_result("exp")
    report = CampaignReport(
        operator_name="exp",
        stop_reason="max_rounds",
        total_rounds=1,
        baseline_result=best,
        best_result=best,
        rounds=(record,),
    )
    payload = report.to_dict()
    rounds = payload["rounds"]
    assert isinstance(rounds, list)
    round_payload = rounds[0]
    assert isinstance(round_payload, Mapping)
    persisted_evidence = round_payload["trackb_evidence"]
    assert isinstance(persisted_evidence, list)
    persisted = persisted_evidence[0]
    assert isinstance(persisted, Mapping)
    assert persisted["round_idx"] == 2
    assert persisted["regime_tag"] == regime.regime_tag
    assert persisted["backend_tier"] == "ascendc_template"
    persisted_blueprint = persisted["blueprint"]
    assert isinstance(persisted_blueprint, Mapping)
    assert persisted_blueprint["ideal_roofline_us"] > 0.0
    persisted_observation = persisted["observation"]
    assert isinstance(persisted_observation, Mapping)
    persisted_residuals = persisted_observation["residuals"]
    assert isinstance(persisted_residuals, Mapping)
    assert persisted_residuals["grid_wave_gap"] > 0.0
    persisted_provenance = persisted["field_provenance"]
    assert isinstance(persisted_provenance, Mapping)
    assert persisted_provenance["has_l2_telemetry"] is True
    restored_record = CampaignRoundRecord.from_dict(round_payload)
    assert restored_record.trackb_evidence == record.trackb_evidence


_RENDER_TRACKB_SEED_MODULE = '''
EXP_TRACKB_CU = """
#include <tl_templates/ascend/kernel_entry.h>
__global__ void gather_kernel() {}
"""


def _select_exp_config(M: int = 1024, D: int = 1024, tl_dtype: str = "float16") -> dict[str, object]:
    """Select the Track-B translation unit for one regime."""
    return {"pass_configs": {}, "compile_flags": [], "trackb_symbol": "EXP_TRACKB_CU"}


def _render_trackb_tu(**dims: object) -> str:
    """Append a launcher comment exposing the representative dimensions."""
    base = str(dims.get("trackb_src", ""))
    rows = int(dims.get("M", 0))
    width = int(dims.get("D", 0))
    return base + "\\n// launch M=" + str(rows) + " D=" + str(width) + "\\n"


def exp(x, index):
    """Dispatch gather through the bisheng-compiled Track-B library."""
    raise RuntimeError("offline test fixture has no compiled Track-B library")


kernel = exp
'''


def _trackb_render_regime_specs() -> tuple[RegimeTargetSpec, ...]:
    """Two Track-B regimes whose representative dims must render distinct launchers."""
    ratio = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.5,
        working_set_vreg_ratio=0.25,
        wave_occupancy_ratio=1.0,
        oi_ridge_ratio=0.5,
        inner_contiguous_bytes=512,
    )
    return (
        RegimeTargetSpec(
            regime_tag="vector",
            builder_candidates=("_select_exp_config",),
            sample_builder_kwargs={"M": 1024, "D": 1024},
            representative_hir_dims={"M": 1024, "D": 1024},
            config_selector_fn="_select_exp_config",
            editable_symbols=("_select_exp_config", "EXP_TRACKB_CU"),
            runtime_backend="ascendc_trackb",
            trackb_symbols=("EXP_TRACKB_CU",),
            representative_case_ratios=ratio,
        ),
        RegimeTargetSpec(
            regime_tag="vector_twopass",
            builder_candidates=("_select_exp_config",),
            sample_builder_kwargs={"M": 8192, "D": 8192},
            representative_hir_dims={"M": 8192, "D": 8192},
            config_selector_fn="_select_exp_config",
            editable_symbols=("_select_exp_config", "EXP_TRACKB_CU"),
            runtime_backend="ascendc_trackb",
            trackb_symbols=("EXP_TRACKB_CU",),
        ),
    )


def test_stage0_trackb_validator_requires_renderer_abi(tmp_path: Path) -> None:
    """Reject a Track-B seed that cannot render per-regime launchers."""
    validator = build_stage0_static_validator(
        _trackb_render_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    passed, error = validator(_TRACKB_SEED_MODULE, "cce_trackb")
    assert passed is False
    assert error.count("Track-B seed must declare def _render_trackb_tu") == 2
    assert "[vector]" in error
    assert "[vector_twopass]" in error


def test_stage0_trackb_validator_instantiates_each_regime_before_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate each regime's rendered launcher instead of the raw translation unit."""
    evaluated: list[tuple[str, CasePhysicsRatios | None]] = []
    compiled: list[str] = []

    def _evaluate(cpp_source: str, *, case_ratios: CasePhysicsRatios | None) -> StageAResult:
        evaluated.append((cpp_source, case_ratios))
        return StageAResult(passed=True, prim_func=None, ledger=None)

    def _compile(
        cpp_source: str,
        *,
        compile_flags: Sequence[str],
        bisheng_bin: str | None,
        cann_root: Path | None,
    ) -> StageBResult:
        compiled.append(cpp_source)
        return StageBResult(passed=True)

    monkeypatch.setattr(marathon, "evaluate_ascendc_candidate", _evaluate)
    monkeypatch.setattr(marathon, "run_stage_b_ascendc_compile", _compile)
    specs = _trackb_render_regime_specs()
    validator = build_stage0_static_validator(
        specs,
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    passed, error = validator(_RENDER_TRACKB_SEED_MODULE, "cce_trackb")
    assert passed is True
    assert error == ""
    assert len(evaluated) == 2
    assert compiled == [tu for tu, _ratios in evaluated]
    assert "// launch M=1024 D=1024" in evaluated[0][0]
    assert "// launch M=8192 D=8192" in evaluated[1][0]
    assert evaluated[0][0] != evaluated[1][0]
    assert "gather_kernel" in evaluated[0][0]
    assert evaluated[0][1] is specs[0].representative_case_ratios
    assert evaluated[1][1] is None


def test_stage0_trackb_validator_aggregates_every_regime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report every regime's Track-B L0 failure without compiling any translation unit."""
    compiled: list[str] = []

    def _evaluate(cpp_source: str, *, case_ratios: CasePhysicsRatios | None) -> StageAResult:
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=None,
            error_message="token purity",
        )

    def _compile(*_args: object, **_kwargs: object) -> StageBResult:
        compiled.append("called")
        return StageBResult(passed=True)

    monkeypatch.setattr(marathon, "evaluate_ascendc_candidate", _evaluate)
    monkeypatch.setattr(marathon, "run_stage_b_ascendc_compile", _compile)
    validator = build_stage0_static_validator(
        _trackb_render_regime_specs(),
        entrypoint="exp",
        worktree=tmp_path,
        env_file=None,
    )
    passed, error = validator(_RENDER_TRACKB_SEED_MODULE, "cce_trackb")
    assert passed is False
    assert error == (
        "[vector] Track-B L0 token purity (EXP_TRACKB_CU): token purity; "
        "[vector_twopass] Track-B L0 token purity (EXP_TRACKB_CU): token purity"
    )
    assert compiled == []


def test_marathon_passes_seed_blueprints_and_topology_axes_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward Stage-0 blueprints and topology axes into the campaign runner."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANNBENCH_REPO", raising=False)
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_EXP_GOLDEN, entry_name="exp", proto=_EXP_PROTO)
    specs = _legacy_regime_specs()
    packets = {
        spec.regime_tag: analyzer.build_universal_blueprint(
            semantics,
            spec.representative_hir_dims,
        )
        for spec in specs
    }
    seed_outcome = SeedSynthesisOutcome(
        mode="cold_start",
        initial_track="tilelang_dsl",
        module_source=_SEED_MODULE,
        semantics=semantics,
        case_ratios=(),
        regime_specs=specs,
        blueprints_by_regime=packets,
        attempts_used=1,
        verification_state="l3_verified",
        l0_l1_passed=True,
        oracle_accuracy_passed=True,
        candidate_sha256=_sha256(_SEED_MODULE),
        l3_run_id="run-stage0-0001",
        l3_receipt_path="/tmp/tilefoundry-l3-receipt.json",
        l3_state="passed",
        l3_elapsed_sec=1.5,
    )
    captured: dict[str, object] = {}

    class _CapturingRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, *, resume: bool) -> object:
            return _StubReport()

    class _StubReport:
        def to_dict(self) -> dict[str, object]:
            return {"status": "stubbed"}

    monkeypatch.setattr(marathon, "_resolve_worktree", lambda _cfg: (tmp_path, tmp_path))
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _worktree: None)
    monkeypatch.setattr(marathon, "_configure_marathon_logging", lambda _path: None)
    monkeypatch.setattr(
        marathon,
        "ensure_marathon_env_file",
        lambda **_kw: tmp_path / "env.sh",
    )
    monkeypatch.setattr(marathon, "configure_marathon_process_env", lambda **_kw: {})
    monkeypatch.setattr(
        marathon,
        "create_llm_backend_from_env",
        lambda **_kw: MockLLMBackend(),
    )
    monkeypatch.setattr(marathon, "prepare_marathon_seed", lambda **_kw: seed_outcome)
    monkeypatch.setattr(
        marathon,
        "build_resilient_l2_evaluator",
        lambda **_kw: (lambda cands, anchors: {}),
    )
    monkeypatch.setattr(
        marathon,
        "build_prewarmed_l3_runner",
        lambda **_kw: (lambda _path: L3AcceptanceResult(passed=True)),
    )
    monkeypatch.setattr(
        marathon,
        "resolve_default_experience_lake_dir",
        lambda **_kw: None,
    )
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _CapturingRunner)

    exit_code = run_operator_marathon(
        MarathonConfig(
            operator_name="exp",
            state_dir=tmp_path / "state",
            target_module_path=tmp_path / "kernels.py",
            salvage_beam_candidates=False,
        )
    )
    assert exit_code == 0
    assert captured["regime_blueprints"] == packets
    assert captured["topology_axes"] == seed_outcome.semantics.base_axes
    assert captured["regime_specs"] == seed_outcome.regime_specs
