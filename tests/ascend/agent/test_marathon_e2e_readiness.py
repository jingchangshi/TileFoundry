"""End-to-end readiness tests for Marathon live SIL wiring, Carbon effort, and Stage-0 seeding."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import marathon as marathon_module
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    default_source_root,
    get_default_regime_specs,
    lookup_case_dims,
    lookup_case_metadata,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    discover_marathon_environment,
    ensure_marathon_env_file,
    prepare_marathon_seed,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
    ensure_npu_compare_shim_dir,
    lookup_case_static_spec,
)
from tilefoundry_ascend.agent.kb import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    check_npu_entrypoint_purity,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    GenerationRequest,
    MockLLMBackend,
    RoleRoutingConfig,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.op_analyzer import OpAdvisoryExtractor

_ROPE_SEED_MODULE = '''
"""TileLang Ascend950PR seed module for apply_rotary_pos_emb (zero golden fallback)."""

import contextlib

import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ascend_target_ctx():
    from tilelang import tvm
    from tilelang.backend.module import create_backend_context

    if tvm.target.Target.current(allow_none=True) is None:
        return create_backend_context("ascend").target
    return contextlib.nullcontext()


def _select_apply_rotary_pos_emb_config(**kwargs) -> dict[str, object]:
    raw_dt = str(kwargs.get("tl_dtype", "float16")).strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "float16"
    return {
        "tile_elems": int(kwargs.get("tile_elems", 1024)),
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": [],
    }


def _build_apply_rotary_pos_emb_prim_func(
    total_elements: int = 1024,
    tile_elems: int = 1024,
    tl_dtype: str = "float16",
    **_extra_kwargs,
):
    blk = max(32, min(4096, ((int(tile_elems) + 31) // 32) * 32))
    tot = max(blk, ((int(total_elements) + blk - 1) // blk) * blk)
    blocks = tot // blk
    dt = tl_dtype if tl_dtype in ("float16", "bfloat16", "float32") else "float16"
    with _ascend_target_ctx():

        @T.prim_func
        def main(X: T.Tensor((tot,), dt), Y: T.Tensor((tot,), dt)):
            with T.Kernel(blocks) as cid:
                ub = T.alloc_shared((blk,), dt)
                T.copy(X[cid * blk : (cid + 1) * blk], ub)
                T.copy(ub, Y[cid * blk : (cid + 1) * blk])

    return main


def _get_kernel(total_elements: int, tl_dtype: str):
    cfg = _select_apply_rotary_pos_emb_config(tl_dtype=tl_dtype)
    key = (int(total_elements), int(cfg["tile_elems"]), tl_dtype)
    if key not in _KERNEL_CACHE:
        with _ascend_target_ctx():
            prog = _build_apply_rotary_pos_emb_prim_func(
                total_elements=total_elements,
                tile_elems=int(cfg["tile_elems"]),
                tl_dtype=tl_dtype,
            )
        _KERNEL_CACHE[key] = tilelang.compile(prog, target="ascend", out_idx=[1])
    return _KERNEL_CACHE[key]


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the pure TileLang NPU seed kernel for apply_rotary_pos_emb."""
    orig_shape = query.shape
    flat = query.contiguous().view(-1)
    blk = 1024
    pad = (blk - (flat.numel() % blk)) % blk
    work = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
    dt_map = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
    fn = _get_kernel(work.numel(), dt_map[work.dtype])
    out = fn(work)
    return out[: flat.numel()].view(orig_shape), key


kernel = apply_rotary_pos_emb
'''

_SOFTMAX_SEED_MODULE = '''
"""TileLang Ascend950PR seed module for softmax (zero golden fallback)."""

import contextlib

import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ascend_target_ctx():
    from tilelang import tvm
    from tilelang.backend.module import create_backend_context

    if tvm.target.Target.current(allow_none=True) is None:
        return create_backend_context("ascend").target
    return contextlib.nullcontext()


def _select_softmax_config(**kwargs) -> dict[str, object]:
    raw_dt = str(kwargs.get("tl_dtype", "float16")).strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "float16"
    return {
        "tile_elems": int(kwargs.get("tile_elems", 1024)),
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": [],
    }


def _build_softmax_prim_func(
    total_elements: int = 1024,
    tile_elems: int = 1024,
    tl_dtype: str = "float16",
    **_extra_kwargs,
):
    blk = max(32, min(4096, ((int(tile_elems) + 31) // 32) * 32))
    tot = max(blk, ((int(total_elements) + blk - 1) // blk) * blk)
    blocks = tot // blk
    dt = tl_dtype if tl_dtype in ("float16", "bfloat16", "float32") else "float16"
    with _ascend_target_ctx():

        @T.prim_func
        def main(X: T.Tensor((tot,), dt), Y: T.Tensor((tot,), dt)):
            with T.Kernel(blocks) as cid:
                ub = T.alloc_shared((blk,), dt)
                T.copy(X[cid * blk : (cid + 1) * blk], ub)
                T.copy(ub, Y[cid * blk : (cid + 1) * blk])

    return main


def _get_kernel(total_elements: int, tl_dtype: str):
    cfg = _select_softmax_config(tl_dtype=tl_dtype)
    key = (int(total_elements), int(cfg["tile_elems"]), tl_dtype)
    if key not in _KERNEL_CACHE:
        with _ascend_target_ctx():
            prog = _build_softmax_prim_func(
                total_elements=total_elements,
                tile_elems=int(cfg["tile_elems"]),
                tl_dtype=tl_dtype,
            )
        _KERNEL_CACHE[key] = tilelang.compile(prog, target="ascend", out_idx=[1])
    return _KERNEL_CACHE[key]


def softmax(x, dim=-1):
    """Execute the pure TileLang NPU seed kernel for softmax."""
    orig_shape = x.shape
    flat = x.contiguous().view(-1)
    blk = 1024
    pad = (blk - (flat.numel() % blk)) % blk
    work = torch.nn.functional.pad(flat, (0, pad)) if pad else flat
    dt_map = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
    fn = _get_kernel(work.numel(), dt_map[work.dtype])
    out = fn(work)
    return out[: flat.numel()].view(orig_shape)


kernel = softmax
'''


class _ScriptedBackend:
    """Deterministic async LLM backend replaying one fixed module source."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return self._reply


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _install_stage0_gate_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    worktree: Path,
) -> list[Path]:
    """Replace device-only L1 compilation and real L3 execution with digest-checking fakes."""
    staged_calls: list[Path] = []

    def _fake_l1(
        module_source: str,
        *,
        builder_name: str,
        builder_kwargs: Mapping[str, object],
        config_selector_name: str | None = None,
        pass_configs: Mapping[str, object] | None = None,
        compile_flags: Sequence[str] | str | None = None,
        dims: Mapping[str, object] | None = None,
        source_root: str | Path | None = None,
        env_file: str | Path | None = None,
        timeout_sec: int = 120,
    ) -> StageBResult:
        assert "def _build_" in module_source
        assert builder_name.startswith("_build_")
        assert pass_configs is None or isinstance(pass_configs, Mapping)
        assert compile_flags is None or isinstance(compile_flags, (Sequence, str))
        assert dims is None or isinstance(dims, Mapping)
        return StageBResult(passed=True)

    def _fake_l3(
        *,
        operator_name: str | None = None,
        candidate_module_path: str | Path | None = None,
        source_root: str | Path | None = None,
        env_file: str | Path | None = None,
        device_id: str = "1",
        timeout_sec: int = 0,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        assert operator_name is not None
        assert candidate_module_path is not None
        assert source_root is not None
        assert Path(source_root).resolve() == worktree.resolve()
        assert env_file is None
        assert resume_execution_id == "" and resume_run_id == ""
        assert on_execution_started is None
        staged = Path(candidate_module_path)
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        assert staged.name == f"{digest}.py"
        assert staged.is_file()
        staged_calls.append(staged)
        return L3AcceptanceResult(
            passed=True,
            run_id=f"run-stage0-{digest[:12]}",
            receipt_path=str(staged.with_suffix(".receipt.json")),
            elapsed_sec=0.25,
            candidate_sha256=digest,
            state="passed",
            device_id=device_id,
        )

    monkeypatch.setattr(marathon_module, "run_l1_compile_check", _fake_l1)
    monkeypatch.setattr(marathon_module, "run_l3_full_contract_via_agent_gates", _fake_l3)
    return staged_calls


def _cannbench_tasks_root() -> Path:
    return default_source_root().parent / "cann-bench" / "tasks"


def test_campaign_runner_live_sil_7facet_cost_calibration_and_shared_stone_pruning(
    tmp_path: Path,
) -> None:
    """Verify live SIL recomputation, 7-facet prompts, hazards, and sibling pruning."""
    mod_file = tmp_path / "demo_op.py"
    initial_src = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 64, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 64, tile_n: int = 64):\n"
        "    return ('v0', tile_m, tile_n)\n\n"
        "def demo_op(x):\n"
        "    return _build_demo_prim_func(), x\n"
    )
    mod_file.write_text(initial_src, encoding="utf-8")

    r1 = RegimeTargetSpec(
        regime_tag="r1_main",
        builder_candidates=("_build_demo_prim_func",),
        sample_builder_kwargs={"tile_m": 64, "tile_n": 64},
        config_selector_fn="_select_demo_config",
        editable_symbols=("_select_demo_config", "_build_demo_prim_func"),
        representative_hir_dims={"M": 128, "N": 128},
        representative_anchor_dims={"M": 128, "N": 128},
    )
    r2 = RegimeTargetSpec(
        regime_tag="r2_tail",
        builder_candidates=("_build_demo_prim_func",),
        sample_builder_kwargs={"tile_m": 32, "tile_n": 64},
        config_selector_fn="_select_demo_config",
        editable_symbols=("_select_demo_config", "_build_demo_prim_func"),
        representative_hir_dims={"M": 32, "N": 64},
        representative_anchor_dims={"M": 32, "N": 64},
    )
    anchors = [
        AnchorCaseSpec("level1/demo_op", 1, "r1_main", 100.0, 20.0),
        AnchorCaseSpec("level1/demo_op", 2, "r2_tail", 100.0, 20.0),
    ]

    overflow_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=253952,
                explicit_buffers={"x": 200000},
                implicit_buffers={"extra_tmp": 120000},
            ),
        },
        gm_roundtrip_bytes=131072,
        dominant_contig_bytes=32768,
        grid_blocks=56,
    )

    improved_shared_slice = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 32, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 32, tile_n: int = 64):\n"
        "    return ('v2_promoted', tile_m, tile_n)\n"
    )
    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Shrink UB tile footprint and eliminate register spill",
            "target_local_metric": "dur_us",
            "dsl_source": improved_shared_slice,
        }
    )

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [improved_shared_slice, improved_shared_slice],
            "explore_creative": [creative_reply, creative_reply, creative_reply, creative_reply],
        }
    )

    comp_hazards = BishengResourceSummary(
        spill_bytes=64,
        vloopv2_loops=0,
        software_loops=2,
        vf_fusion_breaks=2,
        alias_serialized_buffers=1,
        dual_issue_ipc_collapse=0.5,
        dcache_dcci_flushes=1,
        has_local_membar_in_vf_call=True,
    )

    def _l2_eval(cands, ancs):
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, _p in cands:
            lat = 50.0 if cid == "incumbent_baseline" else 22.0
            out[cid] = aggregate_surrogate_result(
                cid,
                "demo_op",
                [build_anchor_metrics(a, accuracy_passed=True, cand_us=lat) for a in ancs],
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=1,
            k_rule_workers=1,
            stagnation_escalate_rounds=1,
            exploit_effort="high",
        ),
        target_module_path=mod_file,
        perf_target=PerfTargetSpec("demo_op", 4.0, 3.5, 0.95),
        anchors=anchors,
        regime_specs=[r1, r2],
        llm_backend=backend,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        l0_checker=lambda _src, _reg: StageAResult(
            passed=True,
            prim_func=None,
            ledger=overflow_ledger,
        ),
        l1_compiler=lambda _src, _reg: StageBResult(
            passed=True,
            spill_bytes=comp_hazards.spill_bytes,
            vloopv2_hit=comp_hazards.vloopv2_hit,
            compiler_summary=comp_hazards,
        ),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        source_root=tmp_path,
    )

    stale_stone = BeamCandidate(
        candidate_id="stale_r2_stone",
        round_idx=0,
        worker_kind="creative_explore",
        rule_id=None,
        dsl_source=(
            "def _build_demo_prim_func(tile_m: int = 16, tile_n: int = 32):\n"
            "    return ('stale_old', tile_m, tile_n)\n"
        ),
        track="developer",
        ledger=overflow_ledger,
        residual=compute_static_dynamic_residual(overflow_ledger, {"dur_us": 45.0}),
        latency_us=45.0,
        accuracy_passed=True,
    )
    runner._beams["r2_tail"].stepping_stone_beam = [stale_stone]

    l1_residual = compute_static_dynamic_residual(
        overflow_ledger,
        {"dur_us": 50.0, "roofline_eta": 0.8},
        compiler_summary=comp_hazards,
    )
    assert l1_residual.has_l1_compiler_summary is True
    initial_slice = (
        "def _select_demo_config(**kwargs):\n"
        "    return {'tile_m': 64, 'tile_n': 64}\n\n"
        "def _build_demo_prim_func(tile_m: int = 64, tile_n: int = 64):\n"
        "    return ('v0', tile_m, tile_n)\n"
    )
    r1_stone = BeamCandidate(
        candidate_id="r1_hazard_stone",
        round_idx=0,
        worker_kind="creative_explore",
        rule_id=None,
        dsl_source=initial_slice,
        track="developer",
        ledger=overflow_ledger,
        residual=l1_residual,
        latency_us=50.0,
        accuracy_passed=True,
    )
    runner._beams["r1_main"].stepping_stone_beam = [r1_stone]
    parent_module_source = patch_regime_into_module(
        initial_src,
        r1,
        initial_slice,
        operator_entrypoint="demo_op",
        sibling_specs=(r1, r2),
    )
    runner._msprof_cache[
        (runner._source_sha256(parent_module_source), "r1_main")
    ] = {
        "dur_us": 40.0,
        "aic_mac_ratio": 0.11,
        "l2_hit_rate": 0.42,
    }

    base_res = runner._evaluate_single_baseline(initial_src, anchors)
    (
        rec,
        new_incumbent,
        _new_best,
        _stag_after,
        promoted,
        _rolled_back,
        escalated,
        _promotion_blocked,
    ) = asyncio.run(
        runner._execute_single_round(
            round_idx=1,
            regime_spec=r1,
            incumbent_source=initial_src,
            best_result=base_res,
            active_anchors=anchors,
            stagnation_rounds=1,
            admitted_rule_ids=[],
        )
    )

    assert escalated is True
    assert promoted is True
    assert rec.promoted is True
    assert "v2_promoted" in new_incumbent

    exploit_reqs = [c for c in backend.calls if c.role == "exploit_rule"]
    creative_reqs = [c for c in backend.calls if c.role == "explore_creative"]
    assert exploit_reqs and creative_reqs
    for req in (*exploit_reqs, *creative_reqs):
        assert "### SIL LIVE POLICY & 7-FACET CONSTRAINT ENVELOPE" in req.user_prompt
        assert "tilelang_7pipe_isa_coverage" in req.user_prompt
        assert "### HARDWARE MSPROF TELEMETRY" in req.user_prompt
        assert "spill_bytes: 64" in req.user_prompt
        assert "vloopv2_hit: False" in req.user_prompt
        assert "vf_fusion_breaks: 2" in req.user_prompt
        assert "aic_mac_ratio: 0.1100" in req.user_prompt
    assert exploit_reqs[0].effort == "high"

    mask = runner._freedom_masks["r1_main"]
    assert mask.sram_layout_and_aliasing is True
    assert mask.exec_mode_and_spr is True
    assert mask.pipeline_and_sync_graph is True

    assert runner._cost_model.sample_count > 0

    assert runner._beams["r2_tail"].stepping_stone_beam == []


def test_agy_effort_defaults_and_campaign_defaults(tmp_path: Path) -> None:
    """Verify Agy compatibility stays in its provider and campaign defaults are neutral."""
    fake_agy = tmp_path / "fake_agy"
    fake_agy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_agy.chmod(0o755)

    carbon_backend = AgyCliLLMBackend(
        bin_path=str(fake_agy),
        routing=RoleRoutingConfig("gemini-pro-orchestrator", "explore", "escalate", "repair"),
    )
    assert carbon_backend.select_effort(GenerationRequest("exploit_rule", "s", "u")) == "high"
    assert (
        carbon_backend.select_effort(GenerationRequest("exploit_rule", "s", "u", effort="xhigh"))
        == "high"
    )

    hydrogen_backend = AgyCliLLMBackend(
        bin_path=str(fake_agy),
        routing=RoleRoutingConfig("gemini-4-hydrogen-medium", "explore", "escalate", "repair"),
    )
    assert hydrogen_backend.select_effort(GenerationRequest("exploit_rule", "s", "u")) == "medium"

    m_cfg = MarathonConfig(operator_name="rms_norm")
    assert m_cfg.llm_backend is None
    assert m_cfg.escalate_model is None
    assert m_cfg.exploit_effort == "high"
    assert m_cfg.auto_bootstrap_seed is True

    c_cfg = CampaignConfig(operator_name="rms_norm")
    assert c_cfg.exploit_effort == "medium"


def test_prepare_marathon_seed_cold_start_rope_semantic_regimes_pass_l0_and_takeover_is_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Stage-0 rope seeding, L3-gated publish, and unverified takeover."""
    if not _cannbench_tasks_root().is_dir():
        pytest.skip("cann-bench repository not present on disk")

    cannbench_repo = _cannbench_tasks_root().parent
    staging_dir = tmp_path / ".tilefoundry-marathon" / "apply_rotary_pos_emb" / "stage0_candidates"
    seed_path = tmp_path / "examples" / "ascend950pr_ops" / "apply_rotary_pos_emb" / "kernels.py"
    staged_calls = _install_stage0_gate_fakes(monkeypatch, worktree=tmp_path)
    backend = _ScriptedBackend(_ROPE_SEED_MODULE)

    outcome = prepare_marathon_seed(
        operator_name="apply_rotary_pos_emb",
        target_module=seed_path,
        staging_dir=staging_dir,
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=cannbench_repo,
        llm_backend=backend,
        cold_start=False,
        device_id="1",
        op_logger=logging.getLogger("test.readiness.rope_seed"),
    )

    assert outcome.mode == "cold_start"
    assert outcome.verification_state == "l3_verified"
    assert outcome.l0_l1_passed is True
    assert outcome.oracle_accuracy_passed is True
    assert outcome.candidate_sha256 == _sha256(outcome.module_source)
    assert outcome.l3_run_id == f"run-stage0-{outcome.candidate_sha256[:12]}"
    assert outcome.l3_receipt_path.endswith(f"{outcome.candidate_sha256}.receipt.json")
    assert len(staged_calls) == 1
    assert staged_calls[0].name == f"{outcome.candidate_sha256}.py"
    assert staged_calls[0].parent == staging_dir.resolve()
    assert staged_calls[0].read_text(encoding="utf-8").strip() == outcome.module_source.strip()

    published = seed_path.read_text(encoding="utf-8")
    assert published.strip() == outcome.module_source.strip()
    assert _sha256(published) == outcome.candidate_sha256
    init_source = (seed_path.parent / "__init__.py").read_text(encoding="utf-8")
    assert "from .kernels import apply_rotary_pos_emb, kernel" in init_source
    assert {regime.regime_tag for regime in outcome.regime_specs} == {
        "vector_latency",
        "vector_throughput",
    }
    default_tags = {regime.regime_tag for regime in get_default_regime_specs("apply_rotary_pos_emb")}
    assert {regime.regime_tag for regime in outcome.regime_specs} == default_tags
    assert set(outcome.blueprints_by_regime) == {
        regime.regime_tag for regime in outcome.regime_specs
    }
    assert all(regime.runtime_backend == "tilelang_dsl" for regime in outcome.regime_specs)

    system_prompt = backend.requests[0].system_prompt
    user_prompt = backend.requests[0].user_prompt
    assert "def _select_apply_rotary_pos_emb_config(" in system_prompt
    assert "def _build_apply_rotary_pos_emb_prim_func(" in system_prompt
    assert "def _get_kernel(" in system_prompt
    assert "def apply_rotary_pos_emb(query: torch.Tensor," in system_prompt
    assert "-> tuple[torch.Tensor, torch.Tensor]:" in system_prompt
    assert "kernel = apply_rotary_pos_emb" in system_prompt
    assert "layout: int =" in system_prompt
    assert "rotaryMode: str =" in system_prompt
    assert "tl_dtype: str =" in system_prompt
    assert "total_elements: int =" in system_prompt
    assert "### OFFICIAL PROTO CONTRACT" in user_prompt
    assert "### OFFICIAL CASES" in user_prompt
    assert "### CAMPAIGN REGIME INSTANTIATIONS" in user_prompt
    assert '"layout": 0' in user_prompt
    assert '"rotaryMode": "half"' in user_prompt
    assert "apply_rotary_pos_emb" in user_prompt

    assert "_load_golden_fn" not in published
    assert "golden.py" not in published
    assert check_npu_entrypoint_purity(published, "apply_rotary_pos_emb") == []

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="apply_rotary_pos_emb", max_rounds=1),
        target_module_path=seed_path,
        source_root=tmp_path,
        l2_batch_evaluator=lambda _c, _a: {},
    )
    for regime in outcome.regime_specs:
        stage_a = runner._run_l0(published, regime)
        assert (
            stage_a.passed is True
        ), f"Regime {regime.regime_tag} failed L0: {stage_a.error_message}"

    mod_spec = importlib.util.spec_from_file_location("_tf_readiness_rope", str(seed_path))
    assert mod_spec is not None and mod_spec.loader is not None
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = mod
    try:
        mod_spec.loader.exec_module(mod)
        assert callable(mod.apply_rotary_pos_emb)
        assert callable(mod._get_kernel)
        assert callable(mod._select_apply_rotary_pos_emb_config)
        assert callable(mod._build_apply_rotary_pos_emb_prim_func)
    finally:
        sys.modules.pop(mod_spec.name, None)

    request_count = len(backend.requests)
    staged_count = len(staged_calls)
    takeover = prepare_marathon_seed(
        operator_name="apply_rotary_pos_emb",
        target_module=seed_path,
        staging_dir=staging_dir,
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=cannbench_repo,
        llm_backend=backend,
        cold_start=False,
        device_id="1",
        op_logger=logging.getLogger("test.readiness.rope_takeover"),
    )
    assert takeover.mode == "incumbent_takeover"
    assert takeover.verification_state == "unverified_incumbent"
    assert takeover.l0_l1_passed is False
    assert takeover.oracle_accuracy_passed is False
    assert takeover.candidate_sha256 == _sha256(published)
    assert takeover.l3_run_id == ""
    assert takeover.l3_receipt_path == ""
    assert takeover.l3_state == ""
    semantic_tags = {regime.regime_tag for regime in outcome.regime_specs}
    assert semantic_tags == default_tags
    assert {regime.regime_tag for regime in takeover.regime_specs} == semantic_tags
    assert set(takeover.blueprints_by_regime) == semantic_tags
    assert all(
        regime.runtime_backend == "tilelang_dsl" for regime in takeover.regime_specs
    )
    assert len(backend.requests) == request_count
    assert len(staged_calls) == staged_count
    assert seed_path.read_text(encoding="utf-8") == published

    forced_cold = prepare_marathon_seed(
        operator_name="apply_rotary_pos_emb",
        target_module=seed_path,
        staging_dir=staging_dir,
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=cannbench_repo,
        llm_backend=backend,
        cold_start=True,
        device_id="1",
        op_logger=logging.getLogger("test.readiness.rope_forced_cold"),
    )
    assert forced_cold.mode == "cold_start"
    assert forced_cold.verification_state == "l3_verified"
    assert forced_cold.candidate_sha256 == _sha256(forced_cold.module_source)
    assert len(backend.requests) == request_count + 1
    assert len(staged_calls) == staged_count + 1
    assert seed_path.read_text(encoding="utf-8").strip() == forced_cold.module_source.strip()


def _create_synthetic_cannbench_repo(repo_root: Path) -> Path:
    tasks_root = repo_root / "tasks"
    op_dir = tasks_root / "level2" / "softmax"
    op_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = tasks_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    (tasks_root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "softmax", "level": "L2", "path": "tasks/level2/softmax"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "950pr.json").write_text(
        json.dumps(
            {
                "level2": {
                    "softmax": {
                        "1": {"baseline_perf_us": 42.5, "t_hw_us": 10.5},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (op_dir / "proto.yaml").write_text(
        "name: softmax\n"
        "inputs:\n"
        "  - name: x\n"
        "outputs:\n"
        "  - name: y\n"
        "attributes:\n"
        "  - name: dim\n"
        "    default: -1\n",
        encoding="utf-8",
    )
    (op_dir / "cases.yaml").write_text(
        "cases:\n"
        "  - case_id: 1\n"
        "    input_shape: [[16, 128]]\n"
        "    dtype: bfloat16\n"
        "    attrs:\n"
        "      dim: -1\n",
        encoding="utf-8",
    )
    (op_dir / "golden.py").write_text(
        "import torch\n"
        "def softmax(x, dim=-1):\n"
        "    return torch.softmax(x, dim=dim)\n",
        encoding="utf-8",
    )
    src_dir = repo_root / "src" / "kernel_eval" / "utils"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    (src_dir.parent / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "compare.py").write_text(
        "from types import SimpleNamespace\n"
        "def compare_tensors(actual, golden, **kwargs):\n"
        "    return SimpleNamespace(passed=True, mere=0.0, mare=0.0, max_diff=0.0, error_msg='')\n",
        encoding="utf-8",
    )
    return repo_root


def test_cannbench_pure_tasks_layout_and_rejects_legacy_operators_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TileFoundry Ascend uses cann-bench tasks/+src/ and cold-starts layout-only ops."""
    repo = _create_synthetic_cannbench_repo(tmp_path / "cann_bench")
    monkeypatch.setenv("TILEFOUNDRY_NPU_CANNBENCH_REPO", str(repo))

    assert resolve_cannbench_repo() == repo.resolve()
    assert resolve_cannbench_tasks_root(repo) == (repo / "tasks").resolve()

    reg = CannbenchOpsRegistry(tasks_root=repo)
    assert reg.has_operator("softmax")
    assert reg.resolve("softmax").task == "level2/softmax"

    baseline_us, t_hw_us = lookup_case_metadata("softmax", 1, cannbench_repo=repo)
    assert baseline_us == pytest.approx(42.5)
    assert t_hw_us == pytest.approx(10.5)

    dims = lookup_case_dims("softmax", 1, cannbench_repo=repo)
    assert dims["dim"] == -1
    assert dims["M"] == 16 and dims["D"] == 128

    static_spec = lookup_case_static_spec("softmax", 1, cannbench_repo=repo)
    assert static_spec["tl_dtype"] == "bfloat16"

    extractor = OpAdvisoryExtractor(repo)
    assert extractor.ops_root == (repo / "tasks")
    advisory = extractor.extract("softmax")
    assert advisory.op_name == "softmax"

    (repo / "tasks" / "suite_manifest.json").unlink()
    assert not (tmp_path / "examples" / "ascend950pr_ops").exists()
    _install_stage0_gate_fakes(monkeypatch, worktree=tmp_path)
    seed_path = tmp_path / "seed_tasks" / "softmax" / "kernels.py"
    seed_outcome = prepare_marathon_seed(
        operator_name="softmax",
        target_module=seed_path,
        staging_dir=tmp_path / "stage0_candidates",
        worktree=tmp_path,
        env_file=None,
        cannbench_repo=repo,
        llm_backend=_ScriptedBackend(_SOFTMAX_SEED_MODULE),
        cold_start=False,
        device_id="1",
        op_logger=logging.getLogger("test.readiness.softmax_seed"),
    )
    assert seed_outcome.mode == "cold_start"
    assert seed_outcome.verification_state == "l3_verified"
    assert seed_outcome.candidate_sha256 == _sha256(seed_outcome.module_source)
    assert {regime.regime_tag for regime in seed_outcome.regime_specs} == {"reduce_tail"}
    seed_src = seed_path.read_text(encoding="utf-8")
    assert seed_src.strip() == seed_outcome.module_source.strip()
    assert (seed_path.parent / "__init__.py").is_file()
    assert "_load_golden_fn" not in seed_src
    assert "tilelang.compile(" in seed_src
    assert "kernel = softmax" in seed_src
    assert "/ \"operators\" /" not in seed_src

    seed_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="softmax", max_rounds=1),
        target_module_path=seed_path,
        source_root=tmp_path,
        l2_batch_evaluator=lambda _c, _a: {},
    )
    for regime in seed_outcome.regime_specs:
        stage_a = seed_runner._run_l0(seed_src, regime)
        assert (
            stage_a.passed is True
        ), f"Regime {regime.regime_tag} failed L0: {stage_a.error_message}"

    shim_dir = ensure_npu_compare_shim_dir(tmp_path / "shim_tasks", cannbench_repo=repo)
    shim_src = (shim_dir / "npu_compare.py").read_text(encoding="utf-8")
    assert 'cann_repo / "src"' in shim_src
    assert "vendor" not in shim_src

    legacy_only = tmp_path / "legacy_repo"
    (legacy_only / "operators" / "level2" / "softmax").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_cannbench_tasks_root(legacy_only)


def test_marathon_env_preserves_cannbench_mcp_keys_and_prefers_cann_bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify marathon env discovery finds cann-bench and preserves CANNBENCH keys."""
    ws = tmp_path / "ws"
    worktree = ws / "TileFoundry"
    worktree.mkdir(parents=True)
    (ws / "cann-bench" / "tasks").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)

    monkeypatch.delenv("TILEFOUNDRY_NPU_CANNBENCH_REPO", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)

    discovered = discover_marathon_environment(
        operator_name="softmax",
        worktree=worktree,
        state_dir=state_dir,
    )
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_REPO"] == str(ws / "cann-bench")

    token_file = tmp_path / "token.txt"
    env_path = state_dir / "env.sh"
    env_path.write_text(
        "\n".join(
            [
                f"TILEFOUNDRY_NPU_CANNBENCH_REPO={ws / 'cann-bench'}",
                "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL=https://example.invalid/mcp",
                f"TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE={token_file}",
                "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION=v2",
                "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID=pool-950pr",
                "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN=agg-123",
                "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT=1",
                "TILEFOUNDRY_NPU_CANNBENCH_FUTURE_FLAG=yes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_env = ensure_marathon_env_file(
        operator_name="softmax",
        worktree=worktree,
        state_dir=state_dir,
        env_file=env_path,
    )
    parsed = parse_simple_env_file(out_env)
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_MCP_URL"] == "https://example.invalid/mcp"
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE"] == str(token_file)
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION"] == "v2"
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_POOL_ID"] == "pool-950pr"
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN"] == "agg-123"
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT"] == "1"
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_FUTURE_FLAG"] == "yes"
