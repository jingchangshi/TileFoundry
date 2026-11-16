"""Offline unit tests for WP4 dual-loop orchestrator, prescreener, funnel, and two-lane beam."""

import asyncio
import json

from tilefoundry_ascend.agent import (
    AgyCliLLMBackend,
    BayesianRuleTracker,
    BeamCandidate,
    GateFunnel,
    GenerationRequest,
    MockLLMBackend,
    PortfolioRunner,
    StaticDynamicResidual,
    TileConfigCandidate,
    TwoLaneBeamSearch,
    prescreen_tile_configs,
)
from tilefoundry_ascend.agent.orchestrator import (
    CREATIVE_RESPONSE_SCHEMA,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_creative_payload,
    run_stage_a_static,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

_SEED_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_OVERSIZED_UB_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(X: T.Tensor((2, 256, 128), "float32"), Y: T.Tensor((2, 256, 128), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256, 128), "float32")
            b_ub = T.alloc_shared((256, 128), "float32")
            for k in T.Pipelined(2, num_stages=2):
                T.copy(X[k, 0:256, 0:128], a_ub)
                T.copy(a_ub, b_ub)
                T.copy(b_ub, Y[k, 0:256, 0:128])
    return main
"""

_CREATIVE_SSBUF_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            ss[sid] = T.Cast("uint32", 0)
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""


def test_agy_backend_command_isolation_and_schema_extraction(monkeypatch) -> None:
    """Verify AgyCliLLMBackend strips --print/--add-dir, routes models, and isolates env."""
    monkeypatch.setenv("AGY_CONVERSATION_ID", "secret-conv")
    monkeypatch.setenv("MCP_TOKEN", "secret-mcp")
    monkeypatch.setenv("CLAUDECODE", "1")
    backend = AgyCliLLMBackend(
        bin_path="/bin/true",
        common_args="--print --input-format text --add-dir /tmp --sandbox --mode plan",
    )
    req_exploit = GenerationRequest("exploit_rule", "sys", "usr")
    cmd_exploit = backend.build_command(req_exploit, "PROMPT_TEXT")
    assert "gemini-4-hydrogen-medium" in cmd_exploit
    assert "--add-dir" not in cmd_exploit
    assert "--print=PROMPT_TEXT" in cmd_exploit
    assert "--print" not in cmd_exploit

    req_explore = GenerationRequest("explore_creative", "sys", "usr", CREATIVE_RESPONSE_SCHEMA)
    cmd_explore = backend.build_command(req_explore, "CREATIVE_PROMPT")
    assert "gemini-4-hydrogen-high" in cmd_explore
    assert "--output-format" in cmd_explore
    assert "--json-schema" in cmd_explore

    sanitized = backend._sanitized_env("/tmp/empty_box")
    assert "AGY_CONVERSATION_ID" not in sanitized
    assert "MCP_TOKEN" not in sanitized
    assert "CLAUDECODE" not in sanitized
    assert sanitized["PWD"] == "/tmp/empty_box"

    envelope = json.dumps(
        {
            "structured_output": {
                "physical_hypothesis": "Reduce FixPipe drain stall",
                "target_local_metric": "aic_fixpipe",
                "dsl_source": "```python\nx = 1\n```",
            }
        }
    )
    parsed = extract_creative_payload(envelope)
    assert parsed["physical_hypothesis"] == "Reduce FixPipe drain stall"
    assert parsed["dsl_source"] == "x = 1"


def test_tilesize_prescreen_and_stage_a_early_reject() -> None:
    """Verify dry-run tile prescreener and Stage A reject infeasible configs without bisheng."""
    cands = [
        TileConfigCandidate(bm=15, bn=64, bk=64),
        TileConfigCandidate(bm=256, bn=512, bk=128),
        TileConfigCandidate(bm=64, bn=128, bk=64),
        TileConfigCandidate(bm=128, bn=128, bk=64),
    ]
    ranked = prescreen_tile_configs(cands)
    assert len(ranked) == 2
    assert ranked[0].candidate.bm == 128

    compile_called = [False]

    def _forbid_compile(_src: str) -> StageBResult:
        compile_called[0] = True
        return StageBResult(passed=True)

    funnel = GateFunnel(compile_runner=_forbid_compile)
    res = funnel.evaluate(_OVERSIZED_UB_DSL, factory_name="make_kernel")
    assert res.passed is False
    assert res.failed_stage == "stage_a"
    assert compile_called[0] is False


def test_portfolio_runner_k2_rule_plus_creative_repair_and_two_lane_beam() -> None:
    """Verify K=2 Rule + 1 Rule-Blind Creative round, syntax repair, and two-lane beam."""
    stage_a_seed = run_stage_a_static(_SEED_DSL, factory_name="make_kernel")
    assert stage_a_seed.passed is True and stage_a_seed.ledger is not None
    seed_ledger = stage_a_seed.ledger
    seed_residual = compute_static_dynamic_residual(
        seed_ledger,
        {"dur_us": 40.0, "aic_mad": 0.45, "aiv_vec": 0.10, "aiv_mte2": 0.25},
    )
    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SEED_DSL,
        track="developer",
        ledger=seed_ledger,
        residual=seed_residual,
        latency_us=40.0,
        accuracy_passed=True,
    )

    broken_syntax = "def broken(:\n    pass"
    creative_json = json.dumps(
        {
            "physical_hypothesis": "Move cross-core metadata into 3 KiB SSBUF",
            "target_local_metric": "sync_stall_ratio",
            "dsl_source": _CREATIVE_SSBUF_DSL,
        }
    )
    mock_backend = MockLLMBackend(
        responses={
            "exploit_rule": [broken_syntax, _SEED_DSL],
            "syntax_repair": [_SEED_DSL],
            "explore_creative": [creative_json],
        }
    )

    def _mock_device(dsl: str, st_a) -> StageCResult:
        lat = 28.0 if "alloc_ss" not in dsl else 34.0
        tel = {"dur_us": lat, "aic_mad": 0.72, "aiv_vec": 0.20, "aiv_mte2": 0.30}
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=lat,
            residual=compute_static_dynamic_residual(st_a.ledger, tel),
            telemetry=tel,
            run_id="mock_run",
        )

    tracker = BayesianRuleTracker.from_seed_yaml()
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_mock_device,
    )
    beam = TwoLaneBeamSearch(exploit_width=2, stepping_stone_width=2)
    runner = PortfolioRunner(
        backend=mock_backend,
        funnel=funnel,
        tracker=tracker,
        beam=beam,
        k_rule_workers=2,
        max_syntax_repairs=1,
    )

    report = asyncio.run(runner.run_round(parent, round_idx=1, factory_name="make_kernel"))
    assert len(report.spawned_workers) == 3
    assert len(report.evaluated_candidates) == 3
    assert report.repaired_worker_ids == ["r1_rule_0"]

    creative_reqs = [c for c in mock_backend.calls if c.role == "explore_creative"]
    assert len(creative_reqs) == 1
    creative_prompt = creative_reqs[0].system_prompt + "\n" + creative_reqs[0].user_prompt
    for rule in tracker.active_rules():
        assert rule.rule_id not in creative_prompt

    assert mock_backend.routed_models.count("gemini-4-hydrogen-high") == 1
    assert mock_backend.routed_models.count("gemini-4-hydrogen-medium") == 3

    assert report.exploit_beam[0].latency_us == 28.0
    assert any("T.alloc_ss" in c.structural_signature for c in report.stepping_stone_beam)
