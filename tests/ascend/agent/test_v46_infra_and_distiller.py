"""Unit tests for infra resilience, analytical ratio telemetry, and distiller rules."""

import asyncio
import logging
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import marathon, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    L3AcceptanceResult,
    _resolve_agent_gates_bin,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AstEditChunk,
    FlywheelDistiller,
    distill_causal_rule,
)
from tilefoundry_ascend.agent.kb import BayesianRuleTracker, SymptomFamily, load_seed_rules
from tilefoundry_ascend.agent.kb.schema import validate_rule_predicate
from tilefoundry_ascend.agent.orchestrator import (
    StageAResult,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger

_BASELINE_DSL = """
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
            for kt in T.Pipelined(2, num_stages=1):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_WINNER_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(8):
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


def test_resolve_agent_gates_bin_relative_and_env_file_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _name: None)

    ag_from_env = tmp_path / "env_ag"
    env_bin = ag_from_env / ".venv" / "bin" / "agent-gates"
    env_bin.parent.mkdir(parents=True)
    env_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    env_file = ag_from_env / "env.sh"
    env_file.write_text("TILEFOUNDRY_NPU_CANNBENCH_ARCH=ascend950\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_ENV_FILE", str(env_file))

    fake_module = (
        tmp_path
        / "workspace_only"
        / "TileFoundry"
        / "targets"
        / "ascend"
        / "tilefoundry_ascend"
        / "agent"
        / "campaign"
        / "surrogate_cannbench.py"
    )
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(surrogate_cannbench, "__file__", str(fake_module))

    assert _resolve_agent_gates_bin() == str(env_bin)

    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    sibling_bin = tmp_path / "workspace_only" / "AgentGates" / ".venv" / "bin" / "agent-gates"
    sibling_bin.parent.mkdir(parents=True)
    sibling_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    assert _resolve_agent_gates_bin() == str(sibling_bin)

    sibling_bin.unlink()
    with pytest.raises(FileNotFoundError, match="agent-gates"):
        _resolve_agent_gates_bin()


def test_cost_prescreen_pruned_includes_analytical_ratio_and_fallbacks() -> None:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 20},
        waterlines={},
    )
    candidates = [
        ("rule_0", "src_r0"),
        ("rule_1", "src_r1"),
        ("rule_2", "src_r2"),
        ("creative_0", "src_c0"),
    ]
    checks_with_ratio = asyncio.run(
        run_l0_l1_round_filter(
            candidates,
            patch_fn=lambda s: s,
            l0_checker=lambda _s: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda _s: StageBResult(passed=True),
            stage_a_cost_ranker=lambda _items: [
                ("rule_0", 10.0),
                ("rule_1", 12.0),
                ("rule_2", 35.0),
                ("creative_0", 25.0),
            ],
            cost_prescreen_top_k=2,
            creative_candidate_ids={"creative_0"},
            parent_analytical_us=10.0,
            stage_a_analytical_scorer=lambda cid, _p, _sa: 35.0 if cid == "rule_2" else 18.0,
        )
    )
    results_by_id = {c.candidate_id: c for c in checks_with_ratio}
    assert results_by_id["creative_0"].passed is True
    pruned = results_by_id["rule_2"]
    assert pruned.failed_tier == "cost_prescreen"
    assert pruned.error_message == "cost_prescreen_pruned:pred_us=35.00,analytical_ratio=3.500"

    checks_no_parent = asyncio.run(
        run_l0_l1_round_filter(
            [("rule_0", "src_r0"), ("rule_1", "src_r1"), ("rule_2", "src_r2")],
            patch_fn=lambda s: s,
            l0_checker=lambda _s: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda _s: StageBResult(passed=True),
            stage_a_cost_ranker=lambda _items: [
                ("rule_0", 10.0),
                ("rule_1", 12.0),
                ("rule_2", 25.0),
            ],
            cost_prescreen_top_k=2,
            parent_analytical_us=0.0,
        )
    )
    pruned_fallback = {c.candidate_id: c for c in checks_no_parent}["rule_2"]
    assert pruned_fallback.error_message == "cost_prescreen_pruned:pred_us=25.00"


def test_stage0_oracle_and_marathon_propagate_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def _missing_ag(**_kwargs: object) -> L3AcceptanceResult:
        calls.append(1)
        raise FileNotFoundError("Cannot locate 'agent-gates' binary")

    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", _missing_ag)
    oracle = marathon.build_stage0_oracle_validator(
        operator_name="level4/gqa",
        staging_dir=tmp_path / "staging",
        worktree=tmp_path,
        device_id="0",
        op_logger=logging.getLogger("test.v46.oracle"),
        blocked_env_max_retries=3,
    )
    with pytest.raises(FileNotFoundError, match="agent-gates"):
        oracle("kernel = object()\n")
    assert len(calls) == 1

    monkeypatch.setattr(
        marathon,
        "run_l3_full_contract_via_agent_gates",
        lambda **_kwargs: L3AcceptanceResult(
            passed=False,
            run_id="",
            receipt_path="",
            elapsed_sec=0.0,
            error_message="Cannot locate 'agent-gates' binary.",
            failure_kind="infrastructure",
        ),
    )
    with pytest.raises(FileNotFoundError, match="agent-gates"):
        oracle("kernel = object()\n")

    fake_module = (
        tmp_path
        / "iso"
        / "TileFoundry"
        / "targets"
        / "ascend"
        / "tilefoundry_ascend"
        / "agent"
        / "campaign"
        / "surrogate_cannbench.py"
    )
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(surrogate_cannbench, "__file__", str(fake_module))
    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _name: None)
    cand = tmp_path / "iso" / "TileFoundry" / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        surrogate_cannbench,
        "get_operator_task",
        lambda _op: "level4/attention_residual",
    )
    with pytest.raises(FileNotFoundError, match="agent-gates"):
        surrogate_cannbench.run_l3_full_contract_via_agent_gates(
            operator_name="attention_residual",
            candidate_module_path=cand,
            source_root=tmp_path / "iso" / "TileFoundry",
        )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.setattr(
        marathon,
        "ensure_marathon_env_file",
        lambda **_kwargs: tmp_path / "env.sh",
    )
    monkeypatch.setattr(
        marathon,
        "configure_marathon_process_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        marathon,
        "resolve_campaign_device_plan",
        lambda **_kwargs: ("0", ("0",)),
    )
    monkeypatch.setattr(
        marathon,
        "create_llm_backend_from_env",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        marathon,
        "prepare_marathon_seed",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing env.sh")),
    )
    with pytest.raises(FileNotFoundError, match="missing env.sh"):
        marathon.run_operator_marathon(
            marathon.MarathonConfig(
                operator_name="level1/exp",
                worktree=worktree,
                state_dir=tmp_path / "state",
                auto_bootstrap_seed=True,
                pure_golden=True,
            )
        )


def test_ablation_distiller_populates_symptom_family_and_anti_precondition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CannbenchOpsRegistry, "default", classmethod(lambda cls: cls(tmp_path)))
    chunk = AstEditChunk(
        chunk_id="chunk_0",
        edit_kind="core_parallelism",
        target_path="main:0",
        before_src="with T.Kernel(1):",
        after_src="with T.Kernel(8):",
        ablated_dsl=_WINNER_DSL,
    )
    inferred_rule = distill_causal_rule(chunk, rule_id="R_FLYWHEEL_01_CORE")
    assert inferred_rule.symptom_family == SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL.value
    assert inferred_rule.anti_precondition != "False"
    validate_rule_predicate(inferred_rule.anti_precondition)

    explicit_rule = distill_causal_rule(
        chunk,
        rule_id="R_FLYWHEEL_02_EXPLICIT",
        active_symptom_family=SymptomFamily.SYNC_BARRIER_SERIALIZATION.value,
    )
    assert explicit_rule.symptom_family == SymptomFamily.SYNC_BARRIER_SERIALIZATION.value

    tracker = BayesianRuleTracker.from_seed_yaml()

    def _device_runner(dsl: str, stage_a: StageAResult) -> StageCResult:
        assert stage_a.ledger is not None
        lat = 42.0 if "T.Kernel(8)" in dsl else 85.0
        tel = {"dur_us": lat, "aic_mad": 0.65, "aiv_vec": 0.25, "aiv_mte2": 0.18}
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=lat,
            residual=compute_static_dynamic_residual(stage_a.ledger, tel),
            telemetry=tel,
        )

    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_device_runner,
    )
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=lambda _spec: {
            "baseline_dur_us": 40.0,
            "optimized_dur_us": 12.0,
            "numerical_ok": 1.0,
        },
        min_edit_gain=0.05,
        min_probe_speedup=1.2,
    )
    kb_path = tmp_path / "flywheel_rules.yaml"
    report = distiller.distill_from_pair(
        baseline_dsl=_BASELINE_DSL,
        winner_dsl=_WINNER_DSL,
        baseline_latency_us=110.0,
        winner_latency_us=35.0,
        factory_name="make_kernel",
        kb_output_path=kb_path,
        active_symptom_family=SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    )
    assert report.admitted_rule_ids
    loaded = load_seed_rules(kb_path, require_symptom_family=True)
    admitted = [r for r in loaded if r.rule_id in set(report.admitted_rule_ids)]
    assert admitted
    for rule in admitted:
        assert rule.symptom_family == SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value
        assert rule.anti_precondition != "False"
        validate_rule_predicate(rule.anti_precondition)
