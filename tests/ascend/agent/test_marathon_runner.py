"""Offline unit tests for standalone zero-config score-sprint marathon module."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.marathon import (
    ExtremeCampaignRunner,
    FastCreativeAgyBackend,
    MarathonConfig,
    build_marathon_arg_parser,
    build_resilient_l2_evaluator,
    discover_marathon_environment,
    ensure_marathon_env_file,
    watch_marathon_rounds,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest


def test_zero_config_env_discovery_and_existing_env_preservation(tmp_path: Path, monkeypatch) -> None:
    """Verify env discovery synthesizes sibling paths and preserves existing env.sh."""
    for env_key in list(os.environ):
        if env_key.startswith("TILEFOUNDRY_NPU_"):
            monkeypatch.delenv(env_key, raising=False)

    ws = tmp_path / "ws"
    worktree = ws / "TileFoundry"
    worktree.mkdir(parents=True)
    (ws / "cann-bench" / "tasks").mkdir(parents=True)
    (ws / "tilelang" / "build").mkdir(parents=True)
    ag_bin = ws / "AgentGates" / ".venv" / "bin" / "agent-gates"
    ag_bin.parent.mkdir(parents=True)
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    discovered = discover_marathon_environment(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
    )
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_REPO"] == str((ws / "cann-bench").resolve())
    assert discovered["TILEFOUNDRY_NPU_TILELANG_SOURCE"] == str((ws / "tilelang").resolve())
    assert discovered["TILEFOUNDRY_NPU_TILELANG_BUILD"] == str((ws / "tilelang" / "build").resolve())
    assert discovered["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] == str(ag_bin.resolve())
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] == (
        "level4/attention_residual:examples/ascend950pr_ops/attention_residual/kernels.py"
    )
    assert discovered["TILEFOUNDRY_NPU_WORK_ROOT"] == str((state_dir / "cannbench-work").resolve())

    env_path = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
    )
    assert env_path == state_dir / "env.sh"
    assert env_path.is_file()
    content = env_path.read_text(encoding="utf-8")
    assert "TILEFOUNDRY_NPU_LLM_BACKEND=agy_cli" in content
    assert "TILEFOUNDRY_NPU_AGENT_GATES_BIN" not in content

    marker_text = content + "\n# keep-existing-env\n"
    env_path.write_text(marker_text, encoding="utf-8")
    reused = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
        env_file=env_path,
    )
    assert reused == env_path
    assert "# keep-existing-env" in reused.read_text(encoding="utf-8")

    env_path.write_text(
        content + "TILEFOUNDRY_NPU_AGENT_GATES_BIN=/tmp/fake_agent_gates\n",
        encoding="utf-8",
    )
    sanitized = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
        env_file=env_path,
    )
    sanitized_text = sanitized.read_text(encoding="utf-8")
    assert "TILEFOUNDRY_NPU_AGENT_GATES_BIN" not in sanitized_text
    assert "TILEFOUNDRY_NPU_LLM_BACKEND=agy_cli" in sanitized_text


def test_extreme_runner_rotation_fast_backend_and_resilient_l2(tmp_path: Path) -> None:
    """Verify stagnation-discounted regime rotation, xhigh clamp, and L2 isolation fallback."""
    backend = FastCreativeAgyBackend(escalate_model="gemini-4-hydrogen-high")
    req = GenerationRequest(
        role="creative_explore",
        system_prompt="sys",
        user_prompt="usr",
        effort="xhigh",
    )
    assert backend.select_effort(req) == "high"
    assert backend.select_effort(req, effort_override="xhigh") == "high"

    target = tmp_path / "kernels.py"
    target.write_text("def _b1():\n    pass\n", encoding="utf-8")
    r_a = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("_b1",),
        sample_builder_kwargs={},
        description="Regime A",
    )
    r_b = RegimeTargetSpec(
        regime_tag="reg_b",
        builder_candidates=("_b1",),
        sample_builder_kwargs={},
        description="Regime B",
    )
    runner = ExtremeCampaignRunner.__new__(ExtremeCampaignRunner)
    runner._regime_specs = (r_a, r_b)
    runner._live_stag_by_regime = {"reg_a": 4, "reg_b": 0}
    runner._stagnation_discount = 0.45
    runner._perf_target = type("_PT", (), {"target_min_case_speedup": 2.0, "target_roofline_eta": 0.99})()
    runner._is_regime_satisfied = lambda *_args, **_kwargs: False  # type: ignore[assignment]

    anchors = (
        AnchorCaseSpec("level4/attention_residual", 1, "reg_a", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 2, "reg_b", 10.0, 4.0),
    )
    best = CandidateSurrogateResult(
        candidate_id="base",
        operator_name="attention_residual",
        all_passed=True,
        anchors=(
            AnchorEvalMetrics(
                case_num=1,
                regime_tag="reg_a",
                accuracy_passed=True,
                ref_perf_us=10.0,
                t_hw_us=2.0,
                cand_us=4.0,
                speedup=2.5,
                roofline_eta=0.50,
                eq3_score=0.80,
            ),
            AnchorEvalMetrics(
                case_num=2,
                regime_tag="reg_b",
                accuracy_passed=True,
                ref_perf_us=10.0,
                t_hw_us=4.0,
                cand_us=6.0,
                speedup=1.67,
                roofline_eta=0.67,
                eq3_score=0.75,
            ),
        ),
        avg_speedup=2.085,
        min_case_speedup=1.67,
        avg_roofline_eta=0.585,
        surrogate_score=65.0,
    )
    picked = runner._select_active_regime(None, set(), best, anchors)
    assert picked is not None
    assert picked.regime_tag == "reg_b"

    def _flaky_batch(
        _op,
        cands,
        anc_seq,
        **_kwargs,
    ):
        if len(cands) > 1:
            raise RuntimeError("transient batch timeout")
        cid, _ = cands[0]
        return {cid: best}, "rid", "rcp"

    eval_fn = build_resilient_l2_evaluator(
        operator_name="attention_residual",
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="1",
        batch_fn=_flaky_batch,
    )
    merged = eval_fn([("c1", target), ("c2", target)], anchors)
    assert set(merged) == {"c1", "c2"}


def test_marathon_cli_and_watch_round_detection(tmp_path: Path, monkeypatch) -> None:
    """Verify CLI parser defaults/shorthand and watch_marathon_rounds event output."""
    from tilefoundry_ascend.agent.campaign import cli, marathon  # noqa: PLC0415

    parser = build_marathon_arg_parser()
    parsed_run = parser.parse_args(["run", "--operator", "mla", "--device-id", "2"])
    assert parsed_run.command == "run"
    assert parsed_run.operator == "mla"
    assert parsed_run.device_id == "2"

    seen_cfgs: list[MarathonConfig] = []
    monkeypatch.setattr(marathon, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])
    monkeypatch.setattr(cli, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])
    assert marathon.main(["--operator", "attention_residual", "--device-id", "1"]) == 0
    assert cli.main(["--operator", "mla", "--marathon", "--device-id", "2"]) == 0
    assert [c.operator_name for c in seen_cfgs] == ["attention_residual", "mla"]
    assert seen_cfgs[0].device_id == "1"
    assert seen_cfgs[1].device_id == "2"

    parsed_watch = parser.parse_args(
        ["watch", "--base-dir", str(tmp_path), "--once"]
    )
    assert parsed_watch.command == "watch"
    assert parsed_watch.once is True

    ar_dir = tmp_path / "ar"
    ar_dir.mkdir(parents=True)
    ckpt = {
        "operator_name": "attention_residual",
        "baseline_result": {"surrogate_score": 52.0, "anchors": []},
        "best_result": {
            "surrogate_score": 77.5,
            "avg_speedup": 1.64,
            "min_case_speedup": 1.12,
            "avg_roofline_eta": 0.65,
            "anchors": [],
        },
        "rounds": [
            {
                "round_idx": 1,
                "regime_tag": "aligned_prefill",
                "parent_lane": "exploit",
                "creative_effort": "high",
                "l0_l1_passed_ids": ["r1_rule_0"],
                "l0_l1_failed": [],
                "best_round_surrogate_score": 77.5,
                "best_so_far_surrogate_score": 77.5,
                "best_round_speedup": 1.64,
                "best_so_far_speedup": 1.64,
                "promoted": True,
                "rolled_back_l3": False,
            }
        ],
        "promoted_rounds": [1],
        "rolled_back_rounds": [],
        "stagnation_by_regime": {"aligned_prefill": 0},
        "frozen_regimes": [],
        "stop_reason": "in_progress",
    }
    (ar_dir / "checkpoint.json").write_text(json.dumps(ckpt), encoding="utf-8")
    state_file = tmp_path / "last_reported.json"
    out = watch_marathon_rounds(
        state_dirs={"ar": ar_dir},
        state_file=state_file,
        once=True,
    )
    assert len(out["events"]) == 1
    assert out["events"][0]["event_type"] == "round_completed"
    assert out["scoreboard"]["ar"]["rounds_len"] == 1
    assert state_file.is_file()

    cfg = MarathonConfig(operator_name="attention_residual")
    assert cfg.max_rounds == 80
    assert cfg.target_surrogate_score == 99.8


def test_cann_opp_integrity_preference_and_source_root_sibling_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify intact CANN OPP is preferred and source_root.parent resolves sibling repos."""
    for env_key in list(os.environ):
        if env_key.startswith("TILEFOUNDRY_NPU_"):
            monkeypatch.delenv(env_key, raising=False)

    vendor_root = tmp_path / "pri"
    broken_cann = vendor_root / "9.3.0-weekly" / "cann-9.3.0"
    (broken_cann / "opp" / "built-in").mkdir(parents=True)
    broken_setup = broken_cann / "set_env.sh"
    broken_setup.write_text("#!/bin/sh\n", encoding="utf-8")

    intact_cann = vendor_root / "Ascend" / "9.3.0-weekly" / "cann-9.3.0"
    (
        intact_cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
        / "ascend950"
    ).mkdir(parents=True)
    intact_setup = intact_cann / "set_env.sh"
    intact_setup.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(broken_cann))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(broken_cann))

    ws = tmp_path / "workspace"
    src_root = ws / "TileFoundry"
    src_root.mkdir(parents=True)
    (ws / "cann-bench" / "tasks").mkdir(parents=True)
    (ws / "tilelang" / "build").mkdir(parents=True)
    ag_bin = ws / "AgentGates" / ".venv" / "bin" / "agent-gates"
    ag_bin.parent.mkdir(parents=True)
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    external_wt = tmp_path / "scratch" / "worktrees" / "grouped_matmul"
    external_wt.mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)

    discovered = discover_marathon_environment(
        operator_name="level3/grouped_matmul",
        worktree=external_wt,
        state_dir=state_dir,
        source_root=src_root,
    )
    assert discovered["TILEFOUNDRY_NPU_CANN_SETUP"] == str(intact_setup.resolve())
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_REPO"] == str((ws / "cann-bench").resolve())
    assert discovered["TILEFOUNDRY_NPU_TILELANG_SOURCE"] == str((ws / "tilelang").resolve())
    assert discovered["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] == str(ag_bin.resolve())
