"""Consolidated tests for thin marathon.py, cli.py parsing, and cost advisor off isolation."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent.campaign import cli, marathon
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    build_marathon_arg_parser,
    configure_marathon_process_env,
    discover_marathon_environment,
    ensure_marathon_env_file,
    run_operators_marathon,
)
from tilefoundry_ascend.agent.cost_advisor import (
    CostModelAdvisor,
    NullCostModelAdvisor,
    PluginCostModelAdvisor,
    create_cost_advisor,
)
from tilefoundry_ascend.agent.specs.anchor_specs import parse_simple_env_file
from tilefoundry_ascend.ascend.cost_model import (
    CostGateDecision,
    CostModelMode,
    HybridAscendCostModel,
    evaluate_cost_gate_policy,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger


def test_marathon_module_remains_thin_and_parses_all_cli_flags() -> None:
    """Verify marathon.py stays under 350 LOC and CLI parsers accept required flags."""
    marathon_lines = Path(marathon.__file__).read_text(encoding="utf-8").splitlines()
    assert len(marathon_lines) < 350

    parser = cli.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--marathon",
            "--operator",
            "mla",
            "--max-rounds",
            "12",
            "--env-file",
            "/tmp/env.sh",
            "--device-ids",
            "0,1,2",
            "--source-root",
            "/tmp/src",
        ]
    )
    assert parsed.marathon is True
    assert parsed.operator == "mla"
    assert parsed.max_rounds == 12
    assert parsed.env_file == "/tmp/env.sh"
    assert parsed.device_ids == ("0", "1", "2")
    assert parsed.source_root == "/tmp/src"
    assert parsed.cost_model_mode == "off"
    assert parsed.enable_tilesize_prescreen is True

    no_pre = parser.parse_args(["--operator", "mla", "--no-enable-tilesize-prescreen"])
    assert no_pre.enable_tilesize_prescreen is False

    m_parsed = build_marathon_arg_parser().parse_args(
        ["run", "--operator", "rms_norm", "--device-id", "2", "--device-ids", "0, 3", "--max-rounds", "5"]
    )
    assert m_parsed.command == "run"
    assert m_parsed.operator == "rms_norm"
    assert m_parsed.device_id == "2"
    assert m_parsed.device_ids == ("0", "3")
    assert m_parsed.max_rounds == 5


def test_zero_config_env_discovery_sanitization_and_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sibling path discovery, env.sh sanitization, and process env configuration."""
    for key in list(os.environ):
        if key.startswith("TILEFOUNDRY_NPU_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_API_KEY", "secret-token-stays-out")

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

    env_path = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
    )
    env_text = env_path.read_text(encoding="utf-8")
    assert "secret-token-stays-out" not in env_text
    assert "TILEFOUNDRY_NPU_AGENT_GATES_BIN" not in env_text

    env_path.write_text(
        env_text + "TILEFOUNDRY_NPU_AGENT_GATES_BIN=/tmp/bad\nTILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,2\n",
        encoding="utf-8",
    )
    sanitized = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
        env_file=env_path,
    )
    parsed = parse_simple_env_file(sanitized)
    assert "TILEFOUNDRY_NPU_AGENT_GATES_BIN" not in parsed
    assert parsed["TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS"] == "0,2"

    monkeypatch.setattr(os, "environ", os.environ.copy())
    proc_env = configure_marathon_process_env(operator_name="attention_residual", env_file=sanitized)
    assert proc_env["TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS"] == "0,2"
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert os.environ["TILEFOUNDRY_NPU_ENV_FILE"] == str(sanitized)


def test_ensure_marathon_env_file_preserves_occupancy_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ensure_marathon_env_file retains AgentGates occupancy policy keys."""
    for key in list(os.environ):
        if key.startswith("TILEFOUNDRY_NPU_"):
            monkeypatch.delenv(key, raising=False)
    ws = tmp_path / "ws"
    worktree = ws / "TileFoundry"
    worktree.mkdir(parents=True)
    (ws / "cann-bench" / "tasks").mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    env_path = state_dir / "env.sh"
    env_path.write_text(
        "TILEFOUNDRY_NPU_OCCUPANCY_POLICY=shared_quiet\n"
        "TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS=6\n"
        "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB=16384\n"
        "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT=3\n",
        encoding="utf-8",
    )
    preserved = ensure_marathon_env_file(
        operator_name="attention_residual",
        worktree=worktree,
        state_dir=state_dir,
        env_file=env_path,
    )
    parsed = parse_simple_env_file(preserved)
    assert parsed["TILEFOUNDRY_NPU_OCCUPANCY_POLICY"] == "shared_quiet"
    assert parsed["TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS"] == "6"
    assert parsed["TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB"] == "16384"
    assert parsed["TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT"] == "3"


def test_cli_and_marathon_multi_operator_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cli.main and marathon.main route single and multi-op sequential/round-robin loops."""
    seen_cfgs: list[MarathonConfig] = []
    monkeypatch.setattr(marathon, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])
    monkeypatch.setattr(cli, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])

    assert (
        marathon.main(
            ["--operator", "attention_residual", "--device-id", "1", "--device-ids", "0,1", "--max-rounds", "7"]
        )
        == 0
    )
    assert cli.main(["--marathon", "--operator", "mla", "--device-id", "2", "--max-rounds", "4"]) == 0
    assert [c.operator_name for c in seen_cfgs] == ["attention_residual", "mla"]
    assert seen_cfgs[0].device_ids == ("0", "1")
    assert seen_cfgs[1].max_rounds == 4
    assert seen_cfgs[1].enable_tilesize_prescreen is False

    rr_calls: list[tuple[str, int, bool | None]] = []
    monkeypatch.setattr(
        marathon,
        "run_operator_marathon",
        lambda _base, **kw: (rr_calls.append((str(kw["operator_name"]), int(kw["max_rounds"]), kw.get("resume"))), 0)[1],
    )
    base_cfg = MarathonConfig(operator_name="mla", max_rounds=4)
    assert run_operators_marathon(["mla", "rms_norm"], base_config=base_cfg, mode="round-robin", round_slice=2) == 0
    assert rr_calls == [
        ("mla", 2, None),
        ("rms_norm", 2, None),
        ("mla", 4, True),
        ("rms_norm", 4, True),
    ]
    with pytest.raises(ValueError, match="must not be empty"):
        run_operators_marathon([], base_config=base_cfg)

    env_path = tmp_path / "env.sh"
    env_path.write_text("TILEFOUNDRY_NPU_LLM_BACKEND=mock\n", encoding="utf-8")
    target = tmp_path / "kernels.py"
    target.write_text("def kernel(x):\n    return x\n", encoding="utf-8")

    class _FakeReport:
        def to_dict(self) -> dict[str, object]:
            return {"status": "ok"}

    captured_cfg = []

    class _FakeRunner:
        def __init__(self, *, config: object, **_kw: object) -> None:
            captured_cfg.append(config)

        async def run(self, *, resume: bool) -> _FakeReport:
            _ = resume
            return _FakeReport()

    monkeypatch.setattr(cli, "PerfTargetCampaignRunner", _FakeRunner)
    rc = cli.main(
        [
            "--operator",
            "mla",
            "--checkpoint",
            str(tmp_path / "ckpt.json"),
            "--target-file",
            str(target),
            "--env-file",
            str(env_path),
            "--device-id",
            "0",
            "--device-ids",
            "1,2",
            "--max-rounds",
            "3",
            "--source-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured_cfg[0].max_rounds == 3
    assert captured_cfg[0].enable_tilesize_prescreen is False
    assert captured_cfg[0].device_ids == ("1", "2")


def test_cost_advisor_off_isolation_and_shadow_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cost_model_mode='off' never invokes HybridAscendCostModel and shadow never prunes."""
    default_decision = evaluate_cost_gate_policy()
    assert default_decision.mode == CostModelMode.OFF
    assert default_decision.engaged is False

    def _forbid(*_a: object, **_k: object) -> object:
        raise AssertionError("HybridAscendCostModel must not run in off mode")

    monkeypatch.setattr(HybridAscendCostModel, "__init__", _forbid)
    monkeypatch.setattr(HybridAscendCostModel, "predict", _forbid)

    advisor = create_cost_advisor("off", calibration_path=tmp_path / "calib.jsonl")
    assert isinstance(advisor, NullCostModelAdvisor)
    assert isinstance(advisor, CostModelAdvisor)
    assert advisor.underlying_cost_model is None
    assert advisor.predictions_invoked == 0
    assert advisor.sync_experience_lake(tmp_path) == {}
    assert advisor.predict_candidate_us({"cube_flops": 1e9, "gm_bytes": 1e6}) is None
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIC.PIPE_M": 400, "AIV0.MTE2": 1200},
        gm_roundtrip_bytes=32768,
        grid_blocks=28,
        waterlines={},
    )
    assert advisor.predict_anchor_mean_us(ledger, anchors=[]) == 0.0
    assert advisor.build_prompt_guidance(decision=default_decision, base_cfg={"bm": 64}, dims={"M": 256}) == ""
    kept, rejected = advisor.filter_stage_a_passers(
        ["c1", "c2"],
        decision=default_decision,
        parent_pred_us=10.0,
        parent_axes=None,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={"c1": 500.0, "c2": 5.0},
        candidate_axes_by_id={},
        candidate_workspace_stages_by_id={},
        creative_ids=(),
    )
    assert kept == ["c1", "c2"] and rejected == []
    advisor.observe_surrogate_batch(observations=[{"workload": {"cube_flops": 1e9}, "measured_us": 20.0}])
    assert not (tmp_path / "calib.jsonl").exists()
    assert advisor.resolve_dominant_bound(ledger, None, fallback_cost_pred=SimpleNamespace(dominant_bound="x")) == "memory_bound"

    monkeypatch.undo()
    shadow = create_cost_advisor("shadow", operator_name="rms_norm", calibration_path=tmp_path / "shadow_calib.jsonl")
    assert isinstance(shadow, PluginCostModelAdvisor)
    sdec = shadow.evaluate_round_policy(enable_prescreen_flag=True, bucket_clean_samples=30, out_of_sample_spearman_rho=0.7)
    assert sdec.engaged is False and sdec.allow_prompt_guidance is False and sdec.allow_candidate_pruning is False
    forced = CostGateDecision(CostModelMode.ADVISORY, True, "forced", True, True)
    assert shadow.build_prompt_guidance(decision=forced, base_cfg={"bm": 64}, dims={"M": 256}) == ""
    s_kept, s_rej = shadow.filter_stage_a_passers(
        ["c1"],
        decision=forced,
        parent_pred_us=10.0,
        parent_axes=None,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={"c1": 100.0},
        candidate_axes_by_id={},
        candidate_workspace_stages_by_id={},
        creative_ids=(),
    )
    assert s_kept == ["c1"] and s_rej == []
