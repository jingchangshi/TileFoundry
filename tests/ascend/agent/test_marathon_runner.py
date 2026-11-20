"""Offline unit tests for standalone zero-config score-sprint marathon module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import cli, marathon
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    RegimeTargetSpec,
    parse_simple_env_file,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    ExtremeCampaignRunner,
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
    _write_temp_env_file,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    MockLLMBackend,
    create_llm_backend_from_env,
)


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
    assert "TILEFOUNDRY_NPU_LLM_BACKEND" not in content
    assert "TILEFOUNDRY_NPU_AGY_" not in content
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
    assert "TILEFOUNDRY_NPU_AGY_" not in sanitized_text


def test_extreme_runner_rotation_and_resilient_l2(tmp_path: Path) -> None:
    """Verify stagnation-discounted regime rotation and L2 isolation fallback."""
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


@pytest.mark.parametrize("entrypoint", ["dedicated", "campaign_marathon", "campaign"])
@pytest.mark.parametrize("override_backend", [False, True])
def test_campaign_entrypoints_inject_selected_provider(
    tmp_path: Path, monkeypatch, entrypoint: str, override_backend: bool
) -> None:
    """Exercise each entry point through runner construction without live evaluations."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
    ambient = {
        "TILEFOUNDRY_NPU_LLM_BACKEND": "unsupported",
        "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL": "ambient-exploit",
        "TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL": "ambient-explore",
        "TILEFOUNDRY_NPU_LLM_MAX_EFFORT": "low",
    }
    os.environ.update(ambient)
    env_path = tmp_path / "env.sh"
    file_backend = "unsupported" if override_backend else "mock"
    env_path.write_text(
        f"TILEFOUNDRY_NPU_LLM_BACKEND={file_backend}\n"
        "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL=file-exploit\n"
        "TILEFOUNDRY_NPU_LLM_EXPLORE_MODEL=file-explore\n"
        "TILEFOUNDRY_NPU_LLM_ESCALATE_MODEL=file-escalate\n"
        "TILEFOUNDRY_NPU_LLM_EFFORT_CAP=xhigh\n",
        encoding="utf-8",
    )
    os.environ["TILEFOUNDRY_NPU_ENV_FILE"] = str(env_path)
    seen_overrides = []

    def make_backend(overrides):
        seen_overrides.append(overrides)
        return create_llm_backend_from_env(overrides=overrides)

    class Report:
        def to_dict(self):
            return {"status": "ok"}

    class Runner:
        def __init__(self, *, llm_backend, **kwargs):
            assert isinstance(llm_backend, MockLLMBackend)
            self.backend = llm_backend

        async def run(self, *, resume):
            for role in ("exploit_rule", "explore_creative", "escalate_trackb"):
                await self.backend.generate(GenerationRequest(role, "s", "u", effort="xhigh"))
            explore_model = "cli-explore" if override_backend else "file-explore"
            assert self.backend.routed_models == ["cli-exploit", explore_model, "cli-escalate"]
            assert self.backend.routed_efforts == ["xhigh"] * 3
            return Report()

    monkeypatch.setattr(cli, "PerfTargetCampaignRunner", Runner)
    monkeypatch.setattr(cli, "create_llm_backend_from_env", make_backend)
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", Runner)
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _: None)
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_: None)
    monkeypatch.setattr(marathon, "build_prewarmed_l3_runner", lambda **_: None)
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_: None)
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", make_backend)
    target = tmp_path / "kernels.py"
    target.touch()
    args = [
        "--operator", "mla",
        "--source-root", str(tmp_path),
        "--target-file", str(target),
        "--checkpoint", str(tmp_path / "checkpoint.json"),
        "--exploit-model", "cli-exploit",
        "--escalate-model", "cli-escalate",
        "--llm-api-base", "https://provider.example/v1",
    ]
    if override_backend:
        args.extend(["--llm-backend", "mock", "--explore-model", "cli-explore"])
    if entrypoint == "dedicated":
        assert marathon.main(["run", *args, "--env-file", str(env_path)]) == 0
    elif entrypoint == "campaign_marathon":
        assert cli.main(["--marathon", *args, "--env-file", str(env_path)]) == 0
    else:
        assert cli.main(args) == 0
    assert len(seen_overrides) == 1
    assert seen_overrides[0]["TILEFOUNDRY_NPU_LLM_API_BASE"] == "https://provider.example/v1"
    assert all(os.environ[key] == value for key, value in ambient.items())


def test_marathon_env_provider_values_round_trip(tmp_path, monkeypatch) -> None:
    """Preserve generic CLI templates and legacy aliases in both env files."""
    template_key = "TILEFOUNDRY_NPU_LLM_CLI_TEMPLATE"
    api_key_env = "TILEFOUNDRY_NPU_LLM_API_KEY_ENV"
    raw_key = "TILEFOUNDRY_NPU_LLM_API_KEY"
    legacy_key = "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL"
    template = "{bin} --tag 'a#b;$c|d' --path \"C:\\bin\\tool\" --prompt {prompt}"
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.setenv(template_key, template)
    monkeypatch.setenv(api_key_env, "EXAMPLE_PROVIDER_TOKEN")
    monkeypatch.setenv("EXAMPLE_PROVIDER_TOKEN", "example-value-that-stays-in-environment")
    monkeypatch.setenv(raw_key, "example-value-that-stays-in-environment")
    monkeypatch.setenv(legacy_key, "legacy-escalate")
    env_file = ensure_marathon_env_file(
        operator_name="mla", worktree=tmp_path, state_dir=tmp_path
    )
    temp_env = _write_temp_env_file(env_file, tmp_path / "temp.sh", {template_key: template})
    for path in (env_file, temp_env):
        values = parse_simple_env_file(path)
        assert values[template_key] == template
        assert values[api_key_env] == "EXAMPLE_PROVIDER_TOKEN"
        assert values[legacy_key] == "legacy-escalate"
        assert raw_key not in values
        assert "example-value-that-stays-in-environment" not in path.read_text(encoding="utf-8")


def test_marathon_cli_and_watch_round_detection(tmp_path: Path, monkeypatch) -> None:
    """Verify CLI parser defaults, shorthand, and watch events."""
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
