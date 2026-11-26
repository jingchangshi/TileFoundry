"""Offline unit tests for standalone zero-config score-sprint marathon module."""

from __future__ import annotations

import hashlib
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
from tilefoundry_ascend.agent.campaign.campaign_runner import CampaignConfig
from tilefoundry_ascend.agent.campaign.marathon import (
    ExtremeCampaignRunner,
    MarathonConfig,
    build_marathon_arg_parser,
    build_resilient_l2_evaluator,
    discover_marathon_environment,
    ensure_marathon_env_file,
    salvage_unpromoted_beam_winner,
    watch_marathon_rounds,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
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
        content
        + "TILEFOUNDRY_NPU_AGENT_GATES_BIN=/tmp/fake_agent_gates\n"
        + "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,2\n",
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
    assert "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,2" in sanitized_text


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
        return {cid: best}, f"rid_{cid}", "rcp"

    eval_fn = build_resilient_l2_evaluator(
        operator_name="attention_residual",
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="1",
        batch_fn=_flaky_batch,
    )
    with pytest.raises(RuntimeError, match="transient batch timeout"):
        eval_fn([("c1", target), ("c2", target)], anchors)

    def _shared_batch(
        _op,
        cands,
        anc_seq,
        **_kwargs,
    ):
        return {cid: best for cid, _path in cands}, "batch-rid", "rcp"

    shared_fn = build_resilient_l2_evaluator(
        operator_name="attention_residual",
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="1",
        batch_fn=_shared_batch,
    )
    shared = shared_fn([("c1", target), ("c2", target)], anchors)
    assert set(shared) == {"c1", "c2"}
    assert shared.run_ids == {"c1": "batch-rid", "c2": "batch-rid"}


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
    seen_configs: list[CampaignConfig] = []

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
            seen_configs.append(kwargs["config"])

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
    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", lambda **_: None)
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_: None)
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", make_backend)
    target = tmp_path / "kernels.py"
    target.touch()
    args = [
        "--operator", "mla",
        "--source-root", str(tmp_path),
        "--target-file", str(target),
        "--checkpoint", str(tmp_path / "checkpoint.json"),
        "--device-id", "0",
        "--device-ids", "1,2",
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
    assert [config.device_ids for config in seen_configs] == [("1", "2")]
    assert seen_configs[0].device_id == "0"
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
    parsed_default = parser.parse_args(["run", "--operator", "mla"])
    assert parsed_default.device_id == ""
    parsed_run = parser.parse_args(
        ["run", "--operator", "mla", "--device-id", "2", "--device-ids", "0, 3"]
    )
    assert parsed_run.command == "run"
    assert parsed_run.operator == "mla"
    assert parsed_run.device_id == "2"
    assert parsed_run.device_ids == ("0", "3")
    assert parsed_run.retry_device_id == ""

    seen_cfgs: list[MarathonConfig] = []
    monkeypatch.setattr(marathon, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])
    monkeypatch.setattr(cli, "run_operator_marathon", lambda c: (seen_cfgs.append(c), 0)[1])
    assert (
        marathon.main(
            [
                "--operator",
                "attention_residual",
                "--device-id",
                "1",
                "--device-ids",
                "0,1,2",
                "--retry-device-id",
                "6",
            ]
        )
        == 0
    )
    assert cli.main(["--operator", "mla", "--marathon", "--device-id", "2"]) == 0
    assert [c.operator_name for c in seen_cfgs] == ["attention_residual", "mla"]
    assert seen_cfgs[0].device_id == "1"
    assert seen_cfgs[0].device_ids == ("0", "1", "2")
    assert seen_cfgs[0].retry_device_id == "6"
    assert seen_cfgs[1].device_id == "2"
    assert seen_cfgs[1].device_ids == ()

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
                "l3_resolution": "PROMOTED",
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


def test_marathon_l3_runner_forwards_resume_and_authorized_card(
    tmp_path: Path, monkeypatch
) -> None:
    """L3 continuation uses the saved execution and the requested card, never a resubmit."""
    seen: list[dict[str, object]] = []

    def _fake_l3(**kwargs):
        seen.append(dict(kwargs))
        return L3AcceptanceResult(passed=True)

    def callback(*_args) -> None:
        return None

    class _Report:
        def to_dict(self):
            return {"status": "ok"}

    class _Runner:
        def __init__(self, **kwargs):
            self.l3_runner = kwargs["l3_contract_runner"]

        async def run(self, *, resume):
            _ = resume
            res = self.l3_runner(
                tmp_path / "kernels.py",
                resume_execution_id="exec-saved",
                resume_run_id="run-saved",
                device_id="6",
                on_execution_started=callback,
            )
            assert res.passed is True
            return _Report()

    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", _fake_l3)
    monkeypatch.setattr(marathon, "_resolve_worktree", lambda _c: (tmp_path, tmp_path))
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _w: None)
    monkeypatch.setattr(marathon, "_configure_marathon_logging", lambda _p: None)
    monkeypatch.setattr(marathon, "ensure_marathon_env_file", lambda **_k: tmp_path / "env.sh")
    monkeypatch.setattr(marathon, "configure_marathon_process_env", lambda **_k: {})
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", lambda **_k: MockLLMBackend())
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_k: (lambda c, a: {}))
    monkeypatch.setattr(marathon, "resolve_default_experience_lake_dir", lambda **_k: None)
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _Runner)
    (tmp_path / "kernels.py").touch()
    rc = marathon.run_operator_marathon(
        marathon.MarathonConfig(
            operator_name="mla",
            state_dir=tmp_path / "state",
            target_module_path=tmp_path / "kernels.py",
            auto_bootstrap_seed=False,
            salvage_beam_candidates=False,
        )
    )
    assert rc == 0
    assert seen == [
        {
            "operator_name": "mla",
            "candidate_module_path": tmp_path / "kernels.py",
            "source_root": tmp_path,
            "device_id": "6",
            "resume_execution_id": "exec-saved",
            "resume_run_id": "run-saved",
            "on_execution_started": callback,
        }
    ]


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


def test_marathon_entrypoint_resolves_managed_device_pool(
    tmp_path: Path, monkeypatch
) -> None:
    """The managed env pool names the primary card; the old default 1 is not injected."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    seen_configs: list[CampaignConfig] = []
    env_path = tmp_path / "env.sh"
    env_path.write_text(
        "TILEFOUNDRY_NPU_LLM_BACKEND=mock\n"
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,2\n",
        encoding="utf-8",
    )

    class Report:
        def to_dict(self):
            return {"status": "ok"}

    class Runner:
        def __init__(self, *, llm_backend, **kwargs):
            assert isinstance(llm_backend, MockLLMBackend)
            seen_configs.append(kwargs["config"])

        async def run(self, *, resume):
            return Report()

    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", Runner)
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _: None)
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_: None)
    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", lambda **_: None)
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_: None)
    target = tmp_path / "kernels.py"
    target.touch()
    assert (
        marathon.main(
            [
                "run",
                "--operator",
                "mla",
                "--source-root",
                str(tmp_path),
                "--target-file",
                str(target),
                "--checkpoint",
                str(tmp_path / "checkpoint.json"),
                "--env-file",
                str(env_path),
            ]
        )
        == 0
    )
    assert seen_configs[0].device_id == "0"
    assert seen_configs[0].device_ids == ("2",)


def test_marathon_entrypoint_narrows_managed_pool_to_parent_visibility(
    tmp_path: Path, monkeypatch
) -> None:
    """The captured parent visibility narrows the pool after the env pop."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS", raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    seen_configs: list[CampaignConfig] = []
    env_path = tmp_path / "env.sh"
    env_path.write_text(
        "TILEFOUNDRY_NPU_LLM_BACKEND=mock\n"
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,1,2\n",
        encoding="utf-8",
    )

    class Report:
        def to_dict(self):
            return {"status": "ok"}

    class Runner:
        def __init__(self, *, llm_backend, **kwargs):
            assert isinstance(llm_backend, MockLLMBackend)
            seen_configs.append(kwargs["config"])

        async def run(self, *, resume):
            return Report()

    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", Runner)
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _: None)
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_: None)
    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", lambda **_: None)
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_: None)
    target = tmp_path / "kernels.py"
    target.touch()
    assert (
        marathon.main(
            [
                "run",
                "--operator",
                "mla",
                "--source-root",
                str(tmp_path),
                "--target-file",
                str(target),
                "--checkpoint",
                str(tmp_path / "checkpoint.json"),
                "--env-file",
                str(env_path),
            ]
        )
        == 0
    )
    assert seen_configs[0].device_id == "2"
    assert seen_configs[0].device_ids == ()
    assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ


def test_salvage_skips_pending_l4_checkpoint_without_touching_target(
    tmp_path: Path,
) -> None:
    """A pending L4 keeps its submitted candidate; salvage never rewrites it first."""
    target = tmp_path / "kernels.py"
    original = "def incumbent():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "incumbent_source": original,
                "pending_l4": {
                    "target_path": str(target),
                    "candidate_relative": target.name,
                    "candidate_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                    "device_id": "1",
                    "state": "BLOCKED",
                    "blocked_env_category": "occupancy",
                },
                "best_result": {},
                "beam_states": {},
            }
        ),
        encoding="utf-8",
    )

    def _must_not_evaluate(*_args, **_kwargs):
        raise AssertionError("salvage must not evaluate a pending-L4 checkpoint")

    promoted = salvage_unpromoted_beam_winner(
        operator_name="attention_residual",
        checkpoint_path=checkpoint,
        target_module_path=target,
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="1",
        min_surrogate_gain=0.1,
        l2_evaluator=_must_not_evaluate,
        l3_runner=_must_not_evaluate,
    )
    assert promoted is False
    assert target.read_text(encoding="utf-8") == original


def test_discover_marathon_env_seeds_adapter_toolchain_without_leaking_llm_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sibling AgentGates adapter env.sh seeds CANN/Conda paths without LLM keys."""
    for env_key in list(os.environ):
        if env_key.startswith(("TILEFOUNDRY_NPU_", "ASCEND_", "CONDA_")):
            monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("PATH", "")

    ws = tmp_path / "ws"
    worktree = ws / "TileFoundry"
    worktree.mkdir(parents=True)
    cann_setup = tmp_path / "cann" / "set_env.sh"
    cann_setup.parent.mkdir(parents=True)
    cann_setup.write_text("#!/bin/sh\n", encoding="utf-8")
    conda_sh = tmp_path / "conda" / "conda.sh"
    conda_sh.parent.mkdir(parents=True)
    conda_sh.write_text("#!/bin/sh\n", encoding="utf-8")

    adapter_env = ws / "AgentGates" / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh"
    adapter_env.parent.mkdir(parents=True)
    adapter_env.write_text(
        f"TILEFOUNDRY_NPU_CANN_SETUP={cann_setup}\n"
        f"TILEFOUNDRY_NPU_CONDA_SH={conda_sh}\n"
        "TILEFOUNDRY_NPU_CONDA_ENV=tilefoundry-npu\n"
        "TILEFOUNDRY_NPU_CANNBENCH_ARCH=ascend950\n"
        "TILEFOUNDRY_NPU_LLM_BACKEND=agy_cli\n"
        "TILEFOUNDRY_NPU_AGY_BIN=/usr/bin/agy\n"
        "TILEFOUNDRY_NPU_WORK_ROOT=/tmp/wrong_work_root\n"
        "TILEFOUNDRY_NPU_CANNBENCH_TARGETS=wrong/op:kernels.py\n",
        encoding="utf-8",
    )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    discovered = discover_marathon_environment(
        operator_name="rms_norm",
        worktree=worktree,
        state_dir=state_dir,
    )
    assert discovered["TILEFOUNDRY_NPU_CANN_SETUP"] == str(cann_setup.resolve())
    assert discovered["TILEFOUNDRY_NPU_CONDA_SH"] == str(conda_sh.resolve())
    assert discovered["TILEFOUNDRY_NPU_CONDA_ENV"] == "tilefoundry-npu"
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_ARCH"] == "ascend950"
    assert "TILEFOUNDRY_NPU_LLM_BACKEND" not in discovered
    assert "TILEFOUNDRY_NPU_AGY_BIN" not in discovered
    assert discovered["TILEFOUNDRY_NPU_WORK_ROOT"] == str((state_dir / "cannbench-work").resolve())
    assert discovered["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] != "wrong/op:kernels.py"
