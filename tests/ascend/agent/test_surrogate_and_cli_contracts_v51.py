"""Offline contract tests for P1-3b, P1-4, and Blind Spot #3 (v5.1)."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import cli, marathon, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import CampaignConfig
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import PortfolioRunner


class _ClockEvent:
    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self._t = 0.0

    def record(self) -> None:
        self._t = self._clock[0]

    def elapsed_time(self, other: _ClockEvent) -> float:
        return (other._t - self._t) / 1000.0


def test_env_vals_probe_host_card_load_and_preflight_grading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom env_vals lock root and HBM threshold grade cards without os.environ mutation."""
    lock_root = tmp_path / "custom_locks"
    lock_root.mkdir()
    monkeypatch.delenv("AGENT_GATES_DEVICE_LOCK_ROOT", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", raising=False)
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _cmd: "/usr/bin/npu-smi")
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda _s: None)

    q_hash = hashlib.sha256(b"npu:2").hexdigest()
    (lock_root / f"{q_hash}.lock.quarantine.json").write_text('{"reason": "busy"}\n', encoding="utf-8")

    l_hash = hashlib.sha256(b"npu:4").hexdigest()
    lock_4 = lock_root / f"{l_hash}.lock"
    lock_4.write_text("", encoding="utf-8")
    fd4 = os.open(lock_4, os.O_RDWR)

    def _fake_smi(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1:] == ["info", "-m"]:
            stdout = (
                "NPU ID  Slot ID  Chip ID  Chip Phy-ID  Chip Name\n"
                "0       74       0        0            Ascend950PR\n"
                "1       76       0        1            Ascend950PR\n"
                "2       84       0        2            Ascend950PR\n"
                "3       86       0        3            Ascend950PR\n"
                "4       88       0        4            Ascend950PR\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "0"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="NPU ID : 0\nProcess id:11 Process name:p Process memory(MB):10000\n",
                stderr="",
            )
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "1"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="NPU ID : 1\nProcess id:22 Process name:p Process memory(MB):4096\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="NPU ID : 3\nNo process in device\n",
            stderr="",
        )

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_smi)
    env_vals = {
        "AGENT_GATES_DEVICE_LOCK_ROOT": str(lock_root),
        "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB": "8192",
    }
    try:
        fcntl.flock(fd4, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert surrogate_cannbench._probe_host_card_load("0", env_vals=env_vals) == (2, 10000, 1)
        assert surrogate_cannbench._probe_host_card_load("1", env_vals=env_vals) == (1, 4096, 1)
        assert surrogate_cannbench._probe_host_card_load("2", env_vals=env_vals) == (3, 999999, 999999)
        assert surrogate_cannbench._probe_host_card_load("3", env_vals=env_vals) == (0, 0, 0)
        assert surrogate_cannbench._probe_host_card_load("4", env_vals=env_vals) == (3, 999999, 999999)

        ordered = surrogate_cannbench.order_candidates_by_idle_preflight(
            "0",
            ("0", "1", "2", "3", "4"),
            env_vals=env_vals,
        )
        assert ordered[:3] == ("3", "1", "0")
        assert set(ordered[3:]) == {"2", "4"}
        assert surrogate_cannbench.select_idle_campaign_device(
            "0", ("0", "1", "2"), env_vals=env_vals
        ) == "1"
    finally:
        fcntl.flock(fd4, fcntl.LOCK_UN)
        os.close(fd4)


def test_submit_fast_gate_and_batch_attach_selected_card_ext_hbm_mb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_submit_fast_gate records selected card ext HBM and batch populates AnchorEvalMetrics."""
    src_root = tmp_path / "repo"
    src_root.mkdir()
    env_file = tmp_path / "env.sh"
    work_root = tmp_path / "work"
    cann_dir = tmp_path / "cann"
    (cann_dir / "tasks").mkdir(parents=True)
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_dir}\n"
        "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB=8192\n",
        encoding="utf-8",
    )
    cand = src_root / "cand.py"
    cand.write_text("def op():\n    return 1\n", encoding="utf-8")
    anchor = AnchorCaseSpec("level2/softmax", 1, "small", 10.0, 2.0)

    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "tilefoundry")
    monkeypatch.setattr(surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *a, **k: None)
    monkeypatch.setattr(surrogate_cannbench, "prewarm_surrogate_tilelang_cache", lambda *a, **k: None)
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)

    seen_probe_envs: list[dict[str, str] | None] = []

    def _fake_probe(
        card: str,
        *,
        max_quiet_hbm_mb: int | None = None,
        env_vals: dict[str, str] | None = None,
    ) -> tuple[int, int, int]:
        _ = max_quiet_hbm_mb
        seen_probe_envs.append(dict(env_vals) if env_vals is not None else None)
        return (1, 4321, 1) if card == "1" else (2, 12000, 1)

    monkeypatch.setattr(surrogate_cannbench, "_probe_host_card_load", _fake_probe)

    def _fake_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        req_rel = str(kwargs["candidate_path"])
        req_payload = json.loads((src_root / req_rel).read_text(encoding="utf-8"))
        req_sha = hashlib.sha256((src_root / req_rel).read_bytes()).hexdigest()
        resp_path = src_root / str(req_payload["response_path"])
        c0 = req_payload["batches"][0]["candidates"][0]
        resp = {
            "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
            "request_id": req_payload["request_id"],
            "request_sha256": req_sha,
            "mode": "batch",
            "status": "ok",
            "candidates": {
                "softmax": {
                    "c0": {
                        "module_path": c0["module_path"],
                        "sha256": c0["sha256"],
                        "post_sha256": c0["sha256"],
                    }
                }
            },
            "coverage": {
                "expected_candidates": 1,
                "evaluated_candidates": 1,
                "expected_msprof_targets": 0,
                "observed_msprof_targets": 0,
            },
            "operators": {
                "softmax": {
                    "c0": {
                        "candidate_id": "c0",
                        "operator_name": "softmax",
                        "all_passed": True,
                        "avg_speedup": 2.0,
                        "min_case_speedup": 2.0,
                        "avg_roofline_eta": 0.9,
                        "surrogate_score": 80.0,
                        "anchors": [
                            {
                                "case_num": 1,
                                "regime_tag": "small",
                                "accuracy_passed": True,
                                "cand_us": 5.0,
                                "ref_perf_us": 10.0,
                                "t_hw_us": 4.5,
                                "speedup": 2.0,
                                "roofline_eta": 0.9,
                                "eq3_score": 80.0,
                            }
                        ],
                    }
                }
            },
        }
        resp_path.write_text(json.dumps(resp), encoding="utf-8")
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-hbm",
            run_id="run-hbm",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True, "current_source_identity": {"root": str(src_root)}},
            device_id=str(kwargs["device_id"]),
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2)
    by_op, _run_id, _receipt, _held = surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
        [("level2/softmax", [("c0", cand)], [anchor])],
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=("0", "1"),
        prewarm_cache=True,
    )
    assert any(
        env is not None and env.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB") == "8192"
        for env in seen_probe_envs
    )
    assert by_op["level2/softmax"]["c0"].anchors[0].card_ext_hbm_mb == 4321


def test_cli_and_marathon_cost_mode_validation_and_forwarding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject --cost-model-mode always and forward cost/distilled flags to CampaignConfig."""
    with pytest.raises(SystemExit):
        marathon.build_marathon_arg_parser().parse_args(
            ["run", "--operator", "rms_norm", "--cost-model-mode", "always"]
        )
    with pytest.raises(SystemExit):
        cli.build_arg_parser().parse_args(
            ["--operator", "rms_norm", "--cost-model-mode", "always"]
        )

    captured_marathon_cfgs: list[marathon.MarathonConfig] = []
    monkeypatch.setattr(
        cli,
        "run_operator_marathon",
        lambda cfg: captured_marathon_cfgs.append(cfg) or 0,
    )
    rc_cli = cli.main(
        [
            "--operator",
            "rms_norm",
            "--marathon",
            "--cost-disengage-eta",
            "0.12",
            "--cost-slow-gain-threshold",
            "0.02",
        ]
    )
    assert rc_cli == 0
    assert captured_marathon_cfgs[0].cost_disengage_eta == pytest.approx(0.12)
    assert captured_marathon_cfgs[0].cost_slow_gain_threshold == pytest.approx(0.02)

    captured_direct: list[marathon.MarathonConfig] = []
    monkeypatch.setattr(
        marathon,
        "run_operator_marathon",
        lambda cfg: captured_direct.append(cfg) or 0,
    )
    rc_m = marathon.main(
        [
            "run",
            "--operator",
            "rms_norm",
            "--cost-disengage-eta",
            "0.12",
            "--cost-slow-gain-threshold",
            "0.02",
            "--allow-unreviewed-distilled-rules",
        ]
    )
    assert rc_m == 0
    assert captured_direct[0].cost_disengage_eta == pytest.approx(0.12)
    assert captured_direct[0].cost_slow_gain_threshold == pytest.approx(0.02)
    assert captured_direct[0].allow_unreviewed_distilled_rules is True

    monkeypatch.undo()
    work_dir = tmp_path / "wt"
    work_dir.mkdir()
    target_mod = work_dir / "kernels.py"
    target_mod.write_text("def k():\n    return 1\n", encoding="utf-8")
    env_sh = tmp_path / "env.sh"
    env_sh.write_text("TILEFOUNDRY_NPU_LLM_BACKEND=mock\n", encoding="utf-8")

    seen_tracker_flags: list[bool] = []
    seen_campaign_cfgs: list[CampaignConfig] = []

    class _FakeReport:
        best_speedup = 1.0
        best_roofline_eta = 0.5
        best_surrogate_score = 50.0
        final_l4_passed = True

        def to_dict(self) -> dict[str, object]:
            return {"status": "ok"}

    class _FakeExtremeRunner:
        def __init__(self, **kwargs: object) -> None:
            cfg_obj = kwargs["config"]
            assert isinstance(cfg_obj, CampaignConfig)
            seen_campaign_cfgs.append(cfg_obj)

        async def run(self, *, resume: bool) -> _FakeReport:
            _ = resume
            return _FakeReport()

    monkeypatch.setattr(marathon, "_resolve_worktree", lambda _c: (work_dir, work_dir))
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _w: None)
    monkeypatch.setattr(marathon, "ensure_marathon_env_file", lambda **_k: env_sh)
    monkeypatch.setattr(
        marathon,
        "configure_marathon_process_env",
        lambda **_k: {"TILEFOUNDRY_NPU_LLM_BACKEND": "mock"},
    )
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", lambda **_k: MockLLMBackend())
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_k: None)
    monkeypatch.setattr(
        marathon.BayesianRuleTracker,
        "from_experience_lake",
        classmethod(
            lambda cls, _dir, allow_unreviewed_distilled_rules=False, **_k: (
                seen_tracker_flags.append(bool(allow_unreviewed_distilled_rules))
                or cls.from_seed_yaml()
            )
        ),
    )
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _FakeExtremeRunner)

    rc_run = marathon.run_operator_marathon(
        marathon.MarathonConfig(
            operator_name="rms_norm",
            source_root=work_dir,
            worktree=work_dir,
            state_dir=tmp_path / "state",
            target_module_path=target_mod,
            env_file=env_sh,
            auto_bootstrap_seed=False,
            salvage_beam_candidates=False,
            cost_model_mode="coarse_only",
            cost_disengage_eta=0.12,
            cost_slow_gain_threshold=0.02,
            allow_unreviewed_distilled_rules=True,
        )
    )
    assert rc_run == 0
    assert seen_tracker_flags == [True]
    assert len(seen_campaign_cfgs) == 1
    assert seen_campaign_cfgs[0].cost_model_mode == "coarse_only"
    assert seen_campaign_cfgs[0].cost_disengage_eta == pytest.approx(0.12)
    assert seen_campaign_cfgs[0].cost_slow_gain_threshold == pytest.approx(0.02)
    assert seen_campaign_cfgs[0].allow_unreviewed_distilled_rules is True


def test_measure_npu_ring_burst_stats_and_anchor_metrics_roundtrip() -> None:
    """measure_npu_ring_burst_stats reports CV/spike_retries and AnchorEvalMetrics roundtrips."""
    clock = [0.0]

    non_burst_steps = [0]
    non_burst_deltas = [100.0, 20.0, 22.0, 26.0]

    def _non_burst_fn(_x: object) -> None:
        val = non_burst_deltas[min(non_burst_steps[0], len(non_burst_deltas) - 1)]
        non_burst_steps[0] += 1
        clock[0] += val

    nb_med, nb_k, nb_cv, nb_retries = surrogate_cannbench.measure_npu_ring_burst_stats(
        _non_burst_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=10.0,
        first_probe_us=20.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert nb_k == 1
    assert nb_retries == 0
    assert nb_med == pytest.approx(22.0)
    assert nb_cv > 0.0

    k = surrogate_cannbench.compute_ring_burst_k(0, t_hw_us=2.0, first_probe_us=10.0)
    warmup_calls = 1 + min(k, 1)
    idx1 = [0]
    pattern_retry_clean = [
        *([10.0] * warmup_calls),
        *([5.0] * k),
        *([6.0] * k),
        *([25.0] * k),
        *([5.0] * (3 * k)),
    ]

    def _retry_clean_fn(_x: object) -> None:
        val = pattern_retry_clean[min(idx1[0], len(pattern_retry_clean) - 1)]
        idx1[0] += 1
        clock[0] += val

    rc_med, rc_k, rc_cv, rc_retries = surrogate_cannbench.measure_npu_ring_burst_stats(
        _retry_clean_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=10.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert rc_k == k
    assert rc_retries == 1
    assert rc_med == pytest.approx(5.0)
    assert rc_cv == pytest.approx(0.0)

    idx2 = [0]
    noisy_pattern = [
        *([10.0] * warmup_calls),
        *([5.0] * k), *([6.0] * k), *([30.0] * k),
        *([5.5] * k), *([6.5] * k), *([35.0] * k),
        *([5.2] * k), *([6.2] * k), *([40.0] * k),
    ]

    def _all_noisy_fn(_x: object) -> None:
        val = noisy_pattern[min(idx2[0], len(noisy_pattern) - 1)]
        idx2[0] += 1
        clock[0] += val

    n_med, n_k, n_cv, n_retries = surrogate_cannbench.measure_npu_ring_burst_stats(
        _all_noisy_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=10.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert n_k == k
    assert n_retries == 2
    assert n_med < 6.0
    assert n_cv > 0.0

    spec = AnchorCaseSpec("level2/rms_norm", 1, "small", 10.0, 2.0)
    built = surrogate_cannbench.build_anchor_metrics(
        spec,
        accuracy_passed=True,
        cand_us=n_med,
        ring_burst_k=n_k,
        burst_batch_cv=n_cv,
        spike_retries=n_retries,
        card_ext_hbm_mb=2048,
    )
    restored = surrogate_cannbench.AnchorEvalMetrics.from_dict(built.to_dict())
    assert restored.burst_batch_cv == pytest.approx(round(n_cv, 6))
    assert restored.spike_retries == 2
    assert restored.card_ext_hbm_mb == 2048


def test_portfolio_runner_default_disengage_roofline_eta() -> None:
    """PortfolioRunner.plan_workers defaults disengage_roofline_eta to 0.65."""
    sig = inspect.signature(PortfolioRunner.plan_workers)
    assert sig.parameters["disengage_roofline_eta"].default == pytest.approx(0.65)
