"""Tests for AgentGates transport, git worktree passthrough, env cleaning, and card preflight."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.npu import ag_transport
from tilefoundry_ascend.agent.specs.anchor_specs import parse_simple_env_file


def _init_repo_with_worktree(base: Path) -> tuple[Path, Path]:
    main_repo = base / "workspace" / "TileFoundry"
    main_repo.mkdir(parents=True)
    subprocess.run(["git", "init", str(main_repo)], check=True, capture_output=True, text=True)
    (main_repo / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(main_repo), "add", "README.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(main_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wt_repo = base / "worktrees" / "TileFoundry-wt"
    wt_repo.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "add", "-b", "wt-branch", str(wt_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    return main_repo, wt_repo


def _write_ag_config(
    cfg_dir: Path,
    *,
    source_root: Path,
    allow_worktrees: bool | None = None,
) -> Path:
    evidence_root = cfg_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    lines = [
        f'evidence_root = "{evidence_root}"',
        "",
        "[workspaces.tilefoundry]",
        f'source_root = "{source_root}"',
        'adapters = ["tilefoundry-npu"]',
    ]
    if allow_worktrees is not None:
        lines.append(f"allow_worktrees = {'true' if allow_worktrees else 'false'}")
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg_path


def _assert_flag_value(cmd: list[str], flag: str, expected: str) -> None:
    assert flag in cmd
    idx = cmd.index(flag)
    assert idx + 1 < len(cmd)
    assert cmd[idx + 1] == expected


def _completed(
    cmd: list[str], payload: dict[str, object], rc: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, rc, stdout=json.dumps(payload), stderr=stderr)


def test_git_common_dir_and_workspace_worktree_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve git common dir with poisoned GIT_DIR and honor allow_worktrees."""
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    unrelated = tmp_path / "unrelated.git"
    unrelated.mkdir()
    monkeypatch.setenv("GIT_DIR", str(unrelated))

    main_common = ag_transport._git_common_dir(main_repo)
    assert main_common == (main_repo / ".git").resolve()
    assert ag_transport._git_common_dir(wt_repo) == main_common
    assert ag_transport._git_common_dir(tmp_path / "nonexistent") is None
    plain = tmp_path / "plain"
    plain.mkdir()
    assert ag_transport._git_common_dir(plain) is None

    cfg_ok = _write_ag_config(tmp_path / "allowed", source_root=main_repo)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_ok))
    assert ag_transport._agent_gates_workspace_name(main_repo) == "tilefoundry"
    assert ag_transport._agent_gates_workspace_name(wt_repo) == "tilefoundry"
    assert ag_transport._agent_gates_source_override("tilefoundry", main_repo) is None
    assert ag_transport._agent_gates_source_override("tilefoundry", wt_repo) == str(
        wt_repo.resolve()
    )

    cfg_no = _write_ag_config(tmp_path / "disallowed", source_root=main_repo, allow_worktrees=False)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_no))
    assert ag_transport._agent_gates_workspace_name(main_repo) == "tilefoundry"
    with pytest.raises(ValueError, match="No AgentGates workspace"):
        ag_transport._agent_gates_workspace_name(wt_repo)

    monkeypatch.setenv("AGENT_GATES_CONFIG", str(tmp_path / "missing.toml"))
    assert ag_transport._agent_gates_source_override("tilefoundry", wt_repo) is None


def test_resolve_agent_gates_bin_discovery_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover agent-gates from explicit env, PATH, env.sh ancestor venv, and sibling venv."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    ag_root = tmp_path / "custom_ag" / "AgentGates"
    ag_bin = ag_root / ".venv" / "bin" / "agent-gates"
    ag_bin.parent.mkdir(parents=True)
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    deep_env = ag_root / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh"
    deep_env.parent.mkdir(parents=True)
    deep_env.write_text("TILEFOUNDRY_NPU_CANNBENCH_ARCH=ascend950\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_ENV_FILE", str(deep_env))
    assert ag_transport._resolve_agent_gates_bin() == str(ag_bin)

    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    sibling_bin = main_repo.parent / "AgentGates" / ".venv" / "bin" / "agent-gates"
    sibling_bin.parent.mkdir(parents=True)
    sibling_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    assert ag_transport._resolve_agent_gates_bin(wt_repo) == str(sibling_bin)

    isolated = tmp_path / "iso" / "TileFoundry"
    isolated.mkdir(parents=True)
    iso_sibling = tmp_path / "iso" / "AgentGates" / ".venv" / "bin" / "agent-gates"
    iso_sibling.parent.mkdir(parents=True)
    iso_sibling.write_text("#!/bin/sh\n", encoding="utf-8")
    assert ag_transport._resolve_agent_gates_bin(isolated) == str(iso_sibling)

    explicit = tmp_path / "explicit_ag"
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", str(explicit))
    assert ag_transport._resolve_agent_gates_bin(isolated) == str(explicit)

    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/agent-gates" if name == "agent-gates" else None)
    assert ag_transport._resolve_agent_gates_bin(isolated) == "/usr/local/bin/agent-gates"


def test_run_v2_gate_forwards_source_for_worktree_only_and_cleans_cli_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward --source on submit and verify-requirements only when root differs from workspace."""
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    cfg_path = _write_ag_config(tmp_path / "cfg", source_root=main_repo)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_path))
    monkeypatch.setenv("PYTHONPATH", "/poison/pythonpath")
    monkeypatch.setenv("TILEFOUNDRY_NPU_SECRET", "strip-me")

    recorded: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd_list = [str(part) for part in cmd]
        if cmd_list[0] == "git":
            return subprocess.CompletedProcess(cmd_list, 1, "", "")
        recorded.append((cmd_list, dict(kwargs["env"])))
        sub = cmd_list[1]
        if sub == "doctor":
            return _completed(cmd_list, {"status": "PASS"})
        if sub == "submit":
            return _completed(cmd_list, {"execution_id": "exec-wt-1", "device_id": "0"})
        if sub == "wait":
            return _completed(cmd_list, {"status": {"state": "FINISHED", "run_id": "run-wt-1"}})
        if sub == "report":
            return _completed(cmd_list, {"receiptPath": str(tmp_path / "receipt.json"), "gates": []})
        assert sub == "verify-requirements"
        return _completed(cmd_list, {"satisfied": True})

    monkeypatch.setattr(subprocess, "run", _fake_run)

    wt_exec = ag_transport._run_v2_gate(
        agent_gates_bin="agent-gates",
        root=wt_repo,
        workspace="tilefoundry",
        gate_id="l4-cannbench",
        target="op/task.py",
        candidate_path="cand.py",
        timeout_sec=30.0,
    )
    assert wt_exec.verified is True
    wt_by_sub = {cmd[1]: cmd for cmd, _env in recorded}
    _assert_flag_value(wt_by_sub["submit"], "--source", str(wt_repo.resolve()))
    _assert_flag_value(wt_by_sub["verify-requirements"], "--source", str(wt_repo.resolve()))
    for _cmd, env in recorded:
        assert "PYTHONPATH" not in env
        assert "TILEFOUNDRY_NPU_SECRET" not in env
        assert env.get("PYTHONNOUSERSITE") == "1"

    recorded.clear()
    main_exec = ag_transport._run_v2_gate(
        agent_gates_bin="agent-gates",
        root=main_repo,
        workspace="tilefoundry",
        gate_id="l4-cannbench",
        target="op/task.py",
        candidate_path="cand.py",
        timeout_sec=30.0,
    )
    assert main_exec.verified is True
    main_by_sub = {cmd[1]: cmd for cmd, _env in recorded}
    assert "--source" not in main_by_sub["submit"]
    assert "--source" not in main_by_sub["verify-requirements"]


def test_env_cleaning_and_temp_env_file_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strip host thread/secret keys in subprocess env and keep API key tokens out of env files."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ATEN_CPU_CAPABILITY",
        "TORCH_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "TILELANG_DISABLE_CACHE",
        "USE_CUDA",
        "ASCEND_RT_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("TILEFOUNDRY_NPU_SECRET", "drop")
    monkeypatch.setenv("KEEP_HOST_VAR", "keep")

    cleaned = ag_transport._clean_agent_gates_subprocess_env({"CANNBENCH_CACHE_DIR": "/tmp/cb"})
    for stripped in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ATEN_CPU_CAPABILITY",
        "TORCH_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "TILELANG_DISABLE_CACHE",
        "USE_CUDA",
        "ASCEND_RT_VISIBLE_DEVICES",
        "TILEFOUNDRY_NPU_SECRET",
    ):
        assert stripped not in cleaned
    assert cleaned["KEEP_HOST_VAR"] == "keep"
    assert cleaned["CANNBENCH_CACHE_DIR"] == "/tmp/cb"
    assert cleaned["PYTHONNOUSERSITE"] == "1"

    base_env = tmp_path / "base.env"
    base_env.write_text(
        "export TILEFOUNDRY_NPU_LLM_API_KEY=raw-secret\n"
        "TILEFOUNDRY_NPU_LLM_API_KEY_ENV=MY_API_KEY\n"
        "TILEFOUNDRY_NPU_AGENT_GATES_BIN=/old/bin\n"
        "TILEFOUNDRY_NPU_CANNBENCH_ARCH=ascend950\n",
        encoding="utf-8",
    )
    dest = ag_transport._write_temp_env_file(
        base_env,
        tmp_path / "out.env",
        {
            "TILEFOUNDRY_NPU_CANNBENCH_ARCH": "ascend950pr",
            "TILEFOUNDRY_NPU_LLM_API_KEY": "override-secret",
        },
    )
    parsed = parse_simple_env_file(dest)
    assert parsed["TILEFOUNDRY_NPU_CANNBENCH_ARCH"] == "ascend950pr"
    assert parsed["TILEFOUNDRY_NPU_LLM_API_KEY_ENV"] == "MY_API_KEY"
    assert "TILEFOUNDRY_NPU_LLM_API_KEY" not in parsed
    assert "TILEFOUNDRY_NPU_AGENT_GATES_BIN" not in parsed
    assert "raw-secret" not in dest.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="separate environment variable"):
        ag_transport._validate_llm_api_key_env("TILEFOUNDRY_NPU_BAD")
    with pytest.raises(ValueError, match="single-line"):
        ag_transport._format_env_line("KEY", "bad\nline")


def test_run_agent_gates_cmd_retries_transient_blocked_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry transient BLOCKED_ENV and DEPENDENCY_NOT_PASSED with bounded exponential backoff."""
    attempts = 0
    sleeps: list[float] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _completed(cmd, {"status": "BLOCKED_ENV"}, rc=75)
        if attempts == 2:
            return _completed(
                cmd,
                {
                    "status": "FAIL",
                    "gates": [
                        {
                            "gateId": ag_transport.AGENT_GATES_DEVICE_FAST_GATE,
                            "status": "NOT_RUN",
                            "failures": ["DEPENDENCY_NOT_PASSED"],
                        }
                    ],
                },
                rc=1,
            )
        return _completed(cmd, {"status": "PASS"}, rc=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    proc, parsed = ag_transport._run_agent_gates_cmd(
        ["agent-gates", "run"],
        cwd=str(tmp_path),
        env={},
        timeout_sec=30,
        max_attempts=4,
        backoff_sec=2.0,
        sleep_fn=lambda sec: sleeps.append(float(sec)),
    )
    assert proc.returncode == 0
    assert parsed["status"] == "PASS"
    assert attempts == 3
    assert sleeps == [2.0, 4.0]
    ag_transport.wait_for_npu_quiescent(max_retries=1)


def test_probe_host_card_load_and_idle_preflight_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grade cards free > unknown > quiet shared > heavy > quarantined/locked via env_vals."""
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.delenv("AGENT_GATES_DEVICE_LOCK_ROOT", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/npu-smi" if "smi" in cmd else None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    q_hash = hashlib.sha256(b"npu:2").hexdigest()
    (lock_root / f"{q_hash}.lock.quarantine.json").write_text('{"reason": "busy"}\n', encoding="utf-8")
    l_hash = hashlib.sha256(b"npu:4").hexdigest()
    lock_4 = lock_root / f"{l_hash}.lock"
    lock_4.write_text("", encoding="utf-8")
    fd4 = os.open(lock_4, os.O_RDWR)

    retry_215 = {"done": False}

    def _fake_smi(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "-c" not in cmd
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
                cmd, 0, stdout="NPU ID : 0\nProcess id:11 Process memory(MB):10000\n", stderr=""
            )
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "1"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="NPU ID : 1\nProcess id:22 Process memory(MB):4096\n", stderr=""
            )
        if not retry_215["done"]:
            retry_215["done"] = True
            return subprocess.CompletedProcess(cmd, 215, stdout="", stderr="transient")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="NPU ID : 3\nNo process in device\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_smi)
    env_vals = {
        "AGENT_GATES_DEVICE_LOCK_ROOT": str(lock_root),
        "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB": "8192",
    }
    try:
        fcntl.flock(fd4, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ag_transport._probe_host_card_load("0", env_vals=env_vals) == (2, 10000, 1)
        assert ag_transport._probe_host_card_load("1", env_vals=env_vals) == (1, 4096, 1)
        assert ag_transport._probe_host_card_load("2", env_vals=env_vals) == (3, 999999, 999999)
        assert ag_transport._probe_host_card_load("3", env_vals=env_vals) == (0, 0, 0)
        assert ag_transport._probe_host_card_load("4", env_vals=env_vals) == (3, 999999, 999999)
        assert retry_215["done"] is True

        ordered = ag_transport.order_candidates_by_idle_preflight(
            "0", ("0", "1", "2", "3", "4"), env_vals=env_vals
        )
        assert ordered[:3] == ("3", "1", "0")
        assert set(ordered[3:]) == {"2", "4"}
        assert ag_transport.select_idle_campaign_device("0", ("0", "1", "2"), env_vals=env_vals) == "1"

        monkeypatch.setenv("AGENT_GATES_DEVICE_LOCK_ROOT", str(lock_root))
        assert ag_transport._probe_host_card_idle("2") is False
        assert ag_transport._probe_host_card_idle("3") is True
    finally:
        fcntl.flock(fd4, fcntl.LOCK_UN)
        os.close(fd4)

    def _raise_perm(*_args: object, **_kwargs: object) -> int:
        raise PermissionError("unreadable lock")

    monkeypatch.setattr(os, "open", _raise_perm)
    assert ag_transport._probe_host_card_idle("4") is False

    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    assert ag_transport.order_candidates_by_idle_preflight("0", ("0", "1", "2"), excluded={"2"}) == (
        "0",
        "1",
    )


def test_submit_fast_gate_occupancy_exhausted_and_contamination_retest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rotate on occupancy/device_fault, unlink stale response, or raise OccupancyExhaustedError."""
    cfg_path = _write_ag_config(tmp_path / "cfg", source_root=tmp_path)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_path))
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"
    resp_path.write_text('{"status": "stale"}\n', encoding="utf-8")

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(float(sec)))

    seen_devices: list[str] = []
    resp_exists_on_card2: list[bool] = []
    active_dev = {"id": ""}

    def _run_retest(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        sub = cmd[1]
        if sub == "doctor":
            active_dev["id"] = cmd[cmd.index("--device-id") + 1]
            return _completed(cmd, {"status": "PASS"})
        if sub == "submit":
            dev = cmd[cmd.index("--device-id") + 1]
            active_dev["id"] = dev
            seen_devices.append(dev)
            if dev == "2":
                resp_exists_on_card2.append(resp_path.exists())
            return _completed(cmd, {"execution_id": f"exec-{dev}", "device_id": dev})
        if sub == "wait":
            return _completed(cmd, {"status": {"state": "FINISHED", "run_id": f"run-{active_dev['id']}"}})
        if sub == "report":
            dev = active_dev["id"]
            if dev == "1":
                return _completed(
                    cmd,
                    {
                        "status": "BLOCKED_ENV",
                        "receiptPath": str(tmp_path / "r1.json"),
                        "gates": [
                            {
                                "gateId": ag_transport.AGENT_GATES_DEVICE_FAST_GATE,
                                "recordedStatus": "BLOCKED_ENV",
                                "failures": ["BLOCKED_EXIT_CODE: 75"],
                                "observations": {"errorCategory": "occupancy"},
                            }
                        ],
                    },
                )
            resp_path.write_text(
                json.dumps(
                    {
                        "schema": ag_transport.FAST_RESPONSE_SCHEMA,
                        "request_id": "req-occ",
                        "request_sha256": req_sha,
                        "mode": "batch",
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            return _completed(
                cmd,
                {
                    "status": "PASS",
                    "receiptPath": str(tmp_path / "r2.json"),
                    "gates": [
                        {
                            "gateId": ag_transport.AGENT_GATES_DEVICE_FAST_GATE,
                            "observations": {"deviceWindow": {"heldSeconds": 1.25}},
                        }
                    ],
                },
            )
        return _completed(
            cmd,
            {"satisfied": active_dev["id"] == "2", "current_source_identity": {"root": str(tmp_path.resolve())}},
        )

    monkeypatch.setattr(subprocess, "run", _run_retest)
    with caplog.at_level(logging.WARNING):
        raw, final_exec = ag_transport._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-occ",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=60.0,
            device_id="1",
            device_pool=("1", "2"),
        )
    assert seen_devices == ["1", "2"]
    assert resp_exists_on_card2 == [False]
    assert final_exec.device_id == "2"
    assert raw["status"] == "ok"
    assert ag_transport._fast_held_seconds(final_exec.report) == pytest.approx(1.25)
    assert "FAST_GATE_CONTAMINATION_RETEST" in caplog.text

    sleeps.clear()
    occ_dev = {"id": ""}

    def _run_always_busy(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        sub = cmd[1]
        if sub == "doctor":
            return _completed(cmd, {"status": "PASS"})
        if sub == "submit":
            occ_dev["id"] = cmd[cmd.index("--device-id") + 1]
            return _completed(cmd, {"execution_id": f"exec-{occ_dev['id']}", "device_id": occ_dev["id"]})
        if sub == "wait":
            return _completed(cmd, {"status": {"state": "FINISHED", "run_id": f"run-{occ_dev['id']}"}})
        if sub == "report":
            return _completed(
                cmd,
                {
                    "status": "BLOCKED_ENV",
                    "receiptPath": str(tmp_path / "r_occ.json"),
                    "gates": [
                        {
                            "gateId": ag_transport.AGENT_GATES_DEVICE_FAST_GATE,
                            "recordedStatus": "BLOCKED_ENV",
                            "failures": ["BLOCKED_EXIT_CODE: 75"],
                            "observations": {"errorCategory": "occupancy"},
                        }
                    ],
                },
            )
        return _completed(cmd, {"satisfied": False, "failures": ["BLOCKED_EXIT_CODE: 75"]})

    monkeypatch.setattr(subprocess, "run", _run_always_busy)
    with pytest.raises(ag_transport.OccupancyExhaustedError) as exc_info:
        ag_transport._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-occ",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=15.0,
            device_id="0",
            device_pool=("0", "1"),
        )
    err = exc_info.value
    assert err.fault_plane == "INFRA_OCCUPANCY"
    assert err.exit_code == 75
    assert err.attempts == 8
    assert err.receipt_reasons == ("occupancy",)
    assert sleeps == [4.0, 6.0, 8.0]


def test_submit_fast_gate_device_fault_and_rejection_raise_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exclude DEVICE_NOT_AUTHORIZED cards and raise AgentGatesDeviceFaultError when all fail."""
    cfg_path = _write_ag_config(tmp_path / "cfg", source_root=tmp_path)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_path))
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"

    def _run_fault(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        sub = cmd[1]
        if sub == "doctor":
            return _completed(cmd, {"status": "PASS"})
        if sub == "submit":
            dev = cmd[cmd.index("--device-id") + 1]
            if dev == "0":
                return _completed(cmd, {"code": "DEVICE_NOT_AUTHORIZED"}, rc=2, stderr="DEVICE_NOT_AUTHORIZED")
            return _completed(cmd, {"execution_id": "exec-1", "device_id": "1"})
        if sub == "wait":
            return _completed(cmd, {"status": {"state": "FINISHED", "run_id": "run-1"}})
        if sub == "report":
            return _completed(
                cmd,
                {
                    "status": "FAIL",
                    "receiptPath": str(tmp_path / "r_df.json"),
                    "gates": [
                        {
                            "gateId": ag_transport.AGENT_GATES_DEVICE_FAST_GATE,
                            "status": "FAIL",
                            "observations": {"errorCategory": "device_fault", "deviceRejected": True},
                        }
                    ],
                },
            )
        return _completed(cmd, {"satisfied": False, "deviceRejected": True})

    monkeypatch.setattr(subprocess, "run", _run_fault)
    with pytest.raises(ag_transport.AgentGatesDeviceFaultError) as exc_info:
        ag_transport._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-df",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=30.0,
            device_id="0",
            device_pool=("0", "1"),
        )
    assert exc_info.value.fault_plane == "INFRA_DEVICE_FAULT"
    assert exc_info.value.fault_code == "NPU_DEVICE_TSD_FAULT"
    assert exc_info.value.rejected_cards == ("0", "1")
