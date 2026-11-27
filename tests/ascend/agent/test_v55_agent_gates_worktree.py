"""Unit tests for linked git worktree support in AgentGates gate wrappers."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _bootstrap_isolated_worktree() -> None:
    """Resolve worktree modules directly even if another checkout is installed."""
    root = Path(__file__).resolve().parents[3]
    ascend_root = str(root / "targets" / "ascend")
    if ascend_root not in sys.path:
        sys.path.insert(0, ascend_root)
    pkg_dir = root / "targets" / "ascend" / "tilefoundry_ascend"
    try:
        mod = importlib.import_module("tilefoundry_ascend.agent.campaign.surrogate_cannbench")
        mod_file = getattr(mod, "__file__", "") or ""
        if Path(mod_file).resolve().is_relative_to(root):
            return
    except ImportError:
        pass
    stale = [
        k
        for k in sys.modules
        if k == "tilefoundry_ascend" or k.startswith("tilefoundry_ascend.")
    ]
    for key in stale:
        del sys.modules[key]
    for mod_name, sub_path in (
        ("tilefoundry_ascend", pkg_dir),
        ("tilefoundry_ascend.agent", pkg_dir / "agent"),
        ("tilefoundry_ascend.agent.campaign", pkg_dir / "agent" / "campaign"),
    ):
        mod = types.ModuleType(mod_name)
        mod.__path__ = [str(sub_path)]
        sys.modules[mod_name] = mod


_bootstrap_isolated_worktree()

from tilefoundry_ascend.agent.campaign import surrogate_cannbench  # noqa: E402


def _init_repo_with_worktree(base: Path) -> tuple[Path, Path]:
    """Create a main git checkout and a separate linked git worktree."""
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


def _write_agent_gates_config(
    tmp_path: Path,
    *,
    source_root: Path,
    allow_worktrees: bool | None = None,
) -> Path:
    """Write a minimal AgentGates config TOML for workspace resolution tests."""
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    lines = [
        f'evidence_root = "{evidence_root}"',
        "",
        "[workspaces.tilefoundry]",
        f'source_root = "{source_root}"',
    ]
    if allow_worktrees is not None:
        flag = "true" if allow_worktrees else "false"
        lines.append(f"allow_worktrees = {flag}")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg_path


def _assert_flag_value(cmd: list[str], flag: str, expected: str) -> None:
    """Assert that CLI flag appears in cmd with the expected argument."""
    assert flag in cmd
    idx = cmd.index(flag)
    assert idx + 1 < len(cmd)
    assert cmd[idx + 1] == expected


def test_git_common_dir_handles_worktree_and_poisoned_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve identical common dir for main checkout and linked worktree even with GIT_DIR."""
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    unrelated = tmp_path / "unrelated.git"
    unrelated.mkdir()
    monkeypatch.setenv("GIT_DIR", str(unrelated))

    main_common = surrogate_cannbench._git_common_dir(main_repo)
    wt_common = surrogate_cannbench._git_common_dir(wt_repo)
    assert main_common == (main_repo / ".git").resolve()
    assert wt_common == main_common
    assert surrogate_cannbench._git_common_dir(tmp_path / "nonexistent") is None
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    assert surrogate_cannbench._git_common_dir(plain_dir) is None


def test_resolve_agent_gates_bin_deep_env_file_and_worktree_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover agent-gates from ancestor venv of env.sh and main repo sibling venv."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _name: None)

    ag_root = tmp_path / "custom_ag" / "AgentGates"
    ag_bin = ag_root / ".venv" / "bin" / "agent-gates"
    ag_bin.parent.mkdir(parents=True)
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    deep_env = ag_root / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh"
    deep_env.parent.mkdir(parents=True)
    deep_env.write_text("TILEFOUNDRY_NPU_CANNBENCH_ARCH=ascend950\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_ENV_FILE", str(deep_env))

    assert surrogate_cannbench._resolve_agent_gates_bin() == str(ag_bin)

    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    sibling_bin = main_repo.parent / "AgentGates" / ".venv" / "bin" / "agent-gates"
    sibling_bin.parent.mkdir(parents=True)
    sibling_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    fake_module = (
        wt_repo
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

    assert surrogate_cannbench._resolve_agent_gates_bin(wt_repo) == str(sibling_bin)


def test_agent_gates_workspace_name_worktree_and_allow_worktrees_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve main checkout and worktree unless allow_worktrees is false."""
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    cfg_allowed = _write_agent_gates_config(tmp_path / "allowed", source_root=main_repo)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_allowed))

    assert surrogate_cannbench._agent_gates_workspace_name(main_repo) == "tilefoundry"
    assert surrogate_cannbench._agent_gates_workspace_name(wt_repo) == "tilefoundry"

    disallowed_dir = tmp_path / "disallowed"
    disallowed_dir.mkdir()
    cfg_disallowed = _write_agent_gates_config(
        disallowed_dir,
        source_root=main_repo,
        allow_worktrees=False,
    )
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_disallowed))

    assert surrogate_cannbench._agent_gates_workspace_name(main_repo) == "tilefoundry"
    with pytest.raises(ValueError, match="No AgentGates workspace"):
        surrogate_cannbench._agent_gates_workspace_name(wt_repo)


def test_run_v2_gate_forwards_source_only_for_worktree_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward --source on submit and verify-requirements only when root differs."""
    main_repo, wt_repo = _init_repo_with_worktree(tmp_path)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_path = _write_agent_gates_config(cfg_dir, source_root=main_repo)
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(cfg_path))

    recorded_cmds: list[list[str]] = []

    def _fake_v2_cmd(
        cmd: list[str] | tuple[str, ...],
        *,
        root: Path,
        timeout_sec: float,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        _ = (root, timeout_sec)
        cmd_list = [str(part) for part in cmd]
        recorded_cmds.append(cmd_list)
        subcmd = cmd_list[1]
        payload: dict[str, object]
        if subcmd == "doctor":
            payload = {"status": "PASS"}
        elif subcmd == "submit":
            payload = {"execution_id": "exec-wt-1", "device_id": "0"}
        elif subcmd == "wait":
            payload = {"status": {"state": "FINISHED", "run_id": "run-wt-1"}}
        elif subcmd == "report":
            payload = {"receiptPath": str(tmp_path / "receipt.json"), "gates": []}
        else:
            assert subcmd == "verify-requirements"
            payload = {"satisfied": True}
        return subprocess.CompletedProcess(cmd_list, 0, "", ""), payload

    monkeypatch.setattr(surrogate_cannbench, "_run_agent_gates_v2_cmd", _fake_v2_cmd)

    wt_exec = surrogate_cannbench._run_v2_gate(
        agent_gates_bin="agent-gates",
        root=wt_repo,
        workspace="tilefoundry",
        gate_id="l4-cannbench",
        target="op/task.py",
        candidate_path="cand.py",
        timeout_sec=30.0,
    )
    assert wt_exec.verified is True
    wt_by_subcmd = {cmd[1]: cmd for cmd in recorded_cmds}
    _assert_flag_value(wt_by_subcmd["submit"], "--source", str(wt_repo.resolve()))
    _assert_flag_value(wt_by_subcmd["verify-requirements"], "--source", str(wt_repo.resolve()))

    recorded_cmds.clear()
    main_exec = surrogate_cannbench._run_v2_gate(
        agent_gates_bin="agent-gates",
        root=main_repo,
        workspace="tilefoundry",
        gate_id="l4-cannbench",
        target="op/task.py",
        candidate_path="cand.py",
        timeout_sec=30.0,
    )
    assert main_exec.verified is True
    main_by_subcmd = {cmd[1]: cmd for cmd in recorded_cmds}
    assert "--source" not in main_by_subcmd["submit"]
    assert "--source" not in main_by_subcmd["verify-requirements"]

    monkeypatch.setenv("AGENT_GATES_CONFIG", str(tmp_path / "missing.toml"))
    assert surrogate_cannbench._agent_gates_source_override("tilefoundry", wt_repo) is None
