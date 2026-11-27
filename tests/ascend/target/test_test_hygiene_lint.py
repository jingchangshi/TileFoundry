"""Unit tests for the Ascend test hygiene linter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import test_hygiene_lint as hygiene_lint  # noqa: E402


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dedent(content).lstrip("\n"), encoding="utf-8")
    return root


def test_clean_tree_and_benchmarks_skip(tmp_path: Path) -> None:
    """Clean names, dunder/public/stdlib/local-fixture patches, and benchmarks pass."""
    root = _write_tree(
        tmp_path / "tests_ascend",
        {
            "target/test_clean_surface.py": """
                import os
                import subprocess
                from types import SimpleNamespace
                from unittest.mock import patch
                import pytest
                from tilefoundry_ascend.ascend.target import AscendTarget

                class LocalStub:
                    _secret = 1

                @pytest.fixture
                def local_stub():
                    return SimpleNamespace(_value=10)

                def test_allowed_patches(monkeypatch, local_stub):
                    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
                    monkeypatch.chdir(os.getcwd())
                    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
                    monkeypatch.setattr(AscendTarget, "__init__", lambda self: None)
                    monkeypatch.setattr(local_stub, "_value", 20)
                    stub = LocalStub()
                    monkeypatch.setattr(stub, "_secret", 2)
                    ns = SimpleNamespace(_field=3)
                    monkeypatch.setattr(ns, "_field", 4)
                    with patch("subprocess.run", lambda *a, **k: None):
                        pass
            """,
            "benchmarks/test_v43_legacy_bench.py": """
                from tilefoundry_ascend.agent import contracts
                def test_ignored(monkeypatch):
                    monkeypatch.setattr(contracts, "_hidden", lambda: None)
            """,
        },
    )
    report = hygiene_lint.run_lint(root)
    assert report.files_scanned == 1
    assert report.total_violations == 0
    assert hygiene_lint.main(["--root", str(root)]) == 0


@pytest.mark.parametrize(
    ("filename", "flagged"),
    [
        ("test_v43_campaign.py", True),
        ("test_v55_w4_dialect.py", True),
        ("test_v501_recurrence.py", True),
        ("test_ops_contracts_v52.py", True),
        ("test_fixes_v4_7.py", True),
        ("test_phase_v2.py", True),
        ("test_observation_space_5d.py", False),
        ("test_target_and_surface.py", False),
        ("test_v_only_word.py", False),
    ],
)
def test_version_filename_rule(tmp_path: Path, filename: str, flagged: bool) -> None:
    """Reject versioned test filenames while allowing clean domain names."""
    root = _write_tree(tmp_path / "suite", {filename: "def test_ok():\n    assert True\n"})
    report = hygiene_lint.run_lint(root, {"version_filename"})
    assert report.counts_by_rule["version_filename"] == (1 if flagged else 0)
    if flagged:
        assert report.violations[0].path == filename
        assert report.violations[0].line == 1


def test_private_monkeypatch_rejects_production_targets(tmp_path: Path) -> None:
    """Flag monkeypatch.setattr, patch.object, and patch on private production members."""
    root = _write_tree(
        tmp_path / "suite",
        {
            "test_bad_patches.py": """
                import unittest.mock
                from unittest.mock import patch
                import tilefoundry_ascend.agent.orchestrator.gate_funnel as gf
                from tilefoundry_ascend.agent.campaign.campaign_runner import Runner

                def test_bad(monkeypatch):
                    r = Runner()
                    monkeypatch.setattr(gf, "_resolve_bisheng_bin", lambda *a: "")
                    monkeypatch.setattr(r, "_run_l1", lambda *a: None)
                    monkeypatch.setattr("tilefoundry_ascend.agent.contracts._secret", 1)
                    with patch.object(gf, "_disassemble_object", return_value=""):
                        pass
                    with unittest.mock.patch("tilefoundry_ascend.ascend.sil._priv"):
                        pass
            """,
        },
    )
    report = hygiene_lint.run_lint(root, {"private_monkeypatch"})
    assert report.counts_by_rule["private_monkeypatch"] == 5
    assert hygiene_lint.main(["--root", str(root)]) == 1


def test_cli_json_error_and_speed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --json schema, missing root exit code 2, and < 0.5s live scan speed."""
    root = _write_tree(tmp_path / "suite", {"test_v56_bad.py": "def test_x():\n    pass\n"})
    rc = hygiene_lint.main(["--root", str(root), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_violations"] == 1
    assert payload["counts_by_rule"]["version_filename"] == 1
    assert payload["violations"][0]["rule"] == "version_filename"

    assert hygiene_lint.main(["--root", str(tmp_path / "missing")]) == 2

    live_dir = _REPO_ROOT / "tests" / "ascend" / "cost_model"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "test_hygiene_lint.py"),
            "--root",
            str(live_dir),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1)
    live_payload = json.loads(proc.stdout)
    assert live_payload["elapsed_ms"] < 500.0
