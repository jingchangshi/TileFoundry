"""Unit tests for the 6-layer lower-triangular DSM architecture linter."""

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

import dsm_lower_triangular_lint as dsm_lint  # noqa: E402


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dedent(content).lstrip("\n"), encoding="utf-8")
    return root


def _base_packages() -> dict[str, str]:
    return {
        "__init__.py": "",
        "ascend/__init__.py": "",
        "ascend/target/__init__.py": "",
        "ascend/target/device.py": "DEVICE = 'dav-3510'\n",
        "ascend/sil/__init__.py": "",
        "ascend/sil/schema.py": "from tilefoundry_ascend.ascend.target.device import DEVICE\n",
        "ascend/templates/__init__.py": "",
        "ascend/templates/shells.py": "from ..sil.schema import DEVICE\n",
        "ascend/templates/registry.py": "from .shells import DEVICE\n",
        "agent/__init__.py": "",
        "agent/contracts.py": "from tilefoundry_ascend.ascend.sil.schema import DEVICE\n",
        "agent/kb/__init__.py": "",
        "agent/kb/schema.py": "from tilefoundry_ascend.agent.contracts import DEVICE\n",
        "agent/router/__init__.py": "",
        "agent/router/capability_gate.py": (
            "from tilefoundry_ascend.agent.kb.schema import DEVICE\n"
        ),
        "agent/gates/__init__.py": "",
        "agent/gates/cpp_slice_protocol.py": (
            "import inspect\n"
            "def cached_callable_signature_info(fn):\n"
            "    return inspect.signature(fn)\n"
        ),
        "agent/gates/gate_funnel.py": (
            "from tilefoundry_ascend.agent.router.capability_gate import DEVICE\n"
        ),
        "agent/orchestrator/__init__.py": "",
        "agent/orchestrator/llm_client.py": (
            "from tilefoundry_ascend.agent.gates.gate_funnel import DEVICE\n"
        ),
        "agent/flywheel/__init__.py": "",
        "agent/flywheel/ablation_distiller.py": (
            "from tilefoundry_ascend.agent.orchestrator.llm_client import DEVICE\n"
        ),
        "agent/probes/__init__.py": "",
        "agent/probes/probe_hbm_bw.py": (
            "from tilefoundry_ascend.agent.kb.schema import DEVICE\n"
        ),
        "agent/npu/__init__.py": "",
        "agent/npu/ag_transport.py": (
            "import sys\n"
            "from tilefoundry_ascend.agent.gates.gate_funnel import DEVICE\n"
            "def has_tl():\n"
            "    return sys.modules.get('tilelang') is not None\n"
        ),
        "agent/campaign/__init__.py": "",
        "agent/campaign/telemetry_ledger_writer.py": (
            "from tilefoundry_ascend.agent.contracts import DEVICE\n"
        ),
        "agent/campaign/campaign_runner.py": (
            "from tilefoundry_ascend.agent.flywheel.ablation_distiller import DEVICE\n"
            "from tilefoundry_ascend.agent.npu.ag_transport import has_tl\n"
            "from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import (\n"
            "    DEVICE as LEDGER_DEV,\n"
            ")\n"
        ),
    }


def test_clean_lower_triangular_tree_passes(tmp_path: Path) -> None:
    """Valid L0..L5 DAG with third-party sys.modules probe passes with 0 violations."""
    root = _write_tree(tmp_path / "tilefoundry_ascend", _base_packages())
    report = dsm_lint.run_lint(root)
    assert report.total_violations == 0
    assert dsm_lint.main(["--root", str(root)]) == 0


def test_upper_triangular_and_lazy_relative_imports_flagged(tmp_path: Path) -> None:
    """Detect top-level, function-local lazy, and relative upper-triangular imports."""
    files = _base_packages()
    files["ascend/target/device.py"] = (
        "def lazy():\n"
        "    from tilefoundry_ascend.agent.contracts import DEVICE\n"
        "    return DEVICE\n"
    )
    files["ascend/sil/schema.py"] = (
        "from ..schedule.regime_solver import SOLVER\n"
    )
    files["ascend/schedule/__init__.py"] = ""
    files["ascend/schedule/regime_solver.py"] = "SOLVER = 1\n"
    files["agent/gates/gate_funnel.py"] = (
        "from tilefoundry_ascend.agent.orchestrator.llm_client import DEVICE\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {"layers"})
    assert report.counts_by_category["layers"] == 3
    msgs = "\n".join(v.message for v in report.violations)
    assert "L0->L1" in msgs
    assert "L0->L2" in msgs
    assert "L2->L3" in msgs


def test_unclassified_module_flagged(tmp_path: Path) -> None:
    """Reject any new module outside the L0..L5 DSM layer mapping."""
    files = _base_packages()
    files["agent/rogue_helper.py"] = "X = 1\n"
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root)
    assert report.counts_by_category["layers"] == 1
    assert report.violations[0].path == "agent/rogue_helper.py"
    assert "unclassified module" in report.violations[0].message


def test_l3_l4_orthogonality_violations(tmp_path: Path) -> None:
    """Enforce A[L4, L3] = 0 and A[L3, L4] = 0 orthogonality."""
    files = _base_packages()
    files["agent/flywheel/ablation_distiller.py"] = (
        "def run_probe():\n"
        "    from tilefoundry_ascend.agent.probes import probe_hbm_bw\n"
        "    return probe_hbm_bw\n"
    )
    files["agent/npu/ag_transport.py"] = (
        "from tilefoundry_ascend.agent.orchestrator.llm_client import DEVICE\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root)
    assert report.counts_by_category["l3_l4"] == 2
    assert report.counts_by_category["layers"] == 0


def test_tarjan_scc_and_self_import_cycles(tmp_path: Path) -> None:
    """Detect 2-node, 3-node, and self-import SCC cycles including lazy imports."""
    files = _base_packages()
    files["ascend/templates/ascendc_templates.py"] = (
        "from .registry import DEVICE\n"
    )
    files["ascend/templates/tilelang_templates.py"] = (
        "from .ascendc_templates import DEVICE\n"
    )
    files["ascend/templates/registry.py"] = (
        "def render():\n"
        "    from .tilelang_templates import DEVICE\n"
        "    return DEVICE\n"
    )
    files["agent/campaign/telemetry_ledger_writer.py"] = (
        "def write_row():\n"
        "    from tilefoundry_ascend.agent.campaign.campaign_runner import LEDGER_DEV\n"
        "    return LEDGER_DEV\n"
    )
    files["agent/kb/schema.py"] = (
        "from tilefoundry_ascend.agent.kb.schema import X\n"
        "X = 1\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {"scc"})
    assert report.counts_by_category["scc"] == 3
    msgs = "\n".join(v.message for v in report.violations)
    assert "3 modules" in msgs
    assert "2 modules" in msgs
    assert "self-import cycle" in msgs


def test_sys_modules_reflection_detection(tmp_path: Path) -> None:
    """Flag internal or dynamic sys.modules probes while allowing third-party checks."""
    files = _base_packages()
    files["agent/campaign/stage_gate_coordinator.py"] = (
        "import sys\n"
        "from sys import modules as sys_mods\n"
        "def resolve(dyn_name):\n"
        "    a = sys.modules.get('tilefoundry_ascend.agent.campaign.campaign_runner')\n"
        "    b = sys_mods['tilefoundry_ascend.agent.contracts']\n"
        "    c = sys.modules.get(dyn_name)\n"
        "    d = 'tilefoundry_ascend.agent.kb' in sys.modules\n"
        "    ok = sys.modules.get('tilelang')\n"
        "    return (a, b, c, d, ok)\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {"sys_modules"})
    assert report.counts_by_category["sys_modules"] == 4


def test_banned_reflection_symbols_and_global_inspect_signature(tmp_path: Path) -> None:
    """Flag banned reflection everywhere and inspect.signature outside the single boundary."""
    files = _base_packages()
    files["agent/campaign/stage_gate_coordinator.py"] = (
        "import inspect\n"
        "from inspect import signature as sig_fn\n"
        "_PROPAGATED_HOOKS = set()\n"
        "def _resolve_hook(name):\n"
        "    return name\n"
        "def _owner_override(self, name):\n"
        "    return name\n"
        "def _invoke_with_optional_kwargs(fn):\n"
        "    return inspect.signature(fn), sig_fn(fn)\n"
    )
    files["agent/gates/gate_funnel.py"] = (
        "import inspect\n"
        "from tilefoundry_ascend.agent.router.capability_gate import DEVICE\n"
        "def sig(fn):\n"
        "    return inspect.signature(fn)\n"
    )
    files["agent/campaign/campaign_runner.py"] = (
        "class Runner:\n"
        "    def _sync_coordinators(self):\n"
        "        pass\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {"reflection"})
    lines = {(v.path, v.line, v.message) for v in report.violations}
    assert any("_PROPAGATED_HOOKS" in m for _, _, m in lines)
    assert any("_resolve_hook" in m for _, _, m in lines)
    assert any("_owner_override" in m for _, _, m in lines)
    assert any("_invoke_with_optional_kwargs" in m for _, _, m in lines)
    assert any("_sync_coordinators" in m for _, _, m in lines)
    assert any(
        p == "agent/campaign/stage_gate_coordinator.py" and "inspect.signature" in m
        for p, _, m in lines
    )
    assert any(
        p == "agent/gates/gate_funnel.py" and "inspect.signature" in m
        for p, _, m in lines
    )
    assert not any(p == "agent/gates/cpp_slice_protocol.py" for p, _, _ in lines)


@pytest.mark.parametrize(
    ("category", "rel_path", "snippet", "expected_substr"),
    [
        (
            "flat_typing",
            "agent/kb/schema.py",
            "from typing import Any\ndef f(x: Any) -> object:\n    return x\n",
            "Any",
        ),
        (
            "flat_typing",
            "agent/kb/schema.py",
            "import typing as t\ndef f(x: object) -> int:\n    return t.cast(int, x)\n",
            "t.cast",
        ),
        (
            "flat_typing",
            "agent/kb/schema.py",
            "from typing import TYPE_" "CHECKING\nif TYPE_" "CHECKING:\n    X = 1\n",
            "TYPE_" "CHECKING",
        ),
        (
            "flat_typing",
            "agent/kb/schema.py",
            "X: int = 'bad'  # type: " "ignore[assignment]\n",
            "# type: ignore",
        ),
        (
            "flat_typing",
            "agent/kb/schema.py",
            "def f():\n    from tilefoundry_ascend.agent.contracts import DEVICE\n"
            "    return DEVICE\n",
            "function-local internal import",
        ),
        (
            "flat_typing",
            "agent/kb/schema.py",
            "def f():\n    from ..contracts import DEVICE\n    return DEVICE\n",
            "function-local internal import",
        ),
        (
            "reflection",
            "agent/gates/stage_b_gate.py",
            "import inspect\ndef probe(fn):\n    return inspect.signature(fn)\n",
            "inspect.signature",
        ),
        (
            "reflection",
            "agent/gates/cpp_slice_protocol.py",
            "from inspect import signature\n"
            "def cached_callable_signature_info(fn):\n    return signature(fn)\n",
            "inspect.signature",
        ),
        (
            "reflection",
            "agent/gates/cpp_slice_protocol.py",
            "import inspect\n"
            "def cached_callable_signature_info(fn):\n    return inspect.signature(fn)\n"
            "def sneaky(fn):\n    return inspect.signature(fn)\n",
            "inspect.signature",
        ),
        (
            "cross_private_write",
            "agent/kb/schema.py",
            "def mutate(obj):\n    obj._secret = 1\n",
            "_secret",
        ),
        (
            "cross_private_write",
            "agent/kb/schema.py",
            "def mutate(obj):\n    obj._count += 1\n",
            "_count",
        ),
        (
            "cross_private_write",
            "agent/kb/schema.py",
            "def mutate(obj):\n    _, obj.pub._hidden = (0, 2)\n",
            "_hidden",
        ),
        (
            "module_loc",
            "agent/kb/schema.py",
            "X = 1\n" * 1501,
            "1501 lines",
        ),
    ],
)
def test_strict_ratchets_reject(
    tmp_path: Path, category: str, rel_path: str, snippet: str, expected_substr: str
) -> None:
    """Reject violations across all four strict ratchets with single-sentence fix hints."""
    files = _base_packages()
    files[rel_path] = snippet
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {category})
    assert report.counts_by_category[category] >= 1
    assert any(expected_substr in v.message for v in report.violations)
    for v in report.violations:
        assert v.category == category
        assert "; " in v.message and v.message.endswith(".")


@pytest.mark.parametrize(
    ("category", "rel_path", "snippet"),
    [
        (
            "flat_typing",
            "agent/kb/schema.py",
            "from typing import Protocol\n"
            "class Reader(Protocol):\n"
            "    def read(self) -> object: ...\n"
            "def f() -> str:\n"
            "    import json\n"
            "    return json.dumps({'note': '# type: ' + 'ignore'})\n",
        ),
        (
            "reflection",
            "agent/gates/cpp_slice_protocol.py",
            "import inspect\n"
            "def cached_callable_signature_info(fn):\n"
            "    return inspect.signature(fn)\n",
        ),
        (
            "reflection",
            "agent/gates/stage_b_gate.py",
            "import inspect\n"
            "def is_empty(param) -> bool:\n"
            "    return param.default is inspect.Parameter.empty\n",
        ),
        (
            "cross_private_write",
            "agent/kb/schema.py",
            "class Box:\n"
            "    def __init__(self) -> None:\n"
            "        self._val = 0\n"
            "    @classmethod\n"
            "    def reset(cls) -> None:\n"
            "        cls._default = 1\n"
            "def clean(obj) -> None:\n"
            "    _local = 1\n"
            "    obj.__doc__ = 'ok'\n"
            "    obj._cache['k'] = _local\n",
        ),
        (
            "module_loc",
            "agent/kb/schema.py",
            "X = 1\n" * 1500,
        ),
    ],
)
def test_strict_ratchets_accept(
    tmp_path: Path, category: str, rel_path: str, snippet: str
) -> None:
    """Accept clean typed modules, boundary exemptions, self/cls/dunder, and <=1500 LOC."""
    files = _base_packages()
    files[rel_path] = snippet
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)
    report = dsm_lint.run_lint(root, {category})
    assert report.counts_by_category[category] == 0
    assert report.total_violations == 0


def test_cli_only_json_and_error_handling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --only filtering, --json schema, and invalid CLI argument exit codes."""
    files = _base_packages()
    files["agent/gates/gate_funnel.py"] = (
        "from tilefoundry_ascend.agent.orchestrator.llm_client import DEVICE\n"
    )
    root = _write_tree(tmp_path / "tilefoundry_ascend", files)

    assert dsm_lint.main(["--root", str(root), "--only", "l3_l4"]) == 0
    capsys.readouterr()

    rc = dsm_lint.main(["--root", str(root), "--only", "layers", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_violations"] == 1
    assert payload["counts_by_category"]["layers"] == 1
    assert payload["violations"][0]["category"] == "layers"

    assert dsm_lint.main(["--root", str(tmp_path / "missing")]) == 2
    assert dsm_lint.main(["--root", str(root), "--only", "bogus"]) == 2


def test_live_tilefoundry_ascend_speed_and_cli() -> None:
    """Live tilefoundry_ascend scan finishes under 0.5s with structured category counts."""
    live_root = _REPO_ROOT / "targets" / "ascend" / "tilefoundry_ascend"
    report = dsm_lint.run_lint(live_root)
    assert report.files_scanned >= 70
    assert report.elapsed_ms < 500.0
    assert set(report.counts_by_category) == set(dsm_lint.VALID_CATEGORIES)

    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "dsm_lower_triangular_lint.py"), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1)
    parsed = json.loads(proc.stdout)
    assert parsed["files_scanned"] == report.files_scanned
    assert parsed["elapsed_ms"] < 500.0
