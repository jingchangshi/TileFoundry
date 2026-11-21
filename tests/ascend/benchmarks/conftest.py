"""Promotion-profile pytest hooks for L3 acceptance evidence.

Ordinary acceptance runs (no ``TF_L3_EVIDENCE_PATH``) keep the historical
skip-on-missing-device behavior.  A promotion run activates the recorder,
restricts collection to the target contract's nodes, probes a required
reference cache before any device work, and turns incomplete coverage or an
environment block into a non-zero exit instead of a pass by omission.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import l3_contract
import pytest

_ACCEPTANCE_FILE = "test_npu_acceptance.py"


def _finalize_and_exit(recorder: l3_contract.EvidenceRecorder, message: str) -> None:
    """Configure-time exits skip sessionfinish, so evidence is closed first."""
    recorder.finalize(interrupted=False)
    pytest.exit(message, returncode=l3_contract.BLOCKED_EXIT_CODE)


def _probe_required_cache(recorder: l3_contract.EvidenceRecorder) -> None:
    mode = os.environ.get("CANNBENCH_CACHE_MODE", "").strip().lower()
    location = os.environ.get("CANNBENCH_CACHE_DIR", "").strip()
    preflight: dict[str, object] = {
        "mode": mode or None,
        "expectation": None,
        "cacheDir": location or None,
        "probed": False,
        "ok": None,
        "error": None,
    }

    def block(detail: str) -> None:
        preflight["error"] = detail
        recorder.note_cache_preflight(preflight)
        recorder.environment_block("reference_cache_unavailable", detail)
        _finalize_and_exit(
            recorder,
            f"L3 promotion blocked: reference_cache_unavailable: {detail}",
        )

    try:
        expectation = l3_contract.cache_expectation()
    except l3_contract.L3ContractError as exc:
        block(str(exc))
        return
    preflight["expectation"] = expectation
    if mode not in ("", "off", "optional", "required"):
        block(f"CANNBENCH_CACHE_MODE has invalid value {mode!r}")
        return
    if expectation in {"cold", "warm"} and mode != "required":
        block(
            f"TF_L3_CACHE_EXPECTATION={expectation} requires "
            "CANNBENCH_CACHE_MODE=required"
        )
        return
    if mode != "required":
        recorder.note_cache_preflight(preflight)
        return
    if not location or location == "off":
        block("CANNBENCH_CACHE_MODE=required but CANNBENCH_CACHE_DIR is unset or off")
        return
    result = l3_contract.probe_reference_cache(Path(location))
    preflight["probed"] = True
    preflight["ok"] = result.ok
    preflight["error"] = result.error or None
    recorder.note_cache_preflight(preflight)
    if not result.ok:
        detail = f"reference cache probe failed in {result.cache_dir}: {result.error}"
        recorder.environment_block("reference_cache_unavailable", detail)
        _finalize_and_exit(
            recorder,
            f"L3 promotion blocked: reference_cache_unavailable: {result.error}",
        )


def _install_signal_handler(recorder: l3_contract.EvidenceRecorder) -> None:
    def handle(signum: int, frame: object) -> None:
        _ = frame
        recorder.note_signal(signum)
        recorder.finalize(interrupted=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle)


def pytest_configure(config: pytest.Config) -> None:
    _ = config
    recorder = l3_contract.active_recorder()
    if recorder is None:
        return
    _install_signal_handler(recorder)
    if recorder.error_category:
        _finalize_and_exit(
            recorder,
            f"L3 promotion blocked: {recorder.error_category}: {recorder.error_detail}",
        )
    _probe_required_cache(recorder)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    recorder = l3_contract.current_recorder()
    if recorder is None:
        return
    allowed = {node.name for node in recorder.contract.nodes}
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        short = item.name.split("[", 1)[0]
        if short in allowed and Path(str(item.path)).name == _ACCEPTANCE_FILE:
            kept.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    recorder = l3_contract.current_recorder()
    if recorder is None:
        return
    longrepr = ""
    if report.longrepr is not None:
        longrepr = str(report.longrepr)
    recorder.note_report(report.nodeid, report.when, report.outcome, longrepr)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = exitstatus
    recorder = l3_contract.current_recorder()
    if recorder is None:
        return
    recorder.finalize(interrupted=False)
    if not recorder.coverage()["qualified"] and session.exitstatus == 0:
        session.exitstatus = 1
