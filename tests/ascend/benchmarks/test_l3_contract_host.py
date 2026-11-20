"""Host-only tests for L3 target contracts, evidence, and reference-cache policy."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree

import pytest
import torch

import cannbench_contract
import l3_contract


def _unit_contract(tmp_path: Path) -> l3_contract.TargetContract:
    return l3_contract.TargetContract(
        operator="unit",
        task="level1/unit",
        task_dir=tmp_path,
        nodes=(
            l3_contract.NodeContract(name="node_one", required_cases=("1", "2")),
            l3_contract.NodeContract(name="node_two", required_cases=("4",)),
        ),
        case_ids=("1", "2", "4"),
    )


def _single_case_contract(tmp_path: Path) -> l3_contract.TargetContract:
    return l3_contract.TargetContract(
        operator="unit",
        task="level1/unit",
        task_dir=tmp_path,
        nodes=(l3_contract.NodeContract(name="node_one", required_cases=("1",)),),
        case_ids=("1",),
    )


def _dynamic_case_contract(tmp_path: Path) -> l3_contract.TargetContract:
    return l3_contract.TargetContract(
        operator="unit",
        task="level1/unit",
        task_dir=tmp_path,
        nodes=(
            l3_contract.NodeContract(
                name=l3_contract.DYNAMIC_NODE, required_cases=("1",)
            ),
        ),
        case_ids=("1",),
    )


def _observation(
    *,
    hits: int = 0,
    misses: int = 0,
    writes: int = 0,
    expectation: str = "any",
    required: bool = False,
    usable: bool = True,
    errors: list[str] | None = None,
) -> dict[str, object]:
    return {
        "mode": "required" if required else "optional",
        "required": required,
        "expectation": expectation,
        "usable": usable,
        "location": "/tmp/cache",
        "hits": hits,
        "misses": misses,
        "writes": writes,
        "errors": list(errors or ()),
        "byKind": {
            "oracle": {"hits": hits, "misses": misses, "writes": writes},
            "native": {"hits": 0, "misses": 0, "writes": 0},
        },
    }


def _pass_first_node(recorder: l3_contract.EvidenceRecorder) -> None:
    with recorder.node("node_one"):
        for case_id in ("1", "2"):
            recorder.case_started("node_one", case_id)
            recorder.case_finished("node_one", case_id, "passed")


def _pass_all_nodes(recorder: l3_contract.EvidenceRecorder) -> None:
    _pass_first_node(recorder)
    with recorder.node("node_two"):
        recorder.case_started("node_two", "4")
        recorder.case_finished("node_two", "4", "passed")


def _write_cases(task_dir: Path, rows: list[dict[str, object]]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "cases.yaml").write_text(json.dumps({"cases": rows}), encoding="utf-8")


def test_ar_contract_covers_full_case_and_node_identity() -> None:
    contract = l3_contract.resolve_contract("level4/attention_residual")
    assert contract.operator == "attention_residual"
    assert contract.task == l3_contract.AR_TASK
    assert tuple(node.name for node in contract.nodes) == (
        l3_contract.AR_NODE,
        l3_contract.AR_PARITY_NODE,
        l3_contract.DYNAMIC_NODE,
    )
    assert contract.case_ids == l3_contract.load_case_ids(l3_contract.AR_DIR)
    assert sorted(int(case_id) for case_id in contract.case_ids) == list(range(1, 21))
    assert contract.nodes[0].required_cases == contract.case_ids
    assert contract.nodes[1].required_cases == ("1", "16", "20")
    assert contract.nodes[2].required_cases == l3_contract.AR_DYNAMIC_CASES


def test_mla_contract_covers_official_and_supplemental_cases() -> None:
    contract = l3_contract.resolve_contract("mla")
    assert contract.operator == "mla"
    assert tuple(node.name for node in contract.nodes) == (
        l3_contract.MLA_NODE,
        l3_contract.MLA_SUPPLEMENTAL_NODE,
        l3_contract.DYNAMIC_NODE,
    )
    assert sorted(int(case_id) for case_id in contract.case_ids[:20]) == list(range(1, 21))
    assert contract.nodes[0].required_cases == contract.case_ids[:20]
    assert contract.nodes[1].required_cases == (
        "gqa_bsnd_kv_tail",
        "gqa_bnsd_scale",
        "gqa_decode_shard448",
        "gqa_query_tail",
    )
    assert contract.nodes[2].required_cases == l3_contract.MLA_DYNAMIC_CASES


def test_empty_target_is_rejected() -> None:
    with pytest.raises(l3_contract.L3ContractError):
        l3_contract.resolve_contract("   ")


def test_duplicate_and_missing_case_ids_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    _write_cases(duplicate, [{"case_id": 1}, {"case_id": 1}])
    with pytest.raises(l3_contract.L3ContractError, match="duplicate"):
        l3_contract.load_case_ids(duplicate)

    missing = tmp_path / "missing"
    _write_cases(missing, [{"input_shape": [[4]]}])
    with pytest.raises(l3_contract.L3ContractError, match="without case_id"):
        l3_contract.load_case_ids(missing)

    empty = tmp_path / "empty"
    _write_cases(empty, [])
    with pytest.raises(l3_contract.L3ContractError, match="declares no cases"):
        l3_contract.load_case_ids(empty)


def test_duplicate_supplemental_case_ids_are_rejected(tmp_path: Path) -> None:
    payload = {
        "cases": [
            {"case_id": "gqa_one", "key": "a"},
            {"case_id": "gqa_one", "key": "b"},
        ]
    }
    path = tmp_path / "supplemental.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(l3_contract.L3ContractError, match="duplicate"):
        l3_contract.load_supplemental_mla_ids(path)


def _write_fake_cannbench(root: Path, case_ids: list[int]) -> Path:
    tasks = root / "cann-bench" / "tasks"
    op_dir = tasks / "level1" / "fake_op"
    op_dir.mkdir(parents=True)
    (tasks / "suite_manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "fake_op", "path": "tasks/level1/fake_op", "level": "L1"}
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "case_id": case_id,
            "input_shape": [[case_id, 16]],
            "dtype": ["float16"],
            "attrs": {},
            "value_range": [[-1, 1]],
        }
        for case_id in case_ids
    ]
    (op_dir / "cases.yaml").write_text(json.dumps({"cases": rows}), encoding="utf-8")
    return root / "repo"


def test_generic_contract_uses_official_cases_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANNBENCH_REPO", raising=False)
    source_root = _write_fake_cannbench(tmp_path, [7, 9, 11])
    contract = l3_contract.resolve_contract("fake_op", source_root=source_root)
    assert contract.operator == "fake_op"
    assert contract.task == "level1/fake_op"
    assert contract.case_ids == ("7", "9", "11")
    assert tuple(node.name for node in contract.nodes) == (l3_contract.GENERIC_NODE,)
    assert contract.nodes[0].required_cases == ("7", "9", "11")


def test_evidence_is_atomic_and_unknown_fields_are_null(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    recorder = l3_contract.EvidenceRecorder(
        path, _unit_contract(tmp_path), expected_python=sys.executable
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == l3_contract.SCHEMA
    assert payload["profile"] == l3_contract.PROFILE
    assert payload["phase"] == "initialized"
    assert payload["candidate"] is None
    assert payload["referenceCache"] is None
    assert payload["interruption"]["interrupted"] is True
    assert payload["interruption"]["lastCompletedNode"] is None
    assert payload["interruption"]["lastCompletedCase"] is None
    assert payload["interpreter"]["sysExecutable"] == str(Path(sys.executable).resolve())
    assert payload["interpreter"]["pid"] == os.getpid()
    assert payload["interpreter"]["match"] is True
    assert payload["coverage"]["qualified"] is False

    recorder.case_started("node_one", "1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "case:node_one:1"
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_target_module_and_candidate_bind_imported_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "candidate.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv(l3_contract.MODULE_ENV, str(module))
    path = tmp_path / "evidence.json"
    recorder = l3_contract.EvidenceRecorder(
        path, _unit_contract(tmp_path), expected_python=sys.executable
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target"]["modulePath"] == str(module.resolve())
    assert payload["candidate"] is None

    recorder.note_candidate(module)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target"]["modulePath"] == str(module.resolve())
    assert payload["candidate"]["modulePath"] == str(module.resolve())
    assert payload["candidate"]["declaredPath"] == str(module.resolve())
    assert (
        payload["candidate"]["sha256"]
        == hashlib.sha256(module.read_bytes()).hexdigest()
    )


def test_qualified_run_requires_every_node_and_case(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.reference_cache(_observation(hits=3))
    recorder.finalize()
    coverage = recorder.coverage()
    assert coverage["qualified"] is True
    assert coverage["nodes"]["completed"] == 2
    assert coverage["cases"]["completed"] == 3
    payload = recorder.payload()
    assert payload["phase"] == "finished"
    assert payload["interruption"]["interrupted"] is False
    assert payload["interruption"]["lastCompletedNode"] == "node_two"
    assert payload["interruption"]["lastCompletedCase"] == {
        "node": "node_two",
        "caseId": "4",
        "status": "passed",
    }


def test_partial_and_skipped_runs_never_qualify(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    with recorder.node("node_one"):
        recorder.case_finished("node_one", "1", "passed")
    recorder.note_report(
        "tests/ascend/benchmarks/test_npu_acceptance.py::node_two",
        "setup",
        "skipped",
        "NPU unavailable",
    )
    coverage = recorder.coverage()
    assert coverage["qualified"] is False
    assert coverage["nodes"]["missing"] == []
    assert coverage["nodes"]["skipped"] == ["node_two"]
    assert sorted(coverage["cases"]["missing"]) == ["node_one:2", "node_two:4"]

    payload = recorder.payload()
    assert payload["interruption"]["interrupted"] is True
    assert payload["interruption"]["lastCompletedCase"]["caseId"] == "1"


def test_all_skipped_run_is_not_qualified(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    for node in ("node_one", "node_two"):
        recorder.note_report(f"test_npu_acceptance.py::{node}", "setup", "skipped", "no device")
    coverage = recorder.coverage()
    assert coverage["qualified"] is False
    assert coverage["nodes"]["completed"] == 0
    assert len(coverage["nodes"]["skipped"]) == 2
    assert len(coverage["cases"]["missing"]) == 3


def test_failed_and_unexpected_cases_never_qualify(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.case_finished("node_one", "extra", "passed")
    recorder.case_finished("node_two", "4", "failed", "boom")
    coverage = recorder.coverage()
    assert coverage["qualified"] is False
    assert coverage["cases"]["unexpected"] == ["node_one:extra"]
    assert coverage["cases"]["missing"] == ["node_two:4"]


def test_interruption_preserves_last_completed_node_and_case(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    with recorder.node("node_one"):
        for case_id in ("1", "2"):
            recorder.case_finished("node_one", case_id, "passed")
    recorder.node_started("node_two")
    recorder.case_started("node_two", "4")
    recorder.note_signal(signal.SIGTERM)
    recorder.finalize(interrupted=True)

    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "finished"
    assert payload["interruption"] == {
        "interrupted": True,
        "lastCompletedNode": "node_one",
        "lastCompletedCase": {"node": "node_one", "caseId": "2", "status": "passed"},
        "errorCategory": "interrupted",
        "detail": None,
        "signal": "SIGTERM",
    }
    assert payload["nodes"][1]["status"] == "running"


def test_environment_block_prevents_qualification(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.reference_cache(_observation(required=True, usable=False, errors=["no cache"]))
    recorder.environment_block("reference_cache_unavailable", "cache is not writable")
    recorder.finalize()
    assert recorder.coverage()["qualified"] is False
    payload = recorder.payload()
    assert payload["coverage"]["referenceCacheProblem"] == "reference_cache_unavailable"
    assert payload["interruption"]["errorCategory"] == "reference_cache_unavailable"


def test_blocked_node_never_qualifies_even_without_a_category(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.node_finished("node_two", "blocked")
    coverage = recorder.coverage()
    assert coverage["nodes"]["blocked"] == ["node_two"]
    assert coverage["qualified"] is False


def test_evidence_write_errors_fail_closed(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.reference_cache(_observation(hits=1))
    recorder.write_errors.append("OSError: disk full")
    recorder.finalize()
    assert recorder.coverage()["qualified"] is False
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert payload["writeErrors"] == ["OSError: disk full"]
    assert payload["interruption"]["errorCategory"] == "evidence_write_error"
    assert payload["interruption"]["interrupted"] is False


def test_cold_and_warm_cache_policy_violations(tmp_path: Path) -> None:
    cold = l3_contract.EvidenceRecorder(
        tmp_path / "cold.json", _unit_contract(tmp_path)
    )
    cold.reference_cache(_observation(misses=2, writes=2, expectation="cold"))
    cold.reference_cache(_observation(hits=1, expectation="cold"))
    assert cold.reference_cache_problem() == ""
    cache = cold.payload()["referenceCache"]
    assert cache["hits"] == 1
    assert cache["misses"] == 2
    assert cache["writes"] == 2
    assert cache["observations"] == 2

    no_cold_miss = l3_contract.EvidenceRecorder(
        tmp_path / "no-cold-miss.json", _unit_contract(tmp_path)
    )
    no_cold_miss.reference_cache(_observation(hits=1, expectation="cold"))
    assert no_cold_miss.reference_cache_problem() == "reference_cache_expectation_cold"

    lost_cold_write = l3_contract.EvidenceRecorder(
        tmp_path / "lost-cold-write.json", _unit_contract(tmp_path)
    )
    lost_cold_write.reference_cache(
        _observation(misses=1, writes=0, expectation="cold")
    )
    assert (
        lost_cold_write.reference_cache_problem()
        == "reference_cache_expectation_cold"
    )

    warm = l3_contract.EvidenceRecorder(
        tmp_path / "warm.json", _unit_contract(tmp_path)
    )
    warm.reference_cache(_observation(hits=2, expectation="warm"))
    assert warm.reference_cache_problem() == ""
    warm.reference_cache(_observation(hits=1, misses=1, expectation="warm"))
    assert warm.reference_cache_problem() == "reference_cache_expectation_warm"

    warm_write = l3_contract.EvidenceRecorder(
        tmp_path / "warm-write.json", _unit_contract(tmp_path)
    )
    warm_write.reference_cache(_observation(hits=1, writes=1, expectation="warm"))
    assert warm_write.reference_cache_problem() == "reference_cache_expectation_warm"

    missing_hits = l3_contract.EvidenceRecorder(
        tmp_path / "missing-hits.json", _unit_contract(tmp_path)
    )
    missing_hits.reference_cache(_observation(expectation="warm"))
    assert missing_hits.reference_cache_problem() == "reference_cache_expectation_warm"


def test_any_cache_expectation_allows_mixed_hits_and_misses(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.reference_cache(
        _observation(misses=2, writes=2, expectation="any", required=True)
    )
    recorder.reference_cache(
        _observation(hits=3, expectation="any", required=True)
    )
    recorder.finalize()
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert recorder.reference_cache_problem() == ""
    assert payload["coverage"]["qualified"] is True
    assert payload["interruption"]["errorCategory"] is None
    cache = payload["referenceCache"]
    assert (cache["hits"], cache["misses"], cache["writes"]) == (3, 2, 2)


def test_cache_expectation_violation_blocks_a_green_run(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.reference_cache(_observation(hits=1, expectation="cold"))
    recorder.finalize()
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert (
        payload["interruption"]["errorCategory"]
        == "reference_cache_expectation_cold"
    )
    assert payload["coverage"]["qualified"] is False


def test_cache_expectation_violation_does_not_mask_a_case_failure(
    tmp_path: Path,
) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    _pass_all_nodes(recorder)
    recorder.case_finished("node_two", "4", "failed", "numeric mismatch")
    recorder.reference_cache(_observation(hits=1, expectation="cold"))
    recorder.finalize()
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert payload["interruption"]["errorCategory"] is None
    assert payload["nodes"][1]["cases"]["4"] == "failed"


def test_cache_policy_change_between_nodes_breaks_qualification(
    tmp_path: Path,
) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    recorder.reference_cache(_observation(misses=1, writes=1, expectation="cold"))
    recorder.reference_cache(_observation(hits=1, expectation="warm"))
    assert recorder.reference_cache_problem() == "reference_cache_policy_changed"
    cache = recorder.payload()["referenceCache"]
    assert cache["usable"] is False
    assert "policy changed" in cache["errors"][0]


def test_required_cache_with_errors_is_a_problem(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    recorder.reference_cache(_observation(required=True, errors=["write failed"]))
    assert recorder.reference_cache_problem() == "reference_cache_error"


def test_cache_probe_writes_reads_and_deletes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    result = l3_contract.probe_reference_cache(cache_dir)
    assert result.ok is True
    assert result.error == ""
    assert cache_dir.is_dir()
    assert list(cache_dir.iterdir()) == []


def test_cache_probe_reports_unusable_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    result = l3_contract.probe_reference_cache(blocker)
    assert result.ok is False
    assert result.error


def test_interpreter_identity_detects_mismatch(tmp_path: Path) -> None:
    matching = l3_contract.EvidenceRecorder(
        tmp_path / "matching.json",
        _unit_contract(tmp_path),
        expected_python=sys.executable,
    )
    assert matching.interpreter["match"] is True
    mismatched = l3_contract.EvidenceRecorder(
        tmp_path / "mismatched.json",
        _unit_contract(tmp_path),
        expected_python="/opt/not-this/python3",
    )
    assert mismatched.interpreter["match"] is False


def test_cache_preflight_is_preserved_with_harness_observation(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    recorder.note_cache_preflight(
        {
            "mode": "required",
            "expectation": "cold",
            "cacheDir": "/tmp/cache",
            "probed": True,
            "ok": True,
            "error": None,
        }
    )
    assert recorder.payload()["referenceCache"] == {
        "preflight": {
            "mode": "required",
            "expectation": "cold",
            "cacheDir": "/tmp/cache",
            "probed": True,
            "ok": True,
            "error": None,
        }
    }
    recorder.reference_cache(_observation(misses=2, writes=2, expectation="cold"))
    payload = recorder.payload()
    reference_cache = payload["referenceCache"]
    assert reference_cache["preflight"]["probed"] is True
    assert reference_cache["misses"] == 2


def test_get_recorder_requires_expected_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(l3_contract.EVIDENCE_ENV, str(tmp_path / "evidence.json"))
    monkeypatch.setenv(l3_contract.TARGET_ENV, "level4/attention_residual")
    monkeypatch.delenv(l3_contract.EXPECTED_PYTHON_ENV, raising=False)
    l3_contract.reset_recorder()
    try:
        recorder = l3_contract.active_recorder()
        assert recorder is not None
        payload = recorder.payload()
        assert payload["interruption"]["errorCategory"] == "interpreter_identity_missing"
        assert payload["coverage"]["qualified"] is False
    finally:
        l3_contract.reset_recorder()


def test_get_recorder_resolves_contract_and_records_broken_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(l3_contract.EVIDENCE_ENV, str(tmp_path / "evidence.json"))
    monkeypatch.setenv(l3_contract.TARGET_ENV, "level4/attention_residual")
    monkeypatch.setenv(l3_contract.EXPECTED_PYTHON_ENV, sys.executable)
    l3_contract.reset_recorder()
    try:
        recorder = l3_contract.active_recorder()
        assert recorder is not None
        assert recorder.contract.operator == "attention_residual"
        assert recorder.payload()["target"]["operator"] == "attention_residual"
    finally:
        l3_contract.reset_recorder()

    monkeypatch.setenv(l3_contract.TARGET_ENV, "definitely/not_an_operator")
    l3_contract.reset_recorder()
    try:
        broken = l3_contract.active_recorder()
        assert broken is not None
        payload = broken.payload()
        assert payload["interruption"]["errorCategory"] == "target_contract_error"
        assert payload["coverage"]["qualified"] is False
    finally:
        l3_contract.reset_recorder()


def test_current_recorder_never_activates_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After-test hooks must not resurrect a recorder from a patched env."""
    monkeypatch.setenv(l3_contract.EVIDENCE_ENV, str(tmp_path / "evidence.json"))
    monkeypatch.setenv(l3_contract.TARGET_ENV, "level4/attention_residual")
    monkeypatch.setenv(l3_contract.EXPECTED_PYTHON_ENV, sys.executable)
    l3_contract.reset_recorder()
    try:
        assert l3_contract.current_recorder() is None
        recorder = l3_contract.active_recorder()
        assert recorder is not None
        assert l3_contract.current_recorder() is recorder
        l3_contract.reset_recorder()
        assert l3_contract.current_recorder() is None
    finally:
        l3_contract.reset_recorder()


def _cache_harness(tmp_path: Path) -> SimpleNamespace:
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "golden.py").write_text("def unit():\n    return None\n", encoding="utf-8")
    (task_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    (task_dir / "proto.yaml").write_text("operator: {}\n", encoding="utf-8")
    return SimpleNamespace(task="level1/unit", task_dir=task_dir)


def _cache_case(case_id: str = "1") -> SimpleNamespace:
    return SimpleNamespace(get_case_id_str=lambda: case_id)


def _cache_value() -> dict[str, object]:
    return {"inputs": [torch.ones(2)], "oracle": torch.zeros(2)}


def test_reference_cache_cold_then_warm_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    case = _cache_case()
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "required")
    monkeypatch.setenv("TF_L3_CACHE_EXPECTATION", "cold")
    cold = cannbench_contract.ReferenceCache(harness)
    with cold:
        cold.get(case, "oracle", _cache_value)
    cold_observation = cold.observation()
    assert cold_observation["required"] is True
    assert cold_observation["usable"] is True
    assert cold_observation["hits"] == 0
    assert cold_observation["misses"] == 1
    assert cold_observation["writes"] == 1
    assert cold_observation["byKind"]["oracle"] == {"hits": 0, "misses": 1, "writes": 1}
    assert cold_observation["errors"] == []

    monkeypatch.setenv("TF_L3_CACHE_EXPECTATION", "warm")
    warm = cannbench_contract.ReferenceCache(harness)
    with warm:
        value = warm.get(
            case,
            "oracle",
            lambda: pytest.fail("warm cache must not recompute the oracle"),
        )
    warm_observation = warm.observation()
    assert warm_observation["hits"] == 1
    assert warm_observation["misses"] == 0
    assert warm_observation["writes"] == 0
    assert value["oracle"].tolist() == [0.0, 0.0]


def test_required_cache_reports_unusable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", str(blocker))
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "required")
    cache = cannbench_contract.ReferenceCache(harness)
    with pytest.raises(cannbench_contract.ReferenceCacheError):
        with cache:
            pass
    observation = cache.observation()
    assert observation["usable"] is False
    assert observation["required"] is True
    assert observation["errors"]


def test_required_cache_write_failure_is_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "required")

    def fail_save(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(cannbench_contract.torch, "save", fail_save)
    cache = cannbench_contract.ReferenceCache(harness)
    with cache:
        with pytest.raises(cannbench_contract.ReferenceCacheError):
            cache.get(_cache_case(), "oracle", _cache_value)
    observation = cache.observation()
    assert observation["writes"] == 0
    assert observation["errors"]


def test_optional_cache_write_failure_keeps_computed_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "optional")

    def fail_save(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(cannbench_contract.torch, "save", fail_save)
    cache = cannbench_contract.ReferenceCache(harness)
    with cache:
        value = cache.get(_cache_case(), "oracle", _cache_value)
    assert value["oracle"].tolist() == [0.0, 0.0]
    assert cache.observation()["writes"] == 0
    assert cache.observation()["errors"]


def test_off_cache_mode_computes_without_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", "off")
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "off")
    cache = cannbench_contract.ReferenceCache(harness)
    with cache:
        value = cache.get(_cache_case(), "oracle", _cache_value)
    observation = cache.observation()
    assert observation["mode"] == "off"
    assert observation["usable"] is True
    assert observation["hits"] == observation["misses"] == observation["writes"] == 0
    assert value["oracle"].tolist() == [0.0, 0.0]


def test_required_cache_lease_contention_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _cache_harness(tmp_path)
    monkeypatch.setenv("CANNBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CANNBENCH_CACHE_MODE", "required")
    first = cannbench_contract.ReferenceCache(harness)
    first.__enter__()
    try:
        second = cannbench_contract.ReferenceCache(harness)
        with pytest.raises(cannbench_contract.ReferenceCacheError):
            second.__enter__()
        assert second.observation()["usable"] is False
    finally:
        first.__exit__(None, None, None)


def _write_fake_comparator(root: Path) -> Path:
    src = root / "src"
    compare_dir = src / "kernel_eval" / "utils"
    compare_dir.mkdir(parents=True)
    compare_dir.joinpath("compare.py").write_text(
        "from types import SimpleNamespace\n"
        "CALLS = []\n"
        "def compare_tensors(actual, golden, *, dtype, native_output=None,\n"
        "                    custom_thresholds=None):\n"
        "    CALLS.append({'dtype': dtype, 'native': native_output,\n"
        "                  'thresholds': custom_thresholds})\n"
        "    passed = native_output is not None\n"
        "    return SimpleNamespace(passed=passed, mere=0.0, mare=0.0,\n"
        "                           max_diff=0.0, error_msg='')\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def isolated_comparator(monkeypatch: pytest.MonkeyPatch):
    _ = monkeypatch
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "kernel_eval" or name.startswith("kernel_eval.")
    }
    saved_comparator = cannbench_contract._COMPARE_CASE_RESULT
    yield
    for name in list(sys.modules):
        if (
            name == "kernel_eval" or name.startswith("kernel_eval.")
        ) and name not in saved_modules:
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)
    cannbench_contract._COMPARE_CASE_RESULT = saved_comparator


def test_comparator_fallback_uses_cannbench_thresholds_and_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_comparator: None
) -> None:
    _ = isolated_comparator
    repo = _write_fake_comparator(tmp_path / "cann-bench")
    monkeypatch.setenv("TF_L3_CANNBENCH_REPO", str(repo))
    monkeypatch.setattr(cannbench_contract, "_COMPARE_CASE_RESULT", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "npu_compare", raising=False)
    for name in [
        name
        for name in sys.modules
        if name == "kernel_eval" or name.startswith("kernel_eval.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "npu_compare":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    compare = cannbench_contract._load_compare_case_result()
    assert cannbench_contract._resolve_cannbench_src() == repo / "src"
    native = torch.ones(2)
    result = compare(
        case_id="case:0",
        actual=torch.ones(2, dtype=torch.bfloat16),
        golden=torch.zeros(2),
        native_ref_operator=lambda _: native,
        extra={"precision_thresholds": {"foo": 1e-3}},
    )
    assert result.passed is True
    module = sys.modules["kernel_eval.utils.compare"]
    calls = getattr(module, "CALLS")
    assert calls[0]["dtype"] == "bfloat16"
    assert calls[0]["thresholds"] == {"foo": 1e-3}
    assert calls[0]["native"] is None
    assert calls[1]["native"] is native


def _acceptance_module() -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_l3_acceptance_under_test",
        Path(__file__).resolve().parent / "test_npu_acceptance.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FailingHarness:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def run(self, operator_api: object, *, observer: object, node: str) -> None:
        _ = (operator_api, observer, node)
        raise self.error


def test_mid_run_required_cache_error_blocks_the_promotion(tmp_path: Path) -> None:
    from _pytest.outcomes import Exit

    acceptance = _acceptance_module()
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    error = cannbench_contract.ReferenceCacheError(
        "required reference cache write failed: disk full", {}
    )
    with pytest.raises(Exit) as excinfo:
        acceptance._run_harness(_FailingHarness(error), None, recorder, "node_one")
    assert excinfo.value.returncode == l3_contract.BLOCKED_EXIT_CODE
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "blocked:reference_cache_error"
    assert payload["interruption"]["errorCategory"] == "reference_cache_error"
    assert payload["coverage"]["qualified"] is False


def test_mid_run_cache_error_is_not_swallowed_without_a_recorder(
    tmp_path: Path,
) -> None:
    acceptance = _acceptance_module()
    error = cannbench_contract.ReferenceCacheError("required cache unavailable", {})
    recorder = SimpleNamespace(active=False)
    with pytest.raises(cannbench_contract.ReferenceCacheError) as excinfo:
        acceptance._run_harness(_FailingHarness(error), None, recorder, "node_one")
    assert excinfo.value is error
    assert not (tmp_path / "evidence.json").exists()


class _StubCase:
    def __init__(self, label: str) -> None:
        self.label = label

    def get_case_id_str(self) -> str:
        return self.label


class _PassThroughCache:
    def get(
        self, case: object, kind: str, compute: object
    ) -> dict[str, object]:
        _ = (case, kind, compute)
        return {"inputs": [], "oracle": []}


class _StubHarness:
    task = "level1/unit"

    def __init__(
        self,
        *,
        passed: bool = True,
        compare_error: BaseException | None = None,
    ) -> None:
        self.cases = (_StubCase("1"),)
        self.passed = passed
        self.compare_error = compare_error

    def prepare(self, case: object) -> list[object]:
        _ = case
        return []

    def params(
        self, case: object, tensors: object, function: object = None
    ) -> dict[str, object]:
        _ = (case, tensors, function)
        return {}

    def reference_output(
        self, case: object, tensors: object, *, native: bool = False
    ) -> list[object]:
        _ = (case, tensors, native)
        return []

    def compare(
        self,
        case: object,
        actual: object,
        oracle: object,
        tensors: object,
        native_fn: object,
    ) -> tuple[bool, list[dict[str, object]]]:
        _ = (case, actual, oracle, tensors, native_fn)
        if self.compare_error is not None:
            raise self.compare_error
        details = [{"output": 0, "passed": False, "message": "numeric mismatch"}]
        return self.passed, details


def _constant_operator(**params: object) -> object:
    _ = params
    return object()


def _raising_operator(**params: object) -> object:
    _ = params
    raise RuntimeError("device execution failed")


def _run_stub_harness(
    tmp_path: Path,
    harness: _StubHarness,
    operator: object = _constant_operator,
) -> dict[str, object]:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    with pytest.raises(AssertionError):
        with recorder.node("node_one"):
            cannbench_contract.Harness._run(
                harness, operator, _PassThroughCache(), recorder, "node_one"
            )
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def test_comparator_deviation_records_candidate_numeric_evidence(
    tmp_path: Path,
) -> None:
    payload = _run_stub_harness(tmp_path, _StubHarness(passed=False))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "candidate"
    assert failure["category"] == "numeric"
    assert failure["stage"] == "comparison"
    assert failure["node"] == "node_one"
    assert failure["case"] == "1"
    assert failure["role"] == "candidate"
    assert failure["comparator"] == "cannbench"
    assert failure["references"] == ["oracle", "native"]
    comparisons = failure["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["output"] == 0
    assert comparisons[0]["passed"] is False
    case = _mapping(_mapping(payload["cases"])["node_one:1"])
    assert _mapping(case["failure"])["kind"] == "candidate"
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    assert _mapping(nodes[0])["failure"] == failure
    assert _mapping(payload["coverage"])["qualified"] is False


def test_kernel_exception_is_infrastructure_not_candidate(tmp_path: Path) -> None:
    payload = _run_stub_harness(
        tmp_path, _StubHarness(), operator=_raising_operator
    )
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "infrastructure"
    assert failure["category"] == "exception"
    assert failure["stage"] == "kernel"
    assert failure["exceptionType"] == "RuntimeError"
    assert failure["node"] == "node_one"
    assert failure["case"] == "1"
    assert failure["role"] is None
    assert failure["comparisons"] == []


def test_non_numeric_comparison_assertion_is_infrastructure(
    tmp_path: Path,
) -> None:
    payload = _run_stub_harness(
        tmp_path,
        _StubHarness(compare_error=AssertionError("output count differs")),
    )
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "infrastructure"
    assert failure["category"] == "assertion"
    assert failure["stage"] == "comparison"
    assert failure["exceptionType"] == "AssertionError"
    assert failure["role"] is None


def test_baseline_deviation_is_infrastructure_with_baseline_role(
    tmp_path: Path,
) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    failure = l3_contract.comparison_failure(
        node="node_one",
        case_id="1",
        role="baseline",
        comparator="torch.testing.assert_close",
        references=("oracle",),
        comparisons=({"output": 0, "passed": False, "message": "baseline differs"},),
    )
    with pytest.raises(l3_contract.ClassifiedTestFailure):
        with recorder.node("node_one"):
            recorder.case_started("node_one", "1")
            recorder.case_finished("node_one", "1", "failed", "baseline differs", failure=failure)
            raise l3_contract.ClassifiedTestFailure("baseline differs")
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    top = _mapping(payload["failure"])
    assert top["kind"] == "infrastructure"
    assert top["category"] == "baseline_mismatch"
    assert top["role"] == "baseline"
    assert top["node"] == "node_one"
    assert top["case"] == "1"
    assert _mapping(payload["coverage"])["qualified"] is False


def test_case_identity_gap_is_coverage(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    failure = l3_contract.coverage_failure(
        node="node_one",
        case_id="1",
        category="case_identity",
        detail="case label does not match the contract",
    )
    with pytest.raises(l3_contract.CoverageContractError):
        with recorder.node("node_one"):
            raise l3_contract.CoverageContractError(
                "case label does not match the contract", failure
            )
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    top = _mapping(payload["failure"])
    assert top["kind"] == "coverage"
    assert top["category"] == "case_identity"
    assert _mapping(payload["coverage"])["qualified"] is False


def test_unexplained_missing_cases_are_coverage(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    payload = recorder.payload()
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "coverage"
    assert failure["category"] in {"missing_node", "missing_case"}
    failures = payload["failures"]
    assert isinstance(failures, list) and len(failures) == 1


def test_plain_assertion_is_infrastructure_not_candidate(tmp_path: Path) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    with pytest.raises(AssertionError):
        with recorder.node("node_one"):
            raise AssertionError("structural check failed")
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "infrastructure"
    assert failure["category"] == "assertion"
    assert failure["stage"] == "report"


def test_infrastructure_precedes_candidate_when_both_are_recorded(
    tmp_path: Path,
) -> None:
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    candidate = l3_contract.comparison_failure(
        node="node_one",
        case_id="1",
        role="candidate",
        comparator="cannbench",
        references=("oracle", "native"),
        comparisons=({"output": 0, "passed": False, "message": "numeric mismatch"},),
    )
    infrastructure = l3_contract.infrastructure_failure(
        node="node_one",
        case_id="1",
        stage="kernel",
        category="exception",
        exception_type="RuntimeError",
        detail="device execution failed",
    )
    recorder.case_started("node_one", "1")
    recorder.case_finished("node_one", "1", "failed", "numeric mismatch", failure=candidate)
    recorder.record_failure(infrastructure)
    payload = recorder.payload()
    assert _mapping(payload["failure"])["kind"] == "infrastructure"
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert [row["kind"] for row in failures] == ["candidate", "infrastructure"]


def test_coverage_precedes_candidate_for_an_unexplained_missing_case(
    tmp_path: Path,
) -> None:
    """Coverage outranks candidate when a contracted case is absent.

    Case 2 is intentionally absent and has no structured failure.
    """
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _unit_contract(tmp_path)
    )
    candidate = l3_contract.comparison_failure(
        node="node_one",
        case_id="1",
        role="candidate",
        comparator="cannbench",
        references=("oracle", "native"),
        comparisons=({"output": 0, "passed": False, "message": "numeric mismatch"},),
    )
    with recorder.node("node_one"):
        recorder.case_finished("node_one", "1", "failed", "numeric mismatch", failure=candidate)
    with recorder.node("node_two"):
        recorder.case_finished("node_two", "4", "passed")
    payload = recorder.payload()
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "coverage"
    assert failure["category"] == "missing_case"
    assert failure["case"] == "2"
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert [row["kind"] for row in failures] == ["candidate", "coverage"]


def test_dynamic_shape_candidate_deviation_is_recorded_as_candidate(
    tmp_path: Path,
) -> None:
    acceptance = _acceptance_module()
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _dynamic_case_contract(tmp_path)
    )

    def check() -> None:
        acceptance._assert_candidate_close(
            node=l3_contract.DYNAMIC_NODE,
            case_id="1",
            actual=torch.zeros(2),
            reference=torch.ones(2),
            rtol=0.0,
            atol=0.0,
        )

    with pytest.raises(l3_contract.CandidateComparisonError):
        acceptance._record_dynamic_shape(recorder, "1", check)
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "candidate"
    assert failure["category"] == "numeric"
    assert failure["node"] == l3_contract.DYNAMIC_NODE
    assert failure["case"] == "1"
    assert failure["role"] == "candidate"
    assert failure["comparator"] == "torch.testing.assert_close"
    comparisons = failure["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["passed"] is False


def test_dynamic_shape_kernel_exception_is_recorded_as_infrastructure(
    tmp_path: Path,
) -> None:
    acceptance = _acceptance_module()
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _dynamic_case_contract(tmp_path)
    )

    def check() -> None:
        raise RuntimeError("device execution failed")

    with pytest.raises(l3_contract.ClassifiedTestFailure):
        acceptance._record_dynamic_shape(recorder, "1", check)
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "infrastructure"
    assert failure["category"] == "exception"
    assert failure["stage"] == "kernel"
    assert failure["exceptionType"] == "RuntimeError"
    assert failure["role"] is None


def test_dynamic_shape_non_numeric_assertion_is_recorded_as_infrastructure(
    tmp_path: Path,
) -> None:
    acceptance = _acceptance_module()
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _dynamic_case_contract(tmp_path)
    )

    def check() -> None:
        raise AssertionError("shape mismatch")

    with pytest.raises(l3_contract.ClassifiedTestFailure):
        acceptance._record_dynamic_shape(recorder, "1", check)
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "infrastructure"
    assert failure["category"] == "assertion"
    assert failure["stage"] == "kernel"
    assert failure["role"] is None


def test_contract_case_identity_mismatch_is_coverage(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    with pytest.raises(l3_contract.CoverageContractError) as excinfo:
        acceptance._require_contract_cases(
            recorder, "node_one", ("1",), ("different",)
        )
    failure = excinfo.value.failure
    assert failure["kind"] == "coverage"
    assert failure["category"] == "case_identity"
    assert failure["node"] == "node_one"


def test_skipped_node_is_coverage_not_infrastructure(tmp_path: Path) -> None:
    from _pytest.outcomes import Skipped

    recorder = l3_contract.EvidenceRecorder(
        tmp_path / "evidence.json", _single_case_contract(tmp_path)
    )
    with pytest.raises(Skipped):
        with recorder.node("node_one"):
            raise Skipped("no available device")
    recorder.note_report(
        "test_npu_acceptance.py::node_one", "setup", "skipped", "no available device"
    )
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    assert failure["kind"] == "coverage"
    assert failure["category"] == "skipped_node"
    assert failure["node"] == "node_one"
    assert _mapping(payload["coverage"])["qualified"] is False


def test_pytest_call_phase_candidate_failure_is_serialized_as_failure(
    tmp_path: Path,
) -> None:
    benchmarks = Path(__file__).resolve().parent
    shutil.copyfile(benchmarks / "conftest.py", tmp_path / "conftest.py")
    test_source = f'''
import l3_contract


def _required(recorder, node):
    return next(
        item.required_cases for item in recorder.contract.nodes if item.name == node
    )


def _pass_node(recorder, node):
    with recorder.node(node):
        for case_id in _required(recorder, node):
            recorder.case_started(node, case_id)
            recorder.case_finished(node, case_id, "passed")


def {l3_contract.AR_NODE}():
    recorder = l3_contract.active_recorder()
    assert recorder is not None
    node = {l3_contract.AR_NODE!r}
    required = _required(recorder, node)
    with recorder.node(node):
        for case_id in required:
            recorder.case_started(node, case_id)
            if case_id == required[0]:
                failure = l3_contract.comparison_failure(
                    node=node,
                    case_id=case_id,
                    role="candidate",
                    comparator="torch.testing.assert_close",
                    references=("oracle",),
                    comparisons=(
                        {{"output": 0, "passed": False, "message": "numeric mismatch"}},
                    ),
                )
                recorder.case_finished(
                    node, case_id, "failed", "numeric mismatch", failure=failure
                )
            else:
                recorder.case_finished(node, case_id, "passed")
        raise l3_contract.ClassifiedTestFailure("numeric mismatch")


def {l3_contract.AR_PARITY_NODE}():
    recorder = l3_contract.active_recorder()
    assert recorder is not None
    _pass_node(recorder, {l3_contract.AR_PARITY_NODE!r})


def {l3_contract.DYNAMIC_NODE}():
    recorder = l3_contract.active_recorder()
    assert recorder is not None
    _pass_node(recorder, {l3_contract.DYNAMIC_NODE!r})
'''
    test_file = tmp_path / "test_npu_acceptance.py"
    test_file.write_text(test_source, encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    junit_path = tmp_path / "junit.xml"
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(benchmarks)
        if not existing_path
        else f"{benchmarks}{os.pathsep}{existing_path}"
    )
    env["TF_L3_EVIDENCE_PATH"] = str(evidence_path)
    env["TF_L3_TARGET_OPERATOR"] = "level4/attention_residual"
    env["TF_L3_EXPECTED_PYTHON"] = str(Path(sys.executable).resolve())
    for key in (
        "TF_L3_TARGET_MODULE_PATH",
        "TF_L3_CACHE_EXPECTATION",
        "TF_L3_PROFILE",
        "CANNBENCH_CACHE_MODE",
        "CANNBENCH_CACHE_DIR",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit_path}",
            str(test_file),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1, (
        f"pytest returncode {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert evidence_path.is_file(), result.stdout + result.stderr
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    failure = _mapping(payload["failure"])
    ar_required = next(
        node.required_cases
        for node in l3_contract.resolve_contract("level4/attention_residual").nodes
        if node.name == l3_contract.AR_NODE
    )
    assert failure["kind"] == "candidate"
    assert failure["category"] == "numeric"
    assert failure["node"] == l3_contract.AR_NODE
    assert failure["case"] == ar_required[0]
    assert failure["role"] == "candidate"
    assert failure["comparator"] == "torch.testing.assert_close"
    comparisons = failure["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["passed"] is False
    assert _mapping(payload["coverage"])["qualified"] is False
    assert _mapping(payload["interruption"])["interrupted"] is False

    root = ElementTree.parse(junit_path).getroot()
    failed_cases = [
        testcase.attrib["name"]
        for testcase in root.iter("testcase")
        if testcase.find("failure") is not None
    ]
    assert failed_cases == [l3_contract.AR_NODE]
    assert not any(testcase.find("error") is not None for testcase in root.iter("testcase"))
