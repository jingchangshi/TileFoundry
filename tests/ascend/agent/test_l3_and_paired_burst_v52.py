"""CPU unit tests for L3 failure precedence, paired burst timing, slice errors, and telemetry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ascend.agent import test_surrogate_batch_npu as batch_harness
from tests.ascend.benchmarks.l3_contract import (
    EvidenceRecorder,
    NodeContract,
    TargetContract,
    comparison_failure,
    coverage_failure,
    infrastructure_failure,
)
from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    DeterministicSearchBoundaryLedger,
    is_clean_calibration_sample,
    normalize_compile_error_code,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    PatchAnchorMissError,
    apply_scoped_trackb_patch,
    splice_cpp_units_into_tu,
)


class _DriftingClockEvent:
    """Synthetic Event reading a shared microsecond clock list."""

    def __init__(self, clock: list[float]) -> None:
        self._clock = clock
        self._t_us = 0.0

    def record(self) -> None:
        self._t_us = self._clock[0]

    def elapsed_time(self, other: _DriftingClockEvent) -> float:
        return (other._t_us - self._t_us) / 1000.0


def _build_two_node_recorder(tmp_path: Path) -> EvidenceRecorder:
    contract = TargetContract(
        operator="mla",
        task="level4/mla",
        task_dir=tmp_path,
        nodes=(
            NodeContract(name="test_op_cases", required_cases=("0", "1", "2", "3", "4", "5")),
            NodeContract(name="test_other", required_cases=("0",)),
        ),
        case_ids=("0", "1", "2", "3", "4", "5"),
    )
    return EvidenceRecorder(
        tmp_path / "l3_evidence.json",
        contract,
    )


def test_l3_contract_numeric_failure_not_preempted_by_collateral_missing_cases(
    tmp_path: Path,
) -> None:
    """Same-node missing cases after a candidate failure must not preempt candidate status."""
    rec = _build_two_node_recorder(tmp_path)
    for cid in ("0", "1", "2"):
        rec.case_finished("test_op_cases", cid, "passed")
    cand_fail = comparison_failure(
        node="test_op_cases",
        case_id="3",
        role="candidate",
        comparator="torch.testing.assert_close",
        references=["oracle"],
        comparisons=[{"output": 0, "passed": False, "message": "max_abs_error"}],
        detail="case 3 numeric mismatch",
    )
    rec.case_finished("test_op_cases", "3", "failed", detail="case 3 mismatch", failure=cand_fail)
    rec.node_finished("test_op_cases", "failed", detail="case 3 mismatch")
    rec.case_finished("test_other", "0", "passed")
    rec.node_finished("test_other", "passed")
    rec.finalize()

    cov = rec.coverage()
    assert cov["cases"]["missing"] == ["test_op_cases:3", "test_op_cases:4", "test_op_cases:5"]
    assert rec._coverage_gap_failure(cov) is None

    payload = rec.payload()
    assert payload["failure"] is not None
    assert payload["failure"]["kind"] == "candidate"
    assert payload["failure"]["category"] == "numeric"
    assert payload["failure"]["node"] == "test_op_cases"
    assert payload["failure"]["case"] == "3"

    evidence = dict(payload)
    evidence["timing"] = {"startedAt": "2026-11-25T00:00:00Z", "finishedAt": "2026-11-25T00:00:05Z"}
    obs = {"failureCategory": "numeric", "evidence": evidence}
    kind, pending = surrogate_cannbench.classify_l3_observation_failure(obs)
    assert kind == surrogate_cannbench.FAILURE_KIND_CANDIDATE
    assert pending is False

    collateral_row = coverage_failure(
        node="test_op_cases",
        case_id="4",
        category="missing_case",
        detail="test_op_cases:4 was not completed",
    )
    evidence_with_collateral = dict(evidence)
    evidence_with_collateral["failures"] = [cand_fail, collateral_row]
    kind_col, pending_col = surrogate_cannbench.classify_l3_observation_failure(
        {"failureCategory": "numeric", "evidence": evidence_with_collateral}
    )
    assert kind_col == surrogate_cannbench.FAILURE_KIND_CANDIDATE
    assert pending_col is False

    kernel_row = {
        "kind": "candidate",
        "category": "candidate",
        "stage": "compile",
        "node": "test_op_cases",
        "case": "3",
        "role": "candidate",
        "comparator": None,
        "references": [],
        "comparisons": [],
        "exceptionType": "RuntimeError",
        "detail": "exceeded vf stack size",
    }
    kernel_evidence = dict(evidence)
    kernel_evidence["failure"] = kernel_row
    kernel_evidence["failures"] = [kernel_row, collateral_row]
    kind_k, pending_k = surrogate_cannbench.classify_l3_observation_failure(
        {"failureCategory": "candidate", "evidence": kernel_evidence}
    )
    assert kind_k == surrogate_cannbench.FAILURE_KIND_CANDIDATE
    assert pending_k is False


def test_l3_cross_node_missing_case_and_infra_still_block(tmp_path: Path) -> None:
    """Missing cases on another node or infrastructure faults keep blocking priority."""
    rec = _build_two_node_recorder(tmp_path)
    rec.case_finished("test_op_cases", "0", "passed")
    cand_fail = comparison_failure(
        node="test_op_cases",
        case_id="3",
        role="candidate",
        comparator="torch.testing.assert_close",
        references=["oracle"],
        comparisons=[{"output": 0, "passed": False, "message": "max_abs_error"}],
        detail="case 3 numeric mismatch",
    )
    rec.case_finished("test_op_cases", "3", "failed", failure=cand_fail)
    rec.node_finished("test_op_cases", "failed")
    rec.finalize()

    gap = rec._coverage_gap_failure()
    assert gap is not None
    assert gap["kind"] == "coverage"
    assert gap["category"] == "missing_node"
    assert gap["node"] == "test_other"

    rec_case_gap = EvidenceRecorder(
        tmp_path / "l3_evidence_cross_case.json",
        TargetContract(
            operator="mla",
            task="level4/mla",
            task_dir=tmp_path,
            nodes=(
                NodeContract(name="test_op_cases", required_cases=("0", "3", "4")),
                NodeContract(name="test_other", required_cases=("0", "1")),
            ),
            case_ids=("0", "1", "3", "4"),
        ),
    )
    rec_case_gap.case_finished("test_op_cases", "0", "passed")
    rec_case_gap.case_finished("test_op_cases", "3", "failed", failure=cand_fail)
    rec_case_gap.node_finished("test_op_cases", "failed")
    rec_case_gap.case_finished("test_other", "0", "passed")
    rec_case_gap.node_finished("test_other", "passed")
    rec_case_gap.finalize()
    cross_gap = rec_case_gap._coverage_gap_failure()
    assert cross_gap is not None
    assert cross_gap["kind"] == "coverage"
    assert cross_gap["category"] == "missing_case"
    assert cross_gap["node"] == "test_other"
    assert cross_gap["case"] == "1"

    payload = rec.payload()
    assert payload["failure"]["kind"] == "coverage"

    cross_node_missing = coverage_failure(
        node="test_other",
        case_id="0",
        category="missing_case",
        detail="test_other:0 was not completed",
    )
    infra_row = infrastructure_failure(
        node="test_op_cases",
        case_id="3",
        stage="run",
        category="exception",
        detail="device offline",
    )
    for bad_row in (cross_node_missing, infra_row):
        evidence = dict(payload)
        evidence["timing"] = {"finishedAt": "2026-11-25T00:00:05Z"}
        evidence["failure"] = cand_fail
        evidence["failures"] = [cand_fail, bad_row]
        kind, pending = surrogate_cannbench.classify_l3_observation_failure(
            {"failureCategory": "numeric", "evidence": evidence}
        )
        assert kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
        assert pending is True


def test_paired_interleaved_micro_burst_cancels_slow_drift_and_roundtrips() -> None:
    """Interleaved pairing achieves low relative MAD despite >20% common-mode clock drift."""
    spec = AnchorCaseSpec("level4/mla", 3, "decode", 20.0, 6.0)
    built = surrogate_cannbench.build_anchor_metrics(
        spec,
        accuracy_passed=True,
        cand_us=10.0,
        burst_batch_cv=0.015,
        paired_speedup_vs_incumbent=1.85,
        paired_rel_mad=0.015,
    )
    restored = surrogate_cannbench.AnchorEvalMetrics.from_dict(built.to_dict())
    assert restored.paired_speedup_vs_incumbent == pytest.approx(1.85)
    assert restored.paired_rel_mad == pytest.approx(0.015)

    legacy = built.to_dict()
    legacy.pop("paired_speedup_vs_incumbent")
    legacy.pop("paired_rel_mad")
    restored_legacy = surrogate_cannbench.AnchorEvalMetrics.from_dict(legacy)
    assert restored_legacy.paired_speedup_vs_incumbent == 0.0
    assert restored_legacy.paired_rel_mad == 0.0

    clock = [0.0]
    drift_factors = [1.0, 1.15, 1.35, 1.60, 1.85, 2.05, 2.30]
    call_counter = {"inc": 0, "fast": 0}

    def _incumbent_fn(_x: object) -> None:
        idx = call_counter["inc"]
        call_counter["inc"] += 1
        drift_idx = max(0, min(len(drift_factors) - 1, idx - 1))
        clock[0] += 20.0 * drift_factors[drift_idx]

    def _fast_fn(_x: object) -> None:
        idx = call_counter["fast"]
        call_counter["fast"] += 1
        drift_idx = max(0, min(len(drift_factors) - 1, idx - 1))
        clock[0] += 10.0 * drift_factors[drift_idx]

    stats = surrogate_cannbench.measure_paired_interleaved_micro_burst(
        [("__incumbent__", _incumbent_fn), ("cand_fast", _fast_fn)],
        object(),
        warmup=1,
        repeat=7,
        t_hw_us=10.0,
        first_probe_us=20.0,
        event_factory=lambda: _DriftingClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda obj: obj,
    )
    inc_us, inc_k, inc_raw_cv, inc_ratio, inc_mad = stats["__incumbent__"]
    fast_us, fast_k, fast_raw_cv, fast_ratio, fast_mad = stats["cand_fast"]
    assert inc_k == 1
    assert fast_k == 1
    assert inc_raw_cv > 0.20
    assert fast_raw_cv > 0.20
    assert inc_ratio == pytest.approx(1.0)
    assert inc_mad == pytest.approx(0.0)
    assert fast_ratio == pytest.approx(2.0, rel=1e-2)
    assert fast_mad < 0.02
    assert inc_us > fast_us

    burst_clock = [0.0]
    burst_state = {"round": 0, "inc_calls": 0, "fast_calls": 0}
    k_burst = surrogate_cannbench.compute_ring_burst_k(0, t_hw_us=2.0)
    assert k_burst >= 24

    def _burst_inc(_x: object) -> None:
        burst_state["inc_calls"] += 1
        r_idx = min(
            len(drift_factors) - 1,
            max(0, (burst_state["inc_calls"] - 2) // k_burst),
        )
        burst_clock[0] += 4.0 * drift_factors[r_idx]

    def _burst_fast(_x: object) -> None:
        burst_state["fast_calls"] += 1
        r_idx = min(
            len(drift_factors) - 1,
            max(0, (burst_state["fast_calls"] - 1) // k_burst),
        )
        burst_clock[0] += 2.0 * drift_factors[r_idx]

    burst_stats = surrogate_cannbench.measure_paired_interleaved_micro_burst(
        [("__incumbent__", _burst_inc), ("cand_fast", _burst_fast)],
        object(),
        warmup=1,
        repeat=7,
        t_hw_us=2.0,
        event_factory=lambda: _DriftingClockEvent(burst_clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda obj: obj,
    )
    assert burst_stats["cand_fast"][1] == k_burst
    assert burst_stats["cand_fast"][2] > 0.20
    assert burst_stats["cand_fast"][3] == pytest.approx(2.0, rel=1e-2)
    assert burst_stats["cand_fast"][4] < 0.02


def test_evaluate_operator_batch_step_a_gate_and_paired_incumbent_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_evaluate_operator_batch filters accuracy first and computes paired speedup vs incumbent."""
    clock = [0.0]
    drift = [1.0, 1.2, 1.45, 1.7, 2.0, 2.25, 2.5]
    round_state = {"idx": 0, "calls": 0}

    class _FakeTensor:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def npu(self) -> _FakeTensor:
            return self

        def clone(self) -> _FakeTensor:
            return _FakeTensor(self.tag)

        def numel(self) -> int:
            return 1024

        def element_size(self) -> int:
            return 2

    class _FakeHarness:
        def prepare(self, _case: object) -> dict[str, _FakeTensor]:
            return {"x": _FakeTensor("in")}

        def reference_output(
            self, _case: object, _ts: object, native: bool = False
        ) -> _FakeTensor:
            _ = native
            return _FakeTensor("ref")

        def params(self, _case: object, tensors: dict[str, _FakeTensor]) -> dict[str, object]:
            return {"x": tensors["x"]}

        def compare(
            self, _case: object, actual: object, _oracle: object, _tensors: object, _native_fn: object
        ) -> tuple[bool, dict[str, object]]:
            if actual == "bad":
                return False, {"reason": "numeric_mismatch"}
            return True, {}

    class _FakeRefCache:
        def __init__(self, _harness: object) -> None:
            self._store: dict[tuple[int, str], object] = {}

        def __enter__(self) -> _FakeRefCache:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, case: object, tag: str, builder: object) -> object:
            key = (int(getattr(case, "case_num", 1)), tag)
            if key not in self._store:
                self._store[key] = builder()
            return self._store[key]

    fake_contract = SimpleNamespace(
        ReferenceCache=_FakeRefCache,
        map_tensors=lambda obj, fn: {k: fn(v) for k, v in obj.items()},
    )
    monkeypatch.setattr(
        batch_harness,
        "load_cannbench_contract_module",
        lambda **_kwargs: fake_contract,
    )
    monkeypatch.setattr(
        batch_harness,
        "load_anchor_harness_and_cases",
        lambda _op, anchors, **_kwargs: (
            _FakeHarness(),
            [SimpleNamespace(case_num=int(a.case_num)) for a in anchors],
        ),
    )

    class _BoundClockEvent(_DriftingClockEvent):
        def __init__(self, enable_timing: bool = True) -> None:
            _ = enable_timing
            super().__init__(clock)

    fake_npu = SimpleNamespace(
        Event=_BoundClockEvent,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: True,
        set_device=lambda _idx: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    inc_mod = tmp_path / "inc.py"
    inc_mod.write_text("def run_op(x):\n    return 'inc'\n", encoding="utf-8")
    fast_mod = tmp_path / "fast.py"
    fast_mod.write_text("def run_op(x):\n    return 'fast'\n", encoding="utf-8")
    bad_mod = tmp_path / "bad.py"
    bad_mod.write_text("def run_op(x):\n    return 'bad'\n", encoding="utf-8")

    bad_calls = [0]

    def _fake_load(mod_path: Path, _entrypoint: str, _tag: str):
        if mod_path == inc_mod:
            def _inc_op(x: object) -> str:
                _ = x
                r = min(len(drift) - 1, max(0, (round_state["calls"] - 3) // 2))
                round_state["calls"] += 1
                clock[0] += 24.0 * drift[r]
                return "inc"

            return _inc_op, SimpleNamespace()
        if mod_path == fast_mod:
            def _fast_op(x: object) -> str:
                _ = x
                r = min(len(drift) - 1, max(0, (round_state["calls"] - 3) // 2))
                round_state["calls"] += 1
                clock[0] += 12.0 * drift[r]
                return "fast"

            return _fast_op, SimpleNamespace()

        def _bad_op(x: object) -> str:
            _ = x
            bad_calls[0] += 1
            return "bad"

        return _bad_op, SimpleNamespace()

    monkeypatch.setattr(batch_harness, "_load_candidate_entrypoint", _fake_load)

    batch = {
        "operator_name": "mla",
        "entrypoint": "run_op",
        "candidates": [
            {"candidate_id": "__incumbent__", "module_path": str(inc_mod)},
            {"candidate_id": "cand_fast", "module_path": str(fast_mod)},
            {"candidate_id": "cand_bad_acc", "module_path": str(bad_mod)},
        ],
        "anchors": [AnchorCaseSpec("level4/mla", 3, "decode", 24.0, 6.0).to_dict()],
    }
    results = batch_harness._evaluate_operator_batch(
        batch,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
        warmup=1,
        repeat=7,
    )
    assert bad_calls[0] == 1
    assert results["cand_bad_acc"].all_passed is False
    assert results["cand_bad_acc"].anchors[0].cand_us == 0.0
    assert results["__incumbent__"].all_passed is True
    assert results["__incumbent__"].anchors[0].paired_speedup_vs_incumbent == pytest.approx(1.0)
    assert results["__incumbent__"].anchors[0].paired_rel_mad == pytest.approx(0.0)

    fast_anchor = results["cand_fast"].anchors[0]
    assert fast_anchor.accuracy_passed is True
    assert fast_anchor.paired_speedup_vs_incumbent == pytest.approx(2.0, rel=0.05)
    assert fast_anchor.paired_rel_mad < 0.02
    assert fast_anchor.burst_batch_cv == pytest.approx(fast_anchor.paired_rel_mad)
    assert is_clean_calibration_sample(fast_anchor.burst_batch_cv) is True

    single_batch = {
        "operator_name": "mla",
        "entrypoint": "run_op",
        "candidates": [{"candidate_id": "cand_fast", "module_path": str(fast_mod)}],
        "anchors": [AnchorCaseSpec("level4/mla", 3, "decode", 24.0, 6.0).to_dict()],
    }
    single_res = batch_harness._evaluate_operator_batch(
        single_batch,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
        warmup=1,
        repeat=5,
    )
    assert single_res["cand_fast"].all_passed is True
    assert single_res["cand_fast"].anchors[0].paired_speedup_vs_incumbent == 0.0


def test_cpp_slice_protocol_raises_patch_anchor_miss_error() -> None:
    """Unmatched, empty, or zero-change Track-B patches raise PatchAnchorMissError."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    int step = 1;\n"
        "    x[0] = x[0] + (float)step;\n"
        "}\n"
    )
    missing_patch = (
        "<<<<<<< SEARCH\n"
        "    int missing = 99;\n"
        "=======\n"
        "    int missing = 100;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(PatchAnchorMissError) as exc_missing:
        apply_scoped_trackb_patch(cpp, missing_patch, active_kernel="kernel_active")
    assert isinstance(exc_missing.value, ValueError)

    with pytest.raises(PatchAnchorMissError):
        apply_scoped_trackb_patch(cpp, "no valid search blocks here", active_kernel="kernel_active")

    empty_search = "<<<<<<< SEARCH\n   \n=======\n    int a = 1;\n>>>>>>> REPLACE\n"
    with pytest.raises(PatchAnchorMissError):
        apply_scoped_trackb_patch(cpp, empty_search, active_kernel="kernel_active")

    identical_patch = (
        "<<<<<<< SEARCH\n"
        "    int step = 1;\n"
        "=======\n"
        "    int step = 1;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(PatchAnchorMissError, match="zero effective changes"):
        apply_scoped_trackb_patch(cpp, identical_patch, active_kernel="kernel_active")

    identical_unit = (
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    int step = 1;\n"
        "    x[0] = x[0] + (float)step;\n"
        "}\n"
    )
    with pytest.raises(PatchAnchorMissError, match="identical to parent Track-B"):
        splice_cpp_units_into_tu(cpp, identical_unit, active_kernel="kernel_active")


def test_funnel_telemetry_error_normalization_and_ledger_sha_pruning() -> None:
    """Funnel telemetry normalizes Stage-A error codes and prunes repeated bad SHAs."""
    assert (
        normalize_compile_error_code(
            "stage_a", "PatchAnchorMissError: SEARCH block did not match target C++ source"
        )
        == "PATCH_ANCHOR_MISS"
    )
    assert (
        normalize_compile_error_code(
            "stage_a", "Spliced C++ units are identical to parent Track-B translation unit"
        )
        == "PATCH_ANCHOR_MISS"
    )
    assert (
        normalize_compile_error_code("stage_a", "DEAD_CODE_TRACKB_EDIT in helper")
        == "TRACKB_SLICE_SPLICE_ERR"
    )
    assert (
        normalize_compile_error_code("stage_a", "signature mismatch: missing regime symbols")
        == "SIGNATURE_CONTRACT_MISMATCH"
    )
    assert (
        normalize_compile_error_code("stage_a", "unexpected keyword argument 'foo'")
        == "SIGNATURE_CONTRACT_MISMATCH"
    )
    assert (
        normalize_compile_error_code("stage_a", "non-contiguous stride contract error")
        == "DYNAMIC_STRIDE_CONTRACT_ERR"
    )
    assert (
        normalize_compile_error_code("stage_b", "implicit declaration of function asc_vadd")
        == "CCE_UNDECLARED_INTRINSIC"
    )
    assert (
        normalize_compile_error_code("stage_b", "unknown type name 'half_t'")
        == "CCE_UNDECLARED_INTRINSIC"
    )

    ledger = DeterministicSearchBoundaryLedger()
    for idx, code in enumerate(
        ("PATCH_ANCHOR_MISS", "TRACKB_SLICE_SPLICE_ERR", "SIGNATURE_CONTRACT_MISMATCH")
    ):
        sha = f"bad_sha_{idx}"
        assert (
            ledger.record_compile_wall(
                regime_tag="decode",
                cand_sha=sha,
                knob_signature="bm=64",
                error_code_norm=code,
            )
            is True
        )
        assert (
            ledger.precheck_known_infeasible(
                regime_tag="decode",
                cand_sha=sha,
                knob_signature="bm=64",
            )
            == f"deterministic_boundary_sha:{code}"
        )
    assert (
        ledger.precheck_known_infeasible(
            regime_tag="decode",
            cand_sha="fresh_sha",
            knob_signature="bm=64",
        )
        is None
    )
