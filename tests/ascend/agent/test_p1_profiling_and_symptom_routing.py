"""Tests for msprof op CSV harness wiring, P2 deep-dive fallback, and symptom routing."""

from __future__ import annotations

import importlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ascend.agent import test_surrogate_batch_npu as batch_harness
from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    SymptomFamily,
)


def _write_executable(path: Path, script_body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if script_body.startswith("#!/usr/bin/env python3\n"):
        script_body = f"#!{sys.executable}\n" + script_body[len("#!/usr/bin/env python3\n") :]
    path.write_text(script_body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_msprof_op_falls_back_to_tilelang_profiler_when_unavailable_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback to tilelang.profiler.do_bench occurs if msprof is absent, disabled, or empty."""
    fallback_calls: list[dict[str, object]] = []

    def _fake_do_bench(_fn: object, **kwargs: object) -> SimpleNamespace:
        fallback_calls.append(dict(kwargs))
        return SimpleNamespace(
            aic_mad=75.0,
            aiv_vec=30.0,
            aiv_mte2=20.0,
            dur_us=14.0,
        )

    real_import = importlib.import_module

    def _mock_import(name: str, package: str | None = None):
        if name == "tilelang.profiler":
            return SimpleNamespace(do_bench=_fake_do_bench)
        return real_import(name, package)

    monkeypatch.setattr(batch_harness.importlib, "import_module", _mock_import)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _i: None), raising=False)
    monkeypatch.setattr(torch, "device", lambda spec: str(spec))

    monkeypatch.setattr(batch_harness.shutil, "which", lambda _name: None)
    no_bin_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
    )
    assert no_bin_metrics["aic_mac_ratio"] == pytest.approx(0.75)
    assert no_bin_metrics["dur_us"] == pytest.approx(14.0)
    assert len(fallback_calls) == 1

    empty_msprof = _write_executable(
        tmp_path / "bin" / "msprof",
        "#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n",
    )
    monkeypatch.setattr(
        batch_harness.shutil,
        "which",
        lambda name: str(empty_msprof) if name == "msprof" else None,
    )
    empty_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
    )
    assert empty_metrics["aiv_vec_ratio"] == pytest.approx(0.30)
    assert len(fallback_calls) == 2

    monkeypatch.setenv("TF_DISABLE_MSPROF_OP", "1")
    disabled_metrics = batch_harness._measure_msprof_detail_metrics(
        lambda: None,
        warmup=1,
        repeat=2,
        operator_name="softmax",
        module_path=tmp_path / "m.py",
        case_num=1,
        source_root=tmp_path,
        cannbench_repo=tmp_path,
    )
    assert disabled_metrics["mte2_wait_ratio"] == pytest.approx(0.20)
    assert len(fallback_calls) == 3


def test_collect_p2_deep_dive_and_bottleneck_msprof_via_agent_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify P2 deep-dive preserves non-numeric fields while bottleneck collector filters them."""
    src_root = tmp_path / "repo"
    src_root.mkdir(parents=True, exist_ok=True)
    cand = src_root / "candidate.py"
    cand.write_text("def run_op(x):\n    return x\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={tmp_path}\n"
        f"TILEFOUNDRY_NPU_WORK_ROOT={tmp_path / 'work'}\n",
        encoding="utf-8",
    )
    from tilefoundry_ascend.agent.npu import ag_transport, msprof_probe  # noqa: PLC0415

    monkeypatch.setattr(msprof_probe, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(msprof_probe, "_agent_gates_workspace_name", lambda _r: "ws")
    monkeypatch.setattr(msprof_probe, "resolve_cannbench_repo", lambda **_k: tmp_path)
    monkeypatch.setattr(msprof_probe, "prewarm_anchor_reference_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(
        msprof_probe, "prewarm_surrogate_tilelang_cache", lambda *_a, **_k: tmp_path
    )
    monkeypatch.setattr(msprof_probe, "_verify_fast_bindings", lambda *_a, **_k: None)
    monkeypatch.setattr(msprof_probe, "_verify_fast_coverage", lambda *_a, **_k: None)

    recorded_payloads: list[dict[str, object]] = []

    def _fake_submit(**kwargs: object) -> tuple[dict[str, object], SimpleNamespace]:
        req_path = Path(str(kwargs["request_path"]))
        req_payload = json.loads(req_path.read_text(encoding="utf-8"))
        recorded_payloads.append(req_payload)
        cid = str(req_payload["msprof_targets"][0]["candidate_id"])
        raw_tel = {
            "dur_us": 18.5,
            "aiv_vec_ratio": 0.42,
            "dominant_subkernel_name": "aiv_kernel",
            "msprof_source": "msprof_op",
            "p2_source": "msprof_p2_trace",
            "p2_mte_waitflag_share": 0.35,
            "p2_issue_interval_us": {"MTE2": {"p50_us": 1.2, "p90_us": 2.4}},
            "p2_hotspots": [{"source_location": "kernel.py:19", "instruction_count": 12}],
            "p2_summary_lines": ["kernel.py:19 (12 instrs, 35.0%)"],
        }
        return (
            {"msprof_telemetry": {"softmax": {cid: {"3": raw_tel}}}},
            SimpleNamespace(state="FINISHED"),
        )

    monkeypatch.setattr(msprof_probe, "_submit_fast_gate", _fake_submit)
    anchors = [AnchorCaseSpec("level1/softmax", 3, "vector_mid", 20.0, 2.0)]

    numeric_only = surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
        "softmax",
        cand,
        anchors,
        source_root=src_root,
        env_file=env_file,
    )
    assert recorded_payloads[0]["mode"] == "profile"
    assert recorded_payloads[0]["msprof_targets"][0]["candidate_id"] == "bottleneck_probe"
    assert numeric_only == {
        "vector_mid": {
            "dur_us": 18.5,
            "aiv_vec_ratio": 0.42,
            "p2_mte_waitflag_share": 0.35,
        }
    }

    deep_dive = surrogate_cannbench.collect_p2_deep_dive_via_agent_gates(
        "softmax",
        cand,
        anchors,
        source_root=src_root,
        env_file=env_file,
    )
    assert recorded_payloads[1]["mode"] == "profile"
    assert recorded_payloads[1]["msprof_targets"][0]["candidate_id"] == "bottleneck_p2_deep_dive"
    reg_payload = deep_dive["vector_mid"]
    assert isinstance(reg_payload, dict)
    assert reg_payload["dur_us"] == pytest.approx(18.5)
    assert reg_payload["p2_source"] == "msprof_p2_trace"
    assert reg_payload["p2_mte_waitflag_share"] == pytest.approx(0.35)
    assert reg_payload["p2_issue_interval_us"]["MTE2"]["p50_us"] == pytest.approx(1.2)
    assert reg_payload["p2_hotspots"][0]["source_location"] == "kernel.py:19"
    assert reg_payload["p2_summary_lines"] == ["kernel.py:19 (12 instrs, 35.0%)"]


def test_bayesian_tracker_symptom_ranking_boost_threshold_and_family_tabu() -> None:
    """Verify symptom_ranking tier priority, >0.15 threshold, and family tabu suppression."""
    high_posterior_other = AscendCausalRule(
        rule_id="R_HIGH_VEC",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Unroll vector loop.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="ref:test",
        alpha_prior=20.0,
        beta_prior=1.0,
    )
    low_posterior_dma_a = AscendCausalRule(
        rule_id="R_DMA_A",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Fuse GM roundtrip into UB.",
        expected_postcondition="GM_BYTES <= prev.GM_BYTES",
        source_provenance="ref:test",
        alpha_prior=2.0,
        beta_prior=2.0,
    )
    low_posterior_dma_b = AscendCausalRule(
        rule_id="R_DMA_B",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Keep activations resident in UB.",
        expected_postcondition="GM_BYTES <= prev.GM_BYTES",
        source_provenance="ref:test",
        alpha_prior=4.0,
        beta_prior=2.0,
    )
    tracker = BayesianRuleTracker.from_rules(
        [high_posterior_other, low_posterior_dma_a, low_posterior_dma_b]
    )
    ctx = {"is_legal": True, "PIPE_V": 100}

    default_order = [r.rule_id for r in tracker.rank_applicable_rules(ctx)]
    assert default_order == ["R_HIGH_VEC", "R_DMA_B", "R_DMA_A"]

    boosted_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.80)],
        )
    ]
    assert boosted_order == ["R_DMA_B", "R_DMA_A", "R_HIGH_VEC"]

    selected_top = [
        r.rule_id
        for r in tracker.select_top_rules(
            ctx,
            top_k=2,
            symptom_ranking=[("dma_redundant_gm_roundtrip", 0.50)],
        )
    ]
    assert selected_top == ["R_DMA_B", "R_DMA_A"]

    threshold_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.15)],
        )
    ]
    assert threshold_order == default_order

    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R_DMA_A",
        parent_sha="sha_p1",
        current_round=1,
        reason="no_speedup",
    )
    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R_DMA_B",
        parent_sha="sha_p1",
        current_round=2,
        reason="no_speedup",
    )
    assert (
        tracker.is_family_tabued(
            regime_tag="reg_a",
            symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP,
            parent_sha="sha_p1",
            current_round=3,
        )
        is True
    )
    tabued_order = [
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx,
            regime_tag="reg_a",
            parent_sha="sha_p1",
            current_round=3,
            symptom_ranking=[(SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, 0.90)],
        )
    ]
    assert tabued_order == ["R_HIGH_VEC"]
