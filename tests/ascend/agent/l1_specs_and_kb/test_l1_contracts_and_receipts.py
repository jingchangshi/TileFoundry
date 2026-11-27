"""Reject and accept tests for frozen L1 stage contracts and banned facade hooks."""

from __future__ import annotations

import ast
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Literal

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import dsm_lower_triangular_lint as dsm_lint  # noqa: E402

from tilefoundry_ascend.agent.contracts import (
    CampaignStateSnapshot,
    CompileTaskSpec,
    RecoveryPort,
    RecoveryReceipt,
    RoundContextPort,
    RoundContextReceipt,
    RoundGatePort,
    StageGateRoundReceipt,
)


class _StubRoundContextPort:
    def __init__(self, receipt: RoundContextReceipt) -> None:
        self._receipt = receipt

    def build_round_bundle(
        self,
        state: object,
        *,
        round_idx: int,
        regime_spec: object = None,
        working_anchors: Sequence[object] | None = None,
        exclude_regime: str | None = None,
        creative_effort: str = "high",
    ) -> RoundContextReceipt:
        _ = (state, round_idx, regime_spec, working_anchors, exclude_regime, creative_effort)
        return self._receipt


class _StubRoundGatePort:
    def __init__(self, receipt: StageGateRoundReceipt) -> None:
        self._receipt = receipt

    async def execute_round_gates(
        self,
        batch: object,
        working_anchors: Sequence[object] = (),
        *,
        snapshot: CampaignStateSnapshot | None = None,
        parent_sha: str = "",
        parent_workload: object = None,
    ) -> StageGateRoundReceipt:
        _ = (batch, working_anchors, snapshot, parent_sha, parent_workload)
        return self._receipt

    async def cancel_pending_speculative(self) -> None:
        return None


class _StubRecoveryPort:
    def __init__(self, receipt: RecoveryReceipt) -> None:
        self._receipt = receipt

    def resolve_pending_promotion(self, pending: object, **kwargs: object) -> RecoveryReceipt:
        _ = (pending, kwargs)
        return self._receipt

    def resolve_pending_l4(
        self, pending: object, *, incumbent_source: str = ""
    ) -> RecoveryReceipt:
        _ = (pending, incumbent_source)
        return self._receipt

    def run_final_l4_tracked(
        self, *, incumbent_source: str, prior_stop_reason: str
    ) -> RecoveryReceipt:
        _ = (incumbent_source, prior_stop_reason)
        return self._receipt


class _ContractStepHarness:
    def __init__(
        self,
        *,
        context_port: RoundContextPort,
        gate_port: RoundGatePort,
        recovery_port: RecoveryPort,
    ) -> None:
        self._context_port = context_port
        self._gate_port = gate_port
        self._recovery_port = recovery_port

    def run_step(
        self, snapshot: CampaignStateSnapshot
    ) -> tuple[object, object, object]:
        ctx = self._context_port.build_round_bundle(snapshot, round_idx=snapshot.round_index)
        gate = asyncio.run(self._gate_port.execute_round_gates("batch", snapshot=snapshot))
        rec = self._recovery_port.resolve_pending_promotion("pending")
        return ctx, gate, rec


@pytest.mark.parametrize(
    ("banned_symbol", "snippet"),
    [
        ("_FACADE_NS", "_FACADE_NS: dict[str, object] = {}\n"),
        ("_bind_facade_globals", "def _bind_facade_globals(ns):\n    return ns\n"),
        ("_hook", "def _hook(name, val):\n    return val\n"),
    ],
)
def test_reject_banned_facade_hooks_and_invalid_receipt_mutations(
    tmp_path: Path, banned_symbol: str, snippet: str
) -> None:
    """Reject banned facade reflection symbols, frozen mutations, and contract typing leaks."""
    pkg_root = tmp_path / "tilefoundry_ascend"
    npu_file = pkg_root / "agent" / "npu" / "ag_transport.py"
    npu_file.parent.mkdir(parents=True, exist_ok=True)
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "agent" / "npu" / "__init__.py").write_text("", encoding="utf-8")
    npu_file.write_text(snippet, encoding="utf-8")

    report = dsm_lint.run_lint(pkg_root, {"reflection"})
    assert report.counts_by_category["reflection"] >= 1
    assert any(banned_symbol in item.message for item in report.violations)

    contracts_path = _REPO_ROOT / "targets/ascend/tilefoundry_ascend/agent/contracts.py"
    contracts_src = contracts_path.read_text(encoding="utf-8")
    assert "# type: ignore" not in contracts_src
    tree = ast.parse(contracts_src, filename=str(contracts_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"Any", "cast", "TYPE_" + "CHECKING"}

    ctx = RoundContextReceipt(1, "decode", "_build", "exploit", "abc123", "tilelang_dsl")
    gate = StageGateRoundReceipt(1, "decode", 2, ("c0",), (), ("c0",))
    rec = RecoveryReceipt("l3_promotion", resolved=True, passed=True)
    spec = CompileTaskSpec("k0", "_build", (), (), ())
    for frozen_obj, attr, val in (
        (ctx, "round_idx", 2),
        (gate, "raw_candidate_count", 9),
        (rec, "passed", False),
        (spec, "unit_kind", "cce_cpp20"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(frozen_obj, attr, val)

    ctx_cls: type[RoundContextReceipt] = RoundContextReceipt
    with pytest.raises(TypeError):
        ctx_cls(1, "decode")


@pytest.mark.parametrize(
    ("unit_kind", "unit_source"),
    [
        ("tilelang_dsl", "def _build_main():\n    return 1\n"),
        ("cce_cpp20", 'extern "C" __global__ __aicore__ void kernel_active() {}\n'),
    ],
)
def test_accept_compile_task_spec_and_frozen_stage_receipts(
    unit_kind: Literal["tilelang_dsl", "cce_cpp20"],
    unit_source: str,
) -> None:
    """Accept CompileTaskSpec unit fields and frozen stage receipts via Protocol ports."""
    default_spec = CompileTaskSpec("k_def", "_build", (), (), ())
    assert default_spec.unit_kind == "tilelang_dsl"
    assert default_spec.unit_source == ""

    spec = CompileTaskSpec(
        task_key="k_explicit",
        builder_name="_build_main",
        static_kwargs=(("bm", 64),),
        pass_configs=(("tl.ascend.opt", 1),),
        compile_flags=("--cce-disable-vf-stack-reserved-ubuf",),
        unit_kind=unit_kind,
        unit_source=unit_source,
    )
    assert spec.unit_kind == unit_kind
    assert spec.unit_source == unit_source
    assert spec.disables_vf_stack_ubuf is True

    ctx_receipt = RoundContextReceipt(
        round_idx=1,
        regime_tag="decode",
        func_name="_build_main",
        parent_lane="exploit",
        parent_sha256="sha0",
        track=unit_kind,
        worker_count=2,
    )
    gate_receipt = StageGateRoundReceipt(
        round_idx=1,
        regime_tag="decode",
        raw_candidate_count=2,
        l0_l1_passed_ids=("c0",),
        l0_l1_failed=(("c1", "stage_b", "overflow"),),
        l2_passed_ids=("c0",),
        l2_run_ids=(("c0", "run-1"),),
    )
    rec_receipt = RecoveryReceipt(
        stage="l3_promotion",
        resolved=True,
        passed=True,
        candidate_sha256="sha1",
        run_id="run-l3-1",
    )
    snapshot = CampaignStateSnapshot(
        run_id="run-0",
        round_index=1,
        incumbent_source="def kernel(): pass",
        incumbent_sha256="sha0",
        dispatch_floor_us=1.5,
        active_regime_tag="decode",
        guardrail_case_ids=("0",),
        stagnation_by_regime={"decode": 0},
    )
    harness = _ContractStepHarness(
        context_port=_StubRoundContextPort(ctx_receipt),
        gate_port=_StubRoundGatePort(gate_receipt),
        recovery_port=_StubRecoveryPort(rec_receipt),
    )
    got_ctx, got_gate, got_rec = harness.run_step(snapshot)
    assert got_ctx == ctx_receipt
    assert got_gate == gate_receipt
    assert got_rec == rec_receipt
