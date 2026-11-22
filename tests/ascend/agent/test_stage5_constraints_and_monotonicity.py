"""Stage-5 prompt redline, rule-blind creative, monotonicity, and passthrough tests."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from textwrap import dedent

from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    prepare_operator_seed_or_takeover,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import run_l0_l1_round_filter
from tilefoundry_ascend.agent.kb.schema import load_seed_rules
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_creative_request,
    build_exploit_request,
    build_syntax_repair_request,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask

_GOLDEN_EXP = dedent(
    """
    import torch

    def exp_op(x):
        return torch.exp(x)
    """
).strip()

_PROTO_EXP: dict[str, object] = {
    "name": "Exp",
    "inputs": [{"name": "x", "dtype": ["float16"], "shape": ["M", "N"]}],
    "outputs": [{"name": "y", "dtype": ["float16"], "shape": ["M", "N"]}],
    "schema": "exp(Tensor x) -> Tensor y",
}

_CASES_EXP: list[dict[str, object]] = [
    {"case_id": 1, "dtype": ["float16"], "input_shape": [[64, 128]]},
]

_PARENT_DSL = dedent(
    """
    import tilelang.ascend.language as T

    def make_kernel():
        @T.prim_func
        def main(X: T.Tensor((64, 128), "float16"), Y: T.Tensor((64, 128), "float16")):
            with T.Kernel(1) as bx:
                x_ub = T.alloc_shared((32, 128), "float16")
                for i in T.Pipelined(2, num_stages=2):
                    T.copy(X[i * 32 : (i + 1) * 32, 0:128], x_ub)
                    with T.SimdVF():
                        T.fill(x_ub, 0.0)
                    T.copy(x_ub, Y[i * 32 : (i + 1) * 32, 0:128])
        return main
    """
).strip()

_REPAIR_DROP_PIPELINED = dedent(
    """
    import tilelang.ascend.language as T

    def make_kernel():
        @T.prim_func
        def main(X: T.Tensor((64, 128), "float16"), Y: T.Tensor((64, 128), "float16")):
            with T.Kernel(1) as bx:
                x_ub = T.alloc_shared((32, 128), "float16")
                for i in range(2):
                    T.copy(X[i * 32 : (i + 1) * 32, 0:128], x_ub)
                    with T.SimdVF():
                        T.fill(x_ub, 0.0)
                    T.copy(x_ub, Y[i * 32 : (i + 1) * 32, 0:128])
        return main
    """
).strip()

_REPAIR_SCALAR_LOOP = dedent(
    """
    import tilelang.ascend.language as T

    def make_kernel():
        @T.prim_func
        def main(X: T.Tensor((64, 128), "float16"), Y: T.Tensor((64, 128), "float16")):
            with T.Kernel(1) as bx:
                x_ub = T.alloc_shared((32, 128), "float16")
                for i in T.Pipelined(2, num_stages=2):
                    T.copy(X[i * 32 : (i + 1) * 32, 0:128], x_ub)
                    for j in range(128):
                        x_ub[0, j] = 0.0
                    T.copy(x_ub, Y[i * 32 : (i + 1) * 32, 0:128])
        return main
    """
).strip()

_REPAIR_VALID_HALVED = dedent(
    """
    import tilelang.ascend.language as T

    def make_kernel():
        @T.prim_func
        def main(X: T.Tensor((64, 128), "float16"), Y: T.Tensor((64, 128), "float16")):
            with T.Kernel(1) as bx:
                x_ub = T.alloc_shared((16, 128), "float16")
                for i in T.Pipelined(4, num_stages=2):
                    T.copy(X[i * 16 : (i + 1) * 16, 0:128], x_ub)
                    with T.SimdVF():
                        T.fill(x_ub, 0.0)
                    T.copy(x_ub, Y[i * 16 : (i + 1) * 16, 0:128])
        return main
    """
).strip()


class _ScriptedBackend:
    """Async LLM backend replaying scripted replies and recording requests."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> str:
        self.requests.append(req)
        idx = min(len(self.requests) - 1, len(self._replies) - 1)
        return self._replies[idx]


def _sample_parent() -> BeamCandidate:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 32},
        waterlines={},
    )
    return BeamCandidate(
        candidate_id="parent_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_PARENT_DSL,
        track="unconstrained",
        ledger=ledger,
        residual=compute_static_dynamic_residual(ledger, {"dur_us": 20.0}),
        latency_us=20.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(_PARENT_DSL),
    )


def _assert_g1_to_g4_present(text: str) -> None:
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in text
    assert "LAYER 1.5" not in text.upper()
    assert "Schedule Integrity Redlines" in text
    for marker in (
        "G1 (No Scalar Compute Emulation)",
        "G2 (No Dense-Loop Narrow DCache-Bypass Stores)",
        "G3 (No Cache-Key Pollution or Static Unrolled 2D Dynamic Copy)",
        "G4 (No Overlapping `coupled_1c2v` Dual-Subcore Writes)",
    ):
        assert marker in text


def test_all_five_roles_include_layer1_g1_to_g4_redlines() -> None:
    """Verify Stage-0 (1 & 2..N), exploit, creative, repair, and track-B carry G1..G4."""
    good_seed = "def exp_op(x):\n    return x\nkernel = exp_op\n"
    seed_backend = _ScriptedBackend([good_seed, good_seed])
    attempt_count = 0

    def _static_factory(regime_specs: tuple[object, ...], *, semantics: object | None = None):
        _ = (regime_specs, semantics)

        def _validate(_src: str, _track: str) -> tuple[bool, str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                return False, "missing @T.prim_func entry"
            return True, ""

        return _validate

    def _oracle(src: str) -> OracleAccuracyResult:
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
        return OracleAccuracyResult(
            passed=True,
            candidate_sha256=digest,
            l3_run_id="run-1",
            l3_receipt_path="/tmp/receipt.json",
            l3_state="PASSED",
        )

    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="level1/exp",
            entrypoint="exp_op",
            pytorch_golden_source=_GOLDEN_EXP,
            proto=_PROTO_EXP,
            cases=_CASES_EXP,
            legacy_regime_specs=None,
            llm_backend=seed_backend,
            l0_l1_validator_factory=_static_factory,
            oracle_accuracy_validator=_oracle,
            max_seed_attempts=2,
        )
    )
    assert outcome.attempts_used == 2
    assert len(seed_backend.requests) == 2
    stage0_first = seed_backend.requests[0]
    stage0_retry = seed_backend.requests[1]
    assert stage0_first.role == "explore_creative"
    assert stage0_retry.role == "syntax_repair"
    for req in (stage0_first, stage0_retry):
        _assert_g1_to_g4_present(req.user_prompt)
        assert "### LAYER 2: TARGET DIALECT API SURFACE" in req.user_prompt
        assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in req.user_prompt
        assert "### PER-REGIME HIR BLUEPRINT SUMMARIES" in req.user_prompt
        assert "torch.exp(x)" in req.user_prompt
    assert "### G5 PARENT MONOTONICITY LOCK" in stage0_retry.user_prompt
    assert "never degrade into a bare blueprint-less syntax patch" in stage0_retry.user_prompt

    parent = _sample_parent()
    rule = load_seed_rules()[0]
    exploit_req = build_exploit_request(parent, rule)
    creative_req = build_creative_request(parent)
    repair_req = build_syntax_repair_request(
        _PARENT_DSL,
        "ub overflow",
        freedom_mask=ConstraintRelaxationMask.bottom(),
    )
    trackb_req = build_creative_request(parent, escalation_tier="ascendc_template")

    assert exploit_req.role == "exploit_rule"
    assert creative_req.role == "explore_creative"
    assert repair_req.role == "syntax_repair"
    assert trackb_req.role == "escalate_trackb"

    for req in (exploit_req, creative_req, repair_req, trackb_req):
        _assert_g1_to_g4_present(req.user_prompt)
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in repair_req.user_prompt
    assert "### G5 PARENT MONOTONICITY LOCK" in repair_req.user_prompt


def test_explore_creative_is_100_percent_rule_blind() -> None:
    """Ensure explore_creative contains zero seed_rules.yaml rule IDs or templates."""
    parent = _sample_parent()
    creative_req = build_creative_request(parent)
    full_prompt = f"{creative_req.system_prompt}\n{creative_req.user_prompt}"
    for rule in load_seed_rules():
        assert rule.rule_id not in full_prompt
        template_head = rule.transform_template.strip().splitlines()[0].strip()
        assert template_head not in full_prompt
        if rule.cpp20_transform_template.strip():
            cpp_head = rule.cpp20_transform_template.strip().splitlines()[0].strip()
            assert cpp_head not in full_prompt


def test_run_l0_l1_round_filter_rejects_repair_monotonicity_violations() -> None:
    """Reject repairs that strip T.Pipelined or degrade T.SimdVF to scalar loops."""
    for bad_reply in (_REPAIR_DROP_PIPELINED, _REPAIR_SCALAR_LOOP):
        backend = _ScriptedBackend([bad_reply])
        checks = asyncio.run(
            run_l0_l1_round_filter(
                [("w_bad", "def broken(:\n pass\n")],
                patch_fn=lambda src: src,
                l0_checker=lambda src: StageAResult(
                    passed="broken(" not in src,
                    prim_func=None,
                    ledger=None,
                    error_message="syntax_err" if "broken(" in src else "",
                ),
                l1_compiler=lambda _src: StageBResult(passed=True),
                repair_backend=backend,
                max_syntax_repairs=1,
                parent_source=_PARENT_DSL,
            )
        )
        assert len(checks) == 1
        assert checks[0].passed is False
        assert "repair_monotonicity_violation" in checks[0].error_message

    valid_backend = _ScriptedBackend([_REPAIR_VALID_HALVED])
    ok_checks = asyncio.run(
        run_l0_l1_round_filter(
            [("w_ok", "def broken(:\n pass\n")],
            patch_fn=lambda src: src,
            l0_checker=lambda src: StageAResult(
                passed="broken(" not in src,
                prim_func=None,
                ledger=None,
                error_message="syntax_err" if "broken(" in src else "",
            ),
            l1_compiler=lambda _src: StageBResult(passed=True),
            repair_backend=valid_backend,
            max_syntax_repairs=1,
            parent_source=_PARENT_DSL,
        )
    )
    assert len(ok_checks) == 1
    assert ok_checks[0].passed is True
    assert ok_checks[0].repaired is True


def test_run_l0_l1_round_filter_bounded_creative_passthrough() -> None:
    """Promote 1 creative candidate into L1 iff analytical_us <= 1.5 * parent_analytical_us."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 20},
        waterlines={},
    )
    candidates = [
        ("rule_0", "src_r0"),
        ("rule_1", "src_r1"),
        ("creative_0", "src_c0"),
    ]

    l1_seen_within: list[str] = []
    checks_within = asyncio.run(
        run_l0_l1_round_filter(
            candidates,
            patch_fn=lambda s: s,
            l0_checker=lambda _s: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda s: (l1_seen_within.append(s), StageBResult(passed=True))[1],
            stage_a_cost_ranker=lambda _items: [
                ("rule_0", 10.0),
                ("rule_1", 12.0),
                ("creative_0", 14.0),
            ],
            cost_prescreen_top_k=2,
            creative_candidate_ids={"creative_0"},
            parent_analytical_us=10.0,
        )
    )
    by_id_within = {chk.candidate_id: chk for chk in checks_within}
    assert by_id_within["rule_0"].passed is True
    assert by_id_within["rule_1"].passed is True
    assert by_id_within["creative_0"].passed is True
    assert set(l1_seen_within) == {"src_r0", "src_r1", "src_c0"}

    l1_seen_over: list[str] = []
    checks_over = asyncio.run(
        run_l0_l1_round_filter(
            candidates,
            patch_fn=lambda s: s,
            l0_checker=lambda _s: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda s: (l1_seen_over.append(s), StageBResult(passed=True))[1],
            stage_a_cost_ranker=lambda _items: [
                ("rule_0", 10.0),
                ("rule_1", 12.0),
                ("creative_0", 16.0),
            ],
            cost_prescreen_top_k=2,
            creative_candidate_ids={"creative_0"},
            parent_analytical_us=10.0,
        )
    )
    by_id_over = {chk.candidate_id: chk for chk in checks_over}
    assert by_id_over["rule_0"].passed is True
    assert by_id_over["rule_1"].passed is True
    assert by_id_over["creative_0"].passed is False
    assert by_id_over["creative_0"].failed_tier == "cost_prescreen"
    assert set(l1_seen_over) == {"src_r0", "src_r1"}
