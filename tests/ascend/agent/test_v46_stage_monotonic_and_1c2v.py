"""Tests for Stage-Monotonic Track Pivot, FileNotFoundError propagation, and 1C2V checks."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.specs.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    prepare_operator_seed_or_takeover,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import run_stage_a_static
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest

_RMS_GOLDEN = dedent(
    """
    import torch

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    """
).strip()

_PROTO: dict[str, object] = {
    "api_name": "rms_norm",
    "inputs": [
        {"name": "x", "shape": ["M", "D"], "dtype": "float16"},
        {"name": "weight", "shape": ["D"], "dtype": "float16"},
    ],
    "outputs": [{"name": "out", "shape": ["M", "D"], "dtype": "float16"}],
    "attrs": [{"name": "eps", "type": "float", "default": 1e-6}],
}
_CASES = [
    {"case_id": 1, "input_shape": [[64, 128], [128]], "dtype": ["float16", "float16"]},
]


class _ScriptedBackend:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self._replies) - 1)
        return self._replies[idx]


def test_stage_monotonic_track_pivot_blocks_oracle_pivot_and_anchors_parent() -> None:
    """Never pivot on oracle mismatch and anchor repair back to last L0/L1-passing source."""
    l0_ok_1 = "def rms_norm(x, weight, eps=1e-6):\n    return x * 1.0\n"
    l0_bad_2 = "def rms_norm_broken(x, weight, eps=1e-6):\n    return x\n"
    l0_ok_3 = "def rms_norm(x, weight, eps=1e-6):\n    return x * 2.0\n"
    backend = _ScriptedBackend([l0_ok_1, l0_bad_2, l0_ok_3])
    seen_backends: list[str] = []

    def _validator_factory(
        specs: tuple[RegimeTargetSpec, ...],
        *,
        semantics: object | None = None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        _ = semantics
        rt = specs[0].runtime_backend

        def _check(src: str, tag: str) -> tuple[bool, str]:
            _ = tag
            seen_backends.append(rt)
            if "rms_norm_broken" in src:
                return False, "syntax_or_l0_error"
            return True, ""

        return _check

    def _oracle(src: str) -> OracleAccuracyResult:
        if "2.0" in src:
            digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
            return OracleAccuracyResult(
                passed=True,
                candidate_sha256=digest,
                l3_run_id="r_ok",
                l3_receipt_path="/tmp/r_ok.json",
                l3_state="passed",
                elapsed_sec=0.1,
            )
        return OracleAccuracyResult(
            passed=False,
            error_message="max_abs_err=0.5",
            l3_state="failed",
            failure_kind="candidate",
        )

    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="level2/rms_norm",
            entrypoint="rms_norm",
            pytorch_golden_source=_RMS_GOLDEN,
            proto=_PROTO,
            cases=_CASES,
            legacy_regime_specs=None,
            llm_backend=backend,
            l0_l1_validator_factory=_validator_factory,
            oracle_accuracy_validator=_oracle,
            force_cold_start=True,
            max_seed_attempts=4,
        )
    )
    assert outcome.oracle_accuracy_passed is True
    assert outcome.initial_track == "tilelang_dsl"
    assert seen_backends == ["tilelang_dsl", "tilelang_dsl", "tilelang_dsl"]
    assert "return x * 1.0" in backend.requests[2].user_prompt
    assert "rms_norm_broken" not in backend.requests[2].user_prompt


def test_seed_synthesizer_propagates_file_not_found_from_validators() -> None:
    """Propagate FileNotFoundError from l0_l1_validator and oracle_accuracy_validator."""
    src_a = "def rms_norm(x, weight, eps=1e-6):\n    return x\n"

    def _fnf_l0_factory(
        specs: tuple[RegimeTargetSpec, ...],
        *,
        semantics: object | None = None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        _ = specs, semantics

        def _check(_src: str, _tag: str) -> tuple[bool, str]:
            raise FileNotFoundError("missing bisheng")

        return _check

    def _ok_l0_factory(
        specs: tuple[RegimeTargetSpec, ...],
        *,
        semantics: object | None = None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        _ = specs, semantics
        return lambda _src, _tag: (True, "")

    def _fnf_oracle(_src: str) -> OracleAccuracyResult:
        raise FileNotFoundError("missing agent-gates")

    with pytest.raises(FileNotFoundError, match="missing bisheng"):
        asyncio.run(
            prepare_operator_seed_or_takeover(
                operator_name="level2/rms_norm",
                entrypoint="rms_norm",
                pytorch_golden_source=_RMS_GOLDEN,
                proto=_PROTO,
                cases=_CASES,
                legacy_regime_specs=None,
                llm_backend=_ScriptedBackend([src_a]),
                l0_l1_validator_factory=_fnf_l0_factory,
                oracle_accuracy_validator=_fnf_oracle,
                force_cold_start=True,
                max_seed_attempts=4,
            )
        )

    with pytest.raises(FileNotFoundError, match="missing agent-gates"):
        asyncio.run(
            prepare_operator_seed_or_takeover(
                operator_name="level2/rms_norm",
                entrypoint="rms_norm",
                pytorch_golden_source=_RMS_GOLDEN,
                proto=_PROTO,
                cases=_CASES,
                legacy_regime_specs=None,
                llm_backend=_ScriptedBackend([src_a]),
                l0_l1_validator_factory=_ok_l0_factory,
                oracle_accuracy_validator=_fnf_oracle,
                force_cold_start=True,
                max_seed_attempts=4,
            )
        )


def test_tracka_mixedkernel_sids2_rejects_uncovered_mte3_write_and_allows_sid_partition() -> None:
    """Reject unpartitioned UB->GM T.copy in T.MixedKernel(sids=2) while allowing sid slices."""
    bad_dsl = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def main(X: T.Tensor((128, 64), "float16"), Out: T.Tensor((128, 64), "float16")):
            with T.MixedKernel(1, sids=2) as (bx, sid):
                out_ub = T.alloc_shared((64, 64), "float16")
                T.copy(X[sid * 64 : (sid + 1) * 64, :64], out_ub)
                T.copy(out_ub, Out[bx * 64 : (bx + 1) * 64, :64])
        """
    ).strip()
    bad_res = run_stage_a_static(bad_dsl)
    assert bad_res.passed is False
    assert "uncovered_1c2v_subcore_mte3_write" in bad_res.error_message

    good_dsl = bad_dsl.replace(
        "Out[bx * 64 : (bx + 1) * 64, :64]",
        "Out[sid * 64 : (sid + 1) * 64, :64]",
    )
    good_res = run_stage_a_static(good_dsl)
    assert "uncovered_1c2v_subcore_mte3_write" not in good_res.error_message
