"""Ascend 950PR NPU numerical acceptance tests for adapted operators."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

BENCHMARKS_DIR = Path(__file__).resolve().parent
AR_DIR = BENCHMARKS_DIR / "attention_residual"
MLA_DIR = BENCHMARKS_DIR / "mla"
SUPP_MLA_PATH = BENCHMARKS_DIR / "cases" / "supplemental_mla.json"

for _path in (BENCHMARKS_DIR, AR_DIR, MLA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cannbench_contract
import test_attention_residual
import test_mla

from examples.ascend950pr_ops.attention_residual import attention_residual
from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels
from examples.ascend950pr_ops.mla import kernels as mla_kernels
from examples.ascend950pr_ops.mla import mla


def _require_npu_env() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is unavailable on this host.")
    if importlib.util.find_spec("npu_compare") is None:
        pytest.skip("npu_compare module is unavailable on PYTHONPATH.")


def _call_single_kernel(kernel_module, fn, **kwargs):
    orig = kernel_module._get_kernel
    calls = 0

    def wrapped(*args, **inner_kwargs):
        nonlocal calls
        calls += 1
        return orig(*args, **inner_kwargs)

    kernel_module._get_kernel = wrapped
    try:
        result = fn(**kwargs)
    finally:
        kernel_module._get_kernel = orig
    assert calls == 1, f"expected 1 kernel invocation, got {calls}"
    return result


def test_attention_residual_cannbench_cases() -> None:
    """Verify all 20 CANNBench AttentionResidual cases on NPU."""
    _require_npu_env()
    test_attention_residual.main(attention_residual)


def test_attention_residual_baseline_vs_tuned_parity() -> None:
    """Verify AttentionResidual baseline schedule numerical parity on NPU."""
    _require_npu_env()
    harness = cannbench_contract.Harness("level4/attention_residual", AR_DIR)
    target_cases = {1, 16, 20}
    checked = 0
    for case in harness.cases:
        if case.case_num not in target_cases:
            continue
        checked += 1
        tensors = harness.prepare(case)
        oracle = harness.reference_output(case, tensors)
        npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
        params = harness.params(case, npu_tensors)
        actual = _call_single_kernel(
            ar_kernels,
            attention_residual,
            **params,
            schedule_mode="baseline",
        )
        passed, details = harness.compare(
            case,
            actual,
            oracle,
            tensors,
            lambda case=case, tensors=tensors: harness.reference_output(
                case, tensors, native=True
            ),
        )
        assert passed, f"{case.get_case_id_str()}: {details}"
    assert checked == len(target_cases)


def test_mla_cannbench_cases() -> None:
    """Verify all 20 CANNBench MLA cases on NPU."""
    _require_npu_env()
    test_mla.main(mla)


def test_mla_supplemental_cases() -> None:
    """Verify supplemental GQA, tail, layout, and scale MLA cases on NPU."""
    _require_npu_env()
    payload = json.loads(SUPP_MLA_PATH.read_text(encoding="utf-8"))
    harness = cannbench_contract.Harness("level4/mla", MLA_DIR)
    for raw in payload["cases"]:
        case = cannbench_contract.Case(
            case_num=raw["case_id"],
            label=raw["key"],
            input_shapes=raw["input_shapes"],
            dtypes=raw["dtypes"],
            attrs=raw["attrs"],
            value_ranges=raw["value_range"],
        )
        tensors = cannbench_contract.DataGenerator().generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges,
            seed=int(raw["seed"]),
        )
        params = {
            item.name: tensors[i] if i < len(tensors) else None
            for i, item in enumerate(harness.info.inputs)
        }
        params.update({k: v for k, v in case.attrs.items() if k not in params})
        tensors = harness.get_input(**params)
        oracle = harness.reference_output(case, tensors)
        native_fn = lambda case=case, tensors=tensors: harness.reference_output(
            case, tensors, native=True
        )
        for mode in ("tuned", "baseline"):
            npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
            call_params = harness.params(case, npu_tensors)
            actual = _call_single_kernel(
                mla_kernels,
                mla,
                **call_params,
                schedule_mode=mode,
            )
            passed, details = harness.compare(case, actual, oracle, tensors, native_fn)
            assert passed, f"{case.get_case_id_str()} ({mode}): {details}"


def test_dynamic_shape_kernel_cache_reuse() -> None:
    """Verify T.dynamic parameterization reuses compiled kernels across dynamic shapes."""
    _require_npu_env()
    from tests.ascend.benchmarks.attention_residual.golden import (
        attention_residual as ar_golden,
    )
    from tests.ascend.benchmarks.mla.golden import mla as mla_golden

    torch.manual_seed(0)
    ar_kernels._KERNEL_CACHE.clear()
    n_depth, d_dim = 4, 256
    w = torch.randn(d_dim, dtype=torch.bfloat16)
    gamma = torch.rand(d_dim, dtype=torch.bfloat16) + 0.5
    for b_dim, t_dim in ((1, 16), (2, 32)):
        v_stack = torch.randn(n_depth, b_dim, t_dim, d_dim, dtype=torch.bfloat16)
        x = torch.randn(b_dim, t_dim, d_dim, dtype=torch.bfloat16)
        ref = ar_golden(v_stack, x, w, gamma, alpha=0.5, epsilon=1e-6)
        actual = _call_single_kernel(
            ar_kernels,
            attention_residual,
            v_stack=v_stack.npu(),
            x=x.npu(),
            w=w.npu(),
            gamma=gamma.npu(),
            alpha=0.5,
            epsilon=1e-6,
        )
        torch.testing.assert_close(actual.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
    assert len(ar_kernels._KERNEL_CACHE) == 1

    mla_kernels._KERNEL_CACHE.clear()
    n_q, n_kv, d_nope, d_rope = 16, 1, 512, 64
    for b_dim, s_dim, s_kv in ((2, 64, 128), (4, 128, 256)):
        qn = torch.randn(b_dim, s_dim, n_q, d_nope, dtype=torch.bfloat16)
        qr = torch.randn(b_dim, s_dim, n_q, d_rope, dtype=torch.bfloat16)
        kn = torch.randn(b_dim, s_kv, n_kv, d_nope, dtype=torch.bfloat16)
        kr = torch.randn(b_dim, s_kv, n_kv, d_rope, dtype=torch.bfloat16)
        v = kn.clone()
        ref = mla_golden(
            qn,
            qr,
            kn,
            kr,
            v,
            numKVHeads=n_kv,
            scaleValue=-1.0,
            inputLayout="BSND",
            is_causal=True,
        )
        actual = _call_single_kernel(
            mla_kernels,
            mla,
            q_nope=qn.npu(),
            q_rope=qr.npu(),
            k_nope=kn.npu(),
            k_rope=kr.npu(),
            v=v.npu(),
            numKVHeads=n_kv,
            scaleValue=-1.0,
            inputLayout="BSND",
            is_causal=True,
        )
        torch.testing.assert_close(actual.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
    assert len(mla_kernels._KERNEL_CACHE) == 1
