"""Ascend 950PR NPU parity tests for Split-KV MLA and AttentionResidual regimes."""

from __future__ import annotations

import pytest
import torch

from examples.ascend950pr_ops.attention_residual import attention_residual
from examples.ascend950pr_ops.mla import mla
from tests.ascend.benchmarks.attention_residual.golden import (
    attention_residual as ar_golden,
)
from tests.ascend.benchmarks.mla.golden import mla as mla_golden


def _require_npu_env() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is unavailable on this host.")


@pytest.mark.parametrize(
    ("b_dim", "s_dim", "s_kv", "n_q", "n_kv", "dtype", "layout", "is_causal"),
    [
        (1, 1, 2048, 128, 1, torch.float16, "BSND", False),
        (1, 1, 257, 64, 1, torch.bfloat16, "BSND", False),
        (2, 2, 260, 128, 1, torch.float16, "BSND", True),
    ],
)
def test_mla_split_kv_single_vs_multi_split_npu_parity(
    b_dim: int,
    s_dim: int,
    s_kv: int,
    n_q: int,
    n_kv: int,
    dtype: torch.dtype,
    layout: str,
    is_causal: bool,
) -> None:
    """Verify mla num_kv_splits=1 and num_kv_splits=4 match PyTorch golden on NPU."""
    _require_npu_env()
    torch.manual_seed(42)
    d_nope, d_rope = 512, 64
    q_nope = torch.randn(b_dim, s_dim, n_q, d_nope, dtype=dtype)
    q_rope = torch.randn(b_dim, s_dim, n_q, d_rope, dtype=dtype)
    k_nope = torch.randn(b_dim, s_kv, n_kv, d_nope, dtype=dtype)
    k_rope = torch.randn(b_dim, s_kv, n_kv, d_rope, dtype=dtype)
    v = k_nope.clone()

    ref = mla_golden(
        q_nope.float(),
        q_rope.float(),
        k_nope.float(),
        k_rope.float(),
        v.float(),
        numKVHeads=n_kv,
        scaleValue=-1.0,
        inputLayout=layout,
        is_causal=is_causal,
    )
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    for splits in (1, 4):
        actual = mla(
            q_nope.npu(),
            q_rope.npu(),
            k_nope.npu(),
            k_rope.npu(),
            v.npu(),
            numKVHeads=n_kv,
            scaleValue=-1.0,
            inputLayout=layout,
            is_causal=is_causal,
            num_kv_splits_override=splits,
        )
        torch.testing.assert_close(actual.cpu().float(), ref, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    ("n_depth", "b_dim", "t_dim", "d_dim", "dtype"),
    [
        (2, 1, 1, 4096, torch.bfloat16),
        (4, 1, 8, 1000, torch.bfloat16),
        (4, 2, 32, 256, torch.bfloat16),
    ],
)
def test_attention_residual_regimes_npu_parity(
    n_depth: int,
    b_dim: int,
    t_dim: int,
    d_dim: int,
    dtype: torch.dtype,
) -> None:
    """Verify AttentionResidual small_decode, unaligned_d, and aligned_prefill on NPU."""
    _require_npu_env()
    torch.manual_seed(7)
    v_stack = torch.randn(n_depth, b_dim, t_dim, d_dim, dtype=dtype)
    x = torch.randn(b_dim, t_dim, d_dim, dtype=dtype)
    w = torch.randn(d_dim, dtype=dtype)
    gamma = torch.rand(d_dim, dtype=dtype) + 0.5
    ref = ar_golden(v_stack, x, w, gamma, alpha=0.5, epsilon=1e-6).float()
    actual = attention_residual(
        v_stack.npu(),
        x.npu(),
        w.npu(),
        gamma.npu(),
        alpha=0.5,
        epsilon=1e-6,
    )
    torch.testing.assert_close(actual.cpu().float(), ref, rtol=2e-2, atol=2e-2)
