"""Ascend 950PR SparseFlashAttention operator package."""

from .kernels import kernel, sparse_flash_attention

__all__ = ["kernel", "sparse_flash_attention"]
