"""Unit tests for operator-agnostic associative outer-recurrence detection and 7-facet unlock."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import yaml

from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask
from tilefoundry_ascend.ascend.templates import registry as template_registry
from tilefoundry_ascend.ascend.templates.registry import (
    unlock_cross_topology_paradigm_shift,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEED_RULES_PATH = (
    _REPO_ROOT
    / "targets"
    / "ascend"
    / "tilefoundry_ascend"
    / "agent"
    / "kb"
    / "seed_rules.yaml"
)


def test_linear_outer_product_recurrence_detected_and_keeps_initial_base_axes() -> None:
    """Detect linear outer-product state recurrence while keeping initial base_axes intact."""
    analyzer = PyTorchSemanticAnalyzer()
    kda_like_source = dedent(
        """
        import torch

        def gated_linear_recurrence(q, k, v, g, beta, initial_state):
            query = q.to(torch.float32)
            key = k.to(torch.float32)
            value = v.to(torch.float32)
            decay = g.to(torch.float32).exp()
            update_strength = beta.to(torch.float32)
            memory = initial_state.to(torch.float32).clone()
            output = torch.empty(value.shape, dtype=torch.float32, device=q.device)
            for token in range(q.shape[1]):
                faded_memory = memory * decay[:, token].unsqueeze(-1)
                token_key = key[:, token]
                predicted_value = torch.sum(faded_memory * token_key.unsqueeze(-1), dim=-2)
                correction = (value[:, token] - predicted_value) * update_strength[
                    :, token
                ].unsqueeze(-1)
                memory = faded_memory + token_key.unsqueeze(-1) * correction.unsqueeze(-2)
                output[:, token] = torch.sum(memory * query[:, token].unsqueeze(-1), dim=-2)
            return output.to(q.dtype), memory
        """
    )
    sem = analyzer.analyze_source(kda_like_source, entry_name="gated_linear_recurrence")
    assert sem.has_recurrent_loop is True
    assert sem.has_associative_outer_recurrence is True
    assert sem.base_axes.compute_engine == "aiv_only"
    assert sem.base_axes.memory_ring == "persistent_onchip_state"

    einsum_recurrence_source = dedent(
        """
        import torch

        def linear_state_scan(k, v, g, state):
            for t in range(k.shape[1]):
                state = g[:, t].unsqueeze(-1) * state + torch.einsum(
                    "bhk,bhv->bhkv", k[:, t], v[:, t]
                )
            return state
        """
    )
    sem_einsum = analyzer.analyze_source(
        einsum_recurrence_source, entry_name="linear_state_scan"
    )
    assert sem_einsum.has_recurrent_loop is True
    assert sem_einsum.has_associative_outer_recurrence is True

    minimal_outer_source = dedent(
        """
        import torch

        def minimal_outer_recurrence(g, k, v, S):
            for t in range(k.shape[1]):
                S = g * S + k.unsqueeze(-1) * v.unsqueeze(-2)
            return S
        """
    )
    sem_min = analyzer.analyze_source(
        minimal_outer_source, entry_name="minimal_outer_recurrence"
    )
    assert sem_min.has_recurrent_loop is True
    assert sem_min.has_associative_outer_recurrence is True
    assert sem_min.base_axes.compute_engine == "aiv_only"
    assert sem_min.base_axes.memory_ring == "persistent_onchip_state"

    aug_outer_source = dedent(
        """
        import torch

        def aug_outer_recurrence(k, v, S):
            for t in range(k.shape[1]):
                S += torch.outer(k[t], v[t])
            return S
        """
    )
    sem_aug = analyzer.analyze_source(
        aug_outer_source, entry_name="aug_outer_recurrence"
    )
    assert sem_aug.has_recurrent_loop is True
    assert sem_aug.has_associative_outer_recurrence is True


def test_nonlinear_rnn_and_non_loop_ops_are_rejected() -> None:
    """Reject nonlinear tanh/sigmoid RNNs and non-loop elementwise or reduction ops."""
    analyzer = PyTorchSemanticAnalyzer()
    nonlinear_rnn_source = dedent(
        """
        import torch

        def elman_rnn(x, weight, h0):
            h = h0
            for t in range(x.shape[1]):
                h = torch.tanh(weight @ h)
            return h
        """
    )
    sem_rnn = analyzer.analyze_source(nonlinear_rnn_source, entry_name="elman_rnn")
    assert sem_rnn.has_recurrent_loop is True
    assert sem_rnn.has_associative_outer_recurrence is False

    nonlinear_affine_rnn_source = dedent(
        """
        import torch

        def affine_tanh_rnn(x, weight, h0):
            h = h0
            for t in range(x.shape[1]):
                h = torch.tanh(weight @ h + x[:, t]).to(torch.float32)
            return h
        """
    )
    sem_affine_rnn = analyzer.analyze_source(
        nonlinear_affine_rnn_source, entry_name="affine_tanh_rnn"
    )
    assert sem_affine_rnn.has_recurrent_loop is True
    assert sem_affine_rnn.has_associative_outer_recurrence is False

    elementwise_source = dedent(
        """
        import torch

        def add_scale(x, y):
            return x + 2.0 * y
        """
    )
    sem_elem = analyzer.analyze_source(elementwise_source, entry_name="add_scale")
    assert sem_elem.has_recurrent_loop is False
    assert sem_elem.has_associative_outer_recurrence is False

    reduction_source = dedent(
        """
        import torch

        def row_sum(x):
            return torch.sum(x, dim=-1)
        """
    )
    sem_red = analyzer.analyze_source(reduction_source, entry_name="row_sum")
    assert sem_red.has_recurrent_loop is False
    assert sem_red.has_associative_outer_recurrence is False


def test_unlock_cross_topology_paradigm_shift_and_seed_rules_invariant() -> None:
    """Unlock all 7 facets to top and preserve the orthogonal seed rules."""
    assert "unlock_cross_topology_paradigm_shift" in template_registry.__all__
    bottom = ConstraintRelaxationMask.bottom()
    assert bottom.is_unrestricted_top is False
    unlocked = unlock_cross_topology_paradigm_shift(bottom)
    assert unlocked.is_unrestricted_top is True
    assert unlocked == ConstraintRelaxationMask.top()

    seed_payload = yaml.safe_load(_SEED_RULES_PATH.read_text(encoding="utf-8"))
    assert isinstance(seed_payload, dict)
    assert len(seed_payload["rules"]) >= 41
