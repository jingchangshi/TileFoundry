"""Unit tests for Ascend 950PR 24-bit coupled core masks, SSBUF, and 3510 facts."""

import pytest

from tilefoundry_ascend import AscendTarget, NodeResourceMask
from tilefoundry_ascend.ascend.target import AscendSpaceFacts


def test_24bit_node_resource_mask_bit_layout_and_conflicts() -> None:
    """Verify bit ranges for AIV0 (0..6), AIC (8..14), AIV1 (16..22), and dual-AIV."""
    m_aiv0 = NodeResourceMask.for_aiv0("MTE2", "VECTOR")
    assert m_aiv0.aiv0_bits == (1 << 1) | (1 << 4)
    assert m_aiv0.aic_bits == 0
    assert m_aiv0.aiv1_bits == 0
    assert int(m_aiv0) == (1 << 1) | (1 << 4)

    m_aic = NodeResourceMask.for_aic("MTE1", "CUBE", "FIXPIPE")
    assert m_aic.aiv0_bits == 0
    assert m_aic.aic_bits == (1 << 0) | (1 << 3) | (1 << 5)
    assert m_aic.aiv1_bits == 0
    assert not m_aiv0.conflicts_with(m_aic)

    m_aiv1 = NodeResourceMask.for_aiv1("MTE3", "SCALAR")
    assert m_aiv1.aiv0_bits == 0
    assert m_aiv1.aic_bits == 0
    assert m_aiv1.aiv1_bits == (1 << 2) | (1 << 6)
    assert not m_aiv0.conflicts_with(m_aiv1)

    m_dual = NodeResourceMask.for_dual_aiv("MTE2", "VECTOR")
    expected_dual = (1 << 1) | (1 << 4) | (1 << 17) | (1 << 20)
    assert m_dual.bits == expected_dual
    assert m_dual.conflicts_with(m_aiv0)
    assert m_dual.conflicts_with(NodeResourceMask.for_aiv1("VECTOR"))
    assert not m_dual.conflicts_with(NodeResourceMask.for_aic("MTE2"))


def test_24bit_node_resource_mask_validation_and_solver_projection() -> None:
    """Verify invalid core-pipe bindings fail and solver resource tuples preserve ownership."""
    with pytest.raises(ValueError, match="does not support pipe"):
        NodeResourceMask.for_aiv0("CUBE")
    with pytest.raises(ValueError, match="does not support pipe"):
        NodeResourceMask.for_aiv1("FIXPIPE")
    with pytest.raises(ValueError, match="does not support pipe"):
        NodeResourceMask.for_aic("VECTOR")
    with pytest.raises(ValueError, match="out of range"):
        NodeResourceMask(1 << 24)
    with pytest.raises(ValueError, match="invalid pipe bits"):
        NodeResourceMask(1 << 3)

    combined = NodeResourceMask.for_dual_aiv("MTE2") | NodeResourceMask.for_aic("PIPE_FIX")
    assert combined.to_solver_resources(duration=2, offset=1) == (
        ("aiv", "aiv0", "MTE2", 2, 1),
        ("aic", "aic", "FIXPIPE", 2, 1),
        ("aiv", "aiv1", "MTE2", 2, 1),
    )


def test_ascend950_ssbuf_calibrated_hbm_and_3510_datapath_facts() -> None:
    """Verify SSBUF specs, calibrated HBM bandwidth, microarch caps, and 3510 DMA rules."""
    target = AscendTarget()
    spaces = target.get_facts(AscendSpaceFacts)
    assert spaces.ssbuf_bytes == 3072
    assert spaces.limit_bytes("SSBUF") == 3072
    assert spaces.ssbuf_align_bytes == 8
    assert spaces.ssbuf_latency_cycles == 8
    assert spaces.ssbuf_owner == "block"
    assert spaces.max_mte2_inflight_per_aiv == 16
    assert spaces.fixpipe_bytes_per_cycle == 128

    assert target.device.hbm_bandwidth_bytes_per_second == 1_633_108_573_296
    hbm_fact = AscendTarget.hardware.documents()["ascend.950pr"].fact("memory.hbm.bandwidth")
    assert hbm_fact.origin == "measured"

    assert spaces.is_dma_path_legal("L0C", "UB") is True
    assert spaces.is_dma_path_legal("UB", "L1") is True
    assert spaces.is_dma_path_legal("L1", "GM") is False
    assert spaces.is_dma_path_legal("GM", "L0A") is False
    assert spaces.is_dma_path_legal("GM", "L0B") is False
