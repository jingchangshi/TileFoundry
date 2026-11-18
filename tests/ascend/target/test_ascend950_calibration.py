"""Tests for Ascend 950 (dav-3510) DType extensions, 950DT, and architectural facts."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tilefoundry.ir.types import BoolDType, DType, FloatDType, IntegerDType
from tilefoundry.target import TopologyFacts, TopologyLevelFacts
from tilefoundry_ascend import AscendTarget
from tilefoundry_ascend.ascend.target import (
    ASCEND_950DT_ID,
    ASCEND_950PR_ID,
    AscendSpaceFacts,
    cube_macs_per_cycle,
    detect_runtime_core_counts,
)
from tilefoundry_ascend.ascend.target.hardware import BUILTIN_HARDWARE_DOCUMENTS


def test_core_dtype_extensions_and_backward_compatibility() -> None:
    """Verify all existing and newly added Ascend 950 DType members resolve cleanly."""
    existing = ("f32", "f16", "bf16", "fp8e4m3", "f8e8m0", "f4e2m1", "i32", "i64", "bool")
    for name in existing:
        assert DType.from_name(name) is getattr(DType, name)

    expected_ints = (
        ("i8", 8, True),
        ("u8", 8, False),
        ("i16", 16, True),
        ("u16", 16, False),
        ("u32", 32, False),
        ("u64", 64, False),
        ("i4", 4, True),
        ("u4", 4, False),
    )
    for name, bits, signed in expected_ints:
        dt = DType.from_name(name)
        assert dt is getattr(DType, name)
        assert isinstance(dt, IntegerDType)
        assert dt == IntegerDType(name=name, bit_width=bits, signed=signed)

    expected_floats = (
        ("fp8e5m2", 8, 5, 2),
        ("hif8", 8, 4, 3),
        ("f4e1m2", 4, 1, 2),
    )
    for name, bits, exp, man in expected_floats:
        dt = DType.from_name(name)
        assert dt is getattr(DType, name)
        assert isinstance(dt, FloatDType)
        assert dt == FloatDType(name=name, bit_width=bits, exponent_bits=exp, mantissa_bits=man)

    assert isinstance(DType.bool, BoolDType)


def test_ascend_950dt_registration_and_topology() -> None:
    """Verify ascend.950dt is registered with 36 AIC / 72 AIV, 144 GiB HBM, 4.0 TB/s, 1.65 GHz."""
    assert set(BUILTIN_HARDWARE_DOCUMENTS) >= {
        "davinci_950.toml",
        "ascend_950pr.toml",
        "ascend_950dt.toml",
    }
    available_ids = {t.device_id for t in AscendTarget.available()}
    assert {ASCEND_950PR_ID, ASCEND_950DT_ID} <= available_ids

    dt = AscendTarget("ascend.950dt")
    assert dt.device_id == "ascend.950dt"
    assert dt.device.aic_count == 36
    assert dt.device.aiv_count == 72
    assert dt.device.sm_count == 36
    assert dt.device.hbm_capacity_bytes == 144 * 1024 * 1024 * 1024
    assert dt.device.hbm_bandwidth_bytes_per_second == 4_000_000_000_000
    assert dt.device.clock_ghz == pytest.approx(1.65)
    assert dt.to_python().text == 'AscendTarget("ascend.950dt")'

    topo = dt.get_facts(TopologyFacts)
    assert topo.level("block") == TopologyLevelFacts("block", 36, 36)
    assert topo.level("aic") == TopologyLevelFacts("aic", 1, 36)
    assert topo.level("aiv") == TopologyLevelFacts("aiv", 2, 72)
    assert topo.level("lane") == TopologyLevelFacts("lane", 64, 72 * 64)

    spaces = dt.get_facts(AscendSpaceFacts)
    assert spaces.num_aic_cores == 36
    assert spaces.num_aiv_cores == 72
    assert spaces.clock_ghz == pytest.approx(1.65)
    assert spaces.hbm_peak_bandwidth_bps == 4_000_000_000_000
    assert spaces.effective_hbm_bandwidth_bps(4096) == pytest.approx(4_000_000_000_000.0)


def test_dav3510_cube_macs_bank_geometry_and_sync_facts() -> None:
    """Verify exact dav-3510 Cube MACs/cycle, L1/UB banks, Bias/FixBuf, L2, L0 layouts, and sync."""
    expected_macs = {
        "f4e2m1": 16384,
        "f4e1m2": 16384,
        "fp8e4m3": 8192,
        "fp8e5m2": 8192,
        "hif8": 8192,
        "i8": 8192,
        "u8": 8192,
        "f16": 4096,
        "bf16": 4096,
        "hf32": 2048,
        "f32": 256,
    }
    for dtype_name, macs in expected_macs.items():
        assert cube_macs_per_cycle(dtype_name) == macs
    assert cube_macs_per_cycle("f32", hf32=True) == 2048
    assert cube_macs_per_cycle("f32", hf32=False) == 256

    with pytest.raises(ValueError, match="unsupported Ascend 950 Cube dtype"):
        cube_macs_per_cycle("i4")

    for target_id in (ASCEND_950PR_ID, ASCEND_950DT_ID):
        target = AscendTarget(target_id)
        arch = target.architecture
        spaces = target.get_facts(AscendSpaceFacts)

        assert arch.cube_macs_per_cycle_map == expected_macs
        assert arch.cube_macs_per_cycle("f16") == 4096
        assert arch.cube_macs_per_cycle("f32", hf32=True) == 2048
        assert spaces.cube_macs_per_cycle("fp8e4m3") == 8192
        assert spaces.cube_macs_per_cycle("f32", hf32=True) == 2048
        assert spaces.clock_ghz == pytest.approx(1.65)

        assert spaces.l1_banks == 2
        assert spaces.l1_bank_bytes == 262144
        assert spaces.ub_bank_groups == 8
        assert spaces.ub_banks_per_group == 2
        assert spaces.ub_bank_bytes == 16384
        assert spaces.ub_datablock_bytes == 32
        assert spaces.bias_table_bytes == 4096
        assert spaces.fixbuf_bytes == 16384
        assert spaces.l2_capacity_bytes == 134217728
        assert spaces.l2_bytes_per_cycle_per_aic == 101
        assert spaces.l0a_layout == "NZ"
        assert spaces.l0b_layout == "ZN"
        assert spaces.l0c_layout == "NZ"
        assert spaces.max_crosscore_flag_depth == 15
        assert spaces.hardware_mutex_count == 28
        assert spaces.max_mmad_mnk == 4095
        assert spaces.forbid_cube_int4 is True


def test_detect_runtime_core_counts_safe_offline_and_mocked_npu_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify runtime core count detection never crashes offline and parses npu-smi output."""
    aic, aiv = detect_runtime_core_counts()
    assert aic > 0 and aiv == 2 * aic

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    assert detect_runtime_core_counts(default_aic=36) == (36, 72)

    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/local/sbin/npu-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\tAicore Freq(MHZ) : 1650\n\tAicore Count : 36\n",
        ),
    )
    assert detect_runtime_core_counts(default_aic=28) == (36, 72)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("npu-smi failed")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert detect_runtime_core_counts(default_aic=28) == (28, 56)
