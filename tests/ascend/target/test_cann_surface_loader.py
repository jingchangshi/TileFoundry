"""Tests for dual-CANN dav-3510 surface snapshots, resolution, and hardware spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry_ascend.ascend.target import (
    ASCEND_950PR_ID,
    DEFAULT_CANN_SERIES,
    VALID_CANN_SERIES,
    AscendTarget,
    load_cann_surface,
    resolve_cann_series,
)
from tilefoundry_ascend.scripts.sync_dav3510_cann_surface import validate_snapshot_schema

_CANN_SURFACE_DIR = (
    Path(__file__).resolve().parents[3]
    / "targets"
    / "ascend"
    / "tilefoundry_ascend"
    / "ascend"
    / "target"
    / "cann_surface"
)


def test_davinci950_hardware_spec_converged_fields() -> None:
    """Verify davinci_950 exposes series, cce_arch, SKUs, and dual-CANN toolchain fields."""
    target = AscendTarget(ASCEND_950PR_ID)
    arch = target.architecture
    assert arch.name == "dav_3510"
    assert arch.series == "ascend_950"
    assert arch.cce_arch == "dav-3510"
    assert arch.supported_skus == ("ascend_950pr", "ascend_950dt")
    assert arch.default_cann_series == "cann_9_3"
    assert arch.supported_cann_series == ("cann_9_2", "cann_9_3")


def test_resolve_cann_series_precedence_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve explicit arg, env override, ASCEND_HOME_PATH version, and fail on bad input."""
    monkeypatch.delenv("TILEFOUNDRY_CANN_SERIES", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    assert resolve_cann_series() == DEFAULT_CANN_SERIES
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/Ascend/cann-9.2.0")
    assert resolve_cann_series() == "cann_9_2"
    monkeypatch.setenv("TILEFOUNDRY_NPU_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_2")
    assert resolve_cann_series() == "cann_9_2"
    assert resolve_cann_series("cann_9_3") == "cann_9_3"

    with pytest.raises(ValueError, match="Unsupported CANN series"):
        resolve_cann_series("cann_8_0")


@pytest.mark.parametrize("series", VALID_CANN_SERIES)
def test_dual_cann_surface_snapshots_schema_and_signatures(series: str) -> None:
    """Validate both cann_9_2 and cann_9_3 snapshots load with ellipsis-free signatures."""
    snap_path = _CANN_SURFACE_DIR / f"{series}.toml"
    assert validate_snapshot_schema(snap_path) == series

    spec = load_cann_surface(series)
    assert spec.dialect_id == "cce_cpp20"
    by_sym = {entry.symbol: entry for entry in spec.primitives}
    assert "ascend_gemm_l1" in by_sym
    assert "asc_copy_l0c2ub" in by_sym
    assert "asc_copy_l0c2gm" in by_sym
    for entry in spec.primitives:
        assert "..." not in entry.signature
        assert entry.pipes
        assert entry.facets
        assert entry.required_headers

    if series == "cann_9_3":
        assert "asc_mmad_mx" in by_sym
        assert "asc_store_l2_cache_mode" in by_sym["asc_copy_l0c2gm"].signature
        assert "#include <c_api/asc_cube.h>" in spec.required_headers_or_imports
    else:
        assert "asc_mmad_mx" not in by_sym
        assert "unit_flag_mode" in by_sym["asc_copy_l0c2gm"].signature
