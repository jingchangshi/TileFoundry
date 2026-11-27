"""Tests for CANN 9.3 dav-3510 surface snapshots, resolution, and hardware spec."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tilefoundry_ascend.ascend.target import (
    ASCEND_950PR_ID,
    DEFAULT_CANN_SERIES,
    VALID_CANN_SERIES,
    AscendTarget,
    get_dialect_surface,
    load_cann_surface,
    resolve_cann_series,
)
from tilefoundry_ascend.ascend.target.dialect_surface import resolve_trackb_cce_surface

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
    """Verify davinci_950 exposes series, cce_arch, SKUs, and CANN 9.3 toolchain fields."""
    target = AscendTarget(ASCEND_950PR_ID)
    arch = target.architecture
    assert arch.name == "dav_3510"
    assert arch.series == "ascend_950"
    assert arch.cce_arch == "dav-3510"
    assert arch.supported_skus == ("ascend_950pr", "ascend_950dt")
    assert arch.default_cann_series == "cann_9_3"
    assert arch.supported_cann_series == ("cann_9_3",)


def test_resolve_cann_series_precedence_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve explicit arg, env override, ASCEND_HOME_PATH version, and fail on bad input."""
    monkeypatch.delenv("TILEFOUNDRY_CANN_SERIES", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    assert resolve_cann_series() == DEFAULT_CANN_SERIES == "cann_9_3"
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/Ascend/cann-9.3.0")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.setenv("TILEFOUNDRY_NPU_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == "cann_9_3"
    assert resolve_cann_series("cann_9_3") == "cann_9_3"

    with pytest.raises(ValueError, match="Unsupported CANN series"):
        resolve_cann_series("cann_8_0")


@pytest.mark.parametrize("series", VALID_CANN_SERIES)
def test_cann_surface_snapshots_schema_and_signatures(series: str) -> None:
    """Validate CANN 9.3 snapshot TOML loads with ellipsis-free signatures."""
    assert VALID_CANN_SERIES == ("cann_9_3",)
    snap_path = _CANN_SURFACE_DIR / f"{series}.toml"
    raw = tomllib.loads(snap_path.read_text(encoding="utf-8"))
    assert raw["surface"]["cann_series"] == series

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

    assert "asc_mmad_mx" in by_sym
    assert "asc_store_l2_cache_mode" in by_sym["asc_copy_l0c2gm"].signature
    assert "#include <c_api/asc_cube.h>" in spec.required_headers_or_imports


def test_dynamic_cann_series_resolution_without_module_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dynamic CANN 9.3 surface resolution without module reload."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
    surf_93 = get_dialect_surface("cce_trackb")
    trackb_93 = resolve_trackb_cce_surface("cce_trackb")
    assert surf_93.display_label == "CCE C++20 (dav-3510 / CANN 9.3)"
    assert trackb_93 is surf_93
    sig_93 = {e.symbol: e.signature for e in surf_93.primitives}["asc_copy_l0c2gm"]
    assert "asc_store_l2_cache_mode" in sig_93
    assert sig_93.count(",") == 13
