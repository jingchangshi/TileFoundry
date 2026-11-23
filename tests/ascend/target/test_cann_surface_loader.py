"""Tests for dual-CANN dav-3510 surface snapshots, resolution, and hardware spec."""

from __future__ import annotations

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
from tilefoundry_ascend.scripts.sync_dav3510_cann_surface import (
    _build_canary_source,
    extract_snapshot,
    validate_snapshot_schema,
)

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


def test_dynamic_cann_series_resolution_without_module_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve cann_9_2 and cann_9_3 dynamically on TILEFOUNDRY_CANN_SERIES toggles."""
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_2")
    surf_92 = get_dialect_surface("cce_trackb")
    trackb_92 = resolve_trackb_cce_surface("cce_trackb")
    assert surf_92.display_label == "CCE C++20 (dav-3510 / CANN 9.2)"
    assert trackb_92 is surf_92
    sig_92 = {e.symbol: e.signature for e in surf_92.primitives}["asc_copy_l0c2gm"]
    assert "asc_store_l2_cache_mode" not in sig_92
    assert sig_92.count(",") == 10

    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
    surf_93 = get_dialect_surface("cce_trackb")
    trackb_93 = resolve_trackb_cce_surface("cce_trackb")
    assert surf_93.display_label == "CCE C++20 (dav-3510 / CANN 9.3)"
    assert trackb_93 is surf_93
    sig_93 = {e.symbol: e.signature for e in surf_93.primitives}["asc_copy_l0c2gm"]
    assert "asc_store_l2_cache_mode" in sig_93
    assert sig_93.count(",") == 13


def test_sync_canary_cann93_nd2nz_8arg_and_extract_snapshot_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CANN 9.3 8-arg nd2nz Canary coverage and extract_snapshot symbol diffing."""
    canary_93 = _build_canary_source("cann_9_3")
    assert "asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM" in canary_93
    assert "asc_copy_gm2l1_nd2nz(\n        l1_a, gm_a, 64ULL," in canary_93
    assert "asc_copy_gm2l1_nd2nz(l1_a, gm_a, 1," not in canary_93

    canary_92 = _build_canary_source("cann_9_2")
    assert "asc_copy_gm2l1_nd2nz(l1_a, gm_a, 1, 64, 64, 0, 64, 1, 64, 0);" in canary_92

    hdr_dir = tmp_path / "x86_64-linux" / "asc" / "include" / "c_api" / "cube_datamove"
    hdr_dir.mkdir(parents=True)
    (hdr_dir / "cube_datamove.h").write_text(
        "__aicore__ inline void asc_copy_gm2l1_nd2nz();\n"
        "__aicore__ inline void asc_synthetic_extra_op();\n",
        encoding="utf-8",
    )
    rc = extract_snapshot(cann_home=tmp_path, series="cann_9_3")
    assert rc == 1
    captured = capsys.readouterr()
    assert "Discovered 2 asc_* C-API symbols" in captured.out
    assert "missing:" in captured.out
    assert "extra: 1" in captured.out
    assert "asc_synthetic_extra_op" in captured.out
    assert "Missing snapshot asc_* symbols in headers:" in captured.err
    assert "asc_copy_l0c2gm" in captured.err

    rc_92 = extract_snapshot(cann_home=tmp_path, series="cann_9_2")
    assert rc_92 == 0
    assert "Skipping live header enforcement" in capsys.readouterr().out
