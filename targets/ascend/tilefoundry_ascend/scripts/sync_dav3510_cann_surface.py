"""Synchronize and bisheng-verify dual-CANN dav-3510 C++20 surface snapshots."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _rel_path in ("src", "targets/ascend"):
    _cand_path = str(_REPO_ROOT / _rel_path)
    if _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _detect_cann_include_dirs,
    _detect_tilelang_include_dir,
    _resolve_bisheng_bin,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    DEFAULT_CANN_SERIES,
    VALID_CANN_SERIES,
    load_cann_surface,
)

_CANN_SURFACE_DIR = (
    Path(__file__).resolve().parents[1] / "ascend" / "target" / "cann_surface"
)
_REQUIRED_CANARY_SYMBOLS_BASE = (
    "ascend_gemm_l1",
    "asc_mmad",
    "asc_copy_gm2l1_nd2nz",
    "asc_set_l0c_copy_nz_para",
    "asc_copy_l0c2ub",
    "asc_copy_l0c2gm",
    "asc_copy_ub2l1",
    "asc_sync_notify",
    "asc_sync_wait",
    "asc_sync_block_arrive",
    "asc_sync_block_wait",
    "asc_lock",
    "asc_unlock",
    "asc_create_mask_b32",
    "simd_inst::update_mask_b32",
    "simd_inst::vlds_brc_elem",
    "simd_inst::vdup",
    "simd_inst::vadd",
    "simd_inst::vsts_1st",
    "asc_vf_call",
    "asc_shfl",
)


def _find_header_matches(cann_include: Path) -> list[str]:
    found: list[str] = []
    targets = (
        cann_include / "c_api" / "cube_datamove" / "cube_datamove.h",
        cann_include / "c_api" / "cube_compute" / "cube_compute.h",
        cann_include / "c_api" / "sync" / "sync.h",
        cann_include / "c_api" / "spr" / "spr.h",
        cann_include / "simt_api" / "device_warp_functions.h",
    )
    decl_pat = re.compile(r"__aicore__\s+inline\s+\w+\s+(asc_[a-zA-Z0-9_]+)\s*\(")
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in decl_pat.finditer(text):
            sym = match.group(1)
            if sym not in found:
                found.append(sym)
    return found


def _build_canary_source(cann_series: str) -> str:
    extra_header = "#include <c_api/asc_cube.h>\n" if cann_series == "cann_9_3" else ""
    if cann_series == "cann_9_3":
        nd2nz_calls = """    asc_copy_gm2l1_nd2nz(
        l1_a, gm_a, 64ULL, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 64U, 64U, 4096ULL, false
    );
    asc_copy_gm2l1_nd2nz(
        l1_b, gm_b, 64ULL, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 64U, 64U, 4096ULL, false
    );"""
        l0c2gm_call = """    asc_copy_l0c2gm(
        gm_c, l0c, 64, 64, 64, 64,
        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
        asc_unit_flag_mode::DISABLE,
        static_cast<asc_quant_mode>(0),
        asc_relu_pre_mode::NONE,
        false, true, false, false
    );"""
    else:
        nd2nz_calls = """    asc_copy_gm2l1_nd2nz(l1_a, gm_a, 1, 64, 64, 0, 64, 1, 64, 0);
    asc_copy_gm2l1_nd2nz(l1_b, gm_b, 1, 64, 64, 0, 64, 1, 64, 0);"""
        l0c2gm_call = """    asc_copy_l0c2gm(
        gm_c, l0c, 64, 64, 64, 64,
        0U, 0ULL, 0U,
        false, true
    );"""
    return (
        "#include <tl_templates/ascend/common.h>\n"
        "#include <tl_templates/ascend/gemm.h>\n"
        f"{extra_header}"
        "#include <c_api/asc_simd.h>\n"
        "#include <simt_api/asc_simt.h>\n\n"
        "__simd_vf__ void vf_simd_canary(\n"
        "    __ubuf__ float* dst_ub, __ubuf__ float* scalar_ub, int rem\n"
        ") {\n"
        "    vector_bool m32 = asc_create_mask_b32(PAT_ALL);\n"
        "    vector_bool m32_tail = simd_inst::update_mask_b32(rem);\n"
        "    auto brc = simd_inst::vlds_brc_elem<float>(scalar_ub, 0);\n"
        "    auto v0 = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);\n"
        "    auto v1 = simd_inst::vadd(v0, brc, m32_tail, MODE_ZEROING);\n"
        "    simd_inst::vsts_1st(v1, dst_ub, 0, m32);\n"
        "}\n\n"
        "__simt_vf__ __launch_bounds__(128) void vf_simt_canary(\n"
        "    __gm__ float* out, __gm__ const float* in, int n\n"
        ") {\n"
        "    uint32_t tid = threadIdx.x;\n"
        "    if ((int)tid < n) {\n"
        "        float v = asc_shfl(in[tid], 0);\n"
        "        out[tid] = v + 1.0f;\n"
        "    }\n"
        "}\n\n"
        'extern "C" __global__ __aicore__ void dav3510_surface_canary(\n'
        "    __gm__ bfloat16_t* gm_a,\n"
        "    __gm__ bfloat16_t* gm_b,\n"
        "    __gm__ float* gm_c,\n"
        "    __gm__ float* gm_out\n"
        ") {\n"
        "    __cbuf__ bfloat16_t* l1_a = reinterpret_cast<__cbuf__ bfloat16_t*>(0);\n"
        "    __cbuf__ bfloat16_t* l1_b = reinterpret_cast<__cbuf__ bfloat16_t*>(32768);\n"
        "    __cc__ float* l0c = reinterpret_cast<__cc__ float*>(0);\n"
        "    __ubuf__ float* ub_tmp = reinterpret_cast<__ubuf__ float*>(0);\n"
        f"{nd2nz_calls}\n"
        "    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);\n"
        "    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);\n"
        "    asc_unlock(PIPE_MTE1, 0, ASC_LOCK_BLOCK);\n"
        "    ascend_gemm_l1<64, 64, 64, 32, false, bfloat16_t>(l0c, l1_a, l1_b, 1, 0, 0);\n"
        "    asc_lock(PIPE_M, 0, ASC_LOCK_BLOCK);\n"
        "    asc_set_l0c_copy_nz_para(1, 0, 0);\n"
        "    asc_copy_l0c2ub(\n"
        "        ub_tmp, l0c, 64, 64, 64, 64, 0,\n"
        "        asc_dual_dst_mode::DUAL_DST_DISABLE,\n"
        "        asc_unit_flag_mode::DISABLE,\n"
        "        static_cast<asc_quant_mode>(0),\n"
        "        asc_relu_pre_mode::NONE,\n"
        "        false, true, false, false\n"
        "    );\n"
        f"{l0c2gm_call}\n"
        "    asc_copy_ub2l1(l1_a, ub_tmp, 1, 16, 0, 0);\n"
        "    asc_sync_block_arrive(PIPE_FIX, 0);\n"
        "    asc_sync_block_wait(PIPE_V, 0);\n"
        "    asc_vf_call<vf_simd_canary>(dim3(1, 1, 1), ub_tmp, ub_tmp, 16);\n"
        "    asc_vf_call<vf_simt_canary>(dim3(128, 1, 1), gm_out, gm_c, 128);\n"
        "}\n"
    )


def validate_snapshot_schema(snapshot_path: Path) -> str:
    """Validate snapshot TOML schema, metadata, and ellipsis-free signatures."""
    data = tomllib.loads(snapshot_path.read_text(encoding="utf-8"))
    surface_meta = data.get("surface")
    if not isinstance(surface_meta, dict):
        raise ValueError(f"{snapshot_path}: missing [surface] table")
    series = str(surface_meta.get("cann_series", "")).strip()
    if series not in VALID_CANN_SERIES:
        raise ValueError(f"{snapshot_path}: invalid cann_series {series!r}")
    if surface_meta.get("cce_arch") != "dav-3510":
        raise ValueError(f"{snapshot_path}: expected cce_arch='dav-3510'")
    primitives = data.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError(f"{snapshot_path}: expected non-empty [[primitives]] list")
    symbols: set[str] = set()
    for idx, entry in enumerate(primitives):
        if not isinstance(entry, dict):
            raise ValueError(f"{snapshot_path}: primitive #{idx} is not a table")
        sym = str(entry.get("symbol", "")).strip()
        sig = str(entry.get("signature", "")).strip()
        if not sym or not sig:
            raise ValueError(f"{snapshot_path}: primitive #{idx} missing symbol/signature")
        if "..." in sig:
            raise ValueError(f"{snapshot_path}: ellipsis '...' forbidden in signature {sym!r}")
        for key in ("pipes", "facets", "required_headers"):
            vals = entry.get(key)
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"{snapshot_path}: {sym!r} missing non-empty {key}")
        symbols.add(sym)
    missing = [s for s in _REQUIRED_CANARY_SYMBOLS_BASE if s not in symbols]
    if missing:
        raise ValueError(f"{snapshot_path}: missing required Canary symbols {missing}")
    return series


def verify_snapshot(
    snapshot_path: Path,
    *,
    cann_home: Path | None = None,
    tilelang_src: Path | None = None,
    bisheng_bin: Path | None = None,
) -> int:
    """Validate TOML structure and compile a dav-3510 C++20 Canary with bisheng."""
    series = validate_snapshot_schema(snapshot_path)
    load_cann_surface(series)
    resolved_bisheng = _resolve_bisheng_bin(
        bisheng_bin=bisheng_bin,
        cann_root=cann_home,
    )
    if resolved_bisheng is None and cann_home is not None:
        cand = cann_home / "bin" / "bisheng"
        if cand.is_file():
            resolved_bisheng = str(cand)
    if resolved_bisheng is None:
        raise FileNotFoundError("bisheng compiler not found for --verify")

    inc_dirs = _detect_cann_include_dirs(cann_root=cann_home, bisheng_bin=resolved_bisheng)
    if cann_home is not None:
        for rel in ("x86_64-linux/asc", "x86_64-linux/asc/include", "x86_64-linux/include"):
            cand_dir = str((cann_home / rel))
            if (cann_home / rel).is_dir() and cand_dir not in inc_dirs:
                inc_dirs.insert(0, cand_dir)
    tl_inc = str(tilelang_src) if tilelang_src is not None else _detect_tilelang_include_dir()
    if not tl_inc or not Path(tl_inc).is_dir():
        raise FileNotFoundError("tilelang include root not found for --verify")

    canary_src = _build_canary_source(series)
    with tempfile.TemporaryDirectory(prefix="tf_dav3510_verify_") as tmp_dir:
        src_path = Path(tmp_dir) / f"canary_{series}.cce"
        src_path.write_text(canary_src, encoding="utf-8")
        cmd = [
            resolved_bisheng,
            "--npu-arch=dav-3510",
            "--cce-aicore-only",
            "-x",
            "cce",
            "-std=c++20",
            f"-I{tl_inc}",
            *[f"-I{d}" for d in inc_dirs],
            "-fsyntax-only",
            str(src_path),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or proc.stdout or "bisheng verification failed\n")
            return proc.returncode or 1
    print(f"Verified {snapshot_path} ({series}) against {resolved_bisheng} cleanly.")
    return 0


def extract_snapshot(
    *,
    cann_home: Path,
    series: str,
    output_path: Path | None = None,
) -> int:
    """Audit live CANN headers against the tracked snapshot and emit synced TOML."""
    if series not in VALID_CANN_SERIES:
        raise ValueError(f"Unsupported series {series!r}; expected {VALID_CANN_SERIES}")
    asc_inc = cann_home / "x86_64-linux" / "asc" / "include"
    if not asc_inc.is_dir():
        asc_inc = cann_home / "asc" / "include"
    discovered = _find_header_matches(asc_inc)
    base_toml = (_CANN_SURFACE_DIR / f"{series}.toml").read_text(encoding="utf-8")
    snapshot_data = tomllib.loads(base_toml)
    snapshot_asc = sorted(
        {
            str(entry.get("symbol", "")).strip()
            for entry in snapshot_data.get("primitives", [])
            if isinstance(entry, dict) and str(entry.get("symbol", "")).startswith("asc_")
        }
    )
    discovered_set = set(discovered)
    snapshot_set = set(snapshot_asc)
    missing_from_headers = sorted(snapshot_set - discovered_set)
    extra_in_headers = sorted(discovered_set - snapshot_set)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(base_toml, encoding="utf-8")
    print(
        f"Discovered {len(discovered)} asc_* C-API symbols under {asc_inc} for {series} "
        f"(snapshot asc_*: {len(snapshot_asc)}, missing: {len(missing_from_headers)}, "
        f"extra: {len(extra_in_headers)})."
    )
    if missing_from_headers:
        print(f"Missing snapshot asc_* symbols in headers: {', '.join(missing_from_headers)}")
    if extra_in_headers:
        print(f"Extra header asc_* symbols not in snapshot: {', '.join(extra_in_headers)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for dav-3510 dual-CANN surface extraction and verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true", help="Extract/audit CANN headers")
    mode.add_argument("--verify", action="store_true", help="Compile Canary against bisheng")
    parser.add_argument("--cann-home", type=Path, default=None, help="Path to CANN toolkit root")
    parser.add_argument("--series", default=DEFAULT_CANN_SERIES, choices=VALID_CANN_SERIES)
    parser.add_argument("--snapshot", type=Path, default=None, help="Snapshot TOML path")
    parser.add_argument("--output", type=Path, default=None, help="Output TOML for --extract")
    parser.add_argument("--tilelang-src", type=Path, default=None, help="TileLang src root")
    parser.add_argument("--bisheng-bin", type=Path, default=None, help="Path to bisheng binary")
    args = parser.parse_args(argv)

    if args.extract:
        cann_home = args.cann_home or Path(os.environ.get("ASCEND_HOME_PATH", ""))
        return extract_snapshot(
            cann_home=cann_home,
            series=args.series,
            output_path=args.output,
        )
    snap = args.snapshot or (_CANN_SURFACE_DIR / f"{args.series}.toml")
    return verify_snapshot(
        snap,
        cann_home=args.cann_home,
        tilelang_src=args.tilelang_src,
        bisheng_bin=args.bisheng_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
