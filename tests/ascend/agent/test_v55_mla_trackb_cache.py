"""Tests for MLA Track-B tile resolution, persistent .so cache, and Stage-A purity."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import examples.ascend950pr_ops.mla.kernels as mla_kernels
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    check_npu_entrypoint_purity,
    run_stage_a_static,
)
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

REPO_ROOT = Path(__file__).resolve().parents[3]


def _expected_trackb_sig(bm: int, d_nope: int, tl_dtype: str) -> str:
    launcher_src = mla_kernels._render_mla_trackb_launcher(
        mla_kernels._MLA_TRACKB_SRC,
        bm=bm,
        d_nope=d_nope,
        tl_dtype=tl_dtype,
    )
    non_inc_flags = () if "__simt_vf__" in launcher_src else tuple(PURE_SIMDVF_UBUF_FLAGS)
    payload = (launcher_src + "|" + " ".join(non_inc_flags)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _make_fake_cdll() -> tuple[list[str], object]:
    loaded_paths: list[str] = []

    def _fake_cdll(path: str) -> object:
        loaded_paths.append(path)
        return SimpleNamespace(launch_mla_trackb=SimpleNamespace(argtypes=None))

    return loaded_paths, _fake_cdll


def test_render_trackb_tu_anchor_bm_mapping_and_fallback() -> None:
    """Verify _render_trackb_tu maps MLA anchor cases to expected bm and preserves bm=64."""
    anchor_cases = (
        (1, {"b_dim": 2, "s_dim": 128, "s_kv": 128, "n_q": 128, "d_nope": 512}, 128, "half"),
        (6, {"b_dim": 2, "s_dim": 256, "s_kv": 256, "n_q": 128, "d_nope": 448}, 128, "half"),
        (7, {"b_dim": 1, "s_dim": 1, "s_kv": 2048, "n_q": 128, "d_nope": 512}, 32, "half"),
        (11, {"b_dim": 128, "s_dim": 1, "s_kv": 1024, "n_q": 128, "d_nope": 512}, 128, "half"),
        (13, {"B": 2, "S_q": 128, "S_kv": 128, "N_q": 128, "d_nope": 512}, 128, "half"),
    )
    for case_id, dims, expected_bm, c_dtype in anchor_cases:
        tu_src = mla_kernels._render_trackb_tu(tl_dtype="float16", **dims)
        d_nope = int(dims["d_nope"])
        expected_inst = f"mla_kv_reuse_c1c1c2<{expected_bm}, 128, {d_nope}, 64, {c_dtype}>"
        assert expected_inst in tu_src, f"Case {case_id} did not emit {expected_inst}"

    explicit_tu = mla_kernels._render_trackb_tu(bm=64, d_nope=512, tl_dtype="float16")
    assert "mla_kv_reuse_c1c1c2<64, 128, 512, 64, half>" in explicit_tu

    override_tu = mla_kernels._render_trackb_tu(
        b_dim=2,
        s_dim=128,
        s_kv=128,
        n_q=128,
        trackb_bm=32,
        d_nope=512,
        tl_dtype="bfloat16",
    )
    assert "mla_kv_reuse_c1c1c2<32, 128, 512, 64, bfloat16_t>" in override_tu


def test_build_mla_trackb_callable_prepopulated_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hit pre-populated TILELANG_CACHE_DIR .so with zero compile_ascend calls."""
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mla_kernels, "_TRACKB_SO_CACHE", {})

    sig = _expected_trackb_sig(bm=128, d_nope=512, tl_dtype="float16")
    so_path = tmp_path / f"trackb_{sig}.so"
    so_path.write_bytes(b"\x7fELF-prepopulated")

    compile_calls: list[dict[str, object]] = []

    def _fail_compile(*args: object, **kwargs: object) -> None:
        compile_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("compile_ascend must not run on cache hit")

    loaded_paths, fake_cdll = _make_fake_cdll()
    monkeypatch.setattr(mla_kernels, "compile_ascend", _fail_compile)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    fn1 = mla_kernels._build_mla_trackb_callable(128, 512, "float16", "BSND", True, 128)
    fn2 = mla_kernels._build_mla_trackb_callable(128, 512, "float16", "BNSD", False, 128)

    assert callable(fn1)
    assert callable(fn2)
    assert compile_calls == []
    assert loaded_paths == [str(so_path)]


def test_build_mla_trackb_callable_miss_compiles_atomically_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile once via atomic temp target on miss, then hit disk and CDLL caches."""
    cache_dir = tmp_path / "tl_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(mla_kernels, "_TRACKB_SO_CACHE", {})

    sig = _expected_trackb_sig(bm=64, d_nope=448, tl_dtype="bfloat16")
    expected_so = cache_dir / f"trackb_{sig}.so"
    compile_targets: list[str] = []

    def _fake_compile(_src: str, **kwargs: object) -> None:
        target = str(kwargs["path_target"])
        compile_targets.append(target)
        Path(target).write_bytes(b"\x7fELF-compiled-once")

    loaded_paths, fake_cdll = _make_fake_cdll()
    monkeypatch.setattr(mla_kernels, "compile_ascend", _fake_compile)
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    fn_bsnd = mla_kernels._build_mla_trackb_callable(128, 448, "bfloat16", "BSND", True, 64)
    fn_bnsd = mla_kernels._build_mla_trackb_callable(128, 448, "bfloat16", "BNSD", False, 64)

    assert callable(fn_bsnd)
    assert callable(fn_bnsd)
    assert len(compile_targets) == 1
    assert Path(compile_targets[0]).name.startswith(f".trackb_{sig}.")
    assert expected_so.is_file()
    assert expected_so.read_bytes() == b"\x7fELF-compiled-once"
    assert not list(cache_dir.glob(f".trackb_{sig}.*"))
    assert loaded_paths == [str(expected_so)]

    monkeypatch.setattr(mla_kernels, "_TRACKB_SO_CACHE", {})
    fn_cold_proc = mla_kernels._build_mla_trackb_callable(64, 448, "bfloat16", "BNSD", True, 64)
    assert callable(fn_cold_proc)
    assert len(compile_targets) == 1
    assert loaded_paths == [str(expected_so), str(expected_so)]


def test_mla_kernels_passes_static_stage_a_policy() -> None:
    """Verify edited MLA kernels module passes Stage-A purity and static DSL checks."""
    mla_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "kernels.py"
    mla_src = mla_path.read_text(encoding="utf-8")

    purity_errors = check_npu_entrypoint_purity(
        mla_src,
        "mla",
        track="hybrid_trackb",
        require_kernel_cache=True,
    )
    assert purity_errors == []

    stage_a = run_stage_a_static(
        mla_src,
        factory_name="_build_mla_prefill_func",
        dims={"B": 2, "S": 128, "S_kv": 128, "N_q": 128, "N_kv": 1, "D_nope": 512, "D_rope": 64},
    )
    assert stage_a.passed is True, stage_a.error_message
