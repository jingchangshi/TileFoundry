"""Tests for v5.2 Bug-3 symbolic var binding, Bug-4 Huber calibration, and Pillar-1 sweeper."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as tp
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    LatticeSweepCandidate,
    sweep_executable_lattice_zero_llm,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures,
    HybridAscendCostModel,
    _huber_location,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_bug3_sparse_attention_dim_mappings_and_scalar_extent_cap() -> None:
    """Verify S1/S2/N1/N2/topK/G/T bindings and conservative 16 cap for unbound scalars."""
    import tilelang.ascend.language as T  # noqa: PLC0415

    dims: dict[str, int] = {
        "B": 2,
        "S1": 64,
        "S2": 2048,
        "N1": 16,
        "N2": 4,
        "topK": 128,
        "T": 32,
    }
    merged = gf._merge_dims_into_kwargs({}, dims)
    assert merged["s1_dim"] == 64
    assert merged["s2_dim"] == 2048
    assert merged["n_q"] == 16
    assert merged["n_kv"] == 4
    assert merged["top_k"] == 128
    assert merged["t_dim"] == 32

    resolved = gf._resolve_stage_a_var_bindings(dims, None)
    assert resolved is not None
    assert resolved["s1_dim"] == 64
    assert resolved["s2_dim"] == 2048
    assert resolved["n_q"] == 16
    assert resolved["n_kv"] == 4
    assert resolved["top_k"] == 128
    assert resolved["q_total_rows"] == 2 * 64 * 16
    assert resolved["g_ratio"] == 4

    def _probe_builder(
        s1_dim: int,
        s2_dim: int,
        n_q: int,
        n_kv: int,
        top_k: int,
        t_dim: int,
    ) -> dict[str, int]:
        return {
            "s1_dim": s1_dim,
            "s2_dim": s2_dim,
            "n_q": n_q,
            "n_kv": n_kv,
            "top_k": top_k,
            "t_dim": t_dim,
        }

    filtered = gf._filter_kwargs_for_callable(_probe_builder, dims)
    assert filtered == {
        "s1_dim": 64,
        "s2_dim": 2048,
        "n_q": 16,
        "n_kv": 4,
        "top_k": 128,
        "t_dim": 32,
    }

    dsl_src = """
import tilelang.ascend.language as T


def _build_sparse_scalar_kernel(s1_dim: int, s2_dim: int, n_kv: int, top_k: int):
    total_elems = T.dynamic("total_elems", "int32")

    @T.prim_func
    def main(
        X: T.Tensor((1048576,), "float16"),
        Y: T.Tensor((total_elems,), "float16"),
        num_blocks: T.int32,
        s1_arg: T.int32,
        s2_arg: T.int32,
        n_kv_arg: T.int32,
        top_k_arg: T.int32,
    ):
        with T.Kernel(56) as cid:
            ub = T.alloc_shared((256,), "float16")
            for b in T.serial(num_blocks):
                idx = (b * 56 + cid) * 256
                T.copy(X[idx : idx + 256], ub)
                T.copy(ub, Y[idx : idx + 256])
            _guard = s1_arg + s2_arg + n_kv_arg + top_k_arg + s1_dim + s2_dim + n_kv + top_k

    return main
"""

    target = AscendTarget("ascend.950pr")
    stage_a = gf.run_stage_a_static(
        dsl_src,
        factory_name="_build_sparse_scalar_kernel",
        dims=dims,
        target=target,
    )
    assert stage_a.passed is True, stage_a.error_message
    assert stage_a.ledger is not None
    assert stage_a.ledger.is_legal is True
    mte2_cycles = max(
        stage_a.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0),
        stage_a.ledger.pipe_cycles.get("AIV1.PIPE_MTE2", 0),
    )
    assert mte2_cycles > 0
    assert mte2_cycles < 1_000_000

    res_direct = inspect_tilelang_primfunc(
        stage_a.prim_func,
        target=target,
        var_bindings=resolved,
    )
    assert res_direct.ledger.is_legal is True


def test_bug4_huber_location_and_two_stage_offset_convergence(tmp_path: Path) -> None:
    """Verify Huber location, 12x bucket intercept convergence in <=4 samples, and roundtrip."""
    assert _huber_location([]) == 0.0
    assert _huber_location([3.0]) == 3.0
    robust = _huber_location([2.8, 2.9, 3.0, 3.1, 25.0])
    assert 2.8 <= robust <= 3.35

    target = AscendTarget("ascend.950pr")
    lake = ExperienceLake(tmp_path / "lake")
    cal_path = tmp_path / "calibration.json"
    model = HybridAscendCostModel(target, calibration_path=cal_path, anomaly_lake=lake)
    axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    workloads = [
        CostFeatures(
            cube_flops=2.0e12 * (i + 1),
            gm_bytes=8.0e6 * (i + 1),
            total_tiles=28,
            contig_burst_bytes=32768,
            topology_axes=axes,
        )
        for i in range(4)
    ]
    multipliers = [12.0 * 0.94, 12.0 * 1.03, 12.0 * 1.05, 12.0 * 0.98]

    measured_list: list[float] = []
    for wl, mult in zip(workloads, multipliers):
        t01 = model._compute_tier0_and_tier1(wl)
        base_us = max(1e-6, t01.tier0_us + t01.tier1_penalty_us)
        analytical = model.observe(wl, base_us * mult)
        measured_list.append(analytical * mult)

    assert model.sample_count == 4
    anomaly_file = lake.root_dir / "unmodeled_anomalies.jsonl"
    assert not anomaly_file.exists()

    log_errors = [
        abs(math.log(m / max(1e-6, model.predict(wl).pred_us)))
        for wl, m in zip(workloads, measured_list)
    ]
    assert statistics.median(log_errors) < 0.20

    base_wl = workloads[0]
    base_analytical = model._compute_tier0_and_tier1(base_wl).tier0_us
    model.observe(base_wl, base_analytical * math.exp(6.5))
    assert model.sample_count == 4
    assert anomaly_file.is_file()
    anomaly_rows = [json.loads(ln) for ln in anomaly_file.read_text(encoding="utf-8").splitlines()]
    assert len(anomaly_rows) == 1

    model.observe(base_wl, base_analytical * math.exp(9.0))
    assert model.sample_count == 4
    assert len(anomaly_file.read_text(encoding="utf-8").splitlines()) == 2

    model.save_calibration(cal_path)
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    assert "global_intercept" in payload
    assert "bucket_intercepts" in payload
    assert "coupled_1c2v:bidirectional_cv_ring" in payload["bucket_intercepts"]

    reloaded = HybridAscendCostModel(target, calibration_path=cal_path)
    assert reloaded.predict(workloads[2]).pred_us == pytest.approx(
        model.predict(workloads[2]).pred_us,
        rel=1e-9,
    )


_SYNTHETIC_VECTOR_SOURCE = """
import tilelang.ascend.language as T


def _select_vec_config(d_dim: int = 128) -> dict[str, int]:
    return {"tile_m": 4, "num_stages": 1, "d_dim": int(d_dim)}


def _build_vec_prim_func(
    d_dim: int = 128,
    tile_m: int = 4,
    num_stages: int = 1,
    **_extra,
):
    rows = max(1, int(tile_m))
    stages = max(1, int(num_stages))

    @T.prim_func
    def main(
        X: T.Tensor((2048, d_dim), "float16"),
        Y: T.Tensor((2048, d_dim), "float16"),
    ):
        with T.Kernel(56) as cid:
            x_ub = T.alloc_shared((rows, d_dim), "float16")
            y_ub = T.alloc_shared((rows, d_dim), "float16")
            for it in T.Pipelined(4, num_stages=stages):
                row_off = cid * rows + it * 56 * rows
                T.copy(X[row_off : row_off + rows, :], x_ub)
                with T.SimdVF():
                    for r, c in T.Parallel(rows, d_dim):
                        y_ub[r, c] = x_ub[r, c] + T.float16(1.0)
                T.copy(y_ub, Y[row_off : row_off + rows, :])

    return main
"""


_SYNTHETIC_CUBE_SOURCE = """
import tilelang.ascend.language as T


def _select_cube_config(m_dim: int = 256, n_dim: int = 256, k_dim: int = 128) -> dict[str, int]:
    return {
        "bm": 64,
        "bn": 64,
        "bk": 32,
        "num_stages": 1,
        "m_dim": int(m_dim),
        "n_dim": int(n_dim),
        "k_dim": int(k_dim),
    }


def _build_cube_prim_func(
    m_dim: int = 256,
    n_dim: int = 256,
    k_dim: int = 128,
    bm: int = 64,
    bn: int = 64,
    bk: int = 32,
    num_stages: int = 1,
    **_extra,
):
    stages = max(1, int(num_stages))

    @T.prim_func
    def main(
        A: T.Tensor((m_dim, k_dim), "float16"),
        B: T.Tensor((n_dim, k_dim), "float16"),
        C: T.Tensor((m_dim, n_dim), "float32"),
    ):
        with T.Kernel((m_dim // bm) * (n_dim // bn)) as cid:
            n_blocks = n_dim // bn
            bx = cid // n_blocks
            by = cid % n_blocks
            a_l1 = T.alloc_l1((bm, bk), "float16")
            b_l1 = T.alloc_l1((bn, bk), "float16")
            c_l0c = T.alloc_l0c((bm, bn), "float32")
            for ko in T.Pipelined(k_dim // bk, num_stages=stages):
                T.copy(A[bx * bm : (bx + 1) * bm, ko * bk : (ko + 1) * bk], a_l1)
                T.copy(B[by * bn : (by + 1) * bn, ko * bk : (ko + 1) * bk], b_l1)
                T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(ko == 0))
            T.copy(c_l0c, C[bx * bm : (bx + 1) * bm, by * bn : (by + 1) * bn])

    return main
"""


def test_pillar1_zero_llm_lattice_sweeper_vector_and_cube() -> None:
    """Verify Contract-D exports, Vector sweep, Cube sweep, and patched selector execution."""
    assert "LatticeSweepCandidate" in tp.__all__
    assert "sweep_executable_lattice_zero_llm" in tp.__all__

    target = AscendTarget("ascend.950pr")
    cost_model = HybridAscendCostModel(target)

    vec_results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_VECTOR_SOURCE,
        selector_fn_name="_select_vec_config",
        builder_fn_name="_build_vec_prim_func",
        sample_builder_kwargs={"d_dim": 128},
        dims={"D": 128},
        cost_model=cost_model,
        max_points=12,
        top_k=3,
    )
    assert len(vec_results) >= 1
    assert len(vec_results) <= 3
    assert all(isinstance(item, LatticeSweepCandidate) for item in vec_results)
    assert all("tile_m" in item.knob_overrides for item in vec_results)
    assert all("num_stages" in item.knob_overrides for item in vec_results)
    assert vec_results == sorted(vec_results, key=lambda c: (c.pred_us, c.ub_bytes))

    ns_vec: dict[str, object] = {}
    exec(compile(vec_results[0].patched_source, "<vec_patched>", "exec"), ns_vec)
    sel_vec = ns_vec["_select_vec_config"]
    assert callable(sel_vec)
    patched_vec_cfg = sel_vec(d_dim=128)
    assert patched_vec_cfg["tile_m"] == vec_results[0].knob_overrides["tile_m"]
    assert patched_vec_cfg["num_stages"] == vec_results[0].knob_overrides["num_stages"]

    cube_results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_CUBE_SOURCE,
        selector_fn_name="_select_cube_config",
        builder_fn_name="_build_cube_prim_func",
        sample_builder_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        cost_model=cost_model,
        max_points=24,
        top_k=3,
    )
    assert len(cube_results) >= 1
    assert len(cube_results) <= 3
    assert cube_results == sorted(cube_results, key=lambda c: (c.pred_us, c.ub_bytes))
    for item in cube_results:
        assert item.knob_overrides["bm"] % 16 == 0
        assert item.knob_overrides["bn"] % 16 == 0
        assert item.knob_overrides["bk"] % 16 == 0
        assert item.pred_us > 0.0
        assert item.l1_bytes > 0

    state_grid = tp._generate_lattice_override_grid(
        {"chunk_t": 32, "dv_tile": 32, "num_stages": 1},
        cost_model.facts,
        max_points=24,
    )
    assert {"chunk_t": 16, "dv_tile": 64, "num_stages": 2} in state_grid
    assert all(set(ov.keys()) == {"chunk_t", "dv_tile", "num_stages"} for ov in state_grid)
