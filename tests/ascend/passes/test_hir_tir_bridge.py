"""Tests for HIR-to-TIR navigation packet bridge and NP1 physical ledger telemetry."""

from __future__ import annotations

import json
import pathlib

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.schedule.finalize import finalize
from tilefoundry_ascend.ascend import (
    AscendSpaceFacts,
    AscendTarget,
    HirCandidateOpEntry,
    HirRooflineSummary,
    HirTirNavigationPacket,
    compile_hir_navigation_packet,
)
from tilefoundry_ascend.ascend.passes import ObjectivePhysicalLedger, build_objective_physical_ledger
from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc, inspect_tir_envelope

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
AR_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual"
MLA_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "mla"


def test_ascend_space_facts_exposes_core_counts() -> None:
    """Verify AscendSpaceFacts carries AIC and AIV core counts from the device model."""
    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    assert spaces.num_aic_cores == 28
    assert spaces.num_aiv_cores == 56


def test_compile_hir_navigation_packet_mla() -> None:
    """Verify MLA model + scheduled HIR produces valid roofline, Z3 II=10, and TIR ledger."""
    authored_src = (MLA_DIR / "model.py").read_text(encoding="utf-8")
    scheduled_src = (MLA_DIR / "scheduled.py").read_text(encoding="utf-8")

    packet = compile_hir_navigation_packet(
        authored_src,
        scheduled_hir_source=scheduled_src,
        anchor_dims={"B": 1, "S": 16, "S_kv": 16, "N_q": 4, "N_kv": 1},
    )
    assert isinstance(packet, HirTirNavigationPacket)
    assert packet.entry_name == "forward"
    assert packet.lowering_diagnostics == ()
    assert packet.tir_script is not None and "tf.pipelined(" in packet.tir_script
    assert "fixed_ii=10" in packet.tir_script
    assert packet.roofline.ideal_roofline_us > 0.0
    assert packet.roofline.compute_flops > 0
    assert packet.roofline.min_gm_bytes > 0
    assert packet.candidate_ops
    assert any(entry.hir_op_name == "tf.matmul" for entry in packet.candidate_ops)

    loop_summary = next(iter(packet.z3_pipeline_summary.values()))
    assert isinstance(loop_summary, dict)
    assert loop_summary["ii"] == 10
    assert loop_summary["stages"]
    assert loop_summary["phases"]
    assert loop_summary["versions"]

    ledger = packet.tir_ledger
    assert ledger is not None
    assert ledger.dominant_contig_bytes == 128
    assert ledger.effective_cube_m == 64
    assert ledger.has_online_rescale_loop is True
    assert ledger.in_loop_ub_reload_stores >= 3

    rebuilt = HirTirNavigationPacket.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert rebuilt == packet
    prompt_text = packet.format_for_prompt()
    assert "Roofline[" in prompt_text
    assert "Candidate Ops" in prompt_text
    assert "Z3 Pipelines:" in prompt_text


def test_compile_hir_navigation_packet_attention_residual_scheduled_and_unscheduled() -> None:
    """Verify AttentionResidual scheduled lowering and graceful unscheduled fallback."""
    authored_src = (AR_DIR / "model.py").read_text(encoding="utf-8")
    scheduled_src = (AR_DIR / "scheduled.py").read_text(encoding="utf-8")

    sched_packet = compile_hir_navigation_packet(
        authored_src,
        scheduled_hir_source=scheduled_src,
        anchor_dims={"N": 4, "B": 1, "S": 8, "D": 64},
    )
    assert sched_packet.lowering_diagnostics == ()
    assert sched_packet.tir_script is not None and len(sched_packet.tir_script) > 0
    assert sched_packet.roofline.ideal_roofline_us > 0.0
    assert sched_packet.roofline.anchor_dims == {"N": 4, "B": 1, "S": 8, "D": 64}
    assert sched_packet.tir_ledger is not None
    assert sched_packet.tir_ledger.dominant_contig_bytes == 128
    assert sched_packet.tir_ledger.effective_cube_m == 0
    assert sched_packet.tir_ledger.has_online_rescale_loop is True
    assert sched_packet.tir_ledger.in_loop_ub_reload_stores == 3

    unsched_packet = compile_hir_navigation_packet(
        authored_src,
        anchor_dims={"N": 4, "B": 2, "S": 8, "D": 64},
    )
    assert unsched_packet.tir_script is None
    assert unsched_packet.tir_ledger is None
    assert unsched_packet.lowering_diagnostics
    assert unsched_packet.roofline.compute_flops > sched_packet.roofline.compute_flops
    assert unsched_packet.roofline.min_gm_bytes > sched_packet.roofline.min_gm_bytes
    assert unsched_packet.roofline.ideal_roofline_us > 0.0
    assert unsched_packet.candidate_ops
    assert any(
        isinstance(entry, HirCandidateOpEntry) and entry.legal_ascend_ops
        for entry in unsched_packet.candidate_ops
    )


def test_compile_hir_navigation_packet_with_dimvars() -> None:
    """Verify DimVar HIR modules specialize cleanly without SpecializationError."""
    dimvar_source = (
        "from __future__ import annotations\n"
        "from tilefoundry.dsl import *\n"
        "from tilefoundry_ascend import AscendTarget\n"
        "TOPOS = (Topology('block', 1), Topology('aic', 1), "
        "Topology('aiv', 2), Topology('lane', 64))\n"
        "M = DimVar('M', lo=1, hi=1024)\n"
        "N = DimVar('N', lo=1, hi=1024)\n"
        "@module(entry='add_scale', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
        "class DimMod:\n"
        "    @func\n"
        "    def add_scale(a: Tensor[(M, N), 'f32'], b: Tensor[(M, N), 'f32']) "
        "-> Tensor[(M, N), 'f32']:\n"
        "        return tf.cast(a + b, dtype='f32')\n"
    )
    packet = compile_hir_navigation_packet(
        dimvar_source,
        anchor_dims={"M": 16, "N": 64, "UNUSED": 99},
    )
    assert isinstance(packet.roofline, HirRooflineSummary)
    assert packet.roofline.compute_flops == 2 * 16 * 64
    assert packet.roofline.min_gm_bytes == 3 * 16 * 64 * 4
    assert packet.roofline.ideal_roofline_us > 0.0
    assert packet.candidate_ops
    assert packet.lowering_diagnostics


def test_np1_tilelang_primfunc_ledger_telemetry() -> None:
    """Verify all 10 NP1 ObjectivePhysicalLedger fields populate on TileLang PrimFuncs."""
    import tilelang.ascend.language as T

    @T.prim_func
    def rich_vector_kernel(
        X: T.Tensor((28, 64, 128), "float16"),
        Idx: T.Tensor((28, 64), "int32"),
        Meta: T.Tensor((64, 1), "float32"),
        Flag: T.Tensor((1,), "int32"),
        TmpGM: T.Tensor((16, 64), "float32"),
        Out: T.Tensor((28, 64, 128), "float16"),
    ):
        with T.Kernel(28) as bx:
            ub = T.alloc_shared((4, 64), "float32")
            acc = T.alloc_shared((4, 64), "float32")
            T.copy(X[bx, 0:4, 0:64], ub)
            T.copy(ub, TmpGM[0:4, 0:64])
            T.copy(TmpGM[0:4, 0:64], ub)
            for _it in T.serial(4):
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    v_acc = T.simd.vld(acc[0, 0], dist="NORM_B32")
                    v_exp = T.simd.vexp(v_acc, mask=m32)
                    T.simd.vsts(acc[0, 0], v_exp, mask=m32, dist="NORM_B32")
                Flag[0] = Flag[0] + 1
            acc[0, 0] = Meta[Idx[bx, 0], 0]
            T.copy(acc, Out[bx, 0:4, 0:64])

    rich_with_attr = rich_vector_kernel.with_attr(
        "tir.cce_flags", ["--cce-disable-vf-stack-reserved-ubuf"]
    )
    res_vec = inspect_tilelang_primfunc(
        rich_with_attr,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    led = res_vec.ledger
    assert led.grid_blocks == 28
    assert led.wave_occupancy == round(28 / 56.0, 4)
    assert led.gm_roundtrip_bytes == 16 * 64 * 4 + 1 * 4
    assert led.dominant_contig_bytes == 128 * 2
    assert led.effective_cube_m == 0
    assert led.has_indirect_gather is True
    assert led.has_lowbit_quant is False
    assert led.has_online_rescale_loop is True
    assert led.in_loop_ub_reload_stores == 1
    assert led.uses_unreserved_ubuf is True

    @T.prim_func
    def quant_cube_kernel(
        A: T.Tensor((64, 128), "int8"),
        B: T.Tensor((128, 64), "int8"),
        C: T.Tensor((64, 64), "int32"),
    ):
        with T.MixedKernel(28, sids=2) as (_bx, _sid):
            a_l1 = T.alloc_l1((64, 128), "int8")
            b_l1 = T.alloc_l1((128, 64), "int8")
            c_l0c = T.alloc_l0c((64, 64), "int32")
            T.copy(A[0:64, 0:128], a_l1)
            T.copy(B[0:128, 0:64], b_l1)
            T.gemm(a_l1, b_l1, c_l0c, clear_accum=True)
            T.copy(c_l0c, C[0:64, 0:64])

    res_cube = inspect_tilelang_primfunc(
        quant_cube_kernel,
        pass_configs={"tl.enable_auto_schedule": False},
        unreserved_ubuf=True,
    )
    c_led = res_cube.ledger
    assert c_led.grid_blocks == 28
    assert c_led.wave_occupancy == 1.0
    assert c_led.effective_cube_m == 64
    assert c_led.has_lowbit_quant is True
    assert c_led.dominant_contig_bytes == 64 * 4
    assert c_led.uses_unreserved_ubuf is True

    roundtrip = ObjectivePhysicalLedger.from_dict(json.loads(c_led.to_json()))
    assert roundtrip == c_led


def test_np1_tir_envelope_ledger_telemetry_and_roundtrip(tmp_path: pathlib.Path) -> None:
    """Verify all new ObjectivePhysicalLedger fields populate on TileFoundry TIR and roundtrip."""
    source = (
        "from __future__ import annotations\n"
        "from tilefoundry.dsl import *\n"
        "from tilefoundry_ascend import AscendTarget\n"
        "TOPOS = (Topology('block', 14), Topology('aic', 1), "
        "Topology('aiv', 2), Topology('lane', 64))\n"
        "@module(entry='tir_probe', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
        "class TirProbe:\n"
        "    @func\n"
        "    def tir_probe(\n"
        "        a: Tensor[(14, 32, 64), 'fp8e4m3'],\n"
        "        b: Tensor[(14, 64, 32), 'fp8e4m3'],\n"
        "        meta: Tensor[(32, 1), 'f32'],\n"
        "        scalar_flag: Tensor[(1,), 'i32'],\n"
        "        mid_gm: Tensor[(32, 32), 'f32'],\n"
        "    ) -> Tensor[(32, 32), 'f32']:\n"
        "        with Mesh(('block', 'aiv'), (14, 2), names=('b', 'v')) as _bv:\n"
        "            acc = tf.zeros(Tensor[(32, 32), 'f32', 'smem'])\n"
        "            for _k in range(2):\n"
        "                with Mesh(('aic',), (1,), names=('c',)) as _c:\n"
        "                    a_l1 = tf.schedule((a[0, :, :], tf.empty(Tensor[(32, 64), 'fp8e4m3', 'tmem'])), "
        "op=T.ascend_gm_to_l1())\n"
        "                    b_l1 = tf.schedule((b[0, :, :], tf.empty(Tensor[(64, 32), 'fp8e4m3', 'tmem'])), "
        "op=T.ascend_gm_to_l1())\n"
        "                    a_l0 = tf.schedule((a_l1, tf.empty(Tensor[(32, 64), 'fp8e4m3', 'tmem'])), "
        "op=T.ascend_l1_to_l0a())\n"
        "                    b_l0 = tf.schedule((b_l1, tf.empty(Tensor[(64, 32), 'fp8e4m3', 'tmem'])), "
        "op=T.ascend_l1_to_l0b())\n"
        "                    c_t = tf.zeros(Tensor[(32, 32), 'f32', 'tmem'])\n"
        "                    c_m = tf.schedule((c_t, a_l0, b_l0), op=T.ascend_mmad(init_zero=True))\n"
        "                    c_ub = tf.schedule((c_m, tf.empty(Tensor[(32, 32), 'f32', 'smem'])), "
        "op=T.ascend_dual_copy())\n"
        "                with T.SimdVF():\n"
        "                    ar = tf.schedule((acc, tf.empty(Tensor[(32, 32), 'f32', 'rmem'])), "
        "op=T.ascend_ub_to_reg())\n"
        "                    cr = tf.schedule((c_ub, tf.empty(Tensor[(32, 32), 'f32', 'rmem'])), "
        "op=T.ascend_ub_to_reg())\n"
        "                    er = tf.schedule((cr, tf.empty(Tensor[(32, 32), 'f32', 'rmem'])), "
        "op=T.ascend_vexp())\n"
        "                    sr = tf.schedule((ar, er, tf.empty(Tensor[(32, 32), 'f32', 'rmem'])), "
        "op=T.ascend_vadd())\n"
        "                    acc = tf.schedule((sr, tf.empty(Tensor[(32, 32), 'f32', 'smem'])), "
        "op=T.ascend_reg_to_ub())\n"
        "            mid_w = tf.schedule((acc, mid_gm), op=T.ascend_ub_to_gm())\n"
        "            mid_r = tf.schedule((mid_w, tf.empty(Tensor[(32, 32), 'f32', 'smem'])), "
        "op=T.ascend_gm_to_ub())\n"
        "            return tf.schedule((mid_r, tf.empty(Tensor[(32, 32), 'f32', 'gmem'])), "
        "op=T.ascend_ub_to_gm())\n"
    )
    mod_path = tmp_path / "tir_probe.py"
    mod_path.write_text(source, encoding="utf-8")
    mod = load_authored_ir(f"{mod_path}:TirProbe")
    authored = mod.entry_function()
    c_mod, c_fn = check_program(mod, authored)
    lowered = finalize(analyze(c_mod, c_fn, analysis="memory"), authored=authored)
    tir_res = inspect_tir_envelope(lowered.entry_function(), unreserved_ubuf=True)
    led = tir_res.ledger

    assert led.grid_blocks == 14
    assert led.wave_occupancy == 0.5
    assert led.gm_roundtrip_bytes == 32 * 32 * 4
    assert led.dominant_contig_bytes == 64
    assert led.effective_cube_m == 32
    assert led.has_lowbit_quant is True
    assert led.has_online_rescale_loop is True
    assert led.in_loop_ub_reload_stores == 1
    assert led.uses_unreserved_ubuf is True

    rebuilt = ObjectivePhysicalLedger.from_dict(led.to_dict())
    assert rebuilt == led

    default_led = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={},
        waterlines={},
    )
    assert ObjectivePhysicalLedger.from_dict(default_led.to_dict()) == default_led
