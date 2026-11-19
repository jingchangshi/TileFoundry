"""Tests for dialect-neutral HIR blueprint compilation and physical ledger telemetry."""

from __future__ import annotations

import json
import pathlib

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.schedule.finalize import finalize
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
    format_blueprint_delta_section,
)
from tilefoundry_ascend.ascend import (
    AscendSpaceFacts,
    AscendTarget,
    HirBlueprintPacket,
    HirRooflineSummary,
    compile_hir_blueprint,
)
from tilefoundry_ascend.ascend.passes import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.schedule import inspect_tir_envelope

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
AR_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual"
MLA_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "mla"


def test_ascend_space_facts_exposes_core_counts() -> None:
    """Verify AscendSpaceFacts carries AIC and AIV core counts from the device model."""
    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    assert spaces.num_aic_cores == 28
    assert spaces.num_aiv_cores == 56


def test_compile_hir_blueprint_mla_without_virtual_tir_leakage() -> None:
    """Verify MLA HIR blueprint emits Roofline, Z3 summary, and zero virtual TIR ops."""
    authored_src = (MLA_DIR / "model.py").read_text(encoding="utf-8")
    scheduled_src = (MLA_DIR / "scheduled.py").read_text(encoding="utf-8")

    packet = compile_hir_blueprint(
        authored_src,
        scheduled_hir_source=scheduled_src,
        anchor_dims={"B": 1, "S": 16, "S_kv": 16, "N_q": 4, "N_kv": 1},
    )
    assert isinstance(packet, HirBlueprintPacket)
    assert packet.entry_name == "forward"
    assert packet.lowering_diagnostics == ()
    assert not hasattr(packet, "candidate_ops")
    assert not hasattr(packet, "tir_script")
    assert packet.roofline.ideal_roofline_us > 0.0
    assert packet.roofline.compute_flops > 0
    assert packet.roofline.min_gm_bytes > 0

    loop_summary = next(iter(packet.z3_pipeline_summary.values()))
    assert isinstance(loop_summary, dict)
    assert loop_summary["ii"] == 32
    assert loop_summary["stages"]
    assert loop_summary["phases"]
    assert loop_summary["versions"]

    ledger = packet.reference_ledger
    assert ledger is not None
    assert ledger.dominant_contig_bytes == 128
    assert ledger.effective_cube_m == 64
    assert ledger.has_online_rescale_loop is True
    assert ledger.in_loop_ub_reload_stores >= 3

    rebuilt = HirBlueprintPacket.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert rebuilt == packet
    prompt_text = packet.format_for_prompt()
    assert "Roofline[" in prompt_text
    assert "Z3 Pipelines:" in prompt_text
    assert "T.ascend_mmad" not in prompt_text

    constraint_block = build_progressive_llm_constraints(
        escalation_tier="tilelang_dsl",
        track="manual_ssbuf",
        enable_track_router=True,
        blueprint=packet,
        current_ledger=ledger,
    )
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in constraint_block
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in constraint_block
    assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in constraint_block
    assert "Routed Execution Track: manual_ssbuf" in constraint_block
    assert "T.gemm(" in constraint_block
    assert "T.gemm_v0" in constraint_block
    assert "Strictly Forbidden Symbols" in constraint_block
    assert "T.ascend_mmad" not in constraint_block


def test_compile_hir_blueprint_attention_residual_scheduled_and_unscheduled() -> None:
    """Verify AttentionResidual scheduled blueprint and graceful unscheduled Roofline."""
    authored_src = (AR_DIR / "model.py").read_text(encoding="utf-8")
    scheduled_src = (AR_DIR / "scheduled.py").read_text(encoding="utf-8")

    sched_packet = compile_hir_blueprint(
        authored_src,
        scheduled_hir_source=scheduled_src,
        anchor_dims={"N": 4, "B": 1, "S": 8, "D": 64},
    )
    assert sched_packet.lowering_diagnostics == ()
    assert sched_packet.roofline.ideal_roofline_us > 0.0
    assert sched_packet.roofline.anchor_dims == {"N": 4, "B": 1, "S": 8, "D": 64}
    assert sched_packet.reference_ledger is not None
    assert sched_packet.reference_ledger.dominant_contig_bytes == 128
    assert sched_packet.reference_ledger.effective_cube_m == 0
    assert sched_packet.reference_ledger.has_online_rescale_loop is True
    assert sched_packet.reference_ledger.in_loop_ub_reload_stores == 3

    unsched_packet = compile_hir_blueprint(
        authored_src,
        anchor_dims={"N": 4, "B": 2, "S": 8, "D": 64},
    )
    assert unsched_packet.reference_ledger is None
    assert unsched_packet.lowering_diagnostics
    assert unsched_packet.roofline.compute_flops > sched_packet.roofline.compute_flops
    assert unsched_packet.roofline.min_gm_bytes > sched_packet.roofline.min_gm_bytes
    assert unsched_packet.roofline.ideal_roofline_us > 0.0


def test_compile_hir_blueprint_with_dimvars() -> None:
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
    packet = compile_hir_blueprint(
        dimvar_source,
        anchor_dims={"M": 16, "N": 64, "UNUSED": 99},
    )
    assert isinstance(packet.roofline, HirRooflineSummary)
    assert packet.roofline.compute_flops == 2 * 16 * 64
    assert packet.roofline.min_gm_bytes == 3 * 16 * 64 * 4
    assert packet.roofline.ideal_roofline_us > 0.0
    assert packet.lowering_diagnostics


def test_np1_tir_envelope_ledger_telemetry_and_roundtrip(tmp_path: pathlib.Path) -> None:
    """Verify ObjectivePhysicalLedger fields populate on canonical TIR and roundtrip."""
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
        "op=T.ascend_vunary(kind='exp'))\n"
        "                    sr = tf.schedule((ar, er, tf.empty(Tensor[(32, 32), 'f32', 'rmem'])), "
        "op=T.ascend_vbinary(kind='add'))\n"
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


def test_hir_blueprint_fallback_provenance_and_delta_gating() -> None:
    """Verify D=1000 fallback sets shape_exact_match=False and suppresses raw byte deltas."""
    authored_src = (AR_DIR / "model.py").read_text(encoding="utf-8")
    scheduled_src = (AR_DIR / "scheduled.py").read_text(encoding="utf-8")

    req_dims = {"N": 8, "B": 1, "S": 4, "D": 1000}
    pkt = compile_hir_blueprint(
        authored_src,
        scheduled_hir_source=scheduled_src,
        anchor_dims=req_dims,
    )
    assert pkt.shape_exact_match is False
    assert pkt.requested_anchor_dims == req_dims
    assert pkt.resolved_hir_dims == {"N": 4, "B": 1, "S": 4, "D": 64}
    assert pkt.fallback_reason is not None and "ValueError" in pkt.fallback_reason
    assert pkt.reference_ledger is not None

    delta_text = format_blueprint_delta_section(pkt, current_ledger=pkt.reference_ledger)
    assert "[Structural Reference Only:" in delta_text
    assert "scope_utilization[" in delta_text
    assert "ub_peak_bytes:" not in delta_text


def test_dtype_aware_cube_floor_precision_scaling() -> None:
    """Verify dav-3510 Cube Roofline scales FP8 (2x), MXFP4 (4x), and FP32 (1/16x) vs BF16."""
    target = AscendTarget("ascend.950pr")

    def _make_matmul_src(dtype_name: str, out_dtype: str) -> str:
        return (
            "from __future__ import annotations\n"
            "from tilefoundry.dsl import *\n"
            "from tilefoundry_ascend import AscendTarget\n"
            "TOPOS = (Topology('block', 1), Topology('aic', 1), "
            "Topology('aiv', 2), Topology('lane', 64))\n"
            "@module(entry='mm', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
            "class MmMod:\n"
            "    @func\n"
            f"    def mm(a: Tensor[(64, 128), '{dtype_name}'], "
            f"b: Tensor[(128, 64), '{dtype_name}']) -> Tensor[(64, 64), '{out_dtype}']:\n"
            f"        return tf.matmul(a, b, out_dtype='{out_dtype}')\n"
        )

    bf16_rf = compile_hir_blueprint(_make_matmul_src("bf16", "f32"), target=target).roofline
    fp8_rf = compile_hir_blueprint(_make_matmul_src("fp8e4m3", "f32"), target=target).roofline
    fp4_rf = compile_hir_blueprint(_make_matmul_src("f4e2m1", "f32"), target=target).roofline
    fp32_rf = compile_hir_blueprint(_make_matmul_src("f32", "f32"), target=target).roofline

    assert bf16_rf.cube_floor_us > 0.0
    assert abs(fp8_rf.cube_floor_us * 2.0 - bf16_rf.cube_floor_us) < 1e-12
    assert abs(fp4_rf.cube_floor_us * 4.0 - bf16_rf.cube_floor_us) < 1e-12
    assert abs(fp32_rf.cube_floor_us - bf16_rf.cube_floor_us * 16.0) < 1e-12
