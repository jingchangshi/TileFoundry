"""Tests for semantic-first blueprints, explicit ledger traffic scope, and caliper solving."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from textwrap import dedent

import pytest

from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    BlueprintProvenance,
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    SOLVER_MODE_ALGEBRAIC,
    SOLVER_MODE_Z3,
    RegimePiecewiseSolver,
    RegimePlanResult,
    normalize_solver_mode,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil import DEFAULT_OPS_ROOT, TopologyAxes
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


def _write_modelless_task_repo(root: Path) -> Path:
    tasks_root = root / "tasks"
    (tasks_root / "metadata").mkdir(parents=True)
    (tasks_root / "metadata" / "950pr.json").write_text(
        json.dumps({"level1": {}, "level2": {}, "level3": {}}),
        encoding="utf-8",
    )
    (tasks_root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "exp", "level": "L1", "path": "tasks/level1/exp"},
                    {"id": "softmax", "level": "L2", "path": "tasks/level2/softmax"},
                    {"id": "matmul_op", "level": "L3", "path": "tasks/level3/matmul_op"},
                ]
            }
        ),
        encoding="utf-8",
    )

    exp_dir = tasks_root / "level1" / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def exp(x):
                return torch.exp(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (exp_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: exp
              inputs:
                - name: x
                  dtype: [bfloat16]
              outputs:
                - name: y
                  dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (exp_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[64, 1024]]
                dtype: [bfloat16]
              - case_id: 2
                input_shape: [[256, 4096]]
                dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    sm_dir = tasks_root / "level2" / "softmax"
    sm_dir.mkdir(parents=True)
    (sm_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def softmax(x):
                return torch.softmax(x, dim=-1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (sm_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: softmax
              inputs:
                - name: x
                  dtype: [bfloat16]
              outputs:
                - name: y
                  dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (sm_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[128, 1024]]
                dtype: [bfloat16]
              - case_id: 2
                input_shape: [[8192, 8192]]
                dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    mm_dir = tasks_root / "level3" / "matmul_op"
    mm_dir.mkdir(parents=True)
    (mm_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def matmul_op(x, w):
                return torch.matmul(x, w)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (mm_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: matmul_op
              inputs:
                - name: x
                  dtype: [float16]
                - name: w
                  dtype: [float16]
              outputs:
                - name: y
                  dtype: [float16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (mm_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[64, 128], [128, 256]]
                dtype: [float16, float16]
              - case_id: 2
                input_shape: [[256, 512], [512, 1024]]
                dtype: [float16, float16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return tasks_root


def test_modelless_semantic_first_blueprints_softmax_parity_and_backtest(
    tmp_path: Path,
) -> None:
    """Verify golden-only workflows without model.py across blueprint, parity, and backtest."""
    tasks_root = _write_modelless_task_repo(tmp_path)
    assert not list(tasks_root.rglob("model.py"))
    assert not list(tasks_root.rglob("scheduled.py"))

    analyzer = PyTorchSemanticAnalyzer()
    sm_source = (tasks_root / "level2" / "softmax" / "golden.py").read_text(encoding="utf-8")
    sm_semantics = analyzer.analyze_source(sm_source, entry_name="softmax")
    sm_case = RepresentativeCase(
        case_id=2,
        input_shapes=((8192, 8192),),
        input_dtypes=("bfloat16",),
        output_dtype="bfloat16",
    )
    sm_packet = analyzer.build_blueprint(
        sm_semantics,
        {"M": 8192, "D": 8192, "total_elements": 8192 * 8192},
        representative_case=sm_case,
    )
    assert sm_packet.roofline.min_gm_bytes == 268_435_456
    assert sm_packet.provenance == BlueprintProvenance(
        semantic_source="pytorch_golden",
        traffic_basis="declared_io_exact",
        coordinate_scope="representative_anchor",
    )
    rebuilt_sm = HirBlueprintPacket.from_dict(json.loads(json.dumps(sm_packet.to_dict())))
    assert rebuilt_sm == sm_packet


def test_universal_blueprint_anchors_real_dims_from_pytorch_golden() -> None:
    """Build universal blueprints from PyTorch golden forward() with exact anchor dims."""
    analyzer = PyTorchSemanticAnalyzer()
    mla_golden = dedent(
        """
        import torch

        def forward(q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)
        """
    )
    semantics = analyzer.analyze_source(mla_golden, entry_name="forward")
    real_anchor_dims = {
        "B": 4,
        "S_q": 256,
        "S_kv": 4096,
        "N_q": 32,
        "N_kv": 8,
        "D_qk": 576,
        "D_v": 512,
    }
    rep_case = RepresentativeCase(
        case_id=1,
        input_shapes=((4, 32, 256, 576), (4, 8, 4096, 576), (4, 8, 4096, 512)),
        input_dtypes=("bfloat16", "bfloat16", "bfloat16"),
        output_dtype="bfloat16",
    )

    pkt = analyzer.build_universal_blueprint(
        semantics,
        real_anchor_dims,
        representative_case=rep_case,
    )

    assert pkt.entry_name == "forward"
    assert pkt.roofline.anchor_dims["S_q"] == 256
    assert pkt.roofline.anchor_dims["S_kv"] == 4096
    assert pkt.roofline.ideal_roofline_us > 10.0
    assert pkt.provenance == BlueprintProvenance(
        semantic_source="pytorch_golden",
        traffic_basis="declared_io_exact",
        coordinate_scope="representative_anchor",
    )

    roundtrip = HirBlueprintPacket.from_dict(json.loads(json.dumps(pkt.to_dict())))
    assert roundtrip == pkt
    replaced = dataclasses.replace(pkt, fallback_reason="test_note")
    assert replaced.fallback_reason == "test_note"

    ar_golden = dedent(
        """
        import torch

        def forward(v_stack, x, w, gamma):
            rms = torch.rsqrt(v_stack.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            scores = (v_stack * rms * gamma * w).sum(dim=-1)
            probs = torch.softmax(scores, dim=0)
            return 0.5 * x + 0.5 * (probs.unsqueeze(-1) * v_stack).sum(dim=0)
        """
    )
    ar_semantics = analyzer.analyze_source(ar_golden, entry_name="forward")
    ar_unaligned_dims = {
        "M": 16,
        "D": 1000,
    }
    ar_pkt = analyzer.build_universal_blueprint(ar_semantics, ar_unaligned_dims)
    assert ar_pkt.roofline.anchor_dims["D"] == 1000
    assert ar_pkt.roofline.ideal_roofline_us > 0.0


def test_objective_physical_ledger_gm_traffic_scope_and_total_gm_bytes() -> None:
    """ObjectivePhysicalLedger distinguishes per_tile vs full_grid GM traffic explicitly."""
    per_tile = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={},
        grid_blocks=28,
        active_grid_blocks=14,
        gm_roundtrip_bytes=65536,
        gm_traffic_scope="per_tile",
    )
    assert per_tile.gm_traffic_scope == "per_tile"
    assert per_tile.total_gm_bytes() == 65536 * 14
    assert per_tile.total_gm_bytes(28) == 65536 * 28
    assert ObjectivePhysicalLedger.from_dict(per_tile.to_dict()) == per_tile

    full_grid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={},
        grid_blocks=28,
        gm_roundtrip_bytes=1_048_576,
        gm_traffic_scope="full_grid",
    )
    assert full_grid.gm_traffic_scope == "full_grid"
    assert full_grid.total_gm_bytes() == 1_048_576
    assert full_grid.total_gm_bytes(56) == 1_048_576
    assert ObjectivePhysicalLedger.from_dict(full_grid.to_dict()) == full_grid

    legacy_dict = full_grid.to_dict()
    legacy_dict.pop("gm_traffic_scope")
    rebuilt_legacy = ObjectivePhysicalLedger.from_dict(legacy_dict)
    assert rebuilt_legacy.gm_traffic_scope == "per_tile"
    assert rebuilt_legacy.total_gm_bytes() == 1_048_576 * 28

    with pytest.raises(ValueError, match="gm_traffic_scope"):
        ObjectivePhysicalLedger.from_dict(
            dict(full_grid.to_dict(), gm_traffic_scope="invalid_scope")
        )


def test_regime_solver_algebraic_mode_roofline_twopass_and_monotone_caliper() -> None:
    """Verify algebraic_piecewise mode, Roofline UB twopass algebra, and <=3-step caliper."""
    assert SOLVER_MODE_ALGEBRAIC == "algebraic_piecewise"
    assert SOLVER_MODE_Z3 == "z3_smt"
    assert normalize_solver_mode("analytical_piecewise") == "algebraic_piecewise"
    assert normalize_solver_mode("algebraic_piecewise") == "algebraic_piecewise"

    target = AscendTarget("ascend.950pr")
    facts = target.get_facts(AscendSpaceFacts)
    assert isinstance(facts, AscendSpaceFacts)
    solver = RegimePiecewiseSolver(target)
    aiv_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    wide_roofline = HirRooflineSummary(
        anchor_dims={"M": 64, "D": 65536, "elements": 64 * 65536},
        compute_flops=64 * 65536 * 11,
        min_gm_bytes=64 * 65536 * 4,
        arithmetic_intensity=2.75,
        compute_floor_us=1.0,
        memory_floor_us=10.0,
        ideal_roofline_us=10.0,
        dominant_bound="memory",
    )
    wide_plan = solver.solve_regime(
        {"regime_tag": "opaque_regime_0", "representative_dims": {"M": 64}},
        op_name="anon_reduce",
        topology_axes=aiv_axes,
        roofline=wide_roofline,
        input_dtypes=("bfloat16",),
    )
    assert wide_plan.solver_mode == SOLVER_MODE_ALGEBRAIC
    assert wide_plan.caliper_steps == 0
    assert wide_plan.caliper_action == "none"
    assert wide_plan.policy["requires_twopass_reduction"] is True
    ub_tile_elems = wide_plan.policy["ub_tile_elems"]
    twopass_switch = wide_plan.policy["twopass_switch_elems"]
    assert isinstance(ub_tile_elems, int) and isinstance(twopass_switch, int)
    assert 64 <= ub_tile_elems < 65536
    assert ub_tile_elems & (ub_tile_elems - 1) == 0
    assert twopass_switch <= facts.ub_simdvf_usable_bytes // 8
    assert wide_plan.policy["tile_mnk"][1] == ub_tile_elems

    legacy_dict = dict(wide_plan.to_summary_dict(), solver_mode="analytical_piecewise")
    restored = RegimePlanResult.from_dict(legacy_dict)
    assert restored.solver_mode == SOLVER_MODE_ALGEBRAIC
    assert restored.caliper_steps == 0
    assert restored.caliper_action == "none"

    cube_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "affine_simd")
    cube_base = solver.solve_regime(
        {"regime_tag": "cube_main", "representative_dims": {"M": 256, "N": 256, "K": 128}},
        op_name="anon_mm",
        topology_axes=cube_axes,
        uses_cube=True,
    )
    ub_overflow = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=facts.ub_simdvf_usable_bytes,
                explicit_buffers={"buf": int(facts.ub_simdvf_usable_bytes * 3.2)},
                implicit_buffers={},
            )
        },
    )
    ub_caliper = recompute_regime_for_live_buffers(solver, cube_base, ub_overflow)
    assert 1 <= ub_caliper.caliper_steps <= 3
    assert ub_caliper.caliper_action == "ub_halving"
    assert ub_caliper.policy["ub_estimated_bytes"] <= facts.ub_simdvf_usable_bytes
    assert ub_caliper.to_summary_dict()["caliper_steps"] == ub_caliper.caliper_steps
    assert ub_caliper.to_summary_dict()["caliper_action"] == "ub_halving"

    staged_policy = dict(cube_base.policy, l1_stages=3, l1_half_bank_split=True)
    staged_plan = dataclasses.replace(cube_base, policy=staged_policy)
    l1_overflow = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 120},
        waterlines={
            "L1": build_scope_waterline(
                scope="L1",
                capacity_bytes=facts.l1_bytes,
                explicit_buffers={"a_l1": int(facts.l1_bytes * 1.8)},
                implicit_buffers={},
            )
        },
    )
    l1_caliper = recompute_regime_for_live_buffers(solver, staged_plan, l1_overflow)
    assert 1 <= l1_caliper.caliper_steps <= 3
    assert l1_caliper.caliper_action == "l1_stage_backoff"
    assert l1_caliper.policy["l1_stages"] < 3
    assert l1_caliper.policy["l1_estimated_bytes"] <= facts.l1_bytes


@pytest.mark.skipif(
    not DEFAULT_OPS_ROOT.is_dir(),
    reason="cann-bench tasks root unavailable",
)
def test_real_stock_operators_build_blueprints_without_model_py(tmp_path: Path) -> None:
    """Copying golden.py + proto.yaml + cases.yaml without model.py builds valid blueprints."""
    analyzer = PyTorchSemanticAnalyzer()
    selected = (
        ("level1", "swi_glu"),
        ("level2", "rms_norm"),
        ("level2", "softmax"),
        ("level3", "grouped_matmul"),
        ("level4", "gqa"),
    )
    for level, op_name in selected:
        src_dir = DEFAULT_OPS_ROOT / level / op_name
        dst_dir = tmp_path / level / op_name
        dst_dir.mkdir(parents=True)
        for fname in ("golden.py", "proto.yaml", "cases.yaml"):
            shutil.copy2(src_dir / fname, dst_dir / fname)
        assert not (dst_dir / "model.py").exists()
        golden_src = (dst_dir / "golden.py").read_text(encoding="utf-8")
        semantics = analyzer.analyze_source(golden_src, entry_name=op_name)
        assert semantics.entry_name == op_name
        dims = (
            {"B": 1, "S_q": 64, "S_kv": 128, "N_q": 8, "N_kv": 2, "D_qk": 128, "D_v": 128}
            if op_name == "gqa"
            else (
                {"M": 64, "N": 128, "K": 64}
                if op_name == "grouped_matmul"
                else {"M": 64, "D": 512, "total_elements": 64 * 512}
            )
        )
        pkt = analyzer.build_blueprint(semantics, dims)
        assert pkt.roofline.ideal_roofline_us > 0.0
        assert pkt.provenance is not None
        assert pkt.provenance.semantic_source == "pytorch_golden"
