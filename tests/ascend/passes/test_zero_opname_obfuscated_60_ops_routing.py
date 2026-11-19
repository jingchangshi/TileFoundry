"""Verify 100% op-name invariance across SIL, router, and regime solver for 60 operators."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.router import capability_gate
from tilefoundry_ascend.agent.router.capability_gate import route_from_op_spec
from tilefoundry_ascend.ascend.schedule import pipeline_solver
from tilefoundry_ascend.ascend.schedule.pipeline_solver import read_from_sil
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil.op_analyzer import (
    OpAdvisoryExtractor,
    synthesize_default_policy,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANNBENCH_TASKS = _REPO_ROOT.parent / "cann-bench" / "tasks"


def _scrub_proto_names(proto: dict[str, object], alias: str) -> dict[str, object]:
    """Return a deep copy of proto with all identifier and name fields obfuscated."""
    copied = copy.deepcopy(proto)
    for key in ("id", "name", "op_name", "cannbot_task", "schema"):
        if key in copied:
            copied[key] = alias
    op_block = copied.get("operator")
    if isinstance(op_block, dict):
        for key in ("id", "name", "schema"):
            if key in op_block:
                op_block[key] = alias
    return copied


@pytest.mark.skipif(not _CANNBENCH_TASKS.is_dir(), reason="cann-bench tasks missing")
def test_obfuscated_60_ops_zero_opname_invariance(tmp_path: Path) -> None:
    """Scrubbing all 60 CANNBench names preserves axes, contraction, routing, and regime plan."""
    assert not hasattr(capability_gate, "_CANNBENCH_OP_ROUTING")
    assert not hasattr(pipeline_solver, "fallback_from_sil")
    assert not hasattr(pipeline_solver, "_FallbackRooflineSummary")
    assert read_from_sil(tmp_path / "nonexistent_sil.json") is None

    proto_files = sorted(_CANNBENCH_TASKS.glob("level*/*/proto.yaml"))
    assert len(proto_files) == 60

    orig_extractor = OpAdvisoryExtractor(_CANNBENCH_TASKS)
    orig_perf = orig_extractor.perf_metadata
    anon_root = tmp_path / "anon_tasks"
    anon_root.mkdir(parents=True, exist_ok=True)
    (anon_root / "metadata").mkdir(parents=True, exist_ok=True)
    anon_perf: dict[str, dict[str, object]] = {}
    anon_map: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
    manifest_ops: list[dict[str, object]] = []

    for idx, pf in enumerate(proto_files):
        op_dir = pf.parent
        level_dir = op_dir.parent.name
        lvl_num = level_dir.removeprefix("level")
        orig_name = op_dir.name
        alias = f"anon_op_{idx:02d}"
        dest_dir = anon_root / level_dir / alias
        dest_dir.mkdir(parents=True, exist_ok=True)

        orig_proto = yaml.safe_load(pf.read_text(encoding="utf-8"))
        obf_proto = _scrub_proto_names(orig_proto, alias)
        (dest_dir / "proto.yaml").write_text(yaml.safe_dump(obf_proto), encoding="utf-8")

        cases_file = op_dir / "cases.yaml"
        if cases_file.is_file():
            shutil.copy2(cases_file, dest_dir / "cases.yaml")

        orig_op_perf = orig_perf.get(level_dir, {}).get(orig_name)
        if isinstance(orig_op_perf, dict):
            anon_perf.setdefault(level_dir, {})[alias] = copy.deepcopy(orig_op_perf)

        manifest_ops.append(
            {
                "id": alias,
                "level": f"L{lvl_num}",
                "path": f"tasks/{level_dir}/{alias}",
            }
        )
        anon_map.append((level_dir, orig_name, orig_proto, obf_proto))

    (anon_root / "metadata" / "950pr.json").write_text(
        json.dumps(anon_perf),
        encoding="utf-8",
    )

    (anon_root / "suite_manifest.json").write_text(
        json.dumps({"operators": manifest_ops}),
        encoding="utf-8",
    )
    anon_extractor = OpAdvisoryExtractor(anon_root)

    for idx, (level_dir, orig_name, orig_proto, obf_proto) in enumerate(anon_map):
        alias = f"anon_op_{idx:02d}"
        orig_adv = orig_extractor.extract(orig_name, level=level_dir)
        anon_adv = anon_extractor.extract(alias, level=level_dir)

        assert anon_adv.topology_axes == orig_adv.topology_axes, orig_name
        orig_sf = {k: v for k, v in orig_adv.structural_features.items() if k != "schema"}
        anon_sf = {k: v for k, v in anon_adv.structural_features.items() if k != "schema"}
        assert anon_sf == orig_sf, orig_name
        assert anon_adv.roofline == orig_adv.roofline, orig_name
        assert [r.representative_dims for r in anon_adv.regimes] == [
            r.representative_dims for r in orig_adv.regimes
        ], orig_name

        orig_pol = synthesize_default_policy(orig_adv)
        anon_pol = synthesize_default_policy(anon_adv)
        assert [
            orig_pol.policies[t].l1_slot_aliasing for t in orig_pol.policies
        ] == [
            anon_pol.policies[t].l1_slot_aliasing for t in anon_pol.policies
        ], orig_name

        orig_route = route_from_op_spec(orig_proto)
        anon_route = route_from_op_spec(obf_proto)
        assert anon_route.topology_axes == orig_route.topology_axes, orig_name
        assert anon_route.track == orig_route.track, orig_name
        assert anon_route.recommended_backend_tier == orig_route.recommended_backend_tier, orig_name
        assert anon_route.features == orig_route.features, orig_name


def test_regime_solver_zero_opname_and_strict_sil_contract() -> None:
    """Verify RegimePiecewiseSolver relies strictly on structural axes and flags."""
    solver = RegimePiecewiseSolver()
    axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    regime = {
        "regime_tag": "main",
        "representative_dims": {"M": 128, "N": 128, "K": 64},
        "has_unaligned_tail": False,
    }
    plan_mla = solver.solve_regime(regime, op_name="mla", topology_axes=axes)
    plan_vec = solver.solve_regime(regime, op_name="unrelated_vec_op", topology_axes=axes)
    assert plan_mla.policy == plan_vec.policy
    assert plan_mla.policy["l1_slot_aliasing"] == {}

    plan_alias = solver.solve_regime(
        regime,
        op_name="unrelated_vec_op",
        topology_axes=axes,
        has_dead_l1_slice=True,
    )
    assert plan_alias.policy["l1_slot_aliasing"] == {"reused_stage": "dead_l1_slice"}
