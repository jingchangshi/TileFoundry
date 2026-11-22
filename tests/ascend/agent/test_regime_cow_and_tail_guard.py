"""Tests for helper closure extraction, CoW helper namespacing, and dual-layer tail guard."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
from pathlib import Path

from tilefoundry_ascend.agent.campaign import (
    AnchorCaseSpec,
    CampaignConfig,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
    extract_regime_snippet,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.orchestrator import MockLLMBackend, StageAResult, StageBResult
from tilefoundry_ascend.agent.router import extract_structural_features, route_execution_track
from tilefoundry_ascend.ascend.cost_model import CostFeatures
from tilefoundry_ascend.ascend.passes import (
    HirBlueprintPacket,
    HirRooflineSummary,
    build_objective_physical_ledger,
)

_SHARED_HELPER_MODULE = '''import tilelang.ascend.language as T

_KERNEL_CACHE = {"kept": True}

def _stage_leaf(val: int) -> int:
    return val + 1

def _emit_pass1_compute(scale: int) -> int:
    return _stage_leaf(scale) * 2

def _exclusive_a_helper(x: int) -> int:
    return x + 3

def _select_regime_a_config(d_dim: int = 128) -> dict[str, object]:
    return {"tile": d_dim}

def _build_regime_a(tile_m: int = 1) -> object:
    step = _emit_pass1_compute(tile_m) + _exclusive_a_helper(1)
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main, step

def _select_regime_b_config(d_dim: int = 64) -> dict[str, object]:
    return {"tile": d_dim}

def _build_regime_b(tile_m: int = 1) -> object:
    step = _emit_pass1_compute(tile_m)
    @T.prim_func
    def main(X: T.Tensor((64, 64), "bfloat16"), Y: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], b_ub)
            T.copy(b_ub, Y[0:64, 0:64])
    return main, step

def entrypoint(x: int) -> object:
    return _KERNEL_CACHE, _build_regime_a(), _build_regime_b(), x
'''


def _top_level_ast_sha256(module_source: str, symbol_name: str) -> str:
    tree = ast.parse(module_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
            return hashlib.sha256(ast.dump(node).encode("utf-8")).hexdigest()
    raise AssertionError(f"Missing symbol {symbol_name}")


def _regime_specs() -> tuple[RegimeTargetSpec, RegimeTargetSpec]:
    return (
        RegimeTargetSpec(
            regime_tag="regime_a",
            builder_candidates=("_build_regime_a",),
            sample_builder_kwargs={"tile_m": 1},
            config_selector_fn="_select_regime_a_config",
            editable_symbols=("_select_regime_a_config", "_build_regime_a"),
            allowed_helper_prefix="_regime_a_",
        ),
        RegimeTargetSpec(
            regime_tag="regime_b",
            builder_candidates=("_build_regime_b",),
            sample_builder_kwargs={"tile_m": 1},
            config_selector_fn="_select_regime_b_config",
            editable_symbols=("_select_regime_b_config", "_build_regime_b"),
            allowed_helper_prefix="_regime_b_",
        ),
    )


def test_extract_closure_cow_namespacing_sibling_invariance_and_pruning() -> None:
    """Verify closure extraction, CoW helper namespacing, sibling AST invariance, and pruning."""
    spec_a, spec_b = _regime_specs()
    specs = (spec_a, spec_b)
    primary, snippet = extract_regime_snippet(
        _SHARED_HELPER_MODULE,
        spec_a,
        operator_entrypoint="entrypoint",
        sibling_specs=specs,
    )
    assert primary == "_build_regime_a"
    assert "def _stage_leaf(" in snippet
    assert "def _emit_pass1_compute(" in snippet
    assert "def _exclusive_a_helper(" in snippet
    assert "def _build_regime_a(" in snippet
    assert "_KERNEL_CACHE" not in snippet
    assert "def entrypoint(" not in snippet
    assert "def _build_regime_b(" not in snippet

    sib_hashes_before = {
        sym: _top_level_ast_sha256(_SHARED_HELPER_MODULE, sym)
        for sym in ("_stage_leaf", "_emit_pass1_compute", "_select_regime_b_config", "_build_regime_b", "entrypoint")
    }

    co_edited = snippet.replace(
        "return _stage_leaf(scale) * 2",
        "return _stage_leaf(scale) * 4",
    ).replace(
        "return x + 3",
        "return x + 9",
    )
    patched = patch_regime_into_module(
        _SHARED_HELPER_MODULE,
        spec_a,
        co_edited,
        operator_entrypoint="entrypoint",
        sibling_specs=specs,
    )

    assert "def _regime_a_emit_pass1_compute(" in patched
    assert "return _stage_leaf(scale) * 4" in patched
    assert "return _stage_leaf(scale) * 2" in patched
    assert "step = _regime_a_emit_pass1_compute(tile_m) + _exclusive_a_helper(1)" in patched
    assert "return x + 9" in patched
    assert "return x + 3" not in patched

    for sym, expected_hash in sib_hashes_before.items():
        assert _top_level_ast_sha256(patched, sym) == expected_hash

    ns: dict[str, object] = {}
    exec_src = patched.replace("import tilelang.ascend.language as T", "class _T:\n    prim_func = staticmethod(lambda fn: fn)\nT = _T")
    exec(compile(exec_src, "<cow_test>", "exec"), ns)
    res_a = ns["_build_regime_a"]()
    res_b = ns["_build_regime_b"]()
    assert res_a[1] == (1 + 1) * 4 + (1 + 9)
    assert res_b[1] == (1 + 1) * 2

    patched_again = patch_regime_into_module(
        patched,
        spec_a,
        extract_regime_snippet(patched, spec_a, operator_entrypoint="entrypoint", sibling_specs=specs)[1].replace(
            "return _stage_leaf(scale) * 4",
            "return _stage_leaf(scale) * 6",
        ),
        operator_entrypoint="entrypoint",
        sibling_specs=specs,
    )
    assert patched_again.count("def _regime_a_emit_pass1_compute(") == 1
    assert "return _stage_leaf(scale) * 6" in patched_again

    drop_cow_snippet = (
        "def _build_regime_a(tile_m: int = 1) -> object:\n"
        "    step = tile_m + 10\n"
        "    return None, step\n"
    )
    pruned = patch_regime_into_module(
        patched_again,
        spec_a,
        drop_cow_snippet,
        operator_entrypoint="entrypoint",
        sibling_specs=specs,
    )
    assert "def _regime_a_emit_pass1_compute(" not in pruned
    for sym, expected_hash in sib_hashes_before.items():
        assert _top_level_ast_sha256(pruned, sym) == expected_hash


def test_extract_closure_and_cow_when_editable_symbols_empty() -> None:
    """Verify closure extraction and CoW work even when editable_symbols is empty."""
    spec_a = RegimeTargetSpec("regime_a", ("_build_regime_a",), {})
    spec_b = RegimeTargetSpec("regime_b", ("_build_regime_b",), {})
    _, snip = extract_regime_snippet(
        _SHARED_HELPER_MODULE,
        spec_a,
        operator_entrypoint="entrypoint",
        sibling_specs=(spec_a, spec_b),
    )
    assert "def _emit_pass1_compute(" in snip
    assert "def _stage_leaf(" in snip
    assert "def _exclusive_a_helper(" in snip
    assert "def _build_regime_b(" not in snip

    co_edited = snip.replace("return _stage_leaf(scale) * 2", "return _stage_leaf(scale) * 5")
    patched = patch_regime_into_module(
        _SHARED_HELPER_MODULE,
        spec_a,
        co_edited,
        operator_entrypoint="entrypoint",
        sibling_specs=(spec_a, spec_b),
    )
    assert "def _regime_a_emit_pass1_compute(" in patched
    assert _top_level_ast_sha256(patched, "_emit_pass1_compute") == _top_level_ast_sha256(
        _SHARED_HELPER_MODULE, "_emit_pass1_compute"
    )
    assert _top_level_ast_sha256(patched, "_build_regime_b") == _top_level_ast_sha256(
        _SHARED_HELPER_MODULE, "_build_regime_b"
    )


def _dummy_stage_a(source: str) -> StageAResult:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
        grid_blocks=20,
        wave_occupancy=0.60,
    )
    feats = extract_structural_features(None, ledger=ledger)
    route = route_execution_track(feats)
    return StageAResult(passed=True, prim_func=None, ledger=ledger, features=feats, route=route)


def _make_surrogate(
    cid: str,
    anchors: tuple[AnchorCaseSpec, ...],
    lats: dict[int, float],
) -> CandidateSurrogateResult:
    metrics = [
        build_anchor_metrics(spec, accuracy_passed=True, cand_us=lats[spec.case_num])
        for spec in anchors
    ]
    return aggregate_surrogate_result(cid, "attention_residual", metrics)


def _tail_guard_runner(
    tmp_path: Path,
    anchors: tuple[AnchorCaseSpec, ...],
    l2_fn,
) -> PerfTargetCampaignRunner:
    target_mod = tmp_path / "kernels.py"
    target_mod.write_text(_SHARED_HELPER_MODULE, encoding="utf-8")
    specs = _regime_specs()
    roofline = HirRooflineSummary(
        anchor_dims={"M": 1024, "D": 1024},
        compute_flops=1,
        min_gm_bytes=1,
        arithmetic_intensity=1.0,
        compute_floor_us=1.0,
        memory_floor_us=1.0,
        ideal_roofline_us=1.0,
        dominant_bound="vector",
    )
    blueprints = {
        spec.regime_tag: HirBlueprintPacket(
            entry_name="exp",
            roofline=roofline,
            z3_pipeline_summary={},
            reference_ledger=None,
            lowering_diagnostics=(),
        )
        for spec in specs
    }
    return PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=0,
            min_surrogate_gain=0.25,
        ),
        target_module_path=target_mod,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=specs,
        regime_blueprints=blueprints,
        l0_checker=lambda src, _reg: _dummy_stage_a(src),
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=l2_fn,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt.json",
    )


def test_dual_layer_tail_guard_active_gain_and_sibling_recheck_protocol(tmp_path: Path) -> None:
    """Verify 15% active gain passes, 1 sibling dip triggers 1 recheck, and 2+ dips block."""
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 1, "regime_a", 100.0, 20.0),
        AnchorCaseSpec("level4/attention_residual", 2, "regime_b", 100.0, 20.0),
        AnchorCaseSpec("level4/attention_residual", 3, "regime_b", 100.0, 20.0),
    )
    spec_a, _ = _regime_specs()
    best_res = _make_surrogate("best", anchors, {1: 100.0, 2: 200.0, 3: 100.0})

    recheck_calls: list[list[int]] = []

    def _l2_recheck_recovers(cands, req_anchors):
        recheck_calls.append([a.case_num for a in req_anchors])
        out = {}
        for cid, _path in cands:
            lats = {a.case_num: 100.0 for a in req_anchors}
            if 2 in lats:
                lats[2] = 200.0
            out[cid] = _make_surrogate(cid, tuple(req_anchors), lats)
        return out

    runner = _tail_guard_runner(tmp_path, anchors, _l2_recheck_recovers)

    active_15pct_clean = _make_surrogate("cand_ok", anchors, {1: 86.95, 2: 203.0, 3: 101.0})
    ok_clean, updated_clean = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_ok",
        candidate_source=_SHARED_HELPER_MODULE,
        top_res=active_15pct_clean,
        best_result=best_res,
        regime_spec=spec_a,
        active_anchors=anchors,
    )
    assert ok_clean is True
    assert updated_clean is active_15pct_clean
    assert recheck_calls == []

    single_dip = _make_surrogate("cand_dip1", anchors, {1: 86.95, 2: 200.0, 3: 103.5})
    ok_recheck, updated_recheck = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_dip1",
        candidate_source=_SHARED_HELPER_MODULE,
        top_res=single_dip,
        best_result=best_res,
        regime_spec=spec_a,
        active_anchors=anchors,
    )
    assert ok_recheck is True
    assert recheck_calls == [[3]]
    by_case = {a.case_num: a for a in updated_recheck.anchors}
    assert by_case[3].cand_us == 100.0

    recheck_calls.clear()
    two_dips = _make_surrogate("cand_dip2", anchors, {1: 86.95, 2: 206.0, 3: 103.5})
    ok_two_dips, _ = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_dip2",
        candidate_source=_SHARED_HELPER_MODULE,
        top_res=two_dips,
        best_result=best_res,
        regime_spec=spec_a,
        active_anchors=anchors,
    )
    assert ok_two_dips is False
    assert recheck_calls == []

    def _l2_recheck_stays_dipped(cands, req_anchors):
        recheck_calls.append([a.case_num for a in req_anchors])
        return {
            cid: _make_surrogate(cid, tuple(req_anchors), {3: 104.0})
            for cid, _path in cands
        }

    runner_fail = _tail_guard_runner(tmp_path, anchors, _l2_recheck_stays_dipped)
    ok_fail_recheck, _ = runner_fail._evaluate_dual_layer_tail_guard(
        candidate_id="cand_dip1_fail",
        candidate_source=_SHARED_HELPER_MODULE,
        top_res=single_dip,
        best_result=best_res,
        regime_spec=spec_a,
        active_anchors=anchors,
    )
    assert ok_fail_recheck is False
    assert recheck_calls == [[3]]

    recheck_calls.clear()
    active_tail_regress = _make_surrogate("cand_active_bad", anchors, {1: 105.0, 2: 200.0, 3: 103.5})
    ok_active_bad, _ = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand_active_bad",
        candidate_source=_SHARED_HELPER_MODULE,
        top_res=active_tail_regress,
        best_result=best_res,
        regime_spec=spec_a,
        active_anchors=anchors,
    )
    assert ok_active_bad is False
    assert recheck_calls == []
    assert "_recheck_single_sibling_anchor" in inspect.getsource(
        PerfTargetCampaignRunner._evaluate_dual_layer_tail_guard
    )


def test_stage6_zero_shot_prescreen_and_experience_lake_anomaly_wiring(
    tmp_path: Path,
) -> None:
    """Verify zero-shot cost_prescreen activation and ExperienceLake anomaly persistence."""
    anchors = (AnchorCaseSpec("level4/attention_residual", 1, "regime_a", 100.0, 20.0),)
    spec_a, _ = _regime_specs()
    target_mod = tmp_path / "kernels.py"
    target_mod.write_text(_SHARED_HELPER_MODULE, encoding="utf-8")
    roofline = HirRooflineSummary(
        anchor_dims={"M": 1024, "D": 1024},
        compute_flops=1,
        min_gm_bytes=1,
        arithmetic_intensity=1.0,
        compute_floor_us=1.0,
        memory_floor_us=1.0,
        ideal_roofline_us=1.0,
        dominant_bound="vector",
    )
    blueprints = {
        spec_a.regime_tag: HirBlueprintPacket(
            entry_name="exp",
            roofline=roofline,
            z3_pipeline_summary={},
            reference_ledger=None,
            lowering_diagnostics=(),
        )
    }
    fast_cand = (
        "def _build_regime_a(tile_m: int = 2) -> object:\n"
        "    return None, tile_m\n"
    )
    slow_cand = (
        "def _build_regime_a(tile_m: int = 99) -> object:\n"
        "    return None, tile_m\n"
    )
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [fast_cand, fast_cand],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": "Slow scalar outlier",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": slow_cand,
                    }
                )
            ],
        }
    )

    def _l0_ranked(src: str, _reg: RegimeTargetSpec) -> StageAResult:
        is_slow = "tile_m: int = 99" in src
        ii = 50000 if is_slow else 80
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=ii,
            pipe_cycles={"AIV0.PIPE_V": ii},
            waterlines={},
            grid_blocks=20,
            wave_occupancy=0.60,
        )
        feats = extract_structural_features(None, ledger=ledger)
        route = route_execution_track(feats)
        return StageAResult(passed=True, prim_func=None, ledger=ledger, features=feats, route=route)

    l1_calls: list[str] = []

    def _l1_track(src: str, _reg: RegimeTargetSpec) -> StageBResult:
        l1_calls.append(src)
        return StageBResult(passed=True)

    lake_dir = tmp_path / "lake"
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=2,
            enable_tilesize_prescreen=True,
            cost_prescreen_top_k=None,
        ),
        target_module_path=target_mod,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec_a,),
        regime_blueprints=blueprints,
        llm_backend=backend,
        l0_checker=_l0_ranked,
        l1_compiler=_l1_track,
        l2_batch_evaluator=lambda cands, req: {
            cid: _make_surrogate(cid, tuple(req), {1: 50.0})
            for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=lake_dir,
    )

    assert runner._cost_model.anomaly_lake is runner._experience_lake
    assert len(runner._cost_model._samples) == 0
    cf = CostFeatures(vec_flops=1e6, gm_bytes=1e5, active_aiv_cores=56)
    ana = runner._cost_model.predict(cf).pred_us
    runner._cost_model.observe(cf, ana * 50.0)
    assert len(runner._cost_model._samples) == 0
    anomaly_file = lake_dir / "unmodeled_anomalies.jsonl"
    assert anomaly_file.is_file()
    assert len(anomaly_file.read_text(encoding="utf-8").splitlines()) == 1

    report = asyncio.run(runner.run())
    assert len(l1_calls) == 3
    assert all("tile_m: int = 99" not in s for s in l1_calls)
    assert any(tier == "cost_prescreen" for _cid, tier, _msg in report.rounds[0].l0_l1_failed)

