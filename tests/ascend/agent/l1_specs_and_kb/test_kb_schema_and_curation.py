"""Unit tests for per-track KB rules, safe predicates, persistence, tabu, and curation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    FamilyTabuEntry,
    SymptomFamily,
    TabuEntry,
    UnsafePredicateError,
    append_kb_event_shard,
    append_rule_misapplication,
    append_winning_edge,
    build_predicate_context,
    default_seed_rules_path,
    load_seed_rules,
    safe_eval_predicate,
    save_distilled_rule_file,
    save_rules_yaml,
    save_staging_candidate_rule,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.kb.schema import (
    DETERMINISTIC_SYMPTOM_FAMILIES,
    normalize_runtime_backend,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    append_counterexample_boundary,
    compute_source_sha256,
    extract_ub_occupancy_ratio,
    load_counterexample_boundaries,
    matches_operator_filter,
    resolve_candidate_for_regime,
    resolve_default_experience_lake_dir,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def test_safe_predicate_validation_and_ast_evaluation() -> None:
    """Verify safe AST whitelist blocks injection/private access and evaluates expressions."""
    for bad_expr in (
        "",
        "   ",
        "def f(): pass",
        "__import__('os').system('echo x')",
        "(lambda x: x > 1)(2)",
        "AIC._secret > 0",
        "_hidden.flag > 0",
        "len([1, 2]) > 0",
        "min(1, 2, key=abs) > 0",
        "[x for x in (1, 2)]",
        "x ** 2 > 1",
    ):
        with pytest.raises(UnsafePredicateError):
            validate_rule_predicate(bad_expr)

    env = {
        "PIPE_V": 80,
        "AIV0": {"PIPE_V": 40},
        "prev": {"PIPE_V": 120, "AIV0.PIPE_V": 80},
        "dominant_pipe": "AIV0.PIPE_V",
        "flag": True,
    }
    assert safe_eval_predicate("PIPE_V > 0 and AIV0.PIPE_V < prev.AIV0.PIPE_V", env) is True
    assert safe_eval_predicate("dominant_pipe in ('PIPE_V', 'AIV0.PIPE_V')", env) is True
    assert safe_eval_predicate("min(abs(-10), PIPE_V) == 10 and not (not flag)", env) is True
    assert safe_eval_predicate("False and missing_var > 0", env) is False
    with pytest.raises(KeyError):
        safe_eval_predicate("missing_var > 0", env)


def test_per_track_rules_backend_isolation_and_cpp20_rendering() -> None:
    """Verify backend normalization, dialect isolation, and C++20 rendering."""
    assert normalize_runtime_backend(None) is None
    assert normalize_runtime_backend("") is None
    assert normalize_runtime_backend("TileLang_DSL") == "tilelang_dsl"
    assert normalize_runtime_backend("inline_cce_vf") == "tilelang_dsl"
    assert normalize_runtime_backend("CCE_CPP20") == "ascendc_trackb"
    assert normalize_runtime_backend("ascendc_template") == "ascendc_trackb"
    assert normalize_runtime_backend("cce_trackb") == "ascendc_trackb"
    assert normalize_runtime_backend("hybrid") == "hybrid_trackb"

    rule = AscendCausalRule(
        rule_id="R_HYBRID_01",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Use T.SimdVF.",
        expected_postcondition="PIPE_V < prev.PIPE_V",
        source_provenance="ref:test",
        compatible_backends=("tilelang_dsl", "cce_cpp20", "hybrid_trackb"),
        cpp20_transform_template="Use simd_inst::vmul in __simd_vf__.",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
        anti_precondition="spill_bytes > 0",
    )
    assert rule.supports_backend("tilelang_dsl") is True
    assert rule.supports_backend("ascendc_trackb") is True
    assert rule.supports_backend(None) is True

    tl_rendered = rule.render_for_backend("tilelang_dsl")
    assert tl_rendered.transform_template == "Use T.SimdVF."
    cpp_rendered = rule.render_for_backend("ascendc_trackb")
    assert cpp_rendered.transform_template == "Use simd_inst::vmul in __simd_vf__."
    hyb_rendered = rule.render_for_backend("hybrid_trackb")
    assert "Use T.SimdVF." in hyb_rendered.transform_template
    assert "Track-B CCE (C++20):" in hyb_rendered.transform_template
    assert hyb_rendered.render_for_backend("hybrid_trackb") == hyb_rendered
    assert len(rule.as_six_tuple()) == 6

    tl_only = AscendCausalRule(
        rule_id="R_TL",
        regime_guard="True",
        precondition="True",
        transform_template="TileLang only.",
        expected_postcondition="True",
        source_provenance="test",
        compatible_backends=("tilelang_dsl",),
    )
    assert tl_only.supports_backend("tilelang_dsl") is True
    assert tl_only.supports_backend("cce_trackb") is False
    assert tl_only.supports_backend("hybrid_trackb") is False

    with pytest.raises(ValueError, match="rule_id and transform_template"):
        AscendCausalRule("", "True", "True", "x", "True", "test")
    with pytest.raises(ValueError, match="invalid rule scope"):
        AscendCausalRule("R1", "True", "True", "x", "True", "test", scope="bad")
    with pytest.raises(ValueError, match="compatible_backends"):
        AscendCausalRule("R1", "True", "True", "x", "True", "test", compatible_backends=())
    with pytest.raises(ValueError, match="invalid symptom_family"):
        AscendCausalRule(
            "R1",
            "True",
            "True",
            "x",
            "True",
            "test",
            symptom_family=SymptomFamily.UNMODELED_SILICON_RESIDUAL.value,
        )


def test_seed_rules_completeness_sanitization_and_yaml_json_persistence(tmp_path: Path) -> None:
    """Verify bundled seed_rules.yaml contracts, YAML/JSON persistence, and predicate context."""
    yaml_text = default_seed_rules_path().read_text(encoding="utf-8")
    assert "/home/" not in yaml_text
    assert "/Users/" not in yaml_text
    assert not re.search(r"\bmla\b", yaml_text.lower())

    rules = load_seed_rules()
    assert len(rules) >= 41
    assert len({r.mechanism_tag for r in rules}) == len(rules)
    assert {SymptomFamily(r.symptom_family) for r in rules} == DETERMINISTIC_SYMPTOM_FAMILIES
    by_id = {r.rule_id: r for r in rules}
    assert by_id["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"].compatible_backends == (
        "ascendc_trackb",
        "hybrid_trackb",
    )
    assert by_id["R19A_CONSTRAINED_STAGE_OFFSET_PIPELINE"].compatible_backends == ("tilelang_dsl",)
    assert by_id["R35_Z3_TASK_CYCLE_LATENCY_OVERRIDE"].compatible_backends == ("tilelang_dsl",)

    for rule in rules:
        validate_rule_predicate(rule.regime_guard)
        validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.anti_precondition)
        validate_rule_predicate(rule.expected_postcondition)
        assert AscendCausalRule.from_json(rule.to_json()) == rule

    rules_yaml = tmp_path / "rules.yaml"
    save_rules_yaml(rules[:3], rules_yaml)
    reloaded = load_seed_rules(rules_yaml)
    assert reloaded == rules[:3]

    flywheel_rule = AscendCausalRule(
        rule_id="R_FW_01",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Optimize.",
        expected_postcondition="PIPE_V < prev.PIPE_V",
        source_provenance="flywheel:ablation",
        symptom_family="",
    )
    fw_yaml = tmp_path / "fw.yaml"
    save_rules_yaml([flywheel_rule], fw_yaml)
    assert load_seed_rules(fw_yaml)[0].rule_id == "R_FW_01"
    with pytest.raises(ValueError, match="must declare a valid deterministic symptom_family"):
        load_seed_rules(fw_yaml, require_symptom_family=True)

    pre_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=3,
        pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 96},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=262144,
                explicit_buffers={"a": 131072},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=32768,
        gm_traffic_scope="full_grid",
    )
    post_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 128, "AIC.PIPE_FIX": 96},
        waterlines={},
        gm_roundtrip_bytes=16384,
        gm_traffic_scope="full_grid",
    )
    ctx_match = build_predicate_context(pre_ledger)
    assert ctx_match["ub_occupancy_ratio"] == pytest.approx(0.5)
    assert by_id["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].matches(ctx_match) is True
    post_ctx = build_predicate_context(
        post_ledger,
        prev_ledger=pre_ledger,
        roofline={"min_gm_bytes": 16384, "compute_floor_us": 5.0},
        extra={"aiv_vec_ratio": 0.42},
    )
    assert post_ctx["transfer_inflation_ratio"] == pytest.approx(2.0)
    assert post_ctx["prev"]["transfer_inflation_ratio"] == pytest.approx(3.0)
    assert post_ctx["l2_pmu_available"] == 1
    assert by_id["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].postcondition_realized(post_ctx) is True


def test_experience_lake_records_deltas_and_curation_files(tmp_path: Path) -> None:
    """Verify ExperienceLake JSON/JSONL records, pairwise deltas, and KB curation artifacts."""
    lake_dir = tmp_path / "lake"
    assert resolve_default_experience_lake_dir(explicit_dir=lake_dir) == lake_dir.resolve()
    assert compute_source_sha256("abc") == compute_source_sha256("abc")
    assert matches_operator_filter("level2/rms_norm", "rms_norm") is True
    assert matches_operator_filter("level2/rms_norm", "level4/rms_norm") is False

    lake = ExperienceLake(lake_dir)
    parent_art = CandidateArtifactRecord(
        cand_sha="sha_parent",
        operator="rms_norm",
        regime_tag="small",
        base_ledger={"grid_blocks": 4},
        topology_axes={"compute_engine": "aiv_only"},
    )
    cand_art = CandidateArtifactRecord(
        cand_sha="sha_child",
        operator="rms_norm",
        regime_tag="small",
        parent_sha="sha_parent",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        bisheng_summary={"vec_regs": 24},
    )
    assert CandidateArtifactRecord.from_json(cand_art.to_json()) == cand_art
    lake.register_candidate(parent_art)
    lake.register_candidate(cand_art)

    parent_obs = CaseObservationRecord(
        run_id="run1",
        operator="rms_norm",
        regime_tag="small",
        cand_sha="sha_parent",
        eval_tier="l2_pass",
        case_num=1,
        measured_us=20.0,
        accuracy_passed=True,
    )
    cand_obs = CaseObservationRecord(
        run_id="run1",
        operator="rms_norm",
        regime_tag="small",
        cand_sha="sha_child",
        parent_sha="sha_parent",
        eval_tier="l2_pass",
        case_num=1,
        measured_us=12.0,
        accuracy_passed=True,
    )
    assert CaseObservationRecord.from_json(cand_obs.to_json()) == cand_obs
    lake.append_observation(parent_obs)
    lake.append_observation(cand_obs)
    lake.append_unmodeled_anomaly({"operator": "rms_norm", "case_num": 1})

    flat_arts, obs_list = lake.load_records()
    assert set(flat_arts) == {"sha_parent", "sha_child"}
    assert len(obs_list) == 2
    by_reg, _ = lake.load_records_by_regime()
    assert resolve_candidate_for_regime(by_reg, flat_arts, "sha_child", "small") == cand_art
    pairs = lake.extract_pairwise_deltas()
    assert len(pairs) == 1
    assert pairs[0][1].measured_us == pytest.approx(12.0)
    assert pairs[0][3].measured_us == pytest.approx(20.0)
    assert lake.extract_pairwise_deltas(exclude_run_id="run1") == []

    assert (
        extract_ub_occupancy_ratio({"UB": {"capacity_bytes": 100, "peak_allocated_bytes": 45}})
        == pytest.approx(0.45)
    )
    append_counterexample_boundary(lake_dir, {"operator": "level2/rms_norm", "case_num": 3})
    assert len(load_counterexample_boundaries(lake_dir, operator="rms_norm")) == 1

    staged_rule = AscendCausalRule(
        rule_id="R_STAGED_01",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Coalesce loads.",
        expected_postcondition="PIPE_V < prev.PIPE_V",
        source_provenance="flywheel:ablation",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
    )
    staging_path = save_staging_candidate_rule(
        lake_dir, staged_rule, ablation_report={"gain": 0.25}
    )
    assert json.loads(staging_path.read_text(encoding="utf-8"))["status"] == "pending_review"
    distilled_path = save_distilled_rule_file(lake_dir, staged_rule)
    assert distilled_path.is_file()
    assert append_winning_edge(lake_dir, {"rule_id": "R_STAGED_01"}).is_file()
    assert append_rule_misapplication(
        lake_dir, {"rule_id": "R04_MTE2_SPARSE_GATHER_COALESCE"}
    ).is_file()

    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    append_kb_event_shard(
        lake_dir,
        run_id="peer_1",
        rule_id="R_STAGED_01",
        postcondition_realized=True,
        speed_gain=0.3,
        topology_axes=axes,
    )
    frozen = BayesianRuleTracker.from_experience_lake(lake_dir)
    assert "R_STAGED_01" not in frozen.rules
    opted_in = BayesianRuleTracker.from_experience_lake(
        lake_dir, allow_unreviewed_distilled_rules=True
    )
    assert "R_STAGED_01" in opted_in.rules
    assert opted_in.posteriors["R_STAGED_01"].successes == 1


def test_symptom_family_tabu_cooldown_escalation_and_roundtrip() -> None:
    """Verify scoped rule cooldown, streak >= 2 family tabu, parent reset, and serialization."""
    tracker = BayesianRuleTracker.from_seed_yaml()
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 64},
        waterlines={},
        mte2_inflight_per_aiv=24,
        unaligned_dma_padding_bytes=16,
        wave_occupancy=0.45,
        dma_burst_efficiency_ratio=0.55,
    )
    ctx = build_predicate_context(ledger)
    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_1",
        current_round=1,
        reason="compile_fail",
    )
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_1",
        current_round=3,
    ) is True
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R04_MTE2_SPARSE_GATHER_COALESCE",
        parent_sha="sha_1",
        current_round=4,
    ) is False
    assert tracker.is_family_tabued(
        regime_tag="reg_a",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST,
        parent_sha="sha_1",
        current_round=2,
    ) is False

    tracker.record_scoped_failure(
        regime_tag="reg_a",
        rule_id="R16_PADDED_PITCH_TAIL_MASKING",
        parent_sha="sha_1",
        current_round=2,
        reason="no_gain",
    )
    assert tracker.is_family_tabued(
        regime_tag="reg_a",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST,
        parent_sha="sha_1",
        current_round=3,
    ) is True
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_1",
        current_round=3,
    ) is True
    ranked_r3 = {
        r.rule_id
        for r in tracker.rank_applicable_rules(
            ctx, regime_tag="reg_a", parent_sha="sha_1", current_round=3
        )
    }
    assert "R18_WAVE_QUANTIZED_BURST_TILING" not in ranked_r3
    assert tracker.is_rule_tabued(
        regime_tag="reg_a",
        rule_id="R18_WAVE_QUANTIZED_BURST_TILING",
        parent_sha="sha_2",
        current_round=3,
    ) is False

    assert isinstance(tracker.tabu_entries[0], TabuEntry)
    assert isinstance(tracker.family_tabu_entries[0], FamilyTabuEntry)
    restored = BayesianRuleTracker.from_dict(tracker.to_dict())
    assert restored.tabu_entries == tracker.tabu_entries
    assert restored.family_tabu_entries == tracker.family_tabu_entries

    tracker.reset_scoped_family_streak(
        regime_tag="reg_a", rule_id="R04_MTE2_SPARSE_GATHER_COALESCE", parent_sha="sha_1"
    )
    streak_key = ("reg_a", SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value, "sha_1")
    assert streak_key not in tracker.family_failure_streaks
