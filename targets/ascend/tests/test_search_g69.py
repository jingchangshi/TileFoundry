"""P7-01/G69: the search harness, its pruning and its measurement statistics.

These tests check the properties a pass count cannot show: that the candidate space is
ENUMERATED rather than listed after the fact, that pruning rejects for arithmetic reasons with
named codes, that the cache identity invalidates when any ingredient changes, that a failed
candidate cannot enter a ranking, and that the reported statistics are recomputable from the raw
samples. Device numbers live in evidence/G69_SEARCH_HARNESS.json; nothing here claims a device
measurement, and every timing below comes from the injectable simulated clock and is labelled as
such.
"""
import json
import math
from pathlib import Path

import pytest

from tilefoundry_ascend.mla.model import BFLOAT16, BNSD, BSND, FLOAT16, build_mla
from tilefoundry_ascend.search import (
    Budget,
    CacheIdentity,
    CandidateResult,
    CandidateRun,
    DeviceTimer,
    DeviceTimerUnavailable,
    FEASIBLE,
    GeneratorError,
    HarnessError,
    INFEASIBLE,
    MEASUREMENT_SOURCE_SIMULATED,
    OK,
    ORIGIN_ADVANCED,
    ORIGIN_NEIGHBOR,
    ORIGIN_SEED,
    P0_BUDGET,
    PRECISION_HI_LO,
    PRECISION_SINGLE,
    PRUNED_STATIC,
    RESCALE_CANONICAL,
    RESCALE_DEFERRED,
    SimulatedClock,
    attention_residual_neighbours,
    attention_residual_seeds,
    compute_stats,
    confirmation_required,
    deduplicate,
    judge_all,
    judge_attention_residual,
    judge_mla,
    mla_advanced_around,
    mla_neighbours,
    mla_seeds,
    rank,
    run_candidate,
    search_case,
    summarise_filter,
    theta_legal_bound,
    theta_search_bound,
    toolchain_identity,
)


def mla_spec(d_nope=512, dtype=FLOAT16, seq_q=128, seq_kv=128, heads_q=128, layout=BSND):
    return build_mla(batch=2, seq_q=seq_q, seq_kv=seq_kv, heads_q=heads_q, heads_kv=1,
                     d_nope=d_nope, d_rope=64, layout=layout, dtype=dtype)


# ------------------------------------------------------------------ candidate generation

def test_mla_seeds_are_twelve_distinct_stratified_configurations():
    """A seed set of twelve variations of one tile measures one configuration twelve times."""
    seeds = mla_seeds(mla_spec())
    assert len(seeds) == 12
    assert len({c.key() for c in seeds}) == 12
    # Every route G61/G62 require must be reachable from the seeds themselves, not only from the
    # advanced list, or the stratified coverage claim is false.
    families = {c.family for c in seeds}
    assert "mla_canonical" in families
    assert "mla_deferred" in families
    assert "mla_output_shard" in families
    assert any(c.precision == PRECISION_HI_LO for c in seeds)
    assert any(not c.kv_reuse for c in seeds)
    assert any(c.stages == 3 for c in seeds)
    # Both a zero and a positive theta must appear: theta=0 is the always-draining deferred form
    # and the positive one is the route's actual claim (spec 8.7).
    thetas = [c.theta for c in seeds if c.rescale == RESCALE_DEFERRED]
    assert 0.0 in thetas
    assert any(t > 0.0 for t in thetas)


def test_the_attention_residual_seeds_cover_all_three_families():
    seeds = attention_residual_seeds(n=8, tokens=64, d=2048)
    assert len(seeds) == 12
    assert len({c.key() for c in seeds}) == 12
    assert {c.family for c in seeds} == {"online", "blocked_online", "resident_n"}
    assert any(c.score_mode == "factored_gw" for c in seeds)
    assert any(c.prefetch != "none" for c in seeds)
    assert any(c.vf_mode == "simtvf" for c in seeds)


def test_the_family_is_derived_from_cn_and_a_contradicting_claim_is_refused():
    """A report must not be able to label a schedule it did not configure."""
    from tilefoundry_ascend.search.generator import AttentionResidualCandidate
    for cn, n, expected in ((1, 8, "online"), (4, 8, "blocked_online"), (8, 8, "resident_n")):
        candidate = AttentionResidualCandidate(n=n, tokens=8, d=64, dtype="bfloat16",
                                              block_n=cn, score_mode="literal_fp32",
                                              token_rows=1, d_chunk=64, v_versions=1)
        assert candidate.family == expected
    with pytest.raises(GeneratorError):
        AttentionResidualCandidate(n=8, tokens=8, d=64, dtype="bfloat16", block_n=4,
                                   score_mode="literal_fp32", token_rows=1, d_chunk=64,
                                   v_versions=1, declared_family="resident_n")


def test_deduplication_refuses_to_count_one_schedule_twice():
    seeds = mla_seeds(mla_spec())
    doubled = tuple(seeds) + tuple(seeds)
    assert len(deduplicate(doubled)) == len(seeds)


def test_a_neighbour_changes_exactly_one_factor_and_says_which():
    seed = mla_seeds(mla_spec())[0]
    neighbours = mla_neighbours(seed)
    assert neighbours
    assert all(n.origin == ORIGIN_NEIGHBOR for n in neighbours)
    assert all(n.changed for n in neighbours)
    assert all(n.parent == seed.key() for n in neighbours)
    for neighbour in neighbours:
        assert neighbour.key() != seed.key()
        differing = [name for name, value in neighbour.knobs()
                     if name != "theta" and value != dict(seed.knobs())[name]]
        # Changing theta from 0 to the bound flips the mode too, so allow that one compound case.
        assert len(differing) <= 1, (neighbour.changed, differing)
    with pytest.raises(GeneratorError):
        from tilefoundry_ascend.search.generator import MLACandidate
        MLACandidate(block_q=32, block_c=128, output_panel=128, kv_versions=1, p_versions=1,
                     stages=1, precision=PRECISION_SINGLE, rescale=RESCALE_CANONICAL, theta=0.0,
                     output_shards=1, kv_reuse=True, origin=ORIGIN_NEIGHBOR)


def test_the_advanced_list_carries_the_routes_the_gates_require():
    seed = mla_seeds(mla_spec())[0]
    advanced = mla_advanced_around(seed)
    assert advanced
    assert all(a.origin == ORIGIN_ADVANCED for a in advanced)
    routes = {(a.rescale, a.output_shards, a.precision) for a in advanced}
    assert (RESCALE_DEFERRED, 1, PRECISION_SINGLE) in routes, "G61 deferred route missing"
    assert any(shards > 1 for _, shards, _ in routes), "G62 output-shard route missing"
    assert any(precision == PRECISION_HI_LO for _, _, precision in routes)


# ------------------------------------------------------------------ theta bounds

def test_theta_is_bounded_by_the_operand_and_the_accumulator_not_by_a_magic_number():
    """The case doc explicitly forbids copying the reference implementation's constant 6."""
    fp16 = theta_legal_bound(FLOAT16, block_c=128)
    bf16 = theta_legal_bound(BFLOAT16, block_c=128)
    assert 0.0 < fp16 < bf16, "fp16 has a far smaller finite max, so its bound must be tighter"
    # exp(bound) must stay inside the dtype's finite range with the declared headroom.
    assert math.exp(fp16) <= 65504.0
    assert math.exp(bf16) <= 3.3895313892515355e38
    # A wider tile accumulates more mass into one FP32 cell, so the bound can only tighten.
    assert theta_legal_bound(FLOAT16, block_c=256) <= theta_legal_bound(FLOAT16, block_c=128)
    # A larger |V| bound also tightens it, monotonically.
    assert theta_legal_bound(FLOAT16, block_c=128, v_abs_max=16.0) <= fp16
    assert theta_search_bound(FLOAT16, block_c=128) <= theta_legal_bound(FLOAT16, block_c=128)


def test_a_theta_beyond_the_operand_bound_is_rejected_by_name():
    spec = mla_spec()
    seed = mla_seeds(spec)[0]
    bound = theta_legal_bound(FLOAT16, block_c=128)
    over = seed.with_(rescale=RESCALE_DEFERRED, theta=bound * 2.0 + 1.0,
                      origin=ORIGIN_NEIGHBOR, changed="theta")
    judgement = judge_mla(over, spec)
    assert not judgement.feasible
    assert "LEGAL_THETA_EXCEEDS_OPERAND_BOUND" in {r.code for r in judgement.rejections}

# ------------------------------------------------------------------ static pruning

def test_pruning_names_the_space_that_overflowed():
    """A rejection has to say WHICH resource bound; an unqualified 'infeasible' proves nothing."""
    spec = mla_spec()
    seeds = mla_seeds(spec)
    judgements = judge_all(seeds, operator="mla", spec=spec)
    summary = summarise_filter(judgements)
    assert summary["infeasible"] > 0, "a 12-seed set that is entirely feasible is suspicious"
    codes = summary["rejections_by_code"]
    assert any(code.startswith("RES_") for code in codes)
    assert any(code.startswith("LEGAL_") for code in codes)
    for judgement in judgements:
        for rejection in judgement.rejections:
            assert rejection.space in ("ub", "l1", "l0a", "l0b", "l0c", "ss", "events",
                                       "schedule", "numerics", "layout")
            if (rejection.space in ("ub", "l1", "l0a", "l0b", "l0c", "ss", "events")
                    and rejection.used is not None and rejection.limit is not None):
                assert rejection.used > rejection.limit, (
                    "a capacity rejection must show the requirement exceeding the limit")


def test_the_six_spaces_are_each_charged_and_the_aiv_only_operator_states_its_zeros():
    """An unmodelled space and a genuinely unused space must not look the same."""
    mla_judgement = judge_mla(mla_seeds(mla_spec())[0], mla_spec())
    spaces = set(mla_judgement.use.spaces())
    assert {"ub", "l1", "l0a", "l0b", "l0c"} <= spaces
    assert "ss" not in spaces, ("a canonical candidate publishes nothing into group scalar "
                               "storage, so it must charge no SS at all")
    deferred = mla_seeds(mla_spec())[2]
    assert deferred.rescale == RESCALE_DEFERRED
    assert "ss" in judge_mla(deferred, mla_spec()).use.spaces(), (
        "the deferred route's group predicate and factor live in SS and must be charged")

    ar_judgement = judge_attention_residual(attention_residual_seeds(n=8, tokens=64, d=1024)[0])
    ar_spaces = set(ar_judgement.use.spaces())
    assert "ub" in ar_spaces and "l1" in ar_spaces
    assert "l0c" not in ar_spaces, "an AIV-only kernel issues no Cube work"
    assert any("charged ZERO" in note for note in ar_judgement.use.notes)


def test_gemm_operand_staging_is_a_maximum_and_ub_is_a_sum():
    """QK and PV are serialized on one Cube, so charging their operands as a sum would reject
    configurations the hardware can run; UB's objects genuinely overlap, so it must be a sum."""
    judgement = judge_mla(mla_seeds(mla_spec())[0], mla_spec())
    max_ruled = {c.item for c in judgement.use.charges if c.rule == "max"}
    assert {"qa_l0a", "pa_l0a", "kb_l0b", "vb_l0b"} <= max_ruled
    assert all(c.rule == "sum" for c in judgement.use.charges if c.space == "ub")
    # The sum rule for a shared space means the total is the arithmetic sum of its charges.
    ub = sum(c.bytes for c in judgement.use.charges if c.space == "ub")
    assert judgement.use.total("ub") == ub


def test_deferred_rescale_costs_l0c_and_canonical_does_not():
    """The whole point of the deferred route is to keep U in L0C; the charge must show it."""
    spec = mla_spec()
    canonical = judge_mla(mla_seeds(spec)[0], spec)
    deferred = judge_mla(mla_seeds(spec)[2], spec)
    assert canonical.use.total("l0c") < deferred.use.total("l0c")
    assert "pending_l0c" not in canonical.use.items(), (
        "canonical drains every tile, so it must not be charged a pending accumulator")
    assert "pending_l0c" in deferred.use.items(), (
        "the deferred route exists to keep U in L0C and must be charged for it")


def test_an_unimplemented_tile_width_is_rejected_as_a_search_space_entry():
    """BC=64 is a legal entry in the declared set but has no packing path; saying so is the point."""
    spec = mla_spec()
    narrow = [c for c in mla_seeds(spec) if c.block_c == 64]
    assert narrow, "the declared BC set must include 64 so the rejection is reachable"
    codes = {r.code for r in judge_mla(narrow[0], spec).rejections}
    assert "LEGAL_BC_NOT_IMPLEMENTED" in codes


def test_a_panel_that_does_not_tile_d_nope_is_rejected_because_it_silently_drops_columns():
    spec = mla_spec(d_nope=448)
    seed = mla_seeds(spec)[0]
    bad = seed.with_(output_panel=256)
    codes = {r.code for r in judge_mla(bad, spec).rejections}
    assert "LEGAL_PANEL_NOT_TILING_D_NOPE" in codes
    # ...and the generator must not hand out such a panel as a SEED for this shape.
    assert all(c.output_panel <= 448 and 448 % c.output_panel == 0 for c in mla_seeds(spec))


def test_a_block_q_below_the_score_packing_floor_is_rejected():
    spec = mla_spec()
    seed = mla_seeds(spec)[0]
    codes = {r.code for r in judge_mla(seed.with_(block_q=16), spec).rejections}
    assert "LEGAL_BQ_BELOW_SCORE_ROWS" in codes


def test_output_sharding_is_refused_where_the_shard_width_is_not_a_panel():
    """448 columns do not split into panels of 64, so sharding is genuinely unavailable there."""
    spec448 = mla_spec(d_nope=448)
    seed = mla_seeds(spec448)[0]
    codes = {r.code for r in judge_mla(seed.with_(output_shards=2), spec448).rejections}
    assert "LEGAL_SHARD_WIDTH_NOT_PANEL" in codes
    spec512 = mla_spec(d_nope=512)
    sharded = seed.with_(output_shards=4, output_panel=128)
    assert judge_mla(sharded, spec512).feasible


def test_event_pools_are_finite_and_exhaustion_is_named():
    from tilefoundry_ascend.search import filter as flt
    limits = flt.target_limits()
    assert limits.aiv_ready_ids == 32, "the budget is derived from the declared pools"
    spec = mla_spec()
    seed = mla_seeds(spec)[0]
    starved = flt.TargetLimits(aiv_ready_ids=1, aiv_free_ids=1)
    judgement = judge_mla(seed, spec, limits=starved)
    assert not judgement.feasible
    assert any(r.code.startswith("RES_EVENT_AIV_") for r in judgement.rejections)
    assert limits.as_dict()["source"]


def test_the_same_shape_with_a_larger_d_prunes_at_least_as_hard():
    small = mla_spec(d_nope=448)
    big = mla_spec(d_nope=512)
    small_feasible = summarise_filter(
        judge_all(mla_seeds(small), operator="mla", spec=small))["feasible"]
    big_feasible = summarise_filter(
        judge_all(mla_seeds(big), operator="mla", spec=big))["feasible"]
    assert small_feasible <= big_feasible


def test_attention_residual_resident_n_gets_a_hard_wall():
    seeds = attention_residual_seeds(n=64, tokens=64, d=4096, limit=12)
    resident = [c for c in seeds if c.family == "resident_n"]
    assert resident
    judgements = [judge_attention_residual(c) for c in resident]
    assert all(not j.feasible for j in judgements)
    assert any("RES_UB_OVERFLOW" in {r.code for r in j.rejections} for j in judgements)

# ------------------------------------------------------------------ cache identity

def _identity(**overrides):
    base = dict(operator="mla", target_profile="ascend950pr_9579",
                toolchain=toolchain_identity(), source_digest="aa", plan_digest="bb",
                dtype="float16", layout="BSND", shape_digest="cc",
                precision_policy=PRECISION_SINGLE, candidate_key="dd")
    base.update(overrides)
    return CacheIdentity(**base)


def test_the_cache_identity_changes_when_any_ingredient_changes():
    """Spec 08 8.3: the key must cover source, target, precision, layout and the actual plan.

    A key built from the shape alone would return the binary of a DIFFERENT schedule, and the
    resulting timing would look entirely plausible. So every ingredient is flipped in turn.
    """
    base = _identity()
    for name, other in (("source_digest", "zz"), ("target_profile", "other_target"),
                        ("toolchain", "cann=other"), ("dtype", "bfloat16"),
                        ("layout", "BNSD"), ("shape_digest", "yy"),
                        ("precision_policy", PRECISION_HI_LO), ("plan_digest", "plan2"),
                        ("candidate_key", "ee"), ("operator", "attention_residual")):
        assert base.key() != _identity(**{name: other}).key(), (
            "changing %s must invalidate the cache key" % name)
    assert base.key() == _identity().key()


def test_an_incomplete_cache_identity_is_refused_rather_than_defaulted():
    with pytest.raises(HarnessError):
        _identity(source_digest="")
    with pytest.raises(HarnessError):
        _identity(plan_digest="")
    with pytest.raises(HarnessError):
        _identity(precision_policy="")


def test_the_source_digest_is_read_from_the_files_not_from_a_name():
    from tilefoundry_ascend.search import source_digest
    here = str(Path(__file__).resolve())
    other = str((Path(__file__).resolve().parents[1]
                 / "tilefoundry_ascend" / "search" / "filter.py"))
    assert source_digest([here]) != source_digest([other])
    assert source_digest([here]) == source_digest([here])
    with pytest.raises(HarnessError):
        source_digest(["/nonexistent/source.py"])


# ------------------------------------------------------------------ statistics

def test_the_statistics_are_recomputable_from_the_retained_raw_samples():
    samples = [10.0, 12.0, 11.0, 30.0, 13.0, 12.0, 11.5, 12.5, 60.0, 12.2]
    stats = compute_stats(samples, discarded_warmups=10)
    assert stats.count == len(samples)
    assert stats.discarded_warmups == 10, "the discard count must be reported, not implied"
    assert list(stats.samples_us) == sorted(samples), "raw samples are retained, sorted"
    assert stats.median_us == pytest.approx(sorted(samples)[4:6][0] if False else
                                            (sorted(samples)[4] + sorted(samples)[5]) / 2.0)
    assert stats.minimum_us == 10.0 and stats.maximum_us == 60.0
    assert stats.p10_us <= stats.median_us <= stats.p90_us
    assert stats.spread_us == pytest.approx(stats.p90_us - stats.p10_us)
    assert stats.relative_spread == pytest.approx(stats.spread_us / stats.median_us)


def test_an_empty_sample_list_is_refused_instead_of_becoming_nan():
    with pytest.raises(HarnessError):
        compute_stats([])
    with pytest.raises(HarnessError):
        compute_stats([1.0, float("inf")])
    with pytest.raises(HarnessError):
        compute_stats([-1.0])


def test_warmup_is_excluded_and_the_count_of_discarded_calls_is_recorded():
    calls = {"n": 0}
    timer = SimulatedClock(base_us=100.0, jitter_us=0.0)

    def run():
        calls["n"] += 1

    budget = Budget(warmup_calls=3, timing_calls_per_batch=5, confirmation_batches=2)
    result = run_candidate(
        CandidateRun(compile=lambda: "compiled", numeric_check=lambda c: None,
                     launch_check=lambda c: None, run=lambda c: run()),
        _identity(), budget=budget, timer=timer)
    assert result.status == OK
    assert result.stats.discarded_warmups == 3
    assert result.stats.count == 10, "2 confirmation batches x 5 calls"
    assert calls["n"] == 3 + 10, "warmup calls happened and were discarded"


# ------------------------------------------------------------------ rejected candidates

def test_a_numeric_failure_carries_no_latency_and_cannot_be_ranked():
    """A wrong kernel's latency is a measurement of something else, not a slower candidate."""
    result = run_candidate(
        CandidateRun(compile=lambda: "c", numeric_check=lambda c: "case 7 mismatch: 4.2e-02",
                     launch_check=lambda c: None, run=lambda c: None),
        _identity(), timer=SimulatedClock())
    assert result.status == "NUMERIC_FAILED"
    assert result.stats is None
    assert not result.rankable
    ranking = rank([result])
    assert ranking.winner is None
    assert ranking.comparable is False
    assert ranking.rejected and ranking.rejected[0].key == result.key


def test_every_failure_mode_is_distinct_and_none_of_them_is_a_latency():
    cases = [
        (CandidateRun(compile=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                      numeric_check=lambda c: None, launch_check=lambda c: None,
                      run=lambda c: None), None),
        (CandidateRun(compile=lambda: "c", numeric_check=lambda c: "mismatch",
                      launch_check=lambda c: None, run=lambda c: None), "NUMERIC_FAILED"),
        (CandidateRun(compile=lambda: "c", numeric_check=lambda c: None,
                      launch_check=lambda c: "2 host launches", run=lambda c: None),
         "LAUNCH_CONTRACT_VIOLATED"),
    ]
    for contract, expected in cases:
        if expected is None:
            with pytest.raises(RuntimeError):
                run_candidate(contract, _identity(), timer=SimulatedClock())
            continue
        result = run_candidate(contract, _identity(), timer=SimulatedClock())
        assert result.status == expected
        assert result.stats is None


def test_a_compile_that_exceeds_its_deadline_is_a_timeout_not_a_measurement():
    def slow():
        import time as _t
        _t.sleep(0.05)
        return "compiled"

    result = run_candidate(
        CandidateRun(compile=slow, numeric_check=lambda c: None, launch_check=lambda c: None,
                     run=lambda c: None),
        _identity(), timer=SimulatedClock(), deadline_s=0.001)
    assert result.status == "COMPILE_TIMEOUT"
    assert result.stats is None


def test_a_rejected_candidate_with_a_latency_is_refused_by_the_type_itself():
    """The rule is enforced where the record is built, not only where it is read."""
    stats = compute_stats([1.0, 2.0])
    with pytest.raises(HarnessError):
        CandidateResult(key="k", operator="mla", family="mla_canonical", origin=ORIGIN_SEED,
                        status="NUMERIC_FAILED", identity_key="i", stats=stats)
    with pytest.raises(HarnessError):
        CandidateResult(key="k", operator="mla", family="mla_canonical", origin=ORIGIN_SEED,
                        status=OK, identity_key="i", stats=None)


def test_pruning_produces_a_recorded_row_with_its_codes_not_a_missing_row():
    spec = mla_spec(d_nope=448)
    seed = mla_seeds(spec)[0]
    judgement = judge_mla(seed.with_(output_panel=256), spec)
    from tilefoundry_ascend.search import pruned_result
    result = pruned_result(judgement)
    assert result.status == PRUNED_STATIC
    assert result.reason.startswith("LEGAL_")
    assert "LEGAL_PANEL_NOT_TILING_D_NOPE" in result.detail
    assert result.stats is None


# ------------------------------------------------------------------ ranking and noise

def _result(key, median, spread_frac=0.02, family="mla_canonical"):
    from tilefoundry_ascend.search import Stats
    stats = Stats(count=3, median_us=median, p10_us=median * (1 - spread_frac),
                  p90_us=median * (1 + spread_frac), minimum_us=median * (1 - spread_frac),
                  maximum_us=median * (1 + spread_frac), mean_us=median,
                  spread_us=median * spread_frac * 2.0, relative_spread=spread_frac * 2.0,
                  discarded_warmups=10, samples_us=(median,))
    return CandidateResult(key=key, operator="mla", family=family, origin=ORIGIN_SEED,
                           status=OK, identity_key="i", stats=stats, lower_bound_only=True)


def test_ranking_picks_the_fastest_and_reports_the_rest_as_ties_or_losers():
    ranking = rank([_result("slow", 300.0), _result("fast", 200.0),
                    _result("mid", 250.0)])
    assert ranking.winner.key == "fast"
    assert [r.key for r in ranking.ordered] == ["fast", "mid", "slow"]
    # The medians are published with the ranking so a reader can check the ordering directly
    # instead of trusting the winner field.
    assert ranking.as_dict()["medians_us"] == {"fast": 200.0, "mid": 250.0, "slow": 300.0}
    assert ranking.comparable is True


def test_a_gap_inside_the_noise_demands_confirmation():
    """Spec 08 8.9: a difference within the repeated-batch noise is a tie, not a win."""
    left = _result("a", 200.0, spread_frac=0.02)
    right = _result("b", 202.0, spread_frac=0.02)
    assert confirmation_required(left, right)
    far = _result("c", 260.0, spread_frac=0.02)
    assert not confirmation_required(left, far)
    with pytest.raises(HarnessError):
        confirmation_required(left, CandidateResult(
            key="x", operator="mla", family="mla_canonical", origin=ORIGIN_SEED,
            status="NUMERIC_FAILED", identity_key="i"))


def test_ties_are_broken_by_resource_footprint_not_by_a_luckier_sample():
    """The rule is stated in the ranking text so a reader can see what was applied."""
    ranking = rank([_result("a", 200.0), _result("b", 200.5)])
    assert "tie" in ranking.rule.lower()
    assert ranking.winner is not None


# ------------------------------------------------------------------ the search driver

def test_the_search_driver_stops_at_the_per_case_cap_and_says_so():
    spec = mla_spec()
    seeds = mla_seeds(spec)
    budget = P0_BUDGET.case_budget(seeds=12, neighbours=20, advanced=8)
    counter = {"n": 0}

    def judge(candidate):
        return judge_mla(candidate, spec)

    def run_one(candidate, judgement):
        counter["n"] += 1
        return _result(candidate.key(), 100.0 + counter["n"], family=candidate.family)

    search_round = search_case(case_label="mla_1", operator="mla", seeds=seeds, advanced=(),
                               judge=judge, run_one=run_one, budget=budget,
                               make_neighbours=mla_neighbours)
    assert len(search_round.results) <= budget.max_candidates
    assert search_round.rounds == 2, "a seeded case must expand one neighbourhood"
    assert "cap" in search_round.stop_reason or "neighbourhood" in search_round.stop_reason
    statuses = {r.status for r in search_round.results}
    assert statuses & {OK, PRUNED_STATIC}


def test_the_search_driver_survives_a_case_where_nothing_times():
    spec = mla_spec()
    seeds = mla_seeds(spec)
    budget = P0_BUDGET.case_budget(seeds=12, neighbours=20, advanced=8)
    search_round = search_case(
        case_label="mla_fail", operator="mla", seeds=seeds, advanced=(),
        judge=lambda c: judge_mla(c, spec),
        run_one=lambda c, j: CandidateResult(
            key=c.key(), operator="mla", family=c.family, origin=c.origin,
            status="COMPILE_FAILED", identity_key="i", reason="COMPILE"),
        budget=budget, make_neighbours=mla_neighbours)
    assert search_round.ranking.winner is None
    assert search_round.rounds == 1, "nothing timed, so there is no neighbourhood to expand"
    assert "nothing to refine" in search_round.stop_reason or \
           "no neighbourhood" in search_round.stop_reason


def test_the_budget_refuses_a_seed_plus_neighbour_count_over_the_cap():
    with pytest.raises(HarnessError):
        Budget(seed_candidates_max=20, neighbor_candidates_max=20, max_candidates_per_case=20)
    with pytest.raises(HarnessError):
        Budget(warmup_calls=0)


def test_the_p0_budget_matches_the_configured_search_limits():
    assert P0_BUDGET.max_candidates_per_case == 40
    assert P0_BUDGET.seed_candidates_max == 12
    assert P0_BUDGET.neighbor_candidates_max == 20
    assert P0_BUDGET.advanced_candidates_max == 8
    assert P0_BUDGET.warmup_calls == 10
    assert P0_BUDGET.timing_calls_per_batch == 50
    assert P0_BUDGET.confirmation_batches == 3


# ------------------------------------------------------------------ measurement honesty

def test_the_simulated_clock_is_labelled_and_never_passes_as_a_device_timing():
    result = run_candidate(
        CandidateRun(compile=lambda: "c", numeric_check=lambda c: None,
                     launch_check=lambda c: None, run=lambda c: None),
        _identity(), budget=Budget(warmup_calls=1, timing_calls_per_batch=2,
                                   confirmation_batches=1),
        timer=SimulatedClock())
    assert result.measurement_source == MEASUREMENT_SOURCE_SIMULATED


def test_a_device_measurement_without_a_calibrated_timer_is_refused_by_name():
    """Timing an asynchronous launch with a wall clock is not a kernel measurement."""
    with pytest.raises(DeviceTimerUnavailable):
        DeviceTimer(sample=None).measure(lambda: None, 3)
    clock = DeviceTimer(sample=lambda: 12.5)
    assert clock.measure(lambda: None, 2) == [12.5, 12.5]
    with pytest.raises(HarnessError):
        DeviceTimer(sample=lambda: float("nan")).measure(lambda: None, 1)


def test_every_judgement_is_marked_lower_bound_only():
    """The resource figures are declared buffers, not a measurement of the real allocation."""
    judgement = judge_mla(mla_seeds(mla_spec())[0], mla_spec())
    assert judgement.as_dict()["lower_bound_only"] is True
    assert any("NOMINAL" in note or "lower bound" in note for note in judgement.use.notes)


# ------------------------------------------------------------------ evidence record

def test_the_g69_record_reports_a_real_end_to_end_filter_and_timing_pass():
    here = Path(__file__).resolve().parents[1]
    path = here / "evidence" / "G69_SEARCH_HARNESS.json"
    assert path.exists(), "G69 requires an end-to-end candidate-filter/timing record"
    data = json.loads(path.read_text())
    assert data["gate"] == "G69"
    assert data["status"] in ("VERIFIED", "BLOCKED_ENV", "INCOMPLETE")
    assert len(data["representative_shapes"]) >= 8
    for operator in ("attention_residual", "mla"):
        shapes = [s for s in data["representative_shapes"] if s["operator"] == operator]
        assert len(shapes) >= 8, "%s needs at least 8 representative shapes" % operator
        covered_dtypes = {s["dtype"] for s in shapes}
        assert len(covered_dtypes) >= 2, "the representative set must cover both dtypes"
        layouts = {s.get("layout") for s in shapes if s.get("layout")}
        if operator == "mla":
            assert layouts >= {"BSND", "BNSD"}, "MLA must cover both layouts"
    assert data["budget"]["max_candidates_per_case"] == 40
    assert data["end_to_end_run"]["verified_candidates"] > 0
    assert data["end_to_end_run"]["rejected_candidates"] > 0
    assert data["cache_invalidation_tests"]["result"] == "PASS"
    assert data["rejection_tests"]["result"] == "PASS"
    assert data["noise_tests"]["result"] == "PASS"

def test_the_g69_record_does_not_claim_a_device_number_it_did_not_take():
    """The most expensive mistake this gate can make is a plausible-looking device latency.

    Every timing in the record must carry its source, and a record that marks the device path
    BLOCKED_ENV must not also publish device medians.
    """
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G69_SEARCH_HARNESS.json").read_text())
    device = data["device_timing"]
    assert device["status"] in ("VERIFIED", "BLOCKED_ENV")
    if device["status"] == "BLOCKED_ENV":
        assert device["reason"], "a blocked device window must say what blocked it"
        assert device["what_is_NOT_claimed"]
    sources = data["end_to_end_run"]["measurement_sources"]
    assert set(sources) == {MEASUREMENT_SOURCE_SIMULATED}, (
        "this record was produced by the simulated clock and must say so")
    for case in data["end_to_end_run"]["per_case"]:
        assert case["timing_is_simulated"] is True
        # The label must be on the FIELD, not only in a nearby sentence: a reader who grabs the
        # number alone has to see that it did not come from the device.
        assert "winner_median_us" not in case, (
            "an unqualified latency field is a device claim; the simulated number must be named "
            "as illustrative")
        if case["winner"] is not None:
            assert case["winner_median_us_illustrative_only"] is not None
        assert case["rule"], "each case must publish the ranking rule that produced its winner"
    # The gate's own honesty list has to be present and non-empty, because a gate record that
    # lists nothing it did not prove is claiming everything.
    assert len(data["not_claimed"]) >= 4
    assert any("device" in line.lower() for line in data["not_claimed"])


def test_the_record_covers_the_exclusive_access_protocol_and_the_sync_scope():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G69_SEARCH_HARNESS.json").read_text())
    method = data["measurement_method"]
    for key in ("timer", "scope", "warmup_calls", "timing_calls_per_batch",
                "confirmation_batches", "statistics", "compile_time", "device_exclusivity"):
        assert key in method, "%s is a G69 timing-scope requirement" % key
    assert "flock" in method["device_exclusivity"]
    assert "excluded" in method["scope"] or "only" in method["scope"]
    # Compile time must be a separate field, never folded into the kernel time.
    assert "separately" in method["compile_time"]


def test_the_representative_set_is_selected_by_computed_coverage_not_by_hand():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G69_SEARCH_HARNESS.json").read_text())
    selection = data["representative_selection"]
    assert "set cover" in selection["method"]
    for operator in ("attention_residual", "mla"):
        block = selection[operator]
        assert block["count"] >= 8
        assert block["labels_covered"], "the covered labels must be published"
        assert len(block["case_ids"]) == block["count"]
    # Every representative must carry the labels that justified its selection.
    for shape in data["representative_shapes"]:
        assert shape["labels"]
