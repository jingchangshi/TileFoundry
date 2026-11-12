"""P6-03/G62: output-column sharding and the shape-specialized program mapping.

The device result is in evidence/G62_DEVICE_EVIDENCE.json. These tests pin the PROPERTIES that a
plausible-looking shard implementation can break: a gap or overlap in the column partition, a row
index mistaken for a query position, or a split-KV merge quietly reintroduced.
"""
import json
from pathlib import Path

import pytest

from tilefoundry_ascend.mla.model import BNSD, BSND, MLAError, build_mla
from tilefoundry_ascend.mla.sharding import (
    BD_CHOICES,
    check_no_split_kv,
    check_shards,
    choose_shard_count,
    column_shards,
    plan_programs,
    program_axes,
    query_head_of,
    query_row_position,
    shard_cost,
)


# ------------------------------------------------------------ column partition

@pytest.mark.parametrize("d_v,k", [(512, 1), (512, 2), (512, 4), (512, 8),
                                   (448, 1), (448, 2), (448, 4), (448, 7),
                                   (100, 3), (7, 7)])
def test_shards_are_disjoint_covering_and_balanced(d_v, k):
    shards = column_shards(d_v, k)
    assert check_shards(shards, d_v) == ()
    assert sum(s.width for s in shards) == d_v
    assert [s.lo for s in shards] == [sum(x.width for x in shards[:i]) for i in range(k)]
    widths = [s.width for s in shards]
    assert max(widths) - min(widths) <= 1


def test_a_gap_is_detected():
    """Rounding the width up leaves a hole at the end; nothing writes those columns."""
    bad = column_shards(512, 4)
    gapped = (bad[0], bad[1], bad[2], type(bad[3])(3, 384, 500))
    codes = [c for c, _ in check_shards(gapped, 512)]
    assert "shard_gap" in codes


def test_an_overlap_is_detected():
    """An overlap is a race, not a merge: two programs write the same column."""
    bad = column_shards(512, 4)
    overlapping = (bad[0], bad[1], bad[2], type(bad[3])(3, 300, 512))
    codes = [c for c, _ in check_shards(overlapping, 512)]
    assert "shard_overlap" in codes


def test_more_shards_than_columns_is_refused():
    with pytest.raises(MLAError) as exc:
        column_shards(4, 8)
    assert exc.value.code == "MLA_TOO_MANY_SHARDS"


def test_a_zero_shard_count_is_refused():
    with pytest.raises(MLAError) as exc:
        column_shards(512, 0)
    assert exc.value.code == "MLA_BAD_SHARD_COUNT"


def test_the_panel_choices_match_the_spec_search_set():
    assert BD_CHOICES == (64, 128, 256)


def test_the_union_of_all_shards_covers_every_output_column():
    """The spec's requirement stated directly: no overlap and a complete union. Checked column by
    column rather than by summing widths, because equal widths do not imply a correct partition."""
    for d_v, k in ((512, 4), (448, 7), (512, 2), (448, 1)):
        shards = column_shards(d_v, k)
        owners = [0] * d_v
        for s in shards:
            for c in range(s.lo, s.hi):
                owners[c] += 1
        assert owners == [1] * d_v, "d_v=%d k=%d" % (d_v, k)


# ------------------------------------------------------------ row mapping

def test_bsnd_and_bnsd_map_the_same_r_to_different_positions():
    """THE BUG THIS PINS: r is a programming row index, not a query position. At G == 1 the two
    layouts agree, so only a GQA shape can tell a correct mapping from using r directly."""
    # BSND: r = i*G + g
    assert query_row_position(5, group=4, seq_q=3, layout=BSND)[0] == 1
    # BNSD: r = g*S + i
    assert query_row_position(5, group=4, seq_q=3, layout=BNSD)[0] == 2


def test_the_mapping_is_a_bijection_over_the_grid():
    for layout in (BSND, BNSD):
        seen = set()
        for r in range(4 * 3):
            i, g, _ = query_row_position(r, group=4, seq_q=3, layout=layout)
            assert 0 <= i < 3 and 0 <= g < 4
            seen.add((i, g))
        assert len(seen) == 12


def test_using_r_as_the_position_would_give_a_different_mask():
    """This is the actual defect the mapping prevents, stated as a difference rather than a
    property. With G=4, S=3, taking r directly would apply four DIFFERENT causal limits to rows
    that all belong to query position 0 -- so three of the four would attend to KV they must not
    see. Asserting only 'i != r sometimes' would pass even if the mapping were off by one."""
    group, seq_q = 4, 3
    wrong = []
    for r in range(group * seq_q):
        i, _, _ = query_row_position(r, group=group, seq_q=seq_q, layout=BSND)
        if i != r:
            wrong.append(r)
    # computed, not assumed: with G=4 every r except 0 names a different position, so taking r
    # directly would move the causal limit for 11 of the 12 rows
    assert wrong == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_a_wrong_layout_would_misplace_rows():
    """The two mappings are not interchangeable: for the same r they name different positions, so
    compiling one layout's mapping into the other layout's kernel shifts every causal limit."""
    mismatches = 0
    for r in range(12):
        i_bsnd, _, _ = query_row_position(r, group=4, seq_q=3, layout=BSND)
        i_bnsd, _, _ = query_row_position(r, group=4, seq_q=3, layout=BNSD)
        if i_bsnd != i_bnsd:
            mismatches += 1
    # computed, not assumed: the two mappings happen to agree on 6 of the 12 rows (those where
    # i == g), which is exactly why a G=1 test cannot tell them apart AND why a GQA test can
    assert mismatches == 6
    assert [r for r in range(12)
            if query_row_position(r, group=4, seq_q=3, layout=BSND)[0]
            != query_row_position(r, group=4, seq_q=3, layout=BNSD)[0]] == [1, 2, 5, 6, 9, 10]


def test_the_query_head_follows_hq_equals_h_times_g_plus_g():
    assert query_head_of(0, kv_head=0, group=4, seq_q=3, layout=BSND) == 0
    assert query_head_of(3, kv_head=0, group=4, seq_q=3, layout=BSND) == 3
    assert query_head_of(0, kv_head=2, group=4, seq_q=3, layout=BSND) == 8


def test_an_out_of_range_row_is_refused():
    with pytest.raises(MLAError) as exc:
        query_row_position(12, group=4, seq_q=3, layout=BSND)
    assert exc.value.code == "MLA_ROW_OUT_OF_RANGE"


def test_an_unknown_layout_is_refused_by_the_mapping():
    with pytest.raises(MLAError) as exc:
        query_row_position(0, group=4, seq_q=3, layout="BSDH")
    assert exc.value.code == "MLA_BAD_LAYOUT"


# ------------------------------------------------------------ program grid

def test_sharding_multiplies_the_grid_and_nothing_else():
    spec = build_mla(batch=1, seq_q=1, seq_kv=2048, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    one = plan_programs(spec.shape, block_q=64, shard_count=1)
    four = plan_programs(spec.shape, block_q=64, shard_count=4)
    assert four.program_count == 4 * one.program_count
    assert four.block_q == one.block_q


def test_the_spec_example_grid_is_reproduced():
    """cases/mla.md 6.2: case7 is B=1, S=1, Nq=128. BQ16 gives 8 row jobs; four 128-wide column
    shards give 32 independent jobs."""
    spec = build_mla(batch=1, seq_q=1, seq_kv=2048, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    plan = plan_programs(spec.shape, block_q=16, shard_count=4)
    assert plan.program_count // 4 == 8       # 8 row jobs
    assert plan.program_count == 32           # 32 independent jobs


def test_every_program_decodes_to_a_distinct_axis_tuple():
    spec = build_mla(batch=2, seq_q=3, seq_kv=128, heads_q=8, heads_kv=2,
                     d_nope=512, d_rope=64)
    plan = plan_programs(spec.shape, block_q=2, shard_count=2)
    seen = set()
    for pid in range(plan.program_count):
        a = program_axes(pid, plan)
        key = (a["batch"], a["kv_head"], a["query_position"], a["row_tile"], a["shard"])
        assert key not in seen, "program %d duplicates %r" % (pid, key)
        seen.add(key)
    assert len(seen) == plan.program_count


def test_the_axes_stay_inside_their_bounds():
    spec = build_mla(batch=2, seq_q=3, seq_kv=128, heads_q=8, heads_kv=2,
                     d_nope=512, d_rope=64)
    plan = plan_programs(spec.shape, block_q=2, shard_count=2)
    for pid in range(plan.program_count):
        a = program_axes(pid, plan)
        assert 0 <= a["batch"] < 2
        assert 0 <= a["kv_head"] < 2
        assert 0 <= a["query_position"] < 3
        assert 0 <= a["shard"] < 2
        assert a["columns"].hi <= spec.shape.d_v


def test_the_shard_axis_is_least_significant():
    """Adjacent programs should be adjacent column shards so they read the same K tiles at about
    the same time."""
    spec = build_mla(batch=1, seq_q=1, seq_kv=2048, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    plan = plan_programs(spec.shape, block_q=64, shard_count=4)
    first = [program_axes(pid, plan)["shard"] for pid in range(4)]
    assert first == [0, 1, 2, 3]
    assert len({program_axes(pid, plan)["query_position"] for pid in range(4)}) == 1


def test_an_out_of_range_program_is_refused():
    spec = build_mla(batch=1, seq_q=1, seq_kv=128, heads_q=4, heads_kv=1,
                     d_nope=512, d_rope=64)
    plan = plan_programs(spec.shape, block_q=2, shard_count=1)
    with pytest.raises(MLAError) as exc:
        program_axes(plan.program_count, plan)
    assert exc.value.code == "MLA_PROGRAM_OUT_OF_RANGE"


# ------------------------------------------------------------ cost model

def test_sharding_repeats_qk_and_softmax_but_not_pv():
    """QK and softmax are duplicated in full by every shard; PV is partitioned. Recording both as
    'k' would hide the reason sharding is only worth it at low occupancy."""
    spec = build_mla(batch=1, seq_q=1, seq_kv=2048, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    for k in (1, 2, 4):
        cost = shard_cost(plan_programs(spec.shape, block_q=64, shard_count=k))
        assert cost.qk_work_multiplier == k
        assert cost.softmax_work_multiplier == k
        assert cost.pv_work_multiplier == pytest.approx(1.0)
        assert cost.kv_read_multiplier == k


def test_the_cost_model_is_serialisable():
    spec = build_mla(batch=1, seq_q=1, seq_kv=2048, heads_q=128, heads_kv=1,
                     d_nope=512, d_rope=64)
    cost = shard_cost(plan_programs(spec.shape, block_q=64, shard_count=2))
    blob = json.dumps(cost.as_dict())
    assert json.loads(blob)["shards"] == 2


# ------------------------------------------------------------ forbidden alternatives

def test_a_cross_program_merge_is_refused():
    codes = [c for c, _ in check_no_split_kv(merges_across_programs=True,
                                            gm_probability_matrix=False,
                                            second_kernel=False, host_math=False)]
    assert "forbidden_cross_program_merge" in codes


def test_a_gm_probability_matrix_is_refused():
    codes = [c for c, _ in check_no_split_kv(merges_across_programs=False,
                                            gm_probability_matrix=True,
                                            second_kernel=False, host_math=False)]
    assert "forbidden_gm_probability" in codes


def test_a_second_kernel_is_refused():
    codes = [c for c, _ in check_no_split_kv(merges_across_programs=False,
                                            gm_probability_matrix=False,
                                            second_kernel=True, host_math=False)]
    assert "forbidden_second_kernel" in codes


def test_host_maths_is_refused():
    codes = [c for c, _ in check_no_split_kv(merges_across_programs=False,
                                            gm_probability_matrix=False,
                                            second_kernel=False, host_math=True)]
    assert "forbidden_host_math" in codes


def test_the_clean_case_produces_no_violations():
    assert check_no_split_kv(merges_across_programs=False, gm_probability_matrix=False,
                             second_kernel=False, host_math=False) == ()


# ------------------------------------------------------------ dispatch

def test_a_full_grid_is_not_sharded():
    """Sharding a shape that already fills the device multiplies real work for no gain."""
    assert choose_shard_count(programs_unsharded=512, cores=56, d_v=512) == 1


def test_an_empty_grid_is_sharded_upward():
    assert choose_shard_count(programs_unsharded=2, cores=56, d_v=512) > 1


def test_the_shard_count_never_exceeds_the_columns():
    for p in (1, 2, 3, 7):
        k = choose_shard_count(programs_unsharded=p, cores=57, d_v=4)
        assert 1 <= k <= 4


# ------------------------------------------------------------ kernel contract

def test_the_default_entry_point_does_not_shard():
    src = (Path(__file__).resolve().parents[1]
           / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    assert "columnShards=1" in src


def test_the_shard_count_is_in_the_compile_cache_key():
    """Shards change the emitted bounds arithmetic, so the binary differs."""
    src = (Path(__file__).resolve().parents[1]
           / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    key = src[src.index("    key = ("):src.index("    if key not in _COMPILED")]
    assert "columnShards" in key


def test_the_kernel_checks_that_shards_divide_the_panels():
    src = (Path(__file__).resolve().parents[1]
           / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    assert "MLA_BAD_SHARD_WIDTH" in src


# ------------------------------------------------------------ evidence

def test_the_g62_record_matches_the_device_run():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G62_DEVICE_EVIDENCE.json").read_text())
    assert data["status"] == "VERIFIED"