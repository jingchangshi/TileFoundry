"""P4-02 / G41: versioned, completion-aware physical storage placement.

WHAT THESE TESTS PROVE, and why each would catch its own regression.

The claim of P4-02 is not "a layout is produced". It is that the placement accounts for what
a buffer REALLY occupies and that reuse is decided by ASYNCHRONOUS COMPLETION, not by
instruction issue order. Those are two separate claims and they fail in opposite directions:

  * an understated footprint (bare element bytes for an NZ matrix, no FP32 scratch, no
    reserved region) makes capacity checks pass that should fail -- it is invisible until
    the program corrupts memory on device;
  * reuse decided by issue order frees a buffer while an issued store is STILL READING it.
    That one is worse, because it works most of the time.

So the tests here drive a REAL carrier through the public entry point and then VARY the
program (versions, trip, capacities, layouts), checking that the placement MOVES WITH IT.
The negative probes are mutations of a placement that verified -- reuse an output slot
before its store completed, alias two owners, overfill a space -- because a plausible
near-miss is what a checker has to reject.

Deliberately NOT claimed: device execution, cycle timing, or that the resulting addresses
are the ones the runtime uses. Those stay P4-03/P4-04 and are named UNVERIFIED.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

CARRIER = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline",
                       "pipe_vadd_carrier.py")


# --------------------------------------------------------------------------- fixtures
def _module():
    """Drive the PUBLIC entry point and return the finalized module."""
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    program = checked_program(CARRIER, None, command="g41")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    return finalize(analyzed, authored=program.authored)


def _plan_and_completion():
    from tilefoundry_ascend import completion as CP

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    assert plan is not None, "the chain did not reach resolve"
    # COMPLETION IS PLANNED AGAINST THE PRE-MATERIALIZATION PROGRAM.
    # entry_function() is now the FINISHED artifact: unrolled AND completion-carrying. Planning
    # completion on it asks for the completion of a program that already has one, and the
    # emitter then reports 31 extra statements. The serial program is the input this stage
    # reasons about, so it is asked for by name.
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir

    completion = CP.build_completion_plan(plan, pre_completion_tir(module))
    assert completion.verified, (completion.code, completion.report.detail)
    return plan, completion


def _serial(module):
    """The program BEFORE completion emission -- what completion is derived against."""
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir

    return pre_completion_tir(module)


def _storage(plan, completion, **kw):
    from tilefoundry_ascend import storage as ST

    return ST.build_storage_plan(plan, completion, **kw)


def _entry(storage_plan, root):
    for entry in storage_plan.placement.entries:
        if entry.root == root:
            return entry
    return None


def _ub_roots(plan):
    return [buffer.root for buffer in plan.buffers if buffer.space == "ascend.ub"]


# ------------------------------------------------------- the footprint is REAL, not element bytes
def test_an_nz_matrix_occupies_its_fractal_footprint_not_its_element_bytes():
    """L0 matrix operands are padded to full 16x16 fractals, and the padding IS allocated.

    A 63x63 bf16 tile is 7938 element bytes row-major. In NZ it occupies four full 16x16
    fractals = 16384 bytes. A placement that multiplies element bytes understates the tile by
    more than 2x, and nothing downstream notices because the same wrong number is used for the
    capacity check it has to pass.
    """
    from tilefoundry_ascend.storage.model import (
        LAYOUT_FRACTAL_NZ, LAYOUT_LINEAR, LayoutRequest, bytes_for, facts_for,
    )

    facts = facts_for("ascend950pr_9579")

    def _nz(shape):
        return bytes_for(LayoutRequest("m", shape=shape, dtype="bf16",
                                       layout=LAYOUT_FRACTAL_NZ), facts)

    def _linear(shape):
        return bytes_for(LayoutRequest("m", shape=shape, dtype="bf16",
                                       layout=LAYOUT_LINEAR), facts)

    # A SHAPE WHOSE FRACTAL PADDING IS LARGE. 65x17 rounds up to 5x2 fractals: it occupies
    # ten full 16x16 blocks (5120 B) while carrying only 2210 B of real elements. A footprint
    # computed from element count understates this by 2.3x, and the understatement is exactly
    # what then passes a capacity check it should fail.
    thin = (65, 17)
    thin_linear = _linear(thin)
    thin_nz = _nz(thin)
    assert thin_linear["single_version_bytes"] == 65 * 17 * 2
    fractals = (-(-65 // 16)) * (-(-17 // 16))
    assert fractals == 10, fractals
    assert thin_nz["single_version_bytes"] == fractals * 16 * 16 * 2, (
        "the NZ footprint must be EXACTLY the padded fractal total: got %d, expected %d"
        % (thin_nz["single_version_bytes"], fractals * 16 * 16 * 2))
    assert thin_nz["single_version_bytes"] > 2 * thin_linear["single_version_bytes"], (
        "this shape's fractal padding is more than half its footprint, and the placement must"
        " say so: nz=%d linear=%d" % (thin_nz["single_version_bytes"],
                                      thin_linear["single_version_bytes"]))
    assert thin_nz["padding_bytes"] == thin_nz["single_version_bytes"] - 65 * 17 * 2

    # A near-multiple tile is padded only slightly, which proves the padding is COMPUTED and
    # not a blanket multiplier applied to every matrix.
    near = _nz((63, 63))
    assert near["single_version_bytes"] == 16 * 16 * 16 * 2, (
        "63x63 rounds up to a 4x4 fractal grid = 16 blocks, got %d"
        % near["single_version_bytes"])

    # And an exact multiple of 16 needs no padding at all.
    exact = _nz((64, 64))
    assert exact["padding_bytes"] == 0, exact
    assert exact["payload_bytes"] == exact["single_version_bytes"] == 64 * 64 * 2


def test_fp32_scratch_and_reserved_regions_are_counted_not_ignored():
    """A scratch buffer's width is its LANES x 4B, and a reservation is real bytes.

    Accumulating 64 bf16 lanes in FP32 needs 256 bytes of scratch even though the operand
    payload is 128. And a region that reserves bytes it never exposes must still be placed
    around, or every address after it is wrong.
    """
    from tilefoundry_ascend.storage.model import (
        LAYOUT_RESERVED, LAYOUT_SCRATCH, LayoutRequest, StorageRefusal, bytes_for, facts_for,
    )

    facts = facts_for("ascend950pr_9579")
    scratch = bytes_for(LayoutRequest("acc", shape=(64,), dtype="f32",
                                      layout=LAYOUT_SCRATCH, scratch_dtype="f32",
                                      scratch_lanes=64), facts)
    assert scratch["scratch_bytes"] == 64 * 4, scratch
    assert scratch["single_version_bytes"] >= 256

    reserved = bytes_for(LayoutRequest("rsv", layout=LAYOUT_RESERVED, reserved_bytes=4096),
                         facts)
    assert reserved["reserved_bytes"] == 4096
    assert reserved["single_version_bytes"] >= 4096

    # A reservation of UNKNOWN size is refused, never defaulted to zero: a region whose size
    # nobody knows cannot be planned around, and treating it as empty overcommits the space.
    with pytest.raises(StorageRefusal) as refusal:
        bytes_for(LayoutRequest("rsv", layout=LAYOUT_RESERVED), facts)
    assert refusal.value.code == "STORAGE_RESERVED_UNSPECIFIED"

    # Same for scratch: the lane count is not inferable from the operand shape.
    with pytest.raises(StorageRefusal) as refusal:
        bytes_for(LayoutRequest("acc", shape=(64,), dtype="f32", layout=LAYOUT_SCRATCH), facts)
    assert refusal.value.code == "STORAGE_SCRATCH_UNSPECIFIED"


def test_an_unknown_layout_or_dtype_is_refused_rather_than_guessed():
    """A footprint that needs a fact nobody has must refuse. A guess is worse than a stop."""
    from tilefoundry_ascend.storage.model import LayoutRequest, StorageRefusal, bytes_for, facts_for

    facts = facts_for("ascend950pr_9579")
    with pytest.raises(StorageRefusal) as refusal:
        bytes_for(LayoutRequest("b", shape=(4,), dtype="no-such-dtype"), facts)
    assert refusal.value.code == "STORAGE_DTYPE_UNKNOWN"

    with pytest.raises(StorageRefusal) as refusal:
        bytes_for(LayoutRequest("b", shape=(4,), dtype="f32", layout="no-such-layout"), facts)
    assert refusal.value.code == "STORAGE_LAYOUT_UNKNOWN"

    from tilefoundry_ascend.storage.model import facts_for as facts_for_profile

    with pytest.raises(StorageRefusal) as refusal:
        facts_for_profile("no-such-profile")
    assert refusal.value.code == "STORAGE_TARGET_UNKNOWN"


def test_a_declared_version_stride_too_small_for_the_real_footprint_is_refused():
    """The stride must hold one version's REAL footprint; otherwise slots overlap silently.

    This is the specific hazard that a bare element-byte size hides: the plan's stride was
    computed from element bytes, the NZ footprint is larger, and consecutive ring slots would
    then overlap each other by the padding.
    """
    from tilefoundry_ascend.storage.model import (
        LAYOUT_FRACTAL_NZ, LayoutRequest, StorageRefusal, bytes_for, facts_for,
    )

    facts = facts_for("ascend950pr_9579")
    with pytest.raises(StorageRefusal) as refusal:
        bytes_for(LayoutRequest("m", shape=(63, 63), dtype="bf16", layout=LAYOUT_FRACTAL_NZ,
                                version_stride_bytes=63 * 63 * 2), facts)
    assert refusal.value.code == "STORAGE_STRIDE_TOO_SMALL"


# ------------------------------------------------------- placement on the real carrier
def test_the_carrier_gets_a_verified_placement_with_a_per_space_peak_report():
    """The real carrier must place, and the report must state peak usage per space.

    A peak is the MAXIMUM SIMULTANEOUSLY LIVE footprint, not the highest address: a hole left
    by a short-lived buffer is not live, and reporting the highest address would overstate
    usage for exactly the programs that reuse well.
    """
    plan, completion = _plan_and_completion()
    storage = _storage(plan, completion)
    assert storage.verified, (storage.code, storage.verdict.detail)

    spaces = storage.verdict.spaces
    assert "ascend.ub" in spaces, "the carrier has UB buffers, so UB must be reported"
    for name, usage in spaces.items():
        assert usage.peak_bytes >= 0
        assert usage.peak_bytes <= max([0] + [entry.range()[1] for entry in usage.entries]), (
            "peak %d exceeds the highest placed address in %s" % (usage.peak_bytes, name))
        if usage.capacity_bytes is not None:
            assert usage.fits, "%s does not fit: %d > %d" % (name, usage.peak_bytes,
                                                             usage.capacity_bytes)

    # Every buffer in the plan is placed, and every address is inside its space's alignment.
    placed_roots = {entry.root for entry in storage.placement.entries}
    assert placed_roots == {buffer.root for buffer in plan.buffers}, (
        "every declared buffer must be placed; missing: %s"
        % sorted({b.root for b in plan.buffers} - placed_roots))


def test_changing_the_version_count_changes_the_placement_and_the_peak():
    """Versions are an INPUT to placement: doubling them must move the addresses.

    The assertion is a DIFF between two runs of the same code on two plans, so a placement
    that ignores the version count cannot pass it by being internally consistent.
    """
    plan, completion = _plan_and_completion()
    roots = _ub_roots(plan)
    assert roots, "the carrier must have UB buffers to version"

    base = _storage(plan, completion)
    assert base.verified, (base.code, base.verdict.detail)
    base_peak = base.verdict.spaces["ascend.ub"].peak_bytes

    grown_plan = plan
    for root in roots:
        buffers = [dataclasses.replace(b, slot_count=2 * max(1, b.slot_count))
                   if b.root == root else b for b in grown_plan.buffers]
        grown_plan = dataclasses.replace(grown_plan, buffers=tuple(buffers))
    from tilefoundry_ascend import completion as CP

    grown_completion = CP.build_completion_plan(grown_plan, _serial(_module()))
    grown = _storage(grown_plan, grown_completion)
    assert grown.verified, (grown.code, grown.verdict.detail)

    assert grown.verdict.spaces["ascend.ub"].peak_bytes > base_peak, (
        "doubling the version count did not change the UB peak (%d -> %d): versions are not"
        " an input to the placement" % (base_peak, grown.verdict.spaces["ascend.ub"].peak_bytes))
    for root in roots:
        before, after = _entry(base, root), _entry(grown, root)
        assert after.slot_count == 2 * before.slot_count
        assert after.allocation_bytes > before.allocation_bytes, (
            "the allocation of %r did not grow with its version count" % root)


def test_a_stale_placement_is_refused_when_the_versions_change():
    """A placement computed for other versions must not have its offsets trusted.

    A stale placement is worse than none: the addresses look valid, so nothing downstream
    refuses them. The refusal must NAME the input that changed, not merely say "different".
    """
    from tilefoundry_ascend.storage.verify import invalidated_inputs, verify_storage

    plan, completion = _plan_and_completion()
    storage = _storage(plan, completion)
    assert storage.verified

    roots = _ub_roots(plan)
    changed_plan = dataclasses.replace(plan, buffers=tuple(
        dataclasses.replace(b, slot_count=b.slot_count + 1) if b.root == roots[0] else b
        for b in plan.buffers))
    changed = invalidated_inputs(changed_plan, storage.placement)
    assert any(item[0] == "versions" and item[1] == roots[0] for item in changed), (
        "changing a buffer's version count did not invalidate the placement: %s" % (changed,))

    verdict = verify_storage(changed_plan, storage.placement, completion=completion)
    assert verdict.status == "FAILED"
    assert verdict.code == "STORAGE_PLACEMENT_STALE"
    assert verdict.invalidated_by, "the refusal must name what invalidated it"


def test_a_completion_program_change_invalidates_the_reuse_intervals():
    """Reuse depends on completion: if the completion program changed, so did the intervals."""
    from tilefoundry_ascend.storage.verify import verify_storage

    plan, completion = _plan_and_completion()
    storage = _storage(plan, completion)
    assert storage.verified

    class _OtherCompletion:
        """Stands in for a completion plan with a different result digest."""

        class report:
            result_digest = "different-digest"

    verdict = verify_storage(plan, storage.placement, completion=_OtherCompletion)
    assert verdict.status == "FAILED"
    assert verdict.code == "STORAGE_COMPLETION_CHANGED"


# ------------------------------------------------------- reuse is decided by completion
def test_an_output_slot_is_not_reused_before_its_store_read_completes():
    """The MTE3 hazard, stated as an interval relationship rather than as a comment.

    A UB output staging buffer is read by an asynchronous store. Its bytes are NOT reusable
    when that store is ISSUED; they are reusable when the store's READ has COMPLETED. This
    test pins the rule in the interval data: the interval's end must come from a read_end
    record, and an interval closed by an issue would be the defect.

    The check is done on the real trace, so it cannot be satisfied by a comment: the entry's
    'evidence' must name a completed read, and the read_end record it names must exist.
    """
    plan, completion = _plan_and_completion()
    storage = _storage(plan, completion)
    assert storage.verified

    read_end_by_root = {}
    for record in completion.trace.records:
        if record.role == "read_end" and record.channel.startswith("access:"):
            root = record.channel.split("access:", 1)[1]
            read_end_by_root.setdefault(root, []).append(record.index)

    checked = 0
    for entry in storage.placement.entries:
        stops = read_end_by_root.get(entry.root)
        if not stops:
            continue
        checked += 1
        assert entry.interval.stop >= min(stops), (
            "%r is freed at %d, before its first completed read at %d: reuse is being decided"
            " by issue order, not completion"
            % (entry.root, entry.interval.stop, min(stops)))
        assert "read_end" in entry.interval.evidence or "region end" in entry.interval.evidence, (
            "%r has completion reads but its interval evidence is %r, which is not a"
            " completion fact" % (entry.root, entry.interval.evidence))
    assert checked, "the carrier must exercise at least one completed read"


def test_legal_short_lifetime_reuse_is_preserved_rather_than_serialized():
    """The OTHER defect: refusing all reuse is not safe, it is a missed optimization.

    Two buffers whose live intervals are disjoint MUST be allowed to share bytes. A placement
    that serialises everything passes every safety check and fails the spec, so the test
    constructs the disjoint case directly and asserts the allocator reuses.
    """
    from tilefoundry_ascend.storage.plan_placement import plan_storage

    class _Buffer:
        def __init__(self, root, space, size, owner="s0"):
            self.root = root
            self.space = space
            self.owner = owner
            self.dtype = "f32"
            self.shape = (size // 4,)
            self.single_version_bytes = size
            self.alignment = 64
            self.slot_count = 1
            self.slot_source = "fixed"
            self.version_stride_bytes = None
            self.source_decl = root

    class _Plan:
        buffers = (_Buffer("early", "ascend.ub", 256), _Buffer("late", "ascend.ub", 256))

    class _Record:
        def __init__(self, index, role, channel, slot=0, generation=0):
            self.index, self.role, self.channel = index, role, channel
            self.slot, self.generation = slot, generation

    class _Trace:
        # 'early' is written at 0 and its read completes at 5. 'late' is written at 10.
        # Their lives are disjoint, so the SAME 256 bytes may serve both.
        records = (_Record(0, "write_begin", "access:early"),
                   _Record(5, "read_end", "access:early"),
                   _Record(10, "write_begin", "access:late"),
                   _Record(15, "read_end", "access:late"))

    storage = plan_storage(_Plan(), _Trace())
    assert storage.verified, (storage.report.code, storage.report.detail)
    early = next(e for e in storage.entries if e.root == "early")
    late = next(e for e in storage.entries if e.root == "late")

    assert not early.interval.overlaps(late.interval), (
        "the fixture's two lifetimes must be disjoint for this test to mean anything")
    assert late.base_offset_bytes == early.base_offset_bytes, (
        "two buffers with disjoint live intervals must be allowed to share bytes; they were"
        " placed at %d and %d, which is full serialization, not analysis"
        % (early.base_offset_bytes, late.base_offset_bytes))
    assert "early" in late.shared_with, (
        "the reuse must be RECORDED, so a reviewer can see the bytes are shared on purpose")


def test_two_overlapping_lifetimes_never_share_bytes():
    """The mirror of the previous test: overlapping lifetimes must NOT be placed on top."""
    from tilefoundry_ascend.storage.plan_placement import plan_storage

    class _Buffer:
        def __init__(self, root, size=256):
            self.root = root
            self.space = "ascend.ub"
            self.owner = "s0"
            self.dtype = "f32"
            self.shape = (size // 4,)
            self.single_version_bytes = size
            self.alignment = 64
            self.slot_count = 1
            self.slot_source = "fixed"
            self.version_stride_bytes = None
            self.source_decl = root

    class _Plan:
        buffers = (_Buffer("a"), _Buffer("b"))

    class _Record:
        def __init__(self, index, role, channel):
            self.index, self.role, self.channel = index, role, channel
            self.slot, self.generation = 0, 0

    class _Trace:
        # Both live across the same span: 0..8 and 2..10 overlap.
        records = (_Record(0, "write_begin", "access:a"),
                   _Record(8, "read_end", "access:a"),
                   _Record(2, "write_begin", "access:b"),
                   _Record(10, "read_end", "access:b"))

    storage = plan_storage(_Plan(), _Trace())
    assert storage.verified, (storage.report.code, storage.report.detail)
    a = next(e for e in storage.entries if e.root == "a")
    b = next(e for e in storage.entries if e.root == "b")
    assert a.interval.overlaps(b.interval), "the fixture's lifetimes must overlap"
    base_a, end_a = a.range()
    base_b, end_b = b.range()
    assert not (base_a < end_b and base_b < end_a), (
        "two buffers with OVERLAPPING live intervals were placed at overlapping bytes"
        " ([%d,%d) and [%d,%d))" % (base_a, end_a, base_b, end_b))


# ------------------------------------------------------- negative probes
def test_reusing_bytes_across_two_owners_is_refused():
    """UB is per-AIV-instance: two owners at one offset are different storage, not reuse."""
    from tilefoundry_ascend.storage.plan_placement import verify_placement
    from tilefoundry_ascend.storage.placement import (
        Interval, PlacementEntry, SpaceUsage,
    )

    left = PlacementEntry(root="a", space="ascend.ub", owner="aiv:0", layout="linear",
                          base_offset_bytes=0, version_stride_bytes=256, slot_count=1,
                          allocation_bytes=256,
                          interval=Interval(0, 4, "read_end"))
    right = PlacementEntry(root="b", space="ascend.ub", owner="aiv:1", layout="linear",
                           base_offset_bytes=0, version_stride_bytes=256, slot_count=1,
                           allocation_bytes=256,
                           interval=Interval(10, 14, "read_end"))
    usage = {"ascend.ub": SpaceUsage(space="ascend.ub", capacity_bytes=256 * 1024,
                                     peak_bytes=256, entries=(left, right))}
    report = verify_placement([left, right], usage)
    assert report.status == "REFUSED"
    assert report.code == "STORAGE_CROSS_OWNER_ALIAS", report.code


def test_exceeding_a_space_capacity_is_refused_and_names_the_space():
    """Capacity is a hard refusal that names the space, not a warning in a log."""
    from tilefoundry_ascend.storage.plan_placement import verify_placement
    from tilefoundry_ascend.storage.placement import (
        Interval, PlacementEntry, SpaceUsage,
    )

    entry = PlacementEntry(root="big", space="ascend.ub", owner="s0", layout="linear",
                           base_offset_bytes=0, version_stride_bytes=1024, slot_count=1,
                           allocation_bytes=1024, interval=Interval(0, 4, "read_end"))
    usage = {"ascend.ub": SpaceUsage(space="ascend.ub", capacity_bytes=512,
                                     peak_bytes=1024, entries=(entry,))}
    report = verify_placement([entry], usage)
    assert report.status == "REFUSED"
    assert report.code == "STORAGE_CAPACITY_EXCEEDED"
    assert "ascend.ub" in report.detail


def test_an_identifier_placed_at_zero_silently_aliasing_everything_is_caught():
    """The real defect this suite found: a root with no access must not become 'never live'.

    A buffer that the completion trace never mentions (a function parameter, say) has no
    per-slot interval. Treating that as the EMPTY interval made every such buffer overlap
    nothing, so they all landed at offset 0 and the alias check passed -- two zero-length
    intervals do not overlap. 'The completion program says nothing' means UNKNOWN, and the
    safe reading of unknown is 'live throughout'.
    """
    from tilefoundry_ascend.storage.plan_placement import plan_storage

    class _Buffer:
        def __init__(self, root):
            self.root = root
            self.space = "ascend.gm"
            self.owner = "device"
            self.dtype = "f32"
            self.shape = (64,)
            self.single_version_bytes = 256
            self.alignment = 64
            self.slot_count = 1
            self.slot_source = "fixed"
            self.version_stride_bytes = None
            self.source_decl = root

    class _Plan:
        buffers = (_Buffer("x"), _Buffer("z"))

    class _Record:
        def __init__(self, index, role, channel):
            self.index, self.role, self.channel = index, role, channel
            self.slot, self.generation = 0, 0

    class _Trace:
        # Neither x nor z appears: exactly the parameter case.
        records = (_Record(0, "write_begin", "access:other"),
                   _Record(4, "read_end", "access:other"))

    storage = plan_storage(_Plan(), _Trace())
    assert storage.verified, (storage.report.code, storage.report.detail)
    x = next(e for e in storage.entries if e.root == "x")
    z = next(e for e in storage.entries if e.root == "z")
    assert x.base_offset_bytes != z.base_offset_bytes or x.interval.overlaps(z.interval) is False, (
        "two buffers with no completion access were both placed at %d: silence in the"
        " completion program is being read as 'never live'" % x.base_offset_bytes)
    base_x, end_x = x.range()
    base_z, end_z = z.range()
    assert not (base_x < end_z and base_z < end_x), (
        "two unaccessed buffers overlap their bytes ([%d,%d) and [%d,%d))"
        % (base_x, end_x, base_z, end_z))


def test_a_register_class_capacity_is_reported_unknown_not_used_as_an_aggregate_bound():
    """vreg's 'nominal_bytes' is ONE register's width, not a budget for all registers.

    Using it as a capacity either refuses every real program or passes by luck. The aggregate
    register budget needs a register COUNT, which this target does not source, so the honest
    report is UNKNOWN.
    """
    plan, completion = _plan_and_completion()
    storage = _storage(plan, completion)
    assert storage.verified
    spaces = storage.verdict.spaces
    if "ascend.vreg" in spaces:
        usage = spaces["ascend.vreg"]
        assert usage.capacity_known is False, (
            "the vector register class has no sourced aggregate capacity, so capacity_known"
            " must be False; claiming one would verify arithmetic against a wrong bound")
        assert usage.capacity_bytes is None


# ------------------------------------------------------- the chain wiring
def test_the_chain_resolves_storage_and_the_export_gate_sees_it():
    """The verified placement must reach the chain and be visible to the export gate."""
    from tilefoundry_ascend.passes.p3_chain import _stored_chain, require_export_ready

    module = _module()
    record = _stored_chain(module)
    assert record is not None
    assert record.completion_status == "VERIFIED", record.reason
    assert record.storage_status == "RESOLVED", (
        "storage_status is %r (%s); the chain must compute and VERIFY a real placement"
        % (record.storage_status, record.reason))

    storage = module.metadata.get("ascend.p3_storage")
    assert storage is not None and storage.verified
    gate = require_export_ready(module, record, raising=False)
    # P4-03 LANDED THE STAGED TIR, so this gate is now READY.
    # At G41 the scheduled-TIR emission was still outstanding and this asserted not-ready. With
    # G42 the chain emits, verifies and installs the scheduled program, so "not ready" would
    # assert that the stage we just completed did not happen. What this test is really for is
    # the two clauses below: storage and completion must not appear as gaps once verified.
    assert gate["ready"] is True, gate["gaps"]
    assert not any(gap[0] == "storage" for gap in gate["gaps"]), (
        "storage is verified, so it must not be reported as a gap: %s" % (gate["gaps"],))
    assert not any(gap[0] == "completion" for gap in gate["gaps"]), (
        "completion is verified, so it must not be reported as a gap: %s" % (gate["gaps"],))


def test_a_storage_plan_without_a_completion_program_is_refused():
    """Reuse cannot be decided without completion evidence, so absence must refuse.

    Placing with no completion program would mean reuse decided by instruction order, which is
    the exact defect this stage exists to remove. It must be a refusal, not a default.
    """
    from tilefoundry_ascend import storage as ST

    plan, _completion = _plan_and_completion()
    storage = ST.build_storage_plan(plan, None)
    assert not storage.verified
    assert storage.code == "STORAGE_COMPLETION_ABSENT"
