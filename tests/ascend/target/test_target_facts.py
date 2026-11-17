"""Verify Ascend Target facts projections integrate with core analysis."""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    MemoryRelationKind,
    ThroughputFacts,
)
from tilefoundry.target import TopologyFacts, TopologyLevelFacts
from tilefoundry_ascend import AscendTarget


def test_ascend_target_owns_memory_hierarchy_and_throughput_facts():
    target = AscendTarget()
    assert target.get_facts(ThroughputFacts).bandwidth_level == "gmem"
    hierarchy = target.get_facts(MemoryHierarchyFacts)
    assert {level.name: level.owner for level in hierarchy.explicit_levels} == {
        "host": "target",
        "gmem": "target",
        "tmem": "aic",
        "smem": "aiv",
        "rmem": "lane",
    }
    by_name = {level.name: level for level in hierarchy.explicit_levels}
    assert by_name["smem"].capacity_bytes == 248 * 1024
    simtvf_levels = {level.name: level for level in target.get_facts(MemoryHierarchyFacts, "simtvf").explicit_levels}
    assert simtvf_levels["smem"].capacity_bytes == 216 * 1024
    unres_levels = {level.name: level for level in target.get_facts(MemoryHierarchyFacts, "simdvf_unreserved").explicit_levels}
    assert unres_levels["smem"].capacity_bytes == 256 * 1024
    assert tuple(level.name for level in hierarchy.implicit_levels) == ("simt_dcache",)
    sharing = [r for r in hierarchy.relations if r.kind is MemoryRelationKind.SHARES_CAPACITY_WITH]
    assert len(sharing) == 1
    assert sharing[0].near == "simt_dcache"
    assert sharing[0].far == "smem"
    assert sharing[0].shared_capacity_bytes == 256 * 1024


def test_ascend_topology_facts():
    levels = AscendTarget().get_facts(TopologyFacts)
    assert levels.level("block") == TopologyLevelFacts("block", 28, 28)
    assert levels.level("aic") == TopologyLevelFacts("aic", 1, 28)
    assert levels.level("aiv") == TopologyLevelFacts("aiv", 2, 56)
    assert levels.level("lane") == TopologyLevelFacts("lane", 64, 56 * 64)
    assert levels.parallel() == levels.level("block")


def test_ascend_hbm_burst_bandwidth_curve():
    import pytest
    from tilefoundry_ascend.ascend.target.facts import ascend_space_facts

    target = AscendTarget()
    facts = ascend_space_facts(target)
    assert facts.hbm_burst_bandwidth_table[0][0] == 4096
    assert facts.hbm_burst_bandwidth_table[-1][0] == 65536
    bw_4k = facts.effective_hbm_bandwidth_bps(4096)
    bw_32k = facts.effective_hbm_bandwidth_bps(32768)
    assert bw_4k < bw_32k
    assert facts.dma_burst_efficiency(4096) < 0.70
    assert facts.dma_burst_efficiency(32768) == pytest.approx(1.0)
