"""Verify Ascend Target facts projections integrate with core analysis."""

from __future__ import annotations

from tilefoundry.analysis.facts import MemoryHierarchyFacts, ThroughputFacts
from tilefoundry.target import TopologyFacts, TopologyLevelFacts
from tilefoundry_ascend import AscendTarget


def test_ascend_target_owns_memory_hierarchy_and_throughput_facts():
    target = AscendTarget()
    assert target.get_facts(ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in target.get_facts(MemoryHierarchyFacts).explicit_levels
    } == {
        "host": "target",
        "gmem": "target",
        "tmem": "aic",
        "smem": "aiv",
        "rmem": "lane",
    }


def test_ascend_topology_facts():
    levels = AscendTarget().get_facts(TopologyFacts)
    assert levels.level("block") == TopologyLevelFacts("block", 28, 28)
    assert levels.level("aic") == TopologyLevelFacts("aic", 1, 28)
    assert levels.level("aiv") == TopologyLevelFacts("aiv", 2, 56)
    assert levels.level("lane") == TopologyLevelFacts("lane", 64, 56 * 64)
    assert levels.parallel() == levels.level("block")
