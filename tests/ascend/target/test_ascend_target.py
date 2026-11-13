"""Pin Ascend target topology validation, facts, and hardware documents."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import Topology
from tilefoundry.target import TopologyFacts, TopologyLevelFacts, UnsupportedCapabilityError
from tilefoundry_ascend import AscendTarget
from tilefoundry_ascend.ascend.target import AscendSpaceFacts


def test_ascend_target_reports_and_validates_its_topology_levels():
    target = AscendTarget()
    levels = target.get_facts(TopologyFacts).topologies
    assert tuple(level.name for level in levels) == ("block", "aic", "aiv", "lane")
    assert target.get_facts(TopologyLevelFacts, "block").max_logical_units == 28
    assert target.get_facts(TopologyLevelFacts, "aic").max_logical_units == 1
    assert target.get_facts(TopologyLevelFacts, "aiv").max_logical_units == 2
    assert target.get_facts(TopologyLevelFacts, "lane").max_logical_units == 64
    assert AscendTarget("ascend.950pr") == target
    assert target.to_python().text == 'AscendTarget("ascend.950pr")'

    with pytest.raises(ValueError) as topology_error:
        target.validate_program_topology(Topology("cta", 4))
    assert "unsupported topology level 'cta'" in str(topology_error.value)
    with pytest.raises(UnsupportedCapabilityError, match="no Facts projection"):
        target.get_facts(TopologyLevelFacts, "cta")

    target.validate_program_topology(Topology("block", 28))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 28"):
        target.validate_program_topology(Topology("block", 29))


def test_ascend_space_facts_expose_exact_950pr_capacities():
    spaces = AscendTarget().get_facts(AscendSpaceFacts)
    assert spaces.ub_nominal_bytes == 256 * 1024
    assert spaces.ub_usable_bytes == 216 * 1024
    assert spaces.l1_bytes == 512 * 1024
    assert spaces.l0a_bytes == 64 * 1024
    assert spaces.l0b_bytes == 64 * 1024
    assert spaces.l0c_bytes == 256 * 1024
