"""Plain-python self-checks for the Ascend950PR Target facts (no pytest needed)."""
from tilefoundry.analysis.facts import MemoryHierarchyFacts
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.target.facts import MemoryExecutionFacts, TopologyFacts

from tilefoundry_ascend.tf_target import DEFAULT_AIC, DEFAULT_AIV, UB_BYTES, Ascend950PRTarget


def check(name, condition):
    assert condition, name
    return name


def run():
    t = Ascend950PRTarget()
    ok = []
    # 1. serialization round-trip for default and nondefault counts
    expr = t.to_python()
    ok.append(check("default_to_python", expr.text == "Ascend950PRTarget()"))
    t2 = Ascend950PRTarget(aiv_count=32, aic_count=16)
    expr2 = t2.to_python()
    ok.append(check("nondefault_preserved", "aiv_count=32" in expr2.text and "aic_count=16" in expr2.text))
    ns = {}
    exec("from tilefoundry_ascend.tf_target import Ascend950PRTarget", ns)
    ok.append(check("nondefault_roundtrip", eval(expr2.text, ns) == t2))
    # 2. topology levels and ordering semantics
    topo = t.get_facts(TopologyFacts)
    ok.append(check("topology_names", tuple(x.name for x in topo.topologies) == ("aiv", "aic")))
    ok.append(check("parallel_level", topo.parallel_level == "aiv"))
    ok.append(check("physical_counts", topo.level("aiv").max_physical_units == DEFAULT_AIV))
    ok.append(check("memory_execution", t.get_facts(MemoryExecutionFacts).topologies == ("aiv",)))
    # 3. memory facts: aggregate rmem capacity unknown, ub nominal
    mem = t.get_facts(MemoryHierarchyFacts)
    ok.append(check("rmem_unknown", mem.explicit("rmem").capacity_bytes is None))
    ok.append(check("smem_nominal", mem.explicit("smem").capacity_bytes == UB_BYTES))
    ok.append(check("gm_target_owner", mem.explicit("gmem").owner == "target"))
    ok.append(check("l2_implicit", mem.implicit("l2") is not None))
    ok.append(check("l2_caches_gmem", mem.cached_level("l2") == "gmem"))
    # 4. storage owners: smem/rmem private to aiv, gmem target-wide
    owners = t.storage_owners
    ok.append(check("smem_owner", owners.owner[StorageKind.SMEM] == "aiv"))
    ok.append(check("rmem_owner", owners.owner[StorageKind.RMEM] == "aiv"))
    ok.append(check("smem_private_aiv", owners.private(StorageKind.SMEM, "aiv")))
    # AIC does not own UB: sibling engines are not a nesting relationship
    ok.append(check("smem_not_private_aic", not owners.private(StorageKind.SMEM, "aic")))
    # 5. l2 is not an allocatable StorageKind
    raised = False
    try:
        resolve_storage("l2")
    except ValueError:
        raised = True
    ok.append(check("l2_not_storagekind", raised))
    # 6. invalid counts rejected
    for bad in (0, -1, True):
        raised = False
        try:
            Ascend950PRTarget(aiv_count=bad)
        except ValueError:
            raised = True
        ok.append(check(f"reject_aiv_{bad}", raised))
    print({"checks": len(ok), "passed": ok})
    return ok


if __name__ == "__main__":
    run()
