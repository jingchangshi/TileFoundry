"""P1: memory-space semantics, capacity model, classification and refusal.

Checks the CONTRACT, not a program: every space states its role/owner/capacity, a
classified access lands in exactly one space, a buffer that fits none is refused with a
reason, and the alias/transfer rules from memory.py hold. No device, no Torch.

Run with the interpreter of record:
    PYTHONPATH=<repo>/src:<repo>/targets/ascend \
      python targets/ascend/tests/test_p1_placement.py
Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import sys

import tilefoundry_ascend  # noqa: F401

from tilefoundry_ascend.memory import (
    MemorySpaceRef,
    SPACE_FACTS,
    assert_transfer_legal,
    transfer_contract,
)
from tilefoundry_ascend.placement import (
    AccessSpec,
    GMEM,
    L1,
    PlacementRefusal,
    RMEM,
    ROLE_AIC_OPERAND,
    ROLE_PARAM,
    ROLE_PRIVATE_TEMP,
    ROLE_RESULT,
    ROLE_SCALAR,
    ROLE_STAGED_TILE,
    ROLE_VERSIONED_SLOT,
    SLAB,
    SMEM,
    classify,
    place,
    require_known_capacity,
    space_semantics,
    space_table,
)


def check_space_table_complete() -> None:
    table = space_table()
    assert set(table) == {GMEM, SMEM, RMEM, L1, SLAB}, sorted(table)
    for space, entry in table.items():
        assert entry["role"], f"{space} has no stated role"
        assert entry["physical"] in SPACE_FACTS, f"{space} physical space not registered"
        assert entry["capacity_rule"], f"{space} has no capacity rule"
    # UB states nominal and provenance; rmem aggregate budget is explicitly unknown.
    assert table[SMEM]["nominal_bytes"] == 256 * 1024
    assert table[SMEM]["capacity_known"] is False, "usable UB capacity must not be fabricated"
    assert table[RMEM]["capacity_known"] is False, "register budget must be stated unknown"


def check_classification() -> None:
    cases = [
        (AccessSpec("x", ROLE_PARAM, 8), GMEM),
        (AccessSpec("out", ROLE_RESULT, 8), GMEM),
        (AccessSpec("tile", ROLE_STAGED_TILE, 8), SMEM),
        (AccessSpec("tmp", ROLE_PRIVATE_TEMP, 8, inside_vf=True), RMEM),
        (AccessSpec("eps", ROLE_SCALAR, 4, inside_vf=True), RMEM),
        (AccessSpec("ring", ROLE_VERSIONED_SLOT, 8, versions=2), SLAB),
        (AccessSpec("w", ROLE_AIC_OPERAND, 8, domain="aic"), L1),
    ]
    for access, expected in cases:
        got = classify(access)
        assert got == expected, f"{access.name}: {got} != {expected}"


def check_refusals() -> None:
    def refused(access, needle):
        try:
            classify(access)
        except PlacementRefusal as error:
            text = str(error)
            assert needle in text, f"refused for the wrong reason: {text}"
            return
        raise AssertionError(f"expected a refusal for {access.name}")

    refused(AccessSpec("l2buf", ROLE_PARAM, 8, wants="ascend.l2"), "implicit cache")
    refused(AccessSpec("l1buf", ROLE_STAGED_TILE, 8, wants="ascend.l1"), "AIC-local")
    refused(AccessSpec("tmpo", ROLE_PRIVATE_TEMP, 8, inside_vf=False), "VF-private")
    refused(AccessSpec("aicbuf", ROLE_AIC_OPERAND, 8, domain="aiv"), "AIC operand")
    refused(AccessSpec("weird", "no-such-role", 8), "fits no declared space")


def check_capacity() -> None:
    # A staged tile larger than UB usable capacity must refuse, not overflow silently.
    try:
        place([AccessSpec("huge", ROLE_STAGED_TILE, 300 * 1024)])
    except PlacementRefusal as error:
        assert "exceeds capacity bound" in str(error), str(error)
    else:
        raise AssertionError("oversized smem placement was accepted")
    # Slab accounting multiplies by versions.
    placed = place([AccessSpec("ring", ROLE_VERSIONED_SLOT, 1024, versions=4)])
    assert placed[SLAB]["bytes"] == 4096, placed
    assert placed[SLAB]["slots"] == 4, placed
    # Unknown capacity is refused when a HARD bound is requested, not fabricated.
    try:
        require_known_capacity([AccessSpec("tmp", ROLE_PRIVATE_TEMP, 8, inside_vf=True)])
    except PlacementRefusal as error:
        assert "UNSOURCED" in str(error), str(error)
    else:
        raise AssertionError("register capacity was treated as known")


def check_alias_semantics() -> None:
    # Two DIFFERENT AIV instances at the same offset are different storage.
    a3 = MemorySpaceRef("ascend.ub", "aiv:3")
    a7 = MemorySpaceRef("ascend.ub", "aiv:7")
    assert a3.alias_key() != a7.alias_key(), "different AIV UBs must not alias"
    # The same owned root at the same space+owner aliases (slices of one root).
    same = MemorySpaceRef("ascend.ub", "aiv:3")
    assert a3.alias_key() == same.alias_key(), "same root must alias"
    # GM is device-global: owner is normalised away.
    assert MemorySpaceRef("ascend.gm", "aiv:3").alias_key() == (
        "ascend.gm",
        None,
    ), "GM must be device-global"


def check_transfer_legality() -> None:
    assert transfer_contract("ascend.gm", "ascend.ub").legal is True
    assert transfer_contract("ascend.ub", "ascend.gm").legal is True
    try:
        assert_transfer_legal("ascend.gm", "ascend.l0a")
    except ValueError as error:
        assert "no GM->L0A direct path" in str(error), str(error)
    else:
        raise AssertionError("GM->L0A direct path was accepted")
    try:
        assert_transfer_legal("ascend.l1", "ascend.gm")
    except ValueError as error:
        assert "no L1->GM direct path" in str(error), str(error)
    else:
        raise AssertionError("L1->GM direct path was accepted")
    # L2 is an implicit cache: no transfer edge may name it.
    assert transfer_contract("ascend.ub", "ascend.l2").legal is False
    assert SPACE_FACTS["ascend.l2"].allocatable is False


CHECKS = [
    ("space_table_complete", check_space_table_complete),
    ("classification", check_classification),
    ("refusals", check_refusals),
    ("capacity", check_capacity),
    ("alias_semantics", check_alias_semantics),
    ("transfer_legality", check_transfer_legality),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}")
            print(f"     {error}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    if failed:
        print(f"{failed}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"{len(CHECKS)}/{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
