"""Gmem dtype-view verifier: sound views accepted; misaligned/out-of-bounds/overlapping
WRITES refused with NAMED reasons. Overlap is ACCESS-BASED, not dtype-based.
Pure Python, no device.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import sys

from tilefoundry.ir.types.dtype_view import (
    DtypeViewError,
    DtypeViewSpec,
    validate_view,
    validate_views,
)


def refused(reason: str, view: DtypeViewSpec) -> None:
    try:
        validate_view(view)
    except DtypeViewError as error:
        assert error.reason == reason, f"reason {error.reason!r} != {reason!r}: {error}"
    else:
        raise AssertionError(f"expected a {reason} refusal")


def check_sound_alias() -> None:
    # 64 bf16 elements = 128 B; 32 f32 elements = 128 B, offset 0, aligned to 4.
    validate_view(DtypeViewSpec("x", "f32", 4, 0, 32, 4, 128, access="read"))


def check_misaligned() -> None:
    refused("misaligned", DtypeViewSpec("x", "f32", 4, 2, 8, 4, 128, access="read"))


def check_out_of_bounds() -> None:
    refused("out-of-bounds", DtypeViewSpec("x", "f32", 4, 0, 64, 4, 128, access="read"))


def check_read_read_cross_dtype_allowed() -> None:
    """PINNED: f32 read [0,64) + bf16 read [32,96) overlap, different dtypes -> ALLOWED."""
    f32_read = DtypeViewSpec("x", "f32", 4, 0, 16, 4, 128, access="read")
    bf16_read = DtypeViewSpec("x", "bf16", 2, 32, 32, 2, 128, access="read")
    validate_views([f32_read, bf16_read])


def check_same_dtype_write_read_forbidden() -> None:
    """PINNED: f32 write [0,64) + f32 read [32,96) overlap, SAME dtype -> FORBIDDEN."""
    f32_write = DtypeViewSpec("x", "f32", 4, 0, 16, 4, 128, access="write")
    f32_read = DtypeViewSpec("x", "f32", 4, 32, 16, 4, 128, access="read")
    try:
        validate_views([f32_write, f32_read])
    except DtypeViewError as error:
        assert error.reason == "overlapping-write", error
    else:
        raise AssertionError("same-dtype overlapping writer/reader was accepted")


def check_write_write_forbidden() -> None:
    first = DtypeViewSpec("x", "f32", 4, 0, 16, 4, 128, access="write")
    second = DtypeViewSpec("x", "f32", 4, 8, 16, 4, 128, access="write")
    try:
        validate_views([first, second])
    except DtypeViewError as error:
        assert error.reason == "overlapping-write", error
    else:
        raise AssertionError("two overlapping writers were accepted")


def check_overlapping_read_same_dtype_allowed() -> None:
    first = DtypeViewSpec("x", "f32", 4, 0, 16, 4, 128, access="read")
    second = DtypeViewSpec("x", "f32", 4, 32, 16, 4, 128, access="read")
    validate_views([first, second])


def check_same_shape_widening_is_refused() -> None:
    """The RMSNorm case: bf16 root read as f32 with the SAME extent needs 2x the bytes."""
    refused("out-of-bounds", DtypeViewSpec("x", "f32", 4, 0, 64, 4, 128, access="read"))


def check_access_is_required() -> None:
    """A view cannot be constructed without an access kind: no silent read default."""
    try:
        DtypeViewSpec("x", "f32", 4, 0, 16, 4, 128)
    except TypeError:
        return
    raise AssertionError("DtypeViewSpec accepted a view without an access kind")


CHECKS = [
    ("sound_alias", check_sound_alias),
    ("misaligned", check_misaligned),
    ("out_of_bounds", check_out_of_bounds),
    ("read_read_cross_dtype_allowed", check_read_read_cross_dtype_allowed),
    ("same_dtype_write_read_forbidden", check_same_dtype_write_read_forbidden),
    ("write_write_forbidden", check_write_write_forbidden),
    ("overlapping_read_same_dtype_allowed", check_overlapping_read_same_dtype_allowed),
    ("same_shape_widening_is_refused", check_same_shape_widening_is_refused),
    ("access_is_required", check_access_is_required),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error}")
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
