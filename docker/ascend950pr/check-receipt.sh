#!/usr/bin/env bash
# Judge a device-run receipt. Report what is THERE; never assume what is not.
#
# Adopted from AgentGates tla-npu README "What the Gate requires". The rule that
# matters most: MISSING DATA IS FALSE. A receipt with no peak allocation is not
# "probably fine" -- it is a run with no evidence that the device allocated
# anything, which is indistinguishable from a run that never touched the device.
#
# Every absent or unparseable field is reported as false with the reason, so a
# reader can tell "the field said zero" from "the field was not there" -- a
# distinction the receipt format alone does not preserve.
#
# Exit codes
#   0  every required receipt field is satisfied
#   1  the receipt does not prove a device run
#   2  the receipt could not be read
set -uo pipefail

PROG="check-receipt.sh"
log() { printf '%s: %s\n' "$PROG" "$*" >&2; }

RECEIPT="${1:-}"
[ -n "$RECEIPT" ] || { log "usage: $0 RECEIPT.json"; exit 2; }
[ -r "$RECEIPT" ] || { log "receipt not readable: $RECEIPT"; exit 2; }

command -v python3 >/dev/null 2>&1 || { log "python3 not found"; exit 2; }

python3 - "$RECEIPT" <<'PY'
import json, sys

path = sys.argv[1]
try:
    with open(path) as handle:
        report = json.load(handle)
except Exception as error:
    print("receipt is not readable JSON: %s: %s" % (type(error).__name__, error))
    sys.exit(2)

if not isinstance(report, dict):
    print("receipt is not a JSON object, so no field can be read")
    sys.exit(2)

failures = []
checked = 0


def require(condition, message):
    """Record one requirement. Missing data is false, never assumed."""
    global checked
    checked += 1
    if not condition:
        failures.append(message)


# 1. Both markers, each DISTINGUISHED -- not one substring test. A plain marker
#    can contain the specific one, so a single containment test would credit both
#    routes from one line.
markers = report.get("markers") or {}
specific = markers.get("tilefoundry")
plain = markers.get("reference")
require(isinstance(specific, str) and bool(specific), "tilefoundry marker absent")
require(isinstance(plain, str) and bool(plain), "reference marker absent")
require(
    not (isinstance(specific, str) and isinstance(plain, str)
         and specific and plain and specific == plain),
    "the two markers are identical, so they do not distinguish the two routes",
)

# 2. Tensors really moved to the device, counted in the process that ran the
#    work. A parent-process counter would be structurally zero.
moves = report.get("deviceMoves")
require(isinstance(moves, int) and not isinstance(moves, bool),
        "deviceMoves missing or not an integer")
require(isinstance(moves, int) and not isinstance(moves, bool) and moves > 0,
        "deviceMoves is %r: no tensor was moved to the device" % (moves,))

# 3. The device actually allocated memory. Zero is not a pass: it means nothing
#    ran on the device, however cleanly the process exited.
peak = report.get("npuPeakMemoryBytes")
require(isinstance(peak, int) and not isinstance(peak, bool),
        "npuPeakMemoryBytes missing or not an integer")
require(isinstance(peak, int) and not isinstance(peak, bool) and peak > 0,
        "npuPeakMemoryBytes is %r: the device allocated nothing" % (peak,))

# 4. At least one compiled kernel image. A .o is counted but not required: the
#    Ascend backend does not emit one, so requiring it would be unsatisfiable
#    rather than strict.
artifacts = report.get("artifacts") or {}
npubin = artifacts.get("npubin")
objects = artifacts.get("object")
# Specifically an INTEGER >= 1. Python's bool is a subclass of int, so
# isinstance(True, int) is True and `true` would satisfy a naive check -- a
# receipt claiming npubin: true would pass while reporting no count at all.
require(isinstance(npubin, int) and not isinstance(npubin, bool),
        "artifacts.npubin must be an integer count, got %r" % (npubin,))
require(isinstance(npubin, int) and not isinstance(npubin, bool) and npubin >= 1,
        "artifacts.npubin is %r: no .npubin produced, the compiler emitted no kernel image"
        % (npubin,))

# 5. Synchronization succeeded; when it did not, the ORIGINAL error text is
#    carried rather than replaced by a summary of it.
sync_ok = report.get("synchronizeOk")
require(sync_ok is True,
        "device synchronization did not succeed (synchronizeOk=%r)" % (sync_ok,))
if sync_ok is not True:
    error = report.get("synchronizeError")
    require(isinstance(error, str) and bool(error.strip()),
            "synchronization failed and the original error text was not carried")

# 6. Exactly one device visible to the process. A filter that silently expanded
#    to every card would make a run on the wrong board look correct.
# An INTEGER COUNT equal to 1, not any single non-empty string. "banana" is one
# non-empty string and would satisfy a split-on-comma test; so would "0". The
# count is what the field must carry, and it must be exactly 1.
visible = report.get("visibleDevices")
require(isinstance(visible, int) and not isinstance(visible, bool),
        "visibleDevices must be an integer count of visible devices, got %r" % (visible,))
require(isinstance(visible, int) and not isinstance(visible, bool) and visible == 1,
        "visibleDevices is %r: expected exactly 1 visible device" % (visible,))

print("receipt: %s" % path)
print("fields checked: %d, failures: %d" % (checked, len(failures)))
print("deviceMoves=%r npuPeakMemoryBytes=%r npubin=%r objects=%r synchronizeOk=%r"
      % (moves, peak, npubin, objects, sync_ok))

if failures:
    print("RESULT: FAIL")
    for item in failures:
        print("  - %s" % item)
    sys.exit(1)

print("RESULT: PASS")
print("NOTE: a PASS here means a compiled kernel ran on one device. It is not a")
print("      claim about timing, which is shared-tenant and provisional.")
sys.exit(0)
PY