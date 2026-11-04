#!/usr/bin/env bash
# Negative control for the receipt checker's own wording.
#
# The checker validates the SHAPE of a receipt and must never say that a kernel
# ran: it has not seen a kernel, and the receipt it reads is written by the same
# process it describes. A checker whose output claims execution is a checker
# whose output can be quoted as execution evidence.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/ok.json" <<'JSON'
{"markers":{"tilefoundry":"A","reference":"B"},"deviceMoves":1,
 "npuPeakMemoryBytes":1,"artifacts":{"object":0,"npubin":1},
 "synchronizeOk":true,"visibleDevices":1}
JSON

out="$("$HERE/check-receipt.sh" "$TMP/ok.json" 2>&1)"; rc=$?
printf '%s\n' "$out"
echo "exit: $rc"

fail=0
[ "$rc" -eq 0 ] || { echo "FAIL: a schema-valid receipt should pass"; fail=1; }

# The forbidden claim, in the words a reader would quote.
for phrase in "compiled kernel ran" "kernel ran" "attest" "device was used"; do
    case "$phrase" in
        attest|"device was used") ;;   # allowed only in a negation; checked below
    esac
done
if printf '%s\n' "$out" | grep -qi 'compiled kernel ran'; then
    echo "FAIL: the checker claims a compiled kernel ran; it can only validate shape"
    fail=1
fi
# A positive claim, not merely the word appearing inside a disclaimer.
if printf '%s\n' "$out" | grep -qi 'does not attest'; then
    echo "ok: the output explicitly disclaims attestation"
else
    echo "FAIL: the output does not disclaim attestation"
    fail=1
fi
if printf '%s\n' "$out" | grep -q 'SCHEMA-OK'; then
    echo "ok: the verdict names itself schema-level"
else
    echo "FAIL: the verdict does not name itself schema-level"
    fail=1
fi

[ "$fail" -eq 0 ] && { echo "RESULT: PASS"; exit 0; }
echo "RESULT: FAIL"; exit 1
