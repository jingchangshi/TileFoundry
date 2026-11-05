#!/bin/sh
# DEVICE PRECONDITION GATE (HANDOFF section 4 policy).
#
# Shared card is NORMAL. Multiple users/processes may hold device 0; a busy card is
# RECORDED, not refused. Refuse ONLY on health / readability / identity / ffi problems:
#   * the occupancy evidence file is missing, unreadable or empty (health evidence absent);
#   * npu-smi health check failed (occupancy UNKNOWN, not empty);
#   * the device id is not the canonical 0;
#   * the ffi environment is the CPU one (0.1.15.post2) instead of the device one (0.1.12).
#
# This gate itself never opens /dev/davinci*, never runs npu-smi, never imports torch_npu.
# Sourcing it is safe. The ONE lock is NOT here: the single canonical holder is
# /home/.tf_device0.lock, taken by run_all_cases.sh (or, for a standalone run, by  # no-machine-path: allow
# run_case.py). Do not add a second flock in this file.
#
# Required evidence:
#   TF_DEVICE_AUTH=1 (or legacy TF_DEVICE_AUTHORISED non-empty)  explicit go-ahead
#   TF_DEVICE_DEVICE_ID=0                                       the canonical device
#   TF_DEVICE_OCCUPANCY=<path>                                  a readable occupancy record
#                                                               (content may say BUSY;
#                                                               busy is allowed/recorded)
#
# Usage:
#   . "$(dirname "$0")/device_precondition.sh"
#   tf_require_device || exit 78      # 78 = EX_CONFIG, refused, didn't attempt

tf_require_device() {
    tf_gate_missing=""

    if [ "$TF_DEVICE_AUTH" != "1" ] && [ -z "$TF_DEVICE_AUTHORISED" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_AUTH=1"
    fi
    if [ "$TF_DEVICE_DEVICE_ID" = "" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_DEVICE_ID"
    elif [ "$TF_DEVICE_DEVICE_ID" != "0" ]; then
        echo "device gate: REFUSING: this backend is pinned to device 0, got TF_DEVICE_DEVICE_ID=$TF_DEVICE_DEVICE_ID." >&2
        echo "device gate: a run on another board is evidence about a different device." >&2
        return 78
    fi
    if [ -z "$TF_DEVICE_OCCUPANCY" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_OCCUPANCY"
    elif [ ! -r "$TF_DEVICE_OCCUPANCY" ]; then
        echo "device gate: REFUSING: occupancy evidence $TF_DEVICE_OCCUPANCY is unreadable (health/readability refusal)." >&2
        return 78
    elif [ ! -s "$TF_DEVICE_OCCUPANCY" ]; then
        echo "device gate: REFUSING: occupancy evidence $TF_DEVICE_OCCUPANCY is empty." >&2
        echo "device gate: absent evidence is not evidence of an idle card." >&2
        return 78
    fi

    if [ -n "$tf_gate_missing" ]; then
        echo "device gate: REFUSING to touch a device. Missing:$tf_gate_missing" >&2
        echo "device gate: nothing was opened. This is a refusal, not a failed attempt." >&2
        return 78
    fi

    # BUSY IS ALLOWED: record it, never refuse on it (HANDOFF section 4, shared card).
    if grep -qiE 'busy|occupied|held' "$TF_DEVICE_OCCUPANCY" 2>/dev/null; then
        echo "device gate: card reports occupancy (BUSY/held); ALLOWED on a shared card and RECORDED only," >&2
        echo "device gate: proceeding -- timings are PROVISIONAL, numerics are unaffected by contention." >&2
    else
        echo "device gate: card reports no active process." >&2
    fi
    echo "device gate: authorised (device $TF_DEVICE_DEVICE_ID); health evidence $TF_DEVICE_OCCUPANCY" >&2
    return 0
}

# CPU-side identity preconditions that cannot be checked by this POSIX shell are asserted
# in run_case.py (_assert_device_identity): uid==0, ASCEND_RT_VISIBLE_DEVICES=0,
# apache-tvm-ffi==0.1.12 (device env; 0.1.15.post2 is CPU and must not be mixed).
