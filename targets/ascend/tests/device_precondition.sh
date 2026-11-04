#!/bin/sh
# DEVICE PRECONDITION GATE. Source this before ANY test that could open /dev/davinci*.
#
# Why this file exists
# -------------------
# A host-wide filesystem wedge (D-state children on the exec flush path, 200+ stuck
# processes, load above 500) correlates with a single davinci0 open at 13:46:26 from
# one of my own probes. The correlation is not proof and nobody has root stacks yet,
# but the shape is consistent across tenants and the cost of being wrong is a stalled
# host for everybody.
#
# The rule that follows is not "be careful with the device". It is:
#
#     A DEVICE PROBE MUST REFUSE WHEN THE CARD IS BUSY, NOT ATTEMPT AND FAIL.
#
# That distinction is the whole point. An attempt that is expected to fail at
# aclInit still OPENS the node, still enters the driver, and still takes whatever
# lock is contended. "It will just error out" is exactly the reasoning that makes a
# shared card unsafe.
#
# What this gate does NOT do
# -------------------------
# It does not check occupancy itself. Checking occupancy on this host means npu-smi,
# and running npu-smi is itself device access of the kind currently forbidden. So the
# gate cannot self-certify; it can only demand evidence that someone else produced.
# It refuses by default and says exactly what it is missing.
#
# It never opens /dev/davinci*, never runs npu-smi, never imports torch_npu. Sourcing
# it is safe. That is deliberate: the safe thing must be the easy thing.
#
# Required evidence, all three or the gate refuses
# ------------------------------------------------
#   1. TF_DEVICE_AUTHORISED=<token>   an explicit go-ahead for THIS probe, set in the
#      environment. Not a default, not inherited from a previous session.
#   2. TF_DEVICE_OCCUPANCY=<path>     a file recording an occupancy check that was
#      made BEFORE this probe: who holds the card, how many processes, and when it
#      was checked. Produced by whatever authorised tool the device owner uses --
#      not by this script and not by me.
#   3. TF_DEVICE_DEVICE_ID=<id>       the specific device being claimed, so a probe
#      cannot wander onto another tenant card.
#
# Usage
# -----
#    . "$(dirname "$0")/device_precondition.sh"
#    tf_require_device || exit 78          # 78 = EX_CONFIG, "refused, did not attempt"
#
# and only then run the probe. A probe that reaches the driver without passing this
# gate is a defect regardless of whether it succeeded.

tf_require_device() {
    tf_gate_missing=""

    if [ -z "${TF_DEVICE_AUTHORISED:-}" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_AUTHORISED"
    fi
    if [ -z "${TF_DEVICE_DEVICE_ID:-}" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_DEVICE_ID"
    fi
    if [ -z "${TF_DEVICE_OCCUPANCY:-}" ]; then
        tf_gate_missing="$tf_gate_missing TF_DEVICE_OCCUPANCY"
    elif [ ! -s "${TF_DEVICE_OCCUPANCY}" ]; then
        echo "device gate: TF_DEVICE_OCCUPANCY=$TF_DEVICE_OCCUPANCY is missing or empty." >&2
        echo "device gate: absent evidence is not evidence of an idle card." >&2
        return 78
    fi

    if [ -n "$tf_gate_missing" ]; then
        echo "device gate: REFUSING to touch a device. Missing:$tf_gate_missing" >&2
        echo "device gate: nothing was opened. This is a refusal, not a failed attempt --" >&2
        echo "device gate: an attempt is what we are trying to stop, because a probe that" >&2
        echo "device gate: is expected to fail at aclInit still opens the node." >&2
        return 78
    fi

    echo "device gate: authorised by TF_DEVICE_AUTHORISED, device $TF_DEVICE_DEVICE_ID," >&2
    echo "device gate: occupancy evidence $TF_DEVICE_OCCUPANCY" >&2
    echo "device gate: a busy card is still refused by the caller, not by this gate." >&2
    return 0
}

# The default posture for this package, stated where it can be read at a glance:
# no device is touched at all. Every test in this directory is CPU arithmetic and IR
# inspection, so a device probe would have to be added deliberately, and it would
# have to come through this gate.
if [ "${TF_DEVICE_AUTHORISED:-}" = "" ]; then
    : # quiet: sourcing without calling the function asserts nothing
fi