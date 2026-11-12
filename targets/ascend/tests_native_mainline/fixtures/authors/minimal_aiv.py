"""Minimal real Ascend author program: a pipelined AIV accumulation over GM.

This is authored source, not assembled TIR. Reading it top to bottom:

    GM x --ascend_gm_to_ub--> UB base                (the first version)
    for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2):
        GM x --ascend_gm_to_ub--> UB tile            (one GM slice per iteration)
        with T.SimdVF():                             (one whole vector region)
            base, tile --ascend_vld--> vregs
            ascend_pset(32) --> a real one-bit-per-lane predicate
            vregs --ascend_vadd--> vreg
            vreg --ascend_vsts--> UB base            (the next carried version)
    UB base --ascend_ub_to_gm--> GM out              (trailing ABI output)

So each iteration is base[k+1] = base[k] + x on the 64 AIV lanes, and the UB
slice is the loop-carried value: the add is real, not an empty declaration.

The four register instructions are inside-only for the declared SimdVF mode and
the two transfers are outside-only, so the placement is the shape of the program
rather than an example of it. tf.pipelined states the single typed pipeline
surface (max_stage bounds stages; max_buffer_versions bounds auto versions);
the loop bounds stay the loop's own start/stop/step.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="vsum", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class MinimalAivPipeline:
    """One pipelined GM tensor accumulation, executed by one AIV core."""

    @func
    def vsum(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            base = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            carried = tf.schedule((x, base), op=T.ascend_gm_to_ub())
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2):
                tile = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                loaded = tf.schedule((x, tile), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    left = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                    right = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                    mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                    acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                    hl = tf.schedule((carried, left), op=T.ascend_vld())
                    hr = tf.schedule((loaded, right), op=T.ascend_vld())
                    pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                    added = tf.schedule((hl, hr, pred, acc), op=T.ascend_vadd())
                    stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                    back = tf.schedule((added, stored), op=T.ascend_vsts())
                carried = back
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((carried, out), op=T.ascend_ub_to_gm())
