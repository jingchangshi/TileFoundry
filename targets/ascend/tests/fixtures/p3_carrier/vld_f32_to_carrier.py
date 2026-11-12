"""NEGATIVE (03-03): f32 vld with no dist landing in the bf16 CARRIER class.

The carrier register class is the UNPK_B16 destination and nothing else: a plain f32
load must land in an f32 element register. This is the reverse direction of the same
signature rule. A second, legitimate f32 load keeps the program otherwise well formed,
so the only defect is the destination class of the first one.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget


@module(entry="load", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VldF32ToCarrier:
    @func
    def load(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                carrier = tf.empty(Tensor[(64,), "bf16", (64,), "carrier"])
                wrong = tf.schedule((staged, carrier), op=T.ascend_vld())
                reg = tf.empty(Tensor[(64,), "f32", (64,), "rmem"])
                right = tf.schedule((staged, reg), op=T.ascend_vld())
                stored = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
                back = tf.schedule((right, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
