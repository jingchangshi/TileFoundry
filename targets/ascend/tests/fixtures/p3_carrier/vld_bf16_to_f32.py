"""NEGATIVE (03-03): bf16 vld with UNPK_B16 landing in an f32 ELEMENT register.

The contract says UNPK_B16 produces a carrier-window and only vcvt may consume it; a
bf16 load whose destination is an f32 rmem would be a numeric cast fused into vld.
This fixture is the shape that must be refused by name.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget


@module(entry="convert", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VldBf16ToF32:
    @func
    def convert(x: Tensor[(64,), "bf16"]) -> Tensor[(64,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(64,), "bf16", (64,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                reg = tf.empty(Tensor[(64,), "f32", (64,), "rmem"])
                loaded = tf.schedule((staged, reg), op=T.ascend_vld(dist="UNPK_B16"))
                stored = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
                back = tf.schedule((loaded, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
