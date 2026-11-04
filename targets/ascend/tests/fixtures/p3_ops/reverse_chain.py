"""03-18 micro: reverse conversion f32 RMEM -> bf16 CARRIER -> vsts(PK_B32) -> bf16 UB -> GM.

    GM f32 -> UB f32 -> vld -> vcvt(direction="reverse", part=0, 16-bit mask)
          -> CARRIER bf16 -> vsts(pack="PK_B32") -> UB bf16 -> ub_to_gm(bf16,bf16) -> GM bf16
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="reverse", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class ReverseVcvtChain:
    @func
    def reverse(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                reg = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                carrier = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "carrier"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                la = tf.schedule((staged, reg), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset16(elem_width=16))
                conv = tf.schedule((la, pred, carrier), op=T.ascend_vcvt(direction="reverse", part=0, elem_width=16))
                ub_bf16 = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "smem"])
                back = tf.schedule((conv, ub_bf16), op=T.ascend_vsts(pack="PK_B32"))
            out = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
