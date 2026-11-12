"""03-04/03-05 positive chain, authored as a normal TF scheduled-HIR program.

    GM bf16 --gm_to_ub--> UB bf16
            --vld(dist=UNPK_B16)--> CARRIER window (bf16, NOT an f32 register)
            --pset16(elem_width=16)--> conversion mask
            --vcvt(direction="forward", part=0, elem_width=16)--> f32 RMEM
            --vsts--> UB f32 --ub_to_gm--> GM f32

UNPK_B16 belongs to vld and is a transport unpack, not a numeric cast. The only op that
consumes a carrier is vcvt. pset32 is untouched and reserved for FP32 arithmetic.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="convert", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Bf16VcvtChain:
    @func
    def convert(x: Tensor[(LANES,), "bf16"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                carrier = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "carrier"])
                loaded = tf.schedule((staged, carrier), op=T.ascend_vld(dist="UNPK_B16"))
                mask16 = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                pred16 = tf.schedule((mask16,), op=T.ascend_pset16(elem_width=16))
                f32reg = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                converted = tf.schedule(
                    (loaded, pred16, f32reg),
                    op=T.ascend_vcvt(direction="forward", part=0, elem_width=16),
                )
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((converted, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
