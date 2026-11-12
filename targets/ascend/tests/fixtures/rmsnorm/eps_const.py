"""Probe: eps as a 0-d f32 RUNTIME scalar operand of ascend_vadds (no magic constant)."""
from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget
L = 64

@module(entry="eps", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class EpsConst:
    @func
    def eps(x: Tensor[(L,), "bf16"]) -> Tensor[(L,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            st = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                c = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                lc = tf.schedule((st, c), op=T.ascend_vld(dist="UNPK_B16"))
                m32 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p32 = tf.schedule((m32,), op=T.ascend_pset(elem_width=32))
                m16 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p16 = tf.schedule((m16,), op=T.ascend_pset16(elem_width=16))
                fr = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                cv = tf.schedule((lc, p16, fr), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                out = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                ad = tf.schedule((cv, p32, out), op=T.ascend_vadds(value=1e-6))
                m2 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p2 = tf.schedule((m2,), op=T.ascend_pset16(elem_width=16))
                co = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                rv = tf.schedule((ad, p2, co), op=T.ascend_vcvt(direction="reverse", part=0, elem_width=16))
                uo = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
                wv = tf.schedule((rv, uo), op=T.ascend_vsts(pack="PK_B32"))
            y = tf.empty(Tensor[(L,), "bf16", (L,), "gmem"])
            return tf.schedule((wv, y), op=T.ascend_ub_to_gm())