"""03-11 micro: typed vmula chain (a*b into acc), authored as scheduled HIR.

    GM f32 -> UB -> vld a, vld b -> pset(32) -> vmula acc -> vsts -> UB -> GM
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="mulacc", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VmulaMicro:
    @func
    def mulacc(x: Tensor[(LANES,), "f32"], y: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ux = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            uy = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            sx = tf.schedule((x, ux), op=T.ascend_gm_to_ub())
            sy = tf.schedule((y, uy), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                vb = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((sx, va), op=T.ascend_vld())
                lb = tf.schedule((sy, vb), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                prod = tf.schedule((la, lb, pred, acc), op=T.ascend_vmula())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((prod, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
