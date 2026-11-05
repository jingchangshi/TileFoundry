"""D0 fixture: a pipelined VF vadd loop (gm->UB, VF vld/vadd/vsts, UB->gm).

Supplied by Lead, who measured checked/analyze/finalize all succeeding on it. It is
the D0 end-to-end carrier: the loop declares tf.pipelined(4, num_stages=2), so the
typed spec must survive parse -> analyze -> finalize and be visible on the TIR For.
"""
from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64

@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class PipeVadd:
    @func
    def k(x: Tensor[(256,), "f32"], z: Tensor[(256,), "f32"]) -> Tensor[(256,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            y = tf.empty(Tensor[(256,), "f32", (256,), "gmem"])
            for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):
                ux = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sx = tf.schedule((x[i * L:i * L + L], ux), op=T.ascend_gm_to_ub())
                uz = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sz = tf.schedule((z[i * L:i * L + L], uz), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    va = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    vb = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    mask = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                    la = tf.schedule((sx, va), op=T.ascend_vld())
                    lb = tf.schedule((sz, vb), op=T.ascend_vld())
                    pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                    acc = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    total = tf.schedule((la, lb, pred, acc), op=T.ascend_vadd())
                    stored = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                    back = tf.schedule((total, stored), op=T.ascend_vsts())
                s = tf.schedule((back, y[i * L:i * L + L]), op=T.ascend_ub_to_gm())
            return y