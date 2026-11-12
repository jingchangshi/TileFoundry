"""Native AR RMS primitive: elementwise product, 4 x 64 float32 lanes.

With z=x this is the squared input before the RMS reduction. This is a
limited native backend bring-up primitive, not the full attention operator.
"""
from __future__ import annotations

# ruff: noqa: F403, F405, I001, F841  -- curated DSL star import; 'as aiv' binds a mesh var

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64


@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class NativeSquare:
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
                    total = tf.schedule((la, lb, pred, acc), op=T.ascend_vmul())
                    stored = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                    back = tf.schedule((total, stored), op=T.ascend_vsts())
                y = tf.insert_slice(y, back, (i * L,))
            return y
