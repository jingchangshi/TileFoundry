"""Two outputs keep the loaded UB tile live across the vector computation."""
from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

@module(entry="vadd", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class IndependentUB:
    @func
    def vadd(x: Tensor[(64,), "f32"]) -> tuple[Tensor[(64,), "f32"], Tensor[(64,), "f32"]]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            loaded = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
            staged = tf.schedule((x, loaded), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                left = tf.empty(Tensor[(64,), "f32", (64,), "rmem"])
                right = tf.empty(Tensor[(64,), "f32", (64,), "rmem"])
                mask = tf.empty(Tensor[(64,), "bool", (64,), "rmem"])
                acc = tf.empty(Tensor[(64,), "f32", (64,), "rmem"])
                hl = tf.schedule((staged, left), op=T.ascend_vld())
                hr = tf.schedule((staged, right), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                added = tf.schedule((hl, hr, pred, acc), op=T.ascend_vadd())
                stored = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
                back = tf.schedule((added, stored), op=T.ascend_vsts())
            y = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            y1 = tf.schedule((back, y), op=T.ascend_ub_to_gm())
            z = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            z1 = tf.schedule((staged, z), op=T.ascend_ub_to_gm())
            return y1, z1
