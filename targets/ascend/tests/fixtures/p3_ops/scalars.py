"""03-12/13/14 micro chains: vdup, vadds, vmuls, each ending in vsts -> UB -> GM.

GM f32 -> UB -> vld -> <op with scalar> -> vsts -> UB -> GM
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="dup", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VdupChain:
    @func
    def dup(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                reg = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                d = tf.schedule((pred, reg), op=T.ascend_vdup(value=1.5))
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((d, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="adds", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VaddsChain:
    @func
    def adds(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                reg = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((staged, va), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                d = tf.schedule((la, pred, reg), op=T.ascend_vadds(value=0.25))
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((d, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="muls", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VmulsChain:
    @func
    def muls(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                reg = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((staged, va), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                d = tf.schedule((la, pred, reg), op=T.ascend_vmuls(value=-2.0))
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((d, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
