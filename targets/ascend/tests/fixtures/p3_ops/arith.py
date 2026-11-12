"""03-15/16/17 micro chains: vmul, vsqrt, vdiv, vcadd (six calls), vdupv."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


def _stage(x):
    with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
        ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
        staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
        return staged


def _finish(reg):
    stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
    back = tf.schedule((reg, stored), op=T.ascend_vsts())
    out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
    return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="mul", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VmulChain:
    @func
    def mul(x: Tensor[(LANES,), "f32"], y: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
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
                reg = tf.schedule((la, lb, pred, acc), op=T.ascend_vmul())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((reg, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="sqrt", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VsqrtChain:
    @func
    def sqrt(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            sx = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((sx, va), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                reg = tf.schedule((la, pred, acc), op=T.ascend_vsqrt())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((reg, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="div", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VdivChain:
    @func
    def div(x: Tensor[(LANES,), "f32"], y: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
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
                reg = tf.schedule((la, lb, pred, acc), op=T.ascend_vdiv())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((reg, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="cadd", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VcaddChain:
    @func
    def cadd(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            sx = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((sx, va), op=T.ascend_vld())
                b1 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b2 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b3 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b4 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b5 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b6 = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                c1 = tf.schedule((la, b1), op=T.ascend_vcadd(lanes=64))
                c2 = tf.schedule((c1, b2), op=T.ascend_vcadd(lanes=32))
                c3 = tf.schedule((c2, b3), op=T.ascend_vcadd(lanes=16))
                c4 = tf.schedule((c3, b4), op=T.ascend_vcadd(lanes=8))
                c5 = tf.schedule((c4, b5), op=T.ascend_vcadd(lanes=4))
                c6 = tf.schedule((c5, b6), op=T.ascend_vcadd(lanes=2))
                bc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b = tf.schedule((c6, bc), op=T.ascend_vdupv(position=0))
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((b, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())


@module(entry="dupv", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VdupvChain:
    @func
    def dupv(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            sx = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                va = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                bc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((sx, va), op=T.ascend_vld())
                b = tf.schedule((la, bc), op=T.ascend_vdupv(position=2))
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((b, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
