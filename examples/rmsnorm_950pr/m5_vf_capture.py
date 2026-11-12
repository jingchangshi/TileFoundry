"""M5 regression shape: an OUTSIDE producer captured into a SimdVF body.

A generic reshard loads gm->smem OUTSIDE any mode (ascend_gm_to_ub), then a SimdVF body
captures that value and moves it smem->rmem (ascend_vld) and back (ascend_vsts), then
smem->gmem outside (ascend_ub_to_gm).

This exists because a data-dependency walk of the region captures marks the outside
producer as "inside" once the VF body uses its result, which drops it from the outside
default set. The producer is lexically OUTSIDE and must stay there; this file is the
shape that proves it.
"""
from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget

N = 64


@module(entry="vmove", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class GenVfCapture:
    @func
    def vmove(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            staged = tf.reshard(x, (N,), "smem")
            with T.SimdVF():
                reg = tf.reshard(staged, (N,), "rmem")
                back = tf.reshard(reg, (N,), "smem")
            return tf.reshard(back, (N,), "gmem")
