"""M2b: one-AIV same-dtype GM->UB->fresh GM copy through normal finalize.

This is the smallest native TF chain proof. It intentionally uses only the
generic Copy instruction and a one-unit AIV mesh; it does not exercise Binary,
global candidate selection, or any core mesh change.
"""
from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget

N = 64


@module(entry="copy1", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class CopySlice:
    @func
    def copy1(x: Tensor[(N,), "f32"], y: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(N,), "f32", (N,), "smem"])
            staged = tf.schedule((x, ub), op=T.copy())
            out = tf.empty(Tensor[(N,), "f32", (N,), "gmem"])
            return tf.schedule((staged, out), op=T.copy())
