"""Negative: bf16 GM into an f32 UB window must be refused by the dtype constraint."""
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget


@module(entry="bad", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Bf16Mismatch:
    @func
    def bad(x: Tensor[(64,), "bf16"]) -> Tensor[(64,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            out = tf.empty(Tensor[(64,), "bf16", (64,), "gmem"])
            return tf.schedule((staged, out), op=T.ascend_ub_to_gm())
