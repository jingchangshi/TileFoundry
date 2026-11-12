"""bf16 staging OUTSIDE the mode: GM bf16 -> UB bf16 -> GM bf16 (no vld)."""
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget


@module(entry="copy", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Bf16Copy:
    @func
    def copy(x: Tensor[(64,), "bf16"]) -> Tensor[(64,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(64,), "bf16", (64,), "smem"])
            staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            out = tf.empty(Tensor[(64,), "bf16", (64,), "gmem"])
            return tf.schedule((staged, out), op=T.ascend_ub_to_gm())
