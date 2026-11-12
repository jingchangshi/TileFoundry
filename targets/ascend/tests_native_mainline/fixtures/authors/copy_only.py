from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

@module(entry="copy", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class CopyOnly:
    @func
    def copy(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            loaded = tf.empty(Tensor[(64,), "f32", (64,), "smem"])
            staged = tf.schedule((x, loaded), op=T.ascend_gm_to_ub())
            out = tf.empty(Tensor[(64,), "f32", (64,), "gmem"])
            return tf.schedule((staged, out), op=T.ascend_ub_to_gm())
