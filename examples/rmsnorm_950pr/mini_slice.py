"""R1 minimal vertical slice: a tiny real TF program on the Ascend Target."""
from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget

N = 64
D = 256


@module(entry="add1", target=Ascend950PRTarget(), topologies=(Topology("aiv", 56),))
class MiniSlice:
    @func
    def add1(x: Tensor[(N, D), "bf16"], t: Tensor[(N, D), "bf16"]) -> Tensor[(N, D), "bf16"]:
        with Mesh(("aiv",), layout=(56,), names=("core",)) as aiv:
            return x + t
