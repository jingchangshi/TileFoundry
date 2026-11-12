from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget


@module(entry="convert", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Bf16VcvtChain:
    @prim_func(target=Ascend950PRTarget())
    def convert(x: Tensor[(64,), "bf16"], out: Tensor[(64,), "f32"]):
        with Mesh((Topology("aiv", 1),), layout=(1,), names=("core",)) as cta:
            ub = T.tensor_view(0, dtype="bf16", storage="smem", layout=(64,), shape=(64,))
            T.ascend_gm_to_ub(x, ub)
            with T.SimdVF():
                carrier = T.alloc_tensor(tensor_type=Tensor[(64,), "bf16", "carrier"])
                ub_2 = T.tensor_view(0, dtype="bf16", storage="smem", layout=(64,), shape=(64,))
                T.ascend_vld(ub_2, carrier, dist="UNPK_B16")
                if True:
                    mask_inside = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", "rmem"])
                    T.ascend_pset16(mask_inside, elem_width=16)
                    f32_inside = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                    T.ascend_vcvt(carrier, mask_inside, f32_inside, direction="forward", part=0, elem_width=16)
                    stored_in = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
                    T.ascend_vsts(f32_inside, stored_in)
            stored_2 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.ascend_ub_to_gm(stored_2, out)
