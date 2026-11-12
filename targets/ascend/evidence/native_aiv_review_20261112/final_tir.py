from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend.tf_target import Ascend950PRTarget


@prim_func(target=Ascend950PRTarget())
def k(x: Tensor[(256,), "f32"], z: Tensor[(256,), "f32"], out: Tensor[(256,), "f32"]):
    with Mesh((Topology("aiv", 1),), layout=(1,), names=("core",)) as cta:
        for i in tf.pipelined(1, num_stages=2, annotations={"enable_offset": False}):
            value__k0 = T.tensor_view(T.ptr_of(x[0 * 64:0 * 64 + 1]), layout=(64,), shape=(64,))
            ux__k0 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.ascend_gm_to_ub(
                value__k0,
                T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            value_1__k0 = T.tensor_view(
                T.ptr_of(z[0 * 64:0 * 64 + 1]), layout=(64,), shape=(64,)
            )
            uz__k0 = T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.ascend_gm_to_ub(
                value_1__k0,
                T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            with T.SimdVF():
                va__k0 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                ux__k0_2 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    va__k0,
                )
                vb__k0 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                uz__k0_2 = T.tensor_view(
                    256, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    vb__k0,
                )
                mask__k0 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", "rmem"])
                T.ascend_pset(mask__k0, elem_width=32)
                acc__k0 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                T.ascend_vmul(va__k0, vb__k0, mask__k0, acc__k0)
                stored__k0 = T.tensor_view(
                    512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vsts(
                    acc__k0,
                    T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                )
            T.flag_set(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_set(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=0,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_set(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=0,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            stored__k0_2 = T.tensor_view(
                512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
            )
            window__k0 = T.tensor_view(
                T.ptr_of(out[0 * 64:0 * 64 + 64]), layout=(64,), shape=(64,)
            )
            T.flag_wait(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=0,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.copy(
                T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                window__k0,
            )
            T.flag_set(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=0,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            value__k1 = T.tensor_view(T.ptr_of(x[1 * 64:1 * 64 + 1]), layout=(64,), shape=(64,))
            ux__k1 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value__k1,
                T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            value_1__k1 = T.tensor_view(
                T.ptr_of(z[1 * 64:1 * 64 + 1]), layout=(64,), shape=(64,)
            )
            uz__k1 = T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value_1__k1,
                T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_wait(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            with T.SimdVF():
                va__k1 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                ux__k1_2 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    va__k1,
                )
                vb__k1 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                uz__k1_2 = T.tensor_view(
                    256, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    vb__k1,
                )
                mask__k1 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", "rmem"])
                T.ascend_pset(mask__k1, elem_width=32)
                acc__k1 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                T.ascend_vmul(va__k1, vb__k1, mask__k1, acc__k1)
                stored__k1 = T.tensor_view(
                    512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vsts(
                    acc__k1,
                    T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                )
            T.flag_set(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_set(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_set(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            stored__k1_2 = T.tensor_view(
                512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
            )
            window__k1 = T.tensor_view(
                T.ptr_of(out[1 * 64:1 * 64 + 64]), layout=(64,), shape=(64,)
            )
            T.flag_wait(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=1,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.copy(
                T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                window__k1,
            )
            T.flag_set(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=1,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            value__k2 = T.tensor_view(T.ptr_of(x[2 * 64:2 * 64 + 1]), layout=(64,), shape=(64,))
            ux__k2 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value__k2,
                T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            value_1__k2 = T.tensor_view(
                T.ptr_of(z[2 * 64:2 * 64 + 1]), layout=(64,), shape=(64,)
            )
            uz__k2 = T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value_1__k2,
                T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_wait(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            with T.SimdVF():
                va__k2 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                ux__k2_2 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    va__k2,
                )
                vb__k2 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                uz__k2_2 = T.tensor_view(
                    256, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    vb__k2,
                )
                mask__k2 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", "rmem"])
                T.ascend_pset(mask__k2, elem_width=32)
                acc__k2 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                T.ascend_vmul(va__k2, vb__k2, mask__k2, acc__k2)
                stored__k2 = T.tensor_view(
                    512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vsts(
                    acc__k2,
                    T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                )
            T.flag_set(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_set(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_set(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            stored__k2_2 = T.tensor_view(
                512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
            )
            window__k2 = T.tensor_view(
                T.ptr_of(out[2 * 64:2 * 64 + 64]), layout=(64,), shape=(64,)
            )
            T.flag_wait(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=2,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.copy(
                T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                window__k2,
            )
            T.flag_set(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=2,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            value__k3 = T.tensor_view(T.ptr_of(x[3 * 64:3 * 64 + 1]), layout=(64,), shape=(64,))
            ux__k3 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value__k3,
                T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            value_1__k3 = T.tensor_view(
                T.ptr_of(z[3 * 64:3 * 64 + 1]), layout=(64,), shape=(64,)
            )
            uz__k3 = T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,))
            T.flag_wait(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.ascend_gm_to_ub(
                value_1__k3,
                T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            T.flag_set(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_wait(
                channel_name="ready.s1.s2.s2",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_wait(
                channel_name="ready.s4.s5.s5",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            with T.SimdVF():
                va__k3 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                ux__k3_2 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(0, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    va__k3,
                )
                vb__k3 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                uz__k3_2 = T.tensor_view(
                    256, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vld(
                    T.tensor_view(256, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                    vb__k3,
                )
                mask__k3 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", "rmem"])
                T.ascend_pset(mask__k3, elem_width=32)
                acc__k3 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", "rmem"])
                T.ascend_vmul(va__k3, vb__k3, mask__k3, acc__k3)
                stored__k3 = T.tensor_view(
                    512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                T.ascend_vsts(
                    acc__k3,
                    T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                )
            T.flag_set(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.flag_set(
                channel_name="free.s1.s2.s2",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=0,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            T.flag_set(
                channel_name="free.s4.s5.s5",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=1,
                producer_pipe="mte2",
                consumer_pipe="vector",
            )
            stored__k3_2 = T.tensor_view(
                512, dtype="f32", storage="smem", layout=(64,), shape=(64,)
            )
            window__k3 = T.tensor_view(
                T.ptr_of(out[3 * 64:3 * 64 + 64]), layout=(64,), shape=(64,)
            )
            T.flag_wait(
                channel_name="ready.s10.s11.s9",
                kind="ready",
                slot=0,
                generation=3,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.copy(
                T.tensor_view(512, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                window__k3,
            )
            T.flag_set(
                channel_name="free.s10.s11.s9",
                kind="free",
                slot=0,
                generation=3,
                versions=1,
                flag_id=2,
                producer_pipe="vector",
                consumer_pipe="mte3",
            )
            T.drain()
