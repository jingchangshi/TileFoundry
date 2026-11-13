from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import AscendTarget


@module(
    entry="forward",
    target=AscendTarget("ascend.950pr"),
    topologies=(Topology("block", 1), Topology("aic", 1), Topology("aiv", 2), Topology("lane", 64),),
)
class MlaScheduled:
    @prim_func(target=AscendTarget("ascend.950pr"))
    def forward(
        q_nope: Tensor[(1, 4, 16, 64), "f16"], q_rope: Tensor[(1, 4, 16, 64), "f16"], k_nope: Tensor[(1, 1, 16, 64), "f16"], k_rope: Tensor[(1, 1, 16, 64), "f16"], v: Tensor[(1, 1, 16, 64), "f16"], causal_mask: Tensor[(64, 16), "f32"], scratch: Tensor[(1024,), "f32"], scratch_1: Tensor[(4096,), "f16"], out: Tensor[(1, 4, 16, 64), "f16"]
    ):
        with Mesh((Topology("aiv", 2),), layout=(1,), names=("v",)) as cta:
            value = T.tensor_view(0, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16))
            T.ascend_gm_to_ub(causal_mask, value)
            value_1 = T.tensor_view(
                4096, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
            )
            T.fill(value_1, 0.0)
            scale_ub = T.tensor_view(
                8192, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
            )
            T.fill(scale_ub, 0.08838834764831843)
            scores_gm_1 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 1024]), layout=(64, 16), shape=(64, 16)
            )
            T.fill(scores_gm_1, 0.0)
            for kt in tf.pipelined(2, max_stage=2, max_buffer_versions=2, enable_offset=True, fixed_ii=4, task_intents=(('t0_fill', 0, 0, (), 1), ('t1_ascend_gm_to_l1', 1, 0, (), 1), ('t2_ascend_l1_to_l0a', 2, 1, (), 1), ('t3_ascend_gm_to_l1', 3, 0, (), 1), ('t4_ascend_l1_to_l0b', 4, 1, (), 1), ('t5_ascend_mmad', 5, 2, (), 1), ('t6_ascend_gm_to_l1', 6, 0, (), 1), ('t7_ascend_l1_to_l0a', 7, 1, (), 1), ('t8_ascend_gm_to_l1', 8, 0, (), 1), ('t9_ascend_l1_to_l0b', 9, 1, (), 1), ('t10_ascend_mmad', 10, 2, (), 1), ('t11_ascend_dual_copy', 11, 2, (), 1), ('t12_ascend_ub_to_gm', 12, 2, (), 1)), buffer_schedules=(('s0', 'fixed', 2), ('value_2', 'fixed', 2), ('value_4', 'fixed', 1), ('value_5', 'fixed', 2), ('value_8', 'fixed', 1), ('value_9', 'fixed', 2), ('value_11', 'fixed', 1), ('value_12', 'fixed', 1), ('value_15', 'fixed', 2), ('value_16', 'fixed', 1))):
                with Mesh((Topology("aic", 1),), layout=(1,), names=("c",)) as scope:
                    s0 = T.tensor_view(
                        0, dtype="f32", storage="tmem", layout=(64, 8), shape=(64, 8)
                    )
                    T.fill(s0, 0.0)
                    value_3 = T.tensor_view(T.ptr_of(q_nope), layout=(64, 64), shape=(64, 64))
                    value_2 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_gm_to_l1(value_3, value_2)
                    value_4 = T.tensor_view(
                        10240, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_l1_to_l0a(value_2, value_4)
                    value_6 = T.tensor_view(T.ptr_of(k_nope), layout=(16, 64), shape=(16, 64))
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6[kt * 8:kt * 8 + 1, 0:0 + 1]), layout=(8, 64), shape=(8, 64)
                    )
                    value_5 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_gm_to_l1(value_7, value_5)
                    value_8 = T.tensor_view(
                        3072, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_l1_to_l0b(value_5, value_8)
                    T.ascend_mmad(
                        s0,
                        value_4,
                        value_8,
                        a_layout="MK",
                        b_layout="NK",
                        init_zero=True,
                    )
                    value_10 = T.tensor_view(T.ptr_of(q_rope), layout=(64, 64), shape=(64, 64))
                    value_9 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_gm_to_l1(value_10, value_9)
                    value_11 = T.tensor_view(
                        10240, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_l1_to_l0a(value_9, value_11)
                    value_13 = T.tensor_view(T.ptr_of(k_rope), layout=(16, 64), shape=(16, 64))
                    value_14 = T.tensor_view(
                        T.ptr_of(value_13[kt * 8:kt * 8 + 1, 0:0 + 1]),
                        layout=(8, 64),
                        shape=(8, 64),
                    )
                    value_12 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_gm_to_l1(value_14, value_12)
                    value_15 = T.tensor_view(
                        3072, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_l1_to_l0b(value_12, value_15)
                    T.ascend_mmad(
                        s0,
                        value_11,
                        value_15,
                        a_layout="MK",
                        b_layout="NK",
                        init_zero=False,
                    )
                    value_16 = T.tensor_view(
                        4096, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_dual_copy(s0, value_16)
                value_16_2 = T.tensor_view(
                    4096, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                )
                scores_gm_2 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 1024]), layout=(64, 16), shape=(64, 16)
                )
                value_17 = T.tensor_view(
                    T.ptr_of(scores_gm_2[0:0 + 1, kt * 8:kt * 8 + 1]),
                    layout=((64, 8), (16, 1)),
                    shape=(64, 8),
                )
                T.ascend_ub_to_gm(value_16_2, value_17)
            value_18 = T.tensor_view(
                4096, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
            )
            T.ascend_gm_to_ub(scores_gm_1, value_18)
            with T.SimtVF():
                value_19 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                value_18_2 = T.tensor_view(
                    4096, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_ub_to_reg(value_18_2, value_19)
                value_20 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                scale_ub_2 = T.tensor_view(
                    8192, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_ub_to_reg(scale_ub_2, value_20)
                value_21 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                T.ascend_vmul(value_19, value_20, value_21)
                value_22 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                value_40 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_ub_to_reg(value_40, value_22)
                value_23 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                T.ascend_vadd(value_21, value_22, value_23)
                value_24 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                T.ascend_vreduce_max(value_23, value_24, axes=(-1,), keepdim=True)
                value_25 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                T.ascend_vsub(value_23, value_24, value_25)
                value_26 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                T.ascend_vexp(value_25, value_26)
                value_27 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                T.ascend_vreduce_sum(value_26, value_27, axes=(-1,), keepdim=True)
                value_28 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f32", "rmem"])
                T.ascend_vdiv(value_26, value_27, value_28)
                value_29 = T.alloc_tensor(tensor_type=Tensor[(64, 16), "f16", "rmem"])
                T.ascend_vcast(value_28, value_29, dtype="f16")
                value_30 = T.tensor_view(
                    0, dtype="f16", storage="smem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_reg_to_ub(value_29, value_30)
            with Mesh((Topology("aic", 1),), layout=(1,), names=("c2",)) as scope_1:
                o0 = T.tensor_view(0, dtype="f32", storage="tmem", layout=(64, 64), shape=(64, 64))
                T.fill(o0, 0.0)
                value_30_2 = T.tensor_view(
                    0, dtype="f16", storage="smem", layout=(64, 16), shape=(64, 16)
                )
                value_31 = T.tensor_view(
                    16384, dtype="f16", storage="tmem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_dual_copy(value_30_2, value_31)
                value_32 = T.tensor_view(
                    18432, dtype="f16", storage="tmem", layout=(64, 16), shape=(64, 16)
                )
                T.ascend_l1_to_l0a(value_31, value_32)
                value_34 = T.tensor_view(T.ptr_of(v), layout=(16, 64), shape=(16, 64))
                value_33 = T.tensor_view(
                    16384, dtype="f16", storage="tmem", layout=(16, 64), shape=(16, 64)
                )
                T.ascend_gm_to_l1(value_34, value_33)
                value_35 = T.tensor_view(
                    20480, dtype="f16", storage="tmem", layout=(16, 64), shape=(16, 64)
                )
                T.ascend_l1_to_l0b(value_33, value_35)
                T.ascend_mmad(o0, value_32, value_35, a_layout="MK", b_layout="KN", init_zero=True)
                value_36 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                )
                T.ascend_dual_copy(o0, value_36)
            with T.SimtVF():
                value_37 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                value_36_2 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                )
                T.ascend_ub_to_reg(value_36_2, value_37)
                value_38 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f16", "rmem"])
                T.ascend_vcast(value_37, value_38, dtype="f16")
                value_39 = T.tensor_view(
                    0, dtype="f16", storage="smem", layout=(64, 64), shape=(64, 64)
                )
                T.ascend_reg_to_ub(value_38, value_39)
            value_39_2 = T.tensor_view(
                0, dtype="f16", storage="smem", layout=(64, 64), shape=(64, 64)
            )
            value_40_1 = T.tensor_view(
                T.ptr_of(scratch_1[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
            )
            T.ascend_ub_to_gm(value_39_2, value_40_1)
