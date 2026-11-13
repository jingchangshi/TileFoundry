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
        q_nope: Tensor[(1, 4, 16, 64), "f16"], q_rope: Tensor[(1, 4, 16, 64), "f16"], k_nope: Tensor[(1, 1, 16, 64), "f16"], k_rope: Tensor[(1, 1, 16, 64), "f16"], v: Tensor[(1, 1, 16, 64), "f16"], causal_mask: Tensor[(64, 16), "f32"], scratch: Tensor[(4096,), "f16"], out: Tensor[(1, 4, 16, 64), "f16"]
    ):
        with Mesh((Topology("aiv", 2),), layout=(2,), names=("v",)) as cta:
            z_m1 = T.tensor_view(0, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1))
            T.fill(z_m1, 0.0)
            m_ub = T.tensor_view(34816, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1))
            T.fill(m_ub, -1e+30)
            l_ub = T.tensor_view(35072, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1))
            T.fill(l_ub, 0.0)
            o_acc_ub = T.tensor_view(
                0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
            )
            T.fill(o_acc_ub, 0.0)
            value = T.tensor_view(16384, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8))
            T.fill(value, 0.0)
            scale_tile_ub = T.tensor_view(
                32768, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
            )
            T.fill(scale_tile_ub, 0.08838834764831843)
            for kt in tf.pipelined(2, max_stage=2, max_buffer_versions=2, enable_offset=True, fixed_ii=10, task_intents=(('t0_ascend_gm_to_ub', 0, 0, (), 1), ('t1_fill', 1, 0, (), 1), ('t2_ascend_gm_to_l1', 2, 0, (), 1), ('t3_ascend_l1_to_l0a', 3, 1, (), 1), ('t4_ascend_gm_to_l1', 4, 0, (), 1), ('t5_ascend_l1_to_l0b', 5, 1, (), 1), ('t6_ascend_mmad', 6, 2, (), 1), ('t7_ascend_gm_to_l1', 7, 0, (), 1), ('t8_ascend_l1_to_l0a', 8, 1, (), 1), ('t9_ascend_gm_to_l1', 9, 0, (), 1), ('t10_ascend_l1_to_l0b', 10, 1, (), 1), ('t11_ascend_mmad', 11, 2, (), 1), ('t12_ascend_dual_copy', 12, 2, (), 1), ('t13_simdvf', 13, 2, (), 2), ('t14_fill', 14, 0, (), 1), ('t15_ascend_dual_copy', 15, 2, (), 1), ('t16_ascend_l1_to_l0a', 16, 2, (), 1), ('t17_ascend_gm_to_l1', 17, 0, (), 1), ('t18_ascend_l1_to_l0b', 18, 1, (), 1), ('t19_ascend_mmad', 19, 2, (), 1), ('t20_ascend_dual_copy', 20, 2, (), 1), ('t21_simdvf', 21, 2, (), 6)), buffer_schedules=(('value_1', 'fixed', 2), ('s0', 'fixed', 2), ('value_3', 'fixed', 2), ('value_5', 'fixed', 2), ('value_6', 'fixed', 1), ('value_9', 'fixed', 1), ('value_10', 'fixed', 2), ('value_12', 'fixed', 1), ('value_13', 'fixed', 1), ('value_16', 'fixed', 2), ('value_17', 'fixed', 1), ('l_ub', 'fixed', 1), ('m_ub', 'fixed', 1), ('scale_tile_ub', 'fixed', 1), ('value_34', 'fixed', 1), ('value_36', 'fixed', 1), ('o0', 'fixed', 2), ('value_37', 'fixed', 1), ('value_38', 'fixed', 1), ('value_39', 'fixed', 1), ('value_42', 'fixed', 1), ('value_43', 'fixed', 1), ('o_acc_ub', 'fixed', 1))):
                value_2 = T.tensor_view(
                    T.ptr_of(causal_mask[0:0 + 1, kt * 8:kt * 8 + 1]),
                    layout=((64, 8), (16, 1)),
                    shape=(64, 8),
                )
                value_1 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                )
                T.ascend_gm_to_ub(value_2, value_1)
                with Mesh((Topology("aic", 1),), layout=(1,), names=("c",)) as scope:
                    s0 = T.tensor_view(
                        0, dtype="f32", storage="tmem", layout=(64, 8), shape=(64, 8)
                    )
                    T.fill(s0, 0.0)
                    value_4 = T.tensor_view(T.ptr_of(q_nope), layout=(64, 64), shape=(64, 64))
                    value_3 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_gm_to_l1(value_4, value_3)
                    value_5 = T.tensor_view(
                        10240, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_l1_to_l0a(value_3, value_5)
                    value_7 = T.tensor_view(T.ptr_of(k_nope), layout=(16, 64), shape=(16, 64))
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7[kt * 8:kt * 8 + 1, 0:0 + 1]), layout=(8, 64), shape=(8, 64)
                    )
                    value_6 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_gm_to_l1(value_8, value_6)
                    value_9 = T.tensor_view(
                        3072, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_l1_to_l0b(value_6, value_9)
                    T.ascend_mmad(
                        s0,
                        value_5,
                        value_9,
                        a_layout="MK",
                        b_layout="NK",
                        init_zero=True,
                    )
                    value_11 = T.tensor_view(T.ptr_of(q_rope), layout=(64, 64), shape=(64, 64))
                    value_10 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_gm_to_l1(value_11, value_10)
                    value_12 = T.tensor_view(
                        10240, dtype="f16", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_l1_to_l0a(value_10, value_12)
                    value_14 = T.tensor_view(T.ptr_of(k_rope), layout=(16, 64), shape=(16, 64))
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14[kt * 8:kt * 8 + 1, 0:0 + 1]),
                        layout=(8, 64),
                        shape=(8, 64),
                    )
                    value_13 = T.tensor_view(
                        2048, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_gm_to_l1(value_15, value_13)
                    value_16 = T.tensor_view(
                        3072, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_l1_to_l0b(value_13, value_16)
                    T.ascend_mmad(
                        s0,
                        value_12,
                        value_16,
                        a_layout="MK",
                        b_layout="NK",
                        init_zero=False,
                    )
                    value_17 = T.tensor_view(
                        18432, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_dual_copy(s0, value_17)
                with T.SimdVF():
                    value_18 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    l_ub_2 = T.tensor_view(
                        35072, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                    )
                    T.ascend_ub_to_reg(l_ub_2, value_18)
                    value_19 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    m_ub_2 = T.tensor_view(
                        34816, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                    )
                    T.ascend_ub_to_reg(m_ub_2, value_19)
                    value_20 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    value_17_2 = T.tensor_view(
                        18432, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_ub_to_reg(value_17_2, value_20)
                    value_21 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    scale_tile_ub_2 = T.tensor_view(
                        32768, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_ub_to_reg(scale_tile_ub_2, value_21)
                    value_22 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    T.ascend_vmul(value_20, value_21, value_22)
                    value_23 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    value_1_2 = T.tensor_view(
                        16384, dtype="f32", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_ub_to_reg(value_1_2, value_23)
                    value_24 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    T.ascend_vadd(value_22, value_23, value_24)
                    value_25 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vreduce_max(value_24, value_25, axes=(-1,), keepdim=True)
                    value_26 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vmax(value_19, value_25, value_26)
                    value_27 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vsub(value_19, value_26, value_27)
                    value_28 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vexp(value_27, value_28)
                    value_29 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vmul(value_18, value_28, value_29)
                    value_30 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    T.ascend_vsub(value_24, value_26, value_30)
                    value_31 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f32", "rmem"])
                    T.ascend_vexp(value_30, value_31)
                    value_32 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vreduce_sum(value_31, value_32, axes=(-1,), keepdim=True)
                    value_33 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    T.ascend_vadd(value_29, value_32, value_33)
                    T.ascend_reg_to_ub(value_33, l_ub_2)
                    T.ascend_reg_to_ub(value_26, m_ub_2)
                    value_34 = T.tensor_view(
                        35328, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                    )
                    T.ascend_reg_to_ub(value_28, value_34)
                    value_35 = T.alloc_tensor(tensor_type=Tensor[(64, 8), "f16", "rmem"])
                    T.ascend_vcast(value_31, value_35, dtype="f16")
                    value_36 = T.tensor_view(
                        16384, dtype="f16", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_reg_to_ub(value_35, value_36)
                with Mesh((Topology("aic", 1),), layout=(1,), names=("c2",)) as scope_1:
                    o0 = T.tensor_view(
                        0, dtype="f32", storage="tmem", layout=(64, 64), shape=(64, 64)
                    )
                    T.fill(o0, 0.0)
                    value_36_2 = T.tensor_view(
                        16384, dtype="f16", storage="smem", layout=(64, 8), shape=(64, 8)
                    )
                    value_37 = T.tensor_view(
                        16384, dtype="f16", storage="tmem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_dual_copy(value_36_2, value_37)
                    value_38 = T.tensor_view(
                        17408, dtype="f16", storage="tmem", layout=(64, 8), shape=(64, 8)
                    )
                    T.ascend_l1_to_l0a(value_37, value_38)
                    value_40 = T.tensor_view(T.ptr_of(v), layout=(16, 64), shape=(16, 64))
                    value_41 = T.tensor_view(
                        T.ptr_of(value_40[kt * 8:kt * 8 + 1, 0:0 + 1]),
                        layout=(8, 64),
                        shape=(8, 64),
                    )
                    value_39 = T.tensor_view(
                        16384, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_gm_to_l1(value_41, value_39)
                    value_42 = T.tensor_view(
                        18432, dtype="f16", storage="tmem", layout=(8, 64), shape=(8, 64)
                    )
                    T.ascend_l1_to_l0b(value_39, value_42)
                    T.ascend_mmad(
                        o0,
                        value_38,
                        value_42,
                        a_layout="MK",
                        b_layout="KN",
                        init_zero=True,
                    )
                    value_43 = T.tensor_view(
                        16384, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_dual_copy(o0, value_43)
                with T.SimdVF():
                    value_44 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                    o_acc_ub_2 = T.tensor_view(
                        0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_ub_to_reg(o_acc_ub_2, value_44)
                    value_45 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                    value_34_2 = T.tensor_view(
                        35328, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                    )
                    T.ascend_ub_to_reg(value_34_2, value_45)
                    value_46 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                    T.ascend_vmul(value_44, value_45, value_46)
                    value_47 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                    value_43_2 = T.tensor_view(
                        16384, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                    )
                    T.ascend_ub_to_reg(value_43_2, value_47)
                    value_48 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                    T.ascend_vadd(value_46, value_47, value_48)
                    T.ascend_reg_to_ub(value_48, o_acc_ub_2)
            with T.SimdVF():
                value_49 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                o_acc_ub_3 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
                )
                T.ascend_ub_to_reg(o_acc_ub_3, value_49)
                value_50 = T.alloc_tensor(tensor_type=Tensor[(64, 1), "f32", "rmem"])
                l_ub_3 = T.tensor_view(
                    35072, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                T.ascend_ub_to_reg(l_ub_3, value_50)
                value_51 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f32", "rmem"])
                T.ascend_vdiv(value_49, value_50, value_51)
                value_52 = T.alloc_tensor(tensor_type=Tensor[(64, 64), "f16", "rmem"])
                T.ascend_vcast(value_51, value_52, dtype="f16")
                value_53 = T.tensor_view(
                    0, dtype="f16", storage="smem", layout=(64, 64), shape=(64, 64)
                )
                T.ascend_reg_to_ub(value_52, value_53)
            value_53_2 = T.tensor_view(
                0, dtype="f16", storage="smem", layout=(64, 64), shape=(64, 64)
            )
            value_54_1 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
            )
            T.ascend_ub_to_gm(value_53_2, value_54_1)
