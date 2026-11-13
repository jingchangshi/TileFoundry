from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import AscendTarget


@module(
    entry="forward",
    target=AscendTarget("ascend.950pr"),
    topologies=(Topology("block", 1), Topology("aic", 1), Topology("aiv", 2), Topology("lane", 64),),
)
class AttentionResidualScheduled:
    @prim_func(target=AscendTarget("ascend.950pr"))
    def forward(
        v_stack: Tensor[(4, 1, 4, 64), "bf16"], x: Tensor[(1, 4, 64), "bf16"], w: Tensor[(64,), "bf16"], gamma: Tensor[(64,), "bf16"], scratch: Tensor[(256,), "bf16"], out: Tensor[(1, 4, 64), "bf16"]
    ):
        with Mesh((Topology("aiv", 2),), layout=(2,), names=("v",)) as cta:
            z_m1 = T.tensor_view(1280, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(z_m1, 0.0)
            m_ub = T.tensor_view(1792, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(m_ub, -1e+30)
            l_ub = T.tensor_view(4096, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(l_ub, 0.0)
            acc_ub = T.tensor_view(0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64))
            T.fill(acc_ub, 0.0)
            eps_ub = T.tensor_view(1808, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(eps_ub, 1e-06)
            inv_d_ub = T.tensor_view(1824, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(inv_d_ub, 0.015625)
            value_1 = T.tensor_view(T.ptr_of(gamma), layout=(1, 64), shape=(1, 64))
            value = T.tensor_view(1024, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64))
            T.ascend_gm_to_ub(value_1, value)
            value_3 = T.tensor_view(T.ptr_of(w), layout=(1, 64), shape=(1, 64))
            value_2 = T.tensor_view(
                1152, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
            )
            T.ascend_gm_to_ub(value_3, value_2)
            with T.SimdVF():
                value_4 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "bf16", "rmem"])
                value_2_2 = T.tensor_view(
                    1152, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
                )
                T.ascend_ub_to_reg(value_2_2, value_4)
                value_5 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                T.ascend_vcast(value_4, value_5, dtype="f32")
                value_6 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "bf16", "rmem"])
                value_56 = T.tensor_view(
                    1024, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
                )
                T.ascend_ub_to_reg(value_56, value_6)
                value_7 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                T.ascend_vcast(value_6, value_7, dtype="f32")
                value_8 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                T.ascend_vmul(value_5, value_7, value_8)
                value_9 = T.tensor_view(
                    1536, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)
                )
                T.ascend_reg_to_ub(value_8, value_9)
            scale_ub = T.tensor_view(1840, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(scale_ub, 0.125)
            for n in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True, fixed_ii=1, task_intents=(('t0_ascend_gm_to_ub', 0, 0, (), 1), ('t1_simdvf', 1, 1, (), 1)), buffer_schedules=(('value_10', 'fixed', 2), ('acc_ub', 'fixed', 1), ('m_ub', 'fixed', 1), ('inv_d_ub', 'fixed', 1), ('eps_ub', 'fixed', 1), ('value_9', 'fixed', 1), ('scale_ub', 'fixed', 1), ('l_ub', 'fixed', 1))):
                value_11 = T.tensor_view(T.ptr_of(v_stack), layout=(4, 4, 64), shape=(4, 4, 64))
                value_12 = T.tensor_view(
                    T.ptr_of(value_11[n:n + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 4, 64),
                    shape=(1, 4, 64),
                )
                value_13 = T.tensor_view(T.ptr_of(value_12), layout=(4, 64), shape=(4, 64))
                value_10 = T.tensor_view(
                    1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_gm_to_ub(value_13, value_10)
                with T.SimdVF():
                    value_14 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    acc_ub_2 = T.tensor_view(
                        0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                    )
                    T.ascend_ub_to_reg(acc_ub_2, value_14)
                    value_15 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    m_ub_2 = T.tensor_view(
                        1792, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(m_ub_2, value_15)
                    value_16 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                    value_10_2 = T.tensor_view(
                        1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                    )
                    T.ascend_ub_to_reg(value_10_2, value_16)
                    value_17 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vcast(value_16, value_17, dtype="f32")
                    value_18 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_17, value_17, value_18)
                    value_19 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vreduce_sum(value_18, value_19, axes=(-1,), keepdim=True)
                    value_20 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    inv_d_ub_2 = T.tensor_view(
                        1824, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(inv_d_ub_2, value_20)
                    value_21 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmul(value_19, value_20, value_21)
                    value_22 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    eps_ub_2 = T.tensor_view(
                        1808, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(eps_ub_2, value_22)
                    value_23 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vadd(value_21, value_22, value_23)
                    value_24 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vrsqrt(value_23, value_24)
                    value_25 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_17, value_24, value_25)
                    value_26 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                    value_9_2 = T.tensor_view(
                        1536, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)
                    )
                    T.ascend_ub_to_reg(value_9_2, value_26)
                    value_27 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_25, value_26, value_27)
                    value_28 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vreduce_sum(value_27, value_28, axes=(-1,), keepdim=True)
                    value_29 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    scale_ub_2 = T.tensor_view(
                        1840, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(scale_ub_2, value_29)
                    value_30 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmul(value_28, value_29, value_30)
                    value_31 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmax(value_15, value_30, value_31)
                    value_32 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vsub(value_15, value_31, value_32)
                    value_33 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vexp(value_32, value_33)
                    value_34 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_14, value_33, value_34)
                    value_35 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vsub(value_30, value_31, value_35)
                    value_36 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vexp(value_35, value_36)
                    value_37 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_17, value_36, value_37)
                    value_38 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vadd(value_34, value_37, value_38)
                    T.ascend_reg_to_ub(value_38, acc_ub_2)
                    value_39 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    l_ub_2 = T.tensor_view(
                        4096, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(l_ub_2, value_39)
                    value_40 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmul(value_39, value_33, value_40)
                    value_41 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vadd(value_40, value_36, value_41)
                    T.ascend_reg_to_ub(value_41, l_ub_2)
                    T.ascend_reg_to_ub(value_31, m_ub_2)
            z_md = T.tensor_view(1024, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64))
            T.fill(z_md, 0.0)
            c_agg_ub = T.tensor_view(
                2048, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            T.fill(c_agg_ub, 0.5)
            c_res_ub = T.tensor_view(
                3072, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            T.fill(c_res_ub, 0.5)
            value_43 = T.tensor_view(T.ptr_of(x), layout=(4, 64), shape=(4, 64))
            value_42 = T.tensor_view(
                1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            T.ascend_gm_to_ub(value_43, value_42)
            with T.SimdVF():
                value_44 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                value_42_2 = T.tensor_view(
                    1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(value_42_2, value_44)
                value_45 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vcast(value_44, value_45, dtype="f32")
                value_46 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                c_res_ub_2 = T.tensor_view(
                    3072, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(c_res_ub_2, value_46)
                value_47 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vmul(value_45, value_46, value_47)
                value_48 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                acc_ub_3 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(acc_ub_3, value_48)
                value_49 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                l_ub_3 = T.tensor_view(
                    4096, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                )
                T.ascend_ub_to_reg(l_ub_3, value_49)
                value_50 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vdiv(value_48, value_49, value_50)
                value_51 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                c_agg_ub_2 = T.tensor_view(
                    2048, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(c_agg_ub_2, value_51)
                value_52 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vmul(value_50, value_51, value_52)
                value_53 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vadd(value_47, value_52, value_53)
                value_54 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                T.ascend_vcast(value_53, value_54, dtype="bf16")
                value_55 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_reg_to_ub(value_54, value_55)
            value_55_2 = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            value_56_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 256]), layout=(4, 64), shape=(4, 64))
            T.ascend_ub_to_gm(value_55_2, value_56_1)
