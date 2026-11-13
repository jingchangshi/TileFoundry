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
        v_stack: Tensor[(4, 1, 4, 64), "bf16"], x: Tensor[(1, 4, 64), "bf16"], w: Tensor[(64,), "bf16"], gamma: Tensor[(64,), "bf16"], scratch: Tensor[(16,), "f32"], scratch_1: Tensor[(4,), "f32"], scratch_2: Tensor[(16,), "f32"], scratch_3: Tensor[(256,), "bf16"], out: Tensor[(1, 4, 64), "bf16"]
    ):
        with Mesh((Topology("aiv", 2),), layout=(1,), names=("v",)) as cta:
            acc_ub = T.tensor_view(0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64))
            T.fill(acc_ub, 0.0)
            scores_gm_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 16]), layout=(4, 4), shape=(4, 4))
            T.fill(scores_gm_1, 0.0)
            z_m1 = T.tensor_view(1024, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(z_m1, 0.0)
            eps_ub = T.tensor_view(1040, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(eps_ub, 1e-06)
            value_1 = T.tensor_view(T.ptr_of(gamma), layout=(1, 64), shape=(1, 64))
            value = T.tensor_view(1056, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64))
            T.ascend_gm_to_ub(value_1, value)
            inv_d_ub = T.tensor_view(1184, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(inv_d_ub, 0.015625)
            scale_ub = T.tensor_view(1200, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1))
            T.fill(scale_ub, 0.125)
            value_3 = T.tensor_view(T.ptr_of(w), layout=(1, 64), shape=(1, 64))
            value_2 = T.tensor_view(
                1216, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
            )
            T.ascend_gm_to_ub(value_3, value_2)
            for n in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True, fixed_ii=2, task_intents=(('t0_ascend_gm_to_ub', 0, 0, (), 1), ('t1_simtvf', 1, 1, (), 1), ('t2_ascend_ub_to_gm', 2, 0, (), 1), ('t3_copy', 3, 0, (), 1)), buffer_schedules=(('value_4', 'fixed', 1), ('inv_d_ub', 'fixed', 1), ('eps_ub', 'fixed', 1), ('value_2', 'fixed', 1), ('value', 'fixed', 1), ('scale_ub', 'fixed', 1), ('value_27', 'fixed', 1), ('value_29', 'fixed', 1))):
                value_5 = T.tensor_view(T.ptr_of(v_stack), layout=(4, 4, 64), shape=(4, 4, 64))
                value_6 = T.tensor_view(
                    T.ptr_of(value_5[n:n + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 4, 64),
                    shape=(1, 4, 64),
                )
                value_7 = T.tensor_view(T.ptr_of(value_6), layout=(4, 64), shape=(4, 64))
                value_4 = T.tensor_view(
                    1344, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_gm_to_ub(value_7, value_4)
                with T.SimtVF():
                    value_8 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                    value_4_2 = T.tensor_view(
                        1344, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                    )
                    T.ascend_ub_to_reg(value_4_2, value_8)
                    value_9 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vcast(value_8, value_9, dtype="f32")
                    value_10 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_9, value_9, value_10)
                    value_11 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vreduce_sum(value_10, value_11, axes=(-1,), keepdim=True)
                    value_12 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    inv_d_ub_2 = T.tensor_view(
                        1184, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(inv_d_ub_2, value_12)
                    value_13 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmul(value_11, value_12, value_13)
                    value_14 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    eps_ub_2 = T.tensor_view(
                        1040, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(eps_ub_2, value_14)
                    value_15 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vadd(value_13, value_14, value_15)
                    value_16 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vrsqrt(value_15, value_16)
                    value_17 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_9, value_16, value_17)
                    value_18 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "bf16", "rmem"])
                    value_2_2 = T.tensor_view(
                        1216, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
                    )
                    T.ascend_ub_to_reg(value_2_2, value_18)
                    value_19 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                    T.ascend_vcast(value_18, value_19, dtype="f32")
                    value_20 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "bf16", "rmem"])
                    value_28 = T.tensor_view(
                        1056, dtype="bf16", storage="smem", layout=(1, 64), shape=(1, 64)
                    )
                    T.ascend_ub_to_reg(value_28, value_20)
                    value_21 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                    T.ascend_vcast(value_20, value_21, dtype="f32")
                    value_22 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", "rmem"])
                    T.ascend_vmul(value_19, value_21, value_22)
                    value_23 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_17, value_22, value_23)
                    value_24 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vreduce_sum(value_23, value_24, axes=(-1,), keepdim=True)
                    value_25 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    scale_ub_2 = T.tensor_view(
                        1200, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(scale_ub_2, value_25)
                    value_26 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    T.ascend_vmul(value_24, value_25, value_26)
                    value_27 = T.tensor_view(
                        1024, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_reg_to_ub(value_26, value_27)
                value_27_2 = T.tensor_view(
                    1024, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                )
                value_29 = T.tensor_view(T.ptr_of(value_27_2), layout=(4,), shape=(4,))
                value_28_1 = T.tensor_view(T.ptr_of(scratch_1[0:0 + 4]), layout=(4,), shape=(4,))
                T.ascend_ub_to_gm(value_29, value_28_1)
                value_30 = T.tensor_view(T.ptr_of(value_28_1), layout=(1, 4), shape=(1, 4))
                scores_gm_2 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 16]), layout=(4, 4), shape=(4, 4)
                )
                window = T.tensor_view(
                    T.ptr_of(scores_gm_2[n:n + 1, 0:0 + 4]), layout=((1, 4), (4, 1)), shape=(1, 4)
                )
                T.copy(value_30, window)
            value_31 = T.tensor_view(1024, dtype="f32", storage="smem", layout=(4, 4), shape=(4, 4))
            T.ascend_gm_to_ub(scores_gm_1, value_31)
            with T.SimtVF():
                value_32 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
                value_31_2 = T.tensor_view(
                    1024, dtype="f32", storage="smem", layout=(4, 4), shape=(4, 4)
                )
                T.ascend_ub_to_reg(value_31_2, value_32)
                value_33 = T.alloc_tensor(tensor_type=Tensor[(1, 4), "f32", "rmem"])
                T.ascend_vreduce_max(value_32, value_33, axes=(0,), keepdim=True)
                value_34 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
                T.ascend_vsub(value_32, value_33, value_34)
                value_35 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
                T.ascend_vexp(value_34, value_35)
                value_36 = T.alloc_tensor(tensor_type=Tensor[(1, 4), "f32", "rmem"])
                T.ascend_vreduce_sum(value_35, value_36, axes=(0,), keepdim=True)
                value_37 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
                T.ascend_vdiv(value_35, value_36, value_37)
                value_38 = T.tensor_view(
                    1024, dtype="f32", storage="smem", layout=(4, 4), shape=(4, 4)
                )
                T.ascend_reg_to_ub(value_37, value_38)
            probs_gm_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 16]), layout=(4, 4), shape=(4, 4))
            T.fill(probs_gm_1, 0.0)
            value_38_2 = T.tensor_view(
                1024, dtype="f32", storage="smem", layout=(4, 4), shape=(4, 4)
            )
            T.ascend_ub_to_gm(value_38_2, probs_gm_1)
            for n_2 in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True, fixed_ii=2, task_intents=(('t0_ascend_gm_to_ub', 0, 0, (), 1), ('t1_ascend_gm_to_ub', 1, 0, (), 1), ('t2_simtvf', 2, 1, (), 1)), buffer_schedules=(('value_39', 'fixed', 1), ('value_43', 'fixed', 2), ('acc_ub', 'fixed', 1))):
                probs_gm_2 = T.tensor_view(
                    T.ptr_of(scratch_2[0:0 + 16]), layout=(4, 4), shape=(4, 4)
                )
                value_40 = T.tensor_view(
                    T.ptr_of(probs_gm_2[n_2:n_2 + 1, 0:0 + 1]), layout=(1, 4), shape=(1, 4)
                )
                value_41 = T.tensor_view(T.ptr_of(value_40), layout=(4,), shape=(4,))
                value_42 = T.tensor_view(T.ptr_of(value_41), layout=(4, 1), shape=(4, 1))
                value_39 = T.tensor_view(
                    1024, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                )
                T.ascend_gm_to_ub(value_42, value_39)
                value_44 = T.tensor_view(T.ptr_of(v_stack), layout=(4, 4, 64), shape=(4, 4, 64))
                value_45 = T.tensor_view(
                    T.ptr_of(value_44[n_2:n_2 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(1, 4, 64),
                    shape=(1, 4, 64),
                )
                value_46 = T.tensor_view(T.ptr_of(value_45), layout=(4, 64), shape=(4, 64))
                value_43 = T.tensor_view(
                    1040, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_gm_to_ub(value_46, value_43)
                with T.SimtVF():
                    value_47 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    acc_ub_2 = T.tensor_view(
                        0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                    )
                    T.ascend_ub_to_reg(acc_ub_2, value_47)
                    value_48 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                    value_43_2 = T.tensor_view(
                        1040, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                    )
                    T.ascend_ub_to_reg(value_43_2, value_48)
                    value_49 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vcast(value_48, value_49, dtype="f32")
                    value_50 = T.alloc_tensor(tensor_type=Tensor[(4, 1), "f32", "rmem"])
                    value_39_2 = T.tensor_view(
                        1024, dtype="f32", storage="smem", layout=(4, 1), shape=(4, 1)
                    )
                    T.ascend_ub_to_reg(value_39_2, value_50)
                    value_51 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vmul(value_49, value_50, value_51)
                    value_52 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                    T.ascend_vadd(value_47, value_51, value_52)
                    T.ascend_reg_to_ub(value_52, acc_ub_2)
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
            value_54 = T.tensor_view(T.ptr_of(x), layout=(4, 64), shape=(4, 64))
            value_53 = T.tensor_view(
                1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            T.ascend_gm_to_ub(value_54, value_53)
            with T.SimtVF():
                value_55 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                value_53_2 = T.tensor_view(
                    1024, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(value_53_2, value_55)
                value_56 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vcast(value_55, value_56, dtype="f32")
                value_57 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                c_res_ub_2 = T.tensor_view(
                    3072, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(c_res_ub_2, value_57)
                value_58 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vmul(value_56, value_57, value_58)
                value_59 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                acc_ub_3 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(acc_ub_3, value_59)
                value_60 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                c_agg_ub_2 = T.tensor_view(
                    2048, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_ub_to_reg(c_agg_ub_2, value_60)
                value_61 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vmul(value_59, value_60, value_61)
                value_62 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "f32", "rmem"])
                T.ascend_vadd(value_58, value_61, value_62)
                value_63 = T.alloc_tensor(tensor_type=Tensor[(4, 64), "bf16", "rmem"])
                T.ascend_vcast(value_62, value_63, dtype="bf16")
                value_64 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.ascend_reg_to_ub(value_63, value_64)
            value_64_2 = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(4, 64), shape=(4, 64)
            )
            value_65_1 = T.tensor_view(
                T.ptr_of(scratch_3[0:0 + 256]), layout=(4, 64), shape=(4, 64)
            )
            T.ascend_ub_to_gm(value_64_2, value_65_1)
