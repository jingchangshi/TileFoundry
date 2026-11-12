# Native AR arithmetic micro-case review

This is the complete review export of the restricted two-input x*z primitive (z=x gives the RMS squared-input step), four tiles of 64 float32 elements. It is not the full attention_residual operator.

**Compilation: PASS. Device numerical execution: NOT_RUN.** Device initialization failed at set_device(0) with 507033/E39006 before kernel launch. These dumps do not establish E2E success. No NPU work was run for this export.

Both HIR and TIR were produced by the formal project printers; no pseudo IR was substituted. Before and after export, emit_for_codegen returned unchanged TIR digest 936c5b3ca6eb44f1a5d3466220446a429024a00c6acf7c98cd0f0de358913da3 and TL SHA256 022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9. Final TIR is the actual static materialization; it is not a claim of symbolic software pipelining.

## Authored Scheduled HIR source

Source file: `author_native_square.py`. Contents below are complete and unchanged.

```python
"""Native AR RMS primitive: elementwise product, 4 x 64 float32 lanes.

With z=x this is the squared input before the RMS reduction. This is a
limited native backend bring-up primitive, not the full attention operator.
"""
from __future__ import annotations

# ruff: noqa: F403, F405, I001, F841  -- curated DSL star import; 'as aiv' binds a mesh var

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64


@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class NativeSquare:
    @func
    def k(x: Tensor[(256,), "f32"], z: Tensor[(256,), "f32"]) -> Tensor[(256,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            y = tf.empty(Tensor[(256,), "f32", (256,), "gmem"])
            for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):
                ux = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sx = tf.schedule((x[i * L:i * L + L], ux), op=T.ascend_gm_to_ub())
                uz = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sz = tf.schedule((z[i * L:i * L + L], uz), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    va = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    vb = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    mask = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                    la = tf.schedule((sx, va), op=T.ascend_vld())
                    lb = tf.schedule((sz, vb), op=T.ascend_vld())
                    pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                    acc = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    total = tf.schedule((la, lb, pred, acc), op=T.ascend_vmul())
                    stored = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                    back = tf.schedule((total, stored), op=T.ascend_vsts())
                y = tf.insert_slice(y, back, (i * L,))
            return y
```

## Canonical Scheduled HIR — formal HirPrinter

Source file: `scheduled_hir.py`. Contents below are complete and unchanged.

```python
from __future__ import annotations

from tilefoundry.dsl import *

@func(mesh=Mesh((Topology("aiv", 1),), layout=(1,), names=("core",)))
def k(
    x: Tensor[(256,), "f32"],
    z: Tensor[(256,), "f32"]
) -> Tensor[(256,), "f32"]:
    y = tf.empty(type=Tensor[(256,), "f32"])
    for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):
        v1 = 64
        v2 = i * 64
        v4 = tf.slice(x, (v2,), sizes=(64,), strides=(1,))
        ux = tf.empty(type=Tensor[(64,), "f32", "smem"])
        sx = tf.schedule((v4, ux), op=T.ascend_gm_to_ub())
        v7 = 64
        v8 = i * 64
        v10 = tf.slice(z, (v8,), sizes=(64,), strides=(1,))
        uz = tf.empty(type=Tensor[(64,), "f32", "smem"])
        sz = tf.schedule((v10, uz), op=T.ascend_gm_to_ub())
        with T.SimdVF():
            va = tf.empty(type=Tensor[(64,), "f32", "rmem"])
            la = tf.schedule((sx, va), op=T.ascend_vld())
            vb = tf.empty(type=Tensor[(64,), "f32", "rmem"])
            lb = tf.schedule((sz, vb), op=T.ascend_vld())
            mask = tf.empty(type=Tensor[(64,), "bool", "rmem"])
            pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
            acc = tf.empty(type=Tensor[(64,), "f32", "rmem"])
            total = tf.schedule((la, lb, pred, acc), op=T.ascend_vmul())
            stored = tf.empty(type=Tensor[(64,), "f32", "smem"])
            back = tf.schedule((total, stored), op=T.ascend_vsts())
        v24 = 64
        v25 = tf.mul(i, v24)
        y_2 = tf.insert_slice(y, back, (v25,))
        y = y_2
    return y
```

## Final materialized TIR — formal TirPrinter

Source file: `final_tir.py`. Contents below are complete and unchanged.

```python
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
```

## Actual compiled TileLang source

Source file: `tilelang.py`. Contents below are complete and unchanged.

```python
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

@T.prim_func
def main(x: T.Tensor((256,), "float32"), z: T.Tensor((256,), "float32"), out: T.Tensor((256,), "float32")):
    with T.Kernel(1) as bx:
        with T.Vector(vector=1):
            ub = T.alloc_shared((192,), "float32")
            T.copy(x[0:64], ub[0:64])
            T.ascend_set_flag("MTE2_V", 0)
            T.copy(z[0:64], ub[64:128])
            T.ascend_set_flag("MTE2_V", 1)
            T.ascend_wait_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 1)
            with T.SimdVF():
                r1 = S.vld(ub[0])
                r2 = S.vld(ub[64])
                r3 = S.pset(32, "PAT_ALL")
                r4 = S.vmul(r1, r2, r3)
                r5 = S.pset(32, "PAT_ALL")
                S.vsts(ub[128], r4, r5)
            T.ascend_set_flag("V_MTE3", 2)
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_wait_flag("V_MTE3", 2)
            T.copy(ub[128:192], out[0:64])
            T.ascend_set_flag("MTE3_V", 2)
            T.ascend_wait_flag("V_MTE2", 0)
            T.copy(x[64:128], ub[0:64])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("V_MTE2", 1)
            T.copy(z[64:128], ub[64:128])
            T.ascend_set_flag("MTE2_V", 1)
            T.ascend_wait_flag("MTE3_V", 2)
            T.ascend_wait_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 1)
            with T.SimdVF():
                r6 = S.vld(ub[0])
                r7 = S.vld(ub[64])
                r8 = S.pset(32, "PAT_ALL")
                r9 = S.vmul(r6, r7, r8)
                r10 = S.pset(32, "PAT_ALL")
                S.vsts(ub[128], r9, r10)
            T.ascend_set_flag("V_MTE3", 2)
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_wait_flag("V_MTE3", 2)
            T.copy(ub[128:192], out[64:128])
            T.ascend_set_flag("MTE3_V", 2)
            T.ascend_wait_flag("V_MTE2", 0)
            T.copy(x[128:192], ub[0:64])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("V_MTE2", 1)
            T.copy(z[128:192], ub[64:128])
            T.ascend_set_flag("MTE2_V", 1)
            T.ascend_wait_flag("MTE3_V", 2)
            T.ascend_wait_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 1)
            with T.SimdVF():
                r11 = S.vld(ub[0])
                r12 = S.vld(ub[64])
                r13 = S.pset(32, "PAT_ALL")
                r14 = S.vmul(r11, r12, r13)
                r15 = S.pset(32, "PAT_ALL")
                S.vsts(ub[128], r14, r15)
            T.ascend_set_flag("V_MTE3", 2)
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_wait_flag("V_MTE3", 2)
            T.copy(ub[128:192], out[128:192])
            T.ascend_set_flag("MTE3_V", 2)
            T.ascend_wait_flag("V_MTE2", 0)
            T.copy(x[192:256], ub[0:64])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("V_MTE2", 1)
            T.copy(z[192:256], ub[64:128])
            T.ascend_set_flag("MTE2_V", 1)
            T.ascend_wait_flag("MTE3_V", 2)
            T.ascend_wait_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 1)
            with T.SimdVF():
                r16 = S.vld(ub[0])
                r17 = S.vld(ub[64])
                r18 = S.pset(32, "PAT_ALL")
                r19 = S.vmul(r16, r17, r18)
                r20 = S.pset(32, "PAT_ALL")
                S.vsts(ub[128], r19, r20)
            T.ascend_set_flag("V_MTE3", 2)
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_wait_flag("V_MTE3", 2)
            T.copy(ub[128:192], out[192:256])
            T.ascend_set_flag("MTE3_V", 2)
            T.ascend_pipe_barrier("PIPE_ALL")
```
