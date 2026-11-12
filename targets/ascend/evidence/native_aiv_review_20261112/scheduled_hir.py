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
