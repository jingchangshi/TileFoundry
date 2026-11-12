"""56-AIV ragged ownership, 1/8 split group (NBASE=1, REM=8, TILES=64), via tf.minimum."""
from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget
L = 64
TILES = 64
CORES = 56
NBASE = 1
REM = 8

@module(entry="own", target=Ascend950PRTarget(), topologies=(Topology("aiv", CORES),))
class Own56R:
    @func
    def own(x: Tensor[(TILES * L,), "bf16"]) -> Tensor[(TILES * L,), "bf16"]:
        with Mesh(("aiv",), layout=(CORES,), names=("core",)) as aiv:
            one = 1
            tail = tf.minimum(aiv.core, REM)
            start = aiv.core * NBASE + tail
            nxt = aiv.core + one
            tail2 = tf.minimum(nxt, REM)
            extra = tail2 - tail
            count = NBASE + extra
            y = tf.empty(Tensor[(TILES * L,), "bf16", (TILES * L,), "gmem"])
            for t in tf.tile(count, 1):
                idx = start + t
                ub = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
                st = tf.schedule((x[idx * L:idx * L + L], ub), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    c = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                    lc = tf.schedule((st, c), op=T.ascend_vld(dist="UNPK_B16"))
                    m = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                    pm = tf.schedule((m,), op=T.ascend_pset16(elem_width=16))
                    fr = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    cv = tf.schedule((lc, pm, fr), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                    m2 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                    pm2 = tf.schedule((m2,), op=T.ascend_pset16(elem_width=16))
                    co = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                    rv = tf.schedule((cv, pm2, co), op=T.ascend_vcvt(direction="reverse", part=0, elem_width=16))
                    uo = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
                    wv = tf.schedule((rv, uo), op=T.ascend_vsts(pack="PK_B32"))
                y = tf.insert_slice(y, wv, (idx * L,))
            return y