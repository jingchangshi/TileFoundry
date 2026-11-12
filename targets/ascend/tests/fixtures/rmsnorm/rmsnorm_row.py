"""03-22 step 1 (path a): single-row RMSNorm structural realization at D=256.

One row of D=256 = FOUR 64-lane tiles. The ABI here names the four tiles explicitly
(x0..x3, w0..w3) instead of one (256,) row; windowing a single row into tiles is step 2.
This structural version computes the FULL row's rstd over all four tiles (four vmula
chains -> balanced ((0+1)+(2+3)) merge -> six vcadd -> explicit vdupv(POS_LOWEST) ->
vmuls(1/D) -> vadds(eps) -> exact vsqrt -> vdiv(1, sqrt)) and applies it to tile 0 with the
weight tile, then the reverse path (reverse vcvt -> carrier -> vsts PK_B32 -> bf16 UB -> GM).

Chains: rows=4 case -> FOUR chains; each chain has one tile here, so acc_i = v_i * v_i.
Weight is converted in a SEPARATE SimdVF (bf16->f32 into wf0 in UB); the main VF vld's it,
because CARRIER/RMEM cannot cross an ExecutionRegion.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64
TILES = 4
D = L * TILES
EPS = 1e-6


@module(entry="rmsnorm", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class RMSNormRow:
    @func
    def rmsnorm(
        x0: Tensor[(L,), "bf16"],
        x1: Tensor[(L,), "bf16"],
        x2: Tensor[(L,), "bf16"],
        x3: Tensor[(L,), "bf16"],
        w0: Tensor[(L,), "bf16"],
        w1: Tensor[(L,), "bf16"],
        w2: Tensor[(L,), "bf16"],
        w3: Tensor[(L,), "bf16"],
    ) -> Tensor[(L,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ux0 = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            ux1 = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            ux2 = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            ux3 = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            sx0 = tf.schedule((x0, ux0), op=T.ascend_gm_to_ub())
            sx1 = tf.schedule((x1, ux1), op=T.ascend_gm_to_ub())
            sx2 = tf.schedule((x2, ux2), op=T.ascend_gm_to_ub())
            sx3 = tf.schedule((x3, ux3), op=T.ascend_gm_to_ub())
            uw0 = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
            sw0 = tf.schedule((w0, uw0), op=T.ascend_gm_to_ub())
            # --- WEIGHT VF (separate): convert w0 bf16 -> f32, store to wf0 in UB ---
            wf0 = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
            with T.SimdVF():
                wc = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                lw = tf.schedule((sw0, wc), op=T.ascend_vld(dist="UNPK_B16"))
                mw = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                pw = tf.schedule((mw,), op=T.ascend_pset16(elem_width=16))
                wfr = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                wcv = tf.schedule((lw, pw, wfr), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                wst = tf.schedule((wcv, wf0), op=T.ascend_vsts())
            # --- MAIN COMPUTE VF (not split) ---
            with T.SimdVF():
                m32 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p32 = tf.schedule((m32,), op=T.ascend_pset(elem_width=32))
                m16 = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p16 = tf.schedule((m16,), op=T.ascend_pset16(elem_width=16))

                c0 = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                c1 = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                c2 = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                c3 = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                l0 = tf.schedule((sx0, c0), op=T.ascend_vld(dist="UNPK_B16"))
                l1 = tf.schedule((sx1, c1), op=T.ascend_vld(dist="UNPK_B16"))
                l2 = tf.schedule((sx2, c2), op=T.ascend_vld(dist="UNPK_B16"))
                l3 = tf.schedule((sx3, c3), op=T.ascend_vld(dist="UNPK_B16"))
                f0 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                f1 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                f2 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                f3 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                v0 = tf.schedule((l0, p16, f0), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                v1 = tf.schedule((l1, p16, f1), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                v2 = tf.schedule((l2, p16, f2), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                v3 = tf.schedule((l3, p16, f3), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                # four chains: acc_i = 0 + v_i * v_i
                a0 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                a1 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                a2 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                a3 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                q0 = tf.schedule((v0, v0, p32, a0), op=T.ascend_vmula())
                q1 = tf.schedule((v1, v1, p32, a1), op=T.ascend_vmula())
                q2 = tf.schedule((v2, v2, p32, a2), op=T.ascend_vmula())
                q3 = tf.schedule((v3, v3, p32, a3), op=T.ascend_vmula())
                # balanced merge ((0+1)+(2+3))
                s01 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                s23 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                mg = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                h01 = tf.schedule((q0, q1, p32, s01), op=T.ascend_vadd())
                h23 = tf.schedule((q2, q3, p32, s23), op=T.ascend_vadd())
                hmg = tf.schedule((h01, h23, p32, mg), op=T.ascend_vadd())
                # six vcadd 64 -> 1
                b1 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                b2 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                b3 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                b4 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                b5 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                b6 = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                r1 = tf.schedule((hmg, b1), op=T.ascend_vcadd(lanes=64))
                r2 = tf.schedule((r1, b2), op=T.ascend_vcadd(lanes=32))
                r3 = tf.schedule((r2, b3), op=T.ascend_vcadd(lanes=16))
                r4 = tf.schedule((r3, b4), op=T.ascend_vcadd(lanes=8))
                r5 = tf.schedule((r4, b5), op=T.ascend_vcadd(lanes=4))
                r6 = tf.schedule((r5, b6), op=T.ascend_vcadd(lanes=2))
                # explicit broadcast
                br = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                bb = tf.schedule((r6, br), op=T.ascend_vdupv(position=0))
                # mean = sum / D ; + eps ; sqrt ; 1/sqrt
                mean = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                mm = tf.schedule((bb, p32, mean), op=T.ascend_vmuls(value=1.0 / D))
                shifted = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                sh = tf.schedule((mm, p32, shifted), op=T.ascend_vadds(value=EPS))
                root = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                rt = tf.schedule((sh, p32, root), op=T.ascend_vsqrt())
                one = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                oo = tf.schedule((p32, one), op=T.ascend_vdup(value=1.0))
                rstd = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                rs = tf.schedule((oo, rt, p32, rstd), op=T.ascend_vdiv())
                # x0 * rstd
                xr = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                # Use the vcvt RESULT v0 (not the raw dst f0): the schedule result is the
                # bound value; the raw empty Var reads as zeros. This was the first real
                # error found by the end-to-end numeric check.
                xrr = tf.schedule((v0, rs, p32, xr), op=T.ascend_vmul())
                # * weight tile 0 (f32 from wf0 in UB)
                wr = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                # wst is the WEIGHT VF's leaving value (the f32 weight tile in UB); using it
                # here is what makes that region compute a value to leave.
                wrr = tf.schedule((wst, wr), op=T.ascend_vld())
                outv = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                ov = tf.schedule((xrr, wrr, p32, outv), op=T.ascend_vmul())
                # reverse: f32 -> carrier -> bf16 UB (PK_B32)
                m16b = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                p16b = tf.schedule((m16b,), op=T.ascend_pset16(elem_width=16))
                cout = tf.empty(Tensor[(L,), "bf16", (L,), "carrier"])
                rv = tf.schedule((ov, p16b, cout), op=T.ascend_vcvt(direction="reverse", part=0, elem_width=16))
                uy = tf.empty(Tensor[(L,), "bf16", (L,), "smem"])
                st = tf.schedule((rv, uy), op=T.ascend_vsts(pack="PK_B32"))
            out = tf.empty(Tensor[(L,), "bf16", (L,), "gmem"])
            return tf.schedule((st, out), op=T.ascend_ub_to_gm())
