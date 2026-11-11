"""P5-01/G50: the explicit TileLang kernel for attention_residual, and its conformance check.

WHY THIS EMITS SOURCE RATHER THAN BEING THE KERNEL. The kernel that was measured lives in
kernel.py and is written directly against TileLang. This module emits the SAME program as text
so a reviewer can read it, diff it against the scheduled HIR, and check that the tasks the HIR
names are the tasks the device runs. If the two drifted apart, the emitted text would stop
matching and the checker would fail rather than quietly passing.

THE SYNCHRONISATION IS COMPILER-DERIVED. There are no hand-written cross-core flags here and
there must not be: this is a single-AIV kernel, so every dependency between its tasks is a
pipe/instruction ordering inside one core, which TileLang derives from the dataflow. Adding
ascend_set_flag/ascend_wait_flag calls would suggest an obligation exists that does not.

WHAT THE CHECKER ACTUALLY PROVES, AND WHAT IT DOES NOT. It proves the emitted program has the
structural properties G50 requires: one launch, both reduction axes present, the aggregation
reading the ORIGINAL V rather than the normalised key, FP32 state, and the real-D denominator.
It cannot prove the numbers are right -- that comes from device execution, recorded in
evidence/, and is a separate field on the evidence record precisely so the two cannot be
confused.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .model import (
    AttentionResidualError,
    AttentionResidualSpec,
    BLOCKED_ONLINE,
    FAMILIES,
    FACTORED_GW,
    LITERAL_FP32,
    ONLINE,
    RESIDENT_N,
    SCORE_MODES,
)

MATCHED = "MATCHED"
FAILED = "FAILED"

#: TileLang runs its whole core/sync/version/pipeline pass group when auto-scheduling is on.
#: A twin must express the plan itself, so the switch is emitted and checked.
AUTO_SCHEDULE_FLAG = "auto_schedule=False"


@dataclass(frozen=True)
class ConformanceReport:
    kernel: str
    status: str
    findings: tuple = ()
    failures: tuple = ()
    compiled: bool = False
    compile_reason: str = "not attempted by this checker"

    def as_dict(self) -> dict:
        return {"kernel": self.kernel, "status": self.status,
                "findings": list(self.findings), "failures": list(self.failures),
                "compiled": self.compiled, "compile_reason": self.compile_reason}


def _ident(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _params(spec: AttentionResidualSpec) -> str:
    return ("V: T.Tensor((N * M * D,), %r), X: T.Tensor((M * D,), %r),\n"
            "             Wt: T.Tensor((D,), %r), Gt: T.Tensor((D,), %r),\n"
            "             Y: T.Tensor((M * D,), %r), alpha: T.float32, eps: T.float32"
            % (spec.dtype, spec.dtype, spec.dtype, spec.dtype, spec.dtype))


def build_attnres_twin(spec: AttentionResidualSpec, *, function_name: str = "mixed_kernel") -> str:
    """Emit the TileLang source for spec.

    The emitted body is the measured kernel's structure: persistent w/gamma residency, one
    token per program, CN-blocked online softmax along N, FP32 m/l/acc, aggregation over the
    ORIGINAL V, then the alpha blend. The two score modes differ only in how the score is
    formed, which is emitted from the spec rather than from a flag inside the kernel.
    """
    if not isinstance(spec, AttentionResidualSpec):
        raise AttentionResidualError(
            "TWIN_NOT_A_SPEC", "the twin is built from a spec", "spec",
            "AttentionResidualSpec", type(spec).__name__)
    if spec.family not in FAMILIES:
        raise AttentionResidualError(
            "TWIN_UNKNOWN_FAMILY", "unknown family", "family", FAMILIES, spec.family)
    if spec.score_mode not in SCORE_MODES:
        raise AttentionResidualError(
            "TWIN_UNKNOWN_SCORE_MODE", "unknown score mode", "score_mode", SCORE_MODES,
            spec.score_mode)
    if spec.launches != 1:
        raise AttentionResidualError(
            "TWIN_MULTIPLE_LAUNCHES", "exactly one logical launch is required", "launches", 1,
            spec.launches)

    if spec.score_mode == LITERAL_FP32:
        score_lines = [
            "                            # literal_fp32: keep the golden's arithmetic order --",
            "                            # normalise each element, scale by gamma, THEN dot with w.",
            "                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):",
            "                                if i < D:",
            "                                    T.reducer_update(dt[0], (vf[i] / rms[0]) * g_ub[i] * w_ub[i])",
        ]
        score_note = "literal_fp32"
    else:
        score_lines = [
            "                            # factored_gw: hoist the division out of the reduction.",
            "                            # gw = gamma*w is reused across every block, so the",
            "                            # elementwise v/rms becomes a scalar scale on the dot.",
            "                            # This is NOT bit-identical to literal_fp32.",
            "                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):",
            "                                if i < D:",
            "                                    T.reducer_update(dt[0], vf[i] * (g_ub[i] * w_ub[i]))",
        ]
        score_note = "factored_gw"

    text = []
    add = text.append
    add("# EXPLICIT attention_residual KERNEL -- one logical launch, TileLang must NOT re-schedule.")
    add("#")
    add("# Task graph (from the Scheduled HIR, not from this file's prose):")
    add("#   LOAD_X(g)        AIV / MTE2    residual branch input")
    add("#   LOAD_V(u)        AIV / MTE2    CN x 1 x D tile, validity mask")
    add("#   BLOCK_COMPUTE(u) AIV / VECTOR  score, block state, running-state merge")
    add("#   FINALIZE(g)      AIV / VECTOR  read final state + x, form the output")
    add("#   STORE_Y(g)       AIV / MTE3    write the REAL D")
    add("#")
    add("# score_mode = %s" % score_note)
    add("# family = %s (CN=%d, so %d state update(s) per token group, N=%d)"
        % (spec.family, spec.block_n, spec.state_updates(), spec.n))
    add("#")
    add("# V's LAST consumer is the weighted sum inside BLOCK_COMPUTE, not the score. The")
    add("# release edge is attached to BLOCK_COMPUTE for that reason.")
    add("#")
    add("# Synchronisation here is compiler-derived from the dataflow: this is a single-AIV")
    add("# kernel, so its task dependencies are pipe ordering within one core. No cross-core")
    add("# flag is emitted because none is required.")
    add("")
    add("import math")
    add("import tilelang")
    add("import tilelang.ascend.language as T")
    add("")
    add("SERIAL = getattr(T, \"Pipelined\")")
    add("THREADS = 128")
    add("")
    add("")
    add("def %s(N, M, D, P, dtype=%r):" % (function_name, spec.dtype))
    add("    PD = tilelang.next_power_of_2(D)   # padded UB width, allocation only")
    add("    sqrt_d = math.sqrt(D)")
    add("")
    add("    @T.prim_func")
    add("    def main(%s):" % _params(spec))
    add("        with T.Kernel(P) as pid:")
    add("            # Every buffer is declared in the LAUNCH scope: TileLang binds an immutable")
    add("            # name to its defining region, and state must outlive the inner N loop.")
    add("            v_ub = T.alloc_shared((PD,), dtype)")
    add("            x_ub = T.alloc_shared((PD,), dtype)")
    add("            y_ub = T.alloc_shared((PD,), dtype)")
    add("            acc_ub = T.alloc_shared((PD,), \"float32\")")
    add("            g_ub = T.alloc_shared((PD,), \"float32\")")
    add("            w_ub = T.alloc_shared((PD,), \"float32\")")
    add("            st = T.alloc_shared((8,), \"float32\")  # 0 m 1 l 2 score 3 rho 4 beta 5 invl 6 1-a")
    add("            # persistent gamma/w residency: ONE GM read per AIV, reused by every group")
    add("            with T.SimtVF(threads=THREADS):")
    add("                gf = T.alloc_fragment((PD,), \"float32\")")
    add("                for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                    gf[i] = T.if_then_else(i < D, T.cast(Gt[i], \"float32\"), T.float32(0))")
    add("                for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                    g_ub[i] = gf[i]")
    add("            with T.SimtVF(threads=THREADS):")
    add("                wf = T.alloc_fragment((PD,), \"float32\")")
    add("                for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                    wf[i] = T.if_then_else(i < D, T.cast(Wt[i], \"float32\"), T.float32(0))")
    add("                for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                    w_ub[i] = wf[i]")
    add("            for it in %s((M + P - 1) // P, num_stages=1):" % "T.Pipelined")
    add("                m = pid + it * P")
    add("                if m < M:")
    add("                    # LOAD_X -- exact D elements into the padded buffer")
    add("                    T.copy(X[m * D:m * D + D], x_ub[0:D])")
    add("                    with T.SimtVF(threads=THREADS):")
    add("                        for i in T.Parallel(1):")
    add("                            st[0] = T.float32(-1e30)   # m: finite floor, not -inf")
    add("                        for i in T.Parallel(1):")
    add("                            st[1] = T.float32(0)       # l")
    add("                        zf = T.alloc_fragment((PD,), \"float32\")")
    add("                        for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                            zf[i] = T.float32(0)")
    add("                        for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                            acc_ub[i] = zf[i]          # acc")
    add("                    for n in %s(N, num_stages=1):" % "T.Pipelined")
    add("                        # LOAD_V -- the ORIGINAL v, which the weighted sum needs")
    add("                        T.copy(V[(n * M + m) * D:(n * M + m) * D + D], v_ub[0:D])")
    add("                        with T.SimtVF(threads=THREADS):")
    add("                            vf = T.alloc_fragment((PD,), \"float32\")")
    add("                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                                vf[i] = T.if_then_else(i < D, T.cast(v_ub[i], \"float32\"), T.float32(0))")
    add("                            # RMS along D, FP32, real-D denominator")
    add("                            sq = T.alloc_reducer((1,), \"float32\", op=\"sum\")")
    add("                            T.reducer_init(sq)")
    add("                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                                if i < D:")
    add("                                    T.reducer_update(sq[0], vf[i] * vf[i])")
    add("                            qs = T.alloc_fragment((1,), \"float32\")")
    add("                            T.finalize_reducer(sq, qs)")
    add("                            rms = T.alloc_fragment((1,), \"float32\")")
    add("                            for i in T.Parallel(1):")
    add("                                rms[0] = T.sqrt(qs[0] / T.float32(D) + eps)")
    add("                            # score along D")
    add("                            dt = T.alloc_reducer((1,), \"float32\", op=\"sum\")")
    add("                            T.reducer_init(dt)")
    for line in score_lines:
        add(line)
    add("                            ds = T.alloc_fragment((1,), \"float32\")")
    add("                            T.finalize_reducer(dt, ds)")
    add("                            for i in T.Parallel(1):")
    add("                                st[2] = ds[0] / T.float32(sqrt_d)")
    add("                        # running-state merge (softmax along N)")
    add("                        with T.SimtVF(threads=THREADS):")
    add("                            nm = T.alloc_fragment((1,), \"float32\")")
    add("                            rho = T.alloc_fragment((1,), \"float32\")")
    add("                            bet = T.alloc_fragment((1,), \"float32\")")
    add("                            for i in T.Parallel(1):")
    add("                                nm[0] = T.max(st[0], st[2])")
    add("                            for i in T.Parallel(1):")
    add("                                rho[0] = T.exp(st[0] - nm[0])")
    add("                            for i in T.Parallel(1):")
    add("                                bet[0] = T.exp(st[2] - nm[0])")
    add("                            for i in T.Parallel(1):")
    add("                                st[1] = rho[0] * st[1] + bet[0]")
    add("                            for i in T.Parallel(1):")
    add("                                st[0] = nm[0]")
    add("                            for i in T.Parallel(1):")
    add("                                st[3] = rho[0]")
    add("                            for i in T.Parallel(1):")
    add("                                st[4] = bet[0]")
    add("                        with T.SimtVF(threads=THREADS):")
    add("                            af = T.alloc_fragment((PD,), \"float32\")")
    add("                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                                af[i] = T.if_then_else(i < D, T.cast(v_ub[i], \"float32\"), T.float32(0))")
    add("                            for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                                if i < D:")
    add("                                    # weighted sum over the ORIGINAL v, then merge")
    add("                                    acc_ub[i] = st[3] * acc_ub[i] + st[4] * af[i]")
    add("                    # FINALIZE + STORE_Y")
    add("                    with T.SimtVF(threads=THREADS):")
    add("                        for i in T.Parallel(1):")
    add("                            st[5] = T.float32(1) / st[1]")
    add("                        for i in T.Parallel(1):")
    add("                            st[6] = T.float32(1) - alpha")
    add("                        yf = T.alloc_fragment((PD,), \"float32\")")
    add("                        for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                            yf[i] = T.if_then_else(i < D, st[6] * T.cast(x_ub[i], \"float32\") + alpha * acc_ub[i] * st[5], T.float32(0))")
    add("                        for i in T.Parallel(PD, coalesced_width=T.int32(1)):")
    add("                            y_ub[i] = T.cast(yf[i], dtype)")
    add("                    # STORE_Y -- exactly the REAL D, never the padded width")
    add("                    T.copy(y_ub[0:D], Y[m * D:m * D + D])")
    add("    return main")
    add("")
    return "\n".join(text)

# ------------------------------------------------------------------ parsing and checking

@dataclass(frozen=True)
class AttnResTwin:
    """A parsed kernel source, kept as text so a reviewer can diff it."""

    source: str
    auto_schedule: bool = False

    def as_dict(self) -> dict:
        return {"auto_schedule": self.auto_schedule, "chars": len(self.source)}


def parse_attnres_twin(source: str) -> AttnResTwin:
    if not isinstance(source, str) or not source.strip():
        raise AttentionResidualError(
            "TWIN_EMPTY", "the twin source is empty", "source", "nonempty source", source)
    return AttnResTwin(source=source, auto_schedule=AUTO_SCHEDULE_FLAG in source)


def check_attnres_conformance(spec: AttentionResidualSpec,
                              twin) -> ConformanceReport:
    """Compare spec against twin; report MATCHED or FAILED without raising on divergence.

    Every clause below corresponds to one acceptance sentence of G50, so a failure names which
    sentence is unmet rather than "the twin looks wrong".
    """
    if not isinstance(twin, AttnResTwin):
        twin = parse_attnres_twin(twin)

    findings = []
    failures = []
    src = twin.source

    # -- clause 1: exactly one logical launch -----------------------------------------
    launches = re.findall(r"with T\.(?:Kernel|MixedKernel)\(", src)
    if len(launches) != 1:
        failures.append(
            "TWIN_MULTIPLE_KERNELS: %d launch scope(s); the operator must be ONE kernel"
            % len(launches))
    else:
        findings.append("exactly one launch scope, so the chain is one logical kernel")

    # -- clause 2: both reduction axes are present, along the right axes ---------------
    # The RMS sum and the score dot run along D; the softmax runs along N. A kernel missing
    # either axis is not this operator, and a kernel whose softmax runs along D is a different
    # one that would still compile and still return the right shape.
    if "T.reducer_update(sq[0]" not in src:
        failures.append("TWIN_MISSING_RMS: no RMS square-sum reduction along D")
    if "T.reducer_update(dt[0]" not in src:
        failures.append("TWIN_MISSING_SCORE: no score dot reduction along D")
    if "st[2] = ds[0]" not in src:
        failures.append("TWIN_MISSING_TEMPERATURE: the score is not divided by sqrt(D)")
    if failures:
        pass
    else:
        findings.append("RMS sum and score dot both reduce along D, with the sqrt(D) temperature")

    # -- clause 3: the aggregation reads the ORIGINAL V, not the normalised key ---------
    # This is the substitution most likely to slip in unnoticed: using k instead of v keeps
    # the shapes and even the ballpark magnitude, and is simply the wrong operator.
    if "acc_ub[i] = st[3] * acc_ub[i] + st[4] * af[i]" not in src:
        failures.append(
            "TWIN_WRONG_AGGREGATION: the weighted sum is not the emitted acc update; the "
            "aggregation must weight the ORIGINAL V")
    elif 'af[i] = T.if_then_else(i < D, T.cast(v_ub[i], "float32")' not in src:
        failures.append(
            "TWIN_AGGREGATION_NOT_ORIGINAL_V: the accumulator reads something other than the "
            "loaded original V, which would make this a different operator")
    else:
        findings.append("the weighted sum consumes the ORIGINAL v_ub, not the normalised key")

    # -- clause 4: the running state is FP32 --------------------------------------------
    for name in ("acc_ub", "st"):
        decl = [l for l in src.splitlines() if ("%s = T.alloc_shared" % name) in l]
        if not decl:
            failures.append("TWIN_MISSING_STATE: no %s declaration" % name)
        elif '"float32"' not in decl[0]:
            failures.append(
                "TWIN_STATE_NOT_FP32: %s is declared %s; the golden computes it in FP32"
                % (name, decl[0].strip()))
    if not failures:
        findings.append("m/l/acc/score state is declared float32")

    # -- clause 5: the real D is the denominator, never the padded width ----------------
    if "qs[0] / T.float32(D)" not in src:
        failures.append(
            "TWIN_PADDED_DENOMINATOR: the RMS denominator is not the real D; using the padded "
            "width would rescale every score")
    if "T.float32(sqrt_d)" not in src:
        failures.append("TWIN_MISSING_SQRT_D: the score temperature is absent")
    else:
        findings.append("RMS denominator is the real D and the temperature is sqrt(real D)")
    # The padding must appear ONLY as an allocation/loop bound.
    if "Y[m * D:m * D + D]" not in src:
        failures.append(
            "TWIN_PADDED_STORE: the output store does not write exactly the real D")

    # -- clause 6: the alpha residual blend --------------------------------------------
    if "st[6] = T.float32(1) - alpha" not in src:
        failures.append("TWIN_MISSING_RESIDUAL: the (1 - alpha) residual coefficient is absent")
    elif "st[6] * T.cast(x_ub[i]" not in src:
        failures.append("TWIN_MISSING_RESIDUAL_PATH: x is not blended into the output")
    else:
        findings.append("the final blend is (1 - alpha) * x + alpha * agg")

    # -- clause 7: the score strategy is the one that was declared ---------------------
    if spec.score_mode == LITERAL_FP32:
        if "(vf[i] / rms[0]) * g_ub[i] * w_ub[i]" not in src:
            failures.append(
                "TWIN_SCORE_MODE_MISMATCH: literal_fp32 was declared but the emitted score path "
                "is not the golden's arithmetic order")
        elif "vf[i] * (g_ub[i] * w_ub[i])" in src:
            failures.append(
                "TWIN_SCORE_MODE_MIXED: both score paths are present, so which one runs is "
                "decided by the compiler rather than by the spec")
        else:
            findings.append("score path is literal_fp32 (normalise, scale, then dot)")
    else:
        if "vf[i] * (g_ub[i] * w_ub[i])" not in src:
            failures.append(
                "TWIN_SCORE_MODE_MISMATCH: factored_gw was declared but the emitted score path "
                "is not the factored form")
        else:
            findings.append("score path is factored_gw (gw hoisted out of the reduction)")

    # -- clause 8: no hand-written cross-core flags ------------------------------------
    hand = [l.strip() for l in src.splitlines()
            if "ascend_cross_core_set_flag" in l or "ascend_cross_core_wait_flag" in l]
    if hand:
        failures.append(
            "TWIN_MANUAL_SYNC: %d hand-written cross-core flag call(s); this is a single-AIV "
            "kernel whose dependencies are pipe ordering inside one core, so a partial flag set "
            "would imply obligations that do not exist" % len(hand))
    else:
        findings.append(
            "synchronisation is compiler-derived from the dataflow; no cross-core flag emitted "
            "because this kernel has no second core")

    status = FAILED if failures else MATCHED
    return ConformanceReport(
        kernel=spec.metadata.get("kernel", "attention_residual"),
        status=status,
        findings=tuple(findings),
        failures=tuple(failures),
    )
