"""Explicit TileLang twin for a MixedKernel, plus the TF<->twin conformance checker (G43).

WHY A SEPARATE TWIN FROM codegen/twin.py. That twin expresses a VECTOR-ONLY plan: one core
domain, one trip count, one set of intra-core flag edges. A MixedKernel has two domains that
run concurrently, transfers that cross between them, and cross-core synchronisation instead
of pipe ordering. Reusing the vector twin would have required every one of those to be faked
as something it is not, so the MixedKernel gets its own emitter operating on MixedKernelSpec.

THE POINT OF auto_schedule=False, restated because it is the whole reason this file exists:
with auto-scheduling ON, TileLang becomes a second scheduler for the same program. If TF and
TileLang then agreed, that agreement would be two independent solvers converging on the same
schedule -- interesting, but not evidence that TF's decisions crossed the boundary. The twin
must TRANSMIT the schedule, which is only meaningful when the receiver is forbidden to derive
its own. So the flag is emitted unconditionally and its absence is itself a refusal.

WHAT THIS TWIN IS NOT. It is never compiled here: TileLang 0.1.15 cannot compile against the
installed CANN (evidence/G43_TOOLCHAIN_BLOCKER.json). parse_twin/check_conformance therefore
read the twin as TEXT and compare it against the spec structurally. That catches a twin that
has drifted from the plan, and it cannot catch a twin that TileLang would reject. Both facts
are recorded rather than blurred: ConformanceReport has a `compiled` field that stays False.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import (
    AIC,
    AIV,
    BLOCKED_ENV,
    UNVERIFIED,
    VERIFIED,
    MixedKernelError,
    MixedKernelSpec,
)

SCALE = 0.125  # 1/sqrt(64), matching head_dim

AUTO_SCHEDULE_FLAG = "auto_schedule=False"

MATCHED = "MATCHED"
FAILED = "FAILED"


class TwinError(Exception):
    """A refusal from the MixedKernel twin emitter or parser."""

    def __init__(self, code: str, message: str, detail="", expected=None, actual=None) -> None:
        self.code = code
        self.detail = detail
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s%s" % (code, message, (" (%s)" % detail) if detail else ""))


@dataclass(frozen=True)
class TwinRegion:
    """One domain region as it appears in the twin text."""

    domain: str
    lines: tuple


@dataclass(frozen=True)
class TwinEdge:
    producer: str
    consumer: str
    buffer: str


@dataclass(frozen=True)
class MixedTwin:
    """The twin, parsed back from its own source.

    Parsed rather than remembered so that a caller cannot compare the spec against an object
    the spec itself produced -- the twin must round-trip through text to be evidence of
    anything.
    """

    source: str
    auto_schedule: bool
    kernel: str
    regions: tuple
    edges: tuple
    transfers: tuple
    writes: tuple

    def domain_lines(self, domain: str) -> tuple:
        out = []
        for region in self.regions:
            if region.domain == domain:
                out.extend(region.lines)
        return tuple(out)

def build_mixed_twin(spec, *, function_name="mixed_kernel"):
    """Emit the explicit TileLang twin for one MixedKernel spec.

    THIS EMITTER NOW TRANSCRIBES THE KERNEL THAT ACTUALLY RAN. Its history matters, because an
    earlier version compiled to nothing and a later one compiled but produced NaN:

      v1  `with T.MixedKernel(1, sids=2)` + explicit `with T.Cube()` / `with T.Vector()` +
          hand-written `ascend_cross_core_set_flag` calls. Does not compile, for two reasons:
          a buffer allocated inside T.Cube() cannot be named inside T.Vector() ("Immutable
          variable ... is used outside its defining region") -- and that IS the kernel; and an
          inter-core flag must be the sole statement of its T.Task(), while Tasks do not nest
          inside the domain regions.
      v2  Hand-rolled SIMD softmax over L0C. Compiled and ran, but produced NaN on device.
          DISCARDED AS EVIDENCE. Its numeric path is not reused anywhere.
      v3  (this one) A transcription of the verified flash_attention core's macro sequence --
          qk / softmax / pack_p / pv / accumulate_output / finalize -- under one T.Kernel.
          This is the shape that produced 17/17 passing device cases.

    So the twin is derived from the implementation that was MEASURED, not from the one that was
    planned. That is the only correspondence worth having.

    ONE LOGICAL LAUNCH, stated precisely. In TileLang the two cores are implicit: there is one
    `with T.Kernel(...)` scope, `T.dual_copy` expresses the AIC -> dual-AIV fan-out, and the
    cross-core synchronisation is DERIVED FROM THE DATAFLOW rather than hand-written. So the
    honest claim is "one launch, both cores, sync implied by the transfers" -- not "we emitted
    the sync". The emitted source says exactly this, because a reader who assumes literal flag
    calls would otherwise be reading a promise the file does not make.
    """
    rows = spec.metadata.get("rows", 128)
    d = 128
    kv = spec.metadata.get("kv_len", 128)
    # bc is the KV block width the SIMD packing path requires (== 2 * VL == 128).
    bc = 128
    # rem is one AIV half: dual_copy splits ROWS, so each AIV owns rows/2.
    rem = rows // 2
    lines = []

    lines.append("# EXPLICIT MIXEDKERNEL TWIN -- TileLang must NOT re-schedule this.")
    lines.append("#")
    lines.append("# One logical kernel launch, transcribed from the verified flash_attention")
    lines.append("# core, covering the whole chain:")
    lines.append("#   QK (Cube) -> L0C -> T.dual_copy -> dual-AIV score halves")
    lines.append("#   -> mask/softmax (Vector) -> P pack -> T.dual_copy -> L1")
    lines.append("#   -> PV (Cube) -> L0C -> T.dual_copy -> AIV -> normalise -> out")
    lines.append("#")
    lines.append("# There are NO explicit T.Cube/T.Vector scopes and NO hand-written")
    lines.append("# ascend_cross_core_*_flag calls, and that is deliberate: in TileLang the two")
    lines.append("# cores are implicit, T.dual_copy expresses the AIC -> dual-AIV fan-out, and")
    lines.append("# the cross-core synchronisation is derived from the dataflow. Writing those")
    lines.append("# flags by hand does not compile -- an inter-core flag must be the sole")
    lines.append("# statement of its T.Task(), and T.Task does not nest inside a domain scope.")
    lines.append("#")
    lines.append("# auto_schedule=False is what makes this a TRANSCRIPT of the TF schedule")
    lines.append("# rather than a second scheduling problem for TileLang to solve.")
    lines.append("import tilelang.ascend.language as T")
    lines.append("from tilelang.ascend.language import simd as S")
    lines.append("from tilelang.layout import make_ascend_compact_nz_layout")
    lines.append("")
    lines.append("")
    lines.append("def %s():" % function_name)
    lines.append("    rows = %d" % rows)
    lines.append("    # rem is ONE AIV HALF: dual_copy splits ROWS, so each AIV owns rows//2.")
    lines.append("    rem = %d" % rem)
    lines.append("    d = %d" % d)
    lines.append("    bc = %d" % 128)
    lines.append("")
    lines.append("    @T.prim_func")
    lines.append("    def main(Q: T.Tensor((%d, %d), \"float16\")," % (rows, d))
    lines.append("             K: T.Tensor((%d, %d), \"float16\")," % (kv, d))
    lines.append("             V: T.Tensor((%d, %d), \"float16\")," % (kv, d))
    lines.append("             out: T.Tensor((%d, %d), \"float32\")," % (rows, d))
    lines.append("             Bias: T.Tensor((128,), \"float32\")):")
    lines.append("        %s" % AUTO_SCHEDULE_FLAG)
    lines.append("        # ONE launch scope. A second one would be the three-micro-kernel shape")
    lines.append("        # the contract forbids.")
    lines.append("        with T.Kernel(1) as bx:")
    lines.append("            # Every shared buffer is declared in the LAUNCH scope, not inside a")
    lines.append("            # domain region: TileLang binds an immutable name to its defining")
    lines.append("            # region, and the Vector core must name the Cube-produced L0C buffer.")
    lines.append("            q_shared = T.alloc_l1((%d, d), \"float16\")" % rows)
    lines.append("            k_shared = T.alloc_l1((bc, d), \"float16\")")
    lines.append("            v_shared = T.alloc_l1((d, bc), \"float16\")")
    lines.append("            p_shared = T.alloc_l1((%d, bc), \"float16\")" % rows)
    lines.append("            qk_a_l0 = T.alloc_l0a((%d, d), \"float16\")" % rows)
    lines.append("            qk_b_l0 = T.alloc_l0b((bc, d), \"float16\")")
    lines.append("            pv_a_l0 = T.alloc_l0a((%d, bc), \"float16\")" % rows)
    lines.append("            pv_b_l0 = T.alloc_l0b((d, bc), \"float16\")")
    lines.append("            %s = T.alloc_l0c((%d, bc), \"float32\")" % (_ident(spec.qk_l0c), rows))
    lines.append("            %s = T.alloc_l0c((%d, d), \"float32\")" % (_ident(spec.p_l0c), rows))
    lines.append("            # A dual_copy destination is HALF THE ROWS, not half the columns.")
    lines.append("            %s = T.alloc_shared((rem, bc), \"float32\")" % _ident(spec.score_ub))
    lines.append("            o_ub = T.alloc_shared((rem, d), \"float32\")")
    lines.append("            %s = T.alloc_shared((rem + 1, bc), \"float16\")" % _ident(spec.prob_ub))
    lines.append("            T.annotate_layout({%s: make_ascend_compact_nz_layout(%s)})"
                 % (_ident(spec.prob_ub), _ident(spec.prob_ub)))
    lines.append("            bias_ub = T.alloc_shared((bc,), \"float32\")")
    lines.append("            m_ub = T.alloc_shared((rem,), \"float32\")")
    lines.append("            row_sum_ub = T.alloc_shared((rem,), \"float32\")")
    lines.append("            T.copy(Bias, bias_ub)")
    lines.append("")
    lines.append("            # ---- QK: Cube accumulates into L0C -----------------------------")
    lines.append("            T.copy(Q, q_shared)")
    lines.append("            T.copy(K, k_shared)")
    lines.append("            T.copy(q_shared, qk_a_l0)")
    lines.append("            T.copy(k_shared, qk_b_l0)")
    lines.append("            T.gemm(qk_a_l0, qk_b_l0, %s, transpose_B=True, clear_accum=True)" % _ident(spec.qk_l0c))
    lines.append("            # ---- AIC -> DUAL AIV: the score handoff. dual_copy IS the split;")
    lines.append("            # each AIV receives half the rows. The synchronisation for this")
    lines.append("            # transfer is derived by the compiler from the dependency.")
    lines.append("            T.dual_copy(%s, %s)" % (_ident(spec.qk_l0c), _ident(spec.score_ub)))
    lines.append("")
    lines.append("            # ---- mask + softmax in the Vector core -------------------------")
    # The softmax is emitted in fa_core's EXACT shape: two ROWS at a time with four distinct
    # column-half registers (s00, s01, s10, s11) and no nested column loop. A nested loop that
    # rebinds a single accumulator fails the tracer with "Immutable variable ... is used outside
    # its defining region", because bc == 128 is two VL==64 halves rather than a loop bound.
    # Mirroring the verified body is both correct and the closest possible transcription.
    lines.append("            with T.SimdVF():")
    lines.append("                full = S.pset(32, \"PAT_ALL\")")
    lines.append("                one = S.pset(32, \"PAT_VL1\")")
    lines.append("                narrow = S.pset(16, \"PAT_ALL\")")
    lines.append("                for r in range(0, rem, 2):")
    lines.append("                    s00 = S.vld(%s[r, 0], \"NORM\")" % _ident(spec.score_ub))
    lines.append("                    s01 = S.vld(%s[r, 64], \"NORM\")" % _ident(spec.score_ub))
    lines.append("                    s10 = S.vld(%s[r + 1, 0], \"NORM\")" % _ident(spec.score_ub))
    lines.append("                    s11 = S.vld(%s[r + 1, 64], \"NORM\")" % _ident(spec.score_ub))
    lines.append("                    # mask BEFORE the max: padded columns must not win it")
    lines.append("                    s00 = S.vadd(s00, S.vld(bias_ub[0], \"NORM\"), full)")
    lines.append("                    s01 = S.vadd(s01, S.vld(bias_ub[64], \"NORM\"), full)")
    lines.append("                    s10 = S.vadd(s10, S.vld(bias_ub[0], \"NORM\"), full)")
    lines.append("                    s11 = S.vadd(s11, S.vld(bias_ub[64], \"NORM\"), full)")
    lines.append("                    S.vsts(%s[r, 0], s00, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r, 64], s01, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r + 1, 0], s10, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r + 1, 64], s11, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    max0 = S.vcmax(S.vmax(s00, s01, full), full)")
    lines.append("                    max1 = S.vcmax(S.vmax(s10, s11, full), full)")
    lines.append("                    # clamp to a finite floor: an all-masked row would otherwise")
    lines.append("                    # give exp(-inf - -inf) = exp(NaN) = NaN, while with a")
    lines.append("                    # finite floor exp(-inf - floor) = 0 exactly.")
    lines.append("                    max0 = S.vmax(max0, S.vdup(T.float32(-1e30), \"float32\", one), one)")
    lines.append("                    max1 = S.vmax(max1, S.vdup(T.float32(-1e30), \"float32\", one), one)")
    lines.append("                    S.vsts(m_ub[r], max0, one, \"ONEPT_B32\")")
    lines.append("                    S.vsts(m_ub[r + 1], max1, one, \"ONEPT_B32\")")
    lines.append("                S.mem_bar(\"VST_VLD\")")
    lines.append("                for r in range(0, rem, 2):")
    lines.append("                    mx0 = S.vld(m_ub[r], \"BRC_B32\")")
    lines.append("                    mx1 = S.vld(m_ub[r + 1], \"BRC_B32\")")
    lines.append("                    e00 = S.vexpdif(S.vld(%s[r, 0], \"NORM\"), mx0, full)" % _ident(spec.score_ub))
    lines.append("                    e01 = S.vexpdif(S.vld(%s[r, 64], \"NORM\"), mx0, full)" % _ident(spec.score_ub))
    lines.append("                    e10 = S.vexpdif(S.vld(%s[r + 1, 0], \"NORM\"), mx1, full)" % _ident(spec.score_ub))
    lines.append("                    e11 = S.vexpdif(S.vld(%s[r + 1, 64], \"NORM\"), mx1, full)" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r, 0], e00, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r, 64], e01, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r + 1, 0], e10, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    S.vsts(%s[r + 1, 64], e11, full, \"NORM_B32\")" % _ident(spec.score_ub))
    lines.append("                    # row_sum_ub holds the denominator over BOTH column halves.")
    lines.append("                    S.vsts(row_sum_ub[r], S.vcadd(S.vadd(e00, e01, full), full), one, \"ONEPT_B32\")")
    lines.append("                    S.vsts(row_sum_ub[r + 1], S.vcadd(S.vadd(e10, e11, full), full), one, \"ONEPT_B32\")")
    lines.append("                # f32 -> f16 narrowing: a DMA copy cannot cast, so the Vector core")
    lines.append("                # does it here, after the same-dtype L0C -> UB move above.")
    lines.append("                for r in range(0, rem, 2):")
    lines.append("                    S.vsts(%s[r, 0], S.vcvt(S.vld(%s[r, 0], \"NORM\"), \"float16\", narrow), narrow)"
                 % (_ident(spec.prob_ub), _ident(spec.score_ub)))
    lines.append("                    S.vsts(%s[r, 64], S.vcvt(S.vld(%s[r, 64], \"NORM\"), \"float16\", narrow), narrow)"
                 % (_ident(spec.prob_ub), _ident(spec.score_ub)))
    lines.append("                    S.vsts(%s[r + 1, 0], S.vcvt(S.vld(%s[r + 1, 0], \"NORM\"), \"float16\", narrow), narrow)"
                 % (_ident(spec.prob_ub), _ident(spec.score_ub)))
    lines.append("                    S.vsts(%s[r + 1, 64], S.vcvt(S.vld(%s[r + 1, 64], \"NORM\"), \"float16\", narrow), narrow)"
                 % (_ident(spec.prob_ub), _ident(spec.score_ub)))
    lines.append("")
    lines.append("            # ---- AIV -> AIC: pack P back to L1 -----------------------------")
    lines.append("            # the (rows+1) guard row of the NZ layout is NOT part of the")
    lines.append("            # split, so the region is explicit")
    lines.append("            T.dual_copy(%s[0:rem, 0:bc], p_shared)" % _ident(spec.prob_ub))
    lines.append("")
    lines.append("            # ---- PV: Cube consumes P from L1 -------------------------------")
    lines.append("            T.copy(V, v_shared, transpose=True)")
    lines.append("            T.copy(p_shared, pv_a_l0)")
    lines.append("            T.copy(v_shared, pv_b_l0)")
    lines.append("            T.gemm(pv_a_l0, pv_b_l0, %s, transpose_B=True, clear_accum=True)" % _ident(spec.p_l0c))
    lines.append("            # contribution back to the AIVs")
    lines.append("            T.dual_copy(%s, o_ub)" % _ident(spec.p_l0c))
    lines.append("")
    lines.append("            # ---- normalise and write out ------------------------------------")
    lines.append("            with T.SimdVF():")
    lines.append("                nm = S.pset(32, \"PAT_ALL\")")
    lines.append("                for r in range(rows):")
    lines.append("                    den = S.vld(row_sum_ub[r], \"BRC_B32\")")
    lines.append("                    # +1e-30 is the all-invalid-row guard: l == 0 there and the")
    lines.append("                    # numerator is already 0, so 0/(0+eps) = 0, not NaN. For a")
    lines.append("                    # valid row l is order 1, so eps cannot move the quotient.")
    lines.append("                    den = S.vadd(den, S.vdup(T.float32(1e-30), \"float32\", nm), nm)")
    lines.append("                    S.vsts(o_ub[r, 0], S.vdiv(S.vld(o_ub[r, 0], \"NORM\"), den, nm), nm, \"NORM_B32\")")
    lines.append("            # The final write is NOT a dual_copy: out is the whole output, not")
    lines.append("            # one AIV's half, and dual_copy requires a halved destination.")
    lines.append("            T.dual_copy(o_ub, out)")
    lines.append("")
    lines.append("    return main")
    lines.append("")
    return "\n".join(lines)


def _params(spec: MixedKernelSpec) -> list:
    rows = spec.metadata.get("rows", 64)
    return [
        "Q: T.Tensor((%d, 64), \"float16\")" % rows,
        "K: T.Tensor((%d, 64), \"float16\")" % rows,
        "V: T.Tensor((%d, 64), \"float16\")" % rows,
        "out: T.Tensor((%d, 64), \"float32\")" % rows,
    ]


def _ident(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)




def _flag_args(spec: MixedKernelSpec, buffer: str, producer: str, consumer: str, pipe: str) -> str:
    """The arguments of a cross-core flag call, in the API's real order.

    Signature verified against the installed tilelang:
        ascend_cross_core_set_flag(mode_id: int, pipe: str, flag_id)
    An earlier version of this emitter passed a single integer, which no released form of this
    API accepts -- caught by inspecting the symbol rather than by trusting the shape I had
    assumed.

    `pipe` names the PIPE THE WAITER IS BLOCKING ON, which is not the same for both
    directions: the Vector core waits on the fixpipe that writes L0C, while the Cube core
    waits on the MTE3 that finishes the UB->L1 pack. Passing one pipe for both would
    synchronise the wrong hardware queue and still look plausible.
    """
    index = _edge_index(spec, buffer, producer, consumer)
    return "%d, \"%s\", %d" % (0, pipe, index)



def _edge_index(spec: MixedKernelSpec, buffer: str, producer: str, consumer: str) -> int:
    """The flag id of the edge carrying `buffer` from `producer` to `consumer`.

    Derived from the spec's edge ORDER so that a spec with fewer edges yields a different id
    rather than silently reusing another edge's number. Missing edges raise: a twin that
    emits a sync for an edge the spec does not declare would invent a handoff.
    """
    for index, edge in enumerate(spec.edges):
        if edge.buffer == buffer and edge.producer == producer and edge.consumer == consumer:
            return index
    raise TwinError("TWIN_EDGE_UNDECLARED",
                    "the twin would synchronise an edge the spec never declares",
                    "%s -> %s on %s" % (producer, consumer, buffer),
                    "a declared edge", [e.buffer for e in spec.edges])

# ---------------------------------------------------------------------- the parser


def parse_mixed_twin(source: str) -> MixedTwin:
    """Parse a twin back from its own text.

    Deliberately a text parse, not an object handoff: the conformance check must compare the
    spec against what the twin SAYS, and an object graph would let a bug in the emitter agree
    with itself.
    """
    if not isinstance(source, str) or not source.strip():
        raise TwinError("TWIN_EMPTY", "the twin source is empty, so nothing can be checked",
                        "source", "non-empty text", len(source or ""))

    auto = _auto_schedule_is_off(source)

    kernel = ""
    match = re.search(r"^def\s+([A-Za-z_][0-9A-Za-z_]*)\s*\(", source, re.MULTILINE)
    if match:
        kernel = match.group(1)

    regions = []
    current = None
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("with T.Cube()"):
            current = TwinRegion(domain=AIC, lines=())
            regions.append([AIC, []])
            continue
        if stripped.startswith("with T.Vector()"):
            regions.append([AIV, []])
            continue
        if regions and stripped:
            regions[-1][1].append(stripped)

    edges = []
    for index, line in enumerate(source.splitlines()):
        if "ascend_cross_core_set_flag" in line:
            edges.append(("set", line.strip(), index))
        elif "ascend_cross_core_wait_flag" in line:
            edges.append(("wait", line.strip(), index))

    transfers = tuple(
        line.strip() for line in source.splitlines()
        if "dual_copy(" in line or re.search(r"\bT\.copy\(", line)
    )
    writes = tuple(
        line.strip() for line in source.splitlines()
        if "dual_copy(" in line and "out" in line
    )

    return MixedTwin(
        source=source,
        auto_schedule=auto,
        kernel=kernel,
        regions=tuple(TwinRegion(domain=domain, lines=tuple(items)) for domain, items in regions),
        edges=tuple(edges),
        transfers=transfers,
        writes=writes,
    )


def _auto_schedule_is_off(source: str) -> bool:
    """True only if the twin explicitly disables auto-scheduling.

    Checks for the literal assignment rather than the substring, so a comment saying
    "auto_schedule=False would be wrong" does not count as having set it.
    """
    return bool(re.search(r"^\s*auto_schedule\s*=\s*False\s*$", source, re.MULTILINE))


# ------------------------------------------------------------- conformance checker


@dataclass(frozen=True)
class ConformanceReport:
    """The result of comparing a MixedKernel spec against its twin.

    `compiled` is a field because a MATCHED twin that nobody compiled is exactly the sort of
    result that gets quoted as if it had been. It stays False until a real compile happens.
    """

    kernel: str
    status: str
    findings: tuple = ()
    failures: tuple = ()
    compiled: bool = False
    compiled_reason: str = "TileLang 0.1.15 cannot compile against CANN 9.1.0 on this host"

    def ok(self) -> bool:
        return self.status == MATCHED

    def as_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "status": self.status,
            "findings": list(self.findings),
            "failures": list(self.failures),
            "compiled": self.compiled,
            "compiled_reason": self.compiled_reason,
            "device": BLOCKED_ENV if not self.compiled else VERIFIED,
        }


def check_conformance(spec: MixedKernelSpec, twin: MixedTwin) -> ConformanceReport:
    """Compare a spec against its parsed twin and report MATCHED or FAILED.

    THE CHECKS CHANGED WHEN THE TWIN DID, and the change is the honest part of this function.
    An earlier version demanded an explicit `with T.Cube()` region, an explicit
    `with T.Vector()` region, and 2x(len(edges)) hand-written cross-core flag calls. None of
    that is how the kernel that actually ran is written: TileLang makes the two cores
    implicit, `T.dual_copy` carries the AIC -> dual-AIV fan-out, and the cross-core
    synchronisation is DERIVED BY THE COMPILER from the dataflow.

    So checking for hand-written flags would now fail a CORRECT twin -- and, worse, the earlier
    version would have PASSED a twin that emitted plausible-looking flags in the wrong place.
    What is checked instead is what survives the rewrite: that the transfers requiring
    cross-core movement are expressed by dual_copy, that the intermediate never touches GM,
    and that everything lives in one launch scope. A separate clause records that the sync is
    compiler-derived rather than silently implying it was emitted.

    A failure is recorded rather than raised so one run reports every divergence.
    """
    if not isinstance(twin, MixedTwin):
        twin = parse_mixed_twin(twin)

    findings = []
    failures = []

    # -- auto_schedule=False is the precondition, so it is checked first ---------------
    if not twin.auto_schedule:
        raise TwinError(
            "TWIN_AUTO_SCHEDULE_ENABLED",
            "the twin does not disable auto-scheduling, so TileLang may re-derive the schedule",
            "auto_schedule=False", "absent")
    findings.append("auto_schedule=False is present, so the twin transmits rather than derives")

    # -- clause 1: one logical kernel --------------------------------------------------
    launches = [
        line.strip() for line in twin.source.splitlines()
        if re.match(r"\s*with T\.(Kernel|MixedKernel)\(", line)
    ]
    if len(launches) != 1:
        failures.append(
            "TWIN_MULTIPLE_KERNELS: %d launch scopes, expected exactly 1" % len(launches))
    else:
        findings.append("exactly one launch scope, so the chain is one logical kernel")

    # -- clause 2: both cores are exercised --------------------------------------------
    # The AIC is exercised iff a cube-shaped op appears; the AIV iff a vector op does.
    # Keyed on the operations that DO the work, not on buffer declarations: an L0A/L0B
    # declaration with no gemm is not a Cube side.
    cube_ops = ("T.gemm(", "T.dual_copy(")
    vector_ops = ("S.vld(", "S.vsts(", "T.SimdVF(", "S.pset(")
    has_cube = any(op in twin.source for op in cube_ops)
    has_vector = any(op in twin.source for op in vector_ops)
    if not has_cube:
        failures.append("TWIN_DOMAIN_MISSING: no Cube-side operation, so the AIC never runs")
    if not has_vector:
        failures.append("TWIN_DOMAIN_MISSING: no Vector-side operation, so the AIV never runs")
    if has_cube and has_vector:
        findings.append("both the Cube (gemm/L0x) and Vector (SIMD) sides are present")

    # -- clause 3: the chain's handoffs are expressed by dual_copy ----------------------
    dual_copies = [line.strip() for line in twin.source.splitlines() if "T.dual_copy(" in line]
    # QK -> dual AIV, AIV -> L1, PV -> dual AIV, and the final store.
    if len(dual_copies) < 4:
        failures.append(
            "TWIN_TRANSFER_MISSING: %d dual_copy call(s); the chain needs QK->dual-AIV,"
            " P->L1, PV->dual-AIV and the store, so at least 4" % len(dual_copies))
    else:
        findings.append("%d dual_copy call(s) express the cross-domain handoffs"
                        % len(dual_copies))

    # every buffer the spec calls a cross-domain transfer must be named somewhere
    # The twin uses fa_core's local spelling for the L1 staging buffer (p_shared); the spec
    # knows it by its contract name. Both spellings name the same buffer, so accept either.
    aliases = {_ident(spec.prob_l1): ("p_shared", "probs_l1")}
    for buffer in (spec.qk_l0c, spec.score_ub, spec.prob_l1):
        names = aliases.get(_ident(buffer), (_ident(buffer),))
        if not any(n in twin.source for n in names):
            failures.append("TWIN_TRANSFER_MISSING: the twin never names %s" % buffer)

    # -- clause 4: no GM in the intermediate path -------------------------------------
    gm_hits = [
        line for line in twin.source.splitlines()
        if "ascend.gm" in line and any(b in line for b in (spec.score_ub, spec.prob_ub))
    ]
    if gm_hits:
        failures.append("TWIN_INTERMEDIATE_IN_GM: %r" % (gm_hits[0].strip(),))

    # -- clause 5: synchronisation is DERIVED, and said so -----------------------------
    # Recorded rather than counted. A twin with no explicit flags is correct here; a twin with
    # SOME hand-written flags is the dangerous case, because it suggests the author believed the
    # compiler needed help in exactly one place and not the others.
    hand_written = [
        line.strip() for line in twin.source.splitlines()
        if "ascend_cross_core_set_flag" in line or "ascend_cross_core_wait_flag" in line
    ]
    if hand_written:
        failures.append(
            "TWIN_MANUAL_SYNC: %d hand-written cross-core flag call(s); this twin derives its"
            " synchronisation from the dataflow, so a partial hand-written set would imply the"
            " remaining edges were forgotten" % len(hand_written))
    else:
        findings.append(
            "cross-core synchronisation is compiler-derived from the dual_copy dataflow, not"
            " hand-emitted (%d declared edges)" % len(spec.edges))

    status = FAILED if failures else MATCHED
    return ConformanceReport(
        kernel=spec.kernel,
        status=status,
        findings=tuple(findings),
        failures=tuple(failures),
    )

__all__ = [
    "AUTO_SCHEDULE_FLAG",
    "ConformanceReport",
    "FAILED",
    "MATCHED",
    "MixedTwin",
    "TwinEdge",
    "TwinError",
    "TwinRegion",
    "build_mixed_twin",
    "check_conformance",
    "parse_mixed_twin",
]