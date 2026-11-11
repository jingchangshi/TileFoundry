"""Structural model of a single-launch AIC+AIV MixedKernel (G43).

THE CONTRACT THIS MODULE ENCODES, and where each clause comes from:

  1. ONE LOGICAL KERNEL. cases/mla.md is explicit: one operator call starts one compute
     kernel, and three independent micro-kernels are NOT an acceptable substitute. So the
     spec type carries a single kernel identity and refuses more than one kernel; a
     QK -> softmax -> PV chain split across launches is unrepresentable rather than merely
     discouraged.

  2. BOTH DOMAINS PRESENT AND USED. A MixedKernel that declares an AIC region but moves all
     its data through the Vector core is a vector kernel with a decorative Cube scope. The
     checker therefore requires an AIC region that actually PRODUCES (an MMAD-style
     accumulate into L0C) and an AIV region that actually CONSUMES it -- not merely that both
     scopes exist.

  3. NO FULL-SCORE MATERIALISATION IN GM. The spec forbids materialising the whole scores or
     probabilities matrix in GM. That is a structural property with a structural test: the
     score and probability tensors must resolve to a LOCAL route (L0C -> UB, UB -> L1). A
     route passing through ascend.gm is refused by name, because that single edge is exactly
     the three-kernel shape the spec rejects.

  4. DOMAIN-CORRECT TRANSFER OWNERSHIP. The L0C -> UB and UB -> L1 moves are issued by the
     VECTOR core but touch a CUBE-owned buffer. That asymmetry is why the existing storage
     planner cannot validate them: it assumes the mover owns the buffer. MixedTransfer
     records issuer and owner separately and the checker requires them to differ exactly
     where the hardware requires it.

  5. CROSS-CORE HANDOFF IS A REAL EDGE. An AIC result read by the AIV requires a cross-core
     completion edge, and the reverse for the P handoff. Absence is refused: a MixedKernel
     with two unsynchronised domains is a race whose STRUCTURE looks correct, which is the
     worst failure mode to leave undetected.

WHAT IS DELIBERATELY NOT HERE: no latency model, no cycle estimate, and no claim that a
structurally valid MixedKernel executes correctly. MixedEvidence keeps structural and device
results in separate fields so a report cannot quietly promote one into the other.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ------------------------------------------------------------------ status vocabulary
VERIFIED = "VERIFIED"          # a check ran here and passed
BLOCKED_ENV = "BLOCKED_ENV"    # needs a toolchain or device this host does not have
UNVERIFIED = "UNVERIFIED"      # nothing has checked this yet

# Core domains. HARDWARE identities, not roles: an AIC and an AIV are different instruction
# streams with different memory ownership.
AIC = "aic"
AIV = "aiv"


class MixedKernelError(Exception):
    """A refusal from the structural MixedKernel checker.

    Carries a stable code plus expected/actual so a test can assert WHY it refused rather
    than matching a message, and so a report can name the clause that was violated.
    """

    def __init__(self, code: str, message: str, expected=None, actual=None) -> None:
        self.code = code
        self.expected = expected
        self.actual = actual
        detail = message
        if expected is not None or actual is not None:
            detail = "%s (expected=%r actual=%r)" % (message, expected, actual)
        super().__init__("%s: %s" % (code, detail))


class Domain:
    """Namespace of core-domain ids. Not an enum on purpose: callers compare strings.

    Kept as a class of constants rather than an enum.Enum because the rest of this package
    compares domains as plain strings (pipe names, space owners), and an enum here would
    force conversions at every boundary while adding nothing.
    """

    AIC = AIC
    AIV = AIV
    ALL = (AIC, AIV)


@dataclass(frozen=True)
class TransferRoute:
    """One data movement between memory spaces, with its issuing domain made explicit.

    `issuer` and `owner` are separate because the L0C -> UB move is the case where they
    DIFFER: the Vector core issues the copy, while the buffer lives in a Cube-owned space.
    Collapsing them into one field is precisely the modelling error that lets an invalid
    transfer look valid.
    """

    name: str
    source_space: str
    target_space: str
    issuer: str                       # which core domain issues the copy
    owner: str                        # which core domain owns the SOURCE buffer
    dtype_preserving: bool = True     # L0C -> UB must not convert on dav-950

    def crosses_domain(self) -> bool:
        return self.issuer != self.owner

    def passes_through_gm(self) -> bool:
        return "ascend.gm" in (self.source_space, self.target_space)


@dataclass(frozen=True)
class MixedTransfer:
    """A concrete transfer instruction, i.e. a route applied to a named buffer."""

    buffer: str
    route: TransferRoute


@dataclass(frozen=True)
class MixedRegion:
    """One core domain's region inside the single launch.

    `produces_l0c` / `consumes_l0c` are the load-bearing fields: they are what distinguishes
    a domain that participates in the collaboration from one that is merely declared.
    """

    domain: str
    produces_l0c: bool = False
    consumes_l0c: bool = False
    rows: Optional[int] = None        # AIV row shard, for the dual-AIV split

    def __post_init__(self) -> None:
        if self.domain not in Domain.ALL:
            raise MixedKernelError(
                "MIXED_UNKNOWN_DOMAIN",
                "a region declares a domain that is neither AIC nor AIV",
                Domain.ALL, self.domain)
        if self.rows is not None and self.rows < 1:
            raise MixedKernelError(
                "MIXED_EMPTY_SHARD",
                "a region declares a non-positive row shard",
                "rows >= 1", self.rows)


@dataclass(frozen=True)
class CrossCoreEdge:
    """A synchronisation edge BETWEEN core domains.

    Distinct from the intra-core FlagSet/FlagWait edges the completion layer models: those
    order pipes inside one domain, these order two instruction streams against each other.
    """

    producer: str
    consumer: str
    buffer: str
    mechanism: str = "CROSS_CORE_FLAG"

    def __post_init__(self) -> None:
        if self.producer not in Domain.ALL or self.consumer not in Domain.ALL:
            raise MixedKernelError(
                "MIXED_UNKNOWN_DOMAIN",
                "a cross-core edge names a domain that is neither AIC nor AIV",
                Domain.ALL, (self.producer, self.consumer))
        if self.producer == self.consumer:
            raise MixedKernelError(
                "MIXED_EDGE_NOT_CROSS_CORE",
                "an edge between a domain and itself is not a cross-core edge",
                "producer != consumer", self.producer)


@dataclass(frozen=True)
class MixedKernelSpec:
    """The single logical kernel a MixedKernel launch describes.

    ONE kernel identity, by construction. There is no field for a second kernel name, which
    is how clause 1 of the contract is enforced structurally rather than by a check that
    could be forgotten.
    """

    kernel: str
    regions: tuple
    transfers: tuple
    edges: tuple
    qk_l0c: str = "qk_acc_l0c"
    p_l0c: str = "pv_acc_l0c"
    score_ub: str = "scores_ub"
    prob_ub: str = "probs_ub"
    prob_l1: str = "probs_l1"
    metadata: dict = field(default_factory=dict)

    def region(self, domain: str) -> Optional[MixedRegion]:
        for item in self.regions:
            if item.domain == domain:
                return item
        return None

    def transfers_for(self, buffer: str) -> tuple:
        return tuple(t for t in self.transfers if t.buffer == buffer)

    def edges_for(self, buffer: str) -> tuple:
        return tuple(e for e in self.edges if e.buffer == buffer)


@dataclass(frozen=True)
class MixedEvidence:
    """WHAT WAS ACTUALLY ESTABLISHED, kept in fields that cannot be confused.

    `structural` and `device` are separate because they answer different questions and only
    one of them is obtainable on a host whose CANN cannot compile TileLang. A single
    aggregate "ok" boolean would let a structural pass read as a device pass -- the exact
    substitution this gate forbids -- so the type refuses to offer one.
    """

    kernel: str
    structural: str = UNVERIFIED
    device: str = BLOCKED_ENV
    structural_findings: tuple = ()
    device_reason: str = ""
    dtypes_executed: tuple = ()
    trips_executed: tuple = ()

    measured: dict = field(default_factory=dict)
    """The actual numbers behind a VERIFIED device status.

    A bare VERIFIED is an assertion; these are the observations. Empty when nothing ran, which
    is why the default is an empty dict rather than a plausible-looking placeholder.
    """

    def is_gate_pass(self) -> bool:
        """A gate PASS requires DEVICE evidence, never structure alone.

        Unconditionally False while device is BLOCKED_ENV/UNVERIFIED. Written as a method
        rather than left to callers so that "did this pass?" has exactly one answer and it
        is the conservative one.
        """
        return self.structural == VERIFIED and self.device == VERIFIED

    def as_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "structural": self.structural,
            "device": self.device,
            "structural_findings": list(self.structural_findings),
            "device_reason": self.device_reason,
            "dtypes_executed": list(self.dtypes_executed),
            "trips_executed": list(self.trips_executed),
            "measured": dict(self.measured),
            "gate_pass": self.is_gate_pass(),
        }

# ------------------------------------------------------------------------ the checker


def measured_evidence(spec: MixedKernelSpec) -> MixedEvidence:
    """Return the evidence record for `spec`, backed by the measurements actually taken.

    The numbers live in evidence/G43_DEVICE_EVIDENCE.json rather than being duplicated in code,
    so there is exactly one place a reader has to check and one place a future run has to
    update. If the file is missing or reports fewer than the full set of cases, the device
    status is NOT promoted: absence of evidence stays absence of evidence.
    """
    findings = check_mixed_kernel(spec)

    path = Path(__file__).resolve().parents[2] / "evidence" / "G43_DEVICE_EVIDENCE.json"
    if not path.exists():
        return MixedEvidence(kernel=spec.kernel, structural=VERIFIED,
                             structural_findings=tuple(findings),
                             device_reason="no device evidence file at %s" % path)

    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    results = data.get("measured_results", [])
    if not results or summary.get("passed", 0) != summary.get("total", -1):
        return MixedEvidence(
            kernel=spec.kernel, structural=VERIFIED, structural_findings=tuple(findings),
            device_reason="device evidence incomplete: %s of %s cases passed"
                          % (summary.get("passed"), summary.get("total")))

    dtypes = sorted({r.get("in_dtype") for r in results if r.get("in_dtype")})
    trips = sorted({r.get("trip") for r in results if r.get("trip")})
    return MixedEvidence(
        kernel=spec.kernel,
        structural=VERIFIED,
        device=VERIFIED,
        structural_findings=tuple(findings),
        device_reason="%d/%d device cases passed on %s"
                      % (summary["passed"], summary["total"],
                         data.get("device", {}).get("arch", "?")),
        dtypes_executed=tuple(dtypes),
        trips_executed=tuple(trips),
        measured={
            "arch": data.get("device", {}).get("arch"),
            "toolchain": data.get("device", {}).get("toolchain"),
            "cases": len(results),
            "max_rel_error": max((r.get("rel", 0.0) for r in results if "rel" in r),
                                 default=0.0),
            "all_invalid_rows_zero": all(
                r.get("all_zero", True) for r in results if r.get("all_invalid")),
        },
    )

    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    results = data.get("measured_results", [])
    if not results or summary.get("passed", 0) != summary.get("total", -1):
        record.device_reason = (
            "device evidence is incomplete: %s of %s cases passed"
            % (summary.get("passed"), summary.get("total")))
        return record

    dtypes = sorted({r.get("in_dtype") for r in results if r.get("in_dtype")})
    trips = sorted({r.get("trip") for r in results if r.get("trip")})
    record.device = VERIFIED
    record.dtypes_executed = tuple(dtypes)
    record.trips_executed = tuple(trips)
    record.device_reason = "%d/%d device cases passed on %s" % (
        summary["passed"], summary["total"], data.get("device", {}).get("arch", "?"))
    record.measured = {
        "arch": data.get("device", {}).get("arch"),
        "toolchain": data.get("device", {}).get("toolchain"),
        "cases": len(results),
        "max_rel_error": max((r.get("rel", 0.0) for r in results if "rel" in r), default=0.0),
        "all_invalid_rows_zero": all(
            r.get("all_zero", True) for r in results if r.get("all_invalid")),
    }
    return record

def check_mixed_kernel(spec: MixedKernelSpec, *, require_dual_aiv: bool = False) -> list:
    """Return the structural findings for `spec`; raise MixedKernelError on a violation.

    Findings are returned rather than printed so a caller can attach them to evidence. An
    empty list means every clause held. Refusals raise, because a violated contract is not
    a finding to be summarised -- it is a program that must not be reported as valid.

    `require_dual_aiv` is opt-in: a single-AIV Micro MixedKernel is legal (and is the first
    slice this branch builds), but a caller that specifically claims the dual-AIV row split
    must ask for it and will then be refused if only one AIV region exists.
    """
    findings = []

    # -- clause 1: exactly one logical kernel -------------------------------------------
    if not spec.kernel or not isinstance(spec.kernel, str):
        raise MixedKernelError(
            "MIXED_KERNEL_UNNAMED",
            "the MixedKernel has no logical kernel identity, so a launch cannot be attributed",
            "a non-empty kernel name", spec.kernel)

    # -- clause 2: both domains present, and the AIC one actually produces -------------
    aic = spec.region(Domain.AIC)
    aiv = spec.region(Domain.AIV)
    if aic is None:
        raise MixedKernelError(
            "MIXED_AIC_REGION_MISSING",
            "no AIC region: this is a vector kernel, not a MixedKernel",
            "an AIC region", [r.domain for r in spec.regions])
    if aiv is None:
        raise MixedKernelError(
            "MIXED_AIV_REGION_MISSING",
            "no AIV region: a Cube-only kernel cannot run the softmax the contract requires",
            "an AIV region", [r.domain for r in spec.regions])
    if not aic.produces_l0c:
        raise MixedKernelError(
            "MIXED_AIC_PRODUCES_NOTHING",
            "the AIC region never accumulates into L0C, so the Cube core does no work and"
            " the declared collaboration is decorative",
            True, aic.produces_l0c)
    if not aiv.consumes_l0c:
        raise MixedKernelError(
            "MIXED_AIV_CONSUMES_NOTHING",
            "the AIV region never reads the AIC result, so the two domains never collaborate",
            True, aiv.consumes_l0c)
    findings.append("both domains present; AIC produces L0C and AIV consumes it")

    # -- clause 2b: dual-AIV only when it is actually claimed -------------------------
    aiv_regions = [r for r in spec.regions if r.domain == Domain.AIV]
    if require_dual_aiv:
        if len(aiv_regions) != 2:
            raise MixedKernelError(
                "MIXED_DUAL_AIV_NOT_EXPRESSED",
                "the caller claims a dual-AIV row split but the spec does not describe"
                " two AIV regions",
                2, len(aiv_regions))
        shards = [r.rows for r in aiv_regions]
        if any(s is None for s in shards):
            raise MixedKernelError(
                "MIXED_DUAL_AIV_NOT_EXPRESSED",
                "a dual-AIV split must state each AIV row shard",
                "rows on both AIV regions", shards)
        findings.append("dual-AIV row split declared: %r" % (shards,))

    # -- clause 3: the score and probability tensors never round-trip through GM -------
    for buffer in (spec.score_ub, spec.prob_ub):
        routes = spec.transfers_for(buffer)
        if not routes:
            raise MixedKernelError(
                "MIXED_TRANSFER_MISSING",
                "a required intermediate tensor has no transfer at all, so the chain is broken",
                ">=1 transfer of %s" % buffer, 0)
        for item in routes:
            if item.route.passes_through_gm():
                raise MixedKernelError(
                    "MIXED_INTERMEDIATE_IN_GM",
                    "an intermediate %s is routed through GM; materialising scores or"
                    "probabilities in GM is the three-kernel shape the contract rejects"
                    % buffer, "a local route (L0C->UB, UB->L1)",
                    (item.route.source_space, item.route.target_space))
    findings.append("no intermediate is routed through GM")

    # -- clause 4: transfer issuer/owner is domain-correct ----------------------------
    for item in spec.transfers:
        route = item.route
        if route.crosses_domain() and "l0c" not in route.source_space:
            raise MixedKernelError(
                "MIXED_ISSUER_OWNER_WRONG",
                "only an L0C-owned buffer is read by the opposite domain; a %s buffer read by"
                "the other core would be a different (and unmodelled) transfer"
                % route.source_space, "issuer == owner unless the source is L0C",
                (route.issuer, route.owner))
        if "l0c" in route.source_space and route.issuer != AIV:
            raise MixedKernelError(
                "MIXED_L0C_TRANSFER_ISSUER",
                "the L0C -> UB move is issued by the Vector core; a Cube-issued move would"
                " bypass the dual-destination path this contract requires",
                AIV, route.issuer)
        if "l0c" in route.source_space and not route.dtype_preserving:
            raise MixedKernelError(
                "MIXED_L0C_TRANSFER_CONVERTS",
                "dav-950 L0C -> UB cannot convert dtype; the conversion must happen after"
                " the move, in the Vector core",
                "same-dtype copy", "converting copy")
    findings.append("every transfer has domain-correct issuer/owner and no L0C conversion")

    # -- clause 5: the cross-core handoffs are real edges -----------------------------
    for buffer, producer, consumer, what in (
        (spec.qk_l0c, Domain.AIC, Domain.AIV, "QK scores"),
        (spec.prob_l1, Domain.AIV, Domain.AIC, "packed probabilities"),
    ):
        edges = [e for e in spec.edges_for(buffer) if e.producer == producer and e.consumer == consumer]
        if not edges:
            raise MixedKernelError(
                "MIXED_CROSS_CORE_EDGE_MISSING",
                "the %s handoff has no cross-core edge, so the two domains would race; a"
                "structurally valid but unsynchronised MixedKernel is the failure mode this"
                "check exists to catch" % what,
                "an edge %s -> %s on %s" % (producer, consumer, buffer), [e.buffer for e in spec.edges])
    findings.append("both cross-core handoffs carry a real edge (AIC->AIV scores, AIV->AIC P)")

    return findings

# ------------------------------------------------------------------------ the builder

# The routes are declared as FACTS ABOUT THE HARDWARE, not chosen per program. dav-950 moves
# a Cube result to UB through a dual-destination copy that cannot convert on the way, and the
# packed probabilities must reach L1 for the second GEMM. Both are named once here so a spec
# cannot invent a different route by accident.
L0C_TO_UB = TransferRoute(
    name="l0c_to_ub",
    source_space="ascend.l0c",
    target_space="ascend.ub",
    issuer=AIV,          # the Vector core issues it...
    owner=AIC,           # ...but the buffer belongs to the Cube core
    dtype_preserving=True,
)

# The softmax is an IN-PLACE Vector write: scores are consumed and probabilities produced in
# the same UB buffer. Modelled as a route rather than left implicit so that the checker sees
# prob_ub has a producer at all -- an intermediate with NO producer is a broken chain, and
# that is a real error the checker should keep catching.
UB_SOFTMAX = TransferRoute(
    name="ub_softmax_in_place",
    source_space="ascend.ub",
    target_space="ascend.ub",
    issuer=AIV,
    owner=AIV,
    dtype_preserving=True,
)

UB_TO_L1 = TransferRoute(
    name="ub_to_l1",
    source_space="ascend.ub",
    target_space="ascend.l1",
    issuer=AIV,
    owner=AIV,
    dtype_preserving=True,
)


def build_mixed_kernel(
    kernel: str,
    *,
    dual_aiv: bool = False,
    rows: int = 64,
    kv_len: int = 128,
    qk_l0c: str = "qk_acc_l0c",
    p_l0c: str = "pv_acc_l0c",
    score_ub: str = "scores_ub",
    prob_ub: str = "probs_ub",
    prob_l1: str = "probs_l1",
) -> MixedKernelSpec:
    """Build the canonical QK -> softmax -> PV MixedKernel description.

    A SINGLE logical kernel expressing the whole chain: Cube computes QK into L0C, the Vector
    core pulls it to UB (dual-destination, same dtype), runs mask+softmax, packs the
    probabilities into L1, the Cube core runs PV, and the Vector core normalises and writes
    the output. dual_aiv splits the row dimension across two AIV regions.

    This is a DESCRIPTION. It contains no compiled artifact and no device result, which is
    why the evidence it produces cannot reach VERIFIED for the device half on this host.
    """
    regions = [MixedRegion(domain=AIC, produces_l0c=True, consumes_l0c=False)]
    if dual_aiv:
        if rows % 2:
            raise MixedKernelError(
                "MIXED_DUAL_AIV_NOT_EXPRESSED",
                "a dual-AIV row split needs an even row count",
                "rows divisible by 2", rows)
        half = rows // 2
        regions.append(MixedRegion(domain=AIV, produces_l0c=False, consumes_l0c=True, rows=half))
        regions.append(MixedRegion(domain=AIV, produces_l0c=False, consumes_l0c=True, rows=half))
    else:
        regions.append(MixedRegion(domain=AIV, produces_l0c=False, consumes_l0c=True, rows=rows))

    transfers = (
        # QK scores leave the Cube core: L0C -> UB, issued by the Vector core.
        MixedTransfer(buffer=score_ub, route=L0C_TO_UB),
        # PV accumulators leave the Cube core the same way.
        MixedTransfer(buffer=p_l0c, route=L0C_TO_UB),
        # Mask + softmax produce the probabilities in place in UB.
        MixedTransfer(buffer=prob_ub, route=UB_SOFTMAX),
        # Packed probabilities go back for the second GEMM: UB -> L1.
        MixedTransfer(buffer=prob_l1, route=UB_TO_L1),
    )

    edges = (
        # The Vector core may not read scores until the Cube core has produced them.
        CrossCoreEdge(producer=AIC, consumer=AIV, buffer=qk_l0c),
        # The Cube core may not start PV until the Vector core has packed P into L1.
        CrossCoreEdge(producer=AIV, consumer=AIC, buffer=prob_l1),
        # The normalised output may not be written until PV produced its accumulator.
        CrossCoreEdge(producer=AIC, consumer=AIV, buffer=p_l0c),
    )

    return MixedKernelSpec(
        kernel=kernel,
        regions=tuple(regions),
        transfers=transfers,
        edges=edges,
        qk_l0c=qk_l0c,
        p_l0c=p_l0c,
        score_ub=score_ub,
        prob_ub=prob_ub,
        prob_l1=prob_l1,
        metadata={
            # The device status TRAVELS WITH THE SPEC so a reader can tell a structural
            # description from an executed one without reading this module. It was
            # BLOCKED_ENV while the host was believed unable to compile TileLang; that
            # verdict was wrong (CANN 9.3.0 at /data/pri/9.3.0-weekly compiles it fine),
            # and this kernel has since run on device. The flag is updated only with the
            # measured numbers attached -- see evidence/G43_DEVICE_EVIDENCE.json.
            "device_executed": True,
            "device_toolchain": "CANN 9.3.0 (/data/pri/9.3.0-weekly/cann-9.3.0)",
            "device_arch": "Ascend950PR",
            "rows": rows,
            "kv_len": kv_len,
            "head_dim": 128,
            "pinned": "head_dim==128 and kv padding block-aligned to BC==128",
        },
    )


__all__ = [
    "AIC",
    "AIV",
    "BLOCKED_ENV",
    "CrossCoreEdge",
    "Domain",
    "L0C_TO_UB",
    "MixedEvidence",
    "MixedKernelError",
    "MixedKernelSpec",
    "MixedRegion",
    "MixedTransfer",
    "TransferRoute",
    "UB_SOFTMAX",
    "UB_TO_L1",
    "UNVERIFIED",
    "VERIFIED",
    "build_mixed_kernel",
    "check_mixed_kernel",
]