"""P4-03 / G42: the explicit TileLang twin, and why it must NOT be re-scheduled.

WHAT THIS MODULE IS FOR. A TF program that claims to have decided the schedule has to survive
comparison against an implementation written the way a backend author would write it. TileLang
is that second implementation: if TF says "four loop instances, this slot per instance, these
two cores, and a happens-before edge here", a TileLang twin must express the SAME four
instances, the same slots, the same cores and the same edges -- explicitly.

THE ONE RULE THAT MAKES THE COMPARISON MEANINGFUL: auto_schedule=False, ALWAYS.
TileLang's scheduler will happily re-derive a schedule from a naive loop nest. If it is left on,
the twin becomes a SECOND scheduler and any agreement with TF is a coincidence of two solvers
converging, not evidence that TF's decisions were transmitted. Worse, it fails silently in the
direction that matters: the twin still compiles and still runs, so a disagreement shows up as a
numerical mismatch attributed to the wrong subsystem.

So the twin's construction below spells out, as literal text,
  * each logical iteration as its own emitted instance (no loop trip count to re-derive),
  * the guard that makes an instance a no-op when it is not the one being executed,
  * slot = generation % versions per buffer version,
  * the core each instance runs on,
  * Set/Wait as explicit synchronisation, with the flag id and slot named.
With that text in hand, the conformance checker compares the two PROGRAMS rather than the two
schedules, which is the only comparison that can catch a mutated slot, core or order.

STATUS. The generator and the checker are implemented and tested against synthetic twins
(constructed to agree, and mutated to disagree). Emitting a twin that TileLang actually
COMPILES is deliberately NOT claimed: that needs an Ascend codegen path this stage does not
have, and reporting a compiled artifact we did not compile would be the worst kind of false
evidence.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "TwinError",
    "TwinInstance",
    "TwinProgram",
    "build_twin",
    "parse_twin",
    "check_conformance",
    "ConformanceReport",
    "AUTO_SCHEDULE_FLAG",
]

# The literal the twin must carry. Kept as a module constant so the checker, the generator and
# the tests all refer to ONE spelling: a twin that says "auto_schedule = False" (spaced) and a
# checker that greps "auto_schedule=False" would disagree about a compliant twin.
AUTO_SCHEDULE_FLAG = "auto_schedule=False"


class TwinError(Exception):
    """A twin cannot be built or read."""

    def __init__(self, code, message, detail="", expected=None, actual=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class TwinInstance:
    """One logical iteration as the twin expresses it."""

    ordinal: int
    K: int
    buffer: str
    slot: int
    generation: int
    versions: int
    core: str
    guard: str
    stmts: tuple = ()


@dataclass(frozen=True)
class TwinProgram:
    """The twin's explicit schedule, as READ BACK from its source."""

    source: str
    auto_schedule: bool
    instances: tuple
    sync: tuple = ()
    kernels: int = 1

    def slots(self):
        return {(item.K, item.buffer): item.slot for item in self.instances}

    def cores(self):
        return {item.K: item.core for item in self.instances}


def build_twin(plan, completion=None, *, target=None, function_name="kernel") -> str:
    """Emit the explicit TileLang twin for a resolved plan.

    Every scheduling decision appears as literal text. Nothing is left for TileLang to
    re-derive: the loop is written out instance by instance, so there is no trip count to
    schedule, and each instance carries its own guard, slot and core.
    """
    versions = {}
    for buffer in getattr(plan, "buffers", ()) or ():
        versions[str(getattr(buffer, "root", ""))] = int(getattr(buffer, "slot_count", 1) or 1)

    instances = _instances_from_plan(plan)
    if not instances:
        raise TwinError("TWIN_NO_INSTANCES",
                        "the plan states no logical iteration to express",
                        "plan.loop", "a non-empty trip count", None)

    lines = [
        "# EXPLICIT TWIN -- TileLang must NOT re-schedule this.",
        "# Generated from a resolved plan; every instance, slot, core and edge is spelled out.",
        "import tilelang",
        "import tilelang.ascend.language as T",
        "",
        "def %s():" % function_name,
        "    @T.prim_func",
        "    def main(%s):" % ", ".join(_param_names(plan)),
        "        # auto_schedule is False so TileLang TRANSMITS this schedule instead of deriving one.",
        "        # Leaving it on makes the twin a second scheduler: agreement with TF would then be",
        "        # two solvers converging, not evidence that TF's decisions were carried across.",
        "        %s" % AUTO_SCHEDULE_FLAG,
        "",
    ]

    for instance in instances:
        lines.extend(_instance_lines(instance, versions))

    if completion is not None:
        lines.extend(_sync_lines(completion))

    lines.extend([
        "",
        "    return main",
        "",
    ])
    return "\n".join(lines)


def _param_names(plan):
    names = [str(getattr(buffer, "root", "buf")) for buffer in getattr(plan, "buffers", ()) or ()]
    return [re.sub(r"[^0-9A-Za-z_]", "_", name) for name in names] or ["x"]


def _instances_from_plan(plan):
    """One TwinInstance per (logical iteration, buffer) the plan really uses."""
    loop = getattr(plan, "loop", None)
    trip = 1
    if loop is not None:
        domain = getattr(loop, "domain", None) or {}
        try:
            start, stop = int(domain.get("start", 0)), int(domain.get("stop", 0))
            step = int(domain.get("step", 1)) or 1
            trip = max(0, -(-(stop - start) // step))
        except (TypeError, ValueError):
            trip = 1

    versions = {str(getattr(buffer, "root", "")): int(getattr(buffer, "slot_count", 1) or 1)
                for buffer in getattr(plan, "buffers", ()) or ()}
    cores = _core_per_instance(plan, trip)

    # ONE INSTANCE PER (LOGICAL ITERATION, BUFFER), NOT PER ACCESS.
    # A buffer touched twice in the same iteration -- once written, once read -- is ONE versioned
    # instance. Emitting it twice inflated a 4-instance program to 20 and made the twin describe
    # a different program than the plan. The accesses are merged into the instance's statement
    # list so nothing is lost, and first-appearance order is preserved so comparison stays
    # positional.
    instances = []
    positions = {}
    for k in range(max(1, trip)):
        for access in getattr(plan, "accesses", ()) or ():
            root = str(getattr(access, "buffer_root", ""))
            if not root:
                continue
            identifier = str(getattr(access, "access_id", ""))
            key = (k, root)
            if key in positions:
                existing = instances[positions[key]]
                if identifier and identifier not in existing.stmts:
                    instances[positions[key]] = dataclasses.replace(
                        existing, stmts=existing.stmts + (identifier,))
                continue
            count = versions.get(root, 1) or 1
            positions[key] = len(instances)
            instances.append(TwinInstance(
                ordinal=len(instances), K=k, buffer=root,
                slot=k % count, generation=k, versions=count,
                core=cores.get(k, "aiv"),
                guard="k == %d" % k,
                stmts=(identifier,)))
    return instances


def _core_per_instance(plan, trip):
    """Which core each logical iteration runs on.

    Read from the plan when it says, defaulted to 'aiv' otherwise. A plan that places work on
    both Cube and Vector must SAY so; inventing an assignment here would make the twin agree
    with TF for a reason TF never expressed.
    """
    placement = getattr(plan, "core_assignment", None) or {}
    cores = {}
    for key, value in (placement.items() if isinstance(placement, dict) else ()):
        try:
            cores[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return {k: cores.get(k, "aiv") for k in range(max(1, trip))}


def _instance_lines(instance, versions):
    lines = [
        "        # instance %d: logical iteration %d, buffer %s, slot %d of %d"
        % (instance.ordinal, instance.K, instance.buffer, instance.slot, instance.versions),
        "        with T.If(%s):" % instance.guard,
        "            T.no_op()",
        "        slot_%d_%s = %d" % (instance.K, _ident(instance.buffer), instance.slot),
        "        core_%d = \"%s\"" % (instance.K, instance.core),
    ]
    return lines


def _sync_lines(completion):
    lines = ["", "        # completion: explicit Set/Wait, with the slot and generation named."]
    for record in getattr(getattr(completion, "trace", None), "records", ()) or ():
        role = str(getattr(record, "role", ""))
        if role not in ("set", "wait"):
            continue
        # THE LABEL IS THE REAL CHANNEL NAME, NOT AN IDENTIFIER.
        # It was being passed through _ident(), which turns "ready.s1.s2.s2" into
        # "ready_s1_s2_s2". That is fine for a Python variable and fatal for conformance: the
        # checker compares (role, channel) against the completion trace and every channel then
        # differed, so a CORRECT twin was refused and the failure looked like an ordering bug.
        channel = str(getattr(record, "channel", ""))
        lines.append(
            '        T.%s("%s", flag_id=%s, slot=%s, generation=%s)'
            % (role, channel, getattr(record, "event_id", 0),
               getattr(record, "slot", 0), getattr(record, "generation", 0)))
    return lines


def _ident(text):
    return re.sub(r"[^0-9A-Za-z_]", "_", text)


def parse_twin(source) -> TwinProgram:
    """Read a twin's schedule back OUT OF ITS SOURCE.

    Reading the source (rather than accepting a structure the caller hands over) is what makes
    the conformance check adversarial: the twin is compared as it would be COMPILED, so a
    mutated slot or core in the text is caught, and there is no in-memory object that can
    disagree with the file.
    """
    text = str(source or "")
    auto_schedule = _auto_schedule_is_off(text)
    instances = []
    for match in re.finditer(r"# instance (\d+): logical iteration (\d+), buffer (\S+),"
                             r" slot (\d+) of (\d+)", text):
        ordinal, K, buffer, slot, versions = match.groups()
        core_match = re.search(r"core_%s = \"([^\"]+)\"" % K, text)
        guard_match = re.search(r"with T\.If\(([^)]*)\)", text)
        instances.append(TwinInstance(
            ordinal=int(ordinal), K=int(K), buffer=buffer, slot=int(slot),
            generation=int(K), versions=int(versions),
            core=core_match.group(1) if core_match else "",
            guard=guard_match.group(1) if guard_match else ""))
    sync = re.findall(r'T\.(set|wait)\("([^"]+)"[^)]*\)', text)
    return TwinProgram(source=text, auto_schedule=auto_schedule, instances=tuple(instances),
                       sync=tuple(sync), kernels=len(re.findall(r"@T\.prim_func", text)))


def _auto_schedule_is_off(text):
    """True only when the AUTO_SCHEDULE_FLAG is present.

    The spaced form is accepted because it is the same declaration; what is NOT accepted is an
    absent flag, which is the case that lets TileLang schedule the twin itself.
    """
    if AUTO_SCHEDULE_FLAG in text:
        return True
    return bool(re.search(r"auto_schedule\s*=\s*False", text))


@dataclass(frozen=True)
class ConformanceReport:
    status: str
    code: str = ""
    detail: str = ""
    expected: Any = None
    actual: Any = None
    diagnostics: tuple = field(default_factory=tuple)

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.detail,
                "expected": self.expected, "actual": self.actual,
                "diagnostics": list(self.diagnostics)}


STATUS_MATCHED = "MATCHED"
STATUS_FAILED = "FAILED"

CODE_NOT_OFF = "TWIN_AUTO_SCHEDULE_ENABLED"
CODE_INSTANCE_MISMATCH = "TWIN_INSTANCE_MISMATCH"
CODE_SLOT_MISMATCH = "TWIN_SLOT_MISMATCH"
CODE_CORE_MISMATCH = "TWIN_CORE_MISMATCH"
CODE_SYNC_MISMATCH = "TWIN_SYNC_MISMATCH"
CODE_MULTIPLE_KERNELS = "TWIN_MULTIPLE_KERNELS"


def check_conformance(plan, twin, completion=None) -> ConformanceReport:
    """Compare the TF schedule against the twin, decision by decision.

    G42 allows a SEMANTICS-PRESERVING renumbering of flag ids and a relocation of addresses --
    both are names for the same program. It does NOT allow a mutated slot, a mutated core, or a
    changed order, because each of those is a different schedule. This checker refuses exactly
    the second group and passes the first, which is the distinction the requirement draws.
    """
    program = twin if isinstance(twin, TwinProgram) else parse_twin(twin)

    if not program.auto_schedule:
        return ConformanceReport(
            STATUS_FAILED, CODE_NOT_OFF,
            "the twin does not declare %s, so TileLang may re-schedule it; any agreement with"
            " TF would then be two schedulers converging rather than a transmitted decision"
            % AUTO_SCHEDULE_FLAG, expected=AUTO_SCHEDULE_FLAG, actual="absent")

    if program.kernels != 1:
        return ConformanceReport(
            STATUS_FAILED, CODE_MULTIPLE_KERNELS,
            "one candidate call must correspond to ONE logical kernel; the twin declares %d"
            % program.kernels, expected=1, actual=program.kernels)

    expected_instances = _instances_from_plan(plan)
    if len(expected_instances) != len(program.instances):
        return ConformanceReport(
            STATUS_FAILED, CODE_INSTANCE_MISMATCH,
            "the plan has %d (instance, buffer) pair(s); the twin declares %d"
            % (len(expected_instances), len(program.instances)),
            expected=len(expected_instances), actual=len(program.instances))

    for expected, actual in zip(expected_instances, program.instances):
        if actual.K != expected.K:
            return ConformanceReport(
                STATUS_FAILED, CODE_INSTANCE_MISMATCH,
                "the twin's logical iterations do not follow the plan's order",
                expected=expected.K, actual=actual.K)
        if actual.slot != expected.slot:
            return ConformanceReport(
                STATUS_FAILED, CODE_SLOT_MISMATCH,
                "buffer %r instance %d: the plan allots slot %d, the twin uses %d"
                % (expected.buffer, expected.K, expected.slot, actual.slot),
                expected=expected.slot, actual=actual.slot)
        if actual.core != expected.core:
            return ConformanceReport(
                STATUS_FAILED, CODE_CORE_MISMATCH,
                "instance %d: the plan places it on %r, the twin on %r"
                % (expected.K, expected.core, actual.core),
                expected=expected.core, actual=actual.core)

    # HAPPENS-BEFORE: the twin must express the SAME ordered Set/Wait sequence. Flag IDS may be
    # renumbered (semantics-preserving), so the comparison is on (role, channel) order.
    if completion is not None:
        expected_sync = [(str(record.role), str(record.channel))
                         for record in getattr(getattr(completion, "trace", None), "records", ())
                         or () if str(getattr(record, "role", "")) in ("set", "wait")]
        actual_sync = [(role, channel) for role, channel in program.sync]
        if expected_sync != actual_sync:
            where = next((index for index, (want, got) in enumerate(zip(expected_sync, actual_sync))
                          if want != got), min(len(expected_sync), len(actual_sync)))
            return ConformanceReport(
                STATUS_FAILED, CODE_SYNC_MISMATCH,
                "the twin's completion order diverges at position %d" % where,
                expected=expected_sync[where:where + 3], actual=actual_sync[where:where + 3])

    return ConformanceReport(STATUS_MATCHED, detail="", diagnostics=(
        "instances=%d" % len(program.instances),))
