"""Where a scheduling delay's number actually comes from (lead ruling D1-D4).

A delay never stands alone as an integer. Each one carries a STATE, a BASIS, a
UNIT and a LOCATABLE record of what produced it:

  PROVEN_ZERO          0 model ticks, justified by the explicit D2 predicate
                       below. The only family allowed to be zero today: one
                       straight-line region, same AIV instance (core_kind AND
                       sid), same vector/scalar pipe, producer completion =
                       ISSUE, and a grid of exactly one instance. Cross-pipe,
                       async DMA (mte2/mte3), data_ready, write_visible,
                       version wraparound reuse and cross-region edges are NOT
                       zero-delay facts.
  DECLARED             an explicit role value in the edge table, with a unit.
  OCCUPANCY_FALLBACK   no role value: the producing instruction's registered
                       resource window, used as a LOWER BOUND.
  MODEL_COST           no occupancy either (pset): a reviewed model entry whose
                       locator states which profile it came from.
  UNSOURCED            nothing above exists. It is never projected into 0 or 1;
                       the consumer of the record refuses by name.

A silent 0 is the bug this module exists to prevent. Units are MODEL TICKS,
never device cycles: optimality claims stay inside the model, and hardware
correctness is the completion plan's obligation (READY/FREE/exit drain), not
this table's.

WHY THE LOCATOR IS STRUCTURE, NOT A SENTENCE. This table's bytes are folded
into the problem and plan identities, so wording must not be able to move an
identity: rephrasing an explanation means the same thing and would otherwise
invalidate every stored plan for no reason. Every locator here is a record of
names plus the value it carries (ModelSource) and every recorded disagreement
is a record of fields (ModelConflict). Renaming a module, symbol or field, or
changing a value, still changes the identity, because that IS a real change.
Human wording exists only in report()/refusal details, which are never hashed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

__all__ = [
    "DECLARED",
    "DELAY_TABLE_SCHEMA",
    "MODEL_COST",
    "OCCUPANCY_FALLBACK",
    "PROVEN_ZERO",
    "UNIT_MODEL_TICK",
    "UNSOURCED",
    "DelayFact",
    "DelayRefusal",
    "ModelConflict",
    "ModelSource",
    "declared_occupancy",
    "delay_table_bytes",
    "edge_delay_fact",
    "occupancy_ticks",
    "proven_zero_predicate",
    "reuse_delay_fact",
]

PROVEN_ZERO = "PROVEN_ZERO"
DECLARED = "DECLARED"
OCCUPANCY_FALLBACK = "OCCUPANCY_FALLBACK"
MODEL_COST = "MODEL_COST"
UNSOURCED = "UNSOURCED"
UNIT_MODEL_TICK = "MODEL_TICK"
REFERENCE = "REFERENCE_IMPLEMENTATION"
DELAY_TABLE_SCHEMA = "native_mainline_delay_table"

#: The pipes whose declared facts make same-pipe ordering a real guarantee.
_SAME_PIPE_ORDERED = ("vector", "scalar")


def _pipe_facts() -> dict:
    """PIPE_FACTS, imported lazily: the completion package imports plan.units."""
    import importlib

    name = "tilefoundry_ascend.completion.model"
    if name not in sys.modules:
        try:
            importlib.import_module(name)
        except ImportError:
            return {}
    return sys.modules[name].PIPE_FACTS


def _encode_value(value):
    """Encode one model value with the shared canonical encoder, no repr path."""
    from .identity import _encode

    return _encode(value, {}, "$.value", 0)


class DelayRefusal(RuntimeError):
    """A delay has no source: refuse, project nothing, default nothing."""

    def __init__(self, code: str, detail: str, subject: str = "") -> None:
        self.code = code
        self.detail = detail
        self.subject = subject
        super().__init__("%s: %s (subject=%r)" % (code, detail, subject))


@dataclass(frozen=True)
class ModelSource:
    """WHERE a model number came from: names and the value, never a sentence."""

    module: str
    symbol: str
    field: str
    value: object = None

    def canonical(self) -> dict:
        return {"#": "model_source", "module": self.module, "symbol": self.symbol,
                "field": self.field, "value": _encode_value(self.value)}

    def report(self) -> str:
        """Wording for reports and refusals; never enters identity bytes."""
        return "%s %r.%s=%r" % (self.module, self.symbol, self.field, self.value)


@dataclass(frozen=True)
class ModelConflict:
    """Two sources disagree, recorded as fields. Nothing is relaxed by it."""

    conflict_field: str
    declared: str
    other_field: str
    other_claim: str
    resolution: str = "DECLARATION_WINS"

    def canonical(self) -> dict:
        return {"#": "model_conflict", "conflict_field": self.conflict_field,
                "declared": self.declared, "other_field": self.other_field,
                "other_claim": self.other_claim, "resolution": self.resolution}

    def report(self) -> str:
        return ("UNRESOLVED: declared %s=%s disagrees with %s=%s; %s applies and "
                "the ordering obligation stays with the completion plan"
                % (self.conflict_field, self.declared, self.other_field,
                   self.other_claim, self.resolution))


@dataclass(frozen=True)
class DelayFact:
    """One delay's state, number, unit and the locatable evidence behind it."""

    state: str
    ticks: int | None
    basis: str
    unit: str
    source: ModelSource
    status: str
    uncertainty: ModelConflict | None = None
    proof: tuple = ()

    def __post_init__(self) -> None:
        # A locator is a record, not a string: accepting prose here would put
        # wording straight back into the identity bytes.
        if not isinstance(self.source, ModelSource):
            raise TypeError("DelayFact.source must be a ModelSource, got %s"
                            % (type(self.source).__name__,))
        if self.uncertainty is not None and not isinstance(
                self.uncertainty, ModelConflict):
            raise TypeError("DelayFact.uncertainty must be a ModelConflict or None")

    def canonical(self) -> dict:
        return {"#": "delay_fact", "state": self.state,
                "ticks": (None if self.ticks is None else int(self.ticks)),
                "basis": self.basis, "unit": self.unit,
                "source": self.source.canonical(), "status": self.status,
                "uncertainty": (None if self.uncertainty is None
                                else self.uncertainty.canonical()),
                "proof": sorted([[str(field), _encode_value(value)]
                                 for field, value in self.proof])}

    def resolve(self) -> int:
        """The model-tick number a constraint needs, or a named refusal."""
        if self.state == UNSOURCED:
            raise DelayRefusal(
                "DELAY_UNSOURCED",
                "this delay has no declared model source; an unsourced delay is "
                "never projected into 0 or 1", self.source.report())
        if self.unit != UNIT_MODEL_TICK:
            raise DelayRefusal("DELAY_UNIT_UNKNOWN",
                               "a delay must state its unit", self.unit)
        if self.ticks is None or int(self.ticks) < 0:
            raise DelayRefusal("DELAY_TICKS_INVALID",
                               "a sourced delay carries a non-negative number",
                               self.source.report())
        if self.ticks == 0 and self.state != PROVEN_ZERO:
            raise DelayRefusal(
                "DELAY_ZERO_UNJUSTIFIED",
                "a zero delay is legal only under the proven-zero predicate",
                self.source.report())
        return int(self.ticks)


# ------------------------------------------------------------------ occupancy
def _occupancy_rows():
    """Resource windows read from the target ISA table where it declares one."""
    from tilefoundry_ascend.ir.isa import ISA

    rows = {}
    for name in sorted(ISA):
        contract = ISA[name]
        if contract.latency_model is None:
            continue
        conflict = None
        facts = _pipe_facts().get(contract.resource, {})
        if contract.completion != "issue" and facts.get("visible_at_issue"):
            # The instruction declares a later completion than the pipe fact
            # claims. Recorded structurally: never used to relax a delay and
            # never stored as a sentence.
            conflict = ModelConflict(
                conflict_field="completion", declared=contract.completion,
                other_field="PIPE_FACTS[%s].visible_at_issue" % (contract.resource,),
                other_claim="True")
        rows[name] = DelayFact(
            state=DECLARED, ticks=int(contract.latency_model),
            basis=DECLARED, unit=UNIT_MODEL_TICK,
            source=ModelSource(module="ir/isa.py", symbol=name,
                               field="latency_model", value=int(contract.latency_model)),
            status=REFERENCE,
            uncertainty=conflict,
            proof=(("latency_model", contract.latency_model),
                   ("ii_model", contract.ii_model),
                   ("completion", contract.completion),
                   ("resource", contract.resource)))
    return rows


_OCCUPANCY = _occupancy_rows()

#: Instructions the ISA does not model at all. Each row is an explicit,
#: reviewable model cost whose locator names the profile it came from. This
#: replaces the old max(1, latency_model or 0), which invented a tick silently.
_MODELLED_GAPS = {
    "pset": DelayFact(
        state=MODEL_COST, ticks=1, basis=MODEL_COST, unit=UNIT_MODEL_TICK,
        source=ModelSource(module="plan/delays.py", symbol="pset",
                           field="model_profile_window", value=1),
        status=REFERENCE,
        proof=(("isa_latency_model", None), ("category", "predicate"),
               ("resource", "vector"), ("completion", "issue"))),
}


def declared_occupancy(instruction: str) -> DelayFact:
    """The registered resource window for one instruction, or UNSOURCED."""
    rows = dict(_OCCUPANCY)
    rows.update(_MODELLED_GAPS)
    fact = rows.get(str(instruction))
    if fact is not None:
        return fact
    return DelayFact(state=UNSOURCED, ticks=None, basis=UNSOURCED, unit="",
                     source=ModelSource(module="plan/delays.py",
                                        symbol=str(instruction),
                                        field="occupancy_row", value=None),
                     status=UNSOURCED)


def occupancy_ticks(instruction: str) -> int:
    return declared_occupancy(instruction).resolve()


# ---------------------------------------------------------------- edge delays
#: Explicit per-role write-visibility / read-completion values with units.
#: Empty by decision: the target declares no such number today. The ladder
#: below falls back to a registered occupancy instead of inventing one, and
#: adding a row here is a reviewable act that changes plan identity.
_EDGE_ROLES: dict = {}


def _owner_instance(owner, path: str):
    """The (core_kind, sid) identity of one execution owner, or refusal.

    'Same AIV' means the same INSTANCE: core_kind alone is a core TYPE. Two
    different AIV sids share a pipe name and share no ordering guarantee, so a
    predicate that compared only core_kind would certify zero latency across
    instances that have no such guarantee. sid is taken as declared; a missing
    owner is refused rather than assumed to match.
    """
    if owner is None:
        raise DelayRefusal("DELAY_OWNER_MISSING",
                           "%s states no execution owner, so same-instance order "
                           "cannot be proven" % path, path)
    core_kind = getattr(owner, "core_kind", None)
    sid = getattr(owner, "sid", "<<absent>>")
    if core_kind is None:
        raise DelayRefusal("DELAY_OWNER_MISSING", "%s carries no core_kind" % path,
                           path)
    return str(core_kind), sid


def proven_zero_predicate(*, producer_pipe: str, producer_completion: str,
                          consumer_pipe: str, same_region: bool,
                          straight_line: bool, producer_owner, consumer_owner,
                          grid_instances, distance: int = 0) -> tuple:
    """The single D2 predicate. A non-empty result IS the proof.

    Every clause is required, and producer_owner/consumer_owner/grid_instances
    are mandatory keywords so no caller can silently assert an instance or a
    single launch it never checked. The result is a tuple of (field, value)
    pairs: names and values only, so the proof can be identity-bound without
    any wording being able to move it.
    """
    if not same_region or not straight_line or distance != 0:
        return ()
    if producer_pipe != consumer_pipe or producer_pipe not in _SAME_PIPE_ORDERED:
        return ()
    producer_core, producer_sid = _owner_instance(producer_owner, "producer")
    consumer_core, consumer_sid = _owner_instance(consumer_owner, "consumer")
    if (producer_core, producer_sid) != (consumer_core, consumer_sid):
        return ()
    try:
        instances = int(grid_instances)
    except (TypeError, ValueError):
        raise DelayRefusal("DELAY_GRID_UNSOURCED",
                           "the region states no explicit grid instance count",
                           str(grid_instances)) from None
    if instances != 1:
        return ()
    if str(producer_core).lower() != "aiv":
        return ()
    facts = _pipe_facts().get(producer_pipe)
    if facts is None or producer_completion != "issue":
        return ()
    if not facts.get("visible_at_issue") or not facts.get("same_pipe_order"):
        return ()
    if str(facts.get("core")).lower() != "aiv":
        return ()
    return (("same_region", True), ("straight_line", True), ("distance", 0),
            ("pipe", producer_pipe),
            ("core_kind", producer_core), ("sid", producer_sid),
            ("grid_instances", instances),
            ("producer_completion", producer_completion),
            ("visible_at_issue", True), ("same_pipe_order", True))


def _occupancy_fallback(producer_occupancy: DelayFact, ticks: int,
                        role_field: str, extra: tuple) -> DelayFact:
    """One edge/reuse delay carried by the producing instruction's window.

    The locator IS the producing row's locator: the number came from there, so
    pointing at it directly keeps the fallback traceable, moves identity when
    that row genuinely changes, and never copies a sentence.
    """
    return DelayFact(
        state=producer_occupancy.state, ticks=ticks,
        basis=OCCUPANCY_FALLBACK, unit=UNIT_MODEL_TICK,
        source=producer_occupancy.source,
        status=producer_occupancy.status,
        uncertainty=producer_occupancy.uncertainty,
        proof=((role_field, None), ("occupancy_ticks", ticks)) + extra)


def edge_delay_fact(*, producer_instruction: str, producer_pipe: str,
                    producer_completion: str, consumer_pipe: str,
                    producer_occupancy: DelayFact, same_region: bool,
                    straight_line: bool, producer_owner, consumer_owner,
                    grid_instances, distance: int = 0) -> DelayFact:
    """One edge's delay by the declared ladder: role, occupancy, model cost.

    Bare zero is impossible here: PROVEN_ZERO needs the predicate and every
    other state resolves through a registered number.
    """
    proof = proven_zero_predicate(
        producer_pipe=producer_pipe, producer_completion=producer_completion,
        consumer_pipe=consumer_pipe, same_region=same_region,
        straight_line=straight_line, producer_owner=producer_owner,
        consumer_owner=consumer_owner, grid_instances=grid_instances,
        distance=distance)
    if proof:
        return DelayFact(state=PROVEN_ZERO, ticks=0, basis=PROVEN_ZERO,
                         unit=UNIT_MODEL_TICK,
                         source=ModelSource(module="plan/delays.py",
                                            symbol="proven_zero_predicate",
                                            field="predicate", value=True),
                         status=REFERENCE, proof=proof)
    role = _EDGE_ROLES.get((producer_pipe, producer_completion, consumer_pipe))
    if role is not None:
        return role
    if producer_occupancy.state == UNSOURCED:
        return DelayFact(
            state=UNSOURCED, ticks=None, basis=UNSOURCED, unit="",
            source=ModelSource(module="plan/delays.py",
                               symbol=str(producer_instruction),
                               field="edge_role", value=None),
            status=UNSOURCED)
    ticks = producer_occupancy.resolve()
    return _occupancy_fallback(producer_occupancy, ticks, "edge_role_value", (
        ("producer_pipe", producer_pipe),
        ("producer_completion", producer_completion),
        ("consumer_pipe", consumer_pipe),
        ("ordering_obligation", "COMPLETION_PLAN")))


def reuse_delay_fact(*, producer_instruction: str, producer_pipe: str,
                     producer_completion: str, producer_occupancy: DelayFact,
                     distance_offset: int) -> DelayFact:
    """A version-wraparound delay. Reuse is never in the proven-zero family."""
    role = _EDGE_ROLES.get(("reuse", producer_pipe, producer_completion))
    if role is not None:
        return role
    if producer_occupancy.state == UNSOURCED:
        return DelayFact(
            state=UNSOURCED, ticks=None, basis=UNSOURCED, unit="",
            source=ModelSource(module="plan/delays.py",
                               symbol=str(producer_instruction),
                               field="reuse_role", value=None),
            status=UNSOURCED)
    ticks = producer_occupancy.resolve()
    return _occupancy_fallback(producer_occupancy, ticks, "reuse_role_value", (
        ("distance_offset", int(distance_offset)),
        ("producer_pipe", producer_pipe),
        ("read_completion", "COMPLETION_PLAN")))


def delay_table_bytes() -> bytes:
    """Canonical bytes of the entire delay surface, for identity binding.

    Names, values and flags only. The policy block states which ladder and
    which unit are legal, as data, so a reworded comment anywhere in this
    module cannot invalidate a stored plan.
    """
    occupancy = dict(_OCCUPANCY)
    occupancy.update(_MODELLED_GAPS)
    payload = {
        "schema": DELAY_TABLE_SCHEMA,
        "occupancy": {name: occupancy[name].canonical() for name in sorted(occupancy)},
        "edge_roles": {str(key): _EDGE_ROLES[key].canonical()
                       for key in sorted(_EDGE_ROLES, key=str)},
        "proven_zero": {
            "pipes": list(_SAME_PIPE_ORDERED),
            "required": ["same_region", "straight_line", "distance", "core_kind",
                         "sid", "grid_instances", "producer_completion",
                         "visible_at_issue", "same_pipe_order"],
            "grid_instances_required": 1,
            "pipe_facts": {name: {key: _pipe_facts()[name].get(key)
                                  for key in ("core", "completion",
                                              "visible_at_issue", "same_pipe_order")}
                           for name in sorted(_SAME_PIPE_ORDERED)
                           if name in _pipe_facts()}},
        "policy": {"bare_zero_allowed_only_for": PROVEN_ZERO,
                   "ladder": [DECLARED, OCCUPANCY_FALLBACK, MODEL_COST, UNSOURCED],
                   "unit": UNIT_MODEL_TICK,
                   "ordering_authority": "COMPLETION_PLAN",
                   "device_claim": False},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")
