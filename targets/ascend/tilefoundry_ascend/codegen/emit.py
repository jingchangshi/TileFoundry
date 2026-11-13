"""TileLang emission for the supported S0 domain: real source, real parse.

WHAT THIS PRODUCES. emit_supported_tl reads the ACTUAL lowered TIR entry
PrimFunction and states the equivalent TileLang program: source text, the file
the deployed builder needs, the parsed tvm.tirx.PrimFunc, the ABI derived from
the real entry parameters, and content identities over those bytes. Nothing is
compiled and no device is touched; parse-of-emitted-source is the only
observation point this slice can prove (docs/handoff/TL_LOWERING_SPEC.md §4).

SUPPORTED DOMAIN = STRUCTURAL PREDICATES, NOT A NAME WHITELIST. The program is
emitted when the lowered entry states exactly:

  one MeshScope (one program instance); no If/While;
  every instruction an Evaluate whose Op has a registered Ascend ISA contract
  with a TileLang surface in TL_SURFACE, used in the execution region its
  contract requires (T.copy for the MTE transfers, T.simd.* inside T.SimdVF());
  operands are bound Vars whose root is a full-tile zero-offset window, a
  register allocation, or a callable parameter, all with one static int extent
  and an f32/i32-carrying dtype the vector surface states;
  a loop only when it carries the author's PipelineSpec, with static bounds,
  zero origin and step 1, emitted as T.Pipelined(extent, num_stages=max_stage);
  the trailing parameters the ABI declares as outputs are exactly the GM
  destinations the emitted copies write, each written once.

Anything else is a NAMED refusal bound to the real node: an instruction with no
TileLang surface, an operand shape the surface cannot state, a window at a
nonzero offset, a rank != 1 tile, an unsupported dtype, a loop this surface has
no contract for, an output the program never writes. The emitter never drops a
statement to make a kernel fit, and never calls tilelang.compile.

WHY THE SOURCE GOES TO DISK. The deployed @T.prim_func binds through
inspect.getsourcelines (tilelang/language/eager/builder.py:1791), so an emitted
kernel that was never a file cannot be parsed. The path is a function of the
source bytes, so one program always maps to one file and one artifact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
from tilefoundry.ir.types.utils import static_dim_value

from ..ir.isa import ISA
from .abi import abi_from_prim_function

__all__ = ["EmittedTl", "EmitRefusal", "TlBlock", "TlBuffer", "TlStatement",
           "emit_supported_tl", "parse_emitted_source", "supported_domain"]

#: The vector element width this target's SimdVF surface states (isa.pset
#: elem_width_bits, and tilelang simd.py _vreg_lanes over a 256-bit predicate).
VECTOR_ELEMENT_WIDTH = 32

#: One ISA contract -> the TileLang surface that states it. Absent means no
#: surface exists in this slice, which is a refusal, not a guess.
TL_SURFACE = {
    "gm_to_ub": "copy",
    "ub_to_gm": "copy",
    "vld": "simd.vld",
    "pset": "simd.pset",
    "vadd": "simd.vadd",
    "vsts": "simd.vsts",
    # Event instructions. These are the ONLY two the deployed TL surface states:
    # T.ascend_set_flag / T.ascend_wait_flag (tilelang/ascend/language/sync.py:28-71).
    "flag_set": "event.set_flag",
    "flag_wait": "event.wait_flag",
}

#: Event names whose TL face does NOT exist, and which must therefore be refused
#: by name instead of being silently dropped or substituted:
#:   drain        - no ascend_drain in the deployed TL surface (verified against
#:                  ascend/language/sync.py's __all__), so a region-exit drain has
#:                  no instruction to emit.
#:   pipe_barrier / threadfence - the lead's ruling: these have NO TileFoundry op
#:                  at all (only DSL-language functions), so they are not even
#:                  candidates; substituting one for a cross-pipe ordering
#:                  requirement would be claiming a sync we cannot emit.
EVENT_TL_SURFACE_MISSING = frozenset({"drain"})
EVENT_TF_OP_MISSING = frozenset({"pipe_barrier", "threadfence"})

#: hard-event token per canonical pipe, copied from the deployed build's own
#: GetResourcePipeName (tilelang/src/ascend/transform/ascend_pipe.h:75-91):
#: MTE1/MTE2/MTE3/M/V/FIX/S. Not our invention - a mismatch here is a wrong
#: string reaching pto.set_flag's first two arguments (codegen_pto.cc:4847-4853).
HARD_EVENT_TOKENS = {"mte1": "MTE1", "mte2": "MTE2", "mte3": "MTE3",
                     "cube": "M", "m": "M", "vector": "V", "v": "V",
                     "fix": "FIX", "fixpipe": "FIX", "scalar": "S", "s": "S"}

#: The four aliases GetHardEvent applies after joining the pair
#: (insert_sync.cc:751-765). Reproduced exactly, because the string we emit must
#: be the string that pass would have produced.
HARD_EVENT_ALIASES = {("MTE2", "M"): "MTE2_MTE1", ("M", "MTE2"): "MTE1_MTE2",
                      ("MTE3", "M"): "MTE3_MTE1", ("M", "MTE3"): "MTE1_MTE3"}

#: Flag attributes, from tilefoundry/ir/tir/flag.py _ATTRS. The emitter reads them
#: off the real Op instance; it does not accept a channel from anywhere else.
_EVENT_ATTRS = ("channel_name", "kind", "slot", "generation", "versions", "flag_id",
                "producer_pipe", "consumer_pipe")


def hard_event_name(producer_pipe, consumer_pipe):
    """The deployed build's hard-event string for one pipe pair.

    Raises (never guesses) for a pipe token the build does not name: an unknown
    pipe would otherwise be emitted as a plausible-looking flag the hardware has
    no such event for.
    """
    from ..completion.model import canonical_pipe

    try:
        left = HARD_EVENT_TOKENS[canonical_pipe(producer_pipe)]
        right = HARD_EVENT_TOKENS[canonical_pipe(consumer_pipe)]
    except KeyError:
        raise EmitRefusal("TL_EVENT_PIPE_TOKEN_UNKNOWN",
                          "the deployed build names no resource pipe for %r->%r, so no "
                          "hard-event string can be stated" % (producer_pipe,
                                                                consumer_pipe),
                          None, None)
    return HARD_EVENT_ALIASES.get((left, right), left + "_" + right)

#: Tile dtype names the deployed TileLang surface states, keyed by the TF dtype
#: name the lowered program carries. Every entry's REGISTER geometry (how many
#: lanes one vector register holds) comes from codegen/vector_model.py with its
#: own provenance: 32-bit widths are SOURCED against the target's declared
#: lane_count, 16-bit widths are UNSOURCED until a device says otherwise, and an
#: unlisted width is a refusal rather than a guess.
#: Element dtype names the deployed vector surface states, keyed by the TF dtype
#: name the lowered program carries. "bool" is the predicate register an pset
#: publishes; it is an ELEMENT name, not a tile dtype - a tile of predicates is
#: not something this slice copies.
ELEMENT_DTYPES = {"f32": "float32", "float32": "float32",
                  "bf16": "bfloat16", "bfloat16": "bfloat16",
                  "f16": "float16", "float16": "float16",
                  "bool": "bool"}
#: Tile dtypes this slice copies into and out of UB.
TL_DTYPES = {name: value for name, value in ELEMENT_DTYPES.items() if name != "bool"}

#: The dtype every deployed pset publishes, regardless of its element width
#: (tilelang/ascend/language/simd.py:105-111). Cited, not spelled locally.
from .vector_model import PREDICATE_DTYPE as PREDICATE_VECTOR_DTYPE

_READ = "READ"
_WRITE = "WRITE"


class EmitRefusal(RuntimeError):
    """A named refusal at the TileLang emit boundary, bound to the real node.

    expected/actual are optional and carry the two sides of a mismatch when the
    refusal is a comparison, so a caller can assert on the numbers rather than on
    prose.
    """

    def __init__(self, code: str, detail: str, source_ref=None,
                 expected=None, actual=None) -> None:
        self.code = code
        self.detail = detail
        self.source_ref = source_ref
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class TlBuffer:
    """One buffer the emitted program declares: a GM parameter or a UB tile."""

    name: str
    scope: str            # "global" | "shared"
    dtype: str            # the TL element dtype name, e.g. "float32"/"bfloat16"
    extent: int           # ELEMENTS in this tile (what a copy region moves)
    origin: int           # the storage address the lowered window states (0 for a param)


@dataclass(frozen=True)
class TlStatement:
    """One emitted TileLang instruction, bound to the real TIR statement."""

    surface: str              # "copy" | "simd.*" | "event.set_flag" | "event.wait_flag"
    instruction: str          # the ISA contract name it came from
    result: str               # the register/buffer name the instruction writes ("" if none)
    reads: tuple              # names consumed, in the real operand order
    writes: tuple             # names written, in the real operand order
    attributes: tuple         # (name, value) attributes actually stated on the Op
    constants: tuple          # (name, int) emitted as literals (extent, elem_width)
    source_order: int
    in_vector_region: bool
    #: The deployed build's hard-event string for a flag edge ("" for a data
    #: instruction). It is a machine fact read from the Op's own pipe pair through
    #: GetResourcePipeName's table, so the reader can compare the flag that will be
    #: emitted without comparing any prose.
    hard_event: str = ""

    @property
    def names(self) -> tuple:
        return tuple(self.reads) + tuple(self.writes)

    def as_dict(self) -> dict:
        return {
            "surface": self.surface, "instruction": self.instruction,
            "result": self.result, "reads": list(self.reads), "writes": list(self.writes),
            "attributes": list(self.attributes), "constants": list(self.constants),
            "hard_event": self.hard_event,
            "source_order": self.source_order, "in_vector_region": self.in_vector_region,
        }


@dataclass(frozen=True)
class TlBlock:
    """One emitted region: the function body, a pipelined loop or a SimdVF scope."""

    kind: str                 # "function" | "kernel" | "loop" | "simdvf"
    members: tuple            # tuple[TlStatement | TlBlock] in real source order
    extent: int = 0
    num_stages: int = 0
    induction_var: str = ""

    def statements(self) -> tuple:
        out = []

        def walk(member):
            if isinstance(member, TlBlock):
                for item in member.members:
                    walk(item)
                return
            out.append(member)

        for member in self.members:
            walk(member)
        return tuple(out)


@dataclass(frozen=True)
class EmittedTl:
    """The emitted program: its structure and the exact bytes it renders to."""

    entry: str
    source: str
    parameters: tuple
    output_count: int
    buffers: tuple
    body: TlBlock
    lane_count: int
    element_width: int
    grid_blocks: int = 0
    #: (register name, vector dtype) for every register this kernel binds.
    registers: tuple = ()
    #: (name, ELEMENT dtype) for every tile AND register this kernel names. This
    #: is the single key a lane count or vector dtype is looked up through, on the
    #: emit side and the read side, so neither one re-derives it from a tile extent
    #: or spells it as a literal.
    elements: tuple = ()
    vector_surfaces: tuple = ()

    @property
    def source_bytes(self) -> bytes:
        return self.source.encode("utf-8")

    @property
    def source_identity(self) -> str:
        return "sha256:" + hashlib.sha256(self.source_bytes).hexdigest()

    def statements(self) -> tuple:
        return self.body.statements()


def _storage_name(tensor_type) -> str:
    storage = getattr(tensor_type, "storage", None)
    return getattr(storage, "name", str(storage)).lower()


def _dtype_name(dtype) -> str:
    return getattr(dtype, "name", str(dtype))


def _is_predicate_type(tensor_type) -> bool:
    """True when the operand's own dtype states a predicate register."""
    if tensor_type is None:
        return False
    dtype = getattr(tensor_type, "dtype", None)
    return _dtype_name(dtype) == "bool"


def _effect_of(parameter) -> str:
    """READ/WRITE from the REAL ParamDef effect, never from position."""
    from tilefoundry.ir.core.param_def import MemoryEffect

    effect = getattr(parameter, "effect", None)
    if effect is None:
        raise EmitRefusal("TL_EFFECT_UNDECLARED",
                          "operand %r declares no MemoryEffect" % (parameter.name,), parameter)
    read = bool(effect & MemoryEffect.READ)
    write = bool(effect & MemoryEffect.WRITE)
    if read and write:
        return "READ_WRITE"
    if write:
        return _WRITE
    if read:
        return _READ
    raise EmitRefusal("TL_EFFECT_EMPTY", "operand %r has no effect" % (parameter.name,),
                      parameter)


def _entry_event_facts(target):
    """The event surface the lowered entry's own target declares, or None.

    None is not a missing default to fill in: it means the target did not answer
    target_event_facts(), so no flag id can be proven to fit a declared pool and
    every event instruction must be refused by name rather than emitted.
    """
    hook = getattr(target, "target_event_facts", None)
    if not callable(hook):
        return None
    try:
        return hook()
    except Exception:  # noqa: BLE001 - a target that errors states no surface
        return None


class _Emitter:
    """Walk the REAL lowered entry once and state the TileLang program."""

    def __init__(self, entry: PrimFunction) -> None:
        if not isinstance(entry, PrimFunction):
            raise EmitRefusal("TL_ENTRY_NOT_PRIM_FUNCTION",
                              "TileLang emission reads the lowered TIR entry PrimFunction",
                              entry)
        self.entry = entry
        self.symbol = getattr(entry, "name", "") or ""
        if not self.symbol:
            raise EmitRefusal("TL_ENTRY_UNNAMED", "the entry PrimFunction has no symbol", entry)
        self.parameters = tuple(parameter.name for parameter in entry.params)
        self.output_count = int(getattr(entry, "output_count", 0) or 0)
        self.buffers: dict[str, TlBuffer] = {}
        self.registers: dict[int, str] = {}
        self.lane_count = 0
        self.order = 0
        self._params_by_id = {id(parameter): parameter for parameter in entry.params}
        self._addresses: dict[int, str] = {}
        self._meshes = 0
        self._grid_blocks = 0
        #: name -> ELEMENT dtype (e.g. "float32", "bfloat16", "bool") for every
        #: tile AND every register this kernel names. The vector dtype and the
        #: register-access extent are then derived from it through
        #: codegen/vector_model.py, never spelled here.
        self.dtypes: dict[str, str] = {}
        self.register_dtypes: dict[str, str] = {}
        self._vector_surfaces: set[str] = set()
        #: The target the entry was lowered against. Event instructions need its
        #: declared flag pool to check a flag id against, and there is no default:
        #: an entry whose target states no event surface cannot emit a flag.
        self.target = getattr(entry, "target", None)
        self.event_facts = _entry_event_facts(self.target)

    # ------------------------------------------------------------------ roots
    def _element_of(self, dtype) -> str:
        """The deployed element name behind a TF dtype, bool included."""
        name = _dtype_name(dtype)
        if name not in ELEMENT_DTYPES:
            raise EmitRefusal("TL_UNSUPPORTED_DTYPE",
                              "this slice states no vector model for element type %r, so "
                              "it emits no lane count or dtype for it" % (name,), dtype,
                              sorted(ELEMENT_DTYPES))
        return ELEMENT_DTYPES[name]

    def _tl_dtype(self, dtype) -> str:
        name = _dtype_name(dtype)
        if name not in TL_DTYPES:
            raise EmitRefusal("TL_UNSUPPORTED_DTYPE",
                              "this slice states no vector model for a %s tile, so it "
                              "emits no form for it" % (name,), dtype,
                              sorted(TL_DTYPES))
        return TL_DTYPES[name]

    def _register(self, name: str, dtype: str) -> None:
        """Record which vector dtype one emitted register binder carries.

        The parsed IR states a Bind's var dtype, so the expectation must state it
        too - from the same cited model, never from the tile's element count.
        """
        self.register_dtypes[name] = dtype

    def _event_statement(self, surface: str, contract, stmt: Evaluate) -> TlStatement:
        """One flag set/wait, emitted exactly as the real Op states it.

        Rules, each a named refusal rather than a fallback:

        - The flag attributes are read off the Op instance and must be CONCRETE.
          tilefoundry/ir/tir/flag.py declares slot/generation/versions/flag_id as
          int attributes and checks slot == generation % versions at construction,
          so a candidate whose generation is still an Expr is a symbolic obligation
          this slice does not materialise. Expanding one symbolic event into N
          concrete ones would make the event count grow with the trip count, which
          is exactly what the finite template exists to avoid - so this refuses and
          leaves the symbolically-looped case to the control-envelope task rather
          than inventing a second, Expr-carrying event op.
        - The hard-event string comes from the deployed build's own pipe naming and
          pair aliases (see hard_event_name); an unmapped pipe token is refused.
        - Cross-core handoffs are refused: the target declares their bank/mode facts
          as UNRESOLVED, so no flag id can be proven to fit.
        - flag_id must sit inside the declared intra-core pool: emitting a slot the
          target does not have would look like a working synchronisation.
        - One statement in, one statement out: nothing is duplicated, dropped or
          reordered here.
        """
        from ..completion.model import core_for_pipe

        _values, _predicates, _writes, attributes = self._partition(stmt)
        stated = dict(attributes)
        missing = [name for name in _EVENT_ATTRS if name not in stated]
        if missing:
            raise EmitRefusal("TL_EVENT_ATTRIBUTE_MISSING",
                              "the %r statement states no %s attribute, so the emitted "
                              "flag cannot be bound to a channel, slot or generation"
                              % (contract.name, missing), stmt)
        concrete = {}
        for name in ("slot", "generation", "versions", "flag_id"):
            value = stated[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise EmitRefusal("TL_EVENT_GENERATION_SYMBOLIC",
                                  "the %r statement carries a non-concrete %s (%r); this "
                                  "slice emits one instruction per stated event and never "
                                  "expands a symbolic generation into per-trip events"
                                  % (contract.name, name, value), stmt)
            concrete[name] = int(value)
        if concrete["versions"] < 1:
            raise EmitRefusal("TL_EVENT_VERSIONS_INVALID",
                              "versions must be positive", stmt, ">= 1",
                              concrete["versions"])
        if concrete["slot"] != concrete["generation"] % concrete["versions"]:
            raise EmitRefusal("TL_EVENT_SLOT_INCONSISTENT",
                              "the statement's slot is not generation modulo versions, so "
                              "the flag names a ring position the plan did not resolve", stmt,
                              concrete["generation"] % concrete["versions"],
                              concrete["slot"])
        channel = str(stated["channel_name"])
        kind = str(stated["kind"])
        if not channel.strip():
            raise EmitRefusal("TL_EVENT_CHANNEL_MISSING",
                              "a flag edge must name the channel it orders", stmt)
        if kind not in ("ready", "free", "once"):
            raise EmitRefusal("TL_EVENT_KIND_UNKNOWN",
                              "the flag kind is not one the completion contract states",
                              stmt, ["ready", "free", "once"], kind)
        producer_pipe = str(stated["producer_pipe"])
        consumer_pipe = str(stated["consumer_pipe"])
        try:
            producer_core = core_for_pipe(producer_pipe)
            consumer_core = core_for_pipe(consumer_pipe)
        except Exception as error:  # noqa: BLE001 - unknown pipe is a named gap
            raise EmitRefusal("TL_EVENT_PIPE_UNKNOWN",
                              "no core domain for pipe pair %r->%r: %s"
                              % (producer_pipe, consumer_pipe, error), stmt) from None
        if producer_core != consumer_core:
            raise EmitRefusal("TL_EVENT_CROSS_CORE_UNSUPPORTED",
                              "this slice emits same-core flags only; a %s->%s hand-off "
                              "needs the cross-core bank/mode facts the target declares "
                              "UNRESOLVED" % (producer_core, consumer_core), stmt)
        hard_event = hard_event_name(producer_pipe, consumer_pipe)
        pool = self._event_pool(producer_core)
        if concrete["flag_id"] >= pool:
            raise EmitRefusal("TL_EVENT_SLOT_OUT_OF_POOL",
                              "flag id %d is outside the %s intra-core pool of %d ids the "
                              "target declares; emitting it would look like a working "
                              "synchronisation" % (concrete["flag_id"], producer_core, pool),
                              stmt, list(range(pool)), concrete["flag_id"])
        self._vector_surfaces.add(surface)
        return TlStatement(
            surface=surface,
            instruction=contract.name,
            result="",
            reads=(),
            writes=(),
            attributes=tuple((name, str(stated[name])) for name in
                             ("channel_name", "kind", "producer_pipe", "consumer_pipe")),
            constants=tuple(sorted(concrete.items())),
            source_order=self._take_order(),
            in_vector_region=False,
            hard_event=hard_event,
        )

    def _event_pool(self, core: str) -> int:
        """The declared intra-core hard-event id count, or a named refusal.

        Two rules this keeps.

        The core token arrives from core_for_pipe(), the target's own mapping, and
        is resolved through declared_core() against the domains the target
        declares. It is NEVER hand-cased here. An earlier version of this very
        function called hard_event_pool(core.upper()) while the target declares
        AIV="aiv" (model.py:32), so the lookup returned None; had None been read
        as "no limit", every flag id would have passed the capacity check
        silently. That is the failure mode this shape forbids, and
        test_event_pool_lookup_refuses_an_undeclared_core pins it.

        A missing pool is a refusal, never an unbounded pass: emitting a flag
        against a pool the target does not declare would look exactly like a
        working synchronisation.
        """
        from ..completion.model import declared_core

        facts = self.event_facts
        if facts is None:
            raise EmitRefusal("TL_EVENT_POOL_UNDECLARED",
                              "the target states no event surface, so no flag id can be "
                              "checked against a declared pool", core)
        resolved = declared_core(facts, core, "codegen emit")
        pool = facts.hard_event_pool(resolved, "intra_core")
        if pool is None:
            raise EmitRefusal("TL_EVENT_POOL_UNDECLARED",
                              "the target declares no intra-core hard-event pool for %r, "
                              "so no flag id can be bounds-checked for this hand-off"
                              % (resolved,), core,
                              [str(item.core_domain) for item in facts.hard_event_pools])
        return int(pool.capacity)

    def _take_order(self) -> int:
        order = self.order
        self.order += 1
        return order

    def _model_of(self, name: str):
        """The cited vector model behind one tile or register name."""
        from .vector_model import vector_model

        element = self.dtypes.get(name)
        if element is None:
            raise EmitRefusal("TL_ELEMENT_DTYPE_UNKNOWN",
                              "%r has no recorded element dtype, so no vector lane count "
                              "or register extent can be stated for it" % (name,),
                              sorted(self.dtypes), None)
        return vector_model(element)

    def _lanes_of(self, name: str) -> int:
        """The vector-register lane count one access to this name occupies.

        Derived from codegen/vector_model.py, whose rule is the deployed
        toolchain's own (2048 bits per element width) and whose provenance is
        stated per width: 32-bit is SOURCED against the target's declared
        lane_count, 16-bit is UNSOURCED pending device evidence. This is not a
        number this repository gets to pick, and it is not the tile's element
        count either: those coincide only for a 32-bit tile of exactly the
        declared lane count.
        """
        return int(self._model_of(name).lanes)

    def _vector_dtype_of(self, name: str) -> str:
        return self._model_of(name).vector_dtype

    def _extent(self, tensor_type, where: str) -> int:
        shape = tuple(getattr(tensor_type, "shape", ()) or ())
        if len(shape) != 1:
            raise EmitRefusal("TL_UNSUPPORTED_RANK",
                              "%s has rank %d; this slice emits one full contiguous tile "
                              "per buffer" % (where, len(shape)), tensor_type)
        value = static_dim_value(shape[0])
        if value is None:
            raise EmitRefusal("TL_DYNAMIC_SHAPE",
                              "%s has a dynamic dimension, so no copy extent can be "
                              "stated" % (where,), tensor_type)
        return int(value)

    def _declare(self, name: str, scope: str, tensor_type, origin: int, node) -> TlBuffer:
        if scope == "shared":
            storage = _storage_name(tensor_type)
            if storage != "smem":
                raise EmitRefusal("TL_BUFFER_SCOPE_MISMATCH",
                                  "%s is declared as a UB tile but lives in %s"
                                  % (name, storage), node)
            owner = self._addresses.get(int(origin))
            if owner is not None and owner != name:
                raise EmitRefusal("TL_BUFFER_ALIAS_UNPROVEN",
                                  "%s and %s window the same UB address %d; this slice "
                                  "declares one distinct tile per address"
                                  % (name, owner, int(origin)), node)
            self._addresses[int(origin)] = name
        buffer = TlBuffer(name=name, scope=scope, dtype=self._tl_dtype(tensor_type.dtype),
                          extent=self._extent(tensor_type, "buffer %s" % (name,)),
                          origin=int(origin))
        existing = self.buffers.get(name)
        if existing is not None:
            if existing != buffer:
                raise EmitRefusal("TL_BUFFER_NAME_COLLISION",
                                  "the lowered program declares %r twice with different "
                                  "storage: %r and %r" % (name, existing, buffer), node)
            return existing
        if self.lane_count and buffer.extent != self.lane_count:
            raise EmitRefusal("TL_LANE_COUNT_MISMATCH",
                              "%s has %d elements but this kernel's vector surface states "
                              "%d lanes; a partial tile is not emitted"
                              % (name, buffer.extent, self.lane_count), node)
        self.buffers[name] = buffer
        self._note_dtype(name, buffer.dtype)
        self.lane_count = buffer.extent
        return buffer

    def _note_dtype(self, name: str, element: str) -> None:
        """Record the ELEMENT dtype behind one tile or register name.

        Keyed by the name the emitted source will use. Two bindings under one
        name with different element widths would make the lane count ambiguous,
        so that is refused rather than resolved by last-writer-wins.
        """
        known = self.dtypes.get(name)
        if known is not None and known != element:
            raise EmitRefusal("TL_ELEMENT_DTYPE_CONFLICT",
                              "the name %r is bound to both %s and %s; its vector lane "
                              "count would be ambiguous" % (name, known, element),
                              name, element)
        self.dtypes[name] = element

    def _bind_definition(self, let: LetStmt) -> None:
        value = let.value
        target = getattr(value, "target", None)
        name = getattr(let.var, "name", "")
        tensor_type = getattr(let.var, "type", None)
        if tensor_type is None or not hasattr(tensor_type, "shape"):
            raise EmitRefusal("TL_DEFINITION_TYPE_MISSING",
                              "the lowered definition %r carries no tensor type" % (name,), value)
        storage = _storage_name(tensor_type)
        if isinstance(target, AllocTensor):
            if storage == "rmem":
                self.registers[id(let.var)] = name
                # A register's own dtype decides its vector shape: bool is the
                # predicate a pset publishes, f32/bf16 is a value register.
                self._note_dtype(name, self._element_of(tensor_type.dtype))
                return
            self._declare(name, "global" if storage == "gmem" else "shared", tensor_type,
                          0, value)
            self.registers[id(let.var)] = name
            return
        if isinstance(target, TensorView):
            if len(value.args) != 1:
                raise EmitRefusal("TL_UNSUPPORTED_WINDOW",
                                  "%s is a window with %d index expression(s); this slice "
                                  "emits whole zero-offset tiles only"
                                  % (name, len(value.args) - 1), value)
            base = value.args[0]
            if not isinstance(base, Constant) or not isinstance(base.value, int):
                raise EmitRefusal("TL_UNSUPPORTED_WINDOW_BASE",
                                  "%s is a window over a non-static address" % (name,), value)
            if storage == "gmem":
                raise EmitRefusal("TL_UNSUPPORTED_WINDOW",
                                  "%s re-windows a GM root, which this slice cannot bind to "
                                  "a declared parameter" % (name,), value)
            self._declare(name, "shared", tensor_type, int(base.value), value)
            self.registers[id(let.var)] = name
            return
        raise EmitRefusal("TL_DEFINITION_UNSUPPORTED",
                          "a lowered LetStmt binds neither an allocation nor a tile window",
                          value)

    def _operand_name(self, operand, where: str) -> str:
        if not isinstance(operand, Var):
            raise EmitRefusal("TL_UNSUPPORTED_OPERAND",
                              "%s is not a bound Var, so no TileLang buffer or register "
                              "name can be stated" % (where,), operand)
        tensor_type = getattr(operand, "type", None)
        if tensor_type is None or not hasattr(tensor_type, "shape"):
            raise EmitRefusal("TL_OPERAND_TYPE_MISSING",
                              "operand %r carries no tensor type"
                              % (getattr(operand, "name", operand),), operand)
        storage = _storage_name(tensor_type)
        name = getattr(operand, "name", "")
        if storage == "rmem":
            bound = self.registers.get(id(operand))
            if bound is None:
                raise EmitRefusal("TL_REGISTER_UNDECLARED",
                                  "the register %r has no allocation in this program" % (name,),
                                  operand)
            return name
        parameter = self._params_by_id.get(id(operand))
        if parameter is not None:
            if name not in self.buffers:
                self._declare(name, "global", tensor_type, 0, operand)
            return name
        bound = self.registers.get(id(operand))
        if bound is None or name not in self.buffers:
            raise EmitRefusal("TL_OPERAND_ROOT_UNKNOWN",
                              "operand %r is not a parameter and has no allocation/window "
                              "definition in this program" % (name,), operand)
        return name

    # ------------------------------------------------------------- instructions
    def _partition(self, stmt: Evaluate) -> tuple:
        schema = getattr(type(stmt.callable), "_op_schema", None)
        if schema is None:
            raise EmitRefusal("TL_INSTRUCTION_UNREGISTERED",
                              "an Evaluate carries an Op with no registered schema", stmt)
        parameters = tuple(p for p in schema.signature if p.kind == "input")
        if len(parameters) != len(stmt.args):
            raise EmitRefusal("TL_OPERAND_ARITY",
                              "instruction %r issues %d operand(s) but its schema declares "
                              "%d" % (getattr(schema, "name", "?"), len(stmt.args),
                                      len(parameters)), stmt)
        values, predicates, writes = [], [], []
        for parameter, operand in zip(parameters, stmt.args):
            name = self._operand_name(operand, "operand %r" % (parameter.name,))
            effect = _effect_of(parameter)
            if _WRITE in effect:
                # A READ_WRITE operand is the DESTINATION (the vector surface
                # keeps the inactive lanes of its own destination), never a
                # source the emitted instruction consumes.
                writes.append(name)
                continue
            # A predicate register is the operand whose own dtype says so: the
            # 1-bit vector the pset publishes. Nothing is classified by position.
            if _is_predicate_type(getattr(operand, "type", None)):
                predicates.append(name)
            elif _READ in effect:
                values.append(name)
        attributes = tuple(
            (parameter.name, getattr(stmt.callable, parameter.name, None))
            for parameter in type(stmt.callable).params() if parameter.kind == "attribute")
        return tuple(values), tuple(predicates), tuple(writes), attributes

    def _statement(self, stmt: Evaluate, in_vector: bool) -> TlStatement:
        contract = self._contract(stmt)
        surface = TL_SURFACE.get(contract.name)
        if surface is None:
            if contract.name in EVENT_TL_SURFACE_MISSING:
                # Stated, not worked around: the deployed TL face has no drain
                # intrinsic, so this program's region-exit obligation cannot be
                # emitted here. A barrier or a fence is NOT a substitute - they
                # order a pipe, they do not retire an outstanding async boundary.
                raise EmitRefusal("TL_EVENT_SURFACE_MISSING",
                                  "instruction %r has no TileLang surface: the deployed "
                                  "ascend/language/sync.py states no drain intrinsic, so "
                                  "this obligation cannot be emitted (and pipe_barrier / "
                                  "threadfence are not substitutes for it)"
                                  % (contract.name,), stmt)
            if contract.name in EVENT_TF_OP_MISSING:
                raise EmitRefusal("TL_EVENT_TF_OP_MISSING",
                                  "instruction %r has no TileFoundry op and is not a "
                                  "substitute for an ordering this slice cannot state"
                                  % (contract.name,), stmt)
            raise EmitRefusal("TL_NO_SURFACE",
                              "instruction %r has no TileLang surface in this slice"
                              % (contract.name,), stmt)
        if surface.startswith("event."):
            return self._event_statement(surface, contract, stmt)
        required = contract.required_region
        if required == "outside" and in_vector:
            raise EmitRefusal("TL_INSTRUCTION_REGION",
                              "%r must be issued outside the vector region" % (contract.name,),
                              stmt)
        if required == "simdvf" and not in_vector:
            raise EmitRefusal("TL_INSTRUCTION_REGION",
                              "%r must be issued inside the vector region" % (contract.name,),
                              stmt)
        if bool(contract.core_affinity == "aic"):
            raise EmitRefusal("TL_UNSUPPORTED_CORE",
                              "%r belongs to the Cube core; this slice emits one AIV "
                              "program" % (contract.name,), stmt)
        values, predicates, writes, attributes = self._partition(stmt)
        constants: list[tuple[str, int]] = []
        reads = values
        mask = ""
        if surface == "copy":
            self._check_copy(contract, values, writes, stmt)
        elif surface == "simd.vld":
            if len(values) != 1 or len(writes) != 1:
                raise EmitRefusal("TL_OPERAND_SHAPE",
                                  "T.simd.vld states one UB read and one register write, not "
                                  "%r/%r" % (values, writes), stmt)
            # A register access extent is the LANE COUNT for this element width,
            # not the tile's element count: measured, a 32-element float32 tile
            # still yields access_ptr extent 64, and a 64-element bfloat16 tile
            # yields 128. Spelling it from the buffer would only ever be right
            # for the one case where the two numbers coincide.
            constants.append(("lanes", self._lanes_of(values[0])))
        elif surface == "simd.pset":
            width = dict(attributes).get("elem_width", None)
            if width is None:
                raise EmitRefusal("TL_ATTRIBUTE_UNDECLARED",
                                  "pset states no elem_width, so the emitted predicate "
                                  "register width is unsourced", stmt)
            if len(writes) != 1:
                raise EmitRefusal("TL_OPERAND_SHAPE",
                                  "pset must publish exactly one predicate register, not %r"
                                  % (writes,), stmt)
            constants.append(("elem_width", int(width)))
        elif surface == "simd.vadd":
            if len(values) != 2 or len(writes) != 1:
                raise EmitRefusal("TL_OPERAND_SHAPE",
                                  "T.simd.vadd states two value reads and one register write, "
                                  "not %r/%r" % (values, writes), stmt)
            mask = self._instruction_mask(predicates, stmt)
            # The mask is an operand of its own width (a predicate register), so
            # it is kept out of the VALUE list and recorded separately.
            reads = values + (mask,)
        elif surface == "simd.vsts":
            if len(values) != 1 or len(writes) != 1:
                raise EmitRefusal("TL_OPERAND_SHAPE",
                                  "T.simd.vsts states one register source and one UB "
                                  "destination, not %r/%r" % (values, writes), stmt)
            constants.append(("lanes", self._lanes_of(writes[0])))
            mask = self._region_mask(predicates, stmt)
            reads = values + (mask,)
        result = writes[0] if surface in ("simd.vld", "simd.pset", "simd.vadd") else ""
        if result:
            # The register was declared with its own element dtype in the lowered
            # program, so its vector dtype is that width's lane count - a bool
            # register becomes the predicate dtype the deployed surface states,
            # and no branch here spells out "float32x64" or "boolx256".
            self._register(result, self._vector_dtype_of(result))
        self._vector_surfaces.add(surface)
        statement = TlStatement(
            surface=surface,
            instruction=contract.name,
            result=result,
            reads=reads,
            writes=writes,
            attributes=attributes,
            constants=tuple(constants),
            source_order=self.order,
            in_vector_region=bool(in_vector),
        )
        self.order += 1
        return statement

    def _check_copy(self, contract, reads, writes, stmt) -> None:
        if len(reads) != 1 or len(writes) != 1:
            raise EmitRefusal("TL_OPERAND_SHAPE",
                              "T.copy states one source and one destination, not %r/%r"
                              % (reads, writes), stmt)
        source, destination = reads[0], writes[0]
        if source not in self.buffers or destination not in self.buffers:
            raise EmitRefusal("TL_COPY_OPERAND_UNDECLARED",
                              "a copy operand names no declared tile/parameter", stmt)
        left, right = self.buffers[source], self.buffers[destination]
        if left.extent != right.extent or left.dtype != right.dtype:
            raise EmitRefusal("TL_COPY_SHAPE_MISMATCH",
                              "T.copy would move %r onto %r" % (left, right), stmt)
        if left.scope == right.scope:
            raise EmitRefusal("TL_COPY_DIRECTION",
                              "a copy between two %s roots is not a supported MTE transfer"
                              % (left.scope,), stmt)
        if left.scope == "shared":
            self._shared_read.add(source)
        if right.scope == "global":
            # Keyed by the destination NAME: the ABI boundary names parameters,
            # and this is what proves each declared output is really written.
            self._gm_write.setdefault(destination, []).append(source)

    @staticmethod
    def _instruction_mask(predicates, stmt) -> str:
        """The predicate the instruction itself passes, from its own operand list."""
        if len(predicates) != 1:
            raise EmitRefusal("TL_MASK_UNPROVEN",
                              "the instruction passes %d predicate operand(s); the emitted "
                              "mask must be exactly the one it states" % (len(predicates),), stmt)
        return predicates[0]

    def _region_mask(self, predicates, stmt) -> str:
        """The store's mask: the lowered vsts passes none, so the region's own
        single predicate register is the only provable value. Two or zero is a
        refusal, never a default."""
        if predicates:
            return self._instruction_mask(predicates, stmt)
        published = tuple(sorted(self._predicates))
        if len(published) != 1:
            raise EmitRefusal("TL_MASK_UNPROVEN",
                              "the vector region publishes %d predicate register(s) %r; a "
                              "store mask must be exactly the one it states"
                              % (len(published), published), stmt)
        return published[0]

    def _contract(self, stmt: Evaluate):
        """The ISA contract behind one lowered statement, data OR event.

        Two lookup rules, because the two op families are named differently:
          - a data instruction is a class named ascend_<op>, and the ISA row is
            keyed by <op>;
          - a sync op is a class named FlagSet/FlagWait/Drain whose registered
            schema name is already flag_set/flag_wait/drain. Stripping an
            "ascend_" prefix there finds nothing, so event_contract_of() is used -
            it looks up the schema name and refuses a data instruction that merely
            carries an event-shaped name, which is what keeps ONE event model.
        Guessing by class name would merge the two vocabularies, so the category
        check is the gate, not a fallback.
        """
        from ..ir.isa import event_contract_of

        schema = getattr(type(stmt.callable), "_op_schema", None)
        name = getattr(schema, "name", "") or ""
        if getattr(schema, "category", "") == "sync":
            try:
                return event_contract_of(stmt.callable)
            except KeyError as error:
                raise EmitRefusal("TL_INSTRUCTION_UNREGISTERED", str(error), stmt) from None
        if not name.startswith("ascend_") or name[len("ascend_"):] not in ISA:
            raise EmitRefusal("TL_INSTRUCTION_UNREGISTERED",
                              "instruction %r has no Ascend ISA contract to map onto a "
                              "TileLang surface" % (type(stmt.callable).__name__,), stmt)
        return ISA[name[len("ascend_"):]]

    # --------------------------------------------------------------------- walk
    def _walk(self, node, in_vector: bool) -> list:
        members: list = []
        if isinstance(node, Sequential):
            for item in node.body:
                members.extend(self._walk(item, in_vector))
            return members
        if isinstance(node, MeshScope):
            self._meshes += 1
            if self._meshes > 1:
                raise EmitRefusal("TL_MULTIPLE_MESH_SCOPES",
                                  "this slice emits one program instance", node)
            mesh = getattr(node, "mesh", None)
            levels = tuple(getattr(mesh, "topologies", ()) or ())
            if len(levels) != 1:
                raise EmitRefusal("TL_UNSUPPORTED_MESH",
                                  "the mesh states %d levels; this slice emits one AIV "
                                  "instance" % (len(levels),), node)
            level = levels[0]
            if getattr(level, "name", "") != "aiv":
                raise EmitRefusal("TL_UNSUPPORTED_MESH_LEVEL",
                                  "the mesh level is %r; this slice emits the AIV vector "
                                  "engine" % (getattr(level, "name", level),), node)
            # The MeshScope IS the launch grid. TL's MakePackedAPI refuses a
            # dynamic shared buffer that no kernel scope owns ("variables
            # (buf_dyn_shmem,) are used, but are not passed in as API
            # arguments"), so the UB tiles must be declared inside
            # T.Kernel(<grid>), exactly as the lowered program puts its body
            # inside the mesh scope.
            self._grid_blocks = int(getattr(level, "size", 1) or 1)
            inner = self._walk(node.body, in_vector)
            members.append(TlBlock(kind="kernel", members=tuple(inner),
                                   extent=self._grid_blocks,
                                   induction_var=getattr(node.binding, "name", "bx")))
            return members
        if isinstance(node, LetStmt):
            self._bind_definition(node)
            return members
        if isinstance(node, For):
            members.append(self._loop(node))
            return members
        if isinstance(node, ExecutionScope):
            members.append(self._vector(node))
            return members
        if isinstance(node, Evaluate):
            if in_vector:
                raise EmitRefusal("TL_VECTOR_NESTING",
                                  "an instruction is inside a vector region that this "
                                  "emitter already closed", node)
            members.append(self._statement(node, False))
            return members
        if isinstance(node, (If, While)):
            raise EmitRefusal("TL_UNSUPPORTED_CONTROL",
                              "%s has no TileLang emission contract in this slice"
                              % (type(node).__name__,), node)
        raise EmitRefusal("TL_UNSUPPORTED_STMT",
                          "%s cannot be emitted to TileLang in this slice"
                          % (type(node).__name__,), node)

    def _loop(self, node: For) -> TlBlock:
        start = static_dim_value(node.start)
        stop = static_dim_value(node.stop)
        step = static_dim_value(node.step)
        if start is None or stop is None or step is None:
            raise EmitRefusal("TL_DYNAMIC_LOOP_BOUNDS",
                              "the lowered loop bounds are not static integers, so the "
                              "pipelined extent has no source", node)
        if int(start) != 0:
            raise EmitRefusal("TL_UNSUPPORTED_LOOP_ORIGIN",
                              "T.Pipelined starts at 0; a loop with origin %d has no "
                              "emitted form in this slice" % (int(start),), node)
        if int(step) != 1:
            raise EmitRefusal("TL_UNSUPPORTED_LOOP_STEP",
                              "a pipelined loop with step %d has no emitted form"
                              % (int(step),), node)
        spec = getattr(node, "pipeline", None)
        if spec is None:
            raise EmitRefusal("TL_UNPIPELINED_LOOP",
                              "a lowered loop without the author's PipelineSpec has no "
                              "pipelining contract to emit", node)
        stages = int(getattr(spec, "max_stage", 0) or 0)
        if stages < 1:
            raise EmitRefusal("TL_MISSING_STAGE_BOUND",
                              "the PipelineSpec states no max_stage for this loop", node)
        inner = self._walk(node.body, False)
        return TlBlock(kind="loop", members=tuple(inner), extent=int(stop) - int(start),
                       num_stages=stages,
                       induction_var=getattr(node.induction_var, "name", "i"))

    def _vector(self, node: ExecutionScope) -> TlBlock:
        if node.kind != "ascend.simdvf":
            raise EmitRefusal("TL_UNSUPPORTED_EXECUTION_REGION",
                              "execution region %r has no TileLang surface in this slice"
                              % (node.kind,), node)
        self._predicates = {}
        inner = self._walk_region(node.body, in_vector=True)
        if not inner:
            raise EmitRefusal("TL_EMPTY_VECTOR_REGION",
                              "the vector region computes nothing to emit", node)
        return TlBlock(kind="simdvf", members=tuple(inner))

    def _walk_region(self, node, in_vector: bool) -> list:
        members: list = []
        if isinstance(node, Sequential):
            for item in node.body:
                members.extend(self._walk_region(item, in_vector))
            return members
        if isinstance(node, LetStmt):
            self._bind_definition(node)
            return members
        if isinstance(node, Evaluate):
            statement = self._statement(node, True)
            if statement.surface == "simd.pset":
                self._predicates[statement.result] = statement.surface
            members.append(statement)
            return members
        if isinstance(node, ExecutionScope):
            raise EmitRefusal("TL_NESTED_VECTOR_REGION",
                              "a vector region inside a vector region has no emitted form",
                              node)
        if isinstance(node, For):
            raise EmitRefusal("TL_LOOP_INSIDE_VECTOR_REGION",
                              "a loop inside a whole-VF region would split the unit", node)
        raise EmitRefusal("TL_UNSUPPORTED_VF_STMT",
                          "%s inside a whole-VF region is outside this slice's emission "
                          "domain" % (type(node).__name__,), node)

    # -------------------------------------------------------------------- build
    def build(self) -> EmittedTl:
        self._shared_read = set()
        self._gm_write = {}
        self._predicates = {}
        body = self._walk(self.entry.body, False)
        if not body:
            raise EmitRefusal("TL_EMPTY_PROGRAM", "the lowered entry emits nothing", self.entry)
        if not self._grid_blocks:
            raise EmitRefusal("TL_MISSING_MESH_SCOPE",
                              "the lowered entry has no mesh scope, so no T.Kernel grid owns "
                              "the UB tiles this program allocates", self.entry)
        self._check_boundary()
        block = TlBlock(kind="function", members=tuple(body))
        emitted = EmittedTl(
            entry=self.symbol,
            source="",
            parameters=self.parameters,
            output_count=self.output_count,
            buffers=tuple(self._ordered_buffers()),
            body=block,
            lane_count=self.lane_count,
            element_width=VECTOR_ELEMENT_WIDTH,
            grid_blocks=self._grid_blocks,
            registers=tuple(sorted(self.register_dtypes.items())),
            elements=tuple(sorted(self.dtypes.items())),
            vector_surfaces=tuple(sorted(self._vector_surfaces)),
        )
        return replace(emitted, source=self.render(emitted))

    def _check_boundary(self) -> None:
        outputs = self.parameters[len(self.parameters) - self.output_count:] \
            if self.output_count else []
        if len(outputs) != self.output_count:
            raise EmitRefusal("TL_ABI_SPLIT", "the ABI output count exceeds the parameter list")
        for name in outputs:
            written = self._gm_write.get(name)
            if not written:
                raise EmitRefusal("TL_OUTPUT_NOT_WRITTEN",
                                  "the declared output %r is never written by the program"
                                  % (name,))
            if len(written) != 1:
                raise EmitRefusal("TL_OUTPUT_MULTI_WRITTEN",
                                  "the output %r is written %d times; this slice's ABI binds "
                                  "one UB tile per output" % (name, len(written)))
        for index, name in enumerate(self.parameters[:len(self.parameters) - self.output_count]):
            buffer = self.buffers.get(name)
            if buffer is None:
                raise EmitRefusal("TL_PARAMETER_UNUSED",
                                  "the input parameter %r is never read by the program" % (name,))
            if buffer.scope != "global":
                raise EmitRefusal("TL_PARAMETER_SCOPE",
                                  "the parameter %r is not a GM root" % (name,))

    def _shared_buffers(self) -> tuple:
        return tuple(buffer for buffer in self._ordered_buffers() if buffer.scope == "shared")

    def _extent_text(self, buffer) -> str:
        """How one tile's element count is spelled in the emitted source.

        Every tile in one emitted kernel carries the same element count (that is
        enforced when the buffers are declared, against the vector region's lane
        geometry), so the readable LANES constant names it and is itself emitted
        from that same count.
        """
        return "LANES"

    def _ordered_buffers(self) -> tuple:
        order = []
        for name in self.parameters:
            if name in self.buffers:
                order.append(self.buffers[name])
        for name, buffer in self.buffers.items():
            if name not in order:
                order.append(buffer)
        return tuple(order)

    # ------------------------------------------------------------------ render
    def render(self, emitted: EmittedTl) -> str:
        lines = [
            "import tilelang",
            "import tilelang.ascend.language as T",
            "",
            "LANES = %d" % (emitted.lane_count,),
            "",
            "",
            "@T.prim_func",
            "def %s(%s):" % (emitted.entry, ", ".join(
                "%s: T.Tensor((LANES,), %r)" % (name, self.buffers[name].dtype)
                for name in emitted.parameters)),
        ]
        # Every statement/declaration is emitted by _render_members: the UB tiles
        # belong to the kernel frame, which is where TL's MakePackedAPI expects a
        # dynamic shared buffer to be owned.
        body: list[tuple[int, str]] = []
        self._render_members(emitted.body.members, 1, body)
        pass  # PROBE: vector check disabled
        for depth, text in body:
            lines.append("    " * depth + text)
        lines.append("")
        return "\n".join(lines)

    def _render_members(self, members, depth, out) -> None:
        index = 0
        while index < len(members):
            member = members[index]
            if isinstance(member, TlBlock):
                if member.kind == "kernel":
                    out.append((depth, "with T.Kernel(%d) as %s:"
                                % (member.extent, member.induction_var)))
                    # UB tiles are declared in the kernel frame: the deployed
                    # builder hoists every alloc_shared to the root SBlock, and a
                    # tile used across frames must be bound in an enclosing one.
                    for buffer in self._shared_buffers():
                        out.append((depth + 1,
                                    "%s = T.alloc_shared((%s,), %r)"
                                    % (buffer.name, self._extent_text(buffer),
                                       buffer.dtype)))
                    self._render_members(member.members, depth + 1, out)
                elif member.kind == "loop":
                    out.append((depth, "for %s in T.Pipelined(%d, num_stages=%d):"
                                % (member.induction_var, member.extent, member.num_stages)))
                    self._render_members(member.members, depth + 1, out)
                elif member.kind == "simdvf":
                    out.append((depth, "with T.SimdVF():"))
                    self._render_members(member.members, depth + 1, out)
                else:
                    raise EmitRefusal("TL_BLOCK_KIND_UNRENDERED",
                                      "no renderer for block %r" % (member.kind,), None)
                index += 1
                continue
            out.append((depth, self._render_statement(member)))
            index += 1

    def _render_statement(self, statement: TlStatement) -> str:
        surface = statement.surface
        constants = dict(statement.constants)
        if surface == "copy":
            return "T.copy(%s, %s)" % (statement.reads[0], statement.writes[0])
        if surface == "simd.vld":
            return "%s = T.simd.vld(%s[0])" % (statement.result, statement.reads[0])
        if surface == "simd.pset":
            return "%s = T.simd.pset(%d)" % (statement.result, int(constants["elem_width"]))
        if surface == "simd.vadd":
            return "%s = T.simd.vadd(%s, %s, %s)" % (statement.result, statement.reads[0],
                                                    statement.reads[1], statement.reads[2])
        if surface == "simd.vsts":
            return "T.simd.vsts(%s[0], %s, %s)" % (statement.writes[0], statement.reads[0],
                                                   statement.reads[-1])
        if surface in ("event.set_flag", "event.wait_flag"):
            # One statement in, one statement out: the event count is decided by how
            # many site obligations the plan states, never by a trip count, so this
            # renderer must not loop or duplicate.
            values = dict(statement.constants)
            call = "T.ascend_set_flag" if surface == "event.set_flag" else "T.ascend_wait_flag"
            return "%s(%r, %d)" % (call, statement.hard_event, int(values["flag_id"]))
        raise EmitRefusal("TL_SURFACE_UNRENDERED", "no renderer for %r" % (surface,), None)


def supported_domain(entry) -> tuple:
    """(True, emitted) or (False, code, detail) for the domain this emitter states."""
    try:
        return (True, _Emitter(entry).build())
    except EmitRefusal as refusal:
        return (False, refusal.code, refusal.detail)


def emit_supported_tl(module, plan, completion, storage, target):
    """Emit, parse, and run the deployed source-level Ascend C codegen.

    Three products with three separate identities, never interchangeable:
      - the TileLang source TEXT        -> source_identity / tl_identity
      - the parsed tvm.tirx.PrimFunc    -> read by the reader, digested there
      - the Ascend C source the backend -> ascend_identity
        produces from that parse
    None of them is an authorization: install/export stay refused until the whole
    transaction is certified elsewhere.
    """
    from . import NativeArtifact, PASS_CONFIGS

    entry = _entry_of(module)
    emitted = _Emitter(entry).build()
    path = _write_source(emitted)
    parsed = parse_emitted_source(path, emitted.entry)
    ascend_source = ascend_source_from(parsed, PASS_CONFIGS)
    abi = abi_from_prim_function(entry, entry=emitted.entry)
    if abi.entry != emitted.entry:
        raise EmitRefusal("TL_ABI_ENTRY_MISMATCH",
                          "the ABI names %r but the kernel declares %r"
                          % (abi.entry, emitted.entry))
    if abi.output_count != emitted.output_count:
        raise EmitRefusal("TL_ABI_OUTPUT_COUNT",
                          "the ABI output count is %d but the emitted program writes %d"
                          % (abi.output_count, emitted.output_count))
    return NativeArtifact(
        source=emitted.source,
        abi=abi,
        parsed_prim_func=parsed,
        source_identity=emitted.source_identity,
        plan_identity=getattr(plan, "plan_identity", "") or "",
        tir_identity=_tir_identity(entry),
        tl_identity=emitted.source_identity,
        pass_configs=PASS_CONFIGS,
        target_identity=_target_identity(target, plan),
        toolchain_identity=_toolchain_identity(),
        entry_symbol=emitted.entry,
        emitted=emitted,
        source_path=path,
        ascend_source=ascend_source,
        ascend_identity=_text_identity(ascend_source),
        completion_identity=getattr(completion, "identity", "") or "",
        storage_identity=getattr(storage, "identity", "") or "",
    )


def _entry_of(module):
    function = getattr(module, "entry_function", None)
    entry = function() if callable(function) else None
    if not isinstance(entry, PrimFunction):
        raise EmitRefusal("TL_ENTRY_NOT_PRIM_FUNCTION",
                          "TileLang emission reads the lowered TIR entry PrimFunction", entry)
    return entry


def ascend_source_from(parsed, pass_configs) -> str:
    """Run the deployed SOURCE-LEVEL Ascend C codegen over the parsed kernel.

    This is lower_with_context(enable_host_codegen=False, enable_device_compile=False)
    plus inspect_source(): the real Ascend C text the backend would hand to
    bisheng. It compiles nothing and touches no device. A failure here is a NAMED
    refusal carrying the backend's own message - the emitted program is what is
    wrong, and a bare InternalError must never reach the caller.

    The deployed MakePackedAPI requires every dynamic shared buffer to be owned by
    a kernel frame, which is why a lowered MeshScope must be emitted as
    T.Kernel(...): without a grid scope the backend raises
    "variables (buf_dyn_shmem,) are used, but are not passed in as API arguments".
    """
    from tilelang import tvm
    from tilelang.backend.module import create_backend_context
    from tilelang.backend.target import determine_target
    from tilelang.engine.lower import lower_with_context

    context = create_backend_context(target=determine_target("ascend", return_object=True),
                                     target_host=None, execution_backend="auto")
    config = dict(pass_configs)
    try:
        with tvm.transform.PassContext(opt_level=3, config=config), context.target:
            artifact = lower_with_context(parsed, context, enable_host_codegen=False,
                                          enable_device_compile=False)
        source = artifact.kernel_source
    except Exception as error:  # noqa: BLE001 - the backend failure is the named result
        raise EmitRefusal("TL_ASCEND_SOURCEGEN_FAILED",
                          "the deployed source-level Ascend C codegen refused the emitted "
                          "kernel: %s: %s" % (type(error).__name__, str(error)[:400])) from error
    if not isinstance(source, str) or not source.strip():
        raise EmitRefusal("TL_ASCEND_SOURCE_EMPTY",
                          "the backend returned no Ascend C source text",
                          type(source).__name__)
    return source


def parse_emitted_source(path: str, symbol: str):
    """Parse the emitted SOURCE FILE with the deployed build: no compile, no device."""
    if not os.path.isfile(path):
        raise EmitRefusal("TL_SOURCE_FILE_MISSING",
                          "the deployed builder parses a source file; %r is not one" % (path,))
    directory = os.path.dirname(path)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("tilefoundry_tl_%s" % (_safe(symbol),), path)
    if spec is None or spec.loader is None:
        raise EmitRefusal("TL_SOURCE_UNPARSEABLE", "the build cannot load %r" % (path,), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - a parse failure is the named result
        raise EmitRefusal("TL_SOURCE_PARSE_FAILED",
                          "the deployed build refused to parse the emitted source: %s: %s"
                          % (type(error).__name__, str(error)[:300]), path) from error
    function = getattr(module, symbol, None)
    if function is None:
        raise EmitRefusal("TL_SYMBOL_MISSING", "the parsed module declares no %r" % (symbol,), path)
    return function


def _write_source(emitted: EmittedTl) -> str:
    directory = os.path.join(tempfile.gettempdir(), "tilefoundry_tl")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "tl_%s_%s.py" % (_safe(emitted.entry),
                                                   emitted.source_identity.split(":")[-1][:16]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(emitted.source)
    return path


def _safe(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text)


def _text_identity(text: str) -> str:
    """Identity of one emitted TEXT layer. Each layer gets its own digest; the
    TL source, the parsed PrimFunc and the Ascend C source are three different
    products and never share an identity field."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tir_identity(entry) -> str:
    from ..plan.identity import function_identity

    return function_identity(entry)


def _toolchain_identity() -> str:
    import tilelang

    version = str(getattr(tilelang, "__version__", ""))
    return "sha256:" + hashlib.sha256(("tilelang:" + version).encode("utf-8")).hexdigest()


def _target_identity(target, plan) -> str:
    if target is None:
        raise EmitRefusal("TL_TARGET_MISSING", "emission states its target")
    hook = getattr(target, "target_event_facts", None)
    facts = hook() if callable(hook) else None
    if facts is None:
        raise EmitRefusal("TL_TARGET_FACTS_MISSING",
                          "the target declares no event surface, so an artifact cannot name "
                          "its target identity", type(target).__name__)
    # The profile the target actually publishes, not the target's class name:
    # two target objects with different declared profiles must not share one
    # artifact identity.
    return _text_identity("%s|%s|%s" % (type(target).__name__, facts.profile_id,
                                        getattr(plan, "target_identity", "") or ""))
