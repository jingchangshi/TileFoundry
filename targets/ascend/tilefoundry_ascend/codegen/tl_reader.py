"""read_actual_tl: certify the emitted TL source against the REAL parsed PrimFunc.

WHAT THIS PROVES, AND ONLY THIS. The observation point that needs no compile and
no device is parse-of-emitted-source (docs/handoff/TL_LOWERING_SPEC.md §4-§5).
This reader therefore

  1. recomputes the artifact's source identity from the artifact's own bytes, so
     a hand-edited source text is caught rather than trusted;
  2. hands the emitted SOURCE FILE back to the deployed build's own parser and
     requires the caller's parsed object to have the same structure, so a
     substituted or hand-built PrimFunc cannot stand in for the emitted program;
  3. reads the spec's field list out of the ACTUAL tvm.tirx objects - params,
     buffer_map (name/dtype/shape/scope), block names/annotations/allocations,
     the SimdVF scope attribute, loop var/min/extent/kind, every instruction's op
     name, its operands, tl.region and tl.access_ptr (buffer, rw, extent),
     IntImm/StringImm literals, Bind vector dtypes and UB buffer scopes - and
     compares them item by item with the expectation built from the real lowered
     TIR through the emitter's own structure;
  4. refuses by name on every disagreement and states what it could not observe.

Expected and observed go through ONE canonicaliser, so a mismatch always means
"the parsed program differs from the program the source states", never "two
readers spell the same thing differently". Nothing here uses string containment
as evidence, and a parse failure is a named refusal, not a silent skip.

WHAT IT DOES NOT CLAIM. Post-compile refinement (RewriteDualCopy, SIMD merging,
ND2NZ, layout inference, MergeUBAllocations, RewriteFlagToBuf, ThreadSync,
FlattenBuffer, MakePackedAPI) is not observable without running the compile
pipeline, so it is reported as UNOBSERVED_BY_BUILD with the observation point
named. The source-level Ascend C text IS a real second-layer product, and its
identity is kept separate from the TL source identity and the parsed PrimFunc
identity; none of them stands in for another. VERIFIED here means
source-level correspondence only: it is not a license to install, export or bind.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .emit import EmitRefusal, ascend_source_from, parse_emitted_source
from .vector_model import PREDICATE_DTYPE, vector_model

__all__ = ["ConformanceVerdict", "RefinementManifest", "TlFacts", "read_actual_tl",
           "read_tl_facts", "expected_facts"]

OBSERVATION_POINT = "parse_of_emitted_source"
#: What cannot be observed at this point, stated rather than passed over: the
#: post-compile refinement passes, any compiled binary or device resource
#: figure, and T.Pipelined's num_stages - the deployed parser does not persist
#: the stage count in the For node, so a verdict here cannot certify stages.
UNOBSERVED_BY_BUILD = ("post_compile_refinement", "device_binary", "device_resources",
                       "tl_pipelined_num_stages")

#: The tl.* op names this slice recognises, mapped to the emitter's surface token.
OP_SURFACE = {
    "tl.tileop.ascend_copy": "copy",
    "tl.simd.vld": "simd.vld",
    "tl.simd.pset": "simd.pset",
    "tl.simd.vadd": "simd.vadd",
    "tl.simd.vsts": "simd.vsts",
    # The deployed flag face: T.ascend_set_flag / T.ascend_wait_flag, each lowered
    # to two arguments - the hard-event string and the flag id. Recognised here
    # (not "unsupported") so a candidate that DOES emit events can be compared
    # obligation by obligation; see event_correspondence().
    "tl.ascend_set_flag": "event.set_flag",
    "tl.ascend_wait_flag": "event.wait_flag",
}

#: Surface tokens for the two event instructions, so a caller can name them
#: without repeating the strings.
EVENT_SURFACES = ("event.set_flag", "event.wait_flag")

#: Constructs the deployed grid frame emits that are NOT kernel instructions:
#: T.Kernel(n) lowers to a threaded For plus a thread-index binding. They are
#: recognised (not "unsupported") and compared through the loop/grid facts.
GRID_FRAME_OPS = frozenset({"tl.launch_thread_idx", "tl.launch_thread_bound"})

#: Read/write tags the deployed tl.region/tl.access_ptr calls state (spec §3.3/§3.7).
REGION_READ, REGION_WRITE = 1, 2

#: Build facts, cited to the deployed surface rather than guessed:
#:  - a T.Tensor parameter handle is named "<buffer>_handle" (builder convention);
#:  - alloc_shared defaults to scope "shared.dyn" (ascend/language/allocate.py:15);
#:  - the vector register dtype for an N-lane f32 tile is float32xN (simd.py
#:    _vreg_lanes); the predicate a pset publishes is boolx256 (spec §3.7);
#:  - T.Kernel lowers to a threaded For (kind 4), T.Pipelined to a serial For
#:    (kind 0) whose num_stages is NOT persisted in the parsed IR (spec §3.4/§4).
#: Normalised name for a kernel frame's induction variable (see
#: TlFacts.loops_comparable): the builder owns it, not the emitted source text.
GRID_NAME = "<grid>"

HANDLE_SUFFIX = "_handle"
UB_SCOPE = "shared.dyn"
GM_SCOPE = "global"
KERNEL_FOR_KIND = "4"
PIPELINED_FOR_KIND = "0"


class TlReadRefusal(RuntimeError):
    """A named refusal from the TL read boundary, with the expected/actual pair."""

    def __init__(self, code: str, detail: str, expected=None, actual=None) -> None:
        self.code = code
        self.detail = detail
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class TlInstruction:
    """One kernel instruction in canonical form (expected or observed)."""

    surface: str
    result: str
    result_dtype: str
    operands: tuple
    in_vector_region: bool


@dataclass(frozen=True)
class TlLoop:
    """One loop in canonical form: the grid frame or a pipelined region."""

    induction_var: str
    minimum: int
    extent: int
    kind: str
    is_grid: bool


@dataclass(frozen=True)
class TlBlock:
    """One SBlock in canonical form."""

    name: str
    annotations: tuple
    alloc_buffers: tuple


@dataclass(frozen=True)
class TlFacts:
    """Everything this slice can state about one TL program at the parse point."""

    symbol: str
    params: tuple
    buffer_map: tuple
    blocks: tuple
    loops: tuple
    instructions: tuple
    allocations: tuple
    vector_dtypes: tuple
    unsupported: tuple  # (kind, name[, surface]) machine facts

    def loops_comparable(self) -> tuple:
        """Loops in the form two layers can be compared in.

        A kernel frame's induction variable belongs to the deployed builder:
        "with T.Kernel(n) as cta" keeps the author's mesh-binding name in the
        source text, but the parsed IR names that loop after the builder's own
        block index. Comparing the two names would be a fabricated claim, so the
        grid frame is compared by what IS provable - that a grid loop with this
        origin, extent and For kind exists - and its variable is normalised. A
        pipelined loop keeps its name: the source and the IR do agree there.
        """
        out = []
        for loop in self.loops:
            if loop.is_grid:
                out.append(replace(loop, induction_var=GRID_NAME))
            else:
                out.append(loop)
        return tuple(out)

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(repr(self.as_payload()).encode()).hexdigest()

    def as_payload(self) -> tuple:
        return (self.symbol, self.params, self.buffer_map, self.blocks, self.loops,
                self.instructions, self.allocations, self.vector_dtypes, self.unsupported)


@dataclass(frozen=True)
class RefinementManifest:
    """A registered refinement observation. This slice produces none."""

    identity: str
    steps: tuple
    observed_by: str


@dataclass(frozen=True)
class ConformanceVerdict:
    """What the reader actually read, what it compared, and what it refused."""

    status: str
    observation_points: tuple
    gaps: tuple
    compiled_facts: object | None
    unobserved: tuple
    source_identity: str
    tl_identity: str
    ascend_identity: str
    parsed_identity: str
    tir_identity: str
    plan_identity: str
    completion_identity: str
    storage_identity: str
    target_identity: str
    toolchain_identity: str
    pass_configs: tuple
    parsed_prim_func: object
    expected_facts: TlFacts | None
    observed_facts: TlFacts | None
    refinement_manifest: object
    refusals: tuple

    @property
    def verified(self) -> bool:
        """True only when every required layer was read AND agreed."""
        return self.status == "VERIFIED" and not self.refusals and not self.gaps


# --------------------------------------------------------------- canonical forms
def _copy_operands(source: str, destination: str, extent: int) -> tuple:
    return (("region", source, REGION_READ, int(extent)),
            ("region", destination, REGION_WRITE, int(extent)))


def _vld_operands(buffer: str, extent: int) -> tuple:
    return (("access_ptr", buffer, int(extent), REGION_READ), ("literal", "NORM"))


def _pset_operands(elem_width: int) -> tuple:
    return (("literal", int(elem_width)), ("literal", "PAT_ALL"))


def _vadd_operands(left: str, right: str, mask: str, value: str) -> tuple:
    return (("var", left, value), ("var", right, value), ("var", mask, PREDICATE_DTYPE),
            ("literal", "MODE_ZEROING"))


def _vsts_operands(destination: str, source: str, mask: str, extent: int,
                  value: str) -> tuple:
    return (("access_ptr", destination, int(extent), REGION_WRITE), ("var", source, value),
            ("var", mask, PREDICATE_DTYPE), ("literal", "NORM_B32"))


def _literal(node):
    cls = type(node).__name__
    if cls in ("IntImm", "StringImm", "FloatImm"):
        return node.value
    return None


def _region_of(arg):
    """Canonical ('region', name, rw, extent) of a tl.region call, or None."""
    if type(arg).__name__ != "Call" or str(getattr(arg.op, "name", arg.op)) != "tl.region":
        return None
    buffer = getattr(arg.args[0], "buffer", None)
    if buffer is None:
        return None
    return ("region", buffer.name, _literal(arg.args[1]), _literal(arg.args[2]))


def _access_ptr_of(arg):
    """Canonical ('access_ptr', name, extent, rw) of a tl.access_ptr call, or None."""
    if type(arg).__name__ != "Call" or str(getattr(arg.op, "name", arg.op)) != "tl.access_ptr":
        return None
    buffer = getattr(arg.args[0], "buffer", None)
    if buffer is None:
        return None
    return ("access_ptr", buffer.name, _literal(arg.args[1]), _literal(arg.args[2]))


def _operand_form(arg):
    """One actual operand, canonicalised; None when the shape is unrecognised."""
    if type(arg).__name__ == "Var":
        return ("var", arg.name, str(arg.dtype))
    literal = _literal(arg)
    if literal is not None:
        return ("literal", literal)
    return _region_of(arg) or _access_ptr_of(arg)


# ------------------------------------------------------------- reading the parse
def read_tl_facts(prim_func) -> TlFacts:
    """Read this slice's field list out of the ACTUAL parsed object."""
    if not hasattr(prim_func, "body") or not hasattr(prim_func, "params"):
        raise TlReadRefusal("TL_PARSED_OBJECT_INVALID",
                            "read_actual_tl needs the parser's own PrimFunc object",
                            "tvm.tirx.PrimFunc", type(prim_func).__name__)
    params = tuple(parameter.name for parameter in prim_func.params)
    buffer_map = tuple(sorted(
        (handle.name, buffer.name, str(buffer.dtype), tuple(int(dim) for dim in buffer.shape),
         str(buffer.scope()))
        for handle, buffer in prim_func.buffer_map.items()))

    blocks: list[TlBlock] = []
    loops: list[TlLoop] = []
    instructions: list[TlInstruction] = []
    allocations: list[tuple] = []
    unsupported: list[tuple] = []
    state = {"in_vector": False, "in_grid": False}

    def walk(node):
        cls = type(node).__name__
        if cls == "SBlockRealize":
            walk(node.block)
            return
        if cls == "SBlock":
            inside = state["in_vector"] or str(node.name_hint) == "SIMD_VF"
            alloc = tuple(sorted(
                (buffer.name, str(buffer.dtype), tuple(int(dim) for dim in buffer.shape),
                 str(buffer.scope()))
                for buffer in (node.alloc_buffers or ())))
            blocks.append(TlBlock(
                name=str(node.name_hint),
                annotations=tuple(sorted((str(key), str(getattr(value, "value", value)))
                                         for key, value in dict(node.annotations or {}).items())),
                alloc_buffers=alloc))
            allocations.extend(alloc)
            outer, state["in_vector"] = state["in_vector"], inside
            walk(node.body)
            state["in_vector"] = outer
            return
        if cls == "SeqStmt":
            for index in range(len(node)):
                walk(node[index])
            return
        if cls == "AttrStmt":
            key = str(getattr(node, "attr_key", ""))
            if key == "tl.simdvf_scope":
                walk(node.body)
                return
            unsupported.append(("attr", str(key)))
            walk(node.body)
            return
        if cls == "For":
            minimum, extent = _literal(node.min), _literal(node.extent)
            is_grid = state["in_grid"] is False and str(node.kind) == KERNEL_FOR_KIND
            state["in_grid"] = is_grid
            loops.append(TlLoop(induction_var=node.loop_var.name, minimum=minimum,
                                extent=extent, kind=str(node.kind), is_grid=is_grid))
            walk(node.body)
            state["in_grid"] = False
            return
        if cls == "Bind":
            value = node.value
            if type(value).__name__ == "Call":
                name = str(getattr(value.op, "name", value.op))
                if name in OP_SURFACE:
                    _instruction(value, node.var.name, str(node.var.dtype))
                    return
                if name in GRID_FRAME_OPS:
                    return
                unsupported.append(("bind-op", str(name)))
                return
            unsupported.append(("bind", type(value).__name__))
            return
        if cls == "Evaluate":
            value = node.value
            if type(value).__name__ == "Call":
                name = str(getattr(value.op, "name", value.op))
                if name in OP_SURFACE:
                    _instruction(value, "", "")
                    return
                if name in GRID_FRAME_OPS:
                    return
                unsupported.append(("op", str(name)))
                return
            unsupported.append(("evaluate", type(value).__name__))
            return
        unsupported.append(cls)

    def _instruction(call, result, result_dtype):
        surface = OP_SURFACE[str(getattr(call.op, "name", call.op))]
        operands = []
        for arg in call.args:
            form = _operand_form(arg)
            if form is None:
                unsupported.append(("operand", type(arg).__name__, surface))
                continue
            operands.append(form)
        instructions.append(TlInstruction(
            surface=surface, result=result, result_dtype=result_dtype,
            operands=tuple(operands), in_vector_region=bool(state["in_vector"])))

    walk(prim_func.body)
    return TlFacts(
        symbol=str(dict(prim_func.attrs).get("global_symbol", "")),
        params=params,
        buffer_map=buffer_map,
        blocks=tuple(blocks),
        loops=tuple(loops),
        instructions=tuple(instructions),
        allocations=tuple(sorted(set(allocations))),
        vector_dtypes=tuple(sorted({item.result_dtype for item in instructions
                                    if item.result_dtype})),
        unsupported=tuple(sorted(set(unsupported))),
    )


# ----------------------------------------------------------- the TIR-side view
def expected_facts(emitted) -> TlFacts:
    """What the REAL lowered TIR says the parsed TL program must look like.

    Built from the emitter's structure - which came from the lowered PrimFunction
    itself - through the same canonical forms the reader uses on the parse.
    """
    buffers = {buffer.name: buffer for buffer in emitted.buffers}
    # name -> ELEMENT dtype for every tile and register this kernel names, so a
    # lane count or vector dtype is always looked up through the cited model and
    # never inferred from a tile's element count or spelled as a literal here.
    elements = dict(emitted.elements)

    def model_of(name):
        if name not in elements:
            raise TlReadRefusal("TL_ELEMENT_NAME_UNKNOWN",
                                "no element dtype is recorded for %r, so neither its "
                                "vector lane count nor its register dtype can be stated"
                                % (name,), sorted(elements), name)
        return vector_model(elements[name])

    def lanes_of(name):
        return int(model_of(name).lanes)

    def vec_of(name):
        return model_of(name).vector_dtype

    params = tuple(name + HANDLE_SUFFIX for name in emitted.parameters)
    buffer_map = tuple(sorted(
        (name + HANDLE_SUFFIX, name, buffer.dtype, (buffer.extent,),
         GM_SCOPE if buffer.scope == "global" else UB_SCOPE)
        for name, buffer in buffers.items()
        if name in emitted.parameters))
    shared = tuple(sorted(
        (buffer.name, buffer.dtype, (buffer.extent,), UB_SCOPE)
        for buffer in emitted.buffers if buffer.scope == "shared"))

    instructions = []

    def emit_statement(statement):
        surface = statement.surface
        # A copy moves ELEMENTS; a register access spans LANES. The two coincide
        # only for a 32-bit tile of exactly the declared lane count, which is why
        # an earlier version that used one number for both passed by accident.
        if surface == "copy":
            operands = _copy_operands(statement.reads[0], statement.writes[0],
                                      buffers[statement.reads[0]].extent)
            result, result_dtype = "", ""
        elif surface == "simd.vld":
            operands = _vld_operands(statement.reads[0], lanes_of(statement.reads[0]))
            result = statement.result
            result_dtype = vec_of(result)
        elif surface == "simd.pset":
            width = int(dict(statement.constants).get("elem_width", 0))
            operands = _pset_operands(width)
            result, result_dtype = statement.result, PREDICATE_DTYPE
        elif surface == "simd.vadd":
            # The add's register shape follows the register it writes, which the
            # lowered program already typed - not the tile it came from.
            result = statement.result
            operands = _vadd_operands(statement.reads[0], statement.reads[1],
                                      statement.reads[2], vec_of(result))
            result_dtype = vec_of(result)
        elif surface == "simd.vsts":
            operands = _vsts_operands(
                statement.writes[0], statement.reads[0], statement.reads[1],
                lanes_of(statement.writes[0]), vec_of(statement.reads[0]))
            result, result_dtype = "", ""
        elif surface in EVENT_SURFACES:
            # Both flag calls take exactly (hard_event_string, flag_id), so the
            # expectation is the same two literals, read from the same machine
            # fields the emitter put there. Nothing about a flag is compared
            # through prose.
            values = dict(statement.constants)
            operands = (("literal", statement.hard_event),
                        ("literal", int(values["flag_id"])))
            result, result_dtype = "", ""
        else:
            raise TlReadRefusal("TL_SURFACE_UNREADABLE",
                                "no canonical form for surface %r" % (surface,))
        instructions.append(TlInstruction(surface=surface, result=result,
                                          result_dtype=result_dtype, operands=operands,
                                          in_vector_region=bool(statement.in_vector_region)))

    def walk_block(block):
        for member in block.members:
            if isinstance(member, type(block)):
                walk_block(member)
                continue
            emit_statement(member)

    walk_block(emitted.body)

    loops = []
    grid = emitted.grid_blocks

    def walk_loops(block, inside_grid):
        for member in block.members:
            if not isinstance(member, type(block)):
                continue
            if member.kind == "kernel":
                loops.append(TlLoop(induction_var=member.induction_var, minimum=0,
                                    extent=member.extent, kind=KERNEL_FOR_KIND,
                                    is_grid=True))
                walk_loops(member, True)
            elif member.kind == "loop":
                loops.append(TlLoop(induction_var=member.induction_var, minimum=0,
                                    extent=member.extent, kind=PIPELINED_FOR_KIND,
                                    is_grid=False))
                walk_loops(member, inside_grid)
            else:
                walk_loops(member, inside_grid)

    walk_loops(emitted.body, False)
    blocks = (TlBlock(name="root", annotations=(), alloc_buffers=()),
              TlBlock(name="tilelang_root", annotations=(), alloc_buffers=shared),
              TlBlock(name="SIMD_VF", annotations=(("tl.vf_source_index", "0"),),
                      alloc_buffers=()))
    return TlFacts(
        symbol=emitted.entry,
        params=params,
        buffer_map=buffer_map,
        blocks=blocks,
        loops=tuple(loops),
        instructions=tuple(instructions),
        allocations=shared,
        vector_dtypes=tuple(sorted({item.result_dtype for item in instructions
                                    if item.result_dtype})),
        unsupported=(),
    )


# -------------------------------------------------------------------- the gate
def _compare(expected: TlFacts, observed: TlFacts) -> list:
    """Item-by-item structural diff; every entry names what it found."""
    out: list[tuple] = []
    if expected.symbol != observed.symbol:
        out.append(("TL_SYMBOL_MISMATCH", expected.symbol, observed.symbol))
    if expected.params != observed.params:
        out.append(("TL_PARAM_MISMATCH", expected.params, observed.params))
    if expected.buffer_map != observed.buffer_map:
        out.append(("TL_BUFFER_MAP_MISMATCH", expected.buffer_map, observed.buffer_map))
    if expected.allocations != observed.allocations:
        out.append(("TL_ALLOCATION_MISMATCH", expected.allocations, observed.allocations))
    if len(expected.instructions) != len(observed.instructions):
        out.append(("TL_INSTRUCTION_COUNT", len(expected.instructions),
                    len(observed.instructions)))
    for index, (want, have) in enumerate(zip(expected.instructions, observed.instructions)):
        if want.surface != have.surface:
            out.append(("TL_INSTRUCTION_OP", (index, want.surface), (index, have.surface)))
        if want.in_vector_region != have.in_vector_region:
            out.append(("TL_INSTRUCTION_REGION", (index, want.surface, want.in_vector_region),
                        (index, have.surface, have.in_vector_region)))
        if want.result != have.result:
            out.append(("TL_INSTRUCTION_RESULT", (index, want.result), (index, have.result)))
        if want.result_dtype != have.result_dtype:
            out.append(("TL_INSTRUCTION_RESULT_DTYPE", (index, want.result_dtype),
                        (index, have.result_dtype)))
        if want.operands != have.operands:
            out.append(("TL_INSTRUCTION_OPERANDS", (index, want.operands),
                        (index, have.operands)))
    want_loops = expected.loops_comparable()
    have_loops = observed.loops_comparable()
    if len(want_loops) != len(have_loops):
        out.append(("TL_LOOP_COUNT", len(want_loops), len(have_loops)))
    for index, (want, have) in enumerate(zip(want_loops, have_loops)):
        if want != have:
            out.append(("TL_LOOP_MISMATCH", (index, want), (index, have)))
    want_blocks = {(block.name, block.annotations, block.alloc_buffers)
                   for block in expected.blocks if block.alloc_buffers or block.annotations}
    have_blocks = {(block.name, block.annotations, block.alloc_buffers)
                   for block in observed.blocks if block.alloc_buffers or block.annotations}
    if want_blocks != have_blocks:
        out.append(("TL_BLOCK_MISMATCH", sorted(map(str, want_blocks)),
                    sorted(map(str, have_blocks))))
    if expected.vector_dtypes != observed.vector_dtypes:
        out.append(("TL_VECTOR_DTYPE_MISMATCH", expected.vector_dtypes, observed.vector_dtypes))
    return out


def lane_provenance_gap(emitted):
    """(code, facts) if any element width this kernel uses lacks a cross-checked lane
    model, else None.

    The decision reads VectorModel.cross_check["holds"] - the outcome of comparing
    the deployed lane count against the target's declared (lanes, element_bits)
    geometry and the register size - and NOT the provenance string. Payload is
    machine facts: no sentence can flip a verdict here, and no verdict text can
    drift an identity.
    """
    from .vector_model import vector_model

    widths = {}
    for name, element in emitted.elements:
        try:
            model = vector_model(element)
        except KeyError:
            # An unmodelled width is handled at the emit boundary
            # (TL_UNSUPPORTED_DTYPE); here it is simply not a lane claim.
            continue
        if not model.cross_check["holds"]:
            widths.setdefault(element, model)
    if not widths:
        return None
    facts = sorted(
        [element, model.lanes, model.vector_dtype,
         list(model.cross_check["declared_geometry"]),
         model.cross_check["computed_lanes"], model.cross_check["register_file_bits"],
         model.cross_check["whole_register"]]
        for element, model in widths.items())
    return ("TL_LANE_MODEL_UNSOURCED", facts)



def read_actual_tl(artifact, parsed_prim_func, refinement_manifest=None):
    """Certify the emitted source, its parse and its Ascend C text; or refuse."""
    from . import NativeArtifact, PASS_CONFIGS

    if not isinstance(artifact, NativeArtifact):
        raise TlReadRefusal("TL_ARTIFACT_NOT_ACTUAL",
                            "read_actual_tl needs the emitter's NativeArtifact",
                            "NativeArtifact", type(artifact).__name__)
    if not isinstance(artifact.source, str) or not artifact.source.strip():
        raise TlReadRefusal("TL_SOURCE_MISSING", "the artifact carries no emitted source text")
    emitted = artifact.emitted
    if emitted is None:
        raise TlReadRefusal("TL_EMITTED_STRUCTURE_MISSING",
                            "the artifact carries no emitted structure to compare against")

    refusals: list[tuple] = []
    text_identity = "sha256:" + hashlib.sha256(artifact.source.encode("utf-8")).hexdigest()
    if artifact.source_identity != text_identity:
        refusals.append(("TL_SOURCE_IDENTITY_MISMATCH", artifact.source_identity, text_identity))
    if artifact.tl_identity != text_identity:
        refusals.append(("TL_IDENTITY_MISMATCH", artifact.tl_identity, text_identity))

    expected = expected_facts(emitted)

    # The caller's object must BE a parse of this artifact's own source file.
    path = artifact.source_path or ""
    if not path:
        raise TlReadRefusal("TL_SOURCE_FILE_MISSING",
                            "the deployed parser reads a source file; the artifact names none")
    with open(path, encoding="utf-8") as handle:
        on_disk = handle.read()
    if on_disk != artifact.source:
        # The file the deployed parser reads must BE the artifact's text, or the
        # identity fields describe one program while the parse shows another.
        refusals.append(("TL_SOURCE_FILE_CONTENT_MISMATCH",
                         "sha256:" + hashlib.sha256(artifact.source.encode()).hexdigest()[:24],
                         "sha256:" + hashlib.sha256(on_disk.encode()).hexdigest()[:24]))
    reparsed = parse_emitted_source(path, artifact.entry_symbol or emitted.entry)
    observed = read_tl_facts(parsed_prim_func)
    if observed.as_payload() != read_tl_facts(reparsed).as_payload():
        refusals.append(("TL_PARSED_PRIMFUNC_NOT_FROM_SOURCE",
                         read_tl_facts(reparsed).digest(), observed.digest()))

    # The Ascend C text is a SEPARATE layer: its own identity, never confused
    # with the TL source text or the parsed PrimFunc structure.
    ascend_source = artifact.ascend_source or ""
    if not ascend_source:
        raise TlReadRefusal("TL_ASCEND_SOURCE_MISSING",
                            "the artifact carries no source-level Ascend C text; emit runs "
                            "the deployed AscendC codegen for exactly this evidence")
    ascend_identity = "sha256:" + hashlib.sha256(ascend_source.encode("utf-8")).hexdigest()
    if artifact.ascend_identity != ascend_identity:
        refusals.append(("TL_ASCEND_IDENTITY_MISMATCH", artifact.ascend_identity,
                         ascend_identity))
    if ascend_source == artifact.source:
        refusals.append(("TL_ASCEND_SOURCE_NOT_DISTINCT", "Ascend C differs from TL source",
                         "identical text"))
    fresh_ascend = ascend_source_from(parsed_prim_func, artifact.pass_configs or PASS_CONFIGS)
    if fresh_ascend != ascend_source:
        # Re-running the same deterministic source codegen on the same parsed
        # object must reproduce the artifact's text; a difference means the
        # artifact was not produced from this parse.
        refusals.append(("TL_ASCEND_SOURCE_NOT_FROM_PARSED",
                         "sha256:" + hashlib.sha256(fresh_ascend.encode()).hexdigest()[:24],
                         ascend_identity[:24]))

    configs = tuple(artifact.pass_configs or ())
    if configs != tuple(PASS_CONFIGS):
        refusals.append(("PASS_CONFIGS_MISMATCH", tuple(PASS_CONFIGS), configs))

    refusals.extend(_compare(expected, observed))

    abi = artifact.abi
    if abi is None:
        refusals.append(("ABI_NOT_ACTUAL", "OrderedABI", None))
    else:
        if abi.entry != observed.symbol:
            refusals.append(("ABI_ENTRY_MISMATCH", abi.entry, observed.symbol))
        if tuple(spec.name for spec in abi.parameters) != emitted.parameters:
            refusals.append(("ABI_PARAMETER_ORDER", tuple(spec.name for spec in abi.parameters),
                             emitted.parameters))
        if abi.output_count != emitted.output_count:
            refusals.append(("ABI_OUTPUT_COUNT", emitted.output_count, abi.output_count))

    if observed.unsupported:
        # A construct this slice cannot state is a refusal, never a silent omit.
        refusals.append(("TL_FIELDS_UNOBSERVED", (), observed.unsupported))
    if not artifact.target_identity:
        refusals.append(("ARTIFACT_TARGET_IDENTITY_MISSING", "non-empty", ""))
    if not artifact.tir_identity:
        refusals.append(("ARTIFACT_TIR_IDENTITY_MISSING", "non-empty", ""))

    # LANE-MODEL PROVENANCE. The vector dtype and the register-access extent this
    # kernel uses are chosen by the DEPLOYED toolchain, and codegen/vector_model.py
    # cross-checks that choice against the target's own declared geometry. A width
    # whose cross-check does not HOLD may be EMITTED and may be STRUCTURALLY
    # CONSISTENT - both sides of our comparison agree - yet the lane count itself
    # rests on nothing, so certifying it would be certifying an unproven number.
    # That is a GAP, not a REFUSAL: our emission did nothing wrong, and the
    # verdict ceiling becomes INCOMPLETE rather than VERIFIED.
    #
    # The judgment reads cross_check["holds"] - the outcome of an arithmetic and
    # table comparison - never the provenance STRING, so editing a label cannot
    # move a verdict. To make 16-bit SOURCED, ir/isa.py must declare a matching
    # (lanes, element_bits) pair that fills one whole vector register; that change
    # moves plan identity and is reviewable.
    gaps: list[tuple] = []
    lane_gap = lane_provenance_gap(emitted)
    if lane_gap is not None:
        gaps.append(lane_gap)

    # The binary layer. It is read so ONE verdict can carry both observation
    # points, and it is never used to certify events: with
    # tl.enable_auto_schedule=False the InsertSync pass does not run, so what a
    # flag/sync lowering looks like is simply not observable until a real aibin
    # exists. That is a GAP (an incomplete certification), not a defect in our
    # emission, and it is not papered over by pointing at a section name -
    # a present .CrossCoreSync proves our events were emitted no more than an
    # absent one proves they were not.
    compiled = getattr(artifact, "compiled", None)
    blockage = getattr(artifact, "compile_blockage", None)
    points = [OBSERVATION_POINT]
    if compiled is not None:
        points.append(compiled.observation_point)
        from .ascend_binary import CompileRefusal, check_compiled_abi

        try:
            check_compiled_abi(compiled, len(emitted.parameters), emitted.output_count)
        except CompileRefusal as error:
            refusals.append((error.code, error.expected, error.actual))
    elif blockage is not None:
        points.append("blocked_env")
        gaps.append(("TL_EVENT_LOWERING_UNOBSERVED",
                     "a compiled kernel.aibin to inspect sync/event lowering",
                     "BLOCKED_ENV: %s" % (str(getattr(blockage, "detail", ""))[:200],)))
    else:
        gaps.append(("TL_EVENT_LOWERING_UNOBSERVED",
                     "a compiled kernel.aibin: flag/sync lowering is not observable at the "
                     "parse point, and no section name may stand in for it",
                     "the compile site was never run on this artifact"))

    # Three honest states. REFUSED: our emitted/parsed structure disagrees with
    # the lowered TIR - a defect in this slice's own product. INCOMPLETE: the
    # structure agrees but a required layer was never observed, so nothing may
    # be certified from this verdict. VERIFIED: every layer read agrees.
    if refusals:
        status = "REFUSED"
    elif gaps:
        status = "INCOMPLETE"
    else:
        status = "VERIFIED"
    return ConformanceVerdict(
        status=status,
        observation_points=tuple(points),
        gaps=tuple(gaps),
        compiled_facts=compiled,
        unobserved=UNOBSERVED_BY_BUILD,
        source_identity=artifact.source_identity,
        tl_identity=artifact.tl_identity,
        ascend_identity=artifact.ascend_identity,
        parsed_identity=observed.digest(),
        tir_identity=artifact.tir_identity,
        plan_identity=artifact.plan_identity,
        completion_identity=artifact.completion_identity,
        storage_identity=artifact.storage_identity,
        target_identity=artifact.target_identity,
        toolchain_identity=artifact.toolchain_identity,
        pass_configs=configs,
        parsed_prim_func=parsed_prim_func,
        expected_facts=None if refusals else expected,
        observed_facts=observed,
        refinement_manifest=refinement_manifest,
        refusals=tuple(refusals),
    )
