"""The TL artifact layer: emitted source, its parse, and the read-back proof.

These tests are the backend's own evidence chain, and they assert on the FIELDS
OF THE PARSED tvm.tirx.PrimFunc, never on whether some text contains a
substring. The program under test is the real lowered TIR of the two author
fixtures (checked_program, analyze, finalize), emitted by codegen/emit.py,
written to disk, parsed by the deployed build, and read back by
codegen/tl_reader.py.

Nothing here compiles a binary, imports torch_npu, opens a device node or
launches a kernel: parse-of-emitted-source is the observation point, and the
source-level Ascend C text comes from the backend's own device-codegen-without-
compile path.

The mutation tests tamper with what was EMITTED and require the reader to notice
that the parsed program no longer matches the lowered TIR. A named refusal is
the pass condition; a green verdict over tampered bytes would be the failure.
"""
from __future__ import annotations

import hashlib
import pathlib
import tempfile
import traceback
from dataclasses import replace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.schedule import finalize

from tilefoundry_ascend.codegen import (PASS_CONFIGS, NativeArtifact,
                                        compile_supported_artifact,
                                        emit_supported_tl)
from tilefoundry_ascend.codegen.emit import (
    EmitRefusal,
    _Emitter,
    _write_source,
    ascend_source_from,
    parse_emitted_source,
    supported_domain,
)
from tilefoundry_ascend.codegen.ascend_binary import (
    BISHENG_INSTALL_DIRS,
    CCE_MACHINE,
    REQUIRED_PRODUCT_FILES,
    BlockedToolchain,
    CompiledArtifactFacts,
    discover_bisheng,
)
from tilefoundry_ascend.codegen.tl_reader import (
    GRID_FRAME_OPS,
    OBSERVATION_POINT,
    TlReadRefusal,
    expected_facts,
    read_actual_tl,
    read_tl_facts,
)

INDEPENDENT_UB = ("targets/ascend/tests_native_mainline/fixtures/authors/"
                  "independent_ub.py:IndependentUB")
MINIMAL_AIV = ("targets/ascend/tests_native_mainline/fixtures/authors/"
                "minimal_aiv.py:MinimalAivPipeline")
COPY_ONLY = ("targets/ascend/tests_native_mainline/fixtures/authors/"
             "copy_only.py:CopyOnly")
LANES = 64


def _lowered(path):
    program = checked_program(path, None, command="test tl emit read")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    return module, module.entry_function()


def _artifact(path):
    module, entry = _lowered(path)
    return emit_supported_tl(module, None, None, None, entry.target)


def _surface_order(facts):
    return tuple(item.surface for item in facts.instructions)


def _source_lines(text, drop=None, replace_pairs=()):
    out = []
    for line in text.splitlines(keepends=True):
        if drop is not None and line.strip() == drop:
            continue
        for old, new in replace_pairs:
            if old in line:
                line = line.replace(old, new)
        out.append(line)
    return "".join(out)


def _mutated(artifact, text):
    """Re-emit, re-write and re-parse a deliberately changed source text."""
    emitted = replace(artifact.emitted, source=text)
    path = _write_source(emitted)
    parsed = parse_emitted_source(path, artifact.entry_symbol)
    # The identity fields stay the ones the PRISTINE emission recorded: that is
    # the point. The reader must catch the file drifting from its own record.
    return replace(artifact, source=text, source_path=path, parsed_prim_func=parsed), parsed


def _codes(verdict):
    return set(entry[0] for entry in verdict.refusals)


# ------------------------------------------------------------------ 1. the parse
def test_parsed_primfunc_fields_correspond_to_the_lowered_tir():
    artifact = _artifact(INDEPENDENT_UB)
    parsed = artifact.parsed_prim_func
    facts = read_tl_facts(parsed)

    # The parse is the deployed build's own object, not a TileFoundry type.
    assert type(parsed).__module__ == "tvm.tirx.function"
    assert type(parsed).__name__ == "PrimFunc"
    assert facts.symbol == "vadd"
    assert facts.params == ("x_handle", "out_handle", "out_1_handle")
    # buffer_map read off the actual buffers: name, dtype, shape, scope.
    assert facts.buffer_map == (
        ("out_1_handle", "out_1", "float32", (LANES,), "global"),
        ("out_handle", "out", "float32", (LANES,), "global"),
        ("x_handle", "x", "float32", (LANES,), "global"),
    )
    # The UB tiles are real allocations in the shared.dyn scope.
    assert facts.allocations == (
        ("loaded", "float32", (LANES,), "shared.dyn"),
        ("stored", "float32", (LANES,), "shared.dyn"),
    )
    # Vector dtypes are read, not assumed: 64 f32 lanes, one 256-bit predicate.
    assert facts.vector_dtypes == ("boolx256", "float32x64")
    # Instruction identity, order and region membership come from the IR tree.
    assert _surface_order(facts) == (
        "copy", "simd.vld", "simd.vld", "simd.pset", "simd.vadd", "simd.vsts",
        "copy", "copy",
    )
    assert tuple(item.in_vector_region for item in facts.instructions) == (
        False, True, True, True, True, True, False, False)
    first_vld = facts.instructions[1]
    assert first_vld.operands == (("access_ptr", "loaded", LANES, 1), ("literal", "NORM"))
    pset = facts.instructions[3]
    assert pset.operands == (("literal", 32), ("literal", "PAT_ALL"))
    vadd = facts.instructions[4]
    assert vadd.operands == (
        ("var", "left", "float32x64"), ("var", "right", "float32x64"),
        ("var", "mask", "boolx256"), ("literal", "MODE_ZEROING"))
    vsts = facts.instructions[5]
    assert vsts.operands == (
        ("access_ptr", "stored", LANES, 2), ("var", "acc", "float32x64"),
        ("var", "mask", "boolx256"), ("literal", "NORM_B32"))
    assert facts.instructions[0].operands == (
        ("region", "x", 1, LANES), ("region", "loaded", 2, LANES))
    # The mask this program states is emitted; nothing defaulted, nothing unread.
    assert facts.unsupported == ()
    assert all(item.surface not in GRID_FRAME_OPS for item in facts.instructions)


def test_grid_frame_and_pipelined_loop_are_read_from_the_parsed_ir():
    artifact = _artifact(MINIMAL_AIV)
    facts = read_tl_facts(artifact.parsed_prim_func)
    # MeshScope becomes T.Kernel(1): a real grid frame, read back with its kind.
    assert tuple((loop.induction_var, loop.minimum, loop.extent, loop.kind, loop.is_grid)
                 for loop in facts.loops) == (("bx", 0, 1, "4", True), ("k", 0, 4, "0", False))
    assert facts.symbol == "vsum"
    assert _surface_order(facts) == (
        "copy", "copy", "simd.vld", "simd.vld", "simd.pset", "simd.vadd", "simd.vsts",
        "copy")
    # Per-iteration load writes tile; the store writes base: read, not assumed.
    assert facts.instructions[1].operands == (("region", "x", 1, LANES),
                                              ("region", "tile", 2, LANES))
    # index 5 is the vadd; the store is the seventh statement.
    assert facts.instructions[6].surface == "simd.vsts"
    assert facts.instructions[6].operands[0] == ("access_ptr", "base", LANES, 2)
    blocks = tuple((block.name, block.annotations) for block in facts.blocks)
    assert ("SIMD_VF", (("tl.vf_source_index", "0"),)) in blocks


# ------------------------------------------------------------------ 2. the verdict
def test_read_actual_tl_certifies_both_supported_programs():
    for path in (INDEPENDENT_UB, MINIMAL_AIV):
        artifact = _artifact(path)
        verdict = read_actual_tl(artifact, artifact.parsed_prim_func)
        # Source-level structure agrees...
        assert verdict.refusals == (), verdict.refusals
        # ...but the event layer is NOT observable without a compiled binary, so
        # a source-only verdict is INCOMPLETE and never claims certification.
        assert verdict.status == "INCOMPLETE"
        assert verdict.verified is False
        assert verdict.gaps and verdict.gaps[0][0] == "TL_EVENT_LOWERING_UNOBSERVED"
        assert verdict.observation_points == (OBSERVATION_POINT,)
        assert verdict.compiled_facts is None
        # Expected and observed share one canonical structure, both read here.
        assert verdict.expected_facts.instructions == verdict.observed_facts.instructions
        assert verdict.expected_facts.buffer_map == verdict.observed_facts.buffer_map
        assert verdict.expected_facts.allocations == verdict.observed_facts.allocations
        assert (verdict.expected_facts.loops_comparable()
                == verdict.observed_facts.loops_comparable())
        # The grid frame really is read as a grid loop, not as a pipelined one.
        assert verdict.observed_facts.loops[0].is_grid is True
        assert verdict.expected_facts.loops[0].is_grid is True
        # The identity fields are separate layers and never impersonate another.
        assert verdict.source_identity == verdict.tl_identity
        assert verdict.ascend_identity != verdict.tl_identity
        assert verdict.parsed_identity != verdict.tl_identity
        assert verdict.tir_identity and verdict.target_identity and verdict.toolchain_identity


def test_ascend_c_text_is_a_real_distinct_layer_of_the_same_kernel():
    artifact = _artifact(INDEPENDENT_UB)
    assert artifact.ascend_source.strip()
    assert artifact.ascend_source != artifact.source
    recomputed = "sha256:" + hashlib.sha256(
        artifact.ascend_source.encode("utf-8")).hexdigest()
    assert artifact.ascend_identity == recomputed
    # Re-running the deployed source-level codegen reproduces it exactly:
    # deterministic, and bound to THIS parse rather than a recorded string.
    again = ascend_source_from(artifact.parsed_prim_func, PASS_CONFIGS)
    assert again == artifact.ascend_source
    assert "simd_inst::vlds_norm" in again
    assert "asc_create_mask_b32(PAT_ALL)" in again


# ------------------------------------------------------------------ 3. mutations
def test_mutated_operand_or_constant_is_refused():
    artifact = _artifact(INDEPENDENT_UB)
    # Swap the two trailing copy destinations: the parsed program no longer
    # writes the outputs the lowered TIR writes.
    swapped = _source_lines(artifact.source, replace_pairs=(
        ("T.copy(stored, out)", "T.copy(stored, out_1)"),
        ("T.copy(loaded, out_1)", "T.copy(loaded, out)"),
    ))
    assert swapped != artifact.source
    tampered, parsed = _mutated(artifact, swapped)
    verdict = read_actual_tl(tampered, parsed)
    assert verdict.status == "REFUSED"
    assert verdict.compiled_facts is None
    assert "TL_INSTRUCTION_OPERANDS" in _codes(verdict)

    # A predicate element width other than the one the ISA contract states.
    narrowed = _source_lines(artifact.source, replace_pairs=(
        ("T.simd.pset(32)", "T.simd.pset(16)",),))
    tampered, parsed = _mutated(artifact, narrowed)
    verdict = read_actual_tl(tampered, parsed)
    assert verdict.status == "REFUSED"
    assert "TL_INSTRUCTION_OPERANDS" in _codes(verdict)


def test_missing_instruction_or_wrong_slot_is_refused():
    artifact = _artifact(INDEPENDENT_UB)
    # Drop one whole MTE transfer: the kernel is shorter than the program.
    dropped = _source_lines(artifact.source,
                            drop="T.copy(loaded, out_1)")
    assert "T.copy(loaded, out_1)" not in dropped
    tampered, parsed = _mutated(artifact, dropped)
    verdict = read_actual_tl(tampered, parsed)
    assert verdict.status == "REFUSED"
    assert "TL_INSTRUCTION_COUNT" in _codes(verdict)

    # Feed the wrong UB tile to an output: it parses (same shape) but the slot
    # no longer carries the value the lowered program stores there.
    wrong_slot = _source_lines(artifact.source, replace_pairs=(
        ("T.copy(stored, out)", "T.copy(loaded, out)"),))
    assert wrong_slot != artifact.source
    tampered, parsed = _mutated(artifact, wrong_slot)
    verdict = read_actual_tl(tampered, parsed)
    assert verdict.status == "REFUSED"
    assert "TL_INSTRUCTION_OPERANDS" in _codes(verdict)

    # A UB tile half the scheduled width is refused by the deployed parser
    # itself. It must surface as a NAMED refusal, never be skipped as unsupported.
    halved = _source_lines(artifact.source, replace_pairs=(
        ("loaded = T.alloc_shared((LANES,), 'float32')",
         "loaded = T.alloc_shared((32,), 'float32')"),))
    assert halved != artifact.source
    with pytest.raises(EmitRefusal) as caught:
        _mutated(artifact, halved)
    assert caught.value.code == "TL_SOURCE_PARSE_FAILED"


def test_pass_configs_and_source_text_are_bound_to_identity():
    artifact = _artifact(MINIMAL_AIV)
    # One key missing.
    short = replace(artifact, pass_configs=(("tl.enable_auto_schedule", False),))
    verdict = read_actual_tl(short, short.parsed_prim_func)
    assert verdict.status == "REFUSED"
    assert "PASS_CONFIGS_MISMATCH" in _codes(verdict)
    # One key with the wrong value.
    flipped = replace(artifact, pass_configs=(
        ("tl.enable_auto_schedule", True), ("tl.disable_shared_memory_reuse", True)))
    verdict = read_actual_tl(flipped, flipped.parsed_prim_func)
    assert "PASS_CONFIGS_MISMATCH" in _codes(verdict)
    # A wrong value must be REFUSED outright, not merely reported incomplete.
    assert verdict.status == "REFUSED"

    # One character of the source text: its identity must move, and the file
    # must no longer match the record it is certified against.
    edited = _source_lines(artifact.source, replace_pairs=(("LANES = 64", "LANES = 65"),))
    tampered, parsed = _mutated(artifact, edited)
    verdict = read_actual_tl(tampered, parsed)
    assert verdict.status == "REFUSED"
    assert "TL_SOURCE_IDENTITY_MISMATCH" in _codes(verdict)
    assert "TL_IDENTITY_MISMATCH" in _codes(verdict)

    # A parsed object from another program cannot stand in for this one.
    foreign = _artifact(INDEPENDENT_UB).parsed_prim_func
    verdict = read_actual_tl(artifact, foreign)
    assert verdict.status == "REFUSED"
    assert "TL_PARSED_PRIMFUNC_NOT_FROM_SOURCE" in _codes(verdict)


def test_unsupported_instruction_is_a_named_refusal_not_a_dropped_statement():
    _, entry = _lowered(INDEPENDENT_UB)
    # Replace the initial GM->UB transfer with an instruction that does have a
    # real ISA contract but no TileLang surface in this slice: nothing may be
    # skipped to make a kernel look emittable.
    from tilefoundry_ascend.ir.ops import Vmul

    def substitute(node):
        if isinstance(node, Evaluate) and type(node.callable).__name__ == "GmToUb":
            return replace(node, callable=Vmul())
        body = getattr(node, "body", None)
        if body is None:
            return node
        if isinstance(body, (tuple, list)):
            fresh = tuple(substitute(item) for item in body)
            if any(new is not old for new, old in zip(fresh, body)):
                return replace(node, body=fresh)
            return node
        rewritten = substitute(body)
        if rewritten is body:
            return node
        return replace(node, body=rewritten)

    mutated = replace(entry, body=substitute(entry.body))
    ok, code, detail = supported_domain(mutated)
    assert ok is False
    assert code == "TL_NO_SURFACE"
    assert "vmul" in detail


def test_emit_and_read_never_touch_a_device_or_a_binary():
    import sys

    _artifact(INDEPENDENT_UB)
    _artifact(MINIMAL_AIV)
    assert "torch_npu" not in sys.modules
    # The frozen two-key contract is what the compile site must consume.
    assert PASS_CONFIGS == (("tl.enable_auto_schedule", False),
                            ("tl.disable_shared_memory_reuse", True))
    artifact = _artifact(MINIMAL_AIV)
    assert isinstance(artifact, NativeArtifact)
    assert artifact.plan_identity == ""          # no plan wired: recorded, not invented
    assert artifact.completion_identity == ""
    assert artifact.storage_identity == ""


# ===================================================================== task-41
# The compile site runs in a SEPARATE INTERPRETER, and the reason is measured
# rather than assumed: this suite's process already holds OR-Tools, and with
# CANN's environment sourced "import tilelang" passively loads torch_npu, after
# which a CP-SAT solve segfaults (rc=139) inside protobuf. The child imports
# TileFoundry first and never solves, so it sits on the safe side of that order
# either way. Nothing here skips: when the toolchain is not resolvable, the
# BLOCKED_ENV shape IS the assertion.
import json
import os as _os
import re as _re
import subprocess as _subprocess
import tempfile as _tempfile
from pathlib import Path as _Path

_PY = __import__("sys").executable
_REPO = _Path(__file__).resolve().parents[3]
_CHILD = _Path("/tmp/tf_tl_compile_site_child.py")

_CHILD_SOURCE = """
import json, os, sys, tempfile
sys.path.insert(0, "src"); sys.path.insert(0, "targets/ascend")
from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.codegen import PASS_CONFIGS
from tilefoundry_ascend.codegen.emit import _Emitter, _write_source, parse_emitted_source
from tilefoundry_ascend.codegen.ascend_binary import (BlockedToolchain, CompileRefusal,
                                                      compile_ascend_artifact,
                                                      check_compiled_abi,
                                                      CompiledArtifactFacts)

source, cache, mode = sys.argv[1], sys.argv[2], sys.argv[3]
program = checked_program(source, None, command="compile site")
analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
module = finalize(analyzed, authored=program.authored)
emitted = _Emitter(module.entry_function()).build()
parsed = parse_emitted_source(_write_source(emitted), emitted.entry)

if mode == "badkey":
    configs = tuple(dict(PASS_CONFIGS, **{"tl.not_a_real_key": True}).items())
elif mode == "flip":
    configs = (("tl.enable_auto_schedule", True), ("tl.disable_shared_memory_reuse", True))
else:
    configs = PASS_CONFIGS

if mode == "jit":
    import tilelang
    try:
        kernel = tilelang.compile(parsed, target="ascend", out_idx=None,
                                  pass_configs=dict(configs))
    except Exception as error:
        print("RESULT" + json.dumps({"nojit": "%s: %s" % (type(error).__name__,
                                                          str(error)[:160])}))
    else:
        from tilefoundry_ascend.codegen.ascend_binary import CompiledArtifactFacts
        print("RESULT" + json.dumps({
            "jit": True,
            "n_regs": getattr(kernel, "n_regs", "ABSENT"),
            "n_spills": getattr(kernel, "n_spills", "ABSENT"),
            "resource_usage": repr(getattr(kernel, "resource_usage", "ABSENT"))[:80],
            "resource_fields": sorted(
                name for name in CompiledArtifactFacts.__dataclass_fields__
                if "reg" in name or "spill" in name or "resource" in name)}))
    raise SystemExit(0)

if mode == "cachetwice":
    # Two compiles of ONE artifact into two DIFFERENT empty cache directories. The
    # second is served from the build cache, which is exactly the path that used to
    # produce a false TL_COMPILE_PRODUCTS_INCOMPLETE: a hit writes nothing into the
    # directory it was given, and that emptiness was read as "no products".
    import os as _os2
    import tempfile as _tempfile2

    results = []
    for _attempt in range(2):
        directory = _tempfile2.mkdtemp()
        try:
            fresh = compile_ascend_artifact(parsed, configs,
                                            cache_directory=directory)
        except CompileRefusal as refusal:
            results.append({"raised": refusal.code, "detail": refusal.detail[:300]})
            break
        if isinstance(fresh, BlockedToolchain):
            results.append({"blocked": fresh.code, "searched": list(fresh.searched)})
            break
        products = all(
            _os2.path.isfile(_os2.path.join(fresh.kernel_directory, name))
            for name in ("manifest.json", "device_kernel.asc"))
        results.append({"compiled": True,
                        "semantic_identity": fresh.semantic_identity,
                        "ascend_source_sha256": fresh.ascend_source_sha256,
                        "products_present": products})
    payload = {"attempts": results}
    if len(results) == 2 and all(entry.get("compiled") for entry in results):
        payload["compiled"] = True
        payload["cache_hit_compiled"] = True
        payload["cache_hit_same_semantics"] = (
            results[0]["semantic_identity"] == results[1]["semantic_identity"])
        payload["cache_hit_products_present"] = all(
            entry["products_present"] for entry in results)
    elif results and results[0].get("blocked"):
        payload["blocked"] = results[0]["blocked"]
        payload["searched"] = results[0]["searched"]
    print("RESULT" + json.dumps(payload))
    raise SystemExit(0)

try:
    facts = compile_ascend_artifact(parsed, configs, cache_directory=cache)
except CompileRefusal as refusal:
    print("RESULT" + json.dumps({"raised": refusal.code,
                                 "detail": refusal.detail[:400]}))
    raise SystemExit(0)
if isinstance(facts, BlockedToolchain):
    print("RESULT" + json.dumps({"blocked": facts.code,
                                 "searched": list(facts.searched),
                                 "observed": facts.observed[:200]}))
    raise SystemExit(0)
check_compiled_abi(facts, len(emitted.parameters), emitted.output_count)
print("RESULT" + json.dumps({
    "compiled": True, "elf_type": facts.elf_type, "machine": facts.machine,
    "argument_size": facts.argument_size, "parameter_count": len(emitted.parameters),
    "manifest_files": [[n, d, z] for n, d, z in facts.manifest_files],
    "binary_sha256": facts.binary_sha256, "host_sha256": facts.host_library_sha256,
    "ascend_source_sha256": facts.ascend_source_sha256,
    "kernel_type_symbols": list(facts.kernel_type_symbols),
    "cross_core_sync_sections": list(facts.cross_core_sync_sections),
    "control_nodes": list(facts.control_nodes),
    "torch_npu_imported": facts.torch_npu_imported,
    "kernel_directory": facts.kernel_directory,
    "binary_directory": facts.binary_directory,
    "semantic_identity": facts.semantic_identity,
    "build_identity": facts.build_identity,
    "cache_directory": facts.cache_directory,
    "resource_fields": sorted(name for name in CompiledArtifactFacts.__dataclass_fields__
                              if "reg" in name or "spill" in name or "resource" in name),
}))
"""


#: The documented toolchain route, DISCOVERED rather than assumed. This mirrors
#: exactly what CANN's own set_env.sh exports: bisheng's bin directory on PATH,
#: ASCEND_HOME_PATH set to the toolkit root that CONTAINS that bin directory, and
#: deliberately NOT BISHENG_HOME (set_env.sh does not set it, and the deployed
#: backend looks at PATH too - measured, a PATH-only setup compiles fine).
#: Measured, ASCEND_HOME_PATH must be the tool root: bisheng itself errors with
#: "LLVM ERROR: PlugIn Err: can not find ASCEND_HOME_PATH" if only PATH is set,
#: so discovery alone is not enough and the site says so by name.
def _toolchain_environment():
    for directory in BISHENG_INSTALL_DIRS:
        executable = _os.path.join(directory, "bisheng")
        if _os.path.isfile(executable):
            return {"PATH": _os.environ.get("PATH", "") + ":" + directory,
                    "ASCEND_HOME_PATH": _os.path.dirname(directory),
                    "BISHENG_HOME": ""}
    return None


TOOLCHAIN = _toolchain_environment()


def _site_run(mode, extra_env=None):
    _CHILD.write_text(_CHILD_SOURCE, encoding="utf-8")
    env = dict(_os.environ)
    env["PYTHONPATH"] = "%s/src:%s/targets/ascend" % (_REPO, _REPO)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["BISHENG_HOME"] = ""
    if mode != "unreachable" and TOOLCHAIN:
        env.update(TOOLCHAIN)
    if mode == "unreachable":
        env["PATH"] = "/usr/bin:/bin"
        env["ASCEND_HOME_PATH"] = ""
    if extra_env:
        env.update(extra_env)
    cache = _tempfile.mkdtemp(prefix="tf-site-")
    result = _subprocess.run([_PY, str(_CHILD), MINIMAL_AIV, cache, mode],
                             cwd=str(_REPO), env=env, capture_output=True, text=True,
                             timeout=900)
    line = next((entry[len("RESULT"):] for entry in result.stdout.splitlines()
                 if entry.startswith("RESULT")), None)
    assert line is not None, ("the compile worker printed no result\n"
                              + (result.stdout or "")[-1200:]
                              + (result.stderr or "")[-1200:])
    return json.loads(line)


def _toolchain_available(outcome):
    return "blocked" not in outcome


def test_compile_site_products_or_blocked_env():
    """Either five real products whose manifest matches the bytes, or BLOCKED_ENV.

    Both branches are asserted, so an absent toolchain surfaces as a named
    external gap instead of a quietly shortened check.
    """
    if TOOLCHAIN is None:
        # No toolchain on this machine: BLOCKED_ENV is the required outcome, and
        # the refusal must name what was searched so the reader can fix it.
        outcome = _site_run("same")
        assert outcome["blocked"] == "TL_BISHENG_NOT_DISCOVERABLE", outcome
        assert outcome["searched"], "the refusal must say what was searched"
        assert "bisheng" in " ".join(outcome["searched"]), outcome["searched"]
        return
    outcome = _site_run("same")
    assert outcome.get("compiled") is True, outcome
    assert outcome.get("raised") is None, outcome
    names = [entry[0] for entry in outcome["manifest_files"]]
    for required in ("device_kernel.asc", "executable.so", "params.json"):
        assert required in names, (required, names)
    assert _Path(outcome["binary_directory"], "kernel.aibin").is_file()
    # The manifest is verified against the BYTES, never trusted as a list.
    for name, digest, size in outcome["manifest_files"]:
        raw = _Path(outcome["kernel_directory"], name).read_bytes()
        assert len(raw) == size, (name, size, len(raw))
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == "sha256:" + digest, name
    assert outcome["machine"] == CCE_MACHINE, hex(outcome["machine"])
    assert outcome["elf_type"] in ("EXEC", "REL")
    assert outcome["argument_size"] == outcome["parameter_count"] * 8
    assert any(name.startswith(".Kernel_Type_")
               for name in outcome["kernel_type_symbols"])
    # Whatever the environment opened passively, no COMPUTE node was taken.
    assert not [node for node in outcome["control_nodes"]
                if _re.match(r"^/dev/davinci\d+$", node)], outcome["control_nodes"]


def test_pass_configs_reach_the_binary_and_build_identity_is_not_reusable():
    """The two keys change the PRODUCTS; a rebuild moves only the build identity.

    Without this, PASS_CONFIGS would be decoration: both keys are consumed by the
    build, and the difference has to be visible in the bytes it wrote.
    """
    first = _site_run("same")
    if TOOLCHAIN is None:
        blocked = _site_run("unreachable")
        assert blocked["blocked"] == "TL_BISHENG_NOT_DISCOVERABLE", blocked
        return
    first = _site_run("same")
    flipped = _site_run("flip")
    assert _toolchain_available(flipped) and flipped.get("raised") is None, flipped
    assert flipped["ascend_source_sha256"] != first["ascend_source_sha256"]
    assert flipped["binary_sha256"] != first["binary_sha256"]
    assert flipped["semantic_identity"] != first["semantic_identity"]

    again = _site_run("same")
    assert again["semantic_identity"] == first["semantic_identity"], (
        "the same config rebuilt into a new cache must be semantically identical")
    assert again["binary_sha256"] == first["binary_sha256"]
    assert again["cache_directory"] != first["cache_directory"]
    assert again["build_identity"] != first["build_identity"], (
        "build identity must carry THIS run, so an old build fingerprint cannot be"
        " reused")


def test_unknown_pass_config_key_is_refused_with_the_builds_own_words():
    """A key the deployed build does not know is a named refusal, verbatim."""
    outcome = _site_run("badkey")
    if TOOLCHAIN is None:
        assert outcome["blocked"] == "TL_BISHENG_NOT_DISCOVERABLE", outcome
        return
    assert outcome["raised"] == "TL_PASS_CONFIG_REJECTED", outcome
    assert "tl.not_a_real_key" in outcome["detail"], outcome["detail"]


def test_resource_facts_are_never_stated_and_never_invented():
    """JITKernel's n_regs/n_spills/resource_usage stay UNKNOWN, so we state none.

    The facts record carries no resource field at all (asserted from the real
    dataclass), and the live kernel object is inspected to confirm the numbers we
    refuse to report really are absent in this build.
    """
    outcome = _site_run("jit")
    assert outcome["resource_fields"] == [], outcome["resource_fields"]
    if "nojit" in outcome or TOOLCHAIN is None:
        # No toolchain here: the compile never produced a kernel, so the only
        # assertion that still binds is that our record states nothing.
        return
    assert outcome["n_regs"] in (None, "ABSENT"), outcome
    assert outcome["n_spills"] in (None, "ABSENT"), outcome
    assert outcome["resource_usage"] in ("{}", "ABSENT", "None"), outcome


def test_blocked_env_shape_when_toolchain_unreachable():
    """No bisheng anywhere reachable: BLOCKED_ENV, never a downgraded pass.

    Run with a PATH stripped to /usr/bin:/bin, so the deployed backend cannot
    resolve bisheng even on a machine where it IS installed. The site must return
    the blocked shape and name the installed path plus the fix, not pretend a
    source-only artifact is complete and not report an external gap as our defect.
    """
    outcome = _site_run("unreachable")
    assert outcome["blocked"] == "TL_BISHENG_NOT_DISCOVERABLE", outcome
    # "searched" names DIRECTORIES the site looked in, so the reader can see the
    # install that exists but is unreachable and put its bin on PATH.
    assert outcome["searched"], outcome["searched"]
    assert all(path.endswith("bin") or path == "/no/such/dir"
               for path in outcome["searched"]), outcome["searched"]
    if TOOLCHAIN is not None:
        installed = [_os.path.dirname(TOOLCHAIN["ASCEND_HOME_PATH"] + "/bin")]
        assert TOOLCHAIN["ASCEND_HOME_PATH"] + "/bin" in " ".join(outcome["searched"]), \
            (outcome["searched"], TOOLCHAIN)
    assert "PATH" in outcome["observed"], outcome["observed"]
    assert "bisheng" in outcome["observed"] or "BISHENG_HOME" in outcome["observed"], \
        outcome["observed"]


def test_device_node_rule_passive_control_versus_active_compute():
    """The boundary as a decision, not a comment: control nodes are recorded,
    compute nodes refuse, and the classifier is what makes that difference.

    (a) Passive. With CANN's set_env.sh sourced, plain "import tilelang" opens
        /dev/davinci_manager and /dev/hisi_hdc during its own initialisation.
        Refusing those would fail every compile in that environment while
        protecting nothing, and nothing here launched a kernel, so they are
        recorded on the facts instead. Evidence and the fd/maps method are in
        docs/handoff/TL_SOURCEGEN_RUN.md.
    (b) Active. /dev/davinci<digits> means the accelerator was taken. That is
        refused regardless of who opened it: our program must never hold one.
    """
    from tilefoundry_ascend.codegen import ascend_binary as site

    assert site.COMPUTE_NODE.match("/dev/davinci0")
    assert site.COMPUTE_NODE.match("/dev/davinci31")
    assert not site.COMPUTE_NODE.match("/dev/davinci_manager")
    assert not site.COMPUTE_NODE.match("/dev/davinci")
    assert not site.COMPUTE_NODE.match("/dev/hisi_hdc")

    # In THIS process nothing was taken: no compile ran here and the compile
    # tests above all used their own interpreter.
    _imported, _loaded, _taken = site._device_state()
    assert _taken == (), _taken

    original = site._device_state
    try:
        site._device_state = lambda: (True, ("/dev/davinci_manager",), ("/dev/davinci0",))
        with pytest.raises(site.CompileRefusal) as caught:
            site._guard_device_nodes()
        assert caught.value.code == "TL_COMPILE_OPENED_DEVICE"
        assert "/dev/davinci0" in caught.value.detail

        # Control nodes alone: tolerated, and REPORTED rather than dropped.
        site._device_state = lambda: (True, ("/dev/davinci_manager", "/dev/hisi_hdc"), ())
        imported, nodes = site._guard_device_nodes()
        assert nodes == ("/dev/davinci_manager", "/dev/hisi_hdc"), nodes
        assert imported is True
    finally:
        site._device_state = original


# ============================================================ lane provenance
# The 16-bit surface can be EMITTED but not CERTIFIED. vector_model derives
# provenance from a cross-check against the target's own declared geometry, so
# these tests assert on the cross-check DATA, not on a label anyone could flip.

def test_thirty_two_bit_lane_is_sourced_by_the_cross_check_not_by_default():
    """32-bit is SOURCED because the equation holds, and the equation is checkable.

    2048//32 = 64 must appear as a (lane_count, elem_width_bits) pair the TARGET
    declares, and 64*32 must be exactly one vector register. If either half
    stopped being true the row would be UNSOURCED, so this is not a default trust
    in fp32; it is the arithmetic and the table agreeing.
    """
    from tilefoundry_ascend.codegen.vector_model import (declared_geometry,
                                                        vector_lanes, vector_model,
                                                        REGISTER_FILE_BITS)

    model = vector_model("float32")
    check = model.cross_check
    assert check["declared_for_this_width"] is True, check
    assert check["whole_register"] is True, check
    assert (64, 32) in declared_geometry(), declared_geometry()
    assert vector_lanes(32) == 64 and 64 * 32 == REGISTER_FILE_BITS
    assert check["holds"] is True and model.provenance == "SOURCED"


def test_sixteen_bit_is_unsourced_because_the_target_declares_no_such_geometry():
    """16-bit UNSOURCED is a MISSING DECLARATION, not our distrust of fp16.

    The deployed build gives 128 lanes for a 16-bit element. The target declares
    only (64, 32): no 128-lane geometry for a 16-bit element exists, and the
    declared 64 lanes at 16 bits is half a register. So the deployed number has
    nothing to be checked AGAINST - which is why the verdict ceiling is
    INCOMPLETE, and why adding the pair to ir/isa.py (a reviewable,
    identity-moving act) is the only way to close it.
    """
    from tilefoundry_ascend.codegen.vector_model import (declared_geometry,
                                                         vector_model,
                                                         REGISTER_FILE_BITS)

    geometry = declared_geometry()
    for name in ("bfloat16", "float16"):
        model = vector_model(name)
        check = model.cross_check
        assert model.lanes == 128, model.lanes
        assert check["declared_for_this_width"] is False, check
        assert (model.lanes, model.element_bits) not in geometry, geometry
        # The declared 64 lanes at this width is only half a register.
        assert 64 * model.element_bits != REGISTER_FILE_BITS
        assert check["holds"] is False and model.provenance == "UNSOURCED"


def test_lane_provenance_reads_the_cross_check_not_a_prose_label():
    """The reader's gap must follow cross_check["holds"], and only that.

    Three probes, so the judgement cannot be a hardcoded dtype list:
      (a) 32-bit: no lane gap at all.
      (b) A 16-bit element recorded against a target that DOES declare the
          matching geometry: the gap disappears. Nothing about the string
          "bfloat16" is special-cased - the equation is what moved.
      (c) Relabelling provenance to "SOURCED" while the cross-check still fails:
          the gap STAYS. So the label is not the input to the decision.
    """
    from tilefoundry_ascend.codegen.tl_reader import lane_provenance_gap

    artifact = _artifact(MINIMAL_AIV)
    assert lane_provenance_gap(artifact.emitted) is None

    # (b) pretend the target declares the 16-bit geometry; gap must clear.
    import tilefoundry_ascend.codegen.vector_model as vm

    emitted_16 = replace(artifact.emitted,
                          elements=(("x", "bfloat16"), ("out", "bfloat16"),
                                    ("out_1", "bfloat16"), ("tile", "bfloat16")))
    assert lane_provenance_gap(emitted_16)[0] == "TL_LANE_MODEL_UNSOURCED"
    original = vm.declared_geometry
    vm.declared_geometry = lambda: ((64, 32), (128, 16))
    try:
        assert (lane_provenance_gap(emitted_16) is None), (
            "a matching declared geometry must clear the gap")
    finally:
        vm.declared_geometry = original

    # (c) Relabelling alone must NOT move the decision. Proven by substituting a
    # model whose label claims SOURCED while its cross_check still fails, and the
    # converse: a label claiming UNSOURCED while the check holds. If the reader
    # consulted the string, one of these two flips the verdict.
    class FakeModel:
        def __init__(self, provenance, holds):
            self.element_bits = 16
            self.lanes = 128
            self.vector_dtype = "bfloat16x128"
            self.provenance = provenance
            self.cross_check = {"holds": holds, "basis": "GEOMETRY",
                                "declared_geometry": [(64, 32)],
                                "declared_for_this_width": False,
                                "whole_register": False,
                                "computed_lanes": 128,
                                "register_file_bits": 2048, "reason": "probe"}

    real = vm.vector_model
    vm.vector_model = lambda name: FakeModel("SOURCED", False)
    try:
        assert lane_provenance_gap(emitted_16)[0] == "TL_LANE_MODEL_UNSOURCED", \
            "a SOURCED label over a failing cross-check must not clear the gap"
    finally:
        vm.vector_model = real
    vm.vector_model = lambda name: FakeModel("UNSOURCED", True)
    try:
        assert lane_provenance_gap(emitted_16) is None, \
            "a holding cross-check must clear the gap whatever the label says"
    finally:
        vm.vector_model = real


#: A correctly authored bf16 program, kept inside this test so the claim
#: stays checkable and does not depend on anyone's scratch directory.
_AUTHORED_16BIT = """from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64

@module(entry="bf16full", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Bf16Full:
    @func
    def bf16full(x: Tensor[(LANES,), "bf16"]) -> Tensor[(LANES,), "bf16"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            base = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "smem"])
            staged = tf.schedule((x, base), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                carr = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "carrier"])
                v = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                m32 = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                m16 = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                carr2 = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "carrier"])
                ld = tf.schedule((staged, carr), op=T.ascend_vld(dist="UNPK_B16"))
                p32 = tf.schedule((m32,), op=T.ascend_pset(elem_width=32))
                p16 = tf.schedule((m16,), op=T.ascend_pset16(elem_width=16))
                conv = tf.schedule((ld, p16, v), op=T.ascend_vcvt(direction="forward", part=0, elem_width=16))
                added = tf.schedule((conv, conv, p32, acc), op=T.ascend_vadd())
                back = tf.schedule((added, p16, carr2), op=T.ascend_vcvt(direction="reverse", part=0, elem_width=16))
                stored = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "smem"])
                carried = tf.schedule((back, stored), op=T.ascend_vsts(pack="PK_B32"))
            out = tf.empty(Tensor[(LANES,), "bf16", (LANES,), "gmem"])
            return tf.schedule((carried, out), op=T.ascend_ub_to_gm())"""


def test_unsourced_lane_model_caps_the_verdict_at_incomplete():
    """The ceiling rule, proven on a program whose structure fully agrees.

    A 32-bit artifact is used deliberately: both sides of the comparison read the
    SAME cited model, so removing the target's declared (64, 32) geometry changes
    nothing about the numbers expected or parsed - only the PROVENANCE of them.
    The structure therefore stays consistent, and the verdict still must drop from
    a possible VERIFIED to INCOMPLETE with verified False. That isolates the rule:
    an unsourced lane model cannot be certified even when nothing disagrees.

    Reversing the patch restores the possibility of VERIFIED, so the test also
    proves the gap is driven by the declaration and not by a dtype name list.
    """
    from tilefoundry_ascend.codegen.tl_reader import lane_provenance_gap
    import tilefoundry_ascend.codegen.vector_model as vm

    artifact = _artifact(MINIMAL_AIV)
    assert lane_provenance_gap(artifact.emitted) is None

    original = vm.declared_geometry
    vm.declared_geometry = lambda: ((128, 16),)      # (64, 32) no longer declared
    try:
        gap = lane_provenance_gap(artifact.emitted)
        assert gap is not None and gap[0] == "TL_LANE_MODEL_UNSOURCED", gap
        verdict = read_actual_tl(artifact, artifact.parsed_prim_func)
        # Structure agrees: no refusal was produced by the patch.
        assert "TL_LANE_MODEL_UNSOURCED" not in _codes(verdict), verdict.refusals
        assert verdict.status == "INCOMPLETE", (verdict.status, verdict.refusals)
        assert verdict.verified is False
        assert "TL_LANE_MODEL_UNSOURCED" in [gap_code for gap_code in
                                             (g[0] for g in verdict.gaps)]
    finally:
        vm.declared_geometry = original

    restored = read_actual_tl(artifact, artifact.parsed_prim_func)
    assert "TL_LANE_MODEL_UNSOURCED" not in [g[0] for g in restored.gaps], restored.gaps


def test_copy_only_is_outside_the_vector_support_domain_by_name():
    """Reporting the support domain honestly: a pure copy is refused, not emitted.

    copy_only declares GM->UB->GM with no vector instruction, and this emitter
    states no TileLang form for a program with no vector region. It must be
    refused by name rather than emitted as a degenerate kernel - and it must NOT
    appear in any list of certified programs.
    """
    from tilefoundry_ascend.codegen.emit import supported_domain

    _module, entry = _lowered(COPY_ONLY)
    ok, code, detail = supported_domain(entry)
    assert ok is False, code
    assert code == "TL_NO_VECTOR_REGION", (code, detail)


def test_new_model_field_participates_in_identity_by_default():
    """Identity is declared-whitelist, never a prose blacklist.

    A field the records did not have before must move target identity WITHOUT
    anyone editing an allow-list - that is the fail-safe direction. And the prose
    set is one explicit collection: the test enumerates every declared leaf of
    the real fact payload and asserts that anything outside PROSE_PATHS
    participates. An earlier version of this module hashed the digest first and
    pruned afterwards, which pruned nothing; that ordering bug is what this test
    exists to keep dead.
    """
    from tilefoundry_ascend.completion.model import ASCEND_EVENT_FACTS
    from tilefoundry_ascend.completion.plan import (PROSE_PATHS, _declared_leaves,
                                                    _identity_payload,
                                                    _target_identity)

    class Plan:
        target_identity = "sha256:plan-target"

    payload = {
        "plan_target": Plan.target_identity,
        "event_facts": {
            "profile_id": ASCEND_EVENT_FACTS.profile_id,
            "core_domains": sorted(str(d) for d in ASCEND_EVENT_FACTS.core_domains),
            "cross_core_mechanism": ASCEND_EVENT_FACTS.cross_core_mechanism,
            "source": ASCEND_EVENT_FACTS.source,
            "pools": [p.as_dict() for p in ASCEND_EVENT_FACTS.pools],
            "hard_event_pools": [p.as_dict() for p in
                                 ASCEND_EVENT_FACTS.hard_event_pools],
        },
    }
    leaves = _declared_leaves(payload, "")
    assert len(leaves) > 40, len(leaves)
    prose = [path for path in leaves if path in PROSE_PATHS]
    assert prose, "PROSE_PATHS must match real declared paths, not a guess"
    assert all(path.endswith("source") for path in prose), prose

    pruned = _identity_payload(payload)
    carried = json.dumps(pruned, sort_keys=True)
    for path in {leaf for leaf in leaves} - set(PROSE_PATHS):
        field = path.split(".")[-1].replace("[]", "")
        assert field in carried, path

    # A brand-new declared field must participate without touching any allow-list.
    grown = [p.as_dict() for p in ASCEND_EVENT_FACTS.pools]
    grown[0]["brand_new_capacity_like_field"] = 7
    assert "brand_new_capacity_like_field" in json.dumps(
        _identity_payload({**payload, "event_facts": {**payload["event_facts"],
                                                     "pools": grown}}),
        sort_keys=True)
    assert _identity_payload({**payload}) != _identity_payload(
        {**payload, "event_facts": {**payload["event_facts"], "pools": grown}})

    # And it must actually reach the digest (the prune-before-digest ordering).
    widened = replace(ASCEND_EVENT_FACTS, pools=tuple(
        replace(p, capacity=p.capacity + 7) if index == 0 else p
        for index, p in enumerate(ASCEND_EVENT_FACTS.pools)))
    assert _target_identity(Plan(), widened) != _target_identity(Plan(),
                                                                ASCEND_EVENT_FACTS)


def test_sixteen_bit_reaches_the_emitter_and_is_refused_before_any_lane_claim():
    """Where 16-bit actually stops, measured at both layers.

    My earlier claim here was WRONG and this test replaces it. I first wrote "no
    16-bit author program reaches either layer", generalising from a measurement
    of my own making: I took minimal_aiv and substituted "f32" -> "bf16" in its
    text. That dies in TileFoundry's HIR verifier (ir/hir/schedule.py:342
    _infer_schedule) because ascend_vld without dist=UNPK_B16 genuinely does not
    accept a bf16 load - so the substitution produced an INVALID program, and its
    failure said nothing about support. The lead's R10/R12 note had it right: a
    properly authored bf16 program FINALIZES and does reach the emitter.

    Two measured facts, not one:
      (1) the text-substituted fixture is not a 16-bit program at all. It is
          refused in the FRONT END, before this slice runs, with a VerifyError
          that carries no .code - so no support claim may be read from it;
      (2) a correctly authored bf16 program (carrier tile, ascend_vld
          dist=UNPK_B16, ascend_pset16, ascend_vcvt forward/reverse, ascend_vsts
          pack=PK_B32) FINALIZES and DOES reach this emitter's support-domain
          gate, where it is refused on TL_BUFFER_SCOPE_MISMATCH: the carrier
          storage class is not admitted. It therefore never reaches a lane claim.

    Neither fact closes anything. G1's three gaps (carrier load transport,
    f16<->f32 conversion, f16 output storage) and the 16-bit packing hardware
    evidence are still missing, and this slice's own gap (TL_LANE_MODEL_UNSOURCED,
    INCOMPLETE ceiling) is the gate for whatever arrives after the carrier scope
    refusal is lifted in S4. If that happens, the assertion below on
    TL_BUFFER_SCOPE_MISMATCH must be re-measured, not deleted.
    """
    import pathlib as _pathlib
    import tempfile as _tempfile

    text = _pathlib.Path("targets/ascend/tests_native_mainline/fixtures/authors/"
                         "minimal_aiv.py").read_text(encoding="utf-8")
    assert '"f32"' in text
    substituted = _pathlib.Path(_tempfile.mkdtemp()) / "bf16_textsub.py"
    substituted.write_text(text.replace('"f32"', '"bf16"'), encoding="utf-8")
    with pytest.raises(Exception) as caught:
        checked_program("%s:MinimalAivPipeline" % substituted, None,
                        command="test 16-bit text substitution")
    error = caught.value
    frames = traceback.extract_tb(error.__traceback__)
    site = [frame for frame in frames if frame.filename.endswith("hir/schedule.py")]
    assert site, [frame.filename for frame in frames][-3:]
    assert type(error).__name__ == "VerifyError", type(error).__name__
    assert not hasattr(error, "code"), (
        "the DSL refused this program, not this slice: a named refusal here would "
        "mean the emitter ever saw it")
    assert "ascend_vld" in str(error), str(error)[:300]

    authored = _pathlib.Path(_tempfile.mkdtemp()) / "bf16_authored.py"
    authored.write_text(_AUTHORED_16BIT, encoding="utf-8")
    program = checked_program("%s:Bf16Full" % authored, None,
                              command="test 16-bit authored reachability")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    entry = module.entry_function()
    params = [(parameter.name, str(parameter.type.dtype.name))
              for parameter in entry.params]
    assert params == [("x", "bf16"), ("out", "bf16")], params

    ok, code, detail = supported_domain(entry)
    assert ok is False, ("16-bit support is not claimed closed", code)
    # Measured refusal: the carrier storage class, BEFORE any dtype or lane claim
    # is made. If the emitter starts admitting carriers this must be re-measured.
    assert code == "TL_BUFFER_SCOPE_MISMATCH", (code, str(detail)[:200])
    assert "carrier" in str(detail), str(detail)[:200]


def test_every_identity_producer_is_audited_for_prose_participation():
    """One audit of ALL identity bytes in this slice, not just the newest one.

    The rule is stated per producer, and each direction is measured, so a future
    field cannot silently start or stop carrying identity by accident:

      _target_identity           prose EXCLUDED via PROSE_PATHS only
      completion_identity        no prose field exists; a value edit moves it
      semantic_identity          digests/machine/arg-size/file table only
      build_identity             the toolchain sentence IS the point here, and it
                                 is deliberately NOT semantic
      source/tl/ascend identity   a hash of the program's own bytes: rewording the
                                 PROGRAM necessarily moves them, which is not
                                 documentation drift and must not be "fixed"

    PROSE_PATHS is kept narrow and explicit: a blacklist entry is a claim that a
    path is documentation, so the set is asserted against the real declared leaf
    paths rather than guessed at.
    """
    import dataclasses
    import json

    from tilefoundry_ascend.codegen.ascend_binary import CompiledArtifactFacts
    from tilefoundry_ascend.completion.model import ASCEND_EVENT_FACTS
    from tilefoundry_ascend.completion.plan import (PROSE_PATHS, _declared_leaves,
                                                    _identity_payload, _target_identity)

    class Plan:
        target_identity = "sha256:plan-target"

    payload = {
        "plan_target": Plan.target_identity,
        "event_facts": {
            "profile_id": ASCEND_EVENT_FACTS.profile_id,
            "core_domains": sorted(str(d) for d in ASCEND_EVENT_FACTS.core_domains),
            "cross_core_mechanism": ASCEND_EVENT_FACTS.cross_core_mechanism,
            "source": ASCEND_EVENT_FACTS.source,
            "pools": [p.as_dict() for p in ASCEND_EVENT_FACTS.pools],
            "hard_event_pools": [p.as_dict() for p in ASCEND_EVENT_FACTS.hard_event_pools],
        },
    }
    leaves = set(_declared_leaves(payload, ""))
    # Every declared prose path must exist; a stale entry would otherwise hide
    # that the field it named was renamed or removed.
    assert PROSE_PATHS <= leaves, sorted(PROSE_PATHS - leaves)
    assert all(path.endswith("source") for path in PROSE_PATHS), sorted(PROSE_PATHS)

    # semantic vs build identity, on a real record.
    fields = {f.name for f in dataclasses.fields(CompiledArtifactFacts)}
    facts = CompiledArtifactFacts(**{
        name: ([] if name.endswith(("files", "symbols", "sections",
                                    "cross_core_sync_sections", "control_nodes"))
                else 0 if name in ("machine", "argument_size", "compile_seconds",
                                   "compiled_at") else "x")
        for name in fields})
    semantic = facts.semantic_identity
    build = facts.build_identity
    assert semantic != build
    moved_toolchain = replace(facts, toolchain_string="clang version 99.0.0")
    assert moved_toolchain.semantic_identity == semantic, (
        "toolchain prose must not be part of the SEMANTIC identity")
    assert moved_toolchain.build_identity != build, (
        "the build identity exists to record this build's toolchain")
    moved_size = replace(facts, argument_size=17)
    assert moved_size.semantic_identity != semantic, (
        "an ABI number must be part of the semantic identity")

    # Every CONTENT digest must participate in the SEMANTIC identity. The kernel
    # manifest does NOT list the aibin (it lives in a separate immutable
    # directory), so an earlier version of semantic_identity left the device binary
    # out entirely - mutation-caught: a build whose host-side manifest matched but
    # whose aibin differed would have claimed to be the same program.
    for digest_field in ("binary_sha256", "host_library_sha256",
                         "ascend_source_sha256"):
        changed = replace(facts, **{digest_field: "different"})
        assert changed.semantic_identity != semantic, (
            "%s did not participate in the semantic identity" % digest_field)

    # The two identities are separated by RUN data, not by content. compiled_at is
    # this build's stamp, so it moves the BUILD identity and must never move the
    # SEMANTIC one - that is what lets a cold rebuild of the same program be
    # recognisably the same program under a different fingerprint. Content digests
    # appear in BOTH, because they are both the program and the bytes just written.
    stamp = replace(facts, compiled_at=facts.compiled_at + 1.0)
    assert stamp.build_identity != build, (
        "the build identity must carry this run's stamp")
    assert stamp.semantic_identity == semantic, (
        "a rebuild stamp is not a new program")

    # No resource figure is carried anywhere, so none can be misreported.
    assert not [name for name in fields
                if "reg" in name or "spill" in name or "resource" in name], sorted(fields)


# ============================================================ event emission
# task-40 step2 + task-48 item 4. The emitter is the only place these claims are
# testable: S0 materialises CONCRETE flag attributes (the lead's ruling), so the
# properties that matter are that a flag id is bounds-checked against a pool the
# TARGET declares, that nothing is expanded per trip, and that a face the deployed
# backend does not have is refused by name instead of substituted.

def _entry_with_events(path, specs):
    """The real lowered entry, with flag statements appended to its body.

    The statements are real Op instances carrying the attributes the flag IR
    declares (channel_name/kind/slot/generation/versions/flag_id/producer_pipe/
    consumer_pipe), so the emitter sees what apply will hand it, not a stand-in
    structure of my own invention.
    """
    from dataclasses import replace as _replace

    from tilefoundry.ir.tir.flag import FlagSet, FlagWait
    from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Sequential

    _module, entry = _lowered(path)
    ops = {"set": FlagSet, "wait": FlagWait}
    statements = []
    for surface, overrides in specs:
        attrs = dict(channel_name="ub:tile", kind="ready", slot=0, generation=0,
                     versions=1, flag_id=0, producer_pipe="mte2",
                     consumer_pipe="vector")
        attrs.update(overrides)
        statements.append(Evaluate(callable=ops[surface](**attrs), args=()))

    def append(node):
        if isinstance(node, MeshScope):
            return _replace(node, body=append(node.body))
        if isinstance(node, Sequential):
            return _replace(node, body=tuple(node.body) + tuple(statements))
        return node

    return _replace(entry, body=append(entry.body))


def _emit_code(path, statement):
    """The named refusal an event-shaped statement earns, or None if emitted."""
    try:
        return None, _Emitter(_entry_with_events(path, statement)).build()
    except EmitRefusal as refusal:
        return refusal.code, refusal


def test_flag_set_and_wait_emit_the_deployeds_own_call_form():
    """A flag statement round-trips through the parser as two literal arguments.

    Measured shape: T.ascend_set_flag("<HARD_EVENT>", <flag_id>) - the deployed
    surface takes exactly (hard_event string, event id), and the parsed IR is read
    back through the same canonical form the data instructions use, so the flag
    that WILL be emitted is compared, not a description of it.
    """
    emitted = _Emitter(_entry_with_events(MINIMAL_AIV, [("set", {"flag_id": 3}),
                                                        ("wait", {"flag_id": 4})])).build()
    events = [s for s in emitted.body.statements() if s.surface.startswith("event.")]
    assert [s.surface for s in events] == ["event.set_flag", "event.wait_flag"]
    assert [s.hard_event for s in events] == ["MTE2_V", "MTE2_V"]
    assert [dict(s.constants)["flag_id"] for s in events] == [3, 4]

    parsed = parse_emitted_source(_write_source(emitted), emitted.entry)
    facts = read_tl_facts(parsed)
    flag_facts = [i for i in facts.instructions if i.surface.startswith("event.")]
    assert [(i.surface, i.operands) for i in flag_facts] == [
        ("event.set_flag", (("literal", "MTE2_V"), ("literal", 3))),
        ("event.wait_flag", (("literal", "MTE2_V"), ("literal", 4)))], flag_facts
    # Recognised, not unknown: an unlisted op would land in unsupported, and an
    # empty unsupported is the difference between "compared" and "ignored".
    assert facts.unsupported == (), facts.unsupported
    expected = expected_facts(emitted)
    assert [i for i in expected.instructions if i.surface.startswith("event.")] == \
        flag_facts


def test_hard_event_strings_come_from_the_builds_own_pipe_table():
    """The flag's hard-event token is the deployed build's, not a guess.

    Authority: GetResourcePipeName (tilelang/src/ascend/transform/ascend_pipe.h:
    75-91) names MTE1/MTE2/MTE3/M/V/FIX/S, and GetHardEvent
    (insert_sync.cc:751-765) joins the pair and applies four aliases. The Python
    surface checks only that hard_event is a str - it validates nothing about the
    name - so correctness here can only come from copying the C++ table, and a
    wrong string would reach pto.set_flag's arguments (codegen_pto.cc:4847-4853)
    as a flag the hardware has no such event for.
    """
    from tilefoundry_ascend.codegen.emit import hard_event_name

    assert hard_event_name("mte2", "vector") == "MTE2_V"
    assert hard_event_name("mte2", "mte3") == "MTE2_MTE3"
    assert hard_event_name("vector", "mte3") == "V_MTE3"
    assert hard_event_name("scalar", "mte3") == "S_MTE3"
    # the four aliases GetHardEvent applies
    assert hard_event_name("mte2", "cube") == "MTE2_MTE1"
    assert hard_event_name("cube", "mte2") == "MTE1_MTE2"
    assert hard_event_name("mte3", "cube") == "MTE3_MTE1"
    assert hard_event_name("cube", "mte3") == "MTE1_MTE3"
    # authored spellings normalise through the target's own canonical_pipe
    assert hard_event_name("MTE2", "V") == "MTE2_V"
    with pytest.raises(EmitRefusal) as caught:
        hard_event_name("mte2", "not-a-pipe")
    assert caught.value.code == "TL_EVENT_PIPE_TOKEN_UNKNOWN"


def test_flag_id_beyond_the_declared_pool_is_refused():
    """A flag id outside the target's declared intra-core pool cannot be emitted.

    The pool is the finite hardware budget (measured: AIV intra-core declares 8).
    Accepting an id past it would emit a synchronisation the hardware does not
    have, which fails at run time rather than here - so it is refused at the
    emit boundary, with both numbers in the refusal.
    """
    from tilefoundry_ascend import Ascend950PRTarget

    facts = Ascend950PRTarget().target_event_facts()
    pool = facts.hard_event_pool("aiv", "intra_core")
    assert pool is not None and int(pool.capacity) == 8, "the declared pool moved"

    emitted = _Emitter(_entry_with_events(MINIMAL_AIV, [("set", {"flag_id": 7})])).build()
    events = [s for s in emitted.body.statements() if s.surface.startswith("event.")]
    assert len(events) == 1 and dict(events[0].constants)["flag_id"] == 7

    with pytest.raises(EmitRefusal) as caught:
        _Emitter(_entry_with_events(MINIMAL_AIV, [("set", {"flag_id": 8})])).build()
    assert caught.value.code == "TL_EVENT_SLOT_OUT_OF_POOL"
    assert caught.value.expected == list(range(8)) and caught.value.actual == 8


def test_event_count_is_one_statement_per_obligation_and_never_expands():
    """One statement in, one statement out, at every trip count.

    The completion-side half of "events do not grow with N" is the ring template
    holding 9 sites / 4 channels from N=1 to N=1024. This half is narrower and
    structural: the renderer never loops or duplicates, so k obligations become
    exactly k instructions whatever the region's trip count. A future renderer
    that expanded a symbolic generation into per-trip flags fails here.
    """
    specs = [("set", {"flag_id": 1}), ("wait", {"flag_id": 2}),
             ("set", {"flag_id": 3, "kind": "free", "producer_pipe": "vector",
                      "consumer_pipe": "mte3"})]
    text = pathlib.Path("targets/ascend/tests_native_mainline/fixtures/authors/"
                        "minimal_aiv.py").read_text(encoding="utf-8")
    assert "tf.pipelined(4," in text
    for trips in (1, 2, 4, 8, 64):
        path = pathlib.Path(tempfile.mkdtemp()) / ("n%d.py" % trips)
        path.write_text(text.replace("tf.pipelined(4,", "tf.pipelined(%d," % trips),
                        encoding="utf-8")
        emitted = _Emitter(_entry_with_events("%s:MinimalAivPipeline" % path,
                                              specs)).build()
        events = [s for s in emitted.body.statements() if s.surface.startswith("event.")]
        assert len(events) == 3, (trips, len(events))
        rendered = [line for line in emitted.source.splitlines() if "flag" in line]
        assert len(rendered) == 3, (trips, rendered)


def test_symbolic_generation_is_refused_not_expanded():
    """S0 emits concrete flags; a still-symbolic generation is refused by name.

    Measured shapes: the flag IR accepts a non-int generation (a Var or a DimVar
    is stored as-is; only Constant folds to int), so this refusal is reachable
    and not defensive decoration. It matters because the alternative - a renderer
    looping a symbolic generation over trips - is exactly how the event count
    would start growing with N. When the symbolic case is scheduled, the fix is a
    control envelope, not a loop here.
    """
    from tilefoundry.ir.core import Var
    from tilefoundry.ir.types import DType, TensorType
    from tilefoundry.ir.types.dim import DimVar

    I64 = TensorType.scalar(DType.i64)
    for symbolic in (Var(name="k", type=I64), DimVar("k", 0, 64)):
        with pytest.raises(EmitRefusal) as caught:
            _Emitter(_entry_with_events(MINIMAL_AIV,
                                        [("set", {"generation": symbolic})])).build()
        assert caught.value.code == "TL_EVENT_GENERATION_SYMBOLIC", (symbolic,
                                                                     caught.value.code)
        assert "generation" in caught.value.detail

    # slot must equal generation % versions: a mismatch names a ring position the
    # plan never resolved, and flagging it here is what keeps the emitted id honest
    # rather than merely plausible.
    with pytest.raises(EmitRefusal) as slot:
        _Emitter(_entry_with_events(MINIMAL_AIV,
                                    [("set", {"generation": 2, "versions": 3,
                                              "slot": 1})])).build()
    assert slot.value.code == "TL_EVENT_SLOT_INCONSISTENT", slot.value.code


def test_drain_and_cross_core_are_named_gaps_not_substitutes():
    """Three absences, three codes, none silently filled.

      drain                - the deployed TL face states no drain intrinsic
                             (ascend/language/sync.py exports set_flag/wait_flag/
                             pipe_barrier/threadfence/buffer ops, nothing that
                             retires a region's outstanding work), so a drain
                             obligation has nothing to be emitted as.
      pipe_barrier / fence - no TileFoundry op exists for either, and neither one
                             orders what a flag orders, so neither may substitute
                             for the missing drain.
      cross-core hand-off  - the target declares the cross-core bank/mode facts
                             UNRESOLVED, so no flag id can be proven to fit and
                             the pair is refused instead of guessed.
    """
    from tilefoundry.ir.tir.flag import Drain
    from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Sequential
    from dataclasses import replace as _replace

    _module, entry = _lowered(MINIMAL_AIV)

    def wrapped(statement):
        def put(node):
            if isinstance(node, MeshScope):
                return _replace(node, body=put(node.body))
            if isinstance(node, Sequential):
                return _replace(node, body=tuple(node.body) + (statement,))
            return node
        return _replace(entry, body=put(entry.body))

    def code_for(statement):
        try:
            _Emitter(wrapped(statement)).build()
        except EmitRefusal as refusal:
            return refusal.code
        raise AssertionError("expected a named refusal, got an emitted kernel")

    assert code_for(Evaluate(callable=Drain(), args=())) == "TL_EVENT_SURFACE_MISSING"
    cross = dict(channel_name="ub:tile", kind="ready", slot=0, generation=0,
                 versions=1, flag_id=0, producer_pipe="mte2", consumer_pipe="cube")
    from tilefoundry.ir.tir.flag import FlagSet

    assert code_for(Evaluate(callable=FlagSet(**cross), args=())) == \
        "TL_EVENT_CROSS_CORE_UNSUPPORTED"


def test_event_pool_lookup_refuses_an_undeclared_core():
    """The pool lookup may never be handed a hand-cased core token.

    Measured defect this closes: this very function once called
    hard_event_pool(core.upper()) while the target declares AIV as "aiv"
    (completion/model.py:32), so the lookup returned None. Had None been read as
    "no limit" instead of refused, EVERY flag id would have passed the capacity
    check - a silent hole in the one rule that keeps a synchronisation real.
    Three assertions, each independent:
      - the resolver normalises case for LOOKUP and returns the declared token;
      - an undeclared core is refused by name, never defaulted;
      - no source line that queries a pool hand-cases its core argument.
    """
    from tilefoundry_ascend.codegen.emit import _Emitter
    from tilefoundry_ascend.completion.model import ASCEND_EVENT_FACTS, declared_core
    from tilefoundry_ascend.completion import CompletionError

    assert [str(d) for d in ASCEND_EVENT_FACTS.core_domains] == ["aiv", "aic", "device"]
    for spelling in ("aiv", "AIV", "AiV"):
        assert declared_core(ASCEND_EVENT_FACTS, spelling, "test") == "aiv", spelling
    with pytest.raises(CompletionError) as undeclared:
        declared_core(ASCEND_EVENT_FACTS, "not-a-core", "test")
    assert undeclared.value.code == "COMPLETION_TARGET_UNKNOWN"
    assert undeclared.value.expected == ["aiv", "aic", "device"]

    emitter = object.__new__(_Emitter)
    emitter.event_facts = None
    with pytest.raises(EmitRefusal) as no_facts:
        emitter._event_pool("aiv")
    assert no_facts.value.code == "TL_EVENT_POOL_UNDECLARED"

    # An empty pool table is a refusal too: "no pool declared" is not "unlimited".
    emitter.event_facts = replace(ASCEND_EVENT_FACTS, hard_event_pools=())
    with pytest.raises(EmitRefusal) as no_pool:
        emitter._event_pool("aiv")
    assert no_pool.value.code == "TL_EVENT_POOL_UNDECLARED"

    # Static guard on the rule itself, since the defect was a casing choice on one
    # line: no pool query may case its core argument. Read through the AST, not by
    # scanning lines: a first version of this check searched raw text and "found"
    # the violation inside this module's own explanatory comment - the same
    # mistake in reverse, since prose that names a pattern is not a use of it.
    import ast

    root = pathlib.Path(__file__).resolve().parents[1] / "tilefoundry_ascend"
    offenders = []
    for name in ("completion/plan.py", "completion/verify.py",
                 "completion/model.py", "codegen/emit.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "hard_event_pool"):
                continue
            given = list(node.args) + [keyword.value for keyword in node.keywords]
            for argument in given:
                for inner in ast.walk(argument):
                    if (isinstance(inner, ast.Call)
                            and isinstance(inner.func, ast.Attribute)
                            and inner.func.attr in ("upper", "lower")):
                        offenders.append("%s:%d hand-cased core argument"
                                         % (name, node.lineno))
    assert offenders == [], offenders


# ================================================================ task-49 gate
# The gate used to end in one unconditional NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED
# byte. These tests pin the replacement: each layer answers for itself, a missing
# layer is named, and no layer accepts a self-reported field as proof.

def _gate_chain(path=INDEPENDENT_UB):
    """A real module taken through the real chain, with real artifacts attached."""
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    program = checked_program(path, None, command="test gate chain")
    module = finalize(analyze(program.module, program.entry,
                              analysis=("compute-cost", "memory")),
                      authored=program.authored)
    entry = module.entry_function()
    problem = analyze_native_problem(build_native_unit_tree(module,
                                                          authored=program.authored),
                                     None, None, entry.target)
    plan = resolve_schedule(problem, {
        region.region_ref.region_id: solve_region(to_schedule_problem(region))
        for region in problem.regions})
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)
    artifact = emit_supported_tl(module, plan, completion, None, entry.target)
    return {"module": module, "entry": entry, "plan": plan,
            "completion": completion, "tl": artifact}

def test_each_gate_layer_names_its_own_missing_evidence():
    """4(a): dropping one layer produces THAT layer's named gap, not a blanket code.

    The old gate could not express any of this: every program got the same single
    NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED byte, so "which layer is still
    missing" was unanswerable from the output. Each assertion below removes exactly
    one piece of evidence and checks the gap that names it - a gate that reported
    everything missing at once would satisfy a naive "gap present" check while
    telling a reader nothing.
    """
    from tilefoundry_ascend.passes import gate_layers

    chain = _gate_chain()
    entry, artifact = chain["entry"], chain["tl"]

    # (i) no verifier receipt for this candidate object.
    assert gate_layers.install_gaps(entry) == [("install", "INSTALL_CANDIDATE_UNVERIFIED")]

    # (ii) the TL artifact: absent, or a dict standing in for one.
    assert gate_layers.tl_conformance_gaps(None, entry) == [("tl", "TL_ARTIFACT_NOT_ACTUAL")]
    assert gate_layers.tl_conformance_gaps({"source": "T.ascend_set_flag('MTE2_V', 0)"},
                                           entry) == [("tl", "TL_ARTIFACT_NOT_ACTUAL")]
    #   A real artifact describing a DIFFERENT program is refused by identity - the
    #   binding the old gate never checked.
    assert gate_layers.tl_conformance_gaps(
        replace(artifact, tir_identity="sha256:" + "c" * 64), entry) == [
        ("tl", "ARTIFACT_TIR_IDENTITY_MISMATCH")]

    # (iii) the reader's recomputed verdict. This program emits no event statement,
    #   so the event layer is UNOBSERVED rather than silently acceptable.
    codes = [code for _label, code in gate_layers.tl_conformance_gaps(artifact, entry)]
    assert "TL_EVENT_LOWERING_UNOBSERVED" in codes, codes

    # (iv) ABI: the real artifact agrees; a wrong count or a renamed parameter is
    #   named specifically, not collapsed into "not actual".
    assert gate_layers.abi_gaps(artifact, entry) == []
    # output_count=1 is constructible (the ABI refuses >= len(params), which is a
    # different, earlier invariant) yet still disagrees with this entry's 2.
    assert artifact.abi.output_count != 1
    assert ("abi", "ABI_OUTPUT_COUNT_MISMATCH") in gate_layers.abi_gaps(
        replace(artifact, abi=replace(artifact.abi, output_count=1)), entry)
    renamed = replace(artifact, abi=replace(
        artifact.abi,
        parameters=tuple(replace(spec, name="not_" + spec.name)
                         for spec in artifact.abi.parameters)))
    assert ("abi", "ABI_PARAMETER_ORDER") in gate_layers.abi_gaps(renamed, entry)

    # (v) events, both directions. Obligations with no statement is the dangerous
    #   one (an unsynchronised ring would publish); statements with no obligation
    #   means the candidate and the plan disagree.
    assert gate_layers.event_gaps(entry, chain["completion"]) == [
        ("event", "TL_EVENT_LOWERING_UNOBSERVED")]
    empty = replace(chain["completion"], sites=(), channels=())
    assert gate_layers.event_gaps(entry, empty) == []

    # (vi) the binary layer keeps its three states apart. "Blocked by the
    #   environment" must never read as "compiled clean", so the codes differ.
    assert gate_layers.binary_gaps(artifact) == [("binary", "COMPILE_SITE_NOT_RUN")]
    assert gate_layers.binary_gaps(replace(
        artifact, compiled=None,
        compile_blockage=EmitRefusal("TL_BISHENG_NOT_DISCOVERABLE", "no toolchain"))) == [
        ("binary", "TL_BISHENG_NOT_DISCOVERABLE")]


def test_gate_recomputes_instead_of_reading_stored_identity_fields():
    """4(b): the all-layers-present branch, via the real judgment functions.

    Reaching this branch is only worth anything if it is reached honestly, so no
    verdict is injected: the plan layer rebuilds the plan's canonical bytes, the TL
    layer re-runs read_actual_tl, and the completion layer goes through
    require_completion_verified, which re-runs the structural pass and the
    obligation correspondence against the stored real objects.

    The negative half is the point. Each satisfied layer must go RED when the object
    behind it is replaced, and a dict claiming the right identity must clear nothing:
    a gate that compares two stored strings would pass all of the positives below
    while being unable to fail any of the negatives.
    """
    from tilefoundry_ascend.passes import gate_layers

    chain = _gate_chain()
    plan, completion = chain["plan"], chain["completion"]
    artifacts = {"plan": plan, "completion": completion, "tl": chain["tl"]}

    class _Record:
        plan_identity = plan.plan_identity
        completion_identity = completion.identity
        storage_identity = ""

    # Plan: honest clears; replaced bytes and a dict stand-in do not.
    assert gate_layers.plan_gaps(artifacts, _Record(), "plan") == []
    assert gate_layers.plan_gaps(dict(artifacts, plan=replace(
        plan, plan_canonical_bytes=b"{}")), _Record(), "plan") == [
        ("plan", "PLAN_CONTENT_CHANGED")]
    claimed = {"plan_identity": plan.plan_identity,
               "plan_canonical_bytes": plan.plan_canonical_bytes}
    assert gate_layers.plan_gaps({"plan": claimed}, _Record(), "plan") == [
        ("plan", "PLAN_ARTIFACT_NOT_ACTUAL")]

    # Completion: verified clears; unverified content and moved content do not.
    assert gate_layers.completion_gaps(artifacts, "completion", "plan") == []
    codes = [code for _label, code in gate_layers.completion_gaps(
        dict(artifacts, completion=replace(completion, verification=None)),
        "completion", "plan")]
    assert codes and codes[0].startswith("COMPLETION_"), codes
    codes = [code for _label, code in gate_layers.completion_gaps(
        dict(artifacts, completion=replace(completion, sites=completion.sites[:-1])),
        "completion", "plan")]
    assert "COMPLETION_CONTENT_CHANGED" in codes, codes

def test_a_cache_hit_is_not_a_missing_product():
    """Compiling the same artifact twice IN ONE PROCESS must succeed twice.

    A deploy-cache HIT writes nothing into the caller-supplied cache directory,
    because the products already exist from the earlier build. The first version of
    _collect_products read that emptiness as evidence ("no directory under root has
    this text") and refused with TL_COMPILE_PRODUCTS_INCOMPLETE - a FALSE refusal,
    which is worse than a missing check: it makes a working path look broken and
    sends the next reader hunting a build problem that does not exist. It is the
    same shape as three earlier bugs (default 0 / empty set as none / consistency
    as obligation): one probe returning empty was treated as a conclusion.

    This runs in a child process like the rest of the compile-site tests, because
    the value being tested is the SECOND compile sharing a cache with the first.
    With no toolchain the required outcome is the named BLOCKED_ENV, asserted -
    never a skip.
    """
    outcome = _site_run("cachetwice")
    if outcome.get("blocked"):
        assert outcome["blocked"] == "TL_BISHENG_NOT_DISCOVERABLE", outcome
        assert outcome["searched"], "the refusal must say what was searched"
        return
    # Two compiles of one artifact, the second served from cache.
    assert outcome.get("compiled") is True, outcome
    assert outcome.get("cache_hit_compiled") is True, (
        "a cache hit must not be reported as missing products", outcome)
    assert outcome.get("cache_hit_same_semantics") is True, outcome
    assert outcome.get("cache_hit_products_present") is True, outcome