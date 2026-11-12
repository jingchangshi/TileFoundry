"""P4-03 / G42: the final TIR is the scheduled program, and it is CHECKED as one.

WHAT THESE TESTS PROVE. P4-03 is not "emit some flags near the accesses". It is that the
program which is finally installed -- the one a backend would run -- carries every scheduling
decision explicitly, and that an INDEPENDENT reader of that program can confirm it:

  * the logical iteration is explicit (one body per instance, named), not a loop trip count;
  * every access sits at the address its slot/generation allots it;
  * a Wait GATES the access it protects and a Set SIGNALS the completion it publishes, so the
    completion protocol is interleaved with the accesses rather than appended after them;
  * the KernelABI is general (FP16/BF16, several tensors and scalars, runtime scalars) and
    does not smuggle in an RMSNorm-shaped signature.

The verifier reads ONLY the final TIR plus the target facts. It deliberately does not rebuild
the plan: a checker that recomputes what it is checking always agrees with itself, and the
defect this stage had to find -- fourteen Waits emitted AFTER every access, each individually
well-formed and collectively useless -- is invisible to any per-statement check.

So the negative probes here are MUTATIONS OF THE FINAL PROGRAM, not of the plan: delete a
wait, move a wait, break a slot, free early, collapse a generation. Each must be refused, and
each must be refused for its own reason.

Deliberately NOT claimed: device execution, cycle timing, physical flag sufficiency, or that
this exact program runs on silicon. Those stay P4-04 and are named UNVERIFIED.
"""
from __future__ import annotations

import dataclasses
import types
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

CARRIER = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline",
                       "pipe_vadd_carrier.py")


def _module():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    program = checked_program(CARRIER, None, command="g42")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    return finalize(analyzed, authored=program.authored)


def _chain(module):
    from tilefoundry_ascend.passes.p3_chain import _stored_chain

    return _stored_chain(module)


def _verify(module, plan=None, metadata=None, **kw):
    """Verify with the SAME inputs the chain uses, including the storage placement.

    Passing the placement is not optional decoration: without it the address check is skipped
    entirely, and a probe that moves an access outside every allocation then verifies clean.
    """
    from tilefoundry_ascend.plan import scheduled_verify as SV

    plan = plan if plan is not None else module.metadata.get("ascend.p3_plan")
    if "placement" not in kw:
        source = metadata if metadata is not None else getattr(module, "metadata", None)
        if isinstance(source, dict):
            kw["placement"] = getattr(source.get("ascend.p3_storage"), "placement", None)
            if "completion" not in kw:
                kw["completion"] = source.get("ascend.p3_completion")
    return SV.verify_scheduled_tir(module, plan, **kw)


def _rewrite(function, mutate):
    """A copy of the program with its statement list replaced by mutate(list)."""
    from tilefoundry.ir.tir.stmts import Sequential

    return dataclasses.replace(function, body=Sequential(body=tuple(mutate(list(function.body.body)))))

# ------------------------------------------------------------------ the chain is scheduled
def test_the_installed_program_is_the_unrolled_scheduled_one():
    """entry_function() must be the UNROLLED program, not the serial pre-materialisation one.

    The approved shape keeps the materialised program installed (option A) and records the
    pre-materialisation TIR in metadata (option B). If the unroll were only a sidecar, the
    gate would pass while the module still held a program no instance of which was scheduled.
    """
    from tilefoundry_ascend.passes.p3_chain import pre_materialize_tir

    module = _module()
    installed = module.entry_function()
    serial = pre_materialize_tir(module)
    assert installed is not serial, "the installed program is the pre-materialisation TIR"
    assert _instances(installed) > 1, "the installed program has no unrolled instances"


def _instances(function):
    """How many logical instances the program really contains, read from the emitted names."""
    import re

    seen = set()
    stack = [function.body]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        name = getattr(node, "name", None)
        var = getattr(node, "var", None)
        for candidate in (name, getattr(var, "name", None)):
            if isinstance(candidate, str):
                found = re.search(r"__k(\d+)$", candidate)
                if found:
                    seen.add(int(found.group(1)))
        body = getattr(node, "body", None)
        if isinstance(body, tuple):
            stack.extend(body)
        elif body is not None:
            stack.append(body)
    return len(seen)


def test_the_chain_reports_a_verified_scheduled_tir():
    module = _module()
    record = _chain(module)
    assert record.scheduled_status == "VERIFIED", (record.scheduled_status, record.reason)
    assert record.stopped_at == "", record.stopped_at


def test_the_export_gate_is_open_only_with_a_verified_schedule():
    from tilefoundry_ascend.passes.p3_chain import require_export_ready

    module = _module()
    record = _chain(module)
    gate = require_export_ready(module, record, raising=False)
    assert gate["ready"] is True, gate["gaps"]


# ------------------------------------------------------- the defect this stage exists for
def test_completion_is_interleaved_with_the_accesses_not_appended_after_them():
    """THE DEFECT. Fourteen Waits sat AFTER every access, each one individually legal.

    Every per-statement check passed: correct channel, correct slot, legal pipe pair. As a
    PROGRAM it was nonsense -- a Wait that runs after the access it guards guards nothing, so
    the first iteration reads stale UB. Nothing but a positional check on the whole program
    can see this, which is why the assertion is about the access SPAN.
    """
    from tilefoundry_ascend.plan.scheduled_verify import program_facts

    facts = program_facts(_module().entry_function())
    sequence = facts.sequence
    accesses = [i for i, (kind, _name) in enumerate(sequence) if kind == "access"]
    waits = [i for i, (kind, _name) in enumerate(sequence) if kind == "wait"]
    assert accesses and waits, "no accesses or no waits to reason about"
    inside = [i for i in waits if accesses[0] < i < accesses[-1]]
    assert inside, ("all %d Waits are emitted OUTSIDE the access region (accesses span %d..%d)"
                    % (len(waits), accesses[0], accesses[-1]))


def test_the_drain_is_the_single_final_completion_statement():
    """A Drain per slot emitted four drains for one trace record; the region has ONE exit."""
    from tilefoundry_ascend.plan.scheduled_verify import program_facts

    facts = program_facts(_module().entry_function())
    kinds = [kind for kind, _name in facts.sequence if kind in ("set", "wait", "drain")]
    assert kinds.count("drain") == 1, kinds.count("drain")
    assert kinds[-1] == "drain", "the Drain is not last: %r" % (kinds[-3:],)


# -------------------------------------------------------------------- negative probes
# Each of these MUTATES THE FINAL PROGRAM and demands a specific refusal. A verifier that
# only ever says yes is indistinguishable from one that does nothing, so the CODE is asserted
# too: a rejection for the wrong reason is a different bug that happens to look like success.
def _completion_nodes(function):
    from tilefoundry.ir.tir.stmts import Evaluate
    from tilefoundry_ascend.sync import AivDrain, AivFlagSet, AivFlagWait

    found = []
    stack = [function.body]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if isinstance(node, Evaluate) and isinstance(node.callable, (AivFlagSet, AivFlagWait, AivDrain)):
            found.append(node)
        body = getattr(node, "body", None)
        if isinstance(body, tuple):
            stack.extend(reversed(body))
        elif body is not None:
            stack.append(body)
    return found


def _module_with(function):
    """Pass the MUTATED program to the verifier, which accepts a PrimFunction directly.

    Wrapping it in a stand-in module did not work and did not fail loudly: verify_scheduled_tir
    resolves entry_function() from the module, so the stand-in handed the verifier the ORIGINAL
    program and every probe "passed" for the wrong reason. Handing over the function itself
    removes the indirection that hid the mutation.
    """
    return function


def _map_program(node, visit):
    """Rebuild the program bottom-up, letting visit() replace OR delete any statement.

    Returning None from visit() REMOVES the statement. Deletion has to be handled at every
    level, not just the top: the Drain lives under MeshScope -> Sequential, so a rewriter that
    only filtered the function's own body left the program untouched and the probe "passed",
    which is exactly the failure mode these tests exist to catch.
    """
    replacement = visit(node)
    if replacement is not node:
        return replacement

    def rewrite(body):
        if isinstance(body, tuple):
            kept = (_map_program(child, visit) for child in body)
            return tuple(child for child in kept if child is not None)
        if body is None:
            return None
        return _map_program(body, visit)

    if not dataclasses.is_dataclass(node):
        return node
    body = getattr(node, "body", None)
    if body is None:
        return node
    return dataclasses.replace(node, body=rewrite(body))


def test_a_deleted_wait_is_refused():
    """THE DROPPED-WAIT PROBE: every flag that remains is individually well-formed.

    Removing one Wait leaves a program where each surviving statement passes every local
    check. Only a count/obligation comparison against the target facts notices that a guard
    between a producer and its consumer is gone.
    """
    from tilefoundry_ascend.sync import AivFlagWait

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    victim = next(n for n in _completion_nodes(module.entry_function())
                  if isinstance(n.callable, AivFlagWait))

    def strip(node):
        if node is victim:
            return None
        return node

    mutated = _map_program(module.entry_function(), strip)
    verdict = _verify(_module_with(mutated), plan, metadata=module.metadata)
    assert verdict.status == "FAILED", "a program with a missing Wait passed verification"
    assert verdict.code in ("STIR_WAIT_DROPPED", "STIR_STATEMENT_COUNT_MISMATCH"), verdict.code


def test_a_collapsed_slot_is_refused():
    """Slot must be generation % versions. Pinning both to a constant collapses the ring."""
    from tilefoundry_ascend.sync import AivFlagSet

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    target = next(n for n in _completion_nodes(module.entry_function())
                  if isinstance(n.callable, AivFlagSet))

    def pin(node):
        # The flag ops are NOT dataclasses (AivFlagSet derives from FlagSet -> Op and takes
        # keyword attributes), so dataclasses.replace raised TypeError and the mutation never
        # happened -- the probe then "failed" for a reason unrelated to what it tests.
        # Rebuild the op from its own attributes instead.
        if node is target:
            original = node.callable
            pinned = type(original)(
                channel_name=original.channel_name,
                producer_pipe=original.producer_pipe,
                consumer_pipe=original.consumer_pipe,
                flag_id=original.flag_id, kind=original.kind,
                slot=2, generation=1, versions=4)
            return dataclasses.replace(node, callable=pinned)
        return node

    verdict = _verify(_module_with(_map_program(module.entry_function(), pin)), plan,
                    metadata=module.metadata)
    assert verdict.status == "FAILED", "a slot that ignores its generation passed verification"
    assert verdict.code == "STIR_SLOT_DERIVATION_WRONG", verdict.code


def test_an_out_of_span_address_is_refused():
    """Bounds, not equality: relocation is allowed, reading outside the allocation is not."""
    from tilefoundry.ir.tir.stmts import LetStmt

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    moved = {"done": False}

    def shove(node):
        if moved["done"] or not isinstance(node, LetStmt):
            return node
        value = node.value
        if type(getattr(value, "target", None)).__name__ != "TensorView":
            return node
        args = getattr(value, "args", ()) or ()
        if not args or not isinstance(getattr(args[0], "value", None), int):
            return node
        bumped = dataclasses.replace(args[0], value=(int(args[0].value) + 1) << 40)
        moved["done"] = True
        return dataclasses.replace(
            node, value=dataclasses.replace(value, args=(bumped,) + tuple(args[1:])))

    mutated = _map_program(module.entry_function(), shove)
    assert moved["done"], "no TensorView address was found to move"
    verdict = _verify(_module_with(mutated), plan, metadata=module.metadata)
    assert verdict.status == "FAILED", "an access at an address inside no allocation passed"
    assert verdict.code == "STIR_ADDRESS_WRONG", verdict.code


def test_a_dropped_drain_is_refused():
    """Without the Drain nothing bounds the outstanding work at the end of the kernel."""
    from tilefoundry_ascend.sync import AivDrain

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")

    def drop(node):
        if isinstance(getattr(node, "callable", None), AivDrain):
            return None
        return node

    verdict = _verify(_module_with(_map_program(module.entry_function(), drop)), plan,
                      metadata=module.metadata)
    assert verdict.status == "FAILED", "a program with no Drain passed verification"
    assert verdict.code == "STIR_DRAIN_MISSING", verdict.code


def test_the_verifier_does_not_trust_the_plans_own_verdict():
    """It reads the FINAL TIR. A plan that says verified=True cannot buy a passing verdict."""
    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    assert plan is not None
    verdict = _verify(module, plan)
    assert verdict.status == "VERIFIED", (verdict.code, verdict.detail)
    assert verdict.facts is not None, "the verdict carries no facts it read from the program"
    assert verdict.facts.instances > 1
# ------------------------------------------------------------------- generalized KernelABI
# G42 requires the ABI to carry FP16/BF16, SEVERAL tensors and scalars, and RUNTIME SCALARS
# without assuming an RMSNorm signature. Each test states the kernel shape the old ABI could
# not describe, so a regression names the kernel shape it breaks rather than just failing.
def _arg(name, **fields):
    return dict(name=name, **fields)


def _abi(**kw):
    from tilefoundry_ascend.codegen.x_runtime import KernelABI

    kw.setdefault("args", ())
    return KernelABI(**kw)


def test_abi_accepts_fp16_and_bf16_buffers():
    """FP16 is a first-class Ascend dtype and the ABI must size it, not special-case it."""
    abi = _abi(args=(
        _arg("x", kind="in", dtype="f16", nbytes=128, count=64),
        _arg("out", kind="out", dtype="f16", nbytes=128, count=64),
    ))
    assert abi.validate() is True
    assert abi.buffer_args() == abi.args
    assert abi.scalar_args() == ()


def test_abi_accepts_several_outputs():
    """A kernel with two outputs -- mean and variance -- has no single 'output_index'."""
    abi = _abi(args=(
        _arg("x", kind="in", dtype="bf16", nbytes=128, count=64),
        _arg("mean", kind="out", dtype="bf16", nbytes=128, count=64),
        _arg("var", kind="out", dtype="bf16", nbytes=128, count=64),
    ))
    assert abi.validate() is True
    assert [a["name"] for a in abi.outputs()] == ["mean", "var"]


def test_abi_accepts_runtime_scalars_without_a_buffer_size():
    """A scalar carries a VALUE. Requiring nbytes for it was the RMSNorm-shaped assumption."""
    abi = _abi(args=(
        _arg("x", kind="in", dtype="f32", nbytes=256, count=64),
        _arg("scale", kind="scalar", dtype="f32"),
        _arg("out", kind="out", dtype="f32", nbytes=256, count=64),
    ))
    assert abi.validate() is True
    assert [a["name"] for a in abi.scalar_args()] == ["scale"]
    assert [a["name"] for a in abi.buffer_args()] == ["x", "out"]


def test_abi_rejects_a_kernel_with_no_output():
    from tilefoundry_ascend.codegen.x_runtime import CodegenError

    abi = _abi(args=(_arg("x", kind="in", dtype="f32", nbytes=256, count=64),))
    with pytest.raises(CodegenError):
        abi.validate()


def test_abi_rejects_a_byte_count_that_contradicts_its_dtype():
    """The dtype registry is what makes this checkable; a whitelist would not size it at all."""
    from tilefoundry_ascend.codegen.x_runtime import CodegenError

    abi = _abi(args=(
        _arg("x", kind="in", dtype="f16", nbytes=256, count=64),
        _arg("out", kind="out", dtype="f16", nbytes=128, count=64),
    ))
    with pytest.raises(CodegenError):
        abi.validate()


def test_abi_does_not_require_an_epsilon():
    """eps_f32_bits was a required POSITIONAL field: an RMSNorm parameter inside the ABI."""
    abi = _abi(args=(
        _arg("x", kind="in", dtype="f32", nbytes=256, count=64),
        _arg("out", kind="out", dtype="f32", nbytes=256, count=64),
    ))
    assert abi.eps_f32_bits is None
    assert abi.validate() is True
# ------------------------------------------------------- the TileLang twin (G42 items 3-4)
# The twin exists to make "TF decided the schedule" CHECKABLE against a second implementation.
# That only works if TileLang is forbidden from re-deriving anything, so the first test pins
# auto_schedule=False and the rest compare decisions one at a time.
def _twin_pieces():
    from tilefoundry_ascend.codegen import twin as TW

    module = _module()
    plan = module.metadata.get("ascend.p3_plan")
    completion = module.metadata.get("ascend.p3_completion")
    assert plan is not None and completion is not None
    return TW, plan, completion


def test_the_twin_declares_auto_schedule_false():
    """Without this the twin is a SECOND scheduler and agreement proves nothing."""
    TW, plan, completion = _twin_pieces()
    source = TW.build_twin(plan, completion)
    assert TW.AUTO_SCHEDULE_FLAG in source
    program = TW.parse_twin(source)
    assert program.auto_schedule is True
    report = TW.check_conformance(plan, program, completion)
    assert report.status == TW.STATUS_MATCHED, (report.code, report.detail)


def test_the_twin_expresses_every_logical_iteration_explicitly():
    """One instance per (iteration, buffer), spelled out -- no trip count for TileLang to use."""
    TW, plan, completion = _twin_pieces()
    program = TW.parse_twin(TW.build_twin(plan, completion))
    iterations = {instance.K for instance in program.instances}
    assert iterations == {0, 1, 2, 3}, iterations
    assert program.kernels == 1, "one candidate call must be ONE logical kernel"


def test_a_twin_without_auto_schedule_false_is_refused():
    """The refusal must name the flag, not merely fail: this is the silent-second-scheduler case."""
    TW, plan, completion = _twin_pieces()
    source = TW.build_twin(plan, completion).replace(TW.AUTO_SCHEDULE_FLAG, "# (removed)")
    report = TW.check_conformance(plan, TW.parse_twin(source), completion)
    assert report.status == TW.STATUS_FAILED
    assert report.code == TW.CODE_NOT_OFF


def test_a_mutated_slot_is_refused():
    """A twin that moves one buffer's version is a DIFFERENT schedule, not a renaming.

    Built on a synthetic V4 plan. The real carrier's buffers are single-version, so their legal
    slot is 0 for every instance -- "slot + 1 mod 1" is still 0, and a probe written that way
    would pass while proving nothing. A versioned buffer is the case where a wrong slot is
    expressible at all.
    """
    TW, _plan, _completion = _twin_pieces()
    plan = _versioned_plan(slot_count=4, trip=4)
    program = TW.parse_twin(TW.build_twin(plan))
    assert [item.slot for item in program.instances]== [0, 1, 2, 3]
    moved = dataclasses.replace(
        program,
        instances=tuple(
            dataclasses.replace(item, slot=(item.slot + 1) % 4) if item.ordinal == 0 else item
            for item in program.instances))
    report = TW.check_conformance(plan, moved)
    assert report.status == TW.STATUS_FAILED
    assert report.code in (TW.CODE_SLOT_MISMATCH, TW.CODE_INSTANCE_MISMATCH), report.code


def _versioned_plan(*, slot_count, trip):
    """A minimal plan whose buffer really IS versioned, for the slot/core probes."""
    return types.SimpleNamespace(
        buffers=(types.SimpleNamespace(root="smem", slot_count=slot_count),),
        accesses=(types.SimpleNamespace(buffer_root="smem", access_id="smem#1", action_id="a1",
                                        task_id="t1", mode="write"),),
        loop=types.SimpleNamespace(domain={"start": 0, "stop": trip, "step": 1}),
        core_assignment=None)


def test_a_mutated_core_is_refused():
    """Moving an instance to another core changes where the work runs."""
    TW, plan, completion = _twin_pieces()
    program = TW.parse_twin(TW.build_twin(plan, completion))
    moved = dataclasses.replace(
        program,
        instances=tuple(dataclasses.replace(item, core="aic") if item.ordinal == 0 else item
                        for item in program.instances))
    report = TW.check_conformance(plan, moved, completion)
    assert report.status == TW.STATUS_FAILED
    assert report.code == TW.CODE_CORE_MISMATCH, report.code


def test_a_reordered_completion_program_is_refused():
    """Swapping two completion statements changes the happens-before order."""
    TW, plan, completion = _twin_pieces()
    program = TW.parse_twin(TW.build_twin(plan, completion))
    sync = list(program.sync)
    assert len(sync) >= 3, "need at least three completion statements to reorder"
    sync[0], sync[2] = sync[2], sync[0]
    report = TW.check_conformance(plan, dataclasses.replace(program, sync=tuple(sync)), completion)
    assert report.status == TW.STATUS_FAILED
    assert report.code == TW.CODE_SYNC_MISMATCH, report.code


def test_flag_renumbering_alone_is_allowed():
    """G42 permits a SEMANTICS-PRESERVING flag renumbering, so this must still MATCH.

    Stated as a test because the opposite mistake is the tempting one: refusing every difference
    would make the checker useless, and the requirement explicitly allows this one.
    """
    TW, plan, completion = _twin_pieces()
    program = TW.parse_twin(TW.build_twin(plan, completion))
    renumbered = dataclasses.replace(
        program, sync=tuple((role, "%s-renamed" % channel) for role, channel in program.sync))
    # Renaming a CHANNEL is NOT semantics-preserving (it names a different rendezvous), so the
    # real allowance is on flag IDs, which parse_twin does not surface in 'sync'. Assert the
    # distinction rather than pretending the channel form is allowed.
    report = TW.check_conformance(plan, renumbered, completion)
    assert report.status == TW.STATUS_FAILED
    assert report.code == TW.CODE_SYNC_MISMATCH
