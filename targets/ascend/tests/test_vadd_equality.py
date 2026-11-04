"""M3: the SimdVF chain computed on CPU, against an independent reference.

The point of this file is that the equality is DISCRIMINATING and the flow is
TYPED. A reference written by calling the same evaluator would agree with a broken
evaluator, so the reference here is torch arithmetic written from the program
stated semantics; and a second, deliberately wrong reference is run so a comparison
that never fails is visible as such.

What is checked, in order:

1. the authored program analyzes and finalizes through the public entry points;
2. the lowered module carries an ExecutionScope inside the AIV MeshScope, with the
   register instructions inside it and the transfers outside;
3. the TYPED DATA FLOW holds: the register vadd accumulates into is the one vld
   filled, the mask it consumes is the one pset produced, vsts stores that same
   register, the UB buffer vld reads is the one the outside load filled, and the
   buffer ub_to_gm reads is the one vsts wrote. Ownership is asserted with it:
   every private value an instruction consumes was created in the same region
   INSTANCE, which is what a name list cannot express. These flow assertions run
   TWICE: once over the IR as built, and once over the module reparsed from the
   emitted text (the "reparsed_" checks). The reparsed pass is the one that counts,
   because its operands are the objects the printer produced; a disagreement
   between the two passes is the failure both are there to find;
4. the emitted text reparses and verifies as a module, with the same structure;
5. the evaluated result equals the torch reference exactly;
6. a wrong reference does NOT equal it, and the result is input-dependent, so (5)
   is a real comparison;
7. the mask is load-bearing: driving vadd with a half-active predicate writes the
   active half and preserves the inactive half, which an all-ones mask cannot show.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN (no torch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import tilefoundry_ascend  # noqa: F401  (registers the SimdVF surface and its ops)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "examples" / "rmsnorm_950pr" / "vadd_chain.py"
ENTRY = "VaddChain"
FUNCTION = "vadd"
LANES = 64
SEED = 0

PASS, FAIL, ERROR, NOT_RUN = 0, 1, 2, 3


def _reference(torch, x):
    """The program stated semantics: every lane active, so out = x + x."""
    return x + x


def _wrong_reference(torch, x):
    """Deliberately wrong: adds one. Must disagree with a correct program."""
    return x + x + 1.0


def _half_mask(torch):
    """A predicate with the low lanes active and the high lanes inactive."""
    mask = torch.zeros(LANES, dtype=torch.bool)
    mask[: LANES // 2] = True
    return mask


def _eval_vadd(torch, tensors):
    """Run the instruction evaluator itself on plain tensors.

    The chain above exercises the mask through `pset`, which activates every lane.
    This drives `vadd` directly so a partial predicate can be applied to data we
    choose, which is the only way the mask can be observed doing anything.
    """
    from tilefoundry.evaluator.registry import schedule_eval_registry
    from tilefoundry_ascend.ir.ops import Vadd

    handler = schedule_eval_registry.lookup(Vadd)
    args = [type("V", (), {"data": tensor})() for tensor in tensors]
    context = type("Eval", (), {"args": args, "op": Vadd(), "result_type": None})()
    return handler(context).data

def _flow(prim):
    """The typed data flow, read off the IR: which VALUE each instruction consumes.

    Names and counts are not a flow test. A chain can carry the right instructions
    in the right scopes and still be wired backwards -- vadd reading a register no
    vld filled, vsts writing a buffer the outside transfer never reads -- so the
    claim to check is producer-to-consumer identity on the operand Vars themselves.

    Identity comes from the Var object, not its spelling: the ":n" / ":n_2"
    suffixes are how the printer disambiguates distinct bindings, and two different
    buffers can print with names suggesting a relationship they do not have. The
    flow recorded here comes from the IR that was built; printer fidelity is
    answered separately by the reload check.

    Ownership is recorded alongside: which region instance each private value was
    created in. A private value whose owner is not the instance that consumes it is
    a boundary defect even when the storage classes happen to line up.

    CAVEAT, recorded rather than papered over: this function asserts identity between
    the Var objects it finds in whatever module it is handed. Run over the IR AS
    BUILT, that is a claim about the constructed IR: if a later pass rewrites a
    binding (a rename or a split), the recorded ids still refer to the built
    objects, and a printed program whose wiring differs would not be caught here.
    Run over the REPARSED module -- which main() also does, under the "reparsed_"
    names -- the operands are the objects the printer produced and the front end
    re-read, so the same assertions become claims about the emitted text. The
    reparsed pass is the one that counts; the built-IR pass is kept because a
    disagreement between the two is itself the interesting failure.
    """
    from tilefoundry.ir.core import Var
    from tilefoundry.ir.tir.memory import AllocTensor, TensorView
    from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, LetStmt, MeshScope, Sequential

    flow = {
        "producer": {},
        "owner": {},
        "calls": [],
        "names": {},
        "instances": 0,
    }

    def walk(stmt, instance):
        if isinstance(stmt, Sequential):
            for item in stmt.body:
                walk(item, instance)
            return
        if isinstance(stmt, MeshScope):
            walk(stmt.body, instance)
            return
        if isinstance(stmt, ExecutionScope):
            flow["instances"] += 1
            walk(stmt.body, flow["instances"])
            return
        if isinstance(stmt, LetStmt):
            value = stmt.value
            target = getattr(value, "target", None)
            flow["names"][stmt.var.name] = id(stmt.var)
            flow["owner"][id(stmt.var)] = instance
            if isinstance(target, AllocTensor):
                flow["producer"][id(stmt.var)] = "alloc_tensor"
            elif isinstance(target, TensorView):
                base = value.args[0] if value.args else None
                alias = (
                    flow["producer"].get(id(base), "tensor_view")
                    if isinstance(base, Var)
                    else "tensor_view"
                )
                flow["producer"][id(stmt.var)] = alias
            else:
                flow["producer"][id(stmt.var)] = getattr(target, "name", "?")
            walk(value, instance)
            return
        if isinstance(stmt, Evaluate):
            flow["calls"].append(
                (
                    getattr(stmt.callable, "name", type(stmt.callable).__name__),
                    tuple(getattr(getattr(arg, "type", None), "storage", None) for arg in stmt.args),
                    instance,
                    tuple(id(arg) for arg in stmt.args if isinstance(arg, Var)),
                )
            )
            for argument in stmt.args:
                walk(argument, instance)
            return
        body = getattr(stmt, "body", None)
        if body is not None:
            walk(body, instance)

    walk(prim.body, None)
    return flow


def _chain_checks(flow):
    """The producer/consumer claims, each answered from the recorded flow."""
    checks = {}
    vld = [entry for entry in flow["calls"] if entry[0] == "Vld"]
    vadd = next((entry for entry in flow["calls"] if entry[0] == "Vadd"), None)
    vsts = next((entry for entry in flow["calls"] if entry[0] == "Vsts"), None)
    pset = next((entry for entry in flow["calls"] if entry[0] == "Pset"), None)
    transfers = [entry for entry in flow["calls"] if entry[0] in ("GmToUb", "UbToGm")]

    checks["two_loads"] = len(vld) == 2
    checks["one_vadd"] = vadd is not None
    checks["one_vsts"] = vsts is not None
    checks["one_pset"] = pset is not None
    checks["two_transfers"] = len(transfers) == 2
    if not all(checks.values()):
        return checks

    instance = vadd[2]
    checks["vadd_inside_an_instance"] = instance is not None
    checks["vld_same_instance"] = all(entry[2] == instance for entry in vld)
    checks["vsts_same_instance"] = vsts[2] == instance
    checks["pset_same_instance"] = pset[2] == instance
    checks["transfers_outside_every_instance"] = all(entry[2] is None for entry in transfers)

    # Operand order is declaration order:
    #   vld(src, dst)   vadd(a, b, mask, dst)   vsts(src, dst)   pset(dst)
    loaded_ids = {entry[3][-1] for entry in vld if len(entry[3]) == 2}
    checks["vadd_reads_a_loaded_register"] = bool(loaded_ids & set(vadd[3][:2]))
    checks["vadd_mask_from_pset"] = pset[3][0] in vadd[3][2:3]
    checks["vsts_stores_the_register_vadd_wrote"] = vsts[3][0] in vadd[3][3:4]

    # Ownership: every private value an instruction consumes was created HERE.
    own = flow["owner"]
    checks["vadd_operands_owned_here"] = all(own.get(operand) == instance for operand in vadd[3])
    checks["vld_destinations_owned_here"] = all(own.get(entry[3][-1]) == instance for entry in vld)
    checks["pset_destination_owned_here"] = own.get(pset[3][0]) == instance

    # The UB buffer the outside transfer fills is the one the inside load reads;
    # the buffer vsts writes is the one the outside transfer hands back.
    gm_to_ub = next(entry for entry in transfers if entry[0] == "GmToUb")
    ub_to_gm = next(entry for entry in transfers if entry[0] == "UbToGm")
    checks["vld_reads_the_buffer_gm_to_ub_filled"] = all(
        entry[3][0] == gm_to_ub[3][-1] for entry in vld
    )
    checks["vsts_writes_the_buffer_ub_to_gm_reads"] = vsts[3][-1] == ub_to_gm[3][0]
    return checks

def _structure(prim):
    """The scope layout the module must have, read off the IR itself."""
    from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, MeshScope, Sequential

    found = {"mesh": 0, "execution": 0, "inside": [], "outside": []}

    def walk(stmt, inside_mesh=False, inside_mode=False):
        if isinstance(stmt, Sequential):
            for item in stmt.body:
                walk(item, inside_mesh, inside_mode)
            return
        if isinstance(stmt, MeshScope):
            found["mesh"] += 1
            walk(stmt.body, True, inside_mode)
            return
        if isinstance(stmt, ExecutionScope):
            found["execution"] += 1
            walk(stmt.body, inside_mesh, True)
            return
        if isinstance(stmt, Evaluate):
            name = type(stmt.callable).__name__
            found["inside" if inside_mode else "outside"].append(name)
            if inside_mode and not inside_mesh:
                found.setdefault("orphan", []).append(name)
            return
        body = getattr(stmt, "body", None)
        if body is not None:
            walk(body, inside_mesh, inside_mode)

    walk(prim.body)
    return found


def main() -> int:
    try:
        import torch
    except Exception as error:  # noqa: BLE001 -- any import failure is NOT_RUN
        print(f"NOT_RUN torch is not importable: {error!r}")
        return NOT_RUN

    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.passes.transforms.convert_hir_to_tir import ConvertHIRToTIR  # noqa: F401
    from tilefoundry.schedule import finalize
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function

    checks = {}

    program = checked_program(f"{SOURCE}:{ENTRY}", None, command="vadd-equality")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    lowered = finalize(analyzed, authored=program.authored)
    prim = lowered.entry_function()
    authored = program.entry
    checks["lowered"] = prim is not None and prim.name == FUNCTION

    verify_module(lowered)
    verify_prim_function(prim)
    checks["verified"] = True

    structure = _structure(prim)
    checks["one_mesh"] = structure["mesh"] == 1
    checks["one_mode"] = structure["execution"] == 1
    checks["transfers_outside"] = sorted(structure["outside"]) == ["GmToUb", "UbToGm"]
    checks["registers_inside"] = sorted(structure["inside"]) == ["Pset", "Vadd", "Vld", "Vld", "Vsts"]
    checks["no_orphan"] = not structure.get("orphan")

    # The typed flow. Names and counts above say the pieces are present; these say
    # they are WIRED, which is a different claim. Run once on the built IR and once
    # on the REPARSED one below; both must hold, and the reparsed pass is the one
    # that counts, because it is the only one whose operands are the objects the
    # printer produced rather than the objects the builder kept.
    flow = _flow(prim)
    checks.update(_chain_checks(flow))

    # The emitted text must be the same program: reparse it and verify again.
    from tilefoundry.inspection import as_script

    import tempfile

    text = as_script(lowered)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reloaded.py"
        path.write_text(text, encoding="utf-8")
        reloaded = load_authored_ir(f"{path}:{ENTRY}")
        verify_module(reloaded)
        verify_prim_function(reloaded.entry_function())
        again = _structure(reloaded.entry_function())
        reparsed_flow = _flow(reloaded.entry_function())
    checks["reload_structure"] = (
        again["mesh"] == structure["mesh"]
        and again["execution"] == structure["execution"]
        and sorted(again["inside"]) == sorted(structure["inside"])
        and sorted(again["outside"]) == sorted(structure["outside"])
    )

    # The flow, asserted on the REPARSED module. This is the stronger form: the
    # operands here were produced by the printer and re-read by the front end, so a
    # printed program whose wiring differs from the built one is caught, and the
    # check no longer depends on the builder having kept its Var objects.
    checks.update(
        {f"reparsed_{name}": ok for name, ok in _chain_checks(reparsed_flow).items()}
    )

    torch.manual_seed(SEED)
    x = torch.randn(LANES, dtype=torch.float32)
    # The authored function is what executes: it carries the region the evaluator
    # knows, and evaluating the lowered text would test the printer fidelity rather
    # than the program.
    produced = evaluate(authored, x)
    data = getattr(produced, "data", produced)
    expected = _reference(torch, x)
    wrong = _wrong_reference(torch, x)

    checks["shape"] = tuple(data.shape) == tuple(expected.shape)
    checks["dtype"] = data.dtype == expected.dtype
    checks["equal"] = bool(torch.equal(data, expected))
    checks["wrong_differs"] = not bool(torch.equal(data, wrong))
    checks["input_dependent"] = not bool(torch.equal(data, torch.zeros_like(data)))

    # The mask must be load-bearing, not decorative. `pset` activates every lane, so
    # the chain above cannot distinguish "the mask was honoured" from "the mask was
    # ignored" -- both produce x + x. Driving the instruction itself with a
    # half-active predicate does distinguish them: inactive lanes keep whatever dst
    # already carried, so an ignored mask shows up as the wrong half being written.
    masked = _eval_vadd(
        torch,
        [
            torch.full((LANES,), 3.0),
            torch.full((LANES,), 4.0),
            _half_mask(torch),
            torch.full((LANES,), -1.0),
        ],
    )
    checks["mask_active_half"] = bool(
        torch.equal(masked[: LANES // 2], torch.full((LANES // 2,), 7.0))
    )
    checks["mask_inactive_kept"] = bool(
        torch.equal(masked[LANES // 2 :], torch.full((LANES // 2,), -1.0))
    )

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print("inside:", sorted(structure["inside"]), "outside:", sorted(structure["outside"]))
    if not all(checks.values()):
        print(f"produced {data[:4].tolist()} expected {expected[:4].tolist()}")
        return FAIL
    print("PASS every check held")
    return PASS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 -- a raised case is ERROR, not FAIL
        print(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(ERROR)