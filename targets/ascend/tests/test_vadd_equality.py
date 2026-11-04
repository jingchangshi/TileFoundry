"""M3: the SimdVF chain computed on CPU, against an independent reference.

The point of this file is that the equality is discriminating. A reference
written by calling the same evaluator would agree with a broken evaluator, so
the reference here is torch arithmetic written from the program stated
semantics; and a second, deliberately wrong reference is run so a comparison
that never fails is visible as such.

What is checked, in order:

1. the authored program analyzes and finalizes through the public entry points;
2. the lowered module carries an ExecutionScope inside the AIV MeshScope, with
   the register instructions inside it and the transfers outside;
3. the emitted text reparses and verifies as a module;
4. the evaluated result equals the torch reference exactly;
5. a wrong reference does NOT equal it, so (4) is a real comparison.

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
    checks["reload_structure"] = (
        again["mesh"] == structure["mesh"]
        and again["execution"] == structure["execution"]
        and sorted(again["inside"]) == sorted(structure["inside"])
        and sorted(again["outside"]) == sorted(structure["outside"])
    )

    torch.manual_seed(SEED)
    x = torch.randn(LANES, dtype=torch.float32)
    # The authored function is what executes: it carries the region the
    # evaluator knows, and evaluating the lowered text would test the printer's
    # fidelity rather than the program.
    produced = evaluate(authored, x)
    data = getattr(produced, "data", produced)
    expected = _reference(torch, x)
    wrong = _wrong_reference(torch, x)

    checks["shape"] = tuple(data.shape) == tuple(expected.shape)
    checks["dtype"] = data.dtype == expected.dtype
    checks["equal"] = bool(torch.equal(data, expected))
    checks["wrong_differs"] = not bool(torch.equal(data, wrong))
    checks["input_dependent"] = not bool(torch.equal(data, torch.zeros_like(data)))

    # The mask must be load-bearing, not decorative. `pset` activates every lane,
    # so the chain above cannot distinguish "the mask was honoured" from "the mask
    # was ignored" -- both produce x + x. Driving the instruction itself with a
    # half-active predicate does distinguish them: inactive lanes keep whatever
    # dst already carried, so a mask that is ignored shows up as the wrong half
    # being written.
    masked = _eval_vadd(torch, [
        torch.full((LANES,), 3.0),
        torch.full((LANES,), 4.0),
        _half_mask(torch),
        torch.full((LANES,), -1.0),
    ])
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