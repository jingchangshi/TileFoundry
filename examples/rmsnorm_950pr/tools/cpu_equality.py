"""CPU equality: the authored HIR against an independent torch reference.

The authored program is executed by the HIR evaluator; the reference is written
here, in torch, from the program's stated semantics. Both run on the same CPU
inputs.

What "exact" means here, precisely. The comparison is torch.equal, which is
exact VALUE equality: two tensors are equal when they have the same shape,
the same dtype, and every element compares equal. For float32 that is not
the same claim as bitwise identity -- it does not distinguish +0.0 from
-0.0, and it does not inspect NaN payloads (a NaN compares unequal to
anything, including another NaN, so a program producing NaN FAILs here
rather than passing quietly). Nothing in this tool claims a bit-level
proof. What it establishes is that every element carries the same value as
the reference, which for a copy is the whole semantic claim. No tolerance
is ever applied, so an approximate match cannot pass as a copy.

Nothing in this tool reaches a device. When torch is missing the case is
reported NOT_RUN with its reason -- never PASS: an equality claim that was not
executed is not a weaker pass, it is no evidence at all.

Exit codes
    0  every case PASSed
    1  a case FAILed (values, dtype, or shape disagree)
    2  ERROR: the artifact could not be loaded or a case raised
    3  NOT_RUN: a dependency (torch / evaluator) is unavailable
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

#: tools -> rmsnorm_950pr -> examples -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]

PASS, FAIL, ERROR, NOT_RUN = 0, 1, 2, 3
_STATUS_EXIT = {"PASS": PASS, "FAIL": FAIL, "ERROR": ERROR, "NOT_RUN": NOT_RUN}

SEED = 0

#: One entry per case. "reference" names a function in REFERENCES; "reason" is
#: the written argument for what the program must compute -- the reference is
#: only as good as that sentence.
CASES = {
    "copy_slice": {
        "source": "examples/rmsnorm_950pr/copy_slice.py",
        "entry": "CopySlice",
        "function": "copy1",
        "reference": "identity",
        "reason": (
            "copy1 stages x through one smem buffer and writes it to a fresh gmem "
            "output; a copy computes nothing, so every output element must equal "
            "the input element at the same index, in the same dtype and shape."
        ),
        "inputs": {"x": (64,)},
    },
}


def _reference_identity(torch, **inputs):
    """out == in: a copy is the identity map on values, dtypes and shapes."""
    return inputs["x"].clone()


REFERENCES = {"identity": _reference_identity}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_torch():
    """torch, or the reason it is not importable."""
    try:
        import torch  # noqa: PLC0415 -- the dependency is what this checks
    except Exception as exc:  # noqa: BLE001 -- any import failure is NOT_RUN
        return None, repr(exc)[:300]
    return torch, None


def run_case(name: str, torch) -> dict:
    """Execute one case and compare it to its reference, exactly."""
    spec = CASES[name]
    source = REPO_ROOT / spec["source"]
    record = {
        "case": name,
        "source": str(source),
        "entry": spec["entry"],
        "function": spec["function"],
        "reference": spec["reference"],
        "comparison": "torch.equal -- exact value, no tolerance (not a bitwise proof)",
        "reason": spec["reason"],
        "device": "cpu",
        "seed": SEED,
    }
    if not source.is_file():
        record.update(status="ERROR", error=f"artifact not found: {source}")
        return record

    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.evaluator import evaluate

    torch.manual_seed(SEED)
    inputs = {
        key: torch.randn(*shape, dtype=torch.float32)
        for key, shape in spec["inputs"].items()
    }
    record["inputs"] = {key: list(v.shape) for key, v in inputs.items()}
    record["input_dtype"] = str(next(iter(inputs.values())).dtype)

    module = load_authored_ir(f"{source}:{spec['entry']}")
    function = module.entry_function()
    record["loaded_function"] = function.name
    record["loaded_function_type"] = type(function).__name__

    produced = evaluate(function, *inputs.values())
    produced_data = getattr(produced, "data", produced)
    expected = REFERENCES[spec["reference"]](torch, **inputs)

    record["shape_ok"] = tuple(produced_data.shape) == tuple(expected.shape)
    record["dtype_ok"] = produced_data.dtype == expected.dtype
    record["shape"] = list(produced_data.shape)
    record["dtype"] = str(produced_data.dtype)
    record["bitwise_equal"] = bool(torch.equal(produced_data, expected))
    if record["shape_ok"] and record["dtype_ok"] and produced_data.numel():
        delta = (produced_data.to(torch.float64) - expected.to(torch.float64)).abs()
        record["max_abs_diff"] = float(delta.max())
    elif record["shape_ok"] and record["dtype_ok"]:
        record["max_abs_diff"] = 0.0
    record["checks"] = {
        "shape": record["shape_ok"],
        "dtype": record["dtype_ok"],
        "bits": record["bitwise_equal"],
    }
    record["status"] = "PASS" if all(record["checks"].values()) else "FAIL"
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an authored HIR program against a torch reference on CPU."
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="case to run; repeatable. Defaults to every case.",
    )
    parser.add_argument("--out", help="write the JSON report here (stdout carries it too)")
    parser.add_argument(
        "--source",
        help=(
            "assert the case's source resolves to this path before running."
            " A caller that names a source gets a refusal if the case it"
            " selected is about a different program, rather than a verdict"
            " about the wrong one."
        ),
    )
    parser.add_argument(
        "--entry",
        help="assert the case's entry name matches, on the same grounds",
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="treat a missing torch as ERROR rather than NOT_RUN",
    )
    args = parser.parse_args(argv)

    cases = args.case or sorted(CASES)

    # A caller-supplied source/entry is a claim about WHICH program to check.
    # If it disagrees with the selected case, refuse: running the case anyway
    # would answer a question nobody asked, and silently checking a different
    # artifact is worse than not checking one.
    for name in cases:
        spec = CASES[name]
        if args.source is not None:
            wanted = Path(args.source).resolve()
            actual = (REPO_ROOT / spec["source"]).resolve()
            if wanted != actual:
                print(
                    "cpu_equality: error: case %r is about %s, not the requested %s"
                    % (name, actual, wanted),
                    file=sys.stderr,
                )
                return ERROR
        if args.entry is not None and spec["entry"] != args.entry:
            print(
                "cpu_equality: error: case %r has entry %r, not the requested %r"
                % (name, spec["entry"], args.entry),
                file=sys.stderr,
            )
            return ERROR

    report = {
        "tool": "cpu_equality",
        "generated_at": _now(),
        "repo_root": str(REPO_ROOT),
        "device": "cpu",
        "cases": [],
    }
    torch, reason = _load_torch()
    report["torch"] = getattr(torch, "__version__", None)
    if torch is None:
        report["torch_import_error"] = reason

    for name in cases:
        if torch is None:
            report["cases"].append(
                {
                    "case": name,
                    "status": "ERROR" if args.require_torch else "NOT_RUN",
                    "reason": f"torch is not importable: {reason}",
                }
            )
            continue
        try:
            report["cases"].append(run_case(name, torch))
        except Exception as exc:  # noqa: BLE001 -- a broken case is ERROR, not FAIL
            report["cases"].append(
                {
                    "case": name,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )

    statuses = [case["status"] for case in report["cases"]]
    if "FAIL" in statuses:
        report["status"] = "FAIL"
    elif "ERROR" in statuses:
        report["status"] = "ERROR"
    elif "NOT_RUN" in statuses:
        report["status"] = "NOT_RUN"
    else:
        report["status"] = "PASS"
    report["exit_code"] = _STATUS_EXIT[report["status"]]

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"cpu_equality: wrote {destination}", file=sys.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
