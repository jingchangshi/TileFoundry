"""M3 pair negatives: each starts from the accepted chain and breaks one rule.

Every case below is the accepted program with a single intended change, and each
asserts on the refusal text of the check that rule belongs to. A parse error, a
missing dependency or an unrelated failure is not accepted as a stand-in: the
expected substring names the rule, not just "something went wrong".

Run with the approved container interpreter. Exit 0 when every case refuses for
its pinned reason, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tilefoundry_ascend  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]
HEAD = """
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="f", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class Case:
    @func
    def f(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
"""

CASES = [
    (
        "private_storage_hoisted_out",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            g2u = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            a = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
            with T.SimdVF():
                la = tf.schedule((g2u, a), op=T.ascend_vld())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((la, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
""",
        "runs outside every execution region",
        "register storage is allocated outside the mode that owns it",
    ),
    (
        "transfer_inside_mode",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            with T.SimdVF():
                staged = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((staged, out), op=T.ascend_ub_to_gm())
""",
        "belongs outside an execution region",
        "the memory transfer runs with the mode in force",
    ),
    (
        "mask_short_of_data",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            g2u = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                a = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                m = tf.empty(Tensor[(LANES // 2,), "bool", (LANES // 2,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((g2u, a), op=T.ascend_vld())
                lb = tf.schedule((g2u, b), op=T.ascend_vld())
                pm = tf.schedule((m,), op=T.ascend_pset(elem_width=32))
                summed = tf.schedule((la, lb, pm, acc), op=T.ascend_vadd())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((summed, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
""",
        "mask covers",
        "the predicate is shorter than the data it selects",
    ),
    (
        "register_used_after_region",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            g2u = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                a = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((g2u, a), op=T.ascend_vld())
                leaked = la
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((leaked, out), op=T.copy())
""",
        "private to execution region",
        "a register value is carried past the mode that owns it",
    ),
    (
        "mode_inside_mode",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            with T.SimdVF():
                with T.SimdVF():
                    y = x
            return y
""",
        "not nestable",
        "a second mode is entered inside the first",
    ),
    (
        "pset_wrong_element_width",
        """        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            g2u = tf.schedule((x, ub), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                a = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                b = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                m = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                la = tf.schedule((g2u, a), op=T.ascend_vld())
                lb = tf.schedule((g2u, b), op=T.ascend_vld())
                pm = tf.schedule((m,), op=T.ascend_pset(elem_width=17))
                summed = tf.schedule((la, lb, pm, acc), op=T.ascend_vadd())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((summed, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())
""",
        "declares 32",
        "the stated element width is not the width the mode declares",
    ),
]


def runCase(name, body, expected, why):
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "case.py"
        path.write_text(HEAD + body, encoding="utf-8")
        try:
            program = checked_program(f"{path}:Case", None, command="negative")
            analyzed = analyze(program.module, program.entry, analysis=("memory",))
            finalize(analyzed, authored=program.authored)
        except Exception as error:  # noqa: BLE001 -- the refusal is the subject
            text = f"{type(error).__name__}: {error}"
            first = text.splitlines()[0]
            if expected not in text:
                print(f"FAIL {name}: refused for the wrong reason ({why})")
                print(f"       expected {expected!r}")
                print(f"       got      {first[:150]}")
                return False
            print(f"ok   {name}: {first[:110]}")
            return True
    print(f"FAIL {name}: accepted -- expected a refusal; {why}")
    return False


def main() -> int:
    results = [runCase(*case) for case in CASES]
    print(f"{sum(results)}/{len(results)} cases refused for their pinned reason")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())