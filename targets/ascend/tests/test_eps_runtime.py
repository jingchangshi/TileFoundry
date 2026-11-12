"""Runtime eps scalar ABI (03-24): eps as a 0-d f32 RUNTIME operand, not a magic constant.

v33 kernel signature is main(X, W, Y, eps: T.float32). Here ascend_vadds accepts EITHER the
value attribute OR exactly one 0-d f32 runtime operand (never a constant-folded literal).

Checks: const-eps and runtime-eps produce IDENTICAL bits on the same input; changing eps changes
the output; the runtime TIR keeps eps as a prim_func parameter. Named refusals: eps shape (1,)
(not 0-d), eps bf16 / i32 / f64 (not f32), and eps deleted (neither attribute nor operand).

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import os, struct, subprocess, sys
import tilefoundry_ascend  # noqa: F401
import torch

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu loaded; CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIXDIR = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm")
L = 64


def cli(*a):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"; env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *a], cwd=ROOT, env=env, capture_output=True, text=True)


def main():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.evaluator import evaluate

    checks = {}
    d = os.environ.get("TMPDIR", "/tmp")
    for name, entry in (("eps_const", "EpsConst"), ("eps_runtime", "EpsRt")):
        r = cli("analyze", f"{FIXDIR}/{name}.py:{entry}")
        checks[f"{name}_analyze_exit0"] = r.returncode == 0
        r = cli("schedule", "finalize", f"{FIXDIR}/{name}.py:{entry}", os.path.join(d, f"{name}_tir.py"))
        checks[f"{name}_finalize_exit0"] = r.returncode == 0
    rt_tir = os.path.join(d, "eps_runtime_tir.py")
    text = open(rt_tir, encoding="utf-8").read() if os.path.isfile(rt_tir) else ""
    checks["runtime_eps_is_a_param"] = 'eps: Tensor[(), "f32"]' in text and "ascend_vadds(" in text
    checks["runtime_vadds_uses_operand_not_attr"] = ("ascend_vadds(" in text) and ("value=" not in text.split("ascend_vadds(")[1].split(")")[0])

    x = torch.tensor([((i % 7) - 3) * 0.5 for i in range(L)], dtype=torch.float32).to(torch.bfloat16)
    const_prog = checked_program(f"{FIXDIR}/eps_const.py:EpsConst", None, command="eps-const")
    rt_prog = checked_program(f"{FIXDIR}/eps_runtime.py:EpsRt", None, command="eps-runtime")

    def bits(t):
        return [b & 0xFFFF for b in t.reshape(-1).view(torch.int16).tolist()]

    got_const = bits(getattr(evaluate(const_prog.entry, x), "data", None))
    got_rt = bits(getattr(evaluate(rt_prog.entry, x, torch.tensor(1e-6, dtype=torch.float32)), "data", None))
    got_rt_big = bits(getattr(evaluate(rt_prog.entry, x, torch.tensor(1e-3, dtype=torch.float32)), "data", None))
    checks["const_and_runtime_eps_identical_bits"] = got_const == got_rt
    checks["changing_eps_changes_output"] = got_rt_big != got_rt

    # named refusals, authored fixtures derived from the runtime one
    base = open(f"{FIXDIR}/eps_runtime.py", encoding="utf-8").read()
    variants = {
        "eps_not_0d": (base.replace('eps: Tensor[(), "f32"]', 'eps: Tensor[(1,), "f32"]'), "0-d f32"),
        "eps_bf16": (base.replace('eps: Tensor[(), "f32"]', 'eps: Tensor[(), "bf16"]'), "0-d f32"),
        "eps_i32": (base.replace('eps: Tensor[(), "f32"]', 'eps: Tensor[(), "i32"]'), "0-d f32"),
        # f64 is not a dtype this DSL knows at all, so it is refused at the dtype-name layer;
        # the scalar rule is what refuses bf16/i32. Both are refusals, by name.
        "eps_f64": (base.replace('eps: Tensor[(), "f32"]', 'eps: Tensor[(), "f64"]'), "0-d f32|unknown value 'f64'"),
        "eps_deleted": (base.replace('eps: Tensor[(), "f32"], ', "").replace(", eps)", ")"), "value must be a real scalar"),
    }
    for name, (src, needle) in variants.items():
        path = os.path.join(d, f"eps_{name}.py")
        open(path, "w", encoding="utf-8").write(src)
        r = cli("analyze", f"{path}:EpsRt")
        msg = r.stdout + r.stderr
        checks[f"neg_{name}_refused"] = r.returncode != 0 and any(n in msg for n in needle.split("|"))

    for k, v in checks.items():
        print(("ok   " if v else "FAIL ") + k)
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}"); raise SystemExit(2)
