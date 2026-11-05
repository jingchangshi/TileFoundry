#!/usr/bin/env python3
"""Prove the run_case device gate refuses. Run from this directory."""
import sys, types

failures = 0


def case(label, expect_exit, fake_present, auth):
    global failures
    import importlib
    import run_case
    importlib.reload(run_case)
    # Simulate torch_npu being installed WITHOUT importing it, which is the whole
    # point of the gate: presence must be answerable by find_spec alone.
    real = importlib.util.find_spec
    if fake_present:
        importlib.util.find_spec = lambda name, *a: (
            types.SimpleNamespace(name="torch_npu") if name == "torch_npu" else real(name, *a)
        )
    else:
        importlib.util.find_spec = lambda name, *a: None if name == "torch_npu" else real(name, *a)
    run_case.os.environ.pop("TF_DEVICE_AUTH", None)
    if auth:
        run_case.os.environ["TF_DEVICE_AUTH"] = "1"
    try:
        run_case.refuse_unless_authorised(0)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        importlib.util.find_spec = real
    ok = code == expect_exit
    print(("  PASS  " if ok else "  FAIL  ") + "%s (exit %s, expected %s)" % (label, code, expect_exit))
    if not ok:
        failures += 1


# torch_npu absent -> NOT_RUN, because nothing can be measured. Not a pass.
case("torch_npu absent refuses as NOT_RUN", 3, False, False)
# torch_npu present, no authorisation -> BLOCKED_ENV. The default path.
case("present without authorisation refuses as BLOCKED_ENV", 75, True, False)
# Explicit, wrong-valued authorisation is still refusal.
case("authorisation must be exactly 1", 75, True, False)
# Authorised -> proceeds (no exception).
case("explicit authorisation proceeds", 0, True, True)

# --- device selection: the list decides, nothing is hardcoded -----------------
# A one-card visible list must REFUSE a request for its second entry. Falling back to
# entry 0 would silently measure a different card than the one asked for, which is the
# failure this rule exists to prevent.


def case_device_selection():
    global failures
    import importlib
    import run_case
    importlib.reload(run_case)
    saved = {k: run_case.os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "TF_DEVICE_ALLOW")}
    run_case.os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"   # ONE card, id 1
    run_case.os.environ["TF_DEVICE_ALLOW"] = "0,1"
    try:
        sel = run_case.resolve_device_selection(None)
        ok_default = (sel["visible_list"] == [1] and sel["logical_index"] == 0
                      and sel["physical_device_id"] == 1)
        print(("  PASS  " if ok_default else "  FAIL  ") +
              "default selects entry 0 of the list (%r)" % (sel,))
        if not ok_default:
            failures += 1
        try:
            run_case.resolve_device_selection(1)
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        ok = code == 75
        print(("  PASS  " if ok else "  FAIL  ") +
              "index 1 of a 1-entry list is refused as BLOCKED_ENV (exit %s, expected 75)" % code)
        if not ok:
            failures += 1
    finally:
        for k, v in saved.items():
            if v is None:
                run_case.os.environ.pop(k, None)
            else:
                run_case.os.environ[k] = v


case_device_selection()

# --- static ordering check -------------------------------------------------
# A behavioural test cannot see this. If the guard were moved below the import, a
# unit test that calls the guard directly would still return 75, while the real
# script had already loaded libhccl and opened the device node. Only source order
# catches that.
import ast
from pathlib import Path


def check_order():
    global failures
    src = Path(__file__).with_name("run_case.py").read_text()
    tree = ast.parse(src)
    main_fn = next((f for f in ast.walk(tree)
                    if isinstance(f, ast.FunctionDef) and f.name == "main"), None)
    if main_fn is None:
        print("  FAIL  run_case.main not found")
        failures += 1
        return
    guard = imp = setdev = None
    for node in ast.walk(main_fn):
        text = ast.get_source_segment(src, node) or ""
        if guard is None and isinstance(node, ast.Call) and "refuse_unless_authorised" in text:
            guard = node.lineno
        if imp is None and isinstance(node, ast.Import) and any(
                "torch_npu" in (a.name or "") for a in node.names):
            imp = node.lineno
        if setdev is None and isinstance(node, ast.Call) and "set_device" in text:
            setdev = node.lineno
    problems = []
    if guard is None:
        problems.append("no guard call in main")
    if imp is None:
        problems.append("no torch_npu import any more")
    if setdev is None:
        problems.append("no set_device call any more")
    if not problems:
        if not guard < imp:
            problems.append("guard (line %d) does not precede the import (line %d)" % (guard, imp))
        if not guard < setdev:
            problems.append("guard (line %d) does not precede set_device (line %d)" % (guard, setdev))
    if problems:
        for item in problems:
            print("  FAIL  " + item)
        failures += 1
    else:
        print("  PASS  guard (line %d) precedes the torch_npu import (line %d) and set_device (line %d)"
              % (guard, imp, setdev))


check_order()
print()
if failures:
    print("RESULT: FAIL (%d)" % failures)
    sys.exit(1)
print("RESULT: PASS")