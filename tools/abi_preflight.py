#!/usr/bin/env python3
"""ABI preflight for test entry points.

The ONLY formal environment is envs/tfnpu-v2. There is no second overlay in
the workspace; a temporary need is created with venv --system-site-packages from it,
used once, then deleted. The removed/deprecated .venv-explorer and the legacy
.conda/envs/tilefoundry-npu are not used for judgment.

Any mismatch fails by name instead of letting a test run in the wrong ABI:
  ABI_PREFLIGHT_INTERPRETER  interpreter prefix is not envs/tfnpu-v2
  ABI_PREFLIGHT_VERSION      tilelang/tvm_ffi/z3 version triple differs
  ABI_PREFLIGHT_SHA          a package file is not the tfnpu-v2 one
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

EXPECTED = {
    "tilelang": "0.1.15+cpu.git0e687c17",
    "apache-tvm-ffi": "0.1.12",
    "z3-solver": "4.15.4.0",
}
Z3_RUNTIME_PREFIX = "4.15.4"


class PreflightError(RuntimeError):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


def _envs_root() -> Path:
    # tools/abi_preflight.py -> workspace/envs
    return Path(__file__).resolve().parents[2] / "envs"


def _prefix() -> Path:
    return Path(sys.prefix).resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError as exc:
        raise PreflightError("ABI_PREFLIGHT_VERSION", "%s is missing: %s" % (name, exc))


def check(expected_prefixes=None) -> dict:
    root = _envs_root()
    prefix = _prefix()
    # task-78 cleanup: only the ONE formal environment may run the preflight.
    allowed = [root / "tfnpu-v2"]
    if expected_prefixes:
        # explicit prefixes are only accepted when they name the same formal env
        allowed = [Path(p).resolve() for p in expected_prefixes
                   if Path(p).resolve() == (root / "tfnpu-v2").resolve()]
        if not allowed:
            raise PreflightError("ABI_PREFLIGHT_INTERPRETER",
                                 "explicit prefix is not envs/tfnpu-v2")
    if not any(prefix == item.resolve() for item in allowed):
        raise PreflightError("ABI_PREFLIGHT_INTERPRETER",
                             "interpreter %s is not one of %s" % (prefix, [str(p) for p in allowed]))
    versions = {name: _version(name) for name in EXPECTED}
    for name, wanted in EXPECTED.items():
        if versions[name] != wanted:
            raise PreflightError("ABI_PREFLIGHT_VERSION",
                                 "%s=%s expected %s" % (name, versions[name], wanted))
    import z3
    z3_runtime = z3.get_version_string()
    if not z3_runtime.startswith(Z3_RUNTIME_PREFIX):
        raise PreflightError("ABI_PREFLIGHT_VERSION",
                             "z3 runtime=%s expected %s*" % (z3_runtime, Z3_RUNTIME_PREFIX))
    import tilelang
    import tvm_ffi
    modules = {"tilelang": Path(tilelang.__file__), "tvm_ffi": Path(tvm_ffi.__file__),
               "z3": Path(z3.__file__)}
    v2_site = (root / "tfnpu-v2" / "lib" / "python3.12" / "site-packages").resolve()
    hashes = {}
    for name, path in modules.items():
        canonical = (v2_site / name / "__init__.py")
        if not canonical.is_file():
            canonical = (v2_site / name)
        actual_sha = _sha(path)
        canonical_sha = _sha(canonical) if canonical.is_file() else None
        if canonical_sha is None or actual_sha != canonical_sha:
            raise PreflightError("ABI_PREFLIGHT_SHA",
                                 "%s file %s sha=%s != tfnpu-v2 %s" %
                                 (name, path, actual_sha, canonical_sha))
        hashes[name] = actual_sha
    return {"status": "PASS", "interpreter": str(Path(sys.prefix) / "bin" / "python"),
            "prefix": str(prefix), "versions": versions,
            "z3_runtime": z3_runtime,
            "files": {name: str(path) for name, path in modules.items()},
            "sha256": hashes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expected-prefix", action="append", default=None)
    args = parser.parse_args(argv)
    try:
        result = check(args.expected_prefix)
    except PreflightError as error:
        payload = {"status": "FAIL", "code": error.code, "detail": error.detail}
        print(json.dumps(payload, sort_keys=True))
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("ABI_PREFLIGHT PASS", result["versions"], "z3-runtime=" + result["z3_runtime"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
