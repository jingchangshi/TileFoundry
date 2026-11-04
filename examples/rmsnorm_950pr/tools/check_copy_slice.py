"""M2b closure: reload verification, storage/effect assertions and negative cases."""
from __future__ import annotations
import json, sys, tempfile, traceback
from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function


def verify_artifact(path: Path, entry: str) -> dict:
    m = load_authored_ir(f"{path}:{entry}")
    verify_module(m)
    verify_prim_function(m.entry_function(), module_fns=m)
    return m


def main(artifact: str, entry: str) -> int:
    p = Path(artifact)
    text = p.read_text()
    out = {"artifact": str(p), "entry": entry, "checks": {}, "negatives": {}}
    # structural / storage / effect assertions on the emitted text
    out["checks"]["two_copies"] = text.count("T.copy(") == 2
    out["checks"]["smem_staging"] = 'storage="smem"' in text
    out["checks"]["distinct_entry_and_output"] = ("x: Tensor[" in text and "out: Tensor[" in text)
    out["checks"]["aiv_mesh"] = 'Topology("aiv", 1)' in text
    # supported reload + verification paths (architect-provided entry)
    try:
        verify_artifact(p, entry)
        out["checks"]["reload_verify_module"] = True
    except Exception as exc:  # noqa: BLE001
        out["checks"]["reload_verify_module"] = False
        out["reload_error"] = repr(exc)[:300]
    # negative cases: tampered artifacts must be rejected
    scratch = Path(artifact).parent
    def negative(name: str, mutated: str) -> None:
        path = scratch / f"__negative_{name}.py"
        path.write_text(mutated)
        try:
            verify_artifact(path, entry)
            out["negatives"][name] = {"rejected": False}
        except Exception as exc:  # noqa: BLE001
            out["negatives"][name] = {"rejected": True, "error": repr(exc)[:200]}
    negative("shape_layout_mismatch", text.replace("shape=(64,)", "shape=(32,)"))
    negative("smem_to_unknown_storage", text.replace('storage="smem"', 'storage="l2"'))
    ok = all(out["checks"].values()) and all(v["rejected"] for v in out["negatives"].values())
    # CPU semantic equality requires torch; record honestly if unavailable
    try:
        import torch  # noqa: F401
        out["cpu_semantics"] = {"status": "AVAILABLE"}
    except Exception as exc:  # noqa: BLE001
        out["cpu_semantics"] = {"status": "NOT_RUN", "reason": repr(exc)[:200]}
    out["ok"] = bool(ok)
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
