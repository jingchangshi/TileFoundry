"""Environment validation only: consume separately emitted exact native TL.
Does not import TileFoundry or ORTools. Not a production runtime change.
"""
import faulthandler
faulthandler.enable()
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
print("SPLIT before_torch_import", flush=True)
import torch
import torch_npu
import tilelang
print("SPLIT imported", torch.__version__, torch_npu.__version__, tilelang.__version__, flush=True)
assert not any(n == "tilefoundry" or n.startswith("tilefoundry.") or n == "ortools" or n.startswith("ortools.") for n in sys.modules)
source = root / "split_native_tl.py"
sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert sha == "022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9"
spec = importlib.util.spec_from_file_location("split_native_runtime_artifact", source)
loaded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaded)
print("SPLIT before_compile", flush=True)
policy = {"tl.enable_auto_schedule": False}
kernel = tilelang.compile(loaded.main, target="ascend", out_idx=None, pass_configs=policy)
print("SPLIT after_compile", flush=True)
cce = kernel.get_kernel_source()
(root / "split_native.cce").write_text(cce)
print("SPLIT before_set_device", flush=True)
torch.npu.set_device(0)
print("SPLIT after_set_device", flush=True)
(root / "split_loaded_maps.txt").write_text(Path("/proc/self/maps").read_text())
x = (torch.arange(256, dtype=torch.float32) - 113) / 16
z = (torch.arange(256, dtype=torch.float32).remainder(17) - 7) / 8
rows = []
for name, right in (("product", z), ("square", x)):
    print("SPLIT before_alloc", name, flush=True)
    xn, zn = x.npu(), right.npu()
    out = torch.empty_like(xn)
    print("SPLIT before_launch", name, flush=True)
    kernel(xn, zn, out)
    print("SPLIT before_sync", name, flush=True)
    torch.npu.synchronize()
    print("SPLIT after_sync", name, flush=True)
    actual = out.cpu()
    expected = x * right
    ok = bool(torch.equal(actual, expected))
    row = {"case": name, "passed": ok, "max_abs_error": (actual - expected).abs().max().item()}
    rows.append(row)
    print("SPLIT_RESULT", json.dumps(row), flush=True)
assert not any(n == "tilefoundry" or n.startswith("tilefoundry.") or n == "ortools" or n.startswith("ortools.") for n in sys.modules)
report = {"status": "PASS" if all(r["passed"] for r in rows) else "FAIL", "scope": "same-source static micro-case runtime only; not full scheme or repaired single-process wrapper", "cases": rows, "tl_sha256": sha, "cce_sha256": hashlib.sha256(cce.encode()).hexdigest(), "pass_configs": policy, "TF_ORtools_imported": False}
(root / "split_result.json").write_text(json.dumps(report, indent=2) + "\n")
if report["status"] != "PASS":
    raise SystemExit(1)
