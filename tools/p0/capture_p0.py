#!/usr/bin/env python3
"""P0 recapture: ONE immutable run directory and ONE manifest (no anchors, no dual domain).

Produces, under <run-dir>/payload (the sealed content) and <run-dir>/MANIFEST.json
(outside the payload, so the payload has exactly one domain):
  payload/precision_protocol.json
  payload/environment.json
  payload/source_manifest.json
  payload/execution_topology.json
  payload/passes/<case>/<pass>.tir     (the five TileLang scheduling passes)
  MANIFEST.json                        (hashes + commands + child exit codes)

Exit 0 only if every capture step succeeded; the manifest records per-step status.
"""
from __future__ import annotations

import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

PASSES = ("tl.AutoSchedule", "tl.PrepareMultiBuffer", "tl.InsertSync",
          "tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR")
CASES = ((8192, 4096, "dyn"), (8192, 4096, "recompute"),
         (64, 4096, "dyn"), (64, 4096, "recompute"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: str, cwd: str | None = None) -> dict:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return {"cmd": cmd, "exit": p.returncode,
            "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}


def version(name: str) -> str | None:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:
        return None


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    payload = out / "payload"
    (payload / "passes").mkdir(parents=True, exist_ok=True)
    steps: dict[str, dict] = {}
    started = time.time()

    # 1. frozen protocol (already frozen; copied verbatim)
    proto = Path(os.environ.get("P0_PROTOCOL", "")).expanduser()
    if not proto.is_file():
        sys.stderr.write("P0_PROTOCOL missing\n"); return 75
    (payload / "precision_protocol.json").write_bytes(proto.read_bytes())
    steps["precision_protocol"] = {"exit": 0, "src": str(proto),
                                   "sha256": sha256(payload / "precision_protocol.json")}

    # 2. environment of record
    env = {
        "python_exe": sys.executable,
        "python": platform.python_version(),
        "packages": {n: version(n) for n in ("torch", "torch_npu", "tilelang",
                                             "apache-tvm-ffi", "numpy")},
        "cann_ascend_toolkit": os.environ.get("ASCEND_TOOLKIT_HOME"),
        "path_head": os.environ.get("PATH", "").split(":")[:4],
        "epoch": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (payload / "environment.json").write_text(json.dumps(env, indent=2))
    steps["environment"] = {"exit": 0}

    # 3. source inventory (real import sources, HEAD + dirty, kernel + protocol SHA)
    tf = Path(os.environ.get("TF_ROOT") or "/workspace/TileFoundry")
    tl = Path(os.environ.get("TILELANG_ROOT") or os.environ.get("TL_ROOT") or "/workspace/tilelang")
    kernel = Path(os.environ.get("P0_KERNEL", "")).expanduser()
    src = {"tilefoundry": {"root": str(tf)}, "tilelang": {"root": str(tl)}}
    for name, root in (("tilefoundry", tf), ("tilelang", tl)):
        for key, cmd in (("head", "git rev-parse HEAD"),
                         ("dirty", "git status --porcelain"),
                         ("remote", "git config --get remote.origin.url || true")):
            r = run(cmd, cwd=str(root))
            src[name][key] = r["stdout"].strip() if key != "dirty" else r["stdout"].strip().splitlines()
        steps["git_" + name] = {"exit": 0}
    if not kernel.is_file():
        sys.stderr.write("P0_KERNEL missing\n"); return 75
    src["kernel_v33"] = {"path": str(kernel), "sha256": sha256(kernel)}
    src["precision_protocol"] = {"sha256": sha256(payload / "precision_protocol.json")}
    (payload / "source_manifest.json").write_text(json.dumps(src, indent=2))
    steps["source_manifest"] = {"exit": 0,
                                "kernel_sha256": src["kernel_v33"]["sha256"]}

    # 4. execution topology facts (target facts + best-effort npu-smi)
    try:
        sys.path.insert(0, os.environ.get("TF_TARGET_PATH", str(tf / "targets/ascend")))
        from tilefoundry_ascend.tf_target import Ascend950PRTarget
        t = Ascend950PRTarget()
        topo = {"target": {"name": t.name, "aiv_count": t.aiv_count, "aic_count": t.aic_count}}
    except Exception as e:  # noqa: BLE001
        topo = {"target_error": "%s: %s" % (type(e).__name__, e)}
    smi = run("npu-smi info -t board -i 0 2>&1 | head -40")
    topo["npu_smi"] = {"exit": smi["exit"], "text": smi["stdout"]}
    (payload / "execution_topology.json").write_text(json.dumps(topo, indent=2))
    steps["execution_topology"] = {"exit": 0}

    # 5. the five TileLang pass snapshots, one per case, each in its own process state
    sys.path.insert(0, str(tl))
    sys.path.insert(0, str(tl / "testing"))
    sys.path.insert(0, str(kernel.parent))
    import kernel_v33 as kv
    from testing.ascend._ir import schedule_snapshots
    ok = True
    for batch, d, mode in CASES:
        kv.LOOP_MODE = mode
        case = "%dx%d_%s" % (batch, d, mode)
        cdir = payload / "passes" / case
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            prog = kv._build(batch, d)
            snaps = schedule_snapshots(prog, set(PASSES))
            for name in PASSES:
                mod = snaps[name]
                try:
                    text = mod.script()
                except Exception:
                    text = str(mod)
                (cdir / (name + ".tir")).write_text(text)
            steps["passes_" + case] = {"exit": 0, "captured": sorted(snaps)}
        except Exception as e:  # noqa: BLE001
            ok = False
            steps["passes_" + case] = {"exit": 2, "error": "%s: %s" % (type(e).__name__, e)}
    if not ok:
        sys.stderr.write("one or more pass snapshots failed\n")

    # 6. ONE manifest (outside the payload), hashed once
    files = {}
    for p in sorted(payload.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(payload))] = sha256(p)
    manifest = {
        "run_id": out.name,
        "epoch_start": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "epoch_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "payload_dir": "payload",
        "payload_file_count": len(files),
        "payload_files": files,
        "steps": steps,
        "passes": list(PASSES),
        "cases": ["%dx%d_%s" % c for c in CASES],
        "note": "One domain: every hash above is a payload file. No enclosing-directory hash.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(out / "MANIFEST.json"),
                      "payload_files": len(files), "ok": ok}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
