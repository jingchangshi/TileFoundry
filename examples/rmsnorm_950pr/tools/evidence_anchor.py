#!/usr/bin/env python3
"""Record an evidence ANCHOR so no future audit can mix epochs.

Anchors bind a corpus to: a run id, an epoch (mtimes), the env path and python_exe that
produced it, the torch/torch_npu/tilelang/tvm_ffi versions from device_info, and a
sha256 over the ordered (path, sha256) list of every file in the corpus. Two runs can
never be silently compared again: anchor(two runs) differ in run_id AND corpus digest.

Exit 0 on success, 2 on missing cases dir / inconsistent env across cases.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None, help="override; else RUN_ID.txt or derived")
    args = ap.parse_args()
    d = args.cases_dir
    if not os.path.isdir(d):
        print("no cases dir: " + d, file=sys.stderr); return 2
    cases = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    infos = {}
    mt = []
    for c in cases:
        di = os.path.join(d, c, "device_info.json")
        if os.path.isfile(di):
            infos[c] = json.load(open(di))
            mt.append(os.stat(di).st_mtime)
    if not infos:
        print("no device_info.json in any case", file=sys.stderr); return 2
    # environment consistency across cases
    pyexes = {v.get("python_exe") for v in infos.values()}
    ffis = {v.get("tvm_ffi") for v in infos.values()}
    tos = {v.get("torch") for v in infos.values()}
    npus = {v.get("torch_npu") for v in infos.values()}
    if len(pyexes) != 1 or len(ffis) != 1:
        print("INCONSISTENT env across cases: pyexe=%s ffi=%s" % (pyexes, ffis), file=sys.stderr)
        return 2
    # corpus digest over ordered relative path + sha256
    files = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            files.append((os.path.relpath(p, d), sha256_file(p)))
    corpus_digest = hashlib.sha256(
        ("\n".join("%s %s" % (rel, h) for rel, h in files)).encode()).hexdigest()
    run_id = args.run_id
    rid_file = os.path.join(d, "RUN_ID.txt")
    if run_id is None and os.path.isfile(rid_file):
        run_id = open(rid_file).read().strip()
    epoch_lo = min(mt); epoch_hi = max(mt)
    import datetime
    anchor = {
        "run_id": run_id or ("unversioned-" + hashlib.sha256(corpus_digest.encode()).hexdigest()[:12]),
        "epoch_start": datetime.datetime.fromtimestamp(epoch_lo).isoformat(),
        "epoch_end": datetime.datetime.fromtimestamp(epoch_hi).isoformat(),
        "env_path": os.path.dirname(os.path.dirname(list(pyexes)[0])) if list(pyexes)[0] else None,
        "python_exe": list(pyexes)[0],
        "torch": list(tos)[0], "torch_npu": list(npus)[0], "tvm_ffi": list(ffis)[0],
        "cases": cases,
        "file_count": len(files),
        "corpus_sha256": corpus_digest,
        "note": "bind a corpus to one epoch; two anchors with different run_id AND corpus_sha256 MUST NOT be compared",
    }
    open(args.out, "w").write(json.dumps(anchor, indent=2))
    print(json.dumps(anchor, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
