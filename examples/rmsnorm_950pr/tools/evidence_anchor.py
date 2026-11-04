#!/usr/bin/env python3
"""Record an evidence ANCHOR with an EXPLICIT corpus domain, so no future audit can mix
epochs OR conflate a whole-directory hash with the payload corpus hash.

A payload directory may contain a root metadata file (e.g. _ARCHIVE_MANIFEST.json) that is
NOT part of the measured payload. Hashing the whole directory then differs from hashing the
payload. This tool reports BOTH, explicitly labelled:
  * payload_corpus_sha256     -- PAYLOAD files only, EXCLUDING root metadata
  * whole_dir_sha256_present  -- EVERY file present (may include root metadata)
Never present whole_dir as the payload corpus hash.

Exit 0 ok, 2 missing cases dir / inconsistent env across cases.
"""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, sys

ROOT_METADATA = {"_ARCHIVE_MANIFEST.json"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(base, exclude_root_metadata):
    payload, whole = [], []
    for p in sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, base)
        h = sha256_file(p)
        whole.append((rel, h))
        if exclude_root_metadata and os.sep not in rel and rel in ROOT_METADATA:
            continue
        payload.append((rel, h))
    pd = hashlib.sha256(("\n".join("%s %s" % (r, h) for r, h in payload)).encode()).hexdigest()
    wd = hashlib.sha256(("\n".join("%s %s" % (r, h) for r, h in whole)).encode()).hexdigest()
    return payload, pd, whole, wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    d = args.cases_dir
    if not os.path.isdir(d):
        print("no cases dir: " + d, file=sys.stderr); return 2
    cases = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    infos, mt = {}, []
    for c in cases:
        di = os.path.join(d, c, "device_info.json")
        if os.path.isfile(di):
            infos[c] = json.load(open(di)); mt.append(os.stat(di).st_mtime)
    if not infos:
        print("no device_info.json in any case", file=sys.stderr); return 2
    pyexes = {v.get("python_exe") for v in infos.values()}
    ffis = {v.get("tvm_ffi") for v in infos.values()}
    tos = {v.get("torch") for v in infos.values()}
    npus = {v.get("torch_npu") for v in infos.values()}
    if len(pyexes) != 1 or len(ffis) != 1:
        print("INCONSISTENT env across cases: pyexe=%s ffi=%s" % (pyexes, ffis), file=sys.stderr)
        return 2
    payload, pd, whole, wd = digest(d, exclude_root_metadata=True)
    excluded_present = sorted(rel for rel, _ in whole if os.sep not in rel and rel in ROOT_METADATA)
    run_id = args.run_id
    rid = os.path.join(d, "RUN_ID.txt")
    if run_id is None and os.path.isfile(rid):
        run_id = open(rid).read().strip()
    anchor = {
        "run_id": run_id or ("unversioned-" + pd[:12]),
        "epoch_start": datetime.datetime.fromtimestamp(min(mt)).isoformat(),
        "epoch_end": datetime.datetime.fromtimestamp(max(mt)).isoformat(),
        "env_path": os.path.dirname(os.path.dirname(list(pyexes)[0])),
        "python_exe": list(pyexes)[0],
        "torch": list(tos)[0], "torch_npu": list(npus)[0], "tvm_ffi": list(ffis)[0],
        "cases": cases,
        "payload_domain": ("%d payload files, EXCLUDING root metadata %s"
                           % (len(payload), sorted(ROOT_METADATA))),
        "payload_file_count": len(payload),
        "payload_corpus_sha256": pd,
        "whole_dir_file_count_present": len(whole),
        "whole_dir_sha256_present": wd,
        "root_metadata_present": excluded_present,
        "note": ("payload_corpus_sha256 EXCLUDES root metadata; whole_dir_sha256_present "
                 "INCLUDES anything present. Never report whole_dir as the payload corpus hash."),
    }
    open(args.out, "w").write(json.dumps(anchor, indent=2))
    print(json.dumps(anchor, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
