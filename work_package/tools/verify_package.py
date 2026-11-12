#!/usr/bin/env python3
"""Verify package integrity and contracts without importing TF, Torch or NPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from next_tasks import load_graph


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    fixture_manifest = json.loads((ROOT/'fixtures/source_manifest.json').read_text())
    for f in fixture_manifest['files']:
        path = ROOT/f['path']
        assert path.is_file(), f['path']
        assert len(path.read_bytes()) == f['bytes'], f['path']
        assert digest(path) == f['sha256'], f['path']
    matrix = json.loads((ROOT/'cases/case_matrix.json').read_text())
    rows = matrix['cases']
    expected = {f'level4/{case}_{i}' for case in ('attention_residual','mla') for i in range(1,21)}
    assert len(rows) == 40 and {r['key'] for r in rows} == expected
    for r in rows:
        seed = int.from_bytes(hashlib.sha256(r['key'].encode()).digest()[:8], 'big') % (2**31)
        assert r['seed'] == seed, r['key']
    graph = load_graph()
    gates = json.loads((ROOT/'plan/GATES.json').read_text())['gates']
    assert {t for g in gates for t in g['required_tasks']} == set(graph)
    assert len(gates) == 8
    python_files = sorted(ROOT.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    manifest_path = ROOT/'PACKAGE_MANIFEST.json'
    checked_manifest = False
    if manifest_path.exists():
        files = json.loads(manifest_path.read_text())['files']
        for f in files:
            assert digest(ROOT/f['path']) == f['sha256'], f['path']
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.name != 'PACKAGE_MANIFEST.json'}
        assert actual == {f['path'] for f in files}, 'manifest file set differs'
        checked_manifest = True
    result = {'status':'PASS', 'scope':'package integrity, AST syntax, fixture identity, task DAG; no NPU',
              'fixture_files':len(fixture_manifest['files']), 'original_cases':len(rows),
              'tasks':len(graph), 'gates':len(gates), 'python_files_syntax_checked':len(python_files),
              'package_manifest_checked':checked_manifest}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
