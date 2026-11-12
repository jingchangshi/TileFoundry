#!/usr/bin/env python3
"""List task readiness. Never infer completion from test counts or edit state."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_graph():
    rows = json.loads((ROOT/'plan/TASKS.json').read_text())['tasks']
    graph = {r['id']: r for r in rows}
    if len(graph) != len(rows):
        raise ValueError('duplicate task ID')
    done, active = set(), set()
    def visit(n):
        if n in active: raise ValueError('cyclic task dependency: '+n)
        if n in done: return
        if n not in graph: raise ValueError('unknown dependency: '+n)
        active.add(n)
        for d in graph[n]['depends_on']: visit(d)
        active.remove(n); done.add(n)
    for n in graph: visit(n)
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=ROOT/'templates/progress.json')
    args = p.parse_args()
    graph = load_graph()
    state = json.loads(args.state.read_text())['tasks']
    if set(state) != set(graph): raise ValueError('state must cover exact task ID set')
    allowed = {'NOT_STARTED', 'IN_PROGRESS', 'BLOCKED', 'FAILED', 'PASS'}
    for n, row in state.items():
        if row['status'] not in allowed: raise ValueError('invalid status for '+n)
        if row['status'] == 'PASS' and not row.get('evidence'):
            raise ValueError('PASS needs concrete evidence: '+n)
    ready, running, blocked = [], [], []
    for n, row in graph.items():
        if state[n]['status'] == 'PASS': continue
        summary = {'id': n, 'phase': row['phase'], 'title': row['title'], 'owner': row['owner']}
        unmet = [d for d in row['depends_on'] if state[d]['status'] != 'PASS']
        if state[n]['status'] == 'IN_PROGRESS': running.append(summary)
        elif unmet or state[n]['status'] == 'BLOCKED': blocked.append({**summary, 'unmet': unmet})
        else: ready.append(summary)
    print(json.dumps({'ready': ready, 'in_progress': running,
                      'blocked_count': len(blocked), 'completed': sum(r['status']=='PASS' for r in state.values())},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
