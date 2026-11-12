"""Real-object incremental negative probes; CPU only, no implementation monkeypatch."""
import argparse
import json
from dataclasses import dataclass, replace
from collections import Counter
from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.stmt import Stmt
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.stmts import Sequential, Evaluate, If
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.plan.problem import analyze_native_problem, _coverage
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.resolved import resolve_schedule
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.storage import place_resolved_versions, verify_storage_current
from tilefoundry_ascend.plan.mainline import install_entry
from tilefoundry_ascend.plan.identity import function_identity
from tilefoundry_ascend.plan.verify import NativeVerdict
from tilefoundry_ascend.passes.p3_chain import require_export_ready
from tilefoundry_ascend.verify.executor import run_entry

AUTHOR='targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline'
parser=argparse.ArgumentParser();parser.add_argument('case',nargs='?');args=parser.parse_args()
program=checked_program(AUTHOR,None,command='incremental refusal probes')
analyzed=analyze(program.module,program.entry,analysis=('compute-cost','memory'))
module=finalize(analyzed,authored=program.authored)
tree=build_native_unit_tree(module,authored=program.authored)
problem=analyze_native_problem(tree,None,None,module.entry_function().target)
root=problem.regions[0]
single=replace(problem,regions=(root,))
assignment=solve_region(to_schedule_problem(root),None)
assert assignment.status=='SAT',assignment
plan=resolve_schedule(single,assignment)
target=module.entry_function().target
storage=place_resolved_versions(plan,None,target,None)
assert verify_storage_current(plan,storage,target).verified
assert problem.coverage.complete
results=[]

def record(key,fn,expected,kind='exception'):
    if args.case and args.case!=key:return
    try:
        result=fn()
    except Exception as e:
        code=getattr(e,'code',None)
        results.append(dict(id=key,status='REFUSED' if code==expected else 'MISSING',
            exception=type(e).__name__,code=code,detail=str(e),expected=expected))
    else:
        if kind=='coverage':
            codes=[r.code for r in result.refusals]
            ok=result.complete is False and expected in codes
            results.append(dict(id=key,status='REFUSED' if ok else '意外通过',complete=result.complete,codes=codes,expected=expected))
        elif kind=='gate':
            results.append(dict(id=key,status='REFUSED' if result['ready'] is False else '意外通过',verdict=result))
        else:
            results.append(dict(id=key,status='意外通过',expected=expected,returned=type(result).__name__))

tid=root.tasks[0].task_id
for key,change,code in [
 ('plan_phase',dict(phases={**assignment.phases,tid:assignment.ii}), 'PLAN_PHASE_OUT_OF_RANGE'),
 ('plan_ii',dict(ii=assignment.ii+1000),'PLAN_II_OUT_OF_DOMAIN'),
 ('plan_start',dict(starts={**assignment.starts,tid:assignment.starts[tid]+1}),'PLAN_START_INCONSISTENT'),
 ('plan_fixed_v',dict(versions={**assignment.versions,root.buffers[0].buffer_id:2}),'PLAN_FIXED_VERSION_CHANGED'),
]:record(key,lambda change=change:resolve_schedule(single,replace(assignment,**change)),code)
ids=tuple(t.task_id for t in root.tasks)
assert len(ids)>=2
ordered_root=replace(root,issue_after=((ids[0],ids[1]),),manual_pipe_order=(),edges=(),reuse=())
same=replace(assignment,phases={k:0 for k in ids},stages={k:0 for k in ids},starts={k:0 for k in ids},issue_order=tuple(reversed(ids)))
record('plan_issue_reverse',lambda:resolve_schedule(replace(single,regions=(ordered_root,)),same),'PLAN_ISSUE_ORDER_VIOLATED')

for key,mode,code in [
 ('coverage_delete','delete','SOURCE_COVERAGE_UNITS_UNMAPPED'),
 ('coverage_duplicate','duplicate','SOURCE_COVERAGE_TASK_DUPLICATE'),
 ('coverage_clone_unit','clone','SOURCE_COVERAGE_TASK_FOREIGN'),
 ('coverage_wrong_region','region','SOURCE_COVERAGE_TASK_REGION_MISMATCH'),
]:
    regions=list(problem.regions); first=regions[0]; task=first.tasks[0]
    if mode=='delete':regions[0]=replace(first,tasks=first.tasks[1:])
    if mode=='duplicate':regions[0]=replace(first,tasks=first.tasks+(task,))
    if mode=='clone':regions[0]=replace(first,tasks=(replace(task,unit_ref=replace(task.unit_ref)),)+first.tasks[1:])
    if mode=='region':
        regions[0]=replace(first,tasks=first.tasks[1:])
        regions[1]=replace(regions[1],tasks=regions[1].tasks+(task,))
    record(key,lambda regions=regions:_coverage(tree,tuple(regions),'probe-mutated'),'{}'.format(code),'coverage')

entry=storage.entries[0]
for key,field,delta,code in [('storage_offset','base_offset_bytes',64,'STORAGE_PLACEMENT_MISMATCH'),('storage_bytes','allocation_bytes',256,'STORAGE_BYTES_MISMATCH')]:
    changed=replace(entry,**{field:getattr(entry,field)+delta})
    modified=replace(storage,entries=(changed,)+storage.entries[1:])
    record(key,lambda modified=modified:verify_storage_current(plan,modified,target),code)
for key,space,shape,code in [('storage_capacity','smem',(131072,),'STORAGE_CAPACITY_EXCEEDED'),('storage_rmem','rmem',(64,),'UNKNOWN_CAPACITY'),('storage_carrier','carrier',(64,),'UNKNOWN_CAPACITY')]:
    old=next(b for b in plan.regions[0].buffers if b.buffer_ref is entry.root)
    var=Var('probe_'+space,type=make_tensor_type(shape,DType.f32,space))
    mutated=replace(old,buffer_ref=var)
    rp=replace(plan.regions[0],buffers=tuple(mutated if b is old else b for b in plan.regions[0].buffers))
    changed=replace(plan,regions=(rp,))
    record(key,lambda changed=changed:place_resolved_versions(changed,None,target,None),code)

@dataclass(unsafe_hash=True)
class UnknownStmt(Stmt):pass
@dataclass(unsafe_hash=True)
class UnknownExpr(Expr):pass
class UnknownInstruction:pass
# Real PrimFunction+Var+Stmt/Expr nodes; unknown subclasses have no registry entries.
x=Var('x',type=make_tensor_type((1,),DType.f32,'gmem'))
y=Var('y',type=x.type)
base=replace(module.entry_function(),name='probe_exec',params=(x,y),output_count=1)
for key,stmt,code in [
 ('execution_instruction',Evaluate(callable=UnknownInstruction(),args=()),'EXEC_INSTRUCTION_UNSUPPORTED'),
 ('execution_statement',UnknownStmt(),'EXEC_STATEMENT_UNSUPPORTED'),
 ('execution_expression',If(UnknownExpr(type=x.type),Sequential(()),Sequential(())),'EXEC_EXPR_UNSUPPORTED'),
]:
    fn=replace(base,body=Sequential((stmt,)))
    record(key,lambda fn=fn:run_entry(fn,{'x':[1.0]}),code)
record('export_false',lambda:require_export_ready(module,raising=False),None,'gate')
record('install_no_verdict',lambda:install_entry(module,replace(module.entry_function())), 'INSTALL_CANDIDATE_UNVERIFIED')
# Highest priority attack: forge the public verdict constructor without calling verifier.
g=replace(module.entry_function(),target=replace(target,aiv_count=1))
candidate=replace(module,functions=(g,))
v=NativeVerdict(status='VERIFIED',original_module_ref=module,candidate_module_ref=candidate,
    original_entry_ref=module.entry_function(),candidate_entry_ref=g,
    original_entry_identity=function_identity(module.entry_function()),candidate_entry_identity=function_identity(g))
record('install_forged_verified',lambda:install_entry(module,g,conformance=v),'INSTALL_CANDIDATE_UNVERIFIED')
for r in results:print(json.dumps(r,ensure_ascii=False,sort_keys=True))
summary=Counter(r['status'] for r in results)
print(json.dumps(dict(summary=dict(summary),baseline_real_tasks=len(tree.units),baseline_sat=assignment.status),ensure_ascii=False))
if args.case and not results:parser.error('unknown case')
raise SystemExit(1 if summary['意外通过'] else 0)
