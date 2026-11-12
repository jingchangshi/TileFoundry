"""CPU-only adversarial probes; run from repository root with prescribed env.
No compiler implementation is monkeypatched. JSONL records observed exceptions/gaps.
"""
import copy
import importlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry_ascend import Ascend950PRTarget
from tilefoundry_ascend.codegen import NativeArtifact, PASS_CONFIGS, emit_supported_tl
from tilefoundry_ascend.completion import CompletionPlan
from tilefoundry_ascend.storage import StoragePlan, place_resolved_versions
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.plan.identity import function_identity
from tilefoundry_ascend.plan.mainline import (
    SINGLE_TRANSACTION, compile_mainline, apply_symbolic_pipeline,
    verify_actual_program, install_entry,
)
from tilefoundry_ascend.passes.p3_chain import (
    CHAIN_KEY, PLAN_KEY, COMPLETION_KEY, STORAGE_KEY, TL_ARTIFACT_KEY,
    TL_CONFORMANCE_KEY, ChainRecord, ir_identity, require_export_ready,
    RunAscendNativeMainline,
)

ONLY = sys.argv[1] if len(sys.argv) > 1 else None

def emit(name, result):
    if ONLY is None or ONLY == name:
        print(json.dumps(dict(probe=name, **result), sort_keys=True))

def fresh():
    target = Ascend950PRTarget()
    fn = PrimFunction('k', (), Sequential(()), output_count=0, target=target)
    return Module('probe', (fn,), entry='k', target=target)

def candidate():
    m = fresh()
    fn = m.functions[0]
    r = ChainRecord(ir_identity(m), entry_identity=function_identity(fn),
        plan_identity='p', completion_identity='c', storage_identity='s',
        tl_identity='tl', pipeline_status='APPLIED', applied=SINGLE_TRANSACTION,
        stopped_at='', reason='')
    parsed = NS(params=(), body=object())
    art = NativeArtifact(source='arbitrary non-TL text', abi=NS(input_count=0, output_count=0),
        parsed_prim_func=parsed, source_identity='src', plan_identity='p',
        tir_identity='tir', tl_identity='tl', target_identity='target', toolchain_identity='tool')
    m.metadata.update({CHAIN_KEY:r, PLAN_KEY:NS(identity='p'),
        COMPLETION_KEY:NS(identity='c'), STORAGE_KEY:NS(identity='s'), TL_ARTIFACT_KEY:art,
        TL_CONFORMANCE_KEY:NS(verified=True, parsed_prim_func=parsed,
            entry_identity=r.entry_identity, plan_identity='p', tl_identity='tl')})
    return m

def gate(name, m, expected=None):
    before = (ir_identity(m), copy.copy(m.metadata), m.functions)
    v = require_export_ready(m, raising=False)
    try:
        require_export_ready(m)
    except Exception as e:
        result = dict(exception=type(e).__name__, detail=str(e), code=getattr(e,'code',None),
            missing=list(getattr(e,'missing',())), ready=v['ready'])
    else:
        result = dict(exception=None, code=None, missing=[], ready=v['ready'])
    result['unchanged'] = before == (ir_identity(m), m.metadata, m.functions)
    if expected:
        result['expected_present'] = expected in result['missing']
    emit(name, result)

m = fresh()
gate('no_chain', m, 'CHAIN_NOT_RUN')
for label,key in [('plan',PLAN_KEY),('completion',COMPLETION_KEY),('storage',STORAGE_KEY)]:
    for kind in ('dict','empty_object','mismatch','matching_object'):
        m = candidate()
        ident = {'plan':'p','completion':'c','storage':'s'}[label]
        m.metadata[key] = ({'identity':ident} if kind=='dict' else
            NS() if kind=='empty_object' else NS(identity='wrong') if kind=='mismatch' else NS(identity=ident))
        expected = label.upper() + ('_IDENTITY_MISMATCH' if kind=='mismatch' else '_ARTIFACT_NOT_ACTUAL')
        gate(label+'_'+kind, m, None if kind=='matching_object' else expected)

for field,value,expected in [
    ('source','', 'TL_SOURCE_MISSING'),
    ('pass_configs', PASS_CONFIGS[:1], 'PASS_CONFIGS_MISMATCH'),
    ('abi', {}, 'ABI_NOT_ACTUAL'),
    ('abi', NS(input_count=1,output_count=0), 'ABI_INPUT_COUNT_MISMATCH'),
    ('abi', NS(input_count=0,output_count=1), 'ABI_OUTPUT_COUNT_MISMATCH'),
    ('parsed_prim_func',None,'PARSED_PRIMFUNC_MISSING'),
    *[(f,'','ARTIFACT_'+f.upper()+'_MISSING') for f in
      ('source_identity','plan_identity','tir_identity','tl_identity','target_identity','toolchain_identity')],
]:
    m=candidate()
    m.metadata[TL_ARTIFACT_KEY]=replace(m.metadata[TL_ARTIFACT_KEY], **{field:value})
    gate('native_'+expected.lower(),m,expected)
m=candidate();m.metadata[TL_ARTIFACT_KEY]={'source':'anything'}
gate('native_dict',m,'TL_ARTIFACT_NOT_ACTUAL')
for label,c in [('none',None),('dict',{}),('verified_dict',{'verified':True}),('object',NS(verified=True))]:
    m=candidate();m.metadata[TL_CONFORMANCE_KEY]=c
    gate('conformance_'+label,m,'TL_CONFORMANCE_NOT_ACTUAL' if label!='object' else 'TL_CONFORMANCE_PARSED_PRIMFUNC_MISMATCH')

for field,code in [('ir_identity','STALE_IR_IDENTITY'),('entry_identity','ENTRY_IDENTITY_MISMATCH')]:
    m=candidate();m.metadata[CHAIN_KEY]=replace(m.metadata[CHAIN_KEY],**{field:'tampered'})
    gate('tamper_'+field,m,code)
gate('all_structural_fields_forged',candidate(),'NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED')
m=candidate()
r=m.metadata[CHAIN_KEY]
m.metadata[CHAIN_KEY]=NS(**r.as_dict())
gate('chain_duck_record',m,'NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED')
# All identity strings except actual IR/entry identities are caller-invented and coordinated.
m=candidate();r=m.metadata[CHAIN_KEY]
r=replace(r,plan_identity='invented-plan',completion_identity='invented-completion',
    storage_identity='invented-storage',tl_identity='invented-tl')
m.metadata.update({CHAIN_KEY:r,PLAN_KEY:NS(identity=r.plan_identity),
    COMPLETION_KEY:NS(identity=r.completion_identity),STORAGE_KEY:NS(identity=r.storage_identity)})
a=m.metadata[TL_ARTIFACT_KEY]
m.metadata[TL_ARTIFACT_KEY]=replace(a,plan_identity=r.plan_identity,tl_identity=r.tl_identity,
    source_identity='invented-source',tir_identity='not-the-real-tir',
    target_identity='wrong-target',toolchain_identity='wrong-toolchain')
m.metadata[TL_CONFORMANCE_KEY]=NS(verified=True,parsed_prim_func=a.parsed_prim_func,
    entry_identity=r.entry_identity,plan_identity=r.plan_identity,tl_identity=r.tl_identity)
gate('coordinated_invented_identities',m,'NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED')
m=candidate();m.metadata[CHAIN_KEY]=replace(m.metadata[CHAIN_KEY],plan_identity=NS(identity='p'))
gate('record_identity_object',m,'PLAN_IDENTITY_MISMATCH')
m=fresh();m.metadata[CHAIN_KEY]={'pipeline_status':'APPLIED'}
try:
    require_export_ready(m)
except Exception as e:
    emit('chain_record_dict',dict(exception=type(e).__name__,code=getattr(e,'code',None),detail=str(e)))

for label,c in [('none',None),('dict',{}),('verified_dict',{'verified':True}),('object',NS(verified=True))]:
    m=fresh();before=(ir_identity(m),copy.copy(m.metadata),m.functions)
    fn=replace(m.functions[0])
    try:
        install_entry(m,fn,conformance=c)
    except Exception as e:
        emit('install_'+label,dict(exception=type(e).__name__,detail=str(e),code=getattr(e,'code',str(e)),
            fresh_candidate=fn is not m.functions[0],unchanged=before==(ir_identity(m),m.metadata,m.functions)))

for name in ('compile_mainline','apply_symbolic_pipeline','verify_actual_program',
             'emit_supported_tl','solve_region','place_resolved_versions'):
    m=fresh();fn=m.functions[0];p=NS(identity='p');c=CompletionPlan();s=StoragePlan()
    args={'compile_mainline':(m,), 'apply_symbolic_pipeline':(fn,p,c,s),
          'verify_actual_program':(fn,fn,p,c,s),'emit_supported_tl':(m,p,c,s,m.target),
          'solve_region':(p,), 'place_resolved_versions':(p,c,m.target,None)}[name]
    before=(ir_identity(m),copy.copy(m.metadata),repr(args))
    try:
        globals()[name](*args)
    except Exception as e:
        emit(name,dict(exception=type(e).__name__,detail=str(e),code=getattr(e,'code',str(e)),
            unchanged=before==(ir_identity(m),m.metadata,repr(args))))

m=Module('empty',(),target=Ascend950PRTarget())
gate('empty_no_chain',m,'CHAIN_NOT_RUN')
m.metadata[CHAIN_KEY]=ChainRecord(ir_identity(m),pipeline_status='APPLIED',applied=SINGLE_TRANSACTION,stopped_at='')
gate('empty_forged_chain',m,'ACTUAL_ENTRY_MISSING')
for label,m in [('minimal',fresh()),('empty',Module('empty',(),target=Ascend950PRTarget()))]:
    before=ir_identity(m)
    result=RunAscendNativeMainline().run(m)
    emit('pass_'+label,dict(same_module=result is m,ir_unchanged=ir_identity(m)==before,
        ready=m.metadata['ascend.export_gate']['ready'],
        reason=m.metadata[CHAIN_KEY].reason,metadata_keys=sorted(m.metadata)))

for name in ('tilefoundry_ascend.attnres.twin','tilefoundry_ascend.mla.kernel',
             'tilefoundry_ascend.completion.materialize', 'tilefoundry_ascend.codegen.native_tl',
             'tilefoundry_ascend.attnres.build','tilefoundry_ascend.mla.build'):
    try:
        mod=importlib.import_module(name)
    except Exception as e:
        emit('import_'+name,dict(exception=type(e).__name__,detail=str(e)))
    else:
        emit('import_'+name,dict(exception=None,file=mod.__file__))
emit('loaded_bypass_modules',dict(modules=sorted(n for n in sys.modules
    if '_archive_historical' in n or n=='tilelang' or n.startswith('tilelang.') or n=='torch_npu' or n.startswith('torch_npu.'))))
