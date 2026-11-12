"""CPU-only evolving refusal matrix. Missing fixture or stub is MISSING, never PASS.
Run from repository root; use prescribed interpreter/environment. No device imports.
"""
import argparse
import importlib
import inspect
import json
from collections import Counter
from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry_ascend import Ascend950PRTarget

# Each unavailable case is an explicit construction recipe, not fabricated test evidence.
CASES = []
def add(key, construction, module, function):
    CASES.append(dict(id=key, construction=construction, module=module, function=function))

for key,recipe in [
    ('solver_unsat','真实双任务正延迟环且 distance=0；UNSAT 的 starts/stages/phases/versions/issue_order 全空'),
    ('solver_unknown_install','真实 deadline 过期 Assignment UNKNOWN 交给 resolve/install；不得产生可安装候选'),
    ('solver_fixed_v','同一真实 buffer Fixed(3) 求 SAT；赋值和 resolve 后 V 均为 3'),
    ('solver_authored_stage','真实 task authored stage=2；SAT stage 必须仍为 2'),
    ('solver_no_offset','enable_offset=False 且 authored stage=2；构造/分析拒绝'),
    ('solver_capacity','真实 buffer Fixed(3)*64 bytes，usable=128；UNSAT/具名容量拒绝'),
    ('solver_deadline','真实最小 problem + monotonic()-1；UNKNOWN 且无赋值'),
]: add(key,recipe,'tilefoundry_ascend.solver','solve_region')
for field in ('II','stage','phase','V','issue_rank'):
    add('plan_'+field,'先取得真实可行 ResolvedPlan，replace region/task/buffer 的 '+field+' 后交下游校验',
        'tilefoundry_ascend.plan.mainline','verify_actual_program')
for field in ('operand','attribute','guard','multiplicity'):
    add('apply_'+field,'先真实 apply 得私有候选/source_map，再改变 '+field+'；按原 entry 和 source_map 复核',
        'tilefoundry_ascend.plan.mainline','verify_actual_program')
for field in ('tail_wait','generation','slot','cross_core','ready_free_pool'):
    add('completion_'+field,'先构建真实 SymbolicCompletionPlan，再删除/篡改 '+field+'；verify_completion',
        'tilefoundry_ascend.completion','verify_completion')
for field in ('capacity','layout','stride','scratch_double_charge'):
    add('storage_'+field,'真实已 resolve 版本及 placement，改变 '+field+'；verify_storage_current/重新 placement',
        'tilefoundry_ascend.storage','verify_storage_current')
for field in ('operand','constant','slot','guard','source_binding'):
    add('tl_'+field,'先 emitter + reader 得真实 TL/parsed/manifest，再单改 '+field+'；read_actual_tl',
        'tilefoundry_ascend.codegen','read_actual_tl')
for key,recipe in [('tl_config_missing','NativeArtifact.pass_configs 删除第二键'),
                   ('tl_config_value','NativeArtifact.pass_configs 第一键改 True'),
                   ('install_before_verified','真实 Module/PrimFunction + conformance=None'),
                   ('install_foreign','不同 Module 中同名 fresh PrimFunction + 未验证 conformance'),
                   ('install_same_name_different','同名 PrimFunction 改 target aiv_count，未验证即安装'),
                   ('export_cached_false','metadata export_gate 塞缓存 False/空 gaps；实际 require_export_ready 仍须重算拒绝')]:
    add(key,recipe,'tilefoundry_ascend.passes.p3_chain' if key.startswith(('tl_','export_')) else
        'tilefoundry_ascend.plan.mainline','require_export_ready' if key.startswith(('tl_','export_')) else 'install_entry')

def fresh():
    target=Ascend950PRTarget()
    entry=PrimFunction('k',(),Sequential(()),output_count=0,target=target)
    return Module('matrix',(entry,),entry='k',target=target)

def location(fn):
    return dict(file=inspect.getsourcefile(fn),line=inspect.getsourcelines(fn)[1])

def missing_case(case):
    try:
        module=importlib.import_module(case['module'])
        fn=getattr(module,case['function'],None)
        if fn is None:
            return dict(status='MISSING',detail='API absent: '+case['function'])
        src=inspect.getsource(fn)
        if 'raise MainlineNotImplemented(' in src or 'raise SolverNotImplemented(' in src:
            return dict(status='MISSING',detail='API is unconditional NOT_IMPLEMENTED stub',**location(fn))
        return dict(status='MISSING',detail='API changed/exists; actual production fixture adapter not yet bound; rerun after adapter implementation',**location(fn))
    except (ImportError,AttributeError) as e:
        return dict(status='MISSING',detail=type(e).__name__+': '+str(e))

# Only these already have real boundary inputs. Others wait for typed production fixtures.
def active_case(case):
    key=case['id']
    if not key.startswith(('install_','export_','tl_config_')):
        return missing_case(case)
    from tilefoundry_ascend.plan.mainline import install_entry, MainlineNotImplemented, PlanError
    from tilefoundry_ascend.passes.p3_chain import (
        require_export_ready, ExportRefusal, ChainRecord, ir_identity, _function_identity,
        CHAIN_KEY, TL_ARTIFACT_KEY, EXPORT_GATE_KEY,
    )
    from tilefoundry_ascend.codegen import NativeArtifact,PASS_CONFIGS
    m=fresh(); original=m.functions; metadata=dict(m.metadata)
    try:
        if key.startswith('install_'):
            candidate=replace(m.functions[0]) if key!='install_foreign' else fresh().functions[0]
            if key=='install_same_name_different':
                candidate=replace(candidate,target=Ascend950PRTarget(aiv_count=1))
            expected='INSTALL_CANDIDATE_UNVERIFIED'; fn=install_entry
            install_entry(m,candidate)
        else:
            fn=require_export_ready
            if key.startswith('tl_config_'):
                configs=PASS_CONFIGS[:1] if key=='tl_config_missing' else ((PASS_CONFIGS[0][0],True),PASS_CONFIGS[1])
                m.metadata[CHAIN_KEY]=ChainRecord(ir_identity(m),entry_identity=_function_identity(m.functions[0]))
                m.metadata[TL_ARTIFACT_KEY]=NativeArtifact(pass_configs=configs)
                expected='PASS_CONFIGS_MISMATCH'
            else:
                m.metadata[EXPORT_GATE_KEY]={'ready':False,'gaps':[]}
                expected='CHAIN_NOT_RUN'
            metadata=dict(m.metadata)
            require_export_ready(m)
    except MainlineNotImplemented as e:
        return dict(status='MISSING',detail=str(e),**location(fn))
    except (ExportRefusal,PlanError) as e:
        codes=list(getattr(e,'missing',())) or [getattr(e,'code','')]
        unchanged=m.functions is original and m.metadata==metadata
        return dict(status='REFUSED' if expected in codes and unchanged else '意外通过',
            exception=type(e).__name__,detail=str(e),codes=codes,unchanged=unchanged,**location(fn))
    return dict(status='意外通过',detail='bad input returned normally',**location(fn))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case',nargs='?')
    args=parser.parse_args()
    selected=[c for c in CASES if args.case is None or c['id']==args.case]
    if not selected:
        parser.error('unknown case id')
    counts=Counter()
    for case in selected:
        try:
            result=dict(case,**active_case(case))
        except Exception as e:
            # API drift and fixture failures never become successful refusal evidence.
            result=dict(case,status='MISSING',detail='probe fixture/API error: '+type(e).__name__+': '+str(e))
        counts[result['status']]+=1
        print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    print(json.dumps(dict(summary=dict(counts)),ensure_ascii=False,sort_keys=True))
    return 1 if counts['意外通过'] else 0

if __name__=='__main__':
    raise SystemExit(main())
