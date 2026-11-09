"""Validate per-cell policies and resume without rerunning completed cells."""
import os,subprocess,shutil,ast
from full_repair_controller import *
def main():
    audit=SOURCE/'00_Audit/FULL_REPAIR02';audit.mkdir(parents=True,exist_ok=True)
    gate=read(audit/'ETTh1_192/CANONICAL_REPRODUCTION.json')
    assert gate['status']=='PASS' and len(gate['rows'])==4
    assert all(r['units']==2689*7 for r in gate['rows'])
    assert all(r['max_stored_prediction_difference']==0 for r in gate['rows'])
    assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
    assert read(OUT/'FULL/ETTh1_96/DELIVERABLE_GATE.json')['status']=='PASS'
    # Preserve failed controller/export evidence; no previous source/result is edited.
    archive=audit/'failed_20261030_2252';archive.mkdir(exist_ok=True)
    for p in [SOURCE/'00_Audit/CONTROLLER_STATUS.json',SOURCE/'00_Audit/IMMEDIATE_ISSUE.json',SOURCE/'full_v0_canonical_scores/ETTh1_192/PROVENANCE.json',SOURCE/'logs/FULL_V0_CANONICAL_ETTh1_192.log']:
        dst=archive/(p.parent.name+'_'+p.name)
        if not dst.exists():shutil.copy2(p,dst)
    for name,h in read(ORIGINAL/'READY_GATE.json')['source_sha256'].items():assert sha(ORIGINAL/name)==h
    assert read(queue.HERE/'READY_GATE.json')['source_hashes']==queue.source_hashes()
    first=ROOT/'experiments/retrieval_full_precision_repair01'
    assert read(first/'READY_GATE.json')['source_hashes']=={p.name:sha(p) for p in first.glob('*.py')}
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    tests=[]
    for label,py in [('torch2',old.PY),('legacy',old.LEGACY)]:
        cmd=[py,'-m','pytest','-q','-p','no:cacheprovider',str(ORIGINAL/'test_diagnostic_core.py'),str(ORIGINAL/'test_diagnostic_runtime.py'),str(HERE/'test_full_precision_policy.py')]
        a=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=audit/f'tests_{label}.log';log.write_text(a.stdout)
        assert a.returncode==0 and '51 passed' in a.stdout,a.stdout
        tests.append(dict(runtime=label,count=51,log=str(log),sha256=sha(log)))
    dump(HERE/'READY_GATE.json',dict(status='PASS',at=old.now(),source_hashes=hashes(),tests=tests,actual_gpu_gate=str(audit/'ETTh1_192/CANONICAL_REPRODUCTION.json'),actual_gpu_gate_sha256=sha(audit/'ETTh1_192/CANONICAL_REPRODUCTION.json'),all_preexisting_sources_unchanged=True))
    report='''# Full per-cell canonical runtime correction\n\nH96 V0 precision was incorrectly generalized to every Full checkpoint. H192 export stopped at validation batch0/channel0, max prediction difference0.31112647, before writing any scores. The failed attempt and controller evidence are preserved.\n\nValidation-only fixed-first-batch audits of all remaining Full V0 checkpoints identify: ETTh1/Weather H192,H336 -> original Torch1 with matmul TF32=True; ETTh1/Weather H96,H720 -> Torch2 TF32=False score bridge. V5 retains its corrected canonical Torch1 TF32=True path. No checkpoint, loss, support, sample, tolerance, metric or future-dependent selection was changed. Runtime policy is fixed per known cell, not globally by architecture.\n\nH192 V0/V5 were checked on2689queries x7channels in each validation/test split, bank8257. Both models reproduce all stored predictions with maximum difference0, plus all canonical test metrics pass. Remaining cells still must pass their own full validation/test reproduction gate; the first-batch audits are provisional only.\n\nCompleted E96 and all12 P100 diagnostics are reused and never recomputed. Repair01 source hashes remain unchanged, version02 is isolated. The queue proceeds E192 -> E336 -> E720 -> Weather96/192/336/720 -> Solar independent baseline/Oracle diagnostics, with absent Full-trained Solar models kept missing. Full12 plus final summary gates are required before Solar8 waiter is released.\n'''
    (audit/'IMPLEMENTATION_REPAIR.md').write_text(report)
    assert not old.gpu_apps(),'GPU1 busy; no duplicate work'
    for p in Path('/proc').iterdir():
        if p.name.isdigit():assert not any(x.endswith('/full_repair_controller.py') for x in queue.command(int(p.name))), 'Another Full controller exists'
    log=audit/'controller_repaired.log'
    with log.open('a') as f:proc=subprocess.Popen([old.PY,'-u',str(HERE/'full_repair_controller.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump(audit/'ACTIVATION.json',dict(status='STARTED',controller_pid=proc.pid,log=str(log),at=old.now(),original_failures_preserved=str(archive),completed_full_cells_reused=['ETTh1_96']))
    print('ACTIVATED',proc.pid,str(log),flush=True)
if __name__=='__main__':main()
