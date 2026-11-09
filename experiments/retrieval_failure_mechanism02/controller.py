"""Durable one-new-worker queue; existing Solar coexistence explicitly allowed.

No process termination, no fixed sleep while GPU work competes for ownership,
no automatic numerical-tolerance relaxation. Each failed stage stops its queue.
"""
import argparse,fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
from protocol import ROOT,CELLS,MODELS,GPU_UUID,PROTOCOL
from io_utils import read,dump,sha,now,source_hashes
from completed_cells import freeze_runtime,verified_completed_cell
HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02';AUDIT=OUT/'00_Audit'
PY2='/data/pjh_workspace/ts-env/bin/python';PY1='/usr/bin/python3'
BRIDGE={'ETTh1_96','ETTh1_720','Weather_96','Weather_720'}

def runtime(cell,model,support):return PY2 if support=='P100' or (model=='V0' and cell in BRIDGE) or (model=='V1' and cell in {'ETTh1_96','ETTh1_720','Weather_96','Weather_720'}) else PY1
def targets():return [(c,m,s) for c in CELLS for s in ('P100','FULL') for m in MODELS]
def state(status,**kw):dump(AUDIT/'CONTROLLER_STATUS.json',dict(status=status,at=now(),controller_pid=os.getpid(),**kw))
def launch(command,log,target,phase):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state('RUNNING',phase=phase,target=target,worker_pid=p.pid,command=command,log=str(log),shared_gpu_authorized=True)
        code=p.wait()
    assert code==0,dict(target=target,phase=phase,returncode=code,log=str(log))

def run(mode):
    AUDIT.mkdir(parents=True,exist_ok=True);lock=(AUDIT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    test=read(AUDIT/'CPU_TEST_GATE.json');assert test['status']=='PASS' and test['runtime_hashes']==source_hashes()
    freeze_runtime(OUT,HERE)
    # Previous Solar completion prerequisite is intentionally superseded ONLY by
    # the explicit 06:48:12 parallel authorization; checkpoint/scientific gates stay.
    dump(AUDIT/'PARALLEL_AUTHORIZATION.json',dict(at=now(),source='Sentinel_c6a78d0f6bec819183319ae4b6c45973',
         authorization_time='2026-11-02T06:48:12Z',maximum_new_workers=1,physical_gpu=1,existing_solar_untouched=True,
         supersedes='Solar completion wait only; not canonical reproduction or resource gates'))
    jobs=[dict(cell=c,model=m,training_support=s,mode=mode,runtime=runtime(c,m,s),status='PENDING') for c,m,s in targets()]
    dump(AUDIT/'EXECUTION_QUEUE.json',dict(mode=mode,targets=jobs,parallel_authorization='06:48:12 explicit user',seed=0))
    for cell in CELLS:
        if verified_completed_cell(OUT,cell):
            state('PRESERVED_VERIFIED_COMPLETE_CELL',cell=cell,all_hashes_counts_and_sources_verified=True)
            continue
        for support in ('P100','FULL'):
            for model in MODELS:
                dest=OUT/'checkpoint_runs'/cell/(model+'_'+support);t=f'{cell}_{model}_{support}';py=runtime(cell,model,support)
                gate=dest/'REPRODUCTION_GATE.json'
                if not gate.exists():
                    launch([py,'-u',str(HERE/'coexist_smoke.py'),cell,model,support],OUT/'logs'/f'{t}_smoke.log',t,'COEXISTENCE_SMOKE')
                    launch([py,'-u',str(HERE/'worker.py'),cell,model,support,'--mode','gate'],OUT/'logs'/f'{t}_gate.log',t,'FULL_POPULATION_REPRODUCTION')
                g=read(gate);assert g['status']=='PASS' and g['identity']['source_hashes']==source_hashes()
                assert {r['split'] for r in g['rows']}=={'val','test'}
                assert all(r['ordered_mismatch_count']==0 and r['units']==r['queries']*r['channels'] for r in g['rows'])
        if mode=='all':
            ready=read(AUDIT/'ANALYSIS_CPU_READINESS.json')
            assert ready['status']=='PASS' and ready['runtime_hashes']==source_hashes()
            assert all(sha(HERE/k)==v for k,v in ready['aggregation_hashes'].items())
            for support in ('P100','FULL'):
                for model in MODELS:
                    t=f'{cell}_{model}_{support}';py=runtime(cell,model,support)
                    smoke=OUT/'analysis_smoke'/cell/(model+'_'+support)/'ANALYSIS_SMOKE_GATE.json'
                    if not smoke.exists():
                        launch([py,'-u',str(HERE/'worker.py'),cell,model,support,'--mode','smoke'],OUT/'logs'/f'{t}_analysis_smoke.log',t,'ADDITIONAL_ANALYSIS_GPU_SMOKE')
                    s=read(smoke);assert s['status']=='SMOKE_PASS' and s['identity']['source_hashes']==source_hashes()
            launch([PY2,str(HERE/'summarize.py'),cell,'--smoke'],OUT/'logs'/f'{cell}_summary_smoke.log',cell,'ADDITIONAL_ANALYSIS_CPU_AGGREGATION_SMOKE')
            sg=read(OUT/'summary_smoke'/cell/'AGGREGATION_GATE.json');assert sg['status']=='SMOKE_PASS' and sg['checkpoint_count']==6
            dump(AUDIT/f'ANALYSIS_IMPLEMENTATION_GATE_{cell}.json',dict(status='PASS',at=now(),cell=cell,runtime_hashes=source_hashes(),
                cpu_readiness_sha256=sha(AUDIT/'ANALYSIS_CPU_READINESS.json'),aggregation_smoke_sha256=sha(OUT/'summary_smoke'/cell/'AGGREGATION_GATE.json'),
                six_model_support_combinations=True,smoke_not_official_results=True))
            for support in ('P100','FULL'):
                for model in MODELS:
                    t=f'{cell}_{model}_{support}';py=runtime(cell,model,support)
                    launch([py,'-u',str(HERE/'worker.py'),cell,model,support,'--mode','analyze'],OUT/'logs'/f'{t}_analysis.log',t,'MECHANISM_DIAGNOSTICS')
            launch([PY2,str(HERE/'summarize.py'),cell],OUT/'logs'/f'{cell}_summary.log',cell,'CPU_REPORT_AND_ARTIFACT_GATE')
    state('ALL_REPRODUCTION_GATES_COMPLETE_ANALYSIS_PENDING' if mode=='gates' else 'CORE_CELLS_COMPLETE_SUPPLEMENTS_AND_FINAL_REPORT_PENDING',mode=mode)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gates','all'],default='gates');args=ap.parse_args()
    try:run(args.mode)
    except BaseException as e:
        state('BLOCKED',error=repr(e),traceback=traceback.format_exc(),results_preserved=True)
        raise
