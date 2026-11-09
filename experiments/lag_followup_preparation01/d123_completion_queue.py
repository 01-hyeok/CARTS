"""Full-query missing D1/D2/D3 completion after D56 and SoftLag exports.

No active GPU worker is interrupted. Locks serialize the new follow-up with
the existing two tracks. Smoke failure blocks that condition, not its criteria.
"""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from d56_aggregate import sha,dump,now
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';EXT=BASE/'extensions_d5_d6';OUT=BASE/'full_population_completion'
ORDER=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336','Weather_720']
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec';PY='/data/pjh_workspace/ts-env/bin/python'

def status(label,**kw):dump(OUT/'CONTROLLER_STATUS.json',dict(status=label,at=now(),pid=os.getpid(),**kw))
def verified(p,expected):
    if not p.exists():return False
    g=json.loads(p.read_text());assert g['status']==expected
    for k,h in g['files'].items():assert sha(p.parent/k)==h
    return True
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h
    jobs=[(c,s) for c in ORDER for s in ['FULL','P100']]
    dump(OUT/'EXECUTION_QUEUE.json',dict(jobs=jobs,model='frozen V5 only; V0/V1 comparator results reused from D56',queries='all val and test',seed=0,all_channels_single_batch=True,
        dependencies=['D56 GPU queue exited','SoftLag pair export completed','per-cell native whole-query reproduction','first/tail exact candidate MSE + ordered Top10 smoke'],
        old_D1_D2_CPU_queue_not_stopped=True,original_results_immutable=True,solar_resume=False))
    while True:
        p=EXT/'POST_EXPORT_STATUS.json';s=json.loads(p.read_text()) if p.exists() else {}
        c=json.loads((EXT/'CONTROLLER_STATUS.json').read_text())
        if s.get('status')=='SOFTLAG_PAIR_EXPORT_COMPLETE' and c.get('status') in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES']:break
        status('WAITING_D56_AND_SOFTLAG_EXPORTS',D56_status=c.get('status'),SoftLag_export_status=s.get('status'),gpu_used=False);time.sleep(20)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX)
    complete=[];failed=[]
    for cell,training in jobs:
        final=OUT/'analysis'/cell/('V5_'+training)/'DELIVERABLE_GATE.json'
        if verified(final,'PASS_FULL_QUERY_D1_D2_D3'):complete.append([cell,training]);continue
        native=EXT/'complete_features'/cell/('V5_'+training+'_to_'+training)/'FEATURE_GATE.json'
        if not native.exists():failed.append(dict(cell=cell,training=training,reason='native V5 sampled inference prerequisite missing'));continue
        ng=json.loads(native.read_text());assert ng['status']=='PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING'
        for k,h in ng['files'].items():assert sha(native.parent/k)==h
        py='/usr/bin/python3' if training=='FULL' else PY
        stages=[];smoke=OUT/'smoke'/cell/('V5_'+training)/'GATE.json';raw=OUT/'raw'/cell/('V5_'+training)/'GATE.json'
        if not verified(smoke,'PASS_FIRST_TAIL_FULL_POPULATION_SMOKE'):stages.append(('SMOKE',[py,'-u',str(HERE/'d123_full_population.py'),cell,training,'--smoke']))
        if not verified(raw,'PASS_FULL_QUERY_D123_RAW'):stages.append(('FULL_QUERY_INFERENCE',[py,'-u',str(HERE/'d123_full_population.py'),cell,training]))
        stages.append(('CPU_AGGREGATION',[PY,'-u',str(HERE/'d123_full_aggregate.py'),cell,training]))
        ok=True
        for phase,cmd in stages:
            log=OUT/'logs'/(cell+'_'+training+'_'+phase+'.log')
            with log.open('a') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if phase=='CPU_AGGREGATION' else GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',cell=cell,training=training,phase=phase,worker_pid=child.pid,command=cmd,log=str(log),complete=len(complete),failed=failed)
                rc=child.wait()
            if rc:
                failed.append(dict(cell=cell,training=training,phase=phase,returncode=rc,log=str(log)));ok=False;break
        if ok:assert verified(final,'PASS_FULL_QUERY_D1_D2_D3');complete.append([cell,training])
        dump(OUT/'PROGRESS.json',dict(at=now(),complete=len(complete),expected=16,failed=failed))
    status('FULL_QUERY_D123_COMPLETE' if len(complete)==16 and not failed else 'PARTIAL_WITH_EXPLICIT_FAILURES',complete=len(complete),failed=failed)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
