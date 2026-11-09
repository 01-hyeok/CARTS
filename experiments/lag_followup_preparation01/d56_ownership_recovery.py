"""Recover one pre-data ownership-handshake failure at the existing queue boundary.

Original numerical code, resource guards, smoke and scientific gates are unchanged.
The live controller/worker is never stopped. Failure evidence remains immutable.
"""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from d56_controller_v2 import ROOT,HERE,BASE,OUT,PY2,GPU,sha,dump,now,jobs,verified_gate
AUDIT=OUT/'ownership_recovery01'
TARGET=['ETTh1_192','V0','P100','P100']
NAME='V0_P100_to_P100'
ERROR='Existing GPU process is not an authenticated other lag-track worker'
def recovery_status(label,**kw):dump(AUDIT/'CURRENT.json',dict(status=label,at=now(),pid=os.getpid(),target=TARGET,**kw))
def require_predata_failure(record,current,files):
    assert record['job']==TARGET and record['phase']=='FEATURES' and record['returncode']==1
    assert current['phase']=='BLOCKED' and ERROR in current['error']
    assert 'before=start_gpu()' in current['traceback']
    assert not files,'Never treat a failure with saved query data as a pre-data handshake'
def preserve(source,target):
    if target.exists():assert sha(source)==sha(target)
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
def preflight():
    AUDIT.mkdir(parents=True,exist_ok=True)
    failures=json.loads((OUT/'FAILED_JOBS.json').read_text());record=next(x for x in failures if x['job']==TARGET)
    folder=OUT/'complete_features'/TARGET[0]/NAME;current=json.loads((folder/'CURRENT.json').read_text())
    require_predata_failure(record,current,list(folder.glob('*/*.npz')))
    smoke=OUT/'feature_smoke'/TARGET[0]/NAME/'FEATURE_GATE.json'
    assert verified_gate(smoke,'PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL')
    ready=json.loads((OUT/'CPU_EXTENSION_READY_v2.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['code_hashes'].items():assert sha(HERE/n)==h
    for p,dest in [(OUT/'FAILED_JOBS.json',AUDIT/'original_FAILED_JOBS.json'),(Path(record['log']),AUDIT/'original_FEATURES.log'),(folder/'CURRENT.json',AUDIT/'original_CURRENT.json')]:preserve(p,dest)
    dump(AUDIT/'PREFLIGHT.json',dict(status='PASS_PRE_DATA_FAILURE_ONLY',at=now(),target=TARGET,smoke_gate_sha256=sha(smoke),source_hashes=ready['code_hashes'],
        failed_log_sha256=sha(Path(record['log'])),saved_queries=0,scientific_or_resource_guard_changes=False,
        cause='Observed startup ownership authentication failure; overlaps SoftLag GPU transition. Exact handshake race inferred, not directly logged.',
        boundary='Existing D56 controller releases d4_controller.lock; then GPU1 must be empty; only this failed condition is retried'))
def run_child(phase,command,gpu):
    log=AUDIT/'logs'/(phase+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        recovery_status('RUNNING_RECOVERY',phase=phase,worker_pid=p.pid,command=command,log=str(log))
        rc=p.wait()
    assert rc==0,('Recovery child failed; no automatic numerical retry',phase,rc,str(log))
def gpu_idle():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
    return not any(x.strip().startswith(GPU+',') for x in raw.splitlines())
def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    lock=(AUDIT/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (AUDIT/'RESOLUTION_GATE.json').exists():
        g=json.loads((AUDIT/'RESOLUTION_GATE.json').read_text());assert g['status']=='PASS_RECOVERED_PRE_DATA_OWNERSHIP_FAILURE';return
    preflight();proof=json.loads((AUDIT/'PREFLIGHT.json').read_text())
    recovery_status('WAITING_EXISTING_D56_CONTROLLER_BOUNDARY',gpu_used=False)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX)
    original=json.loads((OUT/'CONTROLLER_STATUS.json').read_text())
    assert original['status'] in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES'],original
    dump(AUDIT/'original_terminal_controller_status.json',original)
    for n,h in proof['source_hashes'].items():assert sha(HERE/n)==h
    soft=json.loads((ROOT/'results/TRACK-SOFTLAG-HEAD-EXPANSION02/CONTROLLER_STATUS.json').read_text());assert soft['status']=='COMPLETE_ALL_16'
    deadline=time.monotonic()+60
    while not gpu_idle():
        recovery_status('WAITING_GPU1_IDLE_AFTER_BOUNDARY',gpu_used=False)
        assert time.monotonic()<deadline,'GPU still occupied; no process was killed'
        time.sleep(2)
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='RUNNING',at=now(),pid=os.getpid(),phase='TARGETED_OWNERSHIP_RECOVERY',job=TARGET,original_controller_status=str(AUDIT/'original_terminal_controller_status.json')))
    feature=OUT/'complete_features'/TARGET[0]/NAME/'FEATURE_GATE.json'
    if not verified_gate(feature,'PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING'):
        run_child('FEATURES',[PY2,'-u',str(HERE/'d56_complete_run.py')]+TARGET,True)
    assert verified_gate(feature,'PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING')
    analysis=OUT/'complete_analysis'/TARGET[0]/NAME/'ANALYSIS_GATE.json'
    if not verified_gate(analysis,'PASS_D5_D6_CASE'):
        run_child('CPU_AGGREGATION',[PY2,'-u',str(HERE/'d56_aggregate.py')]+TARGET,False)
    assert verified_gate(analysis,'PASS_D5_D6_CASE')
    complete=[]
    for job in jobs():
        c,m,t,e=job;p=OUT/'complete_analysis'/c/(m+'_'+t+'_to_'+e)/'ANALYSIS_GATE.json'
        if verified_gate(p,'PASS_D5_D6_CASE'):complete.append(job)
    failure_history=json.loads((OUT/'FAILED_JOBS.json').read_text());remaining=[x for x in failure_history if x['job']!=TARGET]
    resolution=dict(job=TARGET,status='RESOLVED_ANALYSIS_GATE_PASS',at=now(),feature_gate_sha256=sha(feature),analysis_gate_sha256=sha(analysis),original_failure_preserved=str(AUDIT/'original_CURRENT.json'))
    dump(OUT/'RESOLVED_JOBS.json',[resolution]);dump(AUDIT/'PROGRESS_BEFORE_RECONCILIATION.json',json.loads((OUT/'PROGRESS.json').read_text()))
    dump(OUT/'PROGRESS.json',dict(at=now(),complete=len(complete),native_complete=sum(j[2]==j[3] for j in complete),cross_support_complete=sum(j[2]!=j[3] for j in complete),failed=remaining,resolved=[resolution],waiting=original.get('waiting',[]),total=96,original_failure_history=str(OUT/'FAILED_JOBS.json')))
    terminal='COMPLETE_ALL_CASES' if len(complete)==96 and not remaining else 'AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES'
    dump(OUT/'CONTROLLER_STATUS.json',dict(status=terminal,at=now(),pid=os.getpid(),complete=len(complete),failed=remaining,resolved=[resolution],waiting=original.get('waiting',[])))
    # Existing CPU queue may have observed the original terminal state before
    # this recovery. Its lock prevents concurrent writers; fill only our gap.
    cpu_lock=(OUT/'d14_statistics.lock').open('a');fcntl.flock(cpu_lock,fcntl.LOCK_EX)
    stats=OUT/'d14_statistics'/TARGET[0]/NAME/'SAMPLE_STATISTICS_GATE.json'
    if not verified_gate(stats,'PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS'):
        run_child('CPU_STATISTICS',[PY2,'-u',str(HERE/'d14_sample_completion.py')]+TARGET,False)
    assert verified_gate(stats,'PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS')
    allstats=list((OUT/'d14_statistics').glob('*/*/SAMPLE_STATISTICS_GATE.json'))
    if len(allstats)==96:
        for p in allstats:assert verified_gate(p,'PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS')
        dump(OUT/'D14_STATISTICS_STATUS.json',dict(status='COMPLETE_SAMPLE_STATISTICS',at=now(),pid=os.getpid(),completed=96,failures=[],source_controller_status=terminal,recovered_one_predata_case=True))
    run_child('REPORT',[PY2,'-u',str(HERE/'d56_report.py')],False)
    dump(AUDIT/'RESOLUTION_GATE.json',dict(status='PASS_RECOVERED_PRE_DATA_OWNERSHIP_FAILURE',at=now(),target=TARGET,feature_gate_sha256=sha(feature),analysis_gate_sha256=sha(analysis),statistics_gate_sha256=sha(stats),code_hashes_unchanged=True,guards_unchanged=True,completed=len(complete),remaining_failures=remaining))
    recovery_status('RECOVERED',complete=len(complete),statistics_complete=len(allstats),gpu_used=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:preflight() if a.preflight else main()
    except BaseException as e:recovery_status('BLOCKED_RECOVERY',error=repr(e),traceback=traceback.format_exc(),original_results_preserved=True);raise
