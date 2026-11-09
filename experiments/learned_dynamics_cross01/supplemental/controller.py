"""Serial supplement queue after primary native16 is complete; GPU1 only."""
import sys,os,json,time,subprocess,fcntl,traceback,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import experiment_common as cm
from experiment_common import dump,sha,now,ROOT,GPU,CELLS
OUT=cm.OUT/'supplemental'

def jobs():
    pools=[dict(mode='pool',cell=c,training='P100',pool=p) for c in ('ETTh1_96','ETTh1_720','Weather_96','Weather_720') for p in ('paa16_cosine','saraf_time_aligned')]
    cross=[dict(mode='cross',cell=c,training=t,pool=None) for c in CELLS for t in ('P100','FULL')]
    return pools+cross
def target(j):
    label=j['pool'] if j['mode']=='pool' else 'V5_'+j['training']+'_to_'+('FULL' if j['training']=='P100' else 'P100')
    return OUT/j['mode']/label/j['cell']
def summary():
    rows=[];history=[];done=[]
    for j in jobs():
        dest=target(j)
        if not (dest/'COMPLETE.json').exists():continue
        done.append(j)
        for r in json.loads((dest/'FULL_RESULTS.json').read_text()):
            rows.append(dict(**r,comparison_type=j['mode'],source_training_support=j['training'],pool=j['pool'] or 'original',selection_reused=j['mode']=='cross'))
        if (dest/'TRAIN_HISTORY.json').exists():
            history.extend(dict(**r,cell=j['cell'],pool=j['pool'],comparison_type=j['mode']) for r in json.loads((dest/'TRAIN_HISTORY.json').read_text()))
    for name,data in [('FULL_RESULTS.csv',rows),('TRAIN_HISTORY.csv',history)]:
        keys=sorted({k for r in data for k in r})
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
    dump(OUT/'SUMMARY_STATUS.json',dict(status='COMPLETE' if len(done)==24 else 'IN_PROGRESS',completed=done,expected=24,at=now()))
    (OUT/'RESULTS_SUMMARY.md').write_text('# Supplemental pool and cross-support comparisons\n'+f'{len(done)}/24 comparisons complete.\n'
        +'Pool comparisons train six new scorer arms on the named PAA/SARAF pool using the same original P100 V5 checkpoint. Historical alternative-trained V5 results are not substituted.\n'
        +'Cross-support comparisons reuse all native-selected checkpoints and calibration without refitting or validation reselection. Training support, evaluation support and pool are separate columns.\n'
        +'Every job preserves full-query metrics, per-query selected IDs and Oracle-recovery diagnostics, runtime and peak allocated/reserved memory in its own folder.\n')
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_GATE.json').read_text());assert ready['status']=='PASS'
    assert ready['sources']=={p.name:sha(p) for p in HERE.glob('*.py')}
    dump(OUT/'QUEUE.json',dict(jobs=jobs(),native_required=16,no_solar=True))
    for j in jobs():
        dest=target(j)
        if (dest/'COMPLETE.json').exists():continue
        while True:
            core=json.loads((cm.OUT/'CONTROLLER_STATUS.json').read_text())
            if core['status']=='BLOCKED':raise RuntimeError(('Primary experiment blocked',core))
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],universal_newlines=True)
            busy=any(line.startswith(GPU) for line in active.splitlines())
            if core['status']=='COMPLETE' and not busy:break
            dump(OUT/'CONTROLLER_STATUS.json',dict(status='WAITING_NATIVE16_AND_GPU',native_status=core['status'],gpu_busy=busy,job=j,at=now(),pid=os.getpid()))
            time.sleep(30)
        assert all((cm.OUT/s/c/'COMPLETE.json').exists() for s in ('P100','FULL') for c in CELLS)
        py='/usr/bin/python3' if j['training']=='FULL' else '/data/pjh_workspace/ts-env/bin/python'
        cmd=[py,'-u',str(HERE/'run_supplement.py'),j['mode'],j['cell'],j['training']]
        if j['pool']:cmd+=['--pool',j['pool']]
        logs=OUT/'logs';logs.mkdir(exist_ok=True);log=logs/(j['mode']+'_'+(j['pool'] or j['training'])+'_'+j['cell']+'_'+cm.datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as stream:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            dump(OUT/'CONTROLLER_STATUS.json',dict(status='RUNNING',job=j,worker_pid=p.pid,controller_pid=os.getpid(),log=str(log),at=now()))
            code=p.wait()
        assert code==0,(j,code,str(log))
        assert (dest/'COMPLETE.json').exists();summary()
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='COMPLETE',jobs=24,at=now()))
    dump(cm.OUT/'REQUEST_EXECUTION_STATUS.json',dict(status='ALL_QUEUED_PRIMARY_AND_SUPPLEMENTARY_COMPARISONS_COMPLETE',native_conditions=16,supplementary_comparisons=24,solar_started=False,at=now()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',at=now(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
