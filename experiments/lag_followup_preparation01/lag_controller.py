"""Two independently locked tracks, at most one GPU worker per track."""
import argparse,datetime,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
TRACKS={'softlag':'TRACK-SOFTLAG-HEAD-PILOT01','d4':'TRACK-HEAD-SHIFT-DIAGNOSTIC01'}
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(track):
    out=ROOT/'results'/TRACKS[track];audit=out/'00_Audit'
    lock=(audit/(track+'_controller.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if track=='softlag':jobs=[('ETTh1_96','FULL'),('ETTh1_336','FULL'),('ETTh1_96','P100'),('ETTh1_336','P100')]
    else:jobs=[(c,s) for c in ['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','ETTh1_192','Weather_192','Weather_336','Weather_720'] for s in ['FULL','P100']]
    dump(audit/(track+'_QUEUE.json'),dict(jobs=jobs,seed=0,all_channels=True,solar_resumed=False,max_workers=1,other_lag_track_concurrent_authorized=True))
    for cell,support in jobs:
        dest=out/(cell+'_'+support) if track=='softlag' else out/'D4'/('V5_'+cell+'_'+support)
        if (dest/'COMPONENT_GATE.json').exists():continue
        py='/usr/bin/python3' if support=='FULL' else '/data/pjh_workspace/ts-env/bin/python'
        # First/tail GPU gate at each native checkpoint. A gate is a separate
        # sequential child; no overlapping child of this track is launched.
        gate=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01/00_Audit'/('GPU_GATE_'+cell+'_'+support+'.json')
        if not gate.exists():
            # If the other track is live, its authenticated owner may coexist.
            command=[py,'-u',str(HERE/'gpu_gate.py'),cell,support,'--out',str(gate)]
            launch(out,audit,track,cell,support,'GPU_GATE',command)
        command=[py,'-u',str(HERE/('softlag_run.py' if track=='softlag' else 'd4_run.py')),cell,support]
        launch(out,audit,track,cell,support,'RUN',command)
        assert (dest/'COMPONENT_GATE.json').exists(),'No component gate; no silent progression'
    dump(audit/(track+'_CONTROLLER_STATUS.json'),dict(status='COMPONENT_QUEUE_COMPLETE_FINAL_REVIEW_PENDING',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),jobs=len(jobs)))

def launch(out,audit,track,cell,support,phase,command):
    log=out/'logs'/('%s_%s_%s_%s.log'%(track,cell,support,phase));log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LAG_TRACK=TRACKS[track])
    with log.open('a') as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        dump(audit/(track+'_CONTROLLER_STATUS.json'),dict(status='RUNNING',phase=phase,cell=cell,support=support,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),worker_pid=p.pid,command=command,log=str(log)))
        code=p.wait()
    assert code==0,dict(cell=cell,support=support,phase=phase,returncode=code,log=str(log))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('track',choices=TRACKS);a=p.parse_args()
    try:run(a.track)
    except BaseException as e:
        dump(ROOT/'results'/TRACKS[a.track]/'00_Audit'/(a.track+'_CONTROLLER_STATUS.json'),dict(status='BLOCKED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
