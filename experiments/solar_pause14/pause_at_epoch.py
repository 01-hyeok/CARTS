"""User-requested lossless in-memory Solar pause after a full epoch save.

Signals only the authenticated Solar controller/worker. No termination, code
mutation, checkpoint replacement, or automatic resume. VRAM remains allocated.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path('/data/pjh_workspace/CARTS')
SOLAR=ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04'
MECH=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
CKPT=ROOT/'checkpoints/track_solar_full_all_channel04/Solar_336/V5/seed0'
UUID='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(p))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def process(pid):
    p=Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text();a=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,start=a[19],state=a[0],ppid=int(a[1]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                    children=(p/'task'/str(pid)/'children').read_text().split(),wchan=(p/'wchan').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None
def matches(expected):
    p=process(expected['pid'])
    return p if p and p['start']==expected['start'] and p['cmd']==expected['cmd'] else None
def validate(owner,worker,status):
    assert owner and worker and status['target']=='Solar_336_V5'
    assert status['controller_pid']==owner['pid'] and status['worker_pid']==worker['pid']
    assert '/experiments/solar_parallel13/controller_parallel.py' in owner['cmd']
    assert '/experiments/solar_parallel13/solar_full_train_parallel.py train Solar_336 --heads 5' in worker['cmd']
    assert owner['state']=='S' and owner['wchan']=='do_wait' and owner['children']==[str(worker['pid'])]
    assert worker['ppid']==owner['pid'] and worker['state'] not in ['T','t','Z']
def checked_signal(expected,sig):
    p=matches(expected);assert p is not None,'PID identity changed; do not signal'
    os.kill(expected['pid'],sig)
def await_stopped(expected):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        p=matches(expected);assert p is not None,'process disappeared while stopping'
        if p['state'] in ['T','t']:return p
        time.sleep(.05)
    raise RuntimeError('SIGSTOP delivery did not reach observed stopped state')
def gpu():
    rows=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert rows[0].strip()==UUID
    apps=[x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines() if UUID in x]
    return dict(at=now(),utilization_percent=int(rows[1]),used_mib=int(rows[2]),free_mib=int(rows[3]),processes=apps)
def payload_gate(s,epoch):
    required={'model_state_dict','slot_heads_state_dict','optimizer','rng','loader_rng','history','source_hashes','epoch'}
    assert required<=set(s) and s['epoch']==epoch
    assert s['slot_heads_state_dict'] and s['optimizer']['state']
    assert set(s['rng'])=={'torch','cuda','numpy','python'} and len(s['rng']['cuda'])==1
    assert len(s['loader_rng'])>0 and len(s['history'])==epoch
    assert [x['epoch'] for x in s['history']]==list(range(1,epoch+1))
    for row in s['history']:
        assert row['train_queries']==36121 and row['steps']==1129
        assert row['val']['queries']==4921 and row['val']['channels']==137 and row['val']['units']==4921*137
    for group in s['optimizer']['param_groups']:assert group['lr']==.001
    for state in s['optimizer']['state'].values():
        assert {'step','exp_avg','exp_avg_sq'}<=set(state)
        assert int(state['step'])==epoch*1129
    original=ROOT/'experiments/solar_full_all_channel04'
    assert {p.name:sha(p) for p in original.glob('*.py')}==s['source_hashes']
    return dict(epoch=epoch,optimizer_states=len(s['optimizer']['state']),rng=list(s['rng']),loader_rng_bytes=len(s['loader_rng']),
                scheduler='NOT_USED_FIXED_ADAM_LR_0.001',resume_from_disk_epoch=epoch+1,source_hashes=s['source_hashes'])
def exact_payload(a,b):
    import numpy as np
    import torch
    if type(a)!=type(b):return False
    if torch.is_tensor(a):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_payload(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact_payload(x,y) for x,y in zip(a,b))
    return a==b


def blocks(root):
    rows=[]
    for split in ['val','test']:
        for p in (root/split).glob('batch*.json'):
            if p.stem.endswith('_ranking'):continue
            d=read(p)
            if d.get('status')=='PASS' and d.get('queries')==32:
                rows.append(dict(split=split,batch=int(p.stem[5:]),at=d['at'],seconds=datetime.datetime.fromisoformat(d['at']).timestamp()))
    return sorted(rows,key=lambda x:x['seconds'])
def rate(rows):
    if len(rows)<2 or len({r['split'] for r in rows})!=1:return None
    if any(b['batch']!=a['batch']+1 for a,b in zip(rows,rows[1:])):return None
    sec=rows[-1]['seconds']-rows[0]['seconds']
    return dict(blocks=len(rows),queries=(len(rows)-1)*32,seconds=sec,queries_per_second=(len(rows)-1)*32/sec,
                split=rows[0]['split'],start=rows[0]['at'],end=rows[-1]['at'])


def main(out,epoch):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only pause monitor'
    import torch
    assert not torch.cuda.is_initialized()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    status=read(SOLAR/'CONTROLLER_STATUS.json');owner=process(status['controller_pid']);worker=process(status['worker_pid'])
    validate(owner,worker,status)
    assert owner['pid']==276745 and worker['pid']==321503
    initial=torch.load(CKPT/'latest.pth',map_location='cpu',weights_only=False);proof=payload_gate(initial,initial['epoch'])
    assert epoch==5 and initial['epoch']==4
    write(out/'PRECHECK.json',dict(at=now(),controller=owner,worker=worker,checkpoint=proof,latest_sha256=sha(CKPT/'latest.pth'),
          target_epoch=epoch,mode='SIGSTOP after atomic full checkpoint; no termination; GPU memory retained'))
    checked_signal(owner,signal.SIGSTOP)
    await_stopped(owner)
    write(SOLAR/'PAUSE_STATUS.json',dict(status='USER_PAUSE_REQUESTED_WAITING_EPOCH_CHECKPOINT',at=now(),monitor_pid=os.getpid(),target_epoch=epoch,
         controller_pid=owner['pid'],worker_pid=worker['pid'],monitor_output=str(out),controller_next_job_blocked=True,worker_still_training=True))
    print('SOLAR PAUSE WAIT EPOCH',epoch,flush=True)
    observed=(CKPT/'latest.pth').stat().st_mtime_ns;deadline=time.time()+2*3600;stopped=False
    last_observation=0;before_gpu=[]
    try:
        while True:
            assert matches(owner) and matches(owner)['state'] in ['T','t']
            assert matches(worker) and matches(worker)['state'] not in ['T','t','Z']
            if time.time()-last_observation>=10:
                write(out/'HEARTBEAT.json',dict(at=now(),waiting_epoch=epoch))
                before_gpu=(before_gpu+[gpu()])[-5:]
                write(out/'GPU_BEFORE_PAUSE.json',dict(samples=before_gpu,interpretation='Last concurrent Solar+Mechanism observations before worker pause; 10-second sampling.'))
                last_observation=time.time()
            stat=(CKPT/'latest.pth').stat()
            if stat.st_mtime_ns!=observed:
                saved=torch.load(CKPT/'latest.pth',map_location='cpu',weights_only=False)
                if saved['epoch']>=epoch:
                    assert saved['epoch']==epoch,'missed requested boundary; do not conceal'
                    checked_signal(worker,signal.SIGSTOP);stopped=True;await_stopped(worker);break
                observed=stat.st_mtime_ns
            assert time.time()<deadline,'epoch checkpoint timeout'
            time.sleep(.2)
        assert matches(worker)['state'] in ['T','t']
        saved=torch.load(CKPT/'latest.pth',map_location='cpu',weights_only=False)
        epoch_saved=torch.load(CKPT/('epoch%02d.pth'%epoch),map_location='cpu',weights_only=False)
        proof=payload_gate(saved,epoch);assert exact_payload(saved,epoch_saved)
        checkpoint_file=CKPT/('epoch%02d.pth'%epoch)
        mech=read(MECH/'00_Audit/CONTROLLER_STATUS.json');cell,model,support=mech['target'].rsplit('_',2)
        mechroot=MECH/'checkpoint_runs'/cell/(model+'_'+support);prior=blocks(mechroot)[-16:];paused=time.time()
        state=dict(status='PAUSED_IN_MEMORY_AT_VERIFIED_EPOCH',at=now(),controller_pid=owner['pid'],worker_pid=worker['pid'],
            checkpoint=str(checkpoint_file),checkpoint_sha256=sha(checkpoint_file),latest_sha256=sha(CKPT/'latest.pth'),checkpoint_contents=proof,
            model_optimizer_rng_retained_in_process=True,completed_epoch_loss=0,gpu_memory_released=False,
            resume_same_process='SIGCONT authenticated worker, then controller after authorization',
            disk_resume='Original runner loads latest.pth at epoch6 if process is later intentionally replaced',
            note='Polling can catch a few post-save instructions; all in-flight memory is retained. Do not claim epoch6 was never entered.',
            mechanism_before=mech,mechanism_before_rate=rate(prior))
        write(out/'PAUSED.json',state);write(SOLAR/'PAUSE_STATUS.json',state)
        samples=[]
        for _ in range(5):samples.append(gpu());time.sleep(2)
        write(out/'GPU_AFTER_PAUSE.json',dict(samples=samples,interpretation='Paused Solar CUDA context retains VRAM; only compute submission stops.'))
        while time.time()-paused<600:
            assert matches(worker)['state'] in ['T','t'] and matches(owner)['state'] in ['T','t']
            newer=[r for r in blocks(mechroot) if r['seconds']>paused+5]
            if len(newer)>=16:
                after=rate(newer[:16]);before=state['mechanism_before_rate']
                write(out/'MECHANISM_SPEED_COMPARISON.json',dict(status='OBSERVATION_COMPLETE',at=now(),before=before,after=after,
                    comparable_same_split=bool(before and after and before['split']==after['split']),
                    same_checkpoint_worker=True,worker_pid=mech['worker_pid'],source_unchanged=True,
                    causal_claim=False,limitations='Different queries and GPU scheduling; sequential observation, not a controlled repeated-input benchmark.'))
                break
            write(out/'HEARTBEAT.json',dict(at=now(),phase='PAUSED_MEASURING_MECHANISM'));time.sleep(5)
        write(out/'MONITOR_FINISHED.json',dict(at=now(),solar_stays_paused=True,automatic_resume=False))
    except BaseException as e:
        import traceback
        write(out/'FAILED.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),worker_stopped=stopped))
        # Never kill; if checkpoint verification failed, preserve all live state
        # paused for inspection instead of silently continuing new training.
        if not stopped and matches(worker):checked_signal(worker,signal.SIGSTOP)
        write(SOLAR/'PAUSE_STATUS.json',dict(status='PAUSED_IN_MEMORY_REQUIRES_INSPECTION',at=now(),error=repr(e),
             controller_pid=owner['pid'],worker_pid=worker['pid'],gpu_memory_released=False,monitor_output=str(out)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args();main(a.out,a.epoch)
