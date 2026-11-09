"""Operational recovery only: immutable original scientific code and gates."""
import hashlib,json,math,os,subprocess
from pathlib import Path

ROOT=Path('/data/pjh_workspace/CARTS')
ORIGINAL=ROOT/'experiments/solar_full_all_channel04'
OUT=ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04'
HERE=Path(__file__).resolve().parent
EVIDENCE=OUT/'parallel_recovery13'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
MECH=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit/CONTROLLER_STATUS.json'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wrapper_hashes(): return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
def cmd(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''

def validate_owner(state,controller,command,ppid,env):
    assert state['status']=='RUNNING'
    assert '/experiments/retrieval_failure_mechanism02/controller.py' in controller
    expected=state['command']
    assert command.split()==expected,'Live command differs from authorized controller dispatch'
    assert expected[0] in ['/data/pjh_workspace/ts-env/bin/python','/usr/bin/python3']
    scripts=[x for x in expected if x.endswith('.py')]
    assert len(scripts)==1
    assert Path(scripts[0]).parent==ROOT/'experiments/retrieval_failure_mechanism02'
    assert Path(scripts[0]).name in ['worker.py','coexist_smoke.py','reproduce.py']
    assert ppid==state['controller_pid'],'Worker parent mismatch'
    assert ('CUDA_VISIBLE_DEVICES='+GPU).encode() in env

def snapshot():
    fields=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    index,uuid,free,total=[x.strip() for x in fields]
    assert index=='1' and uuid==GPU
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    apps=[]
    for row in rows.splitlines():
        u,p,m=[x.strip() for x in row.split(',')]
        if u==GPU: apps.append(dict(host_pid=int(p),memory_mib=int(m)))
    state=read(MECH);controller=cmd(state.get('controller_pid',-1))
    active='/experiments/retrieval_failure_mechanism02/controller.py' in controller
    owner=None
    if apps:
        assert len(apps)==1,'More than one existing GPU1 process; no new Solar job'
        assert active and state['status']=='RUNNING','Existing GPU job not authenticated as Mechanism02'
        worker=state['worker_pid'];command=cmd(worker)
        proc=Path('/proc')/str(worker)
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        env=(proc/'environ').read_bytes().split(b'\0')
        validate_owner(state,controller,command,int(stat[1]),env)
        owner=dict(container_pid=worker,command=command,controller_pid=state['controller_pid'],nvidia_process=apps[0],host_container_pid_mapping='not independently exposed; sole GPU1 process and live owner verified')
    return dict(gpu_uuid=GPU,free_mib=int(free),total_mib=int(total),apps=apps,mechanism_active=active,owner=owner)

def budget(smoke,cell,s):
    row=next(r for r in smoke['results'] if r['cell']==cell)
    assert row['status']=='PASS' and row['channels']==137 and row['batch']==32
    assert row['query_chunk']==32 and row['candidate_chunk']==1000000000
    required=math.ceil(row['peak_reserved']/2**20)+2048
    # Reserve Mechanism02's declared allocator cap even between its workers.
    mechanism_cap=math.ceil(s['total_mib']*.18)+1024 if s['mechanism_active'] else 0
    assert required+mechanism_cap<=s['total_mib'],('WAIT_RESOURCE: full-shape Solar plus bounded Mechanism02 cannot safely coexist',cell,required,mechanism_cap)
    assert s['free_mib']>=required,('WAIT_RESOURCE: insufficient current free memory',cell,required,s['free_mib'])
    return dict(required_free_mib=required,reserved_mechanism_mib=mechanism_cap,snapshot=s)

def validate_completed(t,cell,n):
    dest=OUT/cell/f'V{n}'/'seed0';done=read(dest/'COMPLETE.json')
    assert done['status']=='COMPLETE' and done['source_hashes']==t.source_hashes()
    assert done['canonical_source_hashes']==t.rt.source_hashes()
    assert sha(done['checkpoint'])==done['checkpoint_sha256']
    history=read(dest/'history.json');assert [r['epoch'] for r in history]==list(range(1,11))
    selected=min(history,key=lambda r:(r['val']['retmse10'],r['epoch']))
    assert selected['epoch']==done['selection']['epoch'] and not done['selection']['test_used_for_selection']
    assert all(r['train_queries']==done['full_candidates'] and r['steps']==math.ceil(done['full_candidates']/32) for r in history)
    assert all((t.CKPT/cell/f'V{n}'/'seed0'/f'epoch{i:02d}.pth').exists() for i in range(1,11))
    import numpy as np
    for split,key in [('val','validation'),('test','test')]:
        meta=done[key];q=meta['queries'];assert meta['channels']==137 and meta['units']==q*137 and meta['all_channel_batch']
        ids=np.load(dest/'cache'/f'{split}_ids.npy',mmap_mode='r');assert ids.shape==(q,137,10)
        starts=np.load(dest/'cache'/f'{split}_query_start.npy',mmap_mode='r');assert len(starts)==q
        assert read(dest/'cache'/f'{split}_metrics.json')['retmse10']==meta['retmse10']
    return dict(cell=cell,model=f'V{n}',complete_sha256=sha(dest/'COMPLETE.json'),checkpoint_sha256=done['checkpoint_sha256'],ten_epochs=True,full_evaluation=True)
