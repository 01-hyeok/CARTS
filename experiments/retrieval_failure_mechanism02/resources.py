"""Explicitly authorized coexistence, bounded new allocations, no process control."""
import os,subprocess
from pathlib import Path
from protocol import ROOT,GPU_UUID,PROTOCOL
from io_utils import read,now

def snapshot():
    line=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    index,uuid,free,used,total,util=[x.strip() for x in line]
    assert index=='1' and uuid==GPU_UUID
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[]
    for line in apps.splitlines():
        u,p,m=[x.strip() for x in line.split(',')]
        if u==GPU_UUID:processes.append(dict(pid=int(p),memory_mib=int(m)))
    return dict(at=now(),gpu_uuid=uuid,free_mib=int(free),used_mib=int(used),total_mib=int(total),utilization_percent=int(util),processes=processes)

def before_context():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
    s=snapshot();assert len(s['processes'])<=1,'Another new GPU job already exists; do not compete'
    if s['processes']:
        p=ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04/CONTROLLER_STATUS.json';owner=read(p)
        assert owner['status']=='RUNNING' and 'worker_pid' in owner,'Cannot authenticate existing Solar owner'
        proc=Path('/proc')/str(owner['worker_pid']);assert proc.exists(),'Owner worker is not alive'
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'solar_full_train' in cmd or 'diagnostic' in cmd,'Existing owner is not an authorized Solar job'
        s['existing_owner']=dict(container_pid=owner['worker_pid'],command=cmd,status_file=str(p),nvidia_process=s['processes'][0])
    assert s['free_mib']>=PROTOCOL['concurrency']['minimum_free_before_start_gib']*1024,('insufficient measured headroom',s)
    return s

def bound_context():
    import torch
    assert torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['concurrency']['new_worker_allocated_cap_fraction'],0)

def during():
    s=snapshot();assert len(s['processes'])<=2,'More than authorized Solar + one Mechanism02 process'
    assert s['free_mib']>=1024,'Less than 1GiB free; stop at current batch without affecting Solar'
    return s
