"""Expansion launcher. Calls unchanged pilot numerics; distinct result root."""
import sys,os,json,fcntl,traceback,time
from pathlib import Path
OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN));from contracts import *
sys.path.insert(0,str(OWN.parent))
import server_bridge as bridge
import softlag_run as pilot
import gpu_gate as gate
import torch,numpy as np
from d56_complete_run import reproduction

def coexist_start():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    before=bridge.snapshot();assert len(before['processes'])<=1,'Third major GPU worker forbidden'
    owners=[]
    for p in [DIAG/'extensions_d5_d6/CONTROLLER_STATUS.json',DIAG/'full_population_completion/CONTROLLER_STATUS.json']:
        if not p.exists():continue
        s=json.loads(p.read_text());pid=s.get('worker_pid');parent=s.get('pid')
        if not pid or not parent:continue
        proc=Path('/proc')/str(pid);pp=Path('/proc')/str(parent)
        if not proc.exists() or not pp.exists():continue
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        pcmd=(pp/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        # Child provenance must match the recorded live controller as well.
        status=(proc/'status').read_text();actual_parent=int(next(x.split(':')[1] for x in status.splitlines() if x.startswith('PPid:')))
        if actual_parent!=parent:continue
        try:validate_other(s,cmd,pcmd)
        except AssertionError:continue
        owners.append(dict(status_path=str(p),worker_pid=pid,controller_pid=parent,command=cmd,phase=s['phase']))
    if before['processes']:assert len(owners)==1,('Unauthenticated existing GPU process',before,owners)
    assert before['free_mib']>=24576,('Insufficient 24GiB startup headroom',before)
    before['authenticated_other_workers']=owners
    bridge.rt.gpu_guard();torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.40,0)
    return before

def owner(_):
    value=dict(pid=os.getpid(),command=sys.argv,gpu_uuid=GPU,track='TRACK-SOFTLAG-HEAD-EXPANSION02',at=now(),
               reason='Same approved SoftLag track, distinct expansion results; compatibility owner for unchanged diagnostic guard')
    dump(OUT/'00_Audit/GPU_OWNER.json',value)
    # This file is ephemeral worker ownership, never a checkpoint/result.
    prior=OLD/'00_Audit/GPU_OWNER.json'
    if prior.exists():
        p=json.loads(prior.read_text());assert not (Path('/proc')/str(p['pid'])).exists() or p['pid']==os.getpid(),'Another SoftLag worker remains alive'
        dump(OUT/'00_Audit/previous_owners'/('%d.json'%os.getpid()),p)
    dump(prior,value)

def frozen(cell,support):
    os.chdir(ROOT);data=bridge.Data(cell)
    row=next(r for r in bridge.inventory(cell) if r['architecture']=='V5' and r['training_support']==support)
    prior=reproduction(data,row);assert prior is not None,'Full native val/test reproduction prerequisite missing'
    f=bridge.Frozen(data,row)
    return data,f,dict(checkpoint=row,reproduction_gate_sha256=sha(prior),pool_hashes=data.pool_hashes,
        data=data.data_info,source_hashes={n:sha(HERE/n) for n in ['server_bridge.py','past_lag.py','softlag.py']},
        original_runtime='UNKNOWN',current_torch=torch.__version__)

def main(cell,support,phase):
    assert (cell,support) in NEW,'Existing four cells must be reused'
    (OUT/'00_Audit').mkdir(parents=True,exist_ok=True)
    lock=(OLD/'00_Audit/finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    out=OUT/(cell+'_'+support);out.mkdir(parents=True,exist_ok=True)
    proof=dict(wrapper_sha256=sha(__file__),contracts_sha256=sha(OWN/'contracts.py'),
        pilot_runner_sha256=sha(HERE/'softlag_run.py'),protocol_sha256=sha(HERE/'protocol.json'),
        semantics='Unchanged pilot training/scoring/selection. Expanded cell list only; all channels; native support; original runtime UNKNOWN',
        sample_counts={'train':64,'val':32,'test':32},batch8=True,epochs5=True,seed=0)
    p=out/'EXPANSION_PROVENANCE.json'
    if p.exists():assert json.loads(p.read_text())==proof
    else:dump(p,proof)
    for module in [pilot,gate]:module.frozen=frozen;module.start_gpu=coexist_start;module.register_owner=owner
    pilot.BASE=OUT
    if phase=='GATE':gate.main(cell,support,OUT/'00_Audit'/('GPU_GATE_'+cell+'_'+support+'.json'))
    elif phase=='TRAIN':
        assert not (out/'COMPONENT_GATE.json').exists(),'Completed training preserved'
        pilot.main(cell,support)
    elif phase=='FINAL':
        before=coexist_start();owner('expansion');data,f,identity=frozen(cell,support)
        from cell_finalize import complete_cell
        complete_cell(data,f,out,identity,before)
    else:raise ValueError(phase)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('support');p.add_argument('phase',choices=['GATE','TRAIN','FINAL']);a=p.parse_args()
    try:main(a.cell,a.support,a.phase)
    except BaseException as e:
        dump(OUT/'00_Audit/errors'/('%s_%s_%s_%d.json'%(a.cell,a.support,a.phase,os.getpid())),dict(at=now(),error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),existing_results_preserved=True))
        raise
