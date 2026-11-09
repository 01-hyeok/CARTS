"""One-time preserved handoff after gates; no intervention in unrelated work."""
import sys,json,os,subprocess,shutil,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'orchestration_seed0')]
import runtime_vector as rv
import controller_vector as ctl
rt=rv.rt

if __name__=='__main__':
    assert not (HERE/'ACTIVATED.json').exists(),'Already activated; inspect PID instead of duplicating'
    gate=json.loads((HERE/'VECTOR_GATES.json').read_text())
    assert gate['status']=='PASS' and gate['source_hashes']==rv.source_hashes()
    xml=ET.parse(HERE/'unit_tests.xml').getroot()
    assert all(int(s.get('failures',0))==0 and int(s.get('errors',0))==0 for s in xml.iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in xml.iter('testsuite'))>=18
    assert not ctl.track_workers(),'Cannot launch while another track worker exists'
    oldpid=int((rt.OUT/'controller.pid').read_text())
    assert ctl.original.proc_identity(oldpid) is None,'Old supervisor is still running'
    pause=json.loads((HERE/'PAUSED.json').read_text())
    assert rt.sha256_file(pause['checkpoint'])==pause['checkpoint_sha256']
    assert ctl.original.proc_identity(pause['old_worker_pid']) is None
    rows=json.loads((rt.HERE/'configs/run_manifest.json').read_text());ctl.original.validate_manifest(rows)
    completed={r['run_id']:rt.sha256_file(rt.ROOT/r['output_path']/'COMPLETE.json') for r in rows if (rt.ROOT/r['output_path']/'COMPLETE.json').exists()}
    assert len(completed)==34
    for name in ('run_manifest.json','run_manifest.csv'):
        shutil.copy2(rt.HERE/'configs'/name,HERE/f'before_vector_{name}')
    rerouted=[]
    for r in rows:
        if r['run_id'] in completed:continue
        assert r['dataset']=='Solar' and r['seed']==0
        old=dict(r)
        r['output_path']=str((rt.OUT/'runs_vectorized03'/f"Solar_{r['horizon']}"/'seed0'/r['model']).relative_to(rt.ROOT))
        r['checkpoint_path']=f"checkpoints/track_p100_head_retrieval_quality01_legacy_pool02_vectorized03/Solar_{r['horizon']}/seed0/{r['model']}"
        assert not (rt.ROOT/r['output_path']).exists() and not (rt.ROOT/r['checkpoint_path']).exists()
        r['status']='NOT_STARTED';r['blocked_by']=None
        rerouted.append(dict(run_id=r['run_id'],previous_output=old['output_path'],new_output=r['output_path'],new_checkpoint=r['checkpoint_path']))
    assert len(rerouted)==14
    ctl.original.save_manifest(rows);ctl.original.progress(rows)
    for name in ('SEED0_ADOPTION.json','PAUSE_REQUESTED.json'):
        p=rt.OUT/name
        if p.exists():p.rename(HERE/f'archived_{name}')
    after={r['run_id']:rt.sha256_file(rt.ROOT/r['output_path']/'COMPLETE.json') for r in rows if r['run_id'] in completed}
    assert after==completed
    transition=dict(at=rt.now(),completed_34_unchanged=completed,rerouted=rerouted,
        superseded_partial_run=pause,reason='Restart only incomplete Solar H96 V2 from common seed0 initialization; batched FP32 GEMM/reduction not bitwise trajectory-equivalent despite matching Dropout RNG progression',
        total=48,active_seed=0,cancelled_by_user=96,training_channels=137,evaluation='original canonical channelwise',
        stage2=False,source_hashes=rv.source_hashes())
    rt.write_json(HERE/'TRANSITION.json',transition)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=rt.GPU_UUID,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    log=rt.OUT/'logs/controller_vectorized03.log'
    command=[sys.executable,'-u',str(HERE/'controller_vector.py')]
    with log.open('a') as f:
        worker=subprocess.Popen(command,cwd=rt.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(controller_pid=worker.pid,log=str(log),command=command,at=rt.now(),gpu_uuid=rt.GPU_UUID)
    rt.write_json(HERE/'ACTIVATED.json',launch);print(json.dumps(launch),flush=True)
