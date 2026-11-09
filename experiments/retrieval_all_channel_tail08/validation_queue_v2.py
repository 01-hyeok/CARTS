"""Exclusive durable validation; never releases production on smoke alone."""
import os,sys,json,time,fcntl,subprocess,hashlib,traceback,datetime
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
SUPP=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
OUT=SOURCE/'00_Audit/TAIL08'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
PY='/data/pjh_workspace/ts-env/bin/python';LEGACY='/usr/bin/python3'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def current(**x):
    x.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),production_release=False,solar_training_release=False)
    tmp=OUT/'VALIDATION_CURRENT_v2.tmp';tmp.write_text(json.dumps(x,indent=2));tmp.replace(OUT/'VALIDATION_CURRENT_v2.json')


def main():
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert read(SOURCE/'00_Audit/TAIL05/BOUNDED_QUEUE_STOP.json')['status']=='BLOCKED_CANONICAL_TAIL_PARITY'
    for c in ['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336']:
        g=read(SUPP/'FULL'/c/'DELIVERABLE_GATE.json');assert g['status']=='PASS' and g['source_complete_sha256']==sha(SOURCE/'FULL'/c/'COMPLETE.json')
    sources={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.cu','.so']}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    tasks=[('CPU_TESTS',[PY,'-m','pytest','-q',str(HERE/'test_exact_short_cpu.py'),'-p','no:cacheprovider'],dict(env,CUDA_VISIBLE_DEVICES='')),
           ('SMOKE_V0',[PY,'-u',str(HERE/'smoke_packed.py')],env),
           ('SMOKE_V5',[LEGACY,'-u',str(HERE/'smoke_packed.py')],env),
           ('FULL_V0_EXPORT',[PY,'-u',str(HERE/'full_channel_v0_export.py'),'Weather_720'],env),
           ('FULL_V5_AND_CACHE_REPRODUCTION',[LEGACY,'-u',str(HERE/'full_channel_run.py'),'FULL','Weather_720','--reproduction-only'],env)]
    for phase,command,environment in tasks:
        assert sources=={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.cu','.so']}
        assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        log=OUT/(phase+'_v2.log')
        with log.open('x') as stream:
            worker=subprocess.Popen(command,cwd=str(ROOT),env=environment,stdout=stream,stderr=subprocess.STDOUT)
            current(status='RUNNING_VALIDATION',phase=phase,worker_pid=worker.pid,log=str(log))
            code=worker.wait()
        assert code==0,(phase,code,str(log))
    v0=read(SOURCE/'full_v0_canonical_scores/Weather_720/COMPLETE.json')
    v5=read(OUT/'Weather_720/CANONICAL_REPRODUCTION.json')
    assert v0['status']==v5['status']=='PASS'
    assert {r['split'] for r in v0['splits']}=={'val','test'}
    assert {(r['model'],r['split']) for r in v5['rows']}=={(m,s) for m in ['V0_FullTrained','V5_FullTrained'] for s in ['val','test']}
    for r in v0['splits']:
        assert r['queries']=={'val':4551,'test':9820}[r['split']] and r['channels']==21 and r['candidates']==35448
        assert r['ordered_mismatches']==0 and r['ordered_id_sha256']==r['canonical_ordered_id_sha256']
    for r in v5['rows']:
        assert r['units']=={'val':4551*21,'test':9820*21}[r['split']]
        assert r['ordered_mismatches']==0 and r['ordered_candidate_id_sha256']==r['canonical_ordered_id_sha256']
    result=dict(status='PASS_FULL_MODEL_SPLIT_MATRIX',cell='Weather_720',source_hashes=sources,
                v0_gate_sha256=sha(SOURCE/'full_v0_canonical_scores/Weather_720/COMPLETE.json'),
                v5_gate_sha256=sha(OUT/'Weather_720/CANONICAL_REPRODUCTION.json'),
                original_generation_runtime='UNKNOWN',production_release=False,solar_training_release=False)
    (OUT/'RELEASE_VALIDATION.json').write_text(json.dumps(result,indent=2))
    current(status='FULL_VALIDATION_PASS_PRODUCTION_NOT_STARTED',phase='COMPLETE')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        current(status='BLOCKED_VALIDATION',error=repr(e),traceback=traceback.format_exc())
        raise
