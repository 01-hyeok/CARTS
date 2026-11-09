"""CPU waiting controller; cannot acquire GPU until all original diagnostics pass."""
import csv,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import solar_full_train as t
PY='/data/pjh_workspace/ts-env/bin/python'

def status(**kwargs):t.dump(t.OUT/'CONTROLLER_STATUS.json',dict(at=t.rt.now(),controller_pid=os.getpid(),**kwargs))
def ready():
    try:return t.prerequisites()
    except (FileNotFoundError,AssertionError,KeyError):return False
def apps():
    return [r for r in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines() if t.GPU in r]
def launch(command,label,env):
    while apps():status(status='WAITING_GPU1_RELEASE',target=label,gpu_apps=apps());time.sleep(30)
    logs=t.OUT/'logs';logs.mkdir(exist_ok=True);path=logs/(label+'.log')
    with path.open('a') as f:
        f.write('\nSTART '+t.rt.now()+' '+repr(command)+'\n');f.flush()
        proc=subprocess.Popen(command,cwd=t.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        status(status='RUNNING',target=label,worker_pid=proc.pid,log=str(path),gpu_uuid=t.GPU)
        assert proc.wait()==0,(label,'worker failed',str(path))

def main():
    t.OUT.mkdir(parents=True,exist_ok=True);lock=(t.OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=t.read(t.OUT/'CPU_READY_GATE.json');assert gate['status']=='CPU_PASS_GPU_SMOKE_DEFERRED' and gate['source_hashes']==t.source_hashes()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=t.GPU,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    while not ready():
        status(status='WAITING_ALL_PRIOR_DIAGNOSTICS',actual_training_started=False,order='Oracle12 -> P10012 -> Full12 -> summary -> Solar Full8 -> missing diagnostics')
        time.sleep(30)
    # All GPU smoke and optimizer use is downstream of the explicit dependency.
    if not (t.OUT/'SMOKE_GATE.json').exists():launch([PY,'-u',str(t.HERE/'solar_full_train.py'),'--smoke'],'full_smoke',env)
    results=[]
    for h in [96,192,336,720]:
        for n in [0,5]:
            cell=f'Solar_{h}';launch([PY,'-u',str(t.HERE/'solar_full_train.py'),'--cell',cell,'--heads',str(n)],f'{cell}_V{n}',env)
            done=t.read(t.OUT/cell/f'V{n}'/'seed0/COMPLETE.json');assert done['status']=='COMPLETE';results.append(done)
    for h in [96,192,336,720]:
        launch([PY,'-u',str(t.HERE/'solar_full_diagnostic_completion.py'),f'Solar_{h}'],f'final_diagnostic_Solar_{h}',env)
    lines=['# Solar Full 후속 결과','기존 P100/Full 진단 종료 뒤에만 실행한 seed0 V0/V5 8개이다. Stage2/새 loss/추가 seed 없음.',
        '|Cell|Model|Epoch|Test RetMSE|Test AggMSE|Test AggMAE|','|---|---|---:|---:|---:|---:|']
    for row in results:
        lines.append(f"|{row['cell']}|{row['model']}|{row['selection']['epoch']}|{row['test']['retmse10']:.6f}|{row['test']['agg_mse10']:.6f}|{row['test']['agg_mae10']:.6f}|")
    lines+=['','학습 도중 모든 Full 후보에 대한 전역 softmax와 기존 z-score teacher를 사용했다. Query/Key 양쪽에 gradient가 흐르며 후보 bank를 학습용으로 고정하지 않았다.',
        'Grouped dropout RNG 및 Torch2/TF32-off 실행 경로는 역사적 ETTh1/Weather Full 실행과 구분한다. 원본 missing snapshot은 보존하고 새 진단 결과는 별도 TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-FULL-COMPLETION에 저장했다.']
    (t.OUT/'FINAL_REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    t.dump(t.OUT/'FINAL_GATE.json',dict(status='PASS',at=t.rt.now(),trained_models=8,completed_diagnostic_cells=4,seed=0,stage2=False))
    status(status='COMPLETE',trained_models=8,completed_diagnostic_cells=4)

if __name__=='__main__':
    try:main()
    except BlockingIOError:print('Existing post-Solar controller; no duplicate started')
    except BaseException as exc:
        status(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        t.dump(t.OUT/'IMMEDIATE_ISSUE.json',dict(at=t.rt.now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
