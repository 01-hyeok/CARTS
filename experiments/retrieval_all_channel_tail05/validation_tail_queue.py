"""Sequential, fail-closed verification; does not launch production itself."""
import os,sys,json,subprocess,hashlib,datetime,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/pjh_workspace/CARTS')
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL05'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
PY='/data/pjh_workspace/ts-env/bin/python';LEGACY='/usr/bin/python3'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 q=p.with_suffix('.tmp.json');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def hashes():return {p.name:sha(p) for p in HERE.glob('*.py')}
def main():
 lock=(OUT/'validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source=hashes();env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 checks=[
  ('CPU_FINAL',[PY,'-m','pytest','-q','-p','no:cacheprovider',str(HERE/'test_all_tail_cpu.py'),str(HERE/'test_packed_contract.py')],False),
  ('ALL_TAIL_V0',[LEGACY,'-u',str(HERE/'regress_tail_gpu.py'),'--cell','Weather_192','--model','V0'],True),
  ('ALL_TAIL_V5',[LEGACY,'-u',str(HERE/'regress_tail_gpu.py'),'--cell','Weather_192','--model','V5'],True),
  ('PADDING_GPU',[LEGACY,'-u',str(HERE/'padding_gpu_gate.py')],True),
  ('WEATHER192_WHOLE_VAL_TEST',[LEGACY,'-u',str(HERE/'full_channel_run.py'),'FULL','Weather_192','--reproduction-only'],True),
 ]
 history=[]
 for name,cmd,gpu in checks:
  assert hashes()==source,'Repair source changed during verification'
  if gpu:assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True),'GPU1 occupied; no parallel validation'
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S');log=OUT/(name+'_'+stamp+'.log')
  start=now()
  with log.open('x') as out:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env if gpu else dict(env,CUDA_VISIBLE_DEVICES=''),stdout=out,stderr=subprocess.STDOUT)
   dump(OUT/'VALIDATION_CURRENT.json',dict(status='RUNNING',stage=name,worker_pid=proc.pid,controller_pid=os.getpid(),log=str(log),started_at=start,completed_checks=history,source_hashes=source))
   rc=proc.wait()
  history.append(dict(stage=name,returncode=rc,started_at=start,ended_at=now(),log=str(log)))
  assert rc==0,(name,rc,log)
 assert hashes()==source
 for name in ['Weather_192_V0_ALL_TAIL_GPU.json','Weather_192_V5_ALL_TAIL_GPU.json','REAL_GPU_PADDING_GATE.json']:
  assert read(OUT/name)['status']=='PASS',name
 assert read(OUT/'REMAINING_TAIL_PREFLIGHT.json')['status']=='PASS_WITH_SOLAR_MODEL_PARITY_NOT_APPLICABLE_MISSING_CHECKPOINT'
 assert read(OUT/'Weather_192/CANONICAL_REPRODUCTION.json')['status']=='PASS'
 evidence=[OUT/'Weather_192/CANONICAL_REPRODUCTION.json',OUT/'REMAINING_TAIL_PREFLIGHT.json',OUT/'REAL_GPU_PADDING_GATE.json',OUT/'Weather_192_V0_ALL_TAIL_GPU.json',OUT/'Weather_192_V5_ALL_TAIL_GPU.json']
 result=dict(status='PASS_BOUNDED_WEATHER192_WEATHER336',at=now(),source_hashes=source,checks=history,evidence_hashes={str(p):sha(p) for p in evidence},production_started=False,allowed_new_cells=['Weather_192','Weather_336'],blocked_before_cell='Weather_720',blocker='Weather720 V0 Torch2 tail canonical ordered-ID/prediction parity unresolved; no tolerance change or serial fallback',original_generation_runtime='UNKNOWN',no_serial_production_fallback=True,unchanged_tolerances=True,all_channels_gpu_batch=True,solar_model_gate='MISSING_NOT_IMPUTED')
 dump(OUT/'RELEASE_GATE.json',result);dump(HERE/'READY_GATE.json',result);dump(OUT/'VALIDATION_CURRENT.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(OUT/'VALIDATION_FAILED.json',dict(status='FAILED',at=now(),error=repr(e),traceback=traceback.format_exc(),controller_pid=os.getpid()));raise
