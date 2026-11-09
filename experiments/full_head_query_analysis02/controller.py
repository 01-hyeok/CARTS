import os,sys,json,subprocess,time,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02';lock=open(O/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def save(x):(O/'STATUS.json').write_text(json.dumps(x,indent=2))
for c in ['ETTh1_96','ETTh1_720','Weather_96','Weather_720']:
 for script,done,python in [('extract.py','EXTRACTION_PASS.json','/usr/bin/python3'),('analyze.py','ANALYSIS_PASS.json','/data/pjh_workspace/ts-env/bin/python')]:
  if (O/c/done).exists():continue
  log=O/(c+'_'+script.replace('.py','.log'));env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if script=='extract.py' else '',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  with open(log,'a') as f:
   p=subprocess.Popen([python,'-u',str(R/'experiments/full_head_query_analysis02'/script),'--cell',c],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);save(dict(status='RUNNING',cell=c,phase=script,controller_pid=os.getpid(),worker_pid=p.pid,log=str(log),time=time.time()));rc=p.wait()
  if rc:save(dict(status='FAILED',cell=c,phase=script,returncode=rc,log=str(log),time=time.time()));sys.exit(rc)
 print('[CELL COMPLETE]',c,flush=True)
save(dict(status='FOUR_CELLS_COMPUTED_REPORT_PENDING',time=time.time()));print('ALL COMPUTED',flush=True)
