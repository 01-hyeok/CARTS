"""Independent CPU-only follow-up of immutable, completed D5/D6 cases."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from d56_aggregate import OUT,sha,dump,now
HERE=Path(__file__).resolve().parent

def main():
    lock=(OUT/'d14_statistics.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures=[]
    while True:
        sources=sorted((OUT/'complete_analysis').glob('*/*/ANALYSIS_GATE.json'));done=[]
        for p in sources:
            g=json.loads(p.read_text());ident=g['identity'];args=[ident[k] for k in ['cell','model','training_support','evaluation_support']]
            gate=OUT/'d14_statistics'/p.parent.parent.name/p.parent.name/'SAMPLE_STATISTICS_GATE.json'
            if gate.exists():
                v=json.loads(gate.read_text());assert v['source_analysis_gate_sha256']==sha(p)
                for name,h in v['files'].items():assert sha(gate.parent/name)==h
                done.append(str(gate.relative_to(OUT)));continue
            if args in [x['case'] for x in failures]:continue
            log=OUT/'logs'/('D14_'+'_'.join(args)+'.log')
            cmd=['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'d14_sample_completion.py')]+args
            with log.open('a') as f:
                child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
                dump(OUT/'D14_STATISTICS_STATUS.json',dict(status='RUNNING_CPU_ONLY',at=now(),pid=os.getpid(),worker_pid=child.pid,case=args,completed=len(done),log=str(log)))
                rc=child.wait()
            if rc:failures.append(dict(case=args,returncode=rc,log=str(log)))
            else:assert gate.exists();done.append(str(gate.relative_to(OUT)))
            dump(OUT/'D14_STATISTICS_PROGRESS.json',dict(at=now(),complete=len(done),available=len(sources),expected=96,failures=failures,full_query_D1_D2_not_implied=True))
        state=json.loads((OUT/'CONTROLLER_STATUS.json').read_text())
        if state['status'] in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES']:
            dump(OUT/'D14_STATISTICS_STATUS.json',dict(status='COMPLETE_SAMPLE_STATISTICS' if len(done)==96 and not failures else 'PARTIAL_WITH_EXPLICIT_GAPS',at=now(),pid=os.getpid(),completed=len(done),failures=failures,source_controller_status=state['status']))
            return
        dump(OUT/'D14_STATISTICS_STATUS.json',dict(status='WAITING_NEXT_COMPLETED_FEATURE_CASE',at=now(),pid=os.getpid(),completed=len(done),failures=failures,gpu_used=False));time.sleep(20)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'D14_STATISTICS_STATUS.json',dict(status='BLOCKED_CPU',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
