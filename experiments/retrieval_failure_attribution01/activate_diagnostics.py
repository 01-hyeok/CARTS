"""Run only after approved deployment: final CPU gates, source pin, durable wait queue."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=ROOT/'experiments/retrieval_failure_attribution01';OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
PY='/data/pjh_workspace/ts-env/bin/python';LEGACY='/usr/bin/python3'

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    import xml.etree.ElementTree as ET
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        assert str(HERE/'diagnostic_controller.py').encode() not in argv,'Existing diagnostic controller: do not duplicate'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    tests=[]
    for executable,name in [(PY,'CPU_RUNTIME_TESTS.xml'),(LEGACY,'LEGACY_CPU_RUNTIME_TESTS.xml')]:
        path=OUT/'00_Audit'/name
        command=[executable,'-m','pytest','-q','-p','no:cacheprovider',str(HERE/'test_diagnostic_core.py'),str(HERE/'test_diagnostic_runtime.py'),'--junitxml='+str(path)]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else list(root)
        count=sum(int(x.attrib.get('tests',0)) for x in suites);fail=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
        assert count>=46 and fail==0
        tests.append(dict(runtime=executable,passed=count,failed=fail,evidence=str(path)))
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    protocol['probe'].update(implementation='streaming Ridge regression only; fixed alphas [0.1,1,10]',train_query_budget_per_channel=256,
        train_query_selection='uniform deterministic row grid, future-free',full_train_candidate_sample='100 eligible IDs uniformly sampled without replacement per query/channel; seed0; no future selection',
        evaluation='ALL validation/test queries, channels and candidates in evaluation support',selection_objective='mean validation future-MSE regression error over all eligible candidate pairs',
        primary_evaluation_not_subsampled=True)
    protocol['baseline_numerics']=dict(cosine_eps=1e-12,znorm_std='population; denominator std+1e-6',calendar_anchor='last observed history timestamp',period='at least two cycles; otherwise undefined',
        historical_full_v0='raw cosine TopK (single-slot legacy argmax); preserve FP tie behavior',full_v5='canonical global per-head softmax mean, never per-chunk softmax')
    protocol.setdefault('implementation_ready_at',datetime.datetime.now(datetime.timezone.utc).isoformat())
    protocol['scheduler']['required_predecessor_before_p100']='ORACLE_K_SUPPLEMENT_FINAL_GATE_PASS'
    protocol['stage_order']=['ORACLE_K_SUPPLEMENT_ALL_12','P100_ALL_12','FULL_ALL_12','FILL_DEFERRED_FULL_DEPENDENCIES','SOLAR_FULL8_AFTER_DIAGNOSTIC_SUMMARY']
    write(HERE/'PROTOCOL.json',protocol)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list(HERE.glob('*.py'))+[HERE/'PROTOCOL.json'])}
    write(HERE/'READY_GATE.json',dict(status='CPU_READY_REAL_GPU_GATE_AT_EXECUTION',at=protocol['implementation_ready_at'],tests=tests,source_sha256=hashes,
        gpu_smoke='deferred until existing48 release GPU1; fail closed before attribution/probe',all_original_results_preserved=True))
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/'controller.log').open('a') as f:
        proc=subprocess.Popen([PY,'-u',str(HERE/'diagnostic_controller.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'00_Audit/LAUNCH.json',dict(at=protocol['implementation_ready_at'],controller_pid=proc.pid,log=str(logs/'controller.log'),state='WAITING_EXISTING_TRAINING',actual_diagnostic_gpu_started=False,
        order='existing48 -> Oracle K supplement12 -> P10012 -> Full12 -> summary; Solar Full8 afterwards under separate training gates'))
    print(json.dumps(dict(controller_pid=proc.pid,tests=tests,log=str(logs/'controller.log'))))

if __name__=='__main__':main()
