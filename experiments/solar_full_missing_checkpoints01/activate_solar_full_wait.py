"""Install a CPU-only dependency waiter; GPU smoke remains strictly deferred."""
import ast,datetime,json,os,subprocess
from pathlib import Path
import solar_full_train as t

def main():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        assert str(t.HERE/'solar_full_controller.py').encode() not in args,'Existing waiter; no duplicate'
    for p in t.HERE.glob('*.py'):ast.parse(p.read_text())
    t.OUT.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    xml=t.OUT/'CPU_TESTS.xml'
    subprocess.run(['/data/pjh_workspace/ts-env/bin/python','-m','pytest','-q','-p','no:cacheprovider',str(t.HERE/'test_solar_full_core.py'),str(t.HERE/'test_solar_full_queue.py'),'--junitxml='+str(xml)],cwd=t.ROOT,env=env,check=True)
    t.dump(t.OUT/'CPU_READY_GATE.json',dict(status='CPU_PASS_GPU_SMOKE_DEFERRED',at=t.rt.now(),source_hashes=t.source_hashes(),tests=str(xml),actual_training_started=False))
    logs=t.OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/'controller.log').open('a') as f:
        proc=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python','-u',str(t.HERE/'solar_full_controller.py')],cwd=t.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(at=t.rt.now(),controller_pid=proc.pid,status='CPU_WAITER_LAUNCHED',log=str(logs/'controller.log'),actual_training_started=False,required_order='all48 complete -> Oracle12 -> P10012 -> Full12 -> summary PASS -> Solar Full8 -> missing diagnostic supplement')
    t.dump(t.OUT/'LAUNCH.json',state);print(json.dumps(state))

if __name__=='__main__':main()
