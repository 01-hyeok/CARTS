import os,subprocess,time,shutil,json
from full_repair_controller import *
def main():
    assert not old.gpu_apps(),'GPU1 is busy; do not launch concurrent work'
    assert read(HERE/'READY_GATE.json')['source_hashes']==hashes()
    assert read(SOURCE/'00_Audit/FULL_REPAIR01/ETTh1_96/PRECISION_GATE.json')['status']=='PASS'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        argv=queue.command(int(p.name))
        assert not any(x.endswith('/full_repair_controller.py') for x in argv),'Existing repair controller; no duplicate'
    audit=SOURCE/'00_Audit/FULL_REPAIR01';archive=audit/'preserved_failed_attempt_20261030_2207'
    if not archive.exists():
        src=SOURCE/'FULL/ETTh1_96';assert not (src/'COMPLETE.json').exists()
        assert sum(p.stat().st_size for p in src.rglob('*') if p.is_file())<100000000
        shutil.copytree(src,archive)
        for f in ['CONTROLLER_STATUS.json','IMMEDIATE_ISSUE.json']:
            p=SOURCE/'00_Audit'/f
            if p.exists():shutil.copy2(p,archive/f)
    failed=SOURCE/'FULL/ETTh1_96/FAILED.json'
    if failed.exists():failed.rename(failed.with_name('FAILED_attempt1_2207.json'))
    log=audit/'controller_repaired.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with log.open('a') as f:p=subprocess.Popen([old.PY,'-u',str(HERE/'full_repair_controller.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump(audit/'ACTIVATION.json',dict(status='STARTED',controller_pid=p.pid,log=str(log),at=old.now(),order='verified P10012 -> FULL12+supplements -> summary -> held Solar8',original_failed_attempt_preserved=str(archive)))
    print('ACTIVATED',p.pid,str(log),flush=True)
if __name__=='__main__':main()
