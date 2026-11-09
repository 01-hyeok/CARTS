"""Restart ONLY a stopped Mechanism02 controller after verified CPU repairs."""
import os,subprocess,shutil
from pathlib import Path
from protocol import ROOT
from io_utils import read,now,source_hashes
HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
if __name__=='__main__':
    g=read(OUT/'00_Audit/CPU_TEST_GATE.json');assert g['status']=='PASS' and g['runtime_hashes']==source_hashes()
    p=OUT/'00_Audit/CONTROLLER_STATUS.json'
    if p.exists():
        s=read(p);proc=Path('/proc')/str(s['controller_pid'])
        assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z','Existing controller alive; do not duplicate'
        stamp=now().replace(':','').replace('.','');backup=OUT/'00_Audit/failure_history'/stamp;backup.mkdir(parents=True)
        shutil.copy2(p,backup/'CONTROLLER_STATUS.json')
        for f in (OUT/'checkpoint_runs').glob('**/FAILED.json'):
            target=backup/f.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'logs/controller.log').open('a') as log:
        child=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'controller.py'),'--mode','all'],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('STARTED',child.pid,now())
