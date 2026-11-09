import json
import os
from pathlib import Path
import pytest
import controller

def test_controller_sequential_inventory_and_restart_skip(monkeypatch,tmp_path):
    here=tmp_path/'owned';out=tmp_path/'out'
    (here/'configs').mkdir(parents=True);(here/'reports').mkdir()
    rows=[]
    for seed in range(3):
        for ds in ['ETTh1','Weather','Solar']:
            for h in [96,192,336,720]:
                for model in ['V0','V1','V2','V5']:
                    rid=f'{ds}_{h}_{model}_seed{seed}'
                    rows.append(dict(run_id=rid,dataset=ds,horizon=h,model=model,seed=seed,output_path=f'out/{rid}',status='NOT_STARTED',blocked_by=None))
    (here/'configs/run_manifest.json').write_text(json.dumps(rows))
    (here/'reports/RUNTIME_GATES.json').write_text(json.dumps(dict(status='PASS',source_hashes={})))
    monkeypatch.setattr(controller,'HERE',here);monkeypatch.setattr(controller,'ROOT',tmp_path);monkeypatch.setattr(controller,'OUT',out)
    monkeypatch.setattr(controller,'config',lambda:{})
    monkeypatch.setattr(controller,'source_hashes',lambda:{})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',controller.GPU_UUID)
    calls=[]
    pause_once=[True]
    def fake_execute(command,label,status):
        calls.append((command,label,status))
        if status=='POOL_GENERATION':
            cell=command[-1];p=out/'pools'/cell/'COMPLETE.json';controller.write_json(p,dict(status='PASS'))
        elif status=='STAGE1_RUNNING':
            rid=command[-1];ck=out/rid/'checkpoint.pth';ck.parent.mkdir(parents=True);ck.write_bytes(b'synthetic')
            controller.write_json(out/rid/'COMPLETE.json',dict(status='COMPLETE',run_id=rid,selected_checkpoint=str(ck),checkpoint_sha256=controller.sha256_file(ck)))
            if pause_once[0]:
                controller.write_json(out/'PAUSE_REQUESTED.json',dict(reason='synthetic pause'))
                pause_once[0]=False
    monkeypatch.setattr(controller,'execute',fake_execute)
    controller.main()
    assert json.loads((out/'current.json').read_text())['status']=='PAUSED_AT_RUN_BOUNDARY'
    assert len([x for x in calls if x[2]=='STAGE1_RUNNING'])==1
    (out/'PAUSE_REQUESTED.json').unlink()
    controller.main()
    assert [x[1] for x in calls if x[2]=='STAGE1_RUNNING']==[r['run_id'] for r in rows]
    assert len([x for x in calls if x[2]=='POOL_GENERATION'])==12
    assert json.loads((out/'current.json').read_text())['completed']==144
    assert all(r['status']=='COMPLETE' for r in json.loads((here/'configs/run_manifest.json').read_text()))
    calls.clear();controller.main()
    assert all(x[2]=='CPU_SUMMARY' for x in calls)
