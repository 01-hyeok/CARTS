import importlib.util
import json
from pathlib import Path
import pytest

def load(name):
    path=Path(__file__).parent/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
c=load('controller_seed0');s=load('summarize_seed0')

def prepare_case(monkeypatch,tmp):
    here=tmp/'owned';out=tmp/'out';(here/'configs').mkdir(parents=True);(here/'reports').mkdir()
    rows=[dict(run_id=f'{d}_{h}_{m}_seed0',dataset=d,horizon=h,model=m,seed=0,output_path=f'out/{d}_{h}_{m}_seed0',status='NOT_STARTED',blocked_by=None) for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720) for m in ('V0','V1','V2','V5')]
    c.write_json(here/'configs/run_manifest.json',rows);c.write_json(here/'reports/RUNTIME_GATES.json',dict(status='PASS',source_hashes={}))
    monkeypatch.setattr(c,'HERE',here);monkeypatch.setattr(c,'ROOT',tmp);monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setattr(c,'config',lambda:dict(execution=dict(seed_order=[0],total_runs=48)))
    monkeypatch.setattr(c,'source_hashes',lambda:{});monkeypatch.setattr(c,'track_workers',lambda:[])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',c.GPU_UUID)
    return here,out,rows

def complete(row,tmp):
    dest=tmp/row['output_path'];ck=dest/'checkpoint.pth';dest.mkdir(parents=True,exist_ok=True);ck.write_bytes(b'synthetic')
    c.write_json(dest/'COMPLETE.json',dict(status='COMPLETE',run_id=row['run_id'],selected_checkpoint=str(ck),checkpoint_sha256=c.sha256_file(ck)))

def test_exact48_sequence_stops_and_restart_skips(monkeypatch,tmp_path):
    here,out,rows=prepare_case(monkeypatch,tmp_path);calls=[]
    def fake(command,label,status):
        calls.append((label,status))
        if status=='POOL_GENERATION':c.write_json(out/'pools'/command[-1]/'COMPLETE.json',dict(status='PASS'))
        if status=='STAGE1_RUNNING':complete(next(r for r in rows if r['run_id']==label),tmp_path)
    monkeypatch.setattr(c,'execute',fake);c.main()
    assert [label for label,state in calls if state=='STAGE1_RUNNING']==[r['run_id'] for r in rows]
    assert len([1 for _,state in calls if state=='POOL_GENERATION'])==12
    status=json.loads((out/'current.json').read_text());assert status['completed']==status['total']==48 and status['cancelled_by_user']==96
    assert json.loads((out/'progress.json').read_text())['completed']==48
    calls.clear();c.main();assert all(state=='CPU_SUMMARY' for _,state in calls)

@pytest.mark.parametrize('change',['seed','length','order'])
def test_reject_unapproved_inventory(monkeypatch,tmp_path,change):
    _,_,rows=prepare_case(monkeypatch,tmp_path)
    if change=='seed':rows[-1]['seed']=1
    if change=='length':rows.append(dict(rows[-1]))
    if change=='order':rows.reverse()
    with pytest.raises(AssertionError):c.validate_manifest(rows)

def adoption(monkeypatch,tmp_path):
    here,out,rows=prepare_case(monkeypatch,tmp_path);r=rows[34];r['status']='RUNNING'
    identity=dict(pid=12345,start_ticks='123',argv=['python',str(here/'runtime.py'),'train',r['run_id']])
    c.write_json(out/'SEED0_ADOPTION.json',dict(run_id=r['run_id'],identity=identity,previous_current=dict(target=r['run_id'])))
    return out,rows,r,identity

def test_adopt_active_worker_waits_no_duplicate(monkeypatch,tmp_path):
    out,rows,r,identity=adoption(monkeypatch,tmp_path);alive=[True];sleeps=[]
    monkeypatch.setattr(c,'proc_identity',lambda pid:identity if alive[0] else None)
    monkeypatch.setattr(c,'track_workers',lambda:[identity])
    def sleep(seconds):sleeps.append(seconds);complete(r,tmp_path);alive[0]=False
    monkeypatch.setattr(c.time,'sleep',sleep)
    monkeypatch.setattr(c,'execute',lambda *args:pytest.fail('Duplicate execution'))
    c.adopt_existing(rows);assert sleeps==[30] and r['status']=='COMPLETE'
    assert json.loads((out/'SEED0_ADOPTION_COMPLETE.json').read_text())['worker_restarted'] is False

def test_incomplete_adopted_run_stops_queue(monkeypatch,tmp_path):
    _,rows,_,_=adoption(monkeypatch,tmp_path);monkeypatch.setattr(c,'proc_identity',lambda pid:None)
    with pytest.raises(AssertionError,match='without COMPLETE'):c.adopt_existing(rows)

def test_changed_pid_identity_rejected(monkeypatch,tmp_path):
    _,rows,_,identity=adoption(monkeypatch,tmp_path)
    monkeypatch.setattr(c,'proc_identity',lambda pid:dict(identity,start_ticks='different'))
    with pytest.raises(AssertionError,match='identity changed'):c.adopt_existing(rows)

def test_duplicate_runtime_prevents_launch(monkeypatch,tmp_path):
    prepare_case(monkeypatch,tmp_path);monkeypatch.setattr(c,'track_workers',lambda:[dict(pid=99)])
    with pytest.raises(AssertionError,match='duplicated'):c.execute(['never-execute'],'Solar_96_V5_seed0','STAGE1_RUNNING')

def test_seed0_only_summary_and_cancellation(monkeypatch,tmp_path):
    here,out,rows=prepare_case(monkeypatch,tmp_path)
    cancelled=[dict(run_id=f'{d}_{h}_{m}_seed{seed}',status='CANCELLED_BY_USER',seed=seed) for seed in (1,2) for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720) for m in ('V0','V1','V2','V5')]
    c.write_json(here/'configs/cancelled_runs.json',cancelled)
    monkeypatch.setattr(s,'HERE',here);monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'OUT',out)
    s.main();report=(out/'FINAL_REPORT_KO.md').read_text()
    assert '0/48' in report and '/144' not in report and 'CANCELLED_BY_USER' in report
    assert len((out/'CANCELLED_RUNS.csv').read_text().splitlines())==97
