"""Completed-state CPU fixtures and isolated lifecycle/concurrency checks."""
import copy,json,os,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import calibration_adapter as b
import lifecycle as life
import preflight

def fixture(root):
    import numpy as np,torch
    records=[];native={}
    for number,l in enumerate(b.expected()):
        p=root/('original' if number<8 else 'recovery02')/str(number);p.mkdir(parents=True)
        k=(l['cell'],l['training']);encoder=root/(l['cell']+'_'+l['training']+'.encoder');encoder.write_bytes(str(k).encode())
        identity=dict(data=dict(cell=l['cell'],channels=1,counts=dict(train=2,val=1,test=1)),pool_hashes=dict(train='t',val='v',test='s'),checkpoint=dict(cell=l['cell'],training_support=l['training'],seed='0',checkpoint=str(encoder),checkpoint_sha256=b.sha(encoder)))
        selection=dict(status='REUSED_NATIVE_SELECTION_NO_REFIT' if l['mode']=='cross' else 'FROZEN_BEFORE_FINAL_TEST',arms={n:dict(best_epoch=8,best=1.) for n in b.ARMS})
        cps={}
        if l['mode']=='cross':cps=copy.deepcopy(native[k]['checkpoints'])
        else:
            for n in b.ARMS:
                q=p/(n+'.pt');torch.save(dict(epoch=8,model=dict(weight=torch.zeros(1)),state=dict(best_epoch=8,best=1.)),q);cps[n]=dict(path=str(q),sha256=b.sha(q),epoch=8)
        for name,value in [('COMPLETE.json',{}),('SELECTION.json',selection),('IDENTITY.json',identity)]: (p/name).write_text(json.dumps(value))
        raw={};dirs={};rows=[]
        for split in ['val','test']:
            d=p/'final_raw'/split;d.mkdir(parents=True);dirs[split]=str(d);q=d/'batch00000.npz'
            values=dict(query_start=np.array([100]))
            for m in ['S0','D2_MLP','C2_MLP']:
                values[m+'__selected_ids']=np.arange(10).reshape(1,1,10);values[m+'__selected_mse']=np.zeros((1,1,10))
            np.savez(q,**values);raw[str(q)]=b.sha(q);q.with_suffix('.json').write_text(json.dumps(dict(sha256=b.sha(q))))
            for m in ['S0','D0','C0']+b.ARMS+['Oracle10','Random','LastRepeat']:rows.append(dict(split=split,method=m,queries=1,channels=1,support=l['support'],seed=0,retmse=1.,agg_mse=1.,agg_mae=1.))
        (p/'FULL_RESULTS.json').write_text(json.dumps(rows))
        r=dict(label=l,path=str(p),identity=identity,identity_sha256=b.sha(p/'IDENTITY.json'),completion_path=str(p/'COMPLETE.json'),completion_sha256=b.sha(p/'COMPLETE.json'),selection_path=str(p/'SELECTION.json'),selection_sha256=b.sha(p/'SELECTION.json'),checkpoints=cps,raw_hashes=raw,raw_directories=dirs,queries=dict(val=1,test=1),channels=1)
        records.append(r)
        if l['mode']=='native':native[k]=r
    return dict(records=records)

class Tests(unittest.TestCase):
    def test_completed_input_fixture_and_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            root=pathlib.Path(t);data=fixture(root)
            result=preflight.input_records(data,root,connect=False);self.assertEqual(result['records'],40);self.assertEqual(result['raw_blocks'],80)
            bad=copy.deepcopy(data);bad['records'].pop()
            with self.assertRaises(ValueError):preflight.input_records(bad,root,connect=False)
            r=data['records'][0];p=pathlib.Path(r['path'])/'FULL_RESULTS.json';rows=json.loads(p.read_text());rows[0]['channels']=2;p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):preflight.input_records(data,root,connect=False)
    def test_raw_symlink_approved_and_escape(self):
        with tempfile.TemporaryDirectory() as t,tempfile.TemporaryDirectory() as outside:
            root=pathlib.Path(t);data=fixture(root);r=data['records'][0];raw=pathlib.Path(next(iter(r['raw_hashes'])));saved=raw.read_bytes();target=root/'preserved.npz';target.write_bytes(saved);raw.unlink();raw.symlink_to(target)
            b.check_record(r,root)
            raw.unlink();escape=pathlib.Path(outside)/'raw.npz';escape.write_bytes(saved);raw.symlink_to(escape)
            with self.assertRaises(ValueError):b.check_record(r,root)
    def test_pid_identity(self):
        saved=life.identity(os.getpid());self.assertTrue(life.same_process(saved,life.identity(os.getpid())))
        for k,v in [('start_ticks',saved['start_ticks']+1),('boot_id','new'),('command_sha256','new'),('state','Z')]:
            other=dict(saved);other[k]=v;self.assertFalse(life.same_process(saved,other))
        self.assertFalse(life.same_process(saved,None))
    def test_at_most_once_concurrent(self):
        with tempfile.TemporaryDirectory() as t:
            code='import sys;sys.path.insert(0,sys.argv[2]);import lifecycle as l;l.Stage(sys.argv[1],"p100",{})'
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');ps=[subprocess.Popen([sys.executable,'-c',code,t,str(pathlib.Path(__file__).parent)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env) for _ in range(4)]
            codes=[]
            for p in ps:p.communicate();codes.append(p.returncode)
            self.assertEqual(codes.count(0),1)
            before=(pathlib.Path(t)/'P100_STARTED.json').read_bytes()
            with self.assertRaises(FileExistsError):life.Stage(t,'p100',{})
            self.assertEqual(before,(pathlib.Path(t)/'P100_STARTED.json').read_bytes())
    def test_worker_record_and_failure(self):
        with tempfile.TemporaryDirectory() as t:
            stage=life.Stage(t,'p100',dict(hash='fixture'));p=subprocess.Popen([sys.executable,'-c','pass']);stage.started(p,[sys.executable,'-c','pass'],dict(step='fixture'));code=p.wait();stage.ended(code);stage.complete()
            state=json.loads((pathlib.Path(t)/'launch_p100/STATE.json').read_text());self.assertTrue(state['workers'][0]['wait_reaped']);self.assertEqual(state['workers'][0]['exit_code'],0)
            with self.assertRaises(ValueError):life.prior_finished(t,'p100') # test supervisor is still alive
            stage.failed(ValueError('fixture'));self.assertEqual(json.loads((pathlib.Path(t)/'launch_p100/STATE.json').read_text())['status'],'FAILED')
    def test_stale_corrupt_claim_no_takeover(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'P100_STARTED.json';p.write_text('{incomplete')
            with self.assertRaises(FileExistsError):life.Stage(t,'p100',{})
            self.assertEqual(p.read_text(),'{incomplete')
    def test_preflight_missing_upstream_does_not_claim(self):
        with patch.object(b,'seals'),patch.object(b,'upstream',side_effect=b.NotReady('fixture')):
            with self.assertRaises(b.NotReady):preflight.run()
    def test_finished_stage_after_supervisor_exit(self):
        with tempfile.TemporaryDirectory() as t:
            code='import sys;sys.path.insert(0,sys.argv[2]);import lifecycle as l;s=l.Stage(sys.argv[1],"p100",{});s.complete()'
            r=subprocess.run([sys.executable,'-c',code,t,str(pathlib.Path(__file__).parent)],capture_output=True)
            self.assertEqual(r.returncode,0);self.assertEqual(life.prior_finished(t,'p100')['status'],'COMPLETE')
            p=pathlib.Path(t)/'launch_p100/STATE.json';d=json.loads(p.read_text());d['run_id']='other';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):life.prior_finished(t,'p100')
    def test_gpu_check_before_and_after_locks_no_launch(self):
        import contextlib
        class StopFixture(Exception):pass
        with tempfile.TemporaryDirectory() as t:
            root=pathlib.Path(t);final=root/'final';final.write_text('fixture');calls=[]
            data=dict(ready=dict(final_complete=str(final),final_sha256=b.sha(final)));proof=dict(bindings={})
            @contextlib.contextmanager
            def fake_locks(paths):
                calls.append('locks');yield []
            def stop(*args):calls.append('claim');raise StopFixture()
            with patch.object(b,'OUT',root),patch.object(b,'RESULT',root/'cal'),patch.object(preflight,'run',return_value=(data,proof)),patch.object(b,'gpu_check',side_effect=lambda:calls.append('before')),patch.object(b,'locks',fake_locks),patch.object(b,'gpu_recheck',side_effect=lambda:calls.append('after_twice')),patch.object(b,'seals'),patch.object(life,'Stage',side_effect=stop):
                with self.assertRaises(StopFixture):b.launch('p100')
            self.assertEqual(calls,['before','locks','after_twice','claim'])

if __name__=='__main__':unittest.main(verbosity=2)

