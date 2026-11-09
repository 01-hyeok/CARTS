"""CPU-only isolated fixtures. Does not touch production locks or markers."""
import copy,json,os,pathlib,subprocess,sys,tempfile,unittest,types,runpy
from unittest.mock import patch
import calibration_adapter as b

class Checks(unittest.TestCase):
    def test_upstream_exit2(self):
        def fake(*a,**k):return subprocess.CompletedProcess(a,2,'NOT_READY','')
        with self.assertRaises(b.NotReady):b.upstream(['fixture'],fake)
    def test_upstream_exit1(self):
        def fake(*a,**k):return subprocess.CompletedProcess(a,1,'hash mismatch','')
        with self.assertRaises(ValueError):b.upstream(['fixture'],fake)
    def test_identity_set(self):
        rows=[dict(label=x) for x in b.expected()];b.check_identity_set(rows)
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):b.check_identity_set(bad)
        wrong=copy.deepcopy(rows);wrong[0]['label']['cell']='Weather_999'
        with self.assertRaises(ValueError):b.check_identity_set(wrong)
    def test_mixed_resolution(self):
        records={}
        for x in b.expected():
            root='/fixture/original' if x['training']=='P100' else '/fixture/recovery02'
            records[b.key(x)]=dict(label=x,path=root+'/'+x['cell'],checkpoints={n:dict(path='/fixture/original/checkpoints/'+n+'/epoch08.pt',epoch=8) for n in b.ARMS})
        p=b.ResolvedPath(records)
        self.assertEqual(str(p/'FULL'/'Weather_96'/'SELECTION.json'),'/fixture/recovery02/Weather_96/SELECTION.json')
        self.assertEqual(str(p/'P100'/'Weather_96'/'FULL_RESULTS.json'),'/fixture/original/Weather_96/FULL_RESULTS.json')
        self.assertEqual(str(p/'FULL'/'Weather_96'/'checkpoints'/'D2_MLP'/'epoch08.pt'),'/fixture/original/checkpoints/D2_MLP/epoch08.pt')
        self.assertEqual(str(p/'supplemental/cross/V5_FULL_to_P100'/'Weather_96'/'FULL_RESULTS.json'),'/fixture/recovery02/Weather_96/FULL_RESULTS.json')
        with self.assertRaises(ValueError):str(p/'FULL/Weather_96/checkpoints/D2_MLP/epoch09.pt')
    def test_record_integrity_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp);encoder=root/'encoder.pt';encoder.write_bytes(b'encoder');d=root/'result';d.mkdir()
            identity=dict(data=dict(cell='Weather_96'),pool_hashes=dict(train='p'),checkpoint=dict(cell='Weather_96',training_support='FULL',seed='0',checkpoint=str(encoder),checkpoint_sha256=b.sha(encoder)))
            (d/'IDENTITY.json').write_text(json.dumps(identity));(d/'COMPLETE.json').write_text('{}')
            selection=dict(arms={n:dict(best_epoch=8) for n in b.ARMS});(d/'SELECTION.json').write_text(json.dumps(selection))
            cps={}
            for n in b.ARMS:
                p=root/(n+'.pt');p.write_bytes(n.encode());cps[n]=dict(path=str(p),sha256=b.sha(p),epoch=8)
            raw=d/'batch.npz';raw.write_bytes(b'raw')
            r=dict(path=str(d),label=b.label('native','Weather_96','FULL'),completion_path=str(d/'COMPLETE.json'),completion_sha256=b.sha(d/'COMPLETE.json'),selection_path=str(d/'SELECTION.json'),selection_sha256=b.sha(d/'SELECTION.json'),identity=identity,identity_sha256=b.sha(d/'IDENTITY.json'),checkpoints=cps,raw_hashes={str(raw):b.sha(raw)})
            b.check_record(r,root)
            for field in ['completion_sha256','selection_sha256','identity_sha256']:
                bad=copy.deepcopy(r);bad[field]='bad'
                with self.assertRaises(ValueError):b.check_record(bad,root)
            bad=copy.deepcopy(r);bad['identity']['pool_hashes']['train']='changed'
            with self.assertRaises(ValueError):b.check_record(bad,root)
            bad=copy.deepcopy(r);bad['checkpoints']['D2_MLP']['sha256']='bad'
            with self.assertRaises(ValueError):b.check_record(bad,root)
            raw.write_bytes(b'changed')
            with self.assertRaises(ValueError):b.check_record(r,root)
    def test_full_order(self):
        done={c:'PASS_ALL_QUERY_P100' for c in b.CELLS};receipt=dict(status='PASS_BRIDGE_P100',upstream_final_sha256='hash');b.full_prerequisites(receipt,done,'hash')
        for r,d,h in [({},done,'hash'),(receipt,{},'hash'),(receipt,done,'wrong')]:
            with self.assertRaises(ValueError):b.full_prerequisites(r,d,h)
    def test_zero_replay(self):
        d=dict(status='P100_LOCAL_RANKING_COMPLETE_GPU',replay=[dict(cell=c,split=s,ordered_id_mismatches=dict(D2_MLP=0,C2_MLP=0)) for c in b.CELLS for s in ['val','test']]);b.replay_gate(d)
        for value in [1,-1,0.00001,False]:
            bad=copy.deepcopy(d);bad['replay'][0]['ordered_id_mismatches']['D2_MLP']=value
            with self.assertRaises(ValueError):b.replay_gate(bad)
        bad=copy.deepcopy(d);bad['status']='P100_LOCAL_RANKING_COMPLETE'
        with self.assertRaises(ValueError):b.replay_gate(bad)
        gate=dict(status='PASS',upstream_final_sha256='final',replay_sha256='replay')
        b.p0_gate(gate,d,'final','replay')
        for g in [{},dict(gate,status='PARTIAL'),dict(gate,replay_sha256='old')]:
            with self.assertRaises(ValueError):b.p0_gate(g,d,'final','replay')
        bad=copy.deepcopy(d);bad['replay'][-1]=bad['replay'][0]
        with self.assertRaises(ValueError):b.replay_gate(bad)
    def test_gpu_rules(self):
        good='1,'+b.GPU+',4,0';b.idle(good,'')
        for info,apps in [(good,b.GPU+',4250'),('1,'+b.GPU+',128,0',''),('1,'+b.GPU+',4,1',''),('0,'+b.GPU+',4,0','')]:
            with self.assertRaises(ValueError):b.idle(info,apps)
        calls=[];b.gpu_recheck(lambda:calls.append('check'),lambda n:calls.append(n));self.assertEqual(calls,['check',30,'check'])
    def test_isolated_locks(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'lock';p.write_bytes(b'fixture');before=p.read_bytes()
            with b.locks([p]):
                code='import fcntl,sys; f=open(sys.argv[1],"rb"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                r=subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True);self.assertNotEqual(r.returncode,0)
            with b.locks([p]):pass
            self.assertEqual(p.read_bytes(),before)
            with self.assertRaises(FileNotFoundError):
                with b.locks([pathlib.Path(t)/'missing']):pass
    def test_target_module_binding(self):
        # Prove functions use the injected module global, not a source rewrite.
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'target.py';p.write_text('OLD=None\ndef run(): return str(OLD / "FULL" / "Weather_96" / "FULL_RESULTS.json")\n')
            spec=b.importlib.util.spec_from_file_location('fixture_target',p);m=b.importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            r=dict(label=b.label('native','Weather_96','FULL'),path='/fixture/recovered');m.OLD=b.ResolvedPath({b.key(r['label']):r})
            self.assertEqual(m.run(),'/fixture/recovered/FULL_RESULTS.json')
    def test_actual_readiness_exit_contract(self):
        # Execute the real upstream entry point with isolated dependencies/files.
        with tempfile.TemporaryDirectory() as t:
            root=pathlib.Path(t);runtime=types.ModuleType('recovery_runtime')
            runtime.OUT=root;runtime.read=b.read;runtime.sha=b.sha;runtime.json=json;runtime.sys=sys;runtime.now=lambda:'fixture'
            validator=types.ModuleType('recovery02_validate')
            class Incomplete(Exception):pass
            validator.Incomplete=Incomplete;validator.validate_all=lambda full:None
            with patch.dict(sys.modules,{'recovery_runtime':runtime,'recovery02_validate':validator}):
                with self.assertRaises(SystemExit) as ctx:runpy.run_path(str(b.UP/'readiness.py'),run_name='__main__')
                self.assertEqual(ctx.exception.code,2)
                (root/'final_all_comparisons').mkdir();(root/'final_all_comparisons/COMPLETE.json').write_text(json.dumps(dict(status='CORRUPTED')))
                with self.assertRaises(SystemExit) as ctx:runpy.run_path(str(b.UP/'readiness.py'),run_name='__main__')
                self.assertEqual(ctx.exception.code,1)

if __name__=='__main__':unittest.main(verbosity=2)

