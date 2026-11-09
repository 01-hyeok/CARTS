import copy,tempfile,unittest
from pathlib import Path
import torch
from io_utils import dump,read,sha
from completed_cells import verified_completed_cell,release_key
from worker import canonical_p100_metrics

class MetricPrecision(unittest.TestCase):
    def test_original_dtype_without_changing_diagnostics(self):
        sd=torch.tensor([1e8]+[1.]*9,dtype=torch.float64).reshape(1,1,10).expand(2,7,10).clone()
        ids=torch.arange(10).reshape(1,1,10).expand(2,7,10)
        m=torch.zeros(2,7,4,dtype=torch.float64);m[...,0]=sd.mean(-1);m[...,1]=.25;m[...,2]=.125;m[...,3]=1
        ev=dict(selected_mse=sd,selected_ids=ids,oracle_ids=ids,metrics=m);before=m.clone()
        result=canonical_p100_metrics(ev)
        self.assertEqual(result.dtype,torch.float32)
        self.assertTrue(torch.equal(result[...,0],sd.float().mean(-1)))
        self.assertFalse(torch.equal(result[...,0].double(),m[...,0]))
        self.assertTrue(torch.equal(before,ev['metrics']))
        self.assertTrue(torch.equal(result[...,1:],m[...,1:].float()))
    def test_recall_counts_ids_not_rounded_fraction(self):
        ids=torch.arange(10).reshape(1,1,10).expand(3,7,10);oracle=ids+7
        ev=dict(selected_mse=torch.ones(3,7,10,dtype=torch.float64),selected_ids=ids,oracle_ids=oracle,metrics=torch.zeros(3,7,4,dtype=torch.float64))
        self.assertTrue(torch.equal(canonical_p100_metrics(ev)[...,3],torch.full((3,7),.3)))

class CompletedCell(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.cell='ETTh1_96';self.base=self.root/'cells'/self.cell
        source=self.root/'source.py';source.write_text('frozen-runtime\n');hs={'source.py':sha(source)}
        version=self.root/'00_Audit/runtime_releases'/release_key(hs);version.mkdir(parents=True);(version/'source.py').write_bytes(source.read_bytes());dump(version/'MANIFEST.json',dict(hashes=hs))
        self.version=version
        cp=self.root/'checkpoint.pth';cp.write_bytes(b'unchanged weights');self.cp=cp
        proof=dict(source_hashes=hs,checkpoint_sha256=sha(cp),data=dict(counts={'val':2,'test':2},channels=7))
        for support in ['P100','FULL']:
            for model in ['V0','V1','V5']:
                b=self.root/'checkpoint_runs'/self.cell/(model+'_'+support);files={};rows=[]
                for s in ['val','test']:
                    f=b/s/'batch00000.npz';f.parent.mkdir(parents=True);f.write_bytes(b'validated query block');h=sha(f);files[str(f.relative_to(b))]=h
                    dump(f.with_suffix('.json'),dict(status='PASS',identity=proof,sha256=h));dump(f.with_name('batch00000_ranking.json'),[])
                    dump(b/s/'SPLIT_GATE.json',dict(status='PASS',identity=proof,queries=2,channels=7,units=14))
                    rows.append(dict(split=s,ordered_mismatch_count=0,queries=2,units=14))
                dump(b/'CORE_COMPLETE.json',dict(status='CORE_COMPLETE_NOT_ALL_PHASES',identity=proof,files=files,query_counts={'val':2,'test':2},channels=7,source={'checkpoint':str(cp)}))
                dump(b/'REPRODUCTION_GATE.json',dict(status='PASS',identity=proof,rows=rows))
        dump(self.base/'AGGREGATION_GATE.json',dict(checkpoint_count=6,core_full_population=True,all_v1_present=True))
        dump(self.base/'CPU_SUPPLEMENT_GATE.json',dict(status='PASS',full_population=True,v1_included_both_training_supports=True,channels=7))
        self.rehash()
    def rehash(self):
        files={n:sha(self.base/n) for n in ['AGGREGATION_GATE.json','CPU_SUPPLEMENT_GATE.json']}
        dump(self.base/'CELL_ARTIFACT_GATE.json',dict(status='CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING',cell=self.cell,v1_included=True,files=files))
    def run_gate(self):return verified_completed_cell(self.root,self.cell)
    def test_complete_historical_runtime_accepted(self):self.assertTrue(self.run_gate())
    def test_missing_cell_not_complete(self):self.assertFalse(verified_completed_cell(self.root,'ETTh1_192'))
    def test_bad_report_hash_rejected(self):
        (self.base/'AGGREGATION_GATE.json').write_text('{}')
        with self.assertRaises(AssertionError):self.run_gate()
    def test_pending_supplement_rejected(self):
        p=self.base/'CPU_SUPPLEMENT_GATE.json';d=read(p);d['status']='PENDING';dump(p,d);self.rehash()
        with self.assertRaises(AssertionError):self.run_gate()
    def test_missing_v1_rejected(self):
        (self.root/'checkpoint_runs'/self.cell/'V1_FULL/CORE_COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_gate()
    def test_incomplete_population_rejected(self):
        p=self.root/'checkpoint_runs'/self.cell/'V0_FULL/test/SPLIT_GATE.json';d=read(p);d['queries']=1;dump(p,d)
        with self.assertRaises(AssertionError):self.run_gate()
    def test_changed_checkpoint_rejected(self):
        self.cp.write_bytes(b'changed weights')
        with self.assertRaises(AssertionError):self.run_gate()
    def test_changed_frozen_source_rejected(self):
        (self.version/'source.py').write_text('changed runtime')
        with self.assertRaises(AssertionError):self.run_gate()
    def test_changed_block_rejected(self):
        (self.root/'checkpoint_runs'/self.cell/'V0_FULL/test/batch00000.npz').write_bytes(b'changed query')
        with self.assertRaises(AssertionError):self.run_gate()

if __name__=='__main__':unittest.main()
