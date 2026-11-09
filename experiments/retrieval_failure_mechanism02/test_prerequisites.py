import json,tempfile,unittest,os
from pathlib import Path
from prerequisites import sha,verify_solar

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))

class Predecessor(unittest.TestCase):
    def setUp(self):
        # Use the explicitly writable temp root rather than the sandbox package temp.
        base=Path(os.environ['USERPROFILE'])/'AppData/Local/Temp' if os.name=='nt' else None
        self.tmp=tempfile.TemporaryDirectory(prefix='m02_',dir=base);self.root=Path(self.tmp.name)
        self.out=self.root/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04'
        write(self.out/'FINAL_GATE.json',dict(status='PASS',trained_models=8,completed_diagnostic_cells=4,seed=0,stage2=False))
        write(self.out/'CONTROLLER_STATUS.json',dict(status='COMPLETE',controller_pid=123))
        (self.out/'FINAL_REPORT_KO.md').write_text('verified report')
        for h in (96,192,336,720):
            cell=f'Solar_{h}'
            for m in ('V0','V5'):
                dest=self.out/cell/m/'seed0';cp=self.root/'checkpoints'/cell/m/'epoch01.pth'
                cp.parent.mkdir(parents=True)
                for ep in range(1,11):(cp.parent/f'epoch{ep:02d}.pth').write_bytes(str(ep).encode())
                hist=[dict(epoch=ep,val=dict(retmse10=ep,channels=137,all_channel_batch=True)) for ep in range(1,11)]
                write(dest/'history.json',hist);write(dest/'checkpoint_selection.json',dict(epoch=1,sha256=sha(cp),test_used_for_selection=False))
                write(dest/'COMPLETE.json',dict(status='COMPLETE',seed=0,checkpoint=str(cp),checkpoint_sha256=sha(cp),test=dict(channels=137,queries=12,units=12*137)))
            diag=self.root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-ALL-CHANNEL04/FULL'/cell
            source=diag/'COMPLETE.json'
            write(source,dict(status='COMPLETE',missing_models=[],channels=137,models=['V0','V5'],query_counts=dict(val=8,test=12)))
            rep=[dict(model=m,split=s,status='PASS',ordered_ids_exact=True,predictions_exact=True,channels=137,queries=n) for m in ('V0','V5') for s,n in [('val',8),('test',12)]]
            write(diag/'CANONICAL_REPRODUCTION.json',dict(status='PASS',rows=rep))
            supp=self.root/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04-DIAGNOSTIC-SUPPLEMENT/FULL'/cell
            base=dict(status='PASS',source_complete_sha256=sha(source))
            write(supp/'CPU_GATE.json',dict(**base,all_query_channels=True))
            splits=[dict(status='PASS',split=s,channels=137,queries=n,units=n*137,all_methods_ordered_top10_and_oracle_ids_exact=True) for s,n in [('val',8),('test',12)]]
            write(supp/'INFERENCE_GATE.json',dict(**base,splits=splits))
            write(supp/'DELIVERABLE_GATE.json',dict(**base,deferred=[],cpu_gate_sha256=sha(supp/'CPU_GATE.json'),inference_gate_sha256=sha(supp/'INFERENCE_GATE.json')))
    def tearDown(self):self.tmp.cleanup()
    def test_all_artifacts_pass(self):self.assertEqual(verify_solar(self.root)['status'],'PASS')
    def test_status_alone_insufficient(self):
        (self.out/'Solar_720/V5/seed0/COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):verify_solar(self.root)
    def test_missing_epoch_rejected(self):
        (self.root/'checkpoints/Solar_720/V5/epoch10.pth').unlink()
        with self.assertRaises(AssertionError):verify_solar(self.root)
    def test_missing_model_gap_gate_rejected(self):
        (self.root/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04-DIAGNOSTIC-SUPPLEMENT/FULL/Solar_96/DELIVERABLE_GATE.json').unlink()
        with self.assertRaises(FileNotFoundError):verify_solar(self.root)
    def test_modified_checkpoint_rejected(self):
        (self.root/'checkpoints/Solar_96/V0/epoch01.pth').write_bytes(b'changed')
        with self.assertRaises(AssertionError):verify_solar(self.root)
    def test_stale_source_rejected(self):
        p=self.root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-ALL-CHANNEL04/FULL/Solar_96/COMPLETE.json'
        p.write_text(p.read_text()+'\n')
        with self.assertRaises(AssertionError):verify_solar(self.root)
    def test_unfinished_owner_rejected(self):
        write(self.out/'CONTROLLER_STATUS.json',dict(status='RUNNING'))
        with self.assertRaises(AssertionError):verify_solar(self.root)
    def test_test_selection_rejected(self):
        p=self.out/'Solar_96/V0/seed0/checkpoint_selection.json';v=json.loads(p.read_text());v['test_used_for_selection']=True;write(p,v)
        with self.assertRaises(AssertionError):verify_solar(self.root)

if __name__=='__main__':unittest.main()
