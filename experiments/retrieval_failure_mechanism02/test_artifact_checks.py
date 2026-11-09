import unittest,tempfile,os
from pathlib import Path
import numpy as np
from io_utils import dump,save,sha
from artifact_checks import validated_blocks,strict_head_choice,geometry_choice

class Artifacts(unittest.TestCase):
    def fixture(self,path):
        proof={'data':{'counts':{'test':3},'starts':{'test':[10,12]}}}
        dump(path/'REPRODUCTION_GATE.json',{'identity':proof})
        dump(path/'test/SPLIT_GATE.json',dict(status='PASS',identity=proof,channels=2,queries=3,units=6))
        p=path/'test/batch00000.npz';save(p,{'query_start':np.arange(10,13),'FULL__metrics':np.zeros((3,2,25))})
        dump(p.with_suffix('.json'),dict(status='PASS',identity=proof,channels=2,queries=3,sha256=sha(p)))
        return p
    def temp(self):return tempfile.TemporaryDirectory(dir=os.environ.get('TEMP','/tmp'))
    def test_complete(self):
        with self.temp() as t:
            p=Path(t);self.fixture(p);self.assertEqual(len(validated_blocks(p,'test')[1]),3)
    def test_modified_rejected(self):
        with self.temp() as t:
            p=Path(t);f=self.fixture(p);f.write_bytes(f.read_bytes()+b'changed')
            with self.assertRaises(AssertionError):validated_blocks(p,'test')
    def test_missing_query_rejected(self):
        with self.temp() as t:
            p=Path(t);f=self.fixture(p);save(f,{'query_start':np.array([10,11,13]),'FULL__metrics':np.zeros((3,2,25))})
            import json
            s=json.loads(f.with_suffix('.json').read_text());s['sha256']=sha(f);dump(f.with_suffix('.json'),s)
            with self.assertRaises(AssertionError):validated_blocks(p,'test')
    def test_validation_head_selection_not_test(self):
        val=np.ones((4,3,5,25));val[:,:,2,0]=0
        self.assertEqual(strict_head_choice(val),2)
    def test_geometry_null_without_control_advantage(self):
        p=np.zeros((4,3,13));p[...,3:6]=[.9,5,.9]
        self.assertEqual(geometry_choice(p,p),1)

if __name__=='__main__':unittest.main()
