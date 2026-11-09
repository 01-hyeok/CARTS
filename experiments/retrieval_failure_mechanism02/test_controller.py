import unittest,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('mechanism02_controller_under_test',Path(__file__).with_name('controller.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
targets,runtime=mod.targets,mod.runtime

class Queue(unittest.TestCase):
    def test_all_v1_present(self):
        t=targets();self.assertEqual(len(t),48);self.assertEqual(len(set(t)),48)
        self.assertEqual(sum(m=='V1' for c,m,s in t),16)
    def test_order_and_no_solar_retraining(self):
        t=targets();self.assertEqual(t[0],('ETTh1_96','V0','P100'))
        self.assertTrue(all(not c.startswith('Solar') for c,m,s in t))
    def test_runtime_explicit(self):
        self.assertEqual(runtime('Weather_720','V0','FULL'),'/data/pjh_workspace/ts-env/bin/python')
        self.assertEqual(runtime('Weather_720','V1','FULL'),'/data/pjh_workspace/ts-env/bin/python')
        self.assertEqual(runtime('Weather_720','V1','P100'),'/data/pjh_workspace/ts-env/bin/python')
    def test_v1_cache_verified_runtime_is_cell_specific(self):
        for cell in ['ETTh1_96','ETTh1_720','Weather_96','Weather_720']:
            self.assertEqual(runtime(cell,'V1','FULL'),'/data/pjh_workspace/ts-env/bin/python')
        for cell in ['ETTh1_192','ETTh1_336','Weather_192','Weather_336']:
            self.assertEqual(runtime(cell,'V1','FULL'),'/usr/bin/python3')

if __name__=='__main__':unittest.main()
