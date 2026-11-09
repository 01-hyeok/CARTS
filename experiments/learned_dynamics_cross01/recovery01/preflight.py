import sys,os,json,importlib.util,fcntl
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from experiment_common import OUT,ROOT,sha,dump,now,sources
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=module('recovery_controller_check',P/'controller.py')
r=module('recovery_resume_check',P/'resume.py')
assert c.RECOVERY==r.RECOVERY==P
assert Path(c.native.__file__).resolve()==P.parent/'controller.py'
assert json.loads((OUT/'CPU_IMPLEMENTATION_GATE.json').read_text())['source_hashes']==sources()
assert json.loads((OUT/'recovery01/COMPATIBILITY_GATE.json').read_text())['compat_sha256']==sha(P/'compat.py')
manifest=json.loads((OUT/'recovery01/PRESERVATION_MANIFEST.json').read_text())
for path,h in manifest['files'].items():assert sha(ROOT/path)==h,path
for path in (OUT/'CONTROLLER.lock',OUT/'GPU.lock',OUT/'supplemental/controller.lock',OUT/'recovery01/controller.lock'):
    with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'recovery01/evaluation/ETTh1_96_FULL').exists()
assert not (OUT/'recovery01/evaluation_resume.log').exists()
for name in ('final_raw','FULL_RESULTS.json','FULL_RESULTS.csv','FINAL_VAL_GATE.json','FINAL_TEST_GATE.json','COMPLETE.json'):
    assert not (OUT/'FULL/ETTh1_96'/name).exists(),name
dump(OUT/'recovery01/SOURCE_SEAL.json',dict(sources={f.name:sha(f) for f in P.glob('*.py')},at=now(),preflight='PASS imports, original preservation hashes, source gates, exclusive locks, empty evaluation target'))
print('PASS_PREFLIGHT',len(manifest['files']))
