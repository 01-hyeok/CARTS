import ast,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/data/pjh_workspace/CARTS/results/TRACK-PREDICTIVE-SCORE-CALIBRATION01/full_extension01')
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for name in ('TEST_TORCH1.json','TEST_TORCH2.json','RANKING_TESTS.json'):
    assert json.loads((OUT/name).read_text())['passed']
for name in ('FEATURE_INTEGRATION_TORCH1.json','FEATURE_INTEGRATION_TORCH2_V2.json'):
    assert json.loads((OUT/name).read_text())['status']=='PASS_REAL_TRAIN_FEATURE_INTEGRATION'
info=dict(at=time.time(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},parent_dependencies={'calibrators.py':hashlib.sha256((HERE.parent/'calibrators.py').read_bytes()).hexdigest()},scope='Full P0 all8 -> actual Full integration -> Full P1/P2 all8 -> frozen cross-support; after P100 queue; no Solar; no P3/P4 automatic learning',actual_gpu_gate_passed=False)
with (OUT/'SOURCE_SEAL.json').open('x') as f:json.dump(info,f,indent=2)
print(json.dumps(dict(status='SEALED_CPU_VERIFIED_GPU_GATED',files=len(info['files'])),indent=2))
