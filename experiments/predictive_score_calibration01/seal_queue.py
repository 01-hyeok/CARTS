"""Seal reviewed isolated sources after CPU tests. Does not launch GPU work."""
import ast,hashlib,json,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01'
for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
for filename in ('CPU_SYNTHETIC_TESTS.json','P2_CPU_TESTS.json'):
    assert json.loads((OUT/filename).read_text())['passed']
assert json.loads((OUT/'P1_CPU_INTEGRATION.json').read_text())['status']=='PASS_CPU_ACTUAL_TRAIN_BLOCK_INTEGRATION'
assert json.loads((OUT/'REGIME_CPU_COMPLETE.json').read_text())['status']=='PASS_ALL_AVAILABLE_COMPLETED_CONDITIONS'
seal=dict(at=time.time(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},scope='P100 canonical preparation -> complete P100 P0 gate -> P1 all8 -> P2 all8; Full/P3/P4 not launched by this queue',gpu='physical GPU1 only; after original recovery ALL_QUEUES_COMPLETE and GPU idle twice',test_future_selection=False)
with (OUT/'EXECUTION_SOURCE_SEAL.json').open('x') as f:json.dump(seal,f,indent=2)
print(json.dumps(dict(status='SEALED',source_files=len(seal['files']),scope=seal['scope']),indent=2))
