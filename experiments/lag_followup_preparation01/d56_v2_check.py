import json,shutil
from pathlib import Path
import d56_controller_v2 as c
root=c.OUT;folder=root/'controller_repair_v2';folder.mkdir(exist_ok=True)
old=root/'CONTROLLER_STATUS.json'
if old.exists() and not (folder/'ORIGINAL_STATUS.json').exists():shutil.copyfile(old,folder/'ORIGINAL_STATUS.json')
active=json.loads((root/'ACTIVATION.json').read_text());proc=Path('/proc')/str(active['controller_pid'])/'cmdline'
assert not proc.exists() or b'd56_controller.py' not in proc.read_bytes(),'old controller still live; do not duplicate'
gate=root/'feature_smoke/ETTh1_96/V5_FULL_to_FULL/FEATURE_GATE.json'
assert c.verified_gate(gate,'PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL')
c.OUT=folder
c.status('RUNNING',phase='SMOKE',worker_pid=123,job=['fixture'])
assert json.loads((folder/'CONTROLLER_STATUS.json').read_text())['phase']=='SMOKE'
print('CONTROLLER_STATUS_KEYWORD_REGRESSION_PASS; EXISTING_SMOKE_HASHES_PASS',flush=True)
