"""Apply evaluation compatibility only, then invoke immutable worker sources."""
import sys,runpy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import compat;compat.install()
from experiment_common import OUT,sha,dump,now
gate=json.loads((OUT/'recovery01/COMPATIBILITY_GATE.json').read_text())
assert gate['status']=='PASS' and gate['compat_sha256']==sha(HERE/'compat.py')
mode=sys.argv.pop(1)
target=HERE.parent/('run_experiment.py' if mode=='native' else 'supplemental/run_supplement.py')
sys.argv[0]=str(target)
runpy.run_path(str(target),run_name='__main__')
if mode=='native':dest=OUT/sys.argv[2]/sys.argv[1]
else:
    kind,cell,training=sys.argv[1:4]
    label=sys.argv[5] if kind=='pool' else 'V5_'+training+'_to_'+('FULL' if training=='P100' else 'P100')
    dest=OUT/'supplemental'/kind/label/cell
dump(dest/'EVALUATION_COMPATIBILITY.json',dict(compat_sha256=sha(HERE/'compat.py'),gate_sha256=sha(OUT/'recovery01/COMPATIBILITY_GATE.json'),at=now(),training_unchanged=True))
