"""Seed0-only serial controller; new Solar runs use verified vectorized03."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'orchestration_seed0')]
import controller_seed0 as original
import runtime_validation

def track_workers():
    found=[]
    allowed={str(original.HERE/'runtime.py'),str(original.HERE/'vectorized03/runtime_vector.py'),str(HERE/'runtime_validation.py')}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:i=original.proc_identity(int(p.name))
        except PermissionError:continue
        if i and allowed.intersection(i['argv']):found.append(i)
    return found

execute_original=original.execute
def execute(command,label,status):
    if status=='STAGE1_RUNNING' and label.startswith('Solar_'):
        assert label.endswith('_seed0')
        command=list(command);command[2]=str(HERE/'runtime_validation.py')
    return execute_original(command,label,status)

if __name__=='__main__':
    gate=json.loads((HERE/'VALIDATION_RUNTIME_GATES.json').read_text())
    assert gate['status']=='PASS' and gate['source_hashes']==runtime_validation.source_hashes()
    original.track_workers=track_workers;original.execute=execute
    original.main()
