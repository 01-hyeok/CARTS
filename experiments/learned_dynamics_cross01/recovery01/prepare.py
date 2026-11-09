import sys,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from experiment_common import *
root=OUT/'recovery01';root.mkdir(exist_ok=True)
assert not (root/'PRESERVATION_MANIFEST.json').exists()
archive=root/'original_status';archive.mkdir()
for path in (OUT/'CONTROLLER_STATUS.json',OUT/'MONITOR_STATUS.json',OUT/'supplemental/CONTROLLER_STATUS.json'):
    label='supplemental_' if path.parent.name=='supplemental' else ''
    shutil.copy2(path,archive/(label+path.name))
paths=list((OUT/'FULL/ETTh1_96').rglob('*'))+[OUT/'logs/FULL_ETTh1_96_20261104T095524.log']
paths+=list(archive.iterdir())
paths += [p for p in HERE.parent.rglob('*.py') if 'recovery01' not in p.relative_to(HERE.parent).parts]
paths += [OUT/'P100'/c/n for c in CELLS for n in ('COMPLETE.json','FULL_RESULTS.json','SELECTION.json')]
files={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
dump(root/'PRESERVATION_MANIFEST.json',dict(status='ORIGINALS_HASHED_BEFORE_RECOVERY',files=files,at=now()))
print('PRESERVED',len(files),'files')
