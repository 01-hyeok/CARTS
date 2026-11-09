"""Read-only upstream API isolation. No GPU, file writes, or validator replacement."""
import json,sys
from pathlib import Path
UP=Path('/data/pjh_workspace/CARTS/experiments/learned_dynamics_cross01/recovery02')
sys.dont_write_bytecode=True
sys.path.insert(0,str(UP))
from readiness import check,Incomplete
from resolver import resolve
from recovery02_validate import targets
try:
    ready=check()
    records=[resolve(l['mode'],l['cell'],l['training'],l['pool']) for _,l in targets()]
    print(json.dumps(dict(ready=ready,records=records)))
except Incomplete as e:print(json.dumps(dict(status='NOT_READY',reason=str(e))));sys.exit(2)
except Exception as e:print(json.dumps(dict(status='INVALID',error=repr(e))));sys.exit(1)
