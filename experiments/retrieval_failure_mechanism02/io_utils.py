import datetime,hashlib,json,os
from pathlib import Path
import numpy as np

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def save(p,values):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    os.replace(tmp,p)
RUNTIME_FILES=('adapter.py','worker.py','gpu_metrics.py','geometry.py','feature_analysis.py','ranking.py','resources.py','protocol.py','io_utils.py')
def source_hashes():
    here=Path(__file__).parent
    return {name:sha(here/name) for name in RUNTIME_FILES}
def asnumpy(t):return t.detach().cpu().numpy()
def id_bytes(ids):
    # Candidate IDs are semantic integers; hash one explicit representation.
    return np.ascontiguousarray(ids,dtype='<i8').tobytes()
