"""Paths, source seal, process-wide write boundary, and read-only input resolver."""
import sys,os,json,hashlib,fcntl,subprocess
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'core'))
from experiment_common import *
def read(p):return json.loads(Path(p).read_text())
def seal_check():
    s=read(OUT/'SOURCE_SEAL.json')['files']
    assert s=={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')},'Recovery source seal mismatch'
def protect():
    sys.dont_write_bytecode=True
    import tempfile
    temporary=OUT/'tmp';temporary.mkdir(exist_ok=True)
    tempfile.tempdir=str(temporary);os.environ['TMPDIR']=str(temporary)
    def audit(event,args):
        paths=[]
        if event=='open':
            p,mode,flags=args
            if isinstance(p,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):paths=[p]
        elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.symlink','os.link'):paths=[args[0]]
        elif event in ('os.rename','os.replace'):paths=list(args[:2])
        for p in paths:
            q=Path(os.fsdecode(p)).resolve()
            if str(q)=='/dev/null':continue
            if not (q==OUT.resolve() or OUT.resolve() in q.parents):raise PermissionError('Recovery write boundary: '+str(q))
    sys.addaudithook(audit)
def preserved(all_files=True):
    m=read(OUT/'REUSE_MANIFEST.json')
    for p,v in m['files'].items():
        if all_files or '/experiments/' in p:assert Path(p).is_file() and sha(p)==v['sha256'],('Original changed',p)
    return len(m['files'])
def gpu_ready():
    info=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip()
    i,u,m,v=[x.strip() for x in info.split(',')];assert i=='1' and u==GPU
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
    assert not any(x.startswith(GPU) for x in apps.splitlines()),('GPU occupied',apps)
    assert int(m)<128 and int(v)==0,info
    return info
def hold(path):
    # Existing original locks are opened read-only. No original inode is created or changed.
    f=Path(path).open('rb') if ORIGINAL==Path(path).parent or ORIGINAL in Path(path).parents and OUT not in Path(path).parents else Path(path).open('a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
