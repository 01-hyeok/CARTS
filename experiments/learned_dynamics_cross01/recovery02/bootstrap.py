"""Install isolated copies on the approved server; never alter originals."""
from pathlib import Path
import json,hashlib,difflib
ROOT=Path('/data/pjh_workspace/CARTS')
SRC=ROOT/'experiments/learned_dynamics_cross01'
OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
NEW=SRC/'recovery02';OUT=OLD/'recovery02'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(payload):
    assert not NEW.exists() and not OUT.exists(),'Explicit inspection required for existing recovery02'
    gate=json.loads((OLD/'CPU_IMPLEMENTATION_GATE.json').read_text());assert gate['status']=='PASS'
    assert gate['source_hashes']=={p.name:digest(p) for p in SRC.glob('*.py')}
    NEW.mkdir();OUT.mkdir();(NEW/'core').mkdir();(NEW/'core/supplemental').mkdir();(NEW/'core/monitoring').mkdir()
    changes={};originals={}
    def copy(rel,edits=()):
        src=SRC/rel;s=src.read_text();originals[str(src)]=digest(src);before=s
        for a,b in edits:
            assert a in s,(rel,a);s=s.replace(a,b)
        dest=NEW/'core'/rel;dest.write_text(s)
        changes[rel]=''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(dest)))
    copy('experiment_common.py',[("HERE=ROOT/'experiments/learned_dynamics_cross01'","HERE=Path(__file__).resolve().parent"),("OUT=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'","ORIGINAL=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'\nOUT=ORIGINAL/'recovery02'"),("p=Path(p);p.parent.mkdir", "p=Path(p);assert OUT.resolve() in p.resolve().parents,('write outside recovery',str(p));p.parent.mkdir")])
    with (NEW/'core/experiment_common.py').open('a') as f:f.write('''
def native_path(support,cell):
    assert support in ('P100','FULL') and cell in CELLS
    m=json.loads((OUT/'REUSE_MANIFEST.json').read_text())
    return Path(m['native'].get(support+'/'+cell,str(OUT/support/cell)))
def checkpoint_path(dest,name,epoch):
    p=Path(dest)/'checkpoints'/name/('epoch%02d.pt'%epoch)
    if p.exists():return p
    if Path(dest)==OUT/'FULL/Weather_96':
        q=ORIGINAL/'FULL/Weather_96/checkpoints'/name/('epoch%02d.pt'%epoch)
        assert epoch<=8 and q.exists();return q
    raise FileNotFoundError(str(p))
''')
    copy('learned_scores.py');copy('train_models.py')
    copy('run_experiment.py',[("OUT/'features'/cell","ORIGINAL/'features'/cell"),("    path.parent.mkdir(parents=True,exist_ok=True)","    assert OUT.resolve() in path.resolve().parents\n    path.parent.mkdir(parents=True,exist_ok=True)"),("load(self.dest/'checkpoints'/n/('epoch%02d.pt'%state[n]['best_epoch']))","load(checkpoint_path(self.dest,n,state[n]['best_epoch']))")])
    copy('final_evaluation.py')
    copy('summarize.py',[("dest=OUT/support/cell","dest=native_path(support,cell)"),("paired_path=dest/'PAIRED_TIMESTAMP_ANALYSIS.json'","paired_path=OUT/'paired_cache'/support/cell/'PAIRED_TIMESTAMP_ANALYSIS.json'")])
    copy('supplemental/run_supplement.py',[("cm.OUT/'features'/cell","cm.ORIGINAL/'features'/cell"),("source=cm.OUT/self.training/self.cell","source=cm.native_path(self.training,self.cell)"),("p=source/'checkpoints'/name/('epoch%02d.pt'%state[name]['best_epoch'])","p=cm.checkpoint_path(source,name,state[name]['best_epoch'])")])
    copy('monitoring/finalize_all.py',[("yield OUT/s/c,dict", "yield cm.native_path(s,c),dict"),("str(path.relative_to(OUT))","str(path)"),("native=OUT/source/cell/'final_raw/test'","native=cm.native_path(source,cell)/'final_raw/test'"),("dump(dest/'COMPLETE.json',dict(","dump(dest/'REPORT_BUILD.json',dict(")])
    copy('recovery01/compat.py') if False else None
    compat=(SRC/'recovery01/compat.py').read_text().replace("sys.path.insert(0,str(HERE.parent))","sys.path.insert(0,str(HERE/'core'))")
    (NEW/'compat.py').write_text(compat);originals[str(SRC/'recovery01/compat.py')]=digest(SRC/'recovery01/compat.py')
    for name,content in payload.items():
        assert '/' not in name and name.endswith('.py');(NEW/name).write_text(content)
    files={}
    # Snapshot original files now, excluding the new output subtree. No original is copied or modified.
    for folder in (SRC,OLD,ROOT/'experiments/predictive_score_calibration01'):
        for p in folder.rglob('*'):
            if NEW in p.parents or OUT in p.parents or '__pycache__' in p.parts or not p.is_file():continue
            files[str(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
    native={}
    for s in ('P100','FULL'):
        for p in (OLD/s).glob('*/COMPLETE.json'):
            assert json.loads(p.read_text())['status']=='PASS_ALL_QUERY_TRAIN_VAL_TEST'
            native[s+'/'+p.parent.name]=str(p.parent)
    assert len(native)==10
    (OUT/'REUSE_MANIFEST.json').write_text(json.dumps(dict(native=native,files=files,original_source_hashes=originals,hash_scope='post-interruption snapshot; no pre-interruption equality claim'),indent=2))
    for rel in list(changes):
        changes[rel]=''.join(difflib.unified_diff((SRC/rel).read_text().splitlines(True),(NEW/'core'/rel).read_text().splitlines(True),fromfile=str(SRC/rel),tofile=str(NEW/'core'/rel)))
    (OUT/'SOURCE_DIFF.json').write_text(json.dumps(changes,indent=2))
    (OUT/'EXPERIMENT_PROTOCOL.md').write_text((OLD/'EXPERIMENT_PROTOCOL.md').read_text()+'\nRecovery02: Weather96 FULL restarts epoch9 from completed epoch8 model+optimizer. No saved RNG or partial-block state; bitwise uninterrupted equivalence unproven. Partial epoch9 is excluded from valid history. Original paths remain read-only.\n')
    (OUT/'supplemental').mkdir()
    (OUT/'supplemental/POOL_REUSE_GATE.json').write_bytes((OLD/'supplemental/POOL_REUSE_GATE.json').read_bytes())
    core={p.name:digest(p) for p in (NEW/'core').glob('*.py')}
    (OUT/'CPU_IMPLEMENTATION_GATE.json').write_text(json.dumps(dict(status='PENDING',source_hashes=core)))
    supp={p.name:digest(p) for p in (NEW/'core/supplemental').glob('*.py')}
    (OUT/'supplemental/CPU_GATE.json').write_text(json.dumps(dict(status='PENDING',sources=supp)))
    seal={str(p.relative_to(NEW)):digest(p) for p in NEW.rglob('*.py')}
    (OUT/'SOURCE_SEAL.json').write_text(json.dumps(dict(files=seal),indent=2))
    print(json.dumps(dict(installed=str(NEW),output=str(OUT),preserved_files=len(files),native_reused=len(native),seal=seal)))
if __name__=='__main__':
    import sys
    main(json.loads(sys.stdin.read()))
