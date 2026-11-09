"""Isolated report integration smoke test; never creates experiment results."""
import sys,json,tempfile,shutil,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import finalize_all as f

source=f.OUT/'P100/ETTh1_96'
with tempfile.TemporaryDirectory(prefix='ldc-report-smoke-') as tmp:
    root=Path(tmp);f.OUT=root/'results';f.OUT.mkdir();f.cm.ROOT=root
    (f.OUT/'EXPERIMENT_PROTOCOL.md').write_text('Smoke-test fixture only.\n')
    sample=next(iter(sorted((source/'final_raw/test').glob('*.npz'))))
    with np.load(sample) as z:arrays={k:z[k][:2] for k in z.files}
    pool=np.broadcast_to(np.arange(100),(2,arrays['S0__retmse'].shape[1],100)).copy()
    for cell in f.CELLS:
        p=root/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/pools'/cell
        p.mkdir(parents=True);np.save(p/'test.npy',pool)
    for dest,labels in f.targets():
        raw=dest/'final_raw/test';raw.mkdir(parents=True)
        for name in ('COMPLETE.json','FULL_RESULTS.json','TRAIN_HISTORY.json','LEARNED_PARAMETERS.json'):
            shutil.copyfile(source/name,dest/name)
        block=raw/'batch00000.npz';np.savez_compressed(block,**arrays)
        block.with_suffix('.json').write_text(json.dumps({'sha256':f.sha(block)}))
    f.main()
    done=json.loads((f.OUT/'final_all_comparisons/COMPLETE.json').read_text())
    assert done['native']==16 and done['alternative_pools']==8 and done['cross_support']==16
    with (f.OUT/'final_all_comparisons/FULL_RESULTS.csv').open() as stream:
        rows=list(csv.DictReader(stream));assert len(rows)==40*24,len(rows)
    with (f.OUT/'final_all_comparisons/SUPPORT_EFFECTS.csv').open() as stream:
        rows=list(csv.DictReader(stream));assert len(rows)==16*9,len(rows)
    print(json.dumps({'status':'PASS_ISOLATED_REPORT_SMOKE','fixture':'2 real query timestamps, replicated only in temporary directory','reports':len(done['files']),'finalizer_sha256':f.sha(Path(f.__file__))}))
