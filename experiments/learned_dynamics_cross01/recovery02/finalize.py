from recovery_runtime import *
from recovery02_validate import validate_all
def run():
    proof=validate_all(True);dump(OUT/'FINAL_VALIDATION.json',proof)
    sys.path.insert(0,str(CODE/'core/monitoring'))
    import finalize_all
    finalize_all.main()
    folder=OUT/'final_all_comparisons'
    note='Recovery02: Weather96 FULL resumed at epoch8 boundary; partial original epoch9 was discarded. RNG state was not saved; uninterrupted bitwise equivalence is not asserted. Its CUDA peaks cover recovery only; elapsed combines completed original epochs0-8 and recovery, excludes partial epoch9 and downtime. Original resource samples retain evidence of discarded work. ETTh1_96 FULL allocator peaks cover prior recovery evaluation only. No Calibration or Solar was started by this queue.'
    for name in ('EFFICIENCY_REPORT.md','EXPERIMENT_PROTOCOL.md'):
        p=folder/name;p.write_text(p.read_text()+'\n\n'+note+'\n')
    import csv
    p=folder/'EFFICIENCY.csv'
    with p.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['peak_scope']='worker lifetime'
        if r['comparison']=='native' and r['training_support']=='FULL':
            if r['cell']=='Weather_96':r['peak_scope']='recovery02 lifetime only'
            if r['cell']=='ETTh1_96':r['peak_scope']='recovery01 evaluation only'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    final=dict(status='PASS_ALL_REQUESTED_COMPARISONS_AND_REPORTS',native=16,alternative_pools=8,cross_support=16,source_completion_hashes={r['path']:r['completion_sha256'] for r in proof['jobs']},input_manifest_sha256=sha(OUT/'REUSE_MANIFEST.json'),source_seal_sha256=sha(OUT/'SOURCE_SEAL.json'),validation_sha256=sha(OUT/'FINAL_VALIDATION.json'),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='COMPLETE.json'},at=now(),solar_started=False,calibration_started=False,recovery_disclosure=note)
    assert len(final['files'])>=10
    dump(folder/'COMPLETE.json',final)
