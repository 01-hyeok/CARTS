"""Read-only upstream gate. Exit 0 ready; 2 incomplete; 1 invalid/error. No Calibration launch."""
from recovery_runtime import *
from recovery02_validate import validate_all,Incomplete
def check():
    final=OUT/'final_all_comparisons/COMPLETE.json'
    if not final.exists():raise Incomplete('Final PASS absent: '+str(final))
    done=read(final)
    assert done['status']=='PASS_ALL_REQUESTED_COMPARISONS_AND_REPORTS'
    assert (done['native'],done['alternative_pools'],done['cross_support'])==(16,8,16)
    assert done['input_manifest_sha256']==sha(OUT/'REUSE_MANIFEST.json') and done['source_seal_sha256']==sha(OUT/'SOURCE_SEAL.json')
    for name,h in done['files'].items():assert sha(final.parent/name)==h,('Report hash',name)
    proof=validate_all(True)
    assert done['source_completion_hashes']=={r['path']:r['completion_sha256'] for r in proof['jobs']}
    assert sha(OUT/'FINAL_VALIDATION.json')==done['validation_sha256']
    return dict(status='READY_VERIFIED_16_8_16_AND_REPORTS',final_complete=str(final),final_sha256=sha(final),calibration_started=False,at=now())
if __name__=='__main__':
    try:print(json.dumps(check()));sys.exit(0)
    except Incomplete as e:print(json.dumps(dict(status='NOT_READY',reason=str(e),at=now())));sys.exit(2)
    except Exception as e:print(json.dumps(dict(status='INVALID',error=repr(e),at=now())));sys.exit(1)
