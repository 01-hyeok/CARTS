"""Resume Weather720 diagnostics using the unchanged, fully validated adapter."""
import sys,time,traceback
from pathlib import Path
import gate
sys.path.insert(0,str(gate.VALIDATED))
import runtime_all_channels as repair
import diagnostic_run as run


def verified_reuse(data,models,support,out):
    begin=time.perf_counter();release,precise,result,exported=gate.verify()
    assert data.cell=='Weather_720' and support=='FULL'
    assert data.data_info==exported['provenance']['data']
    assert {m.name:m.row['checkpoint_sha256'] for m in models}=={m['model']:m['sha256'] for m in precise['models']}
    # Reuse the original measured rows only after all source/model/data hashes
    # match. No new metric is invented or rounded, and no extra GPU rerun occurs.
    target=dict(result,reused_from=str(gate.AUDIT/'Weather_720/CANONICAL_REPRODUCTION.json'),
                reused_gate_sha256=release['v5_gate_sha256'],original_validation_seconds=result['seconds'],
                seconds=time.perf_counter()-begin,reuse_status='IDENTICAL_CODE_CHECKPOINT_AND_DATA_VERIFIED')
    assert not (out/'CANONICAL_REPRODUCTION.json').exists()
    run.dump(out/'CANONICAL_REPRODUCTION.json',target)
    for split in ['val','test']:
        origin=gate.AUDIT/'Weather_720'/('REAL_BATCH_GATE_'+split+'.json')
        g=gate.read(origin);assert g['status']=='PASS'
        run.dump(out/origin.name,dict(g,reused_from=str(origin),source_sha256=gate.sha(origin)))
    print('FULL FOUR-ROW CANONICAL GATE VERIFIED REUSE; START MAIN DIAGNOSTICS',flush=True)


previous_snapshot=run.snapshot_hashes
def snapshot():
    result=previous_snapshot()
    result.update({str(p.relative_to(gate.ROOT)):gate.sha(p) for p in Path(__file__).parent.glob('*.py')})
    return result

if __name__=='__main__':
    assert sys.argv[1:]==['FULL','Weather_720']
    run.reproduce=verified_reuse;run.snapshot_hashes=snapshot;repair.adapter.snapshot_hashes=snapshot
    try:run.run_cell('FULL','Weather_720')
    except BaseException as error:
        run.dump(gate.SOURCE/'00_Audit/RESUME09/FAILED.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc()))
        raise
