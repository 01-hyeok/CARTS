"""Solar data-only diagnostics, using the validated full all-channel operations."""
import sys,traceback
import gate
sys.path.insert(0,str(gate.ROOT/'experiments/retrieval_all_channel_tail08'))
import runtime_all_channels as runtime
run=runtime.run
base_snapshot=run.snapshot_hashes
def snapshot():
    a=base_snapshot();a.update({str(p.relative_to(gate.ROOT)):gate.sha(p) for p in gate.HERE.glob('*.py')});return a
run.snapshot_hashes=snapshot;runtime.adapter.snapshot_hashes=snapshot

if __name__=='__main__':
    mode,cell=sys.argv[1:];assert mode in ('main','supplement') and cell in gate.SOLAR
    gate.source_check();gate.preflight(cell)
    rows=run.inventory(cell,'FULL')
    assert len(rows)==2 and {r['architecture'] for r in rows}=={'V0','V5'} and all(r['status']=='MISSING' for r in rows)
    try:
        if mode=='main':run.run_cell('FULL',cell)
        else:
            sys.path.insert(0,str(gate.ROOT/'experiments/retrieval_diagnostic_supplement01'))
            import supplement_inference
            supplement_inference.run('FULL',cell)
    except BaseException as e:
        issue=dict(status='FAILED',at=run.now(),cell=cell,mode=mode,error=repr(e),traceback=traceback.format_exc())
        run.dump(gate.AUDIT/cell/(mode+'_FAILED.json'),issue);run.dump(gate.SOURCE/'00_Audit/IMMEDIATE_ISSUE.json',issue);raise
