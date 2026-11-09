import argparse,traceback,sys
from pathlib import Path
import runtime_all_channels as repair
import diagnostic_run as run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('support',choices=['FULL']);p.add_argument('cell');p.add_argument('--reproduction-only',action='store_true');a=p.parse_args()
    try:
        if a.reproduction_only:
            assert run.GPU not in run.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            run.rt.gpu_guard();data=run.Data(a.cell)
            models=[run.Frozen(data,r,a.support) for r in run.inventory(a.cell,a.support) if r['status']=='VALID']
            out=run.OUT/'00_Audit/TAIL08'/a.cell;out.mkdir(parents=True,exist_ok=True)
            run.reproduce(data,models,a.support,out)
            run.dump(out/'PRECISION_GATE.json',dict(status='PASS',models=[m.immutable_check() for m in models],source_hashes=run.snapshot_hashes()))
            print('FULL PRECISION REPRODUCTION PASS',a.cell,flush=True)
        else:run.run_cell(a.support,a.cell)
    except BaseException as e:
        issue=dict(status='FAILED',at=run.now(),error=repr(e),traceback=traceback.format_exc(),support=a.support,cell=a.cell)
        out=run.OUT/'00_Audit/TAIL08'/a.cell if a.reproduction_only else run.OUT/a.support/a.cell
        run.dump(out/'FAILED.json',issue);run.dump(run.OUT/'00_Audit/IMMEDIATE_ISSUE.json',issue)
        raise
