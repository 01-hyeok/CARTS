"""Isolated missing full-population native gates; preserve all previous tracks."""
import argparse, datetime, fcntl, os, sys, time, traceback
from pathlib import Path
from server_bridge import *
import server_bridge as bridge

BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'

def extended_frozen(cell,support):
    old=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/cell/('V5_'+support)/'REPRODUCTION_GATE.json'
    if old.exists():return bridge.frozen(cell,support)
    os.chdir(ROOT);data=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==support)
    prior=BASE/'00_Audit/extended_native_reproduction'/cell/('V5_'+support)/'REPRODUCTION_GATE.json'
    gate=json.loads(prior.read_text())
    assert gate['status']=='PASS' and gate['identity']['checkpoint_sha256']==row['checkpoint_sha256']
    assert gate['identity']['pool_hashes']==data.pool_hashes
    assert {r['split'] for r in gate['rows']}=={'val','test'}
    for r in gate['rows']:
        assert r['queries']==len(data.datasets[r['split']]) and r['channels']==data.c
        assert r['ordered_mismatch_count']==0 and r['units']==r['queries']*r['channels']
    f=Frozen(data,row)
    return data,f,dict(checkpoint=row,reproduction_gate_sha256=sha(prior),pool_hashes=data.pool_hashes,
        data=data.data_info,source_hashes={name:sha(HERE/name) for name in ['server_bridge.py','past_lag.py','softlag.py']},
        original_runtime='UNKNOWN',current_torch=torch.__version__,extended_gate_path=str(prior),
        extended_gate_source_sha256=sha(__file__))

def main(support,phase):
    cell='Weather_720'
    if phase=='reproduce':
        lock=(BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=start_gpu();begin=time.perf_counter();os.chdir(ROOT)
        data=Data(cell);row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==support)
        f=Frozen(data,row)
        from worker import reproduce,identity
        out=BASE/'00_Audit/extended_native_reproduction'/cell/('V5_'+support);out.mkdir(parents=True,exist_ok=True)
        reproduce(f,out,identity(row,data))
        dump(out/'EXTENSION_PROVENANCE.json',dict(before=before,after=snapshot(),seconds=time.perf_counter()-begin,
            gate_sha256=sha(out/'REPRODUCTION_GATE.json'),source_sha256=sha(__file__),
            scope='all validation and test queries/channels; native checkpoint support',original_tracks_modified=False))
    elif phase=='smoke':
        import gpu_gate
        gpu_gate.frozen=extended_frozen
        gpu_gate.main(cell,support,str(BASE/'00_Audit'/('GPU_GATE_'+cell+'_'+support+'_EXTENDED.json')))
    else:
        import d4_run
        d4_run.frozen=extended_frozen
        d4_run.main(cell,support)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('support',choices=['FULL','P100']);p.add_argument('phase',choices=['reproduce','smoke','d4']);a=p.parse_args()
    try:main(a.support,a.phase)
    except BaseException as e:
        dump(BASE/'00_Audit'/('W720_'+a.support+'_'+a.phase+'_ERROR_'+str(os.getpid())+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise
