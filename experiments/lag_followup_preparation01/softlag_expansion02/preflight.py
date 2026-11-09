"""CPU checkpoint/reuse audit and exact deployment identity."""
import sys,json,csv
from pathlib import Path
OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN));from contracts import *
sys.path.insert(0,str(OWN.parent))
from server_bridge import inventory

def main():
    (OUT/'00_Audit').mkdir(parents=True,exist_ok=True)
    rows=[]
    for cell,support in JOBS:
        row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==support)
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        options=[ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/cell/('V5_'+support)/'REPRODUCTION_GATE.json',DIAG/'00_Audit/extended_native_reproduction'/cell/('V5_'+support)/'REPRODUCTION_GATE.json',DIAG/'extensions_d5_d6/native_reproduction'/cell/('V5_'+support)/'REPRODUCTION_GATE.json']
        rp=next((p for p in options if p.exists()),None);assert rp is not None,('No native reproduction',cell,support)
        g=json.loads(rp.read_text());assert g['status']=='PASS' and g['identity']['checkpoint_sha256']==row['checkpoint_sha256']
        assert {r['split'] for r in g['rows']}=={'val','test'}
        expected_c=7 if cell.startswith('ETTh1') else 21
        assert all(r['channels']==expected_c and r['ordered_mismatch_count']==0 and r['units']==r['queries']*r['channels'] for r in g['rows'])
        entry=dict(cell=cell,support=support,channels=expected_c,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],reproduction_gate=str(rp),reproduction_sha256=sha(rp),splits=g['rows'],action='REUSE' if (cell,support) in REUSE else 'NEW_PILOT')
        if (cell,support) in REUSE:entry['reuse']=validate_reuse(OLD/(cell+'_'+support))
        rows.append(entry)
    sources={str(p.relative_to(HERE)):sha(p) for p in OWN.glob('*.py')}
    for n in ['server_bridge.py','softlag_run.py','softlag.py','past_lag.py','gpu_gate.py','softlag_finalize.py','protocol.json']:sources[n]=sha(HERE/n)
    dump(OUT/'00_Audit/CPU_PREFLIGHT.json',dict(status='PASS_CPU_CHECKPOINT_AND_REUSE',at=now(),conditions=rows,source_hashes=sources,reuse=4,new=12,total=16,gpu_started=False))
    dump(OUT/'EXECUTION_PLAN.json',dict(total=16,reuse=REUSE,new=NEW,seed=0,source_protocol=str(HERE/'protocol.json'),protocol_sha256=sha(HERE/'protocol.json'),samples={'train':64,'val':32,'test':32},all_channels=True,full_support_preserved=True,training='Only lag bias+alpha; frozen V5 encoder/queryheads; B1/B2 5epochs batch8 Adam .01; epoch0 eligible; val RetMSE selection',
        concurrency='One SoftLag expansion worker plus one authenticated D1-D6 worker on physical GPU1; >=24GiB startup headroom; 40% memory cap per process',
        old_softlag_exports='Finish current expansion condition then yield to old pair export after D56 terminal',solar_resume=False,no_retraining_completed_four=True))
    print(json.dumps(dict(status='PASS_CPU_CHECKPOINT_AND_REUSE',total=16,reuse=4,new=12,source_hashes=sources)),flush=True)
if __name__=='__main__':main()
