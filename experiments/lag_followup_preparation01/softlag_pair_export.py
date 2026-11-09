"""Missing pair-level overlap export from frozen sampled caches; no training."""
import argparse,csv,datetime,fcntl,json,os
import numpy as np
from server_bridge import *
from softlag_finalize import load_rows
from softlag_run import batch,probability,write_csv,metrics
from softlag import torch_parameters
from past_lag import indices
BASE=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01'

def main(support):
    # Expansion owns this same SoftLag-track lock. Wait for its safe boundary;
    # never create a third GPU worker or fail merely because that worker is live.
    lock=(BASE/'00_Audit/finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if support=='REPORT':
        rows=[];gates=[]
        for c in ['ETTh1_96','ETTh1_336']:
            for s in ['FULL','P100']:
                out=BASE/(c+'_'+s);g=json.loads((out/'PAIR_EXPORT_GATE.json').read_text());assert g['status']=='PASS'
                assert sha(out/'HEAD_PAIR_OVERLAP.csv')==g['csv_sha256'];gates.append(g)
                with (out/'HEAD_PAIR_OVERLAP.csv').open() as stream:rows+=list(csv.DictReader(stream))
        write_csv(BASE/'HEAD_PAIR_OVERLAP.csv',rows)
        dump(BASE/'DELIVERABLE_COMPLETENESS_GATE.json',dict(status='PASS_REQUIRED_ANALYSES_COMPLETE_WITH_TIMING_LIMITATIONS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             original_final_gate_sha256=sha(BASE/'FINAL_GATE.json'),protocol_audit_sha256=sha(BASE/'PROTOCOL_AUDIT_ADDENDUM.json'),
             head_pair_file_sha256=sha(BASE/'HEAD_PAIR_OVERLAP.csv'),pair_gates=gates,
             remaining_analysis_items=[],unmeasured=['separate GPU training kernel time','separate validation wall time'],no_retraining=True))
        return
    start_gpu()
    for cell in ['ETTh1_96','ETTh1_336']:
        out=BASE/(cell+'_'+support)
        if (out/'PAIR_EXPORT_GATE.json').exists():
            g=json.loads((out/'PAIR_EXPORT_GATE.json').read_text());assert sha(out/'HEAD_PAIR_OVERLAP.csv')==g['csv_sha256'];continue
        data,f,identity=frozen(cell,support);test=load_rows(out,'test');_,lags=indices(data.h);rows=[]
        with (out/'HEAD_USEFULNESS.csv').open() as stream:prior=list(csv.DictReader(stream))
        for arm in ['B0','B1','B2']:
            alpha,bias=torch_parameters(arm!='B1',len(lags),device=data.device,dtype=torch.float32)
            if arm!='B0':
                selected=json.loads((out/arm/'SELECTION.json').read_text());assert sha(selected['checkpoint'])==selected['sha256']
                z=torch.load(selected['checkpoint'],map_location=data.device);alpha.data.copy_(z['alpha']);bias.data.copy_(z['bias'])
            picks=[];met=[]
            with torch.no_grad():
                for a in range(0,len(test),8):
                    b=batch(test,range(a,min(a+8,len(test))));u,p=probability(b,alpha,bias,support)
                    hids=topk(u.masked_fill(~b['mask'][:,:,None],-float('inf')),10);picks.append(hids.cpu().numpy())
                    hm=[]
                    for h in range(5):hm.append(metrics(data,b,u[:,:,h])[0])
                    met.append(np.stack(hm,2))
            ids=np.concatenate(picks);m=np.concatenate(met)
            for h in range(5):
                expected=next(r for r in prior if r['arm']==arm and int(r['head'])==h+1)
                # Match softlag_finalize.head_detail aggregation exactly. Reducing
                # the strided scalar column separately can select another NumPy
                # summation kernel despite identical per-query metric values.
                assert float(m[:,:,h].mean((0,1))[0])==float(expected['test_retmse'])
            for h in range(5):
                for k in range(h+1,5):
                    overlap=(ids[:,:,h,:,None]==ids[:,:,k,None,:]).any(-1).mean(-1)
                    rows.append(dict(cell=cell,support=support,arm=arm,head_a=h+1,head_b=k+1,top10_overlap=float(overlap.mean()),queries=32,channels=7))
            target=out/(arm+'_test_head_ids.npz')
            if target.exists():
                with np.load(target) as z:assert np.array_equal(z['local_ids'],ids)
            else:np.savez_compressed(target,local_ids=ids,query_indices=np.concatenate([r['query_index'] for r in test]))
        write_csv(out/'HEAD_PAIR_OVERLAP.csv',rows)
        dump(out/'PAIR_EXPORT_GATE.json',dict(status='PASS',cell=cell,support=support,csv_sha256=sha(out/'HEAD_PAIR_OVERLAP.csv'),
             all_15_head_arm_retmse_exact=True,no_training=True,source_checkpoint_sha256=sha(f.row['checkpoint']),
             source_runtime=identity['current_torch'],all_channels=True))
        print('SOFTLAG PAIR EXPORT PASS',cell,support,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('support',choices=['FULL','P100','REPORT']);a=p.parse_args();main(a.support)
