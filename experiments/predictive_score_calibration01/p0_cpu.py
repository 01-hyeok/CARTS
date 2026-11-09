"""Read-only P0 evidence replay. Never trains or launches CUDA/Solar."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path('/data/pjh_workspace/CARTS')
OLD = ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
OUT = ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(name, obj):
    with (OUT/name).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'P0_CPU_COMPLETE.json').exists(), 'Preserve completed run'
    started = time.time()
    rows, baselines, inputs = [], [], []
    query_path = OUT/'P0_CANDIDATE_SWAP_QUERY.csv'
    with query_path.open('x', newline='') as stream:
        fields = ['support','cell','split','method','query_start','channel','removed_count','added_count','removed_mse_sum','added_mse_sum','delta_retmse','oracle_added','oracle_removed']
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for support in ('P100','FULL'):
            for folder in sorted((OLD/support).glob('*')):
                if not (folder/'COMPLETE.json').exists(): continue
                identity = json.loads((folder/'IDENTITY.json').read_text())
                data = identity['data']
                assert data['seq_len'] == data['pred_len'] == int(folder.name.rsplit('_',1)[1])
                reference = json.loads((folder/'FULL_RESULTS.json').read_text())
                for split in ('val','test'):
                    gate = json.loads((folder/('FINAL_'+split.upper()+'_GATE.json')).read_text())
                    assert gate['status'] == 'PASS_ALL_QUERY'
                    accum, n, base_sum, starts_all = {}, 0, 0., []
                    for path in sorted((folder/'final_raw'/split).glob('batch*.npz')):
                        digest = sha(path)
                        meta = json.loads(path.with_suffix('.json').read_text())
                        assert digest == meta['sha256']
                        inputs.append(dict(path=str(path), sha256=digest))
                        with np.load(path) as z:
                            base = z['S0__selected_ids']; bm = z['S0__selected_mse'].astype(np.float64)
                            assert base.shape[-1] == 10 and base.shape[1] == data['channels']
                            assert np.all(np.diff(np.sort(base,axis=-1),axis=-1) != 0)
                            assert np.allclose(bm.mean(-1),z['S0__retmse'],rtol=0,atol=1e-12)
                            starts = z['query_start']; starts_all.extend(starts.tolist())
                            n += bm.shape[0]*bm.shape[1]; base_sum += bm.mean(-1).sum()
                            oracle = z['Oracle10__selected_ids']
                            for method in ('D2_MLP','C2_MLP'):
                                ids = z[method+'__selected_ids']; mse = z[method+'__selected_mse'].astype(np.float64)
                                assert np.all(np.diff(np.sort(ids,axis=-1),axis=-1) != 0)
                                removed = ~(base[...,None] == ids[...,None,:]).any(-1)
                                added = ~(ids[...,None] == base[...,None,:]).any(-1)
                                assert np.array_equal(removed.sum(-1),added.sum(-1))
                                # Intersection candidate error must not depend on scorer.
                                match = base[...,None] == ids[...,None,:]
                                assert np.allclose(np.broadcast_to(bm[...,None],match.shape)[match],np.broadcast_to(mse[...,None,:],match.shape)[match],rtol=0,atol=0)
                                rs = (bm*removed).sum(-1); ads = (mse*added).sum(-1)
                                delta = (ads-rs)/10
                                assert np.allclose(delta,mse.mean(-1)-bm.mean(-1),rtol=0,atol=1e-12)
                                assert np.allclose(delta,z[method+'__delta_retmse_vs_s0'],rtol=0,atol=1e-12)
                                ao = (added & (ids[...,None] == oracle[...,None,:]).any(-1)).sum(-1)
                                ro = (removed & (base[...,None] == oracle[...,None,:]).any(-1)).sum(-1)
                                state = accum.setdefault(method, dict(delta=[],added_sum=0.,removed_sum=0.,swaps=0,oracle_added=0,oracle_removed=0))
                                state['delta'].extend(delta.mean(-1).tolist())
                                for key,val in [('added_sum',ads.sum()),('removed_sum',rs.sum()),('swaps',added.sum()),('oracle_added',ao.sum()),('oracle_removed',ro.sum())]: state[key] += float(val)
                                for q in range(len(starts)):
                                    for c in range(base.shape[1]):
                                        writer.writerow(dict(support=support,cell=folder.name,split=split,method=method,query_start=int(starts[q]),channel=c,removed_count=int(removed[q,c].sum()),added_count=int(added[q,c].sum()),removed_mse_sum=rs[q,c],added_mse_sum=ads[q,c],delta_retmse=delta[q,c],oracle_added=int(ao[q,c]),oracle_removed=int(ro[q,c])))
                    assert len(starts_all) == data['counts'][split] and len(set(starts_all)) == len(starts_all)
                    assert starts_all == list(range(data['starts'][split][0],data['starts'][split][1]+1))
                    official = next(r for r in reference if r['split']==split and r['method']=='S0')['retmse']
                    assert np.isclose(base_sum/n,official,rtol=0,atol=1e-12)
                    baselines.append(dict(support=support,cell=folder.name,split=split,seq_len=data['seq_len'],pred_len=data['pred_len'],queries=len(starts_all),channels=data['channels'],retmse=base_sum/n,reference=official,status='PASS_SAVED_EVIDENCE_REPLAY',fresh_encoder_reproduction=False,checkpoint=identity['checkpoint'],pool_hashes=identity.get('pool_hashes')))
                    for method,state in accum.items():
                        delta=np.array(state.pop('delta'));swaps=state['swaps']
                        expected=next(r for r in reference if r['split']==split and r['method']==method)['retmse']-official
                        assert np.isclose(delta.mean(),expected,rtol=0,atol=1e-12)
                        row=dict(support=support,cell=folder.name,split=split,method=method,query_channels=n,delta_retmse=float(delta.mean()),expected_delta=expected,added_mean=state['added_sum']/swaps if swaps else None,removed_mean=state['removed_sum']/swaps if swaps else None,swaps_per_query_channel=swaps/n,win_timestamps=int((delta<0).sum()),tie_timestamps=int((delta==0).sum()),loss_timestamps=int((delta>0).sum()),**state)
                        for pct in (1,5,10):
                            tail=np.sort(delta)[-max(1,int(np.ceil(len(delta)*pct/100))):]
                            row['worst_%d_mean_delta'%pct]=float(tail.mean())
                            row['worst_%d_total_mean_contribution'%pct]=float(tail.sum()/len(delta))
                        rows.append(row)
                    print('P0_REPLAY_COMPLETE',support,folder.name,split,flush=True)
    with (OUT/'P0_CANDIDATE_SWAP.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    write('BASELINE_REPRODUCTION.json',dict(status='SAVED_EVIDENCE_REPLAY_PASS_FRESH_ENCODER_PENDING',rows=baselines))
    write('P0_INPUT_MANIFEST.json',inputs)
    write('P0_CPU_COMPLETE.json',dict(status='PARTIAL_P0_COMPLETE',seconds=time.time()-started,conditions=len(baselines)//2,comparisons=len(rows),gpu_used=False,source_sha256=sha(Path(__file__)),pending=['Local Top20/50/100 ranking','head score calibration','train-fixed regime thresholds and regime analysis','fresh baseline encoder and ID/mask verification']))
    write('EXPERIMENT_GATE_STATUS.json',dict(P0='PARTIAL_CPU_SWAP_COMPLETE',P1='BLOCKED_PENDING_FULL_P0_AND_FRESH_BASELINE',P2='WAIT_P1',P3='CONDITIONAL_VALIDATION_ONLY',P4='CONDITIONAL_VALIDATION_ONLY',solar_started=False,new_gpu_training_started=False))

if __name__ == '__main__': main()
