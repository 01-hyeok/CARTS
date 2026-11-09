"""Authenticate the existing frozen ridge, never refit/reselect a model."""
import os,hashlib,json
from pathlib import Path
import numpy as np
from protocol import ROOT,CELLS
from core import train_label_gate
from io_utils import read,dump,sha,now

def metadata_gate(train,selection,val,data):
    assert train['status']=='FITTED_TRAIN_ONLY' and train['test_used_for_fit'] is False
    assert selection['status']=='VALIDATION_SELECTED' and selection['test_used_for_selection'] is False
    a=dict(train['state']);b=dict(selection['state']);chosen=b.pop('selected');a.pop('selected')
    assert a==b,'frozen probe weights or train scaling changed'
    assert chosen==int(np.argmin(val['probe_validation_loss']))
    assert train['channels']==data['channels']
    expected=np.unique(np.linspace(0,data['counts']['train']-1,min(256,data['counts']['train'])).round().astype(np.int64))
    np.testing.assert_array_equal(train['train_query_rows'],expected)
    train_label_gate(expected+data['starts']['train'][0],data['seq_len'],data['pred_len'],data['starts']['train'][0],data['scaler_train_rows'])
    assert train['pairs']==len(expected)*data['channels']*100==train['state']['n']
    return expected

def main():
    os.chdir(ROOT)
    from adapter import Data
    rows=[]
    for cell in CELLS:
        d=Data(cell,'cpu')
        for support in ('P100','FULL'):
            base=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'/support/cell
            done=read(base/'COMPLETE.json');prov=read(base/'PROVENANCE.json')
            assert prov['data']==d.data_info and prov['pool_hashes']==d.pool_hashes
            train=read(base/'probe_train.json');sel=read(base/'probe_selection.json');val=read(base/'val/SPLIT_GATE.json')
            queryrows=metadata_gate(train,sel,val,d.data_info);hasher=hashlib.sha256()
            for a in range(0,len(queryrows),4):
                batch=queryrows[a:a+4];starts=d.datasets['train'].starts[batch]
                if support=='P100':ids=np.array(d.cpu_pool['train'][batch],np.int64,copy=True)
                else:
                    mask=d.mask(starts).numpy();parts=[]
                    for i,q in enumerate(starts):
                        eligible=np.flatnonzero(mask[i]);assert len(eligible)>=100
                        parts.append(np.stack([np.random.default_rng(np.random.SeedSequence([0,int(q),c,8821])).choice(eligible,100,replace=False) for c in range(d.c)]))
                    ids=np.stack(parts)
                assert ids.min()>=0 and ids.max()<len(d.starts)
                assert bool((ids+2*d.h<=d.data_info['scaler_train_rows']).all())
                hasher.update(ids.tobytes())
            assert hasher.hexdigest()==train['candidate_id_sha256']
            rows.append(dict(cell=cell,support=support,status='PASS',train_queries=len(queryrows),channels=d.c,
                             train_pairs=train['pairs'],test_all_queries=done['query_counts']['test'],
                             candidate_id_sha256=hasher.hexdigest(),selected_alpha=sel['state']['alphas'][sel['state']['selected']],
                             fit_train_only=True,entire_label_future_inside_train=True,weights_frozen=True,selection_validation_only=True,
                             files={n:dict(path=str(base/n),sha256=sha(base/n)) for n in ['probe_train.json','probe_selection.json','PAST_ONLY_PROBE_RESULTS.csv','PROVENANCE.json','COMPLETE.json']}))
        del d
    dump(ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit/FROZEN_PROBE_REUSE_GATE.json',dict(status='PASS',at=now(),rows=rows,
          limitation='256 fixed train timestamps × all channels × 100 future-free sampled candidates; full validation/test evaluation, not full training population; no impossibility conclusion'))
    print('PROBE REUSE PASS',len(rows))

if __name__=='__main__':main()
