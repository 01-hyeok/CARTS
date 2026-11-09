"""Non-GPU integration fixture: exact groups, all R, incomplete inputs rejected."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import d56_aggregate as a
from d56_aggregate import dump,sha

def fixture(root):
    src=root/'complete_features'/'ETTh1_96'/'V5_P100_to_P100';src.mkdir(parents=True)
    identity=dict(checkpoint=dict(checkpoint_sha256='fixture-only'))
    files={};C=2;N=100;ids=np.arange(N)[None].repeat(C,0);order=ids[:,::-1].copy()
    heads=np.stack([np.roll(ids,-10*h,axis=1)[:,:10] for h in range(5)],1)
    for split in ['val','test']:
        (src/split).mkdir()
        for i in range(32):
            z=dict(global_ids=ids,valid=np.ones((C,N),bool),eligible=np.ones((C,N),bool),mse=(ids+1.).astype(np.float32),
                ordered_ids=order,ordered_scores=np.tile(np.arange(N,0,-1),(C,1)).astype(float),
                oracle_ids=ids[:,:30],oracle_mse=(ids[:,:30]+1.).astype(np.float32),oracle_initial_rank=(N-ids[:,:30]),head_ids=heads,
                random_ids=np.roll(ids,i,axis=1),full_oracle_ids=ids[:,:10],near_support_count=np.repeat(10,C),near_mask=ids<10,
                query_index=np.array([i]),query_start=np.array([i*96]),query_std=np.ones(C),best_lag=(ids%25)-12,
                zero=np.zeros_like(ids,dtype=float),best=np.ones_like(ids,dtype=float)*.1,gain=np.ones_like(ids,dtype=float)*.1,
                margin=np.ones_like(ids,dtype=float)*.02,entropy=np.ones_like(ids,dtype=float),candidate_std=np.ones_like(ids,dtype=float),
                recent_difference=np.ones_like(ids,dtype=float),near_tie_lag_fraction=np.ones_like(ids,dtype=float)/25)
            p=src/split/('query%05d.npz'%i);np.savez(p,**z);files[str(p.relative_to(src))]=sha(p)
            dump(p.with_suffix('.json'),dict(identity=identity,all_channels_single_batch=True,lag_shard_hashes={}))
    dump(src/'FEATURE_GATE.json',dict(status='PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING',identity=identity,channels=C,queries={'val':32,'test':32},files=files,
          peak_allocated_bytes=0,peak_reserved_bytes=0))
    return src

def test_complete_cpu_aggregation_and_resume(tmp_path,monkeypatch):
    src=fixture(tmp_path);monkeypatch.setattr(a,'OUT',tmp_path)
    monkeypatch.setattr(a,'block_ci',lambda *x:dict(delta=0.,ci_low=0.,ci_high=0.,blocks=32,sufficient=True))
    a.main('ETTh1_96','V5','P100','P100')
    dest=tmp_path/'complete_analysis'/'ETTh1_96'/'V5_P100_to_P100'
    gate=json.loads((dest/'ANALYSIS_GATE.json').read_text());assert gate['status']=='PASS_D5_D6_CASE'
    assert gate['R']==[10,20,50,100] and gate['Oracle30'] and gate['exact_near_support_count']
    a.main('ETTh1_96','V5','P100','P100')

def test_corrupt_source_rejected(tmp_path,monkeypatch):
    src=fixture(tmp_path);monkeypatch.setattr(a,'OUT',tmp_path)
    p=src/'val/query00000.npz';p.write_bytes(p.read_bytes()+b'altered')
    with pytest.raises(AssertionError):a.main('ETTh1_96','V5','P100','P100')

def test_missing_query_rejected(tmp_path,monkeypatch):
    src=fixture(tmp_path);monkeypatch.setattr(a,'OUT',tmp_path)
    (src/'test/query00031.npz').unlink()
    with pytest.raises(AssertionError):a.main('ETTh1_96','V5','P100','P100')
