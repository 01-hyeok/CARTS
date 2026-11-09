import json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.stats import spearmanr,kendalltau
import diagnostic_metrics as m
from diagnostic_core import TrainOnlyRidgeProbe


def test_average_tied_ranks_spearman_and_kendall():
    a=torch.tensor([[1.,1.,4.,3.,5.],[4.,2.,2.,1.,7.]])
    d=torch.tensor([[8.,2.,4.,3.,1.],[4.,9.,9.,3.,2.]])
    mask=torch.ones_like(a,dtype=torch.bool)
    for i in range(2):
        assert m.spearman(a,d,mask)[i].item()==pytest.approx(spearmanr(a[i],-d[i]).statistic)
        assert m.kendall(a,d,mask)[i].item()==pytest.approx(kendalltau(a[i],-d[i]).statistic)


def test_masked_ranks_ignore_invalid_and_constant_undefined():
    a=torch.tensor([[1.,1.,99.,3.,5.]]);d=-a;mask=torch.tensor([[1,1,0,1,1]],dtype=torch.bool)
    assert m.spearman(a,d,mask).item()==pytest.approx(1)
    assert torch.isnan(m.spearman(torch.ones_like(a),d,mask)).all()


def test_full_kendall_sample_deterministic_and_oriented():
    a=torch.arange(200,dtype=torch.float64)[None];mask=torch.ones_like(a,dtype=torch.bool)
    assert m.kendall(a,-a,mask,True).item()==1
    assert m.kendall(a,a,mask,True).item()==-1


def test_moment_probe_matches_direct_train_only_ridge():
    rng=np.random.default_rng(4);x=rng.normal(size=(1000,14));y=x@rng.normal(size=14)+.1*rng.normal(size=1000)
    p=m.MomentRidge();p.update(x[:300],y[:300],'train');p.update(x[300:],y[300:],'train');p.finish()
    direct=TrainOnlyRidgeProbe().fit(x,y,'train',True)
    z=np.c_[(x-direct.mean)/direct.scale,np.ones(len(x))]
    expected=np.stack([z@w for w in direct.states],1)
    assert np.allclose(p.predict(x),expected,atol=1e-10)
    with pytest.raises(AssertionError):p.choose([1,2,3],'test')
    with pytest.raises(AssertionError):p.update(x,y,'val')


@pytest.mark.parametrize('h',[96,192,336,720])
def test_feature_all_horizons_finite_and_no_future(h):
    torch.manual_seed(5);x=torch.randn(2,h);c=torch.randn(2,100,h)
    f=m.features(x,c);assert f.shape==(2,100,len(m.FEATURES))
    assert torch.isfinite(f[...,:14]).all()
    with pytest.raises(TypeError):m.features(x,c,future=c)


def test_features_constant_period_undefined():
    a=m.temporal(torch.ones(4,96));assert torch.isfinite(a[...,:7]).all();assert torch.isnan(a[...,7]).all()


def test_query_metrics_exchange_tail_and_legacy_ndcg():
    torch.manual_seed(8);s=torch.randn(3,100);d=torch.rand(3,100);mask=torch.ones_like(s,dtype=torch.bool)
    out=m.query_metrics(s,d,mask)
    assert out['legacy_ndcg10'].min()>=0 and out['legacy_ndcg10'].max()<=1
    assert torch.allclose(out['exchange_regret'],out['ranking_regret'],atol=1e-6)
    assert torch.equal(out['recall100'],torch.ones(3))


def test_moving_block_bootstrap_preserves_joint_channels():
    from diagnostic_report import moving_ci
    x=np.c_[np.arange(20),-np.arange(20)].astype(float)
    s=moving_ci(x,4,100);assert s['mean']==0 and s['ci_low']==0 and s['ci_high']==0
    x[0,0]=np.nan;assert moving_ci(x,4,100)==moving_ci(x,4,100)


class FakeData:
    h=96;c=3;device=torch.device('cpu');cell='Fixture_96'
    def __init__(self):
        from diagnostic_adapter import Data
        torch.manual_seed(88);self.series=torch.randn(512,self.c);self.future_series=self.series
        self.offsets=torch.arange(self.h);self.starts=np.arange(200)
        self.histories_all=Data.histories_all.__get__(self);self.values_all=Data.values_all.__get__(self)
    def query(self,split,a,b):
        rows=torch.arange(a,b)[:,None]+torch.arange(96)[None]
        return self.series[rows].permute(0,2,1),self.series[rows+96].permute(0,2,1),np.arange(a,b)+300
    def pool_all(self,split,a,b):return torch.arange(100).expand(b-a,self.c,100)
    def mask(self,starts):return torch.ones(len(starts),200,dtype=torch.bool)


@pytest.mark.parametrize('support',['P100','FULL'])
def test_joint_channel_micro_pipeline_cpu(monkeypatch,support):
    import diagnostic_run as run
    monkeypatch.setattr(run,'sync',lambda:None)
    data=FakeData()
    class Cal:
        values=None;names=[]
        def pair(self,*args):return None
    rng=np.random.default_rng(0);p=m.MomentRidge();p.update(rng.normal(size=(100,14)),rng.random(100),'train');p.finish()
    result=run.evaluate_micro(data,'test',0,2,support,Cal(),p,{},0,32)
    assert result['metrics'].shape==(2,3,7,len(m.METRICS))
    assert result['selected_ids'].shape==(2,3,7,10)
    assert np.isfinite(result['metrics'][...,:6]).all()
    assert (result['metrics'][...,m.METRICS.index('ranking_regret')]>=-1e-6).all()
    assert result['group_features'].shape==(2,3,1,4,len(m.FEATURES)+2)
    if support=='FULL':assert result['p100_oracle'].shape==(2,3)


def test_cpu_reporting_end_to_end(tmp_path,monkeypatch):
    import diagnostic_run as run
    import diagnostic_report as report
    monkeypatch.setattr(run,'sync',lambda:None);monkeypatch.setattr(report,'OUT',tmp_path)
    data=FakeData()
    class Cal:
        values=None;names=[]
        def pair(self,*args):return None
    rng=np.random.default_rng(0);p=m.MomentRidge();p.update(rng.normal(size=(100,14)),rng.random(100),'train');p.finish()
    out=tmp_path/'P100'/'Fixture_96';out.mkdir(parents=True)
    for split,pi in [('val',None),('test',0)]:
        part=run.evaluate_micro(data,split,0,2,'P100',Cal(),p,{},pi,32)
        payload={k:part[k] for k in ['metrics','selected_ids','group_features','neighborhood','oracle_ids','query_start']}
        payload.update(method_names=np.array(part['names']),metric_names=np.array(m.METRICS),group_method_names=np.array(part['group_names']),feature_names=np.array(part['feature_names']))
        run.save_npz(out/split/'batch00000.npz',payload)
        report.dump(out/split/'SPLIT_GATE.json',dict(query_count=2,channels=3))
    p.selected=0;report.dump(out/'probe_selection.json',dict(state=p.state()))
    report.summarize_cell('P100','Fixture_96')
    assert (out/'test_QUERY_LEVEL_REGRET.parquet').exists()
    assert (out/'RANKING_QUALITY.csv').exists()
    assert (out/'FEATURE_GROUP_COMPARISON.csv').exists()
    assert (out/'figures/past_future_similarity.png').exists()
    assert 'DEFERRED_FULL_DEPENDENCY' in (out/'PARTIAL_REPORT_KO.md').read_text()


def test_controller_requires_all48_and_no_active_worker(monkeypatch):
    import diagnostic_controller as c
    monkeypatch.setattr(c,'training_state',lambda:(47,[]));assert not c.prerequisites()[0]
    monkeypatch.setattr(c,'training_state',lambda:(48,[123]));assert not c.prerequisites()[0]
    monkeypatch.setattr(c,'training_state',lambda:(48,[]));assert c.prerequisites()[0]
