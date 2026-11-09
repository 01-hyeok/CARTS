"""Frozen-encoder scorer arms; no future/teacher data accepted here."""
import torch
from torch import nn
from learned_scores import *
from experiment_common import load,sha

class Features:
    def __init__(self,d,folder):
        import json
        m=json.loads((folder/'COMPLETE.json').read_text())
        assert m['status']=='PASS_ALL_QUERY_PAST_FEATURES' and m['data']==d.data_info
        for name,h in m['files'].items():assert sha(folder/name)==h
        self.d=d;self.data={};self.provenance=m
        for split in ('train','val','test'):
            z=load(folder/(split+'.pt'));starts=z.pop('starts')
            assert (starts.numpy()==d.datasets[split].starts).all()
            self.data[split]={k:v.to(d.device) for k,v in z.items()}
        self.norm=FixedStandardizer(6).to(d.device)
        self.norm.load_state_dict(load(folder/'feature_normalizer.pt'))
    def pair(self,split,rows,ids,kind):
        rows=torch.as_tensor(rows,device=self.d.device,dtype=torch.long)
        q={k:v[rows][:,:,None] for k,v in self.data[split].items()}
        channels=torch.arange(self.d.c,device=self.d.device)[None,:,None]
        c={k:self.data['train'][k][ids,channels] for k in ('phi','coef','recent','residual')}
        if kind=='L2':return self.norm(q['phi']),self.norm(c['phi'])
        if kind=='DYN':return dynamics_pairs(q,c,self.norm)
        if kind=='CROSS':return cross_pairs(q,c,self.norm)
        if kind=='C0':return -cross_error(q,c)
        raise ValueError(kind)

class Arm(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        self.kind='L2' if name.endswith('L2') else ('DYN' if name.startswith('D') else 'CROSS')
        self.scorer=DynamicsL2() if self.kind=='L2' else PairMLP(15 if self.kind=='DYN' else 18)
        self.fusion=Fusion();self.fused=name.startswith(('D2','C2'))
        if not self.fused:self.fusion.beta.requires_grad_(False)
    def raw(self,features):
        return self.scorer(*features) if self.kind=='L2' else self.scorer(features)
    def calibrate(self,features):
        if self.kind!='L2':self.scorer.standardizer.fit(features,'train')
        with torch.no_grad():self.fusion.calibration.fit(self.raw(features).reshape(-1,1),'train')
    def score(self,p,raw,mask,ranking=False):
        if self.fused:
            return self.fusion.ranking_score(p,raw,mask) if ranking else self.fusion.score(p,raw,mask)
        return self.fusion.calibration(raw[...,None]).squeeze(-1).masked_fill(~mask,-torch.inf)
    def logprob(self,p,raw,mask):
        return masked_log_probability(self.score(p,raw,mask)/.1,mask)
    def raw_chunks(self,features,split,rows,ids,chunk=256):
        return assembled_scores(ids.shape[-1],chunk,lambda a,b:self.raw(features.pair(split,rows,ids[...,a:b],self.kind)))
