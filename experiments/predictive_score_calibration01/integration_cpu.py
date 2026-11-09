"""Real first-block CPU integration, explicitly not canonical GPU reproduction."""
import os,sys,json,copy,io
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data/pjh_workspace/CARTS');sys.path.insert(0,str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'))
import runtime as rt
import torch,numpy as np
from calibrators import ResidualCalibrator,FixedHeadWeights,teacher_kl
torch.set_num_threads(1)
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'

def main():
    os.chdir(ROOT);cell='ETTh1_96';d=rt.CellData(cell,'cpu')
    identity=json.loads((OLD/'P100'/cell/'IDENTITY.json').read_text());row=identity['checkpoint']
    assert rt.sha256_file(row['checkpoint'])==row['checkpoint_sha256']
    model,heads,_,_=rt.model_init(cell,0,5,'cpu')
    payload=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model']);heads.load_state_dict(payload['heads'])
    model.eval();heads.eval()
    for p in list(model.parameters())+list(heads.parameters()):p.requires_grad_(False)
    parts=[];ids=[]
    with torch.no_grad():
        for c in range(d.c):
            x,_,starts=d.channel_batch('train',0,32,c,future=False);pool=d.pool('train',starts,c)
            parts.append(rt.scores(model,heads,x,d,c,pool,query_channel=0));ids.append(pool)
    g=torch.stack(parts,1);x=g.transpose(-1,-2).contiguous().double();ids=torch.stack(ids,1)
    cache=OLD/'P100'/cell/'frozen_scalar_cache/train/batch00000.npz'
    assert rt.sha256_file(cache)==json.loads(cache.with_suffix('.json').read_text())['sha256']
    with np.load(cache) as z:p=torch.from_numpy(z['p'].copy());distance=torch.from_numpy(z['distance'].copy())
    mask=torch.ones_like(p,dtype=torch.bool)
    teacher=rt.normalized_teacher_prob(distance.reshape(-1,100),mask.reshape(-1,100),.1).reshape_as(distance).double()
    cpu_p=(g/.1).softmax(-1).mean(-2)
    order=lambda s:torch.argsort(s,dim=-1,descending=True,stable=True)[...,:10]
    results=[]
    for kind in ('fixed','cosine'):
        torch.manual_seed(0);cal=FixedHeadWeights() if kind=='fixed' else ResidualCalibrator()
        if kind=='cosine':cal.normalizer.fit(x,'train')
        assert torch.equal(order(cal.score(p,x,mask,ranking=True)),order(p))
        initial=cal.logprob(p,x,mask).exp()
        assert torch.allclose(initial,p.double(),atol=2e-6,rtol=2e-6)
        # Invalid fitting split rejected before mutation.
        if kind=='cosine':
            try:cal.normalizer.fit(x,'test');raise AssertionError('fit allowed')
            except ValueError:pass
        before=cal.score(p,x,mask).detach().clone()
        alternate_teacher=torch.roll(teacher,1,-1)
        assert torch.equal(before,cal.score(p,x,mask).detach())
        assert not torch.equal(teacher,alternate_teacher)
        opt=torch.optim.AdamW(cal.parameters(),lr=.001);gradient=[]
        for step in range(2):
            opt.zero_grad();loss=teacher_kl(teacher,cal.logprob(p,x,mask),mask);loss.backward()
            norms={n:float(v.grad.norm()) if v.grad is not None else 0. for n,v in cal.named_parameters()}
            assert all(np.isfinite(v) for v in norms.values()) and sum(norms.values())>0
            if kind=='cosine' and step==1:assert norms['layers.0.weight']>0
            gradient.append(norms);opt.step()
        clone=copy.deepcopy(cal);buf=io.BytesIO();torch.save(cal.state_dict(),buf);buf.seek(0);clone.load_state_dict(torch.load(buf,weights_only=False))
        assert torch.equal(cal.score(p,x,mask),clone.score(p,x,mask))
        results.append(dict(model=kind,gradient=gradient,save_load_exact=True,initial_ordered_ids_exact=True,initial_max_probability_delta=float((initial-p).abs().max())))
    assert all(v.grad is None for v in list(model.parameters())+list(heads.parameters()))
    info=dict(status='PASS_CPU_ACTUAL_TRAIN_BLOCK_INTEGRATION',cell=cell,queries=32,channels=d.c,official_training=False,canonical_gpu_reproduction=False,cpu_encoder_ordered_id_mismatches=int((order(cpu_p)!=order(p)).any(-1).sum()),cpu_encoder_max_probability_delta=float((cpu_p-p).abs().max()),models=results)
    with (OUT/'P1_CPU_INTEGRATION.json').open('x') as f:json.dump(info,f,indent=2)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
