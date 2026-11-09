"""Identify the exact E96 tail variant mismatch; no outputs are substituted."""
import sys,json,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import d123_full_population as pop
from utils.full_candidate_bank import encode_raw_channel_first
import torch
OUT=pop.BASE/'extensions_d5_d6/cublas_recovery03'

@torch.no_grad()
def main():
    pop.start_gpu();d=pop.Data('ETTh1_96');row=next(r for r in pop.inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='FULL')
    f=pop.Frozen(d,row);f.precision();rows=[]
    for split in ['val','test']:
        a=len(d.datasets[split])-1;b=a+1
        g,mask=f.full_logits(split,a,b);pool=d.pool_all(split,a,b)
        for name in ['FULL','P100_CONTROLLED']:
            support='FULL' if name=='FULL' else 'P100'
            allids=torch.arange(len(d.starts),device=d.device).expand(1,d.c,-1) if name=='FULL' else pool
            gg=g if name=='FULL' else g.gather(-1,pool[:,:,None].expand(-1,-1,5,-1))
            mm=mask if name=='FULL' else torch.ones_like(pool,dtype=torch.bool)
            prod=pop.selections(gg,mm,allids);actual=allids.gather(-1,prod['sel'])
            expected=pop.canonical_cross_ids(f,split,a,b,support)
            native=f.canonical_full_ids(split,a,b) if name=='FULL' else None
            key='FULL' if name=='FULL' else 'P100_CONTROLLED'
            old=pop.ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/d.cell/f.name/split/('batch%05d.npz'%(a//32))
            record=dict(split=split,variant=name,query_index=a,channels=d.c,batch=1,ordered_mismatch_count=int((actual!=expected).sum()),
                        actual=actual.cpu().tolist(),canonical=expected.cpu().tolist(),native_full_equal=None if native is None else torch.equal(actual,native))
            if old.exists():
                meta=json.loads(old.with_suffix('.json').read_text());assert pop.sha(old)==meta['sha256'];z=pop.read_npz(old)
                record['stored_controlled_exact']=pop.np.array_equal(actual.cpu().numpy(),z[key+'__selected_ids'])
            # Compare the actual head logits/probabilities with channel-reference
            # tensors to distinguish score errors from mean/softmax reductions.
            ch_logits=[];ch_prob=[]
            for c in range(d.c):
                x,_,_=d.channel_batch(split,a,b,c,future=False)
                z=encode_raw_channel_first(f.model,x,0);q=f.heads(z);k=f.extension_reference_keys[c]
                s=torch.einsum('bsd,nd->bsn',q,k)
                if name!='FULL':s=s.gather(-1,pool[:,c,None].expand(-1,5,-1))
                ch_logits.append(s);ch_prob.append(torch.softmax(s.masked_fill(~mm[:,c,None],-float('inf'))/.1,-1))
            cg=torch.stack(ch_logits,1);cp=torch.stack(ch_prob,1);cs=cp.mean(2)
            record.update(logits_exact=torch.equal(gg,cg),logit_max_delta=float((gg-cg).abs().max()),head_prob_exact=torch.equal(prod['ph'],cp),
                          score_exact=torch.equal(prod['score'],cs),score_max_delta=float((prod['score']-cs).abs().max()))
            rows.append(record);print('POP_TAIL',json.dumps(record),flush=True)
    pop.dump(OUT/'POPULATION_PROBE.json',dict(status='DIAGNOSED_NOT_RECOVERED',rows=rows,production_activated=False,source_checkpoint_unchanged=pop.sha(row['checkpoint'])==row['checkpoint_sha256']))

if __name__=='__main__':
    try:main()
    except BaseException as e:pop.dump(OUT/'POPULATION_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),production_activated=False));raise
