exec(open(__file__.replace('diagnose_lt_score.py','diagnose_score.py')).read().split("if __name__")[0])
from cublaslt_batch import CublasLtBatch
@torch.no_grad()
def probe():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
    end=len(d.datasets['test']);start=end//32*32;x,y,starts=d.query('test',start,end);B,C,H=x.shape
    ref=f.probability('test',start,end);keys=torch.stack([f.keys[c] for c in range(C)])
    q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)],0)
    mask=d.mask(starts)[:,None];top=lambda v:a.rt.stable_topk_indices(v.reshape(B*C,-1),10,largest=True)
    expected=top(ref);backend=CublasLtBatch();rows=[]
    for heuristic_batch in [1,C]:
      for choice in range(12):
        try:
            got=backend.score(q,keys,77,heuristic_batch,choice).permute(1,0,2).masked_fill(~mask,-float('inf'));torch.cuda.synchronize()
            ids=top(got);delta=(got-ref).abs();delta=delta[torch.isfinite(delta)]
            row=dict(heuristic_batch=heuristic_batch,choice=choice,maxdiff=float(delta.max()),ordered=int((ids!=expected).any(-1).sum()),set_mismatch=int((ids.sort(-1).values!=expected.sort(-1).values).any(-1).sum()))
        except RuntimeError as e:row=dict(heuristic_batch=heuristic_batch,choice=choice,error=str(e))
        rows.append(row);print(json.dumps(row),flush=True)
    a.rt.write_json(OUT/'SCORE_LT_ALGORITHM_AUDIT.json',dict(rows=rows))
if __name__=='__main__':probe()
