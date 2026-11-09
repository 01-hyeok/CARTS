exec(open(__file__.replace('capture_score_inputs.py','diagnose_score.py')).read().split("if __name__")[0])
@torch.no_grad()
def capture():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
    end=len(d.datasets['test']);start=end//32*32;B=end-start;C=d.c;x,y,starts=d.query('test',start,end)
    f.probability('test',start,end);saved=[]
    hook=f.model.encoder.register_forward_hook(lambda m,inp,out:saved.append(out.clone()))
    ref=f.probability('test',start,end);hook.remove();assert len(saved)==C
    q=torch.stack([F.normalize(z,dim=-1) for z in saved],1);key=torch.stack([f.keys[c] for c in range(C)])
    direct=torch.stack([q[:,c].contiguous()@key[c].T for c in range(C)],1)
    batch=score_all(q[:,:,None],key)[:,:,0];mask=d.mask(starts)[:,None]
    def compare(v):
        v=v.masked_fill(~mask,-float('inf'));delta=(v-ref).abs();delta=delta[torch.isfinite(delta)]
        top=lambda vv:a.rt.stable_topk_indices(vv.reshape(B*C,-1),10,largest=True).reshape(B,C,10)
        ids=top(v);old=top(ref);pred=(d.values_all(ids)+x[:,:,-1,None,None]).mean(2)
        expected=torch.as_tensor(f.reference('test')[0][start:end].transpose(0,2,1),device=d.device)
        return dict(max_score_delta=float(delta.max()),ordered=int((ids!=old).any(-1).sum()),stored_prediction_maxdiff=float((pred-expected).abs().max()))
    legacy_x=torch.stack([d.channel_batch('test',start,end,c,future=False)[0][:,:,0] for c in range(C)],1)
    f.precision();recomputed=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)],1)
    variants=[]
    for tf32 in [True,False]:
        torch.backends.cuda.matmul.allow_tf32=tf32
        variants.append(dict(tf32=tf32,direct=compare(torch.stack([q[:,c].contiguous()@key[c].T for c in range(C)],1)),batch=compare(score_all(q[:,:,None],key)[:,:,0])))
    result=dict(input_delta=float((x-legacy_x).abs().max()),query_delta=float((q-recomputed).abs().max()),captured_queries=len(saved),ref=compare(ref),direct=compare(direct),batch=compare(batch),variants=variants,precision=full_precision_runtime.precision_policy('FULL','V0',d.cell))
    a.rt.write_json(OUT/'CAPTURED_SCORE_INPUTS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':capture()
