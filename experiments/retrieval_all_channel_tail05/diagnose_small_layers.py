exec(open(__file__.replace('diagnose_small_layers.py','diagnose_score.py')).read().split("if __name__")[0])
from cublaslt_batch import CublasLtBatch
@torch.no_grad()
def probe():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision();backend=CublasLtBatch();records=[]
    for B in [1,2,17,18,19,20]:
        x,_,_=d.query('val',0,B);C=d.c;layers={n:m for n,m in f.model.encoder.named_modules() if isinstance(m,torch.nn.Linear)};saved={n:[] for n in layers};hooks=[]
        for n,m in layers.items():
            def hook(m,inp,out,n=n):saved[n].append((inp[0].clone(),out.clone()))
            hooks.append(m.register_forward_hook(hook))
        for c in range(C):a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
        for h in hooks:h.remove()
        for n,m in layers.items():
            inp=torch.stack([z[0] for z in saved[n]]);exp=torch.stack([z[1] for z in saved[n]]);variants=[]
            for mode in ['flat','lt0','lt1','bmm_bias','lt_fp32','plain_fp32']:
                try:
                    if mode=='flat':got=m(inp.reshape(C*B,-1)).reshape(C,B,-1)
                    elif mode=='lt0':got=backend.linear(inp,m,77,1,0)
                    elif mode=='lt1':got=backend.linear(inp,m,77,1,1)
                    elif mode=='lt_fp32':got=backend.linear(inp,m,68,1,0)
                    elif mode=='bmm_bias':got=torch.bmm(inp,m.weight.t()[None].expand(C,-1,-1))+m.bias
                    else:
                        torch.backends.cuda.matmul.allow_tf32=False;got=m(inp.reshape(C*B,-1)).reshape(C,B,-1);f.precision()
                    variants.append(dict(mode=mode,maxdiff=float((got-exp).abs().max()),exact=torch.equal(got,exp)))
                except RuntimeError as e:variants.append(dict(mode=mode,error=str(e)))
            row=dict(batch=B,layer=n,variants=variants);records.append(row);print(json.dumps(row),flush=True)
    a.rt.write_json(OUT/'ALL_SMALL_LAYER_DIAGNOSIS.json',dict(rows=records))
if __name__=='__main__':probe()
