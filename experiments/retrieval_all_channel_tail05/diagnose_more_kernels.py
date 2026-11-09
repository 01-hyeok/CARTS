exec(open(__file__.replace('diagnose_more_kernels.py','diagnose_score.py')).read().split("if __name__")[0])
from cublaslt_batch import CublasLtBatch
@torch.no_grad()
def probe():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision();lt=CublasLtBatch();blas=CublasBatch();records=[]
    for B in [1,2,17,18,19,20]:
        x,_,_=d.query('val',0,B);C=d.c;layers={n:m for n,m in f.model.encoder.named_modules() if isinstance(m,torch.nn.Linear)};saved={n:[] for n in layers};hooks=[]
        for n,m in layers.items():
            def hook(m,inp,out,n=n):saved[n].append((inp[0].clone(),out.clone()))
            hooks.append(m.register_forward_hook(hook))
        for c in range(C):a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
        for h in hooks:h.remove()
        for n,m in layers.items():
            inp=torch.stack([z[0] for z in saved[n]]);exp=torch.stack([z[1] for z in saved[n]]);variants=[]
            for compute in [77,68]:
              for backend in ['lt','blas']:
               for choice in (range(12) if backend=='lt' else [-1,0,1,2,3,4,99]):
                try:
                    got=lt.linear(inp,m,compute,1,choice) if backend=='lt' else blas.linear(inp,m,compute,choice)
                    variants.append(dict(backend=backend,compute=compute,choice=choice,maxdiff=float((got-exp).abs().max()),exact=torch.equal(got,exp)))
                except RuntimeError as e:variants.append(dict(backend=backend,compute=compute,choice=choice,error=str(e)))
            row=dict(batch=B,layer=n,variants=variants);records.append(row);print(json.dumps(dict(batch=B,layer=n,exact=[v for v in variants if v.get('exact')],best=sorted([v for v in variants if 'maxdiff' in v],key=lambda v:v['maxdiff'])[:2])),flush=True)
        a.rt.write_json(OUT/'MORE_LAYER_KERNELS.json',dict(rows=records))
    keys=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,d.bank_history(c),0),dim=-1) for c in range(C)])
    rows=[]
    for B in [1,2,17,18,19,20]:
        x,_,_=d.query('val',0,B);q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)])
        exp=torch.stack([q[c]@keys[c].T for c in range(C)]);variants=[]
        for compute in [77,68]:
         for backend in ['lt','blas']:
          for choice in (range(12) if backend=='lt' else [-1,0,1,2,3,4,99]):
           try:
            got=lt.score(q,keys,compute,1,choice) if backend=='lt' else blas.score(q,keys,compute,choice)
            variants.append(dict(backend=backend,compute=compute,choice=choice,maxdiff=float((got-exp).abs().max()),exact=torch.equal(got,exp)))
           except RuntimeError as e:variants.append(dict(backend=backend,compute=compute,choice=choice,error=str(e)))
        rows.append(dict(batch=B,variants=variants));print(json.dumps(dict(score_batch=B,exact=[v for v in variants if v.get('exact')],best=sorted([v for v in variants if 'maxdiff' in v],key=lambda v:v['maxdiff'])[:2])),flush=True)
        a.rt.write_json(OUT/'MORE_SCORE_KERNELS.json',dict(rows=rows))
if __name__=='__main__':probe()
