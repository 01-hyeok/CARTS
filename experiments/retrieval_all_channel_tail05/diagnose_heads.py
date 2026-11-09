exec(open(__file__.replace('diagnose_heads.py','diagnose_score.py')).read().split("if __name__")[0])
from cublaslt_batch import CublasLtBatch
@torch.no_grad()
def probe():
 a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5'),'FULL');f.precision();lt=CublasLtBatch();rows=[]
 for B in [1,17,18,19,20]:
  x,_,_=d.query('val',0,B);C=d.c;z=torch.stack([a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0) for c in range(C)])
  expected=torch.stack([f.heads(z[c]) for c in range(C)]);w=f.heads.W.reshape(-1,z.shape[-1]).unsqueeze(0).expand(C,-1,-1).contiguous();v=[]
  for mode in ['flat','bmm','lt770','lt771','lt680']:
   if mode=='flat':got=f.heads(z.reshape(C*B,-1)).reshape(C,B,5,-1)
   else:
    got=torch.bmm(z,w.transpose(1,2)) if mode=='bmm' else lt.score(z,w,int(mode[2:4]),1,int(mode[-1]))
    got=F.normalize(got.reshape(C,B,5,-1),dim=-1)
   v.append(dict(mode=mode,maxdiff=float((got-expected).abs().max()),exact=torch.equal(got,expected)))
  row=dict(batch=B,variants=v);rows.append(row);print(json.dumps(row),flush=True)
 a.rt.write_json(OUT/'V5_HEAD_KERNEL_DIAGNOSIS.json',dict(rows=rows))
if __name__=='__main__':probe()
