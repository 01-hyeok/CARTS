from remaining_tail_preflight import *
@torch.no_grad()
def probe():
 a.rt.gpu_guard();d=a.Data('Weather_192');rows=[]
 for r in [r for r in a.inventory(d.cell,'FULL') if r['status']=='VALID']:
  f=a.Frozen(d,r,'FULL');f.precision();x,_,_=d.query('val',0,1);valid=x.reshape(d.c,d.h,1)
  for count in [23,28,32]:
   z0=torch.cat([valid,torch.zeros(count-d.c,d.h,1,device=d.device)])
   z1=torch.cat([valid,torch.full((count-d.c,d.h,1),17.125,device=d.device)])
   a0=a.encode_raw_channel_first(f.model,z0,0)[:d.c];a1=a.encode_raw_channel_first(f.model,z1,0)[:d.c]
   assert torch.equal(a0,a1)
   rows.append(dict(model=f.name,encoder_rows=count,real_channels=d.c,padding_rows=count-d.c,real_row_outputs_exact=True,padding_values=[0,17.125]))
  print(json.dumps(rows[-3:]),flush=True)
 a.rt.write_json(OUT/'REAL_GPU_PADDING_GATE.json',dict(status='PASS',rows=rows,padding_not_real_candidates=True))
if __name__=='__main__':probe()
