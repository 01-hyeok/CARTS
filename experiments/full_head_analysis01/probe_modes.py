import sys,torch,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));import full_analysis as m
from scripts.train_t_pure_multislot01 import compute_scores_full_grad
with torch.no_grad():
 cfg,can,sel,exp,args,model,heads,base=m.setup('ETTh1_96',32)
 old=torch.load(can/'cache/test.pt',map_location='cpu')['relation_outputs']
 _,loader=exp._get_data(flag='test',shuffle=False)
 x,y,start=list(itertools.islice(loader,1,2))[0] if False else next(iter(loader))
 it=iter(loader);next(it);x,y,start=next(it);x=x.float().cuda();mask,_=exp._candidate_mask(start)
 mem,off=m.memory_value(args,x,exp.memory_y,exp.memory_x_last,0);ref=old[32:64,:,0]
 rows=[]
 for tf32 in [False,True]:
  torch.backends.cuda.matmul.allow_tf32=tf32
  for name,func in [('channel_first',m.compute_scores_full_grad_channel_first),('legacy',compute_scores_full_grad)]:
   scores=func(model,heads,x,exp.memory_x,0);idx,p,_=m.mean_mixture_topk_selection(scores,mask,sel['tau_s'],10)
   r=(mem[idx]+off[:,None,None]).mean(1).cpu()
   row=dict(tf32=tf32,func=name,maxdiff=float((r-ref).abs().max()),equal_rows=int(torch.isclose(r,ref,atol=1e-5,rtol=1e-5).all(-1).sum()))
   rows.append(row);print(row,flush=True)
 m.dump(m.OUT/'precision_probe.json',rows)
