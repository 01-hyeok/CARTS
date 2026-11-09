import sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import full_analysis as m
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
with torch.no_grad():
 cfg,canonical,sel,exp,args,model,heads,base=m.setup('ETTh1_96',32)
 old=torch.load(canonical/'cache/test.pt',map_location='cpu')
 new_R=old['relation_outputs'].clone()
 _,loader=exp._get_data(flag='test',shuffle=False)
 bad=[]
 for c in range(int(args.enc_in)):
  key=m.bank(model,exp.memory_x,c,0);offset=0
  for bi,(x,y,start) in enumerate(loader):
   x=x.float().cuda();mask,_=exp._candidate_mask(start)
   cached=torch.einsum('bsd,nd->bsn',heads(m.encode_raw_channel_first(model,x,c)),key)
   fresh=m.compute_scores_full_grad_channel_first(model,heads,x,exp.memory_x,c)
   pick,p,_=m.mean_mixture_topk_selection(cached,mask,.1,10)
   fp,_,_=m.mean_mixture_topk_selection(fresh,mask,.1,10)
   mem,off=m.memory_value(args,x,exp.memory_y,exp.memory_x_last,c)
   r=(mem[pick]+off[:,None,None]).mean(1).cpu()
   new_R[offset:offset+len(x),:,c]=r
   ref=old['relation_outputs'][offset:offset+len(x),:,c]
   if not torch.allclose(r,ref,atol=1e-5,rtol=1e-5):
    badrow=int(torch.argmax((r-ref).abs().max(-1).values));ordered=p.mean(1)[badrow].sort(descending=True).values
    bad.append(dict(c=c,batch=bi,row=badrow,query_start=int(start[badrow]),max_score_diff=float((cached-fresh).abs().max()),picks_same=torch.equal(pick,fp),max_prediction_diff=float((r-ref).abs().max()),newpick=pick[badrow].tolist(),boundary_probs=ordered[8:13].tolist(),gap_10_11=float(ordered[9]-ordered[10])))
    print(bad[-1],flush=True)
   assert torch.equal(start,old['query_start_idx'][offset:offset+len(x)])
   offset+=len(x)
  print('channel',c,'bad',len(bad),flush=True)
 m.dump(m.OUT/'parity_probe.json',bad)
 torch.backends.cuda.matmul.allow_tf32=False
 _,B,R,Y=m.load_tensors(exp,base,canonical/'cache','test','cuda',int(args.enc_in))
 gate=m.GlobalLambdaGate().cuda();gate.load_state_dict(torch.load(canonical/'stage2/checkpoint.pth',map_location='cuda')['gate_state_dict'])
 new_metrics=m.eval_gate(gate,B,new_R.cuda(),Y);old_metrics=m.eval_gate(gate,B,R,Y)
 record=dict(new=new_metrics,old=old_metrics,changed_query_channel_count=int((new_R-old['relation_outputs']).abs().amax(1).gt(1e-5).sum()),max_R_diff=float((new_R-old['relation_outputs']).abs().max()),torch_version=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
 m.dump(m.OUT/'parity_metrics_probe.json',record);print(record,flush=True)
