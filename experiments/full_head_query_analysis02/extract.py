import os,sys,json,csv,time,importlib.util,argparse,gc
from pathlib import Path
import numpy as np,torch
from core import NAMES,FEATURES,features,dump,sha,swap_masks
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
spec=importlib.util.spec_from_file_location('canonical_full',ROOT/'experiments/full_head_analysis01/full_analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=ROOT/'results/TRACK-V5-FULL-HEAD-ANALYSIS01';OUT=ROOT/'results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02';GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
@torch.no_grad()
def run(cell):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
 assert torch.__version__.startswith('1.12.0'),torch.__version__
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.22,0)
 t0=time.time();p=OUT/cell;p.mkdir(exist_ok=True);local=p/'local';local.mkdir(exist_ok=True)
 if (p/'EXTRACTION_PASS.json').exists():print('REUSE extraction',cell,flush=True);return
 cfg,canon,sel,exp,args,model,heads,base=m.setup(cell,32);h=args.pred_len;cnum=args.enc_in
 oldprov=json.loads((OLD/cell/'CHECKPOINT_PROVENANCE.json').read_text());ck=Path(sel['selected_checkpoint_path']);bc=Path(cfg['base_checkpoint']);assert sha(ck)==oldprov['checkpoint_sha256'];assert sha(bc)==oldprov['base_sha256']
 src={'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'base':str(bc),'base_sha256':sha(bc),'dataset':str((ROOT/args.root_path/args.data_path).resolve()),'dataset_sha256':sha(ROOT/args.root_path/args.data_path),'source_sha256':sha(__file__),'canonical_helper_sha256':sha(ROOT/'experiments/full_head_analysis01/full_analysis.py'),'seed':0,'gpu_uuid':GPU,'batch':32,'chunk':0,'runtime':torch.__version__,'numpy':np.__version__,'cache_hashes':{}}
 metrics=list(csv.DictReader(open(OLD/cell/'HEAD_STAGE2_MSE_MAE.csv')));allrow=next(z for z in metrics if z['subset']=='12345');lam=float(allrow['secondary_fixed_lambda'])
 src['fixed_lambda']=lam;np.save(local/'candidate_start_ids.npy',exp.memory_sampler.starts)
 # Real one-batch gate before any full inference. Same full support/batch/runtime.
 torch.backends.cuda.matmul.allow_tf32=True
 ds,loader=exp._get_data(flag='val',shuffle=False);x,y,start=next(iter(loader));x=x.float().cuda();k=m.bank(model,exp.memory_x,0,0);mask,_=exp._candidate_mask(start)
 scores=torch.einsum('bsd,nd->bsn',heads(m.encode_raw_channel_first(model,x,0)),k);_,prob,_=m.mean_mixture_topk_selection(scores,mask,sel['tau_s'],10);mem,off=m.memory_value(args,x,exp.memory_y,exp.memory_x_last,0)
 for name in NAMES:
  ids=m.selection(prob,tuple(int(a)-1 for a in name),mask);pred=(mem[ids]+off[:,None,None]).mean(1).cpu().numpy();saved=np.load(OLD/cell/'cache'/name/'val.npy',mmap_mode='r')[:len(x),:,0];assert np.array_equal(pred,saved),(cell,name,'one-batch cache parity',abs(pred-saved).max())
 dump(p/'REAL_BATCH_PASS.json',dict(status='PASS',cell=cell,queries=len(x),channel=0,configurations=11,full_candidates=len(k),prediction_max_diff=0.0));print('[SMOKE PASS]',cell,flush=True);del x,k,scores,prob,mem,off;torch.cuda.empty_cache()
 validation=[]
 for split in ['val','test']:
  ds,loader=exp._get_data(flag=split,shuffle=False);n=len(ds);ids=np.asarray(ds.starts);np.save(local/(split+'_query_ids.npy'),ids)
  saved={name:np.load(OLD/cell/'cache'/name/(split+'.npy'),mmap_mode='r') for name in NAMES}
  for name in NAMES:
   pt=torch.load(OLD/cell/'cache'/name/(split+'.pt'),map_location='cpu');assert np.array_equal(pt['query_start_idx'].numpy(),ids);assert np.array_equal(pt['relation_outputs'].numpy(),saved[name]);del pt
   src['cache_hashes'][name+'/'+split]=sha(OLD/cell/'cache'/name/(split+'.npy'))
  bmap=np.lib.format.open_memmap(local/(split+'_base.npy'),mode='w+',dtype='float32',shape=(n,h,cnum));ymap=np.lib.format.open_memmap(local/(split+'_truth.npy'),mode='w+',dtype='float32',shape=(n,h,cnum));fmap=np.lib.format.open_memmap(local/(split+'_input_features.npy'),mode='w+',dtype='float32',shape=(n,cnum,4));pos=0
  torch.backends.cuda.matmul.allow_tf32=False
  for x,y,start in loader:
   b=len(x);assert np.array_equal(ids[pos:pos+b],start.numpy());fmap[pos:pos+b]=features(x.numpy());x=x.float().cuda();bmap[pos:pos+b]=(base(x)+x[:,-1:,:]).cpu().numpy();ymap[pos:pos+b]=y.numpy();pos+=b
  bmap.flush();ymap.flush();fmap.flush()
  if split=='test':
   B=torch.from_numpy(np.array(bmap)).cuda();Y=torch.from_numpy(np.array(ymap)).cuda();R=torch.from_numpy(np.array(saved['12345'])).cuda();fused=B+lam*(R-B);mse=float((fused-Y).square().mean());mae=float((fused-Y).abs().mean())
   assert mse==float(allrow['secondary_mse']) and mae==float(allrow['secondary_mae']),(cell,'All5 fixed-lambda metric exact gate',mse,mae,allrow)
   validation.append(dict(gate='All5 fixed-lambda exact original GPU reduction',mse=mse,mae=mae,status='PASS'));del B,Y,R,fused;torch.cuda.empty_cache()
  idmap=np.lib.format.open_memmap(local/(split+'_top10_ids.npy'),mode='w+',dtype='int32',shape=(n,cnum,11,10));swap=np.lib.format.open_memmap(local/(split+'_candidate_swaps.npy'),mode='w+',dtype='float32',shape=(n,cnum,5,3));maxdiff=0.;torch.backends.cuda.matmul.allow_tf32=True
  for c in range(cnum):
   keys=m.bank(model,exp.memory_x,c,0);pos=0
   for x,y,start in loader:
    x=x.float().cuda();truth=y.float().cuda()[:,:,c];b=len(x);mask,_=exp._candidate_mask(start);assert np.array_equal(mask.cpu().numpy(),m.independent_raft_mask(start.numpy(),exp.memory_sampler.starts,h,h))
    scores=torch.einsum('bsd,nd->bsn',heads(m.encode_raw_channel_first(model,x,c)),keys);_,prob,_=m.mean_mixture_topk_selection(scores,mask,sel['tau_s'],10);mem,off=m.memory_value(args,x,exp.memory_y,exp.memory_x_last,c);picks=[]
    for j,name in enumerate(NAMES):
     idx=m.selection(prob,tuple(int(a)-1 for a in name),mask);assert mask.gather(1,idx).all();assert (idx.sort().values.diff(dim=-1)>0).all();pred=(mem[idx]+off[:,None,None]).mean(1).cpu().numpy();expected=saved[name][pos:pos+b,:,c];d=float(abs(pred-expected).max());maxdiff=max(maxdiff,d);assert np.array_equal(pred,expected),(cell,split,c,pos,name,'cache parity',d)
     idmap[pos:pos+b,c,j]=idx.cpu().numpy();picks.append(idx)
    old=picks[5]
    for j in range(5):
     new=picks[6+j];enter=~(new[:,:,None]==old[:,None,:]).any(-1);leave=~(old[:,:,None]==new[:,None,:]).any(-1);count=enter.sum(-1);assert torch.equal(count,leave.sum(-1))
     ine=(mem[new]+off[:,None,None]-truth[:,None]).square().mean(-1);oute=(mem[old]+off[:,None,None]-truth[:,None]).square().mean(-1)
     vi=(ine*enter).sum(-1)/count.clamp_min(1);vo=(oute*leave).sum(-1)/count.clamp_min(1);vi[count==0]=float('nan');vo[count==0]=float('nan');swap[pos:pos+b,c,j]=torch.stack([count.float(),vi,vo],-1).cpu().numpy()
    pos+=b
   assert pos==n;print('[IDS]',cell,split,'channel',c+1,'/',cnum,'exact R maxdiff',maxdiff,flush=True);del keys;idmap.flush();swap.flush()
  validation.append(dict(gate=split+' all11 R exact + query IDs + mask',status='PASS',max_prediction_diff=maxdiff,n_queries=n,n_channels=cnum,ids_sha256=sha(local/(split+'_top10_ids.npy'))));del saved,bmap,ymap,fmap,idmap,swap;gc.collect();torch.cuda.empty_cache()
 src['checkpoint_sha256_after']=sha(ck);src['base_sha256_after']=sha(bc);assert src['checkpoint_sha256_after']==src['checkpoint_sha256'] and src['base_sha256_after']==src['base_sha256'];dump(p/'PROVENANCE.json',src);dump(p/'EXTRACTION_PASS.json',dict(status='PASS',cell=cell,gates=validation,wall_seconds=time.time()-t0,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20));print('[EXTRACTION COMPLETE]',cell,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);a=ap.parse_args();run(a.cell)
