"""Eight explicitly authorized Solar Full runs, blocked until all prior diagnostics.

Fresh seed0 encoder initialization; existing z-score teacher and canonical V0/V5
losses. Fixed ten epochs, batch32, global Full-candidate normalization. No Stage2.
"""
import argparse,copy,gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import batch_core as core
from prerequisites import verify
from evaluation import evaluate as evaluate_all
rt=core.rt;ROOT=core.original.ROOT;HERE=Path(__file__).resolve().parent
DIAG=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
OUT=ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04'
CKPT=ROOT/'checkpoints/track_solar_full_all_channel04'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):rt.write_json(p,x)
def source_hashes():return {p.name:rt.sha256_file(p) for p in HERE.glob('*.py')}
def prerequisites():return verify(ROOT)

def guard():
    prerequisites()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert GPU not in apps,'Another GPU1 process is present; never compete or stop it'
    rt.gpu_guard();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def batch_update(data,model,heads,opt,x,y,starts,gate):
    core.validate_training_shape(data,gate['query_chunk'],gate['candidate_chunk'])
    opt.zero_grad(set_to_none=True)
    loss=core.backward_batch(data,model,heads,x.permute(0,2,1).contiguous(),y.permute(0,2,1).contiguous(),starts,gate['query_chunk'],gate['candidate_chunk'])
    for p in list(model.encoder.parameters())+([] if heads is None else list(heads.parameters())):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert np.isfinite(loss);opt.step();return loss

def evaluate(data,model,heads,split,dest=None):
    gate=read(OUT/'SMOKE_GATE.json')
    return evaluate_all(data,model,heads,split,gate['query_chunk'],gate['candidate_chunk'],dest)

def smoke():
    from gpu_smoke import run
    run(sys.modules[__name__])


def train(cell,nheads):
    guard();gate=read(OUT/'SMOKE_GATE.json');assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
    assert gate['canonical_source_hashes']==rt.source_hashes()
    assert cell in [f'Solar_{h}' for h in [96,192,336,720]] and nheads in [0,5]
    dest=OUT/cell/f'V{nheads}'/'seed0';ck=CKPT/cell/f'V{nheads}'/'seed0'
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json');assert done['source_hashes']==source_hashes();assert rt.sha256_file(done['checkpoint'])==done['checkpoint_sha256'];return
    dest.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter();data=rt.CellData(cell,'cuda:0');assert data.c==137
    model,heads,init_hash,head_hash=rt.model_init(cell,0,nheads,'cuda:0')
    params=list(model.parameters())+([] if heads is None else list(heads.parameters()))
    opt=torch.optim.Adam(params,lr=.001);loader=data.loader('train',batch=32,seed=0);history=[];start_epoch=1
    if (ck/'latest.pth').exists():
        saved=torch.load(ck/'latest.pth',map_location='cpu',weights_only=False);assert saved['source_hashes']==source_hashes()
        model.load_state_dict(saved['model_state_dict']);opt.load_state_dict(saved['optimizer'])
        if heads is not None:heads.load_state_dict(saved['slot_heads_state_dict'])
        loader.generator.set_state(saved['loader_rng']);core.rng_restore(saved['rng']);history=saved['history'];start_epoch=saved['epoch']+1
    dump(dest/'PROVENANCE.json',dict(cell=cell,model=f'V{nheads}',seed=0,source_hashes=source_hashes(),data=data.data_info,
        full_support=True,teacher='canonical per-query valid FULL z-score FutureMSE, tau=.1',student='global full softmax, tau=.1; V5 mean probabilities',
        objective='historical V0 log-softmax KL' if nheads==0 else 'historical V5 kl_loss_from_prob',batch=32,epochs=10,adam_lr=.001,
        initialization='seed0 fresh Model; saved reference args only, no pretrained weights',init_hash=init_hash,head_hash=head_hash,
        query_key_gradients=True,all_channel_batch=137,query_chunk=gate['query_chunk'],candidate_chunk=gate['candidate_chunk'],dropout=gate['dropout_policy'],selection='minimum validation Mean-Mixture RetMSE after 10 fixed epochs',stage2=False))
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,11):
        model.train()
        if heads is not None:heads.train()
        t=time.perf_counter();loss_sum=0.;count=0;steps=0
        for x,y,starts in loader:
            x=x.to(data.device).float();y=y.to(data.device).float()
            value=batch_update(data,model,heads,opt,x,y,starts,gate)
            loss_sum+=value*len(x);count+=len(x);steps+=1
            if steps%25==0:
                dump(dest/'CURRENT.json',dict(epoch=epoch,steps=steps,queries=count,total_queries=len(data.datasets['train']),at=rt.now(),pid=os.getpid()))
                print('FULL TRAIN',cell,nheads,'epoch',epoch,'step',steps,'/',len(loader),'loss',value,flush=True)
        assert count==len(data.datasets['train'])
        val=evaluate(data,model,heads,'val')
        history.append(dict(epoch=epoch,train_kl=loss_sum/count,train_queries=count,steps=steps,val=val,seconds=time.perf_counter()-t))
        saved=dict(epoch=epoch,model_state_dict=model.state_dict(),slot_heads_state_dict=None if heads is None else heads.state_dict(),optimizer=opt.state_dict(),
            rng=core.rng_snapshot(),loader_rng=loader.generator.get_state(),history=history,source_hashes=source_hashes(),args=vars(data.args))
        rt.atomic_torch(ck/f'epoch{epoch:02d}.pth',saved);rt.atomic_torch(ck/'latest.pth',saved)
        dump(dest/'history.json',history);print('FULL EPOCH COMPLETE',cell,nheads,epoch,json.dumps(val),flush=True)
    assert len(history)==10
    selected=min(history,key=lambda r:(r['val']['retmse10'],r['epoch']));path=ck/f'epoch{selected["epoch"]:02d}.pth'
    payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model_state_dict'])
    if heads is not None:heads.load_state_dict(payload['slot_heads_state_dict'])
    selection=dict(epoch=selected['epoch'],validation_retmse=selected['val']['retmse10'],checkpoint=str(path),sha256=rt.sha256_file(path),test_used_for_selection=False)
    dump(dest/'checkpoint_selection.json',selection)
    val=evaluate(data,model,heads,'val',dest/'cache');test=evaluate(data,model,heads,'test',dest/'cache')
    assert val['retmse10']==selected['val']['retmse10'],'Selected validation metric did not reproduce'
    dump(dest/'COMPLETE.json',dict(status='COMPLETE',at=rt.now(),cell=cell,model=f'V{nheads}',seed=0,checkpoint=str(path),checkpoint_sha256=selection['sha256'],
        selection=selection,validation=val,test=test,source_hashes=source_hashes(),canonical_source_hashes=rt.source_hashes(),wall_seconds_current_attempt=time.perf_counter()-begin,
        completed_epoch_train_and_validation_wall_seconds=sum(r['seconds'] for r in history),final_evaluation_wall_seconds=val['seconds']+test['seconds'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),full_candidates=len(data.starts),channels=137,stage2=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--cell');ap.add_argument('--heads',type=int,choices=[0,5]);args=ap.parse_args()
    try:
        if args.smoke:smoke()
        else:train(args.cell,args.heads)
    except BaseException as exc:
        dump(OUT/'IMMEDIATE_ISSUE.json',dict(status='FAILED',at=rt.now(),cell=args.cell,heads=args.heads,error=repr(exc),traceback=traceback.format_exc()))
        raise
