"""Evaluate frozen selected calibrators on the other support. No refit/reselection."""
import argparse,time,json,os
from native_runtime import *

def run(cell,training,phase):
    evaluation='FULL' if training=='P100' else 'P100';source=OUT/'runs'/training/cell/phase
    done=json.loads((source/'COMPLETE.json').read_text());assert done['status']=='PASS_ALL_QUERY_'+training
    selection=json.loads((source/'SELECTION.json').read_text());assert selection['status']=='FROZEN_BEFORE_TEST'
    protocol=json.loads((source/'PROTOCOL.json').read_text());names=['fixed','cosine'] if phase=='P1' else ['cosine','dynamics','cross','both','shuffled']
    dest=OUT/'cross_support'/('%s_to_%s'%(training,evaluation))/cell/phase;assert not dest.exists(),'Preserve previous cross-support outputs'
    ctx=NativeContext(cell,training,evaluation);d=ctx.d;source_identity=protocol.get('identity',protocol.get('source_identity'))
    assert source_identity['checkpoint']['checkpoint_sha256']==ctx.identity['checkpoint']['checkpoint_sha256']
    assert source_identity['data']==ctx.identity['data'] and source_identity['pool_hashes']==ctx.identity['pool_hashes']
    arms={};hashes={};before={};dest.mkdir(parents=True);started=time.time()
    for name in names:
        selected=source/'checkpoints'/name/('epoch%02d.pt'%selection['best'][name]['epoch']);arm=new_model(name,phase,d.device);arm.load_state_dict(load(selected)['model']);arm.eval()
        for p in arm.parameters():p.requires_grad_(False)
        arms[name]=arm;hashes[name]=sha(selected);before[name]={k:v.detach().cpu().clone() for k,v in arm.state_dict().items()}
    gamma=json.loads((ev.OUT/'checkpoints'/cell/('V5_'+training)/'GAMMA_SELECTION.json').read_text())['chosen_gamma'];summary=[]
    old_cross=OLD/'supplemental/cross'/('V5_'+training+'_to_'+evaluation)/cell
    assert (old_cross/'COMPLETE.json').exists(),'Legacy same-checkpoint cross-support reference missing'
    baseline_rows=json.loads((old_cross/'FULL_RESULTS.json').read_text())
    with torch.no_grad():
        for split in ('val','test'):
            totals={};units=0;begin=time.time()
            for a in range(0,len(d.datasets[split]),32):
                batch=ctx.batch(split,a);parts={}
                for j in range(batch['b']-a):
                    mask=batch['mask'][j:j+1];distance=batch['distance'][j:j+1].double();ids=batch['ids'][j:j+1];p=batch['p'][j:j+1]
                    opos=stable_topk((-distance).masked_fill(~mask,-torch.inf));oracle=ids.gather(-1,opos);om=distance.gather(-1,opos);bpos=stable_topk(p.masked_fill(~mask,-torch.inf));bret=distance.gather(-1,bpos).mean(-1)
                    scores={n:ctx.ranking(arm,n,phase,batch,split,(j,j+1)) for n,arm in arms.items()}
                    scores.update(S0=p.masked_fill(~mask,-torch.inf),E1=ev.score(batch['g'][j:j+1],mask,'E1',None),E4=ev.score(batch['g'][j:j+1],mask,'E4',gamma),Oracle10=(-distance).masked_fill(~mask,-torch.inf))
                    units+=d.c
                    for name,score in scores.items():
                        pos=stable_topk(score);chosen=ids.gather(-1,pos);sm=distance.gather(-1,pos);ret=sm.mean(-1);agg,mae=ctx.prediction_metrics(split,[a+j],chosen)
                        discount=1/torch.log2(torch.arange(2,12,device=d.device,dtype=torch.float64))
                        metrics=dict(retmse=ret,agg_mse=agg,agg_mae=mae,oracle_gap=ret-om.mean(-1),oracle_recall=(chosen[...,None]==oracle[...,None,:]).any(-1).double().mean(-1),ndcg10=((1/(1+sm))*discount).sum(-1)/((1/(1+om))*discount).sum(-1),best_of10=sm.min(-1).values,delta_retmse_vs_s0=ret-bret)
                        parts.setdefault(name+'__selected_ids',[]).append(chosen.cpu().numpy());parts.setdefault(name+'__selected_mse',[]).append(sm.cpu().numpy())
                        for metric,value in metrics.items():parts.setdefault(name+'__'+metric,[]).append(value.cpu().numpy());totals[(name,metric)]=totals.get((name,metric),0.)+float(value.sum())
                folder=dest/'final_raw'/split;folder.mkdir(parents=True,exist_ok=True)
                with (folder/('batch%05d.npz'%(a//32))).open('xb') as f:np.savez(f,query_start=d.datasets[split].starts[a:batch['b']],**{k:np.concatenate(v) for k,v in parts.items()})
                if a%1024==0:dump(dest/'CURRENT.json',dict(phase='FROZEN_CROSS_SUPPORT',split=split,end=batch['b'],total=len(d.datasets[split]),at=time.time()));print('CROSS',cell,training,evaluation,phase,split,batch['b'],flush=True)
            old=next(r for r in baseline_rows if r['split']==split and r['method']=='S0')
            assert np.isclose(totals[('S0','retmse')]/units,old['retmse'],atol=2e-6,rtol=2e-6)
            for (name,metric),value in totals.items():summary.append(dict(cell=cell,split=split,method=name,metric=metric,value=value/units,training_support=training,evaluation_support=evaluation,selected_epoch=selection['best'][name]['epoch'] if name in names else None))
            dump(dest/(split.upper()+'_GATE.json'),dict(status='PASS_ALL_QUERY',queries=len(d.datasets[split]),channels=d.c,seconds=time.time()-begin,full_candidate_reduction=False))
    for name,arm in arms.items():
        assert all(torch.equal(v,arm.state_dict()[k].detach().cpu()) for k,v in before[name].items()),'Cross-support changed fitted model or normalization'
        selected=source/'checkpoints'/name/('epoch%02d.pt'%selection['best'][name]['epoch']);assert hashes[name]==sha(selected)
    dump(dest/'METRICS.json',summary)
    dump(dest/'COMPLETE.json',dict(status='PASS_FROZEN_CROSS_SUPPORT',cell=cell,phase=phase,training_support=training,evaluation_support=evaluation,selection_reused=True,refit=False,optimizer_created=False,checkpoint_hashes=hashes,source_identity=source_identity,source_selection_sha256=sha(source/'SELECTION.json'),seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),shuffle_extension='P100-trained: original fixed Train timestamp/pool donors, candidate-slot permutation cycles modulo100 for Full; Full-trained: fixed Train pair permutation using actual global candidate IDs',exploratory_test=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cell');parser.add_argument('training',choices=['P100','FULL']);parser.add_argument('phase',choices=['P1','P2']);args=parser.parse_args();os.chdir(ROOT)
    try:run(args.cell,args.training,args.phase)
    except BaseException as e:
        import traceback
        dump(OUT/'full_extension01'/('CROSS_FAILURE_%s_%s_%s_%d.json'%(args.cell,args.training,args.phase,int(time.time()))),dict(error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
