"""Versioned vector runtime reusing the canonical training/checkpoint protocol."""
from pathlib import Path
import inspect,sys,json,os,time
LOCAL=Path(__file__).resolve().parent
sys.path[:0]=[str(LOCAL.parent),str(LOCAL.parent/'vectorized03')]
import runtime as rt
from vector_ops import install_dropout,batch_backward,pool_all
import graphed_validation
CURRENT_DEST=None
VALIDATION_CALL=0


def source_hashes():
    result=rt.source_hashes()
    files=[LOCAL/'runtime_validation.py',LOCAL/'cached_validation.py',LOCAL/'graphed_validation.py',LOCAL.parent/'vectorized03/vector_ops.py']
    for p in files:result[str(p.relative_to(rt.ROOT))]=rt.sha256_file(p)
    return result


def evaluate(data,model,heads,split,detailed=False,dest=None,baselines=False):
    global VALIDATION_CALL
    if split=='val' and not detailed and dest is None and not baselines:
        rt.torch.cuda.synchronize();started=time.perf_counter()
        result=graphed_validation.evaluate(data,model,heads,split)
        rt.torch.cuda.synchronize();VALIDATION_CALL+=1
        if CURRENT_DEST is not None:
            rt.write_json(CURRENT_DEST/'validation_acceleration'/f'call{VALIDATION_CALL:02d}.json',dict(at=rt.now(),seconds=time.perf_counter()-started,**result['validation_acceleration']))
        return result
    return rt.evaluate(data,model,heads,split,detailed,dest,baselines)


def train_one(run_id):
    global CURRENT_DEST,VALIDATION_CALL
    row=next(r for r in json.loads((rt.HERE/'configs/run_manifest.json').read_text()) if r['run_id']==run_id)
    CURRENT_DEST=rt.ROOT/row['output_path'];VALIDATION_CALL=len(list((CURRENT_DEST/'validation_acceleration').glob('call*.json')))
    gate=json.loads((LOCAL/'VALIDATION_RUNTIME_GATES.json').read_text())
    assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
    assert '_seed0' in run_id and run_id.startswith('Solar_')
    # Reuse canonical loop/checkpoint/evaluation plumbing, replace only the
    # exact audited channel loop and version-specific gate/provenance fields.
    source=inspect.getsource(rt.train_one)
    source=source.replace("HERE/'reports/RUNTIME_GATES.json'","LOCAL/'VALIDATION_RUNTIME_GATES.json'")
    old="""            for c in range(data.c):
                ids=data.pool('train',starts,c)
                sc=scores(model,heads,x,data,c,ids)
                loss,_,_=kl(sc,data.values(ids,c),x[:,-1,c],y[:,:,c])
                assert torch.isfinite(loss),f'nonfinite loss {run_id} {epoch} {bi} {c}'
                (loss/data.c).backward();bl+=float(loss.detach())/data.c"""
    assert source.count(old)==1
    new="""            ids_all=pool_all(data,'train',starts)
            rng_before=torch.cuda.get_rng_state()
            while True:
                oom_message=None
                try:
                    bl=batch_backward(rt,data,model,heads,dropout,x,y,ids_all,channel_chunk)
                except torch.cuda.OutOfMemoryError as error:
                    oom_message=str(error)
                if oom_message is None:break
                # Leave exception scope before freeing tensors; no partial update.
                opt.zero_grad(set_to_none=True);dropout.mask=None
                torch.cuda.empty_cache();torch.cuda.set_rng_state(rng_before)
                assert channel_chunk>1,'Even single-channel retry cannot fit'
                channel_chunk=max(1,channel_chunk//2)
                write_json(dest/f'OOM_epoch{epoch}_batch{bi}_chunk{channel_chunk}.json',dict(error=oom_message,epoch=epoch,batch=bi,channel_chunk=channel_chunk,at=now()))"""
    source=source.replace(old,new)
    source=source.replace("    loader=data.loader('train',32,seed)","    dropout=install_dropout(model);channel_chunk=137\n    loader=data.loader('train',32,seed)")
    source=source.replace('logical_batch=32,channel_chunk=1,candidate_encode_chunk=3200,',
        "logical_batch=32,channel_chunk=137,candidate_encode_chunk=438400,implementation='vectorized03_train_validation04',evaluation_implementation='nondetailed validation: exact candidate cache + canonical B32 CUDA graph; detailed validation/test: original canonical',dropout_rng='legacy per-channel query/key mask order',numerical_policy='FP32 GEMM/reduction differs; not bitwise trajectory-equivalent',")
    env=dict(rt.__dict__);env.update(LOCAL=LOCAL,rt=rt,source_hashes=source_hashes,evaluate=evaluate,
        install_dropout=install_dropout,batch_backward=batch_backward,pool_all=pool_all)
    exec(compile(source,str(LOCAL/'runtime_validation.py')+':canonical_train_one','exec'),env)
    return env['train_one'](run_id)


if __name__=='__main__':
    assert len(sys.argv)==3 and sys.argv[1]=='train'
    train_one(sys.argv[2])
