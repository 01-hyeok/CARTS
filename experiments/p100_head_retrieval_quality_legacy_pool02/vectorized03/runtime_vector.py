"""Versioned vector runtime reusing the canonical training/checkpoint protocol."""
from pathlib import Path
import inspect,sys,json,os
LOCAL=Path(__file__).resolve().parent
sys.path.insert(0,str(LOCAL.parent))
import runtime as rt
from vector_ops import install_dropout,batch_backward,pool_all
from vector_eval import evaluate as vector_evaluate


def source_hashes():
    result=rt.source_hashes()
    for name in ('runtime_vector.py','vector_ops.py','vector_eval.py'):
        p=LOCAL/name;result[str(p.relative_to(rt.ROOT))]=rt.sha256_file(p)
    return result


def evaluate(data,model,heads,split,detailed=False,dest=None,baselines=False):
    # Batched eval GEMM changed a near-tied selected candidate in the gate.
    # Keep the original score/reduction shapes for checkpoint selection and test.
    return rt.evaluate(data,model,heads,split,detailed,dest,baselines)


def train_one(run_id):
    gate=json.loads((LOCAL/'VECTOR_GATES.json').read_text())
    assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
    assert '_seed0' in run_id and run_id.startswith('Solar_')
    # Reuse canonical loop/checkpoint/evaluation plumbing, replace only the
    # exact audited channel loop and version-specific gate/provenance fields.
    source=inspect.getsource(rt.train_one)
    source=source.replace("HERE/'reports/RUNTIME_GATES.json'","LOCAL/'VECTOR_GATES.json'")
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
        "logical_batch=32,channel_chunk=137,candidate_encode_chunk=438400,implementation='vectorized03',evaluation_implementation='canonical channelwise; vector eval rejected for near-tie selection mismatch',dropout_rng='legacy per-channel query/key mask order',numerical_policy='FP32 GEMM/reduction differs; not bitwise trajectory-equivalent',")
    env=dict(rt.__dict__);env.update(LOCAL=LOCAL,rt=rt,source_hashes=source_hashes,evaluate=evaluate,
        install_dropout=install_dropout,batch_backward=batch_backward,pool_all=pool_all)
    exec(compile(source,str(LOCAL/'runtime_vector.py')+':canonical_train_one','exec'),env)
    return env['train_one'](run_id)


if __name__=='__main__':
    assert len(sys.argv)==3 and sys.argv[1]=='train'
    train_one(sys.argv[2])
