"""Past-only query normalization/projection diagnosis after score-kernel audit."""
import sys,json,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cublas_recovery02'))
import probe as previous
import adapter,packed_batch,torch
from source_contract import transform
OUT=previous.ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/extensions_d5_d6/cublas_recovery03'

@torch.no_grad()
def main():
    previous.bridge.start_gpu();d=previous.bridge.Data('Weather_336')
    row=next(r for r in previous.bridge.inventory(d.cell) if r['architecture']=='V1' and r['training_support']=='P100')
    f=previous.bridge.Frozen(d,row);f.precision();f.batch.build_keys()
    a,b=4928,4935;x,_,_=d.query('val',a,b)
    qref=previous.canonical_query(f,'val',a,b);keys=previous.canonical_keys(f)
    assert torch.equal(keys,f.batch.keys)
    ref=previous.reference_scores(f,qref,keys)
    mask=d.mask(d.datasets['val'].starts[a:b])[:,None].expand(7,21,-1)
    ids=previous.bridge.topk(ref[:,:,0].masked_fill(~mask,-float('inf')),500)
    variants={};chosen=None
    for mode in ['flat_exact_norm','cuda12_lt_exact_norm','cuda11_lt_exact_norm','cuda11_blas_exact_norm']:
        record={}
        try:
            ns=dict(packed_batch.__dict__);exec(compile(transform(adapter._p100_short,mode),str(HERE/'source_contract.py')+'::'+mode,'exec'),ns)
            q=ns['short_query'](f.batch,x);score=previous.Split64()(q[:,:,0].permute(1,0,2).contiguous(),keys).permute(1,0,2)[:,:,None]
            record=dict(query_exact=torch.equal(q,qref),query_max_delta=float((q-qref).abs().max()),scores_exact=torch.equal(score,ref),
                        score_max_delta=float((score-ref).abs().max()),ordered_top500_exact=torch.equal(previous.bridge.topk(score[:,:,0].masked_fill(~mask,-float('inf')),500),ids))
            if chosen is None and record['query_exact'] and record['scores_exact'] and record['ordered_top500_exact']:chosen=mode
        except Exception as e:record={'error':repr(e),'traceback':traceback.format_exc()}
        variants[mode]=record;print(mode,json.dumps(record),flush=True)
    previous.bridge.dump(OUT/'QUERY_PROBE.json',dict(status='PASS_EXACT_QUERY_AND_SCORE_FOUND' if chosen else 'BLOCKED_QUERY_STILL_DIFFERS',chosen=chosen,variants=variants,
        cell=d.cell,model='V1_P100',batch=7,channels=21,bank_count=36216,checkpoint_sha256=row['checkpoint_sha256'],production_activated=False,
        all_channels_single_batch=True,whole_sample_replay_pending=True,source_hashes={p.name:previous.bridge.sha(p) for p in HERE.glob('*.py')}))
    if not chosen:raise RuntimeError('Exact query still unresolved; no production fallback')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        previous.bridge.dump(OUT/'QUERY_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),production_activated=False));raise
