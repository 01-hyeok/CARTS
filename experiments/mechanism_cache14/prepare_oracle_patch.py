"""Generate a review-only, analyze-body-only patch. Never edits source worker."""
import argparse
import ast
import difflib
from pathlib import Path
from scoped_gate import fingerprint


def prepare(source):
    old=source
    start=source.index('def analyze(')
    end=source.index('\ndef main(',start)
    body=source[start:end]
    body=body.replace("    g=read(out/'REPRODUCTION_GATE.json')", "    from oracle_reuse import build_cached_evaluate\n    import gpu_metrics as canonical_metrics\n    evaluate_reused=build_cached_evaluate(canonical_metrics)\n    g=read(out/'REPRODUCTION_GATE.json')",1)
    anchor="            torch.cuda.synchronize();phase_wall['retrieval_scores']+=time.perf_counter()-tick"
    assert body.count(anchor)==1
    body=body.replace(anchor,"""            # Same canonical Oracle expression, once per support/block.
            oracle_full=topk(-d.masked_fill(~mask,float('inf')),100)
            pooled_distance=d.gather(-1,pool)
            pooled_mask=torch.ones_like(pool,dtype=torch.bool)
            oracle_pool=topk(-pooled_distance.masked_fill(~pooled_mask,float('inf')),100)
            if not f.full_trained:assert torch.equal(ni,pool)
"""+anchor)
    replacements={
      'ev=evaluate(f.data,x,y,dist,score,m,ids)':"ev=evaluate_reused(f.data,x,y,dist,score,m,ids,cached_oracle=oracle_full if name=='FULL' else oracle_pool)",
      "he=evaluate(f.data,x,y,dist,hs[:,:,h].masked_fill(~m,-float('inf')),m,ids)":"he=evaluate_reused(f.data,x,y,dist,hs[:,:,h].masked_fill(~m,-float('inf')),m,ids,cached_oracle=oracle_full if name=='FULL' else oracle_pool)",
      "union_ev=evaluate(f.data,x,y,dist,score.masked_fill(~hu['union_mask'],-float('inf')),m,ids)":"union_ev=evaluate_reused(f.data,x,y,dist,score.masked_fill(~hu['union_mask'],-float('inf')),m,ids,cached_oracle=oracle_full if name=='FULL' else oracle_pool)",
      "ue=evaluate(f.data,x,y,dist,union_oracle,m,ids)":"ue=evaluate_reused(f.data,x,y,dist,union_oracle,m,ids,cached_oracle=oracle_full if name=='FULL' else oracle_pool)",
      "loo.append(evaluate(f.data,x,y,dist,p,m,ids)['metrics'])":"loo.append(evaluate_reused(f.data,x,y,dist,p,m,ids,cached_oracle=oracle_full if name=='FULL' else oracle_pool)['metrics'])"}
    for before,after in replacements.items():
        assert body.count(before)==1,before
        body=body.replace(before,after)
    new=source[:start]+body+source[end:]
    ast.parse(new)
    assert fingerprint(old,'worker.py')==fingerprint(new,'worker.py')
    return new,''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/worker.py',tofile='prepared/worker.py'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    new,diff=prepare(Path(a.source).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'worker_prepared.py').write_text(new);(out/'oracle_reuse.patch').write_text(diff)
