"""CPU D1/D2 additions from verified stored Mechanism02 V5 blocks.

This deliberately stops at missing native head arrays and reports missing
rank/probabilities; these are not silently inferred from different variants.
No model inference, GPU context, original file write or process manipulation.
"""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from .fusion import analyze_unit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(base,out,split='test',variant='FULL'):
    base,out=Path(base),Path(out)
    proof=json.loads((base/'REPRODUCTION_GATE.json').read_text())['identity']
    gate=json.loads((base/split/'SPLIT_GATE.json').read_text())
    assert gate['status']=='PASS' and gate['identity']==proof
    assert not out.exists(),'new output required; old diagnostic artifacts preserved'
    blocks=sorted((base/split).glob('batch*.npz'));assert blocks
    # Complete validation before creating an output, including required arrays.
    needed=['head_ids','head_selected_mse','selected_ids','selected_mse','oracle_ids','oracle100_mse']
    starts=[];fingerprints={}
    for p in blocks:
        side=json.loads(p.with_suffix('.json').read_text())
        assert side['status']=='PASS' and side['identity']==proof and sha(p)==side['sha256']
        with np.load(p) as z:
            missing=[variant+'__'+k for k in needed if variant+'__'+k not in z]
            if missing:raise RuntimeError('MISSING native/exact arrays; minimal inference required: '+str(missing))
            assert z[variant+'__head_ids'].shape[1:]==(gate['channels'],5,10)
            starts.extend(z['query_start'].tolist())
        fingerprints[p.name]=side['sha256']
    assert len(starts)==gate['queries'] and len(set(starts))==len(starts) and np.all(np.diff(starts)>0)
    out.mkdir(parents=True)
    totals={k:0. for k in ['r_mix','r_oracle_C','r_oracle','fusion_regret','discovery_regret','lost_oracle_count','union_size']};n=0
    with gzip.open(out/'QUERY_CHANNEL_DIAGNOSTICS.jsonl.gz','wt',encoding='utf8') as f:
        for p in blocks:
            with np.load(p) as z:
                for q,start in enumerate(z['query_start']):
                    for c in range(gate['channels']):
                        get=lambda k:z[variant+'__'+k][q,c]
                        r=analyze_unit(get('head_ids'),get('head_selected_mse').astype(np.float64),get('selected_ids'),get('selected_mse').astype(np.float64),
                                       get('oracle_ids'),get('oracle100_mse')[:10].astype(np.float64))
                        r.update(query_start=int(start),channel=c,variant=variant)
                        f.write(json.dumps(r,allow_nan=False)+'\n')
                        for k in totals:totals[k]+=r[k]
                        n+=1
    assert n==gate['queries']*gate['channels']
    result=dict(status='PARTIAL_CPU_SET_DIAGNOSTICS_PASS',units=n,queries=gate['queries'],channels=gate['channels'],variant=variant,
                means={k:v/n for k,v in totals.items()},source_identity=proof,source_block_sha256=fingerprints,
                pending=['lost mixture rank','lost candidate all-head score/probability','native head arrays if absent'],gpu_used=False)
    (out/'CPU_SET_GATE.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result['means']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--split',default='test');p.add_argument('--variant',default='FULL');a=p.parse_args();run(a.base,a.out,a.split,a.variant)
