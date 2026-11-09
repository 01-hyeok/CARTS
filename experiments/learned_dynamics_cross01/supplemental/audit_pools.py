"""Read-only verification of conditional existing PAA/SARAF pools."""
import sys,os,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/learned_dynamics_cross01'))
from experiment_common import load,sha,dump,OUT,now
sys.path.insert(0,str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'))
import runtime as rt
import numpy as np

def verify(cell,method):
    d=rt.CellData(cell,'cpu');folder=ROOT/'experiments/p100_preselection_comparison/results/pools'/cell/method
    m=json.loads((folder/'COMPLETE.json').read_text());assert not m['smoke']
    rows=[]
    for split in ('train','val','test'):
        p=folder/(split+'.pt');record=next(r for r in m['splits'] if r['split']==split)
        assert sha(p)==record['sha256'];z=load(p);ids=z['pool_idx_global'].numpy();q=z['query_start_idx'].numpy()
        assert ids.shape==(len(d.datasets[split]),d.c,100) and np.array_equal(q,d.datasets[split].starts)
        assert z['meta']['candidate_mask']=='raft' and z['meta']['preselection_method']==method
        for a in range(0,len(q),128):rt.validate_pool(ids[a:a+128],q[a:a+128],d.starts,d.h,d.h,split)
        rows.append(dict(split=split,path=str(p),sha256=record['sha256'],queries=len(q),channels=d.c,
                         temporal_calendar_normalizer=record['time_bonus_global_max'],eligible_unique=True))
    return dict(cell=cell,pool=method,status='PASS_REUSE_POOL_ONLY',splits=rows,
        past_score_formula='PAA16 cosine of last-value-centered history' if method=='paa16_cosine' else '0.9 mean-centered history cosine + 0.1 calendar bonus',
        calendar_normalization='existing split-wide maximum uses timestamps only, no target/feature-value fitting',
        existing_model_results_not_reused='Prior runs trained other V5 parameters; new scorer variants have not been evaluated there')

if __name__=='__main__':
    os.chdir(ROOT)
    rows=[verify(c,m) for c in ('ETTh1_96','ETTh1_720','Weather_96','Weather_720') for m in ('paa16_cosine','saraf_time_aligned')]
    result=dict(status='PASS',rows=rows,pipeline_source_sha256=sha(ROOT/'experiments/p100_preselection_comparison/pipeline.py'),at=now())
    dump(OUT/'supplemental/POOL_REUSE_GATE.json',result)
    print(json.dumps(result,indent=2))
