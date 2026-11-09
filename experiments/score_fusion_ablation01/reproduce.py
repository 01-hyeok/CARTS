"""Reuse hash-bound whole-population evidence, plus current first/tail GPU gate.

The subsequent evaluator additionally checks E0 IDs/scores/distances in EVERY
new block before evaluating any E1-E4 method for that block.
"""
from bridge import *

def verify(d,f,prior,identity,dest):
    smoke=json.loads((dest/'GPU_SMOKE_GATE.json').read_text())
    assert smoke['status']=='PASS_FIRST_TAIL_GPU_SMOKE' and smoke['identity']==identity
    rows=[];training=f.support
    for split in ('val','test'):
        N=len(d.datasets[split]);sums={s:0. for s in ('FULL','P100')};units=0;file_hashes={}
        reference=f.reference(split) if training=='P100' else None
        for a in range(0,N,32):
            b=min(a+32,N);z=prior.batch(split,a,b);pool=d.cpu_pool[split][a:b]
            assert np.array_equal(pool,z['pool_ids'])
            for support in ('FULL','P100'):
                vn=variant_name(training,support);ids=z[vn+'__selected_ids'];ms=z[vn+'__selected_mse']
                assert ids.shape==(b-a,d.c,10) and np.isfinite(ms).all()
                assert np.all(np.diff(np.sort(ids,-1),axis=-1)!=0)
                if support=='P100':assert (ids[...,None]==pool[...,None,:]).any(-1).all()
                if support==training and training=='P100':assert np.array_equal(ids,reference['ids'][a:b])
                sums[support]+=float(ms.astype(np.float64).mean(-1).sum(dtype=np.float64))
            units+=(b-a)*d.c
            p=prior.base/split/('batch%05d.npz'%(a//32));file_hashes[str(p)]=sha(p)
        assert units==N*d.c
        rows.append(dict(split=split,queries=N,channels=d.c,units=units,retmse={s:v/units for s,v in sums.items()},
            full_population_source_hashes=file_hashes,ordered_native_ids='direct stored official P100 IDs' if training=='P100' else 'prior canonical whole-population gate'))
    actual=rows[-1]['retmse'][training]
    expected=float(f.row['canonical_retmse']) if training=='P100' else float(f.reference('test')['metrics']['retmse10'])
    assert np.isclose(actual,expected,atol=2e-6,rtol=2e-6),(actual,expected)
    g=dict(status='PASS_HASH_BOUND_ALL_QUERY_E0_AND_CURRENT_GPU_SMOKE',at=now(),identity=identity,rows=rows,
        actual_test_retmse=actual,official_test_retmse=expected,delta=actual-expected,
        evidence_kind='Prior whole-population replay with exact source/runtime identities, current direct canonical first/tail, and online per-block E0 gate',
        new_full_population_encoder_rerun=False,original_runtime='UNKNOWN',smoke_gate_sha256=sha(dest/'GPU_SMOKE_GATE.json'))
    dump(dest/'REPRODUCTION_GATE.json',g)
    return g
