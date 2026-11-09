"""Preserve completed cells across verified, versioned runtime repairs.

Verify artifacts, blocks, full split gates, frozen sources and checkpoints;
a completion status string by itself never permits skipping work.
"""
import hashlib,json,shutil
from pathlib import Path
from io_utils import read,dump,sha,source_hashes,now

def release_key(hashes):
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()

def freeze_runtime(out,here):
    hashes=source_hashes();dest=Path(out)/'00_Audit/runtime_releases'/release_key(hashes)
    dest.mkdir(parents=True,exist_ok=True)
    for name,h in hashes.items():
        p=dest/name
        if not p.exists():shutil.copy2(Path(here)/name,p)
        assert sha(p)==h,('runtime snapshot mismatch',p)
    manifest=dest/'MANIFEST.json'
    if not manifest.exists():dump(manifest,dict(status='PASS',at=now(),hashes=hashes,immutable_source_snapshot=True))
    assert read(manifest)['hashes']==hashes
    return dest

def verified_completed_cell(out,cell):
    out=Path(out);base=out/'cells'/cell;gate=base/'CELL_ARTIFACT_GATE.json'
    if not gate.exists():return False
    g=read(gate)
    assert g['status']=='CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING'
    assert g['cell']==cell and g['v1_included'] and g['files']
    for name,h in g['files'].items():assert sha(base/name)==h,('changed cell artifact',name)
    agg=read(base/'AGGREGATION_GATE.json');sup=read(base/'CPU_SUPPLEMENT_GATE.json')
    assert agg['checkpoint_count']==6 and agg['core_full_population'] and agg['all_v1_present']
    assert sup['status']=='PASS' and sup['full_population'] and sup['v1_included_both_training_supports']
    for support in ('P100','FULL'):
        for model in ('V0','V1','V5'):
            b=out/'checkpoint_runs'/cell/(model+'_'+support);c=read(b/'CORE_COMPLETE.json');proof=c['identity']
            assert c['status']=='CORE_COMPLETE_NOT_ALL_PHASES' and c['files']
            assert c['query_counts']=={s:proof['data']['counts'][s] for s in ('val','test')}
            assert c['channels']==proof['data']['channels']==sup['channels']
            assert proof['checkpoint_sha256']==sha(c['source']['checkpoint'])
            version=out/'00_Audit/runtime_releases'/release_key(proof['source_hashes'])
            assert read(version/'MANIFEST.json')['hashes']==proof['source_hashes']
            for name,h in proof['source_hashes'].items():assert sha(version/name)==h
            rep=read(b/'REPRODUCTION_GATE.json');assert rep['status']=='PASS' and rep['identity']==proof
            assert len(rep['rows'])==2 and {r['split'] for r in rep['rows']}=={'val','test'}
            for row in rep['rows']:
                s=row['split'];n=proof['data']['counts'][s];ch=proof['data']['channels']
                assert row['ordered_mismatch_count']==0 and row['queries']==n and row['units']==n*ch
                sp=read(b/s/'SPLIT_GATE.json')
                assert sp['status']=='PASS' and sp['identity']==proof and sp['queries']==n and sp['channels']==ch and sp['units']==n*ch
            for name,h in c['files'].items():
                f=b/name;assert sha(f)==h,('changed query block',f)
                side=read(f.with_suffix('.json'));assert side['status']=='PASS' and side['identity']==proof and side['sha256']==h
                assert f.with_name(f.stem+'_ranking.json').exists()
    return True
