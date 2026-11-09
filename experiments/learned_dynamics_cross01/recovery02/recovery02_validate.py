"""Fail-closed full coverage, selection, artifact and provenance validation."""
from recovery_runtime import *
import numpy as np
METHODS=['S0','D0','C0']+ARMS+['Oracle10','Random','LastRepeat']
class Incomplete(Exception):pass
def targets():
    for s in ('P100','FULL'):
        for c in CELLS:yield native_path(s,c),dict(mode='native',cell=c,training=s,support=s,pool=None)
    for c in ('ETTh1_96','ETTh1_720','Weather_96','Weather_720'):
        for p in ('paa16_cosine','saraf_time_aligned'):yield OUT/'supplemental/pool'/p/c,dict(mode='pool',cell=c,training='P100',support='P100',pool=p)
    for c in CELLS:
        for s in ('P100','FULL'):
            t='FULL' if s=='P100' else 'P100'
            yield OUT/'supplemental/cross'/('V5_'+s+'_to_'+t)/c,dict(mode='cross',cell=c,training=s,support=t,pool=None)
def coverage(actual,expected):
    assert actual.ndim==1 and np.array_equal(actual,expected),'Missing, duplicate, reordered or wrong queries'
    assert len(np.unique(actual))==len(actual),'Duplicate query'
def validate_job(path,label,full=True):
    if not (path/'COMPLETE.json').exists():raise Incomplete(str(path))
    done=read(path/'COMPLETE.json');expected='PASS_ALL_QUERY_TRAIN_VAL_TEST' if label['mode']=='native' else 'PASS_ALL_QUERY_SUPPLEMENT'
    assert done['status']==expected and done['cell']==label['cell']
    if label['mode']=='native':assert done['support']==label['support']
    else:assert done['mode']==label['mode'] and done['training_support']==label['training'] and done['evaluation_support']==label['support'] and done['pool']==label['pool']
    selection=read(path/'SELECTION.json');assert set(selection['arms'])==set(ARMS)
    assert selection['status']==('REUSED_NATIVE_SELECTION_NO_REFIT' if label['mode']=='cross' else 'FROZEN_BEFORE_FINAL_TEST')
    rows=read(path/'FULL_RESULTS.json');assert len(rows)==24
    assert {(r['split'],r['method']) for r in rows}=={(s,m) for s in ('val','test') for m in METHODS}
    cp={};raw={}
    source=native_path(label['training'],label['cell']) if label['mode']=='cross' else path
    if label['mode']=='cross':
        assert selection['arms']==read(source/'SELECTION.json')['arms']
        assert read(path/'CROSS_GATE.json')['status']=='PASS_NO_PARAMETER_UPDATE_OR_CALIBRATION_REFIT'
    hist=read(source/'TRAIN_HISTORY.json')
    for n in ARMS:
        chosen=selection['arms'][n]['best_epoch'];candidate=checkpoint_path(source,n,chosen)
        cp[n]=dict(path=str(candidate),sha256=sha(candidate),epoch=chosen)
        z=load(candidate);assert z['epoch']==chosen and z['state']['best_epoch']==chosen and z['state']['best']==selection['arms'][n]['best']
        h=[r for r in hist if r['arm']==n];best=min(h,key=lambda r:(r['val_retmse'],r['epoch']))
        assert best['epoch']==chosen and best['val_retmse']==selection['arms'][n]['best']
        if label['mode']=='cross':
            ref=next(x for x in selection['checkpoints'] if x['arm']==n)
            assert ref['path']==str(candidate) and ref['sha256']==cp[n]['sha256'] and ref['selected_epoch']==chosen
    for split in ('val','test'):
        gate=read(path/('FINAL_'+split.upper()+'_GATE.json'))
        assert gate['status']=='PASS_ALL_QUERY' and gate['queries']==done['queries'][split] and gate['channels']==done['channels']
        for r in [v for v in rows if v['split']==split]:
            assert r['queries']==done['queries'][split] and r['channels']==done['channels'] and r['seed']==0
            assert r['dataset']+'_'+str(r['horizon'])==label['cell'] and r['support']==label['support']
            assert all(np.isfinite(r[k]) for k in ('retmse','agg_mse','agg_mae'))
            if r['method'] in ARMS:assert r['scorer_epoch']==selection['arms'][r['method']]['best_epoch']
        if full:
            starts=load(ORIGINAL/'features'/label['cell']/(split+'.pt'))['starts'].numpy()
            blocks=sorted((path/'final_raw'/split).glob('*.npz'));assert len(blocks)==(len(starts)+31)//32
            found=[]
            for i,b in enumerate(blocks):
                assert b.name=='batch%05d.npz'%i
                meta=read(b.with_suffix('.json'));digest=sha(b);assert digest==meta['sha256'];raw[str(b)]=digest
                with np.load(b,allow_pickle=False) as z:
                    qs=z['query_start'];coverage(qs,starts[i*32:min(i*32+32,len(starts))]);found.extend(qs.tolist())
                    assert meta['queries']==len(qs) and meta['channels']==done['channels']
                    for m in METHODS:
                        for metric in ('retmse','agg_mse','agg_mae'):
                            a=z[m+'__'+metric];assert a.shape==(len(qs),done['channels']) and np.isfinite(a).all()
                        if m!='LastRepeat':
                            ids=z[m+'__selected_ids'];assert ids.shape==(len(qs),done['channels'],10)
                            assert np.all(np.diff(np.sort(ids,axis=-1),axis=-1)>0)
            coverage(np.asarray(found),starts)
    return dict(path=str(path),label=label,status=done['status'],completion_path=str(path/'COMPLETE.json'),completion_sha256=sha(path/'COMPLETE.json'),selection_path=str(path/'SELECTION.json'),selection_sha256=sha(path/'SELECTION.json'),checkpoints=cp,raw_directories={s:str(path/'final_raw'/s) for s in ('val','test')},raw_hashes=raw,identity=read(path/'IDENTITY.json'),identity_sha256=sha(path/'IDENTITY.json'),queries=done['queries'],channels=done['channels'])
def validate_all(full=True):
    jobs=list(targets());assert len(jobs)==40 and len({str(p) for p,l in jobs})==40
    missing=[str(p) for p,l in jobs if not (p/'COMPLETE.json').exists()]
    if missing:raise Incomplete(json.dumps(dict(missing=missing)))
    seal_check();records=[validate_job(p,l,full) for p,l in jobs]
    preserved()
    return dict(status='PASS_16_8_16_ARTIFACTS',native=16,alternative_pools=8,cross_support=16,jobs=records,input_manifest_sha256=sha(OUT/'REUSE_MANIFEST.json'),source_seal_sha256=sha(OUT/'SOURCE_SEAL.json'),at=now())
