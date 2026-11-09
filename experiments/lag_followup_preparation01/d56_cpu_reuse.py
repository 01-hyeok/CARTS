"""D5/D6 verified D4 reuse. Missing populations/metrics never become COMPLETE."""
import csv,datetime,fcntl,gzip,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
from d56_metrics import *

ROOT=Path('/data/pjh_workspace/CARTS')
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'
OUT=BASE/'extensions_d5_d6'
ORDER=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','ETTh1_192','Weather_192','Weather_336','Weather_720']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    t=p.with_suffix('.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    t.replace(p)
def cat(rows,key):return np.concatenate([r[key] for r in rows],0)

def load_d4(src):
    proof=json.loads((src/'PROVENANCE.json').read_text());gate=json.loads((src/'COMPONENT_GATE.json').read_text())
    assert gate['status']=='D4_FEATURES_AND_RERANK_COMPLETE_STATISTICS_PENDING'
    assert gate['all_channels_single_batch'] and gate['original_checkpoint_unchanged']
    cp=proof['identity']['checkpoint'];assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
    assert proof['training_support']==proof['evaluation_support']==cp['training_support']
    files={};splits={}
    for split in ['val','test']:
        rows=[]
        for p in sorted((src/split).glob('query*.npz')):
            side=json.loads(p.with_suffix('.json').read_text());assert side['identity']==proof
            h=sha(p);assert h==side['sha256'] and side['canonical_native_top10_pass'] and side['base_retrieval_entire_support']
            assert side['rerank_pool_future_blind'] and side['channels']==gate['channels']
            with np.load(p) as z:r={k:z[k].copy() for k in z.files}
            assert int(r['query_index'][0])==side['query_index']
            ids=r['global_ids'];R=int(r['rerank_count'][0]);assert ids.shape[:2]==(1,gate['channels'])
            assert all(len(np.unique(v))==R for v in ids[0,:,:R])
            assert ids.min()>=0 and ids.max()<gate['full_bank_count']
            assert all(np.isfinite(r[k]).all() for k in ['mse','base_score','gain','best_lag','entropy','margin'])
            rows.append(r);files[str(p.relative_to(BASE))]=h
        assert len(rows)==gate['queries'][split]==32
        q=cat(rows,'query_index');assert len(np.unique(q))==len(q) and np.all(np.diff(q)>0)
        starts=cat(rows,'query_start');assert np.all(np.diff(starts)>0)
        splits[split]=rows
    return proof,gate,files,splits

def one_case(cell,support):
    src=BASE/'D4'/('V5_'+cell+'_'+support);dest=OUT/'cpu_reuse01'/(cell+'_'+support)
    if not (src/'COMPONENT_GATE.json').exists():return dict(cell=cell,training_support=support,evaluation_support=support,model='V5',status='WAITING_FOR_DEPENDENCY',reason='D4 native V5 component gate absent')
    prior=dest/'CPU_REUSE_GATE.json'
    if prior.exists():
        g=json.loads(prior.read_text())
        for p,h in g['outputs'].items():assert sha(dest/p)==h
        for p,h in g['source_npz_sha256'].items():assert sha(BASE/p)==h
        return g['availability']
    begin=time.perf_counter();proof,gate,files,splits=load_d4(src);dest.mkdir(parents=True,exist_ok=True)
    dump(dest/'SOURCE_PROVENANCE.json',dict(source=str(src),provenance=proof,component_gate_sha256=sha(src/'COMPONENT_GATE.json'),code_sha256=sha(__file__),metric_sha256=sha(Path(__file__).with_name('d56_metrics.py'))))
    group_rows=[];head_rows=[];summary={};hist={};group_hist={};d6rows=[];selections=[];d6units=[]
    qcpath=dest/'D5_QUERY_CHANNEL_GROUPS.jsonl.gz';qctmp=qcpath.with_suffix('.tmp')
    with gzip.open(qctmp,'wt') as stream:
        for split,rows in splits.items():
            R=int(rows[0]['rerank_count'][0])
            for row in rows:
                for c in range(gate['channels']):
                    ids=row['global_ids'][0,c];o=ids[R:R+10];heads=ids[R+10:R+60].reshape(5,10)
                    unique=np.array([i not in ids[:k] for k,i in enumerate(ids)])
                    threshold=float(row['oracle_mse'][0,c,-1]);useful=row['mse'][0,c]<=threshold+max(1e-8,NEAR_FRACTION*threshold)
                    masks={'Oracle10':np.isin(ids,o)&unique,'Random10':np.arange(len(ids))>=len(ids)-10,
                        'NearOracle_cached':useful&unique,
                        **{k:row['group_'+k][0,c] for k in ['correct_oracle','missed_oracle','false_selected']}}
                    for h in range(5):
                        other=np.delete(heads,h,axis=0).ravel();hs=np.isin(ids,heads[h])&unique
                        masks['H%d_all'%(h+1)]=hs;masks['H%d_oracle'%(h+1)]=hs&np.isin(ids,o)
                        masks['H%d_near'%(h+1)]=hs&useful
                        masks['H%d_unique_useful'%(h+1)]=hs&useful&~np.isin(ids,other)
                    for name,mask in masks.items():
                        s=lag_summary(*[row[k][0,c][mask] for k in ['best_lag','gain','margin','entropy','candidate_std']])
                        base=dict(cell=cell,training_support=support,evaluation_support=support,model='V5',split=split,query_index=int(row['query_index'][0]),query_start=int(row['query_start'][0]),channel=c,group=name,
                            coverage='cached union only; not exhaustive NearOracle' if name=='NearOracle_cached' else 'complete group on 32 fixed timestamps/all channels')
                        stream.write(json.dumps(dict(**base,**s),allow_nan=False)+'\n')
                        key=(split,name);summary.setdefault(key,[]).append(s)
                        hh=group_hist.setdefault(key,{})
                        for k,v in s.get('best_lag_histogram',{}).items():hh[k]=hh.get(k,0)+v
            for (sp,name),vals in list(summary.items()):
                if sp!=split:continue
                valid=[v for v in vals if v['count']];numerical=[k for k,v in valid[0].items() if isinstance(v,(int,float))] if valid else []
                r=dict(cell=cell,training_support=support,evaluation_support=support,model='V5',split=sp,group=name,queries=len(rows),channels=gate['channels'],query_channel_groups=len(vals),nonempty_groups=len(valid),
                       metric_aggregation='equal mean over nonempty query-channel groups; empty count separately',
                       best_lag_histogram=json.dumps(group_hist[(sp,name)],sort_keys=True),
                       near_tied_candidate_definition='top1-top2 lag cosine margin<=0.01; NOT fraction of lag alternatives',
                       near_tie_lag_fraction_status='MISSING_RAW_LAG_SCORE_REQUIRED',
                       coverage='cached candidate union only' if name=='NearOracle_cached' else 'complete stated candidate group; sampled timestamps')
                for k in numerical:r[k]=float(np.mean([v[k] for v in valid]))
                (head_rows if name.startswith('H') else group_rows).append(r)
    qctmp.replace(qcpath)
    stability=[]
    for name in sorted({k[1] for k in group_hist}):
        stability.append(dict(cell=cell,support=support,model='V5',group=name,comparison='validation-test',**hist_distance(group_hist.get(('val',name),{}),group_hist.get(('test',name),{}))))
    for split in ['val','test']:
        for suffix in ['all','oracle','near','unique_useful']:
            for a in range(1,6):
                for b in range(a+1,6):stability.append(dict(cell=cell,support=support,model='V5',group=suffix,comparison='H%d-H%d'%(a,b),split=split,**hist_distance(group_hist[(split,'H%d_%s'%(a,suffix))],group_hist[(split,'H%d_%s'%(b,suffix))])))
    val,test=splits['val'],splits['test'];Rmax=int(val[0]['rerank_count'][0]);rs=[x for x in [10,20,50,100,200,500] if x<=Rmax]
    curves={}
    for R in rs:
        selected={}
        for kind in ['ShiftOnly','ShiftRecent']:
            selected[kind]=select_validation(*[cat(val,k)[...,:R] for k in ['base_score','gain','recent_difference','mse']],kind)
            selections.append(dict(R=R,method=kind,**selected[kind]))
        for name in ['Original','ShiftOnly','ShiftRecent']:
            values=[];deltas=np.empty((len(test),gate['channels']))
            for q,row in enumerate(test):
                rcount=int(row['rerank_count'][0]);ids=row['global_ids'][0];o=ids[:,rcount:rcount+10]
                for c in range(gate['channels']):
                    choice=dict(alpha=0.,beta=0.,score_scale=1.,recent_scale=1.) if name=='Original' else dict(**{k:selected[name]['selected'][k] for k in ['alpha','beta']},score_scale=selected[name]['score_scale'][c],recent_scale=selected[name]['recent_scale'][c])
                    threshold=float(row['oracle_mse'][0,c,-1]);near_count=int((row['mse'][0,c,:100]<=threshold+max(1e-8,NEAR_FRACTION*threshold)).sum()) if support=='P100' else None
                    r=d6_unit(ids[c,:R],row['mse'][0,c,:R],row['base_score'][0,c,:R],row['gain'][0,c,:R],row['recent_difference'][0,c,:R],o[c],row['oracle_mse'][0,c],choice,near_count)
                    r.update(cell=cell,model='V5',training_support=support,evaluation_support=support,R=R,method=name,query_index=int(row['query_index'][0]),query_start=int(row['query_start'][0]),channel=c)
                    r['random_retmse']=float(row['mse'][0,c,-10:].astype(np.float64).mean());d6units.append(r)
                    deltas[q,c]=r['reranked_retmse']-r['original_retmse'];values.append(r)
                    if name=='Original':curves.setdefault((q,c),[]).append(r)
            result=dict(cell=cell,model='V5',training_support=support,evaluation_support=support,R=R,method=name,split='test',queries=len(test),channels=gate['channels'],
                sampled_timestamps='32 fixed evenly spaced/all channels; not full population',near_coverage_status='EXACT_SAME_SUPPORT' if support=='P100' else 'WAITING_FULL_SUPPORT_NEAR_COUNT',
                worst1pct_positive_deterioration_share=loss_tail_contribution(deltas,.01),worst5pct_positive_deterioration_share=loss_tail_contribution(deltas,.05))
            for k in ['original_retmse','reranked_retmse','oracle_at_r','support_oracle_retmse','candidate_limitation_regret','reranking_regret','total_regret','recoverable_headroom','oracle10_coverage','near_oracle_coverage','near_oracle_support_count','recall','win','tie','loss','random_retmse']:
                result[k]=float(np.mean([v[k] for v in values])) if all(v[k] is not None for v in values) else None
            d6rows.append(result)
    for rows in curves.values():validate_r_curve(rows)
    p=dest/'D6_QUERY_CHANNEL_REGRET.jsonl.gz';tmp=p.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as stream:
        for r in d6units:stream.write(json.dumps(r,allow_nan=False)+'\n')
    tmp.replace(p)
    csvout(dest/'D5_ORACLE_LAG_DIVERSITY.csv',group_rows)
    csvout(dest/'D5_HEAD_LAG_SPECIALIZATION.csv',head_rows)
    csvout(dest/'D5_LAG_AMBIGUITY.csv',group_rows+head_rows)
    csvout(dest/'D5_LAG_DISTRIBUTION_STABILITY.csv',stability)
    for name in ['D6_TOPR_ORACLE_COVERAGE','D6_TOPR_RESTRICTED_ORACLE','D6_SHIFT_RERANKING','D6_CANDIDATE_VS_RERANKING_REGRET']:csvout(dest/(name+'.csv'),d6rows)
    dump(dest/'D6_VALIDATION_SELECTION.json',selections)
    pending=['Oracle30 lag features','exact fraction of near-tied lag alternatives','complete support NearOracle lag group','same-size random/Future-MSE matched group controls',
             'paired block uncertainty','non-V5 and cross-support comparisons']
    if support=='FULL':pending+=['Top500 future-blind pool and shift features','exact complete-support near-Oracle coverage']
    availability=dict(cell=cell,training_support=support,evaluation_support=support,model='V5',status='PARTIAL_RECOMPUTE',reuse='REUSED_VALID',source=str(src),queries_per_split=32,channels=gate['channels'],D6_R_available=rs,pending=pending)
    dump(prior,dict(status='PASS_CPU_REUSE_COMPONENT_NOT_COMPLETE_D5_D6',at=now(),availability=availability,
        source_npz_sha256=files,source_provenance_sha256=sha(src/'PROVENANCE.json'),
        outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='CPU_REUSE_GATE.json'},
        gates=dict(source_hash_alignment=True,ordered_pool_unique=True,original_checkpoint_unchanged=True,validation_only_selection=True,R10_reproduces_baseline=True,
                   coverage_monotonic=True,restricted_oracle_monotonic=True,regret_identity_query_channel=True,negative_regret_clamped=False),
        seconds=time.perf_counter()-begin,gpu_used=False))
    return availability

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'cpu_reuse.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=dict(source_library='libfile_1b60a244152081918c08fe6d7c03c428',seed=0,no_new_training=True,lag='exact inherited D4',near_fraction=NEAR_FRACTION,
       lag_margin_cosine_epsilon=LAG_MARGIN_EPS,flat_std=FLAT_STD_EPS,lag_attention_tau=.1,topR_FULL=[10,20,50,100,200,500],topR_P100=[10,20,50,100],
       alpha_grid=ALPHAS,beta_grid=BETAS,near_coverage_denominator='all same masked support candidates within d10+max(1e-8,.05*d10)',
       tie_policy='inherited canonical TopR; stable reranking preserves original rank on ties',zero_total_regret='undefined',
       worst_queries='timestamp means across all channels; largest positive deterioration; ceil(1%/5%) of timestamps',
       bootstrap='H/2H/4H paired timestamp blocks/all channels jointly/seed0/2000/percentile95%; insufficient <4 blocks',
       original_results_unchanged=True,scope='D5/D6 separate extension; partial CPU-only reuse, dependencies remain explicit')
    pp=OUT/'PROTOCOL.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(protocol))
    else:dump(pp,protocol)
    available=[]
    for cell in ORDER:
        for support in ['FULL','P100']:
            dump(OUT/'RUN_STATUS.json',dict(status='CPU_REUSE_RUNNING',cell=cell,support=support,at=now(),pid=os.getpid(),gpu_used=False))
            row=one_case(cell,support);available.append(row);print(cell,support,row['status'],flush=True)
            for model in ['V0','V1']:
                available.append(dict(cell=cell,training_support=support,evaluation_support=support,model=model,status='WAITING_FOR_DEPENDENCY',reason='model-specific D4 lag features not yet materialized'))
    csvout(OUT/'AVAILABILITY_MATRIX.csv',[dict(r,**({'D6_R_available':json.dumps(r['D6_R_available'])} if 'D6_R_available'in r else {}),**({'pending':json.dumps(r['pending'])} if 'pending'in r else {})) for r in available])
    all_files=[]
    for p in sorted((OUT/'cpu_reuse01').rglob('*')):
        if p.is_file():all_files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    csvout(OUT/'D5_D6_ARTIFACT_INVENTORY.csv',all_files)
    ready=[r for r in available if r['status']=='PARTIAL_RECOMPUTE']
    dump(OUT/'D5_D6_VALIDATION_GATE.json',dict(status='PARTIAL_CPU_REUSE_GATES_PASS_NOT_FINAL_COMPLETE',at=now(),completed_reuse_components=len(ready),expected_native_model_support_components=48,
       pending_models=len(available)-len(ready),required_metrics_still_pending=True,inventory_sha256=sha(OUT/'D5_D6_ARTIFACT_INVENTORY.csv'),gpu_used=False))
    csvout(OUT/'D5_D6_RESOURCE_SUMMARY.csv',[dict(cell=r['cell'],support=r['training_support'],seconds=json.loads((OUT/'cpu_reuse01'/(r['cell']+'_'+r['training_support'])/'CPU_REUSE_GATE.json').read_text())['seconds'],gpu_used=False) for r in ready])
    text='# D5/D6 CPU reuse status\n\nValidated D4 native V5 components: %d/16. These are 32 fixed timestamps per split and ALL channels, not the entire query population.\n\n'%len(ready)
    text+='D5 reuses exact saved lag/gain/margin/entropy; Oracle10 and head Top10 groups are complete on these timestamps. NearOracle is cached-union-only and explicitly partial. Oracle30, complete near group, same-size controls, near-tied-lag-alternative fraction remain pending.\n\n'
    text+='D6 computes future-blind TopR within available D4 Top100/200, validation-only ShiftOnly and ShiftRecent coefficients, Oracle@R, per-query/channel regret identities, R10 reproduction and monotonicity gates. Full Top500 and full near denominator are waiting for missing inference.\n\n'
    text+='V0/V1 and cross-support rows remain dependencies; no absent number is imputed. Do not claim all D1-D6 complete. Original D4/SoftLag/checkpoints and Solar are unchanged.\n'
    (OUT/'CPU_REUSE_REPORT.md').write_text(text)
    dump(OUT/'RUN_STATUS.json',dict(status='CPU_REUSE_COMPLETE_REQUIRED_ADDITIONS_PENDING',at=now(),pid=os.getpid(),components=len(ready),gpu_used=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/('CPU_REUSE_ERROR_'+str(os.getpid())+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
