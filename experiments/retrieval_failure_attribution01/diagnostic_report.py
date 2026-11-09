"""CPU-only aggregation and Korean reports; never changes model selection."""
import csv,json,math,os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path('/data/pjh_workspace/CARTS');HERE=ROOT/'experiments/retrieval_failure_attribution01';OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def conv(v):
        if isinstance(v,dict):return {k:conv(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [conv(x) for x in v]
        if isinstance(v,np.generic):v=v.item()
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    p.write_text(json.dumps(conv(x),indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def read(p):return json.loads(Path(p).read_text())
def stats(x):
    a=np.asarray(x);v=a[np.isfinite(a)]
    if not len(v):return dict(n=0,mean=None,median=None,std=None,p10=None,p90=None,p95=None,p99=None)
    q=np.quantile(v,[.1,.5,.9,.95,.99])
    return dict(n=len(v),mean=float(v.mean(dtype=np.float64)),median=float(q[1]),std=float(v.std(dtype=np.float64)),p10=float(q[0]),p90=float(q[2]),p95=float(q[3]),p99=float(q[4]))


def moving_ci(values,block,reps=2000):
    """Circular moving-time-block bootstrap; all simultaneous channels jointly."""
    a=np.asarray(values,np.float64);assert a.ndim==2
    valid=np.isfinite(a);s=np.where(valid,a,0).sum(1);n=valid.sum(1);q=len(a);L=min(block,q)
    if not n.sum():return dict(mean=None,ci_low=None,ci_high=None,n=0,query_times=q,channels=a.shape[1],block=L,repetitions=reps)
    b=(q+L-1)//L;rem=q-(b-1)*L
    ix=np.random.default_rng(0).integers(0,q,(reps,b))
    def sample(v):
        ext=np.r_[v,v[:L]];prefix=np.r_[0,np.cumsum(ext)]
        result=(prefix[ix[:,:-1]+L]-prefix[ix[:,:-1]]).sum(1) if b>1 else np.zeros(reps)
        return result+prefix[ix[:,-1]+rem]-prefix[ix[:,-1]]
    den=sample(n);num=sample(s);z=num[den>0]/den[den>0];lo,hi=np.quantile(z,[.025,.975])
    return dict(mean=float(s.sum()/n.sum()),ci_low=float(lo),ci_high=float(hi),n=int(n.sum()),query_times=q,channels=a.shape[1],block=L,repetitions=reps,seed=0,resampling='circular time blocks, channels jointly',effective_nonoverlap_blocks=q/L)


def materialize(out,split):
    files=sorted((out/split).glob('batch[0-9]*.npz'));assert files
    gate=read(out/split/'SPLIT_GATE.json');q,c=gate['query_count'],gate['channels']
    with np.load(files[0]) as a:
        names=a['method_names'].tolist();fields=a['metric_names'].tolist();gn=a['group_method_names'].tolist();fn=a['feature_names'].tolist()
        shape=(q,c,len(names),len(fields));gs=(q,c,len(gn),4,len(fn))
    dest=out/'aggregate';dest.mkdir(exist_ok=True)
    met=np.lib.format.open_memmap(dest/f'{split}_metrics.npy',mode='w+',dtype=np.float64,shape=shape)
    groups=np.lib.format.open_memmap(dest/f'{split}_groups.npy',mode='w+',dtype=np.float32,shape=gs)
    neigh=np.lib.format.open_memmap(dest/f'{split}_neighborhood.npy',mode='w+',dtype=np.float32,shape=(q,c,6,5))
    starts=np.empty(q,np.int64);offset=0
    for f in files:
        with np.load(f) as a:
            n=len(a['query_start']);assert a['method_names'].tolist()==names and a['group_method_names'].tolist()==gn
            met[offset:offset+n]=a['metrics'];groups[offset:offset+n]=a['group_features'];neigh[offset:offset+n]=a['neighborhood'];starts[offset:offset+n]=a['query_start'];offset+=n
    assert offset==q and np.all(np.diff(starts)==1)
    for a in [met,groups,neigh]:a.flush()
    np.save(dest/f'{split}_starts.npy',starts)
    dump(dest/f'{split}_schema.json',dict(methods=names,metrics=fields,group_methods=gn,features=fn,queries=q,channels=c))
    return met,groups,neigh,starts,names,fields,gn,fn


def summarize_cell(support,cell):
    out=OUT/support/cell;h=int(cell.rsplit('_',1)[1]);block=4*h
    va=materialize(out,'val');te=materialize(out,'test')
    rows=[];channels=[];wins=[];ranking=[];missed=[];boundary=[];feature_rows=[];effects=[];ambiguity=[];strata=[];tails=[];probes=[]
    for split,parts in [('val',va),('test',te)]:
        met,groups,neigh,starts,names,fields,gn,fn=parts;idx={k:i for i,k in enumerate(fields)}
        # Raw per-query primary metrics and IDs; a Parquet row represents a query/channel/method.
        writer=None;pqpath=out/f'{split}_QUERY_LEVEL_REGRET.parquet'
        for f in sorted((out/split).glob('batch[0-9]*.npz')):
            with np.load(f) as a:
                b,c,m,d=a['metrics'].shape
                table={'query_start':np.repeat(a['query_start'],c*m),'channel':np.tile(np.repeat(np.arange(c),m),b),'method':np.tile(names,b*c),
                    'cell':[cell]*(b*c*m),'support':[support]*(b*c*m),'split':[split]*(b*c*m)}
                for j,k in enumerate(fields):table[k]=a['metrics'][...,j].reshape(-1)
                table['candidate_ids']=a['selected_ids'].reshape(-1,10).tolist()
                at=pa.Table.from_pydict(table)
                if writer is None:writer=pq.ParquetWriter(pqpath,at.schema,compression='zstd')
                writer.write_table(at)
        writer.close()
        for mi,name in enumerate(names):
            row=dict(cell=cell,support=support,split=split,method=name,seed=0,queries=len(starts),channels=met.shape[1],units=len(starts)*met.shape[1])
            for k,j in idx.items():row[k]=stats(met[:,:,mi,j])['mean']
            row['ndcg_definition']='inverse_MSE_v1 (legacy_ndcg10 separately labelled)'
            row['full_dependency']='DEFERRED_FULL_DEPENDENCY' if support=='P100' else 'AVAILABLE'
            rows.append(row)
            for c in range(met.shape[1]):
                channels.append(dict(cell=cell,support=support,split=split,method=name,channel=c,queries=len(starts),**{k:stats(met[:,c,mi,j])['mean'] for k,j in idx.items()}))
            ranking.append({k:row[k] for k in ['cell','support','split','method','retmse','spearman','kendall','ndcg_inverse_mse','ndcg50','ndcg100','recall','recall50','recall100','spearman_student_top50','spearman_student_top100']})
            ranking[-1]['kendall_definition']='APPROXIMATE fixed 4096 position pairs tau-b' if support=='FULL' else 'EXACT all 4950 pairs tau-b'
            missed.append({k:v for k,v in row.items() if k.startswith('missed_') or k in ['cell','support','split','method','recall']})
            boundary.append({k:row[k] for k in ['cell','support','split','method','retmse','ranking_regret','exchange_regret','boundary_error','oracle_boundary_gap','student_boundary_gap','student_top1_topk_gap','boundary_11_20_exchange']})
            r=np.asarray(met[:,:,mi,idx['ranking_regret']]).reshape(-1);ordered=np.sort(r.astype(np.float64))[::-1];total=ordered.sum()
            tails.append(dict(cell=cell,support=support,split=split,method=name,**stats(r),**{f'worst_{p}pct_loss_share':None if abs(total)<1e-12 else float(ordered[:max(1,math.ceil(len(r)*p/100))].sum()/total) for p in [1,5,10]}))
            for comparator in ['Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine']:
                j=names.index(comparator);delta=np.asarray(met[:,:,mi,idx['retmse']],np.float64)-met[:,:,j,idx['retmse']]
                wins.append(dict(cell=cell,support=support,split=split,method=name,comparator=comparator,win_rate=float((delta<0).mean()),tie_rate=float((delta==0).mean()),loss_rate=float((delta>0).mean()),**moving_ci(delta,block)))
            if name.startswith('Ridge'):probes.append(dict(cell=cell,support=support,split=split,method=name,retmse=row['retmse'],oracle_regret=row['ranking_regret'],spearman=row['spearman'],future_mse_prediction_correlation=row['pearson_score_negative_mse'],pairwise_ranking_accuracy=row['pairwise_ranking_accuracy'],selection='fixed alpha candidate' if split=='val' else 'validation-selected ridge',test_selection=False))
            # Validation thresholds are channel-specific, controlling channel composition.
            valnames=va[4]
            valname=name if name in valnames else ('Ridge_alpha'+str(read(out/'probe_selection.json')['state']['alphas'][read(out/'probe_selection.json')['state']['selected']]))
            vi=valnames.index(valname);vreg=va[0][:,:,vi,va[5].index('ranking_regret')]
            cuts=np.quantile(vreg,[1/3,2/3],axis=0)
            reg=met[:,:,mi,idx['ranking_regret']];labels=(reg>cuts[0][None]).astype(int)+(reg>cuts[1][None]).astype(int)
            for g,label in enumerate(['Low','Medium','High']):
                take=labels==g
                strata.append(dict(cell=cell,support=support,split=split,method=name,regret_group=label,threshold_source='validation per channel terciles',units=int(take.sum()),retmse=stats(met[:,:,mi,idx['retmse']][take])['mean'],future_dispersion=stats(neigh[:,:,1,3][take])['mean'],past_distance=stats(neigh[:,:,1,0][take])['mean'],within_neighborhood_past_future_correlation=stats(neigh[:,:,1,4][take])['mean']))
        for ni,name in enumerate(['PastNearest1','PastNearest10','PastNearest20','PastNearest50','Random10','FutureNearest10_DIAGNOSTIC']):
            ambiguity.append(dict(cell=cell,support=support,split=split,neighborhood=name,**{k:stats(neigh[:,:,ni,j])['mean'] for j,k in enumerate(['past_distance','query_future_mse','future_variance','mean_pairwise_future_mse','past_future_correlation'])}))
        for mi,name in enumerate(gn):
            for fi,feature in enumerate(fn[:-1]):
                for g,label in enumerate(['CorrectlyRetrievedOracle','MissedOracle','FalseSelected','ValidRandom']):
                    feature_rows.append(dict(cell=cell,support=support,split=split,method=name,group=label,feature=feature,aggregation='query/channel group means; empty group undefined',**stats(groups[:,:,mi,g,fi])))
                delta=np.asarray(groups[:,:,mi,1,fi],np.float64)-groups[:,:,mi,2,fi];s=stats(delta)
                effects.append(dict(cell=cell,support=support,split=split,method=name,feature=feature,comparison='MissedOracle minus FalseSelected, paired query/channel',standardized_paired_effect=None if not s['std'] or s['std']<1e-12 else s['mean']/s['std'],**moving_ci(delta,block)))
    tables={'RETRIEVAL_BASELINE_COMPARISON.csv':rows,'RETRIEVAL_QUALITY_BY_CHANNEL.csv':channels,'BASELINE_WIN_RATE.csv':wins,
        'RANKING_QUALITY.csv':ranking,'GLOBAL_VS_TOPK_RANKING.csv':ranking,'MISSED_ORACLE_RANK_DISTRIBUTION.csv':missed,
        'TOPK_BOUNDARY_ERROR.csv':boundary,'EXCHANGE_REGRET.csv':boundary,'BOUNDARY_GAP_ANALYSIS.csv':boundary,
        'MISSED_ORACLE_FEATURES.csv':feature_rows,'FEATURE_GROUP_COMPARISON.csv':effects,'PAST_FUTURE_DISAGREEMENT.csv':effects,
        'PAST_FUTURE_AMBIGUITY.csv':ambiguity,'AMBIGUITY_BY_REGRET_GROUP.csv':strata,'PAST_ONLY_PROBE_RESULTS.csv':probes,
        'PROBE_VS_RETRIEVER_COMPARISON.csv':rows,'REGRET_TAILS.csv':tails}
    for n,v in tables.items():csvwrite(out/n,v)
    test=[r for r in rows if r['split']=='test'];best=min(test,key=lambda r:r['retmse'])
    text=f'# {support} {cell} Retrieval 진단\n\n전체 query/channel, seed0, K10. 주요 지표는 Test RetMSE이며 Stage2는 실행하지 않았다.\n\n'
    text+='| Method | RetMSE | Recall | Oracle regret | Spearman |\n|---|---:|---:|---:|---:|\n'
    for r in test:text+=f"| {r['method']} | {r['retmse']:.6f} | {r['recall']:.6f} | {r['ranking_regret']:.6f} | {r['spearman'] if r['spearman'] is not None else 'undefined'} |\n"
    text+='\n관찰된 상관관계는 인과적 실패 원인이 아니다. 주원인 분류는 INCONCLUSIVE로 유지하며 Full 의존 지표와 함께 판단한다. Probe 결과는 제한된 past-only 특징과 Ridge의 정보량 진단이다.\n'
    text+='\nP100 단계 Full Oracle/coverage/decomposition: DEFERRED_FULL_DEPENDENCY.\n' if support=='P100' else '\nFull-trained MISSING 행은 MODEL_STATUS.json에 남기며 재학습 결과로 대체하지 않았다.\n'
    text+='\n통계: 2000회 circular time-block bootstrap, block=4H, 동일 시점 채널 공동 resampling. 학습 seed 반복이 아니며 seed0 결론의 한계가 있다.\n'
    (out/'PARTIAL_REPORT_KO.md').write_text(text,encoding='utf-8')
    plot_cell(out,test,ranking,missed,boundary,effects,ambiguity,strata)
    if support=='FULL':backfill_decomposition(cell)


def backfill_decomposition(cell):
    """Only called after Full results exist, using P100-selected model errors."""
    from diagnostic_core import regret_decomposition
    full=OUT/'FULL'/cell;p100=OUT/'P100'/cell;summary=[];coverage_rows=[];writer=None
    target=full/'QUERY_LEVEL_REGRET_DECOMPOSITION.parquet'
    for split in ['val','test']:
        schema=read(p100/'aggregate'/f'{split}_schema.json');pmet=np.load(p100/'aggregate'/f'{split}_metrics.npy',mmap_mode='r');pstarts=np.load(p100/'aggregate'/f'{split}_starts.npy')
        names=schema['methods'];j=schema['metrics'].index('retmse');offset=0;allparts={n:[] for n in names}
        for f in sorted((full/split).glob('batch[0-9]*.npz')):
            with np.load(f) as a:
                b,c=a['metrics'].shape[:2];assert np.array_equal(a['query_start'],pstarts[offset:offset+b])
                oi=a['method_names'].tolist().index('IndividualMSE_Oracle');rfull=a['metrics'][:,:,oi,a['metric_names'].tolist().index('retmse')]
                rp100=a['p100_oracle'];cov=a['full_oracle_coverage']
                with np.load(p100/split/f.name) as pa_ids:prior_selected=pa_ids['selected_ids'].copy()
                for mi,name in enumerate(names):
                    rm=pmet[offset:offset+b,:,mi,j];v=regret_decomposition(rfull,rp100,rm,'P100')
                    allparts[name].append((v['pool_regret'].ravel(),v['ranking_regret'].ravel(),v['total_regret'].ravel(),cov.ravel()))
                    frecall=(prior_selected[:,:,mi,:,None]==a['oracle_ids'][:,: ,None,:]).any(-1).mean(-1)
                    tab=dict(cell=[cell]*(b*c),split=[split]*(b*c),model=[name]*(b*c),model_selection_support=['P100']*(b*c),query_start=np.repeat(a['query_start'],c),channel=np.tile(np.arange(c),b),full_oracle=rfull.ravel(),p100_oracle=rp100.ravel(),model_retmse=np.asarray(rm).ravel(),pool_regret=v['pool_regret'].ravel(),ranking_regret=v['ranking_regret'].ravel(),total_regret=v['total_regret'].ravel(),full_oracle_coverage=cov.ravel(),full_oracle_recall=frecall.ravel(),p100_oracle_recall=np.asarray(pmet[offset:offset+b,:,mi,schema['metrics'].index('recall')]).ravel())
                    t=pa.Table.from_pydict(tab)
                    if writer is None:writer=pq.ParquetWriter(target,t.schema,compression='zstd')
                    writer.write_table(t)
                offset+=b
        assert offset==len(pstarts)
        for name,parts in allparts.items():
            pool,rank,total,cov=[np.concatenate([p[i] for p in parts]) for i in range(4)];den=total.sum()
            summary.append(dict(cell=cell,split=split,method=name,model_selection_support='P100',status='COMPLETE',pool_regret=float(pool.mean()),ranking_regret=float(rank.mean()),total_regret=float(total.mean()),pool_share=None if den<=1e-12 else float(pool.sum()/den),ranking_share=None if den<=1e-12 else float(rank.sum()/den),identity_max_abs=float(np.abs(total-pool-rank).max())))
            coverage_rows.append(dict(cell=cell,split=split,method=name,coverage_at100=float(cov.mean())))
    writer.close();csvwrite(full/'RETRIEVAL_REGRET_DECOMPOSITION.csv',summary);csvwrite(full/'POOL_ORACLE_COVERAGE.csv',coverage_rows)


def plot_cell(out,test,ranking,missed,boundary,effects,ambiguity,strata):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def bars(name,labels,values,ylabel,title):
        fig,ax=plt.subplots(figsize=(max(8,len(labels)*.7),4.8));ax.bar(range(len(labels)),values);ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=40,ha='right');ax.set_ylabel(ylabel);ax.set_title(title);fig.tight_layout();fig.savefig(figs/name,dpi=160);plt.close(fig)
    bars('baseline_retrieval_comparison.png',[r['method'] for r in test],[r['retmse'] for r in test],'Test RetMSE@10 (lower is better)',out.parent.name+' '+out.name)
    t=[r for r in ranking if r['split']=='test'];fig,ax=plt.subplots(figsize=(7,5))
    for r in t:
        if r['spearman'] is not None:ax.scatter(r['spearman'],r['retmse']);ax.annotate(r['method'],(r['spearman'],r['retmse']),fontsize=7)
    ax.set_xlabel('Spearman (higher is better)');ax.set_ylabel('RetMSE (lower is better)');fig.tight_layout();fig.savefig(figs/'global_vs_topk_ranking.png',dpi=160);plt.close(fig)
    t=[r for r in boundary if r['split']=='test'];bars('topk_exchange_regret.png',[r['method'] for r in t],[r['exchange_regret'] for r in t],'Exchange regret (lower is better)','Same-support Top-10 exchange loss')
    t=[r for r in missed if r['split']=='test'];bars('missed_oracle_rank_distribution.png',[r['method'] for r in t],[r['missed_rank_mean'] or 0 for r in t],'Mean missed Oracle rank','No misses shown as 0; see CSV undefined values')
    t=[r for r in ambiguity if r['split']=='test'];bars('neighborhood_future_dispersion.png',[r['neighborhood'] for r in t],[r['mean_pairwise_future_mse'] or 0 for r in t],'Pairwise future MSE','Singleton pairwise dispersion undefined (CSV)')
    t=[r for r in strata if r['split']=='test'];fig,ax=plt.subplots(figsize=(7,5))
    for r in t:
        if r['future_dispersion'] is not None:ax.scatter(r['future_dispersion'],r['retmse'],s=12)
    ax.set_xlabel('Observed neighborhood future dispersion');ax.set_ylabel('RetMSE');fig.tight_layout();fig.savefig(figs/'ambiguity_vs_regret.png',dpi=160);plt.close(fig)
    t=[r for r in effects if r['split']=='test' and r['mean'] is not None]
    if t:
        model=t[0]['method'];z=[r for r in t if r['method']==model][:16]
        bars('feature_effect_sizes.png',[r['feature'] for r in z],[r['standardized_paired_effect'] or 0 for r in z],'Paired standardized difference: missed minus false',model+' (descriptive, not causal)')
    f=out/'MISSED_ORACLE_FEATURES.csv';allrows=list(csv.DictReader(f.open()));fig,ax=plt.subplots(figsize=(7,5))
    groups={}
    for r in allrows:
        if r['split']=='test' and r['mean'] and r['feature'] in ['delta_last_cosine','evaluation_future_mse']:groups.setdefault((r['method'],r['group']),{})[r['feature']]=float(r['mean'])
    for key,v in groups.items():
        if len(v)==2:ax.scatter(v['delta_last_cosine'],v['evaluation_future_mse']);ax.annotate('/'.join(key),(v['delta_last_cosine'],v['evaluation_future_mse']),fontsize=5)
    ax.set_xlabel('Group mean past Delta-last cosine');ax.set_ylabel('Group mean future MSE');fig.tight_layout();fig.savefig(figs/'past_future_similarity.png',dpi=160);plt.close(fig)


def summarize_all():
    protocol=read(HERE/'PROTOCOL.json');rollup={};status=[]
    for support in ['P100','FULL']:
        for cell in protocol['cell_order']:
            p=OUT/support/cell;done=p/'COMPLETE.json'
            if not done.exists():status.append(dict(support=support,cell=cell,status='NOT_COMPLETE'));continue
            d=read(done);status.append(dict(support=support,cell=cell,status=d['status'],missing=len(d['missing_models']),seconds=d['total_wall_seconds']))
            for f in p.glob('*.csv'):
                rows=list(csv.DictReader(f.open()));rollup.setdefault(f.name,[]).extend(rows)
    for name,rows in rollup.items():csvwrite(OUT/'07_Summary'/name,rows)
    csvwrite(OUT/'07_Summary/RUN_STATUS.csv',status)
    resources=[]
    for s in status:
        if s['status']=='NOT_COMPLETE':continue
        d=read(OUT/s['support']/s['cell']/'COMPLETE.json');resources.append({k:v for k,v in d.items() if k in ['cell','support','total_wall_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','cpu_ram_peak_bytes','channels','oracle_computation_seconds','retrieval_inference_seconds','ranking_analysis_seconds','feature_extraction_seconds','probe_training_seconds','canonical_reproduction_seconds','full_candidate_count','p100_candidate_count']})
    csvwrite(OUT/'07_Summary/RESOURCE_SUMMARY.csv',resources)
    attribution=[]
    for r in rollup.get('RETRIEVAL_BASELINE_COMPARISON.csv',[]):
        if r['split']!='test':continue
        evidence=[x for x in rollup.get('RETRIEVAL_REGRET_DECOMPOSITION.csv',[]) if x['cell']==r['cell'] and x['method']==r['method'] and x['split']=='test']
        attribution.append(dict(cell=r['cell'],support=r['support'],method=r['method'],retmse=r['retmse'],ranking_regret=r['ranking_regret'],spearman=r['spearman'],boundary_11_20_exchange=r['boundary_11_20_exchange'],pool_share=evidence[0]['pool_share'] if evidence else 'DEFERRED_OR_NOT_APPLICABLE',conclusion='MIXED / INCONCLUSIVE',causal_effect_established=False))
    csvwrite(OUT/'07_Summary/FAILURE_ATTRIBUTION_SUMMARY.csv',attribution)
    csvwrite(OUT/'07_Summary/CHANNEL_LEVEL_FAILURE_SUMMARY.csv',rollup.get('RETRIEVAL_QUALITY_BY_CHANNEL.csv',[]))
    complete=all(s['status']!='NOT_COMPLETE' for s in status)
    text='# Retrieval 실패 진단 결과\n\n'+('모든 P100/Full 셀의 실행 및 검증을 완료했다. 모델 누락 행은 별도 유지한다.' if complete else 'PARTIAL: 미완료 셀이 있어 전체 실험 결론을 확정하지 않는다.')+'\n\n'
    text+='Primary Test RetMSE, seed0, 전체 평가 query/channel. P100과 Full-trained는 파라미터/초기화/학습 support가 달라 support의 인과 효과로 직접 해석하지 않는다. Stage2는 수행하지 않았다.\n\n'
    text+='| Support | Cell | Status |\n|---|---|---|\n'+''.join(f"| {s['support']} | {s['cell']} | {s['status']} |\n" for s in status)
    text+='\n검색 성능, 순위/경계 오류, 이웃 미래 분산, past-only Ridge 결과는 각 셀 PARTIAL_REPORT_KO.md와 요약 CSV에 수록했다. 후보군 밖 누락은 순위 하단 누락으로 취급하지 않았다. 인과적 원인과 새로운 loss의 효과는 이 진단으로 확정되지 않아 INCONCLUSIVE다.\n'
    text+='\nSolar Full-trained V0/V5 8개는 현재 단계 MISSING이며, 별도 승인된 후속 학습은 이 진단 전체 이후 실행 대상으로 분리되어 있다. 단순 baseline과 Full Oracle 결과는 모델 누락과 독립적으로 계산했다.\n'
    text+='\nWeather 데이터의 기존 -9999 sentinel과 해당 train scaler를 보존했다. 지표는 기존 normalized delta-last 공간이며 정제 데이터 성능이나 물리 단위 오차가 아니다. L=H가 함께 변하므로 horizon만의 효과로 해석하지 않는다.\n'
    (OUT/'07_Summary/FINAL_REPORT.md').write_text(text,encoding='utf-8')
    rec=[('Original KL','전체 teacher 분포','기존 목적함수; teacher entropy를 포함','global ranking/RetMSE와 함께 확인','teacher reliability와 top-K 불일치'),
         ('Weighted Cross-Entropy','동일 고정 teacher 분포','KL과 teacher entropy 상수 차이; 동일 gradient이면 새 신호가 아님','기존 KL과 동등성','명칭만 바꾼 개선 주장 금지'),
         ('RankNet / Weighted RankNet','pairwise 순위 오류','pairwise 비교, regret 가중 가능','missed rank 및 exchange regret','pair sampling/teacher noise'),
         ('Margin Ranking','정답 순서 margin','확률분포 전체 fitting 대신 차이 제약','oracle boundary gap과 손실','작은 미래 MSE 차이에 과도한 margin'),
         ('Choice Loss','후보 선택','set/choice 정의와 KL 구별 필요','선택 regret','정의 불명확하면 비교 불가'),
         ('Top-K aligned auxiliary','경계 근처 큰 교환 손실','KL에 보조 신호','rank11–20 regret 기여','aux weight 및 점수 scale'),
         ('Differentiable Top-K','hard Top-K와 학습 정렬','연속 surrogate','hard-vs-soft 재현','temperature/gradient mismatch'),
         ('Hard Negative Mining','중하위 순위의 어려운 후보','샘플링 분포 변경','missed ranks/global correlation','후보 support 변경 confound')]
    csvwrite(OUT/'07_Summary/LOSS_DESIGN_CANDIDATES.csv',[dict(loss=a,target=b,difference=c,required_evidence=d,risk=e,status='PROPOSAL_ONLY_NOT_TRAINED') for a,b,c,d,e in rec])
    (OUT/'07_Summary/LOSS_DESIGN_RECOMMENDATION.md').write_text('# Loss 후속 연구 판단\n\n새 loss는 학습하지 않았다. 각 후보는 LOSS_DESIGN_CANDIDATES.csv에 비교했다. Pool regret이 지배적이면 preselection 개선을 먼저 검토한다. 같은 support의 ranking regret이 크고 missed Oracle이 하위에 집중되면 representation/pairwise ranking을, rank11–20의 실제 교환 손실이 클 때만 boundary 신호를 우선 검토한다. Ridge의 성공/실패와 관찰된 past–future ambiguity는 정보의 활용 가능성에 대한 제한된 근거이며 불가능성 증명이 아니다.\n\n효과 확인에는 동일 init/support/예산에서 validation으로 정한 한 가지 변경의 후속 ablation이 필요하다. 이 보고서의 제안은 학습 효과를 주장하지 않는다.\n',encoding='utf-8')
    dump(OUT/'07_Summary/SUMMARY_GATE.json',dict(status='PASS' if complete else 'PARTIAL',cells=status,causal_attribution='INCONCLUSIVE',new_loss_training=False))
    decomp=[r for r in rollup.get('RETRIEVAL_REGRET_DECOMPOSITION.csv',[]) if r['split']=='test' and r['method']=='V5_P100Trained']
    if decomp:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        dest=OUT/'08_Figures';dest.mkdir(exist_ok=True)
        vals=np.array([[float(r['pool_share'] or 'nan'),float(r['ranking_share'] or 'nan')] for r in decomp]);labels=[r['cell'] for r in decomp]
        fig,ax=plt.subplots(figsize=(10,5));ax.bar(np.arange(len(vals)),vals[:,0],label='Pool');ax.bar(np.arange(len(vals)),vals[:,1],bottom=vals[:,0],label='Ranking');ax.set_xticks(np.arange(len(vals)));ax.set_xticklabels(labels,rotation=45,ha='right');ax.set_ylabel('Share of aggregate RetMSE regret');ax.legend();fig.tight_layout();fig.savefig(dest/'retrieval_regret_decomposition.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,max(4,len(vals)*.45)));im=ax.imshow(vals,vmin=0,vmax=1,aspect='auto');ax.set_xticks([0,1]);ax.set_xticklabels(['Pool regret share','Ranking regret share']);ax.set_yticks(np.arange(len(vals)));ax.set_yticklabels(labels);fig.colorbar(im,ax=ax);ax.set_title('Descriptive attribution; not a causal test');fig.tight_layout();fig.savefig(dest/'dataset_horizon_bottleneck_heatmap.png',dpi=160);plt.close(fig)


if __name__=='__main__':summarize_all()
