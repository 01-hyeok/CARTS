"""CPU-only snapshot summaries; validation selection kept separate from test oracle."""
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from runtime import HERE, ROOT, OUT, FIELDS, write_json, now

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize_pair(v1,v5):
    target=v5/'pair_v5_minus_v1.json'
    if target.exists():return json.loads(target.read_text())
    vm=json.loads((v5/'val_metrics.json').read_text())
    tm=json.loads((v5/'test_metrics.json').read_text())
    n,c=tm['query_count'],tm['channels']
    time_delta=np.zeros((n,2),dtype=np.float64)
    channels=[];strata=[];all_diff=[]
    for ch in range(c):
        with np.load(v1/'queries/test'/f'channel{ch:03d}.npz') as a, np.load(v5/'queries/test'/f'channel{ch:03d}.npz') as b, np.load(v5/'queries/val'/f'channel{ch:03d}.npz') as val:
            assert np.array_equal(a['query_start'],b['query_start'])
            ia=list(a['config_names']).index('All');ib=list(b['config_names']).index('All')
            am=a['metrics'][:,ia];bm=b['metrics'][:,ib]
            diff=(bm[:,:2]-am[:,:2]).astype(np.float64)
            time_delta+=diff/c
            row=dict(channel=ch,n_queries=n,delta_retmse=float(diff[:,0].mean()),delta_agg_mse=float(diff[:,1].mean()),
                ret_improved_fraction=float((diff[:,0]<0).mean()),agg_improved_fraction=float((diff[:,1]<0).mean()),
                ret_worse_fraction=float((diff[:,0]>0).mean()),agg_worse_fraction=float((diff[:,1]>0).mean()),
                agg_tie_fraction=float((diff[:,1]==0).mean()),
                mean_improvement_when_better=float(-diff[diff[:,1]<0,1].mean()) if (diff[:,1]<0).any() else 0.,
                mean_damage_when_worse=float(diff[diff[:,1]>0,1].mean()) if (diff[:,1]>0).any() else 0.)
            channels.append(row)
            features=b['input_features'];vfeatures=val['input_features']
            for label,z,vz,deployable in (
                ('input_std',features[:,0],vfeatures[:,0],True),
                ('input_mean_abs_change',features[:,1],vfeatures[:,1],True),
                ('last_repeat_future_mse',b['last'][:,0],val['last'][:,0],False)):
                threshold=float(np.median(vz))
                for group,mask in [('low',z<=threshold),('high',z>threshold)]:
                    if not mask.any():continue
                    strata.append(dict(channel=ch,feature=label,group=group,validation_threshold=threshold,
                        deployable_input_only=deployable,n_queries=int(mask.sum()),delta_retmse=float(diff[mask,0].mean()),
                        delta_agg_mse=float(diff[mask,1].mean()),agg_improved_fraction=float((diff[mask,1]<0).mean())))
            all_diff.append(diff.astype(np.float32))
    d=np.concatenate(all_diff)
    h=int(tm['cell'].rsplit('_',1)[1]);block=2*h
    blocks=[time_delta[i:i+block].mean(0) for i in range(0,n,block) if len(time_delta[i:i+block])==block]
    ci=None
    if len(blocks)>=5:
        b=np.stack(blocks);g=np.random.default_rng(7281)
        samples=b[g.integers(0,len(b),(2000,len(b)))].mean(1)
        ci=np.quantile(samples,[.025,.975],axis=0).tolist()
    out=dict(status='COMPLETE',cell=tm['cell'],n_queries=n,channels=c,n_query_channel=n*c,
        delta_retmse=float(d[:,0].mean(dtype=np.float64)),delta_agg_mse=float(d[:,1].mean(dtype=np.float64)),
        agg_improved_fraction=float((d[:,1]<0).mean()),agg_worse_fraction=float((d[:,1]>0).mean()),
        ret_improved_fraction=float((d[:,0]<0).mean()),ret_worse_fraction=float((d[:,0]>0).mean()),
        quantile_levels=[0,.05,.25,.5,.75,.95,1],quantiles=np.quantile(d,[0,.05,.25,.5,.75,.95,1],axis=0).tolist(),
        block_length=block,full_time_blocks=len(blocks),ci95=ci,
        ci_note='Time-block bootstrap of channel-mean deltas; incomplete tail excluded from CI only; null if <5 blocks. Seed variation separate. Exploratory multiple comparisons.',
        strata_note='Threshold fixed per channel using validation. Actual Last-repeat-future-error strata are retrospective, nondeployable.')
    csv_write(v5/'pair_v5_minus_v1_channels.csv',channels)
    csv_write(v5/'pair_v5_minus_v1_strata.csv',strata)
    # Small histogram artifact; full query values already stored in per-channel NPZ.
    bins=np.linspace(-2,2,161)
    hist=[np.histogram(d[:,i],bins)[0].tolist() for i in range(2)]
    out['histogram']=dict(edges=bins.tolist(),counts=hist,outside_count=[int((np.abs(d[:,i])>2).sum()) for i in range(2)])
    write_json(target,out)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'configs/run_manifest.json').read_text())
    rows=[];status=[];headrows=[];selections=[];loo=[];curves=[];krows=[];div=[];pairs=[];baselines=[]
    complete={}
    for r in manifest:
        dest=ROOT/r['output_path'];done=dest/'COMPLETE.json'
        status.append(dict(run_id=r['run_id'],status='COMPLETE' if done.exists() else ('PARTIAL' if (dest/'epochs.json').exists() else 'NOT_STARTED')))
        if not done.exists():continue
        meta=json.loads(done.read_text());val=json.loads((dest/'val_metrics.json').read_text());test=json.loads((dest/'test_metrics.json').read_text())
        key=(r['dataset'],r['horizon'],r['seed'],r['model']);complete[key]=dest
        common=dict(dataset=r['dataset'],horizon=r['horizon'],seed=r['seed'],model=r['model'])
        rows.append(dict(**common,selected_epoch=meta['selected_epoch'],**test['rows'][0],
             test_query_count=test['query_count'],channels=test['channels'],last_mse=test['last_mse'],last_mae=test['last_mae'],
             checkpoint=meta['selected_checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],elapsed_seconds=meta['elapsed_seconds']))
        vr={x['name']:x for x in val['rows']};tr={x['name']:x for x in test['rows']}
        for x in test['rows']:
            if x['k']!=10:krows.append(dict(**common,**x))
            elif len(x['heads'])==1 and r['num_query_heads']>1:headrows.append(dict(**common,**x))
        if r['num_query_heads']>1:
            single=[x for x in val['rows'] if x['k']==10 and len(x['heads'])==1]
            for criterion in ('retmse','agg_mse'):
                best=min(single,key=lambda x:(x[criterion],x['name']))
                selections.append(dict(**common,criterion=f'validation {criterion}',selected=best['name'],
                    val_selection_value=best[criterion],**{f'test_{k}':v for k,v in tr[best['name']].items() if k in FIELDS},
                    test_all_retmse=tr['All']['retmse'],test_all_agg_mse=tr['All']['agg_mse']))
            dv=test['diversity'];div.append(dict(**common,**dv))
        if r['model']=='V5':
            for missing in range(1,6):
                x=next(v for v in test['rows'] if v['k']==10 and len(v['heads'])==4 and missing not in v['heads'])
                loo.append(dict(**common,removed_head=missing,delta_retmse=x['retmse']-tr['All']['retmse'],
                    delta_agg_mse=x['agg_mse']-tr['All']['agg_mse'],delta_agg_mae=x['agg_mae']-tr['All']['agg_mae']))
            for size in range(1,6):
                choices=[x for x in val['rows'] if x['k']==10 and len(x['heads'])==size]
                best=min(choices,key=lambda x:(x['retmse'],x['name']))
                corresponding=[tr[x['name']] for x in choices]
                curves.append(dict(**common,head_count=size,selected=best['name'],validation_retmse=best['retmse'],
                    test_retmse=tr[best['name']]['retmse'],test_agg_mse=tr[best['name']]['agg_mse'],test_recall=tr[best['name']]['recall'],
                    subset_count=len(choices),mean_subset_test_retmse=float(np.mean([x['retmse'] for x in corresponding])),
                    mean_subset_test_agg_mse=float(np.mean([x['agg_mse'] for x in corresponding]))))
        if r['model']=='V0' and r['seed']==0:
            baselines.append(dict(dataset=r['dataset'],horizon=r['horizon'],method='Last-repeat',retmse=None,recall=None,regret=None,
                                  agg_mse=test['last_mse'],agg_mae=test['last_mae'],last_skill=0. if test['last_mse']>1e-12 else None))
            for name,m in test['baselines'].items():
                baselines.append(dict(dataset=r['dataset'],horizon=r['horizon'],method=name,**m))
    for (ds,h,s,m),v5 in complete.items():
        if m=='V5' and (ds,h,s,'V1') in complete:
            pairs.append(dict(dataset=ds,horizon=h,seed=s,**summarize_pair(complete[(ds,h,s,'V1')],v5)))
    tables={'RUN_STATUS':status,'MAIN_SEED_RESULTS':rows,'BASELINES':baselines,'HEAD_QUALITY':headrows,
            'BEST_SINGLE_VALIDATION_SELECTED':selections,'LEAVE_ONE_OUT':loo,'HEAD_COUNT_VALIDATION_SELECTED':curves,
            'K_SENSITIVITY':krows,'DIVERSITY':div}
    for name,data in tables.items():csv_write(OUT/f'{name}.csv',data)
    write_json(OUT/'QUERY_PAIR_SUMMARIES.json',pairs)
    seedrows=[]
    for ds in ('ETTh1','Weather','Solar'):
        for h in (96,192,336,720):
            for model in ('V0','V1','V2','V5'):
                values=[x for x in rows if x['dataset']==ds and x['horizon']==h and x['model']==model]
                if len(values)!=3:continue
                sr=dict(dataset=ds,horizon=h,model=model,seeds=3)
                for metric in FIELDS+['last_skill']:
                    v=[x[metric] for x in values]
                    sr[metric+'_mean']=None if any(x is None for x in v) else float(np.mean(v))
                    sr[metric+'_std']=None if any(x is None for x in v) else float(np.std(v,ddof=1))
                seedrows.append(sr)
    csv_write(OUT/'THREE_SEED_MEAN_STD.csv',seedrows)
    summary=['# P100 Head Retrieval Quality — 진행 결과',f'확인 시각: {now()}',
        f'상태: {"COMPLETE" if len(rows)==144 else "PARTIAL"}, Stage1 {len(rows)}/144 완료. Stage2는 수행하지 않는다.',
        '기존 z-score teacher를 유지했다. 공식 checkpoint는 10epoch 전체 중 validation Mean-Mixture RetMSE@10 최소이며 test로 선택하지 않는다.',
        '아래 표는 완료된 seed0 결과이다. 미완료 모델은 빈 값을 추정하지 않는다. Seed1/2와 3seed 평균·표준편차는 CSV에 별도 기록한다.',
        'Oracle은 개별 후보 MSE로 고른 P100 Individual-MSE Oracle@10이다. Aggregate MSE/MAE의 이론적 최적은 아니다.']
    for ds in ('ETTh1','Weather','Solar'):
        for h in (96,192,336,720):
            b=[x for x in baselines if x['dataset']==ds and x['horizon']==h and 'standard deviation' not in x['method']]
            a=[dict(x,method=x['model']) for x in rows if x['dataset']==ds and x['horizon']==h and x['seed']==0]
            if not b and not a:continue
            summary += [f'\n## {ds} L=H={h}', '|Method|RetMSE|Stage1 MSE|Stage1 MAE|Recall|Regret|Last Skill|', '|---|---:|---:|---:|---:|---:|---:|']
            for x in b+a:
                summary.append('|'+x['method']+'|'+'|'.join('—' if x.get(k) is None else f'{x[k]:.6f}' for k in ['retmse','agg_mse','agg_mae','recall','regret','last_skill'])+'|')
    summary += ['\n## 해석과 검증 범위',
        '- RetMSE와 Oracle Regret은 같은 Oracle 항을 공유하므로 독립적인 두 증거가 아니다.',
        '- V0→V1은 projection 추가 효과이며 V1→V2/V5가 head 수 비교다. 파라미터 수 증가도 포함하므로 순수 head 수 효과로 단정하지 않는다.',
        '- Best Single 및 subset은 validation에서 선택했다. V5 내부 subset과 독립 학습 V1/V2는 별개다.',
        '- Last Skill 분모는 동일 query/channel 전체의 Last-repeat 평균 MSE다. Random 변동은 20회 추출 변동이며 학습 seed 표준편차와 다르다.',
        '- 채널별 validation threshold로 입력 변동성 strata를 정했다. Last-repeat 미래오차 strata는 비배포 사후 진단이다.',
        '- Query CI는 채널평균 후 길이 2H 시간블록을 사용한다. 유효 full block이 5개 미만이면 CI를 제시하지 않는다. 다중비교는 탐색적이며 seed 차이와 구분한다.',
        '- Weather 원본의 -9999/scaler 문제를 보존했다. 정제 데이터 결과로 해석할 수 없다.',
        '- 전체 데이터셋 원시 MSE를 합쳐 개선을 주장하지 않는다. 모든 조건·3seed가 완료되기 전 일반화 결론은 미확정이다.',
        '\n## 산출물',
        'MAIN_SEED_RESULTS.csv, BASELINES.csv, THREE_SEED_MEAN_STD.csv, HEAD_QUALITY.csv, BEST_SINGLE_VALIDATION_SELECTED.csv, LEAVE_ONE_OUT.csv, HEAD_COUNT_VALIDATION_SELECTED.csv, K_SENSITIVITY.csv, DIVERSITY.csv, QUERY_PAIR_SUMMARIES.json. 각 run에 checkpoint 선정/해시, epoch별 기록 및 channel별 압축 query 지표·후보ID가 있다.']
    (OUT/'FINAL_REPORT_KO.md').write_text('\n\n'.join(summary)+'\n')
    # Static figures only from verified completed data.
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
        for ds,h in sorted({(x['dataset'],x['horizon']) for x in rows}):
            cellrows=[x for x in rows if x['dataset']==ds and x['horizon']==h and x['seed']==0]
            if not cellrows:continue
            fig,axes=plt.subplots(1,2,figsize=(8,3))
            for ax,key in zip(axes,['retmse','recall']):
                ax.plot([int(x['model'][1:]) for x in cellrows],[x[key] for x in cellrows],'o-');ax.set(xlabel='Independent model heads (V0 has no projection)',ylabel=key)
            fig.tight_layout();fig.savefig(figdir/f'{ds}_{h}_head_count.png',dpi=160);plt.close(fig)
            vals=[dict(x,method=x['model']) for x in cellrows]+[x for x in baselines if x['dataset']==ds and x['horizon']==h and 'standard deviation' not in x['method']]
            fig,ax=plt.subplots(figsize=(9,4));ax.bar([x['method'] for x in vals],[x['agg_mse'] for x in vals]);ax.set_ylabel('Stage1 output MSE');ax.tick_params(axis='x',rotation=35)
            fig.tight_layout();fig.savefig(figdir/f'{ds}_{h}_output_mse.png',dpi=160);plt.close(fig)
        for d in div:
            if d['seed']!=0 or d['model']!='V5':continue
            mat=np.eye(5)
            for key,value in d['pair_overlap'].items():
                a,b=[int(x[1:])-1 for x in key.split('_')];mat[a,b]=mat[b,a]=value
            fig,ax=plt.subplots();im=ax.imshow(mat,vmin=0,vmax=1);fig.colorbar(im,ax=ax);ax.set_title(f'{d["dataset"]} {d["horizon"]} V5 overlap')
            fig.savefig(figdir/f'{d["dataset"]}_{d["horizon"]}_overlap.png',dpi=160);plt.close(fig)
    print(f'SUMMARY complete={len(rows)}/144 three_seed_cells={len(seedrows)}',flush=True)

if __name__=='__main__':main()
