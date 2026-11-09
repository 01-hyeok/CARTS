"""No GPU use. Summarize only hash-verified completed checkpoint evaluations."""
import csv,json,hashlib,math,datetime
from pathlib import Path
import numpy as np
from fusion_config import ROOT,HERE,OUT,CELLS,PROTOCOL

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def csvwrite(name,rows,fields=None):
    cols=fields or list(dict.fromkeys(k for row in rows for k in row))
    p=OUT/name;t=p.with_suffix('.tmp')
    with t.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    t.replace(p)
def ratio(a,b):return float(a/b) if b!=0 else None
def mdtable(rows,cols):
    if not rows:return '아직 완료된 결과가 없습니다.\n'
    def val(v):return ('%.6f'%v) if isinstance(v,float) else str(v if v is not None else 'UNDEFINED/MISSING')
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(val(r.get(c)) for c in cols)+' |' for r in rows)+'\n'

def main():
    audit=json.loads((OUT/'CHECKPOINT_AUDIT.json').read_text());metrics=[];winloss=[];heads=[];resources=[];gamma_rows=[];stabilities=[];reproductions=[];complete=[]
    native={s:[] for s in ('FULL','P100')};cross=[]
    for cell in CELLS:
        for training in ('FULL','P100'):
            dest=OUT/'checkpoints'/cell/('V5_'+training);gp=dest/'COMPLETE.json'
            if not gp.exists():continue
            g=json.loads(gp.read_text());assert g['status']=='PASS_COMPLETE_ALL_QUERY_SCORE_FUSION'
            gamma=g['gamma'];row=g['identity']['checkpoint'];ds,h=cell.rsplit('_',1)
            sel=json.loads((dest/'GAMMA_SELECTION.json').read_text())
            for r in sel['rows']:gamma_rows.append(dict(dataset=ds,horizon=int(h),**r,selected=(r['gamma']==gamma),checkpoint_sha256=row['checkpoint_sha256']))
            rep=json.loads((dest/'REPRODUCTION_GATE.json').read_text());reproductions.append(dict(cell=cell,training=training,status=rep['status'],path=str(dest/'REPRODUCTION_GATE.json'),sha256=sha(dest/'REPRODUCTION_GATE.json'),online_all_query_e0_ordered_mismatches=g['e0_ordered_id_mismatches']))
            chunks={};randoms={s:[] for s in ('FULL','P100')};oracles={s:[] for s in ('FULL','P100')};stab={};blockwall=0.
            methods=['E0','E1','E2','E3','E4_g'+str(gamma)]
            for rel,digest in g['files'].items():
                p=dest/rel;assert sha(p)==digest
                meta=json.loads(p.with_suffix('.json').read_text());blockwall+=meta['seconds']
                if not rel.startswith('test/'):continue
                with np.load(p) as z:
                    for s in ('FULL','P100'):
                        randoms[s].append(z[s+'__random_mse'].astype(float).mean(-1));oracles[s].append(z[s+'__oracle_mse'].astype(float).mean(-1))
                        for method in methods:
                            for metric in ['retmse','oracle_gap','recall','ndcg10','overlap_e0','useful_available','useful_retained','exclusive_available','exclusive_retained']:
                                key=(s,method,metric);chunks.setdefault(key,[]).append(z[s+'__'+method+'__'+metric].copy())
                    for key in z.files:
                        if key.startswith('STABILITY__') and any(key.startswith('STABILITY__'+m+'__') for m in methods):stab.setdefault(key,[]).append(z[key].copy())
            arrays={k:np.concatenate(v) for k,v in chunks.items()}
            assert all(v.shape==(g['queries']['test'],g['channels']) for v in arrays.values())
            for s in ('FULL','P100'):
                random=float(np.concatenate(randoms[s]).mean());oracle=float(np.concatenate(oracles[s]).mean());base=arrays[(s,'E0','retmse')].mean(1)
                comparison=dict(dataset=ds,horizon=int(h),training_support=training,evaluation_support=s,Random=random,Oracle=oracle,queries=g['queries']['test'],channels=g['channels'],selected_epoch=row['selected_epoch'],gamma=gamma,checkpoint_sha256=row['checkpoint_sha256'])
                for method in methods:
                    method_name='E4' if method.startswith('E4') else method
                    shared=dict(dataset=ds,horizon=int(h),training_support=training,evaluation_support=s,method=method_name,gamma=gamma if method_name=='E4' else '',checkpoint_sha256=row['checkpoint_sha256'],queries=g['queries']['test'],channels=g['channels'])
                    r=arrays[(s,method,'retmse')];mean=float(r.mean());comparison[method_name]=mean
                    metrics.append(dict(**shared,RetMSE=mean,OracleGap=mean-oracle,RandomOracleRecovery=ratio(random-mean,random-oracle),Recall10=float(arrays[(s,method,'recall')].mean()),NDCG10=float(arrays[(s,method,'ndcg10')].mean()),Top10OverlapE0=float(arrays[(s,method,'overlap_e0')].mean())))
                    delta=r.mean(1)-base;ordered=np.sort(delta)[::-1]
                    w=dict(**shared,win=int((delta<0).sum()),tie=int((delta==0).sum()),loss=int((delta>0).sum()),mean_delta=float(delta.mean()),worst_query_delta=float(delta.max()),best_query_delta=float(delta.min()))
                    for pct in (1,5,10):w['worst_%d_percent_mean_delta'%pct]=float(ordered[:max(1,math.ceil(len(delta)*pct/100))].mean())
                    winloss.append(w)
                    for kind in ('useful','exclusive'):
                        den=arrays[(s,method,kind+'_available')];num=arrays[(s,method,kind+'_retained')];ok=den>0
                        heads.append(dict(**shared,kind=kind,available=int(den.sum()),retained=int(num.sum()),pooled_retention=ratio(num.sum(),den.sum()),mean_defined_unit_retention=float((num[ok]/den[ok]).mean()) if ok.any() else None,undefined_units=int((~ok).sum())))
                if s==training:
                    for arm in ('V0','V1'):
                        c=next(x for x in audit['comparators'] if x['cell']==cell and x['training_support']==s and x['architecture']==arm)
                        comparison[arm]=c.get('retmse','MISSING');comparison[arm+'_status']=c['status']
                    native[s].append(comparison)
                else:cross.append(comparison)
            for key,parts in stab.items():
                _,method,metric=key.split('__');values=np.concatenate(parts)
                stabilities.append(dict(dataset=ds,horizon=int(h),training_support=training,method=method,metric=metric,mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),queries=g['queries']['test'],channels=g['channels'],definition='actual native kernels' if 'shared' not in metric else 'pure support control: identical Full logits'))
            resources.append(dict(dataset=ds,horizon=int(h),training_support=training,queries_val=g['queries']['val'],queries_test=g['queries']['test'],channels=g['channels'],full_candidates=g['full_candidate_count'],block_wall_seconds=blockwall,invocation_seconds=g['elapsed_this_invocation_seconds'],smoke_seconds=json.loads((dest/'GPU_SMOKE_GATE.json').read_text())['seconds'],peak_allocated_bytes=g['peak_allocated_bytes'],peak_reserved_bytes=g['peak_reserved_bytes'],observed_end_used_mib=g['observed_gpu']['used_mib'],gpu_uuid=g['observed_gpu']['uuid'],utilization_average='UNMEASURED',**g['timings']))
            complete.append(dict(cell=cell,training=training,gate=str(gp),sha256=sha(gp)))
    csvwrite('FULL_NATIVE_TEST_RETMSE.csv',native['FULL']);csvwrite('P100_NATIVE_TEST_RETMSE.csv',native['P100']);csvwrite('CROSS_SUPPORT_TEST_RETMSE.csv',cross)
    csvwrite('ALL_TEST_METRICS.csv',metrics);csvwrite('VALIDATION_GAMMA_SELECTION.csv',gamma_rows);csvwrite('HEAD_USEFUL_CANDIDATE_RETENTION.csv',heads)
    csvwrite('QUERY_LEVEL_WIN_LOSS.csv',winloss);csvwrite('SUPPORT_RANK_STABILITY.csv',stabilities);csvwrite('RESOURCE_SUMMARY.csv',resources)
    dump(OUT/'REPRODUCTION_GATE.json',dict(status='PASS_ALL_16' if len(complete)==16 else 'PARTIAL',complete_checkpoints=len(complete),rows=reproductions))
    done=len(complete)==16;cols=['dataset','horizon','Random','V0','V1','E0','E1','E2','E3','E4','Oracle','gamma']
    lines=['# Multi-Head Score Fusion Ablation 결과',f'검증 완료 checkpoint {len(complete)}/16; 평가 조합 {2*len(complete)}/32. '+('전체 완료.' if done else '미완료 조건은 결과를 채우지 않았습니다.'),
      '전체 validation/test query와 모든 채널을 사용합니다. Frozen V5 동일 head cosine의 결합만 변경했습니다. E4는 native validation에서만 선택하고 cross-support에 고정했습니다. Stage2 및 새 학습은 없습니다.',
      '## Full Native',mdtable(native['FULL'],cols),'## P100 Native',mdtable(native['P100'],cols),'## Cross-support',mdtable(cross,['dataset','horizon','training_support','evaluation_support','E0','E1','E2','E3','E4','Oracle','gamma']),
      '## 관찰과 해석 한계']
    for s in ('FULL','P100'):
        rows=native[s]
        for method in ('E1','E2','E3','E4'):
            wins=sum(r[method]<r['E0'] for r in rows);v1=[r for r in rows if isinstance(r.get('V1'),(int,float))]
            lines.append(f'- {s} {method}: E0 대비 {wins}/{len(rows)} 조건 개선; 독립 V1 대비 {sum(r[method]<r["V1"] for r in v1)}/{len(v1)} 조건 개선. 테스트에서 가장 좋은 방법을 사후 선택한 수치는 배포 성능으로 해석하지 않습니다.')
    lines+=['- Max 또는 Mean의 개선은 해당 결합식의 관찰된 효과입니다. 개별 head의 미래 품질 보정 또는 인과적 실패 원인이 증명된 것은 아닙니다.',
      '- 고유 유용 후보는 정확히 한 head가 확보한 동일-support Oracle10 후보입니다. 유지율의 분모 0은 undefined이며 임의로 0이나 1로 채우지 않습니다.',
      '- Support-independent E1–E4는 동일 cosine의 공통 후보 점수가 동일합니다. 실제 native P100 encoder와 Full encoder의 kernel shape 수치 차이는 별도 actual-native rank 안정성으로 보존합니다.',
      '- E0는 기존 FP32 공식 경로, E1–E4 결합은 FP64입니다. GPU smoke에 FP32/FP64 Top10 차이와 near-tie gap을 기록합니다. 부동소수점 차이를 후보 수 축소로 해결하지 않았습니다.',
      '- 기존 전체-query 결과의 checkpoint·data·pool·source hash를 검증하여 Oracle/Random과 E0 근거를 재사용했습니다. 현재 각 inference block에서 E0 ID/저장 점수/선택 후보 MSE를 다시 검사합니다. 역사적 생성 runtime은 UNKNOWN입니다.',
      '- 하나의 seed 결과이며 seed 일반화나 유의성 검정은 제공하지 않습니다. Head calibration 또는 새로운 점수 학습의 유효성은 여기서 검증하지 않았습니다. 다음 학습 연구의 우선순위는 조건 일관성·V1 비교·Oracle 격차와 함께 판단해야 합니다.',
      '## Support 확대 시 변화']
    for r in cross:
        nr=next(x for x in native[r['training_support']] if x['dataset']==r['dataset'] and x['horizon']==r['horizon'])
        lines.append(f'- {r["dataset"]} H{r["horizon"]} {r["training_support"]}→{r["evaluation_support"]}: '+', '.join(m+' ΔRetMSE=%+.6f'%(r[m]-nr[m]) for m in ('E0','E1','E2','E3','E4')))
    lines+=['## 자원과 출처',mdtable(resources,['dataset','horizon','training_support','block_wall_seconds','smoke_seconds','peak_allocated_bytes','peak_reserved_bytes']),
      '시간은 wall-clock이며 GPU 연산시간이 아닙니다. 평균 GPU utilization은 미측정입니다. 원시 query×channel 지표·후보 ID는 checkpoints/<cell>/V5_<training>/{val,test}/batch*.npz에 저장되어 있습니다. 각 COMPLETE.json과 블록 JSON에 hash·query 수·checkpoint·pool·source가 연결됩니다.',
      '## 고정 규약',json.dumps(PROTOCOL,ensure_ascii=False,indent=2)]
    report='\n\n'.join(lines)+'\n'
    for name in ('FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md'):
        p=OUT/name;t=p.with_suffix('.tmp');t.write_text(report);t.replace(p)
    files={p.name:sha(p) for p in OUT.glob('*.csv')};files.update({n:sha(OUT/n) for n in ('FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md','REPRODUCTION_GATE.json')})
    dump(OUT/'FINAL_GATE.json',dict(status='PASS_ALL_32_COMBINATIONS' if done else 'PARTIAL',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_checkpoints=len(complete),evaluation_combinations=2*len(complete),complete=complete,files=files,missing=[dict(cell=c,training=s) for c in CELLS for s in ('FULL','P100') if not any(x['cell']==c and x['training']==s for x in complete)]))
    print('SUMMARY',len(complete),'/16 checkpoints',flush=True)
if __name__=='__main__':main()
