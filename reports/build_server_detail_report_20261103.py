from pathlib import Path
import csv, io, json, hashlib, datetime, math, collections

BASE=Path('/data/pjh_workspace/CARTS/reports/mechanism02_completed_20261103_v2')
ORIGINAL=Path('/data/pjh_workspace/CARTS/results/TRACK-RETRIEVAL-FAILURE-MECHANISM02')
OUT=Path('/data/pjh_workspace/CARTS/reports/mechanism02_detail_text_20261103')
CELLS=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96']
ABBR=dict(zip(CELLS,['E96','E192','E336','E720','W96']))
blocks=[]; data={}; inventory=[]; validation=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def put(s):blocks.append(s.strip()+'\n')
def fmt(x):
    if x is None or x=='':return '—'
    if isinstance(x,str) and x in ('True','False'):return x
    try:
        v=float(x)
        if not math.isfinite(v):return '미정의'
        return ('%.6f'%v)
    except (ValueError,TypeError):return str(x).replace('|','/')
def vec(values):return ', '.join(fmt(v) for v in values)
def rows(c,f):return data[c,f]
def allrows(f):return [r for c in CELLS for r in rows(c,f)]
def selected(c,f,**kw):return [r for r in rows(c,f) if all(r.get(k)==str(v) for k,v in kw.items())]
def one(c,f,**kw):
    rr=selected(c,f,**kw);assert len(rr)==1,(c,f,kw,len(rr));return rr[0]
def table(title,header,rr,note=''):
    # Split at row boundaries, repeat definitions/headers, never truncate a table.
    assert rr, ('empty generated table',title)
    prefix=title+'\n\n'+note+'\n\n' if note else title+'\n\n'
    h='| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'
    buf=prefix+h
    for r in rr:
        line='| '+' | '.join(str(x).replace('|','/') for x in r)+' |\n'
        if len(buf)+len(line)>10500:
            put(buf);buf=title+' (계속)\n\n'+h
        buf+=line
    put(buf)
def native(r):return (r.get('training_support')=='FULL' and r.get('variant')=='FULL') or (r.get('training_support')=='P100' and r.get('variant')=='P100_CONTROLLED')
def idcols(r):return [ABBR[r['cell']],r.get('training_support',r.get('support','')),r.get('model',''),r.get('variant','')]

def main(expected):
    assert not OUT.exists(),str(OUT)
    for item in expected:
        p=BASE/item['path'];raw=p.read_bytes();assert sha(raw)==item['sha256'],str(p)
        rr=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        assert len(rr)==item['rows'],str(p)
        stem=p.stem;c=p.parent.name
        if item['alias_of']:
            assert raw==(BASE/item['alias_of']).read_bytes();continue
        data[c,stem]=rr
        original=ORIGINAL/item['path']
        verified_original=False
        if original.exists():
            assert sha(original.read_bytes())==sha(raw),str(original)
            verified_original=True
        elif stem!='RANKING_SAMPLED_AGGREGATE_EXPORT':raise AssertionError(str(original))
        else:
            src=ORIGINAL/'cells'/c/'RANKING_QUALITY.csv'
            origsha=sha(src.read_bytes())
            assert all(r['source_sha256']==origsha for r in rr)
        inventory.append(dict(item,server_path=str(p),original_file_verified=verified_original))
    assert len(inventory)==135 and len(data)==135
    for c in CELLS:
        g=BASE/'cells'/c/'CELL_ARTIFACT_GATE.json';j=json.loads(g.read_text())
        assert j['status']=='CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING'
        validation.append(dict(cell=c,gate=str(g),sha256=sha(g.read_bytes()),status=j['status']))
    put('Mechanism02 완료 5조건 상세 해석 보고서\n\n서버 원본 CSV를 직접 읽어 작성한 고정 snapshot이다. 범위는 ETTh1 H96/H192/H336/H720와 Weather H96, seed0 Stage1이다. 195개 CSV 중 60개는 동일 바이트 별칭이며, 고유 135개(129개 비어 있지 않음)를 모두 읽고 SHA256·행 수를 검증했다. 기본 baseline 성능표는 앞서 전달했으므로 반복하지 않는다. 이 본문은 모든 지표 family와 주요 수치·직접 head 지표를 포함하지만, 모든 상세 원시 행을 전재한 문서는 아니다. 전체 컬럼·정의·행 수·원본 hash는 INVENTORY.json, 끝부분의 135파일 목록에 남긴다. 원본 숫자 문자열은 CSV에서 보존한다.\n\n먼저 두 누락을 명시한다. ② 각 head의 좋은 후보가 All5에서 탈락한 후보별 개수·회복 가능한 손실은 부분 구현이다. union Oracle 차이를 그 직접 손실로 해석하면 안 된다. ④ 후보를 시간축으로 이동시켜 미래 MSE가 줄어드는지 평가하는 temporal shift 분석은 미구현이다. delta-last는 마지막 값에 대한 수직 정렬이다.\n\n5셀 gate는 CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING이다. 전체 연구 완료/GO, 남은 Weather 3조건 완료, 인과효과 입증을 뜻하지 않는다. 실행 중 진단 및 Solar 프로세스·결과는 이 보고서 작성에서 변경하지 않았다.')
    cov=[]
    for c in CELLS:
        item=next(x for x in inventory if x['cell_coverage'] and '/'+c+'/' in x['path'])
        cv=item['cell_coverage'];bank={'ETTh1_96':8449,'ETTh1_192':8257,'ETTh1_336':7969,'ETTh1_720':7201,'Weather_96':36696}[c]
        cov.append([ABBR[c],cv['validation_queries'],cv['test_queries'],cv['channels'],bank,cv['test_queries']*cv['channels']])
    table('표기와 평가 범위',['조건','val Q','test Q','C','Full bank N','test Q×C'],cov,'P=P100, F=Full 학습 support. 평가 variant FULL은 전체 유효 후보, P100_CONTROLLED는 같은 checkpoint의 Full-logit을 동일 P100으로 제한하여 재정규화한다. P100_NATIVE와는 수치 경로가 달라 별도로 검증한다. 모델 사이 학습 support 차이와 같은 checkpoint의 평가 support 개입을 혼합하지 않는다.')
    put('공통 정의: train만으로 적합한 정규화 공간, 역변환 없음, 전체 horizon, delta-last future reconstruction을 사용한다. RetMSE는 선택 10개 후보의 개별 future MSE 평균을 Q×C 전체에서 평균한다. AggMSE/MAE는 선택 미래 10개의 균등 평균 예측 오차다. Oracle은 같은 masked support에서 개별 MSE 최소 10개이며, 평균 예측의 최적 조합이라는 뜻이 아니다. 표는 6자리 반올림이며 미세한 비영 값이 0.000000으로 보일 수 있다. 상세 CSV가 기준이다.')

    put('질문 ① — 다른 head가 놓친 유용 후보의 고유 기여\n\nhead h의 Top10 중 다른 4개 head Top10에 없는 ID가 unique이다. useful은 d_i≤Oracle10의 10번째 MSE+ε_c, ε_c=max(1e-8,0.01×validation Oracle 경계 MSE 중앙값)이며 test 전에 채널별 고정했다. exclusive Oracle은 다른 head에 없는 정확한 Oracle10 ID다. 세 값 모두 Q×C당 평균 개수다. 아래 H1…H5 벡터는 순서가 고정되어 있다. pair overlap=모든 10개 head 쌍의 Top10 교집합/10 평균, union=5개 Top10 합집합 크기, signed cosine은 해당 head의 예측 signed error와 다른 4head error cosine 평균이다. 고유 후보 존재가 All5 선택·예측 개선을 보장하지 않는다. 출처 HEAD_UNIQUE_RELEVANCE.csv 100행 전체를 피벗하였다.')
    unique=[]
    for c in CELLS:
      for tr in ['P100','FULL']:
       for v in ['FULL','P100_CONTROLLED']:
        rr=sorted(selected(c,'HEAD_UNIQUE_RELEVANCE',training_support=tr,variant=v),key=lambda r:int(r['head']));assert len(rr)==5
        unique.append([ABBR[c],tr,v,vec(r['unique_count'] for r in rr),vec(r['unique_near_optimal_count'] for r in rr),vec(r['exclusive_oracle_count'] for r in rr),fmt(rr[0]['pair_overlap']),fmt(rr[0]['union_size']),vec(r['signed_error_cosine'] for r in rr)])
    table('① 고유 후보 및 오류 보완성 — 전체 20조합',['조건','학습','평가','unique H1…H5','useful unique H1…H5','exclusive Oracle H1…H5','pair overlap','union','signed error cosine H1…H5'],unique)
    put('해석: native Full V5의 useful unique 총합은 E96 1.236522, E192 1.245444, E336 0.530059, E720 0.758842, W96 1.586475개/Q×C다. W96은 union 약49.97개로 거의 겹치지 않지만 exact exclusive Oracle 총합은0.374378개다. 다양성 증가와 미래 관련성 증가를 동일시할 수 없다. E336의 head별 고유 후보가 존재하더라도 아래 제거 분석에서는 H2 제거가 오히려 유리하다.')
    put('질문 ② — 좋은 head 후보가 평균 결합에서 탈락하는 손실: 부분 구현\n\nAll5는 각 head 확률을 평균한 뒤 Top10을 다시 고른다. head 예측값 평균이 아니다. ValidationBestSingle은 validation RetMSE 최소 head를 고정한 test 값; POSTHOC_ORACLE_HEAD는 전체 test Q×C의 평균 RetMSE로 단일 head를 사후 선택하는 진단이라 배포 성능과 분리한다. WithoutHh는 해당 head를 빼고 남은 4분포를 재평균·재선택한 추론 ablation이며 재학습이 아니다. DeployableUnionK10은 head Top10 union에서 원래 전체 support mixture score로 10개를 선택한다. OracleUnionK10은 그 union에서 미래 MSE로 고른 비배포 진단이다. equal K=10이지만 계산량은 동일하지 않다. 다음 HEAD_UTILITY.csv 300행은 모두 포함하며 H/Without 벡터의 원소 순서는 H1…H5다.')
    for c in CELLS:
      u=[]
      for tr in ['P100','FULL']:
       for v in ['FULL','P100_CONTROLLED']:
        rr=selected(c,'HEAD_UTILITY',training_support=tr,variant=v)
        ix={r['method']:r for r in rr};assert len(ix)==15
        for method in ['All5','ValidationBestSingle','POSTHOC_ORACLE_HEAD','DeployableUnionK10','OracleUnionK10_DIAGNOSTIC']:
            r=ix[method];u.append([tr,v,method,r['validation_selected_head']]+[fmt(r[k]) for k in ['retmse','agg_mse','agg_mae','recall10','ndcg10','oracle_regret','delta_ret_vs_all5']])
        for prefix in ['H','WithoutH']:
            rrs=[ix[prefix+str(i)] for i in range(1,6)]
            u.append([tr,v,prefix+'1…5',rrs[0]['validation_selected_head']]+[vec(r[k] for r in rrs) for k in ['retmse','agg_mse','agg_mae','recall10','ndcg10','oracle_regret','delta_ret_vs_all5']])
      table('② '+ABBR[c]+' head 단독·결합·제거 전체 지표',['학습','평가','방법','val선택head','RetMSE','AggMSE','AggMAE','Recall10','NDCG10','Oracle regret','ΔRet−All5'],u)
    put('②의 확인 범위: All5−OracleUnion 차이는 union 안에 더 좋은 개별 후보가 얼마나 존재하는지 보여준다. All5 자체가 union 밖의 후보를 선택할 수 있고 두 선택의 후보집합도 달라, 이를 정확한 “좋은 후보 탈락 손실”로 치환할 수 없다. 미구현 직접 집계는 (head useful Top10∩All5), useful 후보의 All5 탈락률, 후보별 교환 후 실제 Ret/Agg 변화다. Full E336은 All5 Ret0.764649, validation single0.708939, H2 제거 ΔRet−0.062717로 결합 악화의 증거가 있으나 그 원인을 특정 탈락 후보까지 귀속하지 않았다. Full W96은 All5 .299062 vs single .367890로 결합 이득이 있으며 H5 제거 Δ+.064451이다. 조건별 효과가 다르다.')

    put('질문 ③ — P100/Full score normalization과 support 민감도\n\n동일 checkpoint·logits·P100 ID/순서·mask·temperature에서 softmax 분모만 바꾸었다. FP64 logsumexp 사용. m_h=Z_P100,h/Z_Full,h이면 Full-normalized P100 제한점수=(1/H)Σ_h m_h p_P100,h. P100 내 재정규화 후 암묵적 head 가중치는 m_h/Σ_h m_h다. 단일 head V0/V1은 분모만으로 상대순위가 바뀌면 안 되며 실제20개 대조에서 ordered 변화0, set일치1, ΔRet0이다. V5는 head별 m_h가 달라 순위가 변한다. ordered change와 set change는 다른 값이다.')
    table('③ 분모-only 전체30대조',['조건','학습','모델','ordered 변화율','set 동일율','ΔRet Full분모−local','mass 최소','mass 평균','mass 최대','weight 최소','weight 최대'],
        [idcols(r)[:3]+[fmt(r[k]) for k in ['changed_order_fraction','same_set_fraction','restricted_minus_local_retmse','mass_min','mass_mean','mass_max','implicit_weight_min','implicit_weight_max']] for r in allrows('NORMALIZER_CONTROL')])
    table('③ head별 확률 질량과 암묵적 가중치 — 70행 전체',['조건','학습','모델','head','mass 평균','mass P05','mass P95','weight 평균'],
        [idcols(r)[:3]+[r['head']]+[fmt(r[k]) for k in ['mass_mean','mass_p05','mass_p95','implicit_weight_mean']] for r in allrows('HEAD_PROBABILITY_MASS')])
    cross=[]
    for c in CELLS:
      for tr in ['P100','FULL']:
       for m in ['V0','V1','V5']:
        rr=selected(c,'CROSS_SUPPORT',training_support=tr,model=m);ix={r['variant']:r for r in rr}
        cross.append([ABBR[c],tr,m]+[fmt(ix[v]['retmse']) if v in ix else '—' for v in ['FULL','P100_CONTROLLED','P100_NATIVE','P100_LOCAL_NORMALIZER','P100_FULL_NORMALIZER']])
    table('③ 동일 checkpoint cross-support RetMSE — 30checkpoint 전체',['조건','학습','모델','FULL','P100 controlled','P100 native','P100 local 분모','P100 Full 분모'],cross)
    table('③ controlled vs native 수치경로 — P100 학습15대조',['조건','모델','Top10 overlap','ordered 동일율'],[[ABBR[r['cell']],r['model'],fmt(r['overlap10']),fmt(r['ordered_equal_fraction'])] for r in allrows('METHOD_OVERLAP')])
    put('③ 해석: W96 Full-trained V5의 RetMSE는 Full .299062 → P100 controlled .669230이다. 같은 P100 logits의 분모만 Full로 유지하면 .417833으로 줄지만 Full .299062에는 미치지 못한다. 분모 변화와 후보 support 제한이 모두 관여한다. 분모-only ΔRet는 E96 Full-trained −.002508, E192 −.001446, E336 +.006439, E720 +.004384, W96 −.251397로 방향도 조건 의존적이다. 다른 checkpoint의 native P100/Full 차이를 이 개입의 순수효과로 해석하지 않는다. P100 controlled/native의 near-tie 차이는 위 overlap에 남기며 같다고 합치지 않는다.')

    put('질문 ④ — 놓친 Oracle 후보의 temporal shift: 미구현\n\n현재 코드는 zero-lag delta-last future MSE, missed Oracle의 검색 순위, past-only 통계와 calendar 비교를 제공한다. lag별 future MSE, 최적 lag, 이동 후 감소량, shift-aware Oracle/Recall은 없다. 아래 rank는 시간 이동량이 아니다. 따라서 “놓친 후보는 시간상 밀려 있어서 실패했다”는 결론은 INCONCLUSIVE다. 원인 규명을 위해서는 별도 사전고정 lag 범위·경계/겹침 처리·비배포 oracle 진단이 필요하며 이번 보고 작성에서는 실행하지 않았다.')
    table('④ 대체할 수 없는 기존 missed-rank 증거 — native V5',['조건','학습','평가','missed pooled N','평균rank','중앙rank','P90','11–20 비율','21–50','51–100','101–500','501–1000','1001+'],
       [[ABBR[r['cell']],r['training_support'],r['variant'],r['missed_count']]+[fmt(r[k]) for k in ['pooled_mean','pooled_median','pooled_p90','rank_11_20_fraction','rank_21_50_fraction','rank_51_100_fraction','rank_101_500_fraction','rank_501_1000_fraction','rank_1001_1000000000_fraction']] for r in allrows('MISSED_ORACLE') if r['model']=='V5' and native(r)])
    put('아래부터는 네 질문 외 모든 지표 family의 정의·주요값과 한계다. 원시 상세행 전부의 재출력이 아니라 사전 선택한 native 조건의 핵심 비교와 유효성 확인이다. 직접 head 분석은 앞부분에 전체 값을 실었다.')

    reg=[r for r in allrows('REGRET_DECOMPOSITION') if r['variant']=='P100_CONTROLLED']
    table('Regret 분해 — 같은 P100 내 선택30조합',['조건','학습','모델','pool','ranking','total','pool share','ranking share','Full O10 P100 coverage','Full O10 recall'],
       [idcols(r)[:3]+[fmt(r[k]) for k in ['pool','ranking','total','pool_share','ranking_share','full_oracle_coverage100','recall_to_full_oracle10']] for r in reg],
       'pool=O_P100−O_Full; ranking=M_P100−O_P100; total=M_P100−O_Full. Q×C별 항등식과 부등식을 검증하며 음수 clamp하지 않는다. share=집계 regret의 비율, total≈0이면 미정의. Full 모델 선택에는 O_P100≤M_Full을 강제하지 않는다.')
    table('Regret tail — V5 P100_CONTROLLED ranking 손실',['조건','학습','mean','median','P90','P95','P99','최악1% share','5%','10%'],
       [[ABBR[r['cell']],r['training_support']]+[fmt(r[k]) for k in ['mean','median','p90','p95','p99','worst_0.01_share','worst_0.05_share','worst_0.1_share']] for r in allrows('REGRET_TAILS') if r['model']=='V5' and r['variant']=='P100_CONTROLLED' and r['regret']=='ranking'],
       '채널을 먼저 평균한 timestamp별 regret의 분위수. 최악1%는 timestamp 단위이고, 채널·중첩 window를 독립표본으로 취급하지 않는다.')
    table('Top-K 경계 및 교환 손실 — native V0/V1/V5',['조건','학습','모델','ranking','exchange','Oracle10/11 gap','score10/11 gap','score1/10 gap','missed 평균score','false 평균score'],
       [idcols(r)[:3]+[fmt(r[k]) for k in ['ranking_regret','exchange_regret','oracle_boundary_gap','student_boundary_gap','student_top1_top10_gap','missed_mean_score','false_mean_score']] for r in allrows('CROSS_SUPPORT') if native(r)],
       'Exchange=[Σfalse d−Σmissed d]/10이며 같은 support ranking regret과 동일해야 한다. score 크기는 모델별 정의가 다르므로 모델 간 절대값 비교는 피한다. 작은 Oracle 경계 gap만으로 손실이 작다고 판단하지 않는다.')
    rank=[]
    for c in CELLS:
      for tr in ['P100','FULL']:
       for m in ['V0','V1','V5']:
        v='FULL' if tr=='FULL' else 'P100_CONTROLLED'
        rr=selected(c,'RANKING_SAMPLED_AGGREGATE_EXPORT',training_support=tr.upper(),model=m+'_'+tr.upper(),variant=v,split='test')
        for r in rr:
            rank.append([ABBR[c],tr,m,r['subset'],r['sampled_timestamps'],r['channels'],r['spearman_finite_count'],fmt(r['spearman_mean']),r['kendall_finite_count'],fmt(r['kendall_mean'])])
    table('Global/Top-K 순위 상관 — native test',['조건','학습','모델','subset','시점N','C','Spearman 유효N','평균ρ','Kendall 유효N','평균τ'],rank,
       '각 split 최대256개 고정 timestamp, 모든 채널. ALL_VALID, STUDENT_TOP50/100, ORACLE_TOP100의 집합 정의가 다르다. 전체 Q 커버리지의 상관이라고 쓰지 않는다. 대규모 Full Kendall은4096고정쌍 근사이며 APPROXIMATE다. finite N은 상수/동점 등 undefined 제외수다.')
    table('NDCG 및 Recall의 서로 다른 목표집합 — native V5',['조건','학습','NDCG10','NDCG50','NDCG100','Recall50/O50','Recall100/O100','O10coverage50','O10coverage100','Last win','Last tie'],
       [[ABBR[r['cell']],r['training_support']]+[fmt(r[k]) for k in ['ndcg10_inverse_mse','ndcg50_inverse_mse','ndcg100_inverse_mse','recall50_vs_oracle50','recall100_vs_oracle100','oracle10coverage50','oracle10coverage100','last_win','last_tie']] for r in allrows('CROSS_SUPPORT') if r['model']=='V5' and native(r)],
       'gain=1/(1+d)≥0, 지수변환 없음, log2 discount, IDCG=전체 support의 이상순위. Recall@50/100는 각각 O50/O100, O10coverage는 고정 O10 분모10이다. Last win/tie는 검색 평균예측 MSE와 Last-repeat MSE의 Q×C 비교다.')
    table('Baseline 대비 query별 승률 — V5 vs Raw/Pearson/Random',['조건','support','baseline','Ret win','Ret tie','ΔRet','Agg win','ΔAgg'],
       [[ABBR[r['cell']],r['support'],r['baseline']]+[fmt(r[k]) for k in ['ret_win_fraction','ret_tie_fraction','ret_mean_difference','agg_win_fraction','agg_mean_difference']] for r in allrows('BASELINE_QUERY_WIN_RATE') if r['model']=='V5' and r['baseline'] in ['Raw Cosine','Pearson','Random']],
       '같은 Q×C/support/mask, Δ=model−baseline. 평균 개선과 대다수 query 승리는 다를 수 있다. 기본 평균성능표는 앞서 전달한 표 참조.')
    table('V5 대 독립 V1 — query별 불일치',['조건','학습','평가','Top10 overlap','V5 Ret win','tie','ΔRet 평균','Δ P05','Δ P95'],
       [idcols(r)[:2]+[r['variant']]+[fmt(r[k]) for k in ['top10_overlap','ret_win_fraction','ret_tie_fraction','mean_ret_difference','difference_p05','difference_p95']] for r in allrows('MODEL_DISAGREEMENT') if r['comparison']=='V5 vs V1'],
       '독립 V1과 V5 내부 single은 다른 모델이다. 기존 Full checkpoint 학습 예산/선택 provenance가 완전히 동일하지 않아 head수만의 인과효과라고 부르지 않는다.')
    table('V5−V1 paired CI — primary블록',['조건','support','지표','평균차','CI low','CI high','유효블록','충분?'],
       [[ABBR[r['cell']],r['support'],r['metric']]+[fmt(r[k]) for k in ['mean','ci_low','ci_high']]+[r['effective_blocks'],r['sufficient_blocks']] for r in allrows('PAIRED_TIME_BLOCK_CI') if r['comparison']=='V5 minus V1' and r['block_rule']=='primary'],
       '2000회 seed0, timestamp 연속블록에서 모든 채널을 함께 재표집한다. primary L+H=2H, 민감도H/4H도 원본에 포함. 4블록 미만은 불충분이며 H720 구간을 유의성 증거로 쓰지 않는다. 다중비교 exploratory, 학습 seed 변동 추정이 아니다.')

    table('Near-Oracle 밀도와 exact Recall — native V5',['조건','학습','near 후보수','선택near율','exact Recall10','Oracle경계MSE','선택MSE P50','P90','선택−경계','Recall/Ret 상관','Oracle경계gap'],
       [[ABBR[r['cell']],r['training_support']]+[fmt(r[k]) for k in ['near_optimal_count','selected_near_fraction','exact_recall10','oracle_kth_mse','selected_mse_p50','selected_mse_p90','selected_minus_oracle_kth','query_channel_recall_ret_correlation','oracle_boundary_gap']] for r in allrows('NEAR_ORACLE_CANDIDATE_ANALYSIS') if r['model']=='V5' and native(r)],
       'near-optimal ε는 validation 고정. Recall 낮음은 ID 일치 낮음이지 곧 미래 MSE가 크다는 뜻은 아니다. WEATHER_RECALL_DIAGNOSTIC.csv는5개 모두 빈 placeholder이며 이 표/채널분석이 제공하는 근거와 구분한다.')
    table('Weather96 채널구성 민감도 — native 모델',['학습','모델','채널group','C','Ret','Agg','MAE','Recall','ranking regret','공식?'],
       [[r['training_support'],r['model'],r['group'],r['channels']]+[fmt(r[k]) for k in ['retmse','agg_mse','agg_mae','recall10','ranking_regret']]+[r['official']] for r in rows('Weather_96','WEATHER_CHANNEL_SENSITIVITY') if native(r)],
       'train에서 -9999 sentinel이 나타난 채널 분류를 사용한다. 정제 scaler/재학습 결과가 아니다. 제외평균은 진단용이고 공식 전체21채널 성능을 대체하지 않는다.')
    channel=[]
    for c in CELLS:
      for tr in ['P100','FULL']:
        rr=[r for r in rows(c,'CHANNEL_METRICS') if r['model']=='V5' and r['training_support']==tr and native(r)]
        lo=min(rr,key=lambda r:float(r['retmse']));hi=max(rr,key=lambda r:float(r['retmse']))
        channel.append([ABBR[c],tr,len(rr),lo['channel'],fmt(lo['retmse']),hi['channel'],fmt(hi['retmse']),fmt(sum(float(r['retmse']) for r in rr)/len(rr))])
    table('채널별 이질성 — native V5',['조건','학습','C','최저채널','Ret 최소','최고채널','Ret 최대','채널macro Ret'],channel,'선택된 최저/최고 채널은 사후 기술통계다. 전체 채널 상세값은 CHANNEL_METRICS.csv. 채널 수가 많은 데이터셋의 원시 MSE를 합쳐 전체 우위를 주장하지 않는다.')

    feat=['raw_cosine','delta_last_cosine','pearson','paa16_cosine','diff1_cosine','history_mse','znorm_mse','trend_difference','std_difference','acf1_difference','dominant_period_difference']
    table('MissedOracle−FalseSelected 과거 특성 — native Full V5',['조건','feature','차이','CI low','CI high','표준화효과','finite Q×C','유효블록','충분?'],
       [[ABBR[r['cell']],r['feature']]+[fmt(r[k]) for k in ['mean','ci_low','ci_high','standardized_paired_effect']]+[r['finite_units'],r['effective_blocks'],r['sufficient_blocks']] for r in allrows('FEATURE_PAIRED') if r['model']=='V5' and r['training_support']=='FULL' and r['variant']=='FULL' and r['block_rule']=='primary' and r['feature'] in feat],
       '원본은35특성×두학습support×두평가support×3모델×3블록규칙. query내 group평균의 paired 차이. 35특성 중공통 query특성은두group에서같아차이0이될수있다. 미래를 score입력으로쓰지않으며 결과의상관을원인으로단정하지않는다. 주기차이는lag가아니다.')
    cal=[]
    for r in allrows('CALENDAR_FEATURES'):
        if r['model']=='V5' and r['training_support']=='FULL' and r['variant']=='FULL':
            cal.append([ABBR[r['cell']],r['group'],r['feature'],fmt(r['mean']),r['finite_units'],fmt(r['ci_low']),fmt(r['ci_high']),r['sufficient_blocks']])
    table('실제 timestamp calendar 진단 — Full V5',['조건','group','feature','평균/차이','finite N','CI low','CI high','충분?'],cal,
       '실제 history마지막timestamp의hour주기유사도/weekday/weekend/month. 존재하는timestamp만사용;future입력없음. group평균과paired차이의행정의를구분한다. 이것도time-shift future alignment가아니다.')
    strata=[]
    for c in CELLS:
      groups=collections.defaultdict(list)
      for r in rows(c,'INPUT_STRATA_BY_CHANNEL'):
        if r['model']=='V5' and r['aggregation']=='within-channel':groups[r['support'],r['feature'],r['stratum']].append(r)
      for (s,f,t),rr in sorted(groups.items()):
        n=sum(int(r['queries']) for r in rr)
        strata.append([ABBR[c],s,f,t,n]+[fmt(sum(float(r[k])*int(r['queries']) for r in rr)/n) for k in ['retmse','agg_mse','agg_mae','recall10']])
    table('입력-only strata — V5, 실제 Q×C수로 재집계',['조건','support','특성','stratum','Q×C','Ret','Agg','MAE','Recall'],strata,
       '경계는 validation 고정, test 입력만으로 group배정. 채널별 strata크기가달라 mean의단순평균대신query수로가중. 이가중평균은조건전체채널을제외하지않는다.')
    table('Observed past–future ambiguity — native V5',['조건','학습','regret group','N','ranking regret','past-nearest future variance'],
       [[ABBR[r['cell']],r['training_support'],r['regret_group'],r['units'],fmt(r['ranking_regret']),fmt(r['past_nearest_future_variance'])] for r in allrows('AMBIGUITY_NEW_MODELS') if r['model']=='V5' and native(r)],
       'regret구간은validation고정이나regret자체미래정답을사용하는사후진단이다. 위input-onlystrata와구분한다. 이분산만으로모든encoder에서해결불가능한irreducibleuncertainty를주장하지않는다.')
    table('Past-only frozen Ridge probe — 30비교 전체',['조건','support','모델','probe Ret','model Ret','probe−model','train시점','C'],
       [[ABBR[r['cell']],r['support'],r['model']]+[fmt(r[k]) for k in ['probe_ret','model_ret','probe_minus_model']]+[r['train_queries'],r['channels']] for r in allrows('PROBE_COMPARISON')],
       'train256고정시점×모든채널,train-onlyfeature scaling/학습,validation선택의기존frozenRidge를사용;이번에encoder/probe를재학습하지않음. 후보입력선택은future-free. probe실패는해당특성·모델의한계이지과거에서미래관련성학습불가능증명이아니다.')

    geom=[]
    for r in allrows('GEOMETRY_TEST'):
      if r['model']=='V5' and native(r) and r['space'] in ('10_oracle_candidate_embedding','30_oracle_candidate_embedding'):
        geom.append([ABBR[r['cell']],r['training_support'],r['space'],r['group'],r['selected_k']]+[fmt(r[k]) for k in ['pair_distance_mean','effective_rank','k2_silhouette','k2_stability_ari','k3_silhouette','k3_stability_ari']])
    table('Oracle positive geometry — native V5 candidate embedding',['조건','학습','공간','group','val k','pair거리','유효rank','k2sil','k2ARI','k3sil','k3ARI'],geom,
       '원본은candidate embedding/raw/delta-last/Pearson ×집합10/30 ×Oracle/random/past-nearest. 모든head가candidateembedding공유. validation에서silhouette≥.25,stability≥.8,minsize3,random대비.10우위일때k2/3선택,아니면null k1. k1에도k2/3기술통계가저장될수있으며이를cluster존재로해석하면안된다. 독립future-MSE-matchedcontrol없어multimodality/원인확정은INCONCLUSIVE.')
    table('Geometry validation선택 분포 — 모든 모델/공간 포함',['조건','선택행N','k1(null)','k2','k3','head cluster행N'],
       [[ABBR[c],len(rows(c,'GEOMETRY_VALIDATION_SELECTION'))]+[sum(r['selected_k']==str(k) for r in rows(c,'GEOMETRY_VALIDATION_SELECTION')) for k in [1,2,3]]+[len(rows(c,'HEAD_CLUSTER_COVERAGE'))] for c in CELLS])
    table('Head의 exact Oracle-positive cluster coverage — native V5',['조건','학습','공간','head/방법','val k','cluster포괄비율'],
       [[ABBR[r['cell']],r['training_support'],r['space'],r['head'],r['selected_k'],fmt(r['mean_fraction'])] for r in allrows('HEAD_CLUSTER_COVERAGE') if native(r)],
       '정확한Oracle-positiveID가포함된validation선택cluster의포괄비율. 외부후보는미할당. E96은대상V5공간의val선택이모두k1이라빈파일(N/A)이며누락학습으로채우지않는다. cluster간거리/size 상세는GEOMETRY_TEST원본참조.')
    table('Measured core 자원 — 30checkpoint',['조건','학습','모델','wall 초','peak alloc GiB','peak reserve GiB','거리 wall','score wall','ranking wall','feature/geometry wall','multihead wall'],
       [idcols(r)[:3]+[fmt(r['wall_seconds']),fmt(float(r['peak_allocated_bytes'])/2**30),fmt(float(r['peak_reserved_bytes'])/2**30)]+[fmt(r[k]) for k in ['phase_oracle_distance_wall_seconds','phase_retrieval_scores_wall_seconds','phase_ranking_wall_seconds','phase_features_geometry_wall_seconds','phase_multihead_wall_seconds']] for r in allrows('RESOURCE_SUMMARY')],
       'Solar와공유GPU에서측정한동기화walltime이며GPUkernel시간/사용률이아니다. 선행재현gate·오류복구·controller대기제외. namedphase는total을모두덮지않고distance에는metric내oracle sorting이포함되지않는다. 현재VRAM점유와allocatorpeak도다르다.')
    put('종합 판단 및 남은 작업\n\n① head는서로다른유용후보를일부검색하지만그고유성이최종결합기여를보장하지않는다. ② All5와single/LOO/union의차이는완료됐으나후보별탈락귀속은미완료다. ③ softmax분모가V5의암묵head가중치를바꾸고조건별검색순위·RetMSE에영향을준다. support제한과학습파라미터차이는분리해야한다. ④ 실제temporalshift검증은미구현이다.\n\n다음loss연구방향은조건별검증가설이다: poolregret/coverage가문제면preselection, 깊은missedrank와rankingregret면representation/ranking, 경계교환손실이집중될때boundary/regretweightedloss, past-future불일치가남으면teacherreliability/context를검토한다. 지금새loss학습결과는없다. WCE와KL은동일fixedteacher일때teacherentropy상수차이일수있으므로단순명칭변경을새학습신호라고주장하면안된다. RankNet/WeightedRankNet·margin·choice·TopKaux/differentiableTopK·hardnegative는별도후속통제실험이필요하다. seed0,역사적학습예산차이,중첩window,다중비교,Weather sentinel,고비용상관표본,geometrycontrol부족으로현재인과적단일원인판정은INCONCLUSIVE다.')

    definitions={}
    for x in inventory:definitions.setdefault(Path(x['path']).stem,x['definition'])
    table('모든 27 metric family 정의·coverage inventory',['family','5조건행수','정의/coverage'],
       [[f,','.join(str(len(rows(c,f))) for c in CELLS),d] for f,d in sorted(definitions.items())],
       '행수순서E96/E192/E336/E720/W96. 행수는Q×C표본수가아니다. JSONinventory에원본컬럼이모두있다. 본문에서피벗/부분추출/가중재집계한곳은각표에명시했다.')
    table('고유 135 CSV의 원본 행수 및 SHA256',['파일(서버export root기준)','데이터행N','SHA256'],[[x['path'],x['rows'],x['sha256']] for x in inventory],
       'root=/data/pjh_workspace/CARTS/reports/mechanism02_completed_20261103_v2/. 원본실험root=/data/pjh_workspace/CARTS/results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/. Ranking aggregate는export추가파일이며그원본RANKING_QUALITY.csv hash도검사했다. 60byte-identical별칭은INVENTORY_ALL195.json에만남긴다.')
    put('재현 근거 파일\n\n서버 export root: '+str(BASE)+'\n원본 실험 root: '+str(ORIGINAL)+'\n보고서 root: '+str(OUT)+'\n각셀 CELL_ARTIFACT_GATE.json, AGGREGATION_GATE.json, CPU_SUPPLEMENT_GATE.json, SOURCES.json과 checkpoint_evidence/를같이검토한다. checkpoint 경로·SHA·학습support·선택기준은 CHECKPOINT_INVENTORY_COMPLETED.csv에30개를기록했다. 원래checkpoint재선택없음. Runtime provenance UNKNOWN과현재저장결과재현PASS를구별한다.\n\n구현근거: experiments/retrieval_failure_mechanism02/summarize.py의고유후보/paired/집계, gpu_metrics.py의분모대조, worker.py의head/union/LOO, adapter.py의동일정규화futureMSE. 이보고서는작동중학습/진단코드를변경하거나GPU추가실행하지않고별도CPU읽기/저장만수행했다.')

    full='\n\n'.join(blocks)
    assert len(rank)==120,len(rank)
    OUT.mkdir()
    (OUT/'INVENTORY.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'REPORT_KO.md').write_text(full,encoding='utf-8')
    (OUT/'INVENTORY_ALL195.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2),encoding='utf-8')
    # Each message <=14k chars including headers; keep table blocks intact.
    parts=[];buf=''
    for b in blocks:
        assert len(b)<14000,len(b)
        if buf and len(buf)+len(b)+2>13800:parts.append(buf);buf=''
        buf+=b+'\n'
    if buf:parts.append(buf)
    index=[]
    for i,s in enumerate(parts,1):
        s='Mechanism02 상세 보고서 %d/%d\n\n'%(i,len(parts))+s
        assert len(s)<=14000,len(s)
        p=OUT/('PART_%02d.txt'%i);p.write_text(s,encoding='utf-8')
        index.append(dict(part=p.name,chars=len(s),bytes=len(p.read_bytes()),sha256=sha(p.read_bytes()),start=s[:180]))
    (OUT/'PART_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    proof=dict(status='PASS_SERVER_CSV_READ_HASH_COUNT_REPORT_RENDER',at=datetime.datetime.utcnow().isoformat()+'Z',canonical_csv=135,nonempty=sum(bool(r) for r in data.values()),aliases=60,canonical_rows=sum(len(r) for r in data.values()),gates=validation,all_csv_raw_values_in_body=False,parts=len(parts),report_sha256=sha((OUT/'REPORT_KO.md').read_bytes()),report_chars=len(full),inventory_sha256=sha((OUT/'INVENTORY.json').read_bytes()),no_gpu_execution=True,originals_modified=False)
    (OUT/'VALIDATION.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False));print(json.dumps(index,ensure_ascii=False))

if __name__=='__main__':
    import sys
    main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
