import csv,json,sys
from pathlib import Path
cell=sys.argv[1];p=Path('/data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01')/cell
v=json.loads((p/'COMPLETION_VERIFICATION.json').read_text())
rows=list(csv.DictReader((p/'HEAD_STAGE2_MSE_MAE.csv').open()))
histpath=p/'CANONICAL_COMPARATORS_RAW.csv'
if not histpath.exists():histpath=p/'INDEPENDENT_MODELS.csv'
hist=list(csv.DictReader(histpath.open()))
pairs=[r for r in csv.DictReader((p/'HEAD_ERROR_COMPLEMENTARITY.csv').open()) if r['split']=='test']
union=json.loads((p/'test_union.json').read_text());subset=json.loads((p/'subset_selection.json').read_text())
all5=next(r for r in rows if r['subset']=='12345')
def name(s):
 if s=='12345':return 'All-5'
 if len(s)==1:return 'H'+s
 return 'Without H'+next(h for h in '12345' if h not in s)
def table(rs,keys):
 return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+''.join('| '+' | '.join(str(r.get(k,'')) for k in keys)+' |\n' for r in rs)
display=[dict(Configuration=name(r['subset']),**{k:f"{float(r[k]):.6f}" for k in ['primary_mse','primary_mae','secondary_mse','secondary_mae']}) for r in rows]
better_m=[name(r['subset']) for r in rows if float(r['primary_mse'])<float(all5['primary_mse'])]
better_a=[name(r['subset']) for r in rows if float(r['primary_mae'])<float(all5['primary_mae'])]
text=f'# {cell}: completed seed0 evidence\n\n상태: VERIFIED. 신규 encoder/head 학습 없이 기존 Full V5의 head 확률을 다시 평균하고 Top10을 재선택했다. Primary는 구성별 global lambda 재적합, Secondary는 canonical All5 lambda 고정이다.\n\n'
text+='## Stage2 결과\n'+table(display,['Configuration','primary_mse','primary_mae','secondary_mse','secondary_mae'])
text+=f'\nAll5보다 test primary MSE가 낮은 구성: {better_m or "없음"}. MAE가 낮은 구성: {better_a or "없음"}. 이 비교는 동일 V5 내부 ablation이며 독립적으로 학습한 1-head 모델 비교와 다르다.\n'
text+='\n## 독립 학습 모델 및 Base\n'+table([{k:r.get(k,'') for k in ['model','test_mse','test_mae','status']} for r in hist],['model','test_mse','test_mae','status'])
text+='\n위 독립 모델 수치는 기존 원시 결과를 동일 Base/split forward로 재검증한 값이며 새 학습 결과가 아니다. V0→V1에는 query projection 도입 효과가 포함된다.\n'
text+=f'\n## Validation 선택과 Oracle\n11개 Stage2 구성에서 validation MSE가 가장 낮은 구성은 {name(v["validation_selected_stage2_configuration"]["subset"])}이다. 해당 test MSE/MAE는 {v["validation_selected_stage2_configuration"]["test_mse"]:.6f}/{v["validation_selected_stage2_configuration"]["test_mae"]:.6f}. 31개 Stage1 subset의 validation AggMSE 선택은 {subset["val_selected_subset"]}, test AggMSE는 {subset["test_selected_subset"]["agg_mse10"]:.6f}. Test future 기반 per-query-channel oracle AggMSE {subset["oracle_test_aggmse"]:.6f}는 배포 불가능한 진단이다. 서로 다른 선택 기준을 혼합하지 않는다.\n'
means={k:sum(float(r[k]) for r in pairs)/len(pairs) for k in ['top10_overlap','error_cosine','negative_cosine_fraction']}
text+=f'\n## 발견·선택·예측 기여\n평균 head Top10 overlap={means["top10_overlap"]:.6f}, union size={union["union_size"]:.6f}. Head 예측 오류 cosine 평균={means["error_cosine"]:.6f}, 음수 비율={means["negative_cosine_fraction"]:.6f}. 후보 다양성과 오류 상쇄는 별개의 근거다. Head union의 individual oracle Top10 coverage={union["oracle_coverage_by_heads"][-1]:.6f}; 최종 mixture Top10 중 head Top10 union 내부 비율={union["mixture_top10_in_head_union"]:.6f}. Mixture는 전체 후보 확률을 평균하므로 head별 Top10의 단순 union에서만 고르지 않는다.\n'
res=v['resource'];text+=f'\n## 무결성과 측정 범위\n31개 subset의 val/test, 11개 Stage2 및 validation epoch 선택을 검증했다. 원본 retriever/Base 해시는 변경되지 않았다. 전체 캐시 비교 상세는 newcache_stored_lambda_verification.json이다. 셀 wall={res["wall_seconds"]:.3f}s, 분석 worker peak allocated/reserved={res["peak_allocated_mib"]:.3f}/{res["peak_reserved_mib"]:.3f}MiB. 이 peak는 Stage2 자식과 P100을 제외하며 시간은 GPU1 경합 환경이다.\n'
text+='\n## 발표 결론과 스크립트\n단일 seed에서 학습된 V5 내부 head의 조건부 기여를 측정했다. 독립 모델 우월성, 의미적 specialization, 다양한 후보가 개선의 원인이라는 인과 주장은 별도 근거가 필요하다.\n\n이번에는 기존 Full V5를 고정한 채 head를 단독 사용하거나 하나씩 제거했습니다. 각 구성에서 Top10을 다시 고르고, lambda를 재적합한 결과와 고정한 결과를 분리했습니다. 표의 Stage2 MSE와 MAE가 실제 기여 판단의 기준입니다. Validation 선택과 test 사후 비교도 분리했습니다. Head overlap이 낮다는 사실만으로 오류 상쇄를 결론내리지 않고, 독립 학습 모델과의 비교를 함께 해석해야 합니다.\n'
(p/'EVIDENCE_INTERPRETATION.md').write_text(text)
print(p/'EVIDENCE_INTERPRETATION.md')
