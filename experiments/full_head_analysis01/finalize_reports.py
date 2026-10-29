import csv,json,shutil,hashlib
from pathlib import Path
root=Path('/data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01')
cells=['ETTh1_96','ETTh1_720','Weather_96','Weather_720']
required=['CHECKPOINT_PROVENANCE.json','BASELINE_REPRODUCTION.md','HEAD_QUALITY.csv','HEAD_OVERLAP.csv','HEAD_UNION_COVERAGE.csv','HEAD_ERROR_COMPLEMENTARITY.csv','HEAD_LEAVE_ONE_OUT.csv','HEAD_SUBSET_ANALYSIS.csv','HEAD_STAGE2_MSE_MAE.csv','HEAD_COUNT.csv','COMPLETION_VERIFICATION.json','CANONICAL_COMPARATORS_RAW.csv','HEAD_STAGE1_STAGE2_JOINED.csv','EVIDENCE_INTERPRETATION.md']
def readcsv(p):return list(csv.DictReader(p.open()))
def table(rows,keys):
 def val(x):
  try:return f'{float(x):.6f}'
  except:return str(x)
 return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+''.join('| '+' | '.join(val(r.get(k,'')) for k in keys)+' |\n' for r in rows)
def savecsv(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (root/name).open('w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
audit=[];raw=[];joined=[];resources=[];overview=[]
for cell in cells:
 p=root/cell
 assert all((p/n).exists() for n in required),cell
 v=json.loads((p/'COMPLETION_VERIFICATION.json').read_text());assert v['status']=='VERIFIED'
 rows=readcsv(p/'HEAD_STAGE2_MSE_MAE.csv');subs=readcsv(p/'HEAD_SUBSET_ANALYSIS.csv')
 assert len(rows)==11 and len(subs)==62
 prov=json.loads((p/'CHECKPOINT_PROVENANCE.json').read_text())
 for pathkey,hashkey in [('checkpoint','checkpoint_sha256'),('base_checkpoint','base_sha256')]:
  cp=Path('/data/pjh_workspace/CARTS')/prov[pathkey]
  assert hashlib.sha256(cp.read_bytes()).hexdigest()==prov[hashkey]
 raw+=readcsv(p/'CANONICAL_COMPARATORS_RAW.csv');joined+=readcsv(p/'HEAD_STAGE1_STAGE2_JOINED.csv')
 res=v['resource'];resources.append(dict(cell=cell,wall_seconds=res['wall_seconds'],worker_peak_allocated_mib=res['peak_allocated_mib'],worker_peak_reserved_mib=res['peak_reserved_mib']))
 best=v['validation_selected_stage2_configuration'];a=v['all5']
 overview.append(dict(cell=cell,all5_mse=a['test_mse'],all5_mae=a['test_mae'],validation_selected_subset=best['subset'],selected_test_mse=best['test_mse'],selected_test_mae=best['test_mae']))
 audit.append(dict(cell=cell,status='VERIFIED',stage2=11,val_subsets=31,test_subsets=31,required_files=len(required),original_checkpoints_unchanged=True))
assert all(r['status']=='VERIFIED' for r in raw)
tests=Path('/data/pjh_workspace/CARTS/logs/TRACK-V5-FULL-HEAD-ANALYSIS01/final_unit_tests.log').read_text();assert '9 passed' in tests
savecsv('CANONICAL_COMPARATORS_RAW.csv',raw);savecsv('HEAD_STAGE1_STAGE2_JOINED.csv',joined);savecsv('RESOURCE_SUMMARY.csv',resources)
text='# Full Candidate Multi-Head Retrieval 최종 연구 보고서\n\n실험 상태: 4/4 셀, seed0, 44개 Stage2 구성 및 val/test 248개 subset 행 검증 완료. 로컬 보고서 완료. GitHub 업로드는 미완료이며 전체 전달 완료로 표시하지 않는다.\n\n'
text+='## 핵심 결론\n최종 판정은 MIXED / INCONCLUSIVE이다. ETTh1 H96에서는 All5가 단일 head와 모든 leave-one-out보다 MSE/MAE 모두 우세했다. ETTh1 H720에서는 validation 선택 H3 제거와 validation 선택 단일 H5가 All5보다 좋았다. Weather H96의 H4 제거는 lambda 재적합 시 MSE만 개선하고 MAE는 악화되며, 고정 lambda에서는 개선이 사라진다. Weather H720에서는 H3 단독의 사후 test MSE가 가장 낮지만 All5의 MAE가 가장 낮다. Test 최선 head를 배포 선택으로 취급하지 않는다.\n\n'
text+='## Validation 선택과 최종 예측\n'+table(overview,['cell','all5_mse','all5_mae','validation_selected_subset','selected_test_mse','selected_test_mae'])
text+='\nValidation 선택은 11개 Stage2 구성의 validation MSE로만 결정했다. 31개 Stage1 subset의 validation AggMSE 선택은 각 셀 보고서에 별도로 기록했다. 기준이 다르므로 혼합하지 않는다.\n\n'
text+='## 독립 학습 V0/V1/V2/V5와 Base\n'+table(raw,['cell','model','test_mse','test_mae','status'])
text+='\n표는 기존 원시 metric 값이다. 동일 Base/split forward 검증 값은 CANONICAL_COMPARATORS_RAW.csv의 별도 열에 보존했다. Weather H720 V0/V1은 후속 MeanMix 디렉터리가 없어 원래 Full 단일-head 경로를 찾아 검증했다. 단일 scoring distribution에서는 원래 선택과 MeanMixture 순위가 같으며 multihead Round-Robin 결과로 대체하지 않았다. V0→V1에는 projection 추가 효과가 있다. V5 내부 subset을 독립 학습 V1/V2와 같은 모델로 해석하지 않는다.\n\n'
text+='## 오류 보완성·기여 해석\n추가 head의 후보 발견, MeanMixture 선택, Base와 결합한 forecasting 기여는 별도로 측정했다. 낮은 후보 overlap만으로 오류 상쇄나 의미적 specialization을 주장할 수 없다. Head error cosine과 candidate-level D/C는 서로 다른 단위다. Query-channel 단위 oracle은 미래 정답을 사용한 비배포 진단이며 실제 선택에는 쓰지 않았다.\n\n'
for cell in cells:
 p=root/cell
 text+='\n---\n\n'+(p/'EVIDENCE_INTERPRETATION.md').read_text()+'\n'
 quality=[r for r in readcsv(p/'HEAD_QUALITY.csv') if r['split']=='test']
 text+='\n### Head별 test 검색 품질\n'+table(quality,['subset','retmse10','agg_mse10','agg_mae10','recall10','ndcg10'])
 text+='\n### Stage1→Stage2 연결과 개선율\n'+(p/'HEAD_STAGE1_STAGE2_JOINED.md').read_text()+'\n'
text+='\n## 검증·실행 환경 및 측정 한계\n기존 /usr/bin/python3, PyTorch1.12/CUDA11.3 환경에서 재현했다. 신규 ts-env 환경과의 수치 차이를 허용오차 확대 없이 해결했다. 네 셀 모두 새 All5 train/val/test 캐시 예측 최대 차이0, query ID 정확 일치, 저장 lambda forward 및 primary canonical 재현을 검증했다. Parent forward와 원본 Stage2 subprocess precision 차이에 의한 1e-7 수준 값은 별도 기록했다. 원본 checkpoint 해시 유지, CPU 테스트9개 통과. Seed1/2와 새 retriever 학습은 미실행이다.\n\n'
text+=table(resources,['cell','wall_seconds','worker_peak_allocated_mib','worker_peak_reserved_mib'])
text+='\nWorker peak는 별도 Stage2 자식/P100을 제외한다. Cell wall은 setup 이후부터 cache/gate/Stage2를 포함하고 앞선 smoke를 제외한다. 모든 시간은 GPU1 공유 경합 환경이며 독립 속도 비교가 아니다.\n\n'
text+='## 최소 후속 과제\n추가 학습보다 먼저 validation/test에서 head 선택 순위가 뒤바뀌는 이유와 MSE/MAE tradeoff를 저장된 query-channel 진단으로 확인한다. Weather720에서는 Base가 retrieval 구성보다 좋은지 함께 판단해야 하며 head 다양성만 최적화하지 않는다. 반복 seed 검증은 후속 제안일 뿐 이번 실행에서는 수행하지 않았다. 새로운 loss/architecture가 필요하다고 단정하지 않는다.\n'
p=root/'FINAL_REPORT.md';backup=root/'FINAL_REPORT.autogenerated.md'
if not backup.exists():shutil.copy2(p,backup)
p.write_text(text)
(root/'DELIVERABLE_AUDIT.json').write_text(json.dumps(dict(experiments='COMPLETED_VERIFIED',cells=audit,total_stage2=44,total_val_test_subset_rows=248,tests='9 passed',github_upload='NOT_DONE',large_caches_uploaded=False,research_conclusion='MIXED / INCONCLUSIVE',source_originals_preserved=True),indent=2))
print(json.dumps(dict(cells=4,stage2=44,subset_rows=248,tests='9 passed',report=str(p),github_upload='NOT_DONE')))
