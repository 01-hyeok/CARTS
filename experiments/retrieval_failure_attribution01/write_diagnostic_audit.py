"""Write factual local reports from saved evidence; no CUDA or experiment execution."""
import datetime,hashlib,json,csv,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
BASE=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'
V=BASE/'validation04'
HERE=ROOT/'experiments/retrieval_failure_attribution01'
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def js(p,d):write(p,json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

g=json.loads((V/'GRAPH_GATES.json').read_text())
pause=json.loads((V/'WORKER_PAUSED.json').read_text())
active=json.loads((V/'ACTIVATED.json').read_text())
pause_seconds=(datetime.datetime.fromisoformat(active['at'])-datetime.datetime.fromisoformat(pause['at'])).total_seconds()
table='\n'.join(f"| {x['cell']} V{x['heads']} | {x['full_validation_queries']} × {x['channels']} | {x['canonical_timing']['seconds']:.6f} | {x['cached_timing']['seconds']:.6f} | {x['speedup']:.3f}× | {x['cached_timing']['peak_allocated_bytes']/2**20:.3f} | {x['cached_timing']['peak_reserved_bytes']/2**20:.3f} |" for x in g['cases'])
write(V/'REPORT_KO.md',f'''# Validation04 검증 및 적용 기록

확인 시각: {now}. 상태: 검증 PASS, 이후 Solar 실행에 적용. 현재 학습이나 기존 결과는 다시 시작하거나 덮어쓰지 않았다.

| 검증 조건 | 전체 query × channel | 기존 초 | 새 경로 초 | 속도 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
{table}

두 조건 모두 전체 validation의 score 및 ordered Top-10 ID hash 불일치 0, RetMSE/AggMSE/AggMAE/Recall 차이 0이다. H720 V0는 메모리·정합성 검증용 기존 reference checkpoint이며 새 실험의 공식 성능으로 사용하지 않았다. 원시 증거: GRAPH_GATES.json, Solar_96_CANONICAL_TRACE.json, Solar_720_CANONICAL_TRACE.json. CPU 검증 23개 PASS: unit_tests.xml.

## 적용한 방법

137채널을 한 GEMM으로 합치는 기존 평가안은 near-tie 순위 변경으로 탈락했다. 이번 방법은 canonical B32/P100 행렬 shape를 유지한 candidate embedding cache와 CUDA Graph 묶음이다. 매 validation 호출마다 cache를 새로 생성하며 첫 실제 batch의 embedding이 동일하지 않은 채널은 기존 연산으로 fallback한다. 짧은 마지막 batch도 기존 경로다. 하나의 graph를 모든 채널에서 재사용하여 Python launch overhead를 줄인다. 임의 허용오차 확대나 후보 수 감소는 없다.

일반 epoch validation만 새 경로다. 상세 validation, test, 최종 체크포인트 선택 규칙(min validation RetMSE), P100 ID/order, batch32, 10epoch, 전체채널, teacher/loss는 보존한다. 이후 Solar H192/H336/H720의 V0/V1/V2/V5가 validation04/runtime_validation.py로 dispatch된다. 새 경로의 각 호출 통계는 해당 run/validation_acceleration/callXX.json에 기록한다.

## 안전한 인계와 시간 해석

기존 Solar96 V5 PID {active['worker_pid']}는 10epoch 저장 후 final test에서 {pause['at']}에 정지되었고, {active['at']}에 같은 프로세스로 재개되었다. pause 약 {pause_seconds:.3f}초는 해당 V5 elapsed_seconds에 포함된다. 따라서 이 elapsed를 순수 연산시간 또는 가속 벤치마크로 사용하면 안 된다. 원본 COMPLETE.json은 수정하지 않았다.

새 controller PID {active['controller_pid']}, 기존 worker를 adoption 후 다음 셀로 진행한다. 기존 35 COMPLETE 파일의 SHA는 ACTIVATED.json에 보존했다. GPU1 UUID: GPU-91418a73-202a-2a9b-57d5-2538b19e07ec. 벤치마크 중 학습 worker는 정지 상태여서 GPU 연산 경합은 없었지만 VRAM context는 유지됐다. 메모리 표는 validator 프로세스의 PyTorch peak이며 장치 전체 사용량이 아니다.
''')
js(V/'RESOURCE_TIME_ANNOTATION.json',dict(worker_run='Solar_96_V5_seed0',pause_start=pause['at'],resume_at=active['at'],pause_seconds=pause_seconds,
    original_complete_unchanged=True,elapsed_includes_pause=True,do_not_label_elapsed_gpu_compute_time=True))

inv=list(csv.DictReader((OUT/'00_Audit/CHECKPOINT_INVENTORY.csv').open()))
counts={f'{s}_{t}':sum(x['training_support']==s and x['status']==t for x in inv) for s in ('P100','FULL') for t in ('VALID','MISSING','INCOMPATIBLE')}
protocol=json.loads((HERE/'PROTOCOL.json').read_text())
protocol['ranking']['kendall_full']='APPROXIMATE tau-b from 4096 fixed future-free candidate pairs per query/channel, seed0; report unique pair count and tie exclusions'
protocol['ranking']['kendall_full_pair_budget']=4096
protocol['ranking']['full_global_spearman']='exact average ranks on one query/channel score vector at a time; streaming candidate vectors may spill to local disk'
protocol['protocol_frozen_before_new_diagnostic_val_test']=now
protocol['legacy_ndcg']=dict(name='legacy_shifted_negative_MSE',gain='max_valid(d)-d; direct gain',source='scripts/train_patch_retrieval_expert01.py:88',
    policy='reproduce legacy metric only for canonical gate; separately report NDCG_inverse_MSE_v1')
protocol['scheduler']['pending_gpu']='Existing48-run controller has exclusive major GPU use; new diagnostic inference not launched until it is complete and free-GPU check passes'
js(HERE/'PROTOCOL.json',protocol)
test=ET.parse(OUT/'00_Audit/CPU_TESTS.xml').getroot()
suites=[test] if test.tag=='testsuite' else list(test)
n_tests=sum(int(s.attrib.get('tests',0)) for s in suites)
fail=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
assert n_tests==31 and fail==0
js(OUT/'00_Audit/CPU_GATE.json',dict(status='PASS',tests=n_tests,failures=fail,at=now,device='CPU, CUDA_VISIBLE_DEVICES empty',
    code_sha256={n:sha(HERE/n) for n in ['diagnostic_core.py','test_diagnostic_core.py']},
    scope='synthetic mathematical/leakage prerequisites only; no real-checkpoint reproduction claimed'))

write(OUT/'00_Audit/IMPLEMENTATION_AUDIT.md',f'''# Retrieval Failure Attribution 01 — Phase 0 구현 감사

작성 시각: {now}. 확정 첨부 libfile_bdf819f7a4ac8191b126af541b9cfa1b만 사용했다. 철회된 첨부는 사용하지 않았다.

## 현재 상태

P100 provenance-valid {counts['P100_VALID']}/24, 미완료 기존 학습에 의존하는 체크포인트 {counts['P100_MISSING']}/24. Full-trained provenance-valid {counts['FULL_VALID']}/24, Solar 4horizon × V0/V5 8개 MISSING. VALID는 경로·SHA·선택 기록 확인이며 새 진단의 metric reproduction PASS를 의미하지 않는다. 실제 GPU 재현/검색은 NOT_RUN이다.

CHECKPOINT_INVENTORY.csv에 경로, SHA256, 선택 epoch, 선택 지표와 근거를 기록했다. P100은 LEGACYPOOL02의 min validation Mean-Mixture RetMSE 선택 완료분이다. Full V5는 TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01의 정정 선택분이다. Full V0는 TRACK-V-MULTIQUERY-GENERALIZATION01의 min val model_top10_individual_mse 선택분이다. 소스 commit이 원래 실행 기록에서 확정되지 않은 경우 UNKNOWN으로 남긴다. 현재 checkout HEAD를 과거 학습 commit으로 대체하지 않는다.

## 실제 목적함수 및 검색

- Full V0 실행 경로: scripts/run_v_one_setting01.sh:33–39 → train_j_shared_encoder_drift01.py. 해당 trainer:450–456은 future MSE d, normalized_teacher_prob(d,mask,tau_t), kl_loss(teacher,student score,tau_s)를 학습한다.
- Full V5: 같은 실행 스크립트:55–63 → train_t_pure_multislot01.py; 정상 부호의 z-score teacher와 mean probability KL. 정정된 validation Mean-Mixture checkpoint 선택을 사용한다. 과거 Round-Robin 선택 결과를 대체값으로 쓰지 않는다.
- P100: experiments/p100_head_retrieval_quality_legacy_pool02/runtime.py:182–198. Query/Key 공통 MLP encoder, query projection head, cosine, candidate 축 softmax(score/0.1), head 확률 평균, teacher KL. teacher는 softmax(-(d-mean_valid(d))/std_valid(d)/0.1)이며 raw-MSE teacher로 변경하지 않는다.
- Value: candidate_future - candidate_history_last + query_history_last. 데이터 정규화 공간의 horizon 전체 MSE다.
- Stable TopK는 inherited support order를 동점 기준으로 사용한다. P100은 저장된 후보 순서, Full은 canonical bank ID 순서다.
- P100 All-head 평가의 advanced indexing 및 mean reduction layout도 canonical과 대조해야 한다. 확률 평균과 score 평균을 혼동하지 않는다.

## 재사용과 새 구현 구분

| 부분 | 사용 계획 / 상태 |
|---|---|
| 실제 loader/scaler/RAFT mask/ID | 기존 runtime.CellData, RelationMemorySampler, 원본 P100 캐시 재사용 |
| frozen encoder/heads, delta-last 복원 | 기존 runtime/model/helper 재사용; checkpoint immutable |
| Mean-Mixture/TopK | utils/mean_mixture_selection.py 및 모델 stable_topk 재사용 |
| 기존 Full 분석 | experiments/full_head_analysis01의 재현/provenance 참고; 대용량 결과 복사 금지 |
| 과거 이웃 분석 | utils/past_neighborhood.py의 수학 연산 재사용 가능. irreducible 표현은 사용하지 않음 |
| 전역 normalization/streaming Oracle | diagnostic_core.py 합성 검증 PASS; 실제 Full inference adapter는 아직 미구현/미검증 |
| NDCG/Regret/Exchange/마스크/동점 | CPU 합성 31개 PASS; 실제 checkpoint 재현 별도 필요 |
| features/probe | past-only score primitive, train-only Ridge guard 구현; 실제 전체 데이터 실행 아직 NOT_RUN |
| ranking/feature/ambiguity/probe/그림 전체 산출 | orchestration 및 전체 실행 구현 잔여. 완료라고 표시하지 않음 |

## 공정성 및 비교 제한

Full-trained와 P100-trained는 학습 support뿐 아니라 초기화, 예산, checkpoint trajectory가 다를 수 있다. 양자 차이를 후보군 효과 하나로 해석하지 않는다. CROSS_SUPPORT_CAPABILITY.csv의 P100-trained frozen Full inference는 같은 파라미터의 support counterfactual이며 Full-trained 결과가 아니다. Solar Full-trained 체크포인트 누락을 이 counterfactual로 채워 넣지 않는다. 미래 진단 결과로 checkpoint를 재선택하지 않는다.

Full 과거 재현에는 legacy torch1.12 CUDA11.3와 TF32 설정이 영향을 준 기록이 있다. 신규 runtime의 Full 연산은 실제 ID/지표 재현 gate를 통과해야 하며, near-tie 불일치가 발생하면 허용오차를 임의로 확대하지 않고 보고한다.

## 안전한 실행 큐

RUN_QUEUE.json은 실행 계획이며 현재 실행 중인 daemon이 아니다. 기존 48개 seed0 작업이 GPU1을 사용 중이므로 신규 major GPU task를 시작하지 않았다. P100 12개를 ETTh1 96/192/336/720 → Weather 동일 → Solar 동일 순서로 모두 완료한 뒤 Full 12개로 넘어간다. P100 단계의 Full Oracle/coverage/decomposition은 DEFERRED_FULL_DEPENDENCY다. Full 단계가 끝난 뒤 의존 지표를 채운다. 실제 checkpoint gate 및 추론 구현이 미완료인 항목은 실행하지 않는다.

Stage1/Stage2 재학습, GPU0, 기존 작업 중단, Git push/외부 업로드는 수행하지 않았다. 진단 때문에 원본 데이터·model·checkpoint를 수정하지 않았다.
''')
write(OUT/'00_Audit/DATA_AND_METRIC_AUDIT.md',f'''# 데이터·지표 사전 고정

기록 시각 {now}. 아직 새로운 진단 validation/test GPU 평가를 실행하지 않았다.

- L=H=96/192/336/720; ETTh1 7채널, Weather 21채널, Solar 137채널 전체. seed0, K10. 실제 각 split의 query 수/starts/scaler/data SHA는 기존 CellData 및 pool metadata를 대조해야 한다.
- ETTh1.csv SHA f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.
- Weather weather.csv SHA 34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63. -9999 값 81개가 기존 scaler에 포함된 상태를 보존한다. 정제 모델 성능이나 물리 단위 오차라고 주장하지 않는다.
- Solar solar_AL.txt SHA 230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2. 실제 calendar timestamp가 없어 날짜를 만들어 넣지 않는다.
- scaler는 기존 train fit, 평가 normalized space, delta-last value reconstruction. Train RAFT는 offline non-overlap train bank이며 strict causal retrieval이라고 부르지 않는다. validation/test는 train bank만 사용한다.
- `NDCG_inverse_MSE_v1`: gain=1/(1+d), direct gain, discount=1/log2(rank+1), 동일 valid support ideal. 기존 `legacy_shifted_negative_MSE`의 gain=max_valid(d)-d와 별도 이름이다. 음수 relevance를 exponential gain에 직접 넣지 않는다.
- Spearman은 exact tie average rank. P100 Kendall tau-b는 exact. Full Kendall은 seed0 future-free 고정 4096쌍 APPROXIMATE로 표기하고 유효/중복/동점 수를 기록한다. Full global Spearman은 query/channel별 전체 score 벡터를 처리한다.
- Oracle은 개별 MSE Top10 RetMSE 최적이며 집계 예측 MSE의 최적을 의미하지 않는다.
- P100 모델만 R_full≤R_p100≤R_model 및 Pool+Ranking=Total을 적용한다. Full 모델에는 R_full≤R_model만 강제한다. Exchange=Ranking은 같은 support에서 검증한다. share는 합계 regret의 비율이다.
- Probe candidate 선택과 입력은 future-free; train만 fit/scaler, validation만 선택, test는 최종평가. 기존 encoder와 head는 불변이다. 합성 Ridge leakage gate는 통과했지만 실제 probe는 미실행이다.
- 시간 블록 bootstrap 2000회, seed0, block length=2*(L+H) rows, 동일 query 시점 채널 공동 resampling. 작은 유효 블록 수와 seed0 한계를 보고한다. query별 독립 p-value를 주장하지 않는다.
- GPU 자원 수치는 실측만 기록한다. CPU 합성 테스트 PASS는 실제 fullsample 재현을 대체하지 않는다.
''')
status=json.loads((OUT/'00_Audit/RUN_STATUS.json').read_text())
status.update(status='PHASE0_INVENTORY_AND_CPU_PREREQUISITES_COMPLETE_GPU_REPRODUCTION_PENDING',checked_at=now,
    inventory=counts,cpu_tests=dict(passed=31,failed=0),validation_acceleration='PASS_ACTIVATED_SEPARATE_TRAINING_TRACK',
    queue=dict(entries=24,launched=False,order='P100_ALL_12_THEN_FULL_ALL_12',gpu_dependency='existing48-run training completion'),
    new_diagnostic_gpu_runs_started=0,full_search_started=False,
    remaining_implementation=['frozen inference adapter and canonical selected-ID/metric reproduction','full ranking/features/ambiguity/probe aggregation and figures','durable diagnostic controller with fail-closed phase barriers'])
js(OUT/'00_Audit/RUN_STATUS.json',status)
print(json.dumps(dict(checked_at=now,inventory=counts,cpu_tests=31,validation_report=str(V/'REPORT_KO.md'))))
