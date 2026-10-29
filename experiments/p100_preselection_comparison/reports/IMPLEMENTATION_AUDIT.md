# P100 Candidate Preselection Comparison — 구현 감사

실행 범위: seed0, ETTh1_96 → ETTh1_720 → Weather_96 → Weather_720. 각 셀의 여섯 설정은 Stage2까지 끝낸 뒤 다음 셀로 이동한다. 물리 GPU1 UUID `GPU-91418a73-202a-2a9b-57d5-2538b19e07ec`만 사용한다. Full 분석 파이프라인과 자원을 공유하므로 timing은 독립 속도 실험이 아니다.

## 재사용 구현 및 과거 오류

- Original KL = V0: query projection 없음. 공유 MLP encoder의 normalized query/key cosine 점수, P100 support, temperature .1, future MSE z-score teacher의 negative softmax, KL(teacher || student).
- V5 = 같은 encoder + identity 근처에서 seed별로 초기화한 5개의 선형 query projection. 각 head normalize, per-head softmax 후 확률 평균을 teacher와 KL 비교. 추가 head loss/aggregate loss 없음.
- 학습은 **`scripts/train_v_sharedtop100_01.py`를 import하여 그대로 실행**. 이 파일의 teacher는 `normalized_teacher_prob(d_pool, ...)`로 sign-fixed 상태다. `train_retriever_pool01.py`의 일반 main은 과거 double negative teacher가 남아 있으므로 실행하지 않는다. 그 파일에서 검증된 `compute_scores`, `CandidatePoolCache`만 재사용한다.
- 10 epochs, batch32, Adam LR .001, seed0/init0/loader0, tau_t=tau_s=.1, slot_std=.001을 모든 방법에서 고정한다. OOM이 나도 한 비교군만 batch 변경하지 않는다.
- V0 primary는 validation single-distribution RetMSE. V5는 **모든 epoch의 validation Mean-Mixture RetMSE로 재선택** (`eval_v5_meanmix_checkpoint_selection_pool01.py`). trainer 내부 round-robin 진단 checkpoint를 V5 Stage2에 전달하지 않는다. test에서 checkpoint 선택 금지.
- V0 cache는 `build_retrieval_cache_pool01.py` (단일 분포에서 RR=MeanMixture), V5 cache는 `build_v_meanmix_cache_pool01.py`를 그대로 사용.
- Stage2는 `train_r_stage2_lambda01.py`: 고정 Base/Stage1, train MSE로 전역 sigmoid lambda 하나만 Adam 학습, validation MSE epoch 선택, test MSE/MAE 평가. 과거 professor-style validation-only beta 결과는 기준으로 사용하지 않는다. 재현 기준은 corrected `stage2_lambda_original/metrics.json`.

## Preselector

- baseline_cosine: 기존 `compute_coarse_delta_last_cosine_scores` + `build_coarse_pool`를 직접 호출. 각 split의 **모든 생성 ID를 기존 canonical cache와 동일성 assert**한다. epsilon/normalization/tie 규칙 변경 없음.
- paa16_cosine: 기존 normalized history의 delta-last 표현을 96→16(6시점 평균),720→16(45시점 평균), F.normalize 후 dot. 후보 원본 encoder 입력은 바꾸지 않음.
- saraf_time_aligned: 공식 SARAF의 Pearson + time bonus, alpha=.1. Pearson은 channel별 time-axis centering과 F.normalize(default epsilon1e-12). 시간 기준은 query/candidate **input midpoint global_start+L//2**. timestamp는 원본 CSV date를 검증하여 사용하며 누락·비정상 간격은 실패.

공식 출처: https://arxiv.org/abs/2606.04135 ; https://github.com/ShiqiaoZhou/SARAF/blob/7b49aa8aabb38e8fe63adbc3146b1890958bf3ca/layers/Retrieval.py . 외부 소스는 검토만 하고 실행하지 않는다.

시간 bonus: L<steps/day면 .6 exp(-circular_hour/2)+.2 exact_weekday+.1 exact_weekend+.2 exp(-circular_month/3), 그 외 .3 exp(-hour/4)+.2 weekday+.1 weekend+.4 exp(-month/2). L<steps/day//4일 때만 .1 exp(-circular_minute/15) 추가. ETTh1은 CSV에서 시간당1회(24/day), Weather는 10분(144/day)임을 확인하며 공식 코드의 15분 alias를 그대로 오용하지 않는다. 지정 L=96/720에서는 minute branch가 실행되지 않는다.

공식 batch-global max normalization은 호출 batch에 따라 결과가 달라질 수 있다. 이를 임의 row normalization으로 바꾸지 않고 **전체 split query×train memory timestamp 집합을 한 번 호출한 것과 같은 최대값**을 chunk로 계산해 split 전체에서 고정한다. score=.9 Pearson+.1 normalized bonus. 이 호출 granularity 선택과 CSV sampling adaptation을 method 정의로 기록한다.

## Mask 및 leakage 범위

기존 **RAFT mask** 그대로 사용: train query는 자기 자신 및 symmetric overlap interval을 제외; val/test는 기존 train memory만 검색. 이는 strict_causal mask가 아니며 모든 더 늦은 train window를 배제한다고 주장하지 않는다. query_start_idx는 Stage1WindowDataset에서 이미 border1이 더해진 전역 값이므로 border를 두 번 더하지 않는다. 각 split의 loader starts와 기존 canonical cache starts를 일치 확인한다. 후보 ID100·중복 없음·mask 유효성을 매 생성마다 assert. 유효 후보100 미만은 실패.

## Isolated adapter / provenance

공통 소스 수정 없음. 새 `pipeline.py`는 import한 module의 CandidatePoolCache만 method별 strict metadata 확인 adapter로 감싼다. 내부 pool config는 기존 coarse_topk 분기를 택하는 용도로만 사용하며 실제 후보 점수는 생성된 ID에 의해 결정된다. 새 cache metadata 및 저장된 checkpoint config 모두 실제 method/metric와 train/val/test cache SHA256을 기록한다. 학습 config/init hash, checkpoint SHA, canonical source SHA, git SHA, GPU UUID를 보존하고 mismatch이면 실패한다.

기존 결과를 새 성능으로 복사하지 않는다. 재현은 새 cache forward와 기존 lambda forward로 검증하고 새 training 결과와 분리한다. 모든 생성 파일은 experiments/p100_preselection_comparison 아래만 저장한다.

## 검증 및 진단

CPU synthetic tests 17개. 실제 baseline V0/V5 한 batch smoke와 세 preselector 실데이터 ID 생성 smoke 통과 후 각 셀의 full-method cache를 이용한 V0/V5 gradient smoke를 추가 수행한다. canonical reproduction은 셀별 gate이며 V0/V5 Stage1 6지표와 저장 lambda Stage2 MSE/MAE를 원시 JSON과 tolerance1e-5로 비교한다.

학습 전 quality는 validation 전체에 고르게 분포한 고정32query×모든 channel에서 full oracle coverage, canonical full V5 head/mixture retention, best individual, individual oracle Top10, greedy-without-replacement approximate set oracle, baseline overlap을 측정한다. 미래 값은 이 진단에만 사용. query ID를 저장한다. 학습·검증·test cache 생성은 전체 split을 사용한다.

V5 최종 test의 head RetMSE/AggMSE/AggMAE/Recall, overlap, raw-error cosine와 final mixture AggMSE를 별도 기록한다. head 단위 오류와 selected candidate D/C는 서로 다른 양임을 유지한다.

각 subprocess의 GPU synchronized seconds, peak allocated/reserved를 RESOURCE_METRICS로 로그에 남긴다. 평가 OOM은 batch32→16→8→4 동일 support로 재시도; training OOM은 공정성을 위해 fail-stop하며 임의 비교군 batch 축소 금지. 결과 완료/진행/미실행을 구분하고 에러 traceback을 FAILED.json에 보존한다.

## Baseline singleton-tail regression (2026-10-28)
The failed PID3507830 is preserved in FAILED.json and pipeline.log. All non-singleton ETTh1_96 rows match canonical IDs exactly. Only the final singleton of each split differs: 17 query-channel rows, including one validation support difference. TF32 GEMM with replicated query batch32 reproduces every historical singleton ID exactly; FP32 and FP64 do not. The isolated baseline_scores adapter pins that historical kernel path only for singleton tails, restores the prior backend flag, and leaves cosine, epsilon, valid mask, stable index tie rules and all shared source unchanged. Full ordered ID equality remains mandatory for every split/cell; failure still aborts. See baseline_mismatch_diagnostic.json. Historical runtime software/kernel provenance is incomplete, so this compatibility evidence does not establish which historical library version caused the original numerical path.

## Independent official-source verification (2026-10-28 13:34 UTC)
Read-only verification against https://arxiv.org/html/2606.04135v1 sections 3.4.1-3.4.2 and immutable official source https://github.com/ShiqiaoZhou/SARAF/blob/7b49aa8aabb38e8fe63adbc3146b1890958bf3ca/layers/Retrieval.py .
Official lines 375-378 use midpoint for query and candidate; 395-456 define circular hour/month/minute distances, weekday/weekend exact matches, sequence-frequency weights, and maximum normalization. Lines 463-474 center then F.normalize the inputs before dot product. Our self-only CI implementation applies that definition separately per target channel. No external source executed.
Alpha is fixed at 0.1, score=0.9*Pearson+0.1*TimeBonus. L=H=96 or L=H=720, PAA=16, K=10, P=100. No MMR, stationarity weighting or new objective.
Official frequency aliases omit Weather's actual 10-minute cadence; derive 144/day from strictly validated CSV intervals (ETTh1=24/day). Official global maximum is fixed over the entire split rather than dependent on loader batches; this explicit SARAF-style adaptation is documented, and is not claimed bitwise identical to arbitrary official per-batch calls.
Stage2 source explicitly freezes the base and loads cached retrieval predictions; only a scalar sigmoid lambda is trained with train MSE. Prediction B+lambda*(R-B). Validation MSE chooses epoch, test supplies no selection signal.
Every completed arm is required to pass checkpoint preselection_method/metric, three cache SHA256 values, init_seed=loader_seed=0 and independent Stage2 MSE/MAE checks. Base SHA is recorded for each cell. Initial model hash is compared across all six methods/models within a cell. New checkpoints are never substituted by historical checkpoints; historical reuse is restricted to the reproduction gate and diagnostic full-head reference.

## Efficiency measurement scopes
RESOURCE_METRICS records a synchronized subprocess wall interval and reset PyTorch allocated/reserved peaks separately. Reported run wall time spans STARTED.json to independently verified Stage2 COMPLETE.json; extra V5 head analysis follows that boundary. Preselection COMPLETE.json separates feature tensor arithmetic from split search time. Timestamp CSV loading and full-split temporal-bonus maximum construction are currently outside those two timers; these measurements must not be described as total end-to-end preselection latency. Parallel contention makes all wall times observational rather than isolated method benchmarks.

## Execution environment comparison (2026-10-28 13:44 UTC)
/usr/bin/python3: PyTorch 1.12.0.dev20220327+cu113, CUDA 11.3, default matmul.allow_tf32=True.
/data/pjh_workspace/ts-env/bin/python: PyTorch 2.4.1+cu121, CUDA 12.1, default matmul.allow_tf32=False.
P100 executes all arms consistently with ts-env, following repository activation guidance. Full owner reports using original /usr/bin/python3 for its separate canonical parity. These different execution environments and simultaneous GPU1 contention prohibit interpreting cross-track wall times as isolated algorithm comparisons.
The P100 singleton numerical regression is compatible with the identified runtime/kernel differences; the isolated singleton adapter restores canonical ordered IDs exactly for ETTh1 H96 while retaining full equality gates for every later cell.
New ETTh1 H96 baseline V0 training completed all 10 epochs and reproduces canonical Stage1 RetMSE/AggMSE and Stage2 MSE=0.36924293637275696, MAE=0.3942062556743622 exactly, with independent verification. New V5 canonical comparison remains pending. Any later reproducibility failure will be reported rather than changing a single arm's runtime or training settings.

## New V5 baseline parity closed
ETTh1 H96 V5 completed all 10 epochs, selected epoch10 by validation Mean-Mixture RetMSE, and independently verified Stage2 MSE=0.3695971369743347 / MAE=0.39546850323677063. Canonical Stage1 RetMSE=0.6638551354408264 / AggMSE=0.40980765223503113 also match exactly. This closes the earlier pending V5 numerical compatibility check for this cell; later cells retain their own gates.

## H720 baseline numerical runtime recovery
At ETTh1 H720, new-runtime baseline mismatched from the first batch (offset0/channel0:131 ordered positions), so the previous singleton-only fix was insufficient and the mandatory equality gate correctly stopped PID3512238 before any H720 training.
Baseline generation now uses /usr/bin/python3 (the original PyTorch1.12/CUDA11.3 runtime) and the exact compute_coarse_delta_last_cosine_scores function, followed by unchanged full ordered-ID equality assertions. This is confined to fixed raw baseline precomputation; all six training arms, PAA/SARAF features, selection and Stage2 continue with the same ts-env runtime/config.
Completed H96 caches/runs are preserved and skipped, with their exact canonical ID/result parity already verified. No tolerance relaxation, cache-ID substitution or model/loss change is made. Original failure is preserved as FAILED_H720_baseline_legacy_runtime.json and pipeline_resume_singleton_fix.log.

## Weather actual timestamp anomalies and explicit handling
Original CSV has 52,696 rows, no missing/unparseable dates, no decreasing dates, one repeated timestamp (row19044:2020-05-12 06:00:00), and one 100-minute interval (row21514:2020-05-29 09:30→11:10). Remaining 52,693 intervals are ten minutes. Raw source hash and exact anomalies are in Weather_timestamp_source_audit.json.
The initial strict-uniform/unique validation stopped PID3560741 before Weather training. This was an overly restrictive validator, not evidence of absent date cells. Preserve original row ordering, all existing candidate IDs, windows and actual calendar values. No row sorting, deletion, deduplication, timestamp correction or gap imputation is performed. Query/candidate midpoint remains original global row+L//2. Two records with the same actual time legitimately receive the same temporal features; the gap uses actual subsequent timestamps.
Nominal period144/day is inferred from the overwhelmingly modal positive interval solely for official SARAF sequence-frequency branch selection. Missing/decreasing timestamps, ambiguous cadence (<99% modal positive intervals), nonintegral nominal multiples and non-day-dividing cadence still fail. Duplicate/gap counts are explicitly printed and saved per dataset/horizon. This supersedes the earlier blanket irregularity rejection description without weakening candidate-ID or split leakage gates.

## Runtime dispatch correction
The original dispatch compared resolved executable paths. ts-env/bin/python is a symlink to the same system binary, despite a different sys.prefix/site-packages/PyTorch. PID3586571 therefore incorrectly skipped legacy baseline delegation and hit the unchanged ID gate. Compare invoked absolute interpreter paths without resolving the venv symlink. Failure preserved in FAILED_Weather96_runtime_dispatch.json. No Weather training had started.
