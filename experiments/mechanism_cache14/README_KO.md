# CACHE14 준비 및 적용 조건

목적은 수치 연산을 바꾸지 않고 동일 입력의 모델 독립 계산과 무관한 코드 변경에 따른 재현 검증 중복을 줄이는 것이다. 기존 실행 폴더/결과/체크포인트는 수정하지 않는다. 현재 상태는 **CPU 검증 완료, GPU 검증·운영 적용 미완료**다.

## 구현

- `cache_store.py`: scope/content hash, 원자적 index publish, 세대별 immutable 저장, 파일·배열 SHA256 및 dtype/shape/ordered ID/값의 bit 비교. 손상·불일치는 canonical 재계산하며 이전 세대는 보존한다. writer 경합 시 대기·강제 unlock 없이 canonical 계산한다. 예산 초과도 계산을 유지한다.
- `provenance.py`: 실제 query/channel/bank/pool/mask, 입력 및 정답 배열, train scaler, 데이터 파일, 원본 연산 의존 코드와 수치 runtime을 키로 고정한다. 전체 channel 수와 P100 순서/유효성/중복을 검사한다. 정답은 거리/Oracle 진단 cache 키에만 사용하며 검색 점수에 넣지 않는다.
- `common_ops.py`: 기존 콜백의 결과만 저장·복원한다. future MSE, past delta score, Random 및 past-only feature를 공유하기 위한 API다. encoder/weights는 다루지 않는다. 모델별 missed/false/correct group 집계와 embedding geometry는 공유 대상이 아니다.
- `oracle_reuse.py`, `prepare_oracle_patch.py`: 원래 evaluate 함수의 Oracle Top100 표현만 선택적 기존값으로 대체한다. 나머지 수식/순서/정밀도/ID 동률 규칙은 그대로다. 패치는 analyze 본문만 바꾸고 reproduce/main은 보존한다. 매 블록 Full/P100 Oracle을 각각 1번 계산한다. V0/V1은 기존 4/5번, V5는 28/29번의 동일 Oracle 계산이 2번으로 줄어드는 코드 경로다. wall-time 절감량은 미측정이다.
- `scoped_gate.py`: 분석 전용 함수 본문만 변경됐는지 AST로 확인한다. 재현 호출 경로의 코드, module globals/imports/defaults, checkpoint/data/pool/protocol, 전체 val/test count·ID hash·runtime이 다르면 재검증한다. adapter 변경은 모델명/branch 추정으로 면제하지 않는다. 원본 gate를 수정하지 않고 별도 attestation 후보만 반환한다.

## 저장·자원 제한

기본 예산은 전체 8 GiB, entry 384 MiB이다. dense `[B,C,N]` future-MSE/score 저장은 **기본 비활성화**이며 활성화 시에도 block B<=32·용량 제한을 적용한다. `[Q,N,H,C]` 또는 `[B,C,N,H]` 후보 미래 배열은 저장하지 않는다. 후보나 채널을 줄이는 fallback은 없다. 전체 Weather96 거리/과거점수/Random의 무제한 저장은 수백 GB의 I/O가 될 수 있으므로 속도 측정 없이 켜지 않는다.

## 검증 근거

서버 경로:
`results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit/CACHE14_PREPARATION_20261103/`

- CPU 회귀: 47개. 손상/부분 저장, runtime·data·mask·pool·ID 변경, 용량 제한, 동시 writer, gate coverage 부족, 재현 함수 변경 거부 포함.
- 실제 완료 블록: ETTh1 H96 test 첫 32query/마지막 1query ×7채널 및 Weather H96 test 첫32query×21채널, 각6checkpoint =18블록. Oracle100 ID/MSE, random/past-nearest 선택 ID 및 미래 분산, random group feature가 lossless cache 전후 bit 일치. 해당 6checkpoint 간에도 이 공유 대상 값의 차이가 없었다.
- 원래 canonical topk와 evaluate를 CPU에서 실행하여 B=1/12/32, C=21, N=128, tie 유무의 6개 case에서 모든 13개 반환 tensor가 bit 일치. cache miss도 원본과 일치. GPU context 생성 없음.
- 기존 전체 distance/full score는 저장되지 않아 위 실제 블록만으로 dense 재계산 경로를 검증했다고 주장하지 않는다.
- geometry 전체 query/channel 범위·반복 횟수는 그대로다.

실제 compact 블록 추출 및 캐시 load+bit 비교의 중앙값(각6checkpoint, hot-cache CPU 관측):

| 대표 블록 | cache payload | 기존 NPZ 추출 | 캐시 load+exact 비교 |
|---|---:|---:|---:|
| ETTh1 H96 32×7 | 1,419,520 bytes | 0.017104 s | 0.013904 s |
| ETTh1 H96 1×7 | 44,360 bytes | 0.009418 s | 0.008487 s |
| Weather H96 32×21 | 4,258,048 bytes | 0.031202 s | 0.025202 s |

이는 GPU 계산 절감이나 전체 실행 가속 비율이 아니다. 최초 write·hash 비용이 있으며, 각 최적화의 end-to-end 절감량은 아직 미측정이다.

## 경계 적용 계획과 남은 조건

1. 운영 큐는 계속 유지한다. 현재 worker/import 경로를 덮어쓰거나 controller를 중단하지 않는다.
2. 별도 prepared worker의 전체 source/dependency/runtime manifest를 고정한다. 분석 extension 파일도 별도로 hash에 포함한다.
3. 첫 적용은 memory 내 동일 블록 Oracle 재사용으로 제한한다. 실제 checkpoint·해당 전체 채널의 첫/tail 대표 블록에서 원래/신규 경로 ordered IDs와 모든 저장 tensor를 bit 비교하고 GPU peak/time을 계측한다. GPU0 또는 Solar+진단 외 세 번째 GPU 작업을 시작하지 않는다.
4. dense 공통 계산 디스크 캐시는 위 검증과 별도로 동일 runtime의 canonical fresh 계산→저장→재로드 bit 일치 및 계산/쓰기/읽기 시간·디스크 예산 gate 후에만 선택적으로 적용한다. compact saved-payload 비교만으로 승인하지 않는다.
5. 기존 controller에는 versioned next-checkpoint dispatch hook가 없다. 현재 큐를 건드리지 않는 동안 자동 적용하지 않는다. 자연스러운 승인된 인계 경계에서 새 실행기에 다음 미시작 checkpoint/조건만 연결하고, 완료·진행 결과는 기존 source snapshot으로 유지한다. 운영 적용은 아직 안 했다.
6. 기존 reproduction gate에는 전체 external/BLAS library hash가 없다. 정확한 이전 runtime attestation을 입증할 수 없으면 scope 재사용은 `RECOMPUTE_REQUIRED`다. 현재 수치에 맞춰 과거 runtime을 추정하지 않는다. 새 gate부터 runtime manifest를 기록해야 향후 무관한 분석 변경에 대한 안전한 재사용이 가능하다.

검증 재현:
```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s experiments/mechanism_cache14 -p test_cache14.py -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/mechanism_cache14/validate_cpu.py --repo /data/pjh_workspace/CARTS --out <new-audit-directory>
```

기존 학습·결과·검증 기준·GPU 공존 조건은 변경하지 않는다. 외부 업로드/Git push 없음.
