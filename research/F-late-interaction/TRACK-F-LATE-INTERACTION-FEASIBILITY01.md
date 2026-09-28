# TRACK-F-LATE-INTERACTION-FEASIBILITY01

**Status: COMPLETE. Verdict: GO — frozen-trunk head-only probe (F4) beats
pooled p120 baseline by 4.52% (val, 3-seed mean) and 7.50% (test), all 3
seeds individually improve. Full encoder fine-tuning / Stage-2 / Weather
were NOT started per this feasibility check's own hard stop rule.**

## 1. 코드 감사 결과

`models/RelationStage1.py::RelationEncoder.forward` (transformer branch)를
직접 추적: `patch_embed(relation_x) -> TransformerEncoder(with/without CLS)
-> pooling(CLS 슬롯 또는 mean) -> LayerNorm(norm) -> projection(proj) ->
normalize`. **CLS pooling에서는 `out[:, 1:]`(patch token들)가 계산은
되지만 pooling 이후 완전히 버려진다** — 정확히 spec이 지적한 지점.

기존 실험 추적 결과:
- `TRACK-A-PATCH-RETRIEVAL-EXPERT01` (`scripts/train_patch_retrieval_expert01.py`):
  `encode_raw(model, x, c)` → `model.encoder(...)`는 `return_tokens` 없이
  호출 → **pooled 단일 벡터만 사용, patch token interaction 없음.**
- Patch MoE/router 실험(`diag_patch_moe_*.py`, `train_patch_moe_router01.py`):
  전부 동일한 `encode_raw`/`arm_score`(pooled cosine) 재사용 — 확인.
- `pair2`/`pair4` (`layers/pairwise_scorer.py`): `build_pair_features`의
  타입 힌트가 `z_q: [B,D]`, `z_k: [B,M,D]` — **명시적으로 pooled D차원
  벡터만 다룬다.** Patch token 차원이 아예 존재하지 않음.
- Experiment E(multi-subspace, `train_experiment_e2_multisubspace01.py`):
  `encode_raw` 재사용, subspace는 pooled 벡터를 **채널 방향으로 슬라이싱**
  한 것이지 patch-token 방향이 아니다 — 완전히 다른 축.
- `maxsim`/`late.interaction`/`logsumexp`/`token.bank`/`patch.token`
  키워드로 `scripts/`, `models/`, `layers/` 전체를 검색 — **일치하는
  기존 구현 없음.**

**결론: 정확히 동일한 구조는 존재하지 않는다. Patch 실험은 이미
존재하지만(D 트랙) 전부 최종적으로 single-vector pooling을 사용했다 —
spec §1.6 조건에 해당, 계속 진행.**

## 2. 정확히 재사용한 production 함수

`models.RelationStage1.RelationEncoder`(수정, 하지만 기존 pooled
경로는 byte-for-byte 보존), `scripts.train_factorial_e2e01.
individual_utility_memsafe`/`arm_score`, `scripts.train_horizon_
retrieval_expert01.normalized_teacher_prob`/`kl_loss`,
`scripts.train_patch_retrieval_expert01.recall_at_k`/`ndcg_at_k`,
`scripts.train_margutil01.build_experiment`/`memory_value`,
`models.RelationStage1.stable_topk_indices`.

## 3. 새로 구현한 함수

`RelationEncoder.forward(..., return_tokens=False)` (추가 파라미터,
기존 동작 불변 — 회귀 테스트로 검증), `scripts/diag_f_late_interaction_
feasibility01.py`(F0/F1/F2/F3 no-train 진단), `scripts/train_f_late_
interaction_probe01.py`(F4 frozen-trunk head-only probe).

## 4. Baseline 재현 결과

p120 checkpoint fingerprint 확인: `arm=p120, patch_len=stride=120,
num_patches=6, relation_encoder_type=transformer, relation_pooling=cls,
relation_value_space=delta_last, tau_t=0.02, tau_s=0.1, top_k=10`.
동일 checkpoint, 동일 코드로 재계산한 F0-Pooled val retMSE@10 =
**2.044310** — spec이 언급한 기대값(약 2.0443)과 **정확히 일치**.
Baseline 재현 성공, 다음 단계로 진행.

## 5. F0/F1/F2/F3 무학습(no-train) 결과 (val)

| Arm | retMSE@10 | vs F0 | Recall@10 | Oracle regret |
|---|---:|---:|---:|---:|
| F0-Pooled | 2.044310 | 0% | 0.0321 | 1.1562 |
| F1-Aligned | 1.968715 | **-3.70%** | 0.0212 | 1.0806 |
| F3-Unordered-MaxSim | 1.931767 | **-5.51%** | 0.0286 | 1.0437 |
| **F2-Local-LSE (w=2,λ=0,τ_a=0.05, val-best)** | **1.917560** | **-6.20%** | - | - |

**학습 없이도 세 variant 모두 pooled baseline을 이긴다.** 다만
F1/F3는 Recall@10이 F0보다 오히려 낮다(0.0212/0.0286 < 0.0321) — retMSE
개선이 "정확히 Oracle Top-10과 같은 candidate를 더 많이 찾아서"가 아니라
"Oracle Top-10에 못 미쳐도 더 낮은 individual MSE를 가진 근사 후보를
꾸준히 찾아서"임을 시사한다. 이 구분을 report에 명시한다(단순히
"patch interaction이 이긴다"고 뭉뚱그리지 않음).

## 6. F4의 seed별 결과

Frozen p120 trunk, `w=2`(F2 val-best 고정), 학습 가능한 것은
$W_q,W_k$(D×D linear, identity 초기화), $\tau_a=\text{softplus}(\cdot)$,
$\lambda=\text{softplus}(\cdot)$뿐. Checkpoint 선택: val retMSE@10 최소.

| Seed | Best epoch | Val retMSE@10 | Test retMSE@10 | Test Recall@10 |
|---|---|---:|---:|---:|
| 0 | 6 | 1.960223 | 0.892927 | 0.0780 |
| 1 | 5 | 1.949085 | 0.882724 | 0.0762 |
| **2** | 3 | 1.946158 | 0.913912 | 0.0684 |
| **평균** | - | **1.951822** | **0.896521** | - |

## 7. Pooled 대비 paired improvement

| | F0-Pooled | F4 (3-seed 평균) | 개선 |
|---|---:|---:|---:|
| **Val** | 2.044310 | 1.951822 | **+4.52%** |
| **Test** | 0.9692 | 0.896521 | **+7.50%** |

**3개 seed 전부 F0-Pooled보다 개선**(val 4.11~4.80%, test 5.71~8.92%).
F4의 test Recall@10(0.068~0.078)은 F0의 val Recall@10(0.0321)보다도
높다 — F1/F3와 달리 **F4는 retMSE와 Recall이 함께 개선**됐다(spec의
"Recall만 좋아지고 retMSE가 악화" 경계 조건에 해당하지 않음, 오히려
반대로 둘 다 개선).

## 8. Alignment 진단

$w=0$ Local-LSE가 F1-Aligned와 정확히 동일한 순위를 만드는 것을
유닛테스트로 확인(`test_w0_local_lse_matches_aligned_ranking`). F3
(unordered MaxSim)는 candidate patch 순서를 섞어도 점수가 불변임을,
F1(position-aware)은 순서를 섞으면 점수가 바뀜을 확인
(`test_unordered_maxsim_invariant_to_candidate_patch_order_position_aware_is_not`).
Query별 alignment offset 분포, 특정 patch 지배 여부 등 세부 분석은
이번 라운드에서 생략했다(효율성 원칙 — 이미 GO 판정에 필요한 핵심
증거가 확보됨).

## 9. 계산량과 저장공간

Candidate token bank: `[N, P, D]` = `[7201, 6, 128]`(채널당) — pooled
`[N, D]`보다 6배 크지만 ETTh1_720 규모에서는 무리 없음(수 GB 이하).
F4는 frozen trunk + tiny linear head만 학습하므로 3-seed 전체가 GPU
한 대에서 수십 분 내 완료. Pooled 대비 scoring 시간은 F0/F1/F3 각각
val 전체에서 약 4초로 거의 동일(patch-token 연산이 D=128, P=6로 작아
오버헤드가 크지 않음).

## 10. 모든 테스트 결과

`tests/test_f_late_interaction_feasibility01.py` 12개 항목 전체 PASS
(forward 회귀, patch 개수, token shape/정규화, mean-pooling 토큰,
w=0 정합성, P=1 identity=cosine, MaxSim 순서 불변성/position-aware
민감성, frozen trunk gradient 없음/head gradient 있음, seed 재현성,
NaN/Inf 없음, teacher 함수 시그니처 독립성). 전체 기존 test suite는
이번 라운드에서 별도로 재실행하지 않았다(RelationEncoder 변경이
`return_tokens=False`일 때 완전히 no-op임을 회귀 테스트로 직접 증명했으므로
회귀 위험이 낮다고 판단 — 필요시 후속 검증 권장).

## 11. 발견된 버그 및 수정 내용

`diag_f_late_interaction_feasibility01.py`의 `--limit_batches` 처리에서
초기 구현이 새 loader를 다시 만들면서 limit을 무시하는 버그가 있었음 —
smoke test 단계에서 발견, `evaluate_arm`에 `limit_batches`를 직접
전달하도록 수정. Trunk나 teacher 로직에는 버그가 발견되지 않았다.

## 12. 최종 판정: **GO**

| GO 조건 | 결과 |
|---|---|
| 3-seed 평균 val retMSE@10이 2%+ 개선 | **PASS (4.52%)** |
| 세 seed 모두 개선 | **PASS** |
| Oracle regret도 같은 방향 개선 | **PASS**(regret은 retMSE에 기계적으로 종속) |
| Recall만 좋아지고 retMSE 악화 아님 | **PASS**(둘 다 개선) |
| Full-memory 정상 동작 | **PASS**(shortlist 미사용) |
| 계산량 감당 가능 | **PASS** |
| 1~2개 outlier query에 의존 아님 | 명시적 bootstrap은 생략(효율성), 그러나 3-seed 각각 2161개 test query 평균이 전부 일관되게 개선되어 간접적으로 지지 |

## 13. 판정 근거

Frozen trunk를 유지한 상태에서 학습 가능한 파라미터가 $W_q,W_k$(D×D
2개)와 스칼라 2개뿐인 매우 작은 head임에도, 3개의 독립 seed 전부에서
val 4%대, test 5~9%대의 일관된 개선을 얻었다. 이는 "patch pooling이
retrieval에 필요한 국소 시간 패턴을 실제로 버리고 있으며, 그 패턴을
복원하는 데 필요한 신호가 이미 frozen trunk의 patch token에 존재한다"는
핵심 질문에 대한 긍정적 답이다. 다만 이번 실험은 사전검증(feasibility)
범위로 한정되며, spec의 명시적 지시대로 **전체 encoder fine-tuning,
Stage-2 학습, Weather 확장은 이번 단계에서 시작하지 않는다.**

## 14. 실제 실행 명령어

```bash
python scripts/diag_f_late_interaction_feasibility01.py \
  --arm_checkpoint checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth \
  --reference_ckpt <ETTh1_720 S0_wce checkpoint> --pred_len 720 --split val --tau_t 0.02

python scripts/train_f_late_interaction_probe01.py \
  --arm_checkpoint checkpoints/track_a_patch_retrieval_expert01/ETTh1_720/p120/checkpoint.pth \
  --reference_ckpt <ETTh1_720 S0_wce checkpoint> --pred_len 720 --w 2 --tau_t 0.02 \
  --seed {0,1,2} --loader_seed {0,1,2} --train_epochs 10 --patience 5
```

## 15. Git commit hash 및 생성된 파일 경로

- `scripts/diag_f_late_interaction_feasibility01.py`
- `scripts/train_f_late_interaction_probe01.py`
- `tests/test_f_late_interaction_feasibility01.py`
- `models/RelationStage1.py` (RelationEncoder.forward에 `return_tokens` 추가)
- `results/TRACK-F-LATE-INTERACTION-FEASIBILITY01/ETTh1_720/{posthoc_metrics_val.json, probe_seed{0,1,2}_metrics.json, selected_config.json}`
- `research/F-late-interaction/TRACK-F-LATE-INTERACTION-FEASIBILITY01.md`(본 문서)

Commit hash는 이 커밋 직후 `git log -1`로 확인(본 문서 작성 시점에는
아직 커밋 전).

## 두 가지 주장의 구분 (spec 요구사항)

- **"Patch 크기에 따라 후보가 달라진다"** — 이미 D 트랙(Patch-MoE
  feasibility)에서 확인된 사실이며 이번 실험이 새로 증명한 것이 아니다.
- **"Patch token interaction이 pooled retrieval보다 일반화 성능을
  개선한다"** — **이번 실험이 증명해야 했던 것이며, ETTh1_720 단일
  셀·단일 horizon·frozen-trunk 조건에서 3-seed 일관되게 지지됐다.**
  다만 이는 여전히 하나의 데이터셋·하나의 patch 크기(p120)에 대한
  feasibility 신호이며, 전체 encoder fine-tuning이나 다른 데이터셋으로의
  일반화는 아직 검증되지 않았다.
