# TRACK-C-HORIZON-RETRIEVAL-CLEAN02

**Status: Phase 1 (ETTh1_720, seed=0) COMPLETE. Result: FAIL on this cell/seed.**
Per the spec's own stopping rule ("ETTh1이 명백히 실패하면 Weather 대규모
학습을 자동으로 시작하지 말고 원인을 보고하라"), Weather_720 and the
seed=1,2 repeat were **not started**. Section 22's final Go/No-Go is
therefore a **preliminary NO-GO signal**, not a final multi-dataset/
multi-seed judgment — root-cause diagnosis is still open.

## 1. Research question

하나의 Shared Transformer Retrieval Encoder와 horizon-specific lightweight
residual adapter를 사용하면, 과거 입력만으로 short/mid/long 미래 구간마다
서로 다른 유용한 후보를 검색할 수 있는가?

## 2. 기존 C와 이번 Clean C의 차이

기존 `TRACK-A-HORIZON-RETRIEVAL-EXPERT01`은 그대로 재사용/재학습하지
않았다. 이번 실험은 전부 새로운 경로(`results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02/`,
`checkpoints/track_c_horizon_retrieval_clean02/`)에 저장되며, 기존 경로는
읽기만 하고 전혀 쓰지 않았다(§3 감사 항목 10).

## 3. 구현 감사 결과

전문:
`research/C-horizon-block-retrieval/TRACK-C-HORIZON-RETRIEVAL-CLEAN02_AUDIT.md`.
요약 — 기존 C에서 사용자가 지적한 5가지 결함을 코드로 직접 확인:

| # | 문제 | 확인 결과 |
|---|---|---|
| 1 | Shared encoder가 MLP였다 | **[ISSUE] 확인** — `build_experiment` 호출에 `relation_encoder_type` override 없음 |
| 2 | Block correction이 query에만 적용 | **[ISSUE] 확인** — `BlockCorrectionHeads.forward(z_q, block_idx)`, candidate `E`는 항상 동일 |
| 3 | Stage-1 aggregate가 uniform mean 아님 | 확인 결과 **정상**(`_gather_mean`은 plain `.mean(dim=1)`) |
| 4 | Stage-2 cache가 HostScorer weight 사용 | **[ISSUE] 확인** — `HostScorer(stage2_host, device)` + softmax-alpha 가중 |
| 5 | Global/Block base branch 독립 학습 | **[ISSUE] 확인** — shared init은 공유하지만 학습 후 발산(이번 세션에서 B트랙 재사용 중 직접 재현: 같은 셀에서 base_mse가 arm마다 49% 차이) |

## 4. Oracle 정의

```
d_G(q,i)  = MSE(y_q[0:720],  y_i[0:720])
d_B1(q,i) = MSE(y_q[0:96],   y_i[0:96])
d_B2(q,i) = MSE(y_q[96:336], y_i[96:336])
d_B3(q,i) = MSE(y_q[336:720],y_i[336:720])
```

Query/channel별 z-score 정규화 후 `softmax(-normalized/tau_T)`. `tau_T,G`는
0.10으로 고정(spec 지정). Block별 `tau_T,b`는 train subset에서 Global의
normalized entropy(엔트로피/log(n_valid))에 가장 가까운 값으로 선택
(val/test 미사용). ETTh1_720 결과: 세 block 전부 `tau_T=0.10` 선택
(`results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02/ETTh1_720/teacher_temperature_calibration.json`).

## 5. Shared Transformer 구조

`models.RelationStage1.RelationEncoder`의 기존 transformer branch를
**변경 없이** 재사용(`encode_raw(model, x, c)`), `build_experiment` override로
`relation_encoder_type='transformer'`, `relation_self_fill='zero'`,
`patch_len=16`, `stride=16`, `d_model=128`, `n_heads=4`, `e_layers=2`,
`d_ff=256`, CLS pooling(기존 기본값) 강제. Query/candidate 동일 encoder.
Candidate embedding은 매 batch마다 live encode(캐시/detach 없음) —
candidate-side gradient가 흐름(smoke test 항목 7로 검증).

## 6. Symmetric horizon adapter 구조

```python
class HorizonAdapter(nn.Module):
    def __init__(self, d_model, bottleneck=32):
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
            for _ in range(3)
        ])
        # last Linear weight AND bias zero-initialized
    def forward(self, h, block_idx):
        return F.normalize(h + self.blocks[block_idx](h), dim=-1)
```

`z_{q,b} = A_b(h_q)`, `z_{i,b} = A_b(h_i)` — **동일한 모듈 호출**을 query와
candidate 양쪽에 사용(query-only 경로 자체가 코드에 존재하지 않음). Zero-init
시 `s_{B1}=s_{B2}=s_{B3}=s_G` 정확히 성립함을 유닛테스트로 검증
(`tests/test_c_horizon_clean02.py::test_horizon_adapter_zero_init_equals_global_score`,
atol=1e-6). Adapter 파라미터 수: 25,056 (trunk 2,688,160의 ~0.9%).

## 7. Teacher temperature calibration

§4 참조. 선택 규칙과 결과는
`results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02/ETTh1_720/teacher_temperature_calibration.json`에
저장. Val/test 미사용, train subset(512 query) 고정.

## 8. Listwise loss

```
L_b = -sum_i p_T,b(i) * log p_S,b(i)          (soft listwise CE, KL(p_T||p_S)와 student
                                                 gradient 동일 — 기존 kl_loss 함수 재사용,
                                                 p_T는 detach)
L_G-Transformer = L_G
L_H-Symmetric   = (L_B1 + L_B2 + L_B3)/3 + lambda_G * L_G,   lambda_G = 0.25
```

`tau_S = 0.1`. Hard winner CE, pairwise loss, load-balancing, diversity loss,
별도 Set Oracle/Host Encoder, joint forecasting loss — 전부 추가하지
않았다(spec 명시 사항 준수).

## 9. Smoke test 결과

ETTh1_720, batch_size=4, 3 optimization step. 필수 14개 검사 전부 PASS.
**한 가지 버그 발견 및 수정**: adapter zero-init 동일성 체크(#5)를 처음에는
`train_epoch` 호출 **이후**(이미 3 step 학습된 뒤)에 검사해서 거짓 FAIL이
발생 — adapter가 더 이상 zero-init 상태가 아니었기 때문. 학습 전으로
체크 순서를 옮겨 재검증, PASS 확인
(`scripts/train_c_horizon_clean02.py` 커밋에 수정 내역 포함).
`tests/test_c_horizon_clean02.py` 7개 유닛테스트 전부 PASS(신선한
`HorizonAdapter` 인스턴스로 재검증, 학습된 상태와 무관).
리포트: `results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02/smoke/ETTh1_720/smoke_report_{g_transformer,h_symmetric}.json`.

## 10. ETTh1 Stage-1 결과

seed=0, batch_size=32, train_epochs=10, patience=5. 두 arm이 동일한
`encoder_init_sha256`(`ac3073e368...`)에서 시작, 동일 loader_seed로
batch order 확인(`batch_order_hashes_*.json`, 두 arm의 epoch1 해시가
`ed5ae8650de28798`로 정확히 일치).

| Arm | Best epoch | Test 지표 | 값 |
|---|---|---|---|
| G-Transformer | 8 | `global_h720_mse` | **0.654082** |
| H-Symmetric | **1** | `block_h720_mse` (blockwise concat) | **0.734835** |
| H-Symmetric (자신의 global, adapter 미적용) | 1 | `global_h720_mse` | 0.733689 |

**H-Symmetric의 block 결과가 G-Transformer보다 12.3% 나쁘다.** H-Symmetric은
best_epoch=1에서 조기 수렴하고 이후 계속 악화되어 epoch6에서 조기종료됐다.

## 11. Weather Stage-1 결과

**실행하지 않음.** ETTh1이 §14 Phase-1 통과 기준(개선, oracle headroom
10%+ 회수, block별 불균형 없음 등)을 명백히 만족하지 못했으므로, spec의
명시적 규칙에 따라 Weather 대규모 학습을 자동 시작하지 않았다.

## 12. Seed 반복 결과

**실행하지 않음.** ETTh1과 Weather 둘 다 seed=0에서 통과해야 seed=1,2를
진행한다는 규칙(§14 Phase 3)에 따라, ETTh1 seed=0 자체가 실패했으므로
반복하지 않았다.

## 13. Block별 retrieval 분석

| 지표 | 값 |
|---|---|
| `block1_mse` | 0.568584 |
| `block2_mse` | 0.669537 |
| `block3_mse` | 0.817209 |
| `global_vs_block1_jaccard` | 0.7627 |
| `global_vs_block2_jaccard` | 0.9079 |
| `global_vs_block3_jaccard` | 0.8396 |
| `block1_vs_block2_jaccard` | 0.7361 |
| `block1_vs_block3_jaccard` | 0.6891 |
| `block2_vs_block3_jaccard` | 0.8132 |

Jaccard가 1.0(완전 collapse)은 아니다 — block들이 서로 다른 Top-K를
고르기는 한다. 하지만 그 차이가 §19 실패유형 C에 해당한다: **Top-K는
달라졌지만 (concatenated) block MSE가 Global보다 개선되지 않았다** —
오히려 악화됐다(0.7348 vs 0.6541).

## 14. Headroom recovery

$$
\text{Recovery} = \frac{MSE_{\text{Global Student}} - MSE_{\text{Block Student}}}{MSE_{\text{Global Oracle}} - MSE_{\text{Block Oracle}}}
$$

분자가 이미 **음수**(Block Student가 Global Student보다 나쁨,
0.654082 − 0.734835 = **−0.080753**)이므로 Recovery는 음수다 — 이전
Oracle 진단(25.7% 상대 개선, §1 사전 근거)의 헤드룸을 전혀 회수하지
못했을 뿐 아니라 방향이 반대다. Query-level paired bootstrap CI는 이번
단계에서는 계산하지 않았다(부호가 이미 명백히 반대 방향이라 통계적
유의성 검정이 결론을 바꾸지 않음 — §16 효율성 원칙에 따라 생략, 필요시
후속 진단에서 추가 가능).

## 15. Optional ablation

**실행하지 않음.** spec §15: "Primary H-Symmetric 결과가 유망할 때만"
수행한다는 조건을 충족하지 못했다(결과가 유망하지 않음).

## 16. Clean Stage-2 결과

**실행하지 않음.** spec §16: "Stage 1이 ETTh1과 Weather에서 모두 통과한
경우에만 진행한다"는 조건을 충족하지 못했다.

## 17. Counterfactual 분석

Stage-2를 실행하지 않았으므로 해당 없음. 다만 §3/§10에서 확인한 사실은
기록해둔다: G-Transformer의 `global_h720_mse`(0.654082)와 H-Symmetric
자신의 (adapter 미적용) `global_h720_mse`(0.733689)가 동일한 shared init·
동일 batch order임에도 12.2% 차이가 난다 — H-Symmetric의 joint loss
($\frac{1}{3}(L_{B1}+L_{B2}+L_{B3}) + \lambda_G L_G$)가 순수 $L_G$ 단독
학습(G-Transformer)보다 trunk의 global-score 품질 자체를 더 나쁘게
만들었을 가능성을 시사한다 — multi-task 간섭 후보(§19 원인 분리 참고).

## 18. 계산량

| | G-Transformer | H-Symmetric |
|---|---|---|
| Trunk params | 2,688,160 | 2,688,160 |
| Adapter params | 0 | 25,056 |
| Best epoch | 8 | 1 |
| Wall clock (전체, 10-epoch 예산 내) | 2711s (~45min) | 1836s (조기종료, ~31min) |

Shared Transformer는 양쪽 arm에서 side당 1회만 실행되고(query 1회,
candidate 1회), H-Symmetric은 그 결과에 3개의 작은 adapter + 3회의 score
lookup만 추가한다(설계상 오버헤드가 매우 작음 — 실측 파라미터 수 기준
+0.9%). 계산량 자체는 문제가 아니었다; 문제는 학습 결과다.

## 19. 실패 유형 분류

spec §14(Phase 1 실패 유형)에 따라 분류:

- **Type C** (Top-K는 달라졌지만 block MSE가 개선되지 않음): Jaccard가
  1.0이 아닌데도 concatenated block MSE가 Global보다 나쁨 — **해당**.
- **Type F** (최적 epoch가 1이고 이후 단조 악화): H-Symmetric의 best_epoch=1,
  이후 5 epoch 연속 개선 없어 조기종료 — **해당**.
- Type A(teacher는 다른데 student score가 collapse)는 아님 — Jaccard가
  1.0이 아니므로 완전 collapse는 아님.
- Type D(train은 개선, val 악화)는 직접 확인하지 않았다(epoch_metrics.csv에
  train/val 곡선이 있으나 이 보고서에서 별도 표로 뽑지 않음 — 후속
  진단 후보).
- Type G(구현/aggregation mismatch)는 §3 감사와 smoke test 14항목 전부
  PASS로 배제했다고 보지만, §17에서 발견한 joint-loss 간섭은 "구현
  버그"는 아니고 "loss 설계의 부작용"에 더 가깝다.

## 20. 논문 contribution 가능성

이 단계에서는 판단하지 않는다 — 결과가 negative이고 원인 진단이 끝나지
않았으므로 contribution 여부를 논하는 것은 시기상조다.

## 21. 한계

- ETTh1_720 **단일 셀, 단일 seed**만 실행했다 — 일반화 주장 불가.
- Query-level paired bootstrap, chronological block-level curve 등
  D-트랙에서 사용했던 엄밀한 통계 검증을 이번에는 아직 적용하지 않았다
  (부호가 이미 명백해 우선순위가 낮았음).
- H-Symmetric의 joint-loss 간섭 가설(§17)은 추측이며, `lambda_G` sweep
  등으로 직접 검증하지 않았다(spec이 "ETTh1 seed=0 결과가 block
  specialization 부족 때문에 실패한 것이 확인될 때만" sweep을 허용하므로,
  그 확인 자체가 아직 안 된 상태 — 다음 단계 후보).

## 22. 최종 Go/No-Go 판정

**ETTh1_720 seed=0: 예비 NO-GO 신호.** spec §14의 NO-GO 조건 중
"두 데이터셋 모두 Global보다 악화"는 Weather를 실행하지 않아 확인 불가;
"Top-K가 Global로 collapse"는 해당 안 됨(Jaccard<1); "Oracle gap의
의미 있는 부분을 전혀 회수하지 못함"은 **해당**(Recovery가 음수).

이 결과 하나만으로 spec이 요구하는 최종(멀티데이터셋·멀티시드) GO/NO-GO를
내리는 것은 시기상조이므로, **최종 판정을 아래와 같이 조건부로 기록한다**:

> ETTh1_720 seed=0에서 H-Symmetric은 Global보다 명백히 나쁘고(−12.3%),
> best_epoch=1 직후 단조 악화되는 패턴(Type F)을 보였다. Oracle 레벨의
> block-specific headroom(사전 근거 25.7%)은 이번 학습된 모델에서 전혀
> 회수되지 않았다. spec의 명시적 정지 규칙에 따라 Weather 및 추가 seed는
> 실행하지 않았다. **원인 진단(구현 vs teacher vs student capacity vs
> representation collapse vs aggregation mismatch vs joint-loss
> 간섭)이 먼저 필요하며, 그 결과에 따라 Weather 확대 여부 또는 완전
> 중단을 결정해야 한다.**
