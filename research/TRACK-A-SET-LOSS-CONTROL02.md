```text
STATUS: DONE -- H96 + H720, Stage-1 (TRACK-A-SET-LOSS-CONTROL02) + Stage-2
(EXP-SET-LOSS-STAGE2-RETRAIN02), all 4 arms x 2 horizons x 2 stages = 16 arm-runs, all DONE,
zero FAILED.json.
```

## Background (seed correction)

This experiment corrects `TRACK-A-SET-LOSS-CONTROL01`'s unauthorized `--seed 1` (the Factorial
baseline `A0_hard_choice` is meant to reproduce, `set_onpolicy_cosine`, uses `seed=0`). Full
correction rationale, RNG-control infrastructure (`scripts/rng_control01.py`), the A0-vs-
Factorial exact-equivalence gate (20/20 steps, max_abs_diff=0.0, PASSED), and the H96/H720
batch-order cross-arm hash gates are all documented in this session's earlier turns and in
`research/REVIEW_FOR_CHATGPT.md`'s seed-correction section. `TRACK-A-SET-LOSS-CONTROL01`'s own
results are preserved (not deleted), marked `UNAUTHORIZED_SEED_WARNING.md`.

## Reproducibility (both horizons)

- All 4 arms per horizon share identical `encoder_init_sha256`/`set_conditioner_init_sha256`
  (shared-init hash asserted at load time).
- Batch-order SHA256 identical across all 4 arms at EVERY epoch, both horizons: H96 8/8 epochs,
  H720 10/10 epochs, confirmed via the automated cross-arm gate
  (`scripts/gate_set_loss_control02_batch_order.py`), not just eyeballed.
- Stage-2 FR-Agg cross-check: diff=0.000000 for all 8 Stage-2 arms (delta-space
  `cache_schema_version=corrected_delta_v1`, same corrected pipeline as
  TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED / EXP-ORACLE-WCE-CONTROL01).

## Stage 1 -- FR-Agg (Oracle-agnostic comparison metric)

| horizon | loss | val_FR-Agg | test_FR-Agg |
|---|---|---:|---:|
| H96 | A0 Hard Choice CE | 0.717068 | 0.417953 |
| H96 | A1 Adaptive MultiPos | 0.715223 | 0.419567 |
| H96 | A2 SRM | **0.704820** | 0.418769 |
| H96 | A3 Set-Utility SoftCE | 0.718263 | 0.419389 |
| H720 | A0 Hard Choice CE | 1.622801 | 0.595195 |
| H720 | A1 Adaptive MultiPos | 1.610715 | 0.601932 |
| H720 | A2 SRM | 1.628706 | **0.587121** |
| H720 | A3 Set-Utility SoftCE | 1.585400 | 0.603817 |

A2 (SRM) has the best H96 val FR-Agg and best H720 test FR-Agg, though no single arm wins on
both val and test at both horizons -- val/test rankings do not fully agree (e.g. H720: A3 has
best val FR-Agg, A2 has best test FR-Agg), consistent with single-seed noise in this metric.

## Stage 2 -- corrected, freshly-retrained fusion (frozen-host injection never used)

Independent Base (IB): H96=0.386451, H720=0.491450.

| horizon | loss | final_MSE | delta_vs_IB | gate_mean |
|---|---|---:|---:|---:|
| H96 | A0 Hard Choice CE | 0.37012 | -4.23% | 0.447 |
| H96 | A1 Adaptive MultiPos | 0.37014 | -4.22% | 0.448 |
| H96 | A2 SRM | 0.37092 | -4.02% | 0.442 |
| H96 | A3 Set-Utility SoftCE | **0.36854** | **-4.64%** | 0.453 |
| H720 | A0 Hard Choice CE | 0.52818 | +7.47% | 0.337 |
| H720 | A1 Adaptive MultiPos | 0.52083 | +5.98% | 0.448 |
| H720 | A2 SRM | **0.48810** | **-0.68%** | 0.195 |
| H720 | A3 Set-Utility SoftCE | 0.50953 | +3.68% | 0.461 |

**Key finding**: at H720, **A2 (SRM) is the only arm that beats Independent Base** (-0.68%);
the other three all end up WORSE than IB (+3.7% to +7.5%). This mirrors the pattern already
seen in `TRACK-A-SET-NORMREGRET-CONTROL01` (a separate, later experiment with a corrected
temperature-scaled normalized-regret formula) where H720 was the horizon where loss choice
mattered and Hard CE underperformed relative to alternatives. A2's H720 win here comes with a
notably LOW gate_mean (0.195, lowest of the 4 H720 arms) -- i.e. it relies on retrieval least,
yet still wins, suggesting its base_head/gate combination generalizes better at long horizon
rather than retrieval quality itself being decisive. At H96 the 4 arms are close (0.3685-0.3709,
~1% relative range), all comfortably beating IB, with A3 narrowly best.

## Ten closing questions

**1-2. (config/pairing)** All seed=0, exact-paired within this campaign (batch-order hashes
identical across arms at every horizon/epoch) -- covered above.

**3. 어느 loss가 가장 낮은 H720 FR-Agg를 보였는가?** test 기준 A2(0.587121), val 기준 A3
(1.585400) -- val/test 불일치.

**4. Stage 1 FR-Agg 개선이 Stage 2 개선으로 이어졌는가?** 부분적으로. A2는 H720 test FR-Agg
최저이면서 Stage-2도 유일하게 IB를 이김 -- 방향 일치. 하지만 A3는 H720 val FR-Agg가 최저인데
Stage-2는 IB보다 나쁨(+3.68%) -- 불일치. H96은 4개 arm 모두 FR-Agg/Stage-2 차이가 작아
결론짓기 어려움.

**5. H96/H720 방향이 일관적인가?** 아니다. H96은 4개 arm 모두 IB를 이기고 서로 근접
(A3가 근소 우세). H720은 A2만 IB를 이기고 나머지 3개는 진다 -- Set-Oracle 실험 전반에서
반복적으로 나타나는 H96/H720 분기 패턴.

**6. 단일 seed 변동 가능성?** H96의 4-arm 차이(0.3685-0.3709, ~1%)는 seed noise 수준일 가능성
높음. H720의 A2 우위(다른 arm 대비 4~8%p)는 더 크지만 단일 seed(seed=0)만으로 확정 불가.

**7. 3-seed 검증 추천 arm?** `A2_srm` at H720 -- 유일하게 IB를 이긴 arm이자 가장 낮은 gate
사용률로 이긴 특이 패턴이라, 진짜 효과인지 확인이 가장 필요함.

## Files

- `results/TRACK-A-SET-LOSS-CONTROL02/**` (Stage-1, 4 arms x 2 horizons)
- `results/EXP-SET-LOSS-STAGE2-RETRAIN02/**` (Stage-2, 4 arms x 2 horizons)
- `scripts/train_set_loss_control02.py`, `scripts/run_set_loss_control02_h96.sh`,
  `scripts/run_set_loss_control02_h720.sh`, `scripts/gate_set_loss_control02_batch_order.py`
- `scripts/build_setlossctrl_retrieval_cache02.py`, `scripts/train_setlossctrl_stage2_retrain02.py`,
  `scripts/run_setlossctrl02_stage2_retrain02_h96.sh`, `scripts/run_setlossctrl02_stage2_retrain02_h720.sh`
- `scripts/rng_control01.py` (shared RNG-control infra, also reused by
  EXP-ORACLE-WCE-CONTROL01 / TRACK-A-SET-NORMREGRET-CONTROL01)

No existing file overwritten; `TRACK-A-SET-LOSS-CONTROL01`'s seed=1 results preserved with
`UNAUTHORIZED_SEED_WARNING.md`. commit/push not performed this round.
