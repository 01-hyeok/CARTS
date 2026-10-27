# PHASE C -- Decision Gate

Synthesizes Phase A (Oracle Lambda) and Phase B (past-only
observability) into the pre-registered Case A-D framework (spec
sections 19-22). No new computation in this file.

## Research questions, answered directly

**RQ1 (Weather에서 retrieval이 Base보다 나쁜 이유는 retrieval output
자체가 유용하지 않기 때문인가?)** -- Partially, and it is dataset
-specific, not uniform even within Weather. Phase A found retrieval
alone is less accurate than Base on average in EVERY cell including
ETTh1 (Table 4, mean gain always negative) -- so H1 is always
present to some degree. But its SEVERITY differs: on Weather_720, even
the validation-optimal GLOBAL lambda (a deployable, non-oracle
quantity) cannot quite reach Base (0.3224 vs 0.3194) -- a genuine H1
ceiling. On Weather_96, the SAME validation-optimal global lambda
DOES beat Base (0.1687/0.1671 vs 0.1694) -- so Weather_96's current
failure is NOT fully explained by H1; the deployed Stage2 training
simply picked a bad lambda (H2).

**RQ2 (Query별 최적 Lambda를 사용하면 Weather에서도 Base를 상당히
개선할 수 있는가?)** -- Yes, clearly, in the oracle sense: per-query
-channel oracle utilization headroom is 19-29% on Weather (larger
than ETTh1's 9-18%), and the oracle gain over Base is 13-18% on
Weather. But Phase B shows this oracle headroom is NOT reliably
realizable from past-only information with a deployable gate -- the
binary gate's test-set gain on Weather is +0.0006 (Weather_96) and
-0.0004 (Weather_720), nowhere close to the oracle's 13-18%.

**RQ3 (Retrieval이 유용한 query와 해로운 query를 과거 정보만으로
구분할 수 있는가?)** -- Partially. AUROC is clearly above chance
everywhere (0.51-0.71) and actually highest on Weather. But "partially
observable in AUROC" does not mean "reliably deployable" -- the
binary gate derived from the SAME probe flips sign val-to-test in 3
of 4 cells. Observability and deployability are NOT the same
question, and this track's own data cleanly separates them.

**RQ4 (현재 문제의 주된 원인은 Selection인가, Utilization인가, 아니면
둘 다인가?)** -- **Both, and the mix is cell-specific**:

| Cell | H1 (Selection) severity | H2 (Utilization) severity |
|---|---|---|
| ETTh1_96 | Oracle gain vs Base 16.3% (moderate H1 ceiling) | ~0% global-lambda headroom -- current lambda already optimal |
| ETTh1_720 | Oracle gain vs Base 16.5-17.9% (moderate) | ~0% (slightly negative) -- already optimal |
| Weather_96 | Oracle gain vs Base 18.0-18.4% (moderate, similar to ETTh1) | **2.5-4.5% -- current lambda is measurably suboptimal** |
| Weather_720 | Oracle gain vs Base 13.5-14.0%, but val-opt-global still trails Base slightly (**H1 ceiling is real here**) | **13.3-16.4% -- the LARGEST utilization gap of any cell** |

## Case classification (spec section 19-22)

**Not Case A** (no oracle headroom) -- headroom is large and real
everywhere (9-29% per-query, plus a further 2.5-16.4% at the
global-lambda level on Weather).

**Not cleanly Case B** (headroom exists, zero observability) -- Phase
B's AUROC is clearly above chance everywhere, strongest on Weather;
calling this "no observability" would misstate the data.

**Not cleanly Case C** (headroom exists, observability exists,
deployable gate improves Test) -- the deployable gate's test-set
result is only robustly positive on ETTh1_720; it is negative
(worse than doing nothing) on ETTh1_96 and Weather_720, and
negligible on Weather_96.

**The actual result is a cell-specific mix that the pre-registered
4-case framework does not have a clean label for**, closest to a
**fragile/partial Case C limited to ETTh1_720**, combined with
**Case B-leaning-but-not-quite (detectable-but-undeployable
observability) on the other 3 cells**. This is reported as such
rather than forced into one box.

## Case D -- ETTh1 vs Weather comparison table (spec section 22, completed)

| Item | ETTh1_720 | Weather_96 | Weather_720 |
|---|---:|---:|---:|
| Retrieval win fraction (R better than B, query-channel) | 0.413 | 0.458 | 0.304 |
| Oracle lambda=0 fraction (query-channel) | 0.189 | 0.348 | 0.366 |
| Current (deployed) global lambda | 0.479 | 0.395 | 0.509 |
| Val-optimal global lambda | 0.487 | 0.226 | 0.180 |
| Oracle lambda MSE (query-channel, test) | 0.4677 | 0.1389 | 0.3090 |
| Oracle gain vs Base (%) | 16.5 | 18.0 | 14.0 |
| Global-lambda headroom (%) | -0.07 | 4.51 | 16.38 |
| Probe B2 AUROC (test) | 0.513 | 0.709 | 0.686 |
| Deployable gate implied gain (test) | **+0.0119** | +0.0006 | **-0.0004** |

The deployed CURRENT global lambda is, strikingly, closer to the
val-optimal one on ETTh1_720 (0.479 vs 0.487) than on either Weather
cell (0.395 vs 0.226 on H96; 0.509 vs 0.180 on H720) -- the Stage2
training procedure's own optimization quality is itself
dataset-dependent, not just the underlying retrieval signal's
quality.

## Overall verdict

No single mechanism explanation (pure H1 or pure H2) fits all 4
cells. The honest summary: **ETTh1's current global lambda is
already close to its own achievable optimum, so further gains there
would have to come from per-query adaptation (RQ2/RQ3's territory,
where ETTh1_720 shows the one robust positive signal this track
found). Weather's current global lambda is clearly suboptimal
(fixable with zero new modeling, just better Stage2 optimization),
AND still has a real per-query oracle ceiling above what even a
perfect global lambda reaches -- but the past-only signal needed to
reach that ceiling is not reliably extractable with the tiny probes
tested here.** Per spec section 21 (Case C), this track does NOT
recommend building a new Stage2 architecture (query-conditional
lambda, usefulness predictor, selective fusion) given the
deployable-gate instability found in Phase B -- except possibly as a
narrow, ETTh1_720-specific follow-up, which is out of this track's
own scope to pursue further.
