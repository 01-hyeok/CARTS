# Base vs ALL retrieval methods (Legacy + New), TEST MSE -- correction note

Prompted by a user spot-check: an earlier status update in this session
said "Weather: new SoftTopK architecture beats every legacy variant"
-- true, but incomplete and potentially misleading, since it only
compared retrieval methods to EACH OTHER, never to the plain Base
Forecaster (no retrieval at all). This note corrects that.

## Verification: same Base checkpoint, not a discrepancy

Independently recomputed the Base-only TEST MSE for Weather_96 from
the exact checkpoint
(`checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth`)
on the same TEST split used by both the legacy tracks and this track:

```
재계산한 Base test MSE: 0.1693644558551501
Legacy 보고값 (TRACK-HARD-EXPERT-V5-P100-ALLH01 Table 1): 0.16936
```

Matches to 4 decimal places -- same checkpoint, same data, not a bug or
provenance mismatch.

## The actual pattern (Weather)

| H | Base | Best Legacy retrieval | Best New retrieval (any method/arm) | Base wins? |
|---|---:|---:|---:|---|
| 96  | 0.16936 | 0.17665 | 0.16527 (V5/QA) | **No -- New QA beats Base, by ~2.4%** |
| 192 | 0.19908 | 0.21758 | 0.20009 (V2/QA) | Yes, by ~0.5% |
| 336 | 0.24567 | 0.28080 | 0.24859 (V2/QA) | Yes, by ~1.2% |
| 720 | 0.31943 | 0.38553 | 0.33063 (Joint/QA) | Yes, by ~3.5%, widening |

**Every retrieval method tested on Weather in this entire multi-week
research program -- Legacy V5/Hard/Frozen-Router/Joint-Router AND the
new SoftTopK NOQA/QA, across all 6 methods -- loses to the plain Base
Forecaster at H192/H336/H720.** Only the new architecture's QA arm
manages to beat Base, and only at H96.

(ETTh1 is the opposite: Base loses to every retrieval method at all 4
horizons -- `base < legacy_best` is False at all 4 ETTh1 cells, vs True
at all 4 Weather cells, per the raw numbers in `legacy_baselines.json`.)

## This is not a new finding -- it reproduces a known, already-documented pattern

From `research/REVIEW_FOR_CHATGPT.md` (2026-10-08 addendum, an earlier
track in this same research program): *"Weather's Base Forecaster beats
the best trained retrieval arm at H192/H336/H720 with a widening margin
[0.4%->1.3%->3.4%], but V5/V1 edge out Base by ~0.9% at H96
specifically -- not a universal Base-wins-everywhere result."*

The new SoftTopK E2E numbers reproduce the SAME shape (Base wins at
H192/336/720, widening with horizon; retrieval wins only at H96) with
similar magnitudes (new QA's H96 margin is 2.4% vs the old 0.9%; new's
H720 Base-win margin is 3.5% vs the old 3.4%, nearly identical).
Changing the retrieval architecture (Legacy Uniform Top-10 -> SoftTopK
E2E, with or without Quality-Aware fusion) does not change this
underlying Weather-specific pattern -- it is evidently a property of
the dataset/retrieval-memory relationship at longer horizons, not of
any one method's implementation.

## Correction to earlier framing

"New beats Legacy on Weather" (true, large margins) should NOT be read
as "retrieval augmentation helps on Weather" -- it does not, beyond
H96; what improved is retrieval-vs-retrieval, while retrieval-vs-Base
stayed structurally the same as it has been throughout this research
program. This will be stated plainly in the final REPORT.md, not
folded into a single positive-sounding headline.
