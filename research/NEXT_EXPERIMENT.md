# Question
Do v2 Stage2 gains persist when fusion weight is controlled?
# Observation
Seven of eight saved v2 pairs improve Stage2 MSE; per-arm lambda was refit. Full ablation shows lambda-sensitive effects.
# Interpretation
Candidate accuracy is a supported explanation; semantic head specialization and exclusion of fusion effects are not established.
# What Is Not Yet Established
Fixed-lambda v1/v2 contrast, seed robustness, remaining four P100 settings, cleaned-weather training impact.
# Competing Hypotheses
## H1
Changed retrieval predictions explain most v2 gains.
## H2
Lambda refitting materially contributes to apparent gains.
# Proposed Experiment
NOT RUN. After user approval, evaluate existing v1/v2 caches under a common lambda selected using validation only. First finish the existing P100 queue; no new training was launched during integration.
# Controlled Variables
Same Base, split, checkpoint, candidate support and query IDs; preserve original primary results.
# Metrics
Test MSE/MAE for refit versus fixed lambda, with paired differences; test never selects lambda.
# Expected Outcomes
## If H1 is correct
Most gains remain under fixed lambda.
## If H2 is correct
Gains shrink or reverse under fixed lambda.
# Decision Rule
Report per-cell directions without test-driven tuning; seed0 cannot establish population-level superiority.
