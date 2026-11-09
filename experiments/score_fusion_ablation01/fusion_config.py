from pathlib import Path
ROOT = Path('/data/pjh_workspace/CARTS')
HERE = ROOT / 'experiments/score_fusion_ablation01'
OUT = ROOT / 'results/TRACK-SCORE-FUSION-ABLATION01'
GPU = 'GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
CELLS = [d+'_'+str(h) for d in ('ETTh1','Weather') for h in (96,192,336,720)]
POP = ROOT / 'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/full_population_completion/raw'
PROTOCOL = {
 'seed':0, 'K':10, 'channels':'ALL in every GPU batch', 'population':'all validation and test queries',
 'gamma_grid':[.01,.03,.05,.1,.2,.5,1.], 'gamma_selection':'minimum ALL native validation RetMSE; smaller gamma on exact tie; cross support frozen',
 'checkpoint_selection':'immutable existing validation-selected checkpoints; no reselection',
 'fusion_precision':'E0 canonical FP32; E1-E4 fixed-order FP64 arithmetic on the identical existing FP32 head cosines',
 'score_ties':'existing stable descending score, original support position tie-break (Full global ID; P100 immutable pool position)',
 'ndcg':'nonnegative direct gain 1/(1+MSE), log2(rank+1) discount; IDCG from same-support Oracle10; no exponent',
 'retmse':'mean candidate individual reconstructed future MSE over K=10, then equal Q x C mean in train-standardized space',
 'head_useful':'unique IDs in union of five head Top10 intersect SAME SUPPORT Oracle10; exclusive useful appears in exactly one head',
 'retention':'selected useful / union useful per QxC (zero denominator undefined), plus pooled retained/available counts',
 'query_win_loss':'channel-mean RetMSE delta versus E0 per timestamp; exact zero tie; worst deterioration is max delta and worst 1/5/10 percent timestamps',
 'recovery':'(Random RetMSE - method RetMSE)/(Random RetMSE - Oracle RetMSE); zero denominator undefined; no clipping',
 'support_stability':'Spearman and Top10 overlap on SAME P100 IDs; shared-Full-logit control separated from native-P100 encoding arithmetic',
 'random':'reuse verified stateless seed0 SeedSequence([0,query_start,channel]) random scores, stable Top10 without replacement; ONE draw',
 'batch':'canonical encoder B32 enclosing blocks; consumer query/candidate slicing cannot change canonical embedding semantics',
 'memory':'bounded BxCx5xN scalar scores; future tensors candidate chunks <=128 only; never population QxNxHxC',
 'original_generation_runtime':'UNKNOWN; current numerical reproduction recorded separately',
 'solar_resume':False, 'training':False, 'stage2':False, 'full_candidate_reduction':False,
}
