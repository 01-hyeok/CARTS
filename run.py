import argparse
import hashlib
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_stage1_relation import Exp_Stage1_Relation
from exp.exp_stage2_relation import Exp_Stage2_Relation
from utils.print_args import print_args
import random
import numpy as np


SETTING_COMPONENT_MAX_BYTES = 200


def _shorten_path_component(value, max_bytes=SETTING_COMPONENT_MAX_BYTES):
    """Keep an experiment name within a safe single-component path length."""
    encoded = value.encode('utf-8')
    if len(encoded) <= max_bytes:
        return value

    digest = hashlib.sha256(encoded).hexdigest()[:12]
    prefix_max_bytes = max_bytes - len(digest) - 1
    prefix = encoded[:prefix_max_bytes].decode('utf-8', errors='ignore')
    prefix = prefix.rstrip(' ._-')
    return '{}_{}'.format(prefix, digest)


def build_experiment_setting(args, iteration):
    setting_task_name = args.task_name.replace('_relation', '')
    full_setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        setting_task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        iteration,
    )
    setting = _shorten_path_component(full_setting)
    if setting != full_setting:
        print('[setting] experiment name exceeded {} bytes and was shortened to: {}'.format(
            SETTING_COMPONENT_MAX_BYTES, setting
        ))
    return setting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, stage1_relation, stage2_relation]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='temp', help='model id')
    parser.add_argument('--model', type=str, default='RAFT',
                        help='model name, options: [RAFT]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=10, help='TimesBlock top-k or Stage-2 retrieval top-k')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument(
        '--output_attention', action='store_true',
        help='whether to output attention in ecoder'
    )
    parser.add_argument(
        '--n_period', type=int, default=3,
        help='Number of Periods'
    )
    parser.add_argument(
        '--topm', type=int, default=20,
        help='Number of Retrievals'
    )
    parser.add_argument('--patch_len', type=int, default=16, help='Stage-1 relation patch length')
    parser.add_argument('--stride', type=int, default=16, help='Stage-1 relation patch stride')
    parser.add_argument('--relation_encoder_type', type=str, default='transformer',
                        choices=['transformer', 'mlp', 'tcn'],
                        help='Relation encoder backbone for Stage-1/Stage-2')
    parser.add_argument('--relation_pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'last'],
                        help=('Relation pooling: cls/mean for transformer, '
                              'last/mean for tcn (last reads the causal end state)'))
    parser.add_argument('--relation_tcn_layers', type=int, default=4,
                        help='TCN relation encoder residual blocks; dilation doubles per block')
    parser.add_argument('--relation_tcn_kernel_size', type=int, default=3,
                        help='TCN relation encoder causal convolution kernel size')
    parser.add_argument('--relation_tcn_channels', type=int, default=0,
                        help='TCN relation encoder hidden width; 0 uses d_model')
    parser.add_argument('--relation_tcn_dropout', type=float, default=-1.0,
                        help='TCN relation encoder dropout; negative uses --dropout')
    parser.add_argument('--relation_self_fill', type=str, default='linear',
                        choices=['zero', 'repeat', 'linear'],
                        help=(
                            'MLP relation input mode: zero/repeat pad a length-L relation to 2L; '
                            'linear keeps self at L and projects cross [target,source] '
                            'from 2L to L before a shared L->d_ff MLP'
                        ))
    parser.add_argument('--tau_student', type=float, default=0.1, help='Stage-1 student softmax temperature')
    parser.add_argument('--rank_loss_weight', type=float, default=0.0,
                        help=('Weight on the boundary hard-pair ranking loss. At 0 the mining is '
                              'skipped and Stage-1 is the plain cross-entropy baseline. The two '
                              'terms disagree by design on a candidate that beats a selected one '
                              'without entering the global Oracle Top-K, so this sets which wins'))
    parser.add_argument('--rank_margin', type=float, default=0.01,
                        help=('Score margin the better candidate must clear. Cosine scores occupy a '
                              'far narrower band than [-1, 1], so measure the real rank-10 to '
                              'rank-100 gap before setting this; too large and the hinge is open on '
                              'every pair, which rank_loss_active_fraction will show'))
    parser.add_argument('--rank_gap_threshold', type=float, default=0.0,
                        help='Minimum future-MSE gap for a pair to count, so near-ties are not ranked')
    parser.add_argument('--rank_pairs_per_query', type=int, default=32,
                        help='Hardest pairs kept per query, ordered by future-MSE gap')
    parser.add_argument('--rank_pool_end', type=int, default=100,
                        help=('End of the mining range (ranks top_k+1..this). A mining range only: '
                              'retrieval still scores the full memory'))
    parser.add_argument('--stage1_global_anchor_weight', type=float, default=0.0,
                        help=('Weight on KL(frozen-cosine ranking || current ranking) over the '
                              'full memory. The ranking loss supervises a hundred candidates '
                              'but the scorer weights move all of them, so this holds the '
                              'thousands it never looks at to the ranking the encoder already '
                              'produced. 0 reproduces the rank-only scorer exactly'))
    parser.add_argument('--stage1_global_anchor_tau', type=float, default=-1.0,
                        help='Temperature for the anchor; below 0 reuses --tau_student')
    parser.add_argument('--stage1_imitation_target', type=str, default='individual',
                        choices=['individual', 'set'],
                        help=('Which oracle the scorer is taught to reproduce. The two differ '
                              'only in how the K members were chosen, so the pair isolates '
                              'whether a per-candidate score can express a set picked for how '
                              'its members combine'))
    parser.add_argument('--stage1_imitation_pool', type=int, default=100,
                        help='Shared candidate pool the imitation ranks within')
    parser.add_argument('--stage1_freeze_encoder', type=int, default=0,
                        help=('Hold the Stage-1 encoder fixed and train only the retrieval '
                              'metric. Rank-only fine-tuning collapses the representation '
                              'within ten steps, so this separates whether the ranking '
                              'supervision is unusable from whether updating the encoder with '
                              'it was'))
    parser.add_argument('--rank_mining_mode', type=str, default='pair',
                        choices=['pair', 'candidate', 'persistent'],
                        help=('pair takes the largest future-MSE gaps, which lets one strong '
                              'candidate consume the budget through near-duplicate pairs. '
                              'candidate takes one pair per distinct candidate that beats a '
                              'selected one. persistent fixes that set once and never '
                              're-mines: dynamic mining releases a pair the moment its '
                              'positive enters the Top-K, which is exactly when the gap '
                              'has only just crossed zero and no margin exists yet'))
    parser.add_argument('--rank_gap_weighted', type=int, default=1,
                        help='Weight each pair by its future-MSE gap rather than treating all alike')
    parser.add_argument('--stage1_expected_mse_lambda', type=float, default=1.0,
                        help=('Additive weight on the expected-future-MSE term for '
                              'wce_expected_mse. Separate from --expected_mse_weight, which '
                              'kl_expected_mse mixes convexly and so caps at 1'))
    parser.add_argument('--stage1_wce_weight', type=float, default=1.0,
                        help='scale on the WCE term under wce_soft_set_mse; 0.0 gives '
                             'a pure soft_set_mse arm without reimplementing the loss')
    parser.add_argument('--stage1_set_mse_weight', type=float, default=0.0,
                        help=('lambda on the set-level term. The cross-entropy sits near 7 and the '
                              'normalised set MSE near 0.2, so parity needs roughly 30-40, not 0.5'))
    parser.add_argument('--stage1_set_tau', type=float, default=0.015,
                        help=('Temperature of the full-memory softmax the set loss aggregates over. '
                              'It decides how many candidates form one set: on ETTh1/336 this gives '
                              'effective support ~32 with ~69%% of the mass on the Top-10'))
    parser.add_argument('--stage1_set_mse_normalization', type=str, default='mean',
                        choices=['none', 'mean', 'median'],
                        help='Per-query scale the set MSE is divided by, as the expected-MSE loss does')
    parser.add_argument('--stage1_set_support_k', type=int, default=0,
                        help=('Target effective support for the one-sided entropy hinge; 0 disables '
                              'it. Only a support wider than the target is penalised, so it cannot '
                              'undo the concentration the cross-entropy is building'))
    parser.add_argument('--stage1_set_support_weight', type=float, default=0.0,
                        help='Weight on the support hinge; 0 disables it')
    parser.add_argument('--tau_teacher', type=float, default=0.1, help='Stage-1 teacher softmax temperature')
    parser.add_argument('--stage1_key_chunk_size', type=int, default=1024,
                        help='Stage-1 key encoder chunk size for memory-safe full candidate training')
    parser.add_argument('--stage1_overfit_queries', type=int, default=0,
                        help='Diagnostic mode: fixed number of train queries; 0 disables tiny-set overfit')
    parser.add_argument('--stage1_overfit_candidates', type=int, default=0,
                        help='Diagnostic mode: fixed candidate count shared by all tiny-set queries')
    parser.add_argument('--stage1_overfit_steps', type=int, default=0,
                        help='Diagnostic mode: repeated optimizer steps per Stage-1 epoch')
    parser.add_argument('--stage1_overfit_oracle_per_query', type=int, default=20,
                        help='Oracle-ranked candidates contributed by each tiny-set query')
    parser.add_argument('--stage1_overfit_key_refresh', type=str, default='epoch',
                        choices=['epoch', 'step'],
                        help='Diagnostic mode: rebuild the relation key bank per epoch or optimizer step')
    parser.add_argument('--stage1_overfit_self_only', type=int, default=0,
                        help='Diagnostic mode: restrict every target to its self-relation branch')
    parser.add_argument('--stage1_overfit_differentiable_keys', type=int, default=0,
                        help=('Diagnostic mode: re-encode every tiny-set candidate with the current '
                              'encoder inside the graph instead of reading the key bank, so query '
                              'and key embeddings both carry gradient'))
    parser.add_argument('--stage1_overfit_holdout_val', type=int, default=0,
                        help=('Tiny-overfit only: validate on held-out val queries against the '
                              'same tiny candidate set instead of reusing the training queries. '
                              'Off by default so existing memorization runs are unchanged; on, it '
                              'separates "memorised these queries" from "transfers to new ones"'))
    parser.add_argument('--stage1_overfit_log_every', type=int, default=0,
                        help='Diagnostic mode: log tiny-set retrieval metrics every N optimizer steps; 0 disables')
    parser.add_argument('--stage1_overfit_summary_path', type=str, default='',
                        help='Diagnostic mode: write the final tiny-set Recall@1/5/10 summary to this JSON path')
    parser.add_argument('--stage1_candidate_subset_mode', type=str, default='none',
                        choices=['none', 'selected_detached', 'selected_reencode'],
                        help=('Training-only Stage-1 candidate subset. none keeps the full-bank KL. '
                              'selected_detached mines Bank Top-M (Oracle Top-K injected) but scores '
                              'them with detached bank embeddings. selected_reencode re-encodes the '
                              'same candidates with the current encoder so candidate-side gradient flows. '
                              'Validation and test always score the full bank without Oracle injection'))
    parser.add_argument('--stage1_candidate_mine_top_m', type=int, default=100,
                        help='Candidates mined per query from the memory bank by cosine similarity')
    parser.add_argument('--stage1_candidate_oracle_inject_k', type=int, default=-1,
                        help='Global Oracle Top-K guaranteed inside the mined set; <=0 reuses --top_k')
    parser.add_argument('--stage1_checkpoint_metric', type=str, default='loss',
                        choices=['loss', 'recall10', 'retrieved_mse10', 'hard_aggregate_mse10', 'retrieval_regret10',
                                 'utility_gap_recovery', 'utility_ndcg', 'retrieved_utility'],
                        help=('Stage-1 best-checkpoint criterion on the validation split. '
                              'retrieved_mse10 minimizes the future-MSE of the model own Top-10, '
                              'which is what Stage-2 consumes; recall10 maximizes Oracle Recall@10 '
                              'and retrieval_regret10 minimizes Retrieval Regret@10, both scored '
                              'against the Future-MSE Oracle; the utility_* options select on '
                              'measured downstream forecast gain; all fall back to loss when absent'))
    parser.add_argument('--stage1_teacher_cache', type=str, default='',
                        help=('Directory of precomputed teacher tensors (train/val/test.pt) from '
                              'scripts/precompute_utility_teacher.py. Supplying it pins Stage-1 '
                              'training to that fixed candidate pool'))
    parser.add_argument('--stage1_teacher_target', type=str, default='future',
                        choices=['future', 'residual', 'utility'],
                        help=('Which measured target supervises retrieval. future keeps the '
                              'existing Future-MSE teacher; utility uses the actual Stage-2 '
                              'forecast gain of each candidate'))
    parser.add_argument('--stage1_teacher_loss', type=str, default='kl',
                        choices=['kl', 'expected_utility'],
                        help=('kl matches the teacher distribution; expected_utility directly '
                              'maximizes the gain the student distribution would collect'))
    parser.add_argument('--stage1_teacher_normalize', type=str, default='per_query_scale',
                        choices=['none', 'per_query_scale'],
                        help=('Divide teacher scores by their per-query scale so one tau means '
                              'the same sharpness across targets'))
    parser.add_argument('--stage1_teacher_tau', type=float, default=0.05,
                        help='Temperature of the external teacher distribution')
    parser.add_argument('--stage1_retrieval_metric', type=str, default='cosine',
                        choices=['cosine', 'mahalanobis', 'asymmetric', 'bilinear'],
                        help=('Learnable but still indexable retrieval score, so training can use '
                              'the same candidate support as evaluation. Nested by expressiveness: '
                              'cosine (W = I) < mahalanobis (one shared L, W = L^T L symmetric PSD) '
                              '< asymmetric (separate query and key spaces, W free). bilinear spans '
                              'the same functions as asymmetric and is kept only for earlier runs'))
    parser.add_argument('--stage1_metric_scaled_dot', type=int, default=1,
                        help='Divide the dot product by sqrt(D) so a cosine-tuned tau still applies')
    parser.add_argument('--stage1_metric_layer_norm', type=int, default=1,
                        help='LayerNorm on the projected embeddings')
    parser.add_argument('--stage1_metric_output', type=str, default='dot',
                        choices=['dot', 'cosine'],
                        help=('cosine renormalises after projecting, so every kind scores in '
                              '[-1, 1], one temperature means the same sharpness for all of them, '
                              'and identity init reproduces the incumbent exactly -- use it when '
                              'comparing kinds. dot leaves the projections unnormalised, which '
                              'ranks differently from cosine at step 0 and lets the score scale '
                              'drift per kind'))
    parser.add_argument('--stage1_full_memory_gradient_mode', type=str, default='bank',
                        choices=['bank', 'selected_reencode', 'full_online'],
                        help=('How the candidate branch receives gradient while the softmax '
                              'denominator stays the whole memory. bank trains only the query '
                              'side; selected_reencode re-encodes Oracle + hard + random '
                              'negatives and scatters their scores back into the full logits'))
    parser.add_argument('--stage1_full_memory_hard_negatives', type=int, default=100,
                        help='Model-top candidates outside the Oracle set that get gradient')
    parser.add_argument('--stage1_full_memory_random_negatives', type=int, default=128,
                        help='Uniformly sampled candidates that get gradient')
    parser.add_argument('--stage1_candidate_random_negatives', type=int, default=0,
                        help=('Uniformly sampled candidates appended to the mined training set. '
                              'A fixed score like cosine extrapolates to unseen pairs by '
                              'construction; a learned scorer does not, and mining alone leaves '
                              'it unconstrained on the rest of the bank that evaluation ranks'))
    parser.add_argument('--stage1_mining_score', type=str, default='self',
                        choices=['self', 'reference'],
                        help=('Which score picks the training candidates. self lets each arm '
                              'mine with its own score, which confounds "better score function" '
                              'with "different candidates seen". reference mines with a frozen '
                              'checkpoint (--stage1_pool_reference_ckpt) so every arm trains on '
                              'the same candidate ids'))
    parser.add_argument('--stage1_retrieval_score', type=str, default='cosine',
                        choices=['cosine', 'pairwise_mlp'],
                        help=('Retrieval score function. cosine is the incumbent fixed dot '
                              'product; pairwise_mlp learns a score per (query, candidate) '
                              'pair and requires --stage1_candidate_subset_mode selected_reencode '
                              'so the candidate side stays in the graph'))
    parser.add_argument('--stage1_pairwise_feature', type=str, default='pair4',
                        choices=['pair2', 'pair4'],
                        help=('pair2 feeds [z_q, z_k]; pair4 adds the difference and its '
                              'magnitude, which the scorer would otherwise have to learn'))
    parser.add_argument('--stage1_pairwise_hidden', type=int, default=256)
    parser.add_argument('--stage1_pairwise_hidden2', type=int, default=128)
    parser.add_argument('--stage1_pairwise_dropout', type=float, default=0.1)
    parser.add_argument('--stage1_residual_teacher', type=int, default=0,
                        help=('Supervise retrieval with residual similarity computed inline '
                              'from cached base forecasts, which scales to the full bank'))
    parser.add_argument('--stage1_residual_teacher_cache', type=str, default='',
                        help='Residual cache from scripts/precompute_residual_teacher.py')
    parser.add_argument('--stage1_pool_size', type=int, default=0,
                        help=('Restrict the loss to the frozen reference encoder\'s Top-M '
                              'candidates; 0 keeps the full memory bank'))
    parser.add_argument('--stage1_pool_reference_ckpt', type=str, default='',
                        help=('Stage-1 checkpoint whose frozen scores define the candidate '
                              'pool, so every teacher arm ranks the same candidates'))
    parser.add_argument('--stage1_null_mode', type=str, default='off',
                        choices=['off', 'fixed', 'query'],
                        help=('Explicit no-retrieval action. fixed pins its score at zero; query '
                              'learns it from the query embedding so abstention can be decided '
                              'per query'))
    parser.add_argument('--stage1_direct_eval', type=int, default=0,
                        help='Evaluate encoder-free Diff1 Direct retrieval with the Stage-1 metrics')
    parser.add_argument('--candidate_mask', type=str, default='raft',
                        choices=['raft', 'strict_causal', 'overlap_only', 'none'],
                        help='Stage-1/Stage-2 memory candidate mask')
    parser.add_argument('--source_mode', type=str, default='auto', choices=['auto', 'all', 'topk_corr'],
                        help='Source selection: auto/topk_corr use self plus absolute-Pearson Top-N; all uses every channel')
    parser.add_argument('--relation_top_n', type=int, default=3,
                        help='Total source channels per target including self; remaining sources use absolute Pearson')
    parser.add_argument('--relation_graph_threshold', type=int, default=21,
                        help='Deprecated compatibility option; auto now always uses the sparse Pearson graph')
    parser.add_argument('--relation_graph_path', type=str, default='',
                        help='Shared Stage1/Stage2 Pearson relation graph JSON path')
    parser.add_argument('--relation_target_chunk_size', type=int, default=0,
                        help='Stage1 target channels trained per batch; <=0 uses all targets')
    parser.add_argument('--target_mode', type=str, default='all', choices=['all', 'single'],
                        help='Stage-1 target channel mode')
    parser.add_argument('--target_channel', type=int, default=None, help='Stage-1 single target channel')
    parser.add_argument('--teacher_mse_space', type=str, default='normalized', choices=['normalized', 'raw'],
                        help='Space used for teacher future MSE')
    parser.add_argument('--stage1_teacher_mode', type=str, default='mse',
                        choices=['mse', 'pearson', 'ema_target', 'ema_input'],
                        help=('Stage-1 teacher source. ema_target preserves the legacy future-EMA '
                              'teacher; ema_input uses an EMA copy on the same past input as the student'))
    parser.add_argument('--relation_teacher_type', type=str, default=None,
                        choices=['future_mse', 'ema'],
                        help=('Experiment-2 teacher alias: future_mse maps to stage1_teacher_mode=mse; '
                              'ema maps to stage1_teacher_mode=ema_input'))
    parser.add_argument('--relation_input_space', type=str, default='delta_last',
                        choices=['absolute', 'delta_last', 'diff1', 'delta_last_diff1'],
                        help=('Relation encoder input space: raw normalized values, values minus each '
                              'role last value, first-order differences of length L-1, or '
                              'delta_last and diff1 stacked as two encoder input channels '
                              '(both cropped to the trailing L-1 steps)'))
    parser.add_argument('--relation_teacher_space', type=str, default='delta_last',
                        choices=['absolute', 'delta_last'],
                        help='Stage-1 MSE teacher space for future matching')
    parser.add_argument('--relation_value_space', type=str, default='delta_last',
                        choices=['absolute', 'delta_last'],
                        help='Stage-2 retrieved value space; delta_last retrieves future deltas and restores query last value')
    parser.add_argument('--stage1_ema_momentum_base', type=float, default=0.99,
                        help='Initial Stage-1 EMA teacher momentum for cosine schedule')
    parser.add_argument('--stage1_ema_momentum_final', type=float, default=0.9995,
                        help='Final Stage-1 EMA teacher momentum for cosine schedule')
    parser.add_argument('--stage1_probe_vis', type=int, default=1,
                        help='Write a fixed validation Stage-1 teacher/student distribution probe each epoch')
    parser.add_argument('--stage1_probe_top_n', type=int, default=50,
                        help='Number of teacher-ranked candidates to show in the Stage-1 probe plot')
    parser.add_argument('--stage1_probe_query', type=int, default=0,
                        help='Valid query index inside the fixed validation probe batch')
    parser.add_argument('--stage1_probe_target_channel', type=int, default=0,
                        help='Target channel index for the fixed validation Stage-1 probe')
    parser.add_argument('--stage1_probe_source_channel', type=int, default=0,
                        help='Source channel index for the fixed validation Stage-1 probe')
    parser.add_argument('--stage1_probe_dir', type=str, default='./stage1_vis',
                        help='Directory for fixed validation Stage-1 distribution probe plots')
    parser.add_argument('--stage1_collapse_metrics', type=int, default=1,
                        help='Compute relation-bank representation-collapse metrics each bank refresh')
    parser.add_argument('--stage1_collapse_sample_size', type=int, default=256,
                        help='Candidate embeddings sampled per relation for collapse diagnostics')
    parser.add_argument('--stage1_collapse_dead_std_threshold', type=float, default=1e-3,
                        help='Per-dimension standard-deviation threshold used to count dead dimensions')
    parser.add_argument('--stage1_variance_weight', type=float, default=0.0,
                        help='Weight for relation-wise VICReg variance loss on online query embeddings')
    parser.add_argument('--stage1_covariance_weight', type=float, default=0.0,
                        help='Weight for relation-wise VICReg covariance loss on online query embeddings')
    parser.add_argument('--stage1_variance_target', type=float, default=1.0,
                        help='Minimum pre-L2-normalization embedding standard deviation per dimension')
    parser.add_argument('--stage1_use_rank_loss', type=int, default=0,
                        help='Add future-aware top-k pairwise ranking loss to Stage-1')
    parser.add_argument('--stage1_loss_mode', type=str, default='kl',
                        choices=['kl', 'kl_infonce', 'kl_rank', 'rnc', 'kl_expected_mse',
                                 'topk_coverage', 'weighted_topk_ce', 'wce_soft_set_mse',
                                 'expected_mse', 'wce_expected_mse', 'rank_only',
                                 'oracle_imitation'],
                        help='Stage-1 objective; legacy stage1_use_rank_loss=1 maps kl to kl_rank')
    parser.add_argument('--stage1_infonce_weight', type=float, default=0.5,
                        help='InfoNCE mixture weight for kl_infonce; KL uses one minus this value')
    parser.add_argument('--stage1_infonce_top_k', type=int, default=-1,
                        help='Future-MSE positive count for multi-positive InfoNCE; <=0 reuses --top_k')
    parser.add_argument('--stage1_infonce_positive_source', type=str, default='target_mse',
                        choices=['target_mse', 'ema_cosine'],
                        help='Positive Top-K source for multi-positive InfoNCE: target future MSE or branch-wise EMA future cosine')
    parser.add_argument('--stage1_coverage_top_k', type=int, default=-1,
                        help='Oracle positive count for Top-K Coverage Loss; <=0 reuses --top_k')
    parser.add_argument('--stage1_rank_weight', type=float, default=0.1,
                        help='Weight for Stage-1 future-aware ranking loss')
    parser.add_argument('--stage1_rank_margin', type=float, default=0.1,
                        help='Margin for Stage-1 top-k pairwise ranking loss')
    parser.add_argument('--stage1_rank_min_mse_gap', type=float, default=0.0,
                        help='Minimum future MSE gap required for a valid Stage-1 ranking pair')
    parser.add_argument('--stage1_rank_top_k', type=int, default=-1,
                        help='Stage-1 ranking top-k; <=0 reuses --top_k')
    parser.add_argument('--rnc_temperature', type=float, default=0.2,
                        help='Temperature for query-conditioned Stage-1 RnC loss')
    parser.add_argument('--rnc_tie_epsilon', type=float, default=0.0,
                        help='Future-MSE tolerance used to form RnC tie groups')
    parser.add_argument('--rnc_quality_source', type=str, default='future_mse',
                        choices=['future_mse', 'ema_cosine'],
                        help='RnC ordering source: actual future MSE or EMA future cosine')
    parser.add_argument('--expected_mse_weight', type=float, default=0.1,
                        help='Mixture coefficient: (1-w)*KL + w*expected future MSE')
    parser.add_argument('--expected_mse_normalization', type=str, default='mean',
                        choices=['none', 'mean', 'median'],
                        help='Per-query future-MSE normalization for expected loss')
    parser.add_argument('--build_memory_index', action='store_true',
                        help='Optional Stage-2 TODO hook for memory embedding cache')
    parser.add_argument('--stage2_e2e', type=int, default=0,
                        help=('End-to-end retrieval: re-encode the selected Top-K candidates '
                              'with the live encoder so the forecast loss reaches the Stage-1 '
                              'encoder through the Top-K weights. Selection and the retrieval '
                              'universe are unchanged -- still the full bank'))
    parser.add_argument('--stage2_e2e_full_online', type=int, default=0,
                        help=('Joint training only: re-encode the whole memory each step for '
                              'candidate selection instead of reading the epoch key bank. The '
                              'bank goes stale as the encoder moves, so selection would use '
                              'embeddings the encoder has left behind while only the chosen '
                              'Top-K are re-encoded live. Serving still uses an index'))
    parser.add_argument('--stage2_rank_loss', type=str, default='none',
                        choices=['none', 'ranknet', 'weighted_ranknet', 'margin', 'adaptive_margin'],
                        help=('Auxiliary pairwise loss on retrieval scores. margin and '
                              'adaptive_margin target absolute score separation, which is what '
                              'survives the division by tau_topk that flattens Stage-2 weights'))
    parser.add_argument('--stage2_rank_weight', type=float, default=0.0,
                        help='alpha in L = L_forecast + alpha * L_rank')
    parser.add_argument('--stage2_rank_margin', type=float, default=0.05,
                        help='Required student score gap for the margin losses')
    parser.add_argument('--stage2_rank_top_p', type=int, default=10,
                        help='Teacher-best candidates used as ranking positives')
    parser.add_argument('--stage2_rank_hard_negatives', type=int, default=30,
                        help='Candidates the student ranks highly but the teacher does not')
    parser.add_argument('--stage2_rank_random_negatives', type=int, default=10,
                        help='Random valid candidates, as a background for the pair geometry')
    parser.add_argument('--stage2_rank_topk_gamma', type=float, default=-1.0,
                        help=('Share of the ranking loss taken from pairs inside the Top-K, '
                              'normalized separately from the rest. Negative keeps the v1 '
                              'behaviour of one mean over all pairs, where the Top-K carried '
                              'under 2 percent'))
    parser.add_argument('--stage2_rank_margin_mode', type=str, default='absolute',
                        choices=['absolute', 'topk_relative'],
                        help=('topk_relative asks for the margin times the current Top-K score '
                              'spread rather than an absolute number, so the demand matches the '
                              'scale of the gaps it is meant to widen'))
    parser.add_argument('--stage2_rank_margin_cap', type=float, default=0.2,
                        help='Upper bound on a relative margin, so the demand cannot run away')
    parser.add_argument('--stage2_rank_sigma_mode', type=str, default='fixed',
                        choices=['fixed', 'topk_relative'],
                        help=('topk_relative divides the RankNet logit by the Top-K spread; '
                              'without it sigmoid sits at 0.50 for gaps 24x apart'))
    parser.add_argument('--stage2_residual_cache', type=str, default='',
                        help=('Residual cache from scripts/precompute_residual_teacher.py, '
                              'used as the ranking teacher'))
    parser.add_argument('--use_ema_teacher', type=int, default=1,
                        help=('0 disables the EMA teacher encoder and its updates. The '
                              'end-to-end arms run without it; the flag keeps the original '
                              'baseline reproducible'))
    parser.add_argument('--stage1_ckpt_path', type=str, default='',
                        help='Stage-1 relation checkpoint used to initialize Stage-2 encoder')
    parser.add_argument('--stage1_encoder_init', type=str, default='checkpoint',
                        choices=['checkpoint', 'random', 'none'],
                        help='Initialize Stage-2 relation encoder from Stage-1, random weights, or no Stage-1 encoder')
    parser.add_argument('--stage2_retrieval_encoder', type=str, default='online',
                        choices=['online', 'ema'],
                        help='Stage-1 retrieval backbone to load for Stage-2; encoder and shared projection are loaded as a matched pair')
    parser.add_argument('--stage2_retrieval_backbone', type=str, default='stage1',
                        choices=['stage1', 'identity', 'pearson', 'chronos'],
                        help=(
                            'Retrieval representation used for Stage-2 cosine Top-K; '
                            'identity directly uses the raw relation history without an encoder, '
                            'pearson additionally mean-centers it so the score is the RAFT-style '
                            'raw Pearson correlation'
                        ))
    parser.add_argument('--chronos_model_id', type=str, default='amazon/chronos-t5-base',
                        help='Hugging Face checkpoint used by the Chronos retrieval backbone')
    parser.add_argument('--chronos_embedding_dim', type=int, default=768,
                        help='Encoder hidden size expected from the Chronos checkpoint')
    parser.add_argument('--chronos_pooling', type=str, default='mean',
                        choices=['mean', 'eos'],
                        help=(
                            'How a Chronos window becomes one vector. mean drops the EOS token '
                            'and averages the value tokens (this repo default). eos keeps only '
                            'the EOS summary token, which is what TS-RAG retrieves with '
                            '(embeddings[:, -1, :])'
                        ))
    parser.add_argument('--chronos_context_length', type=int, default=512,
                        help='Maximum history length passed to the frozen Chronos retrieval encoder')
    parser.add_argument('--chronos_dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Weight dtype used for frozen Chronos encoding')
    parser.add_argument('--chronos_random_init', type=int, default=0,
                        help='Reinitialize the Chronos T5 weights before freezing (random-encoder control)')
    parser.add_argument('--chronos_projection_dim', type=int, default=0,
                        help=(
                            'Project the concatenated [target || source] Chronos embeddings to this '
                            'dimension before the cosine Top-K; 0 keeps the raw 2*embedding_dim space'
                        ))
    parser.add_argument('--chronos_projection_mode', type=str, default='cross_only',
                        choices=['cross_only', 'uniform'],
                        help=(
                            'cross_only mirrors shared_cross_projection: self keeps the raw pooled '
                            'embedding and only cross branches are projected 2D -> D. uniform projects '
                            'both branches from 2D to chronos_projection_dim'
                        ))
    parser.add_argument('--chronos_projection_trainable', type=int, default=0,
                        help=(
                            'Train the Chronos projection with the Stage-2 loss while the encoder '
                            'stays frozen; requires --refresh_memory_every_epoch 1'
                        ))
    parser.add_argument('--chronos_finetune', type=int, default=0,
                        help=(
                            'Train the Chronos retrieval encoder with the Stage-2 loss instead of '
                            'freezing it; requires --refresh_memory_every_epoch 1 so the memory keys '
                            'are re-encoded as the query encoder moves'
                        ))
    parser.add_argument('--chronos_lr_decay', type=int, default=0,
                        help=(
                            'Let the Chronos encoder follow the Stage-2 lr schedule. Off by '
                            'default: --lradj type1 halves every epoch, which drives a 1e-5 '
                            'encoder step to ~1e-8 by epoch 10 and stops fine-tuning entirely'
                        ))
    parser.add_argument('--chronos_grad_checkpointing', type=int, default=1,
                        help=(
                            'Recompute Chronos encoder activations during backward instead of '
                            'storing them; only applies with --chronos_finetune 1. Without it a '
                            'seq_len 336 batch of 32 x 7 channels needs more than 79 GiB'
                        ))
    parser.add_argument('--chronos_lr', type=float, default=-1.0,
                        help=(
                            'Learning rate for the Chronos encoder parameters when --chronos_finetune 1; '
                            'negative uses the Stage-2 learning_rate unchanged'
                        ))
    parser.add_argument('--freeze_stage1_encoder', type=int, default=1,
                        help='Freeze Stage-1 relation encoder during Stage-2')
    parser.add_argument('--refresh_memory_every_epoch', type=int, default=0,
                        help='Refresh Stage-2 memory key bank at each epoch; frozen precomputed retrieval defaults to 0')
    parser.add_argument('--memory_cache_mode', type=str, default='precompute',
                        choices=['precompute', 'on_the_fly'], help='Stage-2 memory cache mode')
    parser.add_argument('--memory_chunk_size', type=int, default=1024,
                        help='Stage-2 memory encoder chunk size')
    parser.add_argument('--tau_topk', type=float, default=0.1,
                        help='Stage-2 top-k attention softmax temperature')
    parser.add_argument('--retrieval_soft_all', type=int, default=0,
                        help=(
                            'Weight every valid candidate with softmax(scores/tau_topk) instead '
                            'of selecting a Top-K first. Top-K picks indices, which is not '
                            'differentiable, so the forecasting loss can only reweight candidates '
                            'that were already chosen; weighting the whole bank puts every score '
                            'on the gradient path and lets one end-to-end loss train retrieval. '
                            'tau_topk has to be far smaller here - the softmax runs over N '
                            'candidates, not k'
                        ))
    parser.add_argument('--retrieval_similarity', type=str, default='cosine',
                        choices=['cosine', 'l2'],
                        help=(
                            'Stage-2 candidate score. cosine L2-normalises the query and key '
                            'and takes a dot product. l2 skips that normalisation and scores '
                            'with the negative mean squared distance, which is what TS-RAG '
                            'retrieves with (faiss IndexFlatL2). On normalised vectors the two '
                            'give the same ranking, so l2 only differs because the norm - the '
                            'amplitude the encoder put in the embedding - is kept'
                        ))
    parser.add_argument('--fusion_mode', type=str, default='raft_concat',
                        choices=['residual', 'mixture', 'raft_concat'], help='Stage-2 base/retrieval fusion mode')
    parser.add_argument('--stage2_relation_fusion', type=str, default='gate',
                        choices=['concat_linear', 'gate'],
                        help='Fuse relation retrieved futures with a shared score MLP and softmax gate, or concat+Linear')
    parser.add_argument('--relation_mixer_input', type=str, default='retrieved',
                        choices=['retrieved', 'retrieved_plus_query'], help='Stage-2 relation mixer input')
    parser.add_argument('--relation_mixer_hidden', type=int, default=128,
                        help='Stage-2 relation mixer hidden size')
    parser.add_argument('--gate_mode', type=str, default='scalar',
                        choices=['scalar', 'horizon'], help='Stage-2 retrieval gate mode')
    parser.add_argument('--gate_hidden', type=int, default=128,
                        help='Stage-2 retrieval gate hidden size')
    parser.add_argument('--fixed_lambda', type=float, default=-1.0,
                        help='Use fixed Stage-2 retrieval gate lambda when >= 0; negative keeps trainable gate')
    parser.add_argument('--stage2_retrieval_off', type=int, default=0,
                        help=('Inference-time counterfactual: keep every trained weight but feed '
                              'the fusion a neutral retrieval signal. Pairs with a normal run on '
                              'the same checkpoint to measure what retrieval actually contributed, '
                              'which disable_retrieval cannot do -- that one builds a different '
                              'model and trains a base-only forecaster'))
    parser.add_argument('--disable_retrieval', type=int, default=0,
                        help='Disable Stage-2 memory retrieval and train/evaluate the base forecast head only')
    parser.add_argument('--base_head_mode', type=str, default='shared_target_linear',
                        choices=['per_channel_linear', 'shared_target_linear'],
                        help='Stage-2 base forecast head mode')
    parser.add_argument('--use_aux_base_loss', type=int, default=0,
                        help='Use auxiliary base forecast loss in Stage-2')
    parser.add_argument('--aux_base_weight', type=float, default=0.1,
                        help='Auxiliary base forecast loss weight')
    parser.add_argument('--use_aux_ret_loss', type=int, default=0,
                        help='Use auxiliary retrieval forecast loss in Stage-2')
    parser.add_argument('--aux_ret_weight', type=float, default=0.1,
                        help='Auxiliary retrieval forecast loss weight')
    parser.add_argument('--retrieval_kl_weight', type=float, default=0.0,
                        help=(
                            'End-to-end lambda on KL(future-MSE teacher || cosine student) over all '
                            'candidates. Zero trains retrieval through the forecasting loss only, '
                            'which cannot reorder candidates because Top-K is not differentiable. '
                            'Uses --tau_teacher / --tau_student, as Stage-1 does'
                        ))
    parser.add_argument('--retrieval_kl_teacher', type=str, default='ema',
                        choices=['ema', 'future_mse'],
                        help=(
                            'Teacher for the end-to-end retrieval KL. ema mirrors Stage-1: an EMA '
                            'copy of the encoder embeds candidate futures and the cosine over those '
                            'is the target, so end-to-end keeps the retrieval objective the 2-stage '
                            'pipeline defines. future_mse replaces it with raw future L2 distance'
                        ))
    parser.add_argument('--beta_entropy_reg', type=float, default=0.0,
                        help='Stage-2 relation beta entropy regularization weight')
    parser.add_argument('--oracle_intervention_arms', type=str, default='',
                        help='comma-separated selection arms for the Stage-2 oracle '
                             'intervention (R0,R1,R2-target,R2-relation,R3). Non-empty '
                             'runs the intervention instead of a normal test pass; '
                             'nothing is trained and only the selected candidates differ')
    parser.add_argument('--oracle_intervention_pool', type=int, default=100,
                        help='size of the cosine-induced fixed candidate support every '
                             'intervention arm selects inside')
    parser.add_argument('--oracle_intervention_out', type=str,
                        default='logs/oracle_intervention',
                        help='directory for oracle intervention CSVs and fingerprints')
    parser.add_argument('--stage2_ckpt_path', type=str, default='',
                        help='Stage-2 checkpoint to evaluate; required by the oracle '
                             'intervention so every arm shares one set of weights')
    parser.add_argument('--oracle_candidate_eval', type=int, default=0,
                        help='Evaluate branchwise target/source-concat future Oracle Top-K on the Stage-2 test split only')
    parser.add_argument('--stage2_oracle_train_mode', type=str, default='none',
                        choices=['none', 'candidate', 'relation', 'full'],
                        help='Use Oracle retrieval for Stage-2; full is encoder-free branchwise relation-future MSE Top-K plus MSE-softmax weighting')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=None, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_tensorboard', type=int, default=1,
                        help='Write Stage-1/Stage-2 train and validation curves to TensorBoard')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                        help='TensorBoard root directory')
    parser.add_argument('--metrics_csv_dir', type=str, default='./metrics',
                        help='Root directory for Stage-2 metric CSV summaries')
    parser.add_argument('--focus_channel', type=str, default='OT',
                        help='Focus channel name for Stage-2 OT-style metric summaries')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=0, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    args = parser.parse_args()
    if args.relation_teacher_type is not None:
        mapped_teacher_mode = {
            'future_mse': 'mse',
            'ema': 'ema_input',
        }[args.relation_teacher_type]
        if args.stage1_teacher_mode not in ('mse', mapped_teacher_mode):
            raise ValueError(
                '--relation_teacher_type conflicts with --stage1_teacher_mode: '
                f'{args.relation_teacher_type} vs {args.stage1_teacher_mode}'
            )
        args.stage1_teacher_mode = mapped_teacher_mode
    if args.relation_encoder_type == 'tcn' and args.relation_pooling not in ('last', 'mean'):
        raise ValueError(
            '--relation_encoder_type tcn requires --relation_pooling last or mean, '
            f'got {args.relation_pooling}'
        )
    if args.relation_encoder_type != 'tcn' and args.relation_pooling == 'last':
        raise ValueError(
            f'--relation_pooling last is only supported by the tcn relation encoder, '
            f'got --relation_encoder_type {args.relation_encoder_type}'
        )
    if bool(int(args.stage1_overfit_differentiable_keys)):
        if int(args.stage1_overfit_queries) <= 0:
            raise ValueError(
                '--stage1_overfit_differentiable_keys requires the tiny-set overfit mode '
                '(--stage1_overfit_queries > 0); re-encoding every candidate per step is '
                'only tractable on a small fixed candidate set'
            )
        if bool(int(args.stage1_direct_eval)):
            raise ValueError(
                '--stage1_overfit_differentiable_keys is incompatible with '
                '--stage1_direct_eval, which is encoder-free'
            )
    if args.oracle_intervention_arms:
        from utils.oracle_intervention import ALL_ARMS
        unknown = [a.strip() for a in args.oracle_intervention_arms.split(',')
                   if a.strip() and a.strip() not in ALL_ARMS]
        if unknown:
            raise ValueError(
                f'--oracle_intervention_arms has unknown arms {unknown}; '
                f'expected from {list(ALL_ARMS)}')
        if args.task_name != 'stage2_relation':
            raise ValueError(
                '--oracle_intervention_arms requires --task_name stage2_relation')
        if bool(int(args.is_training)):
            raise ValueError(
                '--oracle_intervention_arms is an evaluation-only intervention; '
                'use --is_training 0 so no arm can train')
        if not args.stage2_ckpt_path:
            raise ValueError(
                '--oracle_intervention_arms requires --stage2_ckpt_path so that '
                'every arm is evaluated under one identical Stage-2 checkpoint')
    if bool(int(args.stage1_direct_eval)):
        if args.task_name != 'stage1_relation':
            raise ValueError('--stage1_direct_eval requires --task_name stage1_relation')
        if bool(int(args.is_training)):
            raise ValueError('--stage1_direct_eval is evaluation-only; use --is_training 0')
    fix_seed = int(args.seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[seed] using seed={fix_seed}')

    if args.learning_rate is None:
        if args.task_name == 'stage1_relation':
            args.learning_rate = 1e-3
        elif args.task_name == 'stage2_relation':
            args.learning_rate = 1e-2
        else:
            args.learning_rate = 1e-4
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'stage1_relation':
        Exp = Exp_Stage1_Relation
    elif args.task_name == 'stage2_relation':
        if args.memory_cache_mode != 'precompute':
            raise NotImplementedError('Stage-2 currently implements memory_cache_mode=precompute')
        Exp = Exp_Stage2_Relation
    else:
        assert(0)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = build_experiment_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_experiment_setting(args, ii)

        exp = Exp(args)  # set experiments
        if args.oracle_intervention_arms:
            print('>>>>>>>oracle intervention : {}<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.load_stage2_checkpoint(args.stage2_ckpt_path)
            exp.oracle_intervention(setting)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
