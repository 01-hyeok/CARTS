import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_stage1_relation import Exp_Stage1_Relation
from exp.exp_stage2_relation import Exp_Stage2_Relation
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 0
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                        choices=['transformer', 'mlp'], help='Relation encoder backbone for Stage-1/Stage-2')
    parser.add_argument('--relation_pooling', type=str, default='cls',
                        choices=['cls', 'mean'], help='Transformer relation pooling for Stage-1/Stage-2')
    parser.add_argument('--relation_self_fill', type=str, default='zero',
                        choices=['zero', 'repeat'], help='MLP self relation second-slot fill mode')
    parser.add_argument('--tau_student', type=float, default=0.07, help='Stage-1 student softmax temperature')
    parser.add_argument('--tau_teacher', type=float, default=0.1, help='Stage-1 teacher softmax temperature')
    parser.add_argument('--stage1_key_chunk_size', type=int, default=1024,
                        help='Stage-1 key encoder chunk size for memory-safe full candidate training')
    parser.add_argument('--candidate_mask', type=str, default='raft',
                        choices=['raft', 'strict_causal', 'overlap_only', 'none'],
                        help='Stage-1/Stage-2 memory candidate mask')
    parser.add_argument('--source_mode', type=str, default='auto', choices=['auto', 'all', 'topk_corr'],
                        help='Source selection: auto enables absolute-Pearson Top-N when enc_in reaches threshold')
    parser.add_argument('--relation_top_n', type=int, default=3,
                        help='Total source channels per target including self; remaining sources use absolute Pearson')
    parser.add_argument('--relation_graph_threshold', type=int, default=21,
                        help='Auto-enable sparse Pearson relation graph when enc_in is at least this value')
    parser.add_argument('--relation_graph_path', type=str, default='',
                        help='Shared Stage1/Stage2 Pearson relation graph JSON path')
    parser.add_argument('--relation_target_chunk_size', type=int, default=0,
                        help='Stage1 target channels trained per batch; <=0 uses all targets')
    parser.add_argument('--target_mode', type=str, default='all', choices=['all', 'single'],
                        help='Stage-1 target channel mode')
    parser.add_argument('--target_channel', type=int, default=None, help='Stage-1 single target channel')
    parser.add_argument('--teacher_mse_space', type=str, default='normalized', choices=['normalized', 'raw'],
                        help='Space used for teacher future MSE')
    parser.add_argument('--stage1_teacher_mode', type=str, default='mse', choices=['mse', 'pearson', 'ema_target'],
                        help='Stage-1 teacher distribution source: future MSE, future Pearson similarity, or EMA target-future embedding similarity')
    parser.add_argument('--relation_input_space', type=str, default='delta_last',
                        choices=['absolute', 'delta_last'],
                        help='Relation encoder input space: raw normalized values or values minus each role last value')
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
    parser.add_argument('--stage1_use_rank_loss', type=int, default=0,
                        help='Add future-aware top-k pairwise ranking loss to Stage-1')
    parser.add_argument('--stage1_loss_mode', type=str, default='kl',
                        choices=['kl', 'kl_rank', 'rnc', 'kl_expected_mse'],
                        help='Stage-1 objective; legacy stage1_use_rank_loss=1 maps kl to kl_rank')
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
    parser.add_argument('--stage1_ckpt_path', type=str, default='',
                        help='Stage-1 relation checkpoint used to initialize Stage-2 encoder')
    parser.add_argument('--stage1_encoder_init', type=str, default='checkpoint',
                        choices=['checkpoint', 'random'],
                        help='Initialize Stage-2 relation encoder from Stage-1 checkpoint or keep random weights')
    parser.add_argument('--freeze_stage1_encoder', type=int, default=1,
                        help='Freeze Stage-1 relation encoder during Stage-2')
    parser.add_argument('--refresh_memory_every_epoch', type=int, default=1,
                        help='Refresh Stage-2 memory key bank at each epoch')
    parser.add_argument('--memory_cache_mode', type=str, default='precompute',
                        choices=['precompute', 'on_the_fly'], help='Stage-2 memory cache mode')
    parser.add_argument('--memory_chunk_size', type=int, default=1024,
                        help='Stage-2 memory encoder chunk size')
    parser.add_argument('--tau_topk', type=float, default=0.07,
                        help='Stage-2 top-k attention softmax temperature')
    parser.add_argument('--fusion_mode', type=str, default='mixture',
                        choices=['residual', 'mixture', 'raft_concat'], help='Stage-2 base/retrieval fusion mode')
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
    parser.add_argument('--beta_entropy_reg', type=float, default=0.0,
                        help='Stage-2 relation beta entropy regularization weight')
    parser.add_argument('--oracle_candidate_eval', type=int, default=0,
                        help='Evaluate ground-truth Top-K candidate and relation oracles on the Stage-2 test split only')

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
            setting_task_name = args.task_name.replace('_relation', '')
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.task_name != 'stage1_relation':
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting_task_name = args.task_name.replace('_relation', '')
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
