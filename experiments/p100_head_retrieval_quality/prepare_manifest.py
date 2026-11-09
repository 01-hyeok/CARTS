"""Build a static plan and validate existing inputs; never creates pools or models."""
import csv
import datetime
import json
import math
from pathlib import Path
import subprocess
import torch
from preparation import sha256_file, GPU_UUID

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HORIZONS = [96, 192, 336, 720]
DATASETS = [('ETTh1', 'ETTh1', 7), ('Weather', 'custom', 21), ('Solar', 'Solar', 137)]

def audit_data(path, dataset, channels):
    rows, sentinel = 0, 0
    with open(path, newline='') as f:
        reader = csv.reader(f)
        names = next(reader) if dataset != 'Solar' else [str(i) for i in range(channels)]
        if dataset != 'Solar' and (len(names) != channels + 1 or names[0] != 'date'):
            raise ValueError('Unexpected dataset header')
        for row in reader:
            values = row if dataset == 'Solar' else row[1:]
            if len(values) != channels:
                raise ValueError(f'Wrong channel count at row {rows}')
            numbers = [float(v) for v in values]
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError(f'Nonfinite input at row {rows}')
            sentinel += sum(v == -9999. for v in numbers)
            rows += 1
    if dataset == 'ETTh1':
        lengths = [8640, 2880, 2880]
    else:
        train, test = int(rows * .7), int(rows * .2)
        lengths = [train, rows - train - test, test]
    if sum(lengths) > rows:
        raise ValueError('Dataset shorter than defined splits')
    return dict(path=str(path), sha256=sha256_file(path), rows=rows,
                channels=channels, columns=names, split_rows=lengths,
                minus9999_count=sentinel, finite=True,
                timestamp_policy='not required; Solar uses row indices' if dataset=='Solar' else 'existing date column; raw cosine uses history only',
                validation_scope='static full-file shape/finite/hash audit; actual loader smoke not run')

def main():
    dataset_records, references = {}, {}
    for dataset, data_key, channels in DATASETS:
        for horizon in HORIZONS:
            directory = ROOT / f'checkpoints/soft_set_mse/stage1/{data_key}/seq{horizon}_pred{horizon}'
            paths = [p for p in directory.glob('*/checkpoint.pth') if f'{dataset}_{horizon}_S0_wce' in str(p)]
            if len(paths) != 1:
                raise ValueError(f'Ambiguous reference {dataset} {horizon}: {paths}')
            path = paths[0]
            saved = torch.load(path, map_location='cpu', weights_only=False)
            args = dict(saved['args'])
            assert args['data'] == data_key and args['enc_in'] == channels
            assert args['seq_len'] == args['pred_len'] == horizon
            assert args['features'] == 'M' and args['teacher_mse_space'] == 'normalized'
            assert args['augmentation_ratio'] == 0
            data_path = (ROOT / args['root_path'] / args['data_path']).resolve()
            assert data_path.is_file()
            if dataset not in dataset_records:
                dataset_records[dataset] = audit_data(data_path, dataset, channels)
            assert str(data_path) == dataset_records[dataset]['path']
            fields = ['data','features','enc_in','seq_len','pred_len','d_model','d_ff','dropout',
                      'teacher_mse_space','augmentation_ratio','num_workers','stage1_full_memory_gradient_mode']
            references[f'{dataset}_{horizon}'] = dict(path=str(path), sha256=sha256_file(path),
                saved_args={k:args[k] for k in fields}, reference_role='configuration only; no model weights loaded',
                common_initialization='fresh CPU initialization by training seed, copied to every model',
                runtime_initialization_status='NOT_RUN',
                expected_query_counts=dict(zip(['train','val','test'],[dataset_records[dataset]['split_rows'][0]-2*horizon+1, dataset_records[dataset]['split_rows'][1]-horizon+1, dataset_records[dataset]['split_rows'][2]-horizon+1])))
            del saved
    plan = dict(status='PREPARATION_ONLY', training_started=False, stage2_enabled=False,
        teacher=dict(status='PENDING_USER_DECISION', mode=None, decision_reference=None,
                     allowed_modes=['zscore_future_mse','raw_future_mse'], tau_t=.1,
                     note='No default mode. Prior canonical code uses zscore; attached raw formula conflicts.'),
        execution=dict(physical_gpu_index=1, gpu_uuid=GPU_UUID, max_concurrent_runs=1,
                       seed_order=[0,1,2], dataset_order=[x[0] for x in DATASETS],
                       horizon_order=HORIZONS, model_order=['V0','V1','V2','V5'],
                       failure_policy='stop queue; preserve partial artifacts; do not skip to another seed',
                       launch_implementation='NOT_IMPLEMENTED_PENDING_TEACHER_AND_RUNTIME_SMOKE'),
        training=dict(epochs=10, logical_batch_size=32, learning_rate=.001, optimizer='Adam',
                      early_stopping=False, save_every_epoch=True, student_temperature=.1,
                      shared_encoder=True, query_and_candidate_gradients=True,
                      encoder_type='mlp', self_fill='linear', feature_space='delta_last',
                      d_model=128, d_ff=256, dropout=.1,
                      head_initialization_seed='1000 + 10000 * training_seed + zero_based_head_index',
                      loader_seed='training_seed', common_encoder_seed='training_seed'),
        retrieval=dict(pool_size=100, k=10, preselection='delta_last_history_cosine',
                       candidate_pool_shared_across_models_and_seeds=True, mask='existing RAFT offline train-bank nonoverlap',
                       train_rule='candidate_end <= query_start OR candidate_start >= query_start+L+H',
                       evaluation_rule='train bank only; candidate_end <= query_start+L',
                       inference='mean head probabilities then unique TopK; uniform aligned-future mean'),
        evaluation=dict(checkpoint='minimum validation Mean-Mixture RetMSE@10',
            primary_k=10, auxiliary_k=[1,5,20], stage2=False,
            oracle='P100 Individual-MSE Oracle; same MSE-selected set for MAE',
            last_skill='1 - mean(model AggMSE)/mean(last MSE), same units; undefined if denominator <= 1e-12',
            random_draws=20, random_seed_namespace=990000, random_variation='MC draw variation, not training-seed variation',
            strata='fit cutoffs on train/validation; actual Last-repeat-error strata are nondeployable retrospective diagnostics',
            subsets='31 V5 subsets, per-head-count validation RetMSE selection; reuse frozen scores',
            uncertainty='paired seed differences and paired time-block diagnostics; windows/channels are not independent replicates'),
        prerequisites=['explicit teacher decision','all synthetic tests passed','all12 real loader/pool checks',
                       'common encoder/head initial hashes and per-epoch batch-order hashes',
                       'Solar720 memory and RNG-aware forward/gradient/update equivalence smoke',
                       'canonical seed0 reproduction where protocol matches'],
        data=dataset_records, references=references,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    runs=[]
    for seed in [0,1,2]:
        for dataset, _, channels in DATASETS:
            for h in HORIZONS:
                cell=f'{dataset}_{h}'
                for head_count in [0,1,2,5]:
                    run_id=f'{cell}_V{head_count}_seed{seed}'
                    runs.append(dict(ordinal=len(runs)+1,run_id=run_id,seed=seed,dataset=dataset,horizon=h,
                        seq_len=h,model=f'V{head_count}',num_query_heads=head_count,channels=channels,
                        teacher_mode=None,status='NOT_STARTED',blocked_by='TEACHER_UNRESOLVED_AND_RUNTIME_SMOKE_PENDING',
                        previous_run=runs[-1]['run_id'] if runs else None,
                        reference_ckpt=references[cell]['path'],reference_sha256=references[cell]['sha256'],
                        dataset_sha256=dataset_records[dataset]['sha256'],
                        n_train_queries=references[cell]['expected_query_counts']['train'],
                        n_val_queries=references[cell]['expected_query_counts']['val'],
                        n_test_queries=references[cell]['expected_query_counts']['test'],
                        optimizer_steps_per_epoch=math.ceil(references[cell]['expected_query_counts']['train']/32),
                        shared_pool_path=f'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01/pools/{cell}',
                        output_path=f'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01/runs/{cell}/seed{seed}/V{head_count}',
                        checkpoint_path=f'checkpoints/track_p100_head_retrieval_quality01/{cell}/seed{seed}/V{head_count}',
                        common_encoder_init_hash=None,head_init_hash=None,batch_order_hashes=None,
                        logical_batch_size=32,train_epochs=10,physical_gpu_index=1))
    assert len(runs)==144 and len({r['run_id'] for r in runs})==144
    assert all(sum(r['seed']==s for r in runs)==48 for s in [0,1,2])
    (HERE/'configs/protocol.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    (HERE/'configs/run_manifest.json').write_text(json.dumps(runs,indent=2))
    with open(HERE/'configs/run_manifest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
    (HERE/'reports/data_reference_audit.json').write_text(json.dumps(dict(status='STATIC_PASS',datasets=dataset_records,references=references,
        actual_loader_smoke='NOT_RUN',gpu_initialization='NOT_RUN',pool_generation='NOT_RUN'),indent=2,ensure_ascii=False))
    print('STATIC INPUT AUDIT PASS: 3 data files, 12 exact reference configurations')
    print('MANIFEST: 144 NOT_STARTED; teacher mode=null; no execution path enabled')

if __name__=='__main__':
    main()
