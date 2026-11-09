"""Read-only checkpoint inventory and pinned diagnostic protocol; CPU only."""
import csv,hashlib,json,subprocess,datetime
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=ROOT/'experiments/retrieval_failure_attribution01'
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    HERE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);audit=OUT/'00_Audit';audit.mkdir(exist_ok=True)
    base=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'
    manifest=json.loads((base/'configs/run_manifest.json').read_text());cfg=json.loads((base/'configs/protocol.json').read_text())
    observed_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cells=[f'{d}_{h}' for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720)]
    inventory=[]
    for support in ('P100','FULL'):
        for cell in cells:
            dataset,h=cell.rsplit('_',1);h=int(h)
            for model in ('V0','V5'):
                row=dict(dataset=dataset,horizon=h,architecture=model,seed=0,training_support=support,
                    intended_evaluation_support=support,pool_method='baseline delta-last history cosine' if support=='P100' else 'all valid train candidates',
                    objective='KL(zscore negative future-MSE teacher || mean per-head softmax); V0 single distribution',
                    status='MISSING',checkpoint='',checkpoint_sha256='',selected_epoch='',selection_criterion='',
                    source_commit='UNKNOWN; source hashes/provenance retained',metric_reproduction='NOT_RUN',evidence='')
                if support=='P100':
                    run=next(x for x in manifest if x['run_id']==f'{cell}_{model}_seed0');dest=ROOT/run['output_path'];done=dest/'COMPLETE.json'
                    row['evidence']=str(dest);row['run_id']=run['run_id']
                    if done.exists():
                        meta=json.loads(done.read_text());selection=json.loads((dest/'checkpoint_selection.json').read_text());prov=json.loads((dest/'provenance.json').read_text())
                        assert sha(meta['selected_checkpoint'])==meta['checkpoint_sha256']==selection['sha256']
                        assert selection['test_used_for_selection'] is False
                        row.update(status='VALID',checkpoint=meta['selected_checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],selected_epoch=selection['epoch'],selection_criterion=prov['selection'],
                            source_hash_manifest=str(dest/'provenance.json'),init_policy='seed0 common random encoder; no reference weight warm start',
                            canonical_retmse=meta['official_test']['retmse'],canonical_recall=meta['official_test']['recall'])
                    else:row.update(status='MISSING',availability='PENDING_EXISTING_APPROVED_TRAINING_OR_FINAL_EVALUATION; do not retrain for diagnostics')
                elif dataset!='Solar':
                    loc=ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{dataset}/H{h}/seed0/{model}/stage1'
                    if model=='V5':
                        sel=ROOT/f'results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{dataset}/H{h}/seed0/checkpoint_selection/selected_checkpoint.json'
                        if sel.exists():
                            meta=json.loads(sel.read_text());ck=ROOT/meta['selected_checkpoint_path'];criterion=meta['selection_criterion'];epoch=meta['selected_epoch']
                            assert 'pool_top100' not in str(ck) and 'Mean-Mixture' in criterion
                            row.update(checkpoint=str(ck),selected_epoch=epoch,selection_criterion=criterion,evidence=str(sel))
                    else:
                        loc=loc/cell;sel=loc/'checkpoint_fingerprints.json';config=loc/'config.json'
                        if sel.exists() and config.exists():
                            meta=json.loads(sel.read_text());conf=json.loads(config.read_text());ck=ROOT/f'checkpoints/track_v_multiquery_generalization01/{dataset}/H{h}/seed0/V0/stage1/{cell}/checkpoint.pth'
                            assert conf['n_candidates']>100
                            row.update(checkpoint=str(ck),selected_epoch=meta['best_epoch'],selection_criterion=conf['checkpoint_criterion'],evidence=str(loc),
                                full_bank_count_recorded=conf['n_candidates'],init_policy='historical Full trainer; do not equate with new P100 scratch init')
                    if row['checkpoint'] and Path(row['checkpoint']).exists():row.update(status='VALID',checkpoint_sha256=sha(row['checkpoint']))
                else:row.update(availability='No audited Full-trained V0/V5 checkpoint in corrected track; reference WCE/config checkpoints excluded until objective/selection compatibility established')
                row['validation_note']='VALID means provenance-ready only; this new diagnostic reproduction gate remains NOT_RUN'
                inventory.append(row)
    csvwrite(audit/'CHECKPOINT_INVENTORY.csv',inventory)
    cross=[]
    for row in inventory:
        if row['training_support']=='P100':cross.append(dict(dataset=row['dataset'],horizon=row['horizon'],architecture=row['architecture'],training_support='P100',evaluation_support='FULL',
            status='DEFERRED_FULL_DEPENDENCY',checkpoint=row['checkpoint'],note='Frozen same-parameter support counterfactual only; never call this Full-trained. Full inference/reproduction gate after all P100 cells.'))
    csvwrite(audit/'CROSS_SUPPORT_CAPABILITY.csv',cross)
    sources=['scripts/train_v_sharedtop100_01.py','scripts/eval_v5_meanmix_checkpoint_selection_pool01.py','utils/mean_mixture_selection.py','utils/candidate_pool.py','utils/full_candidate_bank.py','utils/rank_losses.py','utils/past_neighborhood.py','scripts/analyze_similar_past_stability.py','scripts/build_ambiguity_report.py',
        'experiments/full_head_analysis01/full_analysis.py','experiments/p100_head_retrieval_quality_legacy_pool02/runtime.py']
    write(audit/'SOURCE_HASHES.json',dict(observed_live_head=observed_head,sources={n:sha(ROOT/n) if (ROOT/n).exists() else 'MISSING' for n in sources}))
    protocol=dict(version='retrieval_failure_attribution01',confirmed_attachment='libfile_bdf819f7a4ac8191b126af541b9cfa1b',withdrawn_attachment_not_used='libfile_f87a8892914481918202046e8c2368b7',
        seed=0,gpu_physical_index=1,gpu_uuid='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec',stage1_training=False,stage2=False,
        stage_order=['P100_ALL_12','FULL_ALL_12','FILL_DEFERRED_FULL_DEPENDENCIES'],cell_order=cells,
        primary_metric='test RetMSE@10',topk=10,seq_len_equals_pred_len=True,all_queries=True,all_channels=True,
        ndcg=dict(name='NDCG_inverse_MSE_v1',relevance='g(d)=1/(1+d), d>=0 normalized delta-last aligned future MSE',gain='g directly, not 2**g-1',discount='1/log2(rank+1)',
            ideal='same valid support sorted increasing future MSE',tie='stable inherited support order: P100 stored ID order; FULL canonical bank ID order',legacy_metric='Do not reuse legacy NDCG name or claim parity if transform differs'),
        ranking=dict(spearman='average ranks for exact ties',kendall_p100='exact tau-b',kendall_full='APPROXIMATE tau-b from fixed future-free pairs; sampling budget to freeze before execution'),
        regret=dict(identity='Rmodel-Rfull=(Rp100-Rfull)+(Rmodel-Rp100)',inequality_domain='Rfull<=Rp100<=Rmodel applies ONLY to model selected inside that same P100',
            full_model='For Full-selected model assert Rfull<=Rmodel only; never require Rp100<=Rmodel',p100_phase_full_dependencies='DEFERRED_FULL_DEPENDENCY'),
        full_scoring=dict(streaming=True,global_softmax='exact global per-head logsumexp across all eligible candidates, then mean probabilities; no per-chunk softmax',
            full_oracle='stream all valid candidates, exact running Top10; no query/channel/candidate pruning'),
        probe=dict(allowed=['ridge','small two-layer MLP'],input='past-only',candidate_selection='future-free; no Oracle-chosen training pairs',fit='train only',scaler='train only',selection='validation only',test='one final evaluation',retriever_frozen=True),
        statistics=dict(bootstrap_seed=0,repetitions=2000,time_block_length='2*(seq_len+pred_len)',channels='resample all channels of each query time jointly',uncertainty='exploratory; no independent-window p-values'),
        scheduler=dict(wait_for_existing_48_seed0_complete=True,do_not_interrupt_existing_jobs=True,major_gpu_workers=1,upload=False,push=False,
            pending_gpu='Existing training resumes after separately approved validation accelerator gate; new diagnostics cannot acquire GPU until it finishes'))
    # Once frozen, a refreshed inventory must not silently redefine metrics.
    if not (HERE/'PROTOCOL.json').exists():write(HERE/'PROTOCOL.json',protocol)
    queue=[dict(ordinal=i+1,support=s,cell=c,status='WAITING_EXISTING_TRAINING' if s=='P100' else 'BLOCKED_P100_ALL_COMPLETE') for i,(s,c) in enumerate((s,c) for s in ('P100','FULL') for c in cells)]
    if not (HERE/'RUN_QUEUE.json').exists():write(HERE/'RUN_QUEUE.json',queue)
    statuses={f'{s}_{st}':sum(x['training_support']==s and x['status']==st for x in inventory) for s in ('P100','FULL') for st in ('VALID','MISSING','INCOMPATIBLE')}
    write(audit/'RUN_STATUS.json',dict(status='PHASE0_PROVENANCE_COMPLETE_IMPLEMENTATION_PENDING',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),inventory=statuses,
        diagnostic_gpu_runs_started=0,metric_reproduction='NOT_RUN',full_search_started=False,existing_training_unchanged=True,
        blocking_dependencies=['Existing48-run queue retains exclusive major GPU use','New diagnostic inference adapters and real-checkpoint reproduction gates still required','Missing pending Solar P100 checkpoints and audited Full-trained Solar checkpoints; no diagnostic retraining authorized']))
    print(json.dumps(statuses))
