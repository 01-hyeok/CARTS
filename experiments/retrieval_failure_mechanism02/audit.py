"""Read existing checkpoints/results on CPU; write only new Mechanism02 audit files."""
import argparse,csv,datetime,hashlib,json
from pathlib import Path
from protocol import ROOT,CELLS,MODELS,PROTOCOL

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def audit(root=ROOT):
    import torch
    torch.set_num_threads(1)
    root=Path(root);out=root/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit';out.mkdir(parents=True,exist_ok=True)
    prior=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/CHECKPOINT_INVENTORY.csv'
    old=list(csv.DictReader(prior.open()));rows=[];fair=[];v1_detail=[]
    p100=root/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02'
    main=list(csv.DictReader((p100/'MAIN_SEED_RESULTS.csv').open()))
    for cell in CELLS:
        ds,h=cell.rsplit('_',1);pe=[]
        for model in MODELS:
            found=[x for x in main if x['dataset']==ds and x['horizon']==h and x['model']==model and x['seed']=='0'];assert len(found)==1
            m=found[0];matches=list(p100.glob(f'**/{cell}/seed0/{model}/COMPLETE.json'));assert len(matches)==1
            path=matches[0];done=read(path);prov=read(path.parent/'provenance.json');epochs=read(path.parent/'epochs.json');selection=read(path.parent/'checkpoint_selection.json')
            assert done['status']=='COMPLETE' and selection['test_used_for_selection'] is False
            assert len(epochs)==10 and [x['epoch'] for x in epochs]==list(range(1,11))
            cp=Path(m['checkpoint']);assert sha(cp)==m['checkpoint_sha256']==done['checkpoint_sha256']
            assert selection['epoch']==int(m['selected_epoch'])
            assert all(x['train_queries']==prov['run']['n_train_queries'] for x in epochs)
            pe.append((model,prov,[x['batch_hash'] for x in epochs]))
            rows.append(dict(cell=cell,dataset=ds,horizon=int(h),architecture=model,seed=0,training_support='P100',status='CPU_PROVENANCE_PASS_GPU_REPRODUCTION_PENDING',checkpoint=str(cp),checkpoint_sha256=m['checkpoint_sha256'],selected_epoch=int(m['selected_epoch']),epochs_completed=10,selection_criterion=prov['selection'],teacher='valid-support zscore negative FutureMSE KL, tau=.1',evidence=str(path.parent),canonical_retmse=m['retmse'],canonical_agg_mse=m['agg_mse'],canonical_agg_mae=m['agg_mae'],canonical_recall=m['recall'],runtime='per-run source hashes; native original selection frozen',new_training=False))
        f=dict(cell=cell,models=list(MODELS),common_init=len({x[1]['initial_model_hash'] for x in pe})==1,
               common_pool=len({json.dumps(x[1]['pool_hashes'],sort_keys=True) for x in pe})==1,
               common_batches=all(x[2]==pe[0][2] for x in pe),all_ten_epochs=True,
               teacher_equal=len({json.dumps(x[1]['teacher'],sort_keys=True) for x in pe})==1)
        assert all(f[k] for k in ['common_init','common_pool','common_batches','all_ten_epochs','teacher_equal'])
        fair.append(f)
        for model in ('V0','V5'):
            rr=[x for x in old if x['dataset']==ds and x['horizon']==h and x['architecture']==model and x['training_support']=='FULL'];assert len(rr)==1
            rr=rr[0];assert sha(rr['checkpoint'])==rr['checkpoint_sha256']
            rows.append(dict(cell=cell,**{k:rr[k] for k in ['dataset','horizon','architecture','seed','training_support','checkpoint','checkpoint_sha256','selected_epoch','selection_criterion','evidence']},status='CPU_PROVENANCE_PASS_GPU_REPRODUCTION_PENDING',new_training=False,runtime='historical generation UNKNOWN; existing full reproduction proof retained separately'))
        cp=root/f'checkpoints/track_v_multiquery_generalization01/{ds}/H{h}/seed0/V1/stage1/checkpoint.pth'
        evidence=root/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/V1/stage1'
        if not cp.exists():
            rows.append(dict(cell=cell,dataset=ds,horizon=int(h),architecture='V1',seed=0,training_support='FULL',status='MISSING_V1_TRAINING_AUTHORIZED_AFTER_PREDECESSORS',new_training=True));continue
        v=torch.load(cp,map_location='cpu',weights_only=False);cfg=v['config'];assert cfg['num_slots']==1 and cfg['init_seed']==0
        assert list(v['slot_heads_state_dict']['W'].shape)==[1,128,128]
        assert v['model_state_dict']['encoder.encoder.0.weight'].shape[1]==int(h)
        vr=evidence/'val_metrics.csv';values=list(csv.DictReader(vr.open())) if vr.exists() else []
        if not values:
            for ep in sorted(cp.parent.glob('checkpoint_epoch*.pth')):
                ev=torch.load(ep,map_location='cpu',weights_only=False)
                values.append(dict(epoch=ev['epoch'],retmse10=ev['val_retmse10']))
        assert values,'Original selection cannot be authenticated; do not reselect on fresh evaluation'
        argmin=min(values,key=lambda x:(float(x['retmse10']),int(x['epoch'])))
        assert int(argmin['epoch'])==v['epoch']
        assert abs(float(argmin['retmse10'])-v['val_retmse10'])<1e-12
        ref=evidence/'final_test_metrics.json'
        if not ref.exists():ref=evidence.parent/'cache/stage1_metrics.json'
        meta=read(ref) if ref.exists() else {}
        row=dict(cell=cell,dataset=ds,horizon=int(h),architecture='V1',seed=0,training_support='FULL',status='CPU_PROVENANCE_PASS_GPU_REPRODUCTION_PENDING',checkpoint=str(cp),checkpoint_sha256=sha(cp),selected_epoch=v['epoch'],epochs_completed=len(values),selection_criterion='original min validation single-head raw-score Top10 RetMSE; no reselection',teacher='valid FULL zscore negative FutureMSE KL; u=-MSE then teacher(-u)',evidence=str(evidence),native_metric_source=str(ref),canonical_retmse=meta.get('retmse10'),canonical_agg_mse=meta.get('agg_mse10'),canonical_recall=meta.get('recall10'),canonical_agg_mae='NOT_STORED_REINFERENCE_REQUIRED',runtime='UNKNOWN historical generation; all-channel GPU canonical gate required',new_training=False)
        rows.append(row);v1_detail.append(dict(cell=cell,config=cfg,selected_epoch=v['epoch'],original_val_mse=v['val_retmse10'],validation_epoch_records=values,checkpoint_sha256=row['checkpoint_sha256'],note='S=1 round-robin degenerates to ordinary raw-score Top10; softmax FP32 tie changes must not be hidden; do not reuse multihead Round-Robin cache'))
    assert len(rows)==48
    csvwrite(out/'CHECKPOINT_AUDIT.csv',rows);dump(out/'P100_COMPARABILITY.json',fair);dump(out/'FULL_V1_ORIGINAL_SELECTION.json',v1_detail)
    dump(out/'PROTOCOL.json',PROTOCOL)
    targets=[dict(ordinal=i+1,cell=cell,phase='P100_AND_FULL_WITH_CROSS_SUPPORT',state='WAITING_SOLAR_AND_IMPLEMENTATION_GATES',models=list(MODELS)) for i,cell in enumerate(CELLS)]
    dump(out/'EXECUTION_QUEUE.json',dict(prerequisite='Solar Full8 final gate + 4 complete model-gap diagnostic cells + original24 deliverable/summary gates + owner exit + GPU1 free',targets=targets))
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status=dict(at=now,status='CPU_AUDIT_COMPLETE_GPU_NOT_STARTED',checkpoints=len(rows),p100_v1=8,full_v1_found=len(v1_detail),v1_new_required=sum(x.get('new_training',False) for x in rows),full_v1_gpu_verified=0,existing_models_reselected=False,original_files_modified=False)
    dump(out/'RUN_STATUS.json',status)
    print(json.dumps(status));return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();audit(args.root)
