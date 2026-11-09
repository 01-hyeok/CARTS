"""CPU input integration checks. Does not instantiate GPU contexts or models on CUDA."""
import importlib.util,json,os,sys
from pathlib import Path
import calibration_adapter as b

def input_records(data,allowed=b.OLD,load_checkpoint=None,connect=True):
    import numpy as np
    b.check_identity_set(data['records'])
    if load_checkpoint is None:
        import torch
        if hasattr(torch.serialization,'add_safe_globals'):
            from torch.torch_version import TorchVersion
            torch.serialization.add_safe_globals([TorchVersion])
            load_checkpoint=lambda p:torch.load(p,map_location='cpu',weights_only=True)
        else:
            # Original Torch1 runtime lacks weights_only. Only load owner's
            # approved checkpoints after the record hash checks above.
            load_checkpoint=lambda p:torch.load(p,map_location='cpu')
    mapped={b.key(r['label']):r for r in data['records']};router=b.ResolvedPath(mapped)
    files=0;checkpoints=0;linked=[]
    for r in data['records']:
        b.check_record(r,allowed);label=r['label'];root=Path(r['path']);selection=b.read(r['selection_path']);identity=r['identity']
        expected_status='REUSED_NATIVE_SELECTION_NO_REFIT' if label['mode']=='cross' else 'FROZEN_BEFORE_FINAL_TEST'
        b.require(selection.get('status')==expected_status and set(selection['arms'])==set(b.ARMS),'Selection schema')
        b.require(identity['data']['channels']==r['channels'] and all(identity['data']['counts'][s]==r['queries'][s] for s in ['val','test']),'Population provenance')
        b.require(set(identity['pool_hashes'])=={'train','val','test'},'Pool provenance schema')
        for n,cp in r['checkpoints'].items():
            z=load_checkpoint(cp['path']);b.require(z['epoch']==cp['epoch'] and isinstance(z['model'],dict),'Checkpoint schema')
            b.require(z['state']['best_epoch']==selection['arms'][n]['best_epoch'] and z['state']['best']==selection['arms'][n]['best'],'Checkpoint selection state')
            checkpoints+=1
        rows=b.read(root/'FULL_RESULTS.json');required={(s,m) for s in ['val','test'] for m in ['S0','D0','C0']+b.ARMS+['Oracle10','Random','LastRepeat']}
        b.require(len(rows)==24 and {(v['split'],v['method']) for v in rows}==required,'Results schema')
        for row in rows:
            b.require(row['queries']==r['queries'][row['split']] and row['channels']==r['channels'] and row['support']==label['support'] and row['seed']==0,'Results population mismatch')
            b.require(all(np.isfinite(row[k]) for k in ['retmse','agg_mse','agg_mae']),'Nonfinite results')
        for split in ['val','test']:
            rawdir=Path(r['raw_directories'][split]);b.require(rawdir==root/'final_raw'/split,'Raw directory mismatch')
            blocks=sorted(rawdir.glob('*.npz'));b.require(len(blocks)==(r['queries'][split]+31)//32,'Raw block coverage')
            starts=[]
            for i,p in enumerate(blocks):
                b.require(p.name=='batch%05d.npz'%i and str(p) in r['raw_hashes'],'Raw name or manifest missing')
                b.require(b.read(p.with_suffix('.json'))['sha256']==r['raw_hashes'][str(p)],'Raw sidecar mismatch')
                with np.load(p,allow_pickle=False) as z:
                    qs=z['query_start'];count=min(32,r['queries'][split]-i*32);b.require(qs.shape==(count,),'Query shape');starts.extend(qs.tolist())
                    for method in ['S0','D2_MLP','C2_MLP']:
                        b.require(z[method+'__selected_ids'].shape==(count,r['channels'],10),'Selected ID shape')
                        mse=z[method+'__selected_mse'];b.require(mse.shape==(count,r['channels'],10) and np.isfinite(mse).all(),'Selected MSE schema')
                files+=1
            b.require(len(set(starts))==len(starts) and starts==sorted(starts),'Duplicate or reordered queries')
        if label['mode']=='native' and label['training']=='FULL':
            logical=router/'FULL'/label['cell']
            b.require(Path(logical/'SELECTION.json')==Path(r['selection_path']),'Full selection routing')
            for n in ['D2_MLP','C2_MLP']:
                cp=r['checkpoints'][n];b.require(Path(logical/'checkpoints'/n/('epoch%02d.pt'%cp['epoch']))==Path(cp['path']),'Full checkpoint routing')
            b.require(Path(logical/'FULL_RESULTS.json')==root/'FULL_RESULTS.json','Full result routing');linked.append(str(logical))
        if label['mode']=='cross':
            logical=router/'supplemental/cross'/('V5_'+label['training']+'_to_'+label['support'])/label['cell']
            b.require(Path(logical/'FULL_RESULTS.json')==root/'FULL_RESULTS.json','Cross routing')
            native=mapped[b.key(b.label('native',label['cell'],label['training']))]
            for field in ['data','pool_hashes','checkpoint']:b.require(identity[field]==native['identity'][field],'Cross/native provenance mismatch')
            b.require(r['checkpoints']==native['checkpoints'],'Cross checkpoint differs from native')
    modules=[]
    if connect:
        # Only imports and binds globals: never calls FullRun/NativeContext/run().
        sys.path[:0]=[str(b.CAL/'full_extension01'),str(b.CAL)]
        for name in ['full_p0','full_train','cross_support']:
            spec=importlib.util.spec_from_file_location('bridge02_preflight_'+name,b.CAL/'full_extension01'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OLD=router
            fn=m.FullRun.evaluate if name=='full_train' and hasattr(m.FullRun,'evaluate') else (m.FullRun.run if name=='full_train' else m.run)
            b.require(fn.__globals__['OLD'] is router,'Module binding failed');modules.append(name)
        from train_models import Arm
        for r in data['records']:
            if r['label']['mode']=='native' and r['label']['training']=='FULL':
                for n in ['D2_MLP','C2_MLP']:
                    arm=Arm(n).cpu();arm.load_state_dict(load_checkpoint(r['checkpoints'][n]['path'])['model'],strict=True)
    return dict(status='PASS_CPU_INPUT_INTEGRATION_NOT_GPU',records=40,raw_blocks=files,checkpoints=checkpoints,full_paths=linked,modules=modules)

def run():
    b.seals();data=b.upstream();proof=input_records(data)
    final=Path(data['ready']['final_complete']);done=b.read(final)
    bindings=dict(upstream_final_sha256=data['ready']['final_sha256'],validation_sha256=done['validation_sha256'],input_manifest_sha256=done['input_manifest_sha256'],upstream_source_seal_sha256=done['source_seal_sha256'],adapter_seal_sha256=b.sha(b.OUT/'SOURCE_SEAL.json'))
    b.require(b.sha(final)==bindings['upstream_final_sha256'],'Final changed during preflight');b.seals()
    return data,dict(proof,bindings=bindings)

