"""Explicit epoch-boundary recovery in new output, never original Experiment.__init__."""
from recovery_runtime import *
import time,copy,inspect,textwrap
import numpy as np
import torch
import run_experiment as main
from train_models import Features,Arm
SOURCE=ORIGINAL/'FULL/Weather_96'
def load_optimizer(opt,saved):
    opt.load_state_dict(copy.deepcopy(saved))
    # Legacy load_state_dict casts the scalar Adam step to parameter float64.
    # Preserve the saved float32 CPU step counter, as in uninterrupted AdamW.
    for group,source_group in zip(opt.param_groups,saved['param_groups']):
        for parameter,key in zip(group['params'],source_group['params']):
            step=saved['state'].get(key,{}).get('step')
            if isinstance(step,torch.Tensor):opt.state[parameter]['step']=step.clone()
def restore(device):
    boundary=read(SOURCE/'EPOCH_STATE.json');assert boundary['epoch']==8
    states=copy.deepcopy(boundary['arms']);arms={};opts={};records=[]
    assert [n for n in ARMS if states[n]['active']]==['D1_MLP','D2_L2','D2_MLP','C1_MLP']
    for n in ARMS:
        epoch=8 if states[n]['active'] else 6
        p=SOURCE/'checkpoints'/n/('epoch%02d.pt'%epoch);z=load(p)
        assert z['epoch']==epoch and z['state']==states[n]
        arm=Arm(n).to(device);arm.load_state_dict(z['model'],strict=True)
        opt=torch.optim.AdamW([v for v in arm.parameters() if v.requires_grad],lr=.001);load_optimizer(opt,z['optimizer'])
        arms[n]=arm;opts[n]=opt
        best=SOURCE/'checkpoints'/n/('epoch%02d.pt'%states[n]['best_epoch']);assert best.exists()
        records.append(dict(arm=n,path=str(p),sha256=sha(p),epoch=epoch,best_path=str(best),best_sha256=sha(best),state=states[n]))
    hist=read(SOURCE/'TRAIN_HISTORY.json');assert all(r['epoch']<=8 for r in hist)
    for n in ARMS:
        rows=[r for r in hist if r['arm']==n];best=min(rows,key=lambda r:(r['val_retmse'],r['epoch']))
        assert best['epoch']==states[n]['best_epoch'] and best['val_retmse']==states[n]['best']
    return arms,opts,states,hist,records
def recovered_train(ex):
    # Copy the original optimization/validation body exactly; replace initialization and start epoch only.
    src=textwrap.dedent(inspect.getsource(main.Experiment.train))
    start=src.index('    val=self.validate(ARMS)');end=src.index('    micro=32',start)
    src=src[:start]+"    state=copy.deepcopy(self.resume_state)\n"+src[end:]
    assert 'range(1,31)' in src;src=src.replace('range(1,31)','range(9,31)')
    ns=dict(main.__dict__);ns['copy']=copy;exec(compile(src,str(CODE/'resume_weather.py')+'::epoch_boundary_loop','exec'),ns)
    return ns['train'](ex)
def run():
    dest=OUT/'FULL/Weather_96';assert not dest.exists(),'No overwrite: review partial recovery first'
    ex=main.Experiment.__new__(main.Experiment)
    ex.cell='Weather_96';ex.support='FULL';ex.dest=dest
    ex.d,ex.f,ex.prior,ex.identity,ex.before=main.ev.setup(ex.cell,ex.support)
    proof=read(main.ev.OUT/'recovery02/Weather_96_NORM_PROBE.json');assert all(r['raw_exact'] and r['prob_exact'] and r['ids_match_old'] for r in proof['rows'])
    main.norm_patch.patch(ex.f.batch)
    ex.features=Features(ex.d,ORIGINAL/'features'/ex.cell)
    ex.arms,ex.opts,ex.resume_state,ex.history,records=restore(ex.d.device)
    ex.identity=read(SOURCE/'IDENTITY.json');ex.identity['recovery02']=dict(source=str(SOURCE),boundary_epoch=8,first_replayed_epoch=9,source_seal=sha(OUT/'SOURCE_SEAL.json'),rng_saved=False,uninterrupted_bitwise_equivalence=False)
    ex.cache={};ex.steps=max(r['steps'] for r in ex.history);ex.raw_time=0.;ex.scoring_time=0.;ex.started=time.perf_counter()
    dest.mkdir(parents=True)
    dump(dest/'IDENTITY.json',ex.identity)
    dump(dest/'RESTORE_GATE.json',dict(status='PASS_EXACT_MODEL_OPTIMIZER_BOUNDARY',checkpoints=records,steps=ex.steps,history_rows=len(ex.history),normalizers_refit=False,initial_validation_or_epoch00=False,early_stopped_reactivated=False,at=now()))
    dump(dest/'TRAIN_HISTORY.json',ex.history);dump(dest/'EPOCH_STATE.json',dict(epoch=8,arms=ex.resume_state,at=now()))
    dump(dest/'TRAINING_STARTED.json',dict(at=now(),recovery=True,source=str(SOURCE),start_epoch=9,seed=0,discarded_partial_epoch=9,last_original_block=365))
    ex.gpu_gate()
    print('RECOVERY_BOUNDARY_RESTORED',json.dumps(records),flush=True)
    state=recovered_train(ex)
    from final_evaluation import finish
    finish(ex,state)
    elapsed=time.perf_counter()-ex.started
    dump(dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_TRAIN_VAL_TEST',at=now(),cell=ex.cell,support=ex.support,selection=state,source_hashes=sources(),identity=ex.identity,queries={s:len(d) for s,d in ex.d.datasets.items()},channels=ex.d.c,elapsed_seconds=max(r.get('seconds',0) for r in read(SOURCE/'TRAIN_HISTORY.json'))+elapsed,raw_encoding_distance_seconds=ex.raw_time,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),recovery=dict(peak_scope='recovery process only; original peak unavailable',elapsed_scope='completed original epoch0-8 plus recovery; excludes discarded partial epoch9 and downtime',partial_epoch9_cost='not included in elapsed; original resource/log evidence retained',input_manifest=sha(OUT/'REUSE_MANIFEST.json'))))
