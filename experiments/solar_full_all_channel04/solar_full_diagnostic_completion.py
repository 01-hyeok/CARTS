"""Separate diagnostic supplement for newly trained Solar Full models.

Runs only after the original P100/Full diagnostics and all eight Solar training
runs. Original missing-model reports remain immutable historical snapshots.
"""
import argparse,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import solar_full_train as train
import diagnostic_runtime as dr
run=dr.configure()
import diagnostic_report as report
from diagnostic_adapter import Data,rt,GPU
SolarFrozen=dr.SolarFrozen
OUT=dr.OUT


def inherited_decomposition(cell):
    # The prior P100-selected decomposition already exists and has not changed.
    # New Full model errors are relative to Full Oracle, never forced >= P100 Oracle.
    original=train.DIAG/'FULL'/cell/'RETRIEVAL_REGRET_DECOMPOSITION.csv'
    assert original.exists()
    rt.write_json(OUT/'FULL'/cell/'P100_DECOMPOSITION_REFERENCE.json',dict(path=str(original),sha256=rt.sha256_file(original),
        note='Existing P100-selected decomposition retained; do not apply P100 inequality to new Full-selected models'))

def main(cell):
    train.guard()
    for h in [96,192,336,720]:
        for m in ['V0','V5']:assert train.read(train.OUT/f'Solar_{h}'/m/'seed0/COMPLETE.json')['status']=='COMPLETE'
    out=OUT/'FULL'/cell;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        prior=train.read(out/'COMPLETE.json');assert prior['solar_source_hashes']==train.source_hashes();return
    run.OUT=OUT;report.OUT=OUT;report.backfill_decomposition=inherited_decomposition
    data=Data(cell);calendar=run.Calendar(data);rows=[]
    for m in ['V0','V5']:
        dest=train.OUT/cell/m/'seed0';meta=train.read(dest/'COMPLETE.json')
        rows.append(dict(status='VALID',architecture=m,training_support='FULL',checkpoint=meta['checkpoint'],
            checkpoint_sha256=meta['checkpoint_sha256'],evidence=str(dest),selected_epoch=meta['selection']['epoch'],
            selection_criterion='validation Mean-Mixture RetMSE after ten epochs',seed=0))
    models=[SolarFrozen(data,r,'FULL') for r in rows]
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    rt.write_json(out/'MODEL_STATUS.json',dict(models=rows,missing=[],prior_missing_snapshot=str(train.DIAG/'FULL'/cell/'MODEL_STATUS.json')))
    rt.write_json(out/'PROVENANCE.json',dict(cell=cell,support='FULL',seed=0,data=data.data_info,pool_hashes=data.pool_hashes,
        source_hashes=run.snapshot_hashes(),solar_source_hashes=train.source_hashes(),new_stage1_training_in_this_diagnostic=False,
        all_queries=True,all_channels=True,bootstrap_seed=0,stage2=False))
    run.reproduce(data,models,'FULL',out)
    ridge=run.fit_probe(data,'FULL',calendar,out)
    val=run.evaluate_split(data,models,'FULL','val',calendar,ridge,out,None)
    ridge.choose(val['probe_validation_loss'],'val');rt.write_json(out/'probe_selection.json',dict(status='VALIDATION_SELECTED',state=ridge.state(),test_used_for_selection=False))
    test=run.evaluate_split(data,models,'FULL','test',calendar,ridge,out,ridge.selected)
    report.summarize_cell('FULL',cell)
    rt.write_json(out/'COMPLETE.json',dict(status='COMPLETE',cell=cell,support='FULL',completed_at=rt.now(),models=[m.immutable_check() for m in models],
        missing_models=[],solar_source_hashes=train.source_hashes(),diagnostic_source_hashes=run.snapshot_hashes(),channels=data.c,
        query_counts={s:len(data.datasets[s]) for s in ['val','test']},stage1_retraining=False,stage2=False,
        total_wall_seconds=time.perf_counter()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        cpu_ram_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print('SOLAR FULL DIAGNOSTIC COMPLETE',cell,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cell',choices=[f'Solar_{h}' for h in [96,192,336,720]]);args=ap.parse_args()
    try:main(args.cell)
    except BaseException as exc:
        rt.write_json(OUT/'IMMEDIATE_ISSUE.json',dict(status='FAILED',cell=args.cell,error=repr(exc),traceback=traceback.format_exc(),at=rt.now()))
        raise
