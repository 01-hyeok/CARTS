"""Compare unchanged full diagnostic metrics on actual all-channel batches."""
import sys,os,json,time,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT/'experiments/retrieval_diagnostic_supplement01'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import diagnostic_run as run
from supplement_inference import restore_probe
import torch,numpy as np
from packed_batch import PackedChannelFull

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);args=ap.parse_args()
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();data=a.Data(args.cell);calendar=run.Calendar(data)
    # Existing frozen probe weights are used solely for arithmetic comparison.
    path=a.OUT/'P100'/args.cell/'probe_selection.json';ridge=restore_probe(path)
    row=next(r for r in a.inventory(data.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(data,row,'FULL');fast=PackedChannelFull(f)
    ref=f.probability('val',0,32);got=fast.probability('val',0,32)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    old=run.evaluate_micro(data,'val',0,4,'FULL',calendar,ridge,{f.name:ref[:4]},None,128)
    torch.cuda.synchronize();oldtime=time.perf_counter()-t
    import runtime_all_channels
    torch.cuda.synchronize();t=time.perf_counter()
    new=run.evaluate_micro(data,'val',0,4,'FULL',calendar,ridge,{f.name:got[:4]},None,128)
    torch.cuda.synchronize();newtime=time.perf_counter()-t
    rows=[];passed=True
    for k in ['metrics','selected_ids','oracle_ids','group_features','neighborhood','probe_validation_loss_sums','p100_oracle','coverage','query_start']:
        x,y=old[k],new[k];equal=np.array_equal(x,y,equal_nan=True)
        finite=np.isfinite(x)&np.isfinite(y);diff=float(np.abs(x[finite]-y[finite]).max()) if finite.any() else 0.
        rows.append(dict(key=k,shape=list(x.shape),bit_exact=equal,maxdiff=diff));passed=passed and equal
    pred={n:float((old['predictions'][n]-new['predictions'][n]).abs().max()) for n in old['predictions']}
    passed=passed and all(x==0 for x in pred.values())
    report=dict(status='PASS' if passed else 'FAIL',cell=data.cell,queries=4,channels=data.c,candidates=len(data.starts),rows=rows,prediction_maxdiff=pred,old_seconds=oldtime,batched_seconds=newtime,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),probe_source=str(path),probe_only_arithmetic_fixture=True,checkpoint_sha256=row['checkpoint_sha256'])
    a.rt.write_json(dest/(data.cell+'_METRIC_BATCH_GATE.json'),report);print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
