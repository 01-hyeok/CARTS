"""Resume only the two validated failures, with explicit recovery provenance."""
import sys,argparse,time,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from weather_patch import install,bridge,OUT,torch
sys.path.insert(0,str(HERE.parent))
import d56_complete_run as r

def main(model,smoke):
    assert model in ['V0','V1'];vg=OUT/'WEATHER_VALIDATION_GATE.json';g=r.json.loads(vg.read_text())
    assert g['status']=='PASS_ALL_APPROVED_SAMPLE_QUERIES_EXACT' and len(g['records'])==128
    for n,h in g['source_hashes'].items():assert r.sha(HERE/n)==h
    install();cell='Weather_336';training='P100';evaluation='FULL'
    out=r.OUT/('feature_smoke' if smoke else 'complete_features')/cell/(model+'_P100_to_FULL');out.mkdir(parents=True,exist_ok=True)
    lock=(r.BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=r.start_gpu();begin=time.perf_counter();r.os.chdir(r.ROOT);data=r.Data(cell)
    row=next(v for v in r.inventory(cell) if v['architecture']==model and v['training_support']==training)
    f=r.Frozen(data,row);prior=r.reproduction(data,row);assert prior is not None
    proof=dict(cell=cell,checkpoint=row,training_support=training,evaluation_support=evaluation,seed=0,
        native_reproduction_gate=str(prior),native_reproduction_sha256=r.sha(prior),pool_hashes=data.pool_hashes,data=data.data_info,
        sample='32 linspace validation and test timestamps/all channels',lag_definition=r.common_identity(data),
        source_sha256=r.sha(HERE/'recover_case.py'),original_feature_source_sha256=r.sha(Path(r.__file__)),
        protocol_sha256=r.sha(r.OUT/'PROTOCOL.json'),original_runtime='UNKNOWN',current_torch=torch.__version__,
        exact_tail_recovery_gate_sha256=r.sha(vg),exact_tail_source_hashes=g['source_hashes'],floating_point_tolerance_unchanged=True)
    pp=out/'PROVENANCE.json'
    if pp.exists():assert r.json.loads(pp.read_text())==proof
    else:r.dump(pp,proof)
    torch.cuda.reset_peak_memory_stats();files=r.case(data,f,training,evaluation,out,proof,smoke)
    assert r.sha(row['checkpoint'])==row['checkpoint_sha256'] and len(files)==(4 if smoke else 64)
    r.dump(out/'FEATURE_GATE.json',dict(status='PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL' if smoke else 'PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING',at=r.now(),identity=proof,
        files=files,queries={'val':2 if smoke else 32,'test':2 if smoke else 32},channels=data.c,seconds=time.perf_counter()-begin,gpu_before=before,gpu_after=r.snapshot(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        all_channels_single_batch=True,full_bank_count=len(data.starts),full_support_topR=True,source_checkpoint_unchanged=True))
    r.state(out,'COMPLETE_FEATURES_AGGREGATION_PENDING')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['V0','V1']);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.model,a.smoke)
