"""Same all-channel adapter for supplemental ranking inference; separate outputs."""
import argparse,sys
import diagnostic_runtime as dr
import solar_full_train as train
sys.path.insert(0,str(train.ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_common as common
import supplement_cpu as cpu
import supplement_inference as inference
import supplement_report as report

def configure():
    dr.configure()
    for m in (common,cpu,inference,report):
        m.SOURCE=dr.OUT;m.OUT=dr.SUPP
    # No unrelated P100 rerun. This Full supplement is downstream of the
    # independently verified original P100 gate (hash-bound in prerequisites).
    inference.Frozen=dr.SolarFrozen;inference.inventory=dr.inventory
    inference.prepare_block=dr.run.prepare_block;inference.snapshot_hashes=dr.snapshot

def main(mode,cell):
    train.prerequisites();configure()
    for h in (96,192,336,720):
        for m in ('V0','V5'):assert train.read(train.OUT/f'Solar_{h}'/m/'seed0/COMPLETE.json')['status']=='COMPLETE'
    dr.SUPP.mkdir(parents=True,exist_ok=True)
    source=train.ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01/P100_DELIVERABLE_GATE.json'
    # Explicit reference, not copied experiment results. Original routine only
    # checks status; prerequisites additionally validates all source/hash gates.
    train.dump(dr.SUPP/'P100_DELIVERABLE_GATE.json',dict(status='PASS',reference=str(source),sha256=train.rt.sha256_file(source),kind='PREREQUISITE_REFERENCE_ONLY'))
    if mode=='cpu':cpu.run('FULL',cell)
    elif mode=='inference':inference.run('FULL',cell)
    elif mode=='gate':report.cell_gate('FULL',cell)
    else:raise ValueError(mode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','inference','gate']);p.add_argument('cell',choices=[f'Solar_{h}' for h in (96,192,336,720)]);a=p.parse_args();main(a.mode,a.cell)
