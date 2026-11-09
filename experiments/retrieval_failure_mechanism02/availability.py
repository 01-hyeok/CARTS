"""Read-only reuse/coverage inventory, with current hashes of every cited file."""
import csv,json
from pathlib import Path
from protocol import ROOT,CELLS
from io_utils import read,sha,dump,now

def main():
    out=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit'
    audit=list(csv.DictReader((out/'CHECKPOINT_AUDIT.csv').open()));rows=[];reused=[]
    for cell in CELLS:
        for support in ('P100','FULL'):
            base=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'/support/cell
            done=read(base/'COMPLETE.json');assert done['status']=='COMPLETE'
            supp=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'/support/cell
            dg=read(supp/'DELIVERABLE_GATE.json');assert dg['status']=='PASS' and dg['source_complete_sha256']==sha(base/'COMPLETE.json')
            table=base/'RETRIEVAL_BASELINE_COMPARISON.csv';allrows=list(csv.DictReader(table.open()))
            for method in ('Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine','Random','IndividualMSE_Oracle'):
                found=[r for r in allrows if r['method']==method]
                assert {r['split'] for r in found}=={'val','test'},(cell,support,method)
                for r in found:
                    g=read(base/r['split']/'SPLIT_GATE.json');assert g['status']=='PASS'
                    assert int(r['queries'])==g['query_count']==done['query_counts'][r['split']]
                    assert int(r['channels'])==g['channels']==done['channels']
                    assert int(r['units'])==g['units']==g['query_count']*g['channels']
                    reused.append(dict(**r,source_csv=str(table),source_sha256=sha(table),source_complete_sha256=sha(base/'COMPLETE.json')))
            v1=next(r for r in audit if r['cell']==cell and r['architecture']=='V1' and r['training_support']==support)
            rows.append(dict(cell=cell,support=support,raw_cosine='EXISTING_FULL_POPULATION_PASS',raw_csv=str(table),raw_sha256=sha(table),
              test_queries=done['query_counts']['test'],channels=done['channels'],v1_checkpoint=v1['checkpoint'],v1_sha256=v1['checkpoint_sha256'],
              v1_original_selected_epoch=v1['selected_epoch'],v1_checkpoint_status='CPU_PROVENANCE_PASS',
              v1_basic_test='EXISTING_ALL_QUERIES' if support=='P100' else 'EXISTING_NATIVE_METRICS_REPRODUCTION_PENDING',
              v1_new_cross_support='IMPLEMENTED_GPU_PENDING',v1_new_ranking='IMPLEMENTED_GPU_PENDING',
              v1_new_missed_oracle='IMPLEMENTED_GPU_PENDING',v1_new_regret='IMPLEMENTED_GPU_PENDING',
              v1_learnability_comparison='FROZEN_PROBE_REUSE_PROVENANCE_PENDING',v1_multihead_only='NOT_APPLICABLE',
              v1_new_training_required=False,deliverable_gate=str(supp/'DELIVERABLE_GATE.json'),deliverable_gate_sha256=sha(supp/'DELIVERABLE_GATE.json')))
    def write(path,values):
        keys=list(dict.fromkeys(k for r in values for k in r))
        with path.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(values)
    write(out/'RAW_V1_AVAILABILITY.csv',rows);write(out/'REUSED_BASELINE_ROWS.csv',reused)
    dump(out/'AVAILABILITY_GATE.json',dict(status='PASS_EXISTING_COVERAGE_NOT_NEW_GPU_GATE',at=now(),raw_conditions=len(rows),
         p100_v1_conditions=8,full_v1_checkpoint_conditions=8,new_v1_training_required=0,
         table_sha256=sha(out/'RAW_V1_AVAILABILITY.csv'),baseline_sha256=sha(out/'REUSED_BASELINE_ROWS.csv')))
    print('AVAILABILITY: Raw 16/16; P100 V1 8/8; Full V1 checkpoint 8/8; new full reproduction pending')

if __name__=='__main__':main()
