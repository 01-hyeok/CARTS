"""Disclose the crashed worker's unavailable CUDA allocator peak."""
import csv,json
from pathlib import Path
NOTE='Recovery disclosure: ETTh1_96 FULL finished training before the legacy argsort API failure. Its allocator peak counters were lost with that process. The recorded allocated/reserved peaks for this one native cell describe recovery evaluation only, not its original training peak. Original training nvidia-smi samples remain in RESOURCE_SAMPLES.jsonl. Elapsed time combines the last recorded training-history interval with recovery evaluation and excludes downtime; raw encoding time for this cell is recovery-only.'
def annotate(folder):
    from experiment_common import sha,dump
    p=folder/'EFFICIENCY.csv'
    if p.exists():
        with p.open() as f:rows=list(csv.DictReader(f))
        for r in rows:
            recovered=r.get('cell')=='ETTh1_96' and r.get('support',r.get('training_support'))=='FULL' and r.get('comparison','native')=='native'
            r['peak_memory_scope']='recovery evaluation only; original training peak unavailable' if recovered else 'original worker measured allocation/reservation'
        fields=sorted({k for r in rows for k in r})
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    p=folder/'EFFICIENCY_REPORT.md'
    if p.exists() and NOTE not in p.read_text():p.write_text(p.read_text()+'\n\n'+NOTE+'\n')
    p=folder/'COMPLETE.json'
    if p.exists() and folder.name=='final_all_comparisons':
        m=json.loads(p.read_text());m['recovery_timing_disclosure']=NOTE
        m['files']={f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name!='COMPLETE.json'};dump(p,m)
