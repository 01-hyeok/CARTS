"""CPU-only final artifact gate, including the old four cells' deferred export."""
import csv,json,os,fcntl,time,traceback,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contracts import *
def prerequisites():
    missing=[];gates=[]
    for c,s in REUSE:
        folder=OLD/(c+'_'+s);validate_reuse(folder)
        p=folder/'PAIR_EXPORT_GATE.json'
        if not p.exists():missing.append(str(p));continue
        g=json.loads(p.read_text());assert g['status']=='PASS' and g['all_15_head_arm_retmse_exact']
        assert sha(folder/'HEAD_PAIR_OVERLAP.csv')==g['csv_sha256']
        gates.append(dict(cell=c,support=s,reused=True,path=str(folder/'FINAL_GATE.json'),sha256=sha(folder/'FINAL_GATE.json'),pair_gate_sha256=sha(p)))
    for c,s in NEW:
        p=OUT/(c+'_'+s)/'FINAL_GATE.json';g=checked_gate(p,'PASS_EXPANSION_CELL_COMPLETE')
        if not g:missing.append(str(p));continue
        assert g['selected_test_ids_metrics_exact'] and g['batch_rng_identical'] and g['channels']==(7 if c.startswith('ETTh1') else 21)
        gates.append(dict(cell=c,support=s,reused=False,path=str(p),sha256=sha(p)))
    root=OLD/'DELIVERABLE_COMPLETENESS_GATE.json'
    if not root.exists():missing.append(str(root))
    else:
        g=json.loads(root.read_text());assert g['status']=='PASS_REQUIRED_ANALYSES_COMPLETE_WITH_TIMING_LIMITATIONS'
        assert sha(OLD/'HEAD_PAIR_OVERLAP.csv')==g['head_pair_file_sha256']
    progress=OUT/'PROGRESS.json'
    p=json.loads(progress.read_text()) if progress.exists() else {}
    if p.get('complete')!=16 or p.get('failed'):missing.append(str(progress)+' : successful 16-cell summary pending')
    return missing,gates
def combine(name):
    rows=[];sources=[]
    for c,s in JOBS:
        folder=(OLD if (c,s) in REUSE else OUT)/(c+'_'+s);p=folder/name
        assert p.exists(),p
        with p.open() as f:
            for raw in csv.DictReader(f):
                r=dict(raw);r.update(cell=c,support=s,reused=(c,s) in REUSE,source=str(p));rows.append(r)
        sources.append(dict(path=str(p),sha256=sha(p)))
    cols=list(dict.fromkeys(k for r in rows for k in r))
    dest=OUT/'final_bundle'/name
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    return dict(rows=len(rows),sources=sources,sha256=sha(dest))
def main():
    out=OUT/'final_bundle';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if checked_gate(out/'FINAL_GATE.json','PASS_ALL_16_SOFTLAG_DELIVERABLES'):return
    while True:
        missing,gates=prerequisites()
        if not missing:break
        dump(out/'CURRENT.json',dict(status='WAITING_ACTUAL_CELL_AND_REUSED_EXPORT_GATES',at=now(),pid=os.getpid(),missing=missing,gpu_used=False));time.sleep(20)
    inventory={}
    for n in ['TEST_RESULTS.csv','VALIDATION_RESULTS.csv','HEAD_USEFULNESS.csv','HEAD_PAIR_OVERLAP.csv','BIAS_SIMILARITY.csv','PAIRED_TIME_BLOCK_CI.csv','LEARNED_LAG_PREFERENCES.csv']:
        inventory[n]=combine(n)
    assert inventory['TEST_RESULTS.csv']['rows']==96 and inventory['HEAD_PAIR_OVERLAP.csv']['rows']==480
    assert inventory['HEAD_USEFULNESS.csv']['rows']==240 and inventory['VALIDATION_RESULTS.csv']['rows']==192
    summary=json.loads((OUT/'PROGRESS.json').read_text());assert summary['complete']==16 and not summary['failed']
    content=(Path(summary['summary'])/'REPORT_KO.md').read_text(encoding='utf8')
    content+='\n\n기존 4조건의 보류 head-pair export까지 확인한 최종 산출물 gate이다. 16조건 TEST_RESULTS 96행, head-pair 480행, head usefulness 240행, validation history 192행의 원형 CSV를 병합했다. 원본 숫자 문자열을 유지하고 source 경로와 hash를 보존했다. 학습·재현 완료와 최종 산출물 완료를 분리했다.\n'
    (out/'FINAL_REPORT_KO.md').write_text(content,encoding='utf8')
    dump(out/'SOURCE_INVENTORY.json',inventory)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['FINAL_GATE.json','CURRENT.json','finalizer.lock']}
    dump(out/'FINAL_GATE.json',dict(status='PASS_ALL_16_SOFTLAG_DELIVERABLES',at=now(),conditions=16,reused=4,new=12,source_cell_gates=gates,files=files,verdict='INCONCLUSIVE',seed=0,full_population=False,solar_resumed=False))
    dump(out/'CURRENT.json',dict(status='COMPLETE',at=now(),pid=os.getpid(),gpu_used=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(OUT/'final_bundle/CURRENT.json',dict(status='BLOCKED',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
