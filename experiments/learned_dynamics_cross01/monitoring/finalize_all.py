"""Final request-wide report, separate from immutable native/supplement outputs."""
import sys,json,csv,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import experiment_common as cm
from experiment_common import OUT,CELLS,ARMS,sha,dump,now
from summarize import paired_ci
import numpy as np

def csv_write(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def targets():
    for s in ('P100','FULL'):
        for c in CELLS:yield OUT/s/c,dict(comparison='native',cell=c,training_support=s,evaluation_support=s,pool='original')
    for c in ('ETTh1_96','ETTh1_720','Weather_96','Weather_720'):
        for p in ('paa16_cosine','saraf_time_aligned'):yield OUT/'supplemental/pool'/p/c,dict(comparison='pool',cell=c,training_support='P100',evaluation_support='P100',pool=p)
    for c in CELLS:
        for s in ('P100','FULL'):
            target='FULL' if s=='P100' else 'P100'
            yield OUT/'supplemental/cross'/('V5_'+s+'_to_'+target)/c,dict(comparison='cross',cell=c,training_support=s,evaluation_support=target,pool='original')
def main():
    jobs=list(targets());assert len(jobs)==40
    assert all((p/'COMPLETE.json').exists() for p,_ in jobs),'All 40 jobs must finish'
    dest=OUT/'final_all_comparisons';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    rows=[];history=[];eff=[];diag=[];paired=[];source_manifests={}
    for path,labels in jobs:
        done=json.loads((path/'COMPLETE.json').read_text());source_manifests[str(path.relative_to(OUT))]=sha(path/'COMPLETE.json')
        rows.extend(dict(**r,**labels) for r in json.loads((path/'FULL_RESULTS.json').read_text()))
        if (path/'TRAIN_HISTORY.json').exists():history.extend(dict(**r,**labels) for r in json.loads((path/'TRAIN_HISTORY.json').read_text()))
        parameters=json.loads((path/'LEARNED_PARAMETERS.json').read_text())
        diag.extend(dict(arm=k,**v,**labels) for k,v in parameters.items())
        eff.append(dict(**labels,worker_elapsed_seconds=done['elapsed_seconds'],peak_allocated_GiB=done['peak_gpu_allocated_bytes']/2**30,
            peak_reserved_GiB=done['peak_gpu_reserved_bytes']/2**30,timing_definition='worker timed interval, includes CPU work while CUDA context resident; not kernel-only GPU time'))
        per={m:[] for m in ['D0','C0']+ARMS};recovered={m:[] for m in per};lost={m:[] for m in per}
        for block in sorted((path/'final_raw/test').glob('*.npz')):
            meta=json.loads(block.with_suffix('.json').read_text());assert sha(block)==meta['sha256']
            with np.load(block) as z:
                for m in per:
                    per[m].append(z[m+'__delta_retmse_vs_s0'].mean(-1));recovered[m].append(z[m+'__recovered_missed_oracle'].mean(-1));lost[m].append(z[m+'__lost_baseline_oracle'].mean(-1))
        for m,parts in per.items():
            paired.append(dict(**labels,method=m,**paired_ci(np.concatenate(parts),int(labels['cell'].rsplit('_',1)[1])),
                mean_oracle_recovered=float(np.concatenate(recovered[m]).mean()),mean_oracle_lost=float(np.concatenate(lost[m]).mean())))
    support_rows=[]
    for cell in CELLS:
        for source in ('P100','FULL'):
            other='FULL' if source=='P100' else 'P100'
            native=OUT/source/cell/'final_raw/test';cross=OUT/'supplemental/cross'/('V5_'+source+'_to_'+other)/cell/'final_raw/test'
            full,p100=(cross,native) if source=='P100' else (native,cross)
            diffs={m:[] for m in ['S0','D0','C0']+ARMS};outside=[];changed={m:[] for m in diffs}
            pool=np.load(cm.ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/pools'/cell/'test.npy',mmap_mode='r')
            offset=0
            for fullfile in sorted(full.glob('*.npz')):
                with np.load(fullfile) as f,np.load(p100/fullfile.name) as p:
                    assert np.array_equal(f['query_start'],p['query_start'])
                    n=len(f['query_start']);pp=pool[offset:offset+n]
                    oracle=f['Oracle10__selected_ids'];outside.append((~(oracle[...,None]==pp[...,None,:]).any(-1)).mean((1,2)))
                    for m in diffs:
                        diffs[m].append((f[m+'__retmse']-p[m+'__retmse']).mean(-1))
                        changed[m].append((~(f[m+'__selected_ids'][...,None]==p[m+'__selected_ids'][...,None,:]).any(-1)).mean((1,2)))
                    offset+=n
            assert offset==len(pool)
            for m,v in diffs.items():support_rows.append(dict(cell=cell,source_training_support=source,method=m,
                contrast='Full minus P100; SAME selected model/calibration',**paired_ci(np.concatenate(v),int(cell.rsplit('_',1)[1])),
                full_oracle_outside_p100_fraction=float(np.concatenate(outside).mean()),selected_candidate_changed_fraction=float(np.concatenate(changed[m]).mean())))
    csv_write(dest/'FULL_RESULTS.csv',rows);csv_write(dest/'TRAIN_HISTORY.csv',history)
    csv_write(dest/'SCORE_DIAGNOSTICS.csv',diag);csv_write(dest/'MISSED_ORACLE_ANALYSIS.csv',paired)
    csv_write(dest/'SUPPORT_EFFECTS.csv',support_rows);csv_write(dest/'EFFICIENCY.csv',eff)
    (dest/'EXPERIMENT_PROTOCOL.md').write_text((OUT/'EXPERIMENT_PROTOCOL.md').read_text()+'\n## Completed supplements\nEight PAA/SARAF pool conditions were separately trained on their fixed pools, using the same P100 V5 initialization. Sixteen cross-support comparisons reused native-selected model states and calibration without refitting or validation reselection. Pool provenance and calendar-only normalizers are in supplemental/POOL_REUSE_GATE.json. All 40 comparisons use complete evaluation populations; native Full models are separately trained.\n')
    test=[r for r in rows if r['split']=='test'];table=['# Results summary','All 16 native conditions, 8 alternative-pool conditions and 16 frozen cross-support comparisons completed.','',
        '| Comparison | Cell | Training→Evaluation | Pool | Method | RetMSE | AggMSE |','|---|---|---|---|---|---:|---:|']
    for r in test:table.append(f"| {r['comparison']} | {r['cell']} | {r['training_support']}→{r['evaluation_support']} | {r['pool']} | {r['method']} | {r['retmse']:.8f} | {r['agg_mse']:.8f} |")
    (dest/'RESULTS_SUMMARY.md').write_text('\n'.join(table)+'\n')
    elapsed=sum(r['worker_elapsed_seconds'] for r in eff)
    (dest/'EFFICIENCY_REPORT.md').write_text('# Efficiency\n'+f'Summed timed worker intervals: {elapsed/3600:.3f} hours across 40 serial jobs. CUDA peak allocated/reserved memory is measured separately in each process and tabulated in EFFICIENCY.csv.\n'
        +'These intervals include host computation with CUDA context residency and exclude parts of initial setup; they are not GPU kernel execution time. RESOURCE_SAMPLES.jsonl is the timestamped external GPU observation record. Feature preprocessing is CPU-only and separately timed in features/*/COMPLETE.json. Final validation/test scoring durations are in each job\'s FINAL_VAL_GATE.json and FINAL_TEST_GATE.json. Training plus interleaved validation spans TRAINING_STARTED.json to SELECTION.json; individual optimizer-kernel time was not measured.\n')
    indexed={(r['comparison'],r['cell'],r['training_support'],r['evaluation_support'],r['pool'],r['method']):r for r in test}
    native=[r for r in test if r['comparison']=='native'];nativeidx={(r['cell'],r['training_support'],r['method']):r for r in native}
    def compare(a,b,metric='retmse'):
        delta=[nativeidx[(c,s,a)][metric]-nativeidx[(c,s,b)][metric] for c in CELLS for s in ('P100','FULL')]
        return f'{sum(v<0 for v in delta)}/16 native conditions lower; mean absolute delta {np.mean(delta):.8f}'
    joint={};supported={}
    for method in ('D2_MLP','C2_MLP'):
        cells=[];strong=[]
        for c in CELLS:
            for s in ('P100','FULL'):
                row=nativeidx[(c,s,method)];base=nativeidx[(c,s,'S0')]
                evidence=next(p for p in paired if p['comparison']=='native' and p['cell']==c and p['training_support']==s and p['method']==method)
                if row['retmse']<base['retmse'] and row['agg_mse']<base['agg_mse']:
                    cells.append(c+'/'+s)
                    if evidence['ci95_high']<0 and evidence['mean_oracle_recovered']>evidence['mean_oracle_lost']:strong.append(c+'/'+s)
        joint[method]=cells;supported[method]=strong
    best=max(joint,key=lambda m:(len(supported[m]),len(joint[m])))
    other='C2_MLP' if best=='D2_MLP' else 'D2_MLP'
    advantage=(len(supported[best]),len(joint[best]))>(len(supported[other]),len(joint[other]))
    choice=(best+' has stronger exploratory follow-up evidence under joint retrieval/aggregate gain and Oracle recovery criteria.' if advantage and supported[best]
            else 'Neither family has a clear robust lead under this exploratory joint-evidence rule.')
    review=['# Final research review','Observed comparisons below use seed0 only. Paired intervals use timestamp channel-means with circular blocks H; channels are not independent repetitions.','',
        '1. Dynamics versus cosine: '+compare('D1_L2','S0')+' (L2); '+compare('D1_MLP','S0')+' (MLP).',
        '2. Cross versus Dynamics: '+compare('C1_MLP','D1_MLP')+'.',
        '3. Learned versus unlearned: Dynamics '+compare('D1_MLP','D0')+'; Cross '+compare('C1_MLP','C0')+'.',
        '4. Fusion versus replacement: Dynamics '+compare('D2_MLP','D1_MLP')+'; Cross '+compare('C2_MLP','C1_MLP')+'.',
        '5. Dataset/horizon/support specificity is shown without averaging away reversals in RESULTS_SUMMARY.md; CIs and win/loss proportions are in MISSED_ORACLE_ANALYSIS.csv.',
        '6. Actual recovered/lost Oracle candidates are measured from saved global IDs. Oracle10 is the individually lowest-error set, not an optimal ensemble.',
        '7. Aggregate transfer: Dynamics fusion AggMSE '+compare('D2_MLP','S0','agg_mse')+'; Cross fusion AggMSE '+compare('C2_MLP','S0','agg_mse')+'.',
        '8. Full feasibility is evidenced by separate native Full training and measured memory/time. SUPPORT_EFFECTS.csv isolates support changes with identical learned checkpoints; alternative-pool rows remain separately labeled.',
        '9. '+choice+' Joint RetMSE/AggMSE winning conditions: '+json.dumps(joint)+'. Joint gains with negative RetMSE CI upper bound and net Oracle recovery: '+json.dumps(supported)+'. This is an explicitly post-hoc comparison, not a preregistered significance rule or proof of mechanism; seed0 does not establish robustness.',
        '10. Continue targeted investigation only in cells with joint and Oracle-recovery evidence; if no such cells exist, stop broad expansion of these scorers and reconsider the score hypothesis. Do not automatically unfreeze the encoder or resume Solar. Raw-only replacement scorers have no encoder gradient path.',
        '', 'Interpretation limits: low future correlation suggests limited utility; failed learned-vs-raw comparisons suggest optimization/inductive-bias issues; high V5 rank correlation and overlap suggest redundancy; Full-only Oracle recovery suggests support limitation. These explanations can coexist and are not individually proven by one metric. Validation selection, calibration and timestamps are preserved in source artifacts.']
    (dest/'FINAL_RESEARCH_REVIEW.md').write_text('\n'.join(review)+'\n')
    files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
    dump(dest/'COMPLETE.json',dict(status='PASS_ALL_REQUESTED_COMPARISONS_AND_REPORTS',native=16,alternative_pools=8,cross_support=16,source_completion_hashes=source_manifests,files=files,at=now(),solar_started=False))

if __name__=='__main__':main()
