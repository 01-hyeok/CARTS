"""Aggregate only completed native-support cells; paired timestamp analysis."""
import csv,json,math
import numpy as np
from experiment_common import *

def csv_write(path,rows):
    keys=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def paired_ci(delta,h):
    n=len(delta);rng=np.random.RandomState(0);means=[]
    for _ in range(1000):
        starts=rng.randint(0,n,size=math.ceil(n/h))
        ids=((starts[:,None]+np.arange(h))%n).reshape(-1)[:n]
        means.append(delta[ids].mean())
    lo,hi=np.quantile(means,[.025,.975])
    return dict(mean_delta=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi),
                query_improved_fraction=float((delta<0).mean()),query_worse_fraction=float((delta>0).mean()),
                query_tie_fraction=float((delta==0).mean()),bootstrap_unit='timestamp after averaging channels',
                bootstrap='circular moving blocks; 1000 seed0 replicates',block_length=h)

def summarize():
    rows=[];history=[];diagnostics=[];paired=[];eff=[];completed=[]
    for support in ('P100','FULL'):
        for cell in CELLS:
            dest=OUT/support/cell
            if not (dest/'COMPLETE.json').exists():continue
            done=json.loads((dest/'COMPLETE.json').read_text());completed.append(support+'/'+cell)
            current=json.loads((dest/'FULL_RESULTS.json').read_text());rows.extend(current)
            hist=json.loads((dest/'TRAIN_HISTORY.json').read_text())
            history.extend(dict(cell=cell,support=support,seed=0,**r) for r in hist)
            pars=json.loads((dest/'LEARNED_PARAMETERS.json').read_text())
            for name,p in pars.items():diagnostics.append(dict(cell=cell,support=support,arm=name,**p))
            paired_path=dest/'PAIRED_TIMESTAMP_ANALYSIS.json'
            if paired_path.exists():pr=json.loads(paired_path.read_text())
            else:
                delta={n:[] for n in ['D0','C0']+ARMS};recovery={n:[] for n in delta};loss={n:[] for n in delta}
                for f in sorted((dest/'final_raw/test').glob('*.npz')):
                    meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256']
                    with np.load(f) as z:
                        for name in delta:
                            delta[name].append(z[name+'__delta_retmse_vs_s0'].mean(-1))
                            recovery[name].append(z[name+'__recovered_missed_oracle'].mean(-1))
                            loss[name].append(z[name+'__lost_baseline_oracle'].mean(-1))
                pr=[dict(cell=cell,support=support,method=n,**paired_ci(np.concatenate(v),int(cell.rsplit('_',1)[1])),
                    mean_recovered_oracle=float(np.concatenate(recovery[n]).mean()),mean_lost_oracle=float(np.concatenate(loss[n]).mean())) for n,v in delta.items()]
                dump(paired_path,pr)
            paired.extend(pr)
            eff.append(dict(cell=cell,support=support,elapsed_seconds=done['elapsed_seconds'],peak_allocated_GiB=done['peak_gpu_allocated_bytes']/2**30,
                peak_reserved_GiB=done['peak_gpu_reserved_bytes']/2**30,raw_encoding_distance_seconds=done['raw_encoding_distance_seconds'],
                feature_bytes=done['identity']['feature_manifest']['bytes'],feature_seconds=done['identity']['feature_manifest']['seconds']))
    csv_write(OUT/'FULL_RESULTS.csv',rows);csv_write(OUT/'TRAIN_HISTORY.csv',history)
    csv_write(OUT/'SCORE_DIAGNOSTICS.csv',diagnostics);csv_write(OUT/'MISSED_ORACLE_ANALYSIS.csv',paired)
    csv_write(OUT/'EFFICIENCY.csv',eff)
    complete=len(completed)==16
    lines=['# Learned Dynamics / Cross-Predictability',f'Status: {"COMPLETE" if complete else "IN PROGRESS"}; {len(completed)}/16 native-support conditions.',
        'Seed 0 only. All Train/Validation/Test timestamps and channels; six separately trained scorer arms. Full is separately trained, not support transfer.',
        'Original offline bidirectional non-overlap RAFT Train mask and immutable P100 pools. Val/Test train-bank futures are known by forecast origin.',
        'Frozen V5 + log-mixture fusion; Train-only feature and score calibration; validation RetMSE selection; final test after selection.',
        'Calibration uses 256 Train timestamps; formal optimization/evaluation use all queries. No causal-mask sensitivity or Solar run.',
        '', '| Condition | Arm | Test RetMSE | Test AggMSE | Delta RetMSE vs S0 |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['split']=='test':lines.append(f"| {r['dataset']}_{r['horizon']} {r['support']} | {r['method']} | {r['retmse']:.8f} | {r['agg_mse']:.8f} | {r.get('delta_retmse_vs_s0',0):.8f} |")
    lines+=['','Confidence intervals use temporal blocks of length H over timestamp channel-means. Channels are not independent repeats.',
        'Oracle10 selects individually lowest FutureMSE candidates; it is not an optimal ensemble oracle.',
        'Prior V5 checkpoint-generation runtime remains UNKNOWN; current numerical reproduction is recorded separately.',
        'Cross-support inference, PAA/SARAF additional pools, and Phase B are not represented by these native-support rows.']
    (OUT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    (OUT/'EFFICIENCY_REPORT.md').write_text('# Efficiency\n'+f'{len(completed)}/16 completed conditions. See EFFICIENCY.csv for feature-cache size/time, total worker time, frozen-score computation time and peak CUDA memory.\n'
        +'Feature/scalar preparation and final ranking costs are included; original V5 pretraining costs are excluded. Query counts and channels are in FULL_RESULTS.csv.\n')
    if complete:
        analysis=['# Final research review','Observed results are in FULL_RESULTS.csv; interpretations below are limited to seed0 and these datasets/supports.']
        tests=[r for r in rows if r['split']=='test'];by={(r['dataset'],r['horizon'],r['support'],r['method']):r for r in tests}
        def compare(a,b,key='retmse'):
            ds=[by[(d,h,s,a)][key]-by[(d,h,s,b)][key] for d in ('ETTh1','Weather') for h in (96,192,336,720) for s in ('P100','FULL')]
            return f'{sum(x<0 for x in ds)}/16 conditions lower; mean absolute delta {np.mean(ds):.8f}'
        questions=[('Dynamics versus cosine',compare('D1_L2','S0')+' (L2); '+compare('D1_MLP','S0')+' (MLP)'),
            ('Cross versus Dynamics',compare('C1_MLP','D1_MLP')),
            ('Learning versus untrained scores',compare('D1_MLP','D0')+'; Cross '+compare('C1_MLP','C0')),
            ('Replacement versus fusion',compare('D2_MLP','D1_MLP')+'; Cross '+compare('C2_MLP','C1_MLP')),
            ('Where gains occur','See condition-level table and paired timestamp CIs; do not infer significance from a sign alone.'),
            ('Recovery of V5-missed Oracle candidates','MISSED_ORACLE_ANALYSIS.csv separates recovered and lost Oracle IDs; this is direct retrieval evidence, not causal mechanism proof.'),
            ('Transfer to aggregate error','Dynamics fusion AggMSE '+compare('D2_MLP','S0','agg_mse')+'; Cross fusion '+compare('C2_MLP','S0','agg_mse')),
            ('Full support and efficiency','Full rows were separately trained. Compare with P100 using fixed native checkpoints as labeled; base checkpoints differ, so this is not a pure support causal effect.'),
            ('Paper-level evidence','Single seed cannot establish robustness. Prefer a scorer only if retrieval, aggregate and Oracle-recovery evidence agree across conditions and temporal CIs.'),
            ('Next development decision','Do not automatically start Phase B. Review per-condition results, optimization histories and cost first; raw-only replacement arms cannot train the encoder without changing inputs.')]
        for i,(q,a) in enumerate(questions,1):analysis+=['',f'{i}. {q}: {a}']
        analysis+=['','Unverified: multiple-seed robustness, out-of-dataset generalization, and end-to-end Stage2 improvements.']
        (OUT/'FINAL_RESEARCH_REVIEW.md').write_text('\n'.join(analysis)+'\n')
    dump(OUT/'SUMMARY_STATUS.json',dict(status='COMPLETE' if complete else 'IN_PROGRESS',completed=completed,at=now()))

if __name__=='__main__':summarize()
