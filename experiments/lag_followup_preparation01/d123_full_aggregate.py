"""CPU aggregation for all-query D1/D2/D3, with explicit native/control labels."""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from fusion import analyze_unit
from d56_aggregate import sha,dump,csvout,now
OUT=Path('/data/pjh_workspace/CARTS/results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/full_population_completion')

def spearman_rows(a,b):
    a,b=rankdata(a,axis=-1),rankdata(b,axis=-1)
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1));result=np.full(den.shape,np.nan)
    np.divide((a*b).sum(-1),den,out=result,where=den>0);return result

def worst_share(x,fraction):
    x=np.asarray(x).mean(-1);positive=np.maximum(x,0);den=positive.sum()
    return float(np.sort(positive)[-max(1,int(np.ceil(len(x)*fraction))):].sum()/den) if den>0 else None

def main(cell,training):
    src=OUT/'raw'/cell/('V5_'+training);g=json.loads((src/'GATE.json').read_text());assert g['status']=='PASS_FULL_QUERY_D123_RAW'
    dest=OUT/'analysis'/cell/('V5_'+training);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DELIVERABLE_GATE.json').exists():
        old=json.loads((dest/'DELIVERABLE_GATE.json').read_text());assert old['source_gate_sha256']==sha(src/'GATE.json')
        for name,h in old['files'].items():assert sha(dest/name)==h
        return
    start=time.perf_counter();tags=dict(cell=cell,model='V5',training_support=training,checkpoint_sha256=g['identity']['checkpoint']['checkpoint_sha256'],coverage='all validation/test query timestamps and all channels')
    C=g['channels'];stats={};d3rows=[];massrows=[];counts={};neg=0
    variants=['FULL','P100_CONTROLLED']+(['P100_NATIVE'] if training=='P100' else [])
    names=['D1_D2_QUERY_CHANNEL','D2_LOST_GOOD_CANDIDATES','D3_QUERY_CHANNEL'];streams={}
    for name in names:streams[name]=gzip.open(dest/(name+'.jsonl.gz.tmp'),'wt')
    try:
        for split in ['val','test']:
            count=0;d3all=[]
            for p in sorted((src/split).glob('batch*.npz')):
                assert sha(p)==g['files'][str(p.relative_to(src))]
                with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                B=len(d['query_start']);count+=B
                for variant in variants:
                    a={k.split('__',1)[1]:v for k,v in d.items() if k.startswith(variant+'__')}
                    for q in range(B):
                        for c in range(C):
                            unit=dict(**tags,split=split,variant=variant,evaluation_support='FULL' if variant=='FULL' else 'P100',query_start=int(d['query_start'][q]),channel=c)
                            r=analyze_unit(a['head_ids'][q,c],a['head_mse'][q,c].astype(float),a['selected_ids'][q,c],a['selected_mse'][q,c].astype(float),a['oracle_ids'][q,c],a['oracle_mse'][q,c].astype(float))
                            assert r['fusion_regret']>=-1e-12 and r['discovery_regret']>=-1e-12
                            r['missing_fields']=[]
                            loom=a['loo_mse'][q,c].astype(float).mean(-1);r['loo_delta_all_minus_without']=(r['r_mix']-loom).tolist()
                            r['loo_without_head_retmse']=loom.tolist()
                            r['pair_top10_overlap']=[len(set(a['head_ids'][q,c,h])&set(a['head_ids'][q,c,k]))/10 for h in range(5) for k in range(h+1,5)]
                            streams['D1_D2_QUERY_CHANNEL'].write(json.dumps(dict(**unit,**r),allow_nan=False)+'\n')
                            key=(split,variant,c);s=stats.setdefault(key,dict(n=0,totals={},heads=[{} for _ in range(5)],head_defined=[{} for _ in range(5)],loo=np.zeros(5),overlap=np.zeros(10)))
                            s['n']+=1
                            for k in ['r_mix','r_oracle_C','r_oracle','fusion_regret','discovery_regret','lost_oracle_count','union_size','union_oracle_hits','final_oracle_hits']:
                                s['totals'][k]=s['totals'].get(k,0)+r[k]
                            for h,hr in enumerate(r['heads']):
                                for k,v in hr.items():
                                    if k!='head' and v is not None:s['heads'][h][k]=s['heads'][h].get(k,0)+v;s['head_defined'][h][k]=s['head_defined'][h].get(k,0)+1
                            s['loo']+=r['loo_delta_all_minus_without'];s['overlap']+=r['pair_top10_overlap']
                            oracle=set(a['oracle_ids'][q,c]);mix=set(a['selected_ids'][q,c]);hid=a['head_ids'][q,c].ravel();d10=float(a['oracle_mse'][q,c].max())
                            for gid in sorted(set(hid)-mix):
                                j=int(np.flatnonzero(hid==gid)[0]);dm=float(a['head_mse'][q,c].ravel()[j]);hit=gid in oracle
                                if not hit and dm>d10+max(1e-8,.1*d10):continue
                                streams['D2_LOST_GOOD_CANDIDATES'].write(json.dumps(dict(**unit,candidate_id=int(gid),oracle10=hit,
                                    near1=dm<=d10+max(1e-8,.01*d10),near5=dm<=d10+max(1e-8,.05*d10),near10=dm<=d10+max(1e-8,.1*d10),future_mse=dm,
                                    mixture_rank=int(a['head_candidate_mixture_rank'][q,c].ravel()[j]),mixture_score=float(a['head_candidate_mixture_score'][q,c].ravel()[j]),
                                    all_head_logits=a['head_candidate_all_head_logits'][q,c,:,j].tolist(),all_head_probabilities=a['head_candidate_all_head_probabilities'][q,c,:,j].tolist()),allow_nan=False)+'\n')
                ra=d['D3_A_mse'].mean(-1);rb=d['D3_B_mse'].mean(-1);rc=d['D3_C_mse'].mean(-1);norm=rb-ra;support=rc-rb;total=rc-ra
                np.testing.assert_allclose(total,norm+support,atol=1e-12,rtol=1e-12)
                rho=spearman_rows(d['D3_local_scores'],d['D3_restricted_scores']);overlap=(d['D3_A_ids'][...,None]==d['D3_B_ids'][...,None,:]).any(-1).mean(-1)
                for q in range(B):
                    vals=[]
                    for c in range(C):
                        r=dict(**tags,split=split,query_start=int(d['query_start'][q]),channel=c,RA=float(ra[q,c]),RB=float(rb[q,c]),RC=float(rc[q,c]),
                            normalization_only_delta=float(norm[q,c]),support_after_normalization_delta=float(support[q,c]),total_delta=float(total[q,c]),
                            A_B_top10_overlap=float(overlap[q,c]),A_B_common_P100_spearman=float(rho[q,c]) if np.isfinite(rho[q,c]) else None,
                            head_mass=d['D3_head_mass'][q,c].tolist(),head_mass_variance=float(d['D3_head_mass'][q,c].var()),implicit_head_weight=d['D3_implicit_weights'][q,c].tolist(),
                            logZ_full=d['D3_logZ_full'][q,c].tolist(),logZ_pool=d['D3_logZ_pool'][q,c].tolist())
                        streams['D3_QUERY_CHANNEL'].write(json.dumps(r,allow_nan=False)+'\n');vals.append(r)
                    d3all.append(vals)
            assert count==g['queries'][split];counts[split]=count
            flat=[r for q in d3all for r in q];row=dict(**tags,split=split,queries=count,channels=C)
            for k in ['RA','RB','RC','normalization_only_delta','support_after_normalization_delta','total_delta','A_B_top10_overlap','A_B_common_P100_spearman','head_mass_variance']:
                v=[r[k] for r in flat if r[k] is not None];row[k]=float(np.mean(v)) if v else None;row[k+'_defined_units']=len(v)
            for key in ['normalization_only_delta','support_after_normalization_delta','total_delta']:
                v=np.array([[r[key] for r in q] for q in d3all]);row[key+'_worst1pct_positive_share']=worst_share(v,.01);row[key+'_worst5pct_positive_share']=worst_share(v,.05)
                row[key+'_qc_improvement_rate']=float((v<0).mean());row[key+'_qc_deterioration_rate']=float((v>0).mean())
            d3rows.append(row)
            for h in range(5):massrows.append(dict(**tags,split=split,head=h+1,head_mass_mean=float(np.mean([r['head_mass'][h] for r in flat])),head_mass_std=float(np.std([r['head_mass'][h] for r in flat])),implicit_weight_mean=float(np.mean([r['implicit_head_weight'][h] for r in flat]))))
    finally:
        for stream in streams.values():stream.close()
    for name in names:(dest/(name+'.jsonl.gz.tmp')).replace(dest/(name+'.jsonl.gz'))
    result=[];heads=[];loo=[];diversity=[]
    for (split,variant,c),s in stats.items():
        unit=dict(**tags,split=split,variant=variant,channel=c,queries=s['n'])
        result.append(dict(**unit,**{k:v/s['n'] for k,v in s['totals'].items()}))
        for h in range(5):
            heads.append(dict(**unit,head=h+1,**{k:v/s['head_defined'][h][k] for k,v in s['heads'][h].items()},defined_counts=json.dumps(s['head_defined'][h])))
            loo.append(dict(**unit,removed_head=h+1,delta_all_minus_without=float(s['loo'][h]/s['n']),positive_means='removal improves; inference only'))
        for pair,v in enumerate(s['overlap']):diversity.append(dict(**unit,pair=pair,pair_top10_overlap=float(v/s['n']),union_size=s['totals']['union_size']/s['n'],normalized_union_size=s['totals']['union_size']/s['n']/50))
    for name,rows in [('D1_HEAD_UNIQUE_USEFUL_CANDIDATES',heads),('D1_HEAD_RELEVANCE_CONTRIBUTION',heads),('D1_HEAD_DIVERSITY',diversity),('D2_FUSION_DISCOVERY_REGRET',result),('D2_HEAD_LEAVE_ONE_OUT',loo),('D3_NORMALIZER_ONLY_RESULTS',d3rows),('D3_HEAD_PROBABILITY_MASS',massrows)]:csvout(dest/(name+'.csv'),rows)
    dump(dest/'DELIVERABLE_GATE.json',dict(status='PASS_FULL_QUERY_D1_D2_D3',at=now(),identity=tags,source_gate_sha256=sha(src/'GATE.json'),queries=counts,channels=C,variants=variants,
        lost_candidate_rank_probability_complete=True,full_query_coverage=True,negative_regret_clamp=False,
        normalizer_precision='A/B FP64 from canonical FP32 logits/tau, C canonical FP32 Full mixture; matches prior Mechanism02 path; not causal attribution',
        seconds=time.perf_counter()-start,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='DELIVERABLE_GATE.json'}))
    print('FULL D123 DELIVERABLE PASS',cell,training,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('training');a=p.parse_args();main(a.cell,a.training)
