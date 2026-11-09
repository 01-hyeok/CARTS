"""Expensive correlations only: preregistered shared timestamps/all channels."""
import numpy as np
from scipy.stats import spearmanr,kendalltau

def rows(n):return np.unique(np.linspace(0,n-1,min(n,256)).round().astype(np.int64))

def analyze(score,d,mask,query_ids):
    result=[]
    for q in range(len(score)):
        for c in range(score.shape[1]):
            valid=np.flatnonzero(mask[q,c]);s=score[q,c,valid];r=-d[q,c,valid]
            so=np.argsort(-s,kind='stable');oracle=np.argsort(-r,kind='stable')
            for name,subset in [('ALL_VALID',np.arange(len(s))),('STUDENT_TOP50',so[:50]),('STUDENT_TOP100',so[:100]),('ORACLE_TOP100',oracle[:100])]:
                ss=s[subset];rr=r[subset]
                if np.ptp(ss)==0 or np.ptp(rr)==0:rho=tau=np.nan
                else:rho=float(spearmanr(ss,rr)[0]);tau=float(kendalltau(ss,rr,variant='b')[0])
                result.append(dict(query_id=int(query_ids[q]),channel=c,subset=name,candidates=len(subset),spearman=rho,kendall_tau_b=tau,
                                   exact_within_sample=True,timestamp_sampling='fixed linspace max256/allchannels'))
    return result
