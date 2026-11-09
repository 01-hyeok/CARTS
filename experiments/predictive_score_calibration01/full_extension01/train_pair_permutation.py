"""Bijective shuffle of eligible offline-RAFT Train pairs, without targets."""
import math
import torch

class TrainPairPermutation:
    def __init__(self,count,history,device='cpu',multiplier=104729,offset=13007):
        self.count=count;self.history=history;self.device=torch.device(device)
        q=torch.arange(count,device=self.device,dtype=torch.long)
        self.left=(q-2*history+1).clamp_min(0)
        right=(count-(q+2*history)).clamp_min(0)
        self.counts=self.left+right
        if (self.counts<1).any():raise ValueError('Train row has no eligible candidates')
        self.prefix=torch.cat([torch.zeros(1,device=self.device,dtype=torch.long),self.counts.cumsum(0)])
        self.total=int(self.prefix[-1])
        while math.gcd(multiplier,self.total)!=1:multiplier+=2
        if multiplier*self.total>torch.iinfo(torch.long).max:raise ValueError('Permutation integer overflow')
        self.multiplier=multiplier;self.offset=offset%self.total
    def map(self,query_rows,candidate_ids,split,query_starts=None):
        q=torch.as_tensor(query_rows,device=self.device,dtype=torch.long)[:,None,None]
        c=candidate_ids.to(self.device)
        if split=='train':
            left=self.left[q]
            valid=(c+2*self.history<=q)|(c>=q+2*self.history)
            rank=torch.where(c+2*self.history<=q,c,left+c-(q+2*self.history))
            index=(self.prefix[q]+rank).masked_fill(~valid,0)
        elif split in ('val','test'):
            if query_starts is None:raise ValueError('Evaluation timestamps required')
            starts=torch.as_tensor(query_starts,device=self.device,dtype=torch.long)[:,None,None]
            # Fixed extension uses observed timestamp and candidate ID only.
            index=(starts*self.count+c)%self.total;valid=torch.ones_like(c,dtype=torch.bool)
        else:raise ValueError('Unknown split')
        permuted=(index*self.multiplier+self.offset)%self.total
        donor_q=torch.searchsorted(self.prefix,permuted.contiguous(),right=True)-1
        rank=permuted-self.prefix[donor_q];left=self.left[donor_q]
        donor_c=torch.where(rank<left,rank,donor_q+2*self.history+rank-left)
        assert ((donor_c+2*self.history<=donor_q)|(donor_c>=donor_q+2*self.history)).all()
        assert (donor_c>=0).all() and (donor_c<self.count).all()
        return donor_q,donor_c,valid
