"""Full-support score assembly. Normalize once across every valid candidate."""
import torch

def stable_topk(score,k=10):
    return torch.sort(score,dim=-1,descending=True,stable=True).indices[...,:k]

def assemble(count,chunk,score_chunk):
    if count<1 or chunk<1:raise ValueError('Invalid candidate count/chunk')
    return torch.cat([score_chunk(a,min(a+chunk,count)) for a in range(0,count,chunk)],dim=-1)

def global_logprob(scores,mask,tau=.1):
    if tau<=0 or scores.shape!=mask.shape or not mask.any(-1).all():raise ValueError('Invalid global support')
    return (scores/tau).masked_fill(~mask,-torch.inf).log_softmax(-1)

def future_teacher(distance,mask,tau=.1):
    """Kept out of scorer: canonical implementation is injected by callers."""
    raise NotImplementedError('Use repository normalized_teacher_prob; never guess zscore variance convention')

def candidate_coverage(mask,chunks):
    positions=[torch.arange(a,b,device=mask.device) for a,b in chunks]
    actual=torch.cat(positions)
    expected=torch.arange(mask.shape[-1],device=mask.device)
    if not torch.equal(actual,expected):raise ValueError('Full bank was reduced, reordered or duplicated')
    return int(mask.sum())
