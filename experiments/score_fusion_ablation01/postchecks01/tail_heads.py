import sys,json,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'recovery01'))
import run as recovery
ev=recovery.ev
@torch.no_grad()
def main(cell):
    done=json.loads((ev.OUT/'checkpoints'/cell/'V5_FULL/COMPLETE.json').read_text())
    if all(n%32!=1 for n in done['queries'].values()):
        ev.dump(ev.OUT/'postchecks01'/(cell+'_HEAD_GATE.json'),dict(status='NOT_APPLICABLE_NO_B1_QUERY',queries=done['queries'],tail_sizes={s:n%32 for s,n in done['queries'].items()},at=ev.now(),reason='Recovery01 only changes B1; actual B17 tails retained original exact score path'))
        print(cell,'NOT_APPLICABLE_NO_B1_QUERY',flush=True);return
    d,f,prior,identity,before=recovery.setup(cell,'FULL');rows=[]
    for split in ('val','test'):
        n=len(d.datasets[split]);a=n-1;variants,_=ev.batch_logits(f,split,a,n);raw=prior.batch(split,a,n)
        for support,(g,m,ids) in variants.items():
            current=ids[:,:,None].expand(-1,-1,5,-1).gather(-1,ev.topk(g.masked_fill(~m[:,:,None],-float('inf')),10))
            vn=ev.variant_name('FULL',support);old=ev.torch.as_tensor(raw[vn+'__head_ids'],device=d.device)
            oracle=ev.torch.as_tensor(raw[vn+'__oracle_ids'],device=d.device)
            diff=int((current!=old).sum());setdiff=int((current.sort(-1).values!=old.sort(-1).values).sum())
            membership=(current[:,:,:,None,:]==oracle[:,:,None,:,None]).any(-1)
            previous=(old[:,:,:,None,:]==oracle[:,:,None,:,None]).any(-1)
            mdiff=int((membership!=previous).sum())
            rows.append(dict(split=split,support=support,query_index=a,head_ordered_id_differences=diff,head_set_differences=setdiff,oracle_head_membership_differences=mdiff))
            ev.save(ev.OUT/'postchecks01'/(cell+'_'+split+'_'+support+'_HEADS.npz'),dict(current_head_ids=current.cpu().numpy(),prior_head_ids=old.cpu().numpy(),oracle_ids=oracle.cpu().numpy(),query_start=raw['query_start']))
            if mdiff:
                ev.dump(ev.OUT/'postchecks01'/(cell+'_HEAD_GATE.json'),dict(status='FAIL_RETENTION_MEMBERSHIP_CHANGED',rows=rows,at=ev.now()))
                raise AssertionError(rows[-1])
    ev.dump(ev.OUT/'postchecks01'/(cell+'_HEAD_GATE.json'),dict(status='PASS_USEFUL_EXCLUSIVE_HEAD_MEMBERSHIP_UNCHANGED',rows=rows,at=ev.now(),definition='Retention depends on Oracle membership in each head, not within-head order'))
    print(cell,rows,flush=True)
if __name__=='__main__':main(sys.argv[1])
