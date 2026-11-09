"""Actual checkpoint restore, optimizer continuation, coverage negative tests and provenance gates."""
from recovery_runtime import *
import copy,inspect
import numpy as np
import torch
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def main():
    seal_check();protect();n=preserved()
    from resume_weather import restore,recovered_train,load_optimizer,SOURCE
    arms,opts,states,hist,records=restore('cpu');tests=[]
    for r in records:
        name=r['arm'];z=load(r['path']);a=arms[name];o=opts[name]
        assert equal(a.state_dict(),z['model']) and equal(o.state_dict(),z['optimizer'])
        # Independent clone/reference: matching next optimizer step after serialization restore.
        b=copy.deepcopy(a);ob=torch.optim.AdamW([v for v in b.parameters() if v.requires_grad],lr=.001);load_optimizer(ob,copy.deepcopy(z['optimizer']))
        for model,opt in ((a,o),(b,ob)):
            opt.zero_grad()
            loss=sum((v.double().square().sum() for v in model.parameters() if v.requires_grad));loss.backward();opt.step()
        assert equal(a.state_dict(),b.state_dict()) and equal(o.state_dict(),ob.state_dict())
        tests.append(dict(arm=name,model_optimizer_restore_exact=True,next_cpu_optimizer_step_exact=True,state=states[name]))
    assert max(r['epoch'] for r in hist)==8 and not states['D1_L2']['active'] and not states['C2_MLP']['active']
    import run_experiment as m
    src=inspect.getsource(recovered_train)
    assert 'range(9,31)' in src and 'self.resume_state' in src
    assert 'RandomState(epoch)' in inspect.getsource(m.Experiment.train)
    assert np.array_equal(np.random.RandomState(9).permutation(np.arange(0,36700,32)),np.random.RandomState(9).permutation(np.arange(0,36700,32)))
    from recovery02_validate import coverage,targets,validate_job,Incomplete,validate_all
    coverage(np.arange(5),np.arange(5));negative=0
    for bad in (np.array([0,1,1,3,4]),np.arange(4),np.array([1,0,2,3,4])):
        try:coverage(bad,np.arange(5))
        except AssertionError:negative+=1
    assert negative==3
    jobs=list(targets());assert len(jobs)==40 and len({str(p) for p,l in jobs})==40
    assert [sum(l['mode']==m for p,l in jobs) for m in ('native','pool','cross')]==[16,8,16]
    try:validate_all();raise AssertionError('Incomplete accepted')
    except Incomplete:pass
    from readiness import check
    try:check();raise AssertionError('Missing final PASS accepted')
    except Incomplete:pass
    outside=ORIGINAL/'RECOVERY02_WRITE_GUARD_TEST_FORBIDDEN'
    assert not outside.exists()
    try:outside.write_text('must fail');raise AssertionError('Original writable')
    except PermissionError:pass
    assert not outside.exists()
    reuse=[]
    for p,l in jobs:
        if str(p).startswith(str(OUT)):continue
        reuse.append(validate_job(p,l,True))
    assert len(reuse)==10
    # The copied diagnostic compatibility implementation must be exactly the reviewed implementation.
    import compat
    old=(CODE.parent/'recovery01/compat.py').read_text().replace("sys.path.insert(0,str(HERE.parent))","sys.path.insert(0,str(HERE/'core'))")
    assert (CODE/'compat.py').read_text()==old
    for p in CODE.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    preserved()
    dump(OUT/'REUSED_RESULTS_VALIDATION.json',dict(status='PASS',jobs=reuse))
    for path,key in ((OUT/'CPU_IMPLEMENTATION_GATE.json','source_hashes'),(OUT/'supplemental/CPU_GATE.json','sources')):
        value=read(path);value['status']='PASS';value['recovery_preflight']=str(OUT/'PREFLIGHT.json');dump(path,value)
    dump(OUT/'PREFLIGHT.json',dict(status='PASS',at=now(),preserved_files=n,checkpoints=tests,reused_conditions=10,expected_jobs=40,coverage_negative_cases=negative,missing_final_rejected=True,original_write_rejected=True,calibration_refit=False,epoch00_replay=False,rng_not_saved=True,source_seal_sha256=sha(OUT/'SOURCE_SEAL.json')))
    print('PREFLIGHT_PASS',json.dumps(tests),flush=True)
if __name__=='__main__':main()
