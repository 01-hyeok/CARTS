import copy
import pytest
from parallel_policy import budget,validate_owner,GPU

def inputs():
    g={'results':[dict(cell='Solar_336',status='PASS',channels=137,batch=32,query_chunk=32,candidate_chunk=1000000000,peak_reserved=63472402432)]}
    return g,dict(total_mib=81920,free_mib=80000,mechanism_active=True)

def test_h336_all_channel_full_shape_shared_memory_fits():
    g,s=inputs();x=budget(g,'Solar_336',s);assert x['reserved_mechanism_mib']>14000

def test_h720_waits_while_mechanism_can_start_another_worker():
    g,s=inputs();g['results'][0].update(cell='Solar_720',peak_reserved=82908807168)
    with pytest.raises(AssertionError,match='cannot safely coexist'):budget(g,'Solar_720',s)
    s.update(mechanism_active=False,free_mib=81920);budget(g,'Solar_720',s)

def test_current_free_memory_gate():
    g,s=inputs();s['free_mib']=10000
    with pytest.raises(AssertionError,match='insufficient'):budget(g,'Solar_336',s)

@pytest.mark.parametrize('field,value',[('channels',1),('batch',16),('query_chunk',16),('candidate_chunk',100),('status','FAIL')])
def test_changed_scientific_smoke_refused(field,value):
    g,s=inputs();g['results'][0][field]=value
    with pytest.raises(AssertionError):budget(g,'Solar_336',s)

def identity():
    command=['/data/pjh_workspace/ts-env/bin/python','-u','/data/pjh_workspace/CARTS/experiments/retrieval_failure_mechanism02/worker.py','ETTh1_720','V1','FULL','--mode','analyze']
    return dict(status='RUNNING',controller_pid=123,command=command),'/data/pjh_workspace/CARTS/experiments/retrieval_failure_mechanism02/controller.py',' '.join(command),123,[('CUDA_VISIBLE_DEVICES='+GPU).encode()]

def test_exact_authorized_owner(): validate_owner(*identity())

@pytest.mark.parametrize('mutation',['status','controller','command','ppid','gpu','script'])
def test_unrelated_or_stale_owner_is_refused(mutation):
    state,controller,command,ppid,env=identity()
    if mutation=='status':state['status']='FAILED'
    if mutation=='controller':controller='/unrelated/controller.py'
    if mutation=='command':command+=' --different'
    if mutation=='ppid':ppid=999
    if mutation=='gpu':env=[b'CUDA_VISIBLE_DEVICES=0']
    if mutation=='script':
        state['command'][2]='/data/pjh_workspace/CARTS/experiments/retrieval_failure_mechanism02/unapproved.py';command=' '.join(state['command'])
    with pytest.raises(AssertionError):validate_owner(state,controller,command,ppid,env)
