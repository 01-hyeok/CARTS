import copy
import pytest
import gate
import canonical,hashlib

def fixture():
    done=dict(status='COMPLETE',query_counts={'val':3,'test':4},channels=137)
    cpu=dict(status='PASS',source_complete_sha256='a',all_query_channels=True)
    inf=dict(status='PASS',source_complete_sha256='a',splits=[dict(status='PASS',split=s,queries=n,channels=137,units=n*137,all_methods_ordered_top10_and_oracle_ids_exact=True) for s,n in [('val',3),('test',4)]])
    final=dict(status='PASS',source_complete_sha256='a',cpu_gate_sha256='b',inference_gate_sha256='c',deferred=[])
    return [done,cpu,inf,final,dict(complete='a',cpu='b',inference='c')]
def test_good_artifacts():gate.validate_cell_gate(*fixture())
@pytest.mark.parametrize('kind',['incomplete','stale','channels','queries','units','ids','deferred'])
def test_reject_incomplete_or_unbound(kind):
    a=fixture()
    if kind=='incomplete':a[1]['status']='PENDING'
    if kind=='stale':a[3]['inference_gate_sha256']='bad'
    if kind in ('channels','queries','units'):a[2]['splits'][0][kind]-=1
    if kind=='ids':a[2]['splits'][0]['all_methods_ordered_top10_and_oracle_ids_exact']=False
    if kind=='deferred':a[3]['deferred']=['missing']
    with pytest.raises(AssertionError):gate.validate_cell_gate(*a)
@pytest.mark.parametrize('h',[96,192,336,720])
def test_actual_full_shapes(h):
    n,v,t=gate.EXPECTED[h];gate.validate_shape('Solar_'+str(h),137,n,dict(train=n,val=v,test=t))
    with pytest.raises(AssertionError):gate.validate_shape('Solar_'+str(h),136,n,dict(train=n,val=v,test=t))
    with pytest.raises(AssertionError):gate.validate_shape('Solar_'+str(h),137,100,dict(train=n,val=v,test=t))
def test_waiter_scope():
    cmd=['python','/x/experiments/solar_full_missing_checkpoints01/solar_full_controller.py'];s=dict(status='WAITING_ALL_PRIOR_DIAGNOSTICS',actual_training_started=False)
    gate.validate_old_waiter(cmd,[],s)
    for command,children,state in [(['other'],[],s),(cmd,['123'],s),(cmd,[],dict(s,actual_training_started=True))]:
        with pytest.raises(AssertionError):gate.validate_old_waiter(command,children,state)
def test_canonical_source_is_preserved(tmp_path):
    p=tmp_path/'example.py'
    p.write_text('def feature_block(x):\n    return x * 137\ndef prepare_block(x):\n    return feature_block(x)+100\n')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    fn=canonical.extract(p,{},digest);assert fn(2)==374
    p.write_text(p.read_text().replace('137','136'))
    with pytest.raises(AssertionError):canonical.extract(p,{},digest)
