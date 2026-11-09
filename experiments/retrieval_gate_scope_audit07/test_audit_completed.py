import hashlib
import numpy as np
import pytest
from audit_completed import raw_id_evidence, MODELS


def write_batch(tmp_path, bad=None):
    x=np.zeros((3,2,2,10),dtype=np.int64)
    x[:,:,0]=np.arange(10);x[:,:,1]=np.arange(10,20)
    starts=np.arange(100,103)
    if bad=='duplicate':x[0,0,0,1]=x[0,0,0,0]
    if bad=='out_of_range':x[0,0,0,0]=50
    if bad=='query_alignment':starts[2]+=1
    if bad=='channel_count':x=x[:,:1]
    names=MODELS if bad!='missing_model' else ['V0_FullTrained','other']
    np.savez(tmp_path/'batch00000.npz',selected_ids=x,method_names=np.array(names),query_start=starts)
    return x


def test_raw_provenance_hash_order(tmp_path):
    x=write_batch(tmp_path)
    got=raw_id_evidence(tmp_path,3,2,100,50)
    for i,m in enumerate(MODELS):
        assert got[m]['queries']==3 and got[m]['units']==6
        assert got[m]['raw_to_reproduction_id_sha256']==hashlib.sha256(x[:,:,i].transpose(1,0,2).copy().tobytes()).hexdigest()


@pytest.mark.parametrize('bad',['duplicate','out_of_range','query_alignment','channel_count','missing_model'])
def test_incomplete_or_misaligned_saved_artifact_rejected(tmp_path,bad):
    write_batch(tmp_path,bad)
    with pytest.raises(AssertionError):raw_id_evidence(tmp_path,3,2,100,50)
