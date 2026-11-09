import copy
import pytest
from gate import require_rows

def rows():
    return [dict(model=m,split=s,status='PASS',units={'val':4551*21,'test':9820*21}[s],ordered_mismatches=0,
                 ordered_candidate_id_sha256='a'*64,canonical_ordered_id_sha256='a'*64,max_stored_prediction_difference=0.)
            for m in ['V0_FullTrained','V5_FullTrained'] for s in ['val','test']]
def test_full_four_rows():require_rows(rows())
@pytest.mark.parametrize('index',[0,1,2,3])
def test_missing_row_denied(index):
    r=rows();r.pop(index)
    with pytest.raises(AssertionError):require_rows(r)
@pytest.mark.parametrize('key,value',[('status','RUNNING'),('units',1),('ordered_mismatches',1),('max_stored_prediction_difference',1e-8),('canonical_ordered_id_sha256','b'*64)])
def test_incomplete_or_mismatched_evidence_denied(key,value):
    r=rows();r[0][key]=value
    with pytest.raises(AssertionError):require_rows(r)
