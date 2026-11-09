import pytest
import sys,types
# Pure eligibility tests run on Windows too. Linux server tests import real
# fcntl; no lock implementation or GPU process is exercised by these tests.
if sys.platform=='win32':sys.modules.setdefault('fcntl',types.ModuleType('fcntl'))
from d56_ownership_recovery import TARGET,ERROR,require_predata_failure
def base():return dict(job=TARGET,phase='FEATURES',returncode=1),dict(phase='BLOCKED',error=ERROR,traceback='before=start_gpu(); data=Data(cell)')
def test_only_predata_owner_failure_accepted():
    r,c=base();require_predata_failure(r,c,[])
@pytest.mark.parametrize('change',[{'phase':'CPU_AGGREGATION'},{'job':['Weather_192','V0','P100','P100']},{'returncode':0}])
def test_wrong_job_or_phase_rejected(change):
    r,c=base();r.update(change)
    with pytest.raises(AssertionError):require_predata_failure(r,c,[])
def test_saved_results_forbid_predata_classification():
    r,c=base()
    with pytest.raises(AssertionError):require_predata_failure(r,c,['query00000.npz'])
def test_numerical_failure_not_retried():
    r,c=base();c['error']='ordered Top10 mismatch'
    with pytest.raises(AssertionError):require_predata_failure(r,c,[])
