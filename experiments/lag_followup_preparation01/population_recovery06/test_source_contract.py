import pytest
from source_contract import transform

def test_single_provenance_change():
    source="a=1\nidentity=dict(original_runtime='UNKNOWN',current_torch=torch.__version__)\nunchanged()"
    expected="a=1\nidentity=dict(recovery=RECOVERY_PROVENANCE,original_runtime='UNKNOWN',current_torch=torch.__version__)\nunchanged()"
    assert transform(source)==expected

@pytest.mark.parametrize('source',['different()',"original_runtime='UNKNOWN',current_torch=torch.__version__\n"*2])
def test_source_drift_rejected(source):
    with pytest.raises(AssertionError):transform(source)
