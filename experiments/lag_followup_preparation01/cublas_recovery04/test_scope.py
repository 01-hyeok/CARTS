from scope import applies
import pytest
@pytest.mark.parametrize('heads',[0,1])
@pytest.mark.parametrize('batch',[7,28])
def test_actual_tails(heads,batch):
    assert applies('Weather_336','P100',heads,batch)
@pytest.mark.parametrize('args',[('Weather_720','P100',1,7),('ETTh1_96','P100',1,7),('Weather_336','FULL',1,7),('Weather_336','P100',5,7),('Weather_336','P100',1,32),('Weather_336','P100',1,6)])
def test_other_paths_untouched(args):
    assert not applies(*args)
