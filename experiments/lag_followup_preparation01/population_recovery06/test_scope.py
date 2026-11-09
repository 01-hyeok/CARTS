import copy
import pytest
from scope import applies, choose


def proof():
    row = dict(projection='lt11_tf32', norm='canonical5', reference_only=False,
               raw_exact=True, encoder_exact=True, query_exact=True, scores_exact=True,
               full_ordered_exact=True, controlled_ordered_exact=True,
               channels=7, bank_count=8449, batch=1)
    return dict(status='PASS_EXACT_ALL_CHANNEL_CANDIDATE', checkpoint_unchanged=True,
                rows=[dict(row, split=s) for s in ['val', 'test']])


def test_exact_candidate():
    assert choose(proof()) == ('lt11_tf32', 'canonical5')


@pytest.mark.parametrize('field', ['raw_exact', 'encoder_exact', 'query_exact', 'scores_exact',
                                   'full_ordered_exact', 'controlled_ordered_exact'])
def test_reject_any_mismatch(field):
    p=proof(); p['rows'][1][field]=False
    with pytest.raises(AssertionError): choose(p)


@pytest.mark.parametrize('field,value', [('channels', 1), ('bank_count', 100),
                                        ('reference_only', True), ('error', 'kernel failed')])
def test_reject_changed_scope(field,value):
    p=proof(); p['rows'][0][field]=value
    with pytest.raises(AssertionError): choose(p)


def test_missing_split():
    p=proof();p['rows'].pop()
    with pytest.raises(AssertionError):choose(p)


def test_scope():
    assert applies('ETTh1_96','FULL',5,1)
    for args in [('Weather_96','FULL',5,1), ('ETTh1_192','FULL',5,1),
                 ('ETTh1_96','P100',5,1), ('ETTh1_96','FULL',1,1),
                 ('ETTh1_96','FULL',5,32)]:
        assert not applies(*args)
