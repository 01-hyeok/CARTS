import ast,inspect
from pathlib import Path
HERE=Path(__file__).resolve().parent
def test_scope():
    src=(HERE/'kernel_patch.py').read_text();tree=ast.parse(src)
    scope=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef)) and (isinstance(n,ast.Assign) or n.name=='applicable')],type_ignores=[])
    ns={};exec(compile(scope,'scope','exec'),ns)
    fn=ns['applicable']
    for cell in ['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336','Weather_720']:
        for training in ['FULL','P100']:
            for heads in [0,1,5]:
                for b in range(1,33):assert fn(cell,training,heads,b)==(cell in {'ETTh1_192','ETTh1_336','ETTh1_720'} and training=='FULL' and heads==5 and b==1)
def test_all_channel_no_fallback():
    s=(HERE/'kernel_patch.py').read_text()
    assert 'torch.bmm(q,self.batch.keys.transpose(1,2))' in s
    assert 'canonical_norm128(v,5)' in s
    assert 'for c in' not in s and 'canonical_logits' not in s
def test_no_tolerance_relaxation():
    s=(HERE/'run.py').read_text()
    assert "r['max_canonical_e0_score_delta']==0" in s
    assert "r['max_canonical_raw_cosine_delta']==0" in s
    assert "canonical_tail_score_recovery=ACTIVE['corrected']" in s
