"""CPU checks binding production to the complete Weather720 model/split gate."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
VALIDATED=ROOT/'experiments/retrieval_all_channel_tail08'
AUDIT=SOURCE/'00_Audit/TAIL08'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require_rows(rows):
    expected={(m,s) for m in ['V0_FullTrained','V5_FullTrained'] for s in ['val','test']}
    assert len(rows)==4 and {(r['model'],r['split']) for r in rows}==expected
    for row in rows:
        assert row['status']=='PASS' and row['units']=={'val':4551*21,'test':9820*21}[row['split']]
        assert row['ordered_mismatches']==0
        assert row['ordered_candidate_id_sha256']==row['canonical_ordered_id_sha256']
        assert row['max_stored_prediction_difference']==0

def verify():
    release=read(AUDIT/'RELEASE_VALIDATION.json')
    assert release['status']=='PASS_FULL_MODEL_SPLIT_MATRIX' and release['cell']=='Weather_720'
    assert release['source_hashes']=={p.name:sha(p) for p in VALIDATED.iterdir() if p.suffix in ['.py','.cu','.so']}
    v0=SOURCE/'full_v0_canonical_scores/Weather_720/COMPLETE.json'
    reproduction=AUDIT/'Weather_720/CANONICAL_REPRODUCTION.json'
    assert sha(v0)==release['v0_gate_sha256'] and sha(reproduction)==release['v5_gate_sha256']
    precise=read(AUDIT/'Weather_720/PRECISION_GATE.json')
    assert precise['status']=='PASS'
    for name,digest in precise['source_hashes'].items():assert sha(ROOT/name)==digest,(name,'changed source')
    for model in precise['models']:assert sha(model['checkpoint'])==model['sha256']
    result=read(reproduction);assert result['status']=='PASS';require_rows(result['rows'])
    assert result['all_channel_gpu'] and not result['production_serial_fallback']
    exported=read(v0);assert exported['status']=='PASS'
    for row in exported['splits']:
        assert row['ordered_mismatches']==0 and row['max_prediction_diff']==row['max_score_difference']==0
        assert row['channels']==21 and row['candidates']==35448
    return release,precise,result,exported
