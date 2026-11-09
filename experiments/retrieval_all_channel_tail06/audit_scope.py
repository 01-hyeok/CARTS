"""CPU source/evidence audit. Does not import Torch or modify running code."""
import ast,difflib,hashlib,json,datetime
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL06'
AUDIT=OUT.parent
OLD=ROOT/'experiments/retrieval_all_channel03'
TAIL=ROOT/'experiments/retrieval_all_channel_tail05'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gate_path=AUDIT/'ALL_CHANNEL03/Weather_720_V5_TRUE_BATCH_V2_FULL_GATE.json'
    gate=json.loads(gate_path.read_text())
    assert gate['status']=='PASS'
    assert sha(OLD/'packed_batch.py')==gate['source_hashes']['packed_batch.py']
    queue=(OLD/'validation_queue_v3.py').read_text()
    metric=(OLD/'validate_metric_existing_protocol.py').read_text()
    assert "r['architecture']=='V5'" in metric
    assert 'Weather_720_V5_TRUE_BATCH_V2_FULL_GATE.json' in queue
    assert 'Weather_720_V0_TRUE_BATCH_V2_FULL_GATE.json' not in queue
    cache=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/full_v0_canonical_scores/Weather_720'
    files=list((AUDIT/'ALL_CHANNEL03').glob('*Weather_720*V0*'))
    evidence={
        'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'CPU_SCOPE_AUDIT_COMPLETE_GPU_REPAIR_PENDING',
        'old_weather720_v5_full_split_gate':'PASS',
        'v5_queries':{r['split']:r['queries'] for r in gate['rows']},
        'v5_units':{r['split']:r['units'] for r in gate['rows']},
        'old_packed_source_matches_archived_gate':True,
        'metric_gate_model':'V5', 'metric_gate_queries':4,
        'prior_v0_named_gate_files':[str(p) for p in files],
        'weather720_v0_score_cache_exists':cache.exists(),
        'finding':'Prior V5 full-split PASS was overstated as a cell-wide V0/V5 PASS. V0 was not covered.',
        'regression_attribution':'UNRESOLVED until same Torch2 V0 input/weight/shape compares old03 and tail05; no prior V0 PASS exists in this queue.',
        'original_generation_runtime':'UNKNOWN where original provenance is unknown; verification runtime does not establish historical runtime.',
        'source_hashes':{str(p):sha(p) for folder in (OLD,TAIL) for p in [folder/'packed_batch.py',folder/'cublaslt_batch.py']},
        'evidence_hashes':{str(p):sha(p) for p in [gate_path,OLD/'validation_queue_v3.py',OLD/'validate_metric_existing_protocol.py',
                                               AUDIT/'TAIL05/WEATHER720_BLOCKER.json',AUDIT/'TAIL05/TORCH2_CANONICAL_REPEAT.json']},
        'safe_boundary':'Only after Weather192 and Weather336 deliverable gates pass and bounded controller12449 exits before Weather720',
        'production_code_modified':False,'new_gpu_work_started':False,'solar_training_released':False,
    }
    (OUT/'PRIOR_GATE_SCOPE_AUDIT.json').write_text(json.dumps(evidence,indent=2))
    diff=''.join(difflib.unified_diff((OLD/'packed_batch.py').read_text().splitlines(True),
                                    (TAIL/'packed_batch.py').read_text().splitlines(True),
                                    fromfile='ALL_CHANNEL03/packed_batch.py',tofile='TAIL05/packed_batch.py'))
    (OUT/'OLD03_TO_TAIL05_PACKED.diff').write_text(diff)
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
