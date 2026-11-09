"""Static plan validation; never launches a worker."""
import collections
import json
from pathlib import Path
from preparation import require_teacher_resolution, GPU_UUID

HERE=Path(__file__).resolve().parent

def check_plan():
    cfg=json.loads((HERE/'configs/protocol.json').read_text())
    rows=json.loads((HERE/'configs/run_manifest.json').read_text())
    assert len(rows)==144 and len({r['run_id'] for r in rows})==144
    assert [r['seed'] for r in rows]==[0]*48+[1]*48+[2]*48
    assert require_teacher_resolution(cfg)=='zscore_future_mse'
    assert all(r['teacher_mode']=='zscore_future_mse' for r in rows)
    assert cfg['execution']['gpu_uuid']==GPU_UUID and cfg['execution']['max_concurrent_runs']==1
    assert cfg['stage2_enabled'] is False
    for i,r in enumerate(rows):
        assert r['ordinal']==i+1 and r['previous_run']==(rows[i-1]['run_id'] if i else None)
        assert Path(r['reference_ckpt']).exists()
    gates=HERE/'reports/RUNTIME_GATES.json'
    ready=gates.exists() and json.loads(gates.read_text())['status']=='PASS'
    return dict(plan_valid=True,total_runs=144,teacher_mode='zscore_future_mse',runtime_gates_passed=ready,
                stage2=False,run_status_counts=dict(collections.Counter(r['status'] for r in rows)))

if __name__=='__main__':print(json.dumps(check_plan(),indent=2))
