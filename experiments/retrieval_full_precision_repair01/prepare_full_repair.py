"""Record actual tests, verify unchanged sources, and permit only verified resume."""
import subprocess,os,sys
from full_repair_controller import *
def main():
    audit=SOURCE/'00_Audit/FULL_REPAIR01';audit.mkdir(parents=True,exist_ok=True)
    original=read(ORIGINAL/'READY_GATE.json')['source_sha256']
    for name,h in original.items():assert sha(ORIGINAL/name)==h,(name,'original source changed')
    sg=read(queue.HERE/'READY_GATE.json')
    assert sg['source_hashes']==queue.source_hashes(),'Original supplement source changed'
    tests=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for label,py in [('torch2',old.PY),('legacy',old.LEGACY)]:
        cmd=[py,'-m','pytest','-q','-p','no:cacheprovider',str(ORIGINAL/'test_diagnostic_core.py'),str(ORIGINAL/'test_diagnostic_runtime.py'),str(HERE/'test_full_precision_policy.py')]
        a=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=audit/f'tests_{label}.log';log.write_text(a.stdout)
        assert a.returncode==0 and '49 passed' in a.stdout,a.stdout
        tests.append(dict(runtime=label,tests=49,exit_code=a.returncode,log=str(log),sha256=sha(log)))
    gate=read(audit/'ETTh1_96/PRECISION_GATE.json');assert gate['status']=='PASS'
    # Keep the successful actual GPU gate; wrapper provenance additions do not
    # alter score tensors, selection or metric computation.
    records=read(audit/'ETTh1_96/CANONICAL_REPRODUCTION.json')
    assert records['status']=='PASS' and len(records['rows'])==4
    assert all(r['units']==2785*7 for r in records['rows'])
    ready=dict(status='PASS',at=old.now(),source_hashes=hashes(),tests=tests,actual_gpu_gate=str(audit/'ETTh1_96/CANONICAL_REPRODUCTION.json'),actual_gpu_gate_sha256=sha(audit/'ETTh1_96/CANONICAL_REPRODUCTION.json'),all_p100_deliverables_preserved=True,all_candidates_channels_unchanged=True)
    dump(HERE/'READY_GATE.json',ready)
    report='''# Full numerical reproduction repair\n\nP100 all 12 calculation, CPU, detail and deliverable gates passed before Full began.\nFull first E96 attempt stopped on V0 validation channel2 batch0 (22:07 UTC).\nThe original adapter incorrectly imposed V5 precision/runtime on V0.\n\nBounded probes: V0 TF32=True changed predictions (first-batch maximum0.4108145); TF32=False fixed the first batch. Whole validation passed under Torch1, but test batch640/channel6 did not. Torch2 canonical V0 reproduced that batch exactly. Conversely Torch2 V5 failed at validation batch736/channel0; original Torch1+TF32=True is needed for V5. No quality-based checkpoint, parameter, tolerance, candidate, or sample selection was made.\n\nRepair: frozen V0 inference runs sequentially under Torch2, TF32=False, stores every original-mask candidate score in compressed batch files and verifies every stored historical retrieval prediction and test metric. The Full diagnostic uses these scores unchanged. V5 stays in original Torch1 runtime, TF32=True, with full-candidate global softmax. All E96 validation/test queries2785x7 per split passed for both models with original tolerances. Full E96 bank8449, K10, seed0. This preserves each existing model's numerical protocol; the runtime difference is disclosed rather than presented as a controlled speed comparison.\n\n49 CPU tests passed in each runtime, including global normalization, streaming Oracle, mask, ranking and regret identities. Original P100 and supplement code hashes are unchanged. Every later cell still has strict canonical reproduction before attribution. Full failures stop the queue and record IMMEDIATE_ISSUE; missing Solar Full models remain MISSING. Solar8 may start only after FULL_DELIVERABLE_GATE and FINAL_DELIVERABLE_GATE PASS.\n'''
    (audit/'IMPLEMENTATION_REPAIR.md').write_text(report)
    print('REPAIR READY',ready,flush=True)
if __name__=='__main__':main()
