"""One bounded CPU verification; no controller or GPU work starts here."""
import ast,json,os,subprocess,sys
from pathlib import Path
import torch
import solar_full_train as train

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    train.OUT.mkdir(parents=True,exist_ok=True)
    for p in train.HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(train.HERE)],capture_output=True,text=True)
    log=train.OUT/'CPU_TESTS_FINAL.log';log.write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    from numerical_audit import main as numerical
    numerical(train.OUT/'CPU_NUMERICAL_AUDIT.json')
    try:state=train.prerequisites()
    except (AssertionError,FileNotFoundError,KeyError) as e:state=dict(status='WAITING_PREDECESSOR_ARTIFACTS',reason=str(e))
    plan=dict(order='48 complete -> Oracle K complete -> P10012 -> Full12 + supplements + summary -> GPU smoke -> Solar Full8 -> missing diagnostics + supplements',
        runs=[dict(cell=f'Solar_{h}',model=f'V{m}',seed=0,channels=137,batch=32,epochs=10,status='WAITING_ALL_PREDECESSORS') for h in (96,192,336,720) for m in (0,5)],
        current_prerequisites=state,stage2=False,training_channel_reduction_allowed=False,unverified_encoder_chunk_fallback_allowed=False,
        memory_policy='full grouped encoder required by Adam parity; teacher futures candidate-streamed P128; H720 OOM stops visibly; no smaller channel axis',future_p100_status='CPU_IMPLEMENTED_NOT_GPU_VALIDATED_NOT_ACTIVATED')
    train.dump(train.OUT/'EXECUTION_PLAN.json',plan)
    train.dump(train.OUT/'CPU_READY_GATE.json',dict(status='CPU_PASS_GPU_SMOKE_DEFERRED',at=train.rt.now(),source_hashes=train.source_hashes(),canonical_source_hashes=train.rt.source_hashes(),
        test_log=str(log),test_log_sha256=train.rt.sha256_file(log),numerical_audit_sha256=train.rt.sha256_file(train.OUT/'CPU_NUMERICAL_AUDIT.json'),
        all_channel_batch=True,channels=137,full_grouped_update_cpu_exact=True,smaller_encoder_chunks='REJECTED_ADAM_PARITY',gpu_training_started=False,prerequisite=state))
    print(result.stdout);print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
