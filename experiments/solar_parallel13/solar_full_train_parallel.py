"""Transparent startup-guard replacement; no learning/evaluation math changes."""
import argparse,os,sys,traceback
from pathlib import Path
import parallel_policy as p
sys.path.insert(0,str(p.ORIGINAL))
import solar_full_train as t

def main(mode,cell,heads):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==p.GPU
    proof=t.prerequisites();old=t.read(t.OUT/'SMOKE_GATE.json')
    assert old['status']=='PASS' and old['source_hashes']==t.source_hashes()
    assert old['canonical_source_hashes']==t.rt.source_hashes()
    resource=p.budget(old,cell,p.snapshot())
    evidence=dict(at=t.rt.now(),pid=os.getpid(),mode=mode,cell=cell,heads=heads,
                  prerequisite=proof,resource=resource,wrapper_hashes=p.wrapper_hashes(),
                  immutable_original_source_hashes=t.source_hashes(),original_smoke_sha256=p.sha(t.OUT/'SMOKE_GATE.json'),
                  authorization='2026-11-02 06:48:12 UTC: Solar plus one Mechanism02 GPU1 worker',
                  mutation='startup exclusivity guard only; training, RNG, loss, chunks and evaluation unchanged')
    def approved_guard():
        t.prerequisites();t.rt.gpu_guard()
        t.torch.backends.cuda.matmul.allow_tf32=False;t.torch.backends.cudnn.allow_tf32=False
    t.guard=approved_guard
    if mode=='smoke':
        approved_guard()
        from gpu_smoke import real_attempt
        result=real_attempt(cell,old['query_chunk'],old['candidate_chunk'])
        evidence.update(status='PASS',result=result)
        t.dump(p.EVIDENCE/(cell+'_COEXISTENCE_SMOKE.json'),evidence)
        print('COEXISTENCE SMOKE PASS',cell,result,flush=True)
    else:
        gate=t.read(p.EVIDENCE/(cell+'_COEXISTENCE_SMOKE.json'))
        assert gate['status']=='PASS' and gate['wrapper_hashes']==p.wrapper_hashes()
        assert gate['immutable_original_source_hashes']==t.source_hashes()
        assert gate['result']['channels']==137 and gate['result']['ordered_id_replay_exact']
        t.dump(p.EVIDENCE/(cell+f'_V{heads}_EXECUTION_POLICY.json'),evidence)
        t.train(cell,heads)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train']);ap.add_argument('cell');ap.add_argument('--heads',type=int,choices=[0,5],default=0);a=ap.parse_args()
    try:main(a.mode,a.cell,a.heads)
    except BaseException as exc:
        t.dump(p.EVIDENCE/'IMMEDIATE_ISSUE.json',dict(at=t.rt.now(),error=repr(exc),traceback=traceback.format_exc(),cell=a.cell,mode=a.mode))
        raise
