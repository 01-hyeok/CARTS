import os,subprocess,sys,re
from pathlib import Path
from protocol import ROOT
from io_utils import dump,now,sha,source_hashes
HERE=Path(__file__).resolve().parent
TESTS=['test_core','test_prerequisites','test_gpu_metrics_cpu','test_geometry_cpu','test_controller','test_statistics','test_artifact_checks','test_supplement','test_recovery']
if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU gate requires hidden GPUs'
    out=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit';out.mkdir(parents=True,exist_ok=True)
    p=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=HERE,capture_output=True,text=True)
    log=out/'CPU_TEST.log';log.write_text(p.stdout+p.stderr);print(log.read_text())
    if p.returncode!=0:
        dump(out/'CPU_TEST_GATE.json',dict(status='FAIL',at=now(),runtime_hashes=source_hashes(),log_sha256=sha(log)))
        raise AssertionError('CPU test failure; no release')
    count=int(re.search(r'Ran (\d+) tests',p.stdout+p.stderr).group(1))
    dump(out/'CPU_TEST_GATE.json',dict(status='PASS',at=now(),tests=count,test_modules=TESTS,
         test_hashes={n+'.py':sha(HERE/(n+'.py')) for n in TESTS},runtime_hashes=source_hashes(),log_sha256=sha(log),gpu_used=False))
