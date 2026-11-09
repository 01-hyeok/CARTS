"""CPU readiness only. Actual six-model GPU smoke remains mandatory."""
import py_compile
from pathlib import Path
from protocol import ROOT
from io_utils import read,dump,source_hashes,sha,now
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    out=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit'
    gate=read(out/'CPU_TEST_GATE.json');assert gate['status']=='PASS' and gate['tests']>=50 and gate['runtime_hashes']==source_hashes()
    for p in HERE.glob('*.py'):py_compile.compile(str(p),doraise=True)
    probe=read(out/'FROZEN_PROBE_REUSE_GATE.json');assert probe['status']=='PASS' and len(probe['rows'])==16
    avail=read(out/'AVAILABILITY_GATE.json');assert avail['raw_conditions']==16 and avail['full_v1_checkpoint_conditions']==8
    files=['summarize.py','artifact_checks.py','mechanism_statistics.py','supplement.py','deliverables.py','controller.py','completed_cells.py','resume.py','coexist_smoke.py']
    dump(out/'ANALYSIS_CPU_READINESS.json',dict(status='PASS',at=now(),runtime_hashes=source_hashes(),
      aggregation_hashes={f:sha(HERE/f) for f in files},cpu_test_sha256=sha(out/'CPU_TEST_GATE.json'),tests=gate['tests'],
      actual_gpu_smoke='PENDING_REQUIRED_PER_CELL',complete_diagnostics_claim=False))
    print('CPU ANALYSIS READINESS PASS; actual per-cell GPU smoke still mandatory')
