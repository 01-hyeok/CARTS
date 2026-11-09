"""Verified CPU-only activation; controller waits for exact D4 boundary gates."""
import hashlib,json,os,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/extensions_d5_d6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    test=OUT/'CPU_EXTENSION_TESTS.xml';root=ET.parse(test).getroot();suites=root.findall('testsuite') if root.tag=='testsuites' else [root]
    assert sum(int(s.attrib['tests']) for s in suites)==10
    assert sum(int(s.attrib.get(k,0)) for s in suites for k in ['failures','errors','skipped'])==0
    names=['d56_complete_run.py','d56_aggregate.py','d56_controller_v2.py','d56_report.py','d56_metrics.py','past_lag.py','server_bridge.py']
    ready=dict(status='PASS',tests=10,test_xml_sha256=sha(test),code_hashes={n:sha(HERE/n) for n in names})
    path=OUT/'CPU_EXTENSION_READY_v2.json'
    if path.exists():assert json.loads(path.read_text())==ready
    else:path.write_text(json.dumps(ready,indent=2))
    active=OUT/'ACTIVATION_v2.json'
    if active.exists():
        old=json.loads(active.read_text());proc=Path('/proc')/str(old['controller_pid'])/'cmdline'
        if proc.exists() and b'd56_controller_v2.py' in proc.read_bytes():print('EXISTING_CONTROLLER',old,flush=True);return
    log=OUT/'logs/controller.log';log.parent.mkdir(exist_ok=True)
    command=['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'d56_controller_v2.py')]
    with log.open('a') as stream:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    payload=dict(controller_pid=p.pid,command=command,log=str(log),cpu_ready_sha256=sha(path),gpu_start_only_after_d4_boundary=True)
    active.write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
