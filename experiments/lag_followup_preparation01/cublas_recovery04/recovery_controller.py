"""Serial validated recovery. Preserve failures; never modify completed cases."""
import sys,json,os,fcntl,subprocess,traceback,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from d56_aggregate import sha,dump,now
ROOT=Path('/data/pjh_workspace/CARTS');BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';EXT=BASE/'extensions_d5_d6';OUT=EXT/'cublas_recovery04'
PY='/data/pjh_workspace/ts-env/bin/python';GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def state(s,**kw):dump(OUT/'RECOVERY_CURRENT.json',dict(status=s,at=now(),pid=os.getpid(),**kw))
def verify(p,status):
    g=json.loads(p.read_text());assert g['status']==status
    for n,h in g['files'].items():assert sha(p.parent/n)==h
    return g
def main():
    own=(OUT/'recovery.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads((OUT/'WEATHER_VALIDATION_GATE.json').read_text());assert gate['status']=='PASS_ALL_APPROVED_SAMPLE_QUERIES_EXACT'
    for n,h in gate['source_hashes'].items():assert sha(HERE/n)==h
    preserved=OUT/'preserved_failures';marker=OUT/'PRESERVATION.json'
    if not marker.exists():
        snapshots=[]
        for model in ['V0','V1']:
            for folder in ['feature_smoke','complete_features','complete_analysis','d14_statistics']:
                p=EXT/folder/'Weather_336'/(model+'_P100_to_FULL');dest=preserved/folder/'Weather_336'/p.name
                if not p.exists():continue
                assert EXT.resolve() in p.resolve().parents and not any(p.glob('*GATE.json')),'Completed gate must not be moved'
                dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();p.rename(dest);snapshots.append(dict(original=str(p),preserved=str(dest)))
        for n in ['CONTROLLER_STATUS.json','PROGRESS.json','D14_STATISTICS_STATUS.json','D14_STATISTICS_PROGRESS.json','RESOLVED_JOBS.json']:
            p=EXT/n
            if p.exists():preserved.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,preserved/n)
        dump(marker,dict(at=now(),snapshots=snapshots,completed_results_modified=False))
    results=[]
    for model in ['V0','V1']:
        args=['Weather_336',model,'P100','FULL'];name=model+'_P100_to_FULL'
        stages=[('SMOKE',[PY,'-u',str(HERE/'recover_case.py'),model,'--smoke'],EXT/'feature_smoke/Weather_336'/name/'FEATURE_GATE.json','PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL',True),
                ('FEATURES',[PY,'-u',str(HERE/'recover_case.py'),model],EXT/'complete_features/Weather_336'/name/'FEATURE_GATE.json','PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING',True),
                ('CPU_ANALYSIS',[PY,'-u',str(HERE.parent/'d56_aggregate.py')]+args,EXT/'complete_analysis/Weather_336'/name/'ANALYSIS_GATE.json','PASS_D5_D6_CASE',False),
                ('CPU_STATISTICS',[PY,'-u',str(HERE.parent/'d14_sample_completion.py')]+args,EXT/'d14_statistics/Weather_336'/name/'SAMPLE_STATISTICS_GATE.json','PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS',False)]
        for phase,cmd,p,status,gpu in stages:
            if p.exists():verify(p,status);continue
            log=OUT/(model+'_'+phase+'.log')
            with log.open('a') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                state('RUNNING',model=model,phase=phase,worker_pid=child.pid,log=str(log),command=cmd);rc=child.wait()
            results.append(dict(model=model,phase=phase,returncode=rc,log=str(log)));assert rc==0,(model,phase,rc)
            verify(p,status)
        resolution=dict(job=args,status='RESOLVED_EXACT_GPU_REPLAY_AND_ALL_DELIVERABLE_GATES',at=now(),validation_gate_sha256=sha(OUT/'WEATHER_VALIDATION_GATE.json'),analysis_gate_sha256=sha(EXT/'complete_analysis/Weather_336'/name/'ANALYSIS_GATE.json'))
        dump(OUT/(model+'_RESOLUTION.json'),resolution)
    for folder,fn,status in [('complete_analysis','ANALYSIS_GATE.json','PASS_D5_D6_CASE'),('d14_statistics','SAMPLE_STATISTICS_GATE.json','PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS')]:
        paths=list((EXT/folder).glob('*/*/'+fn));assert len(paths)==96
        for p in paths:verify(p,status)
    report=subprocess.run([PY,'-u',str(HERE.parent/'d56_report.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),stdout=(OUT/'report.log').open('a'),stderr=subprocess.STDOUT);assert report.returncode==0
    resolved=json.loads((EXT/'RESOLVED_JOBS.json').read_text()) if (EXT/'RESOLVED_JOBS.json').exists() else []
    # Retain original format/history without changing the original FAILED_JOBS.
    dump(OUT/'RESOLUTIONS.json',dict(previous=resolved,new=[json.loads((OUT/(m+'_RESOLUTION.json')).read_text()) for m in ['V0','V1']]))
    dump(EXT/'PROGRESS.json',dict(at=now(),complete=96,total=96,native_complete=48,cross_support_complete=48,failed=[],resolution_manifest=str(OUT/'RESOLUTIONS.json'),original_failure_history=str(EXT/'FAILED_JOBS.json')))
    dump(EXT/'CONTROLLER_STATUS.json',dict(status='COMPLETE_ALL_CASES',at=now(),pid=os.getpid(),complete=96,failed=[],resolution_manifest=str(OUT/'RESOLUTIONS.json')))
    dump(EXT/'D14_STATISTICS_STATUS.json',dict(status='COMPLETE_SAMPLE_STATISTICS',at=now(),completed=96,failures=[],resolution_manifest=str(OUT/'RESOLUTIONS.json')))
    state('PASS_TWO_RECOVERIES_ALL_96_CASES_VERIFIED',results=results)
if __name__=='__main__':
    try:main()
    except BaseException as e:state('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
