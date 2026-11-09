"""Explicit, fail-closed Calibration launch adapter. No background wait mode."""
import argparse, contextlib, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

BASE = Path('/data/pjh_workspace/CARTS')
CODE = BASE/'experiments/calibration_bridge02'
OUT = BASE/'results/TRACK-CALIBRATION-BRIDGE02'
CAL = BASE/'experiments/predictive_score_calibration01'
RESULT = BASE/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01'
OLD = BASE/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
UP = BASE/'experiments/learned_dynamics_cross01/recovery02'
UPOUT = OLD/'recovery02'
PY = '/data/pjh_workspace/ts-env/bin/python'
GPU = 'GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
CELLS = [d+'_'+str(h) for d in ('ETTh1','Weather') for h in (96,192,336,720)]
ARMS = ['D1_L2','D1_MLP','D2_L2','D2_MLP','C1_MLP','C2_MLP']

class NotReady(Exception): pass
def require(ok, message):
    if not ok: raise ValueError(message)
def within(p, root): return p==root or root in p.parents
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def label(mode,cell,training,pool=None):
    return dict(mode=mode,cell=cell,training=training,support=('FULL' if training=='P100' else 'P100') if mode=='cross' else training,pool=pool)
def key(d): return tuple(d[k] for k in ('mode','cell','training','support','pool'))
def expected():
    return [label('native',c,s) for c in CELLS for s in ('P100','FULL')]+[label('pool',c,'P100',p) for c in ('ETTh1_96','ETTh1_720','Weather_96','Weather_720') for p in ('paa16_cosine','saraf_time_aligned')]+[label('cross',c,s) for c in CELLS for s in ('P100','FULL')]
def check_identity_set(records):
    keys=[key(r['label']) for r in records]
    require(len(keys)==40 and len(set(keys))==40 and set(keys)=={key(x) for x in expected()},'Missing, duplicate or wrong upstream identities')
def check_record(r, allowed=OLD):
    root=Path(r['path']).resolve();require(within(root,allowed.resolve()),'Result outside approved root')
    for field,hfield in [('completion_path','completion_sha256'),('selection_path','selection_sha256')]:
        p=Path(r[field]);require(within(p.resolve(),root),'Artifact outside resolved result');require(sha(p)==r[hfield],field+' hash mismatch')
    require(sha(root/'IDENTITY.json')==r['identity_sha256'],'Identity hash mismatch')
    require(read(root/'IDENTITY.json')==r['identity'],'Identity provenance mismatch')
    identity=r['identity'];require(all(k in identity for k in ('data','pool_hashes','checkpoint')),'Missing provenance')
    require(identity['data']['cell']==r['label']['cell'],'Data cell mismatch')
    cp=identity['checkpoint'];require(cp['cell']==r['label']['cell'] and cp['training_support']==r['label']['training'] and str(cp['seed'])=='0','Encoder provenance mismatch')
    require(sha(cp['checkpoint'])==cp['checkpoint_sha256'],'Encoder hash mismatch')
    require(set(r['checkpoints'])==set(ARMS),'Missing scorer checkpoint')
    selection=read(r['selection_path'])['arms']
    for n,v in r['checkpoints'].items():
        require(within(Path(v['path']).resolve(),allowed.resolve()),'Checkpoint outside allowed root')
        require(v['epoch']==selection[n]['best_epoch'] and sha(v['path'])==v['sha256'],'Selected checkpoint mismatch')
    for p,h in r['raw_hashes'].items():
        # Original ETTh1 FULL96 has preserved recovery01 raw links. Accept only
        # manifest-declared logical children resolving within the approved OLD tree.
        require(within(Path(p).absolute(),Path(r['path']).absolute()) and within(Path(p).resolve(),allowed.resolve()) and sha(p)==h,'Raw artifact mismatch')

def upstream(command=None, run=subprocess.run):
    # Separate process prevents upstream recovery_runtime/core imports from
    # replacing Calibration's experiment_common (including its output root).
    command=command or [PY,str(CODE/'upstream_probe.py')]
    r=run(command,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0'))
    if r.returncode==2: raise NotReady(r.stdout.strip())
    require(r.returncode==0,'Upstream INVALID: '+r.stdout[-1500:]+r.stderr[-1500:])
    result=json.loads(r.stdout);check_identity_set(result['records']);return result

def seals():
    pinned=read(OUT/'SOURCE_SEAL.json')
    require(pinned['files']=={p.name:sha(p) for p in CODE.glob('*.py')},'Adapter seal mismatch')
    for p,h in pinned['dependencies'].items(): require(sha(p)==h,'Dependency changed: '+p)
    for root,seal in [(CAL,RESULT/'EXECUTION_SOURCE_SEAL.json'),(CAL/'full_extension01',RESULT/'full_extension01/SOURCE_SEAL.json')]:
        d=read(seal);require(d['files']=={p.name:sha(p) for p in root.glob('*.py')},'Original seal mismatch')
        for p,h in d.get('parent_dependencies',{}).items(): require(sha(CAL/p)==h,'Parent dependency mismatch')

class ResolvedPath:
    """Read-only logical path adapter; creates no files or symlinks."""
    def __init__(self, records, parts=()): self.records=records;self.parts=parts
    def __truediv__(self, other): return ResolvedPath(self.records,self.parts+tuple(Path(other).parts))
    def actual(self):
        p=self.parts
        if not p or p[0]=='features': return OLD.joinpath(*p)
        if p[0] in ('P100','FULL') and len(p)>=2:
            wanted=label('native',p[1],p[0]);tail=p[2:]
        elif len(p)>=4 and p[:2]==('supplemental','cross'):
            training=p[2].split('_')[1];wanted=label('cross',p[3],training);tail=p[4:]
            require(p[2]=='V5_'+training+'_to_'+wanted['support'],'Bad cross label')
        else: raise ValueError('Unsupported resolved path: '+str(p))
        r=self.records[key(wanted)]
        if len(tail)==3 and tail[0]=='checkpoints':
            cp=r['checkpoints'][tail[1]];require(tail[2]=='epoch%02d.pt'%cp['epoch'],'Nonselected checkpoint request');return Path(cp['path'])
        return Path(r['path']).joinpath(*tail)
    def __fspath__(self): return str(self.actual())
    def __str__(self): return str(self.actual())
    def exists(self): return self.actual().exists()
    def read_text(self,*a,**kw): return self.actual().read_text(*a,**kw)
    def read_bytes(self): return self.actual().read_bytes()
    def with_suffix(self,s): return self.actual().with_suffix(s)
    @property
    def name(self): return self.actual().name

def replay_gate(d):
    require(d.get('status')=='P100_LOCAL_RANKING_COMPLETE_GPU','Not a GPU replay')
    rows=d['replay'];require(len(rows)==16 and {(r['cell'],r['split']) for r in rows}=={(c,s) for c in CELLS for s in ('val','test')},'Replay coverage')
    for r in rows:
        require(set(r['ordered_id_mismatches'])=={'D2_MLP','C2_MLP'} and all(type(v) is int and v==0 for v in r['ordered_id_mismatches'].values()),'GPU ordered mismatch must be zero')
def idle(info,apps):
    i,u,m,v=[x.strip() for x in info.strip().split(',')]
    require(i=='1' and u==GPU and int(m)<128 and int(v)==0 and not any(x.startswith(GPU) for x in apps.splitlines()),'GPU1 busy or identity changed')
def gpu_check():
    info=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);idle(info,apps)
@contextlib.contextmanager
def locks(paths):
    import fcntl
    handles=[]
    try:
        for p in paths:
            f=Path(p).open('rb');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield handles
    finally:
        for f in reversed(handles):f.close()
def gpu_recheck(check=gpu_check,sleep=time.sleep): check();sleep(30);check()
def full_prerequisites(receipt, completed, manifest_hash):
    require(receipt.get('status')=='PASS_BRIDGE_P100' and receipt.get('upstream_final_sha256')==manifest_hash,'P100 bridge receipt missing/stale')
    require(set(completed)==set(CELLS) and all(x=='PASS_ALL_QUERY_P100' for x in completed.values()),'P100 P2 incomplete')
def p0_gate(gate, replay, manifest_hash, replay_hash):
    replay_gate(replay)
    require(gate.get('status')=='PASS' and gate.get('upstream_final_sha256')==manifest_hash and gate.get('replay_sha256')==replay_hash,'Missing or stale P0 evidence')

def worker(script,args,snapshot):
    seals();data=read(snapshot);check_identity_set(data['records'])
    require(sha(data['ready']['final_complete'])==data['ready']['final_sha256'],'Upstream receipt changed')
    for r in data['records']:check_record(r)
    # Parent exclusively owns the original GPU lock; child verifies the inherited FD.
    fd=int(os.environ['CALIBRATION_GPU_LOCK_FD']);st=os.fstat(fd);target=(OLD/'GPU.lock').stat()
    require((st.st_dev,st.st_ino)==(target.st_dev,target.st_ino),'Wrong GPU lock descriptor')
    sys.path.insert(0,str(CAL/'full_extension01'));sys.path.insert(0,str(CAL))
    spec=importlib.util.spec_from_file_location('calibration_bridge_target',CAL/'full_extension01'/script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OLD=ResolvedPath({key(r['label']):r for r in data['records']})
    if script=='full_train.py':m.FullRun(args[0],args[1],len(args)>2 and args[2]=='--gate-only').run()
    else:m.run(*args)

def launch(stage):
    import preflight,lifecycle
    data,proof=preflight.run()
    finalhash=data['ready']['final_sha256']
    # Any earlier attempt, including stale/corrupt claims, is a manual-review stop.
    require(not (OUT/(stage.upper()+'_STARTED.json')).exists() and not (OUT/('launch_'+stage)).exists(),'Previous stage attempt: review required, no automatic restart')
    if stage=='full':
        previous=lifecycle.prior_finished(OUT,'p100');receipt=read(OUT/'P100_COMPLETE.json')
        require(receipt['run_id']==previous['run_id'] and receipt['state_sha256']==sha(OUT/'launch_p100/STATE.json'),'P100 lifecycle receipt mismatch')
        require(receipt['bindings']==proof['bindings'],'P100 bindings changed')
        for path,digest in receipt['outputs'].items():require(sha(path)==digest,'P100 output receipt mismatch: '+path)
        full_prerequisites(receipt,{c:read(RESULT/'runs/P100'/c/'P2/COMPLETE.json')['status'] for c in CELLS},finalhash)
    prior_paths=['canonical_cache/P100','runs/P100','LOCAL_RANKING_GPU_COMPLETE.json','HEAD_DIAGNOSTICS_COMPLETE.json','P100_P0_GATE.json'] if stage=='p100' else ['full_extension01/P0','gate_runs/FULL','runs/FULL','cross_support']
    require(not any((RESULT/p).exists() for p in prior_paths),'Prior Calibration stage artifacts: reviewed continuation required')
    # Existing inodes are opened read-only; no lock file is replaced or removed.
    paths=[RESULT/'queue.lock',RESULT/'full_extension01/queue.lock',OLD/'CONTROLLER.lock',OLD/'supplemental/controller.lock',OLD/'recovery02/controller.lock',OLD/'GPU.lock']
    gpu_check()
    with locks(paths) as held:
        gpu_recheck();seals();require(sha(data['ready']['final_complete'])==finalhash,'Final changed after locks')
        if stage=='full':lifecycle.prior_finished(OUT,'p100')
        life=lifecycle.Stage(OUT,stage,proof['bindings']);dest=life.directory
        try:
            snapshot=dest/'UPSTREAM.json';snapshot.write_text(json.dumps(data,indent=2))
            output_hashes={}
            def execute(script,args,target,status,extension=False):
                if target.exists():require(read(target).get('status')==status,'Bad existing completion');return
                seals();gpu_check()
                py='/usr/bin/python3' if extension else PY
                if extension and script=='cross_support.py' and args[1]=='P100':py=PY
                cmd=[py,str(CODE/'bridge.py'),'--worker',script,'--snapshot',str(snapshot),'--']+args if extension else [py,str(CAL/script)]+args
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CALIBRATION_GPU_LOCK_FD=str(held[-1].fileno()))
                with (dest/(script+'_'+'_'.join(args)+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,env=env,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,pass_fds=(held[-1].fileno(),))
                    life.started(child,cmd,dict(script=script,args=args,target=str(target)))
                    returncode=child.wait();life.ended(returncode)
                require(returncode==0,'Worker failed; preserve partial outputs');require(read(target).get('status')==status,'Completion missing/invalid')
                output_hashes[str(target)]=sha(target)
                for name in ['SELECTION.json','PROTOCOL.json']:
                    p=target.parent/name
                    if p.exists():output_hashes[str(p)]=sha(p)
                for p in (target.parent/'checkpoints').rglob('*.pt'):output_hashes[str(p)]=sha(p)
            if stage=='p100':
                # Reuse sealed CPU prerequisite checks, never call its stale main().
                spec=importlib.util.spec_from_file_location('sealed_p100_controller',CAL/'controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.prerequisites()
                for c in CELLS:execute('prepare_gpu.py',[c],RESULT/'canonical_cache/P100'/c/'COMPLETE.json','PASS_ALL_QUERY_P100_HEAD_CACHE')
                execute('local_ranking_gpu.py',[],RESULT/'LOCAL_RANKING_GPU_COMPLETE.json','P100_LOCAL_RANKING_COMPLETE_GPU');replay_gate(read(RESULT/'LOCAL_RANKING_GPU_COMPLETE.json'))
                execute('head_diagnostics_gpu.py',[],RESULT/'HEAD_DIAGNOSTICS_COMPLETE.json','PASS_P100_ALL8')
                with (RESULT/'P100_P0_GATE.json').open('x') as f:json.dump(dict(status='PASS',scope='P100 all8 only',upstream_final_sha256=finalhash,replay_sha256=sha(RESULT/'LOCAL_RANKING_GPU_COMPLETE.json'),test_used_for_method_selection=False),f)
                for phase in ('P1','P2'):
                    for c in CELLS:execute('train_p100.py',[c,phase],RESULT/'runs/P100'/c/phase/'COMPLETE.json','PASS_ALL_QUERY_P100')
            else:
                p0_gate(read(RESULT/'P100_P0_GATE.json'),read(RESULT/'LOCAL_RANKING_GPU_COMPLETE.json'),finalhash,sha(RESULT/'LOCAL_RANKING_GPU_COMPLETE.json'))
                for c in CELLS:execute('full_p0.py',[c],RESULT/'full_extension01/P0'/c/'COMPLETE.json','PASS_FULL_P0_ALL_QUERY',True)
                for phase in ('P1','P2'):
                    for c in CELLS:
                        execute('full_train.py',[c,phase,'--gate-only'],RESULT/'gate_runs/FULL'/c/phase/'COMPLETE.json','PASS_FULL_ACTUAL_INTEGRATION_ONLY',True)
                        execute('full_train.py',[c,phase],RESULT/'runs/FULL'/c/phase/'COMPLETE.json','PASS_ALL_QUERY_FULL',True)
                for training in ('P100','FULL'):
                    other='FULL' if training=='P100' else 'P100'
                    for phase in ('P1','P2'):
                        for c in CELLS:execute('cross_support.py',[c,training,phase],RESULT/'cross_support'/(training+'_to_'+other)/c/phase/'COMPLETE.json','PASS_FROZEN_CROSS_SUPPORT',True)
            life.complete()
            if stage=='p100':output_hashes[str(RESULT/'P100_P0_GATE.json')]=sha(RESULT/'P100_P0_GATE.json')
            lifecycle.exclusive_json(OUT/(stage.upper()+'_COMPLETE.json'),dict(status='PASS_BRIDGE_'+stage.upper(),upstream_final_sha256=finalhash,run_id=life.run_id,bindings=proof['bindings'],outputs=output_hashes,state_sha256=sha(dest/'STATE.json')))

        except BaseException as error:
            life.failed(error)
            raise

def main():
    require(sys.flags.optimize==0,'Run without Python optimization; upstream assertions are mandatory')
    a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');a.add_argument('--preflight',action='store_true');a.add_argument('--execute',choices=['p100','full']);a.add_argument('--worker',choices=['full_p0.py','full_train.py','cross_support.py']);a.add_argument('--snapshot');a.add_argument('args',nargs=argparse.REMAINDER);x=a.parse_args()
    try:
        if x.preflight:
            import preflight
            data,proof=preflight.run();print(json.dumps(proof))
        elif x.check:seals();d=upstream();print(json.dumps(d['ready']))
        elif x.execute:launch(x.execute)
        elif x.worker:worker(x.worker,x.args[1:] if x.args[:1]==['--'] else x.args,x.snapshot)
        else:a.error('Explicit --check or --execute required; no automatic wait')
        return 0
    except NotReady as e:print(json.dumps(dict(status='NOT_READY',reason=str(e))));return 2
    except Exception as e:print(json.dumps(dict(status='INVALID',error=repr(e))));return 1
if __name__=='__main__':sys.exit(main())


