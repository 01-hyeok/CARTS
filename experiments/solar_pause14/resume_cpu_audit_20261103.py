"""CPU-only audit of the unchanged production train() resume control flow.

Synthetic small candidate bank; all 137 channels. Never starts CUDA or touches
the real Solar run/checkpoints. Child processes model an actual process restart.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,subprocess,datetime,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/solar_full_all_channel04'))
import solar_full_train as t
torch.set_num_threads(1)
assert not torch.cuda.is_available() and not torch.cuda.is_initialized()

class Queries(Dataset):
    def __init__(self,series,n,h):
        self.data_x=self.data_y=series.numpy();self.starts=np.arange(n);self.h=h
    def __len__(self):return len(self.starts)
    def __getitem__(self,i):
        return self.data_x[i:i+self.h],self.data_y[i+self.h:i+2*self.h],int(i)

class Data:
    def __init__(self,cell,ignored_device):
        self.cell=cell;self.h=int(cell.split('_')[1]);self.c=137;self.device=torch.device('cpu')
        self.starts=np.arange(19);self.offsets=torch.arange(self.h)
        self.series=torch.from_numpy(np.random.RandomState(714).normal(size=(2*self.h+35,137)).astype('float32'))
        self.future_series=self.series
        self.datasets={s:Queries(self.series,n,self.h) for s,n in [('train',35),('val',3),('test',5)]}
        self.args=SimpleNamespace(seq_len=self.h,pred_len=self.h,enc_in=137)
        self.data_info={'synthetic':True,'candidate_count':19,'all_channels':137}
    def loader(self,split,batch,seed):return t.rt.make_training_loader(self.datasets[split],batch,seed)
    def mask(self,starts):return torch.ones(len(starts),len(self.starts),dtype=torch.bool)

def equal(a,b,path='root'):
    if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)

def cleaned_history(h):
    return [{k:({x:y for x,y in v.items() if x!='seconds'} if k=='val' else v)
             for k,v in row.items() if k!='seconds'} for row in h]

def child(base,mode):
    base=Path(base);t.OUT=base/'results';t.CKPT=base/'checkpoints'
    t.OUT.mkdir(parents=True,exist_ok=True)
    original_init=t.rt.model_init;original_atomic=t.rt.atomic_torch;original_update=t.batch_update
    t.rt.CellData=Data
    t.rt.model_init=lambda cell,seed,heads,device:original_init(cell,seed,heads,'cpu')
    t.guard=lambda:None
    torch.cuda.reset_peak_memory_stats=lambda:None
    torch.cuda.max_memory_allocated=lambda:0
    torch.cuda.max_memory_reserved=lambda:0
    gate=dict(status='PASS',source_hashes=t.source_hashes(),canonical_source_hashes=t.rt.source_hashes(),
              query_chunk=32,candidate_chunk=19,dropout_policy='unchanged production dropout_plan on CPU')
    t.dump(t.OUT/'SMOKE_GATE.json',gate)
    records=[]
    def update(data,model,heads,opt,x,y,starts,g):
        value=original_update(data,model,heads,opt,x,y,starts,g)
        records.append(dict(ids=np.asarray(starts).tolist(),loss=value,python=random.random(),numpy=float(np.random.random())))
        return value
    t.batch_update=update
    class AtBoundary(BaseException):pass
    def save(p,payload):
        original_atomic(p,payload)
        if mode=='interrupt' and Path(p).name=='latest.pth' and payload['epoch']==5:raise AtBoundary()
    t.rt.atomic_torch=save
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    try:t.train('Solar_336',5)
    except AtBoundary:assert mode=='interrupt'
    (base/(mode+'_batch_records.json')).write_text(json.dumps(records,indent=2))
    assert not torch.cuda.is_initialized()

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for name,mode in [('continuous','continuous'),('resumed','interrupt'),('resumed','resume')]:
        with (out/(mode+'.log')).open('w') as f:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'child',str(out/name),mode],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        assert p.returncode==0,('child failed',mode,str(out/(mode+'.log')))
    ca=out/'continuous/checkpoints/Solar_336/V5/seed0';cb=out/'resumed/checkpoints/Solar_336/V5/seed0'
    checks=[]
    for epoch in range(1,11):
        a=torch.load(ca/('epoch%02d.pth'%epoch),map_location='cpu',weights_only=False)
        b=torch.load(cb/('epoch%02d.pth'%epoch),map_location='cpu',weights_only=False)
        for key in ['model_state_dict','slot_heads_state_dict','optimizer','rng','loader_rng','source_hashes','args']:
            equal(a[key],b[key],str(epoch)+'.'+key)
        equal(cleaned_history(a['history']),cleaned_history(b['history']),'history')
        checks.append(dict(epoch=epoch,bitwise_model_heads_optimizer_rng_loader=True,metrics_exact=True))
    a=json.loads((out/'continuous/continuous_batch_records.json').read_text())
    b=json.loads((out/'resumed/interrupt_batch_records.json').read_text())+json.loads((out/'resumed/resume_batch_records.json').read_text())
    equal(a,b,'all batch order/loss/Python/NumPy draws')
    actual=ROOT/'checkpoints/track_solar_full_all_channel04/Solar_336/V5/seed0/epoch05.pth'
    saved=torch.load(actual,map_location='cpu',weights_only=False)
    model,heads,_,_=t.rt.model_init('Solar_336',0,5,'cpu');opt=torch.optim.Adam(list(model.parameters())+list(heads.parameters()),lr=.001)
    model.load_state_dict(saved['model_state_dict']);heads.load_state_dict(saved['slot_heads_state_dict']);opt.load_state_dict(saved['optimizer'])
    equal(model.state_dict(),saved['model_state_dict']);equal(heads.state_dict(),saved['slot_heads_state_dict']);equal(opt.state_dict(),saved['optimizer'])
    assert saved['epoch']==5 and saved['source_hashes']==t.source_hashes()
    proof=dict(status='PASS_CPU_ONLY_GPU_REPLAY_NOT_VERIFIED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),started=started,
      original_train_source_sha256=hashlib.sha256((ROOT/'experiments/solar_full_all_channel04/solar_full_train.py').read_bytes()).hexdigest(),
      test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_device='CPU',cuda_initialized=torch.cuda.is_initialized(),
      synthetic_fixture=dict(horizon=336,channels=137,candidates=19,train_queries=35,batch32=True,tail_batch=3),
      fresh_process_resume_epoch=6,epochs=checks,all_20_batch_losses_ids_and_random_draws_exact=True,
      actual_checkpoint_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),actual_checkpoint_model_heads_optimizer_restore_exact=True,
      actual_optimizer_steps=sorted(set(float(v['step']) for v in saved['optimizer'].values())) if False else sorted(set(float(v['step']) for v in saved['optimizer']['state'].values())),
      limitations=['Synthetic bank/query population, not a new research result','CPU test does not prove CUDA kernel/RNG replay across process restart','No real process signal, checkpoint edit, or CUDA allocation'])
    (out/'RESULT.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='child':child(sys.argv[2],sys.argv[3])
    else:main(sys.argv[1])
