"""Separate pool-training and frozen-checkpoint cross-support jobs.

Imports immutable primary implementation; does not modify its source/results.
"""
import sys,os,json,time,copy,inspect,traceback,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import experiment_common as cm
from experiment_common import load,sha,dump,now,torch_save,ARMS,CELLS,GPU
import run_experiment as main
from run_experiment import Experiment,ev,norm_patch
from train_models import Features,Arm
import torch,numpy as np
from final_evaluation import finish as native_finish

OUT=cm.OUT/'supplemental'

def finish_with_reference(ex,state):
    """Keep metric implementation; use the appropriate same-support S0 reference."""
    src=inspect.getsource(native_finish)
    old="official=(float(ex.f.row['canonical_retmse']) if ex.support=='P100' else float(ex.f.reference('test')['metrics']['retmse10']))"
    assert src.count(old)==1;src=src.replace(old,"official=ex.reference.expected(split,'retmse')")
    old="official_agg=float(ex.f.row['canonical_agg_mse']) if ex.support=='P100' else float(ex.f.reference('test')['metrics']['agg_mse10'])"
    assert src.count(old)==1;src=src.replace(old,"official_agg=ex.reference.expected('test','agg_mse')")
    ns=dict(native_finish.__wrapped__.__globals__);exec(compile(src,__file__+'::same_metrics_new_support_reference','exec'),ns)
    ns['finish'](ex,state)

class Reference:
    def __init__(self,ex,prior):
        self.ex=ex;self.original=prior;self.saved={};self.totals={}
    @torch.no_grad()
    def batch(self,split,a,b):
        key=(split,a)
        if key in self.saved:return self.saved[key]
        ex=self.ex;d=ex.d;vn=ev.variant_name(ex.support,ex.support)
        x,y,_=d.query(split,a,b)
        if ex.mode=='cross':
            old=self.original.batch(split,a,b);source_vn=ev.variant_name(ex.training,ex.support)
            ids=torch.tensor(old[source_vn+'__selected_ids'],device=d.device)
            mse=torch.tensor(old[source_vn+'__selected_mse'],device=d.device)
            provenance='prior independently reproduced same frozen checkpoint, requested cross support'
        else:
            g=ev.canonical_logits(ex.f,split,a,b,'P100');mask=torch.ones((b-a,d.c,100),dtype=torch.bool,device=d.device)
            p=ev.normalize(g,mask);pool=d.pool_all(split,a,b);ids=pool.gather(-1,ev.topk(p,10))
            mse=ev.candidate_distances(d,x,y,ids)
            provenance='direct canonical per-channel V5 on alternate pool; new reference, not reused old-trained model metrics'
        future=d.values_all(ids)+x[:,:,-1,None,None]
        # Reference forecasting arithmetic follows the existing FP32 aggregation.
        agg=(future.mean(-2)-y).square().mean(-1)
        self.totals[key]=dict(ret=float(mse.double().mean(-1).sum()),agg=float(agg.double().sum()),units=mse.shape[0]*mse.shape[1])
        result={vn+'__selected_ids':ids.cpu().numpy(),vn+'__selected_mse':mse.cpu().numpy()}
        path=ex.dest/'baseline_reference'/split/('batch%05d.npz'%(a//32))
        main.npz_save(path,dict(query_start=d.datasets[split].starts[a:b],**result))
        dump(path.with_suffix('.json'),dict(sha256=sha(path),reference=provenance,at=now()))
        self.saved[key]=result
        return result
    def expected(self,split,metric):
        records=[v for (s,a),v in self.totals.items() if s==split]
        total=sum(v['units'] for v in records)
        assert total==len(self.ex.d.datasets[split])*self.ex.d.c,('Incomplete baseline reference',split,total)
        return sum(v['ret' if metric=='retmse' else 'agg'] for v in records)/total
    @torch.no_grad()
    def baseline(self,support,split,a,b):
        ex=self.ex;d=ex.d
        if ex.mode=='cross':return self.original.baseline(support,split,a,b)
        ids=d.pool_all(split,a,b);x,y,_=d.query(split,a,b);distance=ev.candidate_distances(d,x,y,ids)
        oracle_pos=ev.topk(-distance,10);oracle=ids.gather(-1,oracle_pos)
        positions=[]
        for row in range(a,b):
            channels=[]
            for c in range(d.c):
                rng=np.random.default_rng(np.random.SeedSequence([0,int(d.datasets[split].starts[row]),c]))
                channels.append(rng.choice(100,size=10,replace=False))
            positions.append(channels)
        pos=torch.tensor(np.asarray(positions),device=d.device)
        random=ids.gather(-1,pos)
        return oracle,distance.gather(-1,oracle_pos).double().mean(-1).cpu().numpy(),random,distance.gather(-1,pos).double().mean(-1).cpu().numpy(),dict(random='one stateless seed0 uniform-without-replacement draw per timestamp/channel on alternate pool')

class Supplement(Experiment):
    def __init__(self,cell,mode,training,pool=None):
        ready=json.loads((OUT/'CPU_GATE.json').read_text());assert ready['status']=='PASS'
        assert ready['sources']=={p.name:sha(p) for p in HERE.glob('*.py')}
        self.cell=cell;self.mode=mode;self.training=training;self.pool=pool
        self.support=('FULL' if training=='P100' else 'P100') if mode=='cross' else 'P100'
        label=('V5_'+training+'_to_'+self.support) if mode=='cross' else pool
        self.dest=OUT/mode/label/cell
        assert not self.dest.exists(),'Supplement is immutable; explicit recovery required for partial output'
        self.dest.mkdir(parents=True);os.chdir(ROOT)
        self.d,self.f,original,self.identity,self.before=ev.setup(cell,training)
        if training=='FULL' and cell.startswith('Weather'):norm_patch.patch(self.f.batch)
        self.features=Features(self.d,cm.ORIGINAL/'features'/cell)
        if mode=='pool':
            gate=json.loads((OUT/'POOL_REUSE_GATE.json').read_text());r=next(x for x in gate['rows'] if x['cell']==cell and x['pool']==pool)
            for item in r['splits']:
                assert sha(item['path'])==item['sha256'];z=load(item['path']);split=item['split']
                assert np.array_equal(z['query_start_idx'].numpy(),self.d.datasets[split].starts)
                self.d.cpu_pool[split]=z['pool_idx_global'].numpy().astype(np.int64)
                self.d.pool_hashes[split]=item['sha256']
            self.identity['alternate_pool']=r
        self.reference=Reference(self,original);self.prior=self.reference
        self.identity.update(mode=mode,source_training_support=training,evaluation_support=self.support,
            pool=pool or 'original_cosine_P100_or_FULL',supplement_sources={p.name:sha(p) for p in HERE.glob('*.py')},
            future_selection=False,validation_selection='native validation only; unchanged in cross-support evaluation' if mode=='cross' else 'own alternate-pool full validation RetMSE',
            calibration_refit=(mode=='pool'),features_provenance=self.features.provenance)
        dump(self.dest/'IDENTITY.json',self.identity)
        self.cache={};self.arms={};self.opts={};self.history=[];self.steps=0;self.started=time.perf_counter();self.raw_time=0;self.scoring_time=0
    @torch.no_grad()
    def native(self,split,a,b):
        d=self.d
        if self.support=='P100' and not self.f.full_trained:
            g,ids=self.f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
        else:
            g,mask=self.f.full_logits(split,a,b)
            if self.support=='P100':
                ids=d.pool_all(split,a,b);g=g.gather(-1,ids[:,:,None].expand(-1,-1,5,-1));mask=torch.ones_like(ids,dtype=torch.bool)
            else:ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1)
        return g,ev.normalize(g,mask),mask,ids
    @torch.no_grad()
    def batch(self,split,a):
        result=super().batch(split,a)
        if split!='train':self.reference.batch(split,a,result[0])
        return result
    def run(self):
        self.gpu_gate()
        if self.mode=='pool':
            self.calibrate();self.scorer_gate();state=self.train()
        else:
            source=cm.native_path(self.training,self.cell)
            complete=json.loads((source/'COMPLETE.json').read_text());assert complete['status']=='PASS_ALL_QUERY_TRAIN_VAL_TEST'
            state=json.loads((source/'SELECTION.json').read_text())['arms'];cp=[]
            for name in ARMS:
                p=cm.checkpoint_path(source,name,state[name]['best_epoch']);z=load(p)
                arm=Arm(name).to(self.d.device);arm.load_state_dict(z['model']);arm.eval();self.arms[name]=arm
                cp.append(dict(arm=name,path=str(p),sha256=sha(p),selected_epoch=state[name]['best_epoch']))
            dump(self.dest/'SELECTION.json',dict(status='REUSED_NATIVE_SELECTION_NO_REFIT',arms=state,checkpoints=cp,at=now()))
            # Check consumer chunking on actual Train histories; do not fit or update.
            b,ids,mask,p,dist,t=self.batch('train',0);rows=np.arange(2)
            with torch.no_grad():
                for name,arm in self.arms.items():
                    a=arm.raw_chunks(self.features,'train',rows,ids[:2],31);bscore=arm.raw_chunks(self.features,'train',rows,ids[:2],503)
                    assert torch.allclose(a,bscore,rtol=1e-10,atol=1e-10)
            dump(self.dest/'CROSS_GATE.json',dict(status='PASS_NO_PARAMETER_UPDATE_OR_CALIBRATION_REFIT',at=now()))
        # Reuse exact metric implementation. Only reference lookup changes:
        # original native aggregate numbers are not valid for a new support/pool.
        finish_with_reference(self,state)
        dump(self.dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_SUPPLEMENT',at=now(),mode=self.mode,cell=self.cell,
            training_support=self.training,evaluation_support=self.support,pool=self.pool,selection=state,identity=self.identity,
            elapsed_seconds=time.perf_counter()-self.started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),queries={s:len(d) for s,d in self.d.datasets.items()},channels=self.d.c))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cross','pool']);ap.add_argument('cell');ap.add_argument('training',choices=['P100','FULL']);ap.add_argument('--pool',choices=['paa16_cosine','saraf_time_aligned']);a=ap.parse_args()
    try:Supplement(a.cell,a.mode,a.training,a.pool).run()
    except BaseException as e:
        dump(OUT/'errors'/('FAILURE_'+cm.datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),at=now(),args=vars(a),preserved=True));raise
