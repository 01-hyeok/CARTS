import sys,json,time,hashlib,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import runtime as rt
import torch,numpy as np
import graphed_validation as cv

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class Trace:
    def __init__(self,expected=None):self.rows=[];self.expected=expected;self.units=0;self.score_mismatch=0;self.pick_mismatch=0
    def __call__(self,model,heads,x,data,c,ids,score=None,**kw):
        p=rt.stable_topk_indices(torch.softmax(score/.1,-1).mean(1),10,largest=True)
        row=dict(channel=c,batch=len(x),ids_hash=digest(ids),score_hash=digest(score),picks_hash=digest(ids.gather(1,p)))
        if self.expected is not None:
            old=self.expected[len(self.rows)];assert old['channel']==c and old['batch']==len(x) and old['ids_hash']==row['ids_hash']
            self.score_mismatch+=int(old['score_hash']!=row['score_hash']);self.pick_mismatch+=int(old['picks_hash']!=row['picks_hash'])
        self.rows.append(row);self.units+=len(x)

def timed_eval(data,m,h,kind,trace):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    if kind=='canonical':
        original=rt.scores
        def wrapped(*args,**kwargs):
            sc=original(*args,**kwargs);trace(*args,score=sc,**kwargs);return sc
        rt.scores=wrapped
        try:out=rt.evaluate(data,m,h,'val')
        finally:rt.scores=original
    else:out=cv.evaluate(data,m,h,'val',trace=trace)
    torch.cuda.synchronize()
    return out,dict(seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

if __name__=='__main__':
    rt.gpu_guard();torch.set_num_threads(2)
    pause=json.loads((HERE/'WORKER_PAUSED.json').read_text());pid=pause.get('identity',{}).get('pid')
    if pid and Path(f'/proc/{pid}').exists():assert Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]=='T'
    report=dict(status='RUNNING',started_at=rt.now(),scope='validation accelerator correctness, not retrieval failure attribution experiment',cases=[])
    hand=json.loads((HERE/'HANDOFF.json').read_text())
    for cell,heads,checkpoint in [('Solar_96',5,hand['selected_checkpoint']),('Solar_720',0,rt.config()['references']['Solar_720']['path'])]:
        data=rt.CellData(cell,'cuda:0')
        if not (data.pool_dir/'val.npy').exists():
            data.pool_dir=HERE/'fixture_pools'/cell;data.pool_dir.mkdir(parents=True,exist_ok=True)
            if not (data.pool_dir/'val.npy').exists():
                adapter=rt.CanonicalPoolAdapter(data)
                starts,ids=rt.precompute_split(adapter,data.args,'val',list(range(data.c)),100,data.device,256)
                assert np.array_equal(starts.numpy(),data.datasets['val'].starts)
                np.save(data.pool_dir/'val.npy',ids.numpy().astype(np.int32));rt.ordered_equal(np.load(data.pool_dir/'val.npy',mmap_mode='r'),ids.numpy())
                for start in range(0,len(starts),32):rt.validate_pool(ids[start:start+32].numpy(),starts[start:start+32].numpy(),data.starts,data.h,data.h,'val')
                rt.write_json(data.pool_dir/'VALIDATION_FIXTURE.json',dict(status='PASS',cell=cell,queries=len(starts),channels=137,P=100,canonical_order_exact=True,all_eligible_unique=True,sha256=rt.sha256_file(data.pool_dir/'val.npy'),not_official_pool=True))
                del adapter,ids,starts;gc.collect();torch.cuda.empty_cache()
        m,h,_,_=rt.model_init(cell,0,heads,'cuda:0');payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload['model'] if 'model' in payload else payload.get('model_state_dict',payload.get('state_dict',payload))
        m.load_state_dict(state)
        if heads:h.load_state_dict(payload['heads'])
        m.eval()
        if h is not None:h.eval()
        reference=json.loads((HERE/'CACHE_GATES.json').read_text())
        prior=next(c for c in reference['cases'] if c['cell']==cell)
        oldtrace=Trace();oldtrace.rows=json.loads((HERE/f'{cell}_CANONICAL_TRACE.json').read_text())
        old,ot=prior['canonical_metrics'],prior['canonical_timing']
        trace=Trace(oldtrace.rows);new,nt=timed_eval(data,m,h,'cached',trace)
        diff={k:abs(old['rows'][0][k]-new['rows'][0][k]) for k in rt.FIELDS}
        case=dict(cell=cell,heads=heads,checkpoint=checkpoint,checkpoint_sha256=rt.sha256_file(checkpoint),
            full_validation_queries=len(data.datasets['val']),channels=137,units=trace.units,score_mismatch_batches=trace.score_mismatch,pick_mismatch_batches=trace.pick_mismatch,
            metric_abs_differences=diff,canonical_metrics=old,cached_metrics=new,canonical_timing=ot,cached_timing=nt,speedup=ot['seconds']/nt['seconds'])
        print('CASE',json.dumps(case),flush=True);report['cases'].append(case);rt.write_json(HERE/'GRAPH_GATES.json',report)
        assert trace.units==len(data.datasets['val'])*137 and trace.pick_mismatch==0 and trace.score_mismatch==0
        assert max(diff.values())<1e-12
        assert nt['seconds']<ot['seconds'],'No measured speed improvement'
        del data,m,h,payload,oldtrace,trace;gc.collect();torch.cuda.empty_cache()
    report.update(status='PASS',finished_at=rt.now(),cache_source_sha256=rt.sha256_file(HERE/'cached_validation.py'),graph_source_sha256=rt.sha256_file(HERE/'graphed_validation.py'))
    rt.write_json(HERE/'GRAPH_GATES.json',report)
