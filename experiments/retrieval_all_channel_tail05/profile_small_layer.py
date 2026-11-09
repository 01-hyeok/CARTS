exec(open(__file__.replace('profile_small_layer.py','diagnose_score.py')).read().split("if __name__")[0])
@torch.no_grad()
def probe():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
    x,_,_=d.query('val',0,1);saved=[];m=f.model.encoder.encoder[3]
    hook=m.register_forward_hook(lambda mod,inp,out:saved.append(inp[0].clone()))
    a.encode_raw_channel_first(f.model,x[:,0,:,None].contiguous(),0);hook.remove();z=saved[0]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:m(z);torch.cuda.synchronize()
    p.export_chrome_trace(str(OUT/'B1_LAYER3_TRACE.json'))
    trace=json.load(open(OUT/'B1_LAYER3_TRACE.json'));print(json.dumps([v for v in trace['traceEvents'] if v.get('cat','').lower()=='kernel']))
    saved=[];hook=m.register_forward_hook(lambda mod,inp,out:saved.append((inp[0].clone(),out.clone())))
    for c in range(d.c):a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
    hook.remove();inp=torch.stack([v[0] for v in saved]);expected=torch.stack([v[1] for v in saved])
    variants=[];blas=CublasBatch()
    for pointer in [False,True]:
      for mode in [0,1,3]:
        got=blas.sgemm_linear(inp,m,pointer,mode)
        variants.append(dict(pointer=pointer,math_mode=mode,exact=torch.equal(got,expected),maxdiff=float((got-expected).abs().max())))
    result=dict(batch=1,channels=d.c,variants=variants)
    a.rt.write_json(OUT/'B1_GEMV_BATCH_GATE.json',result);print(json.dumps(result))
if __name__=='__main__':probe()
