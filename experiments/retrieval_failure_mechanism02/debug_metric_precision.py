"""Reference-only metric dtype audit; no production fallback or tolerance change."""
import os,json
import numpy as np
import torch
from adapter import Data,Frozen,inventory,distances,rt,topk
from worker import guard,ROOT,OUT,native_batch
from io_utils import dump,now,asnumpy

@torch.no_grad()
def main():
    os.chdir(ROOT);guard();torch.set_num_threads(1)
    d=Data('ETTh1_336');f=Frozen(d,next(r for r in inventory(d.cell) if r['architecture']=='V0' and r['training_support']=='P100'));f.precision()
    split='val';a=480;b=512;x,y,starts,dd=distances(d,split,a,b);ev=native_batch(f,split,a,b,dd,x,y)
    old=f.reference(split)['metrics'][a:b,:,:4];new=asnumpy(ev['metrics'])[...,:4]
    ids=d.pool_all(split,a,b);selected=ev['selected_ids'];D=dd.gather(-1,ids)
    pick=(selected[:,:,:,None]==ids[:,:,None,:]).long().argmax(-1);oracle=topk(-D,10)
    values=d.values_all(ids);ql=x[:,:,-1];last=(ql[:,:,None]-y).square().mean(-1)
    B,C,P=ids.shape
    native=rt.metric_rows(D.reshape(B*C,P),values.reshape(B*C,P,d.h),ql.reshape(B*C),y.reshape(B*C,d.h),pick.reshape(B*C,10),oracle.reshape(B*C,10),last.reshape(B*C)).reshape(B,C,-1)
    serial=[];dist_serial=[]
    for c in range(C):
        xx,yy,_=d.channel_batch(split,a,b,c);v=d.values(ids[:,c],c);ds=rt.pooled_future_mse(v,xx[:,-1,0],yy[:,:,0]);oo=rt.stable_topk_indices(ds,10,largest=False)
        serial.append(rt.metric_rows(ds,v,xx[:,-1,0],yy[:,:,0],pick[:,c],oo,last[:,c]));dist_serial.append(ds)
    serial=torch.stack(serial,1);ds=torch.stack(dist_serial,1)
    dc=(values+ql[:,:,None,None]-y[:,:,None]).square().mean(-1)
    err=np.abs(old-new);idx=np.unravel_index(err.argmax(),err.shape)
    result=dict(at=now(),cell=d.cell,batch=[a,b],all_channels=C,ordered_ids_equal=bool(np.array_equal(asnumpy(selected),f.reference(split)['ids'][a:b])),
        maximum_index=list(map(int,idx)),stored=float(old[idx]),diagnostic_fp64=float(new[idx]),
        per_metric_new_error=err.max((0,1)).tolist(),
        all_channel_canonical_fp32_error=np.abs(old-asnumpy(native)[...,:4]).max((0,1)).tolist(),
        serial_reference_fp32_error=np.abs(old-asnumpy(serial)[...,:4]).max((0,1)).tolist(),
        direct_vs_streamed_distance_max=float((D-dc).abs().max()),serial_vs_streamed_distance_max=float((D-ds).abs().max()),
        precision_only_ret_error=float((ev['selected_mse'].mean(-1)-ev['selected_mse'].float().reshape(B*C,10).mean(-1).reshape(B,C)).abs().max()),
        original_metric_source='runtime.py:339-347 float32 metric_rows',criteria_changed=False)
    dump(OUT/'00_Audit/E336_METRIC_PRECISION_AUDIT.json',result);print(json.dumps(result))

if __name__=='__main__':main()
