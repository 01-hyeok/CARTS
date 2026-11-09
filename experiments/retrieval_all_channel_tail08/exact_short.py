"""Explicit canonical FP32 arithmetic over all channels, no serial fallback."""
import ctypes as ct
from pathlib import Path
import torch


def canonical_sum_squares128(x,canonical_query_rows):
    assert x.shape[-1]==128 and x.dtype==torch.float32
    # Installed Torch2 Reduce.cuh: block_width for canonical [B,128].
    height=min(1<<(canonical_query_rows.bit_length()-1),16)
    width=min(128,512//height)
    squares=x*x
    value=squares[...,:width]
    for start in range(width,128,width):value=value+squares[...,start:start+width]
    while width>32:
        width//=2
        value=value[...,:width]+value[...,width:]
    for offset in [1,2,4,8,16]:value=value[...,:-offset]+value[...,offset:]
    return value


def canonical_norm128(x,canonical_query_rows):
    return x/canonical_sum_squares128(x,canonical_query_rows).sqrt().clamp_min(1e-12)


class Split64:
    def __init__(self):
        self.lib=ct.CDLL(str(Path(__file__).with_name('split64.so')))
        self.call=self.lib.split64_all_channels
        self.call.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_size_t]
        self.call.restype=ct.c_int

    def __call__(self,q,k):
        assert q.ndim==k.ndim==3 and q.is_cuda and k.is_cuda
        assert q.dtype==k.dtype==torch.float32 and q.shape[0]==k.shape[0] and q.shape[-1]==k.shape[-1]==128
        q,k=q.contiguous(),k.contiguous();c,b,_=q.shape;n=k.shape[1]
        out=q.new_empty(c,b,n)
        status=self.call(q.data_ptr(),k.data_ptr(),out.data_ptr(),c,b,n,torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError('split64 CUDA launch failed: %d'%status)
        return out
