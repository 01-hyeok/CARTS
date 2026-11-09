"""Single strided-batched GEMM across all channels; frozen diagnostic only."""
import ctypes as ct
import torch


class CublasBatch:
    def __init__(self):
        self.lib = ct.CDLL('libcublas.so.11')
        self.handle = ct.c_void_p()
        self.check(self.lib.cublasCreate_v2(ct.byref(self.handle)))
        p, i, ll = ct.c_void_p, ct.c_int, ct.c_longlong
        self.fn = self.lib.cublasGemmStridedBatchedEx
        self.fn.argtypes = [p,i,i,i,i,i,p,p,i,i,ll,p,i,i,ll,p,p,i,i,ll,i,i,i]
        self.fn.restype = i

    @staticmethod
    def check(status):
        if status: raise RuntimeError('CUBLAS status %d' % status)

    def linear(self, x, layer, compute=77, algo=-1):
        assert x.is_cuda and x.dtype == torch.float32 and x.ndim == 3
        x=x.contiguous(); c,b,k=x.shape; n=layer.out_features
        out=layer.bias[None,None,:].expand(c,b,n).clone() if layer.bias is not None else x.new_zeros(c,b,n)
        self.check(self.lib.cublasSetStream_v2(self.handle,ct.c_void_p(torch.cuda.current_stream().cuda_stream)))
        alpha,beta=ct.c_float(1.),ct.c_float(1.)
        self.check(self.fn(self.handle,1,0,n,b,k,ct.byref(alpha),ct.c_void_p(layer.weight.data_ptr()),0,k,0,
                           ct.c_void_p(x.data_ptr()),0,k,b*k,ct.byref(beta),ct.c_void_p(out.data_ptr()),0,n,b*n,c,compute,algo))
        return out

    def score(self,query,keys,compute=77,algo=-1):
        query=query.contiguous();keys=keys.contiguous();c,b,k=query.shape;n=keys.shape[1]
        assert keys.shape==(c,n,k) and query.dtype==keys.dtype==torch.float32
        out=query.new_empty(c,b,n)
        self.check(self.lib.cublasSetStream_v2(self.handle,ct.c_void_p(torch.cuda.current_stream().cuda_stream)))
        alpha,beta=ct.c_float(1.),ct.c_float(0.)
        self.check(self.fn(self.handle,1,0,n,b,k,ct.byref(alpha),ct.c_void_p(keys.data_ptr()),0,k,n*k,
                          ct.c_void_p(query.data_ptr()),0,k,b*k,ct.byref(beta),ct.c_void_p(out.data_ptr()),0,n,b*n,c,compute,algo))
        return out

    def sgemm_linear(self,x,layer,pointer=False,math_mode=0):
        x=x.contiguous();c,b,k=x.shape;n=layer.out_features
        out=layer.bias[None,None,:].expand(c,b,n).clone()
        p,i,ll=ct.c_void_p,ct.c_int,ct.c_longlong
        self.check(self.lib.cublasSetMathMode(self.handle,math_mode))
        self.check(self.lib.cublasSetStream_v2(self.handle,ct.c_void_p(torch.cuda.current_stream().cuda_stream)))
        alpha,beta=ct.c_float(1.),ct.c_float(1.)
        if pointer:
            aa=torch.full((c,),layer.weight.data_ptr(),device=x.device,dtype=torch.int64)
            bb=torch.arange(c,device=x.device,dtype=torch.int64)*(b*k*4)+x.data_ptr()
            cc=torch.arange(c,device=x.device,dtype=torch.int64)*(b*n*4)+out.data_ptr()
            fn=self.lib.cublasSgemmBatched;fn.argtypes=[p,i,i,i,i,i,p,p,i,p,i,p,p,i,i];fn.restype=i
            status=fn(self.handle,1,0,n,b,k,ct.byref(alpha),ct.c_void_p(aa.data_ptr()),k,ct.c_void_p(bb.data_ptr()),k,ct.byref(beta),ct.c_void_p(cc.data_ptr()),n,c)
        else:
            fn=self.lib.cublasSgemmStridedBatched;fn.argtypes=[p,i,i,i,i,i,p,p,i,ll,p,i,ll,p,p,i,ll,i];fn.restype=i
            status=fn(self.handle,1,0,n,b,k,ct.byref(alpha),ct.c_void_p(layer.weight.data_ptr()),k,0,ct.c_void_p(x.data_ptr()),k,b*k,ct.byref(beta),ct.c_void_p(out.data_ptr()),n,b*n,c)
        self.check(status);return out
