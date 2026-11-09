"""One cuBLASLt operation over the complete channel batch, fused bias."""
import ctypes as ct
import torch

class Algo(ct.Structure):
    _fields_=[('data',ct.c_uint64*8)]
class Result(ct.Structure):
    _fields_=[('algo',Algo),('workspace',ct.c_size_t),('state',ct.c_int),('waves',ct.c_float),('reserved',ct.c_int*4)]

class CublasLtBatch:
    def __init__(self):
        self.lib=ct.CDLL('libcublasLt.so.11');self.handle=ct.c_void_p();self.check(self.lib.cublasLtCreate(ct.byref(self.handle)))
        p,i,z=ct.c_void_p,ct.c_int,ct.c_size_t
        for name,args in {
            'cublasLtMatmulDescCreate':[p,i,i],
            'cublasLtMatrixLayoutCreate':[p,i,ct.c_uint64,ct.c_uint64,ct.c_int64],
            'cublasLtMatrixLayoutSetAttribute':[p,i,p,z],
            'cublasLtMatmulDescSetAttribute':[p,i,p,z],
            'cublasLtMatmulPreferenceSetAttribute':[p,i,p,z],
            'cublasLtMatmulAlgoGetHeuristic':[p,p,p,p,p,p,p,i,p,p],
            'cublasLtMatmul':[p,p,p,p,p,p,p,p,p,p,p,p,p,p,z,p],
        }.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=i
        self.workspace=torch.empty(1<<20,dtype=torch.uint8,device='cuda')

    @staticmethod
    def check(status):
        if status:raise RuntimeError('CUBLASLT status %d'%status)

    def attribute(self,kind,desc,key,value):
        self.check(getattr(self.lib,'cublasLt'+kind+'SetAttribute')(desc,key,ct.byref(value),ct.sizeof(value)))

    def linear(self,x,layer,compute=77,heuristic_batch=1,choice=0,canonical_rows=None):
        x=x.contiguous();c,b,k=x.shape;n=layer.out_features
        assert x.is_cuda and x.dtype==torch.float32
        out=x.new_empty(c,b,n);desc=ct.c_void_p();pref=ct.c_void_p();layouts=[]
        self.check(self.lib.cublasLtMatmulDescCreate(ct.byref(desc),compute,0))
        self.attribute('MatmulDesc',desc,3,ct.c_int(1))
        self.attribute('MatmulDesc',desc,7,ct.c_int(4))
        self.attribute('MatmulDesc',desc,8,ct.c_void_p(layer.bias.data_ptr()))
        original_b=b if canonical_rows is None else canonical_rows
        for rows,cols,ld,stride in [(k,n,k,0),(k,original_b,k,b*k),(n,original_b,n,b*n),(n,original_b,n,b*n)]:
            layout=ct.c_void_p();self.check(self.lib.cublasLtMatrixLayoutCreate(ct.byref(layout),0,rows,cols,ld))
            self.attribute('MatrixLayout',layout,5,ct.c_int(heuristic_batch))
            self.attribute('MatrixLayout',layout,6,ct.c_int64(stride));layouts.append(layout)
        self.check(self.lib.cublasLtMatmulPreferenceCreate(ct.byref(pref)))
        self.attribute('MatmulPreference',pref,1,ct.c_size_t(self.workspace.numel()))
        results=(Result*32)();count=ct.c_int()
        self.check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle,desc,*layouts,pref,32,ct.byref(results),ct.byref(count)))
        if choice>=count.value:raise RuntimeError('No heuristic %d of %d'%(choice,count.value))
        self.check(results[choice].state)
        for layout in layouts:self.attribute('MatrixLayout',layout,5,ct.c_int(c))
        if canonical_rows is not None:
            for layout in layouts[1:]:self.attribute('MatrixLayout',layout,3,ct.c_uint64(b))
        alpha,beta=ct.c_float(1.),ct.c_float(0.)
        status=self.lib.cublasLtMatmul(self.handle,desc,ct.byref(alpha),ct.c_void_p(layer.weight.data_ptr()),layouts[0],
                ct.c_void_p(x.data_ptr()),layouts[1],ct.byref(beta),ct.c_void_p(out.data_ptr()),layouts[2],
                ct.c_void_p(out.data_ptr()),layouts[3],ct.byref(results[choice].algo),ct.c_void_p(self.workspace.data_ptr()),
                self.workspace.numel(),ct.c_void_p(torch.cuda.current_stream().cuda_stream))
        for layout in layouts:self.lib.cublasLtMatrixLayoutDestroy(layout)
        self.lib.cublasLtMatmulDescDestroy(desc);self.lib.cublasLtMatmulPreferenceDestroy(pref)
        self.check(status)
        return out
