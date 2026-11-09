"""Diagnostic candidates, NOT production-selected: one CUDA12 cuBLAS batch.

Uses the current Torch2 cuBLAS handle (same math mode/workspace/runtime),
without changing it. Both layouts retain all C matrices and all N keys.
"""
import ctypes as ct
import torch


class BorrowedCublas12:
    def __init__(self):
        assert int(torch.__version__.split('.')[0]) >= 2
        self.lib = ct.CDLL('libcublas.so.12')
        self.handle = ct.c_void_p(torch.cuda.current_blas_handle())
        p, i, ll = ct.c_void_p, ct.c_int, ct.c_longlong
        self.strided = self.lib.cublasSgemmStridedBatched
        self.strided.argtypes = [p,i,i,i,i,i,p,p,i,ll,p,i,ll,p,p,i,ll,i]
        self.strided.restype = i
        self.pointer = self.lib.cublasSgemmBatched
        self.pointer.argtypes = [p,i,i,i,i,i,p,p,i,p,i,p,p,i,i]
        self.pointer.restype = i

    def score(self, query, keys, pointer=False, transpose_output=False):
        q, k = query.contiguous(), keys.contiguous()
        c, b, d = q.shape
        assert k.shape[0] == c and k.shape[2] == d
        assert q.dtype == k.dtype == torch.float32 and q.is_cuda and k.is_cuda
        n = k.shape[1]
        if transpose_output:
            # C-major [C,N,B], then expose [C,B,N]. Same mathematical dots.
            out = q.new_empty(c, n, b)
            a, z, ta, tb, m, nn, lda, ldb, ldc = q, k, 1, 0, b, n, d, d, b
            sa, sb, sc = b*d, n*d, b*n
        else:
            out = q.new_empty(c, b, n)
            a, z, ta, tb, m, nn, lda, ldb, ldc = k, q, 1, 0, n, b, d, d, n
            sa, sb, sc = n*d, b*d, b*n
        alpha, beta = ct.c_float(1.), ct.c_float(0.)
        p = ct.c_void_p
        if pointer:
            idx = torch.arange(c, device=q.device, dtype=torch.int64)
            aa, bb, cc = idx*sa*4+a.data_ptr(), idx*sb*4+z.data_ptr(), idx*sc*4+out.data_ptr()
            result = self.pointer(self.handle,ta,tb,m,nn,d,ct.byref(alpha),p(aa.data_ptr()),lda,
                                  p(bb.data_ptr()),ldb,ct.byref(beta),p(cc.data_ptr()),ldc,c)
        else:
            result = self.strided(self.handle,ta,tb,m,nn,d,ct.byref(alpha),p(a.data_ptr()),lda,sa,
                                  p(z.data_ptr()),ldb,sb,ct.byref(beta),p(out.data_ptr()),ldc,sc,c)
        if result: raise RuntimeError('cuBLAS12 batch error %s' % result)
        return out.transpose(1,2).contiguous() if transpose_output else out


def packed_all_pairs(query, keys):
    """One all-channel GEMM and diagonal extraction; diagnostic only.

    Cross-channel outputs are discarded. No channel loop or score correction.
    This costs C times more score memory and is not presumed exact or faster.
    """
    c,b,d = query.shape
    n = keys.shape[1]
    assert query.dtype == keys.dtype == torch.float32
    all_scores = query.reshape(c*b,d) @ keys.reshape(c*n,d).T
    grid = all_scores.reshape(c,b,c,n)
    idx = torch.arange(c, device=query.device)
    return grid[idx,:,idx,:].contiguous()
