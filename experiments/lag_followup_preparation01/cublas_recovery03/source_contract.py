"""Pure source transformation, confined to the actual FP32 single-head tail."""
def transform(source, projection):
    norm='exact_torch2_tail=(compute==68 and self.frozen.heads is None)'
    final='return self.frozen.heads(z.reshape(b*c,dim)).reshape(b,c,-1,dim)'
    assert source.count(norm)==source.count(final)==1
    assert projection in ['flat_exact_norm','cuda12_lt_exact_norm','cuda11_lt_exact_norm','cuda11_blas_exact_norm']
    source=source.replace(norm,'exact_torch2_tail=(compute==68)')
    body='''assert compute==68 and self.frozen.n==1
    zz=z.permute(1,0,2).contiguous()
    weight=self.frozen.heads.W[0][None].expand(c,-1,-1).contiguous()
    if PROJECTION=='flat_exact_norm':
        projected=torch.einsum('bd,sod->bso',z.reshape(b*c,dim),self.frozen.heads.W).reshape(b,c,dim).permute(1,0,2).contiguous()
    elif PROJECTION=='cuda12_lt_exact_norm':
        if not hasattr(self,'probe_lt12'):self.probe_lt12=CublasLtBatch('libcublasLt.so.12')
        projected=self.probe_lt12.score(zz,weight,68,1,0)
    elif PROJECTION=='cuda11_lt_exact_norm':
        projected=self.lt.score(zz,weight,68,1,0)
    else:
        if self.blas is None:self.blas=CublasBatch()
        projected=self.blas.score(zz,weight,68,-1)
    return canonical_norm128(projected,b).permute(1,0,2)[:,:,None].contiguous()'''
    return source.replace(final,body.replace('PROJECTION',repr(projection)))
