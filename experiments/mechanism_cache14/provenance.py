"""Construct cache keys from actual IDs/values, never filenames or mtimes alone."""
import numpy as np
from cache_store import array_info, digest, file_sha, scope


def block_proof(*, cell, split, query_ids, channel_ids, expected_channels,
                candidate_ids, pool_ids, mask, x, y, normalization,
                data_files, dependency_files, runtime, operation):
    q,c,n,p,m = [np.asarray(z) for z in (query_ids,channel_ids,candidate_ids,pool_ids,mask)]
    x,y=np.asarray(x),np.asarray(y)
    assert q.ndim==c.ndim==n.ndim==1
    assert c.size==expected_channels and len(set(c.tolist()))==c.size
    assert q.size<=32 and len(set(q.tolist()))==q.size
    assert n.dtype.kind in 'iu' and np.array_equal(n,np.arange(len(n)))
    assert m.dtype==np.bool_ and m.shape==(len(q),len(c),len(n))
    assert p.dtype.kind in 'iu' and p.shape==(len(q),len(c),100)
    assert ((p>=0)&(p<len(n))).all()
    assert (np.diff(np.sort(p,axis=-1),axis=-1)>0).all()
    assert np.take_along_axis(m,p,axis=-1).all()
    assert x.ndim==y.ndim==3 and x.shape[:2]==y.shape[:2]==(len(q),len(c))
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert data_files and dependency_files and normalization and runtime
    return scope(cell=cell,split=split,query_ids_sha256=digest(array_info(q)),
        channel_ids_sha256=digest(array_info(c)),candidate_ids_sha256=digest(array_info(n)),
        pool_ids_sha256=digest(array_info(p)),mask_sha256=digest(array_info(m)),
        data_sha256=digest({k:file_sha(v) for k,v in data_files.items()}),
        normalization_sha256=digest(normalization),
        code_sha256=digest({k:file_sha(v) for k,v in dependency_files.items()}),
        query_history_sha256=digest(array_info(x)),query_target_sha256=digest(array_info(y)),
        runtime=runtime,seed=0,batch_shape=[len(q),len(c),len(n)],operation=operation)
