"""Exact CPU-only ID lookup for immutable, already validated feature rows.

No score, distance, candidate membership, tie rule or floating-point operation
is changed. Sorted integer lookup is cached once per query/channel in memory.
"""
import numpy as np

def positions_cached(row,c,ids):
    cache=row.setdefault('_position_cache',{})
    if c not in cache:
        ix=np.flatnonzero(row['valid'][c]);values=row['global_ids'][c,ix]
        order=np.argsort(values,kind='stable');values=values[order];ix=ix[order]
        assert np.all(values[1:]!=values[:-1]),'duplicate valid global IDs'
        cache[c]=(values,ix)
    values,ix=cache[c];wanted=np.asarray(ids,dtype=np.int64)
    offset=np.searchsorted(values,wanted)
    assert np.all(offset<len(values)),'candidate ID not present in valid support'
    assert np.array_equal(values[offset],wanted),'candidate ID not present in valid support'
    return ix[offset]
