"""Cache wrappers preserve the passed canonical function and all its input shapes.

Only usable in a fresh, validated worker. This module neither patches nor starts
any worker, and has no checkpoint/model/encoder API.
"""
import numpy as np
from cache_store import array_info, exact_equal


class CommonOperations:
    def __init__(self, store):
        self.store = store

    def array(self, proof, original_compute, *, expected=None):
        def compute():
            value = original_compute()
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            return {'value': np.asarray(value)}
        exp = None if expected is None else {'value': np.asarray(expected)}
        result, state = self.store.get_or_compute(proof, compute, expected=exp)
        return result['value'], state

    def oracle(self, proof, original_compute, distance, mask, ordered_ids, *, expected=None):
        # Pin full values and ID order once per block/support; keys for scores or
        # another model never enter the Oracle operation.
        p = dict(proof, distance_sha256=array_info(distance)['sha256'],
                 mask_sha256=array_info(mask)['sha256'],
                 support_order_sha256=array_info(ordered_ids)['sha256'],
                 operation='oracle100')
        oracle, state = self.array(p, original_compute, expected=expected)
        b, c, n = distance.shape
        valid = (oracle.dtype == np.dtype('int64') and oracle.shape == (b,c,100)
                 and np.all((oracle >= 0) & (oracle < n)))
        if valid:
            valid = (np.take_along_axis(mask, oracle, -1).all()
                     and np.all(np.diff(np.sort(oracle, axis=-1), axis=-1) > 0))
        if not valid:
            # Corrupt-but-self-consistent payload must not escape validation.
            oracle = np.asarray(original_compute())
            if expected is not None and not exact_equal(oracle, expected):
                raise ValueError('canonical Oracle recomputation mismatch')
            if oracle.dtype != np.dtype('int64') or oracle.shape != (b,c,100):
                raise ValueError('invalid canonical Oracle')
            if not (np.all((oracle >= 0) & (oracle < n)) and
                    np.take_along_axis(mask, oracle, -1).all() and
                    np.all(np.diff(np.sort(oracle, axis=-1), axis=-1) > 0)):
                raise ValueError('canonical Oracle violates mask/uniqueness')
            self.store.put(p, {'value': oracle})
            state = 'RECOMPUTED_INVALID_IDS'
        return oracle, state
