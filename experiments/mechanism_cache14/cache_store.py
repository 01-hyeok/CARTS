"""Immutable, content-addressed model-independent blocks. No GPU imports.

Not enabled in the production queue. Dense B*C*N storage is opt-in and bounded;
candidate futures B*C*N*H are never accepted. Corrupt generations are preserved.
"""
import hashlib
import json
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

REQUIRED = {'cell', 'split', 'query_ids_sha256', 'channel_ids_sha256',
            'candidate_ids_sha256', 'pool_ids_sha256', 'mask_sha256',
            'data_sha256', 'normalization_sha256', 'code_sha256',
            'runtime', 'seed', 'batch_shape', 'operation'}
OPERATIONS = {'future_mse', 'oracle100', 'past_delta_scores', 'random_scores',
              'past_features', 'compact_completed_block'}
DENSE = {'future_mse', 'past_delta_scores', 'random_scores'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def array_info(a):
    a = np.asarray(a)
    if a.dtype.hasobject:
        raise ValueError('object arrays are forbidden')
    raw = np.ascontiguousarray(a).view(np.uint8)
    return dict(shape=list(a.shape), dtype=a.dtype.str, nbytes=a.nbytes,
                sha256=hashlib.sha256(raw).hexdigest())


def exact_equal(a, b):
    # Includes dtype, ordered IDs, signed zero, NaN payload, and every value bit.
    return array_info(a) == array_info(b)


def scope(**kw):
    if not REQUIRED <= set(kw):
        raise ValueError('missing provenance: ' + str(sorted(REQUIRED - set(kw))))
    if kw['operation'] not in OPERATIONS or kw['seed'] != 0:
        raise ValueError('unsupported operation or seed')
    if kw['split'] not in ('train', 'val', 'test'):
        raise ValueError('unknown split')
    if not isinstance(kw['runtime'], dict) or not kw['runtime']:
        raise ValueError('explicit numerical runtime required')
    if any(k in kw for k in ('checkpoint', 'model', 'embedding', 'head')):
        raise ValueError('model-dependent values cannot use shared cache')
    b, c, n = kw['batch_shape']
    if not (0 < b <= 32 and c > 0 and n >= 100):
        raise ValueError('invalid full-channel block shape')
    # Round trip provides a detached, JSON-only immutable key payload.
    return json.loads(encoded(kw))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as f:
        f.write(encoded(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


class ExactBlockCache:
    def __init__(self, root, *, dense=False, max_entry_bytes=384 * 1024**2,
                 max_total_bytes=8 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dense = dense
        self.max_entry_bytes = max_entry_bytes
        self.max_total_bytes = max_total_bytes
        self.events = []

    def _paths(self, proof):
        proof = scope(**proof)
        key = digest(proof)
        return proof, key, self.root / (key + '.index.json')

    def load(self, proof, expected=None):
        start = time.perf_counter()
        proof, key, index = self._paths(proof)
        try:
            obj = json.loads(index.read_text())
            # Names cannot escape the dedicated cache directory.
            name = obj['generation']
            if Path(name).name != name or not name.startswith(key + '.'):
                raise ValueError('unsafe generation path')
            generation = self.root / name
            if file_sha(generation / 'manifest.json') != obj['manifest_sha256']:
                raise ValueError('manifest hash mismatch')
            meta = json.loads((generation / 'manifest.json').read_text())
            if meta['scope'] != proof or meta['status'] != 'COMMITTED':
                raise ValueError('provenance mismatch')
            data_path = generation / 'arrays.npz'
            if file_sha(data_path) != meta['file_sha256']:
                raise ValueError('file hash mismatch')
            with np.load(data_path, allow_pickle=False) as z:
                values = {k: z[k].copy() for k in z.files}
            if set(values) != set(meta['arrays']):
                raise ValueError('array set mismatch')
            if any(array_info(a) != meta['arrays'][k] for k, a in values.items()):
                raise ValueError('array ID/value/hash mismatch')
            if expected is not None and (set(expected) != set(values) or
                    any(not exact_equal(values[k], expected[k]) for k in values)):
                raise ValueError('canonical exact-value mismatch')
            self.events.append(dict(event='HIT', key=key, seconds=time.perf_counter()-start))
            return values
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as e:
            self.events.append(dict(event='MISS_RECOMPUTE', key=key,
                                    reason=repr(e), seconds=time.perf_counter()-start))
            return None

    def put(self, proof, values):
        proof, key, index = self._paths(proof)
        values = {k: np.asarray(v) for k, v in values.items()}
        info = {k: array_info(v) for k, v in values.items()}
        if not values or any('/' in k or '\\' in k for k in values):
            raise ValueError('empty or unsafe array set')
        if proof['operation'] in DENSE:
            if any(list(v.shape) != proof['batch_shape'] for v in values.values()):
                raise ValueError('dense cache accepts B*C*N only, never candidate futures')
            if not self.dense:
                return 'BYPASS_DENSE_DISABLED'
        elif any(v.ndim > 4 for v in values.values()):
            raise ValueError('unbounded tensor rank')
        size = sum(v['nbytes'] for v in info.values())
        if size > self.max_entry_bytes:
            return 'BYPASS_CAPACITY_RECOMPUTE'
        lock = self.root / 'cache.writer.lock'
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # No waiting or stale-lock deletion; canonical computation continues.
            return 'BYPASS_WRITER_BUSY'
        try:
            os.close(fd)
            used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if used + size * 2 + 65536 > self.max_total_bytes:
                return 'BYPASS_CAPACITY_RECOMPUTE'
            gen = Path(tempfile.mkdtemp(prefix=key + '.', dir=str(self.root)))
            path = gen / 'arrays.npz'
            with path.open('wb') as f:
                np.savez(f, **values)  # Lossless; no compression CPU overhead.
                f.flush()
                os.fsync(f.fileno())
            meta = dict(status='COMMITTED', scope=proof, arrays=info,
                        file_sha256=file_sha(path), uncompressed_bytes=size)
            atomic_json(gen / 'manifest.json', meta)
            # Reopen and bit-compare the unpublished payload before publishing.
            with np.load(path, allow_pickle=False) as z:
                if set(z.files) != set(values) or any(not exact_equal(z[k], values[k]) for k in values):
                    raise ValueError('write verification failed')
            atomic_json(index, dict(generation=gen.name,
                                   manifest_sha256=file_sha(gen / 'manifest.json')))
            return 'STORED'
        finally:
            lock.unlink()

    def get_or_compute(self, proof, compute, expected=None):
        value = self.load(proof, expected=expected)
        if value is not None:
            return value, 'HIT'
        value = compute()
        if expected is not None and (set(value) != set(expected) or
                any(not exact_equal(value[k], expected[k]) for k in value)):
            raise ValueError('canonical recomputation differs from expected proof')
        return value, self.put(proof, value)
