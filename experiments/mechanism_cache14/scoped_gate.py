"""Fail-closed gate reuse candidate, never rewrites existing PASS records.

Only exact unchanged reproduction dependencies qualify. Adapter/runtime changes
invalidate every model; branch-specific equivalence is NOT inferred from names.
"""
import ast
import copy
import json
from pathlib import Path
from cache_store import digest, file_sha

# Explicit analyzed call-closure exclusions. New definitions are not excluded.
EXCLUDED = {
    'worker.py': {'analyze'},
    'gpu_metrics.py': {'normalization_control', 'head_union'},
    'geometry.py': {'lloyd', 'ari', 'silhouette', 'describe'},
    'feature_analysis.py': {'past_delta_scores', 'random_scores', 'groups', 'geometry_sets'},
    'ranking.py': {'rows', 'analyze'},
}
RUNTIME_FIELDS = {'torch_version', 'cuda_runtime', 'device_uuid', 'matmul_tf32',
                  'dtype', 'external_dependencies_sha256', 'blas_libraries_sha256'}


def known_runtime(runtime):
    def known(x):
        if isinstance(x, dict):return bool(x) and all(known(v) for v in x.values())
        if isinstance(x, list):return bool(x) and all(known(v) for v in x)
        return x is not None and x != 'UNKNOWN' and x != ''
    return isinstance(runtime,dict) and RUNTIME_FIELDS <= set(runtime) and known(runtime)


def fingerprint(text, filename):
    tree = ast.parse(text)
    # Preserve signatures, decorators and top-level side effects/imports.
    # Skip only function BODY changes outside the reproduction call closure.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in EXCLUDED.get(filename, set()):
            node.body = [ast.Pass()]
    return digest(ast.dump(tree, include_attributes=False))


def assess(gate, current_identity, old_root, new_root, old_runtime, new_runtime,
           checkpoint_path, gate_path):
    reasons = []
    if json.loads(Path(gate_path).read_text()) != gate:
        reasons.append('gate object differs from immutable source file')
    if gate.get('status') != 'PASS':
        reasons.append('old gate not PASS')
    if not known_runtime(old_runtime) or old_runtime != new_runtime:
        reasons.append('current numerical runtime changed or unknown')
    a = copy.deepcopy(gate.get('identity', {}))
    b = copy.deepcopy(current_identity)
    old = a.pop('source_hashes', {})
    new = b.pop('source_hashes', {})
    if not old or set(old) != set(new):
        reasons.append('source inventory changed')
    if a != b:
        reasons.append('checkpoint/data/pool/protocol identity changed')
    changes = []
    for name in sorted(set(old) & set(new)):
        if Path(name).name != name:
            reasons.append('unsafe source path'); continue
        p, q = Path(old_root)/name, Path(new_root)/name
        if not p.is_file() or not q.is_file():
            reasons.append('source snapshot missing: '+name); continue
        if file_sha(p) != old[name] or file_sha(q) != new[name]:
            reasons.append('source snapshot hash mismatch: '+name); continue
        if old[name] != new[name]:
            if fingerprint(p.read_text(), name) != fingerprint(q.read_text(), name):
                reasons.append('reproduction dependency changed: '+name)
            changes.append(name)
    if not Path(checkpoint_path).is_file() or file_sha(checkpoint_path) != a.get('checkpoint_sha256'):
        reasons.append('checkpoint hash mismatch')
    rows = gate.get('rows', [])
    if len(rows) != 2 or {r.get('split') for r in rows} != {'val', 'test'}:
        reasons.append('full val/test coverage missing')
    for row in rows:
        data = a.get('data', {})
        n = data.get('counts', {}).get(row.get('split'))
        c = data.get('channels')
        if (row.get('status') != 'PASS' or n is None or c is None or
            row.get('queries') != n or row.get('channels') != c or row.get('units') != n*c or
            row.get('ordered_mismatch_count') != 0 or
            not row.get('ordered_id_sha256') or row.get('ordered_id_sha256') != row.get('canonical_ordered_id_sha256') or
            row.get('source_checkpoint_sha256') != a.get('checkpoint_sha256')):
            reasons.append('invalid population/ID/checkpoint evidence: '+str(row.get('split')))
    # Audited runtime proof must be supplied independently; original training
    # runtime UNKNOWN is not silently converted into current runtime provenance.
    return dict(status='ELIGIBLE_PENDING_BOUNDARY_GPU_SMOKE' if not reasons else 'RECOMPUTE_REQUIRED',
                reasons=reasons, changed_files=changes, source_gate_sha256=file_sha(gate_path),
                old_sources=old, current_sources=new, numerical_runtime=old_runtime,
                old_gate_immutable=True, production_activation=False)
