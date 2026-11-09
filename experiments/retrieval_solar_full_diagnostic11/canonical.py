"""Load the actual source-bound legacy baseline path, never invent a new formula."""
import ast,hashlib
from pathlib import Path
EXPECTED='27d90c73c353983c1cf05506df7d22088b4003db0e538e68c98ebb89969167b7'
def extract(path,namespace,expected=EXPECTED):
    path=Path(path);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    tree=ast.parse(raw.decode());nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    # Original feature_block already batches the channel axis. Original
    # prepare_block differs only in reference distance reductions, not scores.
    assert set(('feature_block','prepare_block')).issubset(nodes)
    ns=dict(namespace)
    unit=ast.Module(body=[nodes['feature_block'],nodes['prepare_block']],type_ignores=[])
    exec(compile(unit,str(path)+'::unchanged_canonical_reference','exec'),ns)
    return ns['prepare_block']
