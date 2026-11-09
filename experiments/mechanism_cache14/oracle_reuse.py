"""Prepared evaluate() adapter: original arithmetic, optional precomputed Oracle.

No monkeypatch of an imported/running module and no global source-file changes.
The caller computes the original topk ONCE for identical d/mask/ID-order inputs.
"""
import ast
import copy


def build_cached_evaluate(original_module):
    import inspect
    source = inspect.getsource(original_module.evaluate)
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    fn.args.kwonlyargs.append(ast.arg(arg='cached_oracle'))
    fn.args.kw_defaults.append(ast.Constant(value=None))
    matches = 0
    for i, node in enumerate(fn.body):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id=='oracle':
            # Same original expression on misses. No reduction/order changes.
            fn.body[i] = ast.Assign(targets=copy.deepcopy(node.targets), value=ast.IfExp(
                test=ast.Compare(left=ast.Name(id='cached_oracle', ctx=ast.Load()),
                                 ops=[ast.Is()], comparators=[ast.Constant(value=None)]),
                body=node.value, orelse=ast.Name(id='cached_oracle', ctx=ast.Load())))
            matches += 1
    if matches != 1:
        raise ValueError('unsupported evaluator; do not guess Oracle expression')
    namespace = dict(original_module.__dict__)
    exec(compile(ast.fix_missing_locations(tree), '<cache14-original-evaluate>', 'exec'), namespace)
    return namespace['evaluate']


def validate_oracle_inputs(d, mask, ids, oracle):
    import torch
    assert oracle.dtype == torch.int64 and oracle.shape == (*d.shape[:2], 100)
    assert d.shape == mask.shape == ids.shape
    assert bool(((oracle >= 0) & (oracle < d.shape[-1])).all())
    assert bool(mask.gather(-1, oracle).all())
    assert bool((oracle.sort(-1).values[..., 1:] != oracle.sort(-1).values[..., :-1]).all())
    # This structural check is additional; it does not prove optimality.
    # Activation requires bit-exact original topk comparison on actual GPU.

