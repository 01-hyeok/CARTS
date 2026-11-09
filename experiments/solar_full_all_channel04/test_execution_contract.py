import ast
from pathlib import Path
import batch_core as core
from test_batch_core import data_fixture

def test_no_production_channel_serial_loop_or_channel_chunk():
    here=Path(__file__).parent
    for name in ('batch_core.py','evaluation.py','future_p100.py','solar_full_train.py','diagnostic_runtime.py'):
        text=(here/name).read_text();tree=ast.parse(text)
        assert 'channel_chunk' not in text,name
        for n in ast.walk(tree):
            if isinstance(n,(ast.For,ast.comprehension)):
                target=n.target
                assert not isinstance(target,ast.Name) or target.id not in ('c','ci','channel'),(name,ast.dump(target))
def test_only_verified_full_group_shape_can_train():
    assert core.validate_training_shape(data_fixture(720),32,10**9) is None

def test_new_modules_do_not_resolve_to_old_waiter_directory():
    import solar_full_train,solar_full_controller,diagnostic_runtime,diagnostic_supplement
    here=Path(__file__).resolve().parent
    for m in (solar_full_train,solar_full_controller,diagnostic_runtime,diagnostic_supplement):assert Path(m.__file__).resolve().parent==here
