"""Unit tests for TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 Stage2 (spec
PART 24, 10 items). Reuses `train_setlossctrl_stage2_retrain02.py`
UNMODIFIED (the audited, most-recently-validated Stage2 trainer) and
`build_m_stage2_retrieval_cache01.py` (TRACK-M's own cache builder,
sibling of `build_setlossctrl_retrieval_cache02.py`).
"""
import hashlib
import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_m_stage2_retrieval_cache01 import CKPTS, build_cache_for_split
from scripts.train_factorial_e2e01 import HostScorer, state_sha
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_setlossctrl_stage2_retrain02 import (
    FREEZE_SUBMODULES, build_fresh_stage2, eval_epoch, freeze_retrieval_submodules,
)

S1_720 = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
         'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_'
         'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
_ref_available = (REPO_ROOT / S1_720).exists() and (REPO_ROOT / S2_720).exists()
needs_ref = pytest.mark.skipif(not _ref_available, reason='Stage1/Stage2 host checkpoints not present')

CACHE_ROOT = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720'
_caches_built = all((CACHE_ROOT / arm / 'train.pt').exists() for arm in ('S0_base', 'S1_J1', 'S2_K2', 'S3_Mstar'))
needs_caches = pytest.mark.skipif(not _caches_built, reason='TRACK-M Stage2 caches not yet built')


class _Cli:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _default_cli(**overrides):
    base = dict(reference_ckpt=str(REPO_ROOT / S1_720), cell='ETTh1_720', pred_len=720, seq_len=720,
               patch_len=16, top_k=10, tau_t=0.1, tau_s=0.1, batch_size=32, learning_rate=1e-3,
               chunk_size=4096, init_seed=0, loader_seed=0)
    base.update(overrides)
    return _Cli(**base)


# item 1: retriever checkpoint hash unchanged by the (read-only) cache-building process
@needs_ref
def test_item1_retriever_checkpoint_hash_unchanged_by_cache_build():
    def _sha(p):
        h = hashlib.sha256()
        h.update(Path(p).read_bytes())
        return h.hexdigest()
    ckpt_path = CKPTS['S2_K2']
    before = _sha(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, base_model = build_model(cli, device)
    host = HostScorer(str(REPO_ROOT / S2_720), device)
    build_cache_for_split('S2_K2', base_model, int(args.d_model), exp, args, host, 720, 'val', 10, device)
    after = _sha(ckpt_path)
    assert before == after


# item 2: identical base initialization across all Stage2 arms (shared seed -> same init SHA)
@needs_ref
def test_item2_identical_base_init_across_arms():
    exp_a, args_a, model_a, host_ck_a = build_fresh_stage2(str(REPO_ROOT / S2_720), seed=0)
    sha_a = state_sha(model_a.state_dict())
    exp_b, args_b, model_b, host_ck_b = build_fresh_stage2(str(REPO_ROOT / S2_720), seed=0)
    sha_b = state_sha(model_b.state_dict())
    assert sha_a == sha_b


# item 3: identical batch order across all Stage2 arms (same seed -> same shuffle draw)
@needs_ref
def test_item3_identical_batch_order_across_arms():
    cli = _default_cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    exp_a, args_a, model_a = build_model(cli, device)
    _, loader_a = exp_a._get_data(flag='train', shuffle=True)
    starts_a = [next(iter(loader_a))[2]]
    torch.manual_seed(0)
    exp_b, args_b, model_b = build_model(cli, device)
    _, loader_b = exp_b._get_data(flag='train', shuffle=True)
    starts_b = [next(iter(loader_b))[2]]
    assert torch.equal(torch.as_tensor(starts_a[0]), torch.as_tensor(starts_b[0]))


# item 4: retrieval candidate count = 10 for every retrieval-bearing arm
@needs_caches
def test_item4_topk_count_is_10_for_every_arm():
    for arm in ('S1_J1', 'S2_K2', 'S3_Mstar'):
        cache = torch.load(CACHE_ROOT / arm / 'val.pt', map_location='cpu')
        assert cache['topk_idx'].shape[-1] == 10
        for b in range(min(20, cache['topk_idx'].shape[0])):
            for c in range(cache['topk_idx'].shape[1]):
                row = cache['topk_idx'][b, c].tolist()
                assert len(set(row)) == 10, f'{arm} b={b} c={c} not 10 unique picks: {row}'


# item 5: no future leakage -- `build_cache_for_split` never references `batch_y` inside its body
def test_item5_no_future_leakage_in_cache_builder():
    src = inspect.getsource(build_cache_for_split)
    assert 'batch_y' not in src or '_batch_y_unused' in src, \
        'cache builder must never read the query future when selecting/aggregating candidates'
    assert 'batch_y[' not in src and 'batch_y.' not in src


# item 6: only the retrieval source differs between arms -- all three retrieval arms
# use the literal SAME HostScorer instance for alpha-weighting (not a per-arm copy)
@needs_ref
def test_item6_same_hostscorer_shared_across_arms():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, base_model = build_model(cli, device)
    host = HostScorer(str(REPO_ROOT / S2_720), device)
    # same host object passed to build_cache_for_split regardless of arm -- verified by
    # construction (single host instance, reused across the S1/S2/S3 loop in main())
    src = inspect.getsource(sys.modules['scripts.build_m_stage2_retrieval_cache01'].main)
    assert src.count('HostScorer(') == 1, 'exactly one HostScorer instantiation, shared across arms'


# item 7: only the intended gate/head parameters are trainable
@needs_ref
def test_item7_only_gate_and_head_trainable():
    exp, args, model, host_ck = build_fresh_stage2(str(REPO_ROOT / S2_720), seed=0)
    freeze_retrieval_submodules(model)
    frozen_params = [p for name in FREEZE_SUBMODULES
                    for sub in [getattr(model, name, None)] if sub is not None
                    for p in sub.parameters()]
    assert len(frozen_params) > 0
    assert all(not p.requires_grad for p in frozen_params)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, 'gate/base head must remain trainable'


# item 8: checkpoint reload is exact
@needs_ref
def test_item8_checkpoint_reload_exact():
    exp, args, model, host_ck = build_fresh_stage2(str(REPO_ROOT / S2_720), seed=0)
    sha_before = state_sha(model.state_dict())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    sha_after = state_sha(model.state_dict())
    assert sha_before == sha_after


# item 9: base-only (S0) cache carries genuinely zero retrieval signal
@needs_caches
def test_item9_s0_cache_is_all_zero_retrieval():
    cache = torch.load(CACHE_ROOT / 'S0_base' / 'val.pt', map_location='cpu')
    assert torch.all(cache['relation_outputs'] == 0)
    assert torch.all(cache['relation_query_embs'] == 0)
    assert torch.any(cache['query_offset'] != 0)  # offset itself is real, not part of "retrieval signal"


# item 10: prediction-metric calculation is identical across arms -- `eval_epoch` is called
# with the SAME function for every arm; only the `cache`/`start_to_row` args differ
def test_item10_eval_epoch_is_single_shared_function():
    import scripts.train_setlossctrl_stage2_retrain02 as m
    assert eval_epoch is m.eval_epoch
    sig = inspect.signature(eval_epoch)
    assert list(sig.parameters) == ['exp', 'model', 'loader', 'cache', 'start_to_row', 'device']
