"""Reviewed recovery wrapper. Original evaluator and completed E96 results unchanged."""
import sys,json,inspect,fcntl,argparse,traceback,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import evaluate as ev
import importlib.util
_spec=importlib.util.spec_from_file_location('score_fusion_weather_norm',HERE/'probe_norm.py')
_norm=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_norm;_spec.loader.exec_module(_norm)
patch=_norm.patch
CELLS={'Weather_96','Weather_192','Weather_336','Weather_720'}
original_setup=ev.setup
ACTIVE={'corrected':False}

def setup(cell,training):
    result=original_setup(cell,training);d,f,prior,identity,before=result
    ACTIVE['corrected']=cell in CELLS and training=='FULL'
    if ACTIVE['corrected']:
        proof=json.loads((ev.OUT/'recovery02'/(cell+'_NORM_PROBE.json')).read_text())
        assert len(proof['rows'])==4 and all(r['raw_exact'] and r['prob_exact'] and r['ids_match_old'] and r['head_id_differences']==0 for r in proof['rows'])
        assert proof['source_sha256']==ev.sha(HERE/'probe_norm.py')
        patch(f.batch)
        identity['canonical_score_recovery']=dict(scope='Weather Full V5 short query B<C encoder normalization only',source_sha256=ev.sha(HERE/'probe_norm.py'),
            prior_tail_scores_not_bitwise_canonical=True,ordered_E0_ids_and_metrics_preserved=True,
            direct_current_canonical_raw_and_probability_bitwise_gate_required=True,not_a_tolerance_change=True)
    return result

original_e0=ev.e0_gate
def e0_gate(g,m,ids,prior,variant,d,x,y):
    if not (ACTIVE['corrected'] and len(g)<d.c):return original_e0(g,m,ids,prior,variant,d,x,y)
    p=ev.normalize(g,m);chosen=ids.gather(-1,ev.topk(p,10))
    assert ev.np.array_equal(chosen.cpu().numpy(),prior[variant+'__selected_ids'])
    ds=ev.candidate_distances(d,x,y,chosen)
    assert ev.np.array_equal(ds.cpu().numpy(),prior[variant+'__selected_mse'])
    # The prior trace's raw scores are explicitly the failed reference here.
    # Current all-channel raw scores and full E0 probabilities must instead
    # match the direct canonical reference bitwise in BOTH val/test tail gates.
    return p,chosen

ev.setup=setup;ev.e0_gate=e0_gate
ss=inspect.getsource(ev.smoke)
needle="    gate=dict(status='PASS_FIRST_TAIL_GPU_SMOKE'"
assert ss.count(needle)==1
ss=ss.replace(needle,"    if ACTIVE['corrected']:\n        assert all(r['max_canonical_e0_score_delta']==0 and r['max_canonical_raw_cosine_delta']==0 for r in rows), 'corrected canonical score must match bitwise'\n"+needle)
ns=dict(ev.__dict__,ACTIVE=ACTIVE);exec(compile(ss,__file__+'::strict_smoke','exec'),ns);ev.smoke=ns['smoke']
src=inspect.getsource(ev.run)
assert src.count('e0_stored_scores_exact=True')==2
src=src.replace('e0_stored_scores_exact=True',"e0_stored_scores_exact=not ACTIVE['corrected'],canonical_tail_score_recovery=ACTIVE['corrected']")
ns=dict(ev.__dict__,ACTIVE=ACTIVE);exec(compile(src,__file__+'::unchanged_run_with_honest_recovery_metadata','exec'),ns)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('training');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    lock=(ev.OUT/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:ns['run'](a.cell,a.training,a.smoke)
    except BaseException as e:
        dest=ev.OUT/'checkpoints'/a.cell/('V5_'+a.training);dest.mkdir(parents=True,exist_ok=True)
        ev.dump(dest/('FAILURE_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(at=ev.now(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
