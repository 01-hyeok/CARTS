"""Run only the missing E96 Full V5 condition after exact population validation."""
import sys,inspect,json,fcntl,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
from exact_patch import pop,install,OUT
from source_contract import transform

def main(smoke):
    patch=install();gate=OUT/'VALIDATION_GATE.json';g=json.loads(gate.read_text())
    assert g['status']=='PASS_ALL_QUERY_NATIVE_AND_CONTROLLED_ORDERED_IDS' and g['patch']==patch
    assert g['queries']==dict(val=2785,test=2785) and g['channels']==7 and g['bank_count']==8449
    assert g['checkpoint_unchanged'] and pop.sha(HERE/'validate.py')==g['source_sha256']
    proof=dict(validation_gate_sha256=pop.sha(gate),patch=patch,
        source_hashes={n:pop.sha(HERE/n) for n in ['recover_run.py','source_contract.py']},
        prior_results_unchanged=True,scope='Only missing ETTh1_96 V5_FULL; only B1 kernel differs',
        canonical_full_and_controlled_all_query_validation=True)
    namespace=dict(pop.__dict__,RECOVERY_PROVENANCE=proof)
    exec(compile(transform(inspect.getsource(pop.run)),__file__+'::original_run_plus_provenance','exec'),namespace)
    namespace['run']('ETTh1_96','FULL',smoke)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    lock=(pop.BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    main(args.smoke)
