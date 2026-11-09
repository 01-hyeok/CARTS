"""External Calibration adapter interface; never launches workers or changes original paths."""
from recovery_runtime import *
from recovery02_validate import targets
def resolve(mode,cell,training,pool=None):
    wanted=[(p,l) for p,l in targets() if l['mode']==mode and l['cell']==cell and l['training']==training and l['pool']==pool]
    assert len(wanted)==1,('Unknown or ambiguous identity',mode,cell,training,pool)
    p,label=wanted[0]
    final=read(OUT/'final_all_comparisons/COMPLETE.json')
    assert final['status']=='PASS_ALL_REQUESTED_COMPARISONS_AND_REPORTS'
    assert sha(OUT/'FINAL_VALIDATION.json')==final['validation_sha256']
    m=read(OUT/'FINAL_VALIDATION.json');record=next(r for r in m['jobs'] if r['label']==label)
    assert record['path']==str(p)
    return record
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['native','pool','cross']);a.add_argument('cell');a.add_argument('training',choices=['P100','FULL']);a.add_argument('--pool');x=a.parse_args()
    try:print(json.dumps(resolve(x.mode,x.cell,x.training,x.pool)))
    except FileNotFoundError as e:print(json.dumps(dict(status='NOT_READY',reason=str(e))));sys.exit(2)
    except Exception as e:print(json.dumps(dict(status='INVALID',error=repr(e))));sys.exit(1)
