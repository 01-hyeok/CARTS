import sys,traceback
import full_precision_runtime
sys.path.insert(0,str(full_precision_runtime.ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_inference as inference
if __name__=='__main__':
    assert sys.argv[1]=='FULL'
    try:inference.run(sys.argv[1],sys.argv[2])
    except BaseException as e:
        inference.dump(inference.OUT/'IMMEDIATE_ISSUE.json',dict(status='FAILED',support='FULL',cell=sys.argv[2],error=repr(e),traceback=traceback.format_exc()));raise
