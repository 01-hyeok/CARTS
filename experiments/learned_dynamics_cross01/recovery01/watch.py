"""Restart one existing monitor with recovery progress routing, no worker control."""
import sys,json,importlib.util,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from experiment_common import OUT,dump,now
import report_notes
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
watch=module('original_monitor',HERE.parent/'monitoring/watch.py')
final=module('finalize_all',HERE.parent/'monitoring/finalize_all.py');original_finish=final.main
def finish():
    original_finish();report_notes.annotate(OUT/'final_all_comparisons')
final.main=finish;sys.modules['finalize_all']=final
original_read=watch.read
def read(path):
    if path.name=='CURRENT.json':
        state=original_read(OUT/'CONTROLLER_STATUS.json') or {}
        if state.get('progress_path'):return original_read(Path(state['progress_path']))
    return original_read(path)
watch.read=read
if __name__=='__main__':
    lock=(OUT/'recovery01/monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:watch.main()
    except BaseException as e:
        dump(OUT/'MONITOR_STATUS.json',dict(status='MONITOR_ERROR',error=repr(e),traceback=traceback.format_exc(),at=now(),workers_untouched=True));raise
