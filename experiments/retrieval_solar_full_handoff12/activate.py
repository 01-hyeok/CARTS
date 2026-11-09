"""Pause only the CPU queue, leave its current GPU child uninterrupted."""
import os,signal,subprocess,time
from pathlib import Path
import controller as c

def main():
    ready=c.sources();plan=c.read(c.HERE/'PLAN.json')
    assert not (c.AUDIT/'ACTIVATION.json').exists(),'Already activated; do not create a duplicate'
    for name in ('controller','worker','held_waiter','replacement'):
        actual=c.snapshot(plan[name]['pid']);assert actual
        assert all(actual[k]==plan[name][k] for k in ('pid','start_ticks','command'))
    pid=plan['controller']['pid'];worker=plan['worker']['pid']
    assert c.snapshot(pid)['state']=='S'
    assert Path(f'/proc/{pid}/task/{pid}/children').read_text().split()==[str(worker)]
    assert c.snapshot(worker)['state'] not in ('T','Z')
    # Only this child-waiting CPU parent is paused. The GPU child receives no signal.
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(30):
            if c.snapshot(pid)['state']=='T':break
            time.sleep(.1)
        assert c.snapshot(pid)['state']=='T'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (c.AUDIT/'controller.log').open('ab') as log:
            proc=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python','-u',str(c.HERE/'controller.py')],cwd=c.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(30):
            assert proc.poll() is None,'New CPU waiter failed to start'
            if (c.AUDIT/'STATUS.json').exists():break
            time.sleep(.2)
        state=c.read(c.AUDIT/'STATUS.json')
        assert state['status']=='WAITING_UNINTERRUPTED_MAIN_WORKER' and state['controller_pid']==proc.pid
        c.dump(c.AUDIT/'ACTIVATION.json',dict(status='CPU_WAITER_ACTIVE_BOUNDARY_PENDING',at=c.old.now(),new_controller_pid=proc.pid,paused_cpu_controller=pid,uninterrupted_gpu_worker=worker,worker_start_ticks=c.snapshot(worker)['start_ticks'],gpu_signal_sent=False,training_released=False,plan_sha256=ready['plan_sha256']))
        print((c.AUDIT/'ACTIVATION.json').read_text())
    except BaseException:
        # Restore the previous CPU queue if activation fails before handoff.
        if c.snapshot(pid) and c.snapshot(pid)['state']=='T':os.kill(pid,signal.SIGCONT)
        raise

if __name__=='__main__':main()
