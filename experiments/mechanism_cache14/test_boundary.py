import sys
import unittest
from unittest.mock import patch


@unittest.skipIf(sys.platform=='win32','Linux /proc boundary checks run on target server CPU')
class BoundaryTests(unittest.TestCase):
    def setUp(self):
        import boundary_probe as b
        self.b=b
        self.owner=dict(pid=100,start='10',state='S',wchan='do_wait',children=['101'],
            cmd='/data/pjh_workspace/ts-env/bin/python -u '+str(b.CONTROLLER)+' --mode all')
        self.worker=dict(pid=101,start='11',state='R',ppid=100,
            cmd='/data/pjh_workspace/ts-env/bin/python -u '+str(b.WORKER)+' Weather_192 V0 P100 --mode analyze')
        self.status=dict(controller_pid=100,worker_pid=101,phase='MECHANISM_DIAGNOSTICS',command=self.worker['cmd'].split())
    def test_valid_wait_pair(self):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_controller_not_waiting(self):
        self.owner['wchan']='other'
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_extra_worker_rejected(self):
        self.owner['children'].append('102')
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_wrong_parent_rejected(self):
        self.worker['ppid']=999
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_solar_rejected(self):
        self.worker['cmd']='python solar_full_train.py'
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_stale_status_rejected(self):
        self.status['worker_pid']=102
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_command_tamper_rejected(self):
        self.status['command'][-1]='gate'
        with self.assertRaises(AssertionError):self.b.validate_pair(self.owner,self.worker,self.status)
    def test_pid_reuse_not_signalled(self):
        with patch.object(self.b,'process',return_value=dict(self.owner,start='different')),patch.object(self.b.os,'kill') as kill:
            self.assertFalse(self.b.resume(self.owner));kill.assert_not_called()
    def test_running_controller_not_signalled(self):
        with patch.object(self.b,'process',return_value=self.owner),patch.object(self.b.os,'kill') as kill:
            self.assertFalse(self.b.resume(self.owner));kill.assert_not_called()
    def test_stopped_matching_controller_resumed(self):
        with patch.object(self.b,'process',return_value=dict(self.owner,state='T')),patch.object(self.b.os,'kill') as kill:
            self.assertTrue(self.b.resume(self.owner));kill.assert_called_once_with(100,self.b.signal.SIGCONT)
    def test_zombie_identity_kept(self):
        with patch.object(self.b,'process',return_value=dict(self.worker,state='Z',cmd='')):
            self.assertIsNotNone(self.b.same_process(self.worker))
    def test_waits_for_owned_probe(self):
        with patch.object(self.b,'owned_probes',side_effect=[[{'pid':222}],[]]),patch.object(self.b,'atomic_json'),patch.object(self.b.time,'sleep'),patch.object(self.b,'resume',return_value=True) as resume:
            self.assertTrue(self.b.safe_resume(self.owner,'/dedicated/out'));resume.assert_called_once()


if __name__=='__main__':unittest.main(verbosity=2)
