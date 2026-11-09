import unittest
from unittest.mock import patch
import pause_at_epoch as p


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.owner=dict(pid=1,start='a',cmd='python /experiments/solar_parallel13/controller_parallel.py',state='S',wchan='do_wait',children=['2'])
        self.worker=dict(pid=2,start='b',cmd='python /experiments/solar_parallel13/solar_full_train_parallel.py train Solar_336 --heads 5',state='R',ppid=1)
        self.status=dict(controller_pid=1,worker_pid=2,target='Solar_336_V5')
    def test_valid(self):p.validate(self.owner,self.worker,self.status)
    def test_other_worker_rejected(self):
        self.worker['cmd']='python Mechanism02 worker'
        with self.assertRaises(AssertionError):p.validate(self.owner,self.worker,self.status)
    def test_wrong_target_rejected(self):
        self.status['target']='Solar_720_V5'
        with self.assertRaises(AssertionError):p.validate(self.owner,self.worker,self.status)
    def test_extra_child_rejected(self):
        self.owner['children'].append('3')
        with self.assertRaises(AssertionError):p.validate(self.owner,self.worker,self.status)
    def test_pid_reuse_never_signalled(self):
        with patch.object(p,'process',return_value=dict(self.worker,start='new')),patch.object(p.os,'kill') as kill:
            with self.assertRaises(AssertionError):p.checked_signal(self.worker,19)
            kill.assert_not_called()
    def test_waits_for_stop_delivery(self):
        with patch.object(p,'matches',side_effect=[dict(self.worker,state='R'),dict(self.worker,state='T')]),patch.object(p.time,'sleep'):
            self.assertEqual(p.await_stopped(self.worker)['state'],'T')
    def test_same_split_rate(self):
        self.assertEqual(p.rate([dict(split='test',batch=1,seconds=0,at='a'),dict(split='test',batch=2,seconds=8,at='b')])['queries_per_second'],4)
    def test_split_mixing_rejected(self):
        self.assertIsNone(p.rate([dict(split='val',batch=1,seconds=0,at='a'),dict(split='test',batch=2,seconds=8,at='b')]))
    def test_gap_rejected(self):
        self.assertIsNone(p.rate([dict(split='test',batch=1,seconds=0,at='a'),dict(split='test',batch=3,seconds=8,at='b')]))


if __name__=='__main__':unittest.main(verbosity=2)
