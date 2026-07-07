import os
import torch
from models import RAFT, RelationStage1, RelationStage2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'RAFT': RAFT,
            'RelationStage1': RelationStage1,
            'RelationStage2': RelationStage2,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0 (visible devices: {})'.format(os.environ.get("CUDA_VISIBLE_DEVICES", "all")))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
