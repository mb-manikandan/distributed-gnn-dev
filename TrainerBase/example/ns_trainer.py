from typing import Type, List
import torch
from collections import OrderedDict

import importlib

if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler


from Commons.torch_rpc.rpc_connect import RPCConfig
from Commons.torch_ddp.ddp_connect import DDPConfig

from TrainerBase.inner_frame.trainer_base import TrainerBase
from TrainerBase.example.ns_trainer_common import train, test


class NSTrainerBase(TrainerBase):

    def __init__(self, args, rank, devices: List[torch.device],
                 model_type: Type[torch.nn.Module], model_state_dict: OrderedDict,
                 ddp_cfg: DDPConfig, rpc_config: RPCConfig):

        super().__init__(args, rank, devices, model_type, model_state_dict, ddp_cfg, rpc_config)


    # Set train function(Model computation phase)
    # serial_train(NeighborSampler)
    def train_impl(self, lr_scheduler=None, cb=None):
        return train.serial_train(
            self.model, self.optimizer, self.receiver,
            log_file=self.args.log_file if self.args.logger else None, devices=[self.main_device]
        )

    # Set test function(Model computation phase)
    # test_ns(NeighborSampler)
    def test_impl(self, name, lr_scheduler=None, cb=None):
        return test.test_ns(self.model_noddp, self.receiver, [self.main_device], name)

