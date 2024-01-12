from abc import abstractmethod
from typing import List, Type, Any
from collections import OrderedDict
from pathlib import Path
import importlib

import torch

if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        pass
    else:
        pass
else:
    pass

import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel

from Commons.output_tool import output_func
from Commons.torch_ddp.ddp_connect import DDPConfig
from Commons.torch_rpc.rpc_connect import RPCConfig, set_master_opt
from Commons.torch_rpc.rpc_data_transfer import get_receiver_object

# from TrainerBase.inner_frame.trainer.concepts import TrainImpl

from Optional_tools import progres_bar
from Optional_tools.lr_scheduler import get_lr_scheduler


class TrainerBase:

    devices: List[torch.device]
    lr: float
    model: torch.nn.Module
    log_file: Path

    train_max_num_batches: int

    # DDP Settings
    global_rank: int

    # RPC Settings
    trainer_rpc_rank: int
    global_rpc_rank: int
    receiver_name: str
    sender_name: str
    rpc_cfg: RPCConfig

    def __init__(self, args, rank, devices: List[torch.device],
                 model_type: Type[torch.nn.Module], model_state_dict: OrderedDict,
                 ddp_cfg: DDPConfig, rpc_cfg: RPCConfig):

        # TODO:Remove
        # assert torch.cuda.is_available()

        self.args = args

        self.proc_rank = rank

        # RPC通信処理
        self.make_global_rank(ddp_cfg.node_num, ddp_cfg.num_devices_per_node, self.proc_rank)
        self.rpc_cfg = rpc_cfg
        self.rpc_connect(rpc_cfg)

        self.receiver = get_receiver_object(self.receiver_name, self.sender_name)

        output_func(self.args.logger,
                    "%s, rpc_connect, %d, %d, %d, %d, %d, %d, %d, %d"
                     %(self.receiver_name, rpc_cfg.rpc_node_num, rpc_cfg.server_num_worker_node, rpc_cfg.server_total_num_nodes,
                       rpc_cfg.trainer_num_devices_per_node, rpc_cfg.trainer_total_num_nodes,
                       rpc_cfg.trainer_world_size, rpc_cfg.server_world_size, rpc_cfg.world_size),
                    self.args.log_file)

        # DDP通信処理
        self.ddp_cfg = ddp_cfg
        self.ddp_connect(ddp_cfg, devices[0])
        self.orig_model = None

        output_func(self.args.logger,
                    "%s, ddp_connect, %d, %d, %d, %d"
                    %(self.receiver_name, ddp_cfg.node_num, ddp_cfg.num_devices_per_node, ddp_cfg.total_num_nodes, ddp_cfg.world_size),
                    self.args.log_file)

        output_func(self.args.logger,
                    "%s, cuda device number(or CPU), node_num=%d, proc_rank=%d, device_mode=%s, device_index=%d"
                    %(self.receiver_name, ddp_cfg.node_num, self.proc_rank, args.dev_mode,
                      devices[0].index if self.args.dev_mode == "cuda" else self.args.dev_mode),
                    self.args.log_file)

        # Set General values
        self.devices = devices
        self.model_type = model_type
        self.lr = args.lr
        self.log_file = Path(args.log_file)
        self.logs = []
        self.firstRun = True
        self.TRIAL_NUM = 0

        if len(self.devices) > 1:
            raise ValueError('Cannot serial train with more than one device.')

        # Set train GNN model(Model computation phase)
        # default=args.model_name(SAGE etc)
        self._set_model()

        # モデルの初期化
        # self.reset()
        self.init_model()

        # modelのload
        self.model.load_state_dict(model_state_dict)

        # print("\nmodel_update:get_share_model_dict:after")
        # for key, value in self.model.state_dict().items():
        #     print(key, value)

        # modelのddp化
        self.init_ddp_model()
        # optimizer
        self._reset_optimizer()



    def __del__(self):
        dist.destroy_process_group()
        rpc.shutdown()

        if len(self.logs) > 0:
            raise RuntimeError('Had unflushed logs when deleting BaseDriver')
        # self.flush_logs()

    # Set train GNN model(Model computation phase)
    # default=args.model_name(SAGE etc)
    def _set_model(self):
        self.model = self.model_type(
            self.args.num_features, self.args.hidden_features,
            self.args.num_classes,
            num_layers=self.args.num_layers).to(self.main_device)

        self.model_noddp = self.model_type(
            self.args.num_features, self.args.hidden_features,
            self.args.num_classes,
            num_layers=self.args.num_layers).to(self.main_device)

    def get_model_dict(self):
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.cpu().module.state_dict()
        else:
            model_state_dict = self.model.cpu().state_dict()

        return model_state_dict

    # def _set_model_from_ps(self):
    #     model_state_dict = remote_get_model_from_ps(self.ps_rref, self.devices[0])
    #     self.model.load_state_dict(model_state_dict)
    #     # print("\nmodel_update:get_share_model_dict:before")
    #     # for key, value in model_state_dict.items():
    #     #     print(key, value)

    def get_minibatch(self, mode, idx):
        flg_id, inputs = self.receiver.get_minibatch_from_server(mode, idx)
        return flg_id, inputs


    @abstractmethod
    def train_impl(self, lr_scheduler=None, cb=None):
        pass

    # Trainのフレームワーク
    def train(self, epochs) -> None:
        self.model.train()

        # print("\nmodel_update:get_share_model_dict:after")
        # for key, value in self.model.state_dict().items():
        #     print(self.receiver.name, "model_dict", key, value)

        # Optional lr_scheduler
        # lr_scheduler = self.set_lr_scheduler()

        for epoch in epochs:

            # Optional プログレスバー処理
            # cb, pbar = self.set_pbar(epoch)

            output_func(self.args.logger, "%s, %d, Start serial_train_ns" % (self.receiver.name, self.receiver.epoch),
                        self.args.log_file)

            # TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
            self.receiver.set_epoch(epoch)
            self.train_impl()

            # Optional プログレスバー処理
            # if self.is_main_proc and self.args.pbar and pbar is not None:
            #     progres_bar.close_pbar(pbar)

            # Barrier
            if dist.is_initialized():
                dist.barrier()

    # Set test function(Model computation phase)
    @abstractmethod
    def test_impl(self, name, lr_scheduler=None, cb=None):
        pass

    # Testのフレームワーク
    def test(self, sets):
        self.model.eval()

        results = {}
        for name in sets:
            # cb = None

            if hasattr(self.model, 'module'):
                self.model_noddp.load_state_dict(self.model.module.state_dict())
            else:
                self.model_noddp.load_state_dict(self.model.state_dict())

            output_func(self.args.logger, "%s, %s, %d, Start test_ns" % (self.receiver.name, name, self.receiver.epoch),
                        self.args.log_file)
            result = self.test_impl(name)
            results[name] = result

        return results


    # モデルと最適化関数のReset処理
    def _reset_model(self):
        if self.ps_mode is False:
            if self.orig_model is None:
                self.orig_model = self.model
            self.orig_model.reset_parameters()
            # TODO:CPU
            self.model = DistributedDataParallel(
                self.orig_model, device_ids=[self.main_device] if self.args.dev_mode == "cuda" else None)
            # , find_unused_parameters=True)

    def _reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print("Reset optimizer")

    def reset(self):
        self._reset_model()
        self._reset_optimizer()
        self.TRIAL_NUM += 1

    def init_model(self):
        if self.orig_model is None:
            self.orig_model = self.model
        self.orig_model.reset_parameters()

    def init_ddp_model(self):
        # print("before: init_ddp_model")
        self.model = DistributedDataParallel(
            self.orig_model, device_ids=[self.main_device] if self.args.dev_mode == "cuda" else None)
        # , find_unused_parameters=True)
        # print("after: init_ddp_model")


    # DDP処理
    def make_global_rank(self, node_num, num_devices_per_node, index):
        self.global_rank = (
                node_num * num_devices_per_node + index
        )
    def ddp_connect(self, ddp_cfg, device):
        # self.global_rank = (
        #         ddp_cfg.node_num * ddp_cfg.num_devices_per_node + device.index
        # )

        backend = "nccl" if self.args.dev_mode == "cuda" else "gloo"
        # DDP Init Process
        dist.init_process_group(
            backend, rank=self.global_rank, world_size=ddp_cfg.world_size)

    # その他DDP関連処理
    @property
    def my_name(self):
        return f'{super().my_name}_{self.ddp_cfg.node_num}_' \
               f'{self.main_device.index if self.args.dev_mode == "cuda" else self.args.dev_mode}'

    @property
    def is_main_proc(self):
        return self.global_rank == 0

    # device(GPU番号)取得処理
    @property
    def main_device(self) -> torch.device:
        if self.args.dev_mode == "cuda":
            return self.devices[0]
        elif self.args.dev_mode == "cpu":
            return self.args.dev_mode

    # Logger関連処理
    def log(self, t) -> None:
        self.logs.append(t)
        if self.is_main_proc and self.args.verbose:
            print(str(t))

    def flush_logs(self) -> None:
        if len(self.logs) == 0:
            return

        with self.log_file.open('a') as f:
            f.writelines(repr(item) + '\n' for item in self.logs)
        self.logs = []

    # Optional lr_scheduler
    def set_lr_scheduler(self):
        if self.args.use_lrs:
            lr_scheduler = get_lr_scheduler(self.optimizer, self.args.patience, self.args.use_lrs)
        else:
            lr_scheduler = None
        return lr_scheduler

    # Optional pbar
    def set_pbar(self, epoch):
        if self.is_main_proc and self.args.pbar:
            cb, pbar = progres_bar.get_pbar(f'Train epoch {epoch}', self.train_loader.node_idx.numel())
        else:
            cb, pbar = None, None
        return cb, pbar


    # RPC処理
    def rpc_connect(self, rpc_cfg: RPCConfig):
        self.trainer_rpc_rank = self.global_rank

        # self.global_rpc_rank = (
        #         rpc_cfg.parameter_server +
        #         rpc_cfg.server_total_num_nodes * rpc_cfg.server_num_worker_node + self.trainer_rpc_rank
        # )

        self.global_rpc_rank = rpc_cfg.model_store_server + self.trainer_rpc_rank

        self.set_data_transfer_name()
        output_func(self.args.logger,
                    "%s, rpc_node_num=%d, trainer_rpc_rank=%d, global_rpc_rank=%d"
                    % (self.receiver_name, self.rpc_cfg.rpc_node_num, self.trainer_rpc_rank, self.global_rpc_rank),
                    self.args.log_file)

        rpc.init_rpc(
            name=self.receiver_name,
            rank=self.global_rpc_rank,
            world_size=rpc_cfg.world_size,
            rpc_backend_options=set_master_opt(self.args.rpc_addr, self.args.rpc_port)
        )

        # winfo = rpc.get_worker_info()
        # rpc_val = rpc.BackendType
        # rpc_opt = rpc.RpcBackendOptions

    def set_data_transfer_name(self):
        self.receiver_name = f"trainer_{self.trainer_rpc_rank}"
        self.sender_name = f"server_{self.trainer_rpc_rank}"

