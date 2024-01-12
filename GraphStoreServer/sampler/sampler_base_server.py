from abc import abstractmethod
from typing import List, Any
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

import torch.distributed.rpc as rpc

from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_connect import RPCConfig, set_master_opt
from Commons.torch_rpc import rpc_data_transfer
from GraphStoreServer.sampler.dist_shuffler.shufflers import DistributedShuffler
from GraphStoreServer.commons.dataset import PyGDataset

from Optional_tools import progres_bar


# TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
class SamplerBaseServer:

    devices: List[torch.device]
    dataset: PyGDataset
    lr: float
    model: torch.nn.Module
    log_file: Path

    # RPC Settings
    server_rpc_rank: int
    global_rpc_rank: int
    receiver_name: str
    sender_name: str
    rpc_cfg: RPCConfig

    # Sampler
    train_loader: Any # Sampler Class Type(NeighborSampler)
    test_loader: Any  # Sampler Class Type(NeighborSampler)
    val_loader: Any   # Sampler Class Type(NeighborSampler)

    def __init__(self, args, num_worker: int,
                 dataset: PyGDataset, rpc_cfg: RPCConfig, devices: List[torch.device]):

        self.args = args
        self.log_file = Path(args.log_file)

        # RPC通信処理
        self.rpc_cfg = rpc_cfg
        self.rpc_connect(rpc_cfg, num_worker)
        log_file = self.log_file if self.args.logger else None

        self.sender = rpc_data_transfer.get_sender_object(self.sender_name, self.receiver_name, log_file)

        output_func(self.args.logger,
                    "%s, rpc_connect, %d, %d, %d, %d, %d, %d, %d, %d"
                    % (self.sender_name, rpc_cfg.rpc_node_num, rpc_cfg.server_num_worker_node,
                       rpc_cfg.server_total_num_nodes,
                       rpc_cfg.trainer_num_devices_per_node, rpc_cfg.trainer_total_num_nodes,
                       rpc_cfg.trainer_world_size, rpc_cfg.server_world_size, rpc_cfg.world_size),
                    self.args.log_file)

        # Set General values
        self.devices = devices
        self.dataset = dataset
        self.logs = []
        self.firstRun = True
        self.TRIAL_NUM = 0

        if len(self.devices) > 1:
            raise ValueError('Cannot serial train with more than one device.')

        # Set Sampler(Sampler phase)
        self._set_sampler()

        # Set transferer(Sampler phase, make devices and sampler)
        self._set_transferer()

        self.set_shuffler()


    def __del__(self):
        rpc.shutdown()

        # pass
        # if len(self.logs) > 0:
        #     raise RuntimeError('Had unflushed logs when deleting BaseDriver')
        # self.flush_logs()


    def set_shuffler(self):
        # DistributedShufflerの定義(Sampler phase)
        self.train_shuffler = DistributedShuffler(
            self.dataset.split_idx['train'], self.rpc_cfg.server_world_size)
        self.test_shuffler = DistributedShuffler(
            self.dataset.split_idx['test'], self.rpc_cfg.server_world_size)
        self.valid_shuffler = DistributedShuffler(
            self.dataset.split_idx['valid'], self.rpc_cfg.server_world_size)

    # Set Sampler(Sampler phase)
    @abstractmethod
    def _set_sampler(self):
        pass

    # Set transferer(Sampler phase, make devices and sampler)
    @abstractmethod
    def _set_transferer(self):
        pass

    # Sampler関連の処理
    def get_idx_test(self, name):
        if name == 'test':
            return self.test_shuffler.get_idx(self.server_rpc_rank)
        elif name == 'valid':
            return self.valid_shuffler.get_idx(self.server_rpc_rank)
        else:
            raise ValueError('invalid test dataset name')

    def get_idx(self, epoch: int):
        self.train_shuffler.set_epoch(10000*self.TRIAL_NUM + epoch)
        return self.train_shuffler.get_idx(self.server_rpc_rank)

    # DistributedSampler設定処理
    def get_dist_sampler(self, _seed, name):
        return torch.utils.data.distributed.DistributedSampler(
            self.dataset.split_idx[name],
            num_replicas=self.rpc_cfg.server_world_size,
            rank=self.server_rpc_rank, seed=_seed)

    # Train用sampling関連メソッド
    @abstractmethod
    def get_train_sampler(self, epoch):
        pass

    # Test用sampling関連メソッド
    @abstractmethod
    def get_test_sampler(self, name):
        pass

    def main(self, args, train_mode):
        delta = min(args.test_epoch_frequency, args.epochs)
        do_eval = args.epochs >= args.test_epoch_frequency

        output_func(self.args.logger, "%s, %s, Start Sampler loop process" % (self.sender_name, train_mode),
                    self.args.log_file)
        for epoch_1 in range(0, args.epochs, delta):
            for epoch_2 in range(epoch_1, epoch_1 + delta):
                sampler = self.get_train_sampler(epoch_2)
                self.minibatch_loop(train_mode, sampler, epoch_2)

            if do_eval:
                test_mode = "valid"
                sampler = self.get_test_sampler(test_mode)
                self.minibatch_loop(test_mode, sampler, epoch_1 + delta - 1)

        for test_mode in ('valid', 'test'):
            sampler = self.get_test_sampler(test_mode)
            # self.minibatch_loop(test_mode, sampler, args.epochs)
            # 最終テスト処理はepoch=-1を渡す。
            self.minibatch_loop(test_mode, sampler, -1)


        output_func(self.args.logger, "%s, %s, End Sampler loop process" % (self.sender_name, train_mode),
                    self.args.log_file)

    def minibatch_loop(self, train_mode, sampler, epoch):
        dataset = self.dataset
        idx = 0
        output_func(self.args.logger, "%s, %s, Sampling Start, epoch=%d, idx=%d" % (self.sender_name, train_mode, epoch, idx),
                    self.args.log_file)
        for inputs in sampler:
            # POINT Sender
            send_data = self.set_minibatch(inputs)
            self.set_minibatch_store(train_mode, epoch, [idx, send_data])
            idx += 1

        self.set_minibatch_store(train_mode, epoch, [-1, []])
        output_func(self.args.logger,
                    "%s, %s, Sampling End, epoch=%d, idx=%d" % (self.sender_name, train_mode, epoch, idx),
                    self.args.log_file)

    # POINT: samplerから取得したinputをtrainer送信用のミニバッチに変換する処理。
    @abstractmethod
    def set_minibatch(self, inputs):
        return inputs # 送信用ミニバッチをリターンする


    # RPC処理
    def rpc_connect(self, rpc_cfg: RPCConfig, num_worker):
        self.server_rpc_rank = (
                rpc_cfg.rpc_node_num * rpc_cfg.server_num_worker_node + num_worker
        )
        # self.global_rpc_rank = rpc_cfg.parameter_server + self.server_rpc_rank
        self.global_rpc_rank = (
                rpc_cfg.model_store_server + rpc_cfg.trainer_world_size + self.server_rpc_rank
        )

        self.set_data_transfer_name()
        output_func(self.args.logger,
                    "%s, rpc_node_num=%d, server_rpc_rank=%d, global_rpc_rank=%d"
                    % (self.sender_name, self.rpc_cfg.rpc_node_num, self.server_rpc_rank, self.global_rpc_rank),
                    self.args.log_file)

        rpc_backend_options = set_master_opt(self.args.rpc_addr, self.args.rpc_port)

        rpc.init_rpc(
            name=self.sender_name,
            rank=self.global_rpc_rank,
            world_size=rpc_cfg.world_size,
            rpc_backend_options=rpc_backend_options
        )

    def set_data_transfer_name(self):
        self.receiver_name = f"trainer_{self.server_rpc_rank}"
        self.sender_name = f"server_{self.server_rpc_rank}"

    def set_minibatch_store(self, train_mode, epoch, inputs):
        self.sender.set_minibatch_store(train_mode, epoch, inputs)

    # def send_minibatch_to_trainer(self, train_mode, epoch, idx):
    #     self.sender.send_minibatch_to_trainer(train_mode, epoch, idx)

    # device(GPU番号)取得処理
    @property
    def main_device(self) -> torch.device:
        return self.devices[0]

    # # Logger関連処理
    # def log(self, t) -> None:
    #     self.logs.append(t)
    #     if self.args.verbose:
    #         print(str(t))
    #
    # def flush_logs(self) -> None:
    #     if len(self.logs) == 0:
    #         return
    #
    #     with self.log_file.open('a') as f:
    #         f.writelines(repr(item) + '\n' for item in self.logs)
    #     self.logs = []

    # Optional pbar
    def set_pbar(self, epoch):
        if self.args.pbar:
            cb, pbar = progres_bar.get_pbar(f'Train epoch {epoch}', self.train_loader.node_idx.numel())
        else:
            cb, pbar = None, None
        return cb, pbar
