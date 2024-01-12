import torch

import importlib

if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler

from GraphStoreServer.commons.dataset import PyGDataset

from Commons.torch_rpc.rpc_connect import RPCConfig
from GraphStoreServer.sampler.sampler_base_server import SamplerBaseServer


# TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
class NSServer(SamplerBaseServer):
    def __init__(self, args, num_worker: int, dataset: PyGDataset, rpc_cfg: RPCConfig, device: torch.device):

        super().__init__(args, num_worker, dataset, rpc_cfg, [device])
        # self.minibatch_store = {}

    # Point: Sampler定義(Sampler phase)
    # default=NeighborSampler
    def _set_sampler(self):
        self.train_max_num_batches = self.args.train_max_num_batches

        # train sampler
        kwargs = dict(sampler=self.get_dist_sampler(self.TRIAL_NUM * 1000 + self.global_rpc_rank, "train"),
                      persistent_workers=True)
        self.train_loader = NeighborSampler(
            self.dataset.adj_t(), node_idx=self.dataset.split_idx['train'],
            batch_size=self.args.train_batch_size, sizes=self.args.train_fanouts,
            num_workers=self.args.num_workers, **kwargs)

        # test sampler
        kwargs = dict(sampler=self.get_dist_sampler(self.TRIAL_NUM * 1000 + self.global_rpc_rank, "test"),
                      persistent_workers=True)
        self.test_loader = NeighborSampler(
            self.dataset.adj_t(), node_idx=self.dataset.split_idx['test'],
            batch_size=self.args.final_test_batchsize, sizes=self.args.final_test_fanouts,
            num_workers=self.args.num_workers, **kwargs)

        # valid sampler
        kwargs = dict(sampler=self.get_dist_sampler(self.TRIAL_NUM * 1000 + self.global_rpc_rank, "valid"),
                      persistent_workers=True)
        self.val_loader = NeighborSampler(
            self.dataset.adj_t(), node_idx=self.dataset.split_idx['valid'],
            batch_size=self.args.test_batch_size, sizes=self.args.batchwise_test_fanouts,
            num_workers=self.args.num_workers, **kwargs)

    # Set transferer(Sampler phase, make devices and sampler)
    # default=None
    def _set_transferer(self):
        pass

    # Train用sampling関連メソッド
    def get_train_sampler(self, epoch):
        # Set shuffle slicing node ids
        self.train_loader.node_idx = self.get_idx(epoch)
        devit = self.train_loader
        return devit

    # Test用sampling関連メソッド
    def get_test_sampler(self, name):
        devit = None
        if name == 'test':
            devit = self.test_loader
        if name == 'valid':
            devit = self.val_loader

        devit.node_idx = self.get_idx_test(name)
        return devit

    # def main(self, args, train_mode):
    #     delta = min(args.test_epoch_frequency, args.epochs)
    #     do_eval = args.epochs >= args.test_epoch_frequency
    #
    #     print("%s, %s, Start Sampler loop process" %(self.sender_name, train_mode))
    #     for epoch_1 in range(0, args.epochs, delta):
    #         for epoch_2 in range(epoch_1, epoch_1 + delta):
    #             sampler = self.get_train_sampler(epoch_2)
    #             self.minibatch_loop(train_mode, sampler, epoch_2)
    #
    #         if do_eval:
    #             test_mode = "valid"
    #             sampler = self.get_test_sampler(test_mode)
    #             self.minibatch_loop(test_mode, sampler, epoch_1 + delta - 1)
    #
    #     for test_mode in ('valid', 'test'):
    #         sampler = self.get_test_sampler(test_mode)
    #         self.minibatch_loop(test_mode, sampler, args.epochs)
    #
    #     print("%s, %s, End Sampler loop process" %(self.sender_name, train_mode))
    #
    # def minibatch_loop(self, train_mode, sampler, epoch):
    #     dataset = self.dataset
    #     idx = 0
    #     print("%s, %s, Sampling Start, epoch=%d, idx=%d" % (self.sender_name, train_mode, epoch, idx))
    #     for inputs in sampler:
    #         # POINT Sender
    #         send_data = self.set_minibatch(inputs, idx)
    #         self.set_minibatch_store(train_mode, epoch, send_data)
    #
    #         idx += 1
    #
    #     self.set_minibatch_store(train_mode, epoch, [-1, []])
    #     print("%s, %s, Sampling End, epoch=%d, idx=%d" % (self.sender_name, train_mode, epoch, idx))

    # POINT: samplerから取得したinputをtrainer送信用のミニバッチに変換する処理。
    def set_minibatch(self, inputs):
        # POINT Sender
        batch_size, n_id, adjs = inputs

        xs = torch.empty(len(n_id), self.dataset.x.shape[1], dtype=self.dataset.x.dtype,
                         layout=self.dataset.x.layout)
        torch.index_select(self.dataset.x, 0, n_id, out=xs)
        ys = torch.empty(batch_size, dtype=self.dataset.y.dtype,
                         layout=self.dataset.y.layout)
        torch.index_select(self.dataset.y, 0, n_id[:batch_size], out=ys)

        return [batch_size, n_id, adjs, xs, ys] # 送信用ミニバッチ(n_idに対応するfeatureと正解ラベルを追加)をリターンする

