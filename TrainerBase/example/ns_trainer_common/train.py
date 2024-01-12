from typing import Optional

import torch
import torch.nn.functional as F

from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_data_transfer import RPCDataReceiver

from TrainerBase.inner_frame.trainer.concepts import TrainCallback
from Optional_tools.lr_scheduler import update_lr_by_SALIENT
from Optional_tools import utils
from Optional_tools.utils_decorator import rs_process_epoch, rs_process_region


import torch.distributed


def serial_train_ns(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    receiver: RPCDataReceiver,
                    log_file=None,
                    lr_scheduler = None,
                    cb: Optional[TrainCallback] = None,
                    devices=None) -> None:

    ''' Serial training code that uses PyG's NeighborSampler '''
    model.train()
    logger = True if log_file is not None else False

    if devices is not None:
        assert len(devices) == 1
        device = devices[0]

    # iterator = iter(devit)
    # print("%s, %d, Start serial_train_ns" %(receiver.name, receiver.epoch))
    idx = 0
    while True:

        # TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
        # inputs = next(iterator, [])
        flg_id, inputs = receiver.get_minibatch_from_server("train", idx)
        if flg_id < 0 and len(inputs) == 0:
            break

        batch_size, n_id, adjs, xs, ys = inputs

        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]

        optimizer.zero_grad()
        out = model(xs, adjs)
        loss = F.nll_loss(out, ys)
        loss.backward()
        result = loss
        optimizer.step()

        # Optional  学習率調整関数
        update_lr_by_SALIENT(lr_scheduler, result)

        # Optional プログレスバー処理
        if cb is not None:
            cb(sum(bs[0] for bs in [inputs]))

        # print("%s, epoch=%d, idx=%d, loss=%f" %(receiver.name, receiver.epoch, idx, result.item()))
        output_func(logger, "%s, epoch=%d, idx=%d, loss=%f" %(receiver.name, receiver.epoch, idx, result.item()),
                    log_file)

        idx += 1


# Optional RuntimeStatics Function
# @decorator_rs_process_epoch
@rs_process_epoch("SamplerBase ogbn-arxiv")
def serial_train_ns_rs(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    receiver: RPCDataReceiver,
                    lr_scheduler,
                    cb: Optional[TrainCallback]=None,
                    dataset=None,
                    devices=None) -> None:

    ''' Serial training code that uses PyG's NeighborSampler '''
    model.train()


    if devices is not None:
        assert len(devices) == 1
        device = devices[0]

    idx = 0
    while True:
        do_flg = batch_loop_process(model, optimizer, receiver, lr_scheduler, cb, dataset, device, idx)
        if not do_flg: break
        idx += 1


# @decorator_rs_process_region # "total"
@rs_process_region("total", True, False)
def batch_loop_process(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       receiver: RPCDataReceiver,
                       lr_scheduler,
                       cb: Optional[TrainCallback] = None,
                       dataset=None,
                       device=None,
                       idx=None):

    # @decorator_rs_process_region
    @rs_process_region("sampling", True, False)
    def sampling_inputs(idx):

        flg_id, inputs = receiver.get_minibatch_from_server("train", idx)
        if flg_id < 0 and len(inputs) == 0:
            return False, ()

        batch_size, n_id, adjs = inputs
        xs = torch.empty(len(n_id), dataset.x.shape[1], dtype=dataset.x.dtype,
                         layout=dataset.x.layout, pin_memory=True)
        torch.index_select(dataset.x, 0, n_id, out=xs)
        ys = torch.empty(batch_size, dtype=dataset.y.dtype,
                         layout=dataset.y.layout, pin_memory=True)
        torch.index_select(dataset.y, 0, n_id[:batch_size], out=ys)

        return True, (inputs, batch_size, n_id, adjs, xs, ys)

    # do_flg, ret = sampling_inputs("sampling", True, False)
    do_flg, ret = sampling_inputs(idx)
    if not do_flg: return False
    inputs, batch_size, n_id, adjs, xs, ys = ret


    # @decorator_rs_process_region
    @rs_process_region("data_transfer", True, False)
    def data_transfer(xs, ys, adjs):
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]
        return xs, ys, adjs

    xs, ys, adjs = data_transfer(xs, ys, adjs)


    # @decorator_rs_process_region
    @rs_process_region("train", True, False)
    def train_core():
        optimizer.zero_grad()
        out = model(xs, adjs)
        loss = F.nll_loss(out, ys)
        loss.backward()
        result = loss
        optimizer.step()

        # Optional  学習率調整関数
        update_lr_by_SALIENT(lr_scheduler, result)

        # Optional プログレスバー処理
        if cb is not None:
            cb(sum(bs[0] for bs in [inputs]))

    train_core()

    return True


def serial_train(model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 receiver: RPCDataReceiver,
                 log_file=None,
                 lr_scheduler=None,
                 cb: Optional[TrainCallback]=None,
                 devices=None) -> None:

    # Optional 処理時間の統計データ出力
    if utils.is_performance_stats_enabled():
        serial_train_ns_rs(model, optimizer, receiver, lr_scheduler,
                           cb=cb, dataset=None, devices=devices)
    # Simple serial_train_ns
    else:
        serial_train_ns(model, optimizer, receiver, log_file=log_file, lr_scheduler=lr_scheduler,
                        cb=cb, devices=devices)

    return



