import torch
import torch.distributed as dist

from Commons.output_tool import output_func


def test_ns(model, receiver, devices, test_name, cb=None):
    if devices is not None:
        assert len(devices) == 1
        device = devices[0]

    # results = torch.empty(len(devit), dtype=torch.long, pin_memory=True)
    results = []
    total = 0

    # iterator = iter(devit)

    # print("%s, %s, %d, Start test_ns" % (receiver.name, test_name, receiver.epoch))
    i = 0
    while True:

        # TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
        flg_id, inputs = receiver.get_minibatch_from_server(test_name, i)
        if flg_id < 0 and len(inputs) == 0:
            break

        batch_size, n_id, adjs, xs, ys = inputs

        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]
        out = model(xs, adjs)

        out = out.argmax(dim=-1, keepdim=True).reshape(-1)
        correct = (out == ys).sum()

        results.append(torch.tensor([correct]))
        total += batch_size
        i += 1

    # TODO: CPU処理できるように修正
    if device == "cpu":
        dist.barrier()
    else:
        torch.cuda.current_stream(device).synchronize()

    result = torch.cat(results, dim=0).sum().item()


    if dist.is_initialized():
        output_0 = torch.tensor([result]).to(device)
        output_1 = torch.tensor([total]).to(device)
        _ = dist.all_reduce(output_0)
        _ = dist.all_reduce(output_1)
        result, total = output_0.item(), output_1.item()

    return result/total

