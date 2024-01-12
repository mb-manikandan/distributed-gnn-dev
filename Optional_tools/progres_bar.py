import torch
import time
from tqdm import tqdm


# Progress Barの設定(Barのヘッダー, 100%となる時の値)
def get_pbar(pbar_mes, total):
    # if self.args.train_sampler == 'NeighborSampler':
    #     pbar = tqdm(total=self.train_loader.node_idx.numel())
    # else:
    #     pbar = tqdm(total=self.train_loader.idx.numel())
    # pbar.set_description(f'Train epoch {epoch}')

    pbar = tqdm(total=total)
    pbar.set_description(pbar_mes)

    # def cb(inputs):
    #     pbar.update(sum(batch.batch_size for batch in inputs))
    #
    # def cb_NS(inputs):
    #     pbar.update(sum(bs[0] for bs in inputs))

    # cb関数
    def cb(update_value):
        pbar.update(update_value)

    return cb, pbar


# Progress Barの更新(cb関数を指定, 更新数(指定した数だけBarが進む))
# def update_pbar(cb, update_value):
#     if cb is not None:
#         cb(update_value)


# Progress Barの更新(pbar = tqdm(total=total)クラスの指定)
def close_pbar(pbar):
    pbar.close()
    del pbar


###########################################################################
# Progress Barの使用例
#
def example_func0():
    epoch_all = 10
    cb, pbar = get_pbar(f'Train epoch', epoch_all) # Progress Barの設定(Barのヘッダー, 100%となる時の値)
    for epoch in range(epoch_all):

        print(epoch)
        time.sleep(2)
        # update_pbar(cb, 1) # Progress Barの更新(cb関数を指定, 更新数(指定した数だけBarが進む))
        if cb is not None:
            cb(1)

    close_pbar(pbar)
    result = None
    return result

# example_func0()
# Train epoch:   0%|                                       | 0/10 [00:00<?, ?it/s]0
# Train epoch:  10%|███                            | 1/10 [00:02<00:18,  2.00s/it]1
# Train epoch:  20%|██████▏                        | 2/10 [00:04<00:16,  2.00s/it]2
# Train epoch:  30%|█████████▎                     | 3/10 [00:06<00:14,  2.00s/it]3
# Train epoch:  40%|████████████▍                  | 4/10 [00:08<00:12,  2.00s/it]4
# Train epoch:  50%|███████████████▌               | 5/10 [00:10<00:10,  2.00s/it]5
# Train epoch:  60%|██████████████████▌            | 6/10 [00:12<00:08,  2.00s/it]6
# Train epoch:  70%|█████████████████████▋         | 7/10 [00:14<00:06,  2.00s/it]7
# Train epoch:  80%|████████████████████████▊      | 8/10 [00:16<00:04,  2.00s/it]8
# Train epoch:  90%|███████████████████████████▉   | 9/10 [00:18<00:02,  2.00s/it]9
# Train epoch: 100%|██████████████████████████████| 10/10 [00:20<00:00,  2.00s/it]


def example_func1():
    epoch_all = [10+_ for _ in range(10)]
    print(epoch_all, sum(epoch_all))
    cb, pbar = get_pbar(f'Train epoch', sum(epoch_all)) # Progress Barの設定(Barのヘッダー, 100%となる時の値)
    for epoch in epoch_all:

        print(epoch)
        time.sleep(2)
        # update_pbar(cb, epoch) # Progress Barの更新(cb関数を指定, 更新数(指定した数だけBarが進む))
        if cb is not None:
            cb(epoch)

    close_pbar(pbar)
    result = None
    return result

# example_func1()
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 145
# Train epoch:   0%|                                      | 0/145 [00:00<?, ?it/s]10
# Train epoch:   7%|██                           | 10/145 [00:02<00:27,  4.99it/s]11
# Train epoch:  14%|████▏                        | 21/145 [00:04<00:23,  5.29it/s]12
# Train epoch:  23%|██████▌                      | 33/145 [00:06<00:19,  5.61it/s]13
# Train epoch:  32%|█████████▏                   | 46/145 [00:08<00:16,  5.96it/s]14
# Train epoch:  41%|████████████                 | 60/145 [00:10<00:13,  6.33it/s]15
# Train epoch:  52%|███████████████              | 75/145 [00:12<00:10,  6.72it/s]16
# Train epoch:  63%|██████████████████▏          | 91/145 [00:14<00:07,  7.14it/s]17
# Train epoch:  74%|████████████████████▊       | 108/145 [00:16<00:04,  7.57it/s]18
# Train epoch:  87%|████████████████████████▎   | 126/145 [00:18<00:02,  8.01it/s]19
# Train epoch: 100%|████████████████████████████| 145/145 [00:20<00:00,  7.24it/s]

