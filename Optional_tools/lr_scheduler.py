import torch
import torch.distributed


def get_lr_scheduler(optimizer, patience, use_lrs=False):
    # Optional 学習率調整関数
    if use_lrs:
        lr_scheduler = \
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, factor=0.8,
                patience=patience, verbose=True
            )
    else:
        lr_scheduler = None

    return lr_scheduler


def update_lr_by_SALIENT(lr_scheduler, result):
    if lr_scheduler is not None:
        world_size = 1.0
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(result)
            world_size = 1.0*torch.distributed.get_world_size()
        lr_scheduler.step(result / world_size)

