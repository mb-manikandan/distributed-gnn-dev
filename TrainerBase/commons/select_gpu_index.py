import torch
import random


def select_gpu_index(args):
    # node_num:trainerのnode番号, proc_num:1trainer当たりの子プロセス数
    node_num, proc_num = args.node_num, args.max_num_devices_per_node
    if args.dev_select_mode == 0:
        return node_mul_proc_select_list(node_num=node_num, proc_num=proc_num)
    # elif args.dev_select_mode == 1:
    #     return random_index_list(proc_num=proc_num)
    else:
        return default_select_list(proc_num)


def default_select_list(proc_num):
    return [rank for rank in range(proc_num)]


def node_mul_proc_select(rank, node_num, proc_num):
    index = (node_num * proc_num + rank) % torch.cuda.device_count()
    return index


def node_mul_proc_select_list(node_num, proc_num):
    return [node_mul_proc_select(rank, node_num, proc_num) for rank in range(proc_num)]


def random_index():
    return int(random.random()*torch.cuda.device_count())


def random_index_list(proc_num):
    return [random_index() for _ in range(proc_num)]

