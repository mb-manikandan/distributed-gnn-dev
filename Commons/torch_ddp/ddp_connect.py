import os
from typing import NamedTuple


# Set DDP Init Process Env
def set_master(addr: str, port=1884):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)


# DDPConfig
class DDPConfig(NamedTuple):
    node_num: int
    num_devices_per_node: int
    total_num_nodes: int

    @property
    def world_size(self):
        return self.total_num_nodes * self.num_devices_per_node


# make DDPConfig
def get_ddp_config(total_num_nodes: int, num_devices_per_node: int,
                   addr: str, port: int, node_num: int):
    assert total_num_nodes > 0
    assert num_devices_per_node > 0

    # Point1: Set DDP Init Process Env
    set_master(addr, port)

    return DDPConfig(node_num=node_num,
                     num_devices_per_node=num_devices_per_node,
                     total_num_nodes=total_num_nodes)
