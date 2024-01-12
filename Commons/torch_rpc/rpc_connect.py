import os
from typing import NamedTuple
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# Set RPC Init Process Env
def set_master(addr: str, port=1885):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)

# Set RPC Init Process Env
def set_master_opt(addr: str, port=1885):
    rpc_backend_options = TensorPipeRpcBackendOptions(rpc_timeout=120)
    rpc_backend_options.init_method = f"tcp://{addr}:{port}"
    return rpc_backend_options

# RPCConfig
class RPCConfig(NamedTuple):
    rpc_node_num: int
    server_num_worker_node: int
    server_total_num_nodes: int
    # trainer_node_num: int
    trainer_num_devices_per_node: int
    trainer_total_num_nodes: int
    model_store_server: int

    @property
    def trainer_world_size(self):
        return self.trainer_total_num_nodes * self.trainer_num_devices_per_node

    @property
    def server_world_size(self):
        return self.server_total_num_nodes * self.server_num_worker_node

    @property
    def world_size(self):
        return self.trainer_world_size + self.server_world_size + self.model_store_server


# make RPCConfig
def get_rpc_config(rpc_node_num: int, server_num_worker_node:int, server_total_num_nodes:int,
                   total_num_nodes: int, num_devices_per_node: int, model_store_server:int):
    assert total_num_nodes > 0
    assert num_devices_per_node > 0

    return RPCConfig(rpc_node_num=rpc_node_num,
                     server_num_worker_node=server_num_worker_node, server_total_num_nodes=server_total_num_nodes,
                     trainer_num_devices_per_node=num_devices_per_node, trainer_total_num_nodes=total_num_nodes,
                     model_store_server=model_store_server)

