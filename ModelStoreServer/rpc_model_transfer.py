# Pytorch
import torch
import torch.distributed.rpc as rpc


# General lib
from threading import Lock
import sys
from time import sleep
from datetime import datetime
import os

from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_remote_method import remote_method, remote_method_async


# Pytorch sample code
# RPC Data Sender/Receiver Class
rpc_model_store = None
global_rmp_lock = Lock()


def get_server_object(name, args, log_file=None):
    global rpc_model_store
    # Ensure that we get only one handle to the RPC Data Sender.
    with global_rmp_lock:
        # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object start: "
        #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))
        if not rpc_model_store:
            # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object start: "
            #       % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))
            rpc_model_store = RPCModelStoreServer(name, args)
            # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object end: "
            #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))

        if log_file is not None:
            rpc_model_store.set_logger(log_file)

        # print("rpc_data_sender id =", id(rpc_data_sender))
        return rpc_model_store


# --------- RPC Model ParameterServer Class --------------------
# 参考:IMPLEMENTING BATCH RPC PROCESSING USING ASYNCHRONOUS EXECUTIONS
#    :https://pytorch.org/tutorials/intermediate/rpc_async_execution.html
# メモ:Parameter Server機能の勾配値集約とモデル更新メソッドは削除。
#　　　モデルパラメータを保持する仕組みは持たせており、TrainerとModelStoreServer(Parameter Server)間でモデルパラメータを共有する仕組みは導入している。
#
class RPCModelStoreServer:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.log_file = None
        self.logger = False

        self.model = None

        self.update_num = 0
        # print("init self.update_num", self.update_num)

    def set_logger(self, log_file):
        self.log_file = log_file
        self.logger = True if log_file is not None else False

    # def get_to_rref(self, to_name):
    #     self.to_rref = rpc.remote(
    #         to_name, get_server_object, args=(to_name, self.name)
    #     )

    def set_model(self, model):
        self.model = model

    def set_model_state_dict(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)

    def get_model(self):
        return self.model

    def get_model_state_dict(self):
        return self.model.state_dict()



def remote_get_model_from_mss(mss_rref):
    # return ps_rref.rpc_sync().get_model().cuda(device)
    return mss_rref.rpc_sync().get_model_state_dict()


def remote_set_model_to_mss(mss_rref, model_state_dict):
    # ps_rref.rpc_sync().set_model_state_dict(model.state_dict())
    ret = remote_method(RPCModelStoreServer.set_model_state_dict, mss_rref, model_state_dict)
    return


