# Pytorch
# import torch
import torch.distributed.rpc as rpc


# General lib
from threading import Lock
from time import sleep
# from datetime import datetime
# import os

# from torch.distributed.rpc import RRef
from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_remote_method import remote_method_async


# Pytorch sample code
# RPC Data Sender Class
# グローバル変数(グローバルスコープの上に)として用意。
rpc_data_sender = None
global_rds_lock = Lock()


def get_sender_object(name, receiver_name, log_file=None):
    global rpc_data_sender
    # Ensure that we get only one handle to the RPC Data Sender.
    with global_rds_lock:
        # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object start: "
        #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))
        if not rpc_data_sender:
            # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object start: "
            #       % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))
            rpc_data_sender = RPCDataSender(name, receiver_name)
            # print("%s: process_id=%d: (RPC Data Sender)Create RPC Data Sender object end: "
            #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid()))

        if log_file is not None:
            rpc_data_sender.set_logger(log_file)

        # print("rpc_data_sender id =", id(rpc_data_sender))
        return rpc_data_sender


# TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
# --------- RPC Data Sender Class --------------------
class RPCDataSender:
    def __init__(self, name, receiver_name):
        # self.sender_rref = RRef(self)
        self.name = name
        self.receiver_name = receiver_name
        self.minibatch_store = {
            "train" :{},
            "valid": {},
            "test": {},
        }
        self.end_flg = False

        self.log_file = None
        self.logger = False

    def set_logger(self, log_file):
        self.log_file = log_file
        self.logger = True if log_file is not None else False


    # def send_sender_rref(self):
    #     return self.sender_rref

    def set_minibatch_store(self, train_mode, epoch, inputs):

        minibatchs = self.minibatch_store[train_mode]
        if epoch in minibatchs.keys():
            pass
        else:
            minibatchs[epoch] = []

        with global_rds_lock:
            minibatchs[epoch].append(inputs)


    def check_minibatch_store(self, train_mode, epoch):
        count, c_limit = 0, 10*1000 # 初期値=0秒, 1000秒
        while count < c_limit:

            if epoch in self.minibatch_store[train_mode].keys():
                minibatchs = self.minibatch_store[train_mode]
                if len(minibatchs[epoch]) > 0:
                    break
                else:
                    sleep(1e-1)
                    count += 1
            else:
                sleep(1e-1)
                count += 1

        return count < c_limit


    # RPC method: Executed from trainer.
    def send_minibatch_to_trainer(self, train_mode, epoch, idx):

        make_flg = self.check_minibatch_store(train_mode, epoch)

        if make_flg:
            with global_rds_lock:
                inputs = self.minibatch_store[train_mode][epoch].pop(0)

            if inputs[0] >= 0:
                # print("%s, %s, Send Minibatch, epoch=%d, idx=%d, inputs_type=%s"
                #       %(self.name, train_mode, epoch, inputs[0], str(type(inputs[1])))
                # )
                output_func(self.logger,
                            "%s, %s, Send Minibatch, epoch=%d, idx=%d, inputs_type=%s"
                            %(self.name, train_mode, epoch, inputs[0], str(type(inputs[1]))),
                            self.log_file)

            else:
                # print("%s, %s, Send Minibatch, epoch=%d, idx=%d, data_size=%s"
                #       %(self.name, "Epoch loop end", epoch, inputs[0], len(self.minibatch_store[train_mode][epoch]))
                # )
                output_func(self.logger,
                            "%s, %s, Send Minibatch, epoch=%d, idx=%d, data_size=%s"
                            %(self.name, "Epoch loop end", epoch, inputs[0], len(self.minibatch_store[train_mode][epoch])),
                            self.log_file)
                self.minibatch_store[train_mode].pop(epoch)
        else:
            inputs = False
            self.end_flg = True

        return inputs



# RPC Data Receiver Class
# グローバル変数(グローバルスコープの上に)として用意。
rpc_data_receiver = None
global_tr_lock = Lock()


def get_receiver_object(name, sender_name):
    global rpc_data_receiver
    # Ensure that we get only one handle to the Trainer Server.
    with global_tr_lock:
        # print("%s: process_id=%d: (RPC Data Receiver)Create RPC Data Receiver object start: name=%s"
        #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid(), name))
        if not rpc_data_receiver:
            rpc_data_receiver = RPCDataReceiver(name, sender_name)
        # ret = trainer_object.update_model_from_main()
        # print("%s: process_id=%d: (RPC Data Receiver)Create RPC Data Receiver object end: name=%s"
        #                   % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), os.getpid(), name))

        return rpc_data_receiver


# TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
# --------- RPC Data Receiver Class --------------------
class RPCDataReceiver:
    def __init__(self, name, sender_name):
        self.name = name
        self.sender_name = sender_name
        # self.receiver_rref = RRef(self)
        self.sender_rref = None
        self.epoch = 0
        self.end_flg = False
        self.get_sender_rref(self.sender_name)

    def get_sender_rref(self, sender_name):
        self.sender_rref = rpc.remote(
            sender_name, get_sender_object, args=(sender_name, self.name)
        )

    def set_epoch(self, epoch):
        self.epoch = epoch


    def get_minibatch_from_server(self, train_mode, idx):

        epoch = self.epoch
        sender_rref = self.sender_rref

        fut = remote_method_async(
            RPCDataSender.send_minibatch_to_trainer,
            sender_rref, train_mode, epoch, idx
        )

        inputs = fut.wait()

        if isinstance(inputs, bool):
            raise Exception
        else:
            # if inputs[0] >= 0:
            #     print("Send Minibatch", epoch, inputs[0],
            #       type(inputs[1][0]),type(inputs[1][1]),type(inputs[1][2]))
            #
            # else:
            #     print("Epoch loop end", epoch, inputs[0])
            pass

        return inputs[0], inputs[1]

