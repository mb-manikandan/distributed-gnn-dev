from typing import Any, Dict
from pathlib import Path
import sys, os

import torch
import torch.multiprocessing as mp
import multiprocessing

# Setting
from ModelStoreServer import Settings
from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_connect import get_rpc_config
from Commons.communication_tools.launch_sampler_process import MessageReceiver, loads_args_str

from ModelStoreServer.commons.model_store_server_parser import make_parser
from Commons.logger_function import prepare_log_file
from ModelStoreServer.commons.save_model_function import load_model_dict, save_model_dict

from ModelStoreServer.model_transfer_process import ModelTransferProcess


class ModelStoreProcess:
    args: Any
    log_file: Path
    model_dicts:Dict
    sampler_type: Any


    def __init__(self):
        self.args = None
        self.sampler_type = None
        self.model_dicts = {}
        self._proc = {}
        # self._save_model_name = Settings.MODEL_LIST
        self._save_model_name = []
        self._model_path = Settings.MODEL_PATH
        self._model_prefix = Settings.MODEL_PREFIX
        self._model_suffix = Settings.MODEL_SUFFIX

        # Socket Message Receive Process
        self.msg_rec = MessageReceiver(addr=Settings.SOCKET_HOST, port=Settings.SOCKET_PORT)

    def get_parser(self):
        return make_parser()

    def prepare_log_file(self, args, file_name=""):
        return prepare_log_file(args, file_name)

    def get_model_state_dict(self, model_path):
        return load_model_dict(model_path)

    def set_model_state_dict(self, model_name, model_path):
        self.model_dicts[model_name] = self.get_model_state_dict(model_path)

    def check_model_dict(self, model_name):
        return model_name in self.model_dicts

    def run_server(self):
        args = self.get_parser().parse_args()
        args.log_file = self.prepare_log_file(args, "ModelStoreServer_ModelStoreProcess")
        self.args = args

        # for key, value in self._save_model_name.items():
        #     self.set_model_state_dict(key, self._model_path+value)
        # for model_name in self._save_model_name:
        #     self.set_model_state_dict(model_name, self._model_path+model_name)

        model_files = os.listdir(self._model_path)

        for model_name in model_files:
            output_func(self.args.logger,
                        "%s, %d, %s" %(model_name, model_name.find(self._model_prefix), self._model_suffix in model_name)
                        ,self.args.log_file)
            # ファイル命名規則の確認。
            # 先頭から0番目にself._model_prefixが、拡張子がself._model_suffixが対応しているかを確認。
            if model_name.find(self._model_prefix) == 0 and self._model_suffix in model_name:
                self.set_model_state_dict(model_name, self._model_path+model_name)
            else:
                output_func(self.args.logger,
                            "%s, Does not meet file naming conventions. Please rename file. "
                            "prefix='%s',suffix='%s'" %(model_name, self._model_prefix, self._model_suffix),
                            self.args.log_file)
                continue


        output_func(self.args.logger,
                    "\n" + "+" * 68 + "\n" \
                    "+" + " " * 10 + "Start: Run ModelStoreServer:ModelStoreProcess" + " " * 11 + "+" \
                    "\n" + "+" * 68 + "\n",
                    self.args.log_file)

        for key, value in self.model_dicts.items():
            output_func(self.args.logger, "model_name=%s, model_size(sys.getsizeof)=%d" % (key, sys.getsizeof(value)),
                        self.args.log_file)

    def receiver_connect_loop(self):
        while True:
            ret = self.msg_rec.accept()
            if ret:
                # print("Connect trainer program")
                output_func(self.args.logger, "Connect Trainer Process", self.args.log_file)
                self.receive_message()
                # break

    def receive_message(self):
        while True:
            msg_str = self.msg_rec.receive_message()
            args = loads_args_str(msg_str)
            # args.log_file = self.prepare_log_file(args, "ModelStoreServer_ModelStoreProcess")
            args.logger = self.args.logger

            output_func(self.args.logger,
                        "Receive message:  %s: %s" % (str(type(args)), args),
                        self.args.log_file)

            if msg_str == "timeout":
                break

            if args.process == "launch":
                # print("Start: launch_sampler_parent_process")
                output_func(self.args.logger, "Start: launch_parent_process", self.args.log_file)
                self.launch_parent_process(args, args.use_model_name)

                output_func(self.args.logger,
                            "\n" + "+" * 68 + "\n" \
                            "+" + " " * 11 + "Running ModelStoreServer:ModelStoreProcess " + " " * 12 + "+" \
                            "\n" + "+" * 68 + "\n",
                            self.args.log_file)
                break

            elif args.process == "stop":
                self.stop_child_process(args, args.use_model_name)
                output_func(self.args.logger, "End  : launch_parent_process\n", self.args.log_file)

                output_func(self.args.logger,
                            "\n" + "+" * 68 + "\n" \
                            "+" + " " * 11 + "Running ModelStoreServer:ModelStoreProcess " + " " * 12 + "+" \
                            "\n" + "+" * 68 + "\n",
                            self.args.log_file)
                break


    def set_mp_to_spawn(self):
        if mp.get_start_method() == 'fork':
            mp.set_start_method('spawn', force=True)
            output_func(self.args.logger, "{} setup done".format(mp.get_start_method()), self.args.log_file)

    def launch_parent_process(self, args, model_name):

        exit_flg = self.check_model_dict(model_name)
        if exit_flg:
            model_dict = self.model_dicts[model_name]
            output_func(self.args.logger, "Load model_state_dict of '%s'." %model_name, self.args.log_file)
        else:
            # self.model_dicts[model_name] = None
            model_dict = None
            output_func(self.args.logger, "Can not load model_state_dict of '%s'. Make new model_state_dict" % model_name,
                        self.args.log_file)
        # mp.spawn(fn=self.launch_child_process, args=(args, model),
        #          nprocs=args.server_num_worker_node, join=True)
        # self.set_mp_to_spawn()

        # 子プロセス(ModelTransferProcess)から本プロセス(ModelStoreProcess)へモデル共有するための辞書方データを用意。
        manager = multiprocessing.Manager()
        share_dict = manager.dict()

        self.set_mp_to_spawn()
        # if args.jobname not in self._proc:
        proc = mp.Process(target=self.launch_child_process, args=(args, model_dict, share_dict))
        proc.start()

        # _procメンバ変数に、jobnameのkey名にprocとshare_dict(モデル共有するための辞書方データ)を格納。
        self._proc[args.jobname] = (proc, share_dict)
        # proc.join()
        # proc.close()

    def launch_child_process(self, args, model_dict, share_dict):

        # RPC通信準備
        rpc_cfg = get_rpc_config(args.node_num, args.server_num_worker_node, args.server_total_num_nodes,
                                 args.total_num_nodes, args.max_num_devices_per_node, 1)

        log_file = prepare_log_file(args, "ModelStoreServer_ModelTransferProcess", args.date_time)
        args.log_file = log_file

        mt_ins = ModelTransferProcess(args, args.model_name, model_dict, rpc_cfg)
        mt_ins.run_process(share_dict)

    def stop_child_process(self, args, model_name):
        # _procメンバ変数から、jobnameのkey名でprocとshare_dict(モデル共有するための辞書方データ)を取り出す。
        proc, share_dict = self._proc.pop(args.jobname)

        # close処理
        proc.join()
        proc.close()

        # if model_name in self.model_dicts:
        #     diff_model_dict(self.model_dicts[model_name], share_dict["model"])
        # else:
        #     print("\nNew model_dict")

        # share_dict(モデル共有するための辞書方データ)から学習したmodelを取り出し、ModelStoreProcessとファイル上に上書き保存する。
        self.model_dicts[model_name] = share_dict["model"]
        # save_model_dict(self.model_dicts[model_name], self._model_path + self._save_model_name[model_name])
        save_model_dict(self.model_dicts[model_name], self._model_path + model_name)
        output_func(self.args.logger, "Save model_state_dict of '%s'." % model_name, self.args.log_file)



def diff_model_dict(model_dict_old, model_dict_new):
    print("\ndiff_model_dict")
    for key, value in model_dict_old.items():
        diff_model_dict = model_dict_new[key] - value
        if diff_model_dict.sum() > 0:
            print(key, diff_model_dict, torch.where(diff_model_dict>0))

# def to_cpu_model_dicts(model_dicts):
#     for model_dict in model_dicts:
#         for key, value in model_dict.items():
#             model_dict[key] = value.cpu()
#     return model_dicts


if __name__ == '__main__':

    mstores_ins = ModelStoreProcess()
    mstores_ins.run_server()
    mstores_ins.receiver_connect_loop()

