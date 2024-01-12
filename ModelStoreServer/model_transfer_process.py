from typing import Any
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed.rpc as rpc

# Setting
# from GraphStoreServer import Settings
from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_connect import get_rpc_config
from Commons.torch_rpc.rpc_connect import RPCConfig, set_master_opt

from ModelStoreServer import rpc_model_transfer, Settings
from ModelStoreServer.commons.model_store_server_parser import make_parser
from Commons.logger_function import prepare_log_file
from TrainerBase.outer_frame.frame_base import get_model_type


class ModelTransferProcess:
    args: Any
    model: torch.nn.Module
    model_state_dict: OrderedDict
    log_file: Path

    # RPC Settings
    server_rpc_rank: int
    mss2tr_rpc_rank: int
    process_name: str
    rpc_cfg: RPCConfig


    def __init__(self, args, model_type_name, model_state_dict, rpc_cfg: RPCConfig):
        self.rpc_model_object = None
        self.process_name = "ModelStoreServer"

        self.args = args
        self.log_file = Path(args.log_file)

        # Set model_state_dict
        self.model_state_dict = model_state_dict

        # RPC通信処理
        self.rpc_cfg = rpc_cfg

        # Set General values
        self.logs = []

        # Set Model
        self._set_model(model_type_name)

    def __del__(self):
        pass

    def get_parser(self):
        return make_parser()

    def prepare_log_file(self, args, file_name=""):
        return prepare_log_file(args, file_name, args.date_time)


    def run_process(self, share_dict):

        output_func(self.args.logger,
                    "\n" + "+" * 68 + "\n" \
                    "+" + " " * 9 + "Start: Run ModelStoreServer:ModelTransferProcess" + " " * 9 + "+" \
                    "\n" + "+" * 68 + "\n",
                    self.args.log_file)

        # モデルパラメータのやとりするためのRPC通信クラスオブジェクトの定義(rpc_model_object)
        # memo:リモート操作したいクラスオブジェクトは、グローバル変数(グローバルスコープ上に)として配置しないといけない
        self.rpc_model_object = rpc_model_transfer.get_server_object(
            self.process_name, self.args, self.log_file
        )

        # モデルの初期化
        self.rpc_model_object.set_model(self.model)

        # RPC通信開始処理
        self.rpc_init()

        # RPC通信が終了するまで(学習Jobなどが終了するまで)待機
        rpc.shutdown()

        # ModelStoreServerから最新のモデルパラメータ(tensor型)を取得し上書きする
        self.model_state_dict = self.rpc_model_object.get_model_state_dict()
        self.set_model_dict()

        # Overwrite share_dict
        self.set_model_to_share_dict(share_dict)

        output_func(self.args.logger,
                    "\n" + "+" * 68 + "\n" \
                    "+" + " " * 9 + "End  : Run ModelStoreServer:ModelTransferProcess" + " " * 9 + "+" \
                    "\n" + "+" * 68 + "\n",
                    self.args.log_file)


    # Set train GNN model(Model computation phase)
    # default=args.model_type_name(SAGE etc)
    def _set_model(self, model_type_name):
        model_type = get_model_type(model_type_name)
        self.model = model_type(
            self.args.num_features, self.args.hidden_features,
            self.args.num_classes,
            num_layers=self.args.num_layers
        )
        if self.model_state_dict is not None:
            self.set_model_dict()
        else:
            self.init_model()

    def init_model(self):
        self.model.reset_parameters()
        self.model_state_dict = self.model.state_dict()

    def set_model_dict(self):
        self.model.load_state_dict(self.model_state_dict)
        # self.set_model_to_parameter_server()

    # def set_model_to_parameter_server(self):
    #     self.rpc_model_object.set_model_dict(self.model)

    def set_model_to_share_dict(self, share_dict):
        # RPC通信クラスオブジェクト(rpc_model_object)から最新のモデル状態を取得する。
        # share_dict["model"] = self.rpc_model_object.get_model_dict()
        share_dict["model"] = self.model_state_dict

    # RPCコネクション処理
    def rpc_connect(self, rpc_cfg: RPCConfig, num_worker):

        self.mss2tr_rpc_rank = num_worker

        output_func(self.args.logger, "%s, mss2tr_rpc_rank=%d" % (self.process_name, self.mss2tr_rpc_rank),
                    self.args.log_file)

        rpc_backend_options = set_master_opt(Settings.RPC_MSS2TR_ADDR, Settings.RPC_MSS2TR_PORT)

        rpc.init_rpc(
            name=self.process_name,
            rank=self.mss2tr_rpc_rank,
            world_size=rpc_cfg.trainer_total_num_nodes + rpc_cfg.model_store_server,
            rpc_backend_options=rpc_backend_options
        )

        # TODO: Remove
        # rpc_backend_options = set_master_opt(self.args.rpc_addr, self.args.rpc_port)
        #
        # rpc.init_rpc(
        #     name=self.process_name,
        #     rank=self.ps2tr_rpc_rank,
        #     world_size=rpc_cfg.world_size,
        #     rpc_backend_options=rpc_backend_options
        # )

    def rpc_init(self):
        num_worker = 0
        self.rpc_connect(self.rpc_cfg, num_worker)
        log_file = self.log_file if self.args.logger else None


if __name__ == '__main__':
    pass



