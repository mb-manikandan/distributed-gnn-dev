from typing import Any, Dict
from pathlib import Path
from datetime import datetime

import torch.multiprocessing as mp

# Setting
from GraphStoreServer import Settings
from Commons.output_tool import output_func
from Commons.torch_rpc.rpc_connect import get_rpc_config
from Commons.communication_tools.launch_sampler_process import MessageReceiver, loads_args_str

from GraphStoreServer.commons.graph_store_parser import make_parser
from GraphStoreServer.commons.common_func import get_dataset, prepare_log_file

# TODO: 後々PyFlink(Pregel)プログラムと入れ替える予定
class GraphStoreServer:
    args: Any
    datasets: Dict
    log_file: Path

    sampler_type: Any


    def __init__(self, trainer_num=1):
        self.args = None
        self.sampler_type = None
        self.datasets = {}

        # self.num_workers = trainer_num
        self._proc = None

        self.msg_rec = MessageReceiver()


    def get_parser(self):
        return make_parser()

    def prepare_log_file(self, args, file_name="", date_time="_%s" % datetime.now().strftime("%Y%m%d-%H%M%S")):
        return prepare_log_file(args, file_name, date_time)

    def get_dataset(self, dataset_name, root):
        return get_dataset(dataset_name, root)

    def get_sampler_type(self, sampler_name):
        if sampler_name == "NSServer":
            from GraphStoreServer.example.ns_server import NSServer
            sampler_type = NSServer
        else:
            raise
        return sampler_type

    def make_ogb2pyg_graphs_from_file(self, dataset_name_list):
        for dataset_name, root in dataset_name_list:
            self.datasets[dataset_name] = self.get_dataset(dataset_name, root)

    # def make_graph(self, dataset_name, root):
    #         self.datasets[dataset_name] = self.get_dataset(dataset_name, root)

    def run_server(self, dataset_name_list):
        args = self.get_parser().parse_args()
        args.log_file = self.prepare_log_file(args, "GraphStorServer")
        self.args = args
        # self.num_workers = args.server_num_worker_node

        self.make_ogb2pyg_graphs_from_file(dataset_name_list)
        output_func(self.args.logger,
                    "\n" + "+" * 50 + "\n" \
                    "+" + " " * 11 + "Start: Run GraphStorServer" + " " * 11 + "+" \
                    "\n" + "+" * 50 + "\n",
                    self.args.log_file)
        # print("\n")
        # print("+" * 50)
        # print("+" + " " * 11 + "Start: Run GraphStorServer" + " " * 11 + "+")
        # print("+" * 50)
        # print("\n")

    def receiver_connect_loop(self):
        while True:
            ret = self.msg_rec.accept()
            if ret:
                # print("Connect trainer program")
                output_func(self.args.logger, "Connect trainer program", self.args.log_file)
                self.receive_message()
                # break

    def receive_message(self):
        while True:
            msg_str = self.msg_rec.receive_message()
            args = loads_args_str(msg_str)
            args.log_file = self.prepare_log_file(args, args.sampler_name + "_process", args.date_time)
            args.logger = self.args.logger
            # print("Receive message:  %s: %s" % (str(type(args)), args))
            # print(args.epochs)
            output_func(self.args.logger,
                        "Receive message:  %s: %s" % (str(type(args)), args),
                        self.args.log_file)

            sampler_type = self.get_sampler_type(args.sampler_name)
            if msg_str != "timeout":
                # print("Start: launch_sampler_parent_process")
                output_func(self.args.logger, "Start: launch_sampler_parent_process", self.args.log_file)
                self.launch_sampler_parent_process(args, sampler_type, args.dataset_name)
                # self.launch_sampler_parent_process2(sampler_type, args.dataset_name)

                # print("End  : launch_sampler_parent_process\n")
                output_func(self.args.logger, "End  : launch_sampler_parent_process\n", self.args.log_file)


                output_func(self.args.logger,
                            "\n" + "+" * 50 + "\n" \
                            "+" + " " * 12 + "Running GraphStorServer " + " " * 12 + "+" \
                            "\n" + "+" * 50 + "\n",
                            self.args.log_file)
                break


    def launch_sampler_parent_process2(self, sampler_type, dataset_name):

        # TODO:Graphデータの渡し方、サブグラフの場合はどうする？
        dataset = self.datasets[dataset_name]
        mp.spawn(fn=self.launch_sampler_child_process2, args=(sampler_type, dataset),
                 nprocs=self.args.server_num_worker_node, join=True)

    # TODO:Graphデータの渡し方
    def launch_sampler_child_process2(self, num_worker, sampler_type, dataset):
        rpc_cfg = get_rpc_config(self.args.node_num, self.args.server_num_worker_node, self.args.server_total_num_nodes,
                                 self.args.total_num_nodes, self.args.num_devices_per_node,
                                 self.args.rpc_addr, self.args.rpc_port)
        sample_server = sampler_type(self.args, num_worker, dataset, rpc_cfg, None)
        sample_server.main(self.args, "train")


    def launch_sampler_parent_process(self, args, sampler_type, dataset_name):

        # TODO:Graphデータの渡し方、サブグラフ化はどうする？
        dataset = self.datasets[dataset_name]
        mp.spawn(fn=self.launch_sampler_child_process, args=(args, sampler_type, dataset),
                 nprocs=args.server_num_worker_node, join=True)

    def launch_sampler_child_process(self, num_worker, args, sampler_type, dataset):
        ps_num = 1 if args.ps_mode == 1 else 0
        rpc_cfg = get_rpc_config(args.node_num, args.server_num_worker_node, args.server_total_num_nodes,
                                 args.total_num_nodes, args.max_num_devices_per_node, 0)
        sample_server = sampler_type(args, num_worker, dataset, rpc_cfg, None)
        sample_server.main(args, "train")


if __name__ == '__main__':

    ggs_ins = GraphStoreServer(1)
    ggs_ins.run_server(Settings.OGB_DATASET_LIST)
    ggs_ins.receiver_connect_loop()
    # ggs_ins.launch_sampler_parent_process(NSServer, "ogbn-arxiv")

