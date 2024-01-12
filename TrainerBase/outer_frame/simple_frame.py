from datetime import datetime
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
import multiprocessing

from TrainerBase.commons.sampler_base_parser import make_parser
from Commons.output_tool import output_func
from Commons.communication_tools.launch_sampler_process import MessageSender, dumps_args_str
from Commons.torch_rpc.rpc_connect import get_rpc_config, RPCConfig, set_master_opt
from Commons.torch_ddp.ddp_connect import get_ddp_config, DDPConfig

from TrainerBase.outer_frame.frame_base import get_job_dir, get_model_type, prepare_log_file
from TrainerBase.inner_frame.trainer_base import TrainerBase
from TrainerBase.commons.select_gpu_index import select_gpu_index

from ModelStoreServer import Settings, rpc_model_transfer
from ModelStoreServer.rpc_model_transfer import remote_set_model_to_mss, remote_get_model_from_mss


class SimpleFrame:
    drv_type: TrainerBase
    def __init__(self, drv_type: TrainerBase):
        self.drv_type = drv_type
        self.log_file = None
        # self.ps_rref = None
    def __del__(self):
        # pass
        rpc.shutdown()

    def get_parser(self):
        return make_parser()

    def get_model_type(self, model_name):
        return get_model_type(model_name)

    def prepare_log_file(self, args):
        return prepare_log_file(args, args.date_time)

    def test_process(self, drv: TrainerBase, acc_type):

        acc = drv.test((acc_type,))[acc_type]
        if isinstance(drv, TrainerBase):
            dist.barrier()

        return acc

    def final_test_process(self, drv: TrainerBase):
        # 最終テスト処理はepoch=-1を渡す。
        drv.receiver.set_epoch(-1)
        acc_dict = {}
        for acc_type in ('valid', 'test'):
            acc = self.test_process(drv, acc_type)
            # if drv.is_main_proc:
            #     drv.log((acc_type, 'Accurracy', acc))
            acc_dict[acc_type] = acc

        return acc_dict

    def train_process(self, args, drv: TrainerBase):

        delta = min(args.test_epoch_frequency, args.epochs)
        do_eval = args.epochs >= args.test_epoch_frequency

        for epoch in range(0, args.epochs, delta):

            # Sync process
            if isinstance(drv, TrainerBase):
                dist.barrier()

            # Driver Train
            drv.train(range(epoch, epoch + delta))

            if do_eval:
                if isinstance(drv, TrainerBase):
                    dist.barrier()
                # Driver Test(valid data)
                acc_type = 'valid'
                acc = self.test_process(drv, acc_type)
                if drv.is_main_proc:
                    output_func(args.logger,
                                "\nEpoch=%d, validation accuracy=: %f\n" %(epoch + delta - 1, acc),
                                args.log_file)
                    # drv.log((acc_type, 'Accurracy', acc))

            # drv.flush_logs()

        # Driver Test(valid & test data)
        acc_dict = self.final_test_process(drv)
        final_valid_acc, final_test_acc = acc_dict['valid'], acc_dict['test']

        if drv.is_main_proc:
            output_func(args.logger,
                        "\nFinal validation,test accuracy is: %f, %f\n" % (final_valid_acc, final_test_acc),
                        args.log_file)

        # # Save model
        # job_dir = get_job_dir(args)
        #
        # if hasattr(drv.model, 'module'):
        #     torch.save(drv.model.module.state_dict(), job_dir.joinpath(f'{drv.receiver_name}_model.pt'))
        # else:
        #     torch.save(drv.model.state_dict(), job_dir.joinpath(f'{drv.receiver_name}_model.pt'))

    # Socket通信処理
    def send_massage(self, args, log_file, socket_addr='127.0.0.1', socket_port=8200):

        args_str = dumps_args_str(args)
        # Socket Message Sender Process
        msg_sen = MessageSender(addr=socket_addr, port=socket_port)

        output_func(args.logger, "Send message: %s: %s" % (str(type(args_str)), args_str), log_file)
        ret = msg_sen.send_message(args_str)

        del msg_sen

        if ret is False:
            raise Exception

    # Socket通信によるModelStoreServerの子プロセス(ModelTransferProcess)を起動/停止合図処理
    def send_massage_to_model_store_server(self, args, log_file, process_name, job_name):
        args.process = process_name
        args.jobname = job_name
        args.log_file = None
        # ModelStoreServerの子プロセス(ModelTransferProcess)を起動合図処理
        self.send_massage(args, log_file, Settings.SOCKET_HOST, Settings.SOCKET_PORT)


    def main(self):

        args = self.get_parser().parse_args()
        # TODO: Remove
        # assert args.max_num_devices_per_node <= torch.cuda.device_count()
        assert (args.dev_mode == "cpu") or (args.dev_mode == "cuda")

        # ADD issue-8
        args.date_time = "_%s" % datetime.now().strftime("%Y%m%d-%H%M%S")

        num_devices_per_node = args.max_num_devices_per_node
        log_file = self.prepare_log_file(args)
        self.log_file = log_file
        args.log_file = None

        # ModelStoreServer connect process
        if args.node_num == 0:
            self.send_massage_to_model_store_server(args, log_file, "launch", "test")

        # ModelStoreServer connection process and prepare process
        mss_rref, share_dict = self.prepare_connection_to_mss(args, num_devices_per_node)

        self.send_massage(args, log_file)

        # HACK: Pass the log_file CommonBaseDriver
        args.log_file = log_file

        output_func(args.logger, f'Using {num_devices_per_node} devices per node', args.log_file)

        ddp_cfg = get_ddp_config(args.total_num_nodes, num_devices_per_node,
                                 args.ddp_addr, args.ddp_port, args.node_num)
        output_func(args.logger, f'Using DDP trainer with {ddp_cfg.total_num_nodes} nodes', args.log_file)

        rpc_cfg = get_rpc_config(args.node_num, args.server_num_worker_node, args.server_total_num_nodes,
                                 args.total_num_nodes, num_devices_per_node, 0)
        output_func(args.logger, f'Using RPC server with {rpc_cfg.server_total_num_nodes} nodes', args.log_file)
        # output_func(args.logger,
        #             f'Using RPC trainer&server with {rpc_cfg.server_total_num_nodes+rpc_cfg.trainer_total_num_nodes} nodes',
        #             args.log_file)

        model_type = self.get_model_type(args.model_name)

        # ADD issue-5
        available_devices = select_gpu_index(args=args)

        mp.spawn(self.ddp_main, args=(args, model_type, share_dict, ddp_cfg, rpc_cfg, available_devices),
                 nprocs=num_devices_per_node, join=True)

        # Finish process
        if args.node_num == 0:
            model_state_dict = share_dict["model"]
            remote_set_model_to_mss(mss_rref, model_state_dict)

            self.send_massage_to_model_store_server(args, log_file, "stop", "test")


    def prepare_connection_to_mss(self, args, num_devices_per_node):
        rpc_cfg_mss = get_rpc_config(args.node_num, args.server_num_worker_node, args.server_total_num_nodes,
                                    args.total_num_nodes, num_devices_per_node, 1)

        self.rpc_connect(args, rpc_cfg_mss)

        mss_name = "ModelStoreServer"
        mss_rref = rpc.remote(
            mss_name, rpc_model_transfer.get_server_object, args=(mss_name, args)
        )

        model_state_dict = remote_get_model_from_mss(mss_rref)

        # 子プロセス(Trainer子プロセス)から本プロセス(Trainer親プロセス)へモデル共有するための辞書方データを用意。
        manager = multiprocessing.Manager()
        share_dict = manager.dict()

        share_dict["model"] = model_state_dict

        return mss_rref, share_dict


    # RPC処理
    def rpc_connect(self, args, rpc_cfg: RPCConfig):
        self.mss2tr_rpc_rank = (
                rpc_cfg.rpc_node_num + rpc_cfg.model_store_server
        )
        self.tr_parent_rank = rpc_cfg.rpc_node_num
        self.set_data_transfer_name()

        output_func(args.logger,
                    "%s, rpc_node_num=%d, mss2tr_rpc_rank=%d"
                    % (self.receiver_name, rpc_cfg.rpc_node_num, self.mss2tr_rpc_rank),
                    self.log_file)

        rpc.init_rpc(
            name=self.receiver_name,
            rank=self.mss2tr_rpc_rank,
            world_size=rpc_cfg.trainer_total_num_nodes + rpc_cfg.model_store_server,
            rpc_backend_options=set_master_opt(Settings.RPC_MSS2TR_ADDR, Settings.RPC_MSS2TR_PORT)
        )

    def set_data_transfer_name(self):
        self.receiver_name = f"trainer_parent_{self.tr_parent_rank}"
        self.sender_name = "ModelStoreServer"

    def ddp_main(self, rank, args, model_type, share_dict, ddp_cfg: DDPConfig, rpc_cfg: RPCConfig, available_devices:list):
        # ADD issue-5
        torch.cuda.set_device(available_devices[rank])
        # device = torch.device(type='cuda', index=rank)
        device = torch.device(type='cuda', index=available_devices[rank])

        # Set model_state_dict
        model_state_dict = share_dict["model"]
        drv = self.drv_type(args, rank, [device], model_type, model_state_dict, ddp_cfg, rpc_cfg)

        self.run_trainer(args, drv)

        # return model_state_dict
        share_dict["model"] = drv.get_model_dict()
        # share_dict["model"] = model_state_dict


    def run_trainer(self, args, drv: TrainerBase):
        if drv.is_main_proc:
            output_func(args.logger,
                        "\n" + "+" * 50 + "\n" \
                        "+" + " " * 16 + "Run main process" + " " * 16 + "+" \
                        "\n" + "+" * 50 + "\n" \
                        "Performing training",
                        args.log_file)

        # Train process(よりシンプルな学習処理)
        self.train_process(args, drv)

        # log出力
        # drv.flush_logs()


from TrainerBase.example.ns_trainer import NSTrainerBase
if __name__ == '__main__':
    assert torch.cuda.is_available()
    # simple_ins = SimpleFrame(drv_type=DDPDriver)
    simple_ins = SimpleFrame(drv_type=NSTrainerBase)
    simple_ins.main()
