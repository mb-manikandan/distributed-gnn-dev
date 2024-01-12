import argparse


class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()
            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                if k not in ["job_name"]:
                    setattr(namespace, k, v)


def make_parser():
    parser = argparse.ArgumentParser(description="Start an experiment")
    # Dataset name, Job_name
    # parser.add_argument("dataset_name",
    #                     help="Name of the OGB dataset",
    #                     type=str)
    parser.add_argument("job_name",
                        help="Name of the Job",
                        type=str)
    # Config file
    parser.add_argument("--config_file",
                        help="Use config file rather than command line arg",
                        type=open, action=LoadFromFile, default=None)

    # Input/Output file path
    parser.add_argument("--output_root",
                        help="The root of output storage",
                        type=str, default=f"job_output/")
    parser.add_argument("--overwrite_job_dir",
                        help="If a job directory exists, delete it",
                        action="store_true")
    # Print log message
    parser.add_argument("--logger",
                        help="logger or print mode",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="Print log entries to stdout",
                        action="store_true")

    # DDP settings of nodes and devices, workers(Sampler process) number
    parser.add_argument("--total_num_nodes",
                        help="Total number of nodes to use",
                        type=int, default=1)
    parser.add_argument("--max_num_devices_per_node",
                        help="Maximum number of devices per node",
                        type=int, default=1)
    parser.add_argument("--rpc_node_num",
                        help="RPC node number",
                        type=int, default=1)

    # RPC settings of nodes and devices, workers(Sampler process) number
    parser.add_argument("--server_rpc_node_num",
                        help="RPC node number",
                        type=int, default=0)
    parser.add_argument("--server_num_worker_node",
                        help="RPC Number of workers",
                        type=int, default=1)
    parser.add_argument("--server_total_num_nodes",
                        help="Total number of nodes to use by RPC",
                        type=int, default=1)
    parser.add_argument("--num_workers",
                        help="Number of workers",
                        type=int, default=10)

    # New add, DDP communication settings
    parser.add_argument("--ddp_addr",
                        help="addr",
                        type=str, default="localhost")
    parser.add_argument("--ddp_port",
                        help="port",
                        type=int, default=1884)
    parser.add_argument("--ddp_node_num",
                        help="node_num(rank)",
                        type=int, default=0)

    # New add, RPC communication settings
    parser.add_argument("--rpc_addr",
                        help="addr",
                        type=str, default="localhost")
    parser.add_argument("--rpc_port",
                        help="port",
                        type=int, default=1885)

    # Model type
    parser.add_argument("--model_type_name",
                        help="Name of the model to use",
                        type=str, default="SAGE")
    # See driver/main.py/get_model_type() for available choicess
    # Model layers settings
    parser.add_argument("--hidden_features",
                        help="Number of hidden features",
                        type=int, default=256)
    parser.add_argument("--num_layers",
                        help="Number of layers",
                        type=int, default=3)
    parser.add_argument("--num_features",
                        help="Number of features",
                        type=int, default=128)
    parser.add_argument("--num_classes",
                        help="Number of classes",
                        type=int, default=40)


    parser.add_argument("--use_model_name",
                        help="Name of the model to use",
                        type=str, default="model1")

    # Learning rate
    parser.add_argument("--lr",
                        help="Learning rate",
                        type=float, default=0.003)


    return parser
