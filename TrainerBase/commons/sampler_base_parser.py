import argparse
from TrainerBase import Settings

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()
            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                if k not in ["dataset_name", "job_name"]:
                    setattr(namespace, k, v)


def make_parser():
    parser = argparse.ArgumentParser(description="Start an experiment")
    # Dataset name, Job_name
    parser.add_argument("dataset_name",
                        help="Name of the OGB dataset",
                        type=str)
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

    # RPC settings of nodes and devices, workers(Sampler process) number
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
    parser.add_argument("--node_num",
                        help="node_num(rank)",
                        type=int, default=0)

    # New add, RPC communication settings
    parser.add_argument("--rpc_addr",
                        help="addr",
                        type=str, default="localhost")
    parser.add_argument("--rpc_port",
                        help="port",
                        type=int, default=1885)

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

    # Epochs settings
    parser.add_argument("--epochs",
                        help="Total number of epochs to train",
                        type=int, default=21)
    # Model type
    parser.add_argument("--model_name",
                        help="Name of the model to use",
                        type=str, default="SAGE")
    # See driver/main.py/get_model_type() for available choices
    parser.add_argument("--use_model_name",
                        help="Name of the model to use",
                        type=str, default="model_sage.pt")

    # Setting Parameter Server
    parser.add_argument("--ps_mode",
                        help="Parameter Server(PS) mode. False(0)/True(1).",
                        type=int, default=1)
    # Setting Distributed Data Parallel(DDP)
    parser.add_argument("--ddp_mode",
                        help="Distributed Data Parallel(DDP) mode. False(0)/True(1).",
                        type=int, default=1)

    # Sampler type
    parser.add_argument("--sampler_name",
                        help="Name of the sampler to use",
                        type=str, default="NSServer")

    # Learning rate
    parser.add_argument("--lr",
                        help="Learning rate",
                        type=float, default=0.003)
    # Train settings
    parser.add_argument("--train_batch_size",
                        help="Size of training batches",
                        type=int, default=1024)
    parser.add_argument("--train_max_num_batches",
                        help="Max number of training batches waiting in queue",
                        type=int, default=100)
    parser.add_argument("--train_fanouts",
                        help="Training fanouts",
                        type=int, default=[15, 10, 5], nargs="*")
    # Test settings
    parser.add_argument("--test_epoch_frequency",
                        help="Number of epochs to train before testing occurs",
                        type=int, default=20)
    parser.add_argument("--test_batch_size",
                        help="Size of testing batches",
                        type=int, default=4096)
    parser.add_argument("--test_max_num_batches",
                        help="Max number of testing batches waiting in queue",
                        type=int, default=50)
    parser.add_argument("--batchwise_test_fanouts",
                        help="Testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")
    # Final test settings
    parser.add_argument("--final_test_batchsize",
                        help="Size of testing batches",
                        type=int, default=1024)
    parser.add_argument("--final_test_fanouts",
                        help="Testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")

    # Optional Argument
    # New add, Progress bar
    parser.add_argument("--pbar",
                        help="Use progress bar function",
                        action="store_true")
    # Trial
    parser.add_argument("--use_trial",
                        help="Use trial run",
                        action="store_true")
    parser.add_argument("--trials",
                        help="Number of trials to run",
                        type=int, default=10)
    # Test only mode
    parser.add_argument("--do_test_run",
                        help="Only run inference on the test set",
                        action="store_true")
    parser.add_argument("--do_test_run_filename",
                        help="The filename of model to load for the test run",
                        type=str, default=f"NONE", nargs='*')
    # Runtime Statistics
    parser.add_argument("--performance_stats",
                        help="Collect detailed performance statistics.",
                        action="store_true")
    # LR scheduler
    parser.add_argument("--use_lrs",
                        help="Use learning rate scheduler",
                        action="store_true")
    parser.add_argument("--patience",
                        help="Patience to use in the LR scheduler",
                        type=int, default=1000)

    # ADD issue-5
    parser.add_argument("--dev_mode",
                        help="Set device mode(cuda or cpu)",
                        type=str, default=Settings.DEVICE_MODE)

    parser.add_argument("--dev_select_mode",
                        help="Select device index mode()",
                        type=str, default=Settings.SELECT_DEVICE_MODE)


    return parser
