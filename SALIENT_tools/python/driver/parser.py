import argparse


def add_parser_SALIENT(parser):
    # SALIENT Argument
    # DDP settings
    parser.add_argument("--ddp_dir",
                        help="Coordination directory for ddp multinode jobs",
                        type=str, default=f"NONE")
    parser.add_argument("--one_node_ddp",
                        help="Do DDP when total_num_nodes=1",
                        action="store_true")
    # Train settings
    parser.add_argument("--train_sampler",
                        help="Training sampler",
                        type=str, default="FastSampler")
    parser.add_argument("--train_prefetch",
                        help="Prefetch for training",
                        type=int, default=1)
    parser.add_argument("--train_type",
                        help="Training Type",
                        type=str, default="serial",
                        choices=("serial", "dp"))
    # Test settings
    parser.add_argument("--test_prefetch",
                        help="Prefetch for testing",
                        type=int, default=1)
    parser.add_argument("--test_type",
                        help="Testing type",
                        type=str, default="batchwise",
                        choices=("layerwise", "batchwise"))

    return parser
