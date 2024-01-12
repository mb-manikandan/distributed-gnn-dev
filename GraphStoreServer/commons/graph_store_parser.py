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
    # Job_name
    parser.add_argument("job_name",
                        help="Name of the Job",
                        type=str)
    # Config file
    parser.add_argument("--config_file",
                        help="Use config file rather than command line arg",
                        type=open, action=LoadFromFile, default=None)

    # Input/Output file path
    parser.add_argument("--dataset_root",
                        help="Dataset root path",
                        type=str, default=f"fast_dataset/")
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


    return parser
