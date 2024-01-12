import shutil
from pathlib import Path
import os
from datetime import datetime

from GraphStoreServer.commons.dataset import PyGDataset


def get_dataset(dataset_name, root):
    assert dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']
    return PyGDataset.from_path(root, dataset_name)


def get_job_dir(args):
    return Path(args.output_root).joinpath(args.job_name)


def prepare_log_file(args, file_name="",  date_time="_%s" % datetime.now().strftime("%Y%m%d-%H%M%S")):
    job_dir = get_job_dir(args)

    if job_dir.exists():
        assert job_dir.is_dir()
        # if args.overwrite_job_dir:
        #     shutil.rmtree(job_dir)
        # else:
        #     raise ValueError(
        #         f'job_dir {job_dir} exists. Use a different job name ' +
        #         'or set --overwrite_job_dir')
        # job_dir.mkdir(parents=True)
    else:
        job_dir.mkdir(parents=True)

    # Get node_name
    node_name = str(os.environ['NODENAME']) + "_"

    # Optional write args.txt
    # Write the args to the job dir for reproducibility
    # with job_dir.joinpath(node_name + "args.txt").open('w') as f:
    #     f.write(repr(args))

    return job_dir.joinpath(node_name + file_name + date_time + ".txt")
