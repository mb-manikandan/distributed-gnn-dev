from pathlib import Path
import os
from datetime import datetime

from TrainerBase.commons.models import SAGE, GAT, GIN, SAGEResInception
from TrainerBase.commons.models import SAGEClassic, JKNet, GCN, ARMA


def get_model_type(model_name):
    assert model_name.lower() in ['sage', 'gat', 'gin', 'sageresinception',
                                  'sageclassic', 'jknet', 'gcn', 'arma']

    if model_name.lower() == 'sage':
        return SAGE                                                   # works
    if model_name.lower() == 'gat':
        return GAT                                                    # works
    if model_name.lower() == 'gin':
        return GIN              # works. does not support layerwise inference
    if model_name.lower() == 'sageresinception':
        return SAGEResInception # works. does not support layerwise inference
    if model_name.lower() == 'sageclassic':
        return SAGEClassic                                # not used in paper
    if model_name.lower() == 'jknet':
        return JKNet                                      # not used in paper
    if model_name.lower() == 'gcn':
        return GCN                                        # not used in paper
    if model_name.lower() == 'arma':
        return ARMA                                                  # broken


def get_job_dir(args):
    return Path(args.output_root).joinpath(args.job_name)


def prepare_log_file(args, date_time="_%s" % datetime.now().strftime("%Y%m%d-%H%M%S")):
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
    node_name = str(os.environ['NODENAME'])

    # Optional write args.txt
    # Write the args to the job dir for reproducibility
    with job_dir.joinpath(node_name + "_args" + date_time + ".txt").open('w') as f:
        f.write(repr(args))

    return job_dir.joinpath(node_name + "_trainer_logs" + date_time + ".txt")

