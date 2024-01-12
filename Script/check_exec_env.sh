#!/bin/bash

# Set JOB_NAME used for this script
JOB_NAME=example_multi_machine

# Set Node name
export NODENAME=`hostname`

# Set PROGRAM root and PYTHONPATH
# PROGRAM_ROOTはDistributedGNNフォルダのパスを指定する。利用者が任意の場所を指定する(下記は記述例)。
PROGRAM_ROOT=$HOME/work/gnn_src/distGNN/DistributedGNN
export PYTHONPATH=$PROGRAM_ROOT:$PROGRAM_ROOT/SamplerBase


# Set the data paths
# DATASET_ROOTはデータセットの配置場所。利用者が任意の場所を指定する(下記は記述例)。
#DATASET_ROOT=$HOME/dataset
DATASET_ROOT=$HOME/work/gnn_src/distGNN/dataset
OUTPUT_ROOT=$HOME/work/slurm_work/job_output

export PYTHONBIN=$HOME/.pyenv/versions/SALIENT_2_230605/bin


echo $NODENAME
echo $PROGRAM_ROOT $JOB_NAME $DATASET_ROOT $OUTPUT_ROOT
echo $PYTHONPATH $PYTHONBIN


