#!/bin/bash

# Set Node name
export NODENAME=`hostname`

ulimit -n

# Set PROGRAM root and PYTHONPATH
# PROGRAM_ROOTはDistributedGNNフォルダのパスを指定する。利用者が任意の場所を指定する(下記は記述例)。
PROGRAM_ROOT=$HOME/work/gnn_src/distGNN/DistributedGNN
export PYTHONPATH=$PROGRAM_ROOT

# Set the log paths
# logの出力先の設定。利用者が任意の場所を指定する(下記は記述例)。
DATASET_ROOT=$HOME/work/gnn_src/distGNN/dataset
OUTPUT_ROOT=$HOME/work/slurm_work/job_output


echo "PROGRAM_ROOT" $PROGRAM_ROOT
echo "DATASET_ROOT" $DATASET_ROOT
echo "OUTPUT_ROOT" $OUTPUT_ROOT
echo "PYTHONPATH" $PYTHONPATH


# PYTHONBINはPythonインタープリタの配置場所。利用者が任意の場所を指定する(下記は記述例)。
export PYTHONBIN=$HOME/.pyenv/versions/SALIENT_2_230605/bin
$PYTHONBIN/python $PROGRAM_ROOT/GraphStoreServer/server/graph_store_server.py \
      GraphStoreServer --output_root $OUTPUT_ROOT --overwrite_job_dir \
#      --logger # コメントを外すと標準出力からloggerへの出力に変わる。


# 実行例(全ノードで実行)
# source SlurmScript/run_graph_store_server_ns.sh

