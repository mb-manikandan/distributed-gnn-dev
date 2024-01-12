#!/bin/bash


# parameters
echo "node_num" $1
echo "total_num_nodes" $2
echo "max_num_devices_per_node, server_num_worker_node" $3
echo "ddp_addr" $4
echo "rpc_addr" $5
echo "device_mode" $6


ulimit -n

# Set JOB_NAME used for this script
JOB_NAME=example_multi_machine

# Set Node name
export NODENAME=`hostname`

# Set PROGRAM root and PYTHONPATH
# PROGRAM_ROOTはDistributedGNNフォルダのパスを指定する。利用者が任意の場所を指定する(下記は記述例)。
PROGRAM_ROOT=$HOME/work/gnn_src/distGNN/DistributedGNN
export PYTHONPATH=$PROGRAM_ROOT:$PROGRAM_ROOT/TrainerBase


# Set the log paths
# logの出力先の設定。利用者が任意の場所を指定する(下記は記述例)。
OUTPUT_ROOT=$HOME/work/slurm_work/job_output

echo "PROGRAM_ROOT" $PROGRAM_ROOT
echo "JOB_NAME" $JOB_NAME
echo "OUTPUT_ROOT" $OUTPUT_ROOT
echo "PYTHONPATH" $PYTHONPATH

# PYTHONBINはPythonインタープリタの配置場所。利用者が任意の場所を指定する(下記は記述例)。
export PYTHONBIN=$HOME/.pyenv/versions/SALIENT_2_230605/bin
$PYTHONBIN/python $PROGRAM_ROOT/TrainerBase/example/ns_main.py ogbn-arxiv $JOB_NAME \
       --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --overwrite_job_dir --num_workers 1 \
       --node_num $1 \
       --total_num_nodes $2 --max_num_devices_per_node $3  \
       --server_total_num_nodes $2 --server_num_worker_node $3 \
       --ddp_addr $4 --rpc_addr $5 \
       --use_model_name model_sage.pt \
       --dev_mode $6 \
#       --logger # コメントを外すと標準出力からloggerへの出力に変わる。



# 実行時の引数例
# source Script/exec_multi_machine_ns.sh node番号 nodeの合計数 1node当たりのtrainer(またはserver)の数(1以上) host名(DDP通信のマスターとなるホスト名) host名(RPC通信のマスターとなるホスト名) device_mode(cpu, cuda)

# 実行例(1nodeの場合)
# source Script/exec_multi_machine_ns.sh 0 1 3 localhost localhost "cuda"

# 実行例(2node以上の場合)
# マスタノード(rank=0): source Script/exec_multi_machine_ns.sh 0 [2~] 3 マスタノードのホスト名 マスタノードのホスト名 device_mode
# ※マスターノード実行時は、node番号は0でつける。また、node番号は0のhost名でDDP、RPC通信のマスターとなるホスト名を指定する。

# 他ノード(rank=2~): source Script/exec_multi_machine_ns.sh [1~] [2~] 3 マスタノードのホスト名 マスタノードのホスト名 device_mode
# ※他ノード実行時は、node番号は0から連番(1,2...)でつける。
# 　nodeの合計数と1node当たりのtrainer(またはserver)の数(1以上)、host名は、全ノード(マスターノードと他ノード)で必ず同じ値で設定する。

