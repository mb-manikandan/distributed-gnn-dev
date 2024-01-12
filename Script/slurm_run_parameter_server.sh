#!/bin/bash

#not_list=("")

node_name="gnn_node1"
start_number=0
end_number=1

#slurm_workはSlurmの作業フォルダのパスを指定する。利用者が任意の場所を指定する(下記は記述例)。
#また管理ノードと全計算ノードの作業フォルダは同じフォルダ構造にする。
slurm_work=$HOME"/work/slurm_work"
output=$slurm_work"/logs/slurm-%j.out" # 「$slurm_work"/logs」は事前に作成しておく。
program_path=$slurm_work"/SlurmScript"

device_per_node=1


sbatch -p ${node_name} -c $device_per_node --mincpus 1 -o ${output} ${program_path}/exec_model_store_process.sh


sleep 2

sinfo -s
squeue
