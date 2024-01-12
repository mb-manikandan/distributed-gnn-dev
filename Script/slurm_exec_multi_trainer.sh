#!/bin/bash

#実行したいnodeのpartition名を指定
gnn_node_list=("gnn_node1")
#gnn_node_list=("gnn_node1" "gnn_node0")

#not_list=("")

#slurm_workはSlurmの作業フォルダのパスを指定する。利用者が任意の場所を指定する(下記は記述例)。
#また管理ノードと全計算ノードの作業フォルダは同じフォルダ構造にする。
slurm_work=$HOME"/work/slurm_work"
output=$slurm_work"/logs/slurm-%j.out" # 「$slurm_work/logs」は事前に作成しておく。
program_path=$slurm_work"/SlurmScript"


num_devices=3

node_name="localhost"

size_node_list=${#gnn_node_list[*]}
echo "Size of gnn_node_list" $size_node_list

if [ $size_node_list > 1 ]; then
  sinfo_arr=($(sinfo -s | grep ${gnn_node_list[0]} | tr " " "\n"))
  node_name=${sinfo_arr[4]} # 対応するpartition名からノード名を取得
fi
echo "trainer node" $node_name


node_num=0
for gnn_node in ${gnn_node_list[@]}
do
  echo "exec trainer" ${gnn_node}

#  ddp_node_num=$node_num
#  rpc_server_node_num=$node_num
#  rpc_trainer_node_num=$((${node_num}+${size_node_list}))

#  echo "ddp_node_num" $ddp_node_num
#  echo "rpc_server_node_num" $rpc_server_node_num
#  echo "rpc_trainer_node_num" $rpc_trainer_node_num

  echo "node_num" $node_num

  sbatch -p ${gnn_node} -c $num_devices --mincpus $num_devices -o ${output} ${program_path}/exec_multi_machine_ns.sh \
        $node_num $size_node_list $num_devices $node_name $node_name

  echo ""

  node_num=${node_num+1}

done

sleep 2

echo "slurm info"
sinfo -s
squeue
