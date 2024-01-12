# GNN Model default device
####################################################################################
DEVICE_MODE = "cuda"  # "cuda" or "cpu"
# 下記以外(default): rank番号(trainerの子processのrank番号)
# 0: ((trainer番号×総子プロセス数)+rank番号) % device総数(コンテナ内のGPU総数),
SELECT_DEVICE_MODE = 0
####################################################################################