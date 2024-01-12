from typing import Any, Optional, Callable, List
from pathlib import Path

import torch

# 学習コア処理(forward, backward): Callable[[学習モデル(nn.Module), グラフのバッチデータ(Any)], 戻り値(Any)]
TrainCore = Callable[[torch.nn.Module, Any], Any]

# 学習のCallBack関数: Callable[[グラフのバッチデータのList(List[Any]), List[Any]], 戻り値(None)]
TrainCallback = Callable[[List[Any], List[Any]], None]

# 学習処理のインターフェース関数: Callable[[学習モデル(nn.Module), 最適化関数(optim.Optimizer),
#                           receiverクラス(Any), logファイルパス
#                           学習のCallBack関数(Optional[TrainCallback]], 学習のCallBack関数(Optional[TrainCallback]],
#                           GPUデバイス(List[torch.device])), 戻り値(None)]
TrainImpl = Callable[[torch.nn.Module, torch.optim.Optimizer, Any, Path,
                      Optional[TrainCallback], Optional[TrainCallback], List[torch.device]], None]

# テスト処理のCallBack関数: Callable[[グラフのバッチデータのList(List[Any]), List[Any]], 戻り値(None)]
TestCallback = Callable[[Any], None]  # should not return anything
