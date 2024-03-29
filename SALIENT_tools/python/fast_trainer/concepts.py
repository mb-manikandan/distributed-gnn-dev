from typing import Any, Optional, Callable, List
import torch

from SALIENT_tools.python.fast_trainer.samplers import PreparedBatch
from SALIENT_tools.python.fast_trainer.transferers import DeviceIterator

TrainCore = Callable[[torch.nn.Module, PreparedBatch], Any]
TrainCallback = Callable[[List[PreparedBatch], List[Any]], None]
TrainImpl = Callable[[torch.nn.Module, TrainCore, DeviceIterator,
                      torch.optim.Optimizer, Optional[TrainCallback]], None]
TestCallback = Callable[[PreparedBatch], None]  # should not return anything
