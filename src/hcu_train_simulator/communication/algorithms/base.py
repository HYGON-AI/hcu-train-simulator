# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from abc import ABC, abstractmethod
from hcu_train_simulator.communication.flows import FlowGraph
from hcu_train_simulator.communication.groups import GroupInfo
from hcu_train_simulator.communication.types import CommType

class CommunicationAlgorithm(ABC):
    """通信算法抽象基类"""
    
    @abstractmethod
    def decompose(self, rank: int, group_info: GroupInfo,
                  com_type: CommType, data_size: int) -> FlowGraph:
        """将集合通信拆解为通信原语"""
        pass
