# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from dataclasses import dataclass, field
from typing import List, Dict
from hcu_train_simulator.communication.types import SingleFlow


@dataclass
class FlowGraph:
    """一组通信原语"""
    flows: Dict[tuple, SingleFlow] = field(default_factory=dict)
    
    def add(self, key: tuple, flow: SingleFlow):
        self.flows[key] = flow
    
    def get(self, key: tuple) -> SingleFlow:
        return self.flows.get(key)
    
    def get_by_rank(self, rank: int) -> List[SingleFlow]:
        """获取某个 rank 参与的所有流"""
        result = []
        for flow in self.flows.values():
            if flow.dest == rank:
                result.append(flow)
        return result
    def get_by_rank_p2p(self, rank: int) -> List[SingleFlow]:
        """获取某个 rank 参与的所有流"""
        result = []
        for flow in self.flows.values():            
            result.append(flow)
        return result    
    def __len__(self):
        return len(self.flows)
