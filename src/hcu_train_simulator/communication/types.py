# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List
from enum import Enum

class GroupType(str, Enum):
    """通信组类型"""
    TP = "TP"
    CP = "CP"
    DP = "DP"
    EP = "EP"
    ETP = "ETP"
    PP = "PP"
    EDP = "EDP"


class CommType(str, Enum):
    """集合通信类型"""
    ALL_REDUCE = "ALL_REDUCE"
    ALL_GATHER = "ALL_GATHER"
    REDUCE_SCATTER = "REDUCE_SCATTER"
    ALL_TO_ALL = "ALL_TO_ALL"
    P2P = "P2P"


class GPUType(str, Enum):
    """GPU类型"""
    BW1100 = "bw1100"
    BW1000 = "bw1000"
    BW500SM = "bw500SM"


class Algorithm(str, Enum):
    """通信算法"""
    RING = "RING"
    TREE = "TREE"
    NVLS = "NVLS"
    NVLS_TREE = "NVLS_TREE"

@dataclass
class SingleFlow:
    """点对点通信原语"""
    flow_id: int
    src: int
    dest: int
    size: int  # 字节
    prev_flows: List[int] = field(default_factory=list)
    child_flows: List[int] = field(default_factory=list)
    channel_id: int = 0
    chunk_id: int = 0
    chunk_count: int = 0
    tag: str = ""
    
    # def __repr__(self):
    #     return (f"Flow({self.flow_id}: {self.src}→{self.dest}, "
    #             f"size={self.size}, tag={self.tag})")
