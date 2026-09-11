# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List
from hcu_train_simulator.communication.types import GPUType


@dataclass
class GPU:
    rank: int
    node_id: int


@dataclass
class Node:
    id: int
    gpu_ranks: List[int] = field(default_factory=list)
    nvswitch_ids: List[int] = field(default_factory=list)


@dataclass
class Topology:
    nodes: List[Node]
    gpus: List[GPU]
    gpu_type: GPUType
    
    @classmethod
    def create_uniform(cls, num_nodes: int, gpus_per_node: int, 
                       gpu_type: GPUType = GPUType.BW1000) -> 'Topology':
        """创建均匀拓扑：每节点相同数量的 GPU"""
        nodes = []
        gpus = []
        rank = 0
        for node_id in range(num_nodes):
            gpu_ranks = []
            for _ in range(gpus_per_node):
                gpus.append(GPU(rank, node_id))
                gpu_ranks.append(rank)
                rank += 1
            nodes.append(Node(id=node_id, gpu_ranks=gpu_ranks))
        return cls(nodes=nodes, gpus=gpus, gpu_type=gpu_type)
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
    
    def get_node_id(self, rank: int) -> int:
        return self.gpus[rank].node_id
    
    def get_gpu_ranks_in_node(self, node_id: int) -> List[int]:
        return self.nodes[node_id].gpu_ranks
