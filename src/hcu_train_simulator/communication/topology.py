# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List
from hcu_train_simulator.communication.types import GPUType, LinkType


@dataclass
class GPU:
    rank: int
    node_id: int


@dataclass
class Node:
    id: int
    gpu_ranks: List[int] = field(default_factory=list)
    nvswitch_ids: List[int] = field(default_factory=list)
    supernode_id: int = 0


@dataclass
class Topology:
    nodes: List[Node]
    gpus: List[GPU]
    gpu_type: GPUType
    nodes_per_supernode: int = 1
    use_supernode: bool = False

    @classmethod
    def create_uniform(cls, num_nodes: int, gpus_per_node: int,
                       gpu_type: GPUType = GPUType.BW1000,
                       nodes_per_supernode: int = 1,
                       use_supernode: bool = False) -> 'Topology':
        """创建均匀拓扑：每节点相同数量的 GPU"""
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        if nodes_per_supernode <= 0:
            raise ValueError("nodes_per_supernode must be positive")
        nodes = []
        gpus = []
        rank = 0
        for node_id in range(num_nodes):
            gpu_ranks = []
            for _ in range(gpus_per_node):
                gpus.append(GPU(rank, node_id))
                gpu_ranks.append(rank)
                rank += 1
            nodes.append(Node(
                id=node_id,
                supernode_id=node_id // nodes_per_supernode,
                gpu_ranks=gpu_ranks,
            ))
        return cls(
            nodes=nodes,
            gpus=gpus,
            gpu_type=gpu_type,
            nodes_per_supernode=nodes_per_supernode,
            use_supernode=use_supernode,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def num_supernodes(self) -> int:
        return len({node.supernode_id for node in self.nodes})

    def get_node_id(self, rank: int) -> int:
        return self.gpus[rank].node_id

    def get_gpu_ranks_in_node(self, node_id: int) -> List[int]:
        return self.nodes[node_id].gpu_ranks

    def get_supernode_id(self, rank: int) -> int:
        return self.nodes[self.get_node_id(rank)].supernode_id

    def get_link_type(self, src_rank: int, dest_rank: int) -> LinkType:
        """Classify a flow using its physical placement, independent of parallelism."""
        src_node = self.get_node_id(src_rank)
        dest_node = self.get_node_id(dest_rank)
        if not self.use_supernode:
            return (
                LinkType.INTRA_NODE
                if src_node == dest_node
                else LinkType.SCALE_OUT
            )
        if src_node == dest_node:
            return LinkType.SCALE_UP_1
        if self.nodes[src_node].supernode_id == self.nodes[dest_node].supernode_id:
            return LinkType.SCALE_UP_2
        return LinkType.SCALE_OUT
