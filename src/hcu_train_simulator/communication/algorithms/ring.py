# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple
from hcu_train_simulator.communication.algorithms.base import CommunicationAlgorithm
from hcu_train_simulator.communication.flows import FlowGraph
from hcu_train_simulator.communication.groups import GroupInfo
from hcu_train_simulator.communication.types import CommType, SingleFlow

class RingAlgorithm(CommunicationAlgorithm):
    """Ring 通信算法实现。"""
    
    def __init__(self, gpus_per_node: int, enable_pxn: bool = False):
        super().__init__()
        self.gpus_per_node = gpus_per_node
        self.enable_pxn = enable_pxn
        self._flow_counter = 0
        self._n_ringchennals = 32
    
    def _next_flow_id(self) -> int:
        fid = self._flow_counter
        self._flow_counter += 1
        return fid
    
    def decompose(self, rank: int, group_info: GroupInfo, 
                  com_type: CommType, data_size: int) -> FlowGraph:
        """将集合通信拆解为 Ring 上的点对点通信流。"""
        if com_type == CommType.ALL_REDUCE:
            return self._decompose_all_reduce(rank, group_info, data_size)
        elif com_type == CommType.REDUCE_SCATTER:
            return self._decompose_reduce_scatter(rank, group_info, data_size)
        elif com_type == CommType.ALL_GATHER:
            return self._decompose_all_gather(rank, group_info, data_size)
        elif com_type == CommType.ALL_TO_ALL:
            return self._decompose_all_to_all(rank, group_info, data_size)
        elif com_type == CommType.P2P:
            return self._decompose_p2p(rank, group_info, data_size)
        else:
            raise ValueError(f"Unsupported CommType: {com_type}")

    def _decompose_reduce_scatter(self, rank: int, group_info: GroupInfo,
                                data_size: int) -> FlowGraph:
        """拆解 Reduce-Scatter 通信。"""
        result = FlowGraph()
        nranks = group_info.n_ranks
        
        if nranks == 1:
            return result
        
        # 构建 Ring 拓扑。
        ring = self._build_ring_topology(rank, group_info)
        
        # 计算每个数据块的大小。
        chunksize = data_size // nranks
        chunk_count = nranks - 1
        task_list = {}
        
        # 第一轮：每个节点发送自己的数据块。
        for cur_rank, next_rank, prev_rank in ring:
            if (next_rank // self.gpus_per_node) != (cur_rank // self.gpus_per_node):
                tag = "PXN"
            else:
                tag = "RING"
            
            flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=cur_rank,
                dest=next_rank,
                size=chunksize,
                prev_flows=[],  # 第一轮没有前置依赖。
                child_flows=[],
                channel_id=0,
                chunk_id=0,
                chunk_count=chunk_count,
                tag=tag
            )
            result.add((0, flow.flow_id), flow)
            task_list[cur_rank] = flow
        
        # 后续轮次转发上一轮收到的数据。
        for step in range(1, nranks - 1):
            new_task_list = {}
            for cur_rank, prev_flow in task_list.items():
                next_rank = self._get_next_rank(ring, cur_rank)
                prev_rank = self._get_prev_rank(ring, cur_rank)
                
                if (next_rank // self.gpus_per_node) != (cur_rank // self.gpus_per_node):
                    tag = "PXN"
                else:
                    tag = "RING"
                
                # 当前节点需要等待自身和前序节点上一轮的发送流。
                prev_flows = [prev_flow.flow_id]
                if prev_rank in task_list:
                    recv_flow = task_list[prev_rank]
                    prev_flows.append(recv_flow.flow_id)
                
                new_flow = SingleFlow(
                    flow_id=self._next_flow_id(),
                    src=cur_rank,
                    dest=next_rank,
                    size=chunksize,
                    prev_flows=prev_flows,
                    child_flows=[],
                    channel_id=0,
                    chunk_id=step,
                    chunk_count=chunk_count,
                    tag=tag
                )
                result.add((0, new_flow.flow_id), new_flow)
                
                # 建立通信流依赖。
                for pfid in prev_flows:
                    parent_flow = result.get((0, pfid))
                    if parent_flow:
                        parent_flow.child_flows.append(new_flow.flow_id)
                
                new_task_list[cur_rank] = new_flow
            
            task_list = new_task_list
        
        return result

    def _decompose_all_gather(self, rank: int, group_info: GroupInfo,
                            data_size: int) -> FlowGraph:
        """拆解 AllGather 通信。"""
        result = FlowGraph()
        nranks = group_info.n_ranks
        
        if nranks == 1:
            return result
        
        ring = self._build_ring_topology(rank, group_info)
        
        chunksize = data_size // nranks
        chunk_count = nranks - 1
        task_list = {}
        
        # 第一轮：每个节点发送自己的数据块。
        for cur_rank, next_rank, prev_rank in ring:
            if (next_rank // self.gpus_per_node) != (cur_rank // self.gpus_per_node):
                tag = "PXN"
            else:
                tag = "RING"
            
            flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=cur_rank,
                dest=next_rank,
                size=chunksize,
                prev_flows=[],  # 第一轮没有前置依赖。
                child_flows=[],
                channel_id=0,
                chunk_id=0,
                chunk_count=chunk_count,
                tag=tag
            )
            result.add((0, flow.flow_id), flow)
            task_list[cur_rank] = flow
        
        # 后续轮次转发上一轮收到的数据。
        for step in range(1, nranks - 1):
            new_task_list = {}
            for cur_rank, prev_flow in task_list.items():
                next_rank = self._get_next_rank(ring, cur_rank)
                prev_rank = self._get_prev_rank(ring, cur_rank)
                
                if (next_rank // self.gpus_per_node) != (cur_rank // self.gpus_per_node):
                    tag = "PXN"
                else:
                    tag = "RING"
                
                # 当前节点需要等待自身和前序节点上一轮的发送流。
                prev_flows = [prev_flow.flow_id]
                if prev_rank in task_list:
                    recv_flow = task_list[prev_rank]
                    prev_flows.append(recv_flow.flow_id)
                
                new_flow = SingleFlow(
                    flow_id=self._next_flow_id(),
                    src=cur_rank,
                    dest=next_rank,
                    size=chunksize,
                    prev_flows=prev_flows,
                    child_flows=[],
                    channel_id=0,
                    chunk_id=step,
                    chunk_count=chunk_count,
                    tag=tag
                )
                result.add((0, new_flow.flow_id), new_flow)
                
                # 建立通信流依赖。
                for pfid in prev_flows:
                    parent_flow = result.get((0, pfid))
                    if parent_flow:
                        parent_flow.child_flows.append(new_flow.flow_id)
                
                new_task_list[cur_rank] = new_flow
            
            task_list = new_task_list
        
        return result

    def _decompose_all_reduce(self, rank: int, group_info: GroupInfo,
                            data_size: int) -> FlowGraph:
        """AllReduce = Reduce-Scatter + AllGather"""
        nranks = group_info.n_ranks
        
        if nranks == 1:
            return FlowGraph()
        
        # 每个 GPU 在 Reduce-Scatter 后持有一个数据块。
        chunk_size = data_size // nranks

        # 第一阶段：Reduce-Scatter。
        rs_flows = self._decompose_reduce_scatter(rank, group_info, data_size)

        # 第二阶段：AllGather。
        ag_flows = self._decompose_all_gather(rank, group_info, data_size)

        # 合并两个阶段的通信流。
        result = FlowGraph()
        for key, flow in rs_flows.flows.items():
            result.add(key, flow)
        for key, flow in ag_flows.flows.items():
            result.add(key, flow)

        # AllGather 的第一轮必须等待 Reduce-Scatter 的最后一轮完成。
        # Reduce-Scatter 最后一轮的 chunk_id 为 nranks - 2。
        last_rs_chunk_id = nranks - 2
        last_rs_flows = {}  # rank -> flow_id
        
        for key, flow in rs_flows.flows.items():
            if flow.chunk_id == last_rs_chunk_id:
                last_rs_flows[flow.src] = flow.flow_id

        # AllGather 第一轮的 chunk_id 为 0。
        first_ag_flows = {}
        for key, flow in ag_flows.flows.items():
            if flow.chunk_id == 0 and flow.src not in first_ag_flows:
                first_ag_flows[flow.src] = flow

        # 建立跨阶段依赖。
        for rank_id, ag_flow in first_ag_flows.items():
            if rank_id in last_rs_flows:
                ag_flow.prev_flows.append(last_rs_flows[rank_id])
                parent_flow = rs_flows.get((0, last_rs_flows[rank_id]))
                if parent_flow:
                    parent_flow.child_flows.append(ag_flow.flow_id)

        return result

    def _decompose_all_to_all(self, rank: int, group_info: GroupInfo,
                            data_size: int) -> FlowGraph:
        """拆解 AllToAll 通信。"""
        result = FlowGraph()
        nranks = group_info.n_ranks
        
        if nranks == 1:
            return result
        
        ranks = sorted(group_info.ranks)
        # 每个 GPU 向其他 GPU 发送的数据块大小。
        chunk_size = data_size // nranks
        
        # 每个 rank 向组内其他 rank 发送数据。
        for src in ranks:
            for dst in ranks:
                if src == dst:
                    continue
                
                # 跨节点通信使用 PXN 标签。
                if (dst // self.gpus_per_node) != (src // self.gpus_per_node):
                    tag = "PXN"
                else:
                    tag = "RING"
                
                flow = SingleFlow(
                    flow_id=self._next_flow_id(),
                    src=src,
                    dest=dst,
                    size=chunk_size,
                    prev_flows=[],  # AllToAll 中的通信流可并行执行。
                    child_flows=[],
                    channel_id=0,
                    chunk_id=0,
                    chunk_count=1,
                    tag=tag
                )
                result.add((0, flow.flow_id), flow)
        
        return result
    
    def _decompose_p2p(self, rank: int, group_info: GroupInfo,
                    data_size: int) -> FlowGraph:
        """
        Megatron 风格的 P2P 通信。

        通信仅发生在相邻 stage 之间，并按流水线方向串联。
        """
        result = FlowGraph()
        ranks = sorted(group_info.ranks)
        if len(ranks) <= 1:
            return result
        
        # 前向：当前 stage 发送到下一个 stage。
        prev_flows = []
        child_flows = []
        for i in range(len(ranks) - 1):
            src = ranks[i]
            dst = ranks[i + 1]

            if i != 0 :prev_flows = [self._flow_counter - 1]
            if i == len(ranks) - 1 : child_flows = []
            else:child_flows = [self._flow_counter + 1 ]
            
            tag = "PXN" if (dst // self.gpus_per_node) != (src // self.gpus_per_node) else "RING"
            
            flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=src,
                dest=dst,
                size=data_size,
                prev_flows=prev_flows,
                child_flows=child_flows,
                channel_id=0,
                chunk_id=0,
                chunk_count=1,
                tag=tag
            )
            result.add((0, flow.flow_id), flow)
        # 反向：当前 stage 将梯度发送到上一个 stage。
        prev_flows = []
        child_flows = []
        for i in range(len(ranks) - 1, 0, -1):
            src = ranks[i]
            dst = ranks[i - 1]
            if i != 1  : child_flows = [self._flow_counter + 1]
            else:child_flows = []
            if i == len(ranks) - 1: prev_flows = []
            else:prev_flows = [self._flow_counter - 1]
            
            tag = "PXN" if (dst // self.gpus_per_node) != (src // self.gpus_per_node) else "RING"
            
            flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=src,
                dest=dst,
                size=data_size,
                prev_flows=prev_flows,
                child_flows=child_flows,
                channel_id=0,
                chunk_id=0,
                chunk_count=1,
                tag=tag
            )
            result.add((0, flow.flow_id), flow)
        return result

    def _build_ring_topology(self, rank: int, group_info: GroupInfo) -> List[Tuple[int, int, int]]:
        """构建从当前 rank 开始的 Ring 拓扑。"""
        ranks = sorted(group_info.ranks)
        n = len(ranks)
        # 找到当前 rank 在 Ring 中的位置。
        try:
            idx = ranks.index(rank)
        except ValueError:
            idx = 0
        
        ring = []
        for i in range(n):
            cur = ranks[(idx + i) % n]
            nxt = ranks[(idx + i + 1) % n]
            prv = ranks[(idx + i - 1) % n]
            ring.append((cur, nxt, prv))
        
        return ring
    
    def _get_next_rank(self, ring: List[Tuple[int, int, int]], cur_rank: int) -> int:
        for cur, nxt, _ in ring:
            if cur == cur_rank:
                return nxt
        return -1
    
    def _get_prev_rank(self, ring: List[Tuple[int, int, int]], cur_rank: int) -> int:
        for cur, _, prv in ring:
            if cur == cur_rank:
                return prv
        return -1

