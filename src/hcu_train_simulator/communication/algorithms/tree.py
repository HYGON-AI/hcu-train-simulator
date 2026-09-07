# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import List, Tuple, Optional
from hcu_train_simulator.communication.algorithms.base import CommunicationAlgorithm
from hcu_train_simulator.communication.flows import FlowGraph
from hcu_train_simulator.communication.groups import GroupInfo
from hcu_train_simulator.communication.types import CommType, SingleFlow


class TreeAlgorithm(CommunicationAlgorithm):
    """Tree-based collective communication algorithm."""

    def __init__(self, enable_pxn: bool = False):
        super().__init__()
        self.enable_pxn = enable_pxn
        self._flow_counter = 0

    def _next_flow_id(self) -> int:
        fid = self._flow_counter
        self._flow_counter += 1
        return fid

    def decompose(self, rank: int, group_info: GroupInfo,
                  com_type: CommType, data_size: int) -> FlowGraph:
        """Decompose a collective operation into tree communication flows."""
        if com_type == CommType.ALL_REDUCE:
            return self._decompose_all_reduce(rank, group_info, data_size)
        else:
            raise ValueError(f"Tree algorithm only supports ALL_REDUCE, got {com_type}")

    # ==================== 树拓扑构建 ====================

    def _build_tree_topology(self, rank: int, group_info: GroupInfo) -> Tuple[int, List[int]]:
        """
        构建完全二叉树拓扑。

        返回 ``(parent, children)``，其中 ``parent == -1`` 表示根节点。
        """
        ranks = sorted(group_info.ranks)
        n = len(ranks)
        try:
            idx = ranks.index(rank)
        except ValueError:
            idx = 0

        # 父节点索引。
        parent_idx = (idx - 1) // 2 if idx > 0 else -1
        parent = ranks[parent_idx] if parent_idx >= 0 else -1

        # 子节点索引。
        children = []
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        if left_idx < n:
            children.append(ranks[left_idx])
        if right_idx < n:
            children.append(ranks[right_idx])

        return parent, children

    # ==================== ALL_REDUCE ====================

    def _decompose_all_reduce(self, rank: int, group_info: GroupInfo,
                              data_size: int) -> FlowGraph:
        """Tree AllReduce = upward Reduce + downward Broadcast."""
        nranks = group_info.n_ranks
        if nranks == 1:
            return FlowGraph()

        parent, children = self._build_tree_topology(rank, group_info)
        gpus_per_node = 8

        result = FlowGraph()

        # 第一阶段：向上 Reduce。
        reduce_flow = self._build_tree_reduce(rank, parent, children, data_size,
                                               gpus_per_node, result, group_info)

        # 第二阶段：向下 Broadcast。
        self._build_tree_broadcast(rank, parent, children, data_size,
                                    gpus_per_node, reduce_flow, result, group_info)

        return result

    def _build_tree_reduce(self, rank: int, parent: int, children: List[int],
                           data_size: int, gpus_per_node: int,
                           result: FlowGraph, group_info: GroupInfo) -> Optional[SingleFlow]:
        """
        递归构建 Reduce 阶段的通信流。

        返回当前节点向父节点发送的流，根节点返回 ``None``。
        """
        # 先处理所有子节点并收集它们的发送流。
        child_flows = []
        for child in children:
            child_parent, child_children = self._build_tree_topology(child, group_info)
            child_flow = self._build_tree_reduce(child, child_parent, child_children,
                                                  data_size, gpus_per_node, result, group_info)
            if child_flow:
                child_flows.append(child_flow)

        # 等待所有子节点完成。
        prev_flows = [cf.flow_id for cf in child_flows]

        # 非根节点向父节点发送归约后的数据。
        if parent != -1:
            tag = "PXN" if (parent // gpus_per_node) != (rank // gpus_per_node) else "RING"
            send_flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=rank,
                dest=parent,
                size=data_size,
                prev_flows=prev_flows,
                child_flows=[],
                channel_id=0,
                chunk_id=0,
                chunk_count=1,
                tag=tag
            )
            result.add((0, send_flow.flow_id), send_flow)

            # 建立通信流依赖。
            for pfid in prev_flows:
                parent_flow = result.get((0, pfid))
                if parent_flow:
                    parent_flow.child_flows.append(send_flow.flow_id)

            return send_flow
        else:
            # 根节点只需等待所有子节点完成。
            return None

    def _build_tree_broadcast(self, rank: int, parent: int, children: List[int],
                              data_size: int, gpus_per_node: int,
                              parent_flow: Optional[SingleFlow],
                              result: FlowGraph, group_info: GroupInfo):
        """
        递归构建 Broadcast 阶段的通信流。
        """
        # 非根节点需要等待父节点的广播流。
        prev_flows = []
        if parent_flow is not None:
            prev_flows = [parent_flow.flow_id]

        # 向每个子节点发送数据。
        for child in children:
            tag = "PXN" if (child // gpus_per_node) != (rank // gpus_per_node) else "RING"
            send_flow = SingleFlow(
                flow_id=self._next_flow_id(),
                src=rank,
                dest=child,
                size=data_size,
                prev_flows=prev_flows,
                child_flows=[],
                channel_id=0,
                chunk_id=0,
                chunk_count=1,
                tag=tag
            )
            result.add((0, send_flow.flow_id), send_flow)

            # 建立通信流依赖。
            for pfid in prev_flows:
                pflow = result.get((0, pfid))
                if pflow:
                    pflow.child_flows.append(send_flow.flow_id)

            # 递归广播到后续子节点。
            _, grand_children = self._build_tree_topology(child, group_info)
            self._build_tree_broadcast(child, rank, grand_children, data_size,
                                        gpus_per_node, send_flow, result, group_info)

