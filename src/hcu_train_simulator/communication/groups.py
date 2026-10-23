# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Optional
from hcu_train_simulator.communication.topology import Topology
from hcu_train_simulator.communication.types import GroupType
from dataclasses import dataclass


@dataclass
class GroupInfo:
    group_id: int
    group_type: GroupType
    n_nodes: int
    n_ranks: int
    ranks: List[int]
    nvswitch_ids: List[int]
    n_supernodes: int = 1

    @property
    def ranks_per_node(self) -> int:
        return self.n_ranks // self.n_nodes if self.n_nodes > 0 else self.n_ranks


class GroupManager:
    """通信组管理器 - Megatron风格分组"""

    def __init__(self, topology: Topology):
        self.topology = topology
        self.rank_to_group: Dict[Tuple[int, GroupType], int] = {}
        self.groups: Dict[int, GroupInfo] = {}
        self._next_group_id = 0

    def _add_group(self, group_info: GroupInfo) -> int:
        group_id = self._next_group_id
        self._next_group_id += 1
        group_info.group_id = group_id
        self.groups[group_id] = group_info
        for rank in group_info.ranks:
            self.rank_to_group[(rank, group_info.group_type)] = group_id
        return group_id

    def _get_nvswitch_ids_for_ranks(self, ranks: List[int]) -> List[int]:
        """根据rank列表获取涉及的NVSwitch ID列表（去重）"""
        gpus_per_node = self.topology.num_gpus // self.topology.num_nodes
        nodes_set = set(r // gpus_per_node for r in ranks)
        nvswitch_ids = []
        for nid in nodes_set:
            if nid < len(self.topology.nodes) and self.topology.nodes[nid].nvswitch_ids:
                nvswitch_ids.append(self.topology.nodes[nid].nvswitch_ids[0])
        return nvswitch_ids

    def build_groups(self, tp_size: int = 1, cp_size: int = 1, dp_size: int = 1,
                     ep_size: int = 1, etp_size: int = 1, edp_size: int = 1, pp_size: int = 1):
        """
        按照两种独立的并行维度构建通信组：
        1) TP -> CP -> DP -> PP
        2) EP -> ETP -> EDP -> PP
        要求：TP*CP*DP*PP == total_gpus 且 EP*ETP*EDP*PP == total_gpus
        """
        total_gpus = self.topology.num_gpus
        gpus_per_node = self.topology.num_gpus // self.topology.num_nodes

        # ========== 验证乘积条件 ==========
        assert tp_size * cp_size * dp_size * pp_size == total_gpus, \
            f"TP*CP*DP*PP ({tp_size * cp_size * dp_size * pp_size}) != total_gpus ({total_gpus})"
        assert ep_size * etp_size * edp_size * pp_size == total_gpus, \
            f"EP*ETP*EDP*PP ({ep_size * etp_size * edp_size * pp_size}) != total_gpus ({total_gpus})"

        # ========== 辅助函数：根据坐标筛选 rank ==========
        def get_ranks_by_coord(rank_to_coord, coord_filter):
            ranks = []
            for rank, coord in rank_to_coord.items():
                if coord_filter(coord):
                    ranks.append(rank)
            return sorted(ranks)

        def collect_groups(rank_to_coord, dim_sizes, dim_names, group_type_map):
            """
            dim_sizes: dict 例如 {'pp': pp_size, 'dp': dp_size, 'cp': cp_size, 'tp': tp_size}
            dim_names: 顺序列表，例如 ['pp', 'dp', 'cp', 'tp']
            group_type_map: 每个维度对应的 GroupType，例如 {'tp': GroupType.TP, ...}
            返回各类型组的 ranks 列表字典
            """
            # 提取各维度范围
            ranges = {name: range(dim_sizes[name]) for name in dim_names}
            groups = {gtype: [] for gtype in group_type_map.values()}

            # 对于每个需要构建的维度，固定其他所有维度，变化该维度
            for target_dim, target_gtype in group_type_map.items():
                other_dims = [d for d in dim_names if d != target_dim]
                # 遍历其他维度的所有组合
                from itertools import product
                other_values_product = product(*[ranges[d] for d in other_dims])
                for other_vals in other_values_product:
                    # 构建过滤条件：其他维度固定为给定值
                    def make_filter(other_dims, other_vals, target_dim):
                        def filter_func(coord):
                            # coord 是一个元组，顺序与 dim_names 一致
                            for d, val in zip(other_dims, other_vals):
                                idx = dim_names.index(d)
                                if coord[idx] != val:
                                    return False
                            return True

                        return filter_func

                    filter_func = make_filter(other_dims, other_vals, target_dim)
                    ranks = get_ranks_by_coord(rank_to_coord, filter_func)
                    if ranks:
                        groups[target_gtype].append(ranks)
            return groups

        # ========== 1. 构建 TP/CP/DP/PP 坐标系统 ==========
        # 坐标顺序: (pp, dp, cp, tp)
        dim_names1 = ['pp', 'dp', 'cp', 'tp']
        dim_sizes1 = {'pp': pp_size, 'dp': dp_size, 'cp': cp_size, 'tp': tp_size}
        rank_to_coord1 = {}
        current_rank = 0
        for pp in range(pp_size):
            for dp in range(dp_size):
                for cp in range(cp_size):
                    for tp in range(tp_size):
                        rank_to_coord1[current_rank] = (pp, dp, cp, tp)
                        current_rank += 1
        assert current_rank == total_gpus

        group_type_map1 = {
            'tp': GroupType.TP,
            'cp': GroupType.CP,
            'dp': GroupType.DP,
            'pp': GroupType.PP,
        }
        groups1 = collect_groups(rank_to_coord1, dim_sizes1, dim_names1, group_type_map1)

        # Megatron's distributed optimizer shards dense parameters over the
        # data-parallel-with-context-parallel group.  Keep the ordinary DP
        # groups for callers that specifically need them, and build the
        # combined group explicitly for parameter reduce-scatter/all-gather.
        dp_cp_groups = []
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = get_ranks_by_coord(
                    rank_to_coord1,
                    lambda coord, pp=pp, tp=tp: coord[0] == pp and coord[3] == tp,
                )
                if ranks:
                    dp_cp_groups.append(ranks)
        groups1[GroupType.DP_CP] = dp_cp_groups

        # ========== 2. 构建 EP/ETP/EDP/PP 坐标系统 ==========
        # 坐标顺序: (pp, edp, etp, ep)  注意 PP 顺序一致，EDP 对应原来的 DP 角色
        dim_names2 = ['pp', 'edp', 'ep', 'etp']
        dim_sizes2 = {'pp': pp_size, 'edp': edp_size, 'ep': ep_size, 'etp': etp_size}
        rank_to_coord2 = {}
        current_rank = 0
        for pp in range(pp_size):
            for edp in range(edp_size):
                for ep in range(ep_size):
                    for etp in range(etp_size):
                        rank_to_coord2[current_rank] = (pp, edp, ep, etp)
                        current_rank += 1
        assert current_rank == total_gpus

        group_type_map2 = {
            'etp': GroupType.ETP,
            'ep': GroupType.EP,
            'edp': GroupType.EDP,  # 需要确保 GroupType 中有 EDP
            'pp': GroupType.PP,
        }
        groups2 = collect_groups(rank_to_coord2, dim_sizes2, dim_names2, group_type_map2)

        # ========== 3. 合并并去重（特别是 PP 组可能重复） ==========
        all_groups = {}
        for gtype, ranks_list in groups1.items():
            all_groups.setdefault(gtype, []).extend(ranks_list)
        for gtype, ranks_list in groups2.items():
            all_groups.setdefault(gtype, []).extend(ranks_list)

        # 去重
        for gtype in all_groups:
            unique = []
            seen = set()
            for ranks in all_groups[gtype]:
                key = tuple(ranks)
                if key not in seen:
                    seen.add(key)
                    unique.append(ranks)
            all_groups[gtype] = unique

        # ========== 4. 按顺序添加组 ==========
        # 顺序：TP -> CP -> DP -> ETP -> EP -> EDP -> PP
        order = [GroupType.TP, GroupType.CP, GroupType.DP, GroupType.DP_CP,
                 GroupType.ETP, GroupType.EP, GroupType.EDP, GroupType.PP]

        for gtype in order:
            if gtype not in all_groups:
                continue
            for ranks in all_groups[gtype]:
                if not ranks:
                    continue
                n_nodes = len(set(r // gpus_per_node for r in ranks))
                nvswitch_ids = self._get_nvswitch_ids_for_ranks(ranks)
                self._add_group(GroupInfo(
                    group_id=-1, group_type=gtype,
                    n_nodes=n_nodes, n_ranks=len(ranks),
                    ranks=ranks, nvswitch_ids=nvswitch_ids,
                    n_supernodes=len({self.topology.get_supernode_id(rank) for rank in ranks}),
                ))

    def get_group_info(self, rank: int, group_type: GroupType) -> Optional[GroupInfo]:
        key = (rank, group_type)
        if key not in self.rank_to_group:
            return None
        group_id = self.rank_to_group[key]
        return self.groups.get(group_id)

    def get_group_ranks(self, rank: int, group_type: GroupType) -> List[int]:
        info = self.get_group_info(rank, group_type)
        return info.ranks if info else []
