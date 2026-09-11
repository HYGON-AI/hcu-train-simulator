# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.communication.algorithms.ring import RingAlgorithm
from hcu_train_simulator.communication.groups import GroupManager
from hcu_train_simulator.communication.topology import Topology
from hcu_train_simulator.communication.types import Algorithm, CommType, GPUType, GroupType, SingleFlow
from hcu_train_simulator.logging import get_logger

import os
import re
from typing import List, Dict, Tuple


logger = get_logger(__name__)


class BandwidthLookup:
    """带宽查找表管理器 - 支持多操作、多GPU数量的表格式"""

    def __init__(self, lookup_dir: str):
        self.lookup_dir = lookup_dir
        # 缓存结构: {(gpu_model, num_nodes, com_type, nranks): [(size, bandwidth), ...]}
        self.cache: Dict[Tuple[str, int, str, int], List[Tuple[int, float]]] = {}

    @staticmethod
    def _normalize_gpu_model(gpu_model) -> str:
        return gpu_model.value if hasattr(gpu_model, "value") else str(gpu_model)

    def _get_table_filename(self, gpu_model: str, num_nodes: int) -> str:
        gpu_model = self._normalize_gpu_model(gpu_model)
        return f"{gpu_model}_{num_nodes}.txt"

    def _parse_table_file(self, filepath: str) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
        """
        解析表文件，返回结构:
        {
            (com_type, nranks): [(size, bandwidth), ...]
        }
        com_type: 字符串如 "all_reduce", "all_gather"
        nranks: GPU数量 (2,4,8,16)
        """
        result = {}
        # utf-8-sig 去除txt文件bom头信息
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        i = 0
        while i < len(lines):
            # 操作名行，例如 "all_reduce"
            op_name = lines[i].lower()
            i += 1
            # 接下来的数据行直到遇到下一个操作名或文件结束
            data_lines = []
            while i < len(lines) and not re.match(r'^[a-z_]+$', lines[i]):
                data_lines.append(lines[i])
                i += 1

            # 解析数据行
            # 第一行数据行应该包含列头？但根据用户示例，数据行直接是数值，没有列头。
            # 第一行数据行格式: "4	0	0	0	0"  => size=4, 带宽对应2,4,8,16 GPUs
            # 我们需要知道每列对应的GPU数量。约定列顺序: 第2列 -> 2 GPUs, 第3列 -> 4 GPUs, 第4列 -> 8 GPUs, 第5列 -> 16 GPUs
            # 因此对于每个size，我们为每个nranks建立条目。
            nranks_list = [2, 4, 8, 16]  # 固定顺序
            for line in data_lines:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    size = int(parts[0])
                    # 解析带宽，跳过可能为0的值（但0也是有效值，保留）
                    for idx, nranks in enumerate(nranks_list):
                        if idx + 1 < len(parts):
                            bw = float(parts[idx + 1])
                            key = (op_name, nranks)
                            if key not in result:
                                result[key] = []
                            result[key].append((size, bw))
                except ValueError:
                    continue

        # 对每个key的列表按size排序
        for key in result:
            result[key].sort(key=lambda x: x[0])
        return result

    def _load_table(self, gpu_model: str, num_nodes: int) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
        """加载指定GPU和节点数的完整表"""
        if num_nodes > 2: num_nodes = 2
        filename = self._get_table_filename(gpu_model, num_nodes)
        # filepath = os.path.join(self.lookup_dir, filename)
        filepath = self.lookup_dir.joinpath(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bandwidth table not found: {filepath}")
        return self._parse_table_file(filepath)

    def get_bandwidth(self, gpu_model: str, num_nodes: int, com_type: CommType,
                      size: int, nranks: int) -> float:
        """
        获取带宽
        :param gpu_model: GPU型号字符串，例如 "bw500SM"
        :param num_nodes: 节点数
        :param com_type: 通信类型枚举
        :param size: 消息大小（字节）
        :param nranks: 通信组内的GPU数量
        :return: 带宽 (GB/s)
        """
        # Map communication types to operation names in the measured tables.
        op_name_map = {
            CommType.ALL_REDUCE: "all_reduce",
            CommType.ALL_GATHER: "all_gather",
            CommType.REDUCE_SCATTER: "reduce_scatter",
            CommType.ALL_TO_ALL: "alltoall",
            CommType.P2P: "sendrecv"
            # 其他操作可根据需要扩展
        }
        op_str = op_name_map.get(com_type)
        if op_str is None:
            # 如果不支持的操作，直接报错
            raise RuntimeError(f"Unsupported communication operation: '{op_str}'.")

        key = (op_str, nranks)
        table = self._load_table(gpu_model, num_nodes)
        if key not in table:
            # 如果没有对应操作或GPU数量的数据，尝试使用最接近的nranks
            # 例如找小于当前nranks的最大值
            available_nranks = [k[1] for k in table.keys() if k[0] == op_str]
            if available_nranks:
                available_nranks.sort()
                # 选择最接近且不大于当前nranks的值，如果没有则选最小
                closest = min(available_nranks, key=lambda x: abs(x - nranks))
                key = (op_str, closest)
            else:
                raise RuntimeError(f"Unsupported communication operation: '{op_str}'.")

        data = table[key]
        return self._lookup_bandwidth(size, data)

    @staticmethod
    def _lookup_bandwidth(size: int, table: List[Tuple[int, float]]) -> float:
        if not table:
            return 0.0
        if size <= table[0][0]:
            return table[0][1]
        if size >= table[-1][0]:
            return table[-1][1]
        for i in range(len(table) - 1):
            s1, b1 = table[i]
            s2, b2 = table[i + 1]
            if size == s1:
                return b1
            if size == s2:
                return b2
            if s1 < size < s2:
                # 线性插值
                ratio = (size - s1) / (s2 - s1)
                return b1 + ratio * (b2 - b1)
        return 0.0


class CommunicationEngine:
    """通信模拟器主类"""

    def __init__(self, topology: Topology,
                 intra_node_bandwidth_gbps: float,
                 inter_node_bandwidth_gbps: float,
                 bandwidth_table_dir: str = None,
                 gpu_model: str = None):
        self.topology = topology
        self.group_manager = GroupManager(topology)
        self.algorithms: Dict[CommType, object] = {}
        self._configured = False
        self.intra_node_bandwidth_gbps = intra_node_bandwidth_gbps
        self.inter_node_bandwidth_gbps = inter_node_bandwidth_gbps
        self.bandwidth_lookup = None
        if bandwidth_table_dir and gpu_model:
            self.bandwidth_lookup = BandwidthLookup(bandwidth_table_dir)
            self.gpu_model = gpu_model
            self.num_nodes = topology.num_nodes

    def configure_parallelism(self, tp_size: int = 1, cp_size: int = 1, dp_size: int = 1,
                              pp_size: int = 1, ep_size: int = 1, etp_size: int = 1, edp_size: int = 1):
        """配置并行策略"""
        dp_size = len(self.topology.gpus) // tp_size // cp_size // pp_size
        edp_size = len(self.topology.gpus) // ep_size // etp_size // pp_size
        logger.debug(
            "parallel groups: tp=%s cp=%s dp=%s pp=%s ep=%s etp=%s edp=%s",
            tp_size,
            cp_size,
            dp_size,
            pp_size,
            ep_size,
            etp_size,
            edp_size,
        )
        self.group_manager.build_groups(tp_size, cp_size, dp_size, ep_size, etp_size, edp_size, pp_size)
        self._configured = True

    def set_algorithm(self, com_type: CommType, algo: Algorithm):
        """设置通信算法"""
        if algo == Algorithm.RING:
            gpus_per_node = self.topology.num_gpus // self.topology.num_nodes
            self.algorithms[com_type] = RingAlgorithm(
                gpus_per_node=gpus_per_node,
            )
        else:
            raise ValueError(f"Algorithm {algo} not yet implemented")

    def get_communication_flows(self, rank: int, com_type: CommType,
                                data_size: int, group_type: GroupType) -> List[SingleFlow]:
        """获取通信原语列表"""
        if not self._configured:
            raise RuntimeError("Must call configure_parallelism() first")
        group_info = self.group_manager.get_group_info(rank, group_type)
        if group_info is None:
            return []
        algo = self.algorithms.get(com_type)
        if algo is None:
            raise ValueError(f"No algorithm set for {com_type}")
        flow_models = algo.decompose(rank, group_info, com_type, data_size)
        if group_type == GroupType.PP:
            return flow_models.get_by_rank_p2p(rank)
        else:
            return flow_models.get_by_rank(rank)

    def get_group_info(self, rank: int, group_type: GroupType):
        return self.group_manager.get_group_info(rank, group_type)

    def simulate_communication_time(self, flows: List[SingleFlow], com_type: CommType = None, nranks: int = None,
                                    data_size: int = None, group_info=None) -> float:
        """
        根据流依赖关系和网络带宽模拟集合通信的总时间
        如果提供了com_type和nranks，且启用了带宽表，则使用动态带宽查询

        Args:
            flows: 通信流列表
            com_type: 通信类型
            nranks: 通信组内的GPU数量
            data_size: 数据大小
            group_info: 通信组信息（用于获取该组实际跨越的节点数）
        """
        if not flows:
            return 0.0

        total_time = 0.0
        # 尝试推断通信组大小（从流中获取src/dest的最大rank范围？简单方法：从第一个流中获取通信组大小？）
        # 更可靠的方式：由调用者传入nranks，或者在生成流时记录在flow_models中。
        # 这里简单起见，如果未提供nranks，则从flows中统计唯一rank数量作为近似
        if nranks is None and flows:
            ranks = set()
            for f in flows:
                ranks.add(f.src)
                ranks.add(f.dest)
            nranks = len(ranks)
        for flow in flows:
            if self.bandwidth_lookup and com_type is not None:
                # 获取当前通信组实际跨越的节点数
                if group_info is not None:
                    num_nodes_for_group = group_info.n_nodes
                else:
                    # 兼容旧调用：从flow中推断节点数（通过rank和每节点GPU数）
                    gpus_per_node = self.topology.num_gpus // self.topology.num_nodes
                    src_node = flow.src // gpus_per_node
                    dst_node = flow.dest // gpus_per_node
                    num_nodes_for_group = len(set([src_node, dst_node]))
                # 使用带宽表
                bandwidth = self.bandwidth_lookup.get_bandwidth(
                    self.gpu_model, num_nodes_for_group, com_type, data_size, nranks
                )
                total_time = data_size / 1024 / 1024 / 1024 / bandwidth
                return total_time
            else:
                # 使用固定带宽
                if flow.tag in ["PXN", "PXN_INIT", "NVLS"]:
                    bandwidth = self.inter_node_bandwidth_gbps
                else:
                    bandwidth = self.intra_node_bandwidth_gbps
            data_size_gb = flow.size / (1024 * 1024 * 1024)
            transfer_time = data_size_gb / bandwidth if bandwidth > 0 else 0
            total_time += transfer_time
        return total_time

    def get_flow_details(self, flows: List[SingleFlow]) -> Dict:
        total_data = sum(flow.size for flow in flows)
        total_data_gb = total_data / (1024 * 1024 * 1024)
        tag_stats = {}
        for flow in flows:
            tag = flow.tag
            if tag not in tag_stats:
                tag_stats[tag] = {'count': 0, 'total_size_bytes': 0, 'total_size_gb': 0}
            tag_stats[tag]['count'] += 1
            tag_stats[tag]['total_size_bytes'] += flow.size
            tag_stats[tag]['total_size_gb'] = tag_stats[tag]['total_size_bytes'] / (1024 * 1024 * 1024)
        pair_stats = {}
        for flow in flows:
            pair = (flow.src, flow.dest)
            if pair not in pair_stats:
                pair_stats[pair] = {'count': 0, 'total_size_bytes': 0, 'total_size_gb': 0}
            pair_stats[pair]['count'] += 1
            pair_stats[pair]['total_size_bytes'] += flow.size
            pair_stats[pair]['total_size_gb'] = pair_stats[pair]['total_size_bytes'] / (1024 * 1024 * 1024)
        tag_time_stats = {}
        for tag, stats in tag_stats.items():
            if tag in ["PXN", "PXN_INIT", "NVLS"]:
                bandwidth = 800.0
            else:
                bandwidth = 400.0
            transfer_time = stats['total_size_gb'] / bandwidth if bandwidth > 0 else 0
            tag_time_stats[tag] = {
                'bandwidth_gbps': bandwidth,
                'theoretical_time_sec': transfer_time
            }
        return {
            'total_flows': len(flows),
            'total_data_bytes': total_data,
            'total_data_gb': total_data_gb,
            'tag_stats': tag_stats,
            'tag_time_stats': tag_time_stats,
            'pair_stats': pair_stats,
            'flows': flows
        }


class CommunicationSimulator:
    """通信模拟器封装类 - 提供简化的接口"""

    def __init__(self):
        self.engine = None
        self.topology = None
        self._initialized = False

    def initialize_parallelism(self,
                         num_gpus: int,
                         tp_size: int,
                         cp_size: int,
                         pp_size: int,
                         ep_size: int = 1,
                         etp_size: int = 1,
                         edp_size: int = 1,
                         gpus_per_node: int = 8,
                         gpu_model: str = "bw500SM",  # 改为字符串，用于匹配文件名
                         intra_node_bandwidth_gbps: float = 800.0,
                         inter_node_bandwidth_gbps: float = 400.0,
                         bandwidth_table_dir: str = None):
        """
        初始化并行配置
        :param gpu_model: GPU型号字符串，如 "bw500SM"，将用于查找表文件名
        """
        num_nodes = num_gpus // gpus_per_node
        if num_gpus % gpus_per_node != 0:
            raise ValueError(f"num_gpus ({num_gpus}) must be divisible by gpus_per_node ({gpus_per_node})")
        dp_size = num_gpus // (tp_size * cp_size * pp_size)
        total_parallel = tp_size * cp_size * pp_size * dp_size
        if total_parallel != num_gpus:
            raise ValueError(f"Parallelism product (tp*cp*pp*dp = {total_parallel}) must equal num_gpus ({num_gpus})")
        self.topology = Topology.create_uniform(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            gpu_type=GPUType.BW1100
        )
        logger.debug("topology: nodes=%s gpus=%s", self.topology.num_nodes, self.topology.num_gpus)
        self.engine = CommunicationEngine(
            self.topology,
            intra_node_bandwidth_gbps,
            inter_node_bandwidth_gbps,
            bandwidth_table_dir=bandwidth_table_dir,
            gpu_model=gpu_model
        )
        dp_size = num_gpus // (tp_size * cp_size * pp_size)
        if dp_size == 0:
            dp_size = 1
        self.engine.configure_parallelism(
            tp_size=tp_size,
            cp_size=cp_size,
            dp_size=dp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            etp_size=etp_size,
            edp_size=edp_size
        )
        # 打印组信息（略）
        for group_type in [GroupType.TP, GroupType.CP, GroupType.DP,
                           GroupType.EP, GroupType.ETP, GroupType.EDP, GroupType.PP]:
            for rank in range(min(8, num_gpus)):
                info = self.engine.get_group_info(rank, group_type)
                if info:
                    logger.debug(
                        "rank %s %s: group_id=%s n_ranks=%s ranks=%s",
                        rank,
                        group_type.value,
                        info.group_id,
                        info.n_ranks,
                        info.ranks[:],
                    )
                    break
                else:
                    if rank == 0:
                        logger.debug("%s: None", group_type.value)
                    break
        self._initialized = True
        logger.debug("communication initialization complete")

    def get_communication_time(self,
                               com_type: CommType,
                               algorithm: Algorithm,
                               group_type: GroupType,
                               data_size: int,
                               rank: int = 0) -> float:
        if not self._initialized:
            raise RuntimeError("Must call initialize_parallelism() first")
        self.engine.set_algorithm(com_type, algorithm)
        flows = self.engine.get_communication_flows(rank, com_type, data_size, group_type)
        if not flows:
            logger.warning("no flows generated for %s on %s", com_type.value, group_type.value)
            return 0.0
        # 获取通信组大小
        group_info = self.engine.get_group_info(rank, group_type)
        nranks = group_info.n_ranks if group_info else 0
        total_time = self.engine.simulate_communication_time(
            flows,
            com_type=com_type,
            nranks=nranks,
            data_size=data_size,
            group_info=group_info,
        )
        return total_time

    def get_communication_details(self,
                                  com_type: CommType,
                                  algorithm: Algorithm,
                                  group_type: GroupType,
                                  data_size: int,
                                  rank: int = 0) -> dict:
        if not self._initialized:
            raise RuntimeError("Must call initialize_parallelism() first")
        self.engine.set_algorithm(com_type, algorithm)
        flows = self.engine.get_communication_flows(rank, com_type, data_size, group_type)
        if not flows:
            return {'total_time': 0.0, 'total_flows': 0, 'total_data_gb': 0.0,
                    'tag_stats': {}, 'tag_time_stats': {}}
        group_info = self.engine.get_group_info(rank, group_type)
        nranks = group_info.n_ranks if group_info else 0
        total_time = self.engine.simulate_communication_time(
            flows,
            com_type=com_type,
            nranks=nranks,
            data_size=data_size,
            group_info=group_info,
        )
        details = self.engine.get_flow_details(flows)
        return {
            'total_time': total_time,
            'total_flows': details['total_flows'],
            'total_data_gb': details['total_data_gb'],
            'tag_stats': details['tag_stats'],
            'tag_time_stats': details['tag_time_stats']
        }
