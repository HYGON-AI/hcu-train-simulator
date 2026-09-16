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

try:
    # python version >= 3.11
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib.abc import Traversable


logger = get_logger(__name__)


class BandwidthLookup:
    """带宽查找表管理器 - 支持多操作、多GPU数量的表格式"""

    MAX_MEASURED_NODES = 2

    def __init__(self, lookup_dir: str):
        self.lookup_dir = lookup_dir
        # 缓存结构: {(gpu_model, num_nodes, com_type, nranks): [(size, bandwidth), ...]}
        self.cache: Dict[Tuple[str, int, str, int], List[Tuple[int, float]]] = {}
        self._warned_fallbacks = set()

    def _warn_once(self, key, message, *args):
        if key in self._warned_fallbacks:
            return
        self._warned_fallbacks.add(key)
        logger.warning(message, *args)

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
        filename = self._get_table_filename(gpu_model, num_nodes)
        # filepath = os.path.join(self.lookup_dir, filename)
        filepath = self.lookup_dir.joinpath(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bandwidth table not found: {filepath}")
        return self._parse_table_file(filepath)

    def get_bandwidth(self, gpu_model: str, num_nodes: int, com_type: CommType,
                      size: int, nranks: int, group_info=None) -> float:
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

        actual_nodes = num_nodes
        table_nodes = min(num_nodes, self.MAX_MEASURED_NODES)
        lookup_nranks = nranks
        table = self._load_table(gpu_model, table_nodes)
        available_nranks = sorted(k[1] for k in table.keys() if k[0] == op_str)
        if not available_nranks:
            raise RuntimeError(f"Unsupported communication operation: '{op_str}'.")

        if lookup_nranks in available_nranks:
            selected_nranks = lookup_nranks
        else:
            selected_nranks = min(
                available_nranks,
                key=lambda candidate: (abs(candidate - lookup_nranks), candidate),
            )
            self._warn_once(
                ("rank-fallback", gpu_model, table_nodes, op_str, lookup_nranks, selected_nranks),
                "Bandwidth table rank fallback: operation=%s requested_concurrency_ranks=%s "
                "available_columns=%s; using %s-rank column from %s_%s.txt. This is an "
                "extrapolation, not an equivalent measurement.",
                op_str,
                lookup_nranks,
                available_nranks,
                selected_nranks,
                self._normalize_gpu_model(gpu_model),
                table_nodes,
            )

        if actual_nodes > self.MAX_MEASURED_NODES:
            self._warn_once(
                ("node-fallback", gpu_model, op_str, actual_nodes, table_nodes),
                "Bandwidth table node fallback: operation=%s actual_group_nodes=%s exceeds "
                "measured maximum=%s; using dual-node table %s_%s.txt. The dual-node result "
                "is treated as the best measured estimate, but fabric contention and scaling "
                "beyond two nodes are not represented.",
                op_str,
                actual_nodes,
                self.MAX_MEASURED_NODES,
                self._normalize_gpu_model(gpu_model),
                table_nodes,
            )

        if com_type == CommType.P2P:
            self._warn_once(
                ("p2p-concurrency", gpu_model, actual_nodes, nranks, table_nodes, selected_nranks),
                "P2P bandwidth-table convention: sendrecv uses configured "
                "concurrency_ranks=%s (an isolated sender/receiver pair is 2), independent of "
                "pp_size=%s. "
                "Actual PP group topology is %s ranks across %s nodes (%.2f ranks/node); selected "
                "measurement is %s_%s.txt, sendrecv %s-rank column (%.2f ranks/node if uniform). "
                "This models one transfer and does not model simultaneous contention among all "
                "PP stage boundaries.",
                lookup_nranks,
                nranks,
                group_info.n_ranks if group_info is not None else nranks,
                actual_nodes,
                (group_info.n_ranks if group_info is not None else nranks) / actual_nodes,
                self._normalize_gpu_model(gpu_model),
                table_nodes,
                selected_nranks,
                selected_nranks / table_nodes,
            )
        elif group_info is not None:
            actual_ranks_per_node = group_info.n_ranks / group_info.n_nodes
            measured_ranks_per_node = selected_nranks / table_nodes
            if (
                group_info.n_nodes != table_nodes
                or abs(actual_ranks_per_node - measured_ranks_per_node) > 1e-9
            ):
                self._warn_once(
                    (
                        "topology-mismatch",
                        gpu_model,
                        op_str,
                        group_info.n_nodes,
                        group_info.n_ranks,
                        table_nodes,
                        selected_nranks,
                    ),
                    "Bandwidth table topology mismatch: operation=%s actual_group=%s ranks "
                    "across %s nodes (%.2f ranks/node), but selected measurement=%s_%s.txt "
                    "%s-rank column (%.2f ranks/node if uniform). Equal total rank count does "
                    "not make these layouts equivalent; the lookup cannot model this topology "
                    "exactly and uses the selected measured bandwidth as an approximation.",
                    op_str,
                    group_info.n_ranks,
                    group_info.n_nodes,
                    actual_ranks_per_node,
                    self._normalize_gpu_model(gpu_model),
                    table_nodes,
                    selected_nranks,
                    measured_ranks_per_node,
                )

        key = (op_str, selected_nranks)

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
                 gpu_model: str = None,
                 p2p_intra_efficiency: float = 1.0,
                 collective_intra_efficiency: float = 1.0,
                 collective_inter_efficiency: float = 1.0):
        self.topology = topology
        self.group_manager = GroupManager(topology)
        self.algorithms: Dict[CommType, object] = {}
        self._configured = False
        self.intra_node_bandwidth_gbps = intra_node_bandwidth_gbps
        self.inter_node_bandwidth_gbps = inter_node_bandwidth_gbps
        self.p2p_intra_efficiency = p2p_intra_efficiency
        self.collective_intra_efficiency = collective_intra_efficiency
        self.collective_inter_efficiency = collective_inter_efficiency
        self.bandwidth_lookup = None
        if bandwidth_table_dir and gpu_model:
            self.bandwidth_lookup = BandwidthLookup(bandwidth_table_dir)
            self.gpu_model = gpu_model
            self.num_nodes = topology.num_nodes
        elif intra_node_bandwidth_gbps <= 0 or inter_node_bandwidth_gbps <= 0:
            raise ValueError(
                "intra_node_bandwidth_gbps and inter_node_bandwidth_gbps must "
                "be positive when no bandwidth table is configured"
            )

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

    def get_effective_bandwidth(self, flow: SingleFlow, com_type: CommType) -> float:
        """Return effective one-direction bandwidth for a flow in GB/s."""
        if flow.tag in ["PXN", "PXN_INIT", "NVLS"]:
            return self.inter_node_bandwidth_gbps * self.collective_inter_efficiency
        if com_type == CommType.P2P:
            return self.intra_node_bandwidth_gbps * self.p2p_intra_efficiency
        return self.intra_node_bandwidth_gbps * self.collective_intra_efficiency

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
        if nranks is None and flows:
            ranks = set()
            for f in flows:
                ranks.add(f.src)
                ranks.add(f.dest)
            nranks = len(ranks)

        if self.bandwidth_lookup and com_type is not None:
            # Table entries are measured end-to-end operation bandwidths and
            # already include real-world efficiency losses.
            if group_info is not None:
                num_nodes_for_group = group_info.n_nodes
            else:
                gpus_per_node = self.topology.num_gpus // self.topology.num_nodes
                nodes = {
                    flow_rank // gpus_per_node
                    for flow in flows
                    for flow_rank in (flow.src, flow.dest)
                }
                num_nodes_for_group = len(nodes)
            bandwidth = self.bandwidth_lookup.get_bandwidth(
                self.gpu_model,
                num_nodes_for_group,
                com_type,
                data_size,
                nranks,
                group_info=group_info,
            )
            if bandwidth <= 0:
                raise ValueError("bandwidth table returned a non-positive bandwidth")
            return data_size / 1_000_000_000 / bandwidth

        for flow in flows:
            bandwidth = self.get_effective_bandwidth(flow, com_type)
            if bandwidth <= 0:
                raise ValueError("effective communication bandwidth must be positive")
            data_size_gb = flow.size / 1_000_000_000
            transfer_time = data_size_gb / bandwidth
            total_time += transfer_time
        return total_time

    def get_flow_details(self, flows: List[SingleFlow], com_type: CommType = None) -> Dict:
        total_data = sum(flow.size for flow in flows)
        total_data_gb = total_data / 1_000_000_000
        tag_stats = {}
        for flow in flows:
            tag = flow.tag
            if tag not in tag_stats:
                tag_stats[tag] = {'count': 0, 'total_size_bytes': 0, 'total_size_gb': 0}
            tag_stats[tag]['count'] += 1
            tag_stats[tag]['total_size_bytes'] += flow.size
            tag_stats[tag]['total_size_gb'] = tag_stats[tag]['total_size_bytes'] / 1_000_000_000
        pair_stats = {}
        for flow in flows:
            pair = (flow.src, flow.dest)
            if pair not in pair_stats:
                pair_stats[pair] = {'count': 0, 'total_size_bytes': 0, 'total_size_gb': 0}
            pair_stats[pair]['count'] += 1
            pair_stats[pair]['total_size_bytes'] += flow.size
            pair_stats[pair]['total_size_gb'] = pair_stats[pair]['total_size_bytes'] / 1_000_000_000
        tag_time_stats = {}
        for tag, stats in tag_stats.items():
            representative = next(flow for flow in flows if flow.tag == tag)
            bandwidth = (
                None
                if self.bandwidth_lookup
                else self.get_effective_bandwidth(
                    representative, com_type or CommType.ALL_REDUCE
                )
            )
            transfer_time = (
                stats['total_size_gb'] / bandwidth
                if bandwidth is not None and bandwidth > 0
                else None
            )
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
                         p2p_intra_efficiency: float = 1.0,
                         collective_intra_efficiency: float = 1.0,
                         collective_inter_efficiency: float = 1.0,
                         bandwidth_table_dir: Traversable = None):
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
            gpu_model=gpu_model,
            p2p_intra_efficiency=p2p_intra_efficiency,
            collective_intra_efficiency=collective_intra_efficiency,
            collective_inter_efficiency=collective_inter_efficiency,
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
        group_info = self.engine.get_group_info(rank, group_type)
        if group_info is None:
            logger.warning("no flows generated for %s on %s", com_type.value, group_type.value)
            return 0.0
        flow_graph = self.engine.algorithms[com_type].decompose(
            rank, group_info, com_type, data_size
        )
        if not flow_graph.flows:
            logger.warning("no flows generated for %s on %s", com_type.value, group_type.value)
            return 0.0

        if self.engine.bandwidth_lookup:
            # The table represents the measured end-to-end collective, so one
            # lookup is sufficient and per-rank flow critical paths do not apply.
            flows = (
                flow_graph.get_by_rank_p2p(rank)
                if group_type == GroupType.PP
                else flow_graph.get_by_rank(rank)
            )
            return self.engine.simulate_communication_time(
                flows, com_type, group_info.n_ranks, data_size, group_info
            )

        if group_type == GroupType.PP:
            # Preserve the legacy PP accounting requested by the estimator:
            # sum every generated forward/backward stage-boundary flow.
            flow_times = [
                self.engine.simulate_communication_time(
                    [flow], com_type, group_info.n_ranks, data_size, group_info
                )
                for flow in flow_graph.get_by_rank_p2p(rank)
            ]
            # return sum(flow_times)
            return sum(flow_times) / len(flow_times)

        # Collective ranks and links operate concurrently. Completion time is
        # determined by the slowest rank path, not by an arbitrary rank.
        rank_times = [
            self.engine.simulate_communication_time(
                flow_graph.get_by_rank(group_rank),
                com_type,
                group_info.n_ranks,
                data_size,
                group_info,
            )
            for group_rank in group_info.ranks
        ]
        return max(rank_times, default=0.0)

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
        total_time = self.get_communication_time(
            com_type=com_type,
            algorithm=algorithm,
            group_type=group_type,
            data_size=data_size,
            rank=rank,
        )
        details = self.engine.get_flow_details(flows, com_type)
        return {
            'total_time': total_time,
            'total_flows': details['total_flows'],
            'total_data_gb': details['total_data_gb'],
            'tag_stats': details['tag_stats'],
            'tag_time_stats': details['tag_time_stats']
        }
