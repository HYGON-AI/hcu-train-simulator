#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# ============================================
# 加载模块和环境配置（来自 run.txt）
# ============================================

# 加载必要的模块
module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1 
module load app/rccl/topos/default 
module load app/rccl/shca_rdma_plugins/v8

# 设置 MPI 和 NCCL/RCCL 环境变量
export OMPI_MCA_btl_tcp_if_include=eno2
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=WARN
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"  # 使用默认值如果未设置
export NCCL_PXN_DISABLE=0
export RCCL_PXN_GPU_BALANCE=1
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export UCX_IB_NUM_PATHS=1
export NCCL_TOPO_FILE="/opt/hpc/software/app/rccl/topos/508/built-in-508-topo-input-tj-default.xml"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ROCM_PATH}/lib"

# ============================================
# 测试配置
# ============================================

# 定义所有集合操作
collective_ops=(
    "all_reduce_perf"
    "all_gather_perf"
    "reduce_scatter_perf"
    "alltoall_perf"
    "alltoallv_perf"
    "broadcast_perf"
    "gather_perf"
    "reduce_perf"
    "scatter_perf"
    "sendrecv_perf"
)

# 定义进程数列表（内层循环）
np_list=(8 4 2)

# 节点配置（可根据需要修改）
# 单机模式: 
# 双机模式（2节点各8卡）
HOSTS="f09r2n06:8"  # 默认dan机各8卡，可根据需要修改

# 日志目录
log_dir="./logs/bw1000_1"
mkdir -p "$log_dir"

# MPI 基础参数（来自 run.txt 的风格）
MPI_BASE_ARGS="--allow-run-as-root -bind-to none"

# ============================================
# 开始测试
# ============================================

echo "=========================================="
echo "Environment loaded:"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  NCCL_DEBUG: $NCCL_DEBUG"
echo "  Hosts: $HOSTS"
echo "=========================================="

# 外层循环：遍历每个集合操作
for op in "${collective_ops[@]}"; do
    echo "=========================================="
    echo "Starting test: $op"
    echo "=========================================="
    
    # 内层循环：遍历不同的进程数
    for np in "${np_list[@]}"; do
        echo "----------------------------------------"
        echo "Running $op - np=$np (total processes: $np)"
        echo "----------------------------------------"
        
        # 构建主机列表（根据 np 动态调整每节点的进程数）
        # 这里假设两个节点，每个节点运行 np 个进程
        HOSTS_PER_NODE="f09r2n06:$np"
        
        # 构建完整的 mpirun 命令
        cmd="mpirun $MPI_BASE_ARGS \
            -H $HOSTS_PER_NODE \
            -np $np \
            -x OMPI_MCA_btl_tcp_if_include \
            -x NCCL_SOCKET_IFNAME \
            -x NCCL_DEBUG \
            -x ROCM_PATH \
            -x NCCL_PXN_DISABLE \
            -x RCCL_PXN_GPU_BALANCE \
            -x NCCL_NET_PLUGIN \
            -x NCCL_PLUGIN_P2P \
            -x UCX_IB_NUM_PATHS \
            -x NCCL_TOPO_FILE \
            -x LD_LIBRARY_PATH \
            ./build/$op -b 8 -e 1g -f 2 -g 1 -d int8"
        
        echo "Executing: $cmd"
        
        # 执行命令，并将输出保存到日志文件
        eval $cmd 2>&1 | tee "$log_dir/${op}_np${np}.log"
        
        # 检查命令是否成功
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ $op (np=$np, total=$((np * 2))) completed"
        else
            echo "✗ $op (np=$np, total=$((np * 2))) failed"
        fi
        
        echo ""
    done
    echo ""
done

echo "=========================================="
echo "All tests completed! Logs saved in: $log_dir"
echo "=========================================="
