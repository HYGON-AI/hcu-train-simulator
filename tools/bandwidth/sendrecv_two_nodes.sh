#!/usr/bin/env bash

module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1 
module load app/rccl/topos/default 
module load app/rccl/shca_rdma_plugins/v8


mpirun --allow-run-as-root -N 8 \
-H f09r2n06:8,f09r2n07:8 \
-x OMPI_MCA_btl_tcp_if_include=eno2 \
-x NCCL_SOCKET_IFNAME=eno2 \
-x NCCL_DEBUG=WARN \
-x ROCM_PATH \
-x NCCL_PXN_DISABLE=0 \
-x RCCL_PXN_GPU_BALANCE=1 \
-x NCCL_NET_PLUGIN=shca \
-x NCCL_PLUGIN_P2P=ib \
-x UCX_IB_NUM_PATHS=1 \
-x NCCL_TOPO_FILE="/opt/hpc/software/app/rccl/topos/508/built-in-508-topo-input-tj-default.xml" \
-x LD_LIBRARY_PATH \
--bind-to none \
/opt/hpc/software/app/rccl/tests/sendrecv_perf -g 1 -b 4 -e 1g -f 2 -I 1 -n 100 -w 10 -d double

