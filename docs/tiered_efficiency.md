# 理论估算的分层效率

在 `hardware_config` 中，`use_tiered_gemm_efficiency` 和
`use_tiered_communication_efficiency` 分别控制 GEMM 与通信的分层修正，默认均为
`true`。设为 `false` 时沿用固定效率。这两个开关不接受自定义分层系数。

## GEMM

只对理论计算耗时生效。有效效率为 `gemm_efficiency × shape 系数`；FP16/FP8
分别使用其配置峰值算力。根据 GEMM 的 `m, n, k` 中最小维度分层：

| 最小维度 | shape 系数 |
| ---: | ---: |
| ≤ 512 | 0.15 |
| 513–1024 | 0.40 |
| 1025–2048 | 0.75 |
| > 2048 | 1.00 |

内置 BW1000、BW1100 的 BF16 GEMM profile 中，小维度算子的实测 TFLOPS
明显低于大维度算子；以上系数是按两个 profile 的分档中位性能相对大维度平台期
近似取整。注意力矩阵乘的理论估算也按其 token、序列、head 维度修正。
如果一个算子有有效的实测 profile 或 benchmark 耗时，报告直接采用实测耗时，
不再叠加 shape 系数。

## 通信

`use_bandwidth_table=false` 时，有效单向带宽为对应拓扑层级的配置带宽 ×
既有通信效率 × 消息大小系数。按每条通信 flow 的传输字节数分层：

| 每条 flow 的大小 | 消息大小系数 |
| ---: | ---: |
| ≤ 64 KiB | 0.02 |
| 64 KiB–1 MiB | 0.15 |
| 1–16 MiB | 0.55 |
| 16–64 MiB | 0.85 |
| > 64 MiB | 1.00 |

系数参考 `communication/bandwidth/bw1000_1.txt` 和 `bw1000_2.txt`
中 all-gather、all-reduce 等操作从小消息到大消息的有效带宽爬升，按量级
近似取整。不同集合通信有不同的曲线，因此这里是通用近似，不代替实测表。
`use_bandwidth_table=true` 时直接使用表中的实测有效带宽，不叠加配置效率或
消息大小系数。
