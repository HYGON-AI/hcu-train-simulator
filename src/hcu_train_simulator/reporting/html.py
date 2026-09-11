# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import html as html_lib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from hcu_train_simulator.logging import get_logger, log_step


logger = get_logger(__name__)


def generate_training_report(
        model_b: int,
        total_gpus: int,
        gpus_per_node: int,
        tgs: float,
        mfu: float,
        pp_stages: List[Dict[str, Any]],
        # each dict: {"pp_rank": int, "param_elems": int, "act_elems": int, "wgo_gib": float, "act_gib": float, "total_gib": float}
        compute_ms: float,
        comm_ms: float,
        compute_details: List[Dict[str, Any]],
        # each dict: {"model_part": str, "b": int, "m": int, "n": int, "k": int, "compute_count": int, "forward_ms": float, "backward_ms": float}
        comm_domains: List[Dict[str, Any]],
        # each dict: {"domain": str, "raw_time_ms": float,
        #             "overlap_percent": float, "time_ms": float, "percent": float}
        parallel_strategy: Dict,  # {"tp": int, "cp": int, "ep": int, "dp": int, "pp": int, "etp": int}
        benchmark_env_info: Optional[Dict[str, Any]] = None,
        benchmark_details: Optional[List[Dict[str, Any]]] = None,
        model_details: Optional[Dict[str, Any]] = None,
        profile_summary: Optional[Dict[str, Any]] = None,
        ep_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    生成大模型训练模拟报告HTML

    Args:
        model_b: 模型尺寸 (B)
        total_gpus: GPU总数
        gpus_per_node: 单节点 GPU 数量
        tgs: TGS (tokens/p/s)
        mfu: MFU百分比
        pp_stages: 各pipeline stage显存占用数据列表
        compute_ms: 计算时间 (毫秒)
        comm_ms: 通信时间 (毫秒)
        compute_details: 各模块计算时间详情列表
        comm_domains: 通信域时间分布列表
        parallel_strategy: 并行策略
    """

    def display_ms(value):
        if value == "/" or value is None:
            return "/"
        return f"{value:.2f}"

    def display_value(value, digits=2):
        if value == "/" or value is None:
            return "/"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return html_lib.escape(str(value))

    def display_int(value):
        if value == "/" or value is None:
            return "/"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return html_lib.escape(str(value))

    def escape_value(value):
        if value is None:
            return "/"
        return html_lib.escape(str(value))

    benchmark_details = benchmark_details or []
    ep_details = ep_details or {}
    has_benchmark_data = any(
        item.get("status") == "ok" and item.get("mean_ms") is not None
        for item in benchmark_details
    )

    total_ms = compute_ms + comm_ms
    compute_percent = (compute_ms / total_ms * 100) if total_ms > 0 else 0
    comm_percent = (comm_ms / total_ms * 100) if total_ms > 0 else 0

    # 并行策略排版
    ps_text = ""
    for k, v in parallel_strategy.items():
        ps_text += (k.upper() + str(v) + "-")
    ps_text = ps_text[:-1]

    # 处理通信域百分比（如果传入的percent为0或未正确计算，则基于time_ms重新计算）
    total_comm_ms = sum(d.get("time_ms", 0) for d in comm_domains)
    for domain in comm_domains:
        if domain.get("percent", 0) == 0 and total_comm_ms > 0:
            domain["percent"] = domain["time_ms"] / total_comm_ms * 100
        domain["percent_display"] = f"{domain['percent']:.2f}%"

    # 生成PP Stage表格行
    pp_rows_html = []
    for stage in pp_stages:
        pp_rows_html.append(f"""
        <tr>
            <td>{stage.get("pp_rank", 0)}</td>
            <td>{stage.get("activation_microbatches") if stage.get("activation_microbatches") is not None else "/"}</td>
            <td>{stage.get("param_elems", 0):,}</td>
            <td>{stage.get("act_elems", 0):,}</td>
            <td>{stage.get("wgo_gib", 0):.2f}</td>
            <td>{stage.get("act_gib", 0):.2f}</td>
            <td><b>{stage.get("total_gib", 0):.2f}</b></td>
        </tr>
        """)

    # 生成计算详情表格行
    gemm_compute_rows_html = []
    non_gemm_compute_rows_html = []
    for comp in compute_details:
        op_type = comp.get("op_type", "gemm")
        if op_type == "non_gemm":
            row_class = (
                ' class="non-gemm-extra"'
                if len(non_gemm_compute_rows_html) >= 5
                else ""
            )
            non_gemm_compute_rows_html.append(f"""
        <tr{row_class}>
            <td>{escape_value(comp.get("model_part", ""))}</td>
            <td>{escape_value(comp.get("shape", ""))}</td>
            <td>{display_int(comp.get("elements"))}</td>
            <td>{display_int(comp.get("total_elements"))}</td>
            <td>{display_value(comp.get("compute_count"), 4)}</td>
            <td>{display_value(comp.get("forward_ops_per_element"), 2)}</td>
            <td>{display_value(comp.get("backward_ops_per_element"), 2)}</td>
            <td>{comp.get("forward_ms", 0):.2f}</td>
            <td>{comp.get("backward_ms", 0):.2f}</td>
            <td>{comp.get("all_time_ms", 0):.2f}</td>
            <td>{display_ms(comp.get("benchmark_time_ms", "/"))}</td>
            <td>{escape_value(comp.get("benchmark_source", "/"))}</td>
        </tr>
            """)
            continue

        gemm_compute_rows_html.append(f"""
        <tr>
            <td>{escape_value(comp.get("model_part", ""))}</td>
            <td>{comp.get("b", 0)}</td>
            <td>{escape_value(comp.get("m", 0))}</td>
            <td>{escape_value(comp.get("n", 0))}</td>
            <td>{escape_value(comp.get("k", 0))}</td>
            <td>{comp.get("compute_count", 0)}</td>
            <td>{comp.get("forward_ms", 0):.2f}</td>
            <td>{comp.get("backward_ms", 0):.2f}</td>
            <td>{comp.get("all_time_ms", 0):.2f}</td>
            <td>{display_ms(comp.get("benchmark_time_ms", "/"))}</td>
            <td>{escape_value(comp.get("benchmark_source", "/"))}</td>
        </tr>
        """)

    # 生成通信域表格行
    non_gemm_toggle_html = ""
    if len(non_gemm_compute_rows_html) > 5:
        non_gemm_toggle_html = f"""
                <tr class="non-gemm-toggle-row">
                    <td colspan="12">
                        <label for="non-gemm-toggle" class="non-gemm-toggle-button" title="展开或收起全部 Non-GEMM 行">
                            <span class="non-gemm-toggle-arrow" aria-hidden="true">⌄</span>
                            <span class="non-gemm-show-label">展开全部 {len(non_gemm_compute_rows_html)} 行</span>
                            <span class="non-gemm-hide-label">收起至前 5 行</span>
                        </label>
                    </td>
                </tr>
        """

    comm_rows_html = []
    for comm in comm_domains:
        domain = comm.get("domain", "")
        raw_time_ms = comm.get("raw_time_ms", comm.get("time_ms", 0))
        time_ms = comm.get("time_ms", 0)
        percent = comm.get("percent", 0)
        has_data = raw_time_ms > 0
        hidden_time_ms = max(0.0, raw_time_ms - time_ms)
        exposed_width = (
            min(100.0, time_ms / raw_time_ms * 100)
            if raw_time_ms > 0 else 0.0
        )
        hidden_width = max(0.0, 100.0 - exposed_width)
        time_bar_label = (
            f"总计 {raw_time_ms:.2f} ms · 未掩盖 {time_ms:.2f} ms · 已掩盖 {hidden_time_ms:.2f} ms"
            if has_data else "该通信域无通信数据"
        )
        time_display = f"{time_ms:.2f}" if has_data else "—"
        percent_display = f"{percent:.2f}%" if has_data else "—"
        timeline_html = ""
        if has_data:
            timeline_html = f"""
                    <div class="comm-timeline">
                        <div class="comm-exposed" style="width:{exposed_width:.2f}%;"></div>
                        <div class="comm-hidden" style="width:{hidden_width:.2f}%;"></div>
                    </div>
            """
        row_class = "" if has_data else ' class="comm-empty"'
        percent_class = "comm-percent has-data" if has_data else "comm-percent"
        comm_rows_html.append(f"""
        <tr{row_class}>
            <td>{domain}</td>
            <td class="comm-time">{time_display}</td>
            <td class="{percent_class}">{percent_display}</td>
            <td class="comm-timeline-cell">
                <div class="comm-timeline-track" aria-label="{time_bar_label}" title="{time_bar_label}">
                    {timeline_html}
                </div>
            </td>
        </tr>
        """)

    deepep_section_html = ""
    if ep_details:
        deepep_rows = []
        for phase in ep_details.get("phases", []):
            deepep_rows.append(f"""
        <tr>
            <td>{escape_value(phase.get("phase", ""))}</td>
            <td>{display_value(phase.get("calls"), 2)}</td>
            <td>{phase.get("raw_time_s", 0) * 1000:.4f}</td>
            <td>{phase.get("effective_overlap_ratio", 0) * 100:.2f}%</td>
            <td>{phase.get("hidden_time_s", 0) * 1000:.4f}</td>
            <td>{phase.get("exposed_time_s", 0) * 1000:.4f}</td>
        </tr>
            """)
        deepep_section_html = f"""
    <div class="section">
        <div class="section-title">DeepEP heuristic phase estimate</div>
        <div class="env-note-desc">
            Embedded heuristic model (no performance profile):
            payload bucket {escape_value(ep_details.get("payload_bucket"))},
            topology {escape_value(ep_details.get("topology"))},
            overlap policy {escape_value(ep_details.get("overlap_policy"))}.
        </div>
        <table>
            <thead>
                <tr><th>Phase</th><th>Calls</th><th>Raw (ms)</th><th>Overlap</th><th>Hidden (ms)</th><th>Exposed (ms)</th></tr>
            </thead>
            <tbody>{''.join(deepep_rows)}</tbody>
        </table>
    </div>
        """

    # 生成算子实测原始数据表格行
    # benchmark_rows_html = []
    # if has_benchmark_data:
    #     for item in benchmark_details:
    #         benchmark_rows_html.append(f"""
    #     <tr>
    #         <td>{escape_value(item.get("name", ""))}</td>
    #         <td>{escape_value(item.get("group", ""))}</td>
    #         <td>{escape_value(item.get("status", ""))}</td>
    #         <td>{display_value(item.get("mean_ms"), 4)}</td>
    #         <td>{display_value(item.get("p50_ms"), 4)}</td>
    #         <td>{display_value(item.get("min_ms"), 4)}</td>
    #         <td>{display_value(item.get("max_ms"), 4)}</td>
    #         <td>{display_value(item.get("tflops"), 2)}</td>
    #         <td>{display_value(item.get("peak_memory_mb"), 1)}</td>
    #         <td style="text-align: left;">{escape_value(item.get("shape", ""))}</td>
    #         <td style="text-align: left;">{escape_value(item.get("message", ""))}</td>
    #     </tr>
    #     """)
    #
    # benchmark_section_html = ""
    # if has_benchmark_data:
    #     benchmark_section_html = f"""
    # <!-- 4. 算子实测原始数据 -->
    # <div class="section">
    #     <div class="section-title">
    #         <span class="icon">🧪</span> 算子实测原始数据
    #     </div>
    #     <table>
    #         <thead>
    #             <tr><th>Operator</th><th>Group</th><th>Status</th><th>Mean (ms)</th><th>P50 (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>TFLOPS</th><th>Peak MB</th><th>Shape</th><th>Message</th></tr>
    #         </thead>
    #         <tbody>
    #             {''.join(benchmark_rows_html)}
    #         </tbody>
    #     </table>
    # </div>
    #     """

    env_section_html = ""
    if has_benchmark_data and benchmark_env_info:
        env_items = []
        env_display_order = [
            "accelerator",
            "driver",
            "torch_version",
            "fa_version",
            "te_version",
            "triton_version",
            "dtk_version",
            "cuda_available",
            "gpu_count",
        ]
        for key in env_display_order:
            if key in benchmark_env_info:
                env_items.append(f"""
            <div class="env-item">
                <div class="env-key">{escape_value(key)}</div>
                <div class="env-value">{escape_value(benchmark_env_info.get(key))}</div>
            </div>
        """)
        for key, value in benchmark_env_info.items():
            if key not in env_display_order:
                env_items.append(f"""
            <div class="env-item">
                <div class="env-key">{escape_value(key)}</div>
                <div class="env-value">{escape_value(value)}</div>
            </div>
        """)
        env_section_html = f"""
    <!-- 实测环境信息 -->
    <div class="section env-note">
        <div class="env-note-title">实测环境信息</div>
        <div class="env-note-desc">
            以下为本次算子实测运行环境的关键版本和设备信息，仅用于结果复现和横向对比参考。
        </div>
        <div class="env-grid">
            {''.join(env_items)}
        </div>
    </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Training Simulation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', 'Roboto', Arial, sans-serif; background: #f0f2f5; padding: 24px; }}
        .container {{ max-width: 1300px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); overflow: hidden; }}

        /* 头部 */
        .header {{ background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%); color: white; padding: 28px 32px; }}
        .header h1 {{ font-size: 28px; font-weight: 600; margin-bottom: 8px; letter-spacing: -0.5px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}

        /* 卡片区域 */
        .cards {{ display: flex; flex-wrap: wrap; gap: 20px; padding: 32px; background: #f8f9fa; }}
        .card {{ width: calc(33.333% - 13.33px); padding: 20px; box-sizing: border-box; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); text-align: center; }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .card-label {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 8px; }}
        .card-value {{ font-size: 32px; font-weight: 700; color: #0f3460; line-height: 1.2; }}
        .card-unit {{ font-size: 14px; font-weight: normal; color: #6c757d; }}

        /* 章节 */
        .section {{ padding: 24px 32px; border-bottom: 1px solid #e9ecef; }}
        .section:last-child {{ border-bottom: none; }}
        .section-title {{ display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 600; color: #1a1a2e; margin-bottom: 20px; }}
        .section-title .icon {{ font-size: 24px; }}

        /* 表格 */
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px 12px; border-bottom: 1px solid #e9ecef; text-align: center; vertical-align: middle; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #495057; border-bottom: 2px solid #dee2e6; }}
        td:first-child, th:first-child {{ text-align: center; font-weight: 500; }}
        tr:hover {{ background: #f8f9fa; }}

        /* 通信条形图 */
        .comm-section-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 16px; }}
        .comm-section-heading .section-title {{ margin-bottom: 0; }}
        .comm-table-wrap {{ border: 1px solid #e4eaf0; border-radius: 12px; overflow-x: auto; background: #fff; }}
        .comm-table {{ table-layout: fixed; min-width: 720px; }}
        .comm-table th {{ background: #f5f8fb; color: #405266; }}
        .comm-table tr:last-child td {{ border-bottom: none; }}
        .comm-table td:first-child {{ color: #0f3460; font-weight: 700; }}
        .comm-time {{ color: #405266; }}
        .comm-percent.has-data {{ color: #244a68; font-size: 15px; font-weight: 700; }}
        .comm-timeline-cell {{ padding-left: 18px; padding-right: 18px; }}
        .comm-timeline-track {{ height: 12px; width: min(100%, 240px); margin: 0 auto; background: #eef2f6; border-radius: 6px; overflow: hidden; box-shadow: inset 0 0 0 1px rgba(72,130,175,0.08); }}
        .comm-timeline {{ display: flex; width: 100%; height: 100%; min-width: 2px; overflow: hidden; border-radius: 6px; }}
        .comm-exposed {{ height: 100%; background: linear-gradient(90deg, #76b7e8, #a4c5f5); }}
        .comm-hidden {{ height: 100%; background: #d6eeff; }}
        .comm-empty td {{ color: #a1a9b2 !important; background: #fafbfc; }}
        .comm-empty .comm-timeline-track {{ background: #e5e8eb; box-shadow: none; }}
        .comm-legend {{ display: flex; align-items: center; gap: 12px; color: #607080; font-size: 12px; white-space: nowrap; }}
        .comm-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
        .comm-legend i {{ display: inline-block; width: 16px; height: 8px; border-radius: 4px; }}

        .non-gemm-toggle {{ position: absolute; opacity: 0; pointer-events: none; }}
        .non-gemm-toggle:not(:checked) + .non-gemm-table .non-gemm-extra {{ display: none; }}
        .non-gemm-toggle-row:hover {{ background: transparent; }}
        .non-gemm-toggle-row td {{ padding: 10px 12px 4px; border-bottom: none; }}
        .non-gemm-toggle-button {{ display: inline-flex; align-items: center; gap: 8px; cursor: pointer; color: #47627c; font-size: 13px; font-weight: 600; user-select: none; }}
        .non-gemm-toggle-button:hover {{ color: #0f3460; }}
        .non-gemm-toggle-arrow {{ display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border: 1px solid #cbd8e4; border-radius: 50%; background: #f6f9fb; color: #0f3460; font-size: 18px; line-height: 1; transition: transform 0.2s ease, background 0.2s ease; }}
        .non-gemm-toggle-button:hover .non-gemm-toggle-arrow {{ background: #eaf1f7; }}
        .non-gemm-hide-label {{ display: none; }}
        .non-gemm-toggle:checked + .non-gemm-table .non-gemm-toggle-arrow {{ transform: rotate(180deg); }}
        .non-gemm-toggle:checked + .non-gemm-table .non-gemm-show-label {{ display: none; }}
        .non-gemm-toggle:checked + .non-gemm-table .non-gemm-hide-label {{ display: inline; }}

        /* 时间总览双栏 */
        .time-overview {{ display: flex; gap: 24px; flex-wrap: wrap; }}
        .time-stat {{ flex: 1; background: #f8f9fa; border-radius: 12px; padding: 20px; text-align: center; }}
        .time-stat .label {{ font-size: 13px; color: #6c757d; margin-bottom: 8px; }}
        .time-stat .value {{ font-size: 28px; font-weight: 700; color: #1a1a2e; }}
        .time-stat .ratio {{ font-size: 14px; color: #28a745; margin-top: 8px; }}

        /* 进度条 */
        .progress-bar {{ background: #e9ecef; border-radius: 20px; height: 24px; margin: 15px 0; overflow: hidden; }}
        .progress-fill {{ background: linear-gradient(90deg, #0f3460, #1a4a7a); height: 100%; border-radius: 20px; display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; color: white; font-size: 12px; font-weight: 500; }}
        .progress-fill.compute {{ background: linear-gradient(90deg, #28a745, #20c997); }}
        .progress-fill.comm {{ background: linear-gradient(90deg, #fd7e14, #ffc107); }}
        .time-breakdown {{ margin-top: 15px; }}
        .time-breakdown-bar {{ display: flex; width: 100%; height: 28px; background: #e9ecef; border-radius: 14px; overflow: hidden; }}
        .time-breakdown-segment.compute {{ background: #8bd6b2; }}
        .time-breakdown-segment.comm {{ background: #ff7a5f; }}
        .time-breakdown-labels {{ display: flex; justify-content: space-between; gap: 16px; margin-top: 7px; font-size: 14px; font-weight: 700; }}
        .time-breakdown-label.compute {{ color: #8bd6b2; }}
        .time-breakdown-label.comm {{ color: #ff7a5f; text-align: right; }}

        /* 实测环境说明 */
        .env-note {{ background: #fbfcfd; }}
        .env-note-title {{ font-size: 16px; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }}
        .env-note-desc {{ font-size: 13px; line-height: 1.7; color: #6c757d; margin-bottom: 14px; }}
        .env-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px 12px; }}
        .env-item {{ border: 1px solid #e6eaf0; background: #ffffff; border-radius: 8px; padding: 10px 12px; }}
        .env-key {{ font-size: 12px; color: #6c757d; margin-bottom: 4px; }}
        .env-value {{ font-size: 13px; color: #1a1a2e; font-weight: 600; overflow-wrap: anywhere; }}

        /* 脚注 */
        .footer {{ background: #f8f9fa; padding: 16px 32px; font-size: 12px; color: #6c757d; border-top: 1px solid #e9ecef; }}

        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            .section {{ padding: 16px 20px; }}
            .cards {{ padding: 20px; gap: 12px; }}
            .card-value {{ font-size: 24px; }}
            table {{ font-size: 12px; }}
            th, td {{ padding: 8px 6px; }}
            .env-grid {{ grid-template-columns: 1fr; }}
            .comm-section-heading {{ align-items: flex-start; flex-direction: column; gap: 10px; }}
            .comm-timeline-cell {{ padding-left: 12px; padding-right: 12px; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🚀 大模型训练模拟报告</h1>
        <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <!-- 关键指标卡片 -->
    <div class="cards">
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">模型尺寸</div>
            <div class="card-value">{model_b}<span class="card-unit"> B</span></div>
        </div>
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">GPU总数</div>
            <div class="card-value">{total_gpus}</div>
        </div>
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">ScaleUp</div>
            <div class="card-value">{gpus_per_node}</div>
        </div>
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">TGS</div>
            <div class="card-value">{tgs:.2f}<span class="card-unit"> tokens/p/s</span></div>
        </div>
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">MFU</div>
            <div class="card-value">{mfu:.2f}<span class="card-unit">%</span></div>
        </div>
        <div class="card">
            <div class="card-label" style="font-size: 15px; font-weight: bold;">并行策略</div>
            <div class="card-value" style="font-size: 18px; font-weight: bold;">{ps_text}</div>
        </div>
    </div>

    <!-- 1. PP Stage 显存分析 -->
    <div class="section">
        <div class="section-title">
            <span class="icon">📊</span> 各 Pipeline Stage 显存占用
        </div>
        <table>
            <thead>
                <tr><th>PP Rank</th><th>Peak In-flight Microbatches</th><th>Parameters</th><th>Activations</th><th>Weight Grad Opt (GB)</th><th>Activations (GB)</th><th>总显存占用 (GB)</th></tr>
            </thead>
            <tbody>
                {''.join(pp_rows_html)}
            </tbody>
        </table>
    </div>

    <!-- 2. 时间开销概览 -->
    <div class="section">
        <div class="section-title">
            <span class="icon">⏱️</span> 时间开销分析
        </div>
        <div class="time-overview">
            <div class="time-stat">
                <div class="label" style="font-size: 15px;">🖥️ 计算时间</div>
                <div class="value">{compute_ms:.1f}<span class="card-unit"> ms</span></div>
            </div>
            <div class="time-stat">
                <div class="label" style="font-size: 15px;">📡 通信时间</div>
                <div class="value">{comm_ms:.1f}<span class="card-unit"> ms</span></div>
            </div>
            <div class="time-stat">
                <div class="label" style="font-size: 15px;">⏲️ 总时间</div>
                <div class="value">{total_ms:.1f}<span class="card-unit"> ms</span></div>
            </div>
        </div>
        <div class="time-breakdown">
            <div class="time-breakdown-bar" aria-label="计算 {compute_percent:.1f}%，通信 {comm_percent:.1f}%">
                <div class="time-breakdown-segment compute" style="width: {compute_percent:.2f}%;"></div>
                <div class="time-breakdown-segment comm" style="width: {comm_percent:.2f}%;"></div>
            </div>
            <div class="time-breakdown-labels">
                <span class="time-breakdown-label compute">计算 {compute_percent:.1f}%</span>
                <span class="time-breakdown-label comm">通信 {comm_percent:.1f}%</span>
            </div>
        </div>
    </div>

    <!-- 3. 计算时间详细 -->
    <div class="section">
        <div class="section-title">
            <span class="icon">📈</span> GEMM 计算时间详情
        </div>
        <table>
            <thead>
                <tr><th>Model part</th><th>B</th><th>M</th><th>N</th><th>K</th><th>计算次数</th><th>forward时间(ms)</th><th>backward时间(ms)</th><th>总时间(ms)</th><th>实测时间(ms)</th><th>Source</th></tr>
            </thead>
            <tbody>
                {''.join(gemm_compute_rows_html)}
            </tbody>
        </table>
    </div>

    <!-- 4. Non-GEMM 计算时间详细 -->
    <div class="section">
        <div class="section-title">
            <span class="icon">📈</span> Non-GEMM 计算时间详情
        </div>
        <input type="checkbox" id="non-gemm-toggle" class="non-gemm-toggle">
        <table class="non-gemm-table">
            <thead>
                <tr><th>Model part</th><th>Shape / Elements</th><th>Elements</th><th>Total Elements</th><th>计算次数</th><th>Fwd ops/elem</th><th>Bwd ops/elem</th><th>forward时间(ms)</th><th>backward时间(ms)</th><th>总时间(ms)</th><th>实测时间(ms)</th><th>Source</th></tr>
            </thead>
            <tbody>
                {''.join(non_gemm_compute_rows_html)}
                {non_gemm_toggle_html}
            </tbody>
        </table>
    </div>

    <!-- 5. 通信域时间分布 -->
    <div class="section">
        <div class="comm-section-heading">
            <div class="section-title">
                <span class="icon">📡</span> 通信域时间分布
            </div>
            <div class="comm-legend">
                <span><i style="background:#a4c5f5;"></i>暴露通信时间</span>
                <span><i style="background:#d6eeff;"></i>已掩盖通信时间</span>
            </div>
        </div>
        <div class="comm-table-wrap">
            <table class="comm-table">
                <colgroup>
                    <col style="width:16%;">
                    <col style="width:22%;">
                    <col style="width:16%;">
                    <col style="width:46%;">
                </colgroup>
                <thead>
                    <tr><th>通信域</th><th>时间 (ms)</th><th>占比</th><th>通信 Overlap 图示</th></tr>
                </thead>
                <tbody>
                    {''.join(comm_rows_html)}
                </tbody>
            </table>
        </div>
    </div>

    {deepep_section_html}

    {env_section_html}

    <div class="footer">
        报告生成于HCU大模型训练模拟工具 | 数据仅供参考
    </div>
</div>
</body>
</html>
"""
    return html


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 示例数据
    pp_stages_data = [
        {"pp_rank": 0, "param_elems": 290_980_000_000, "act_elems": 28_660_000_000, "wgo_gib": 4877.88,
         "act_gib": 53.38, "total_gib": 4931.26},
        {"pp_rank": 1, "param_elems": 290_670_000_000, "act_elems": 21_290_000_000, "wgo_gib": 4872.67,
         "act_gib": 39.66, "total_gib": 4912.32},
        {"pp_rank": 2, "param_elems": 290_670_000_000, "act_elems": 14_190_000_000, "wgo_gib": 4872.67,
         "act_gib": 26.44, "total_gib": 4899.10},
        {"pp_rank": 3, "param_elems": 290_980_000_000, "act_elems": 8_100_000_000, "wgo_gib": 4877.88,
         "act_gib": 15.08, "total_gib": 4892.97},
    ]

    compute_details_data = [
        {"model_part": "qkv_weight", "b": 1, "m": 5120, "n": 4096, "k": 2048, "compute_count": 96, "forward_ms": 58.9,
         "backward_ms": 117.82},
        {"model_part": "flash_attn", "b": 1, "m": 0, "n": 0, "k": 0, "compute_count": 96, "forward_ms": 58.9,
         "backward_ms": 117.82},
        {"model_part": "attn_proj", "b": 1, "m": 2048, "n": 4096, "k": 2048, "compute_count": 96, "forward_ms": 58.9,
         "backward_ms": 117.82},
        {"model_part": "topk_router", "b": 1, "m": 128, "n": 2048, "k": 4096, "compute_count": 96, "forward_ms": 58.9,
         "backward_ms": 117.82},
        {"model_part": "moe_linear_fc1", "b": 1, "m": 1536, "n": 2048, "k": 256, "compute_count": 1536,
         "forward_ms": 58.9, "backward_ms": 117.82},
        {"model_part": "moe_linear_fc2", "b": 1, "m": 2048, "n": 768, "k": 256, "compute_count": 1536,
         "forward_ms": 58.9, "backward_ms": 117.82},
        {"model_part": "lm_head", "b": 1, "m": 151936, "n": 4096, "k": 2048, "compute_count": 8, "forward_ms": 58.9,
         "backward_ms": 117.82},
    ]

    comm_domains_data = [
        {"domain": "TP", "time_ms": 160.08, "percent": 22.13},
        {"domain": "CP", "time_ms": 0.00, "percent": 0.00},
        {"domain": "DP", "time_ms": 0.00, "percent": 0.00},
        {"domain": "PP", "time_ms": 193.84, "percent": 26.80},
        {"domain": "EP", "time_ms": 369.49, "percent": 51.08},
        {"domain": "ETP", "time_ms": 0.00, "percent": 0.00},
    ]

    html_output = generate_training_report(
        model_b=4653,
        total_gpus=128,
        gpus_per_node=8,
        tgs=356.32,
        mfu=25.20,
        pp_stages=pp_stages_data,
        compute_ms=5162.2,
        comm_ms=723.4,
        compute_details=compute_details_data,
        comm_domains=comm_domains_data,
        parallel_strategy={"tp": 1, "cp": 1, "ep": 4, "dp": 2, "pp": 4}
    )

    # 保存到文件
    with open("training_report_output.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    log_step(logger, "training report generated: training_report_output.html")
