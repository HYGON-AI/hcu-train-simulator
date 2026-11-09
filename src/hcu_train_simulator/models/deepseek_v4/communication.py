# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Communication payloads for the expanded residual stream and shared KV."""

from functools import cache

from hcu_train_simulator.communication import Algorithm, CommType, GroupType
from hcu_train_simulator.modeling.spec import get_attention_spec, get_layer_specs
from hcu_train_simulator.parallelism import build_pipeline_layer_layout


def _stage_layout(config):
    return build_pipeline_layer_layout(config.model_spec, config.parallel)


def pipeline_activation_width(config):
    t = config.transformer
    p = config.parallel
    options = t.architecture_options
    width = t.hidden_size * options["hc_mult"]
    if not options["csa2"] or p.pp_size <= 1:
        return width

    # V4.1 also carries the FP32 mix coefficients and the shared state
    # required on the next stage. Use the heaviest boundary with 1F1B/VPP.
    attention = [get_attention_spec(layer).params for layer in get_layer_specs(config.model_spec)]
    layout = _stage_layout(config)
    max_shared = 0
    for stages in layout:
        for chunk in stages:
            if not chunk:
                continue
            next_id = get_attention_spec(chunk[-1]).params["layer_id"] + 1
            if next_id >= len(attention):
                continue
            following = attention[next_id]
            shared = 0
            ratio = following["ratio"]
            if ratio and not following["owns_kv"]:
                shared += (t.kv_channels + t.index_head_dim) / ratio
                if not following["runs_indexer"]:
                    shared += 2 * min(t.index_topk, p.seq_length // ratio)
            if 0 <= options["candidate_source"] < next_id and any(
                layer["candidate_limit"] for layer in attention[next_id:]
            ):
                candidate_ratio = attention[options["candidate_source"]]["ratio"]
                block_size = options["candidate_block_size"]
                visible_blocks = (p.seq_length // candidate_ratio + block_size - 1) // block_size
                shared += 2 * min(options["candidate_blocks"], visible_blocks)
            max_shared = max(max_shared, shared)
    return width + 2 * options["hc_mult"] + max_shared


def _time(simulator, group, collective, payload):
    return simulator.get_communication_time(
        com_type=collective, algorithm=Algorithm.RING, group_type=group,
        data_size=payload, rank=0,
    )


def context_parallel_time(simulator, config, num_microbatches):
    p = config.parallel
    if p.cp_size == 1:
        return 0.0
    t = config.transformer
    options = t.architecture_options
    seq = p.seq_length
    local_seq = seq // p.cp_size
    # SWA needs the preceding shard's window tail. Use a full KV exchange
    # when the window spans multiple neighboring shards.
    halo = min(seq - local_seq, options["window"] - 1)
    halo_bytes = 2 * p.micro_batch_size * halo * t.kv_channels
    if halo <= local_seq:
        window_time = 2 * _time(simulator, GroupType.CP, CommType.P2P, halo_bytes) if halo else 0.0
    else:
        payload = 2 * p.micro_batch_size * seq * t.kv_channels
        window_time = _time(simulator, GroupType.CP, CommType.ALL_GATHER, payload)
        window_time += _time(simulator, GroupType.CP, CommType.REDUCE_SCATTER, payload)

    @cache
    def compressed_time(ratio, indexed):
        width = t.kv_channels + (t.index_head_dim if indexed else 0)
        payload = 2 * p.micro_batch_size * (seq // ratio) * width
        if not payload:
            return 0.0
        return (
            _time(simulator, GroupType.CP, CommType.ALL_GATHER, payload)
            + _time(simulator, GroupType.CP, CommType.REDUCE_SCATTER, payload)
        )

    def stage_time(stages):
        total = 0.0
        for chunk in stages:
            has_main_kv = False
            for layer in chunk:
                a = get_attention_spec(layer).params
                total += window_time
                if a["ratio"] and (a["owns_kv"] or not has_main_kv):
                    total += compressed_time(a["ratio"], a["indexed"])
                    has_main_kv = True
        return total

    return num_microbatches * max(map(stage_time, _stage_layout(config)))


def engram_parallel_time(simulator, config, num_microbatches):
    p = config.parallel
    options = config.transformer.architecture_options
    if not options.get("engram_rows") or p.ep_size * p.etp_size == 1:
        return 0.0
    tables_on_stage = [
        sum(get_attention_spec(layer).params["layer_id"] in options["engram_rows"]
            for chunk in stages for layer in chunk)
        for stages in _stage_layout(config)
    ]
    tokens = p.micro_batch_size * (p.seq_length // p.cp_size)
    if p.sequence_parallel:
        tokens /= p.tp_size
    request = 4 * tokens * options["engram_lookup_heads"]
    values = 2 * tokens * options["engram_lookup_heads"] * options["engram_head_dim"]
    time = 0.0
    for group, size in ((GroupType.EP, p.ep_size), (GroupType.ETP, p.etp_size)):
        if size > 1:
            time += _time(simulator, group, CommType.ALL_TO_ALL, request)
            time += 2 * _time(simulator, group, CommType.ALL_TO_ALL, values)
    return time * num_microbatches * max(tables_on_stage)
