# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


def get_num_microbatches(global_batch_size, micro_batch_size, dp_size):
    return global_batch_size / micro_batch_size / dp_size


def get_vpp_layer_candidates(num_hidden_layers, pp_size):
    """Return legal Megatron layers-per-virtual-stage candidates.

    The physical PP layout must first divide the transformer layers evenly.
    Every returned chunk size creates at least two virtual stages per physical
    rank; VP size itself remains derived rather than user-configured.
    """

    if pp_size <= 1 or num_hidden_layers % pp_size:
        return []
    layers_per_pp_rank = num_hidden_layers // pp_size
    vp_sizes = [
        value
        for value in range(2, layers_per_pp_rank + 1)
        if layers_per_pp_rank % value == 0
    ]
    return [layers_per_pp_rank // vp_size for vp_size in vp_sizes]


def build_interleaved_schedule_table(num_microbatches, vp_size, microbatch_group_size):
    """Return Megatron's tunable interleaved forward schedule.

    Each entry is ``(microbatch_id, vp_rank)``.  Megatron executes one group of
    microbatches through every virtual model chunk before advancing to the next
    group.
    """

    num_microbatches = int(num_microbatches)
    microbatch_group_size = max(1, int(microbatch_group_size))
    schedule = []
    for group_start in range(0, num_microbatches, microbatch_group_size):
        group_end = min(group_start + microbatch_group_size, num_microbatches)
        for vp_rank in range(vp_size):
            schedule.extend(
                (microbatch_id, vp_rank)
                for microbatch_id in range(group_start, group_end)
            )
    return schedule


def interleaved_peak_activation_counts(
    pp_rank,
    pp_size,
    vp_size,
    num_microbatches,
    chunk_activation_weights,
    microbatch_group_size=None,
):
    """Find the peak retained activations for an interleaved 1F1B rank.

    The simulator previously maximized every virtual chunk independently.  The
    resulting peaks cannot occur at the same instant and grow quadratically in
    ``vp_size``.  This replays Megatron's warmup/steady/cooldown ordering and
    returns the simultaneously-live microbatch count for every chunk at the
    weighted peak.  The Megatron default group size is one physical pipeline.
    """

    num_microbatches = int(num_microbatches)
    if num_microbatches <= 0:
        return [0] * vp_size
    if len(chunk_activation_weights) != vp_size:
        raise ValueError("chunk_activation_weights must contain one value per VP stage")

    group_size = microbatch_group_size or pp_size
    group_size = min(max(1, int(group_size)), num_microbatches)
    schedule = build_interleaved_schedule_table(
        num_microbatches,
        vp_size,
        group_size,
    )
    total_virtual_microbatches = len(schedule)
    warmup_microbatches = (
        2 * max(pp_size - pp_rank - 1, 0)
        + (vp_size - 1) * group_size
    )
    warmup_microbatches = min(warmup_microbatches, total_virtual_microbatches)

    live = [0] * vp_size
    peak = list(live)
    peak_weight = 0

    def update_peak():
        nonlocal peak, peak_weight
        weight = sum(
            count * chunk_weight
            for count, chunk_weight in zip(live, chunk_activation_weights)
        )
        if weight > peak_weight:
            peak_weight = weight
            peak = list(live)

    def forward(virtual_microbatch_id):
        _, vp_rank = schedule[virtual_microbatch_id]
        live[vp_rank] += 1
        update_peak()

    def backward(virtual_microbatch_id):
        _, forward_vp_rank = schedule[virtual_microbatch_id]
        vp_rank = vp_size - forward_vp_rank - 1
        if live[vp_rank] <= 0:
            raise ValueError("invalid interleaved schedule produced a negative live count")
        live[vp_rank] -= 1

    for virtual_microbatch_id in range(warmup_microbatches):
        forward(virtual_microbatch_id)

    remaining = total_virtual_microbatches - warmup_microbatches
    for index in range(remaining):
        forward(warmup_microbatches + index)
        backward(index)

    for virtual_microbatch_id in range(remaining, total_virtual_microbatches):
        backward(virtual_microbatch_id)

    if any(live):
        raise ValueError("interleaved schedule did not release all retained activations")
    return peak


def build_pp_layer_counts(num_hidden_layers, pp_size, first_layers=0, last_layers=0):
    num_special_stages = int(bool(first_layers)) + int(bool(last_layers))

    if num_special_stages == pp_size:
        layer_counts = []
        if first_layers:
            layer_counts.append(first_layers)
        if last_layers:
            layer_counts.append(last_layers)
        return layer_counts

    remaining_layers = num_hidden_layers - first_layers - last_layers
    middle_stages = pp_size - num_special_stages
    layers_per_middle_stage = remaining_layers // middle_stages

    layer_counts = []
    if first_layers:
        layer_counts.append(first_layers)
    layer_counts.extend([layers_per_middle_stage] * middle_stages)
    if last_layers:
        layer_counts.append(last_layers)
    return layer_counts


def build_pipeline_layer_layout(model_spec, parallel_config):
    """Assign concrete transformer-layer ModuleSpecs to PP/VP ranks.

    The return value is ``layout[pp_rank][vp_rank] -> list[ModuleSpec]``.
    MTP layers are always placed in the final virtual stage, matching the
    simulator's existing scheduling assumption.
    """

    from hcu_train_simulator.modeling.spec import get_layer_specs

    all_layers = get_layer_specs(model_spec)
    base_layers = [layer for layer in all_layers if not layer.metainfo.get("is_mtp_layer", False)]
    mtp_layers = [layer for layer in all_layers if layer.metainfo.get("is_mtp_layer", False)]
    pp_size = parallel_config.pp_size

    if parallel_config.num_layers_per_vp_stage:
        chunk_size = parallel_config.num_layers_per_vp_stage
        vp_size = max(parallel_config.vp_size, 1)
        layout = [[[] for _ in range(vp_size)] for _ in range(pp_size)]
        cursor = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                layout[pp_rank][vp_rank] = base_layers[cursor:cursor + chunk_size]
                cursor += chunk_size
        if cursor != len(base_layers):
            raise ValueError("pipeline layout does not consume all transformer layers")
        layout[-1][-1].extend(mtp_layers)
        return layout

    first_layers = parallel_config.decoder_first_pipeline_num_layers or 0
    last_layers = parallel_config.decoder_last_pipeline_num_layers or 0
    counts = build_pp_layer_counts(len(base_layers), pp_size, first_layers, last_layers)
    layout = [[[]] for _ in range(pp_size)]
    cursor = 0
    for pp_rank, count in enumerate(counts):
        layout[pp_rank][0] = base_layers[cursor:cursor + count]
        cursor += count
    if cursor != len(base_layers):
        raise ValueError("pipeline layout does not consume all transformer layers")
    layout[-1][0].extend(mtp_layers)
    return layout
