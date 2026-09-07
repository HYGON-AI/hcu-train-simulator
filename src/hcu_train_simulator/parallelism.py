# Copyright (c) 2026 Hygon Information Technology Co., Ltd.


def get_num_microbatches(global_batch_size, micro_batch_size, dp_size):
    return global_batch_size / micro_batch_size / dp_size


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
