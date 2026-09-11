# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.config import initialize_simulation
from hcu_train_simulator.modeling import ModuleSpecMemoryModel
from hcu_train_simulator.parallelism import build_pipeline_layer_layout, get_num_microbatches
from hcu_train_simulator.context import get_config
from hcu_train_simulator.logging import get_logger, log_table


BYTES_PER_GIB = 1024 * 1024 * 1024
logger = get_logger(__name__)


def non_interleaved_1f1b_activation_microbatches(pp_rank, pp_size, num_microbatches):
    """Peak retained microbatches for one stage in non-interleaved 1F1B.

    A stage first retains ``pp_size - pp_rank - 1`` warmup forwards.  If a
    steady-state microbatch remains, its next forward briefly raises the peak
    by one before the oldest retained microbatch is consumed by backward.
    """

    warmup_microbatches = min(
        max(pp_size - pp_rank - 1, 0),
        num_microbatches,
    )
    steady_state_forward = int(num_microbatches > warmup_microbatches)
    return warmup_microbatches + steady_state_forward


def regular_1f1b_activation_microbatches(pp_rank, pp_size, num_microbatches):
    """Backward-compatible alias for the non-interleaved 1F1B calculation."""

    return non_interleaved_1f1b_activation_microbatches(
        pp_rank,
        pp_size,
        num_microbatches,
    )


def interleaved_vpp_activation_microbatches(pp_rank, vp_rank, pp_size, vp_size, num_microbatches):
    if pp_rank == 0:
        num_mbs_vp_rank = pp_size * (vp_size - vp_rank)
        return min(num_mbs_vp_rank, num_microbatches)

    if vp_rank == 0:
        num_mbs_vp_rank = pp_size + max((pp_size - pp_rank) * 2 - 1 - pp_size, 0)
    elif vp_rank == vp_size - 1:
        num_mbs_vp_rank = min((pp_size - pp_rank) * 2 + 1, pp_size)
    else:
        num_mbs_vp_rank = pp_size

    return min(num_mbs_vp_rank, num_microbatches)


def interleaved_vpp_output_microbatches(pp_rank, pp_size, num_microbatches):
    return min((pp_size - pp_rank) * 2 + 1, num_microbatches)


def estimate_memory(log_result=False):

    config = get_config()

    pp_size = config.parallel.pp_size
    num_microbatches = get_num_microbatches(
        config.parallel.global_batch_size,
        config.parallel.micro_batch_size,
        config.parallel.dp_size,
    )

    model = ModuleSpecMemoryModel(config)
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    embedding_spec = config.model_spec.submodules.embedding
    decoder_norm_spec = config.model_spec.submodules.decoder.submodules.layer_norm
    output_spec = config.model_spec.submodules.output_layer
    vision_spec = getattr(config.model_spec.submodules, "vision_model", None)
    mtp_spec = getattr(config.model_spec.submodules, "mtp", None)

    report = []
    for pp_rank in range(pp_size):
        static_mem = 0
        expert_static_mem = 0
        saved_activation_mem = 0
        peak_workspace_mem = 0
        activation_microbatches = None
        module_breakdown = []

        def add_spec(spec, path, activation_microbatches=0):
            nonlocal static_mem, expert_static_mem
            nonlocal saved_activation_mem, peak_workspace_mem
            params = model.parameter_estimate(spec)
            activations = model.activation_estimate(spec)
            static_mem += params.total_elements
            expert_static_mem += params.expert_elements
            saved_activation_mem += (
                activations.saved_elements * activation_microbatches
            )
            if activation_microbatches:
                peak_workspace_mem = max(
                    peak_workspace_mem,
                    activations.workspace_elements,
                )
            for row in model.memory_rows(spec, path=path):
                row = dict(row)
                row["saved_act_elems"] = int(
                    row["saved_act_elems"] * activation_microbatches / config.parallel.cp_size
                )
                row["workspace_elems"] = int(row["workspace_elems"] / config.parallel.cp_size)
                module_breakdown.append(row)

        if config.parallel.num_layers_per_vp_stage:
            # vpp
            for vp_rank, layer_specs in enumerate(layout[pp_rank]):
                num_mbs_vp_rank = interleaved_vpp_activation_microbatches(
                    pp_rank,
                    vp_rank,
                    pp_size,
                    config.parallel.vp_size,
                    num_microbatches,
                )

                for layer_index, layer_spec in enumerate(layer_specs):
                    add_spec(
                        layer_spec,
                        f"model.decoder.pp[{pp_rank}].vp[{vp_rank}].layers[{layer_index}]",
                        num_mbs_vp_rank,
                    )

                if pp_rank == 0 and vp_rank == 0:
                    if vision_spec is not None:
                        add_spec(vision_spec, "model.vision_model", num_mbs_vp_rank)
                    add_spec(embedding_spec, "model.embedding", num_mbs_vp_rank)

                if pp_rank == pp_size - 1 and vp_rank == config.parallel.vp_size - 1:
                    num_mbs_output = interleaved_vpp_output_microbatches(
                        pp_rank,
                        pp_size,
                        num_microbatches,
                    )
                    if mtp_spec is not None:
                        add_spec(mtp_spec, "model.mtp", num_mbs_output)
                    add_spec(decoder_norm_spec, "model.decoder.final_norm", num_mbs_output)
                    add_spec(output_spec, "model.output_layer", num_mbs_output)

        else:
            # for each pprank activation calc
            num_mbs_pp_rank = non_interleaved_1f1b_activation_microbatches(
                pp_rank,
                pp_size,
                num_microbatches,
            )
            activation_microbatches = num_mbs_pp_rank
            layer_specs = layout[pp_rank][0]
            for layer_index, layer_spec in enumerate(layer_specs):
                add_spec(
                    layer_spec,
                    f"model.decoder.pp[{pp_rank}].layers[{layer_index}]",
                    num_mbs_pp_rank,
                )

            if pp_rank == 0:
                if vision_spec is not None:
                    add_spec(vision_spec, "model.vision_model", num_mbs_pp_rank)
                add_spec(embedding_spec, "model.embedding", num_mbs_pp_rank)

            if pp_rank == pp_size - 1:
                if mtp_spec is not None:
                    add_spec(mtp_spec, "model.mtp", num_mbs_pp_rank)
                add_spec(decoder_norm_spec, "model.decoder.final_norm", num_mbs_pp_rank)
                add_spec(output_spec, "model.output_layer", num_mbs_pp_rank)

        # cp
        saved_activation_mem = saved_activation_mem // config.parallel.cp_size
        peak_workspace_mem = peak_workspace_mem // config.parallel.cp_size
        dynamic_mem = saved_activation_mem + peak_workspace_mem

        # static process fp16
        # weight_grad_optim = static_mem * (6 + 12/config.dp_size/config.cp_size)
        num_bytes_per_params = (
            6 + 12 / config.parallel.dp_size / config.parallel.cp_size
            if config.parallel.use_distributed_optimizer
            else 18
        )
        if expert_static_mem and config.parallel.ep_size * config.parallel.etp_size > 1:
            dense_params = static_mem - expert_static_mem
            num_bytes_per_params_dense = num_bytes_per_params
            num_bytes_per_params_moe = (18 if not config.parallel.use_distributed_optimizer else
                                        6 + (12 / (config.parallel.num_gpus / config.parallel.pp_size / config.parallel.ep_size / config.parallel.etp_size)))

            weight_grad_optim = (
                dense_params * num_bytes_per_params_dense
                + expert_static_mem * num_bytes_per_params_moe
            )
        else:
            weight_grad_optim = static_mem * num_bytes_per_params

        wgo_gib = round(weight_grad_optim / BYTES_PER_GIB, 2)

        # dynamic process fp16
        activation = dynamic_mem * 2
        act_gib = round(activation / BYTES_PER_GIB, 2)
        total_gib = round((weight_grad_optim + activation) / BYTES_PER_GIB, 2)

        cur_rpt = {
            "pp_rank": pp_rank,
            "param_elems": int(static_mem),
            "act_elems": int(dynamic_mem),
            "saved_act_elems": int(saved_activation_mem),
            "workspace_elems": int(peak_workspace_mem),
            "activation_microbatches": activation_microbatches,
            "expert_param_elems": int(expert_static_mem),
            "wgo_gib": wgo_gib,
            "act_gib": act_gib,
            "total_gib": total_gib,
            "module_breakdown": module_breakdown,
        }
        report.append(cur_rpt)

    display_fields = (
        "pp_rank",
        "activation_microbatches",
        "param_elems",
        "act_elems",
        "wgo_gib",
        "act_gib",
        "total_gib",
    )
    display_report = [{key: row[key] for key in display_fields} for row in report]
    if log_result:
        log_table(logger, "memory estimate", display_report)
    return report

if __name__ == '__main__':
    initialize_simulation("templates/config.yaml")
    _ = estimate_memory(log_result=True)
