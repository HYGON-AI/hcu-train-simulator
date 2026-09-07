# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from hcu_train_simulator.config.utils import flatten_dataclass
from hcu_train_simulator.context import get_config
from hcu_train_simulator.logging import get_logger
from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    attention_score_dim,
    kv_projection_dim,
    output_loss_factor,
    query_projection_dim,
    shared_expert_intermediate_total,
    value_head_dim,
)
from hcu_train_simulator.modeling.module_cost import ModuleSpecMemoryModel
from hcu_train_simulator.modeling.spec import (
    MLASelfAttention,
    MoELayer,
    SelfAttention,
    get_attention_spec,
    get_layer_specs,
    get_mlp_spec,
    module_is,
)
from hcu_train_simulator.modeling.statistics_registry import (
    get_attention_flops,
    get_component_flops,
)


logger = get_logger(__name__)


class ModelStatistics:
    def __init__(self):
        config = get_config()
        self.config = config
        for key, value in flatten_dataclass(config).items():
            setattr(self, key, value)
        self.model_spec = config.model_spec

    def compute_model_size(self):
        all_params = ModuleSpecMemoryModel(get_config()).model_parameter_estimate().total_elements
        logger.debug("model size: %.2fB", round(all_params / 1e9, 2))
        return all_params

    def compute_flops(self):
        """Return global training FLOPs for the resolved heterogeneous spec."""

        hidden = self.hidden_size
        seq = self.seq_length
        num_tokens = self.global_batch_size * seq
        train_matmul = 6
        total_per_token = 0

        for layer in get_layer_specs(self.model_spec):
            attention = get_attention_spec(layer)
            registered = get_attention_flops(attention)
            if registered is not None:
                total_per_token += registered(self, attention, seq, train_matmul)
            elif module_is(attention, MLASelfAttention):
                query_dim = query_projection_dim(self)
                q_rank = self.q_lora_rank or 0
                q_weights = hidden * q_rank + q_rank * query_dim if q_rank else hidden * query_dim
                kv_weights = hidden * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank * kv_projection_dim(self)
                total_per_token += train_matmul * (q_weights + kv_weights + attention_output_dim(self) * hidden)
                total_per_token += (
                    train_matmul
                    * self.num_attention_heads
                    * (attention_score_dim(self) + value_head_dim(self))
                    * seq
                )
            elif module_is(attention, SelfAttention):
                query_dim = query_projection_dim(self)
                kv_dim = self.num_query_groups * self.kv_channels
                total_per_token += train_matmul * (
                    hidden * (query_dim + 2 * kv_dim) + query_dim * hidden
                )
                total_per_token += (
                    train_matmul
                    * self.num_attention_heads
                    * (attention_score_dim(self) + value_head_dim(self))
                    * seq
                )

            if module_is(get_mlp_spec(layer), MoELayer):
                routed = 3 * hidden * self.moe_ffn_hidden_size * self.moe_router_topk
                shared = 3 * hidden * shared_expert_intermediate_total(self)
                shared_gate = (
                    hidden
                    if shared_expert_intermediate_total(self)
                    and self.moe_shared_expert_gate
                    else 0
                )
                total_per_token += train_matmul * (
                    routed
                    + shared
                    + shared_gate
                    + hidden * self.num_moe_experts
                )
            else:
                total_per_token += train_matmul * 3 * hidden * self.ffn_hidden_size

        total = num_tokens * total_per_token
        total += num_tokens * train_matmul * self.vocab_size * hidden * output_loss_factor(self)

        for component in (
            getattr(self.model_spec.submodules, "vision_model", None),
            getattr(self.model_spec.submodules, "mtp", None),
        ):
            if component is None:
                continue
            registered = get_component_flops(component)
            if registered is None:
                raise NotImplementedError(
                    f"no analytical FLOP handler registered for component {component.module!r}"
                )
            total += registered(self, component, train_matmul)
        return total
