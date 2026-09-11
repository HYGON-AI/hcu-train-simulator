# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SIM = REPO_ROOT / "train_sim.py"


class MlaMtpSimulationST(unittest.TestCase):
    def test_mla_mtp_shared_expert_model_runs_full_simulation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_config_path = tmp_path / "config.json"
            model_config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["DeepseekV3ForCausalLM"],
                        "vocab_size": 1001,
                        "padded_vocab_size": 1024,
                        "hidden_size": 128,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 1,
                        "intermediate_size": 512,
                        "head_dim": 32,
                        "tie_word_embeddings": False,
                        "moe_intermediate_size": 64,
                        "n_routed_experts": 8,
                        "topk": 2,
                        "n_shared_experts": 1,
                        "shared_expert_intermediate_size": 64,
                        "q_lora_rank": 16,
                        "kv_lora_rank": 16,
                        "qk_nope_head_dim": 24,
                        "qk_rope_head_dim": 8,
                        "v_head_dim": 32,
                        "num_nextn_predict_layers": 1,
                    }
                ),
                encoding="utf-8",
            )

            sim_config_path = tmp_path / "mla_mtp_sim.yaml"
            sim_config_path.write_text(
                textwrap.dedent(
                    f"""\
                    model_path: '{model_config_path}'

                    parallel_config:
                      tp_size: 2
                      cp_size: 1
                      ep_size: 2
                      etp_size: 1
                      pp_size: 1
                      pp_schedule: 1f1b
                      num_layers_per_vp_stage: null
                      use_distributed_optimizer: true
                      decoder_first_pipeline_num_layers: null
                      decoder_last_pipeline_num_layers: null
                      full_recompute: false
                      micro_batch_size: 1
                      global_batch_size: 4
                      seq_length: 16
                      num_gpus: 4
                      overlap_mode: auto
                      tp_overlap_ratio: 0
                      cp_overlap_ratio: 0
                      dp_overlap_ratio: 0
                      ep_overlap_ratio: 0
                      pp_overlap_ratio: 0

                    hardware_config:
                      fp16_tflops: 480
                      fp8_tflops: 960
                      gpus_per_node: 4
                      hbm_gib: 64
                      intra_bw_gbps: 448
                      inter_bw_gbps: 128
                      gemm_efficiency: 0.4
                      p2p_intra_efficiency: 0.8
                      collective_intra_efficiency: 0.7
                      collective_inter_efficiency: 0.8
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [sys.executable, str(TRAIN_SIM), str(sim_config_path), "--no-report"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )

        output = result.stdout + result.stderr
        self.assertEqual(
            result.returncode,
            0,
            msg=f"MLA/MTP simulation failed with output:\n{output}",
        )
        self.assertIn("Arguments validation successful", output)
        self.assertIn("mla_q_down", output)
        self.assertIn("mla_kv_up", output)
        self.assertIn("shared_expert_fc1", output)
        self.assertIn("TP_shared_expert_raw_time", output)
        self.assertIn("communication ratio", output)


if __name__ == "__main__":
    unittest.main()
