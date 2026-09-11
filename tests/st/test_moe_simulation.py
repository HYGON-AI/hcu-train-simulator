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


class MoeSimulationST(unittest.TestCase):
    def test_moe_model_runs_full_simulation_from_repo_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_config_path = tmp_path / "config.json"
            model_config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3ForCausalLM"],
                        "vocab_size": 151936,
                        "hidden_size": 2048,
                        "num_hidden_layers": 48,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 4,
                        "intermediate_size": 6144,
                        "head_dim": 128,
                        "tie_word_embeddings": False,
                        "moe_intermediate_size": 768,
                        "num_experts": 128,
                        "num_experts_per_tok": 8,
                    }
                ),
                encoding="utf-8",
            )

            sim_config_path = tmp_path / "moe_sim.yaml"
            sim_config_path.write_text(
                textwrap.dedent(
                    f"""\
                    model_path: '{model_config_path}'

                    parallel_config:
                      tp_size: 1
                      cp_size: 1
                      ep_size: 8
                      etp_size: 1
                      pp_size: 4
                      pp_schedule: 1f1b
                      num_layers_per_vp_stage: null
                      use_distributed_optimizer: true
                      decoder_first_pipeline_num_layers: null
                      decoder_last_pipeline_num_layers: null
                      full_recompute: false
                      micro_batch_size: 1
                      global_batch_size: 64
                      seq_length: 4096
                      num_gpus: 32
                      overlap_mode: auto
                      tp_overlap_ratio: 0
                      cp_overlap_ratio: 0
                      dp_overlap_ratio: 0
                      ep_overlap_ratio: 0
                      pp_overlap_ratio: 0

                    hardware_config:
                      fp16_tflops: 480
                      fp8_tflops: 960
                      gpus_per_node: 8
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
                [sys.executable, str(TRAIN_SIM), str(sim_config_path)],
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
            msg=f"moe simulation failed with output:\n{output}",
        )
        self.assertIn("Arguments validation successful", output)
        self.assertIn("Model Size:", output)
        self.assertIn("model_part", output)
        self.assertIn("communication ratio", output)
        self.assertIn("'dp': ", output)
        self.assertIn("topk_router", output)


if __name__ == "__main__":
    unittest.main()
