# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = SRC_PATH / "hcu_train_simulator" / "templates" / "config.yaml"


def main(config_path=None, generate_report=True):
    sys.path.insert(0, str(SRC_PATH))

    from hcu_train_simulator.simulator import run_simulation

    return run_simulation(
        str(config_path or DEFAULT_CONFIG_PATH),
        generate_report=generate_report,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training simulator from the source tree.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-report", action="store_false", dest="generate_report")
    cli_args = parser.parse_args()
    main(cli_args.config, generate_report=cli_args.generate_report)
