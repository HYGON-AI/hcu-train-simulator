# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.config import initialize_simulation, validate_config
from hcu_train_simulator.context import get_config
from hcu_train_simulator.estimators import estimate_communication
from hcu_train_simulator.logging import get_logger
from hcu_train_simulator.runtime_summary import log_parallel_strategy


logger = get_logger(__name__)


def analyze_communication(config_path):
    initialize_simulation(config_path)
    validate_config()
    log_parallel_strategy(logger, get_config())
    return estimate_communication(log_result=True)
