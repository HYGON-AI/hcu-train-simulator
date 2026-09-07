# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from hcu_train_simulator.config import initialize_simulation
from hcu_train_simulator.modeling import ComputeModel
from hcu_train_simulator.logging import get_logger, log_table


logger = get_logger(__name__)


def estimate_compute(log_result=False):

    mc = ComputeModel()
    rst = mc.spec_summary()

    if log_result:
        log_table(logger, "compute estimate", rst)
    return rst

if __name__ == '__main__':
    initialize_simulation("templates/config.yaml")
    _ = estimate_compute(log_result=True)

