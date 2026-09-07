# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from importlib.resources import files
from pathlib import Path

from hcu_train_simulator.logging import get_logger, log_step


logger = get_logger(__name__)


def init_config(template_name="config"):
    resource = files("hcu_train_simulator.templates").joinpath(
        f"{template_name}.yaml"
    )

    output_file = f"{template_name}.yaml"

    Path(output_file).write_text(
        resource.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    log_step(logger, "generated config: %s", output_file)
