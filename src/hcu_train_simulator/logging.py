# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys


DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class MultilineFormatter(logging.Formatter):
    @staticmethod
    def _level_prefix(levelname):
        if levelname == "INFO":
            return ""
        return f"{levelname} | "

    def format(self, record):
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        if getattr(record, "with_time", False):
            prefix = f"{self.formatTime(record, self.datefmt)} | "
            if record.levelname != "INFO":
                prefix = f"{prefix}{record.levelname} | "
        else:
            prefix = self._level_prefix(record.levelname)
        lines = message.splitlines() or [""]
        return "\n".join(f"{prefix}{line}" for line in lines)


def get_logger(name: str = "", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            MultilineFormatter(
                datefmt=DEFAULT_DATE_FORMAT,
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_step(logger: logging.Logger, message: str, *args, level: int = logging.INFO) -> None:
    logger.log(level, message, *args, extra={"with_time": True})


def log_table(logger: logging.Logger, title: str, rows, *, tablefmt: str = "pretty") -> None:
    from tabulate import tabulate

    logger.info("\n=== %s ===\n%s", title, tabulate(rows, headers="keys", tablefmt=tablefmt))
