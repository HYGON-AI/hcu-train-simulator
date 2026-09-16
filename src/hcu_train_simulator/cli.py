# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse

from hcu_train_simulator.commands.init import init_config
from hcu_train_simulator.simulator import run_simulation
from hcu_train_simulator.commands.analyze import analyze_communication, analyze_compute, analyze_memory
from hcu_train_simulator.commands.search import search_best_perf
from hcu_train_simulator.commands.solve import SolveOptions, solve_training_config


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument(
        "template",
        nargs="?",
        default="config",
    )

    # run
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument(
        "--no-report",
        action="store_false",
        dest="generate_report",
        help="skip HTML report generation",
    )

    # analyze
    analyze_parser = subparsers.add_parser("analyze")

    # analyze 再嵌套一层
    analyze_subparsers = analyze_parser.add_subparsers(
        dest="analyze_target",
        required=True,
    )

    memory_parser = analyze_subparsers.add_parser(
        "memory",
        help="analyze memory usage",
    )

    memory_parser.add_argument(
        "config",
        help="path to config yaml",
    )

    compute_parser = analyze_subparsers.add_parser(
        "compute",
        help="analyze compute time",
    )

    compute_parser.add_argument(
        "config",
        help="path to config yaml",
    )

    comm_parser = analyze_subparsers.add_parser(
        "comm",
        help="analyze communication time",
    )

    comm_parser.add_argument(
        "config",
        help="path to config yaml",
    )

    # search
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument(
        "config",
        help="path to config yaml",
    )

    # solve
    solve_parser = subparsers.add_parser(
        "solve",
        help="solve minimum-GPU and higher-performance training configurations",
    )
    solve_parser.add_argument("config", help="path to the existing config yaml")
    solve_parser.add_argument("--model-path", help="override model_path from config yaml")
    solve_parser.add_argument("--seq-length", type=int, help="override sequence length")
    solve_parser.add_argument("--micro-batch-size", type=int, help="override micro batch size")
    solve_parser.add_argument(
        "--global-batch-size",
        type=int,
        help="preferred minimum global batch size; raised when needed for the bubble limit",
    )
    solve_parser.add_argument(
        "--max-bubble-ratio",
        type=float,
        default=0.15,
        help="maximum pipeline bubble ratio (default: 0.15)",
    )
    solve_parser.add_argument(
        "--memory-margin-gib",
        type=float,
        help="HBM safety margin (default: 5 GiB)",
    )

    args = parser.parse_args()

    if args.command == "init":
        init_config(args.template)

    elif args.command == "run":
        run_simulation(args.config, generate_report=args.generate_report)
    elif args.command == "analyze":
        if args.analyze_target == "memory":
            analyze_memory(args.config)
        elif args.analyze_target == "compute":
            analyze_compute(args.config)
        elif args.analyze_target == "comm":
            analyze_communication(args.config)
        else:
            raise NotImplementedError(args.analyze_target)
    elif args.command == "search":
        search_best_perf(args.config)
    elif args.command == "solve":
        solve_training_config(
            args.config,
            SolveOptions(
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                global_batch_size=args.global_batch_size,
                max_bubble_ratio=args.max_bubble_ratio,
                memory_margin_gib=args.memory_margin_gib,
                model_path=args.model_path,
            ),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
