# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""
Collect bus bandwidth (busbw) data from multiple nccl-test log files
and produce a combined output similar to bw500SM.txt.

Usage:
    python collect_bandwidth.py --input-dir ./logs/bw1000_1 --output bw1000_1.txt
"""

import re
import os
import argparse
import logging
from collections import defaultdict


class HumanFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        if record.levelname == "INFO":
            return message
        return f"{record.levelname} | {message}"


logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().handlers[0].setFormatter(HumanFormatter())
logger = logging.getLogger(__name__)


def parse_log_file(filepath):
    """
    Parse a single log file and return a list of (size, busbw) tuples.
    busbw is the out-of-place bus bandwidth (column 8, index 7).
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment lines, warnings, summary lines
            if line.startswith('#') or line.startswith('Launch') or line.startswith('Avg'):
                continue
            # Check if line starts with a digit (size)
            if not re.match(r'^\d', line):
                continue
            parts = line.split()
            if len(parts) < 8:  # Need at least up to busbw column (index 7)
                continue
            try:
                size = int(parts[0])
                # busbw is the 8th field (index 7)
                busbw = float(parts[6])
                data.append((size, busbw))
            except (ValueError, IndexError):
                continue
    return data

def extract_op_np_from_filename(filename):
    """
    Extract operation name and np from filename like 'all_reduce_perf_np8.log'
    Returns (op, np) or (None, None) if pattern doesn't match.
    """
    base = os.path.splitext(filename)[0]
    # Match pattern: anything ending with '_perf_np<digits>'
    match = re.match(r'(.+)_perf_np(\d+)$', base)
    if match:
        op = match.group(1)
        np_val = int(match.group(2))
        return op, np_val
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Collect bus bandwidth from nccl-test logs')
    parser.add_argument('--input-dir', '-i', default='.', help='Directory containing log files')
    parser.add_argument('--output', '-o', default='bandwidth_combined.txt', help='Output file path')
    args = parser.parse_args()

    # Structure: ops[op][size][np] = busbw
    ops = defaultdict(lambda: defaultdict(dict))

    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.log'):
            continue
        op, np_val = extract_op_np_from_filename(fname)
        if op is None:
            logger.warning("skipping %s: cannot extract op and np", fname)
            continue
        filepath = os.path.join(args.input_dir, fname)
        data = parse_log_file(filepath)
        if not data:
            logger.warning("no data extracted from %s", fname)
        for size, bw in data:
            ops[op][size][np_val] = bw

    # Collect all unique np values across all ops, sorted
    all_nps = set()
    for op_dict in ops.values():
        for size_dict in op_dict.values():
            all_nps.update(size_dict.keys())
    all_nps = sorted(all_nps)

    # Write output
    with open(args.output, 'w') as f:
        for op in sorted(ops.keys()):
            f.write(f"{op}\n")
            sizes = sorted(ops[op].keys())
            for size in sizes:
                row = [str(size)]
                for np_val in all_nps:
                    bw = ops[op][size].get(np_val, 0.0)
                    # Format bandwidth: keep two decimals, but if integer show without decimal
                    if bw == int(bw):
                        row.append(str(int(bw)))
                    else:
                        row.append(f"{bw:.2f}")
                f.write("\t".join(row) + "\n")
            f.write("\n")  # blank line between ops

    logger.info("output written to %s", args.output)
    logger.info("found %s operations: %s", len(ops), ", ".join(ops.keys()))
    logger.info("NP values present: %s", all_nps)

if __name__ == "__main__":
    main()
