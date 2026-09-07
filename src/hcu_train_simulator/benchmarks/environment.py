# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcceleratorOccupancy:
    known: bool
    idle: bool
    foreign_pids: tuple[int, ...] = ()
    command: str = ""
    reason: str = ""
    device_index: int | None = None
    vram_percent: float | None = None
    utilization_percent: float | None = None

    def as_quality(self) -> dict:
        return asdict(self)


def read_dtk_version() -> str:
    rocm_path = os.environ.get("ROCM_PATH")
    if not rocm_path:
        return ""
    version_file = Path(rocm_path) / ".dtk_version"
    try:
        return version_file.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


def _extract_pids(output: str) -> tuple[set[int], bool]:
    no_process_patterns = (
        r"\bno\s+kfd\s+pids?\s+currently\s+running\b",
        r"\bno\s+running\s+process(?:es)?\b",
        r"\bno\s+process(?:es)?\b",
    )
    if any(re.search(pattern, output, re.IGNORECASE) for pattern in no_process_patterns):
        return set(), True

    # hy-smi --showpids prints one process block per KFD process. Match only
    # the exact PID field at the beginning of a line so PASID, GPUID, HCU node
    # indexes, and memory counters cannot be mistaken for process IDs.
    labelled_pids = {
        int(match.group(1))
        for match in re.finditer(r"^\s*PID\s*[:=]\s*(\d+)\b", output, re.IGNORECASE | re.MULTILINE)
    }
    if labelled_pids:
        return labelled_pids, True

    pids: set[int] = set()
    process_section_seen = False
    lines = output.splitlines()
    pid_column = None

    for line in lines:
        lowered = line.lower()
        if "pid" in lowered and ("process" in lowered or "command" in lowered or "name" in lowered):
            process_section_seen = True
            pid_column = lowered.find("pid")
            continue
        if pid_column is not None:
            segment = line[pid_column:pid_column + 16]
            match = re.match(r"\s*(\d+)\b", segment)
            if match:
                pids.add(int(match.group(1)))

    return pids, process_section_seen


def _visible_device_index() -> tuple[int | None, str]:
    """Return the physical HCU used by the default logical CUDA device."""

    for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(variable)
        if value is None:
            continue
        first = value.split(",", 1)[0].strip()
        if re.fullmatch(r"\d+", first):
            return int(first), variable
        return None, variable
    return 0, "default cuda:0"


def _extract_hy_smi_device_metrics(output: str, device_index: int) -> tuple[float, float] | None:
    """Extract (VRAM%, HCU%) for one physical HCU from the hy-smi summary."""

    lines = output.splitlines()
    vram_column = None
    utilization_column = None
    header_line = None
    for index, line in enumerate(lines):
        columns = line.split()
        if columns and columns[0] == "HCU" and "VRAM%" in columns and "HCU%" in columns:
            header_line = index
            vram_column = columns.index("VRAM%")
            utilization_column = columns.index("HCU%")
            break
    if header_line is None or vram_column is None or utilization_column is None:
        return None

    for line in lines[header_line + 1:]:
        columns = line.split()
        if not columns or columns[0] != str(device_index):
            continue
        if len(columns) <= max(vram_column, utilization_column):
            return None
        try:
            vram = float(columns[vram_column].rstrip("%"))
            utilization = float(columns[utilization_column].rstrip("%"))
        except ValueError:
            return None
        return vram, utilization
    return None


def _extract_hy_smi_process_devices(output: str) -> tuple[dict[int, set[int]], bool]:
    """Return KFD PID-to-HCU mappings from hy-smi --showpids output."""

    if re.search(r"\bno\s+kfd\s+pids?\s+currently\s+running\b", output, re.IGNORECASE):
        return {}, True

    matches = list(
        re.finditer(r"^\s*PID\s*:\s*(\d+)\b", output, re.IGNORECASE | re.MULTILINE)
    )
    if not matches:
        return {}, False

    processes: dict[int, set[int]] = {}
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        block = output[match.end():block_end]
        device_line = re.search(
            r"^\s*HCU\s+Index\s*:\s*(.*)$",
            block,
            re.IGNORECASE | re.MULTILINE,
        )
        devices = set()
        if device_line:
            devices = {int(value) for value in re.findall(r"\d+", device_line.group(1))}
        processes[int(match.group(1))] = devices
    return processes, True


def _run(command: list[str], timeout: int = 5):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def check_accelerator_occupancy(*, check_vram: bool = True) -> AcceleratorOccupancy:
    """Check the physical HCU selected as the default CUDA device.

    Pre-benchmark checks should keep ``check_vram=True`` so an accelerator
    with allocated memory is not treated as available. Post-benchmark checks
    may set it to ``False`` because the current process can retain allocator
    cache after all kernels have completed.
    """

    current_pid = os.getpid()
    device_index, visibility_source = _visible_device_index()
    if device_index is None:
        return AcceleratorOccupancy(
            known=False,
            idle=False,
            reason=f"selected accelerator from {visibility_source} is not a numeric HCU index",
        )

    completed = _run(["hy-smi"])
    if completed is not None and completed.returncode == 0:
        output = "\n".join((completed.stdout or "", completed.stderr or ""))
        metrics = _extract_hy_smi_device_metrics(output, device_index)
        if metrics is not None:
            vram_percent, utilization_percent = metrics
            common = {
                "command": f"hy-smi (HCU {device_index})",
                "device_index": device_index,
                "vram_percent": vram_percent,
                "utilization_percent": utilization_percent,
            }
            metrics_idle = utilization_percent == 0 and (
                not check_vram or vram_percent == 0
            )
            if metrics_idle:
                reason = (
                    "selected accelerator compute is idle "
                    "(HCU=0%; post-benchmark VRAM ignored)"
                    if not check_vram
                    else "selected accelerator is idle (VRAM=0%, HCU=0%)"
                )
                return AcceleratorOccupancy(
                    known=True,
                    idle=True,
                    reason=reason,
                    **common,
                )

            if not check_vram:
                return AcceleratorOccupancy(
                    known=True,
                    idle=False,
                    reason="selected accelerator has non-zero HCU utilization",
                    **common,
                )

            # After a benchmark the current process may retain its CUDA/HIP
            # context and cached VRAM. Attribute non-zero metrics before
            # deciding whether another process is interfering with this HCU.
            process_result = _run(["hy-smi", "--showpids"])
            if process_result is not None and process_result.returncode == 0:
                process_output = "\n".join(
                    (process_result.stdout or "", process_result.stderr or "")
                )
                processes, recognized = _extract_hy_smi_process_devices(process_output)
                if recognized:
                    selected_pids = {
                        pid for pid, devices in processes.items() if device_index in devices
                    }
                    foreign = tuple(sorted(pid for pid in selected_pids if pid != current_pid))
                    if foreign:
                        return AcceleratorOccupancy(
                            known=True,
                            idle=False,
                            foreign_pids=foreign,
                            reason="foreign processes detected on selected accelerator",
                            **common,
                        )
                    if current_pid in selected_pids:
                        return AcceleratorOccupancy(
                            known=True,
                            idle=True,
                            reason="non-zero selected-accelerator usage belongs only to current process",
                            **common,
                        )

            return AcceleratorOccupancy(
                known=True,
                idle=False,
                reason="selected accelerator has non-zero VRAM or HCU utilization",
                **common,
            )

    # Compatibility fallbacks are used only when the hy-smi utilization table
    # cannot be read. They remain conservative because device attribution is
    # unavailable in these formats.
    for command in (["hy-smi", "--showpids"], ["rocm-smi", "--showpids"]):
        completed = _run(command)
        if completed is None or completed.returncode != 0:
            continue
        output = "\n".join((completed.stdout or "", completed.stderr or ""))
        pids, recognized = _extract_pids(output)
        if not recognized:
            continue
        foreign = tuple(sorted(pid for pid in pids if pid != current_pid))
        return AcceleratorOccupancy(
            known=True,
            idle=not foreign,
            foreign_pids=foreign,
            command=" ".join(command),
            reason=("foreign accelerator processes detected" if foreign else "accelerator is idle"),
            device_index=device_index,
        )

    # /dev/kfd is shared by ROCm/DTK compute processes. It is conservative and
    # intentionally treats any foreign user of the device stack as interference.
    completed = _run(["fuser", "/dev/kfd"])
    if completed is not None and completed.returncode in {0, 1}:
        output = " ".join((completed.stdout or "", completed.stderr or ""))
        pids = {int(value) for value in re.findall(r"\b\d+\b", output)}
        foreign = tuple(sorted(pid for pid in pids if pid != current_pid))
        return AcceleratorOccupancy(
            known=True,
            idle=not foreign,
            foreign_pids=foreign,
            command="fuser /dev/kfd",
            reason=("foreign accelerator processes detected" if foreign else "accelerator is idle"),
            device_index=device_index,
        )

    return AcceleratorOccupancy(
        known=False,
        idle=False,
        reason="accelerator occupancy could not be verified",
    )
