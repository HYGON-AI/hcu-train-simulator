# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from datetime import datetime
from pathlib import Path


def write_training_report(html, output_dir="training_report"):
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    report_path = report_dir / f"training_report_{timestamp}.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
