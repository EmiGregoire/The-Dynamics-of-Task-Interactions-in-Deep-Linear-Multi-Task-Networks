from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvProgressLogger:
    def __init__(self, path: Path, interval: int = 1000) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.path = Path(path)
        self.interval = int(interval)
        self.fieldnames = [
            "step",
            "loss1",
            "loss2",
            "taskweight1",
            "taskweight2",
            "cossim",
            "gradmag1",
            "gradmag2",
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def __call__(self, metrics: dict[str, Any]) -> None:
        step = int(metrics["step"])
        total_steps = int(metrics.get("total_steps", step))
        if step % self.interval != 0 and step != total_steps:
            return

        row = {name: metrics.get(name, "") for name in self.fieldnames}
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CsvProgressLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
