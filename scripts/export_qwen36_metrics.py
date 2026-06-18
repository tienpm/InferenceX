#!/usr/bin/env python3
"""Export Qwen3.6-27B benchmark JSON files to a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"isl(?P<isl>\d+)_osl(?P<osl>\d+)_conc(?P<conc>\d+)")


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        match = CASE_RE.search(path.stem)
        isl = int(match.group("isl")) if match else mean_int(data.get("input_lens"))
        osl = int(match.group("osl")) if match else mean_int(data.get("output_lens"))
        conc = int(match.group("conc")) if match else data.get("max_concurrency")

        rows.append(
            {
                "file": path.name,
                "date": data.get("date", ""),
                "model": data.get("model_id", ""),
                "isl": isl,
                "osl": osl,
                "conc": conc,
                "num_prompts": data.get("num_prompts"),
                "completed": data.get("completed"),
                "duration": data.get("duration"),
                "request_throughput": data.get("request_throughput"),
                "output_throughput": data.get("output_throughput"),
                "total_token_throughput": data.get("total_token_throughput"),
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "p99_ttft_ms": data.get("p99_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "p99_tpot_ms": data.get("p99_tpot_ms"),
                "mean_itl_ms": data.get("mean_itl_ms"),
                "p99_itl_ms": data.get("p99_itl_ms"),
                "mean_e2el_ms": data.get("mean_e2el_ms"),
                "p99_e2el_ms": data.get("p99_e2el_ms"),
            }
        )
    return rows


def mean_int(values: Any) -> int | None:
    if not isinstance(values, list) or not values:
        return None
    nums = [number(item) for item in values]
    nums = [item for item in nums if item is not None]
    if not nums:
        return None
    return int(round(sum(nums) / len(nums)))


def summarize_gpu_csv(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {}
        for raw_row in reader:
            for key, value in raw_row.items():
                if key is None:
                    continue
                parsed = number(str(value).replace("W", "").replace("%", ""))
                if parsed is not None:
                    columns.setdefault(key.strip(), []).append(parsed)

    wanted = [
        ("power", "Power"),
        ("temperature", "Temperature"),
        ("utilization", "Utilization"),
        ("memory", "Memory"),
        ("vram", "VRAM"),
        ("clock", "Clock"),
    ]
    summary: list[tuple[str, str]] = []
    for needle, label in wanted:
        matches = [
            (name, values)
            for name, values in columns.items()
            if needle.lower() in name.lower() and values
        ]
        for name, values in matches[:3]:
            avg = sum(values) / len(values)
            summary.append((f"{label}: {name}", f"avg {avg:.2f}, max {max(values):.2f}"))
    return summary


def render_markdown(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    image: str,
    results_dir: Path,
    gpu_metrics: list[tuple[str, str]],
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Qwen3.6-27B Inference Metrics",
        "",
        f"- Generated: {generated}",
        f"- Model: `{model_id}`",
        f"- SGLang image: `{image}`",
        f"- Results directory: `{results_dir}`",
        "",
    ]

    if not rows:
        lines.extend(["No benchmark JSON files were found.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Serving Results",
            "",
            "| Case | Completed | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) | Mean E2EL (ms) | P99 E2EL (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        case_name = f"{row['isl']}:{row['osl']} c{row['conc']}"
        completed = f"{row.get('completed', '')}/{row.get('num_prompts', '')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    completed,
                    fmt(row.get("duration")),
                    fmt(row.get("request_throughput")),
                    fmt(row.get("output_throughput")),
                    fmt(row.get("total_token_throughput")),
                    fmt(row.get("mean_ttft_ms")),
                    fmt(row.get("p99_ttft_ms")),
                    fmt(row.get("mean_tpot_ms")),
                    fmt(row.get("p99_tpot_ms")),
                    fmt(row.get("mean_itl_ms")),
                    fmt(row.get("p99_itl_ms")),
                    fmt(row.get("mean_e2el_ms")),
                    fmt(row.get("p99_e2el_ms")),
                ]
            )
            + " |"
        )

    if gpu_metrics:
        lines.extend(["", "## GPU Telemetry", "", "| Metric | Value |", "| --- | --- |"])
        for metric, value in gpu_metrics:
            lines.append(f"| {metric} | {value} |")

    lines.extend(["", "## Raw Files", ""])
    for row in rows:
        lines.append(f"- `{row['file']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--image", default="")
    parser.add_argument("--gpu-csv", type=Path)
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    gpu_csv = args.gpu_csv or args.results_dir / "gpu_metrics.csv"
    markdown = render_markdown(
        rows,
        model_id=args.model_id,
        image=args.image,
        results_dir=args.results_dir,
        gpu_metrics=summarize_gpu_csv(gpu_csv),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
