from __future__ import annotations

import json
from pathlib import Path


def compare_latest(report_dir: Path) -> Path:
    v2 = _latest(report_dir, "backtest_summary_*.json", exclude_prefix="v3_")
    v3 = _latest(report_dir, "v3_backtest_summary_*.json")
    if not v2 or not v3:
        raise RuntimeError("Both v2 and v3 summary reports are required")
    a, b = json.loads(v2.read_text(encoding="utf-8")), json.loads(v3.read_text(encoding="utf-8"))
    keys = ["trades", "final_equity", "net_pnl", "win_rate", "profit_factor", "max_drawdown"]
    lines = ["# v2 vs v3 walk-forward comparison", "", f"- v2: `{v2.name}`", f"- v3: `{v3.name}`", "",
             "| Metric | v2 | v3 |", "|---|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {_fmt(a.get(key))} | {_fmt(b.get(key))} |")
    lines.extend(["", "The comparison is descriptive. Prefer the strategy that remains stable across multiple untouched periods, not the one with the largest single backtest result."])
    output = report_dir / "latest_v2_vs_v3.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _latest(directory: Path, pattern: str, exclude_prefix: str | None = None) -> Path | None:
    files = [p for p in directory.glob(pattern) if not exclude_prefix or not p.name.startswith(exclude_prefix)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}" if isinstance(value, float) else str(value)
