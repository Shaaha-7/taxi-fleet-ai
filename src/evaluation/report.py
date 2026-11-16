"""Markdown result tables."""
from __future__ import annotations

from pathlib import Path


def _write(text: str, path: Path) -> str:
    path.write_text(text, encoding="utf-8")
    return text


def write_forecast_table(res: dict, path: Path) -> str:
    horizons = [k for k in next(iter(res.values())) if k != "all"]
    head = "| Model | " + " | ".join(f"MAE {h}" for h in horizons) + " | MAE all | RMSE all | MAPE all (%) | Coverage (%) | Width |"
    lines = ["## Pickup forecast, test period (15-min slots)", "", head, "|" + "---|" * (len(horizons) + 6)]
    for name, m in res.items():
        a = m["all"]
        lines.append(f"| {name} | " + " | ".join(f"{m[h]['MAE']:.3f}" for h in horizons)
                     + f" | {a['MAE']:.3f} | {a['RMSE']:.3f} | {a['MAPE']:.1f} | {a['coverage']:.1f} | {a['width']:.2f} |")
    return _write("\n".join(lines) + "\n", path)


CONTROL_COLS = ["served_rate_%", "mean_wait_min", "p90_wait_min", "empty_km_share_%", "empty_km_per_trip", "empty_km",
                "rebalance_km", "profit_AED", "utilization_%", "service_gini", "decision_ms"]


def write_control_table(res: dict, path: Path, title: str = "") -> str:
    lines = [f"## Control comparison ({title})", "",
             "Mean ± 95% CI over test days. Δ empty km is relative to `none` (no rebalancing).", "",
             "| Method | " + " | ".join(CONTROL_COLS) + " | Δ empty km vs none |",
             "|" + "---|" * (len(CONTROL_COLS) + 2)]
    base = res.get("none", {}).get("empty_km", (None,))[0]
    for name, m in res.items():
        cells = []
        for c in CONTROL_COLS:
            mean, ci = m[c]
            cells.append(f"{mean:,.1f} ± {ci:,.1f}" if abs(mean) >= 100 else f"{mean:.2f} ± {ci:.2f}")
        delta = f"{100 * (m['empty_km'][0] / base - 1):+.1f}%" if base else "n/a"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {delta} |")
    return _write("\n".join(lines) + "\n", path)


def write_sweep_table(results: dict, path: Path) -> str:
    sizes = list(results)
    methods = list(next(iter(results.values())))
    lines = ["## Fleet-size sweep (policies trained at the config fleet size)", ""]
    for metric in ("served_rate_%", "mean_wait_min", "empty_km_per_trip", "profit_AED"):
        lines += [f"### {metric}", "", "| Method | " + " | ".join(str(s) for s in sizes) + " |",
                  "|" + "---|" * (len(sizes) + 1)]
        for m in methods:
            lines.append(f"| {m} | " + " | ".join(f"{results[s][m][metric][0]:,.1f}" for s in sizes) + " |")
        lines.append("")
    return _write("\n".join(lines), path)


def write_transfer_table(res: dict, path: Path) -> str:
    lines = ["## NYC TLC -> Dubai-like city transfer (pickup forecast, Dubai test days)", "",
             "| Setting | MAE all | RMSE all | MAE h1 | MAE h4 |", "|---|---|---|---|---|"]
    for name, m in res.items():
        lines.append(f"| {name} | {m['all']['MAE']:.3f} | {m['all']['RMSE']:.3f} | {m['h1']['MAE']:.3f} | {m['h4']['MAE']:.3f} |")
    return _write("\n".join(lines) + "\n", path)
