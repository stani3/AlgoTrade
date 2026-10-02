"""Render performance reports to one self-contained HTML file (charts embedded as PNG)."""

from __future__ import annotations

import base64
import html
import io
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .report import DAVEY_CRITERIA, FIELDS, MC_KEYS, PerformanceReport

GREEN, RED, GREY, BLUE = "#1a7f37", "#cf222e", "#8c959f", "#0969da"
# Fields where the sign means good or bad; costs and drawdown sizes stay uncoloured.
TONED = {"net_profit", "total_return", "cagr", "benchmark_cagr", "avg_trade", "avg_trade_usd",
         "gross_profit", "gross_loss", "avg_win", "avg_loss", "largest_win", "largest_loss"}  # fmt: skip


def fmt(value: object, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return "–"
    if kind == "date":
        return f"{pd.Timestamp(value):%Y-%m-%d}"
    if isinstance(value, float) and math.isinf(value):
        return "∞" if value > 0 else "-∞"
    match kind:
        case "usd":
            return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"
        case "pct":
            return f"{value:.1%}"
        case "pct2":
            return f"{value:.2%}"
        case "ratio":
            return f"{value:.2f}"
        case "int":
            return f"{int(value):,}"
        case "days":
            return f"{value:,.0f} days"
        case _:
            return f"{value:,.1f}"


def tone(value: object) -> str:
    """CSS class for signed numbers."""

    if isinstance(value, int | float) and not math.isnan(value) and value != 0:
        return "pos" if value > 0 else "neg"
    return ""


def png(fig) -> str:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<img alt="chart" src="data:image/png;base64,{encoded}">'


def equity_chart(report: PerformanceReport) -> str:
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot(report.equity.index, report.equity.values, color=BLUE, lw=1.2, label="Strategy")
    if report.benchmark is not None:
        top.plot(
            report.benchmark.index, report.benchmark.values, color=GREY, lw=0.9, label="Buy & hold"
        )
    top.axhline(report.account["capital"], color="black", lw=0.6, ls=":")
    if report.equity.nunique() > 1:
        top.set_yscale("log")
    top.set_ylabel("Equity ($, log scale)")
    top.legend(loc="upper left", fontsize=8)
    top.grid(alpha=0.2)
    drawdown = report.drawdown * 100
    bottom.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.35, lw=0)
    bottom.set_ylabel("Drawdown (%)")
    bottom.grid(alpha=0.2)
    fig.tight_layout()
    return png(fig)


def trade_chart(report: PerformanceReport) -> str:
    import matplotlib.pyplot as plt

    closed = report.trades[~report.trades["open"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.4))
    returns = closed["return"].to_numpy() * 100
    bins = np.histogram_bin_edges(returns, bins=40)
    left.hist(returns[returns > 0], bins=bins, color=GREEN, alpha=0.8, label="Winners")
    left.hist(returns[returns <= 0], bins=bins, color=RED, alpha=0.8, label="Losers")
    left.set_xlabel("Trade return (% of equity at entry)")
    left.set_ylabel("Trades")
    left.legend(fontsize=8)
    left.grid(alpha=0.2)
    cumulative = closed["pnl_usd"].cumsum().to_numpy()
    numbers = np.arange(1, len(cumulative) + 1)
    right.step(numbers, cumulative, where="post", color=BLUE, lw=1)
    right.axhline(0, color="black", lw=0.6)
    right.set_xlabel("Trade number")
    right.set_ylabel("Cumulative trade P&L ($)")
    right.grid(alpha=0.2)
    fig.tight_layout()
    return png(fig)


def monte_carlo_chart(report: PerformanceReport) -> str:
    import matplotlib.pyplot as plt

    mc = report.monte_carlo
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.4))
    steps = np.arange(mc.paths.shape[1])
    for path in mc.paths[:100]:
        left.plot(steps, path, color=GREY, lw=0.5, alpha=0.35)
    band = np.quantile(mc.paths, [0.05, 0.5, 0.95], axis=0)
    left.plot(steps, band[1], color=BLUE, lw=1.6, label="Median")
    left.plot(steps, band[0], color=BLUE, lw=0.9, ls="--", label="5th / 95th percentile")
    left.plot(steps, band[2], color=BLUE, lw=0.9, ls="--")
    left.axhline(1 - mc.ruin, color=RED, lw=0.9, label=f"Ruin ({mc.ruin:.0%} loss)")
    left.set_ylim(0, max(float(band[2].max()) * 1.25, 1.2))  # a few lucky paths would squash it
    left.set_xlabel(f"Trade (one year = {mc.trades_per_run} trades)")
    left.set_ylabel("Equity (start = 1)")
    left.legend(fontsize=7, loc="upper left")
    left.grid(alpha=0.2)
    max_dd = mc.max_dd * 100
    right.hist(max_dd, bins=30, color=BLUE, alpha=0.7)
    right.axvline(
        DAVEY_CRITERIA["median_max_dd"] * 100, color=RED, lw=1, label="Davey limit (median)"
    )
    right.axvline(np.median(max_dd), color="black", lw=1, ls="--", label="Median")
    right.set_xlabel(f"Max drawdown over one year (%), {mc.runs:,} runs")
    right.set_ylabel("Runs")
    right.legend(fontsize=7)
    right.grid(alpha=0.2)
    fig.tight_layout()
    return png(fig)


def scroll(table: str) -> str:
    return f'<div class="scroll">{table}</div>'


def esc(text: object) -> str:
    return html.escape(str(text))


def key_table(values: dict, keys: list[str], title: str | None = None) -> str:
    rows = []
    for key in keys:
        if key not in values:
            continue
        label, kind = FIELDS[key]
        value = values[key]
        cls = tone(value) if key in TONED else ""
        rows.append(f'<tr><th>{esc(label)}</th><td class="{cls}">{fmt(value, kind)}</td></tr>')
    caption = f"<caption>{esc(title)}</caption>" if title else ""
    return f'<table class="kv">{caption}{"".join(rows)}</table>'


def trade_summary_table(report: PerformanceReport) -> str:
    frame = report.trade_summary
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    rows = []
    for key, values in frame.iterrows():
        label, kind = FIELDS[key]
        cells = "".join(
            f'<td class="{tone(v) if key in TONED else ""}">{fmt(v, kind)}</td>' for v in values
        )
        rows.append(f"<tr><th>{esc(label)}</th>{cells}</tr>")
    return f'<table class="grid"><tr><th></th>{head}</tr>{"".join(rows)}</table>'


def annual_table(report: PerformanceReport) -> str:
    kinds = {"return": "pct", "net_profit": "usd", "max_drawdown": "pct", "trades": "int",
             "win_rate": "pct"}  # fmt: skip
    labels = ["Year", "Return", "Net profit", "Max drawdown", "Trades", "% profitable"]
    head = "".join(f"<th>{label}</th>" for label in labels)
    rows = []
    for year, row in report.annual.iterrows():
        cells = "".join(
            f'<td class="{tone(row[k]) if k in {"return", "net_profit"} else ""}">'
            f"{fmt(row[k], kind)}</td>"
            for k, kind in kinds.items()
        )
        rows.append(f"<tr><th>{year}</th>{cells}</tr>")
    return f'<table class="grid"><tr>{head}</tr>{"".join(rows)}</table>'


def heat(value: float, scale: float) -> str:
    if value is None or math.isnan(value):
        return ""
    strength = min(abs(value) / scale, 1.0) * 0.55
    rgb = "26,127,55" if value > 0 else "207,34,46"
    return f' style="background: rgba({rgb},{strength:.2f})"'


def monthly_table(report: PerformanceReport) -> str:
    frame = report.monthly
    months = frame.drop(columns="Year").to_numpy(dtype=float)
    scale = np.nanquantile(np.abs(months), 0.9) if np.isfinite(months).any() else 1.0
    scale = scale or 1.0
    head = "".join(f"<th>{c}</th>" for c in frame.columns)
    rows = []
    for year, row in frame.iterrows():
        cells = "".join(
            f"<td{heat(v, scale if c != 'Year' else scale * 3)}>{fmt(v, 'pct')}</td>"
            for c, v in row.items()
        )
        rows.append(f"<tr><th>{year}</th>{cells}</tr>")
    values = months[np.isfinite(months)]
    summary = (
        f"<p class='muted'>Profitable months: {(values > 0).mean():.0%} of {len(values)} · "
        f"best {values.max():+.1%} · worst {values.min():+.1%}</p>"
        if len(values)
        else ""
    )
    return f'<table class="grid months"><tr><th></th>{head}</tr>{"".join(rows)}</table>{summary}'


def monte_carlo_section(report: PerformanceReport) -> str:
    mc = report.monte_carlo
    if mc is None:
        return "<h3>Monte Carlo</h3>" + "".join(
            f"<p class='muted'>{esc(n)}</p>" for n in report.notes
        )
    checks = "".join(
        f"<tr><th>{esc(name)}</th><td>{fmt(value, 'ratio' if 'Return' in name else 'pct')}</td>"
        f"<td>{esc(limit)}</td><td class='{'pos' if ok else 'neg'}'>{'PASS' if ok else 'FAIL'}"
        f"</td></tr>"
        for name, value, limit, ok in mc.checks()
    )
    head = "".join(f"<th>{esc(FIELDS[k][0])}</th>" for k in MC_KEYS)
    rows = []
    for size, row in mc.table.iterrows():
        cells = "".join(f"<td>{fmt(row[k], FIELDS[k][1])}</td>" for k in MC_KEYS)
        marker = " (as tested)" if size == 1.0 else ""
        rows.append(f"<tr><th>{size:g}x size{marker}</th>{cells}</tr>")
    return f"""
<h3>Monte Carlo (Davey)</h3>
<p class="muted">{mc.runs:,} runs of one year ({mc.trades_per_run} trades) drawn with replacement
from the closed trades. A run is ruined when equity falls {mc.ruin:.0%} below the start and then
stops trading.</p>
<table class="grid"><tr><th>Davey check (1x)</th><th>Value</th><th>Goal</th><th></th></tr>{checks}
</table>
{scroll(f'<table class="grid"><tr><th></th>{head}</tr>{"".join(rows)}</table>')}
{monte_carlo_chart(report)}"""


def trade_rows(report: PerformanceReport) -> str:
    trades = report.trades
    has_prices = "entry_price" in trades
    head = ["#", "Side", "Entry", "Exit", "Bars", "Return", "P&L", "Cum. P&L"]
    if has_prices:
        head[3:3] = ["Entry price"]
        head[5:5] = ["Exit price", "Exit reason"]
    rows = []
    cumulative = 0.0
    for number, trade in enumerate(trades.to_dict("records"), start=1):
        cumulative += 0.0 if trade["open"] else trade["pnl_usd"]
        cells = [
            str(number),
            "Long" if trade["direction"] > 0 else "Short",
            f"{trade['entry']:%Y-%m-%d %H:%M}",
        ]
        if has_prices:
            cells.append(f"{trade['entry_price']:,.6g}")
        cells.append(f"{trade['exit']:%Y-%m-%d %H:%M}")
        if has_prices:
            cells += [f"{trade['exit_price']:,.6g}", esc(trade["exit_reason"])]
        cls = tone(trade["pnl_usd"])
        cells += [
            str(int(trade["bars"])),
            f'<span class="{cls}">{trade["return"]:+.2%}</span>',
            f'<span class="{cls}">{fmt(trade["pnl_usd"], "usd")}</span>',
            fmt(cumulative, "usd") + (" (open)" if trade["open"] else ""),
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    header = "".join(f"<th>{h}</th>" for h in head)
    return f'<table class="grid trades"><tr>{header}</tr>{"".join(rows)}</table>'


ACCOUNT_KEYS = {
    "Account": ["capital", "ending_equity", "net_profit", "total_return", "cagr", "ann_vol",
                "sharpe", "sortino", "exposure", "trades_per_year"],
    "Drawdown": ["max_drawdown", "max_drawdown_usd", "dd_peak", "dd_trough", "dd_recovery",
                 "longest_drawdown_days", "time_underwater", "return_on_account", "calmar"],
    "Costs and benchmark": ["costs_paid", "funding_paid", "cost_drag_per_year",
                            "funding_drag_per_year", "stopped_at", "benchmark_cagr",
                            "benchmark_sharpe", "benchmark_max_drawdown"],
}  # fmt: skip


def symbol_section(report: PerformanceReport, anchor: str, is_open: bool) -> str:
    meta = report.meta
    closed = (~report.trades["open"]).sum() if len(report.trades) else 0
    tables = "".join(key_table(report.account, keys, title) for title, keys in ACCOUNT_KEYS.items())
    notes = "".join(
        f"<p class='muted'>{esc(n)}</p>"
        for n in report.notes
        if report.monte_carlo is not None or "Monte Carlo" not in n
    )
    trades_block = (
        f"<h3>Trade analysis</h3>{scroll(trade_summary_table(report))}{trade_chart(report)}"
        if closed
        else "<h3>Trade analysis</h3><p class='muted'>No closed trades.</p>"
    )
    return f"""
<details id="{anchor}"{" open" if is_open else ""}>
<summary>{esc(report.name)} <span class="muted">{meta["start"]:%Y-%m-%d} → {meta["end"]:%Y-%m-%d}
· {meta["bars"]:,} bars</span></summary>
<h3>Equity curve and drawdown</h3>
{equity_chart(report)}
<div class="cols">{tables}</div>
{notes}
{trades_block}
<h3>Annual analysis</h3>
<div class="scroll">{annual_table(report)}</div>
<h3>Monthly returns</h3>
<div class="scroll">{monthly_table(report)}</div>
{monte_carlo_section(report)}
<details class="inner"><summary>List of trades ({len(report.trades)})</summary>
{scroll(trade_rows(report)) if len(report.trades) else "<p class='muted'>No trades.</p>"}
</details>
</details>"""


OVERVIEW = [
    ("Net profit", lambda r: r.account["net_profit"], "usd"),
    ("CAGR", lambda r: r.account["cagr"], "pct"),
    ("Sharpe", lambda r: r.account["sharpe"], "ratio"),
    ("Max DD", lambda r: r.account["max_drawdown"], "pct"),
    ("Return on account", lambda r: r.account["return_on_account"], "ratio"),
    ("Profit factor", lambda r: r.trade_summary.loc["profit_factor", "All trades"], "ratio"),
    ("Trades", lambda r: r.trade_summary.loc["trades", "All trades"], "int"),
    ("% profitable", lambda r: r.trade_summary.loc["win_rate", "All trades"], "pct"),
    ("Avg trade", lambda r: r.trade_summary.loc["avg_trade", "All trades"], "pct2"),
    ("MC ruin", lambda r: r.monte_carlo and r.monte_carlo.table.loc[1.0, "risk_of_ruin"], "pct"),
    (
        "MC return/DD",
        lambda r: r.monte_carlo and r.monte_carlo.table.loc[1.0, "return_dd"],
        "ratio",
    ),
]


def davey_score(report: PerformanceReport) -> str:
    if report.monte_carlo is None:
        return "–"
    passed = sum(ok for *_, ok in report.monte_carlo.checks())
    return f'<span class="{"pos" if passed == 3 else "neg"}">{passed}/3</span>'


def overview_table(reports: list[PerformanceReport]) -> str:
    head = "".join(f"<th>{name}</th>" for name, *_ in OVERVIEW)
    rows = []
    for number, report in enumerate(reports):
        cells = []
        for name, getter, kind in OVERVIEW:
            value = getter(report)
            value = float("nan") if value is None else value
            cls = tone(value) if name in {"Net profit", "CAGR", "Avg trade"} else ""
            cells.append(f'<td class="{cls}">{fmt(value, kind)}</td>')
        rows.append(
            f'<tr><th><a href="#s{number}">{esc(report.name)}</a></th>{"".join(cells)}'
            f"<td>{davey_score(report)}</td></tr>"
        )
    return (
        f'<table class="grid"><tr><th>Market</th>{head}<th>Davey MC checks</th></tr>'
        f"{''.join(rows)}</table>"
    )


def summary_frame(reports: list[PerformanceReport]) -> pd.DataFrame:
    rows = {}
    for report in reports:
        row = {name: getter(report) for name, getter, _ in OVERVIEW}
        if report.monte_carlo is not None:
            row["davey_checks_passed"] = sum(ok for *_, ok in report.monte_carlo.checks())
        rows[report.name] = row
    return pd.DataFrame(rows).T


STYLE = """
:root { --fg:#1f2328; --muted:#656d76; --line:#d0d7de; --bg:#ffffff; --panel:#f6f8fa;
        --pos:#1a7f37; --neg:#cf222e; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 8px; }
h3 { font-size: 15px; margin: 22px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
code, pre { font: 12px/1.4 ui-monospace, Consolas, monospace; }
pre { background: var(--panel); padding: 8px 10px; border-radius: 6px; overflow-x: auto; }
.muted { color: var(--muted); font-size: 13px; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
img { max-width: 100%; height: auto; display: block; margin: 8px 0; }
table { border-collapse: collapse; margin: 6px 0 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 9px; border-bottom: 1px solid var(--line); text-align: right;
         white-space: nowrap; }
th:first-child { text-align: left; }
.grid th { background: var(--panel); font-weight: 600; }
.grid tr th:first-child { background: var(--panel); }
.kv { min-width: 330px; }
.kv th { font-weight: 500; text-align: left; color: var(--muted); }
.kv caption { text-align: left; font-weight: 600; padding: 4px 0; }
.cols { display: flex; flex-wrap: wrap; gap: 6px 28px; align-items: flex-start; }
.months td { min-width: 58px; }
.scroll { overflow-x: auto; }
details { border: 1px solid var(--line); border-radius: 8px; padding: 0 14px; margin: 12px 0; }
details > summary { cursor: pointer; padding: 10px 0; font-weight: 600; font-size: 16px; }
details.inner { border: none; padding: 0; }
details.inner > summary { font-size: 14px; }
.trades td { font-size: 12.5px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; margin: 6px 0; }
dt { color: var(--muted); } dd { margin: 0; }
"""

GLOSSARY = """
<h2>How to read this report</h2>
<ul class="muted">
<li><b>Trade figures</b> are a percentage of the equity at entry, because every trade is sized
from current equity and the account compounds. Money figures follow that compounded account.</li>
<li><b>Profit factor</b> = gross profit / gross loss; above 1 makes money after costs.
<b>Return on account</b> = net profit / largest money drawdown.</li>
<li><b>Sharpe</b> = mean / standard deviation of bar returns × √(bars per year), risk-free rate 0.
</li>
<li><b>Drawdowns</b> are measured close to close, so intrabar dips are not included.</li>
<li><b>Monte Carlo</b>: Davey's goals are risk of ruin under 10%, median max drawdown under 40% and
median return / median drawdown above 2. The larger and smaller sizes show how quickly risk grows
with position size.</li>
<li>Everything here is in-sample unless the date range was held out. A good report is a reason to
test further (walk-forward, other markets, incubation), not a reason to trade.</li>
</ul>"""


def render_html(reports: list[PerformanceReport], title: str, settings: dict) -> str:
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in settings.items())
    sections = "".join(
        symbol_section(report, f"s{number}", number == 0) for number, report in enumerate(reports)
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Performance Report</title><style>{STYLE}</style></head>
<body><main>
<h1>Strategy performance report</h1>
<p class="muted">{esc(title)} · generated {generated}</p>
<dl>{rows}</dl>
<h2>Overview</h2>
<div class="scroll">{overview_table(reports)}</div>
<h2>Markets</h2>
{sections}
{GLOSSARY}
</main></body></html>"""


def write_report(
    reports: list[PerformanceReport], folder: Path, title: str, settings: dict
) -> Path:
    """Write ``report.html``, ``summary.csv`` and one ``trades_<market>.csv`` per market."""

    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "report.html"
    page.write_text(render_html(reports, title, settings), encoding="utf-8")
    summary_frame(reports).to_csv(folder / "summary.csv")
    for report in reports:
        safe = "".join(ch if ch.isalnum() else "_" for ch in report.name)
        report.trades.to_csv(folder / f"trades_{safe}.csv", index=False)
    return page
