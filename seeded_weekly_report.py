"""Shared helpers for `notebooks/backtest_analysis_seeded_weekly_compare.ipynb`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from paths import DATA_DIR, TEST_RESULTS_DIR

DEFAULT_ASYNC_CSV = Path(TEST_RESULTS_DIR) / "seeded_weekly_full" / "seeded_weekly_async_by_ticker.csv"
WEEKLY_DATA = f"{DATA_DIR}/{{t}}_weekly_stock_data.csv"

_COLORS = {"dqn": "#4C72B0", "nollm": "#DD8452", "bah": "#55A868"}


def fmt_mean_std(mean: float, std: float, dm: int = 2, ds: int | None = None) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "-"
    ds = dm if ds is None else ds
    return f"{mean:.{dm}f} ± {std:.{ds}f}"


def winner_icon(
    dqn_val: float,
    nollm_val: float,
    bah_val: float | None = None,
    *,
    lower_is_better: bool = False,
) -> str:
    vals = {"DQN": dqn_val, "No-LLM": nollm_val}
    if bah_val is not None:
        vals["B&H"] = bah_val
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not vals:
        return "-"
    if lower_is_better:
        key = min(vals, key=vals.get)
    else:
        key = max(vals, key=vals.get)
    return key


def load_async_csv(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    wide = pd.read_csv(csv_path)
    summary = _wide_to_long(wide)
    bah_pct = dict(zip(wide["ticker"], wide["bah_pct"]))
    tickers = sorted(summary["ticker"].unique())
    return wide, summary, bah_pct, tickers


def _runs_for_row(wide: pd.DataFrame, row: pd.Series) -> int:
    if "runs" not in wide.columns or pd.isna(row["runs"]):
        return 5
    return int(row["runs"])


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    nan = np.nan
    specs = (("dqn", "dqn"), ("dqn_nollm", "nollm"))
    for _, r in wide.iterrows():
        t, n = r["ticker"], _runs_for_row(wide, r)
        base = dict(ticker=t, runs=n, final_mean=nan, final_std=nan, win_rate_mean=nan, win_rate_std=nan)
        for method, pfx in specs:
            rows.append(
                {
                    **base,
                    "method": method,
                    "return_mean_pct": r[f"{pfx}_return_mean"],
                    "return_std_pct": r[f"{pfx}_return_std"],
                    "sharpe_mean": r[f"{pfx}_sharpe_mean"],
                    "sharpe_std": r[f"{pfx}_sharpe_std"],
                    "max_drawdown_mean": r[f"{pfx}_maxdd_mean"],
                    "max_drawdown_std": r[f"{pfx}_maxdd_std"],
                    "return_max_pct": r[f"{pfx}_max_return"],
                }
            )
    return pd.DataFrame(rows)


def bah_preview(
    tickers: Sequence[str],
    bah_pct: dict[str, float],
    *,
    start_date: str = "2024-01-01",
    template: str = WEEKLY_DATA,
) -> pd.DataFrame:
    out: list[dict] = []
    for t in tickers:
        path = template.format(t=t)
        if not os.path.exists(path):
            out.append(
                dict(
                    ticker=t,
                    start_date="N/A",
                    end_date="N/A",
                    start_price=np.nan,
                    end_price=np.nan,
                    ret=bah_pct[t],
                )
            )
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[df.index >= start_date].sort_index().dropna(subset=["adjusted_close"])
        p0, p1 = df["adjusted_close"].iloc[0], df["adjusted_close"].iloc[-1]
        out.append(
            dict(
                ticker=t,
                start_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
                start_price=round(float(p0), 2),
                end_price=round(float(p1), 2),
                ret=bah_pct[t],
            )
        )
    return pd.DataFrame(out)


def split_methods(summary: pd.DataFrame, tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tix = list(tickers)
    dqn = summary[summary["method"] == "dqn"].set_index("ticker").loc[tix]
    nollm = summary[summary["method"] == "dqn_nollm"].set_index("ticker").loc[tix]
    return dqn, nollm


def comparison_table(
    summary: pd.DataFrame,
    bah_df: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    s = summary.set_index(["ticker", "method"])
    bah_ix = bah_df.set_index("ticker")["ret"].astype(float)
    rows = []
    for t in tickers:
        d = s.loc[(t, "dqn")]
        nl = s.loc[(t, "dqn_nollm")]
        bh = bah_ix.loc[t]
        wr_ok = pd.notna(d["win_rate_mean"]) or pd.notna(nl["win_rate_mean"])
        wr_star = winner_icon(d["win_rate_mean"], nl["win_rate_mean"]) if wr_ok else "-"
        rows.append(
            {
                "Ticker": t,
                "B&H Return %": f"{bh:.2f}%",
                "DQN Return %": fmt_mean_std(d["return_mean_pct"], d["return_std_pct"], 2, 2) + "%",
                "No-LLM Return %": fmt_mean_std(nl["return_mean_pct"], nl["return_std_pct"], 2, 2) + "%",
                "* Return": winner_icon(d["return_mean_pct"], nl["return_mean_pct"], bh),
                "DQN Sharpe": fmt_mean_std(d["sharpe_mean"], d["sharpe_std"], 2, 2),
                "No-LLM Sharpe": fmt_mean_std(nl["sharpe_mean"], nl["sharpe_std"], 2, 2),
                "* Sharpe": winner_icon(d["sharpe_mean"], nl["sharpe_mean"]),
                "DQN MaxDD %": fmt_mean_std(d["max_drawdown_mean"], d["max_drawdown_std"], 1, 1),
                "No-LLM MaxDD %": fmt_mean_std(nl["max_drawdown_mean"], nl["max_drawdown_std"], 1, 1),
                "* MaxDD": winner_icon(
                    d["max_drawdown_mean"], nl["max_drawdown_mean"], lower_is_better=True
                ),
                "DQN Win Rate %": fmt_mean_std(d["win_rate_mean"], d["win_rate_std"]),
                "No-LLM Win Rate %": fmt_mean_std(nl["win_rate_mean"], nl["win_rate_std"]),
                "* Win Rate": wr_star,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def plot_panels(
    dqn: pd.DataFrame,
    nollm: pd.DataFrame,
    bah_ret: Sequence[float],
    tickers: Sequence[str],
    *,
    title: str = "Seeded Weekly Comparison by Ticker",
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(tickers))
    w = 0.28
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    panels = [
        ("Total Return %", "return_mean_pct", "return_std_pct", False),
        ("Sharpe Ratio", "sharpe_mean", "sharpe_std", False),
        ("Max Drawdown %", "max_drawdown_mean", "max_drawdown_std", True),
    ]
    for ax, (ptitle, col, err, invert_y) in zip(axes, panels):
        ax.bar(x - w, dqn[col], w, yerr=dqn[err], capsize=3, label="DQN (LLM)", color=_COLORS["dqn"])
        ax.bar(x, nollm[col], w, yerr=nollm[err], capsize=3, label="No-LLM", color=_COLORS["nollm"])
        if ptitle == "Total Return %":
            ax.bar(x + w, bah_ret, w, label="Buy & Hold", color=_COLORS["bah"], alpha=0.85)
        ax.set_title(ptitle + ("  (lower = better)" if invert_y else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(fontsize=8)
        if invert_y:
            ax.invert_yaxis()
    plt.tight_layout()


def print_scorecard(
    dqn: pd.DataFrame,
    nollm: pd.DataFrame,
    bah_ix: pd.Series,
    tickers: Sequence[str],
) -> None:
    n = len(tickers)
    wins_ret = sum(dqn.loc[t, "return_mean_pct"] > nollm.loc[t, "return_mean_pct"] for t in tickers)
    wins_sr = sum(dqn.loc[t, "sharpe_mean"] > nollm.loc[t, "sharpe_mean"] for t in tickers)
    wins_wr = sum(
        pd.notna(dqn.loc[t, "win_rate_mean"])
        and pd.notna(nollm.loc[t, "win_rate_mean"])
        and dqn.loc[t, "win_rate_mean"] > nollm.loc[t, "win_rate_mean"]
        for t in tickers
    )
    wins_dd = sum(dqn.loc[t, "max_drawdown_mean"] < nollm.loc[t, "max_drawdown_mean"] for t in tickers)

    print("=== Score Card ===\n")
    if dqn["win_rate_mean"].notna().any():
        print(
            f"DQN (LLM) beats No-LLM | Return: {wins_ret}/{n} | Sharpe: {wins_sr}/{n} "
            f"| Win Rate: {wins_wr}/{n} | Lower MaxDD: {wins_dd}/{n}"
        )
    else:
        print(
            f"DQN (LLM) beats No-LLM | Return: {wins_ret}/{n} | Sharpe: {wins_sr}/{n} "
            f"| Lower MaxDD: {wins_dd}/{n}"
        )
        print("(Win rate not in CSV export.)")

    d_bah = sum(dqn.loc[t, "return_mean_pct"] > bah_ix.loc[t] for t in tickers)
    nl_bah = sum(nollm.loc[t, "return_mean_pct"] > bah_ix.loc[t] for t in tickers)
    print(f"Beats Buy & Hold | DQN: {d_bah}/{n} | No-LLM: {nl_bah}/{n}")
    print("\nAverages across all tickers:")
    print(f"  Sharpe: DQN {dqn['sharpe_mean'].mean():.2f} | No-LLM {nollm['sharpe_mean'].mean():.2f}")
    print(f"  MaxDD %: DQN {dqn['max_drawdown_mean'].mean():.1f} | No-LLM {nollm['max_drawdown_mean'].mean():.1f}")
    if dqn["win_rate_mean"].notna().any():
        print(
            f"  Win Rate%: DQN {dqn['win_rate_mean'].mean():.1f} "
            f"| No-LLM {nollm['win_rate_mean'].mean():.1f}"
        )


def max_return_table(summary: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    p = summary.pivot(index="ticker", columns="method", values="return_max_pct").loc[list(tickers)]
    p = p.rename(columns={"dqn": "DQN (LLM) max%", "dqn_nollm": "No-LLM max%"})
    p["Winner"] = np.where(p["DQN (LLM) max%"] > p["No-LLM max%"], "DQN (LLM)", "No-LLM")
    gap = (p["DQN (LLM) max%"] - p["No-LLM max%"]).round(1)
    p["Gap"] = gap.apply(lambda v: f"+{v}" if v > 0 else str(v))
    return p.round(1)
