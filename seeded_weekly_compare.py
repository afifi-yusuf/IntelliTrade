#!/usr/bin/env python3
"""
Seeded weekly benchmark:
- Train 5 seeded runs for DQN and DQN no-LLM
- Evaluate on selected tickers with 10% stop-loss (inference-only)
- Save raw + aggregated CSVs
- Generate a comparison notebook (mean/std per ticker)
"""

import argparse
import contextlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from data import TICKERS
from run import run_backtest
from train import train_agent


DEFAULT_SEEDS = [11, 22, 33, 44, 55]
DEFAULT_TICKERS = [t for t in TICKERS if t != "SPY"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def parse_seed_list(seed_text: str) -> List[int]:
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def weekly_final_model_path(method: str) -> str:
    if method == "dqn":
        return "backtest_results/dqn_model_walkforward_final.pth"
    if method == "dqn_nollm":
        return "backtest_results/dqn_nollm_model_walkforward_final.pth"
    raise ValueError(f"Unsupported method: {method}")


def build_notebook(
    notebook_path: Path,
    summary_csv: Path,
    raw_csv: Path,
    tickers: List[str],
) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Seeded Weekly Comparison (DQN vs DQN No-LLM)\n",
                "\n",
                "This notebook summarizes 5 seeded weekly training runs with 10% stop-loss at inference.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                f"summary_path = '{summary_csv.as_posix()}'\n",
                f"raw_path = '{raw_csv.as_posix()}'\n",
                "summary = pd.read_csv(summary_path)\n",
                "raw = pd.read_csv(raw_path)\n",
                "summary.head()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def fmt(mean, std):\n",
                "    return f\"{mean:.2f} ± {std:.2f}\"\n",
                "\n",
                "ret = summary.pivot(index='ticker', columns='method', values=['return_mean_pct', 'return_std_pct'])\n",
                "fin = summary.pivot(index='ticker', columns='method', values=['final_mean', 'final_std'])\n",
                "\n",
                "rows = []\n",
                "for ticker in sorted(summary['ticker'].unique()):\n",
                "    row = {\n",
                "        'Ticker': ticker,\n",
                "        'DQN Return % (mean±std)': fmt(ret.loc[ticker, ('return_mean_pct', 'dqn')], ret.loc[ticker, ('return_std_pct', 'dqn')]),\n",
                "        'DQN No-LLM Return % (mean±std)': fmt(ret.loc[ticker, ('return_mean_pct', 'dqn_nollm')], ret.loc[ticker, ('return_std_pct', 'dqn_nollm')]),\n",
                "        'DQN Final $ (mean±std)': fmt(fin.loc[ticker, ('final_mean', 'dqn')], fin.loc[ticker, ('final_std', 'dqn')]),\n",
                "        'DQN No-LLM Final $ (mean±std)': fmt(fin.loc[ticker, ('final_mean', 'dqn_nollm')], fin.loc[ticker, ('final_std', 'dqn_nollm')]),\n",
                "    }\n",
                "    rows.append(row)\n",
                "display(pd.DataFrame(rows))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tickers = sorted(summary['ticker'].unique())\n",
                "x = np.arange(len(tickers))\n",
                "width = 0.38\n",
                "\n",
                "dqn = summary[summary['method']=='dqn'].set_index('ticker').loc[tickers]\n",
                "nollm = summary[summary['method']=='dqn_nollm'].set_index('ticker').loc[tickers]\n",
                "\n",
                "plt.figure(figsize=(14, 5))\n",
                "plt.bar(x - width/2, dqn['return_mean_pct'], width, yerr=dqn['return_std_pct'], capsize=4, label='DQN')\n",
                "plt.bar(x + width/2, nollm['return_mean_pct'], width, yerr=nollm['return_std_pct'], capsize=4, label='DQN No-LLM')\n",
                "plt.xticks(x, tickers, rotation=45)\n",
                "plt.ylabel('Return % (mean ± std)')\n",
                "plt.title('Weekly Seeded Comparison by Ticker')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded weekly benchmark for DQN vs DQN no-LLM")
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55", help="Comma-separated seed list")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per run")
    parser.add_argument("--cutoff", type=str, default="2024-01-01", help="Training cutoff date")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--stop-loss-pct", type=float, default=0.10, help="Inference stop loss pct")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Inference start date")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Output directory")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("backtest_results/seed_runs")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "seeded_weekly_compare.log"

    print(f"Seeds: {seeds}")
    print(f"Tickers: {tickers}")
    print(f"Stop-Loss: {args.stop_loss_pct:.2%}")

    records = []
    methods = ["dqn", "dqn_nollm"]

    for method in methods:
        for seed in seeds:
            print(f"\n=== Training {method} | seed={seed} | weekly ===")
            set_global_seed(seed)
            train_agent(
                method=method,
                epochs=args.epochs,
                train_cutoff_date=args.cutoff,
                rebalance_frequency="weekly",
                seed=seed,
            )

            src = Path(weekly_final_model_path(method))
            if not src.exists():
                raise FileNotFoundError(f"Expected model not found: {src}")
            dst = model_dir / f"{method}_weekly_seed_{seed}.pth"
            shutil.copy2(src, dst)
            print(f"Saved seeded model: {dst.as_posix()}")

            for ticker in tickers:
                print(f"  Backtest {method} seed={seed} ticker={ticker}")
                with open(log_path, "a", encoding="utf-8") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                    metrics = run_backtest(
                        ticker=ticker,
                        method=method,
                        initial_capital=10000.0,
                        model_path=dst.as_posix(),
                        enable_learning=False,
                        save_model=False,
                        use_confidence_sizing=True,
                        stop_loss_pct=args.stop_loss_pct,
                        start_date=args.start_date,
                        rebalance_frequency="weekly",
                        read_only=True,
                    )

                records.append(
                    {
                        "seed": seed,
                        "method": method,
                        "ticker": ticker,
                        "final_portfolio_value": metrics.final_portfolio_value,
                        "total_return_pct": metrics.total_return_pct,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "win_rate": metrics.win_rate,
                        "max_drawdown_pct": metrics.max_drawdown_pct,
                        "total_trades": metrics.total_trades,
                    }
                )
                print(
                    f"    -> return={metrics.total_return_pct:+.2f}% "
                    f"final=${metrics.final_portfolio_value:,.2f}"
                )

    raw_df = pd.DataFrame(records)
    raw_csv = output_dir / "seeded_weekly_runs.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_df = (
        raw_df.groupby(["ticker", "method"], as_index=False)
        .agg(
            runs=("seed", "count"),
            return_mean_pct=("total_return_pct", "mean"),
            return_std_pct=("total_return_pct", "std"),
            final_mean=("final_portfolio_value", "mean"),
            final_std=("final_portfolio_value", "std"),
            sharpe_mean=("sharpe_ratio", "mean"),
            sharpe_std=("sharpe_ratio", "std"),
            win_rate_mean=("win_rate", "mean"),
            win_rate_std=("win_rate", "std"),
            max_drawdown_mean=("max_drawdown_pct", "mean"),
            max_drawdown_std=("max_drawdown_pct", "std"),
        )
        .sort_values(["ticker", "method"])
    )
    summary_csv = output_dir / "seeded_weekly_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    comparison_rows = []
    for ticker in sorted(summary_df["ticker"].unique()):
        dqn_row = summary_df[(summary_df["ticker"] == ticker) & (summary_df["method"] == "dqn")]
        nollm_row = summary_df[(summary_df["ticker"] == ticker) & (summary_df["method"] == "dqn_nollm")]
        if dqn_row.empty or nollm_row.empty:
            continue
        dqn_ret = float(dqn_row["return_mean_pct"].iloc[0])
        nollm_ret = float(nollm_row["return_mean_pct"].iloc[0])
        comparison_rows.append(
            {
                "ticker": ticker,
                "dqn_return_mean_pct": dqn_ret,
                "dqn_nollm_return_mean_pct": nollm_ret,
                "llm_minus_nollm_return_mean_pct": dqn_ret - nollm_ret,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("ticker")
    comparison_csv = output_dir / "seeded_weekly_delta.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    notebook_path = Path("backtest_analysis_seeded_weekly_compare.ipynb")
    build_notebook(
        notebook_path=notebook_path,
        summary_csv=summary_csv,
        raw_csv=raw_csv,
        tickers=tickers,
    )

    print("\nDone.")
    print(f"Raw runs: {raw_csv.as_posix()}")
    print(f"Summary: {summary_csv.as_posix()}")
    print(f"Delta: {comparison_csv.as_posix()}")
    print(f"Notebook: {notebook_path.as_posix()}")


if __name__ == "__main__":
    main()
