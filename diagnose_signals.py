#!/usr/bin/env python3
"""
Signal Quality Diagnostic
=========================
Queries Neo4j for all stored LLM structured signals, cross-references them
with actual next-quarter price returns, and computes:
  - Signal distribution (are they clustered near zero?)
  - Directional accuracy (does positive sentiment → positive next-quarter return?)
  - Per-signal Pearson correlation with next-quarter return
  - Per-ticker signal history

Run from project root:
    ./venv/bin/python3 diagnose_signals.py
"""

import json
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from graph_manager import Neo4jHandler

# ── Price data ────────────────────────────────────────────────────────────────

def load_quarterly_returns() -> dict:
    """Load quarterly price returns from local CSV files."""
    returns = {}
    data_dir = "data"
    from data import TICKERS
    for ticker in [t for t in TICKERS if t != "SPY"]:
        for fname in [
            os.path.join(data_dir, f"{ticker}_daily_stock_data_5y.csv"),
            os.path.join(data_dir, f"{ticker.lower()}_daily_stock_data_5y.csv"),
        ]:
            if os.path.exists(fname):
                df = pd.read_csv(fname, index_col=0)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna(subset=["close"])
                df["quarter"] = df.index.to_period("Q")
                q_close = df.groupby("quarter")["close"].last()
                q_ret = q_close.pct_change().dropna() * 100
                # Clamp extreme values (stock splits create spurious -90%+ returns)
                q_ret = q_ret.clip(-50, 200)
                # Convert Period keys to "YYYY-Qn" strings to match Neo4j quarter IDs
                returns[ticker] = {
                    f"{p.year}-Q{p.quarter}": v for p, v in q_ret.items()
                }
                break
    return returns


def quarter_id_to_period(quarter_id: str) -> pd.Period:
    """Convert 'YYYY-Qn' (Neo4j format) → pandas Period."""
    year, q = quarter_id.split("-")
    return pd.Period(f"{year}Q{q[1]}", freq="Q")


def next_quarter_id(quarter_id: str) -> str:
    """Return the next quarter as a 'YYYY-Qn' string (matches Neo4j / price-return keys)."""
    period = quarter_id_to_period(quarter_id)
    nxt = period + 1
    return f"{nxt.year}-Q{nxt.quarter}"


# ── Signal query ──────────────────────────────────────────────────────────────

SIGNAL_KEYS = [
    "revenue_trend", "margin_trend", "peer_relative",
    "historical_momentum", "risk_level", "overall_sentiment",
]

def fetch_signals(gdb: Neo4jHandler) -> pd.DataFrame:
    query = """
    MATCH (t:Ticker)-[:HAS_SUMMARY]->(s:Summary)
    WHERE s.structured_signals_json IS NOT NULL
      AND s.structured_signals_json <> '{}'
    RETURN t.symbol              AS ticker,
           s.source_quarter      AS quarter,
           s.structured_signals_json AS signals_json,
           s.reliability_json    AS reliability_json
    ORDER BY t.symbol, s.source_quarter
    """
    rows = gdb._run_query(query) or []
    records = []
    for r in rows:
        try:
            sig = json.loads(r["signals_json"] or "{}")
            rel = json.loads(r["reliability_json"] or "{}")
        except Exception:
            continue
        rec = {
            "ticker":  r["ticker"],
            "quarter": r["quarter"],
            "evidence_count": rel.get("evidence_count", 0),
            "agent_agreement": rel.get("agent_agreement", 0.5),
            "parse_success":   rel.get("parse_success", 0.0),
        }
        for k in SIGNAL_KEYS:
            rec[k] = sig.get(k, 0.0)
        records.append(rec)
    return pd.DataFrame(records)


# ── Analysis ──────────────────────────────────────────────────────────────────

SEP = "─" * 72

def run():
    print(SEP)
    print("  IntelliSense — LLM Signal Quality Diagnostic")
    print(SEP)

    gdb = Neo4jHandler()
    if not gdb.driver:
        print("❌  Neo4j not connected. Check .env credentials.")
        return

    print("⏳  Fetching signals from Neo4j …")
    df = fetch_signals(gdb)
    if df.empty:
        print("❌  No Summary nodes with structured signals found.")
        print("    Run a full DQN backtest first to populate the cache.")
        return

    print(f"✅  Found {len(df)} signal records across "
          f"{df['ticker'].nunique()} tickers, "
          f"{df['quarter'].nunique()} quarters\n")

    q_returns = load_quarterly_returns()

    # Map next-quarter return onto each signal row
    def get_next_ret(row):
        ticker = row["ticker"]
        if ticker not in q_returns:
            return np.nan
        nxt = next_quarter_id(row["quarter"])  # e.g. "2024-Q1"
        return q_returns[ticker].get(nxt, np.nan)

    df["next_q_return"] = df.apply(get_next_ret, axis=1)
    df_valid = df.dropna(subset=["next_q_return"])

    # ── 1. Signal distribution ────────────────────────────────────────────────
    print(SEP)
    print("  1. SIGNAL DISTRIBUTION  (are signals actually non-zero?)")
    print(SEP)
    stats = df[SIGNAL_KEYS].describe().T[["mean", "std", "min", "max"]]
    # risk_level lives in [0,1] where 0.5 = neutral; all others use [-1,1] where 0 = neutral
    near_neutral = {}
    for k in SIGNAL_KEYS:
        if k == "risk_level":
            near_neutral[k] = ((df[k] - 0.5).abs() < 0.1).mean() * 100
        else:
            near_neutral[k] = (df[k].abs() < 0.05).mean() * 100
    stats["near_neutral%"] = pd.Series(near_neutral)
    print(stats.round(3).to_string())

    if stats["near_neutral%"].mean() > 50:
        print("\n⚠️  WARNING: >50% of signal values are near-neutral — "
              "the synthesiser is not producing discriminatory signals.")

    # ── 2. Per-signal correlation with next-quarter return ────────────────────
    print(f"\n{SEP}")
    print("  2. PEARSON CORRELATION: signal → next-quarter return")
    print(f"     (n={len(df_valid)} valid pairs)")
    print(SEP)
    for k in SIGNAL_KEYS:
        corr = np.corrcoef(df_valid[k], df_valid["next_q_return"])[0, 1]
        direction = "✅ correct" if abs(corr) > 0.15 else ("⚠️  weak" if abs(corr) > 0.05 else "❌ none")
        print(f"  {k:<24}  r = {corr:+.3f}  {direction}")

    # ── 3. Directional accuracy of overall_sentiment ──────────────────────────
    print(f"\n{SEP}")
    print("  3. DIRECTIONAL ACCURACY of overall_sentiment")
    print(SEP)
    df_dir = df_valid[df_valid["overall_sentiment"].abs() > 0.1].copy()
    df_dir["predicted_up"] = df_dir["overall_sentiment"] > 0
    df_dir["actual_up"]    = df_dir["next_q_return"]    > 0
    accuracy = (df_dir["predicted_up"] == df_dir["actual_up"]).mean()
    n_dir = len(df_dir)
    print(f"  Rows with |sentiment| > 0.1:  {n_dir}")
    print(f"  Directional accuracy:          {accuracy*100:.1f}%  "
          f"({'above chance' if accuracy > 0.55 else 'near/below random chance (50%)'})")

    # Breakdown by positive vs negative calls
    pos_calls = df_dir[df_dir["predicted_up"]]
    neg_calls  = df_dir[~df_dir["predicted_up"]]
    if len(pos_calls):
        pos_acc = (pos_calls["actual_up"]).mean()
        print(f"  When sentiment > 0 ({len(pos_calls)} calls): market went up {pos_acc*100:.1f}% of the time")
    if len(neg_calls):
        neg_acc = (~neg_calls["actual_up"]).mean()
        print(f"  When sentiment < 0 ({len(neg_calls)} calls): market went down {neg_acc*100:.1f}% of the time")

    # ── 4. Per-ticker signal history ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  4. PER-TICKER SIGNAL HISTORY  (sentiment | revenue | risk | next_q_ret)")
    print(SEP)
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker].sort_values("quarter")
        print(f"\n  {ticker}")
        print(f"  {'Quarter':<10} {'sentiment':>10} {'revenue':>9} {'margin':>8} "
              f"{'peer':>7} {'risk':>6} {'evidence':>9} {'next_q_ret':>11}")
        for _, r in sub.iterrows():
            nqr = df_valid.loc[
                (df_valid["ticker"] == ticker) & (df_valid["quarter"] == r["quarter"]),
                "next_q_return"
            ]
            nqr_str = f"{nqr.values[0]:+7.1f}%" if len(nqr) else "    n/a"
            # Flag mismatch: sentiment positive but return negative (or vice versa)
            flag = ""
            if len(nqr) and abs(r["overall_sentiment"]) > 0.1:
                if (r["overall_sentiment"] > 0) != (nqr.values[0] > 0):
                    flag = " ◀ contrarian"
            print(f"  {r['quarter']:<10} {r['overall_sentiment']:+10.3f} "
                  f"{r['revenue_trend']:+9.3f} {r['margin_trend']:+8.3f} "
                  f"{r['peer_relative']:+7.3f} {r['risk_level']:6.3f} "
                  f"{int(r['evidence_count']):9d} {nqr_str:>11}{flag}")

    # ── 5. Reliability quality ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  5. RELIABILITY / EVIDENCE QUALITY")
    print(SEP)
    print(f"  Mean evidence count:   {df['evidence_count'].mean():.1f}  "
          f"(target: ≥15 facts per quarter)")
    print(f"  Mean agent agreement:  {df['agent_agreement'].mean():.3f}  "
          f"(1.0 = all agents agree)")
    print(f"  Mean parse success:    {df['parse_success'].mean():.3f}  "
          f"(1.0 = synthesiser produced valid output)")
    low_evidence = (df["evidence_count"] < 5).mean() * 100
    print(f"  Quarters with <5 facts: {low_evidence:.1f}%")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    run()
