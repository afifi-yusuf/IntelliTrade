# NVIDIA Data Collection Guide

## 📋 Overview

The `data.py` script collects **comprehensive historical data** for NVIDIA (NVDA) from Alpha Vantage Premium API and saves everything as CSV files for offline analysis.

## 🚀 Quick Start

1. **Install dependencies:**
```bash
cd /Users/yusufafifi/Desktop/IntelliSense
source venv/bin/activate
pip install pandas
```

2. **Ensure your `.env` file has:**
```bash
ALPHA_VANTAGE_API_KEY=your_premium_api_key_here
```

3. **Run the script:**
```bash
python data.py
```

4. **Find your data:**
All CSV files will be in the `data/` directory

## 📊 Data Categories Collected

### 1️⃣ Stock Price Data (Multiple Timeframes)
- ✅ **Daily OHLCV** - 5 years of adjusted daily prices
- ✅ **Weekly OHLCV** - Full history, weekly aggregation
- ✅ **Monthly OHLCV** - Full history, monthly aggregation
- ✅ **Latest Quote** - Real-time snapshot

**Files:** `NVDA_daily_stock_data_5y.csv`, `NVDA_weekly_stock_data.csv`, `NVDA_monthly_stock_data.csv`, `NVDA_latest_quote.csv`

---

### 2️⃣ Fundamental Data
- ✅ **Company Overview** - Metrics, ratios, valuation multiples
- ✅ **Income Statements** - Annual & quarterly revenue, expenses, profitability
- ✅ **Balance Sheets** - Annual & quarterly assets, liabilities, equity
- ✅ **Cash Flow Statements** - Annual & quarterly operating, investing, financing cash flows
- ✅ **Shares Outstanding** - Historical share count data

**Files:** `NVDA_company_overview.csv`, `NVDA_income_statement_annual.csv`, `NVDA_income_statement_quarterly.csv`, `NVDA_balance_sheet_annual.csv`, `NVDA_balance_sheet_quarterly.csv`, `NVDA_cash_flow_annual.csv`, `NVDA_cash_flow_quarterly.csv`, `NVDA_shares_outstanding.csv`

---

### 3️⃣ Earnings Data
- ✅ **Historical Earnings** - Annual & quarterly EPS with beat/miss data
- ✅ **Earnings Estimates** - Forward-looking analyst estimates
- ✅ **Earnings Calendar** - Upcoming earnings dates

**Files:** `NVDA_earnings_annual.csv`, `NVDA_earnings_quarterly.csv`, `NVDA_earnings_estimates.csv`, `NVDA_earnings_calendar.csv`

---

### 4️⃣ Earnings Call Transcripts (Premium) 🌟
- ✅ **Last 20 Quarters (5 Years)** - Full transcripts of earnings calls (2024-2019)

**Files:** `NVDA_earnings_transcript_2024_Q4.txt`, `NVDA_earnings_transcript_2024_Q3.txt`, `NVDA_earnings_transcript_2024_Q2.txt`, etc. (up to 20 quarterly transcripts)

---

### 5️⃣ Corporate Actions
- ✅ **Dividend History** - All dividend payments (if any)
- ✅ **Stock Split History** - Historical stock splits

**Files:** `NVDA_dividends.csv`, `NVDA_stock_splits.csv`

---

### 6️⃣ News & Sentiment (Premium) 🌟
- ✅ **Up to 1000 Articles** - Recent news with sentiment analysis
- ✅ **Sentiment Scores** - Overall + NVDA-specific sentiment
- ✅ **Relevance Scores** - How relevant each article is to NVDA

**Files:** `NVDA_news_sentiment.csv`

---

### 7️⃣ Insider Transactions (Premium) 🌟
- ✅ **Insider Trading Activity** - Buying/selling by executives and board members

**Files:** `NVDA_insider_transactions.csv`

---

### 8️⃣ Technical Indicators (5 Years, Daily)
**Moving Averages:**
- ✅ SMA (50-day, 200-day)
- ✅ EMA (12-day, 26-day)

**Momentum Indicators:**
- ✅ RSI (Relative Strength Index)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ STOCH (Stochastic Oscillator)
- ✅ CCI (Commodity Channel Index)

**Volatility Indicators:**
- ✅ Bollinger Bands
- ✅ ATR (Average True Range)

**Trend Indicators:**
- ✅ ADX (Average Directional Index)
- ✅ AROON

**Volume Indicators:**
- ✅ OBV (On Balance Volume)
- ✅ AD (Accumulation/Distribution)

**Premium Indicators:**
- ✅ VWAP (Volume Weighted Average Price) 🌟

**Files:** `NVDA_sma_50.csv`, `NVDA_sma_200.csv`, `NVDA_ema_12.csv`, `NVDA_ema_26.csv`, `NVDA_rsi.csv`, `NVDA_macd.csv`, `NVDA_stoch.csv`, `NVDA_cci.csv`, `NVDA_bbands.csv`, `NVDA_atr.csv`, `NVDA_adx.csv`, `NVDA_aroon.csv`, `NVDA_obv.csv`, `NVDA_ad.csv`, `NVDA_vwap.csv`

---

### 9️⃣ Economic Indicators (Federal Reserve Data)
**Interest Rates:**
- ✅ Federal Funds Rate
- ✅ 10-Year Treasury Yield

**GDP & Growth:**
- ✅ Real GDP
- ✅ Real GDP per Capita

**Inflation:**
- ✅ CPI (Consumer Price Index)
- ✅ Inflation Rate

**Employment:**
- ✅ Unemployment Rate
- ✅ Nonfarm Payroll

**Consumer & Business Activity:**
- ✅ Retail Sales
- ✅ Durable Goods Orders

**Files:** `economic_federal_funds_rate.csv`, `economic_treasury_yield_10y.csv`, `economic_real_gdp.csv`, `economic_real_gdp_per_capita.csv`, `economic_cpi.csv`, `economic_inflation.csv`, `economic_unemployment.csv`, `economic_nonfarm_payroll.csv`, `economic_retail_sales.csv`, `economic_durable_goods_orders.csv`

---

### 🔟 Analytics (Premium) 🌟
- ✅ **Fixed Window Analysis** - Statistical analysis over full history
- ✅ **Sliding Window Analysis** - 20-day rolling statistics

**Files:** `NVDA_analytics_fixed_window.csv`, `NVDA_analytics_sliding_window.csv`

---

### 1️⃣1️⃣ Market Context
- ✅ **Top Gainers** - Biggest winners in the market
- ✅ **Top Losers** - Biggest losers in the market
- ✅ **Most Actively Traded** - Highest volume stocks
- ✅ **ETF Holdings** - Which major ETFs hold NVDA

**Files:** `market_top_gainers.csv`, `market_top_losers.csv`, `market_most_actively_traded.csv`, `NVDA_etf_holdings.csv`

---

## 🎯 What Makes This Comprehensive

### Comparison to Original Request:
| Data Type | Requested | Included | Notes |
|-----------|-----------|----------|-------|
| Daily Stock History (5 years) | ✅ | ✅ | Plus weekly & monthly |
| Fed Data | ✅ | ✅ | 10 economic indicators |
| Earnings Data | ✅ | ✅ | History + estimates + calendar |
| **Earnings Call Transcripts** | ✅ | ✅ | Last 8 quarters |
| News & Sentiment | ✅ | ✅ | Up to 1000 articles |
| Insider Transactions | ✅ | ✅ | Full history |
| Fundamental Data | ✅ | ✅ | Income, balance, cash flow statements |
| Technical Indicators | ✅ | ✅ | 15 different indicators |
| Economic Data | ✅ | ✅ | 10 Fed indicators |

### Added Bonuses:
- ✅ Weekly & monthly price data (not just daily)
- ✅ Shares outstanding history
- ✅ Corporate actions (dividends & splits)
- ✅ Earnings estimates (forward-looking)
- ✅ Latest real-time quote
- ✅ Advanced analytics (fixed & sliding window)
- ✅ Market context (top gainers/losers)
- ✅ ETF holdings information

---

## 📁 Expected Output Structure

```
data/
├── Stock Price Data (4 files)
│   ├── NVDA_daily_stock_data_5y.csv
│   ├── NVDA_weekly_stock_data.csv
│   ├── NVDA_monthly_stock_data.csv
│   └── NVDA_latest_quote.csv
│
├── Fundamental Data (8 files)
│   ├── NVDA_company_overview.csv
│   ├── NVDA_income_statement_annual.csv
│   ├── NVDA_income_statement_quarterly.csv
│   ├── NVDA_balance_sheet_annual.csv
│   ├── NVDA_balance_sheet_quarterly.csv
│   ├── NVDA_cash_flow_annual.csv
│   ├── NVDA_cash_flow_quarterly.csv
│   └── NVDA_shares_outstanding.csv
│
├── Earnings Data (4 files)
│   ├── NVDA_earnings_annual.csv
│   ├── NVDA_earnings_quarterly.csv
│   ├── NVDA_earnings_estimates.csv
│   └── NVDA_earnings_calendar.csv
│
├── Earnings Transcripts (up to 20 files)
│   ├── NVDA_earnings_transcript_2024_Q4.txt
│   ├── NVDA_earnings_transcript_2024_Q3.txt
│   ├── NVDA_earnings_transcript_2024_Q2.txt
│   ├── NVDA_earnings_transcript_2024_Q1.txt
│   └── ... (up to 16 more quarters going back 5 years)
│
├── Corporate Actions (2 files)
│   ├── NVDA_dividends.csv
│   └── NVDA_stock_splits.csv
│
├── Market Intelligence (2 files)
│   ├── NVDA_news_sentiment.csv
│   └── NVDA_insider_transactions.csv
│
├── Technical Indicators (15 files)
│   ├── NVDA_sma_50.csv
│   ├── NVDA_sma_200.csv
│   ├── NVDA_ema_12.csv
│   ├── NVDA_ema_26.csv
│   ├── NVDA_rsi.csv
│   ├── NVDA_macd.csv
│   ├── NVDA_stoch.csv
│   ├── NVDA_cci.csv
│   ├── NVDA_bbands.csv
│   ├── NVDA_atr.csv
│   ├── NVDA_adx.csv
│   ├── NVDA_aroon.csv
│   ├── NVDA_obv.csv
│   ├── NVDA_ad.csv
│   └── NVDA_vwap.csv
│
├── Economic Indicators (10 files)
│   ├── economic_federal_funds_rate.csv
│   ├── economic_treasury_yield_10y.csv
│   ├── economic_real_gdp.csv
│   ├── economic_real_gdp_per_capita.csv
│   ├── economic_cpi.csv
│   ├── economic_inflation.csv
│   ├── economic_unemployment.csv
│   ├── economic_nonfarm_payroll.csv
│   ├── economic_retail_sales.csv
│   └── economic_durable_goods_orders.csv
│
├── Analytics (2 files)
│   ├── NVDA_analytics_fixed_window.csv
│   └── NVDA_analytics_sliding_window.csv
│
└── Market Context (4 files)
    ├── market_top_gainers.csv
    ├── market_top_losers.csv
    ├── market_most_actively_traded.csv
    └── NVDA_etf_holdings.csv
```

**Total: ~70+ files with comprehensive NVIDIA data!**

---

## ⚡ Performance & API Usage

- **Total API Calls:** ~50-60 calls (Premium allows much higher limits)
- **Estimated Runtime:** 5-10 minutes (with rate limiting)
- **Total Data Size:** Typically 10-50 MB depending on data availability
- **Rate Limiting:** Built-in delays between requests to avoid issues

---

## 🔧 Customization

To modify the script for your needs:

1. **Change Ticker:** Edit `TICKER = "NVDA"` at the top
2. **Change Time Period:** Modify `timedelta(days=5*365)` for different ranges
3. **Add More Technical Indicators:** See [Alpha Vantage docs](https://www.alphavantage.co/documentation/) for more options
4. **Adjust Output Directory:** Change `OUTPUT_DIR = "data"`

---

## ⚠️ Important Notes

### Premium Features (require Premium API):
- Earnings call transcripts
- News sentiment analysis
- Insider transactions
- VWAP indicator
- Advanced analytics

### Free Tier Limitations:
If you don't have Premium, some endpoints may return errors. The script will continue and collect all available free-tier data.

### Data Availability:
- Not all data is available for all companies
- Earnings transcripts depend on Alpha Vantage's coverage
- Some endpoints may require specific Premium plan tiers
- ETF holdings data depends on which ETFs Alpha Vantage covers

---

## 📚 Data Analysis Ideas

Once you have the data, you can:

1. **Correlation Analysis:** See how NVDA correlates with economic indicators
2. **Technical Analysis:** Backtest trading strategies using technical indicators
3. **Sentiment Analysis:** Analyze how news sentiment affects stock price
4. **Insider Trading Patterns:** See if insider activity predicts price movements
5. **Earnings Analysis:** Study earnings surprise patterns and transcript sentiment
6. **Fundamental Analysis:** Track financial metrics over time
7. **Market Context:** Compare NVDA performance to market leaders
8. **ETF Flow:** Understand institutional ownership through ETF holdings

---

## 🆘 Troubleshooting

**Problem:** `ALPHA_VANTAGE_API_KEY not found`
- **Solution:** Create/update `.env` file with your API key

**Problem:** API returns errors for some endpoints
- **Solution:** Some endpoints require specific Premium tiers. The script will continue with available data.

**Problem:** Rate limit errors
- **Solution:** Increase `rate_limit_sleep()` values in the code

**Problem:** Missing data for certain categories
- **Solution:** Not all data is available for all stocks. This is normal.

---

## 📖 References

- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
- [Alpha Vantage Premium Plans](https://www.alphavantage.co/premium/)
- [Alpha Vantage Support](https://www.alphavantage.co/support/)

---

## ✅ Summary

You now have a **production-ready script** that collects:
- ✅ **70+ data files** covering all aspects of NVIDIA
- ✅ **5 years of price history** (daily, weekly, monthly)
- ✅ **Complete fundamental data** (financial statements)
- ✅ **Earnings transcripts** (last 20 quarters = 5 years)
- ✅ **News & sentiment** (up to 1000 articles)
- ✅ **15 technical indicators** (5 years)
- ✅ **10 economic indicators**
- ✅ **Insider transactions**
- ✅ **Market context & ETF holdings**

**No real-time API calls needed** - everything is saved locally as CSV for your analysis!

---

**Created:** January 2026  
**For:** NVIDIA (NVDA) Stock Analysis  
**API:** Alpha Vantage Premium
