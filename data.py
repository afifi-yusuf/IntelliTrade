"""
Alpha Vantage Premium Data Collection Script for NVIDIA (NVDA)
==============================================================

Comprehensive data collection from Alpha Vantage API covering:

1. STOCK PRICE DATA (Multiple Timeframes)
   - Daily adjusted OHLCV (5 years)
   - Weekly adjusted OHLCV (full history)
   - Monthly adjusted OHLCV (full history)
   - Latest real-time quote

2. FUNDAMENTAL DATA
   - Company overview & metrics
   - Income statements (annual & quarterly)
   - Balance sheets (annual & quarterly)
   - Cash flow statements (annual & quarterly)
   - Shares outstanding history

3. EARNINGS DATA
   - Historical earnings (annual & quarterly with EPS surprises)
   - Forward earnings estimates
   - Earnings calendar

4. EARNINGS CALL TRANSCRIPTS (Premium)
   - Last 20 quarters of transcripts (5 years)

5. CORPORATE ACTIONS
   - Dividend history
   - Stock split history

6. NEWS & SENTIMENT (Premium)
   - Up to 1000 recent news articles
   - Sentiment scores & relevance

7. INSIDER TRANSACTIONS (Premium)
   - Insider buying/selling activity

8. TECHNICAL INDICATORS (5 years)
   - Moving Averages: SMA (50, 200), EMA (12, 26)
   - Momentum: RSI, MACD, STOCH, CCI
   - Volatility: Bollinger Bands, ATR
   - Trend: ADX, AROON
   - Volume: OBV, AD
   - VWAP (Premium)

9. ECONOMIC INDICATORS (Federal Reserve Data)
   - Federal Funds Rate
   - Treasury Yield (10-year)
   - Real GDP & GDP per Capita
   - CPI & Inflation Rate
   - Unemployment Rate
   - Nonfarm Payroll
   - Retail Sales
   - Durable Goods Orders

10. ANALYTICS (Premium)
    - Fixed window analysis
    - Sliding window analysis

11. MARKET CONTEXT
    - Top gainers & losers
    - Most actively traded
    - ETF holdings (which major ETFs hold NVDA)

All data saved as CSV files for offline analysis.

Requires: Alpha Vantage Premium API key in .env file
Output: data/ directory with CSV files and transcript text files
"""

import os
import httpx
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional

load_dotenv()

# Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
TICKER = "NVDA"  # NVIDIA
OUTPUT_DIR = "data"  # Directory to save CSV files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rate limiting helper (Premium allows more calls, but still good to be safe)
def rate_limit_sleep(seconds=0.5):
    """Sleep to avoid rate limiting"""
    time.sleep(seconds)


def make_request(params: Dict) -> Optional[Dict]:
    """Make API request with error handling"""
    params['apikey'] = ALPHA_VANTAGE_API_KEY
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None
            if "Note" in data and "API call frequency" in data["Note"]:
                print(f"⚠️  Rate limit warning: {data['Note']}")
                return None
                
            return data
    except httpx.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# 1. STOCK PRICE DATA (DAILY, WEEKLY, MONTHLY)
# ============================================================================

def fetch_daily_stock_data():
    """Fetch daily stock data for NVDA (5 years of history)"""
    print(f"\n📊 Fetching daily stock data for {TICKER}...")
    
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": TICKER,
        "outputsize": "full",  # Get full history (20+ years)
    }
    
    data = make_request(params)
    if not data or "Time Series (Daily)" not in data:
        print("❌ Failed to fetch daily stock data")
        return
    
    # Convert to DataFrame
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns
    df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 
                  'dividend_amount', 'split_coefficient']
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Sort by date (oldest first)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter last 5 years
    five_years_ago = datetime.now() - timedelta(days=5*365)
    df = df[df.index >= five_years_ago]
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_daily_stock_data_5y.csv"
    df.to_csv(output_file)
    print(f"✅ Saved daily stock data: {output_file} ({len(df)} trading days)")
    
    rate_limit_sleep()


def fetch_weekly_stock_data():
    """Fetch weekly adjusted stock data"""
    print(f"\n📊 Fetching weekly stock data for {TICKER}...")
    
    params = {
        "function": "TIME_SERIES_WEEKLY_ADJUSTED",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "Weekly Adjusted Time Series" not in data:
        print("❌ Failed to fetch weekly stock data")
        return
    
    # Convert to DataFrame
    time_series = data["Weekly Adjusted Time Series"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns
    df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount']
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_weekly_stock_data.csv"
    df.to_csv(output_file)
    print(f"✅ Saved weekly stock data: {output_file} ({len(df)} weeks)")
    
    rate_limit_sleep()


def fetch_monthly_stock_data():
    """Fetch monthly adjusted stock data"""
    print(f"\n📊 Fetching monthly stock data for {TICKER}...")
    
    params = {
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "Monthly Adjusted Time Series" not in data:
        print("❌ Failed to fetch monthly stock data")
        return
    
    # Convert to DataFrame
    time_series = data["Monthly Adjusted Time Series"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns
    df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount']
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_monthly_stock_data.csv"
    df.to_csv(output_file)
    print(f"✅ Saved monthly stock data: {output_file} ({len(df)} months)")
    
    rate_limit_sleep()


def fetch_quote_endpoint():
    """Fetch latest quote snapshot"""
    print(f"\n💱 Fetching latest quote for {TICKER}...")
    
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "Global Quote" not in data:
        print("❌ Failed to fetch quote")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([data["Global Quote"]])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_latest_quote.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved latest quote: {output_file}")
    
    rate_limit_sleep()


# ============================================================================
# 2. FUNDAMENTAL DATA
# ============================================================================

def fetch_company_overview():
    """Fetch company overview and fundamentals"""
    print(f"\n🏢 Fetching company overview for {TICKER}...")
    
    params = {
        "function": "OVERVIEW",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch company overview")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_company_overview.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved company overview: {output_file}")
    
    rate_limit_sleep()


def fetch_income_statement():
    """Fetch income statements"""
    print(f"\n💰 Fetching income statements for {TICKER}...")
    
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch income statements")
        return
    
    # Annual reports
    if "annualReports" in data:
        df_annual = pd.DataFrame(data["annualReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_income_statement_annual.csv"
        df_annual.to_csv(output_file, index=False)
        print(f"✅ Saved annual income statements: {output_file} ({len(df_annual)} years)")
    
    # Quarterly reports
    if "quarterlyReports" in data:
        df_quarterly = pd.DataFrame(data["quarterlyReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_income_statement_quarterly.csv"
        df_quarterly.to_csv(output_file, index=False)
        print(f"✅ Saved quarterly income statements: {output_file} ({len(df_quarterly)} quarters)")
    
    rate_limit_sleep()


def fetch_balance_sheet():
    """Fetch balance sheets"""
    print(f"\n📋 Fetching balance sheets for {TICKER}...")
    
    params = {
        "function": "BALANCE_SHEET",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch balance sheets")
        return
    
    # Annual reports
    if "annualReports" in data:
        df_annual = pd.DataFrame(data["annualReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_balance_sheet_annual.csv"
        df_annual.to_csv(output_file, index=False)
        print(f"✅ Saved annual balance sheets: {output_file} ({len(df_annual)} years)")
    
    # Quarterly reports
    if "quarterlyReports" in data:
        df_quarterly = pd.DataFrame(data["quarterlyReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_balance_sheet_quarterly.csv"
        df_quarterly.to_csv(output_file, index=False)
        print(f"✅ Saved quarterly balance sheets: {output_file} ({len(df_quarterly)} quarters)")
    
    rate_limit_sleep()


def fetch_cash_flow():
    """Fetch cash flow statements"""
    print(f"\n💸 Fetching cash flow statements for {TICKER}...")
    
    params = {
        "function": "CASH_FLOW",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch cash flow statements")
        return
    
    # Annual reports
    if "annualReports" in data:
        df_annual = pd.DataFrame(data["annualReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_cash_flow_annual.csv"
        df_annual.to_csv(output_file, index=False)
        print(f"✅ Saved annual cash flow statements: {output_file} ({len(df_annual)} years)")
    
    # Quarterly reports
    if "quarterlyReports" in data:
        df_quarterly = pd.DataFrame(data["quarterlyReports"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_cash_flow_quarterly.csv"
        df_quarterly.to_csv(output_file, index=False)
        print(f"✅ Saved quarterly cash flow statements: {output_file} ({len(df_quarterly)} quarters)")
    
    rate_limit_sleep()


def fetch_earnings():
    """Fetch earnings history"""
    print(f"\n📊 Fetching earnings history for {TICKER}...")
    
    params = {
        "function": "EARNINGS",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch earnings")
        return
    
    # Annual earnings
    if "annualEarnings" in data:
        df_annual = pd.DataFrame(data["annualEarnings"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_earnings_annual.csv"
        df_annual.to_csv(output_file, index=False)
        print(f"✅ Saved annual earnings: {output_file} ({len(df_annual)} years)")
    
    # Quarterly earnings
    if "quarterlyEarnings" in data:
        df_quarterly = pd.DataFrame(data["quarterlyEarnings"])
        output_file = f"{OUTPUT_DIR}/{TICKER}_earnings_quarterly.csv"
        df_quarterly.to_csv(output_file, index=False)
        print(f"✅ Saved quarterly earnings: {output_file} ({len(df_quarterly)} quarters)")
    
    rate_limit_sleep()


def fetch_earnings_estimates():
    """Fetch forward earnings estimates"""
    print(f"\n🔮 Fetching earnings estimates for {TICKER}...")
    
    params = {
        "function": "EARNINGS_ESTIMATES",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "estimates" not in data:
        print("❌ Failed to fetch earnings estimates")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["estimates"])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_earnings_estimates.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved earnings estimates: {output_file} ({len(df)} estimates)")
    
    rate_limit_sleep()


def fetch_shares_outstanding():
    """Fetch shares outstanding data"""
    print(f"\n📊 Fetching shares outstanding for {TICKER}...")
    
    params = {
        "function": "SHARES_OUTSTANDING",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch shares outstanding")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_shares_outstanding.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved shares outstanding: {output_file} ({len(df)} data points)")
    
    rate_limit_sleep()


def fetch_dividends():
    """Fetch dividend history"""
    print(f"\n💰 Fetching dividend history for {TICKER}...")
    
    params = {
        "function": "DIVIDENDS",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch dividends (Note: NVDA may not have dividends)")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    if len(df) > 0:
        # Save to CSV
        output_file = f"{OUTPUT_DIR}/{TICKER}_dividends.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved dividends: {output_file} ({len(df)} dividend payments)")
    else:
        print("  ℹ️  No dividend data available")
    
    rate_limit_sleep()


def fetch_splits():
    """Fetch stock split history"""
    print(f"\n🔀 Fetching stock split history for {TICKER}...")
    
    params = {
        "function": "SPLITS",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch stock splits")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    if len(df) > 0:
        # Save to CSV
        output_file = f"{OUTPUT_DIR}/{TICKER}_stock_splits.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved stock splits: {output_file} ({len(df)} splits)")
    else:
        print("  ℹ️  No stock split data available")
    
    rate_limit_sleep()


# ============================================================================
# 3. EARNINGS CALL TRANSCRIPTS AND CALENDAR (PREMIUM FEATURE)
# ============================================================================

def fetch_earnings_calendar():
    """Fetch earnings calendar"""
    print(f"\n📅 Fetching earnings calendar for {TICKER}...")
    
    params = {
        "function": "EARNINGS_CALENDAR",
        "symbol": TICKER,
        "horizon": "12month",
    }
    
    # Make raw request for CSV data
    params['apikey'] = ALPHA_VANTAGE_API_KEY
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(BASE_URL, params=params)
            response.raise_for_status()
            
            # Save CSV directly
            output_file = f"{OUTPUT_DIR}/{TICKER}_earnings_calendar.csv"
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"✅ Saved earnings calendar: {output_file}")
    except Exception as e:
        print(f"❌ Failed to fetch earnings calendar: {e}")
    
    rate_limit_sleep()


def fetch_earnings_call_transcript(year: str = "2024", quarter: str = "Q3"):
    """Fetch earnings call transcript for a specific quarter"""
    print(f"\n📝 Fetching earnings call transcript for {TICKER} {quarter} {year}...")
    
    # Alpha Vantage format: quarter=2024Q1 (year+quarter combined)
    quarter_param = f"{year}{quarter}"
    
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": TICKER,
        "quarter": quarter_param,
    }
    
    data = make_request(params)
    if not data:
        print(f"  ⚠️  No transcript available for {quarter} {year}")
        return False
    
    # Check if there's actual transcript data
    if isinstance(data, dict):
        # Check for various "no data" indicators
        if "Information" in data or "Note" in data or "Error Message" in data:
            print(f"  ⚠️  No transcript data for {quarter} {year}")
            return False
        
        # Check if response is empty or has no content
        if len(data) == 0 or (len(data) == 1 and list(data.keys())[0] in ["Information", "Note"]):
            print(f"  ⚠️  No transcript data for {quarter} {year}")
            return False
    
    # Save transcript - handle different response formats
    output_file = f"{OUTPUT_DIR}/{TICKER}_earnings_transcript_{year}_{quarter}.txt"
    
    try:
        # Alpha Vantage returns transcript in different possible formats
        transcript_text = None
        
        if isinstance(data, dict):
            # Try different possible keys
            if "transcript" in data:
                transcript_data = data["transcript"]
                
                # Handle list format
                if isinstance(transcript_data, list):
                    transcript_text = "\n".join(str(item) for item in transcript_data)
                # Handle string format
                elif isinstance(transcript_data, str):
                    transcript_text = transcript_data
                # Handle dict format
                else:
                    transcript_text = json.dumps(transcript_data, indent=2)
            
            # Sometimes the entire response is the transcript
            elif len(data) > 0 and "symbol" not in data and "Meta Data" not in data:
                # Likely the whole dict is transcript data
                transcript_text = json.dumps(data, indent=2)
        
        # If we got transcript text, save it
        if transcript_text and len(transcript_text.strip()) > 50:  # At least 50 chars
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            # Get size in KB for display
            size_kb = len(transcript_text) / 1024
            print(f"  ✅ Saved transcript ({size_kb:.1f} KB): {os.path.basename(output_file)}")
            rate_limit_sleep()
            return True
        else:
            print(f"  ⚠️  No transcript content for {quarter} {year}")
            rate_limit_sleep()
            return False
            
    except Exception as e:
        print(f"  ❌ Error saving transcript: {e}")
        rate_limit_sleep()
        return False


def fetch_all_recent_earnings_transcripts():
    """Fetch earnings call transcripts for recent quarters (5 years = 20 quarters)"""
    print(f"\n📝 Fetching earnings call transcripts for {TICKER} (last 5 years)...")
    
    # Fetch transcripts for the last 20 quarters (5 years)
    # Start from the most recent completed quarter
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_quarter = (current_month - 1) // 3 + 1
    
    # Generate list of quarters to fetch (going backwards from current)
    quarters_to_fetch = []
    year = current_year
    quarter = current_quarter
    
    for i in range(20):  # Fetch last 20 quarters (5 years)
        # Move to previous quarter
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
        
        quarters_to_fetch.append((year, f"Q{quarter}"))
    
    print(f"  Attempting to fetch transcripts from Q{quarters_to_fetch[-1][1]} {quarters_to_fetch[-1][0]} to Q{quarters_to_fetch[0][1]} {quarters_to_fetch[0][0]}...")
    
    success_count = 0
    for year, quarter in quarters_to_fetch:
        if fetch_earnings_call_transcript(str(year), quarter):
            success_count += 1
    
    if success_count > 0:
        print(f"\n  ✅ Successfully fetched {success_count} earnings call transcripts (out of 20 quarters)")
    else:
        print(f"\n  ⚠️  No transcripts available (may require specific Premium tier)")


# ============================================================================
# 4. NEWS AND SENTIMENT (PREMIUM FEATURE)
# ============================================================================

def fetch_news_sentiment():
    """Fetch news and sentiment data"""
    print(f"\n📰 Fetching news & sentiment for {TICKER}...")
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": TICKER,
        "limit": 1000,  # Get up to 1000 news articles
    }
    
    data = make_request(params)
    if not data or "feed" not in data:
        print("❌ Failed to fetch news sentiment")
        return
    
    # Convert news feed to DataFrame
    news_items = []
    for item in data["feed"]:
        news_item = {
            "time_published": item.get("time_published"),
            "title": item.get("title"),
            "url": item.get("url"),
            "summary": item.get("summary"),
            "source": item.get("source"),
            "category_within_source": item.get("category_within_source"),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
        }
        
        # Extract ticker-specific sentiment
        if "ticker_sentiment" in item:
            for ticker_data in item["ticker_sentiment"]:
                if ticker_data.get("ticker") == TICKER:
                    news_item["ticker_sentiment_score"] = ticker_data.get("ticker_sentiment_score")
                    news_item["ticker_sentiment_label"] = ticker_data.get("ticker_sentiment_label")
                    news_item["ticker_relevance_score"] = ticker_data.get("relevance_score")
        
        news_items.append(news_item)
    
    df = pd.DataFrame(news_items)
    output_file = f"{OUTPUT_DIR}/{TICKER}_news_sentiment.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved news & sentiment: {output_file} ({len(df)} articles)")
    
    rate_limit_sleep()


# ============================================================================
# 5. INSIDER TRANSACTIONS
# ============================================================================

def fetch_insider_transactions():
    """Fetch insider transaction data"""
    print(f"\n👔 Fetching insider transactions for {TICKER}...")
    
    # Note: Alpha Vantage uses INSIDER_TRANSACTIONS function (check if available in your plan)
    params = {
        "function": "INSIDER_TRANSACTIONS",
        "symbol": TICKER,
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch insider transactions (may require specific premium plan)")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    output_file = f"{OUTPUT_DIR}/{TICKER}_insider_transactions.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved insider transactions: {output_file} ({len(df)} transactions)")
    
    rate_limit_sleep()


# ============================================================================
# 6. TECHNICAL INDICATORS
# ============================================================================

def fetch_technical_indicator(indicator: str, params: Dict, filename: str):
    """Generic function to fetch technical indicators"""
    print(f"  📈 Fetching {indicator}...")
    
    params.update({
        "symbol": TICKER,
        "interval": "daily",
    })
    
    data = make_request(params)
    if not data:
        print(f"  ❌ Failed to fetch {indicator}")
        return
    
    # Find the technical analysis key (varies by indicator)
    tech_key = None
    for key in data.keys():
        if "Technical Analysis" in key or key == "data":
            tech_key = key
            break
    
    if not tech_key or tech_key not in data:
        print(f"  ❌ No technical data found for {indicator}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data[tech_key], orient='index')
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter last 5 years
    five_years_ago = datetime.now() - timedelta(days=5*365)
    df = df[df.index >= five_years_ago]
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_{filename}.csv"
    df.to_csv(output_file)
    print(f"  ✅ Saved {indicator}: {output_file} ({len(df)} days)")
    
    rate_limit_sleep()


def fetch_all_technical_indicators():
    """Fetch various technical indicators"""
    print(f"\n📊 Fetching technical indicators for {TICKER}...")
    
    indicators = [
        # Moving Averages
        ("SMA (50-day)", {"function": "SMA", "time_period": 50, "series_type": "close"}, "sma_50"),
        ("SMA (200-day)", {"function": "SMA", "time_period": 200, "series_type": "close"}, "sma_200"),
        ("EMA (12-day)", {"function": "EMA", "time_period": 12, "series_type": "close"}, "ema_12"),
        ("EMA (26-day)", {"function": "EMA", "time_period": 26, "series_type": "close"}, "ema_26"),
        
        # Momentum Indicators
        ("RSI", {"function": "RSI", "time_period": 14, "series_type": "close"}, "rsi"),
        ("MACD", {"function": "MACD", "series_type": "close"}, "macd"),
        ("STOCH (Stochastic)", {"function": "STOCH"}, "stoch"),
        ("CCI (Commodity Channel Index)", {"function": "CCI", "time_period": 20}, "cci"),
        
        # Volatility Indicators
        ("Bollinger Bands", {"function": "BBANDS", "time_period": 20, "series_type": "close"}, "bbands"),
        ("ATR (Average True Range)", {"function": "ATR", "time_period": 14}, "atr"),
        
        # Trend Indicators
        ("ADX (Average Directional Index)", {"function": "ADX", "time_period": 14}, "adx"),
        ("AROON", {"function": "AROON", "time_period": 14}, "aroon"),
        
        # Volume Indicators
        ("OBV (On Balance Volume)", {"function": "OBV"}, "obv"),
        ("AD (Accumulation/Distribution)", {"function": "AD"}, "ad"),
    ]
    
    # Add VWAP (Premium feature)
    indicators.append(("VWAP", {"function": "VWAP"}, "vwap"))
    
    for name, params, filename in indicators:
        fetch_technical_indicator(name, params, filename)


# ============================================================================
# 7. ECONOMIC INDICATORS (FED DATA)
# ============================================================================

def fetch_economic_indicator(indicator_name: str, function: str, filename: str):
    """Generic function to fetch economic indicators"""
    print(f"  📊 Fetching {indicator_name}...")
    
    params = {
        "function": function,
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print(f"  ❌ Failed to fetch {indicator_name}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    # Sort by date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/economic_{filename}.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✅ Saved {indicator_name}: {output_file} ({len(df)} data points)")
    
    rate_limit_sleep()


def fetch_all_economic_indicators():
    """Fetch various economic indicators"""
    print(f"\n🏛️  Fetching economic indicators (Fed data)...")
    
    indicators = [
        # Interest Rates
        ("Federal Funds Rate", "FEDERAL_FUNDS_RATE", "federal_funds_rate"),
        ("10-Year Treasury Yield", "TREASURY_YIELD", "treasury_yield_10y"),
        
        # GDP & Growth
        ("Real GDP", "REAL_GDP", "real_gdp"),
        ("Real GDP per Capita", "REAL_GDP_PER_CAPITA", "real_gdp_per_capita"),
        
        # Inflation
        ("CPI (Consumer Price Index)", "CPI", "cpi"),
        ("Inflation Rate", "INFLATION", "inflation"),
        
        # Employment
        ("Unemployment Rate", "UNEMPLOYMENT", "unemployment"),
        ("Nonfarm Payroll", "NONFARM_PAYROLL", "nonfarm_payroll"),
        
        # Consumer & Business Activity
        ("Retail Sales", "RETAIL_SALES", "retail_sales"),
        ("Durable Goods Orders", "DURABLES", "durable_goods_orders"),
    ]
    
    for name, function, filename in indicators:
        fetch_economic_indicator(name, function, filename)


# ============================================================================
# 8. ANALYTICS (PREMIUM FEATURE)
# ============================================================================

def fetch_analytics_fixed_window():
    """Fetch analytics with fixed window"""
    print(f"\n📊 Fetching analytics (fixed window) for {TICKER}...")
    
    params = {
        "function": "ANALYTICS_FIXED_WINDOW",
        "symbol": TICKER,
        "RANGE": "full",  # Full historical analysis
        "INTERVAL": "DAILY",
        "CALCULATIONS": "MEAN,MEDIAN,STDDEV,MIN,MAX",
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch analytics (may require specific premium plan)")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_analytics_fixed_window.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved analytics (fixed window): {output_file}")
    
    rate_limit_sleep()


def fetch_analytics_sliding_window():
    """Fetch analytics with sliding window"""
    print(f"\n📊 Fetching analytics (sliding window) for {TICKER}...")
    
    params = {
        "function": "ANALYTICS_SLIDING_WINDOW",
        "symbol": TICKER,
        "RANGE": "full",
        "INTERVAL": "DAILY",
        "WINDOW_SIZE": "20",  # 20-day sliding window
        "CALCULATIONS": "MEAN,STDDEV",
    }
    
    data = make_request(params)
    if not data or "data" not in data:
        print("❌ Failed to fetch sliding window analytics (may require specific premium plan)")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{TICKER}_analytics_sliding_window.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved analytics (sliding window): {output_file}")
    
    rate_limit_sleep()


def fetch_top_gainers_losers():
    """Fetch top gainers and losers for market context"""
    print(f"\n📈📉 Fetching top gainers & losers...")
    
    params = {
        "function": "TOP_GAINERS_LOSERS",
    }
    
    data = make_request(params)
    if not data:
        print("❌ Failed to fetch top gainers & losers")
        return
    
    # Save each category
    for category in ["top_gainers", "top_losers", "most_actively_traded"]:
        if category in data:
            df = pd.DataFrame(data[category])
            output_file = f"{OUTPUT_DIR}/market_{category}.csv"
            df.to_csv(output_file, index=False)
            print(f"  ✅ Saved {category}: {output_file}")
    
    rate_limit_sleep()


# ============================================================================
# 9. ADDITIONAL CONTEXT
# ============================================================================

def fetch_etf_holdings_that_own_stock():
    """Find which ETFs hold NVDA stock (requires knowing major ETF symbols)"""
    print(f"\n📦 Checking major ETF holdings of {TICKER}...")
    
    # Major ETFs that likely hold NVDA (tech-focused and broad market)
    major_etfs = ["QQQ", "SPY", "VGT", "XLK", "SOXX", "SMH", "VTI", "IWF"]
    
    all_holdings = []
    
    for etf_symbol in major_etfs:
        print(f"  Checking {etf_symbol}...")
        params = {
            "function": "ETF_PROFILE",
            "symbol": etf_symbol,
        }
        
        data = make_request(params)
        if data and "holdings" in data:
            # Look for NVDA in holdings
            holdings = data.get("holdings", [])
            for holding in holdings:
                if holding.get("symbol") == TICKER:
                    all_holdings.append({
                        "etf": etf_symbol,
                        "etf_name": data.get("name", ""),
                        "weight": holding.get("weight", ""),
                        "shares": holding.get("shares", ""),
                    })
                    print(f"    ✅ Found {TICKER} in {etf_symbol}")
                    break
        
        rate_limit_sleep(0.3)
    
    if all_holdings:
        df = pd.DataFrame(all_holdings)
        output_file = f"{OUTPUT_DIR}/{TICKER}_etf_holdings.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved ETF holdings: {output_file} ({len(all_holdings)} ETFs)")
    else:
        print("  ℹ️  No ETF holdings data found")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to fetch all data"""
    print("=" * 80)
    print("🚀 ALPHA VANTAGE PREMIUM DATA COLLECTION FOR NVIDIA (NVDA)")
    print("=" * 80)
    print(f"Ticker: {TICKER}")
    print(f"Output Directory: {OUTPUT_DIR}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not ALPHA_VANTAGE_API_KEY:
        print("❌ ERROR: ALPHA_VANTAGE_API_KEY not found in .env file")
        return
    
    try:
        # 1. Stock Price Data (Multiple Timeframes)
        print("\n" + "=" * 80)
        print("1️⃣  STOCK PRICE DATA")
        print("=" * 80)
        fetch_daily_stock_data()
        fetch_weekly_stock_data()
        fetch_monthly_stock_data()
        fetch_quote_endpoint()
        
        # 2. Fundamental Data
        print("\n" + "=" * 80)
        print("2️⃣  FUNDAMENTAL DATA")
        print("=" * 80)
        fetch_company_overview()
        fetch_income_statement()
        fetch_balance_sheet()
        fetch_cash_flow()
        fetch_shares_outstanding()
        
        # 3. Earnings Data
        print("\n" + "=" * 80)
        print("3️⃣  EARNINGS DATA")
        print("=" * 80)
        fetch_earnings()
        fetch_earnings_estimates()
        fetch_earnings_calendar()
        
        # 4. Earnings Call Transcripts (Premium)
        print("\n" + "=" * 80)
        print("4️⃣  EARNINGS CALL TRANSCRIPTS (Premium)")
        print("=" * 80)
        fetch_all_recent_earnings_transcripts()
        
        # 5. Corporate Actions
        print("\n" + "=" * 80)
        print("5️⃣  CORPORATE ACTIONS")
        print("=" * 80)
        fetch_dividends()
        fetch_splits()
        
        # 6. News & Sentiment (Premium)
        print("\n" + "=" * 80)
        print("6️⃣  NEWS & SENTIMENT (Premium)")
        print("=" * 80)
        fetch_news_sentiment()
        
        # 7. Insider Transactions (Premium)
        print("\n" + "=" * 80)
        print("7️⃣  INSIDER TRANSACTIONS (Premium)")
        print("=" * 80)
        fetch_insider_transactions()
        
        # 8. Technical Indicators
        print("\n" + "=" * 80)
        print("8️⃣  TECHNICAL INDICATORS")
        print("=" * 80)
        fetch_all_technical_indicators()
        
        # 9. Economic Indicators
        print("\n" + "=" * 80)
        print("9️⃣  ECONOMIC INDICATORS (Fed Data)")
        print("=" * 80)
        fetch_all_economic_indicators()
        
        # 10. Analytics (Premium)
        print("\n" + "=" * 80)
        print("🔟 ANALYTICS (Premium)")
        print("=" * 80)
        fetch_analytics_fixed_window()
        fetch_analytics_sliding_window()
        
        # 11. Market Context
        print("\n" + "=" * 80)
        print("1️⃣1️⃣  MARKET CONTEXT")
        print("=" * 80)
        fetch_top_gainers_losers()
        fetch_etf_holdings_that_own_stock()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 All data saved to: {OUTPUT_DIR}/")
    print("\n📊 Files created:")
    
    # List all files created
    import glob
    all_files = glob.glob(f"{OUTPUT_DIR}/*")
    
    # Separate by file type
    csv_files = [f for f in all_files if f.endswith('.csv')]
    txt_files = [f for f in all_files if f.endswith('.txt')]
    json_files = [f for f in all_files if f.endswith('.json')]
    
    print(f"\n  CSV Files ({len(csv_files)}):")
    for i, file in enumerate(sorted(csv_files), 1):
        size = os.path.getsize(file) / 1024  # Size in KB
        print(f"    {i}. {os.path.basename(file)} ({size:.1f} KB)")
    
    if txt_files:
        print(f"\n  Transcript Files ({len(txt_files)}):")
        for i, file in enumerate(sorted(txt_files), 1):
            size = os.path.getsize(file) / 1024
            print(f"    {i}. {os.path.basename(file)} ({size:.1f} KB)")
    
    if json_files:
        print(f"\n  JSON Files ({len(json_files)}):")
        for i, file in enumerate(sorted(json_files), 1):
            size = os.path.getsize(file) / 1024
            print(f"    {i}. {os.path.basename(file)} ({size:.1f} KB)")
    
    total_size = sum(os.path.getsize(f) for f in all_files) / (1024 * 1024)  # MB
    print(f"\n  📦 Total data collected: {total_size:.2f} MB")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
