import os
import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
import json

load_dotenv()

# Initialize clients
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)


def fetch_earnings_calendar(ticker: str) -> dict:
    """
    Fetch earnings calendar from Finnhub API.
    Note: Free tier primarily provides upcoming earnings dates.
    Historical data may be limited or unavailable.
    """
    url = "https://finnhub.io/api/v1/calendar/earnings"
    
    params = {
        "symbol": ticker.upper(),
        "token": finnhub_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.HTTPError as e:
        print(f"Error fetching earnings calendar: {e}")
        return None


def fetch_earnings_surprise(ticker: str) -> dict:
    """Fetch earnings surprise data from Finnhub API (free tier: up to 1 year of data)."""
    url = "https://finnhub.io/api/v1/stock/earnings"
    params = {
        "symbol": ticker.upper(),
        "token": finnhub_api_key
    }
    
    # Free tier provides up to 1 year of earnings data without needing limit parameter
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching earnings surprise: {e}")
        return None


def fetch_company_profile(ticker: str) -> dict:
    """Fetch company profile from Finnhub API (free tier)."""
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {
        "symbol": ticker.upper(),
        "token": finnhub_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching company profile: {e}")
        return None


def fetch_basic_financials(ticker: str) -> dict:
    """Fetch basic financial metrics from Finnhub API (free tier)."""
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {
        "symbol": ticker.upper(),
        "metric": "all",
        "token": finnhub_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching basic financials: {e}")
        return None


# ============================================================================
# ALPHA VANTAGE API FUNCTIONS
# ============================================================================

def fetch_av_earnings(ticker: str) -> dict:
    """Fetch earnings data from Alpha Vantage (more historical data than Finnhub)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS",
        "symbol": ticker.upper(),
        "apikey": alpha_vantage_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching Alpha Vantage earnings: {e}")
        return None


def fetch_av_income_statement(ticker: str) -> dict:
    """Fetch income statement from Alpha Vantage (annual and quarterly)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker.upper(),
        "apikey": alpha_vantage_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching income statement: {e}")
        return None


def fetch_av_balance_sheet(ticker: str) -> dict:
    """Fetch balance sheet from Alpha Vantage (annual and quarterly)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "BALANCE_SHEET",
        "symbol": ticker.upper(),
        "apikey": alpha_vantage_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching balance sheet: {e}")
        return None


def fetch_av_cash_flow(ticker: str) -> dict:
    """Fetch cash flow statement from Alpha Vantage (annual and quarterly)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CASH_FLOW",
        "symbol": ticker.upper(),
        "apikey": alpha_vantage_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching cash flow: {e}")
        return None


def fetch_av_overview(ticker: str) -> dict:
    """Fetch company overview from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker.upper(),
        "apikey": alpha_vantage_api_key
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"Error fetching company overview: {e}")
        return None


def analyze_with_ai(
    # Finnhub data
    calendar_data: dict, 
    surprise_data: dict, 
    profile_data: dict,
    financials_data: dict,
    # Alpha Vantage data
    av_earnings: dict = None,
    av_income: dict = None,
    av_balance: dict = None,
    av_cashflow: dict = None,
    av_overview: dict = None,
    user_query: str = None
) -> str:
    """Send comprehensive company data to Azure OpenAI for analysis."""
    
    # Format Finnhub data
    calendar_text = json.dumps(calendar_data, indent=2) if calendar_data else "No calendar data available"
    surprise_text = json.dumps(surprise_data, indent=2) if surprise_data else "No surprise data available"
    profile_text = json.dumps(profile_data, indent=2) if profile_data else "No profile data available"
    financials_text = json.dumps(financials_data, indent=2) if financials_data else "No financials data available"
    
    # Format Alpha Vantage data
    av_earnings_text = json.dumps(av_earnings, indent=2) if av_earnings else "No extended earnings data available"
    av_income_text = json.dumps(av_income, indent=2) if av_income else "No income statement available"
    av_balance_text = json.dumps(av_balance, indent=2) if av_balance else "No balance sheet available"
    av_cashflow_text = json.dumps(av_cashflow, indent=2) if av_cashflow else "No cash flow statement available"
    av_overview_text = json.dumps(av_overview, indent=2) if av_overview else "No overview available"
    
    # Build the prompt
    system_prompt = """You are an expert financial analyst assistant. Analyze the provided comprehensive company data including:
    
    **Company Information:**
    - Company profile (industry, sector, market cap, description)
    - Detailed overview and fundamentals
    
    **Financial Statements:**
    - Income statement (revenue, expenses, net income trends)
    - Balance sheet (assets, liabilities, equity)
    - Cash flow statement (operating, investing, financing activities)
    
    **Performance Metrics:**
    - Financial ratios (P/E, P/B, PEG, ROE, ROA, margins, debt ratios)
    - Earnings calendar and estimates
    - Extended earnings history (multiple quarters/years)
    - Earnings surprises and beat/miss patterns
    
    Provide comprehensive analysis including:
    - Executive summary and investment thesis
    - Company overview and business model
    - Valuation assessment (compare to industry, historical trends)
    - Profitability analysis (margins, returns, efficiency)
    - Financial health (liquidity, solvency, debt management)
    - Revenue and earnings growth trajectory
    - Cash flow quality and trends
    - Beat/miss patterns and earnings consistency
    - Risks and opportunities
    - Conclusion and investment recommendation"""
    
    user_prompt = f"""Here is the complete company data:

**FINNHUB DATA**

**Company Profile:**
{profile_text}

**Financial Metrics & Ratios:**
{financials_text}

**Earnings Calendar:**
{calendar_text}

**Earnings Surprise (1 year):**
{surprise_text}

---

**ALPHA VANTAGE DATA**

**Company Overview:**
{av_overview_text}

**Extended Earnings History:**
{av_earnings_text}

**Income Statement (Annual & Quarterly):**
{av_income_text}

**Balance Sheet (Annual & Quarterly):**
{av_balance_text}

**Cash Flow Statement (Annual & Quarterly):**
{av_cashflow_text}

"""
    
    if user_query:
        user_prompt += f"\n**User Question:** {user_query}"
    else:
        user_prompt += "\n**Task:** Please provide a comprehensive fundamental and earnings analysis of this company with an investment thesis."
    
    try:
        response = azure_client.chat.completions.create(
    model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return None


def rag_pipeline(ticker: str, user_query: str = None):
    """Main RAG pipeline to fetch and analyze company data from Finnhub and Alpha Vantage."""
    print(f"\n🔍 Fetching comprehensive data for {ticker.upper()}...")
    print("=" * 80)
    
    # ========== FINNHUB DATA ==========
    print("\n📊 FINNHUB DATA (4 API calls)")
    print("-" * 80)
    
    # Step 1: Fetch company profile
    print("  🏢 Fetching company profile...")
    profile_data = fetch_company_profile(ticker)
    
    # Step 2: Fetch basic financials
    print("  💰 Fetching financial metrics...")
    financials_data = fetch_basic_financials(ticker)
    
    # Step 3: Fetch earnings calendar
    print("  📅 Fetching earnings calendar...")
    calendar_data = fetch_earnings_calendar(ticker)
    
    # Step 4: Fetch earnings surprise history
    print("  📊 Fetching earnings surprise history...")
    surprise_data = fetch_earnings_surprise(ticker)
    
    # ========== ALPHA VANTAGE DATA ==========
    print("\n📈 ALPHA VANTAGE DATA (5 API calls - Free tier: 25/day limit)")
    print("-" * 80)
    
    # Step 5: Fetch Alpha Vantage company overview
    print("  🏢 Fetching company overview...")
    av_overview = fetch_av_overview(ticker)
    
    # Step 6: Fetch Alpha Vantage extended earnings
    print("  📊 Fetching extended earnings history...")
    av_earnings = fetch_av_earnings(ticker)
    
    # Step 7: Fetch income statement
    print("  💵 Fetching income statement...")
    av_income = fetch_av_income_statement(ticker)
    
    # Step 8: Fetch balance sheet
    print("  📋 Fetching balance sheet...")
    av_balance = fetch_av_balance_sheet(ticker)
    
    # Step 9: Fetch cash flow statement
    print("  💸 Fetching cash flow statement...")
    av_cashflow = fetch_av_cash_flow(ticker)
    
    # Check if we have at least some data
    has_data = any([
        profile_data, financials_data, calendar_data, surprise_data,
        av_overview, av_earnings, av_income, av_balance, av_cashflow
    ])
    
    if not has_data:
        print("\n❌ No data found or error occurred.")
        return
    
    # ========== DATA SUMMARY ==========
    print("\n" + "=" * 80)
    print("📦 DATA SUMMARY")
    print("=" * 80)
    
    print("\n🔹 FINNHUB:")
    if profile_data and profile_data.get("name"):
        print(f"  ✅ Company: {profile_data.get('name')} ({profile_data.get('ticker')})")
        print(f"     Industry: {profile_data.get('finnhubIndustry', 'N/A')}")
    else:
        print("  ⚠️  No company profile")
    
    if financials_data and financials_data.get("metric"):
        metrics = financials_data.get("metric", {})
        print(f"  ✅ Financial metrics: {len(metrics)} ratios")
    else:
        print("  ⚠️  No financial metrics")
    
    if calendar_data and calendar_data.get("earningsCalendar"):
        print(f"  ✅ Earnings calendar: {len(calendar_data['earningsCalendar'])} entries")
    else:
        print("  ⚠️  No calendar data")
    
    if surprise_data:
        print(f"  ✅ Earnings surprise: {len(surprise_data)} records (1 year)")
    else:
        print("  ⚠️  No surprise data")
    
    print("\n🔹 ALPHA VANTAGE:")
    if av_overview and av_overview.get("Name"):
        print(f"  ✅ Company overview: {av_overview.get('Name')}")
        print(f"     Sector: {av_overview.get('Sector', 'N/A')} | Industry: {av_overview.get('Industry', 'N/A')}")
    else:
        print("  ⚠️  No company overview")
    
    if av_earnings and av_earnings.get("quarterlyEarnings"):
        print(f"  ✅ Extended earnings: {len(av_earnings.get('quarterlyEarnings', []))} quarters")
    else:
        print("  ⚠️  No extended earnings")
    
    if av_income and av_income.get("quarterlyReports"):
        print(f"  ✅ Income statement: {len(av_income.get('quarterlyReports', []))} quarters")
    else:
        print("  ⚠️  No income statement")
    
    if av_balance and av_balance.get("quarterlyReports"):
        print(f"  ✅ Balance sheet: {len(av_balance.get('quarterlyReports', []))} quarters")
    else:
        print("  ⚠️  No balance sheet")
    
    if av_cashflow and av_cashflow.get("quarterlyReports"):
        print(f"  ✅ Cash flow: {len(av_cashflow.get('quarterlyReports', []))} quarters")
    else:
        print("  ⚠️  No cash flow")
    
    # ========== AI ANALYSIS ==========
    print("\n" + "=" * 80)
    print("🤖 ANALYZING WITH AI...")
    print("=" * 80)
    
    analysis = analyze_with_ai(
        # Finnhub data
        calendar_data, surprise_data, profile_data, financials_data,
        # Alpha Vantage data
        av_earnings, av_income, av_balance, av_cashflow, av_overview,
        # User query
        user_query
    )
    
    if analysis:
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE INVESTMENT ANALYSIS")
        print("="*80)
        print(analysis)
        print("="*80)
    else:
        print("\n❌ Failed to generate analysis.")


def main():
    """Main entry point."""
    print("="*80)
    print("📊 INSTITUTIONAL-GRADE STOCK ANALYSIS RAG PIPELINE")
    print("="*80)
    print("Powered by: Finnhub + Alpha Vantage + Azure OpenAI")
    print("\n🔹 Data Sources (9 FREE API endpoints):")
    print("\n  FINNHUB (4 endpoints):")
    print("    • Company Profile & Metrics")
    print("    • Financial Ratios (50+ metrics)")
    print("    • Earnings Calendar")
    print("    • Earnings Surprise (1 year)")
    print("\n  ALPHA VANTAGE (5 endpoints):")
    print("    • Company Overview & Fundamentals")
    print("    • Extended Earnings History (All quarters)")
    print("    • Income Statement (Annual & Quarterly)")
    print("    • Balance Sheet (Annual & Quarterly)")
    print("    • Cash Flow Statement (Annual & Quarterly)")
    print("\n⚠️  Note: Alpha Vantage free tier has 25 API calls/day limit")
    print("💡 Tip: You can analyze 2-3 companies per day on free tier")
    
    # Get ticker from user
    ticker = input("\nEnter stock ticker symbol: ").strip()
    
    if not ticker:
        print("No ticker provided. Exiting.")
        return
    
    # Optional: Get specific question from user
    print("\nEnter a specific question (or press Enter for comprehensive analysis):")
    user_query = input("> ").strip()
    
    if not user_query:
        user_query = None
    
    # Run the RAG pipeline
    rag_pipeline(ticker, user_query)


if __name__ == "__main__":
    main()
