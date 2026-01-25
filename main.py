import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
import json
import glob
# Import Graph Manager
from graph_manager import Neo4jHandler

load_dotenv()

# Initialize API clients (Client for Azure only)
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)

# Initialize Neo4j Handler
try:
    graph_db = Neo4jHandler()
except:
    graph_db = None
    print("⚠️ Could not initialize Neo4j connection. Graph features will be disabled.")

# Directory where local data is stored
DATA_DIR = "data"

def load_csv_data(ticker: str, file_suffix: str) -> dict:
    """Helper to load local CSV data and convert to dict/json-compatible format."""
    # Try both uppercase and original case ticker just in case
    filename = os.path.join(DATA_DIR, f"{ticker.upper()}_{file_suffix}")
    if not os.path.exists(filename):
        # Try lowercase?
        filename_lower = os.path.join(DATA_DIR, f"{ticker.lower()}_{file_suffix}")
        if os.path.exists(filename_lower):
            filename = filename_lower
        else:
            print(f"  ⚠️  File not found: {filename}")
            return None
            
    try:
        df = pd.read_csv(filename)
        # Convert to dict (list of records)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"  ❌ Error reading {filename}: {e}")
        return None

# ============================================================================
# DATA LOADING FUNCTIONS (LOCAL)
# ============================================================================

def fetch_company_profile(ticker: str) -> dict:
    """Load company profile/overview from local CSV."""
    # We use AV company overview as the main profile source now
    data = load_csv_data(ticker, "company_overview.csv")
    if data and len(data) > 0:
        return data[0] # Return single object
    return None

def fetch_basic_financials(ticker: str) -> dict:
    """Load basic financials. using Company Overview as proxy for metrics."""
    # Returning the same overview data as "metrics" since AV Overview is rich
    return fetch_company_profile(ticker)

def fetch_earnings_calendar(ticker: str) -> dict:
    """Load earnings calendar."""
    data = load_csv_data(ticker, "earnings_calendar.csv")
    return {"earningsCalendar": data} if data else None

def fetch_earnings_surprise(ticker: str) -> dict:
    """Load earnings surprise history (from earnings quarterly)."""
    # AV earnings data contains surprise info
    data = load_csv_data(ticker, "earnings_quarterly.csv")
    return data # List of quarters

# ============================================================================
# ALPHA VANTAGE DATA (LOCAL)
# ============================================================================

def fetch_av_earnings(ticker: str) -> dict:
    """Load extended earnings history."""
    # We load both annual and quarterly if needed, but here just return quarterly wrapper
    quarterly = load_csv_data(ticker, "earnings_quarterly.csv")
    annual = load_csv_data(ticker, "earnings_annual.csv")
    return {
        "quarterlyEarnings": quarterly,
        "annualEarnings": annual
    }

def fetch_av_income_statement(ticker: str) -> dict:
    """Load income statement."""
    quarterly = load_csv_data(ticker, "income_statement_quarterly.csv")
    return {"quarterlyReports": quarterly}

def fetch_av_balance_sheet(ticker: str) -> dict:
    """Load balance sheet."""
    quarterly = load_csv_data(ticker, "balance_sheet_quarterly.csv")
    return {"quarterlyReports": quarterly}

def fetch_av_cash_flow(ticker: str) -> dict:
    """Load cash flow statement."""
    quarterly = load_csv_data(ticker, "cash_flow_quarterly.csv")
    return {"quarterlyReports": quarterly}

def fetch_av_overview(ticker: str) -> dict:
    """Load company overview."""
    return fetch_company_profile(ticker)


def analyze_with_ai(
    # Data arguments
    calendar_data: dict, 
    surprise_data: dict, 
    profile_data: dict,
    financials_data: dict,
    av_earnings: dict = None,
    av_income: dict = None,
    av_balance: dict = None,
    av_cashflow: dict = None,
    av_overview: dict = None,
    # Graph Context
    graph_context: str = None,
    user_query: str = None
) -> str:
    """Send comprehensive company data to Azure OpenAI for analysis."""
    
    # Helper to safe-serialize
    def safe_json(data, name):
        if not data: return f"No {name} data available"
        # Truncate large lists to avoid token limits
        if isinstance(data, list) and len(data) > 10:
            return json.dumps(data[:10], indent=2) + f"\n... ({len(data)-10} more records)"
        if isinstance(data, dict):
            # Check for inner lists
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 10:
                    data = data.copy()
                    data[k] = v[:10]
        return json.dumps(data, indent=2)

    # Format data strings
    profile_text = safe_json(profile_data, "profile")
    financials_text = safe_json(financials_data, "financials")
    calendar_text = safe_json(calendar_data, "calendar")
    surprise_text = safe_json(surprise_data, "surprise")
    
    av_earnings_text = safe_json(av_earnings, "earnings")
    av_income_text = safe_json(av_income, "income")
    av_balance_text = safe_json(av_balance, "balance sheet")
    av_cashflow_text = safe_json(av_cashflow, "cash flow")
    av_overview_text = safe_json(av_overview, "overview")
    
    # Build the prompt
    system_prompt = """You are an expert financial analyst assistant utilizing a Context-Enriched Agentic RAG system.
    You have access to:
    1. Comprehensive Financial Data (Fundamentals, Financial Statements, Earnings).
    2. A Causal-Temporal Knowledge Graph (Neo4j) containing historical reasoning extracted from earnings calls.
    
    Analyze the provided data. **CRITICALLY: Connect current performance to historical context provided in the Knowledge Graph section.**
    If the KG shows a recurring reason for missed earnings (e.g. "supply chain issues" appearing in Q1 and Q2), highlight this pattern.
    
    Structure your response as an Institutional Investment Memo.
    
    NOTE: Data is provided in JSON format. 'None' or 'null' indicates missing data."""
    
    user_prompt = f"""
    Please analyze {profile_data.get('Symbol', profile_data.get('ticker', 'the company'))} based on the following:
    
    === CAUSAL-TEMPORAL KNOWLEDGE GRAPH (HISTORICAL CONTEXT) ===
    {graph_context if graph_context else "No historical graph context available."}
    ============================================================

    === FINANCIAL DATA ===
    
    **Company Profile & Overview:**
    {profile_text}
    
    **Key Financial Metrics:**
    {financials_text}
    
    **Income Statement (Recent):**
    {av_income_text}
    
    **Balance Sheet (Recent):**
    {av_balance_text}
    
    **Cash Flow (Recent):**
    {av_cashflow_text}
    
    **Earnings History & Surprises:**
    {av_earnings_text}
    {surprise_text}
    """
    
    if user_query:
        user_prompt += f"\n**User Question:** {user_query}"
    
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

# ============================================================================
# GRAPH RAG & TRANSCRIPT PROCESSING
# ============================================================================

def process_transcripts_for_graph(ticker: str):
    """
    Look for local transcript files, extract facts using LLM, and ingest into Neo4j.
    """
    print(f"\n🧠 Graph Agent: Processing local earnings transcripts for {ticker}...")
    
    if not graph_db:
        print("   ⚠️ Graph DB not available. Skipping.")
        return

    # Find text files in data dir matching pattern
    # Pattern: data/{TICKER}_earnings_transcript_{YEAR}_{QUARTER}.txt
    # or lowercase
    pattern = os.path.join(DATA_DIR, f"{ticker.upper()}_earnings_transcript_*.txt")
    files = glob.glob(pattern)
    
    if not files:
        # Try lowercase
        pattern = os.path.join(DATA_DIR, f"{ticker.lower()}_earnings_transcript_*.txt")
        files = glob.glob(pattern)
        
    if not files:
        print("   ⚠️ No transcript files found in data/ directory.")
        return
        
    print(f"   Found {len(files)} transcript(s). processing recent 2...")
    
    # Sort files to get most recent (by filename usually works if YYYY_Q works)
    files.sort(reverse=True)
    
    # Process only top 2 to save time/tokens
    for filename in files[:2]: 
        print(f"   📖 Reading {os.path.basename(filename)}...")
        
        # Extract Quarter from filename
        # Expected format: TICKER_earnings_transcript_YYYY_QQ.txt
        try:
            parts = os.path.basename(filename).replace(".txt", "").split("_")
            # Usually: NVDA_earnings_transcript_2026_Q2.txt -> ["NVDA", "earnings", "transcript", "2026", "Q2"]
            year = parts[-2]
            quarter = parts[-1] 
            quarter_id = f"{year}-{quarter}"
        except:
            quarter_id = "UNKNOWN"
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            continue
            
        if len(text) < 100:
            print("   ⚠️ File empty or too short.")
            continue
            
        # Extract Facts
        print(f"   🤖 Extracting Causal Facts from {quarter_id}...")
        facts = extract_facts_with_llm(ticker, quarter_id, text[:15000]) # Limit tokens
        
        # Ingest
        if facts:
            print(f"   💾 Ingesting {len(facts)} facts to Neo4j...")
            for fact in facts:
                graph_db.ingest_financial_fact(
                    ticker=ticker,
                    quarter=quarter_id,
                    metric=fact.get("metric"),
                    value=fact.get("value"),
                    unit=fact.get("unit", "USD")
                )
                if fact.get("reason"):
                    graph_db.ingest_transcript_reason(
                        ticker=ticker,
                        quarter=quarter_id,
                        metric=fact.get("metric"),
                        reason_text=fact.get("reason"),
                        sentiment=fact.get("sentiment", "NEUTRAL")
                    )
    print("   ✅ Graph update complete.")


def extract_facts_with_llm(ticker: str, quarter: str, transcript_text: str) -> list:
    """
    Use Azure OpenAI to extract structured facts and causal reasons.
    """
    system_prompt = """You are a financial analyst agent.
    Extract key financial metrics and the *specific reason* provided by management.
    Return JSON: {"facts": [{"metric": "...", "value": "...", "reason": "..."}]}
    """
    
    user_prompt = f"""
    Ticker: {ticker}
    Quarter: {quarter}
    TRANSCRIPT EXCERPT:
    {transcript_text[:10000]} 
    """
    # Truncated context for speed
    
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("facts", [])
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return []


def rag_pipeline(ticker: str, user_query: str = None):
    """Main RAG pipeline."""
    print(f"\n🔍 Analyzing {ticker.upper()} using LOCAL DATA...")
    print("=" * 80)
    
    # Fetch DataLocally
    print("  📂 Loading local files...")
    profile_data = fetch_company_profile(ticker)
    financials_data = fetch_basic_financials(ticker)
    calendar_data = fetch_earnings_calendar(ticker)
    surprise_data = fetch_earnings_surprise(ticker)
    
    av_overview = fetch_av_overview(ticker)
    av_earnings = fetch_av_earnings(ticker)
    av_income = fetch_av_income_statement(ticker)
    av_balance = fetch_av_balance_sheet(ticker)
    av_cashflow = fetch_av_cash_flow(ticker)
    
    if not profile_data and not av_overview:
        print("\n❌ No local data found. Please run data collection script first.")
        return

    # Graph
    print("\n🕸️  NEO4J GRAPH AGENT (Context Enrichment)")
    print("-" * 80)
    if profile_data and graph_db:
        graph_db.ingest_company_profile(profile_data)
        
    process_transcripts_for_graph(ticker)
    
    graph_context = ""
    if graph_db:
        print("  🧠 Retrieving Causal-Temporal Context from Graph...")
        graph_context = graph_db.retrieve_all_context(ticker)
        print(f"  ✅ Context loaded ({len(graph_context)} chars)")

    # Analysis
    print("\n" + "=" * 80)
    print("🤖 ANALYZING WITH AI...")
    print("=" * 80)
    
    analysis = analyze_with_ai(
        calendar_data, surprise_data, profile_data, financials_data,
        av_earnings, av_income, av_balance, av_cashflow, av_overview,
        graph_context, user_query
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
    print("📊 CONTEXT-ENRICHED AGENTIC RAG PIPELINE")
    print("="*80)
    print("Powered by: Finnhub + Alpha Vantage + Neo4j + Azure OpenAI")
    
    # Get ticker from user
    ticker = "NVDA"
    print(f"\n🚀 Analyzing: {ticker}")
    
    # Optional: Get specific question from user
    print("\nEnter a specific question (or press Enter for comprehensive analysis):")
    user_query = input("> ").strip()
    
    if not user_query:
        user_query = None
    
    # Run the RAG pipeline
    rag_pipeline(ticker, user_query)
    
    # Close Graph Connection
    if graph_db:
        graph_db.close()


if __name__ == "__main__":
    main()
