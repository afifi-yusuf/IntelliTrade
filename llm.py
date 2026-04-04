import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import glob
from datetime import datetime

# Import Graph & Signal Agents
from graph_manager import Neo4jHandler
from vectorizer import NarrativeVectorizer
from rl_agent import MarkovRLAgent
from agents import FactExtractorAgent, PeerAnalystAgent, HistoricalAnalystAgent, SynthesizerAgent

# Import sector mapping for graph initialization
from agents.peer_analyst import SECTOR_MAP
from paths import DATA_DIR

load_dotenv()

# Initialize Client
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)

# Initialize Neo4j
try:
    graph_db = Neo4jHandler()
    # Initialize enhanced graph schema for multi-agent system
    if graph_db and graph_db.driver:
        # Setup vector index for semantic search on facts
        graph_db.setup_vector_index(embedding_dim=384)
        # Create Sector nodes and link tickers
        for ticker, sector in SECTOR_MAP.items():
            graph_db.ingest_sector(ticker, sector)
        # Create temporal edges between quarters
        graph_db.setup_temporal_edges()
except Exception:
    graph_db = None
    print("⚠️ Neo4j not connected.")


def load_csv_data(ticker: str, file_suffix: str) -> dict:
    filename = os.path.join(DATA_DIR, f"{ticker.upper()}_{file_suffix}")
    if not os.path.exists(filename):
        # Try lowercase
        filename_lower = os.path.join(DATA_DIR, f"{ticker.lower()}_{file_suffix}")
        if os.path.exists(filename_lower):
            filename = filename_lower
        else:
            return None
    try:
        df = pd.read_csv(filename)
        return df.to_dict(orient="records")
    except Exception:
        return None


def fetch_company_profile(ticker: str) -> dict:
    data = load_csv_data(ticker, "company_overview.csv")
    return data[0] if data else {}

def fetch_financial_metrics(ticker: str) -> dict:
    # Combining overview and basic financials for the agent
    data = load_csv_data(ticker, "company_overview.csv")
    surprise = load_csv_data(ticker, "earnings_quarterly.csv")
    return {
        "financials": {"metric": data[0] if data else {}},
        "surprise": surprise if surprise else []
    }

def fetch_stock_price_history(ticker: str) -> pd.DataFrame:
    """
    Load stock price history for technical analysis.
    Tries daily data first, then weekly if daily not available.
    """
    # Try daily data (5 years)
    filename = os.path.join(DATA_DIR, f"{ticker.upper()}_daily_stock_data_5y.csv")
    if not os.path.exists(filename):
        filename = os.path.join(DATA_DIR, f"{ticker.lower()}_daily_stock_data_5y.csv")
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            # Sort by date to ensure chronological order
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            elif 'date' in df.columns:
                df = df.sort_values('date')
            return df
        except Exception as e:
            print(f"   ⚠️ Error loading daily data: {e}")
    
    # Fallback to weekly data
    filename = os.path.join(DATA_DIR, f"{ticker.upper()}_weekly_stock_data.csv")
    if not os.path.exists(filename):
        filename = os.path.join(DATA_DIR, f"{ticker.lower()}_weekly_stock_data.csv")
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            elif 'date' in df.columns:
                df = df.sort_values('date')
            return df
        except Exception as e:
            print(f"   ⚠️ Error loading weekly data: {e}")
    
    return None

def get_economic_data_for_quarter(quarter_id: str, current_date: datetime = None) -> dict:
    """
    Load economic indicators for a specific quarter.
    quarter_id format: "2025-Q4"
    """
    year, quarter = quarter_id.split("-")
    year = int(year)
    quarter_num = int(quarter[1])
    
    # Calculate approximate date range for the quarter
    quarter_months = {
        1: (1, 3),   # Q1: Jan-Mar
        2: (4, 6),   # Q2: Apr-Jun  
        3: (7, 9),   # Q3: Jul-Sep
        4: (10, 12)  # Q4: Oct-Dec
    }
    start_month, end_month = quarter_months[quarter_num]
    
    # Enforce availability: data assumed available ~1 month after quarter end
    if current_date is not None:
        if end_month == 12:
            reporting_date = datetime(year + 1, 1, 31)
        else:
            reporting_date = datetime(year, end_month + 1, 30)
        if current_date <= reporting_date:
            return {}
    
    def _filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
        if current_date is None:
            return df
        return df[df['date'] <= pd.Timestamp(current_date)]
    
    # Load economic data files
    economic_data = {}
    
    try:
        # CPI
        cpi_df = pd.read_csv(os.path.join(DATA_DIR, "economic_cpi.csv"))
        cpi_df['date'] = pd.to_datetime(cpi_df['date'], errors='coerce')
        # Convert value column to numeric, coercing errors to NaN
        cpi_df['value'] = pd.to_numeric(cpi_df['value'], errors='coerce')
        cpi_df = _filter_by_date(cpi_df)
        cpi_value = cpi_df[
            (cpi_df['date'].dt.year == year) & 
            (cpi_df['date'].dt.month >= start_month) & 
            (cpi_df['date'].dt.month <= end_month)
        ]['value'].mean()
        economic_data['cpi'] = float(cpi_value) if not pd.isna(cpi_value) else None
        
        # Inflation
        inflation_df = pd.read_csv(os.path.join(DATA_DIR, "economic_inflation.csv"))
        inflation_df['date'] = pd.to_datetime(inflation_df['date'], errors='coerce')
        inflation_df['value'] = pd.to_numeric(inflation_df['value'], errors='coerce')
        inflation_df = _filter_by_date(inflation_df)
        inflation_value = inflation_df[
            (inflation_df['date'].dt.year == year)
        ]['value'].mean()
        economic_data['inflation'] = float(inflation_value) if not pd.isna(inflation_value) else None
        
        # Unemployment
        unemployment_df = pd.read_csv(os.path.join(DATA_DIR, "economic_unemployment.csv"))
        unemployment_df['date'] = pd.to_datetime(unemployment_df['date'], errors='coerce')
        unemployment_df['value'] = pd.to_numeric(unemployment_df['value'], errors='coerce')
        unemployment_df = _filter_by_date(unemployment_df)
        unemployment_value = unemployment_df[
            (unemployment_df['date'].dt.year == year) & 
            (unemployment_df['date'].dt.month >= start_month) & 
            (unemployment_df['date'].dt.month <= end_month)
        ]['value'].mean()
        economic_data['unemployment'] = float(unemployment_value) if not pd.isna(unemployment_value) else None
        
        # GDP (annual data)
        gdp_df = pd.read_csv(os.path.join(DATA_DIR, "economic_real_gdp.csv"))
        gdp_df['date'] = pd.to_datetime(gdp_df['date'], errors='coerce')
        gdp_df['value'] = pd.to_numeric(gdp_df['value'], errors='coerce')
        gdp_df = _filter_by_date(gdp_df)
        gdp_value = gdp_df[
            (gdp_df['date'].dt.year == year)
        ]['value'].mean()
        economic_data['gdp'] = float(gdp_value) if not pd.isna(gdp_value) else None
        
        # Federal Funds Rate
        fed_df = pd.read_csv(os.path.join(DATA_DIR, "economic_federal_funds_rate.csv"))
        fed_df['date'] = pd.to_datetime(fed_df['date'], errors='coerce')
        fed_df['value'] = pd.to_numeric(fed_df['value'], errors='coerce')
        fed_df = _filter_by_date(fed_df)
        fed_value = fed_df[
            (fed_df['date'].dt.year == year) & 
            (fed_df['date'].dt.month >= start_month) & 
            (fed_df['date'].dt.month <= end_month)
        ]['value'].mean()
        economic_data['federal_funds_rate'] = float(fed_value) if not pd.isna(fed_value) else None
        
        # Treasury Yield
        treasury_df = pd.read_csv(os.path.join(DATA_DIR, "economic_treasury_yield_10y.csv"))
        treasury_df['date'] = pd.to_datetime(treasury_df['date'], errors='coerce')
        treasury_df['value'] = pd.to_numeric(treasury_df['value'], errors='coerce')
        treasury_df = _filter_by_date(treasury_df)
        treasury_value = treasury_df[
            (treasury_df['date'].dt.year == year) & 
            (treasury_df['date'].dt.month >= start_month) & 
            (treasury_df['date'].dt.month <= end_month)
        ]['value'].mean()
        economic_data['treasury_yield'] = float(treasury_value) if not pd.isna(treasury_value) else None
        
    except Exception as e:
        print(f"   ⚠️ Error loading economic data for {quarter_id}: {e}")
    
    return economic_data

def load_earnings_transcript(ticker: str, quarter_id: str, current_date: datetime = None) -> str:
    """
    Load earnings transcript for a specific quarter.
    """
    year, quarter = quarter_id.split("-")
    year = int(year)
    quarter_num = int(quarter[1])
    
    # Enforce availability: transcript assumed available ~1 month after quarter end
    if current_date is not None:
        quarter_months = {
            1: (1, 3),
            2: (4, 6),
            3: (7, 9),
            4: (10, 12)
        }
        _, end_month = quarter_months[quarter_num]
        if end_month == 12:
            reporting_date = datetime(year + 1, 1, 31)
        else:
            reporting_date = datetime(year, end_month + 1, 30)
        if current_date <= reporting_date:
            return ""
    
    # Try different filename formats
    patterns = [
        f"{ticker.upper()}_earnings_transcript_{year}_{quarter}.txt",
        f"{ticker.lower()}_earnings_transcript_{year}_{quarter}.txt"
    ]
    
    for pattern in patterns:
        filename = os.path.join(DATA_DIR, pattern)
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:
                        return content
            except Exception as e:
                print(f"   ⚠️ Error reading transcript: {e}")
    
    return ""

def build_comprehensive_context(ticker: str, quarter_id: str, 
                                financial_metrics: dict, current_date: datetime = None) -> str:
    """
    Build comprehensive context including:
    - Economic indicators
    - Earnings transcript
    - Financial metrics
    """
    # Get economic data
    economic_data = get_economic_data_for_quarter(quarter_id, current_date=current_date)
    
    # Get transcript
    transcript = load_earnings_transcript(ticker, quarter_id, current_date=current_date)
    
    # Build context string
    context = f"""
=== QUARTERLY CONTEXT FOR {ticker} {quarter_id} ===

## MACROECONOMIC INDICATORS
- CPI (Consumer Price Index): {economic_data.get('cpi', 'N/A')}
- Inflation Rate: {economic_data.get('inflation', 'N/A')}%
- Unemployment Rate: {economic_data.get('unemployment', 'N/A')}%
- Real GDP: {economic_data.get('gdp', 'N/A')}
- Federal Funds Rate: {economic_data.get('federal_funds_rate', 'N/A')}%
- 10Y Treasury Yield: {economic_data.get('treasury_yield', 'N/A')}%

## FINANCIAL METRICS
"""
    
    # Add financial metrics
    if financial_metrics:
        fin_data = financial_metrics.get('financials', {}).get('metric', {})
        for key, value in list(fin_data.items())[:10]:  # Limit to key metrics
            context += f"- {key}: {value}\n"
    
    context += f"\n## EARNINGS TRANSCRIPT\n{transcript[:15000]}\n"
    
    return context, economic_data, transcript


def generate_llm_analysis(comprehensive_context: str, graph_context: str) -> str:
    """
    Generate LLM analysis from comprehensive quarterly context.
    This is the core of the Context Enriched Agentic RAG approach.
    """
    system_prompt = """You are an Expert Investment Analyst with access to comprehensive quarterly data.

Analyze the provided context which includes:
- Macroeconomic indicators (CPI, inflation, unemployment, GDP, interest rates)
- Earnings transcript with management commentary
- Financial metrics and fundamentals

Provide a strategic analysis (200-300 words) covering:
1. Revenue and Growth Trajectory
2. Macro Environment Impact (how economic conditions affect the company)
3. Key Business Drivers and Catalysts
4. Risk Factors and Headwinds
5. Forward-Looking Sentiment

CRITICAL: If historical context is provided, identify RECURRING patterns and trends."""
    
    user_prompt = f"""
=== HISTORICAL CONTEXT (PREVIOUS QUARTERS) ===
{graph_context[:3000] if graph_context else "No historical data available."}
=========================================

{comprehensive_context}

Provide your strategic analysis:"""
    
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   ❌ Error generating LLM analysis: {e}")
        return ""


def generate_agentic_analysis(
    ticker: str, quarter: str, transcript: str,
    economic_data: dict, financial_metrics: dict,
    graph_db_instance=None, vectorizer=None, read_only: bool = False
) -> dict:
    """
    Multi-Agent Agentic RAG Pipeline:
    1. FactExtractor agent extracts structured facts from transcript
    2. Facts stored in Neo4j with embeddings
    3. PeerAnalyst queries graph for peer comparison
    4. HistoricalAnalyst queries graph for historical trends
    5. Synthesizer combines everything → narrative + structured signals
    
    Returns: {
        "narrative": str,
        "signals": {"revenue_trend": float, "margin_trend": float, ...},
        "facts": List[Dict],
        "peer_analysis": str,
        "historical_analysis": str
    }
    """
    gdb = graph_db_instance or graph_db
    default_signals = {
        "revenue_trend": 0.0, "margin_trend": 0.0, "peer_relative": 0.0,
        "historical_momentum": 0.0, "risk_level": 0.5, "overall_sentiment": 0.0
    }
    
    def _clamp_signals(raw_signals: dict) -> dict:
        bounded = default_signals.copy()
        for key, default_value in default_signals.items():
            try:
                value = float(raw_signals.get(key, default_value))
            except Exception:
                value = default_value
            lo, hi = (0.0, 1.0) if key == "risk_level" else (-1.0, 1.0)
            bounded[key] = float(np.clip(value, lo, hi))
        return bounded

    def _delegate_fact(fact: dict) -> list:
        category = str(fact.get("category", "")).lower()
        if category in {"revenue", "profitability", "growth"}:
            return ["historical_performance", "historical_earnings"]
        if category == "guidance":
            return ["historical_earnings"]
        if category == "risk":
            return ["competitor", "historical_earnings"]
        return ["historical_performance"]

    def _score_from_text(text: str) -> float:
        if not text:
            return 0.0
        return float(np.clip(len(text) / 800.0, 0.0, 1.0))

    # Initialize agents (all share the same azure client)
    fact_extractor = FactExtractorAgent(azure_client=azure_client)
    peer_analyst = PeerAnalystAgent(graph_db=gdb, azure_client=azure_client)
    historical_analyst = HistoricalAnalystAgent(graph_db=gdb, azure_client=azure_client)
    synthesizer = SynthesizerAgent(azure_client=azure_client)
    
    # ---- Agent 1: Extract Facts ----
    print(f"   🔬 [FactExtractor] Extracting structured facts...")
    facts = fact_extractor.extract(transcript, ticker, quarter)
    print(f"   ✅ Extracted {len(facts)} facts")

    # Delegation map (paper-style routing metadata)
    delegation = {}
    for fact in facts:
        fact_id = fact.get("fact_id", "")
        delegation[fact_id] = _delegate_fact(fact)
    
    # Store facts in Neo4j with embeddings (skip writes in read-only mode)
    if gdb and vectorizer and facts and not read_only:
        for fact in facts:
            try:
                fact_text = f"{fact['metric']}: {fact['value']}"
                if fact.get('context'):
                    fact_text += f" — {fact['context']}"
                fact_embedding = vectorizer.get_embedding(fact_text)
                if fact_embedding is not None:
                    gdb.ingest_fact_with_embedding(
                        fact, 
                        fact_embedding.tolist() if hasattr(fact_embedding, 'tolist') else fact_embedding
                    )
            except Exception as e:
                print(f"   ⚠️ Could not store fact in graph: {e}")
    
    # ---- Agent 2: Peer Comparison ----
    print(f"   🏢 [PeerAnalyst] Comparing against sector peers...")
    peer_result = peer_analyst.analyze(facts, ticker, quarter, embedder=vectorizer)
    print(f"   ✅ Peer score: {peer_result.get('peer_score', 0):+.2f}")
    
    # ---- Agent 3: Historical Analysis ----
    print(f"   📈 [HistoricalAnalyst] Analyzing multi-quarter trends...")
    historical_result = historical_analyst.analyze(facts, ticker, quarter, embedder=vectorizer)
    print(f"   ✅ Momentum: {historical_result.get('momentum', 0):+.2f}")
    
    # ---- Agent 4: Synthesize Everything ----
    print(f"   🧠 [Synthesizer] Combining all analyses...")
    synthesis = synthesizer.synthesize(
        ticker=ticker,
        quarter=quarter,
        facts=facts,
        peer_result=peer_result,
        historical_result=historical_result,
        economic_data=economic_data
    )
    
    signals = _clamp_signals(synthesis.get("signals", default_signals))
    narrative = synthesis.get("narrative", "")

    peer_score = float(np.clip(peer_result.get("peer_score", 0.0), -1.0, 1.0))
    hist_momentum = float(np.clip(historical_result.get("momentum", 0.0), -1.0, 1.0))
    synth_sentiment = float(np.clip(signals.get("overall_sentiment", 0.0), -1.0, 1.0))
    directional = [peer_score, hist_momentum, synth_sentiment]
    same_direction = [np.sign(v) for v in directional if abs(v) >= 0.05]
    if same_direction:
        agreement = abs(sum(same_direction)) / len(same_direction)
    else:
        agreement = 0.5

    evidence_count = int(
        len(facts)
        + len(peer_result.get("peer_facts", []) or [])
        + len(historical_result.get("historical_facts", []) or [])
    )
    evidence_strength = float(np.clip(evidence_count / 30.0, 0.0, 1.0))
    parse_success = 1.0 if narrative and narrative.strip() else 0.0
    agent_confidences = {
        "peer_analyst": float(np.clip(abs(peer_score), 0.0, 1.0)),
        "historical_analyst": float(np.clip(abs(hist_momentum), 0.0, 1.0)),
        "synthesizer": _score_from_text(narrative),
    }
    reliability = {
        "evidence_count": evidence_count,
        "evidence_strength": evidence_strength,
        "agent_agreement": float(np.clip(agreement, 0.0, 1.0)),
        "parse_success": parse_success,
        "agent_confidences": agent_confidences,
    }

    print(f"   ✅ Synthesis complete | Sentiment: {signals.get('overall_sentiment', 0):+.2f} | Risk: {signals.get('risk_level', 0.5):.2f}")
    
    return {
        "narrative": narrative,
        "signals": signals,
        "reliability": reliability,
        "delegation": delegation,
        "facts": facts,
        "peer_analysis": peer_result.get("analysis", ""),
        "historical_analysis": historical_result.get("analysis", ""),
        "source_quarter": quarter
    }

def process_quarterly_context_and_generate_analysis(ticker: str, num_quarters: int = 3) -> tuple:
    """
    Context Enriched Agentic RAG Pipeline:
    1. Build comprehensive context (economic + transcript + financial data)
    2. Feed to LLM for analysis
    3. Vectorize LLM output (embeddings)
    4. Store in Neo4j
    5. Return analyses, embeddings, and vectorizer for RL agent
    
    Returns: (results_list, vectorizer_instance)
    """
    print(f"\n🧠 Context Enriched RAG: Processing Quarterly Data...")
    
    # Initialize vectorizer once for all quarters
    print(f"   🔢 Initializing vectorizer...")
    vectorizer = NarrativeVectorizer()
    
    # Find available transcripts
    pattern = os.path.join(DATA_DIR, f"{ticker.upper()}_earnings_transcript_*.txt")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(DATA_DIR, f"{ticker.lower()}_earnings_transcript_*.txt")
        files = glob.glob(pattern)
        
    if not files:
        print("   ⚠️ No transcripts found.")
        return [], vectorizer
        
    files.sort(reverse=True)  # Newest first
    results = []
    
    for filename in files[:num_quarters]:
        try:
            # Extract quarter ID
            parts = os.path.basename(filename).replace(".txt", "").split("_")
            year, quarter = parts[-2], parts[-1]
            quarter_id = f"{year}-{quarter}"
            
            print(f"\n   📊 Processing Quarter: {quarter_id}")
            
            # 1. CHECK NEO4J CACHE
            if graph_db:
                cached_data = graph_db.retrieve_quarterly_summary(ticker, quarter_id)
                if cached_data and cached_data.get('embedding'):
                    print(f"   ✅ [Cache Hit] Using stored analysis for {quarter_id}")
                    results.append({
                        'quarter_id': quarter_id,
                        'analysis': cached_data['text'],
                        'embedding': np.array(cached_data['embedding'])
                    })
                    continue
            
            # 2. BUILD COMPREHENSIVE CONTEXT
            print(f"   📦 Building comprehensive context...")
            metrics = fetch_financial_metrics(ticker)
            context, economic_data, transcript = build_comprehensive_context(
                ticker, quarter_id, metrics
            )
            
            # Store raw context in Neo4j first
            if graph_db and len(transcript) > 100:
                graph_db.ingest_quarterly_context(
                    ticker, quarter_id, transcript, 
                    economic_data, metrics
                )
            
            # 3. GET HISTORICAL CONTEXT FROM GRAPH
            historical_context = ""
            if graph_db:
                historical_context = graph_db.retrieve_all_context(ticker, max_quarter=quarter_id)
            
            # 4. GENERATE LLM ANALYSIS
            print(f"   🤖 Generating LLM analysis...")
            llm_analysis = generate_llm_analysis(context, historical_context)
            
            if not llm_analysis:
                print(f"   ⚠️ Failed to generate analysis for {quarter_id}")
                continue
            
            # 5. VECTORIZE LLM OUTPUT (this is the key!)
            print(f"   🔢 Vectorizing LLM output...")
            embedding = vectorizer.get_embedding(llm_analysis)
            
            # 6. STORE IN NEO4J
            if graph_db:
                graph_db.ingest_quarterly_summary(
                    ticker, quarter_id, llm_analysis, 
                    embedding.tolist()
                )
            
            results.append({
                'quarter_id': quarter_id,
                'analysis': llm_analysis,
                'embedding': embedding
            })
            
            print(f"   ✅ Completed {quarter_id}")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    return results, vectorizer


def analyze_with_ai(profile, metrics, signal_data, user_query):
    """Final Investment Memo based on Signal."""
    
    system_prompt = """You are a Portfolio Manager acting on a Quantitative Signal.
    You have received a Signal from the 'Markov Graph Agent'.
    Write a brief Investment Memo justifying this signal."""
    
    user_prompt = f"""
    TICKER: {profile.get('Symbol', 'Unknown')}
    
    === QUANT SIGNAL ===
    ACTION: {signal_data.get('action')}
    CONFIDENCE: {signal_data.get('confidence')}
    RATIONALE: {signal_data.get('rationale')}
    ====================
    
    FINANCIAL CONTEXT:
    PE Ratio: {metrics.get('financials', {}).get('metric', {}).get('PEData', 'N/A')}
    Profit Margin: {metrics.get('financials', {}).get('metric', {}).get('ProfitMargin', 'N/A')}
    
    Please write a concise memo."""
    
    if user_query:
        user_prompt += f"\n User Note: {user_query}"
        
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception:
        return "Analysis failed."


def main():
    print("="*80)
    print("🤖 INTELLISENSE: CONTEXT ENRICHED AGENTIC RAG SYSTEM")
    print("="*80)
    
    tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "ASML", "TSM"]
    
    for ticker in tickers:
        print("\n" + "#"*80)
        print(f"🚀 Processing Target: {ticker}")
        print("#"*80)
        
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

def process_ticker(ticker):
    
    # 1. Load Financials
    print("\n📂 Loading Financial Data...")
    profile = fetch_company_profile(ticker)
    metrics = fetch_financial_metrics(ticker)
    
    if not profile:
        print("   ❌ Data not found.")
        return

    # 2. Load Stock Price History
    print("\n📈 Loading Stock Price History...")
    price_history = fetch_stock_price_history(ticker)
    if price_history is not None:
        print(f"   ✅ Loaded {len(price_history)} data points")
    else:
        print("   ⚠️ No price history found, proceeding without technical indicators")

    # 3. CONTEXT ENRICHED RAG PIPELINE
    print("\n" + "="*80)
    print("🔄 CONTEXT ENRICHED AGENTIC RAG PIPELINE")
    print("="*80)
    
    # Process quarterly contexts -> LLM analysis -> Vectorization
    quarterly_results, vectorizer = process_quarterly_context_and_generate_analysis(ticker, num_quarters=3)
    
    if not quarterly_results:
        print("❌ Could not process quarterly data.")
        return
    
    # Get most recent analysis
    current_result = quarterly_results[0]
    current_embedding = current_result['embedding']
    current_analysis = current_result['analysis']
    
    print(f"\n📊 LLM Analysis Summary (Most Recent Quarter):")
    print("="*80)
    print(current_analysis[:500] + "...")
    print("="*80)
    
    # 4. Initialize RL Agent (vectorizer already created and returned)
    embedding_dim = len(current_embedding)
    print(f"\n🔢 Embedding Dimension: {embedding_dim}")
    
    rl_agent = MarkovRLAgent(narrative_dim=embedding_dim)
    
    # Fit vectorizer on historical embeddings
    if len(quarterly_results) >= 3:
        historical_analyses = [r['analysis'] for r in quarterly_results]
        historical_embeddings = [r['embedding'] for r in quarterly_results]
        vectorizer.fit(historical_analyses, historical_embeddings)
    
    # Get narrative state from embedding
    narrative_state = vectorizer.get_narrative_state(
        text=current_analysis, 
        embedding=current_embedding
    )
    print(f"   🧩 Detected Narrative State: {narrative_state} (0=Bullish, 1=Neutral, 2=Bearish)")
    
    # 5. RL AGENT SIGNAL GENERATION
    print("\n🧠 RL Agent: Generating Trading Signal...")
    print("   Inputs: LLM Embedding + Stock Price History + Financial Metrics")
    
    signal = rl_agent.get_signal(
        narrative_vector=current_embedding,
        price_history=price_history,
        financial_data=metrics,
        narrative_state=narrative_state
    )
    
    # 6. DISPLAY RESULTS
    print("\n" + "="*80)
    print("📢 FINAL TRADING SIGNAL")
    print("="*80)
    print(f"   ACTION: {signal['action']}")
    print(f"   CONFIDENCE: {signal['confidence']:.2f}")
    print(f"\n   RATIONALE: {signal['rationale']}")
    
    if 'stock_features' in signal:
        features = signal['stock_features']
        print(f"\n   📊 Technical Indicators:")
        print(f"      • Momentum (5d-20d avg): {features['momentum']:.2f}%")
        print(f"      • Trend (SMA20/SMA50): {features['trend']:.2f}%")
        print(f"      • RSI (14-day): {features['rsi']:.1f}")
        print(f"      • Volatility: {features['volatility']:.2f}%")
        print(f"      • 1-day change: {features['price_change_1d']:.2f}%")
        print(f"      • 5-day change: {features['price_change_5d']:.2f}%")
        print(f"      • 20-day change: {features['price_change_20d']:.2f}%")
    print("="*80 + "\n")
    
    # 7. Final Investment Memo
    print("📝 GENERATING INVESTMENT MEMO...")
    memo = analyze_with_ai(profile, metrics, signal, None)
    print("\n" + "="*80)
    print("INVESTMENT MEMO")
    print("="*80)
    print(memo)
    print("="*80)

if __name__ == "__main__":
    main()
