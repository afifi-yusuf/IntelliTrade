import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import glob
from datetime import datetime

# Import Graph & Signal Agents
from graph_manager import Neo4jHandler
from vectorizer import NarrativeVectorizer
from rl_agent import MarkovRLAgent

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
except:
    graph_db = None
    print("⚠️ Neo4j not connected.")

DATA_DIR = "data"

# ============================================================================
# DATA LOADING (LOCAL)
# ============================================================================
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
    except:
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

# ============================================================================
# SIGNAL GENERATION & ANALYSIS
# ============================================================================


def generate_strategic_summary(transcript_text: str, graph_context: str) -> str:
    """
    Generate a high-density strategic summary for vectorization,
    Enriched by Neo4j Historical Context.
    """
    system_prompt = """You are a Strategic Signal Compressor.
    Summarize the transcript in strict 50 words.
    
    CRITICAL: Incorporate the 'Historical Context' provided.
    If the current transcript mentions an issue (e.g. Supply Chain) that also appeared in the History, emphasize it as a RECURRING pattern.
    
    Focus ONLY on: Growth Drivers, Macro Headwinds, and Demand durability."""
    
    user_prompt = f"""
    === HISTORICAL CONTEXT (NEO4J) ===
    {graph_context[:2000] if graph_context else "No history available."}
    ==================================
    
    === CURRENT TRANSCRIPT ===
    {transcript_text[:10000]}
    """
    
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def process_transcripts_and_get_summaries(ticker: str) -> list:
    """
    Process transcripts to get Strategic Summaries.
    CHECK NEO4J FIRST. If missing, generate (with context) and Store.
    """
    print(f"\n🧠 Processing Transcripts for Signal Generation...")
    
    # Pattern matching
    pattern = os.path.join(DATA_DIR, f"{ticker.upper()}_earnings_transcript_*.txt")
    files = glob.glob(pattern)
    if not files: # try lower
        pattern = os.path.join(DATA_DIR, f"{ticker.lower()}_earnings_transcript_*.txt")
        files = glob.glob(pattern)
        
    if not files:
        print("   ⚠️ No transcripts found.")
        return []
        
    files.sort(reverse=True) # Newest first
    summaries = []
    
    # We process the top 3-4 to build enough context for the vectorizer
    # In a real app, this would be cached.
    for filename in files[:4]:
        try:
            # Extract Quarter info
            parts = os.path.basename(filename).replace(".txt", "").split("_")
            year, quarter = parts[-2], parts[-1]
            quarter_id = f"{year}-{quarter}"
            
            # 1. CHECK GRAPH CACHE
            if graph_db:
                cached_summary = graph_db.retrieve_quarterly_summary(ticker, quarter_id)
                if cached_summary:
                    print(f"   ✅ [Graph Cache] Found summary for {quarter_id}")
                    summaries.append(cached_summary)
                    continue

            # 2. GENERATE NEW (Enriched)
            print(f"   ⚙️  Generative New Summary for {quarter_id}...")
            
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            if len(text) < 100: continue
            
            # Retrieve Context (What happened BEFORE this quarter?)
            # Ideally we'd filter context to be strictly < current date, but for now we take general history
            context = ""
            if graph_db:
                context = graph_db.retrieve_all_context(ticker)

            # Generate Summary
            summary = generate_strategic_summary(text, context)
            
            if summary:
                summaries.append(summary)
                # 3. STORE IN GRAPH
                if graph_db:
                    graph_db.ingest_quarterly_summary(ticker, quarter_id, summary)
            
        except Exception as e:
            print(f"   Error reading {filename}: {e}")
            
    return summaries


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
    except:
        return "Analysis failed."


def main():
    print("="*80)
    print("🤖 INTELLISENSE: QUANTITATIVE SIGNAL AGENT")
    print("="*80)
    
    ticker = "NVDA"
    print(f"🚀 Target: {ticker}")
    
    # 1. Load Financials
    print("   📂 Loading Financial Data...")
    profile = fetch_company_profile(ticker)
    metrics = fetch_financial_metrics(ticker)
    
    if not profile:
        print("   ❌ Data not found.")
        return

    # 2. Vectorization Pipeline
    vectorizer = NarrativeVectorizer()
    rl_agent = MarkovRLAgent()
    
    summaries = process_transcripts_and_get_summaries(ticker)
    
    if summaries:
        print(f"   📊 Vectorizing {len(summaries)} Historical Narratives...")
        vectorizer.fit(summaries)
        
        # Current State
        current_summary = summaries[0] # Most recent
        narrative_state = vectorizer.get_narrative_state(current_summary)
        print(f"   🧩 Detected Narrative State: {narrative_state}")
        
        # 3. RL Signal
        print("   🧠 Markov Agent Evaluating Policy...")
        signal = rl_agent.get_signal(narrative_state, metrics)
        
        print("\n" + "="*40)
        print(f"📢 FINAL SIGNAL: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.2f}")
        print(f"   Rationale: {signal['rationale']}")
        print("="*40 + "\n")
        
        # 4. Final Memo
        memo = analyze_with_ai(profile, metrics, signal, None)
        print("📝 INVESTMENT MEMO:")
        print(memo)
        
    else:
        print("❌ Could not generate signal (missing transcripts).")

if __name__ == "__main__":
    main()
