# IntelliSense: Context Enriched Agentic RAG Architecture

## Overview

IntelliSense implements a **Context Enriched Agentic RAG** (Retrieval-Augmented Generation) system that combines:
- Comprehensive quarterly context (economic + earnings + financial data)
- LLM-powered strategic analysis
- Embedding-based vectorization (HuggingFace or Azure OpenAI)
- RL agent for trading signal generation

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Per Quarter)                    │
├─────────────────────────────────────────────────────────────────┤
│  • Economic Indicators (CPI, Inflation, GDP, Fed Rate, etc.)    │
│  • Earnings Transcripts (Management Commentary)                 │
│  • Financial Metrics (Balance Sheet, Income Statement, etc.)    │
│  • Stock Price History (Daily/Weekly OHLCV data)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              COMPREHENSIVE CONTEXT BUILDER                       │
│  (build_comprehensive_context)                                  │
│                                                                 │
│  Aggregates quarterly data:                                     │
│  • Economic: CPI, Inflation, Unemployment, GDP, Rates           │
│  • Transcript: Full earnings call text (~50 pages)              │
│  • Financial: Revenue, Margins, Growth, Surprises               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEO4J KNOWLEDGE GRAPH                         │
│  • Stores raw quarterly contexts                                │
│  • Links economic data → quarters                               │
│  • Retrieves historical context (previous quarters)             │
│  • Caches LLM analyses and embeddings                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM STRATEGIC ANALYSIS                        │
│  (generate_llm_analysis)                                        │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ INPUTS TO LLM:                                         │    │
│  │ 1. Comprehensive Context (current quarter):           │    │
│  │    • Macroeconomic indicators                         │    │
│  │    • Full earnings transcript                         │    │
│  │    • Financial metrics                                │    │
│  │ 2. Historical Context (from Neo4j):                   │    │
│  │    • Previous quarters' summaries                     │    │
│  │    • Recurring patterns/themes                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  OUTPUT: 200-300 word strategic analysis covering:              │
│    - Revenue and growth trajectory                              │
│    - Macro environment impact on business                       │
│    - Key business drivers and catalysts                         │
│    - Risk factors and headwinds                                 │
│    - Forward-looking sentiment                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING VECTORIZATION                         │
│  (NarrativeVectorizer.get_embedding)                           │
│                                                                 │
│  Converts LLM analysis text to dense vector:                    │
│    • HuggingFace (all-MiniLM-L6-v2): 384-dim [DEFAULT]         │
│    • Azure OpenAI (ada-002): 1536-dim [OPTIONAL]               │
│    • TF-IDF Fallback: 100-dim [BACKUP]                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RL AGENT (MarkovRLAgent)                      │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ INPUTS TO RL AGENT:                                    │    │
│  │ 1. LLM Embedding Vector (384-dim from HuggingFace)    │    │
│  │    → Semantic representation of LLM analysis          │    │
│  │                                                        │    │
│  │ 2. Stock Price History (DataFrame)                    │    │
│  │    → Daily/weekly OHLCV data (past 5 years)           │    │
│  │    → Generates: RSI, momentum, trend, volatility      │    │
│  │                                                        │    │
│  │ 3. Financial Metrics (dict)                           │    │
│  │    → Earnings surprises, margins, growth rates        │    │
│  │    → Discretized to financial state (Good/Weak)       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  PROCESSING STEPS:                                              │
│  1. Extract technical indicators from price history:            │
│     • RSI (14-day)                                              │
│     • Momentum (5d, 20d price changes)                          │
│     • Trend (SMA20 vs SMA50)                                    │
│     • Volatility (std of returns)                               │
│                                                                 │
│  2. Map embedding to narrative state via KMeans:                │
│     • 0 = Bullish (growth, strong, record keywords)             │
│     • 1 = Neutral (mixed signals)                               │
│     • 2 = Bearish (decline, weak, headwinds)                    │
│                                                                 │
│  3. Look up base Q-values from policy table:                    │
│     Q[(narrative_state, financial_state)] → {BUY, HOLD, SELL}  │
│                                                                 │
│  4. Adjust Q-values using technical indicators:                 │
│     • Strong momentum → favor BUY/SELL direction                │
│     • Uptrend → increase BUY score                              │
│     • RSI overbought (>70) → increase SELL score                │
│     • RSI oversold (<30) → increase BUY score                   │
│     • High volatility → favor HOLD                              │
│                                                                 │
│  5. Select action with max adjusted Q-value                     │
│                                                                 │
│  OUTPUT: Trading Signal + Rationale                             │
│    • Action: BUY / HOLD / SELL                                  │
│    • Confidence: 0.0 - 1.0                                      │
│    • Rationale: Detailed explanation                            │
│    • Stock Features: Technical indicator values                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL INVESTMENT MEMO                          │
│  (analyze_with_ai)                                              │
│                                                                 │
│  LLM synthesizes all signals into actionable recommendation     │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Input/Output Specifications

### LLM Inputs (Azure GPT)

**Function:** `generate_llm_analysis(comprehensive_context, graph_context)`

**Input 1 - Comprehensive Context (Current Quarter):**
```
=== QUARTERLY CONTEXT FOR NVDA 2025-Q4 ===

## MACROECONOMIC INDICATORS
- CPI (Consumer Price Index): 315.605
- Inflation Rate: 2.8%
- Unemployment Rate: 4.1%
- Real GDP: $21,500B
- Federal Funds Rate: 5.25%
- 10Y Treasury Yield: 4.35%

## FINANCIAL METRICS
- Revenue: $39.3B (+78% YoY)
- Gross Margin: 73%
- Net Profit Margin: 53%
- EPS Surprise: +8.5%
- Operating Cash Flow: $28.5B
... (10+ key metrics)

## EARNINGS TRANSCRIPT
[Full 50-page earnings call transcript]
- Management commentary
- Q&A with analysts
- Forward guidance
- Risk disclosures
```

**Input 2 - Historical Context (from Neo4j):**
```
### Causal-Temporal Knowledge Graph Context for NVDA

**2025-Q3**
- Revenue: $35.1B | Reason: Blackwell ramp began
- Margins: 74% | Reason: Product mix optimization
- Theme: Supply chain complexity mentioned

**2025-Q2**
- Revenue: $30.0B | Reason: Data center demand surge
- Margins: 72% | Reason: Hopper production scale
- Theme: Supply chain complexity mentioned [RECURRING]

**2025-Q1**
... (up to 5 previous quarters)
```

**Output:** 200-300 word strategic analysis
```
"NVIDIA's Q4 2025 demonstrates exceptional performance with revenue
of $39.3B (+78% YoY), driven by accelerating demand for Blackwell
architecture. Macroeconomic conditions remain supportive with
inflation moderating to 2.8% and stable Fed rates...

[Detailed multi-dimensional analysis]

Primary risk factors include supply chain complexity (RECURRING
theme across Q2-Q4) and near-term gross margin compression..."
```

---

### RL Agent Inputs (Markov RL Agent)

**Function:** `get_signal(narrative_vector, price_history, financial_data, narrative_state)`

**Input 1 - Narrative Vector (LLM Embedding):**
```python
narrative_vector = np.array([0.023, -0.142, 0.087, ..., 0.031])
# Shape: (384,) for HuggingFace, or (1536,) for Azure
# Type: float64
# Source: Embedding of LLM's strategic analysis text
```

**Input 2 - Stock Price History (DataFrame):**
```python
price_history = pd.DataFrame({
    'timestamp': ['2024-01-01', '2024-01-02', ...],
    'close': [495.22, 497.50, ...],  # 1253 rows (5 years daily)
    'volume': [50000000, 48000000, ...],
    # ... other OHLCV columns
})

# From this, RL agent extracts:
# - RSI (14-day): 68.2
# - Momentum (5d): +3.2%
# - Momentum (20d): +12.5%
# - Trend (SMA20/SMA50): +8.3%
# - Volatility (std): 2.4%
```

**Input 3 - Financial Data (dict):**
```python
financial_data = {
    'financials': {
        'metric': {
            'netProfitMarginTTM': 53.0,
            'PEData': 45.2,
            'ProfitMargin': 53.0,
            # ... other metrics
        }
    },
    'surprise': [
        {'surprisePercent': 8.5, 'quarter': '2025-Q4'},
        {'surprisePercent': 7.2, 'quarter': '2025-Q3'},
        # ... historical surprises
    ]
}

# From this, RL agent derives:
# - Financial State: 0 (Good) or 1 (Weak)
```

**Output:** Trading signal with detailed breakdown
```python
{
    "action": "BUY",
    "confidence": 0.92,
    "rationale": "State: Narrative=Bullish, Financials=Good. Stock Metrics: Momentum=+12.50%, Trend=+8.30%, RSI=68.2. Policy suggests BUY.",
    "stock_features": {
        "momentum": 12.5,
        "trend": 8.3,
        "rsi": 68.2,
        "volatility": 2.4,
        "price_change_1d": 0.04,
        "price_change_5d": -1.23,
        "price_change_20d": 5.67
    }
}
```

---

## Key Components

### 1. Quarterly Context Storage (Neo4j)

**graph_manager.py**

```python
# New structure for quarterly data
Quarter {
  id: "2025-Q4"
} -[:HAS_ECONOMIC_CONTEXT]-> EconomicContext {
  cpi, inflation, unemployment, gdp, fed_rate, treasury_yield
}

Company {
  ticker: "NVDA"
} -[:HAS_TRANSCRIPT]-> Transcript {
  content: "earnings call transcript..."
} -[:IN_QUARTER]-> Quarter

Company -[:HAS_METRICS]-> FinancialMetrics {
  data: {...}
} -[:IN_QUARTER]-> Quarter

Company -[:HAS_SUMMARY]-> Summary {
  text: "LLM analysis...",
  embedding: [1536-dim vector]
} -[:IN_QUARTER]-> Quarter
```

### 2. Context Builder

**Functions in main.py:**

- `get_economic_data_for_quarter(quarter_id)`: Loads CPI, inflation, unemployment, GDP, fed rate, treasury yield for a specific quarter
- `load_earnings_transcript(ticker, quarter_id)`: Loads earnings call transcript
- `build_comprehensive_context(ticker, quarter_id, financial_metrics)`: Combines all data sources into structured context

### 3. LLM Analysis Engine

**generate_llm_analysis(comprehensive_context, graph_context)**

Takes comprehensive quarterly context and generates strategic analysis:
- Analyzes macroeconomic impact on the company
- Identifies growth drivers and headwinds
- Assesses management sentiment from transcripts
- Incorporates historical patterns from Neo4j
- Produces 200-300 word investment-grade analysis

### 4. Embedding Vectorizer

**NarrativeVectorizer** (vectorizer.py)

- **Primary**: Azure OpenAI embeddings (1536 dimensions)
- **Fallback**: TF-IDF vectorization (100 dimensions)
- **Clustering**: KMeans to discretize into narrative states
- **Caching**: Embeddings cached to avoid redundant API calls

### 5. RL Trading Agent

**MarkovRLAgent** (rl_agent.py)

**State Space:**
- Narrative embedding (1536-dim continuous features from LLM)
- Stock technical features (RSI, momentum, trend, volatility)
- Financial state (discretized fundamentals)

**Action Space:**
- BUY (aggressive long position)
- HOLD (maintain current position)
- SELL (exit or short)

**Decision Process:**
1. Extract technical indicators from price history
2. Map embedding to discrete narrative state via clustering
3. Lookup base Q-values from heuristic policy table
4. Adjust Q-values based on technical signals:
   - Momentum: ±0.2 adjustment
   - Trend: ±0.15 adjustment
   - RSI (overbought/oversold): ±0.15 adjustment
   - Volatility: +0.1 to HOLD in high vol
5. Select action with highest adjusted Q-value

## Complete Data Flow Example

### Quarter: 2025-Q4, Ticker: NVDA

#### Step 1: Context Gathering (build_comprehensive_context)
```python
# Inputs collected:
economic_data = {
    'cpi': 315.605,
    'inflation': 2.8,
    'unemployment': 4.1,
    'gdp': 21500.0,
    'federal_funds_rate': 5.25,
    'treasury_yield': 4.35
}

transcript = """[50 pages of earnings call]
Management: Q4 revenue of $39.3B, up 78% YoY...
Analyst Q&A: Discussion of Blackwell ramp...
"""

financial_metrics = {
    'netProfitMarginTTM': 53.0,
    'surprisePercent': 8.5,
    'revenue': 39.3e9,
    # ... 10+ metrics
}

price_history = pd.DataFrame(...)  # 1253 days of OHLCV data

# Output: Comprehensive context string (15,000 characters)
```

#### Step 2: LLM Analysis (generate_llm_analysis)
```python
# LLM receives:
input_to_llm = f"""
=== HISTORICAL CONTEXT (PREVIOUS QUARTERS) ===
Q3: Revenue $35.1B, Blackwell ramp began...
Q2: Revenue $30.0B, Supply chain issues noted...
==================================

=== CURRENT QUARTERLY CONTEXT ===
## MACROECONOMIC INDICATORS
- CPI: 315.605
- Inflation: 2.8%
...

## FINANCIAL METRICS  
- Revenue: $39.3B (+78% YoY)
...

## EARNINGS TRANSCRIPT
{full_transcript}
"""

# LLM outputs:
llm_analysis = """
NVIDIA's Q4 2025 demonstrates exceptional performance with
revenue of $39.3B (+78% YoY), significantly exceeding guidance.
The Blackwell architecture ramp achieved $11B in its first quarter...

From a macroeconomic perspective, the moderating inflation
environment (2.8% vs 3.2% prior quarter) and stable Fed rates...

Primary risk factors include supply chain complexity (RECURRING
theme across Q2-Q4) and near-term gross margin compression...
[250 words total]
"""
```

#### Step 3: Vectorization (NarrativeVectorizer.get_embedding)
```python
# Input: LLM analysis text (250 words)
vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')

# Output: Dense embedding vector
embedding = vectorizer.get_embedding(llm_analysis)
# Shape: (384,) using HuggingFace
# Type: float64
# Values: [0.023, -0.142, 0.087, ..., 0.031]
```

#### Step 4: RL Agent Processing (MarkovRLAgent.get_signal)
```python
# Inputs prepared:
narrative_vector = embedding  # (384,) array
price_history = DataFrame(1253 rows, 5 years)
financial_data = {'financials': {...}, 'surprise': [...]}

# RL Agent processes:
signal = rl_agent.get_signal(
    narrative_vector=narrative_vector,
    price_history=price_history,
    financial_data=financial_data
)

# Internal processing:
# 1. Extract stock features from price_history:
stock_features = {
    'rsi': 68.2,               # From RSI calculation
    'momentum': 12.5,          # Average of 5d and 20d changes
    'trend': 8.3,              # SMA20/SMA50 comparison
    'volatility': 2.4,         # Std of returns
    'price_change_1d': 0.04,
    'price_change_5d': -1.23,
    'price_change_20d': 5.67
}

# 2. Map embedding to narrative state (via KMeans):
narrative_state = 0  # Bullish

# 3. Assess financial state:
financial_state = 0  # Good (positive surprises, healthy margins)

# 4. Look up base Q-values:
base_q = q_table[(0, 0)]  # (Bullish, Good)
# {BUY: 0.9, HOLD: 0.1, SELL: -0.8}

# 5. Adjust Q-values with technical indicators:
adjusted_q = {
    'BUY': 0.9 + 0.2 (momentum) + 0.15 (trend) = 1.25,
    'HOLD': 0.1 + 0.1 (volatility) = 0.2,
    'SELL': -0.8 - 0.2 (momentum) - 0.1 (trend) = -1.1
}

# 6. Select best action:
# Output:
signal = {
    "action": "BUY",
    "confidence": 0.92,
    "rationale": "State: Narrative=Bullish, Financials=Good. Stock Metrics: Momentum=+12.50%, Trend=+8.30%, RSI=68.2. Policy suggests BUY.",
    "stock_features": stock_features
}
```

#### Step 5: Storage in Neo4j
```cypher
# Store comprehensive context
MERGE (c:Company {ticker: "NVDA"})
MERGE (q:Quarter {id: "2025-Q4"})
MERGE (t:Transcript {id: "NVDA_2025-Q4"})
SET t.content = transcript
MERGE (ec:EconomicContext {id: "2025-Q4"})
SET ec.cpi = 315.605, ec.inflation = 2.8, ...

# Store LLM analysis and embedding
MERGE (s:Summary {id: "NVDA_2025-Q4"})
SET s.text = llm_analysis,
    s.embedding = [0.023, -0.142, ..., 0.031]
MERGE (c)-[:HAS_SUMMARY]->(s)-[:IN_QUARTER]->(q)
```

## Advantages of This Architecture

1. **Comprehensive Context**: Unlike simple transcript analysis, we incorporate macroeconomic conditions that directly impact business performance

2. **Semantic Understanding**: LLM analyzes complex relationships between macro factors, business drivers, and management sentiment

3. **Dense Vector Representation**: Embeddings capture nuanced semantic meaning better than TF-IDF

4. **Multi-Modal RL**: Agent considers narrative, technical, and fundamental signals simultaneously

5. **Knowledge Persistence**: Neo4j stores rich quarterly contexts for historical analysis

6. **Caching & Efficiency**: Avoids redundant LLM calls and embedding computations

7. **Temporal Patterns**: System identifies recurring themes across quarters (e.g., supply chain issues)

## Usage

```bash
# Set environment variables
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export AZURE_DEPLOYMENT="gpt-4"
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
export NEO4J_PASSWORD="your_password"

# Run the system
python main.py
```

## Output

The system produces:
1. **Quarterly LLM Analyses**: Strategic summaries for each quarter
2. **Embedding Vectors**: 1536-dimensional semantic representations
3. **Narrative States**: Discrete bullish/neutral/bearish classification
4. **Technical Indicators**: RSI, momentum, trend, volatility metrics
5. **Trading Signal**: BUY/HOLD/SELL with confidence score
6. **Investment Memo**: Final LLM-generated recommendation

## Future Enhancements

- **Real-time Updates**: Stream new economic data and earnings calls
- **Multi-Asset**: Extend to portfolios and sector analysis
- **Reinforcement Learning**: Train Q-values from actual market outcomes
- **Attention Mechanisms**: Weight different context components dynamically
- **Risk Management**: Incorporate position sizing and portfolio constraints
