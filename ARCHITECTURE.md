# IntelliSense: Context Enriched Agentic RAG Architecture

## Overview

IntelliSense now implements a **Context Enriched Agentic RAG** (Retrieval-Augmented Generation) system that combines:
- Comprehensive quarterly context (economic + earnings + financial data)
- LLM-powered strategic analysis
- Embedding-based vectorization
- RL agent for trading signal generation

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Per Quarter)                    │
├─────────────────────────────────────────────────────────────────┤
│  • Economic Indicators (CPI, Inflation, GDP, Fed Rate, etc.)    │
│  • Earnings Transcripts (Management Commentary)                 │
│  • Financial Metrics (Balance Sheet, Income Statement, etc.)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              COMPREHENSIVE CONTEXT BUILDER                       │
│  (build_comprehensive_context)                                  │
│                                                                 │
│  Combines all data sources into rich contextual input           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEO4J KNOWLEDGE GRAPH                         │
│  • Stores quarterly contexts                                    │
│  • Links economic data to quarters                              │
│  • Retrieves historical context                                 │
│  • Caches LLM analyses and embeddings                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM STRATEGIC ANALYSIS                        │
│  (generate_llm_analysis)                                        │
│                                                                 │
│  Input: Comprehensive Context + Historical Context              │
│  Output: 200-300 word strategic analysis covering:              │
│    - Revenue and growth trajectory                              │
│    - Macro environment impact                                   │
│    - Key business drivers                                       │
│    - Risk factors and headwinds                                 │
│    - Forward-looking sentiment                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING VECTORIZATION                         │
│  (NarrativeVectorizer.get_embedding)                           │
│                                                                 │
│  Converts LLM analysis to dense vector:                         │
│    - Azure OpenAI: 1536-dimensional embedding                   │
│    - Fallback: 100-dimensional TF-IDF                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RL AGENT (MarkovRLAgent)                      │
│                                                                 │
│  Multi-Modal Inputs:                                            │
│    1. LLM Embedding Vector (1536-dim)                           │
│    2. Stock Price History (technical indicators)                │
│    3. Financial Metrics (fundamentals)                          │
│                                                                 │
│  Processing:                                                    │
│    • Extracts stock features (RSI, momentum, trend, volatility) │
│    • Maps embedding to narrative state (Bullish/Neutral/Bearish)│
│    • Looks up base Q-values from policy table                   │
│    • Adjusts Q-values based on technical indicators             │
│                                                                 │
│  Output: Trading Signal (BUY/HOLD/SELL) + Confidence            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL INVESTMENT MEMO                          │
│  (analyze_with_ai)                                              │
│                                                                 │
│  LLM generates actionable investment recommendation             │
└─────────────────────────────────────────────────────────────────┘
```

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

## Data Flow Example

### Quarter: 2025-Q4, Ticker: NVDA

1. **Context Gathering:**
   ```
   Economic: CPI=296.5, Inflation=2.8%, Unemployment=4.1%
   Transcript: 50 pages of earnings call
   Financials: Revenue=$39.3B, GM=73%, etc.
   ```

2. **LLM Analysis:**
   ```
   "NVIDIA's Q4 2025 demonstrates exceptional performance with
   revenue of $39.3B (+78% YoY), driven by accelerating demand
   for Blackwell architecture. Macroeconomic conditions remain
   supportive with inflation moderating to 2.8%... [detailed analysis]"
   ```

3. **Vectorization:**
   ```
   embedding = [0.023, -0.142, 0.087, ..., 0.031]  # 1536 dims
   ```

4. **RL Agent Processing:**
   ```
   Narrative State: 0 (Bullish)
   Stock Features: RSI=68, Momentum=+12%, Trend=+8%
   Base Q-values: {BUY: 0.9, HOLD: 0.1, SELL: -0.8}
   Adjusted: {BUY: 1.15, HOLD: 0.2, SELL: -1.0}
   
   Signal: BUY with confidence 0.92
   ```

5. **Storage in Neo4j:**
   ```cypher
   MERGE (c:Company {ticker: "NVDA"})
   MERGE (q:Quarter {id: "2025-Q4"})
   MERGE (s:Summary {id: "NVDA_2025-Q4"})
   SET s.text = "...", s.embedding = [...]
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
