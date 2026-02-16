import os
import time
import json
import neo4j
from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy Neo4j driver notifications
logging.getLogger("neo4j").setLevel(logging.ERROR)

class Neo4jHandler:
    """
    Handles interactions with the Neo4j Graph Database for the Causal-Temporal Knowledge Graph.
    Uses singleton pattern to share one driver across the entire application.
    """
    _instance = None
    _driver = None
    
    def __new__(cls):
        """Singleton: reuse the same instance and driver connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return  # Already initialized
        self._initialized = True
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            logger.warning("NEO4J_PASSWORD not found in environment. Connection may fail.")

        try:
            self.driver = GraphDatabase.driver(
                uri, 
                auth=(username, password),
                max_connection_lifetime=200,
                max_connection_pool_size=5,  # Aura Free tier has limited connections
                connection_acquisition_timeout=30,
                keep_alive=True
            )
            Neo4jHandler._driver = self.driver
            self.verify_connection()
            self.setup_constraints()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def verify_connection(self):
        """Verify connectivity to Neo4j."""
        if not self.driver:
            return
        try:
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Neo4j connection verification failed: {e}")
            raise

    def _run_query(self, query: str, params: dict = None, single: bool = False, 
                   retries: int = 3, backoff: float = 1.0):
        """
        Execute a Cypher query with automatic retry on transient failures.
        
        Args:
            query: Cypher query string
            params: Query parameters
            single: If True, return single record; otherwise return list of records
            retries: Number of retry attempts
            backoff: Initial backoff in seconds (doubles each retry)
        
        Returns:
            Single record dict, list of record dicts, or None on failure
        """
        if not self.driver:
            return None
        
        for attempt in range(retries):
            try:
                with self.driver.session() as session:
                    result = session.run(query, params or {})
                    if single:
                        record = result.single()
                        return dict(record) if record else None
                    else:
                        return [dict(r) for r in result]
            except (neo4j.exceptions.ServiceUnavailable, 
                    neo4j.exceptions.SessionExpired,
                    ConnectionResetError,
                    OSError) as e:
                wait = backoff * (2 ** attempt)
                logger.warning(f"Neo4j transient error (attempt {attempt+1}/{retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"Neo4j query error: {e}")
                return None
        
        logger.error(f"Neo4j query failed after {retries} retries")
        return None

    def _write_query(self, query: str, params: dict = None, 
                     retries: int = 3, backoff: float = 1.0) -> bool:
        """
        Execute a write query with retry. Returns True on success.
        """
        if not self.driver:
            return False
        
        for attempt in range(retries):
            try:
                with self.driver.session() as session:
                    session.run(query, params or {})
                return True
            except (neo4j.exceptions.ServiceUnavailable,
                    neo4j.exceptions.SessionExpired,
                    ConnectionResetError,
                    OSError) as e:
                wait = backoff * (2 ** attempt)
                logger.warning(f"Neo4j write error (attempt {attempt+1}/{retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"Neo4j write error: {e}")
                return False
        
        return False

    def setup_constraints(self):
        """Create uniqueness constraints for the graph schema."""
        if not self.driver:
            return
        
        queries = [
            "CREATE CONSTRAINT ticker_symbol IF NOT EXISTS FOR (t:Ticker) REQUIRE t.symbol IS UNIQUE",
            "CREATE CONSTRAINT quarter_id IF NOT EXISTS FOR (q:Quarter) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Constraint creation failed (might already exist): {e}")

    def clear_graph(self):
        """
        Clear all data from the Neo4j database.
        WARNING: This deletes everything!
        """
        if not self.driver:
            return
        
        query = "MATCH (n) DETACH DELETE n"
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info("⚠️  CLEARED ENTIRE GRAPH DATABASE")
        except Exception as e:
            logger.error(f"Error clearing graph: {e}")

    # =========================================================================
    # INGESTION
    # =========================================================================

    def ingest_company_profile(self, profile_data: Dict[str, Any]):
        """
        Ingest company/ticker profile node.
        """
        if not self.driver or not profile_data:
            return

        # Using Ticker label as per reference
        query = """
        MERGE (t:Ticker {symbol: $ticker})
        SET t.name = $name,
            t.industry = $industry,
            t.sector = $sector,
            t.description = $description,
            t.last_updated = datetime()
        """
        
        params = {
            "ticker": profile_data.get("ticker", "UNKNOWN"),
            "name": profile_data.get("name"),
            "industry": profile_data.get("finnhubIndustry"),
            "sector": profile_data.get("sector"),  # From AlphaVantage maybe
            "description": profile_data.get("description")
        }

        try:
            with self.driver.session() as session:
                session.run(query, params)
            logger.info(f"Ingested Ticker profile for {params['ticker']}")
        except Exception as e:
            logger.error(f"Error ingesting company profile: {e}")

    def ingest_financial_facts_batch(self, facts_data: list):
        """
        Ingest a batch of financial facts for improved performance.
        Expected facts_data format:
        [
            {
                "ticker": "AAPL",
                "quarter": "2024-Q1",
                "metric": "Revenue",
                "value": 100.0,
                "unit": "USD",
                "fact_id": "AAPL_2024-Q1_Revenue",
                "prev_fact_id": "AAPL_2023-Q4_Revenue" (optional)
            },
            ...
        ]
        """
        if not self.driver or not facts_data:
            return

        query = """
        UNWIND $batch AS item
        MERGE (t:Ticker {symbol: item.ticker})
        MERGE (q:Quarter {id: item.quarter})
        MERGE (t)-[:REPORTS_IN]->(q)
        
        MERGE (m:Metric {name: item.metric})
        
        MERGE (f:Fact {id: item.fact_id})
        SET f.ticker = item.ticker,
            f.quarter = item.quarter,
            f.metric_name = item.metric,
            f.value = item.value,
            f.unit = item.unit,
            f.created_at = datetime()
        
        MERGE (v:Value {id: item.fact_id + "_val"})
        SET v.content = toString(item.value)
        
        MERGE (q)-[:HAS_FACT]->(f)
        MERGE (f)-[:FOR_FIRM]->(t)
        MERGE (f)-[:HAS_METRIC]->(m)
        MERGE (f)-[:HAS_VALUE]->(v)
        
        // Handle QoQ comparison if previous fact exists
        // Note: In pure Cypher UNWIND, conditional matching is tricky. 
        // We use FOREACH hack or separate query. Here we use a conditional merge via CASE/FOREACH simulation if needed, 
        // but simpler is to just try to match.
        WITH f, item
        WHERE item.prev_fact_id IS NOT NULL
        MATCH (prev_f:Fact {id: item.prev_fact_id})
        MERGE (f)-[rel:QoQ_COMPARES]->(prev_f)
        SET rel.diff = f.value - prev_f.value,
            rel.pct_change = CASE WHEN prev_f.value <> 0 THEN (f.value - prev_f.value) / abs(prev_f.value) ELSE null END
        """
        
        # Process in chunks to avoid memory issues with massive queries
        chunk_size = 500
        for i in range(0, len(facts_data), chunk_size):
            batch = facts_data[i:i + chunk_size]
            try:
                with self.driver.session() as session:
                    session.run(query, {"batch": batch})
                logger.info(f"Ingested batch of {len(batch)} financial facts")
            except Exception as e:
                logger.error(f"Error ingesting batch facts: {e}")

    def ingest_financial_fact(self, ticker: str, quarter: str, metric: str, value: float, unit: str = "USD"):
        """
        Wrapper for single fact ingestion using the batch method.
        """
        # Calculate previous quarter logic here or in wrapper
        try:
            year, q_part = quarter.split("-")
            q_num = int(q_part.replace("Q", ""))
            
            if q_num == 1:
                prev_year = int(year) - 1
                prev_q_num = 4
            else:
                prev_year = int(year)
                prev_q_num = q_num - 1
            
            prev_quarter = f"{prev_year}-Q{prev_q_num}"
            prev_fact_id = f"{ticker}_{prev_quarter}_{metric}"
        except:
            prev_fact_id = None

        fact_data = {
            "ticker": ticker,
            "quarter": quarter,
            "metric": metric,
            "value": value,
            "unit": unit,
            "fact_id": f"{ticker}_{quarter}_{metric}",
            "prev_fact_id": prev_fact_id
        }
        self.ingest_financial_facts_batch([fact_data])

    def ingest_transcript_reason(self, ticker: str, quarter: str, metric: str, reason_text: str, sentiment: str = "NEUTRAL"):
        """
        Ingest a 'Reason' extracted from transcript and link to the relevant Fact.
        
        Path: (:Fact)-[:EXPLAINED_BY]->(:Reason)
        """
        if not self.driver:
            return

        fact_id = f"{ticker}_{quarter}_{metric}"

        query = """
        MATCH (f:Fact {id: $fact_id})
        CREATE (r:Reason {content: $text, sentiment: $sentiment, created_at: datetime()})
        MERGE (f)-[:EXPLAINED_BY]->(r)
        """
        
        params = {
            "fact_id": fact_id,
            "text": reason_text,
            "sentiment": sentiment
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                summary = result.consume()
                if summary.counters.nodes_created == 0:
                    logger.warning(f"Could not link Reason to Fact {fact_id} (Fact may not exist).")
                else:
                    logger.info(f"Ingested Reason for {fact_id}")
        except Exception as e:
            logger.error(f"Error ingesting reason for {fact_id}: {e}")

    def ingest_quarterly_context(self, ticker: str, quarter: str, 
                                 transcript: str, economic_data: dict, 
                                 financial_metrics: dict):
        """
        Ingest comprehensive quarterly context including:
        - Earnings transcript
        - Economic indicators (CPI, inflation, GDP, etc.)
        - Financial metrics
        
        This creates a rich knowledge graph for context-enriched RAG.
        """
        if not self.driver: return

        query = """
        MERGE (t:Ticker {symbol: $ticker})
        MERGE (q:Quarter {id: $quarter})
        
        // Store transcript
        MERGE (tr:Transcript {id: $ticker + "_" + $quarter})
        SET tr.content = $transcript,
            tr.created_at = datetime()
        MERGE (t)-[:HAS_TRANSCRIPT]->(tr)
        MERGE (tr)-[:IN_QUARTER]->(q)
        
        // Store economic context
        MERGE (ec:EconomicContext {id: $quarter})
        SET ec.cpi = $cpi,
            ec.inflation = $inflation,
            ec.unemployment = $unemployment,
            ec.gdp = $gdp,
            ec.federal_funds_rate = $fed_rate,
            ec.treasury_yield = $treasury_yield,
            ec.created_at = datetime()
        MERGE (q)-[:HAS_ECONOMIC_CONTEXT]->(ec)
        
        // Store financial metrics (snapshot node for easy RAG, separate from atomic Facts)
        MERGE (fm:FinancialMetrics {id: $ticker + "_" + $quarter})
        SET fm.data = $financial_metrics,
            fm.created_at = datetime()
        MERGE (t)-[:HAS_METRICS]->(fm)
        MERGE (fm)-[:IN_QUARTER]->(q)
        """
        
        params = {
            "ticker": ticker,
            "quarter": quarter,
            "transcript": transcript[:50000],  # Limit size
            "cpi": economic_data.get("cpi"),
            "inflation": economic_data.get("inflation"),
            "unemployment": economic_data.get("unemployment"),
            "gdp": economic_data.get("gdp"),
            "fed_rate": economic_data.get("federal_funds_rate"),
            "treasury_yield": economic_data.get("treasury_yield"),
            "financial_metrics": str(financial_metrics)[:10000]
        }
        
        try:
            with self.driver.session() as session:
                session.run(query, params)
            logger.info(f"Ingested quarterly context for {ticker} {quarter}")
        except Exception as e:
            logger.error(f"Error ingesting quarterly context: {e}")

    def ingest_quarterly_summary(
        self,
        ticker: str,
        quarter: str,
        summary_text: str,
        embedding: list = None,
        max_quarter: str = None,
        generated_for_date: str = None,
        structured_signals: Dict[str, float] = None,
        reliability: Dict[str, Any] = None,
        source_quarter: str = None
    ):
        """
        Ingest the LLM-generated analysis and its embedding vector.
        Path: (:Ticker)-[:HAS_SUMMARY]->(:Summary)-[:IN_QUARTER]->(:Quarter)
        """
        if not self.driver: return

        query = """
        MERGE (t:Ticker {symbol: $ticker})
        MERGE (q:Quarter {id: $quarter})
        MERGE (s:Summary {id: $ticker + "_" + $quarter})
        SET s.text = $text,
            s.embedding = $embedding,
            s.created_at = datetime(),
            s.max_quarter = $max_quarter,
            s.generated_for_date = $generated_for_date,
            s.structured_signals_json = $structured_signals_json,
            s.reliability_json = $reliability_json,
            s.source_quarter = $source_quarter
        MERGE (t)-[:HAS_SUMMARY]->(s)
        MERGE (s)-[:IN_QUARTER]->(q)
        """
        
        params = {
            "ticker": ticker,
            "quarter": quarter,
            "text": summary_text,
            "embedding": embedding,
            "max_quarter": max_quarter,
            "generated_for_date": generated_for_date,
            "structured_signals_json": json.dumps(structured_signals or {}),
            "reliability_json": json.dumps(reliability or {}),
            "source_quarter": source_quarter or quarter
        }
        
        try:
            self._write_query(query, params)
            logger.info(f"Ingested LLM summary for {ticker} {quarter}")
        except Exception as e:
            logger.error(f"Error ingesting summary: {e}")

    def retrieve_quarterly_context(self, ticker: str, quarter: str) -> Optional[dict]:
        """
        Retrieve comprehensive quarterly context for RAG.
        Returns: dict with transcript, economic_data, financial_metrics
        """
        if not self.driver: return None
        
        query = """
        MATCH (t:Ticker {symbol: $ticker})
        MATCH (q:Quarter {id: $quarter})
        
        OPTIONAL MATCH (t)-[:HAS_TRANSCRIPT]->(tr:Transcript)-[:IN_QUARTER]->(q)
        OPTIONAL MATCH (q)-[:HAS_ECONOMIC_CONTEXT]->(ec:EconomicContext)
        OPTIONAL MATCH (t)-[:HAS_METRICS]->(fm:FinancialMetrics)-[:IN_QUARTER]->(q)
        
        RETURN tr.content as transcript,
               ec.cpi as cpi,
               ec.inflation as inflation,
               ec.unemployment as unemployment,
               ec.gdp as gdp,
               ec.federal_funds_rate as fed_rate,
               ec.treasury_yield as treasury_yield,
               fm.data as financial_metrics
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"ticker": ticker, "quarter": quarter})
                record = result.single()
                if record:
                    return {
                        "transcript": record.get("transcript"),
                        "economic_data": {
                            "cpi": record.get("cpi"),
                            "inflation": record.get("inflation"),
                            "unemployment": record.get("unemployment"),
                            "gdp": record.get("gdp"),
                            "federal_funds_rate": record.get("fed_rate"),
                            "treasury_yield": record.get("treasury_yield")
                        },
                        "financial_metrics": record.get("financial_metrics")
                    }
        except Exception as e:
            logger.error(f"Error retrieving quarterly context: {e}")
        return None
    
    def retrieve_quarterly_summary(self, ticker: str, quarter: str) -> Optional[dict]:
        """Retrieve stored LLM summary and embedding for a quarter."""
        if not self.driver: return None
        
        query = """
        MATCH (s:Summary {id: $id})
        RETURN s.text as text,
               s.embedding as embedding,
               s.max_quarter as max_quarter,
               s.generated_for_date as generated_for_date,
               s.structured_signals_json as structured_signals_json,
               s.reliability_json as reliability_json,
               s.source_quarter as source_quarter
        """
        record = self._run_query(query, {"id": f"{ticker}_{quarter}"}, single=True)
        if record:
            signals = {}
            reliability = {}
            try:
                raw_signals = record.get("structured_signals_json")
                if raw_signals:
                    signals = json.loads(raw_signals)
            except Exception:
                signals = {}
            try:
                raw_reliability = record.get("reliability_json")
                if raw_reliability:
                    reliability = json.loads(raw_reliability)
            except Exception:
                reliability = {}
            return {
                "text": record.get("text"),
                "embedding": record.get("embedding"),
                "max_quarter": record.get("max_quarter"),
                "generated_for_date": record.get("generated_for_date"),
                "structured_signals": signals,
                "reliability": reliability,
                "source_quarter": record.get("source_quarter")
            }
        return None

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def retrieve_context(self, ticker: str, metric_name: str, lookback_quarters: int = 4) -> str:
        """
        Retrieve historical context (Values + Reasons) for a specific metric.
        Useful for providing "Memory" to the LLM.
        """
        if not self.driver:
            return "Graph Database unavailable."

        query = """
        MATCH (t:Ticker {symbol: $ticker})<-[:FOR_FIRM]-(f:Fact)-[:HAS_METRIC]->(m:Metric {name: $metric})
        MATCH (f)-[:HAS_VALUE]->(v:Value)
        MATCH (f)<-[:HAS_FACT]-(q:Quarter)
        OPTIONAL MATCH (f)-[:EXPLAINED_BY]->(r:Reason)
        WITH f, q, v, r
        ORDER BY q.id DESC
        LIMIT $limit
        RETURN q.id as quarter, v.content as value, collect(r.content) as reasons
        """
        
        params = {
            "ticker": ticker,
            "metric": metric_name,
            "limit": lookback_quarters
        }

        context_lines = []
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                for record in result:
                    q = record["quarter"]
                    v = record["value"]
                    reasons = record["reasons"]
                    
                    line = f"- {q}: {metric_name} = {v}"
                    if reasons:
                        line += f"\n  Reasons cited: {'; '.join(reasons)}"
                    context_lines.append(line)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"Error retrieving graph context: {e}"

        if not context_lines:
            return f"No historical graph data found for {metric_name}."
            
        return "\n".join(context_lines)

    def retrieve_all_context(self, ticker: str, max_quarter: str = None) -> str:
        """
        Retrieve a summary of all available facts and reasons for a company.
        """
        if not self.driver:
            return ""

        query = """
        MATCH (t:Ticker {symbol: $ticker})<-[:FOR_FIRM]-(f:Fact)-[:HAS_METRIC]->(m:Metric)
        MATCH (f)-[:HAS_VALUE]->(v:Value)
        MATCH (f)<-[:HAS_FACT]-(q:Quarter)
        OPTIONAL MATCH (f)-[:EXPLAINED_BY]->(r:Reason)
        {quarter_filter}
        RETURN q.id as quarter, m.name as metric, v.content as value, collect(r.content) as reasons
        ORDER BY q.id DESC, m.name
        LIMIT 20
        """
        quarter_filter = ""
        params = {"ticker": ticker}
        if max_quarter:
            quarter_filter = "WHERE q.id <= $max_quarter"
            params["max_quarter"] = max_quarter
        query = query.replace("{quarter_filter}", quarter_filter)
        
        output = [f"### Causal-Temporal Knowledge Graph Context for {ticker}"]
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                current_q = None
                for record in result:
                    q = record["quarter"]
                    if q != current_q:
                        output.append(f"\n**{q}**")
                        current_q = q
                    
                    metric = record["metric"]
                    val = record["value"]
                    reasons = record["reasons"]
                    
                    item = f"- {metric}: {val}"
                    if reasons and reasons[0]:
                        item += f" | Reason: {reasons[0]}"
                    output.append(item)
        except Exception as e:
            return f"Error: {e}"
            
        return "\n".join(output)

    # =========================================================================
    # MULTI-AGENT GRAPH METHODS
    # =========================================================================

    def setup_vector_index(self, embedding_dim: int = 384):
        """
        Create a Neo4j vector index on Fact.embedding for semantic search.
        Requires Neo4j 5.x+ with vector index support.
        """
        if not self.driver:
            return
        
        query = """
        CREATE VECTOR INDEX fact_embedding_index IF NOT EXISTS
        FOR (f:Fact) ON (f.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            with self.driver.session() as session:
                session.run(query, {"dim": embedding_dim})
            logger.info(f"Vector index created/verified (dim={embedding_dim})")
        except Exception as e:
            logger.warning(f"Vector index creation failed (may not be supported): {e}")

    def ingest_sector(self, ticker: str, sector_name: str):
        """
        Create Sector node and link Ticker to it.
        Path: (:Ticker)-[:IN_SECTOR]->(:Sector)
        """
        if not self.driver or not sector_name:
            return
        
        query = """
        MERGE (t:Ticker {symbol: $ticker})
        MERGE (s:Sector {name: $sector})
        MERGE (t)-[:IN_SECTOR]->(s)
        """
        try:
            with self.driver.session() as session:
                session.run(query, {"ticker": ticker, "sector": sector_name})
            logger.info(f"Linked {ticker} to sector {sector_name}")
        except Exception as e:
            logger.error(f"Error ingesting sector for {ticker}: {e}")

    def setup_temporal_edges(self):
        """
        Create NEXT_QUARTER relationships between consecutive Quarter nodes.
        Enables temporal graph traversal.
        """
        if not self.driver:
            return
        
        query = """
        MATCH (q:Quarter)
        WITH q ORDER BY q.id
        WITH collect(q) AS quarters
        UNWIND range(0, size(quarters)-2) AS i
        WITH quarters[i] AS current, quarters[i+1] AS next
        MERGE (current)-[:NEXT_QUARTER]->(next)
        """
        try:
            with self.driver.session() as session:
                session.run(query)
            logger.info("Temporal NEXT_QUARTER edges created/verified")
        except Exception as e:
            logger.error(f"Error creating temporal edges: {e}")

    def ingest_fact_with_embedding(self, fact_data: Dict[str, Any], embedding: list):
        """
        Ingest a structured fact with its embedding vector for vector search.
        
        Args:
            fact_data: Dict with keys: ticker, quarter, fact_id, metric, value, 
                       change, sentiment, context, category
            embedding: List[float] embedding vector
        """
        if not self.driver or not fact_data:
            return
        
        query = """
        MERGE (t:Ticker {symbol: $ticker})
        MERGE (q:Quarter {id: $quarter})
        MERGE (t)-[:REPORTS_IN]->(q)
        
        MERGE (m:Metric {name: $metric})
        
        MERGE (f:Fact {id: $fact_id})
        SET f.ticker = $ticker,
            f.quarter = $quarter,
            f.metric_name = $metric,
            f.value = $value,
            f.change = $change,
            f.sentiment = $sentiment,
            f.context = $context,
            f.category = $category,
            f.embedding = $embedding,
            f.source = 'transcript_extraction',
            f.created_at = datetime()
        
        MERGE (q)-[:HAS_FACT]->(f)
        MERGE (f)-[:FOR_FIRM]->(t)
        MERGE (f)-[:HAS_METRIC]->(m)
        
        MERGE (v:Value {id: $fact_id + "_val"})
        SET v.content = $value
        MERGE (f)-[:HAS_VALUE]->(v)
        """
        
        params = {
            "ticker": fact_data.get("ticker"),
            "quarter": fact_data.get("quarter"),
            "fact_id": fact_data.get("fact_id"),
            "metric": fact_data.get("metric", "Unknown"),
            "value": str(fact_data.get("value", "")),
            "change": str(fact_data.get("change", "")),
            "sentiment": str(fact_data.get("sentiment", "neutral")),
            "context": str(fact_data.get("context", ""))[:500],
            "category": str(fact_data.get("category", "operational")),
            "embedding": embedding
        }
        
        try:
            with self.driver.session() as session:
                session.run(query, params)
        except Exception as e:
            logger.error(f"Error ingesting fact with embedding: {e}")

    def search_peer_facts(self, query_embedding: list, exclude_ticker: str,
                          peer_tickers: List[str], top_k: int = 10,
                          max_quarter: str = None) -> List[Dict]:
        """
        Vector search for semantically similar facts across peer companies.
        Uses the fact_embedding_index.
        """
        if not self.driver or not peer_tickers:
            return []
        
        quarter_clause = ""
        params = {
            "topK": top_k,
            "vec": query_embedding,
            "peers": peer_tickers,
            "exclude": exclude_ticker
        }
        if max_quarter:
            quarter_clause = "AND node.quarter <= $max_quarter"
            params["max_quarter"] = max_quarter

        query = f"""
        CALL db.index.vector.queryNodes('fact_embedding_index', $topK, $vec)
        YIELD node, score
        WHERE node.ticker IN $peers AND node.ticker <> $exclude AND score > 0.3 {quarter_clause}
        OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
        RETURN node.ticker AS ticker, node.metric_name AS metric,
               v.content AS value, node.quarter AS quarter,
               node.sentiment AS sentiment, node.change AS change, score
        ORDER BY score DESC
        LIMIT $topK
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"Vector peer search failed: {e}")
            return []

    def search_historical_facts(self, query_embedding: list, ticker: str,
                                 max_quarter: str = None, top_k: int = 10) -> List[Dict]:
        """
        Vector search for semantically similar facts from the same company's history.
        """
        if not self.driver:
            return []
        
        # Build quarter filter
        quarter_clause = ""
        params = {
            "topK": top_k,
            "vec": query_embedding,
            "ticker": ticker
        }
        if max_quarter:
            quarter_clause = "AND node.quarter < $max_quarter"
            params["max_quarter"] = max_quarter
        
        query = f"""
        CALL db.index.vector.queryNodes('fact_embedding_index', $topK, $vec)
        YIELD node, score
        WHERE node.ticker = $ticker {quarter_clause} AND score > 0.3
        OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
        RETURN node.metric_name AS metric, v.content AS value,
               node.quarter AS quarter, node.sentiment AS sentiment,
               node.change AS change, score
        ORDER BY score DESC
        LIMIT $topK
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"Vector historical search failed: {e}")
            return []

    def get_metric_trend(self, ticker: str, metric: str, 
                         num_quarters: int = 4) -> List[Dict]:
        """
        Traverse QoQ_COMPARES edges to get a metric's trend over N quarters.
        Returns: [{"quarter": "2024-Q1", "value": "100.0"}, ...] oldest to newest.
        """
        if not self.driver:
            return []
        
        query = """
        MATCH (t:Ticker {symbol: $ticker})<-[:FOR_FIRM]-(f:Fact)-[:HAS_METRIC]->(m:Metric {name: $metric})
        MATCH (f)-[:HAS_VALUE]->(v:Value)
        MATCH (f)<-[:HAS_FACT]-(q:Quarter)
        RETURN q.id AS quarter, v.content AS value, f.change AS change
        ORDER BY q.id DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    "ticker": ticker,
                    "metric": metric,
                    "limit": num_quarters
                })
                records = [dict(r) for r in result]
                records.reverse()  # Return oldest first
                return records
        except Exception as e:
            logger.warning(f"Metric trend query failed for {metric}: {e}")
            return []

    def retrieve_peer_facts_by_quarter(self, peer_tickers: List[str], 
                                       quarter: str) -> List[Dict]:
        """
        Fallback: retrieve peer facts by direct quarter match (no vector search).
        """
        if not self.driver or not peer_tickers:
            return []
        
        query = """
        MATCH (q:Quarter {id: $quarter})-[:HAS_FACT]->(f:Fact)-[:FOR_FIRM]->(t:Ticker)
        WHERE t.symbol IN $peers
        MATCH (f)-[:HAS_VALUE]->(v:Value)
        MATCH (f)-[:HAS_METRIC]->(m:Metric)
        RETURN t.symbol AS ticker, m.name AS metric, v.content AS value,
               f.quarter AS quarter, f.sentiment AS sentiment, f.change AS change
        ORDER BY t.symbol, m.name
        LIMIT 30
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    "quarter": quarter,
                    "peers": peer_tickers
                })
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"Direct peer query failed: {e}")
            return []
