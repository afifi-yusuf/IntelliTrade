import os
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
    """
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            logger.warning("NEO4J_PASSWORD not found in environment. Connection may fail.")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
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

    def setup_constraints(self):
        """Create uniqueness constraints for the graph schema."""
        if not self.driver:
            return
        
        queries = [
            "CREATE CONSTRAINT ticker_symbol IF NOT EXISTS FOR (t:Ticker) REQUIRE t.symbol IS UNIQUE",
            "CREATE CONSTRAINT quarter_id IF NOT EXISTS FOR (q:Quarter) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE"
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
        generated_for_date: str = None
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
            s.generated_for_date = $generated_for_date
        MERGE (t)-[:HAS_SUMMARY]->(s)
        MERGE (s)-[:IN_QUARTER]->(q)
        """
        
        params = {
            "ticker": ticker,
            "quarter": quarter,
            "text": summary_text,
            "embedding": embedding,
            "max_quarter": max_quarter,
            "generated_for_date": generated_for_date
        }
        
        try:
            with self.driver.session() as session:
                session.run(query, params)
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
               s.generated_for_date as generated_for_date
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"id": f"{ticker}_{quarter}"})
                record = result.single()
                if record:
                    return {
                        "text": record["text"],
                        "embedding": record.get("embedding"),
                        "max_quarter": record.get("max_quarter"),
                        "generated_for_date": record.get("generated_for_date")
                    }
        except Exception as e:
            logger.error(f"Error retrieving summary: {e}")
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
