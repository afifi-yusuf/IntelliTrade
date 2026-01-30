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
            "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT quarter_id IF NOT EXISTS FOR (q:Quarter) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE" 
        ]
        
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Constraint creation failed (might already exist): {e}")

    # =========================================================================
    # INGESTION
    # =========================================================================

    def ingest_company_profile(self, profile_data: Dict[str, Any]):
        """
        Ingest company profile node.
        Expects keys: ticker, name, finnhubIndustry, etc.
        """
        if not self.driver or not profile_data:
            return

        query = """
        MERGE (c:Company {ticker: $ticker})
        SET c.name = $name,
            c.industry = $industry,
            c.sector = $sector,
            c.description = $description,
            c.last_updated = datetime()
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
            logger.info(f"Ingested company profile for {params['ticker']}")
        except Exception as e:
            logger.error(f"Error ingesting company profile: {e}")

    def ingest_financial_fact(self, ticker: str, quarter: str, metric: str, value: float, unit: str = "USD"):
        """
        Ingest a specific financial fact (node) and link it to Company and Quarter.
        
        Node Schema: (:Fact {id, metric, value, unit, quarter})
        Path: (:Company)-[:REPORTED]->(:Fact)-[:IN_QUARTER]->(:Quarter)
        """
        if not self.driver:
            return

        # Create a unique ID for the fact to avoid duplicates
        fact_id = f"{ticker}_{quarter}_{metric}"

        query = """
        MERGE (c:Company {ticker: $ticker})
        MERGE (q:Quarter {id: $quarter})
        
        MERGE (f:Fact {id: $fact_id})
        SET f.metric = $metric,
            f.value = $value,
            f.unit = $unit,
            f.quarter = $quarter
        
        MERGE (c)-[:REPORTED]->(f)
        MERGE (f)-[:IN_QUARTER]->(q)
        """
        
        params = {
            "ticker": ticker,
            "quarter": quarter,  # e.g., "2024-Q1"
            "fact_id": fact_id,
            "metric": metric,
            "value": value,
            "unit": unit
        }

        try:
            with self.driver.session() as session:
                session.run(query, params)
        except Exception as e:
            logger.error(f"Error ingesting fact {fact_id}: {e}")

    def ingest_transcript_reason(self, ticker: str, quarter: str, metric: str, reason_text: str, sentiment: str = "NEUTRAL"):
        """
        Ingest a 'Reason' extracted from transcript and link to the relevant Fact.
        
        Path: (:Fact)-[:HAS_REASON]->(:Reason)
        """
        if not self.driver:
            return

        fact_id = f"{ticker}_{quarter}_{metric}"

        query = """
        MATCH (f:Fact {id: $fact_id})
        CREATE (r:Reason {text: $text, sentiment: $sentiment, created_at: datetime()})
        MERGE (f)-[:HAS_REASON]->(r)
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
        MERGE (c:Company {ticker: $ticker})
        MERGE (q:Quarter {id: $quarter})
        
        // Store transcript
        MERGE (t:Transcript {id: $ticker + "_" + $quarter})
        SET t.content = $transcript,
            t.created_at = datetime()
        MERGE (c)-[:HAS_TRANSCRIPT]->(t)
        MERGE (t)-[:IN_QUARTER]->(q)
        
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
        
        // Store financial metrics
        MERGE (fm:FinancialMetrics {id: $ticker + "_" + $quarter})
        SET fm.data = $financial_metrics,
            fm.created_at = datetime()
        MERGE (c)-[:HAS_METRICS]->(fm)
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

    def ingest_quarterly_summary(self, ticker: str, quarter: str, summary_text: str, embedding: list = None):
        """
        Ingest the LLM-generated analysis and its embedding vector.
        Path: (:Company)-[:HAS_SUMMARY]->(:Summary)-[:IN_QUARTER]->(:Quarter)
        """
        if not self.driver: return

        query = """
        MERGE (c:Company {ticker: $ticker})
        MERGE (q:Quarter {id: $quarter})
        MERGE (s:Summary {id: $ticker + "_" + $quarter})
        SET s.text = $text,
            s.embedding = $embedding,
            s.created_at = datetime()
        MERGE (c)-[:HAS_SUMMARY]->(s)
        MERGE (s)-[:IN_QUARTER]->(q)
        """
        
        params = {
            "ticker": ticker,
            "quarter": quarter,
            "text": summary_text,
            "embedding": embedding
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
        MATCH (c:Company {ticker: $ticker})
        MATCH (q:Quarter {id: $quarter})
        
        OPTIONAL MATCH (c)-[:HAS_TRANSCRIPT]->(t:Transcript)-[:IN_QUARTER]->(q)
        OPTIONAL MATCH (q)-[:HAS_ECONOMIC_CONTEXT]->(ec:EconomicContext)
        OPTIONAL MATCH (c)-[:HAS_METRICS]->(fm:FinancialMetrics)-[:IN_QUARTER]->(q)
        
        RETURN t.content as transcript,
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
        RETURN s.text as text, s.embedding as embedding
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"id": f"{ticker}_{quarter}"})
                record = result.single()
                if record:
                    return {
                        "text": record["text"],
                        "embedding": record.get("embedding")
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
        MATCH (c:Company {ticker: $ticker})-[:REPORTED]->(f:Fact)-[:IN_QUARTER]->(q:Quarter)
        WHERE f.metric = $metric
        OPTIONAL MATCH (f)-[:HAS_REASON]->(r:Reason)
        WITH f, q, r
        ORDER BY q.id DESC
        LIMIT $limit
        RETURN q.id as quarter, f.value as value, collect(r.text) as reasons
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
        MATCH (c:Company {ticker: $ticker})-[:REPORTED]->(f:Fact)-[:IN_QUARTER]->(q:Quarter)
        OPTIONAL MATCH (f)-[:HAS_REASON]->(r:Reason)
        {quarter_filter}
        RETURN q.id as quarter, f.metric as metric, f.value as value, collect(r.text) as reasons
        ORDER BY q.id DESC, f.metric
        LIMIT 20
        """
        quarter_filter = ""
        params = {"ticker": ticker}
        if max_quarter:
            quarter_filter = "WHERE q.id <= $max_quarter"
            params["max_quarter"] = max_quarter
        query = query.format(quarter_filter=quarter_filter)
        
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
