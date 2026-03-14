"""
Peer Analyst Agent
==================
Compares company facts against peer companies using Neo4j vector search.
Uses sector-based peer grouping and semantic similarity to find relevant comparisons.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Sub-sector mapping for the 10 tracked tickers
SECTOR_MAP = {
    "NVDA": "Semiconductors", "ASML": "Semiconductors", "TSM": "Semiconductors",
    "AAPL": "Consumer Tech", "MSFT": "Enterprise Tech", "AMZN": "Cloud/E-Commerce",
    "GOOGL": "Digital Advertising", "META": "Digital Advertising",
    "TSLA": "EV/Energy", "SPY": "Index"
}

# Broader peer groups (for cross-sector comparison)
PEER_GROUPS = {
    "Semiconductors": ["NVDA", "ASML", "TSM"],
    "Big Tech": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "AI Beneficiaries": ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
    # TSLA included in AI/Growth group; FSD/Dojo positions it alongside AI spend leaders
    "Mega Cap Growth": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "TSM", "ASML"],
}

PEER_ANALYSIS_PROMPT = """You are a comparative financial analyst. Your peer score feeds \
directly into a quantitative trading model — use the full -1.0 to +1.0 range.

Compare the target company's current quarter facts against peer company data from the knowledge graph.

Analyze:
1. How does the company perform RELATIVE to peers on key metrics (revenue growth rate, margins, EPS)?
2. Is the company an outlier — positively or negatively — versus its peers?
3. Are there sector-wide headwinds/tailwinds, and is this company better or worse positioned?

Be specific and quantitative. Reference actual numbers.
Keep your analysis to 150-200 words.

SCORE CALIBRATION:
- +0.8 to +1.0: clear sector leader — materially outperforming peers on multiple metrics
- +0.3 to +0.7: modest outperformance — ahead on 1-2 metrics
- -0.1 to +0.2: roughly in-line with peers
- -0.3 to -0.7: underperforming — behind on key metrics
- -0.8 to -1.0: clear sector laggard
Only output 0.0 if peer data is absent or truly insufficient for comparison.

On the LAST line, output the peer score:
PEER_SCORE: <float>"""


class PeerAnalystAgent:
    """
    Compares company performance against sector peers using Neo4j graph data.
    Uses vector similarity search to find semantically relevant peer facts.
    """
    
    def __init__(self, graph_db=None, azure_client: Optional[AzureOpenAI] = None):
        self.graph_db = graph_db
        if azure_client:
            self.client = azure_client
        else:
            self.client = AzureOpenAI()
        self.model = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    
    def get_peers(self, ticker: str) -> List[str]:
        """Get peer tickers for comparison."""
        # Find all groups this ticker belongs to
        peers = set()
        for group_name, group_tickers in PEER_GROUPS.items():
            if ticker in group_tickers:
                peers.update(group_tickers)
        
        # Also add same sub-sector peers
        my_sector = SECTOR_MAP.get(ticker)
        if my_sector:
            for t, s in SECTOR_MAP.items():
                if s == my_sector:
                    peers.add(t)
        
        peers.discard(ticker)  # Remove self
        peers.discard("SPY")   # SPY is an index, not a peer
        return list(peers)
    
    def analyze(self, facts: List[Dict], ticker: str, quarter: str,
                embedder=None) -> Dict[str, Any]:
        """
        Compare company facts against peers using graph retrieval.
        
        Args:
            facts: Extracted facts for the target company
            ticker: Target company ticker
            quarter: Current quarter
            embedder: Vectorizer instance for creating embeddings
            
        Returns:
            {"analysis": str, "peer_score": float, "peer_facts": List}
        """
        peers = self.get_peers(ticker)
        if not peers:
            return {"analysis": "No peer data available.", "peer_score": 0.0, "peer_facts": []}
        
        # Retrieve peer facts from Neo4j
        peer_facts = []
        if self.graph_db and embedder:
            peer_facts = self._retrieve_peer_facts(facts, ticker, peers, quarter, embedder)
        
        if not peer_facts:
            # Fallback: try direct graph query without vector search
            peer_facts = self._retrieve_peer_facts_direct(ticker, peers, quarter)
        
        if not peer_facts:
            return {"analysis": "No peer comparison data found in graph.", "peer_score": 0.0, "peer_facts": []}
        
        # Format context for LLM
        target_text = self._format_facts(facts, ticker)
        peer_text = self._format_peer_facts(peer_facts)
        
        prompt = f"""TARGET COMPANY: {ticker} ({quarter})
{target_text}

PEER DATA FROM KNOWLEDGE GRAPH:
{peer_text}

Compare {ticker} against its peers and assess relative performance:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PEER_ANALYSIS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_completion_tokens=500
            )
            
            raw = response.choices[0].message.content.strip()
            analysis, score = self._parse_response(raw)
            
            return {
                "analysis": analysis,
                "peer_score": score,
                "peer_facts": peer_facts
            }
            
        except Exception as e:
            logger.error(f"Peer analysis failed for {ticker}: {e}")
            return {"analysis": f"Peer analysis error: {e}", "peer_score": 0.0, "peer_facts": []}
    
    def _retrieve_peer_facts(self, facts: List[Dict], ticker: str, 
                              peers: List[str], quarter: str, embedder) -> List[Dict]:
        """Retrieve semantically similar facts from peer companies via vector search."""
        if not self.graph_db or not hasattr(self.graph_db, 'search_peer_facts'):
            return []
        
        all_peer_facts = []
        
        # Create a query embedding from the target company's key facts
        fact_text = " | ".join([f"{f['metric']}: {f['value']}" for f in facts[:5]])
        
        try:
            query_embedding = embedder.get_embedding(fact_text)
            if query_embedding is not None:
                results = self.graph_db.search_peer_facts(
                    query_embedding=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                    exclude_ticker=ticker,
                    peer_tickers=peers,
                    top_k=10,
                    max_quarter=quarter
                )
                all_peer_facts.extend(results)
        except Exception as e:
            logger.warning(f"Vector peer search failed: {e}")
        
        return all_peer_facts
    
    def _retrieve_peer_facts_direct(self, ticker: str, peers: List[str], 
                                     quarter: str) -> List[Dict]:
        """Fallback: retrieve peer facts via direct Cypher query (no vector search)."""
        if not self.graph_db:
            return []
        
        try:
            return self.graph_db.retrieve_peer_facts_by_quarter(peers, quarter)
        except Exception as e:
            logger.warning(f"Direct peer query failed: {e}")
            return []
    
    def _format_facts(self, facts: List[Dict], ticker: str) -> str:
        """Format target facts for the prompt."""
        lines = [f"Key Facts for {ticker}:"]
        for f in facts:
            line = f"  • {f['metric']}: {f['value']}"
            if f.get('change'):
                line += f" ({f['change']})"
            lines.append(line)
        return "\n".join(lines)
    
    def _format_peer_facts(self, peer_facts: List[Dict]) -> str:
        """Format peer facts for the prompt."""
        by_ticker = {}
        for f in peer_facts:
            t = f.get("ticker", "?")
            by_ticker.setdefault(t, []).append(f)
        
        lines = []
        for t, facts in by_ticker.items():
            lines.append(f"\n  {t}:")
            for f in facts[:5]:  # Max 5 facts per peer
                metric = f.get("metric", f.get("metric_name", "?"))
                value = f.get("value", "?")
                quarter = f.get("quarter", "")
                lines.append(f"    • {metric}: {value} ({quarter})")
        
        return "\n".join(lines) if lines else "No peer data available."
    
    def _parse_response(self, raw: str) -> tuple:
        """Parse LLM response to extract analysis text and peer score."""
        lines = raw.strip().split("\n")
        score = 0.0
        analysis_lines = []
        
        for line in lines:
            if line.strip().startswith("PEER_SCORE:"):
                try:
                    score = float(line.split(":")[-1].strip())
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    score = 0.0
            else:
                analysis_lines.append(line)
        
        return "\n".join(analysis_lines).strip(), score
