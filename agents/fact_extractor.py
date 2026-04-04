"""
Fact Extractor Agent
====================
Extracts structured financial facts from earnings call transcripts using LLM.
Each fact is a discrete, embeddable unit stored in Neo4j for graph-based retrieval.
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

EXTRACTION_SYSTEM_PROMPT = """You are a precise financial fact extraction bot. 
Extract discrete, quantitative financial facts from earnings call transcripts.

For each fact, provide:
- metric: The financial metric name (e.g., "Revenue", "Operating Margin", "EPS")
- value: The stated value with units (e.g., "$35.1B", "28.5%", "$1.23")
- change: Year-over-year or quarter-over-quarter change if mentioned (e.g., "+12% YoY", "-3pp QoQ")
- sentiment: "positive", "negative", or "neutral" based on management tone
- context: Brief context from the transcript explaining the metric (1 sentence max)
- category: One of ["revenue", "profitability", "growth", "risk", "guidance", "operational"]

CRITICAL RULES:
- Extract ONLY explicitly stated facts, do NOT infer or calculate
- Include forward guidance as separate facts with category "guidance"
- Capture risk factors mentioned by management with category "risk"
- Maximum 15 facts per transcript (focus on the most material ones)

Respond ONLY with a valid JSON array. No markdown, no explanation."""


EXTRACTION_USER_PROMPT = """Extract the key financial facts from this earnings transcript:

TICKER: {ticker}
QUARTER: {quarter}

TRANSCRIPT (truncated):
{transcript}

Respond with a JSON array of facts:"""


class FactExtractorAgent:
    """
    Extracts structured financial facts from earnings call transcripts.
    Each fact becomes a node in the Neo4j knowledge graph.
    """
    
    def __init__(self, azure_client: Optional[AzureOpenAI] = None):
        if azure_client:
            self.client = azure_client
        else:
            self.client = AzureOpenAI()
        self.model = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    
    def extract(self, transcript: str, ticker: str, quarter: str) -> List[Dict[str, Any]]:
        """
        Extract structured facts from an earnings transcript.
        
        Args:
            transcript: Raw earnings call transcript text
            ticker: Company ticker symbol
            quarter: Quarter identifier (e.g., "2024-Q3")
            
        Returns:
            List of fact dictionaries with keys:
            metric, value, change, sentiment, context, category
        """
        if not transcript or len(transcript.strip()) < 100:
            logger.warning(f"Transcript too short for {ticker} {quarter}, returning empty facts")
            return []
        
        # Truncate very long transcripts to stay within token limits
        truncated = transcript[:12000]
        
        prompt = EXTRACTION_USER_PROMPT.format(
            ticker=ticker,
            quarter=quarter,
            transcript=truncated
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic extraction for reproducible runs
                max_completion_tokens=2000
            )
            
            raw = response.choices[0].message.content.strip()
            
            # Parse JSON response
            facts = self._parse_response(raw, ticker, quarter)
            logger.info(f"Extracted {len(facts)} facts from {ticker} {quarter}")
            return facts
            
        except Exception as e:
            logger.error(f"Fact extraction failed for {ticker} {quarter}: {e}")
            return []
    
    def _parse_response(self, raw: str, ticker: str, quarter: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured facts."""
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        try:
            facts = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    facts = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error(f"Could not parse facts JSON for {ticker} {quarter}")
                    return []
            else:
                logger.error(f"No JSON array found in response for {ticker} {quarter}")
                return []
        
        if not isinstance(facts, list):
            return []
        
        # Normalize and validate each fact
        validated = []
        for i, fact in enumerate(facts[:15]):  # Cap at 15
            if not isinstance(fact, dict):
                continue
            
            normalized = {
                "ticker": ticker,
                "quarter": quarter,
                "fact_id": f"{ticker}_{quarter}_fact_{i}",
                "metric": str(fact.get("metric", "Unknown")),
                "value": str(fact.get("value", "N/A")),
                "change": str(fact.get("change", "")),
                "sentiment": str(fact.get("sentiment", "neutral")).lower(),
                "context": str(fact.get("context", ""))[:500],
                "category": str(fact.get("category", "operational")).lower()
            }
            
            # Validate sentiment
            if normalized["sentiment"] not in ("positive", "negative", "neutral"):
                normalized["sentiment"] = "neutral"
            
            # Validate category
            valid_categories = {"revenue", "profitability", "growth", "risk", "guidance", "operational"}
            if normalized["category"] not in valid_categories:
                normalized["category"] = "operational"
            
            validated.append(normalized)
        
        return validated
    
    def facts_to_text(self, facts: List[Dict]) -> str:
        """Convert structured facts to a text representation for embedding."""
        lines = []
        for f in facts:
            line = f"{f['metric']}: {f['value']}"
            if f.get('change'):
                line += f" ({f['change']})"
            if f.get('context'):
                line += f" - {f['context']}"
            lines.append(line)
        return "\n".join(lines)
