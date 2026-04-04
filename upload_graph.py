import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from graph_manager import Neo4jHandler
from paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "TSM", "ASML"]


class Mag7Uploader:
    def __init__(self):
        self.graph = Neo4jHandler()
        self.economic_data = {}
        self._load_economic_data()

    def clear_database(self):
        """Clear Neo4j data and re-apply constraints (IF NOT EXISTS)."""
        self.graph.clear_graph()
        self.graph.setup_constraints()

    def close(self):
        self.graph.close()

    def _load_economic_data(self):
        """Pre-load economic indicators for quick lookup by date."""
        indicators = {
            "cpi": "economic_cpi.csv",
            "inflation": "economic_inflation.csv",
            "unemployment": "economic_unemployment.csv",
            "real_gdp": "economic_real_gdp.csv",
            "federal_funds_rate": "economic_federal_funds_rate.csv",
            "treasury_yield_10y": "economic_treasury_yield_10y.csv"
        }
        
        for key, filename in indicators.items():
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # Ensure date format
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        self.economic_data[key] = df
                        logger.info(f"Loaded {key} data: {len(df)} records")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"Economic data file not found: {path}")

    def get_economic_context_for_date(self, target_date: str) -> Dict[str, float]:
        """Find the closest economic indicators for a given date."""
        context = {}
        try:
            dt = pd.to_datetime(target_date)
            for key, df in self.economic_data.items():
                if df is not None and not df.empty:
                    # Find closest date using asof (assumes sorted index)
                    # default to backward search (data available at that time)
                    try:
                        idx = df.index.get_indexer([dt], method='nearest')[0]
                        if idx != -1:
                            val = df.iloc[idx]['value']
                            context[key] = float(val)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Error getting economic context for {target_date}: {e}")
        return context

    def process_company_profiles(self):
        """Upload company overview data."""
        logger.info("Processing company profiles...")
        for ticker in MAG7_TICKERS:
            path = os.path.join(DATA_DIR, f"{ticker}_company_overview.csv")
            if not os.path.exists(path):
                logger.warning(f"Profile not found for {ticker}")
                continue

            try:
                df = pd.read_csv(path)
                if df.empty:
                    continue
                
                # Alpha Vantage Overview returns a single row usually
                row = df.iloc[0]
                
                profile_data = {
                    "ticker": row.get("Symbol", ticker),
                    "name": row.get("Name"),
                    "finnhubIndustry": row.get("Industry"), # Mapping field name expected by graph_manager
                    "sector": row.get("Sector"),
                    "description": row.get("Description")
                }
                
                self.graph.ingest_company_profile(profile_data)
                
            except Exception as e:
                logger.error(f"Error processing profile for {ticker}: {e}")

    def process_financials(self):
        """Upload financial facts from income, balance sheet, cash flow, and earnings."""
        logger.info("Processing financials...")
        
        for ticker in MAG7_TICKERS:
            logger.info(f"  > {ticker}")
            
            # Map of file suffixes to metric prefix/types
            files = [
                (f"{ticker}_income_statement_quarterly.csv", ["totalRevenue", "netIncome", "grossProfit", "operatingIncome", "ebitda", "eps"]), # eps might not be here, check columns
                (f"{ticker}_balance_sheet_quarterly.csv", ["totalAssets", "totalLiabilities", "cashAndCashEquivalentsAtCarryingValue", "totalShareholderEquity"]),
                (f"{ticker}_cash_flow_quarterly.csv", ["operatingCashflow", "capitalExpenditures", "cashflowFromInvestment", "cashflowFromFinancing"]),
                (f"{ticker}_earnings_quarterly.csv", ["reportedEPS", "estimatedEPS", "surprise"])
            ]
            
            for filename, metrics in files:
                path = os.path.join(DATA_DIR, filename)
                if not os.path.exists(path):
                    continue
                
                try:
                    df = pd.read_csv(path)
                    
                    # Ensure we have date column
                    date_col = 'fiscalDateEnding' if 'fiscalDateEnding' in df.columns else 'reportedDate'
                    if date_col not in df.columns:
                        continue
                        
                    # Collect all facts for this file to batch ingest
                    batch_facts = []
                    
                    for _, row in df.iterrows():
                        date_str = str(row[date_col])
                        if date_str == 'nan' or date_str == 'None':
                            continue
                        
                        # Convert date to quarter ID (YYYY-Q#)
                        try:
                            dt = pd.to_datetime(date_str)
                            quarter_id = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
                        except (ValueError, TypeError):
                            continue

                        for metric in metrics:
                            if metric in row:
                                val = row[metric]
                                # Handle numeric conversion
                                try:
                                    if val == 'None' or val == 'nan' or pd.isna(val):
                                        continue
                                    f_val = float(val)
                                    
                                    # Unit determination (simplified)
                                    unit = "USD"
                                    if "EPS" in metric or "ratio" in metric.lower():
                                        unit = "Ratio/Share"
                                    elif "percentage" in metric.lower():
                                        unit = "Percent"
                                    
                                    # Prepare prev_fact_id logic for batch
                                    try:
                                        year_str, q_part = quarter_id.split("-")
                                        q_curr = int(q_part.replace("Q", ""))
                                        if q_curr == 1:
                                            prev_y = int(year_str) - 1
                                            prev_q = 4
                                        else:
                                            prev_y = int(year_str)
                                            prev_q = q_curr - 1
                                        prev_fact_id = f"{ticker}_{prev_y}-Q{prev_q}_{metric}"
                                    except (ValueError, TypeError):
                                        prev_fact_id = None
                                        
                                    fact_data = {
                                        "ticker": ticker,
                                        "quarter": quarter_id,
                                        "metric": metric,
                                        "value": f_val,
                                        "unit": unit,
                                        "fact_id": f"{ticker}_{quarter_id}_{metric}",
                                        "prev_fact_id": prev_fact_id
                                    }
                                    batch_facts.append(fact_data)

                                except ValueError:
                                    pass
                    
                    # Ingest batch
                    if batch_facts:
                        self.graph.ingest_financial_facts_batch(batch_facts)
                                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

    def process_transcripts_and_context(self):
        """Upload transcripts and link full quarterly context."""
        logger.info("Processing transcripts and context...")
        
        for ticker in MAG7_TICKERS:
            # Find transcript files for this ticker
            # Pattern: {ticker}_earnings_transcript_{year}_{quarter}.txt
            try:
                files = os.listdir(DATA_DIR)
                transcript_files = [f for f in files if f.startswith(f"{ticker}_earnings_transcript_") and f.endswith(".txt")]
                
                for t_file in transcript_files:
                    # Parse year and quarter from filename
                    # Format: TICKER_earnings_transcript_YYYY_QX.txt
                    parts = t_file.replace(".txt", "").split("_")
                    if len(parts) < 4:
                        continue
                        
                    year = parts[-2]
                    q_str = parts[-1] # Q1, Q2, etc.
                    quarter_id = f"{year}-{q_str}"
                    
                    # Read transcript
                    file_path = os.path.join(DATA_DIR, t_file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        transcript_text = f.read()
                        
                    # Estimate date for economic data lookup
                    # Q1 -> March 31, Q2 -> June 30, Q3 -> Sept 30, Q4 -> Dec 31
                    q_num = int(q_str.replace("Q", ""))
                    approx_month = q_num * 3
                    approx_date = f"{year}-{approx_month:02d}-28" # Slightly before end
                    
                    # 1. Get Economic Context
                    econ_data = self.get_economic_context_for_date(approx_date)
                    
                    # 2. Get Financial Metrics (Snapshot for this quarter)
                    # We rebuild a dict of metrics for this specific quarter by querying what we just uploaded 
                    # OR by reading CSVs again. Reading CSVs again for specific quarter is safer locally.
                    fin_metrics = self._get_financials_for_quarter(ticker, quarter_id, approx_date)
                    
                    # 3. Ingest Context
                    self.graph.ingest_quarterly_context(
                        ticker=ticker,
                        quarter=quarter_id,
                        transcript=transcript_text,
                        economic_data=econ_data,
                        financial_metrics=fin_metrics
                    )
                    
            except Exception as e:
                logger.error(f"Error finding transcripts for {ticker}: {e}")

    def _get_financials_for_quarter(self, ticker: str, quarter_id: str, approx_date: str) -> Dict[str, float]:
        """Helper to extract a metrics dictionary for a single quarter from CSVs."""
        # This is a simplified lookup to bundle data for the 'FinancialMetrics' node
        # We try to match 'fiscalDateEnding' to the quarter.
        metrics = {}
        
        metric_files = [
            (f"{ticker}_income_statement_quarterly.csv", ["totalRevenue", "netIncome", "grossProfit", "ebitda"]),
            (f"{ticker}_balance_sheet_quarterly.csv", ["totalAssets", "totalLiabilities", "cashAndCashEquivalentsAtCarryingValue"]),
            (f"{ticker}_earnings_quarterly.csv", ["reportedEPS"])
        ]
        
        for filename, fields in metric_files:
             path = os.path.join(DATA_DIR, filename)
             if not os.path.exists(path):
                 continue
             try:
                 df = pd.read_csv(path)
                 date_col = 'fiscalDateEnding' if 'fiscalDateEnding' in df.columns else 'reportedDate'
                 if date_col not in df.columns:
                     continue
                 
                 # Look for row matching this quarter
                 # We can convert quarter_id (2024-Q1) back to a date range or check if date falls in it
                 # Simple approximation: Match year and Month roughly
                 
                 target_year = int(quarter_id.split("-")[0])
                 target_q = int(quarter_id.split("-")[1].replace("Q", ""))
                 
                 for _, row in df.iterrows():
                     row_date_str = str(row[date_col])
                     if row_date_str == 'nan': continue
                     row_dt = pd.to_datetime(row_date_str)
                     
                     # Check if this date belongs to the quarter
                     # Quarter 1: Jan-Mar, Q2: Apr-Jun, etc.
                     row_q = (row_dt.month - 1) // 3 + 1
                     if row_dt.year == target_year and row_q == target_q:
                         for f in fields:
                             if f in row:
                                try:
                                    metrics[f] = float(row[f])
                                except (ValueError, TypeError):
                                    pass
                         break
             except Exception:
                 pass
                 
        return metrics

def main():
    print("Initializing Neo4j Context Uploader for Mag7...")
    uploader = Mag7Uploader()
    
    try:
        uploader.clear_database() # Clear first as requested
        uploader.process_company_profiles()
        uploader.process_financials()
        uploader.process_transcripts_and_context()
        print("Upload complete!")
    finally:
        uploader.close()

if __name__ == "__main__":
    main()
