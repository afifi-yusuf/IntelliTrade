# API Endpoints Documentation

This document describes all API endpoints used in the IntelliSense RAG pipeline from both Finnhub and Alpha Vantage.

---

## Overview

The pipeline integrates **two data providers** to maximize data coverage while staying within free tier limits:

1. **Finnhub** - Real-time market data, company profiles, and recent earnings
2. **Alpha Vantage** - Historical financial statements and extended earnings data

**Total API Calls per Analysis**: 9 (4 Finnhub + 5 Alpha Vantage) - **ALL FREE TIER**

---

# FINNHUB API

**Base URL**: `https://finnhub.io/api/v1/`

**Rate Limit**: 60 calls/minute (free tier)

**Authentication**: `token` parameter or `X-Finnhub-Token` header

## Finnhub Endpoints

### 1. Company Profile 2

**Endpoint**: `/stock/profile2`

**Method**: `GET`

**Documentation**: https://finnhub.io/docs/api/company-profile2

**Parameters**:
- `symbol` (required): Stock ticker symbol (e.g., "AAPL", "MSFT")
- `token` (required): Your Finnhub API key

**What it Does**:
- Retrieves comprehensive company information and overview
- Provides business context for fundamental analysis

**Data Returned**:
- Company name, ticker, exchange
- Industry and sector classification
- Country, currency
- Market capitalization
- Number of shares outstanding
- IPO date
- Company logo, website, phone
- Business description

**Example Request**:
```
GET https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token=YOUR_API_KEY
```

**Use Case**: Provides company context and business overview

---

### 2. Basic Financials / Company Metrics

**Endpoint**: `/stock/metric`

**Method**: `GET`

**Documentation**: https://finnhub.io/docs/api/company-basic-financials

**Parameters**:
- `symbol` (required): Stock ticker symbol
- `metric` (required): Set to "all" to get all available metrics
- `token` (required): Your Finnhub API key

**What it Does**:
- Retrieves 50+ financial metrics and ratios
- Provides key performance indicators

**Data Returned**:
- **Valuation**: P/E, P/B, P/S, PEG ratios
- **Profitability**: ROE, ROA, profit margins
- **Efficiency**: Asset turnover, inventory turnover
- **Financial Health**: Current ratio, quick ratio, debt ratios
- **Growth**: Revenue growth, EPS growth
- **Market**: 52-week high/low, beta

**Example Request**:
```
GET https://finnhub.io/api/v1/stock/metric?symbol=AAPL&metric=all&token=YOUR_API_KEY
```

**Use Case**: Core fundamental analysis ratios and metrics

---

### 3. Earnings Calendar

**Endpoint**: `/calendar/earnings`

**Method**: `GET`

**Documentation**: https://finnhub.io/docs/api/earnings-calendar

**Parameters**:
- `symbol` (required): Stock ticker symbol
- `token` (required): Your Finnhub API key

**What it Does**:
- Retrieves earnings calendar information
- **Free tier primarily provides upcoming earnings dates**

**Data Returned**:
- Earnings date
- EPS estimate and actual
- Revenue estimate and actual
- Quarter and year

**Example Request**:
```
GET https://finnhub.io/api/v1/calendar/earnings?symbol=AAPL&token=YOUR_API_KEY
```

**Use Case**: Track upcoming earnings dates and recent estimates

**Free Tier Limitations**: Primarily upcoming/future earnings, limited historical data

---

### 4. Earnings Surprises

**Endpoint**: `/stock/earnings`

**Method**: `GET`

**Documentation**: https://finnhub.io/docs/api/earnings-surprises

**Parameters**:
- `symbol` (required): Stock ticker symbol
- `token` (required): Your Finnhub API key

**What it Does**:
- Retrieves historical earnings surprise data
- Shows actual vs estimated EPS
- **Free tier provides up to 1 year of data**

**Data Returned**:
- Period (earnings date)
- Actual EPS
- Estimate EPS
- Surprise amount and percentage

**Example Request**:
```
GET https://finnhub.io/api/v1/stock/earnings?symbol=AAPL&token=YOUR_API_KEY
```

**Use Case**: Analyze beat/miss patterns over past year

**Free Tier Limitations**: Up to 1 year of historical data

---

# ALPHA VANTAGE API

**Base URL**: `https://www.alphavantage.co/query`

**Rate Limit**: 25 calls/day (free tier)

**Authentication**: `apikey` parameter

**Key Advantage**: Extended historical data including full financial statements

## Alpha Vantage Endpoints

### 1. Company Overview

**Endpoint**: `function=OVERVIEW`

**Method**: `GET`

**Documentation**: https://www.alphavantage.co/documentation/#company-overview

**Parameters**:
- `function`: Set to "OVERVIEW"
- `symbol` (required): Stock ticker symbol
- `apikey` (required): Your Alpha Vantage API key

**What it Does**:
- Provides comprehensive company overview and fundamental data
- Complements Finnhub profile with additional metrics

**Data Returned**:
- Company name, description, sector, industry
- Market cap, P/E ratio, PEG ratio
- Dividend data (yield, date, payout ratio)
- 52-week high/low
- Shares outstanding, EPS, revenue
- Profit margin, operating margin
- ROE, ROA, revenue per share
- Quarterly revenue/earnings growth
- Analyst target price
- Beta

**Example Request**:
```
GET https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey=YOUR_API_KEY
```

**Use Case**: Comprehensive fundamental overview with additional metrics not in Finnhub

---

### 2. Earnings History

**Endpoint**: `function=EARNINGS`

**Method**: `GET`

**Documentation**: https://www.alphavantage.co/documentation/#earnings

**Parameters**:
- `function`: Set to "EARNINGS"
- `symbol` (required): Stock ticker symbol
- `apikey` (required): Your Alpha Vantage API key

**What it Does**:
- Provides **extended earnings history** beyond Finnhub's 1-year limit
- Includes both annual and quarterly earnings data

**Data Returned**:
- **Annual Earnings**: Fiscal date, reported EPS
- **Quarterly Earnings**: 
  - Fiscal date ending
  - Reported date
  - Reported EPS
  - Estimated EPS
  - Surprise amount
  - Surprise percentage

**Example Request**:
```
GET https://www.alphavantage.co/query?function=EARNINGS&symbol=IBM&apikey=YOUR_API_KEY
```

**Example Response**:
```json
{
  "symbol": "IBM",
  "annualEarnings": [
    {
      "fiscalDateEnding": "2023-12-31",
      "reportedEPS": "8.67"
    }
  ],
  "quarterlyEarnings": [
    {
      "fiscalDateEnding": "2023-09-30",
      "reportedDate": "2023-10-19",
      "reportedEPS": "1.81",
      "estimatedEPS": "1.78",
      "surprise": "0.03",
      "surprisePercentage": "1.69"
    }
  ]
}
```

**Use Case**: Long-term earnings trend analysis (multiple years of quarterly data)

**Advantage over Finnhub**: Not limited to 1 year, includes all historical quarters

---

### 3. Income Statement

**Endpoint**: `function=INCOME_STATEMENT`

**Method**: `GET`

**Documentation**: https://www.alphavantage.co/documentation/#income-statement

**Parameters**:
- `function`: Set to "INCOME_STATEMENT"
- `symbol` (required): Stock ticker symbol
- `apikey` (required): Your Alpha Vantage API key

**What it Does**:
- Provides detailed income statements
- Annual and quarterly reports
- Shows revenue, expenses, and profitability trends

**Data Returned** (per report):
- Fiscal date ending
- Total revenue
- Cost of revenue / Cost of goods sold (COGS)
- Gross profit
- Operating expenses (R&D, SG&A, etc.)
- Operating income
- Interest income/expense
- Income before tax
- Income tax expense
- Net income
- EPS (basic and diluted)
- EBITDA

**Example Request**:
```
GET https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=YOUR_API_KEY
```

**Use Case**: Revenue and profitability trend analysis, margin analysis

**Not Available in Finnhub Free Tier**: Detailed income statement line items

---

### 4. Balance Sheet

**Endpoint**: `function=BALANCE_SHEET`

**Method**: `GET`

**Documentation**: https://www.alphavantage.co/documentation/#balance-sheet

**Parameters**:
- `function`: Set to "BALANCE_SHEET"
- `symbol` (required): Stock ticker symbol
- `apikey` (required): Your Alpha Vantage API key

**What it Does**:
- Provides detailed balance sheet statements
- Annual and quarterly reports
- Shows assets, liabilities, and equity

**Data Returned** (per report):
- Fiscal date ending
- **Assets**:
  - Current assets (cash, receivables, inventory)
  - Non-current assets (PP&E, intangibles, goodwill)
  - Total assets
- **Liabilities**:
  - Current liabilities (payables, short-term debt)
  - Non-current liabilities (long-term debt)
  - Total liabilities
- **Equity**:
  - Common stock
  - Retained earnings
  - Total shareholder equity

**Example Request**:
```
GET https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=IBM&apikey=YOUR_API_KEY
```

**Use Case**: Financial health assessment, debt analysis, working capital analysis

**Not Available in Finnhub Free Tier**: Detailed balance sheet line items

---

### 5. Cash Flow Statement

**Endpoint**: `function=CASH_FLOW`

**Method**: `GET`

**Documentation**: https://www.alphavantage.co/documentation/#cash-flow

**Parameters**:
- `function`: Set to "CASH_FLOW"
- `symbol` (required): Stock ticker symbol
- `apikey` (required): Your Alpha Vantage API key

**What it Does**:
- Provides detailed cash flow statements
- Annual and quarterly reports
- Shows cash generation and usage

**Data Returned** (per report):
- Fiscal date ending
- **Operating Activities**:
  - Operating cash flow
  - Depreciation & amortization
  - Changes in working capital
- **Investing Activities**:
  - Capital expenditures
  - Investments
  - Acquisitions
- **Financing Activities**:
  - Dividends paid
  - Debt issuance/repayment
  - Stock buybacks
- Free cash flow

**Example Request**:
```
GET https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey=YOUR_API_KEY
```

**Use Case**: Cash generation quality, capital allocation analysis, dividend sustainability

**Not Available in Finnhub Free Tier**: Detailed cash flow statement

---

## Data Flow in RAG Pipeline

```
┌─────────────────────────────────────────────┐
│         User Input: Stock Ticker            │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│   FINNHUB    │                  │ ALPHA VANTAGE│
│  (4 calls)   │                  │  (5 calls)   │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       ├─ Company Profile                ├─ Company Overview
       ├─ Financial Metrics              ├─ Extended Earnings
       ├─ Earnings Calendar              ├─ Income Statement
       └─ Earnings Surprise (1yr)        ├─ Balance Sheet
                                         └─ Cash Flow
       │                                 │
       └────────────┬────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Azure OpenAI API    │
        │   (GPT-4 Analysis)    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Institutional-Grade   │
        │ Investment Analysis   │
        └───────────────────────┘
```

---

## API Comparison & Strategy

### Why Use Both APIs?

| Feature | Finnhub Free | Alpha Vantage Free | Strategy |
|---------|--------------|-------------------|----------|
| **Rate Limit** | 60/min | 25/day | Use both to maximize data |
| **Company Profile** | ✅ Excellent | ✅ Good | Use both for complete picture |
| **Financial Ratios** | ✅ 50+ metrics | ✅ Key metrics | Finnhub primary, AV supplement |
| **Earnings Calendar** | ✅ Upcoming | ❌ Not available | Finnhub only |
| **Earnings History** | ⚠️ 1 year only | ✅ All quarters | **AV provides extended history** |
| **Income Statement** | ❌ Not available | ✅ Full statements | **AV exclusive** |
| **Balance Sheet** | ❌ Not available | ✅ Full statements | **AV exclusive** |
| **Cash Flow** | ❌ Not available | ✅ Full statements | **AV exclusive** |

### Optimal Usage Strategy

**Total Free API Calls per Analysis**: 9 (ALL FREE TIER ✅)

- **Finnhub** (4 calls): Real-time metrics, recent earnings, upcoming dates
- **Alpha Vantage** (5 calls): Historical financials, extended earnings, detailed statements

**Daily Limit on Free Tier**:
- Finnhub: ~1,440 calls/day (60/min × 24h) ✅ No practical limit for individual use
- Alpha Vantage: 25 calls/day ⚠️ **Limiting factor: ~2-3 companies per day**

**Note**: Earnings call transcripts require premium subscriptions on both Finnhub and Alpha Vantage and are NOT included in this pipeline.

---

## Authentication Setup

### Required Environment Variables (.env)

```bash
# Finnhub API
FINNHUB_API_KEY=your_finnhub_api_key_here

# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_DEPLOYMENT=your_deployment_name_here
```

### Get API Keys

- **Finnhub**: https://finnhub.io/register
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **Azure OpenAI**: https://azure.microsoft.com/en-us/products/ai-services/openai-service

---

## Error Handling

The pipeline includes comprehensive error handling for:
- HTTP errors (4xx, 5xx responses)
- Timeout errors (30 second timeout)
- Missing/invalid API keys
- Rate limiting
- Empty responses
- Premium-only features

---

## Data Completeness Matrix

What data you get from each ticker analysis:

| Data Type | Source | Availability |
|-----------|--------|--------------|
| Company Info | Both | Always |
| Market Cap & Valuation | Both | Always |
| Financial Ratios | Finnhub | Always |
| Upcoming Earnings | Finnhub | Usually |
| 1-Year Earnings | Finnhub | Usually |
| Multi-Year Earnings | Alpha Vantage | Usually |
| Income Statement | Alpha Vantage | Usually |
| Balance Sheet | Alpha Vantage | Usually |
| Cash Flow | Alpha Vantage | Usually |

---

## Upgrade Considerations

### When to Upgrade Finnhub

- Need earnings call transcripts (premium feature)
- Need historical financial statements (premium feature)
- Need more than 60 API calls/minute
- Need premium data feeds (options, forex, crypto)

### When to Upgrade Alpha Vantage

- Need more than 25 API calls/day (>2-3 analyses/day)
- Need real-time intraday data
- Need earnings call transcripts (premium feature)
- Need extended historical data (10+ years)
- Need technical indicators for algorithmic trading

---

## Additional Resources

### Finnhub
- **Documentation**: https://finnhub.io/docs/api
- **Pricing**: https://finnhub.io/pricing
- **Dashboard**: https://finnhub.io/dashboard
- **Status**: https://status.finnhub.io/

### Alpha Vantage
- **Documentation**: https://www.alphavantage.co/documentation/
- **Premium**: https://www.alphavantage.co/premium/
- **Support**: https://www.alphavantage.co/support/
- **API Key**: https://www.alphavantage.co/support/#api-key

---

## Best Practices

1. **Cache Results**: Alpha Vantage's 25/day limit is strict - cache results when possible
2. **Handle Errors Gracefully**: Not all data is available for all companies
3. **Check Response Format**: Both APIs return JSON but structures differ
4. **Respect Rate Limits**: Implement delays if making multiple requests
5. **Monitor Usage**: Track your daily API usage to stay within limits (Alpha Vantage: 25/day = ~2-3 companies)
6. **Free Tier Only**: This pipeline uses ONLY free tier endpoints - no premium features required
7. **Earnings Transcripts**: Not available on free tier for either API - consider upgrading if needed

---

## Notes

1. **Alpha Vantage Advantage**: Full financial statements (income, balance, cash flow) not available in Finnhub free tier
2. **Finnhub Advantage**: Higher rate limit (60/min vs 25/day), better for real-time data
3. **Data Freshness**: Finnhub updates more frequently; Alpha Vantage focuses on historical accuracy
4. **Coverage**: Both have excellent US stock coverage; international stocks may vary
5. **API Calls**: Pipeline makes 9 total calls per analysis (4 Finnhub + 5 Alpha Vantage) - **ALL FREE TIER**
6. **Earnings Transcripts**: NOT available on free tier for either API - requires premium subscription

