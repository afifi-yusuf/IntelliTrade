# Embedding Options

IntelliSense now supports **three embedding backends** with automatic fallback:

## 🤗 Option 1: Hugging Face (Recommended) - **FREE & OPEN SOURCE**

**Advantages:**
- ✅ Completely free (no API costs)
- ✅ Runs locally on your machine
- ✅ No rate limits
- ✅ Fast inference
- ✅ No API key required
- ✅ Privacy (data stays local)

**Installation:**
```bash
pip install sentence-transformers torch
```

**Available Models:**

| Model | Dimensions | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ Fast | Good | **Default, balanced** |
| `all-mpnet-base-v2` | 768 | ⚡⚡ Medium | Excellent | High accuracy needed |
| `all-MiniLM-L12-v2` | 384 | ⚡⚡ Medium | Better | Balanced option |
| `paraphrase-MiniLM-L6-v2` | 384 | ⚡⚡⚡ Fast | Good | Paraphrase detection |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ⚡⚡⚡ Fast | Good | Question answering |

**Usage in Code:**
```python
# Default (all-MiniLM-L6-v2, 384 dimensions)
vectorizer = NarrativeVectorizer()

# Or specify a different model
vectorizer = NarrativeVectorizer(model_name='all-mpnet-base-v2')
```

**Current Setup:** This is now the **default** option! No changes needed.

---

## ☁️ Option 2: Azure OpenAI - **PAID SERVICE**

**Advantages:**
- ✅ Highest quality embeddings (1536 dimensions)
- ✅ Consistent with Azure LLM pipeline
- ✅ Managed service (no local compute)

**Disadvantages:**
- ❌ Requires API key and deployment
- ❌ Costs money per token
- ❌ Rate limits apply
- ❌ Data sent to Azure

**Setup:**
1. Create embedding deployment in Azure OpenAI Studio:
   - Model: `text-embedding-ada-002`
   - Deployment name: `text-embedding-ada-002` (or your choice)

2. Set environment variable:
```bash
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
```

**Usage in Code:**
```python
vectorizer = NarrativeVectorizer(
    azure_client=azure_client,
    model_name=None  # Will use Azure
)
```

---

## 📊 Option 3: TF-IDF - **FALLBACK**

**Automatically used if:**
- Hugging Face not installed
- Azure not configured
- Both fail

**Characteristics:**
- 100 dimensions
- Simple bag-of-words
- No semantic understanding
- Fast but limited

---

## Comparison

| Feature | Hugging Face | Azure OpenAI | TF-IDF |
|---------|-------------|--------------|--------|
| **Cost** | Free | ~$0.0001/1K tokens | Free |
| **Dimensions** | 384-768 | 1536 | 100 |
| **Semantic Understanding** | ✅ Excellent | ✅ Excellent | ❌ Limited |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ API latency | ⚡⚡⚡ Fastest |
| **Privacy** | ✅ Local | ❌ Cloud | ✅ Local |
| **Setup Complexity** | Easy (pip install) | Medium (Azure setup) | None |

---

## Recommended Setup

### For Development/Testing:
```python
# Use Hugging Face (default)
vectorizer = NarrativeVectorizer()
```

### For Production (High Quality):
```python
# Use larger Hugging Face model
vectorizer = NarrativeVectorizer(model_name='all-mpnet-base-v2')
```

### For Azure Integration:
```python
# Use Azure (if you already have it)
vectorizer = NarrativeVectorizer(
    azure_client=azure_client
)
```

---

## Installation

### Install Hugging Face Support:
```bash
pip install sentence-transformers torch
```

**First run will download the model (~80-200MB depending on model):**
```
🤗 Loading Hugging Face model: all-MiniLM-L6-v2
Downloading model... [100%]
✅ Using Hugging Face embeddings (384 dimensions)
```

**Subsequent runs use cached model (instant).**

---

## Performance Comparison (on 3 quarterly analyses)

| Backend | Embedding Time | Quality | Cost |
|---------|---------------|---------|------|
| **all-MiniLM-L6-v2** | ~0.1s total | Excellent | $0 |
| **all-mpnet-base-v2** | ~0.3s total | Better | $0 |
| **Azure ada-002** | ~1.5s (API) | Excellent | ~$0.0003 |
| **TF-IDF** | ~0.05s | Limited | $0 |

**Recommendation:** Use **Hugging Face** (default) for best balance of cost, speed, and quality!

---

## Switching Between Backends

The system automatically detects and uses the best available option:

1. **Tries Hugging Face** (if installed)
2. Falls back to **Azure OpenAI** (if configured)
3. Falls back to **TF-IDF** (always available)

You'll see messages like:
```
🤗 Loading Hugging Face model: all-MiniLM-L6-v2
✅ Using Hugging Face embeddings (384 dimensions)
```

Or:
```
⚠️ sentence-transformers not installed
☁️ Using Azure OpenAI embeddings (1536 dimensions)
```

---

## Quick Start

```bash
# Install Hugging Face support
pip install sentence-transformers torch

# Run the system (will use Hugging Face automatically)
python main.py
```

That's it! No Azure setup needed. 🎉
