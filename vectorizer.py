import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Appended to narrative embedding in DQN state (must match `backtest` feature layout)
DQN_STATE_SCALAR_DIM = 7 + 3 + 2 + 1 + 6 + 6 + 5


def dqn_state_dim(embedding_dim: int) -> int:
    return embedding_dim + DQN_STATE_SCALAR_DIM


class NarrativeVectorizer:
    """
    Handles the Vectorization of LLM outputs using embeddings.
    Supports multiple embedding backends:
    1. Hugging Face sentence-transformers (recommended, open-source)
    2. Azure OpenAI (requires API key)
    3. TF-IDF (fallback)
    """
    def __init__(self, azure_client=None, model_name='all-MiniLM-L6-v2', embedding_dim=None):
        """
        Initialize vectorizer with embedding model.
        
        Args:
            azure_client: Optional Azure OpenAI client
            model_name: Hugging Face model name for sentence-transformers
                       Popular options:
                       - 'all-MiniLM-L6-v2' (384-dim, fast, recommended)
                       - 'all-mpnet-base-v2' (768-dim, more accurate)
                       - 'all-MiniLM-L12-v2' (384-dim, balanced)
            embedding_dim: Override embedding dimension (auto-detected if None)
        """
        self.azure_client = azure_client
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_type = None  # 'huggingface', 'azure', or 'tfidf'
        
        # Try to initialize embedding models in order of preference
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"   🤗 Loading Hugging Face model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.embedding_type = 'huggingface'
                print(f"   ✅ Using Hugging Face embeddings ({self.embedding_dim} dimensions)")
            except Exception as e:
                print(f"   ⚠️ Failed to load Hugging Face model: {e}")
                self.embedding_model = None
        
        # Fallback to Azure if HF not available
        if self.embedding_model is None and azure_client is not None and OPENAI_AVAILABLE:
            self.embedding_dim = embedding_dim or 1536
            self.embedding_type = 'azure'
            print(f"   ☁️ Using Azure OpenAI embeddings ({self.embedding_dim} dimensions)")
        
        # Final fallback to TF-IDF
        if self.embedding_model is None and azure_client is None:
            self.vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.embedding_dim = 100
            self.embedding_type = 'tfidf'
            print(f"   📊 Using TF-IDF vectorization ({self.embedding_dim} dimensions)")
        
        # KMeans for discrete state clustering
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self._ensure_models_dir()
        self.is_fitted = False
        self.embedding_cache = {}
        
        # Sentiment anchors for zero-shot classification
        self.sentiment_anchors = None
        self._initialize_sentiment_anchors()

    def _ensure_models_dir(self):
        if not os.path.exists("models"):
            os.makedirs("models")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using available backend.
        Priority: Hugging Face > Azure OpenAI > TF-IDF
        """
        # Check cache
        cache_key = f"{self.embedding_type}:{text[:100]}"  # Use truncated text as key
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = None
        
        # Try Hugging Face
        if self.embedding_type == 'huggingface' and self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                # Convert to float64 for sklearn compatibility
                embedding = embedding.astype(np.float64)
                self.embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                print(f"   ⚠️ Hugging Face embedding error: {e}")
        
        # Try Azure OpenAI
        if self.embedding_type == 'azure' and self.azure_client is not None:
            try:
                response = self.azure_client.embeddings.create(
                    input=text,
                    model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float64)
                self.embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                print(f"   ⚠️ Azure embedding error: {e}. Falling back to TF-IDF.")
        
        # Fallback to TF-IDF
        embedding = self._get_tfidf_embedding(text)
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def _get_tfidf_embedding(self, text: str) -> np.ndarray:
        """Fallback TF-IDF embedding."""
        if not self.is_fitted:
            return np.zeros(self.embedding_dim, dtype=np.float64)
        
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].astype(np.float64)

    def _initialize_sentiment_anchors(self):
        """
        Create sentiment anchor embeddings for zero-shot classification.
        These are computed once and reused for all sentiment detection.
        """
        if self.embedding_type == 'huggingface' and self.embedding_model is not None:
            # Financial sentiment anchors
            bullish_anchor_text = """
            The company delivered exceptional results with record revenue growth.
            Strong demand, expanding margins, and raised guidance indicate robust momentum.
            Management is optimistic about future opportunities and accelerating business trends.
            """
            
            neutral_anchor_text = """
            The company reported results in line with expectations.
            Performance was stable with no major surprises or changes to outlook.
            Business conditions remain mixed with balanced opportunities and challenges.
            """
            
            bearish_anchor_text = """
            The company disappointed with declining revenue and weakening fundamentals.
            Concerns about headwinds, margin pressure, and lowered guidance create uncertainty.
            Management highlighted significant challenges and risks facing the business.
            """
            
            self.sentiment_anchors = {
                'bullish': self.get_embedding(bullish_anchor_text),
                'neutral': self.get_embedding(neutral_anchor_text),
                'bearish': self.get_embedding(bearish_anchor_text)
            }
            logger.info("   ✅ Sentiment anchors initialized")
    
    def fit(self, texts: list, embeddings: Optional[list] = None):
        """
        No-op function kept for backward compatibility.
        Sentiment now determined via zero-shot cosine similarity to anchors.
        """
        print("   ✅ [Vectorizer] Ready (using zero-shot sentiment classification)")
        self.is_fitted = True  # Always "fitted" since we don't need training

    def get_narrative_state(self, text: str = None, embedding: np.ndarray = None) -> int:
        """
        Determine sentiment using zero-shot classification with cosine similarity.
        
        Compares the input embedding against bullish/neutral/bearish anchor embeddings
        and returns the state with highest similarity.
        
        Returns: 
            0 = Bullish  (positive sentiment, growth-oriented)
            1 = Neutral  (mixed or moderate sentiment)
            2 = Bearish  (negative sentiment, concerns/risks)
        """
        # Get embedding if not provided
        if embedding is None:
            if text is None:
                return 1  # Default to neutral
            embedding = self.get_embedding(text)
        
        embedding = np.array(embedding, dtype=np.float64)
        
        # If sentiment anchors not initialized (non-HuggingFace mode), use fallback
        if self.sentiment_anchors is None:
            # Fallback to simple keyword matching
            if text:
                text_lower = text.lower()
                bullish_score = sum(1 for kw in ['strong', 'growth', 'record', 'exceed', 'robust', 'positive'] if kw in text_lower)
                bearish_score = sum(1 for kw in ['weak', 'decline', 'concern', 'risk', 'pressure', 'negative'] if kw in text_lower)
                if bullish_score > bearish_score + 2:
                    return 0
                elif bearish_score > bullish_score + 2:
                    return 2
            return 1
        
        # Zero-shot classification using cosine similarity
        embedding_reshaped = embedding.reshape(1, -1)
        
        similarities = {
            'bullish': cosine_similarity(embedding_reshaped, self.sentiment_anchors['bullish'].reshape(1, -1))[0][0],
            'neutral': cosine_similarity(embedding_reshaped, self.sentiment_anchors['neutral'].reshape(1, -1))[0][0],
            'bearish': cosine_similarity(embedding_reshaped, self.sentiment_anchors['bearish'].reshape(1, -1))[0][0]
        }
        
        # Return state with highest similarity
        max_sentiment = max(similarities, key=similarities.get)
        
        state_map = {'bullish': 0, 'neutral': 1, 'bearish': 2}
        return state_map[max_sentiment]
    
    def get_narrative_vector(self, text: str) -> np.ndarray:
        """
        Transform text into a continuous embedding vector.
        Returns: Embedding vector (1536-dim for Azure, 100-dim for TF-IDF).
        """
        return self.get_embedding(text)
