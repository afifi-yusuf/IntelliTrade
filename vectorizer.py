import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os
from typing import Optional

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

    def fit(self, texts: list, embeddings: Optional[list] = None):
        """
        Fit the Clustering model on historical embeddings or summaries.
        """
        print(f"   📊 [Vectorizer] Fitting on {len(texts)} historical summaries...")
        
        if len(texts) < 3:
            print("   ⚠️ Not enough data to fit KMeans (need > 3 samples). Using fallback logic.")
            self.is_fitted = False
            return

        # Get vectors (embeddings or TF-IDF)
        if embeddings is not None and len(embeddings) == len(texts):
            vectors = np.array(embeddings, dtype=np.float64)
        elif self.embedding_type == 'huggingface' or self.embedding_type == 'azure':
            vectors = np.array([self.get_embedding(text) for text in texts], dtype=np.float64)
        else:
            # TF-IDF fallback
            self.vectorizer.fit_transform(texts)
            vectors = self.vectorizer.transform(texts).toarray().astype(np.float64)
        
        # Ensure float64 dtype for sklearn
        vectors = vectors.astype(np.float64)
        
        # Fit KMeans
        self.kmeans.fit(vectors)
        self.is_fitted = True
        print("   ✅ [Vectorizer] Models fitted successfully.")

    def get_narrative_state(self, text: str = None, embedding: np.ndarray = None) -> int:
        """
        Transform text/embedding into a Discrete Narrative State (Cluster ID).
        Returns: 0, 1, or 2.
        """
        if not self.is_fitted:
            # Fallback simple logic if not enough data to fit
            if text:
                lower = text.lower()
                if "growth" in lower or "strong" in lower or "record" in lower:
                    return 0 # "Bullish"
                elif "weak" in lower or "decline" in lower or "headwinds" in lower:
                    return 2 # "Bearish"
            return 1 # "Neutral"

        # Get embedding if not provided
        if embedding is None:
            if text is None:
                return 1
            embedding = self.get_embedding(text)
        
        # Ensure float64 dtype for sklearn compatibility
        embedding = np.array(embedding, dtype=np.float64)
        
        cluster_id = self.kmeans.predict([embedding])[0]
        return int(cluster_id)
    
    def get_narrative_vector(self, text: str) -> np.ndarray:
        """
        Transform text into a continuous embedding vector.
        Returns: Embedding vector (1536-dim for Azure, 100-dim for TF-IDF).
        """
        return self.get_embedding(text)
