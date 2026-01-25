import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os

class NarrativeVectorizer:
    """
    Handles the Vectorization of "Market Narratives" (LLM Summaries) using TF-IDF.
    Discretizes these vectors into "Narrative States" using KMeans Clustering.
    """
    def __init__(self, vocab_path="models/tfidf_vocab.pkl", kmeans_path="models/kmeans_model.pkl"):
        self.vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        # We start with 3 discrete narrative states (e.g. Bullish, Bearish, Uncertain)
        # In a real system, you'd tune this.
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
        self.vocab_path = vocab_path
        self.kmeans_path = kmeans_path
        self._ensure_models_dir()
        
        self.is_fitted = False

    def _ensure_models_dir(self):
        if not os.path.exists("models"):
            os.makedirs("models")

    def fit(self, texts: list):
        """
        Fit the Vectorizer and Clustering model on historical summaries.
        """
        print(f"   📊 [Vectorizer] Fitting on {len(texts)} historical summaries...")
        
        if len(texts) < 3:
            print("   ⚠️ Not enough data to fit KMeans (need > 3 samples). Using dummy logic.")
            # For demo purposes with limited data, we'll skip fitting
            self.is_fitted = False
            return

        # 1. TF-IDF
        vectors = self.vectorizer.fit_transform(texts)
        
        # 2. KMeans
        self.kmeans.fit(vectors)
        
        self.is_fitted = True
        print("   ✅ [Vectorizer] Models fitted successfully.")
        
        # Save models (optional, for persistency)
        # joblib.dump(self.vectorizer, self.vocab_path)
        # joblib.dump(self.kmeans, self.kmeans_path)

    def get_narrative_state(self, text: str) -> int:
        """
        Transform a new text summary into a Discrete Narrative State (Cluster ID).
        Returns: 0, 1, or 2.
        """
        if not self.is_fitted:
            # Fallback simple logic if not enough data to fit
            # Just simple keyword matching for "State"
            lower = text.lower()
            if "growth" in lower or "strong" in lower or "record" in lower:
                return 0 # "Bullish"
            elif "weak" in lower or "decline" in lower or "headwinds" in lower:
                return 2 # "Bearish"
            else:
                return 1 # "Neutral"

        vector = self.vectorizer.transform([text])
        cluster_id = self.kmeans.predict(vector)[0]
        return int(cluster_id)
