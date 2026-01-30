"""
Quick test to see what embedding values look like for different sentiments
"""

from vectorizer import NarrativeVectorizer
import numpy as np

vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')

# Test different sentiment texts
test_texts = {
    "Bullish": """
    NVIDIA delivered exceptional Q2 results with record revenue of $30.0B, up 122% year-over-year.
    The data center segment exploded to $26.3B driven by insatiable demand for H100 GPUs.
    Gross margins expanded to 78% as the company benefits from strong pricing power.
    Management raised full-year guidance significantly citing unprecedented AI adoption.
    All business segments are accelerating with gaming returning to growth.
    The Blackwell architecture launch positions NVIDIA for continued market dominance.
    """,
    
    "Neutral": """
    NVIDIA's Q2 showed mixed results with revenue meeting expectations at $15B.
    Data center growth moderated while gaming remained challenged by inventory corrections.
    Margins held steady around 65% as product mix normalized.
    Management maintained prior guidance with no major changes to outlook.
    The quarter lacked significant positive or negative surprises.
    """,
    
    "Bearish": """
    NVIDIA disappointed with Q2 revenue declining 20% to $6.7B amid weak demand.
    Data center sales fell sharply as cloud providers paused infrastructure spending.
    Gross margins compressed to 43% due to excess inventory and pricing pressure.
    Management cut guidance citing prolonged headwinds and crypto collapse.
    Gaming segment remains severely depressed with no recovery in sight.
    Regulatory risks in China create additional uncertainty for the business.
    """
}

print("\n" + "="*80)
print("EMBEDDING ANALYSIS FOR DIFFERENT SENTIMENTS")
print("="*80)

from sklearn.metrics.pairwise import cosine_similarity

for sentiment, text in test_texts.items():
    embedding = vectorizer.get_embedding(text)
    
    # Get sentiment via zero-shot classification (with text for proper detection)
    detected_state = vectorizer.get_narrative_state(text=text, embedding=embedding)
    state_names = {0: "Bullish", 1: "Neutral", 2: "Bearish"}
    
    # Show cosine similarities to each anchor
    if vectorizer.sentiment_anchors:
        emb_reshaped = embedding.reshape(1, -1)
        similarities = {
            'Bullish': cosine_similarity(emb_reshaped, vectorizer.sentiment_anchors['bullish'].reshape(1, -1))[0][0],
            'Neutral': cosine_similarity(emb_reshaped, vectorizer.sentiment_anchors['neutral'].reshape(1, -1))[0][0],
            'Bearish': cosine_similarity(emb_reshaped, vectorizer.sentiment_anchors['bearish'].reshape(1, -1))[0][0]
        }
    
    print(f"\n{sentiment.upper()}:")
    if vectorizer.sentiment_anchors:
        print(f"  Cosine Similarities:")
        for anchor_name, sim_score in similarities.items():
            marker = " ✓" if (anchor_name == state_names[detected_state]) else ""
            print(f"    • vs {anchor_name}: {sim_score:.4f}{marker}")
    print(f"  → Detected State: {detected_state} ({state_names[detected_state]})")

print("\n" + "="*80)
print("METHOD: Zero-Shot Classification with Cosine Similarity")
print("  - Compare LLM output embedding to anchor embeddings")
print("  - Select anchor with highest cosine similarity")
print("  - No training data required!")
print("="*80)
