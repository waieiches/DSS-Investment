import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('baconnier/Finance2_embedding_small_en-V1.5')

def build_feature_vector(news_item):
    """
    news_item: {
        "sentiment": "positive",
        "summary": "Apple exceeded earnings expectations...",
        "keywords": ["earnings", "iPhone"]
    }
    """
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_score = sentiment_map.get(news_item["sentiment"], 0)

    summary_vec = model.encode(news_item["summary"])
    keywords_vec = model.encode(" ".join(news_item["keywords"]))

    return np.concatenate([[sentiment_score], summary_vec, keywords_vec])
