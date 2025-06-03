import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('baconnier/Finance2_embedding_small_en-V1.5')

def build_feature_vector(news_item):
    """
    news_item: {
        "sentiment_label": "positive",
        "summary": "Apple exceeded earnings expectations...",
        "keywords": ["earnings", "iPhone"]
    }
    """
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    sentiment_label = news_item.get("sentiment_label", "neutral")
    sentiment_score = sentiment_map.get(sentiment_label, 0)

    summary_text = news_item.get("summary", "").strip()
    keywords_text = " ".join(news_item.get("keywords", [])).strip()

    if summary_text:
        summary_vec = model.encode(summary_text)
    else:
        summary_vec = np.zeros(model.get_sentence_embedding_dimension())

    if keywords_text:
        keywords_vec = model.encode(keywords_text)
    else:
        keywords_vec = np.zeros(model.get_sentence_embedding_dimension())

    return np.concatenate([[sentiment_score], summary_vec, keywords_vec])
