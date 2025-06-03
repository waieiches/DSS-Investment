import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import joblib
import numpy as np
from src.features.feature_engineering import build_feature_vector
from src.llm.llm_fallback import gpt_risk_decision, gpt_risk_decision_with_reason

model = joblib.load("models/risk_model.pkl")
label_map_rev = {0: "매수", 1: "관망", 2: "매도"}

def predict_with_fallback(news_item, threshold=0.6):
    feature = build_feature_vector(news_item).reshape(1, -1)
    proba = model.predict_proba(feature)[0]
    confidence = max(proba)
    prediction = model.predict(feature)[0]
    
    if confidence < threshold:
        print(f"🤔 확신 낮음 ({confidence:.2f}) → GPT-4 호출")
        return gpt_risk_decision(news_item)
    else:
        return label_map_rev[prediction]

def predict_with_explanation(sentiment_label: str, score: float, full_content: str, summary: str = "", keywords=None, threshold=0.6):
    if keywords is None:
        keywords = []

    news_item = {
        "sentiment_label": sentiment_label,
        "sentiment_score": score,
        "full_content": full_content,
        "summary": summary,
        "keywords": keywords,
    }
    print("📝 Summary:", summary)
    feature = build_feature_vector(news_item).reshape(1, -1)
    print("✅ shape:", feature.shape)
    proba = model.predict_proba(feature)[0]
    print("🧪 확률 분포:", proba)
    confidence = max(proba)
    prediction = model.predict(feature)[0]
    label = label_map_rev[prediction]
    print("🧬 Feature vector:", build_feature_vector(news_item)[:10])  # 앞 10개만 출력

    if confidence < threshold:
        print(f"🤔 확신 낮음 ({confidence:.2f}) → GPT-4 호출")
        fallback_result = gpt_risk_decision_with_reason(news_item)  # GPT 설명 포함
        return {
            "judgment": fallback_result["label"],
            "confidence": confidence,
            "source": "GPT fallback",
            "reason": fallback_result["reason"],
            "keywords": fallback_result.get("keywords", [])
        }
    else:
        return {
            "judgment": label,
            "confidence": confidence,
            "source": "ML model",
            "reason": f"모델이 {confidence:.2f}의 높은 확신도로 예측",
            "keywords": keywords
        }


# 예시
if __name__ == "__main__":
    test_news = {
        "summary": "Mixed financial results and pending regulatory approvals have left investors uncertain.",
        "keywords": [
            "mixed results",
            "regulatory review",
            "market uncertainty"
        ],
        "sentiment_label": "negative",
        "sentiment_score": 0.62,
        "full_content": "Mixed financial results and regulatory challenges continue to cause volatility."
    }
    print("최종 판단:", predict_with_explanation(
        sentiment_label=test_news["sentiment_label"],
        score=test_news["sentiment_score"],
        full_content=test_news["full_content"],
        summary=test_news["summary"],
        keywords=test_news["keywords"]
    ))
