import joblib
import numpy as np
from src.features.feature_engineering import build_feature_vector
from src.llm.llm_fallback import gpt_risk_decision

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

# 예시
if __name__ == "__main__":
    test_news = {
        "sentiment": "positive",
        "summary": "Apple exceeded earnings expectations...",
        "keywords": ["earnings", "iPhone"]
    }
    print("최종 판단:", predict_with_fallback(test_news))
