from src.risk_model.predict_risk import predict_with_explanation
from src.sentiment.finbert_sentiment import analyze_sentiment_finbert  # 예시 경로
import json
import os

input_dir = "data/news/"
output_dir = "data/predictions/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    with open(input_path, "r", encoding="utf-8") as f:
        news_items = json.load(f)

    results = []
    for item in news_items:
        try:
            # ✅ 감성 분석 먼저 수행
            sentiment = analyze_sentiment_finbert(item.get("full_content", ""))
            item["sentiment_label"] = sentiment

            # ✅ 투자 판단 수행
            result = predict_with_explanation(item)

        except Exception as e:
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "predict_label": "예측 실패",
                "reason": str(e)
            }

        results.append(result)

    output_path = os.path.join(output_dir, filename.replace(".json", "_predictions.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 모든 뉴스 예측 완료")
