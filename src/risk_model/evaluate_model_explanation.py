import json
from src.risk_model.predict_risk import predict_with_explanation

# 평가용 데이터 파일 경로
eval_path = "data/test_labeled_news.json"

# 결과 저장 리스트
results = []

with open(eval_path, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# 평가 진행
for news_item in news_data:
    result = predict_with_explanation(news_item)
    result_entry = {
        "summary": news_item.get("summary"),
        "keywords": news_item.get("keywords"),
        "true_label": news_item.get("label"),
        "predicted_label": result["judgment"],
        "confidence": result["confidence"],
        "source": result["source"],
        "reason": result["reason"],
        "extracted_keywords": result["keywords"]
    }
    results.append(result_entry)

# 결과 저장
with open("evaluation_results_with_explanation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ 평가 완료: evaluation_results_with_explanation.json 저장됨")
