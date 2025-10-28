import json
import os
from src.sentiment.finbert_sentiment import analyze_sentiment_finbert
from src.risk_model.predict_risk import predict_with_explanation

# 테스트할 하나의 뉴스 JSON 파일 지정
test_file = "data/news/AAPL_2025-06-02.json"

# 파일 존재 여부 확인
if not os.path.exists(test_file):
    raise FileNotFoundError(f"{test_file} 파일이 존재하지 않습니다.")

# 파일 읽기
with open(test_file, "r", encoding="utf-8") as f:
    news_list = json.load(f)

results = []

for article in news_list:
    full_content = article.get("full_content", "")
    summary = article.get("summary", "")
    keywords = article.get("keywords", [])
    label, score = analyze_sentiment_finbert(full_content)

    print(f"📄 기사 제목: {article.get('title', '제목 없음')}")
    print(f"🧠 감성 분석 결과: {label} ({score})")

    try:
        result = predict_with_explanation(
            sentiment_label=label,
            score=score,
            full_content=full_content,
            summary=summary,
            keywords=keywords
        )
        result_entry = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "predict_label": result["judgment"],
            "confidence": result["confidence"],
            "source": result["source"],
            "reason": result["reason"],
            "keywords": result["keywords"],
        }
    except Exception as e:
        result_entry = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "predict_label": "예측 실패",
            "reason": str(e),
        }

    results.append(result_entry)
    print(f"✅ 예측 결과: {result_entry['predict_label']}")
    print(f"📌 이유: {result_entry.get('reason')}")
    print("")

# 결과 저장
output_path = "predict_result.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📝 예측 결과 저장 완료: {output_path}")