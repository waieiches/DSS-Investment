from datetime import datetime
from src.process_news import process_and_prepare
from src.risk_model.predict_risk import predict_with_explanation
import json, os

tickers = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("IonQ", "IonQ"),
    ("RGTI", "Rigetti Computing"),
    ("QBTS", "D-Wave Quantum"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet (Google)")
]

for ticker, name in tickers:
    # 1. 뉴스 수집 + 요약 + 감성 분석
    process_and_prepare(ticker, name)

    # 2. 예측
    path = f"data/news/{ticker}_{datetime.now().strftime('%Y-%m-%d')}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        results = []
        for article in articles:
            try:
                result = predict_with_explanation(
                    sentiment_label=article["sentiment_label"],
                    score=article["sentiment_score"],
                    full_content=article["full_content"],
                    summary=article.get("summary", ""),
                    keywords=article.get("keywords", [])
                )
                results.append({
                    "title": article.get("title", ""),
                    "predict_label": result["judgment"],
                    "confidence": result["confidence"],
                    "source": result["source"],
                    "reason": result["reason"]
                })
            except Exception as e:
                results.append({
                    "title": article.get("title", ""),
                    "predict_label": "예측 실패",
                    "reason": str(e)
                })

        # 결과 저장
        out_path = f"predict_results/{ticker}_predict.json"
        os.makedirs("predict_results", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📦 예측 결과 저장: {out_path}")
