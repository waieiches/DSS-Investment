from datetime import datetime
from src.summary.gpt_summary import summarize_article
from src.sentiment.finbert_sentiment import analyze_sentiment_finbert
from src.process_news import extract_keywords
from src.process_news import process_and_prepare
from src.risk_model.predict_risk import predict_with_explanation
import json, os

def process_and_predict(ticker, name, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    path = f"data/news/{ticker}_{date_str}.json"

    process_and_prepare(ticker, name, date_str)

    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    for article in articles:
        result = predict_with_explanation(
            sentiment_label=article["sentiment_label"],
            score=article["sentiment_score"],
            full_content=article["full_content"],
            summary=article["summary"],
            keywords=article["keywords"]
        )
        results.append({
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "predict_label": result["judgment"],
            "confidence": result["confidence"],
            "source": result["source"],
            "reason": result["reason"],
            "keywords": result["keywords"],
        })

    out_path = f"result/predict_{ticker}_{date_str}.json"
    os.makedirs("result", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ {ticker} 예측 저장 완료: {out_path}")


#run
target_list = [
    ("PLTR", "Palantir Technologies"),
    ("NVDA", "NVIDIA"),
    ("CGC", "Canopy Growth Corporation"),
]

for ticker, name in target_list:
    process_and_predict(ticker, name)