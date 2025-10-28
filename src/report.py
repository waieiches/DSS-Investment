import os
import json
from datetime import datetime

def generate_report(ticker, name=None, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    path = f"result/predict_{ticker}_{date_str}.json"

    if not os.path.exists(path):
        print(f"❌ 파일이 존재하지 않음: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    total = len(articles)
    if total == 0:
        print(f"⚠️ 기사 없음: {ticker}")
        return None

    # 집계
    buy = sum(1 for a in articles if a.get("predict_label") == "매수")
    sell = sum(1 for a in articles if a.get("predict_label") == "매도")
    avg_conf = round(sum(a.get("confidence", 0) for a in articles) / total, 3)
    fallback = sum(1 for a in articles if a.get("source") == "GPT fallback")
    fallback_rate = round(fallback / total * 100, 1)

    sample_titles = [a.get("title", "") for a in articles[:2]]

    # 출력
    report = {
        "ticker": ticker,
        "name": name or ticker,
        "total_articles": total,
        "매수": f"{buy}건 ({round(buy/total*100, 1)}%)",
        "매도": f"{sell}건 ({round(sell/total*100, 1)}%)",
        "평균 confidence": avg_conf,
        "GPT fallback 비율": f"{fallback_rate}%",
        "최근 기사": sample_titles
    }

    return report

#run
tickers = [
    ("CGC", "Canopy Growth"),
    ("NVDA", "NVIDIA"),
    ("PLTR", "Palantir Technologies")
]

print("📋 종합 요약 보고서")
for ticker, name in tickers:
    r = generate_report(ticker, name)
    if r:
        print(f"\n[{r['name']} ({r['ticker']})]")
        print(f"총 기사 수: {r['total_articles']}")
        print(f"📈 매수: {r['매수']} | 📉 매도: {r['매도']}")
        print(f"🔍 평균 confidence: {r['평균 confidence']}")
        print(f"🤖 GPT fallback: {r['GPT fallback 비율']}")
        print(f"📰 최근 기사 예시:")
        for title in r["최근 기사"]:
            print(f" - {title}")
