import os
import json
from datetime import datetime
from collect_news import collect_news, save_news
from sentiment.finbert_sentiment import analyze_sentiment_finbert

# GPT 요약 모듈은 추후 연결
def dummy_summary(text):
    return "This is a placeholder summary."

def load_articles(ticker):
    date_str = datetime.now().strftime('%Y-%m-%d')
    path = f"data/news/{ticker}_{date_str}.json"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print(f"📁 기존 뉴스 불러오기: {path}")
            return json.load(f)
    return None

def process_news(ticker, name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    path = f"data/news/{ticker}_{date_str}.json"

    # 1. 뉴스 수집 (이미 있으면 스킵)
    articles = load_articles(ticker)
    if not articles:
        print(f"📡 뉴스 수집 시작: {ticker} ({name})")
        articles = collect_news(ticker, name)
        save_news(ticker, articles)

    # 2. 감성 분석 추가 (없을 경우만)
    for article in articles:
        if "sentiment_label" not in article:
            label, score = analyze_sentiment_finbert(article.get("full_content", ""))
            article["sentiment_label"] = label
            article["sentiment_score"] = score

    # 3. 요약 (없을 경우만)
    for article in articles:
        if "summary" not in article:
            article["summary"] = dummy_summary(article.get("full_content", ""))

    # 4. 저장
    save_news(ticker, articles)

if __name__ == "__main__":
    # 테스트용 실행
    process_news("TSLA", "Tesla")
    # 실제 사용 시에는 collect_all_from_json() 함수를 호출하여 전체 기업에 대해 처리