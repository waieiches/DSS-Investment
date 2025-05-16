import os
import json
from datetime import datetime
from collect_news import collect_news, save_news
from sentiment.finbert_sentiment import analyze_sentiment_finbert
from summary.gpt_summary import summarize_article

def get_article_path(ticker, date_str):
    return f"data/news/{ticker}_{date_str}.json"

def load_articles(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print(f"📁 기존 뉴스 불러오기: {path}")
            return json.load(f)
    return None

def process_news(ticker, name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    path = get_article_path(ticker, date_str)

    # 1. 뉴스 수집 (없을 경우에만)
    articles = load_articles(path)
    if not articles:
        print(f"📡 뉴스 수집 시작: {ticker} ({name})")
        articles = collect_news(ticker, name)
        save_news(ticker, articles)

    # 2. 감성 분석 수행 (없는 경우에만)
    for article in articles:
        if "sentiment_label" not in article:
            label, score = analyze_sentiment_finbert(article.get("full_content", ""))
            article["sentiment_label"] = label
            article["sentiment_score"] = score

    # 3. GPT 요약 수행 (없는 경우에만)
    for article in articles:
        if "summary" not in article:
            article["summary"] = summarize_article(article.get("full_content", ""))

    # 4. 저장
    save_news(ticker, articles)

if __name__ == "__main__":
    # 테스트 실행용
    process_news("TSLA", "Tesla")
    # 실제 사용 시에는 collect_all_from_json() 함수를 호출하여 전체 기업에 대해 처리