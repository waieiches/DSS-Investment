import os
import json
from datetime import datetime
from src.collect_news import collect_news, save_news
from src.sentiment.finbert_sentiment import analyze_sentiment_finbert
from src.summary.gpt_summary import summarize_article
import re
from collections import Counter

def get_article_path(ticker, date_str):
    return f"data/news/{ticker}_{date_str}.json"

def load_articles(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print(f"📁 기존 뉴스 불러오기: {path}")
            return json.load(f)
    return None

def extract_keywords(text, top_n=5):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # 길이 4이상 영어 단어
    common = Counter(words).most_common(top_n)
    return [word for word, _ in common]

def ensure_fields(article):
    """
    summary → sentiment_label, sentiment_score 순으로 반드시 실행되도록 처리.
    """
    full_content = article.get("full_content", "")

    if not full_content or len(full_content.strip()) < 40:
        print("❗ 본문 없음 → 요약/감성분석 건너뜀")
        article["summary"] = "본문 없음"
        article["sentiment_label"] = "neutral"
        article["sentiment_score"] = 0.0
        return

    # 1. summary
    if "summary" not in article or not article["summary"]:
        article["summary"] = summarize_article(full_content)

    # 2. keywords (간단 키워드 생성기)
    if "keywords" not in article or not article["keywords"]:
        article["keywords"] = extract_keywords(article["summary"])

    # 3. sentiment
    if "sentiment_label" not in article or "sentiment_score" not in article:
        label, score = analyze_sentiment_finbert(full_content)
        article["sentiment_label"] = label
        article["sentiment_score"] = score

def process_and_prepare(ticker, name, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    path = get_article_path(ticker, date_str)

    # 1. Load or collect
    articles = load_articles(path)
    if not articles:
        print(f"📡 뉴스 수집: {ticker} ({name})")
        articles = collect_news(ticker, name)
        save_news(ticker, articles)

    # 2. Process each article (요약 → 감성 분석)
    for article in articles:
        ensure_fields(article)

    # ✅ 사후 정제: 본문 누락 or 너무 짧은 기사 제거
    original_len = len(articles)
    articles = [a for a in articles if a.get("full_content") and len(a["full_content"].strip()) >= 40]
    removed = original_len - len(articles)
    print(f"🧹 본문 누락 기사 {removed}개 제거됨")

    # 3. Save updated articles
    save_news(ticker, articles)
    print(f"✅ 저장 완료: {path}")

if __name__ == "__main__":
    # 테스트 ticker와 날짜 입력
    ticker = "AAPL"
    date_str = datetime.now().strftime("%Y%m%d")  # 예: 20250602
    process_and_prepare(ticker, date_str)