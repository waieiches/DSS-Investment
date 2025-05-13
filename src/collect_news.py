import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import click
from newspaper import Article

# .env에서 API 키 불러오기
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def extract_full_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"❌ 본문 추출 실패 ({url}): {e}")
        return None

# 뉴스 수집 함수
def collect_news(ticker, name, days=7, page_size=20):
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        "q": f"{name} OR {ticker}",
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API 에러: {response.status_code} - {response.text}")

    articles = response.json().get("articles", [])
    print(f"✅ {ticker} 기사 수: {len(articles)}")
    
    for article in articles:
        url = article.get("url")
        if url:
            full_content = extract_full_content(url)
            article["full_content"] = full_content
        else:
            article["full_content"] = None
    return articles

# 저장 함수
def save_news(ticker, articles):
    date_str = datetime.now().strftime('%Y-%m-%d')
    os.makedirs("data/news", exist_ok=True)
    path = f"data/news/{ticker}_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"📁 저장 완료: {path}")

# CLI 실행 모드
@click.command()
@click.option('--ticker', prompt='기업 티커', help='예: TSLA')
@click.option('--name', prompt='기업 이름', help='예: Tesla')
@click.option('--days', default=7, help='며칠 전부터 수집할지')
@click.option('--page_size', default=20, help='뉴스 개수')
def cli_mode(ticker, name, days, page_size):
    print(f"🔎 {ticker} ({name}) 뉴스 수집 시작")
    articles = collect_news(ticker, name, days, page_size)
    save_news(ticker, articles)

# 자동 순회 수집 모드
def collect_all_from_json():
    with open("data/companies.json", "r") as f:
        all_companies = json.load(f)

    for theme, companies in all_companies.items():
        print(f"\n🔷 [테마: {theme}]")
        for company in companies:
            ticker = company.get("ticker")
            name = company.get("name")
            if not ticker or not name:
                continue
            try:
                articles = collect_news(ticker, name)
                save_news(ticker, articles)
            except Exception as e:
                print(f"❌ {ticker} 실패: {e}")

if __name__ == '__main__':
    import sys
    if '--ticker' in sys.argv:
        cli_mode()  # 명령어 기반 실행
    else:
        collect_all_from_json()  # 전체 기업 자동 수집

# python src/collect_news.py (--ticker TSLA --name Tesla --days 5)
# --ticker → 종목코드
# --name → 검색에 쓸 기업 이름
# --days → 최근 N일 간의 뉴스
# --page_size → 가져올 기사 수

# collect_news(ticker, name) → 개별 뉴스 수집
# save_news(ticker, articles) → JSON 저장
# cli_mode() → CLI 명령어로 단일 기업 수집
# collect_all_from_json() → companies.json 자동 순회 수집
# --ticker가 있으면 CLI 실행, 없으면 자동 전체 수집


