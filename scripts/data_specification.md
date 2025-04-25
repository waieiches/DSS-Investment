# 📄 Data Specification (수집 데이터 스펙)

## 뉴스 데이터
-  `ticker` : 기업 티커커
- `title`: 뉴스 제목
- `content`: 뉴스 본문
- `date`: 발행 날짜 (YYYY-MM-DD)
- `source`: 뉴스 출처
- `sentiment_lable` : 감성 분석 결과 (positive, neutral, negetive)
- `sentiment_score`: 감성 분석 점수 (0~1, FinBERT)
- `summary`: GPT-4 요약 결과

## 공시자료 데이터
- `title`: 공시 제목
- `date`: 공시 날짜 (YYYY-MM-DD)
- `summary`: 공시 주요 내용 요약
- `risk_keywords`

## 주가 데이터
- `date`: 날짜 (YYYY-MM-DD)
- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `volume`: 거래량